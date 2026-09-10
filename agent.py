"""IslandChess — super-Capablanca tournament agent for AI Chessathon.

One method to a crazy extreme: Capablanca's. Flawless pawn structure,
classical development, king safety first, simplification when ahead,
winning endgames — plus preemptive attack (hanging pieces and exposed
kings get struck first) and preemptive defense (castle early, never
loosen the shield, never hang a piece).

Strategy: single-threaded iterative-deepening negamax with alpha-beta,
quiescence (captures+promotions), transposition table, MVV-LVA + killer +
history ordering, null-move pruning, late-move reduction, tapered
material + piece-square evaluation (numba-jitted), repetition-aware move
choice, and wall-clock time management for 120s + 0.5s.

Only uses the allowed stack: stdlib + python-chess + numpy + numba.
No third-party engine code, no published network. All code below is
original for this event.

Protocol: platform imports this file (90s init budget) and calls
  get_move(fen: str, time_left_ms: int) -> str  (UCI, e.g. 'e2e4'/'e7e8q')
Process persists across moves of one game, suspended on opponent turn.
"""

import os
import time

# Keep everything single-threaded: one core is fastest with one thread.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
try:
    import numba

    numba.set_num_threads(1)
except Exception:
    numba = None  # type: ignore

import chess
import numpy as np

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

INF = 100000
MATE = 90000
MAX_PLY = 64

# Material in centipawns (K=0, handled by mate scores + PST).
PIECE_VALUE = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
               chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0}

# -- Simplified Evaluation Function tables (T. Michniewski, public domain
# -- teaching tables), written rank8-first as published. Flipped at load into
# -- python-chess square order (a1=0..h8=63).
_P_PST = [
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
]
_N_PST = [
   -50,-40,-30,-30,-30,-30,-40,-50,
   -40,-20,  0,  0,  0,  0,-20,-40,
   -30,  0, 10, 15, 15, 10,  0,-30,
   -30,  5, 15, 20, 20, 15,  5,-30,
   -30,  0, 15, 20, 20, 15,  0,-30,
   -30,  5, 10, 15, 15, 10,  5,-30,
   -40,-20,  0,  5,  5,  0,-20,-40,
   -50,-40,-30,-30,-30,-30,-40,-50,
]
_B_PST = [
   -20,-10,-10,-10,-10,-10,-10,-20,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -10,  0,  5, 10, 10,  5,  0,-10,
   -10,  5,  5, 10, 10,  5,  5,-10,
   -10,  0, 10, 10, 10, 10,  0,-10,
   -10, 10, 10, 10, 10, 10, 10,-10,
   -10,  5,  0,  0,  0,  0,  5,-10,
   -20,-10,-10,-10,-10,-10,-10,-20,
]
_R_PST = [
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0,
]
_Q_PST = [
   -20,-10,-10, -5, -5,-10,-10,-20,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -10,  0,  5,  5,  5,  5,  0,-10,
    -5,  0,  5,  5,  5,  5,  0, -5,
     0,  0,  5,  5,  5,  5,  0, -5,
   -10,  5,  5,  5,  5,  5,  0,-10,
   -10,  0,  5,  0,  0,  0,  0,-10,
   -20,-10,-10, -5, -5,-10,-10,-20,
]
_KMG_PST = [
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -20,-30,-30,-40,-40,-30,-30,-20,
   -10,-20,-20,-20,-20,-20,-20,-10,
    20, 20,  0,  0,  0,  0, 20, 20,
    20, 30, 10,  0,  0, 10, 30, 20,
]
_KEG_PST = [
   -50,-40,-30,-20,-20,-30,-40,-50,
   -30,-20,-10,  0,  0,-10,-20,-30,
   -30,-10, 20, 30, 30, 20,-10,-30,
   -30,-10, 30, 40, 40, 30,-10,-30,
   -30,-10, 30, 40, 40, 30,-10,-30,
   -30,-10, 20, 30, 30, 20,-10,-30,
   -30,-30,  0,  0,  0,  0,-30,-30,
   -50,-30,-30,-30,-30,-30,-30,-50,
]


def _flip_to_chess_order(published):
    """Convert rank8-first table to python-chess square order (a1=0)."""
    out = [0] * 64
    for sq in range(64):
        p = (7 - sq // 8) * 8 + (sq % 8)
        out[sq] = published[p]
    return np.array(out, dtype=np.int16)


MG_P = _flip_to_chess_order(_P_PST)
MG_N = _flip_to_chess_order(_N_PST)
MG_B = _flip_to_chess_order(_B_PST)
MG_R = _flip_to_chess_order(_R_PST)
MG_Q = _flip_to_chess_order(_Q_PST)
MG_K = _flip_to_chess_order(_KMG_PST)
EG_K = _flip_to_chess_order(_KEG_PST)

# Passed-pawn bonus by rank (white perspective, index = rank 0..7).
# Super-Capablanca: outside/connected passers decide games — cranked.
PASSED_BONUS = np.array([0, 8, 18, 35, 65, 100, 150, 0], dtype=np.int16)


# --------------------------------------------------------------------------
# Fast static evaluation (numba)
# --------------------------------------------------------------------------
# Board encoding: int8[64], 0 empty, +P/N/B/R/Q/K = 1..6 white, -1..-6 black.
# Square i corresponds to chess.square(i), i.e. 0=a1.

if numba is not None:
    import numba as _nb

    @_nb.njit(fastmath=True)
    def _evaluate_nb(b, mg_p, mg_n, mg_b, mg_r, mg_q, mg_k, eg_k, passed):
        mg = 0
        eg = 0
        phase = 0  # 0..24, 24 = full middlegame
        w_bishops = 0
        b_bishops = 0
        # pawn file counts for doubled/isolated/passed detection
        w_pf = np.zeros(8, dtype=np.int64)
        b_pf = np.zeros(8, dtype=np.int64)
        w_prank = np.full(8, -1, dtype=np.int64)  # highest rank per file (white)
        b_prank = np.full(8, 8, dtype=np.int64)   # lowest rank per file (black)
        wk_sq = -1
        bk_sq = -1
        for sq in range(64):
            p = b[sq]
            if p == 0:
                continue
            rank = sq // 8
            file = sq % 8
            ap = p if p > 0 else -p
            if ap == 1:
                if p > 0:
                    mg += 100 + mg_p[sq]
                    eg += 100 + mg_p[sq]
                    w_pf[file] += 1
                    if rank > w_prank[file]:
                        w_prank[file] = rank
                else:
                    mg -= 100 + mg_p[sq ^ 56]
                    eg -= 100 + mg_p[sq ^ 56]
                    b_pf[file] += 1
                    if rank < b_prank[file]:
                        b_prank[file] = rank
            elif ap == 2:
                if p > 0:
                    mg += 320 + mg_n[sq]
                    eg += 320 + mg_n[sq]
                else:
                    mg -= 320 + mg_n[sq ^ 56]
                    eg -= 320 + mg_n[sq ^ 56]
                phase += 1
            elif ap == 3:
                if p > 0:
                    mg += 330 + mg_b[sq]
                    eg += 330 + mg_b[sq]
                    w_bishops += 1
                else:
                    mg -= 330 + mg_b[sq ^ 56]
                    eg -= 330 + mg_b[sq ^ 56]
                    b_bishops += 1
                phase += 1
            elif ap == 4:
                if p > 0:
                    mg += 500 + mg_r[sq]
                    eg += 500 + mg_r[sq]
                    # rook on 7th — preemptive attack from Capablanca's playbook
                    if rank == 6:
                        mg += 20
                else:
                    mg -= 500 + mg_r[sq ^ 56]
                    eg -= 500 + mg_r[sq ^ 56]
                    if rank == 1:
                        mg -= 20
                phase += 2
            elif ap == 5:
                if p > 0:
                    mg += 900 + mg_q[sq]
                    eg += 900 + mg_q[sq]
                else:
                    mg -= 900 + mg_q[sq ^ 56]
                    eg -= 900 + mg_q[sq ^ 56]
                phase += 4
            else:  # king
                if p > 0:
                    mg += mg_k[sq]
                    eg += eg_k[sq]
                    wk_sq = sq
                else:
                    mg -= mg_k[sq ^ 56]
                    eg -= eg_k[sq ^ 56]
                    bk_sq = sq
        if phase > 24:
            phase = 24
        score = (mg * phase + eg * (24 - phase)) // 24
        # -- Super-Capablanca: no weaknesses, ever. Bishop pair, flawless
        # -- pawns, rooks on open files, knights on outposts, king safety.
        # bishop pair
        if w_bishops >= 2:
            score += 45
        if b_bishops >= 2:
            score -= 45
        # pawn structure per file
        for f in range(8):
            # doubled pawns — Capablanca hated them: extreme penalty
            if w_pf[f] > 1:
                score -= 22 * (w_pf[f] - 1)
            if b_pf[f] > 1:
                score += 22 * (b_pf[f] - 1)
            # isolated pawns
            w_iso = True
            b_iso = True
            if f > 0:
                if w_pf[f - 1] > 0:
                    w_iso = False
                if b_pf[f - 1] > 0:
                    b_iso = False
            if f < 7:
                if w_pf[f + 1] > 0:
                    w_iso = False
                if b_pf[f + 1] > 0:
                    b_iso = False
            if w_pf[f] > 0 and w_iso:
                score -= 28
            if b_pf[f] > 0 and b_iso:
                score += 28
            # passed pawns (no enemy pawn ahead on same/adjacent files)
            if w_pf[f] > 0:
                wr = w_prank[f]
                blocked = False
                for df in (-1, 0, 1):
                    nf = f + df
                    if 0 <= nf < 8 and b_pf[nf] > 0 and b_prank[nf] > wr:
                        blocked = True
                        break
                if not blocked:
                    score += passed[wr]
            if b_pf[f] > 0:
                br = b_prank[f]
                blocked = False
                for df in (-1, 0, 1):
                    nf = f + df
                    if 0 <= nf < 8 and w_pf[nf] > 0 and w_prank[nf] < br:
                        blocked = True
                        break
                if not blocked:
                    score -= passed[7 - br]
        # rooks belong on open files; knights on supported outposts.
        for sq in range(64):
            p = b[sq]
            if p == 4:  # white rook
                rf = sq % 8
                if w_pf[rf] == 0 and b_pf[rf] == 0:
                    score += 25
                elif w_pf[rf] == 0:
                    score += 12
            elif p == -4:  # black rook
                rf = sq % 8
                if w_pf[rf] == 0 and b_pf[rf] == 0:
                    score -= 25
                elif b_pf[rf] == 0:
                    score -= 12
            elif p == 2:  # white knight outpost: advanced, central, pawn-supported
                nr = sq // 8
                nf = sq % 8
                if nr >= 3 and 1 <= nf <= 6:
                    if nr >= 1:
                        if (nf > 0 and b[(nr - 1) * 8 + nf - 1] == 1) or (
                            nf < 7 and b[(nr - 1) * 8 + nf + 1] == 1
                        ):
                            score += 18
            elif p == -2:  # black knight outpost
                nr = sq // 8
                nf = sq % 8
                if nr <= 4 and 1 <= nf <= 6:
                    if nr <= 6:
                        if (nf > 0 and b[(nr + 1) * 8 + nf - 1] == -1) or (
                            nf < 7 and b[(nr + 1) * 8 + nf + 1] == -1
                        ):
                            score -= 18
        # king pawn shield (middlegame-ish): bonus for friendly pawns near king
        if wk_sq >= 0:
            wkf = wk_sq % 8
            wkr = wk_sq // 8
            if wkr <= 1:
                for df in (-1, 0, 1):
                    nf = wkf + df
                    if 0 <= nf < 8 and w_pf[nf] > 0:
                        score += 12
                    else:
                        score -= 16
        if bk_sq >= 0:
            bkf = bk_sq % 8
            bkr = bk_sq // 8
            if bkr >= 6:
                for df in (-1, 0, 1):
                    nf = bkf + df
                    if 0 <= nf < 8 and b_pf[nf] > 0:
                        score -= 12
                    else:
                        score += 16
        return score

    def _eval_board_array(arr):
        return int(_evaluate_nb(arr, MG_P, MG_N, MG_B, MG_R, MG_Q, MG_K, EG_K, PASSED_BONUS))

else:  # pragma: no cover - fallback if numba missing (never on platform)

    def _eval_board_array(arr):
        return 0


def _board_to_array(board):
    arr = np.zeros(64, dtype=np.int8)
    for sq in chess.SQUARES:
        pc = board.piece_at(sq)
        if pc is None:
            continue
        v = pc.piece_type
        arr[sq] = v if pc.color == chess.WHITE else -v
    return arr


# --------------------------------------------------------------------------
# Search state (persists across moves within one game/process)
# --------------------------------------------------------------------------

_TT = {}            # transposition_key -> (depth, score, flag, best_uci)
_TT_MAX = 250000
_KILLERS = [[None, None] for _ in range(MAX_PLY + 8)]
_HISTORY = np.zeros((2, 64, 64), dtype=np.int32)
_POS_COUNTS = {}    # transposition_key -> times seen (repetition avoidance)
_NODES = 0
_DEADLINE = 0.0
_TIME_UP = False
_BEST_MOVE = None


class _Timeout(Exception):
    pass


def _check_time():
    # Called every 1024 nodes; raises to unwind search.
    if _TIME_UP:
        raise _Timeout()
    if (_NODES & 1023) == 0 and time.monotonic() >= _DEADLINE:
        raise _Timeout()


def _capa_fast_adjust(board, arr):
    """Cheap Capablanca steering on top of the numba score (white perspective).

    Numpy/array only — no piece_at scans, no attacker loops, safe to run at
    every node: simplify when ahead (trade into winning endgames), own the
    center, castle + develop classically, never rush the queen out early.
    """
    try:
        adj = 0
        # -- simplification: ahead => fewer pieces is better (trade everything)
        # values/units indexed by abs(piece): 0, P, N, B, R, Q, K
        vals = (0, 0, 100, 330, 500, 900, 0)
        units = (0, 0, 1, 1, 2, 4, 0)
        w_nonpawn = 0
        b_nonpawn = 0
        w_units = 0
        b_units = 0
        for v in arr:
            a = int(v)
            if a == 0 or a == 1 or a == -1 or a == 6 or a == -6:
                continue
            if a > 0:
                w_nonpawn += vals[a]
                w_units += units[a]
            else:
                b_nonpawn += vals[-a]
                b_units += units[-a]
        lead = w_nonpawn - b_nonpawn
        total_units = w_units + b_units
        if lead > 150:
            adj += (24 - total_units) * 5   # white ahead: push trades/endgame
        elif lead < -150:
            adj -= (24 - total_units) * 5   # black ahead: same, mirrored
        # -- center: own pawns on d4/e4/d5/e5 (squares 27,28,35,36)
        if arr[27] == 1:
            adj += 10
        elif arr[27] == -1:
            adj -= 10
        if arr[28] == 1:
            adj += 10
        elif arr[28] == -1:
            adj -= 10
        if arr[35] == 1:
            adj += 10
        elif arr[35] == -1:
            adj -= 10
        if arr[36] == 1:
            adj += 10
        elif arr[36] == -1:
            adj -= 10
        # -- classical development + castling (opening only), all from arr
        try:
            ply_no = board.ply()
        except Exception:
            ply_no = 99
        if ply_no < 28:
            # castled = king reached g1/c1 (white) or g8/c8 (black).
            # (has_castled does not exist in python-chess 1.11.2.)
            try:
                wk = int(np.where(arr == 6)[0][0]) if 6 in arr else -1
            except Exception:
                wk = -1
            try:
                bk = int(np.where(arr == -6)[0][0]) if -6 in arr else -1
            except Exception:
                bk = -1
            if wk in (2, 6):
                adj += 30
            if bk in (58, 62):
                adj -= 30
            # knights still home on b1/g1 / b8/g8
            if arr[1] == 2:
                adj -= 8
            if arr[6] == 2:
                adj -= 8
            if arr[57] == -2:
                adj += 8
            if arr[62] == -2:
                adj += 8
            # queen off back rank while a knight still home: premature sortie
            try:
                wq = np.where(arr == 5)[0]
                if len(wq) and int(wq[0]) // 8 != 0 and (arr[1] == 2 or arr[6] == 2):
                    adj -= 22
            except Exception:
                pass
            try:
                bq = np.where(arr == -5)[0]
                if len(bq) and int(bq[0]) // 8 != 7 and (arr[57] == -2 or arr[62] == -2):
                    adj += 22
            except Exception:
                pass
        return adj
    except Exception:
        return 0


def _pawn_attacks_square(board, sq, color):
    """True if a pawn of `color` attacks `sq`. Cheap and precise."""
    try:
        r, f = chess.square_rank(sq), chess.square_file(sq)
        back = r - 1 if color == chess.WHITE else r + 1
        if 0 <= back <= 7:
            for df in (-1, 1):
                nf = f + df
                if 0 <= nf <= 7:
                    pc = board.piece_at(chess.square(nf, back))
                    if pc is not None and pc.piece_type == chess.PAWN and pc.color == color:
                        return True
        return False
    except Exception:
        return False


def _slow_solidity(board):
    """Prophylactic solidity for the side that just moved (call on the
    position AFTER our move, so board.turn is the enemy).

    Preemptive attack counts only SOUND threats: enemy pieces hit by our
    pawns (pawns don't hang in one move), our checks, our pawn pressure on
    the enemy king. A piece "attack" by a hanging piece is a phantom the
    search already refuted — never rewarded.
    Preemptive defense counts genuine hangs: our pieces hit by the enemy
    and uncovered by us, plus real enemy pawn pressure on our king.
    Called only at root among near-equal moves (a handful of positions), so
    attacker maps here cost nothing. Bounded to roughly [-120, +120].
    """
    try:
        me = not board.turn   # side that just moved (us)
        opp = board.turn      # enemy to move
        score = 0
        # sound preemptive attack: enemy heavies/minors hit by OUR PAWNS and
        # uncovered by them — they must move or lose material.
        for sq, pc in board.piece_map().items():
            if pc.piece_type in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
                if pc.color != me:
                    if (_pawn_attacks_square(board, sq, me)
                            and not board.is_attacked_by(opp, sq)):
                        score += {chess.KNIGHT: 25, chess.BISHOP: 25,
                                  chess.ROOK: 35, chess.QUEEN: 50}[pc.piece_type]
                else:
                    if board.is_attacked_by(opp, sq) and not board.is_attacked_by(me, sq):
                        score -= {chess.KNIGHT: 50, chess.BISHOP: 50,
                                  chess.ROOK: 70, chess.QUEEN: 110}[pc.piece_type]
        # giving check is always real preemption (forcing).
        try:
            if board.is_check():
                score += 15
        except Exception:
            pass
        # pawn storms: our pawns leaning on their king zone (good), their
        # pawns leaning on ours (bad). Pawn pressure can't hang, so it's real.
        try:
            my_king = board.king(me)
            if my_king is not None:
                zone = [my_king] + [s for s in chess.SquareSet(chess.BB_KING_ATTACKS[my_king])]
                score -= sum(10 for s in zone if _pawn_attacks_square(board, s, opp))
        except Exception:
            pass
        try:
            ek = board.king(opp)
            if ek is not None:
                zone = [ek] + [s for s in chess.SquareSet(chess.BB_KING_ATTACKS[ek])]
                score += sum(10 for s in zone if _pawn_attacks_square(board, s, me))
        except Exception:
            pass
        return max(-120, min(120, score))
    except Exception:
        return 0


def _static_score(board, arr_cache):
    """Centipawns from side-to-move perspective (Capablanca-steered)."""
    s = _eval_board_array(arr_cache) + _capa_fast_adjust(board, arr_cache)
    if board.turn == chess.BLACK:
        s = -s
    # tempo bonus for side to move
    return s + 8


def _capture_value(board, move):
    """MVV-LVA score for ordering."""
    victim = None
    if board.is_en_passant(move):
        victim = chess.PAWN
    else:
        pc = board.piece_at(move.to_square)
        if pc is not None:
            victim = pc.piece_type
    if victim is None:
        # promotion (non-capture): order high
        if move.promotion:
            return 900 + (move.promotion or 0)
        return 0
    attacker = board.piece_at(move.from_square)
    a = attacker.piece_type if attacker is not None else chess.PAWN
    score = 10 * PIECE_VALUE[victim] - PIECE_VALUE[a]
    if move.promotion:
        score += 900
    return score


def _ordered_moves(board, moves, tt_move_uci, ply):
    scored = []
    kills = _KILLERS[ply] if ply < len(_KILLERS) else (None, None)
    color_idx = 0 if board.turn == chess.WHITE else 1
    # preemptive-defense info, computed once per node (cheap attributes)
    try:
        ksq = board.king(board.turn)
        kfile = chess.square_file(ksq) if ksq is not None else -9
        krank = chess.square_rank(ksq) if ksq is not None else -9
    except Exception:
        kfile, krank = -9, -9
    for m in moves:
        u = m.uci()
        if tt_move_uci is not None and u == tt_move_uci:
            scored.append((1000000, m))
            continue
        if board.is_castling(m):
            # Capablanca: king safety first, always look at castling early
            scored.append((9500, m))
            continue
        if board.is_capture(m) or m.promotion:
            scored.append((10000 + _capture_value(board, m), m))
            continue
        bonus = 0
        if kills[0] == u:
            bonus = 5000
        elif kills[1] == u:
            bonus = 4000
        else:
            try:
                bonus = int(_HISTORY[color_idx, m.from_square, m.to_square]) // 16
                if bonus > 3000:
                    bonus = 3000
            except Exception:
                bonus = 0
        # classical centralization: e4/d4/e5/d5 + developing pieces
        to = m.to_square
        tfile, trank = chess.square_file(to), chess.square_rank(to)
        if 2 <= tfile <= 5 and 2 <= trank <= 5:
            bonus += 120
        try:
            pc = board.piece_at(m.from_square)
            if pc is not None and pc.piece_type in (chess.KNIGHT, chess.BISHOP):
                home = 0 if board.turn == chess.WHITE else 7
                if chess.square_rank(m.from_square) == home:
                    bonus += 200  # develop a new piece
            # knights toward the center (ordering only affects speed, but
            # central moves cut off more, so they buy real depth)
            if pc is not None and pc.piece_type == chess.KNIGHT:
                tf = chess.square_file(m.to_square)
                if tf == 0 or tf == 7:
                    bonus -= 400
                elif 2 <= tf <= 5:
                    bonus += 100
        except Exception:
            pass
        # preemptive defense: never loosen the king's pawn shield cheaply
        try:
            if ksq is not None:
                home_rank = 0 if board.turn == chess.WHITE else 7
                if krank == home_rank and abs(kfile - 4) <= 1:
                    fpc = board.piece_at(m.from_square)
                    if fpc is not None and fpc.piece_type == chess.PAWN:
                        ff = chess.square_file(m.from_square)
                        if abs(ff - kfile) <= 1:
                            bonus -= 1500
        except Exception:
            pass
        scored.append((bonus, m))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [m for _, m in scored]


def _quiescence(board, alpha, beta, ply, arr):
    global _NODES
    _NODES += 1
    _check_time()
    if board.is_checkmate():
        return -MATE + ply
    if board.is_stalemate() or board.is_insufficient_material():
        return 0
    if ply >= MAX_PLY:
        return _static_score(board, arr)
    # In check there is no "quiet" stand pat — every evasion must be
    # searched, otherwise mates and winning checks at the horizon are
    # invisible. Evasions are few, so search them all (no delta pruning).
    if board.is_check():
        try:
            evasions = list(board.legal_moves)
        except Exception:
            return _static_score(board, arr)
        if not evasions:
            return -MATE + ply
        evasions.sort(key=lambda m: _capture_value(board, m), reverse=True)
        for m in evasions:
            board.push(m)
            try:
                narr = _board_to_array(board)
                score = -_quiescence(board, -beta, -alpha, ply + 1, narr)
            finally:
                board.pop()
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        return alpha
    stand = _static_score(board, arr)
    if stand >= beta:
        return beta
    if stand > alpha:
        alpha = stand
    # captures + queen-promotions only
    moves = [m for m in board.legal_moves if board.is_capture(m) or m.promotion]
    moves.sort(key=lambda m: _capture_value(board, m), reverse=True)
    for m in moves:
        # delta pruning: capture can't recover if even max gain + margin fails
        vic = 0
        if board.is_en_passant(m):
            vic = 100
        else:
            pc = board.piece_at(m.to_square)
            if pc is not None:
                vic = PIECE_VALUE.get(pc.piece_type, 0)
        if m.promotion:
            vic += 800
        if stand + vic + 150 < alpha:
            continue
        board.push(m)
        try:
            narr = _board_to_array(board)
            score = -_quiescence(board, -beta, -alpha, ply + 1, narr)
        finally:
            board.pop()
        if score >= beta:
            return beta
        if score > alpha:
            alpha = score
    return alpha


def _negamax(board, depth, alpha, beta, ply, arr, allow_null=True):
    global _NODES, _HISTORY
    _NODES += 1
    _check_time()
    if board.is_checkmate():
        return -MATE + ply
    if board.is_stalemate() or board.is_insufficient_material():
        return 0
    # repetition: position seen twice before -> treat as draw
    try:
        key = board._transposition_key()
    except Exception:
        key = None
    if key is not None and _POS_COUNTS.get(key, 0) >= 2:
        return 0
    if board.halfmove_clock >= 100:
        return 0
    if ply >= MAX_PLY:
        return _static_score(board, arr)
    in_check = board.is_check()
    # check extension
    if in_check:
        depth += 1
    if depth <= 0:
        return _quiescence(board, alpha, beta, ply, arr)
    # TT probe
    tt_move = None
    if key is not None:
        hit = _TT.get(key)
        if hit is not None:
            hdepth, hscore, hflag, hmove = hit
            tt_move = hmove
            if hdepth >= depth:
                if hflag == 0:
                    return hscore
                if hflag == 1 and hscore >= beta:
                    return hscore
                if hflag == 2 and hscore <= alpha:
                    return hscore
    # null-move pruning (not in check, enough material, reasonable depth)
    if allow_null and not in_check and depth >= 3 and ply > 0:
        try:
            # need a non-pawn piece to null move safely (from arr, not scans)
            if board.turn == chess.WHITE:
                has_major = bool(np.any((arr >= 2) & (arr <= 5)))
            else:
                has_major = bool(np.any((arr >= -5) & (arr <= -2)))
        except Exception:
            has_major = True
        if has_major:
            board.push(chess.Move.null())
            try:
                narr = arr  # null move: eval flips via side-to-move anyway
                score = -_negamax(board, depth - 3, -beta, -beta + 1, ply + 1, narr, False)
            finally:
                board.pop()
            if score >= beta:
                return beta
    # move loop with LMR
    try:
        legal = list(board.legal_moves)
    except Exception:
        return _static_score(board, arr)
    if not legal:
        if in_check:
            return -MATE + ply
        return 0
    ordered = _ordered_moves(board, legal, tt_move, ply)
    best_score = -INF
    best_uci = ordered[0].uci()
    best_red = 0  # was the best move's final search depth-reduced (LMR)?
    alpha_orig = alpha
    for i, m in enumerate(ordered):
        was_capture = board.is_capture(m) or m.promotion
        board.push(m)
        narr = _board_to_array(board)
        # late-move reduction: late quiet moves searched shallower
        red = 0
        if i >= 4 and depth >= 3 and not in_check and not was_capture:
            red = 1
        alpha_before = alpha
        try:
            if i == 0:
                score = -_negamax(board, depth - 1, -beta, -alpha, ply + 1, narr)
            else:
                score = -_negamax(board, depth - 1 - red, -alpha - 1, -alpha, ply + 1, narr)
                if red and score > alpha:
                    score = -_negamax(board, depth - 1, -alpha - 1, -alpha, ply + 1, narr)
                if score > alpha and score < beta:
                    score = -_negamax(board, depth - 1, -beta, -alpha, ply + 1, narr)
        finally:
            board.pop()
        if score > best_score:
            best_score = score
            best_uci = m.uci()
            # honest depth: only claim full depth if the final search for
            # this move actually ran at full depth (researched or unreduced)
            best_red = 1 if (red and score <= alpha_before) else 0
        if score > alpha:
            alpha = score
            # update killer/history on quiet cutoffs
            if not was_capture:
                try:
                    if ply < len(_KILLERS):
                        k = _KILLERS[ply]
                        if k[0] != m.uci():
                            k[1] = k[0]
                            k[0] = m.uci()
                    ci = 0 if board.turn == chess.WHITE else 1
                    _HISTORY[ci, m.from_square, m.to_square] += depth * depth
                    if _HISTORY[ci, m.from_square, m.to_square] > 100000:
                        _HISTORY[ci] //= 2
                except Exception:
                    pass
        if alpha >= beta:
            break
    # TT store (honest depth: a reduced-only search is stored as the
    # depth it actually ran, so it can never masquerade as a full-depth
    # result and poison later cutoffs with a weakened bound).
    if key is not None:
        try:
            if len(_TT) > _TT_MAX:
                _TT.clear()
            if best_score <= alpha_orig:
                flag = 2
            elif best_score >= beta:
                flag = 1
            else:
                flag = 0
            _TT[key] = (depth - best_red, best_score, flag, best_uci)
        except Exception:
            pass
    return best_score


# --------------------------------------------------------------------------
# Time management + root search
# --------------------------------------------------------------------------

def _time_budget_ms(time_left_ms, move_number):
    """Conservative but not passive budget for 120s+0.5s."""
    try:
        tl = max(0, int(time_left_ms))
    except Exception:
        tl = 1000
    inc = 500
    if tl <= 800:
        return 60
    if tl <= 2000:
        return max(60, tl // 6)
    # base: ~1/32 of remaining + part of increment, clamped.
    # Conservative on purpose: match CPU (EPYC 2.6GHz) is slower than dev
    # machines, and iterative deepening already uses whatever it is given.
    base = tl // 32 + 300
    if move_number < 12:
        base = min(base, 2000)  # don't burn clock in curated openings
    else:
        base = min(base, 3200)
    # keep reserve: never spend more than 1/4 of remaining (minus safety)
    cap = max(100, tl // 4 - 120)
    return max(80, min(base, cap))


def _search_root(board, deadline, max_depth):
    """Iterative deepening. Returns (best_uci, best_score)."""
    global _NODES, _DEADLINE, _TIME_UP
    legal = list(board.legal_moves)
    if not legal:
        return None, -MATE
    if len(legal) == 1:
        return legal[0].uci(), 0
    # order root by quick static guess + TT move
    try:
        rkey = board._transposition_key()
    except Exception:
        rkey = None
    tt_move = None
    if rkey is not None and rkey in _TT:
        try:
            tt_move = _TT[rkey][3]
        except Exception:
            tt_move = None
    ordered = _ordered_moves(board, legal, tt_move, 0)
    best = ordered[0].uci()
    best_score = 0
    arr0 = _board_to_array(board)
    # root scores from the latest completed depth: used for the
    # Capablanca solidity tie-break among near-equal moves.
    root_scores = {}
    # depth 1 first for instant move
    for depth in range(1, max_depth + 1):
        alpha = -INF
        beta = INF
        local_best = best
        local_score = best_score
        try:
            for i, m in enumerate(ordered):
                board.push(m)
                narr = _board_to_array(board)
                if i == 0:
                    s = -_negamax(board, depth - 1, -beta, -alpha, 1, narr)
                else:
                    s = -_negamax(board, depth - 1, -alpha - 1, -alpha, 1, narr)
                    if s > alpha and s < beta:
                        s = -_negamax(board, depth - 1, -beta, -alpha, 1, narr)
                board.pop()
                root_scores[m.uci()] = s
                if s > local_score or i == 0:
                    local_score = s
                    local_best = m.uci()
                if s > alpha:
                    alpha = s
            # completed depth: commit + reorder for next depth
            best, best_score = local_best, local_score
            # move best to front
            ordered.sort(key=lambda mm: 0 if mm.uci() == best else 1)
            # early exit on forced mate found
            if abs(best_score) > MATE - 50:
                break
            if time.monotonic() >= deadline:
                break
        except _Timeout:
            # restore board stack safety: pop any leftover pushes
            while True:
                try:
                    if len(board.move_stack) <= _ROOT_STACK_LEN:
                        break
                    board.pop()
                except Exception:
                    break
            break
        except Exception:
            try:
                while len(board.move_stack) > _ROOT_STACK_LEN:
                    board.pop()
            except Exception:
                pass
            break
    # Super-Capablanca finish: PVS null-window bounds are coarse — several
    # moves can share the same bound while their true scores differ by
    # hundreds. So first VERIFY the near-leaders with a full-window
    # re-search (cheap at low depth, and 1-ply refutations show up
    # immediately), then among truly near-equal moves bank forcing moves
    # first — castle > capture/check > quiet — then the most prophylactically
    # solid (development, pawn pressure, genuine hangs). `force` outranks
    # capped `sol`, and both sit far below the verified score (counts 20x),
    # so they only break genuine near-ties; the rest falls back to search
    # order (principled), never to anything arbitrary.
    # Super-Capablanca finish: PVS null-window bounds are coarse — several
    # moves can share the same bound while their true scores differ by
    # hundreds. So first VERIFY the near-leaders with a full-window
    # re-search (cheap at low depth, and 1-ply refutations show up
    # immediately), then among truly near-equal moves bank forcing moves
    # first — castle > capture/check > quiet — then the most prophylactically
    # solid (development, pawn pressure, genuine hangs). `force` outranks
    # capped `sol`, and both sit far below the verified score (counts 20x),
    # so they only break genuine near-ties. The search-best is always an
    # eligible winner (seeded first): verification can only dethrone it
    # with a strictly better verified key, never by timeout attrition.
    def _force_sol(u):
        m = by_uci.get(u)
        if m is None:
            return 0, 0
        # forcing moves first: a proven tactic banked now never
        # evaporates over the horizon; threatening it again next
        # move is strictly worse. `force` outranks `sol` so a
        # lingering threat bonus can never beat the capture.
        force = 0
        if board.is_castling(m):
            force = 60
        elif board.is_capture(m):
            force = 50
        else:
            try:
                board.push(m)
                gave_check = board.is_check()
                board.pop()
                if gave_check:
                    force = 50
            except Exception:
                try:
                    board.pop()
                except Exception:
                    pass
        sol = 0
        try:
            pc = board.piece_at(m.from_square)
            if pc is not None and pc.piece_type in (chess.KNIGHT, chess.BISHOP):
                home = 0 if board.turn == chess.WHITE else 7
                if chess.square_rank(m.from_square) == home:
                    sol += 10
            # knights belong in the center, never on the rim:
            # Na6/Nh6-style moves lost us a rated game.
            if pc is not None and pc.piece_type == chess.KNIGHT:
                tf = chess.square_file(m.to_square)
                if tf == 0 or tf == 7:
                    sol -= 12
                elif 2 <= tf <= 5:
                    sol += 6
        except Exception:
            pass
        try:
            board.push(m)
            sol += _slow_solidity(board)
            board.pop()
        except Exception:
            try:
                board.pop()
            except Exception:
                pass
        if sol > 25:
            sol = 25
        elif sol < -25:
            sol = -25
        return force, sol

    try:
        if root_scores and abs(best_score) < MATE - 100:
            by_uci = {m.uci(): m for m in legal}
            order_idx = {m.uci(): i for i, m in enumerate(ordered)}
            # seed the decision with the search-best itself: verification
            # may only replace it with something strictly better.
            _bf, _bs = _force_sol(best)
            best_key = (root_scores.get(best, best_score) * 20 + _bf + _bs,
                        -order_idx.get(best, 999))
            best_cand = best
            # wide net on coarse bounds, then verify exactly (cap cost).
            # Search-best goes first so scarce time verifies it first; if
            # the clock is already gone, verify nothing else.
            prelim = [u for u, s in root_scores.items()
                      if s >= best_score - 60 and s > -8000 and u in by_uci
                      and u != best]
            prelim.sort(key=lambda u: (root_scores[u], -order_idx.get(u, 999)),
                        reverse=True)
            prelim = [best] + prelim
            if time.monotonic() >= deadline:
                prelim = prelim[:1]
            verified = {}
            # vdepth=1 + quiescence: exposes 1-ply refutations (hanging
            # pieces), which is all the coarse PVS bounds can hide, at a
            # few ms per candidate. Deeper tactics were already settled by
            # the main search.
            vdepth = 1
            for u in prelim[:8]:
                m = by_uci[u]
                board.push(m)
                try:
                    narr = _board_to_array(board)
                    s = -_negamax(board, vdepth, -INF, INF, 1, narr)
                except _Timeout:
                    try:
                        while len(board.move_stack) > _ROOT_STACK_LEN:
                            board.pop()
                    except Exception:
                        pass
                    break
                except Exception:
                    s = root_scores[u]
                finally:
                    try:
                        while len(board.move_stack) > _ROOT_STACK_LEN:
                            board.pop()
                    except Exception:
                        pass
                verified[u] = s
                if time.monotonic() >= deadline:
                    break
            if verified:
                vbest = max(verified.values())
                cands = [u for u, s in verified.items() if s >= vbest - 4]
                for u in cands:
                    if u == best_cand and u == best:
                        continue
                    force, sol = _force_sol(u)
                    key = (verified[u] * 20 + force + sol, -order_idx.get(u, 999))
                    if key > best_key:
                        best_key = key
                        best_cand = u
                if best_cand != best:
                    best = best_cand
                    best_score = verified[best]
    except Exception:
        pass
    return best, best_score


_ROOT_STACK_LEN = 0


def _repetition_penalty(board, move):
    """Return adjustment (centipawns, from our perspective) for repetition risk.

    If we are ahead and the move repeats a position -> strongly avoid.
    If behind and it repeats -> allow (opponent may avoid).
    """
    try:
        board.push(move)
        try:
            k = board._transposition_key()
        except Exception:
            k = None
        board.pop()
        if k is None:
            return 0
        cnt = _POS_COUNTS.get(k, 0)
        if cnt < 1:
            return 0
        # quick material-based "am I ahead" using eval sign
        arr = _board_to_array(board)
        s = _eval_board_array(arr)
        mine_ahead = (s > 40 and board.turn == chess.WHITE) or (s < -40 and board.turn == chess.BLACK)
        if cnt >= 2:
            return -500 if mine_ahead else 0  # third occurrence = auto draw
        if cnt == 1:
            return -120 if mine_ahead else 20
        return 0
    except Exception:
        return 0


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def _finalize_move(fen, candidate_uci, safe_uci):
    """Last line of defense: validate against a FRESH board parsed from the
    asked FEN, so no internal state corruption can ever leak an illegal
    move. Returns a move that is legal in `fen`, guaranteed unless the
    position itself has no legal moves (mate/stalemate — game already over).
    """
    try:
        fresh = chess.Board(fen)
        fresh_legal = list(fresh.legal_moves)
        if not fresh_legal:
            return "0000"
        try:
            if candidate_uci and chess.Move.from_uci(candidate_uci) in fresh_legal:
                return candidate_uci
        except Exception:
            pass
        try:
            if safe_uci and chess.Move.from_uci(safe_uci) in fresh_legal:
                return safe_uci
        except Exception:
            pass
        return fresh_legal[0].uci()
    except Exception:
        # fen itself unparseable — nothing legal exists to return
        try:
            return safe_uci or "0000"
        except Exception:
            return "0000"


def get_move(fen: str, time_left_ms: int) -> str:
    """Return a legal UCI move for the side to move in `fen`."""
    global _DEADLINE, _TIME_UP, _NODES, _ROOT_STACK_LEN
    try:
        board = chess.Board(fen)
    except Exception:
        return "0000"
    try:
        legal = list(board.legal_moves)
    except Exception:
        return "0000"
    if not legal:
        # No legal moves (mate/stalemate): referee ends game.
        return "0000"
    # safe_move: captured up front, legal by construction. Every fallback
    # below returns this (or a validated better move) — never a hardcoded
    # string that could be illegal in this position.
    try:
        safe_move = legal[0].uci()
    except Exception:
        return "0000"
    if len(legal) == 1:
        u = legal[0].uci()
        try:
            _POS_COUNTS[board._transposition_key()] = _POS_COUNTS.get(board._transposition_key(), 0) + 1
        except Exception:
            pass
        return _finalize_move(fen, u, u)
    # track repetition (positions we are asked about, from game start)
    try:
        rk = board._transposition_key()
        _POS_COUNTS[rk] = _POS_COUNTS.get(rk, 0) + 1
    except Exception:
        rk = None
    move_number = board.ply() // 2 + 1
    budget_ms = _time_budget_ms(time_left_ms, move_number)
    # max depth by time: more time -> deeper; hard cap for safety
    if budget_ms < 150:
        max_depth = 2
    elif budget_ms < 400:
        max_depth = 3
    elif budget_ms < 900:
        max_depth = 4
    elif budget_ms < 1800:
        max_depth = 5
    else:
        max_depth = 7
    # endgame with few pieces: search deeper (branching is low)
    try:
        if len(board.piece_map()) <= 10:
            max_depth = min(9, max_depth + 1)
    except Exception:
        pass
    _ROOT_STACK_LEN = len(board.move_stack)
    _NODES = 0
    _TIME_UP = False
    safety = 0.12  # 120ms safety margin for wall-time watchdog
    deadline = time.monotonic() + max(0.05, budget_ms / 1000.0 - safety)
    _DEADLINE = deadline
    # safety: absolute clamp so we never overrun 1/3 of remaining clock
    try:
        hard_cap = time.monotonic() + max(0.05, (int(time_left_ms) / 1000.0) / 3)
        if deadline > hard_cap:
            deadline = hard_cap
            _DEADLINE = deadline
    except Exception:
        pass
    try:
        best_uci, score = _search_root(board, deadline, max_depth)
    except Exception:
        best_uci, score = None, 0
    # fallback chain, best-first: search result -> greedy capture -> safe.
    # Nothing is returned yet — everything passes _finalize_move below.
    if not best_uci:
        try:
            best_uci = max(legal, key=lambda m: _capture_value(board, m)).uci()
        except Exception:
            best_uci = safe_move
    # repetition avoidance: if best repeats while ahead and an alternative
    # is close, switch. Re-verify legality first.
    try:
        mv = chess.Move.from_uci(best_uci)
        if mv not in legal:
            best_uci = safe_move
        else:
            pen = _repetition_penalty(board, mv)
            if pen <= -100 and abs(score) < 800:
                # look for best non-repeating alternative at depth 1 + quiescence
                alt_best, alt_score = best_uci, score
                for m in legal:
                    if m.uci() == best_uci:
                        continue
                    if _repetition_penalty(board, m) <= -100:
                        continue
                    board.push(m)
                    try:
                        narr = _board_to_array(board)
                        s = -_quiescence(board, -INF, INF, 1, narr)
                    finally:
                        board.pop()
                    # s is from the post-move (opponent) perspective, negated
                    # back to root perspective by the leading minus.
                    if s > alt_score - 30:  # close enough -> prefer non-draw
                        alt_best, alt_score = m.uci(), s
                        break
                best_uci = alt_best
    except Exception:
        best_uci = safe_move
    # FINAL GUARANTEE: re-validate on a fresh parse of the asked position.
    return _finalize_move(fen, best_uci, safe_move)


# --------------------------------------------------------------------------
# Import-time warmup (runs inside 90s init budget, off the match clock).
# --------------------------------------------------------------------------
try:
    _w = np.zeros(64, dtype=np.int8)
    _w[12] = 1
    _w[52] = -1
    _w[60] = -6
    _w[4] = 6
    _eval_board_array(_w)
    _wb = chess.Board()
    _wa = _board_to_array(_wb)
    _evaluate_nb(_wa, MG_P, MG_N, MG_B, MG_R, MG_Q, MG_K, EG_K, PASSED_BONUS) if numba is not None else None
    # tiny search warmup (exercises ordering/TT code paths)
    _DEADLINE = time.monotonic() + 0.2
    try:
        _quiescence(_wb, -INF, INF, 0, _wa)
    except Exception:
        pass
except Exception:
    pass
finally:
    _NODES = 0
    _TIME_UP = False
