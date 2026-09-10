"""IslandChess local simulator (dev tooling — NOT part of the submission).

Referee-style games on this machine: legal-move enforcement for both
sides, wall-clock accounting with increment, draw claims like the
tournament referee, plus an edge-case suite. Opponents: seeded random
mover (stupid but always legal) and Stockfish at a capped Elo.

Usage:
    python simulator.py            # edge suite + 4 random games
    python simulator.py --sf       # + 2 games vs Stockfish Elo 2400
    python simulator.py --sf-only  # only the Stockfish games

Stockfish binary path is taken from STOCKFISH_EXE, defaulting to the
copy used during development.
"""

import os
import random
import sys
import time

import chess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agent  # noqa: E402  (the submission under test)

SF_EXE = os.environ.get(
    "STOCKFISH_EXE",
    r"C:\Users\amins\AppData\Local\Temp\opencode\sf\stockfish\stockfish-windows-x86-64-universal.exe",
)
BASE_MS = 15000
INC_MS = 500
MAX_PLIES = 600
FLAG_GRACE_MS = 500


def random_mover(seed):
    rnd = random.Random(seed)

    def fn(fen, clock_ms):
        board = chess.Board(fen)
        legal = list(board.legal_moves)
        if not legal:
            return "0000"
        return rnd.choice(legal).uci()

    return fn


def island_mover():
    def fn(fen, clock_ms):
        return agent.get_move(fen, int(clock_ms))

    return fn


def stockfish_mover(exe=SF_EXE, elo=2400, depth=None):
    import chess.engine

    eng = chess.engine.SimpleEngine.popen_uci(exe)
    if elo is not None:
        eng.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})
    eng.configure({"Threads": 1, "Hash": 64})

    def fn(fen, clock_ms):
        board = chess.Board(fen)
        if depth is not None:
            res = eng.play(board, chess.engine.Limit(depth=depth))
        else:
            res = eng.play(
                board,
                chess.engine.Limit(
                    white_clock=clock_ms / 1000.0,
                    black_clock=clock_ms / 1000.0,
                    white_inc=INC_MS / 1000.0,
                    black_inc=INC_MS / 1000.0,
                ),
            )
        return res.move.uci()

    fn.close = eng.quit
    return fn


def run_game(white_fn, black_fn, base_ms=BASE_MS, inc_ms=INC_MS,
             start_fen=chess.STARTING_FEN, max_plies=MAX_PLIES):
    """Play one referee-style game. Returns a result dict."""
    board = chess.Board(start_fen)
    clocks = {chess.WHITE: float(base_ms), chess.BLACK: float(base_ms)}
    fns = {chess.WHITE: white_fn, chess.BLACK: black_fn}
    names = {chess.WHITE: "white", chess.BLACK: "black"}
    plies = 0
    illegals = 0
    while plies < max_plies:
        if board.is_checkmate():
            winner = "black" if board.turn == chess.WHITE else "white"
            return {"end": "checkmate", "winner": winner, "plies": plies,
                    "fen": board.fen(), "illegals": illegals,
                    "clocks": dict(clocks)}
        if (board.is_stalemate() or board.is_insufficient_material()
                or board.is_seventyfive_moves()
                or board.is_fivefold_repetition()):
            return {"end": "draw-rule", "winner": "draw", "plies": plies,
                    "fen": board.fen(), "illegals": illegals,
                    "clocks": dict(clocks)}
        side = board.turn
        try:
            t0 = time.monotonic()
            mv_uci = fns[side](board.fen(), clocks[side])
            dt_ms = (time.monotonic() - t0) * 1000.0
        except Exception as exc:  # crash
            return {"end": "crash", "winner": "draw" if False else (
                "black" if side == chess.WHITE else "white"),
                "plies": plies, "fen": board.fen(), "illegals": illegals,
                "clocks": dict(clocks), "detail": repr(exc)}
        clocks[side] = clocks[side] - dt_ms + inc_ms
        if clocks[side] < -FLAG_GRACE_MS:
            winner = "black" if side == chess.WHITE else "white"
            return {"end": "flag", "winner": winner, "plies": plies,
                    "fen": board.fen(), "illegals": illegals,
                    "clocks": dict(clocks)}
        try:
            mv = chess.Move.from_uci(mv_uci)
        except Exception:
            mv = None
        if mv is None or mv not in board.legal_moves:
            illegals += 1
            winner = "black" if side == chess.WHITE else "white"
            return {"end": "illegal", "winner": winner, "plies": plies,
                    "fen": board.fen(), "illegals": illegals,
                    "clocks": dict(clocks), "detail": mv_uci}
        board.push(mv)
        plies += 1
    return {"end": "ply-cap", "winner": "draw", "plies": plies,
            "fen": board.fen(), "illegals": illegals, "clocks": dict(clocks)}


EDGE_CASES = [
    ("startpos", chess.STARTING_FEN, 120000),
    ("promotion-queen", "8/5P1k/8/8/8/8/6K1/8 w - - 0 1", 10000),
    ("promotion-knight-needed", "6k1/5P1p/6K1/8/8/8/8/8 w - - 0 1", 10000),
    ("stalemate-avoid", "k7/1P6/2K5/8/8/8/8/8 w - - 0 1", 10000),
    ("en-passant", "rnbqkbnr/1pp1pppp/p7/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3", 10000),
    ("en-passant-pinned", "8/8/8/3pP1k1/8/8/5K2/8 w - d6 0 1", 10000),
    ("castle-both", "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1", 10000),
    ("no-castle-through-check", "r3k2r/pppppqpp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1", 10000),
    ("mated-no-moves", "7k/5Q2/6p1/6pp/8/8/5PPP/5RK1 b - - 0 1", 10000),
    ("bare-kings", "8/8/4k3/8/8/4K3/8/8 w - - 0 1", 10000),
    ("fifty-move-clock", "8/8/4k3/8/8/4K3/4P3/8 w - - 100 60", 10000),
    ("tiny-clock-300ms", "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1", 300),
    ("tiny-clock-800ms", "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1", 800),
    ("mate-in-1", "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 1", 10000),
    ("curated-petroff", "rnbqkb1r/ppp2ppp/8/3p4/3Pn3/5N2/PPP1BPPP/RNBQK2R b KQkq - 1 6", 120000),
]


def edge_suite():
    print("== edge suite ==", flush=True)
    ok = True
    for name, fen, clk in EDGE_CASES:
        try:
            m = agent.get_move(fen, clk)
        except Exception as exc:
            print(f"{name}: CRASH {exc!r}", flush=True)
            ok = False
            continue
        board = chess.Board(fen)
        legal = list(board.legal_moves)
        good = (m == "0000" and not legal) or (
            m != "0000" and chess.Move.from_uci(m) in legal)
        # stalemate discipline: never stalemate a winning position volunarily
        extra = ""
        if good and m != "0000" and legal:
            nxt = chess.Board(fen)
            nxt.push(chess.Move.from_uci(m))
            if not nxt.is_checkmate() and not list(nxt.legal_moves):
                extra = " STALEMATED-OPPONENT!"
                good = False
        ok = ok and good
        print(f"{name}: {m} legal={good}{extra}", flush=True)
    print("EDGE:", "ALL OK" if ok else "FAILURES", flush=True)
    return ok


def main():
    args = set(sys.argv[1:])
    ok = edge_suite()
    if "--sf-only" not in args:
        print("== random games (stupid but legal) ==", flush=True)
        for i in range(4):
            ours_white = (i % 2 == 0)
            if ours_white:
                r = run_game(island_mover(), random_mover(900 + i))
            else:
                r = run_game(random_mover(900 + i), island_mover())
            us = "W" if ours_white else "B"
            print(f"random{i} us={us}: {r['end']} winner={r['winner']} "
                  f"plies={r['plies']} illegals={r['illegals']}", flush=True)
            ok = ok and r["end"] not in ("illegal", "crash", "flag")
    if "--sf" in args or "--sf-only" in args:
        print("== Stockfish Elo 2400 ==", flush=True)
        try:
            for i, ours_white in enumerate([True, False]):
                sf = stockfish_mover(elo=2400)
                try:
                    if ours_white:
                        r = run_game(island_mover(), sf, base_ms=30000)
                    else:
                        r = run_game(sf, island_mover(), base_ms=30000)
                finally:
                    sf.close()
                us = "W" if ours_white else "B"
                print(f"sf2400-{us}: {r['end']} winner={r['winner']} "
                      f"plies={r['plies']} illegals={r['illegals']}", flush=True)
                ok = ok and r["end"] not in ("illegal", "crash", "flag")
        except Exception as exc:
            print(f"stockfish unavailable: {exc!r}", flush=True)
    print("SIMULATOR:", "ALL OK" if ok else "FAILURES PRESENT", flush=True)


if __name__ == "__main__":
    main()
