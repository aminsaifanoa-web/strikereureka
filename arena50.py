"""50-round tournament-format arena: full 120s+0.5s clock, curated openings,
alternating colours, seeded random opposition, referee rules. Appends one
line per game to arena50.log and prints a final tally."""

import os
import random
import sys
import time

import chess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agent  # noqa: E402

OPENINGS = (
    ("English", "r1bqk2r/pp1pppbp/2n2np1/2p5/2P5/2N1PNP1/PP1P1PBP/R1BQK2R b KQkq - 0 6"),
    ("French Winawer", "r1b1k2r/pp2nppp/2n1p3/q1ppP3/P2P4/2P2N2/2PB1PPP/R2QKB1R b KQkq - 4 9"),
    ("Petroff", "rnbqkb1r/pp3ppp/2p5/1B1p4/3Pn3/5N2/PPP2PPP/RNBQK2R w KQkq - 0 7"),
    ("Scotch", "r1bq1rk1/pppp1ppp/2n2n2/1Bb5/3NP3/2P5/PP3PPP/RNBQ1RK1 w - - 3 8"),
    ("Grunfeld", "rnbqk2r/pp2ppbp/6p1/2p5/3PP3/2P1BN2/P4PPP/R2QKB1R b KQkq - 1 8"),
    ("French Classical", "r1bqk2r/pp1n1ppp/2n1p3/2bpP3/5P2/2NB4/PPP3PP/R1BQK1NR w KQkq - 0 8"),
    ("Sicilian Closed", "1rbqk1nr/pp2ppbp/2np2p1/2p5/P3P3/2NP2P1/1PP1NPBP/R1BQK2R b KQk - 0 7"),
    ("Sveshnikov", "r1bqkb1r/pp3ppp/2np4/1N1Pp3/8/8/PPP2PPP/R1BQKB1R b KQkq - 0 8"),
)

BASE_MS = 120000
INC_MS = 500
LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "arena50.log")


def play_game(start_fen, ours_white, seed):
    rnd = random.Random(seed)
    board = chess.Board(start_fen)
    clocks = {chess.WHITE: float(BASE_MS), chess.BLACK: float(BASE_MS)}
    plies = 0
    while plies < 600:
        if board.is_checkmate():
            winner_is_us = (board.turn == (not ours_white)) == True and (
                (board.turn == chess.BLACK) == ours_white)
            # side to move got mated -> other side won
            us_won = ((board.turn == chess.BLACK) == ours_white)
            return ("win" if us_won else "loss", "checkmate", plies)
        if (board.is_stalemate() or board.is_insufficient_material()
                or board.is_seventyfive_moves()
                or board.is_fivefold_repetition()):
            return ("draw", "draw-rule", plies)
        ours = (board.turn == chess.WHITE) == ours_white
        if ours:
            side = board.turn
            t0 = time.monotonic()
            try:
                mv_uci = agent.get_move(board.fen(), int(clocks[side]))
            except Exception:
                return ("loss", "crash", plies)
            dt = (time.monotonic() - t0) * 1000.0
            clocks[side] = clocks[side] - dt + INC_MS
            if clocks[side] < -500:
                return ("loss", "flag", plies)
            try:
                mv = chess.Move.from_uci(mv_uci)
            except Exception:
                return ("loss", "illegal", plies)
            if mv not in board.legal_moves:
                return ("loss", "illegal", plies)
            board.push(mv)
        else:
            legal = list(board.legal_moves)
            if not legal:
                return ("win", "opp-no-move", plies)
            board.push(rnd.choice(legal))
        plies += 1
    return ("draw", "ply-cap", plies)


def main():
    wins = draws = losses = 0
    done = set()
    if os.path.exists(LOG):
        with open(LOG) as f:
            for line in f:
                parts = line.split()
                # line form: "<i> <W/B> <opening words...> <result> <end> <plies>"
                if parts and parts[0].isdigit() and len(parts) >= 5:
                    done.add(int(parts[0]))
                    res = parts[-3]
                    if res == "win":
                        wins += 1
                    elif res == "draw":
                        draws += 1
                    else:
                        losses += 1
    mode = "a" if done else "w"
    with open(LOG, mode) as f:
        if not done:
            f.write("game colour opening result end plies\n")
        for i in range(50):
            if i in done:
                continue
            name, fen = OPENINGS[i % len(OPENINGS)]
            ours_white = (i % 2 == 0)
            res, end, pl = play_game(fen, ours_white, 5000 + i)
            if res == "win":
                wins += 1
            elif res == "draw":
                draws += 1
            else:
                losses += 1
            f.write(f"{i} {'W' if ours_white else 'B'} {name} {res} {end} {pl}\n")
            f.flush()
            print(f"game {i}: {'W' if ours_white else 'B'} {name} "
                  f"{res} ({end}, {pl} plies) [{wins}W-{draws}D-{losses}L]",
                  flush=True)
        f.write(f"TOTAL {wins}W-{draws}D-{losses}L\n")
    print(f"FINAL: {wins}W-{draws}D-{losses}L", flush=True)


if __name__ == "__main__":
    main()
