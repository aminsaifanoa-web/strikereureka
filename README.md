# IslandChess — super-Capablanca

Tournament chess agent for AI Chessathon. Single-file submission: `agent.py`.

## Style: one method to a crazy extreme
Capablanca's. Flawless pawn structure (doubled/isolated harshly punished),
classical development (knights out before queen sorties), king safety
first (castle early, never loosen the shield), simplification when ahead
(trade into winning endgames), open files for rooks, supported outposts
for knights, connected passed pawns pushed — plus:

- **Preemptive attack:** sound threats only — pawns hitting uncovered
  pieces, checks, pawn storms on the enemy king. Phantom attacks by
  hanging pieces are never rewarded (the search already refuted them).
- **Preemptive defense:** genuine hangs penalized, king pawn shield
  guarded in ordering (no cheap f/g-pawn loosening), near-ties broken
  toward the solid developing move, never an arbitrary one.

## What it is
Single-threaded iterative-deepening negamax with alpha-beta, quiescence
(captures + promotions), transposition table persistent across moves,
MVV-LVA + killer + history ordering, null-move pruning, late-move
reduction, check extension, tapered material + piece-square eval
(numba-jitted), repetition avoidance, wall-clock time management for
120s + 0.5s.

Uses only the allowed stack: `python-chess`, `numpy`, `numba`, stdlib.
No third-party engine, no published network, no binaries.

## Layout (submission zip has agent.py at root)
```
islandchess/
  agent.py          submission entrypoint: get_move(fen, time_left_ms) -> uci
  README.md
  submission.zip    built artifact (agent.py at root)
```

## Local test
```
python -c "import sys; sys.path.insert(0,'.'); import agent; print(agent.get_move('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1', 120000))"
```

## Build zip (agent.py at root, not inside a folder)
```
python -m zipfile -c submission.zip agent.py
```

## Notes
- Import warms numba (~9s, inside 90s init budget, off match clock).
- One thread only. State (TT, repetition counts) persists across moves
  in-process, reset every game by fresh process.
- Time budget ~1/28 remaining + increment, 120ms safety margin, hard cap
  1/3 remaining. Depth 2-8 (10 in sparse endgames). Always returns a
  legal move; never flags on tested positions.
