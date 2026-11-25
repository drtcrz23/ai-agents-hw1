# Task: optimal 8-puzzle solver experiment

Study the classic 8-puzzle (sliding puzzle on a 3x3 grid) by writing `experiment.py`.

## Representation (use EXACTLY this everywhere)

- Board: a tuple of 9 ints, row-major, `0` = blank.
  Goal: `(1, 2, 3, 4, 5, 6, 7, 8, 0)`.
- Move: one char `'U' | 'D' | 'L' | 'R'` — the direction the BLANK moves.
- `solve(start)` returns the optimal solution as a list of move chars,
  `[]` when `start == GOAL`, `None` when the board is unsolvable.

## Requirements

1. Sanity checks BEFORE the main loop. On failure print
   `SANITY FAILED: <reason>` and exit with code 1:
   - `solve(GOAL) == []`
   - `solve((1, 2, 3, 4, 5, 6, 7, 0, 8)) == ['R']`
2. Generate exactly 100 random solvable boards with `random.Random(42)`:
   shuffle `list(range(9))`, keep a board only if its inversion count
   (ignoring the blank) is even, otherwise resample.
3. Solve each board optimally with A* + the Manhattan distance heuristic
   (`heapq` from the stdlib for the priority queue).
4. Verify each solution INDEPENDENTLY of the solver:
   `verify(start, moves)` applies each move char to the board and checks each
   move is legal (the blank stays inside the 3x3 grid); it returns True only
   if the final board equals GOAL. If any solution fails verification, print
   an error and exit with code 1.
5. Write progress to `train.log` (open in write mode), one line per solved
   puzzle, `puzzle i/100 moves=<len> time_ms=<t>` with `i` running from 1
   to 100, where `<t>` is the wall-clock time in MILLISECONDS the solver
   spent on THAT board (measure the time immediately before and after the
   `solve()` call — do NOT print a Unix timestamp). Then write ONE final
   summary line as the LAST line of `train.log`, exactly of this form:
   `Summary solved=<n>/100 avg_moves=<x>`
   where `<n>` is the number of boards solved and verified, and `<x>` is
   the mean solution length (the same value as in the METRIC line).
6. At the end print EXACTLY these two lines to stdout:
   ```
   METRIC avg_moves <float>
   METRIC solved_rate <float>
   ```
   where avg_moves is the mean optimal solution length over the 100 boards,
   and solved_rate is the fraction of boards that were solved AND passed
   verification — compute it as `verified / 100` from a counter incremented
   in the main loop (do NOT hardcode the value).
7. Stdlib only. The whole run must finish well within 120 seconds.

## Notes from the task author

- If `solve` returns `None` for a solvable board, the bug is in the goal
  check or in move generation — write a tiny debug script that prints the
  states A* expands for one small board, run it, and fix that specific bug.
  Do NOT rewrite the whole `experiment.py` when one function is broken.