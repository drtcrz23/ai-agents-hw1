# 8-Puzzle Solver Experiment Report

## Approach

The experiment implements an optimal 8-puzzle solver using A* search with the Manhattan distance heuristic. The algorithm:

1. **Representation**: Board is a tuple of 9 integers (row-major order), where `0` represents the blank tile. Goal state is `(1, 2, 3, 4, 5, 6, 7, 8, 0)`.

2. **Move semantics**: Each move character (`U`, `D`, `L`, `R`) indicates the direction the BLANK moves.

3. **A* Search Algorithm**:
   - Priority queue stores tuples of `(f_score, counter, current_board, path)`
   - `f_score = g_score + h_score` where:
     - `g_score` = number of moves taken to reach current state
     - `h_score` = Manhattan distance heuristic (sum of distances each tile is from its goal position)
   - Visited dictionary tracks best f-score for each board state to avoid re-expanding

4. **Solvability Check**: Uses inversion count (ignoring blank). For 3x3 puzzle, even inversions = solvable.

5. **Verification**: Independently applies each move in the solution path and checks:
   - Each move is legal (blank stays within grid)
   - Final board equals GOAL state

## Algorithm Details

### Manhattan Distance Heuristic
For each tile `v` (where v ≠ 0):
- Current position: `(row, col)` = `(i // 3, i % 3)` where `i` is index
- Goal position: `(goal_row, goal_col)` = `((v-1) // 3, (v-1) % 3)`
- Distance contribution: `|row - goal_row| + |col - goal_col|`

### Move Generation
From blank at `(row, col)`:
- `U`: new position `(row-1, col)` if row > 0
- `D`: new position `(row+1, col)` if row < 2
- `L`: new position `(row, col-1)` if col > 0
- `R`: new position `(row, col+1)` if col < 2

## Metric Values

| Metric | Value | Expected Range | Status |
|--------|-------|----------------|--------|
| avg_moves | 22.33 | [20.0, 24.0] | PASS |
| solved_rate | 1.0 | Must be 1.0 | PASS |

## Errors Fixed During Development

1. **Initial bug in `is_valid_move` function**: The first version had a `NameError` because it tried to use `board` variable that wasn't passed correctly. Fixed by restructuring the function to accept `current_board` as parameter and properly reconstruct the new board state.

2. **Time measurement issue**: The initial time_ms values were extremely large (e.g., 1795609054005) because I was using `time.time()` which returns seconds since epoch, not the elapsed solve time. Fixed by measuring wall-clock time around the `solve()` call with `time.perf_counter()` and writing the per-board elapsed milliseconds.

3. **Sanity check verification**: The sanity checks for `solve(GOAL) == []` and `solve((1,2,3,4,5,6,7,0,8)) == ['R']` were implemented correctly after fixing the move generation logic.

## Conclusion

The experiment successfully solves 100 random solvable 8-puzzle boards with an average optimal solution length of 22.33 moves (within expected range [20, 24]) and achieves 100% verification success rate. The A* algorithm with Manhattan distance heuristic provides optimal solutions for all generated puzzles.
