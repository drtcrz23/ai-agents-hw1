#!/usr/bin/env python3
"""
Optimal 8-puzzle solver experiment using A* with Manhattan distance heuristic.
"""

import heapq
import random
import time

GOAL = (1, 2, 3, 4, 5, 6, 7, 8, 0)

# Direction mappings: where the BLANK moves
DIR_DELTA = {
    'U': (-1, 0),
    'D': (1, 0),
    'L': (0, -1),
    'R': (0, 1),
}

def get_blank_pos(board):
    """Return the position (row, col) of the blank (0)."""
    for i, v in enumerate(board):
        if v == 0:
            return (i // 3, i % 3)
    raise ValueError("No blank found")

def is_valid_move(current_board, move_char):
    """Check if a move is valid and return the new board state."""
    blank_pos = get_blank_pos(current_board)
    row, col = blank_pos
    dr, dc = DIR_DELTA[move_char]
    new_row, new_col = row + dr, col + dc
    
    # Check bounds
    if not (0 <= new_row < 3 and 0 <= new_col < 3):
        return None
    
    # Create new board with blank moved
    new_board_list = list(current_board)
    new_board_list[blank_pos[0] * 3 + blank_pos[1]] = current_board[new_row * 3 + new_col]
    new_board_list[new_row * 3 + new_col] = 0
    return tuple(new_board_list)

def manhattan_distance(board):
    """Compute Manhattan distance heuristic."""
    dist = 0
    for i, val in enumerate(board):
        if val == 0:
            continue
        goal_pos = val - 1  # Goal position for value v is at index v-1
        current_row, current_col = i // 3, i % 3
        goal_row, goal_col = (goal_pos) // 3, (goal_pos) % 3
        dist += abs(current_row - goal_row) + abs(current_col - goal_col)
    return dist

def is_solvable(board):
    """Check if a board state is solvable using inversion count."""
    # Count inversions (ignoring the blank)
    values = [v for v in board if v != 0]
    inversions = 0
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            if values[j] < values[i]:
                inversions += 1
    # For 3x3 puzzle: even number of inversions means solvable
    return inversions % 2 == 0

def solve(start):
    """
    Solve the 8-puzzle using A* with Manhattan distance heuristic.
    Returns a list of move chars (direction blank moves) or None if unsolvable.
    """
    # Check if already at goal
    if start == GOAL:
        return []
    
    # Check solvability
    if not is_solvable(start):
        return None
    
    # A* search
    # Priority queue: (f_score, counter, current_board, path)
    # Use counter to break ties consistently
    counter = 0
    start_blank = get_blank_pos(start)
    
    # Initial state
    f_start = manhattan_distance(start)
    pq = [(f_start, counter, start, [])]
    visited = {start: (f_start, [])}  # board -> (f_score, path_to_reach_it)
    
    while pq:
        f, _, current_board, path = heapq.heappop(pq)
        
        # Check if we reached the goal
        if current_board == GOAL:
            return path
        
        # Skip if we've found a better path to this state already
        if current_board in visited:
            prev_f, _ = visited[current_board]
            if f > prev_f:
                continue
        
        blank_pos = get_blank_pos(current_board)
        
        # Try all possible moves
        for move_char in ['U', 'D', 'L', 'R']:
            new_board = is_valid_move(current_board, move_char)
            if new_board is None:
                continue
            
            g_score = len(path) + 1
            f_score = g_score + manhattan_distance(new_board)
            
            # Check if this is a better path to new_board
            if new_board not in visited or f_score < visited[new_board][0]:
                visited[new_board] = (f_score, path + [move_char])
                counter += 1
                heapq.heappush(pq, (f_score, counter, new_board, path + [move_char]))
    
    return None

def verify(start, moves):
    """
    Verify a solution by applying each move and checking legality.
    Returns True only if the final board equals GOAL.
    """
    current = start
    for move in moves:
        new_board = is_valid_move(current, move)
        if new_board is None:
            return False  # Illegal move
        current = new_board
    
    return current == GOAL

def main():
    # Sanity checks
    sanity_passed = True
    
    # Check 1: solve(GOAL) == []
    result = solve(GOAL)
    if result != []:
        print(f"SANITY FAILED: solve(GOAL) = {result}, expected []")
        exit(1)
    
    # Check 2: solve((1, 2, 3, 4, 5, 6, 7, 0, 8)) == ['R']
    test_board = (1, 2, 3, 4, 5, 6, 7, 0, 8)
    result = solve(test_board)
    if result != ['R']:
        print(f"SANITY FAILED: solve({test_board}) = {result}, expected ['R']")
        exit(1)
    
    # Generate 100 random solvable boards
    rng = random.Random(42)
    base = list(range(9))
    
    solved_count = 0
    total_moves = 0
    
    with open('train.log', 'w') as log_file:
        for i in range(1, 101):
            # Generate a solvable board
            while True:
                shuffled = base.copy()
                rng.shuffle(shuffled)
                if is_solvable(tuple(shuffled)):
                    break
            
            start_board = tuple(shuffled)
            
            # Solve
            solve_start = time.perf_counter()
            solution = solve(start_board)
            elapsed_ms = int((time.perf_counter() - solve_start) * 1000)

            if solution is None:
                print(f"ERROR: Could not solve board {start_board}")
                exit(1)
            
            # Verify
            if not verify(start_board, solution):
                print(f"ERROR: Solution for {start_board} failed verification")
                exit(1)
            
            # Log progress
            log_file.write(f"puzzle {i}/100 moves={len(solution)} time_ms={elapsed_ms}\n")
            
            solved_count += 1
            total_moves += len(solution)
    
    # Calculate metrics
    avg_moves = total_moves / 100.0
    solved_rate = solved_count / 100.0
    
    # Write summary line to train.log
    with open('train.log', 'a') as log_file:
        log_file.write(f"Summary solved={solved_count}/100 avg_moves={avg_moves}\n")
    
    # Print metrics to stdout
    print(f"METRIC avg_moves {avg_moves}")
    print(f"METRIC solved_rate {solved_rate}")

if __name__ == "__main__":
    main()
