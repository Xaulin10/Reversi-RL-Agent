"""
Core game logic for Reversi (Othello).

This module defines a minimal, self-contained implementation of the
game rules. It represents the board as an 8×8 matrix with the
following conventions:

* ``EMPTY = 0`` – no piece on the square
* ``BLACK = 1`` – the AI/player using black pieces
* ``WHITE = -1`` – the opposing player using white pieces

The functions provided allow you to create a new board, compute
valid moves, apply moves (including flipping captured discs), check
for the end of the game, and count pieces for scoring. They do not
depend on any graphics library and are intended to be used in
non-interactive simulations (e.g. training reinforcement learning
agents) as well as in conjunction with a graphical frontend.
"""

from __future__ import annotations

import copy
from typing import List, Tuple, Optional

# Constants to represent board cells
EMPTY: int = 0
BLACK: int = 1
WHITE: int = -1

def create_board() -> List[List[int]]:
    """Return a new game board initialized with the standard starting position.

    The starting position has two black pieces and two white pieces in the
    centre of the board arranged diagonally.
    """
    board: List[List[int]] = [[EMPTY for _ in range(8)] for _ in range(8)]
    board[3][3] = WHITE
    board[3][4] = BLACK
    board[4][3] = BLACK
    board[4][4] = WHITE
    return board

def is_on_board(x: int, y: int) -> bool:
    """Return True if coordinates (x, y) are within the 8×8 board."""
    return 0 <= x < 8 and 0 <= y < 8

def _get_flips(board: List[List[int]], player: int, x: int, y: int) -> List[Tuple[int, int]]:
    """Return a list of opponent positions that would be flipped for a move.

    If the move at (x, y) is invalid (does not capture in any direction),
    an empty list is returned.

    This function does not modify the board.
    """
    if board[x][y] != EMPTY:
        return []  # Cannot play on an occupied square
    opponent = -player
    flips: List[Tuple[int, int]] = []
    # Directions: N, NE, E, SE, S, SW, W, NW
    directions = [(-1, 0), (-1, 1), (0, 1), (1, 1),
                  (1, 0), (1, -1), (0, -1), (-1, -1)]
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        captured: List[Tuple[int, int]] = []
        # First move must be an opponent's piece
        while is_on_board(nx, ny) and board[nx][ny] == opponent:
            captured.append((nx, ny))
            nx += dx
            ny += dy
        # If we ended on our own piece after capturing at least one
        if is_on_board(nx, ny) and board[nx][ny] == player and captured:
            flips.extend(captured)
    return flips

def get_valid_moves(board: List[List[int]], player: int) -> List[Tuple[int, int]]:
    """Return a list of all valid moves for ``player`` on ``board``.

    A move is valid if placing a piece of ``player`` at the coordinate
    captures at least one of the opponent's pieces.
    """
    moves: List[Tuple[int, int]] = []
    for x in range(8):
        for y in range(8):
            if board[x][y] != EMPTY:
                continue
            if _get_flips(board, player, x, y):
                moves.append((x, y))
    return moves

def apply_move(board: List[List[int]], player: int, move: Tuple[int, int]) -> List[List[int]]:
    """Return a new board after applying ``move`` for ``player``.

    The returned board reflects the piece placed at ``move`` and all
    captured opponent pieces flipped. The original board is not
    modified.
    """
    x, y = move
    flips = _get_flips(board, player, x, y)
    if not flips:
        raise ValueError(f"Invalid move {move} for player {player}")
    new_board = copy.deepcopy(board)
    # Place the player's piece
    new_board[x][y] = player
    # Flip captured opponent pieces
    for fx, fy in flips:
        new_board[fx][fy] = player
    return new_board

def has_any_moves(board: List[List[int]], player: int) -> bool:
    """Return True if ``player`` has at least one valid move on ``board``."""
    return any(_get_flips(board, player, x, y) for x in range(8) for y in range(8) if board[x][y] == EMPTY)

def is_terminal(board: List[List[int]]) -> bool:
    """Return True if the game is over.

    The game ends when neither player has a valid move or when the
    board is full.
    """
    # Check board full
    for row in board:
        if any(cell == EMPTY for cell in row):
            break
    else:
        # No empty cells
        return True
    # Check no moves for both players
    return not (has_any_moves(board, BLACK) or has_any_moves(board, WHITE))

def count_pieces(board: List[List[int]]) -> Tuple[int, int]:
    """Return a tuple (black_count, white_count) of pieces on ``board``."""
    black = 0
    white = 0
    for row in board:
        for cell in row:
            if cell == BLACK:
                black += 1
            elif cell == WHITE:
                white += 1
    return black, white

def print_board(board: List[List[int]]) -> None:
    """Print the board to stdout for debugging."""
    symbols = {EMPTY: '.', BLACK: 'B', WHITE: 'W'}
    print('  0 1 2 3 4 5 6 7')
    for i, row in enumerate(board):
        print(i, ' '.join(symbols[cell] for cell in row))