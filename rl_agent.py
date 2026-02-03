
from __future__ import annotations

import json
from typing import List, Tuple, Optional, Sequence, Iterable
import game_logic
import random


def compute_features(board: List[List[int]], player: int) -> List[float]:

    # Piece difference
    black, white = game_logic.count_pieces(board)
    piece_diff = (black - white) if player == game_logic.BLACK else (white - black)
    # Mobility difference
    player_moves = len(game_logic.get_valid_moves(board, player))
    opponent_moves = len(game_logic.get_valid_moves(board, -player))
    mobility_diff = player_moves - opponent_moves
    # Corner occupancy difference
    corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
    corner_diff = 0
    for cx, cy in corners:
        if board[cx][cy] == player:
            corner_diff += 1
        elif board[cx][cy] == -player:
            corner_diff -= 1
    # Edge occupancy difference (exclude corners)
    edge_diff = 0
    # top and bottom rows
    for y in range(8):
        if (0, y) not in corners:
            cell = board[0][y]
            if cell == player:
                edge_diff += 1
            elif cell == -player:
                edge_diff -= 1
            cell = board[7][y]
            if cell == player:
                edge_diff += 1
            elif cell == -player:
                edge_diff -= 1
    # left and right columns (excluding corners)
    for x in range(1, 7):
        cell = board[x][0]
        if cell == player:
            edge_diff += 1
        elif cell == -player:
            edge_diff -= 1
        cell = board[x][7]
        if cell == player:
            edge_diff += 1
        elif cell == -player:
            edge_diff -= 1
    # Normalize by board area or number of positions to keep features in similar range
    normalization = 64.0
    return [piece_diff / normalization,
            mobility_diff / 8.0,  # max diff is 8 moves at start
            corner_diff / 4.0,
            edge_diff / 24.0]  # 24 non-corner edge squares


def evaluate_move(board: List[List[int]], player: int, move: Tuple[int, int], weights: Sequence[float]) -> Tuple[float, List[float]]:
    new_board = game_logic.apply_move(board, player, move)
    features = compute_features(new_board, player)
    # Dot product of weights and features
    score = sum(w * f for w, f in zip(weights, features))
    return score, features


def choose_action(board: List[List[int]], player: int, weights: Sequence[float], epsilon: float) -> Tuple[Optional[Tuple[int, int]], Optional[List[float]]]:
    
    moves = game_logic.get_valid_moves(board, player)
    if not moves:
        return None, None
    # Exploration
    if random.random() < epsilon:
        move = random.choice(moves)
        return move, None
    # Exploitation: choose the best evaluated move
    best_move: Optional[Tuple[int, int]] = None
    best_score: float = float('-inf')
    best_features: Optional[List[float]] = None
    for m in moves:
        score, features = evaluate_move(board, player, m, weights)
        if score > best_score:
            best_score = score
            best_move = m
            best_features = features
    # best_move must be non-None because moves is non-empty
    return best_move, best_features


def update_weights(weights: List[float], trajectory: Iterable[List[float]], reward: float, alpha: float) -> None:
    for features in trajectory:
        for i in range(len(weights)):
            weights[i] += alpha * reward * features[i]


def play_game(weights: List[float], epsilon: float, alpha: float) -> float:
    board = game_logic.create_board()
    player = game_logic.BLACK  # AI always starts as black
    # Trajectories of feature vectors for the AI
    ai_trajectory: List[List[float]] = []
    while not game_logic.is_terminal(board):
        # AI's turn
        if player == game_logic.BLACK:
            move, features = choose_action(board, player, weights, epsilon)
            if move is None:
                # No valid move: pass
                player = -player
                continue
            # If move selected randomly, compute its features after applying move
            if features is None:
                _, features = evaluate_move(board, player, move, weights)
            board = game_logic.apply_move(board, player, move)
            ai_trajectory.append(features)
        else:
            # Opponent (white) plays random move
            moves = game_logic.get_valid_moves(board, player)
            if moves:
                opp_move = random.choice(moves)
                board = game_logic.apply_move(board, player, opp_move)
            # If no move: pass
        # Switch player
        player = -player
    # Game ended: compute reward from AI perspective
    black_count, white_count = game_logic.count_pieces(board)
    if black_count > white_count:
        reward = 1.0
    elif black_count < white_count:
        reward = -1.0
    else:
        reward = 0.0
    # Update weights based on the AI's trajectory
    update_weights(weights, ai_trajectory, reward, alpha)
    return reward


def train_agent(num_games: int, epsilon_start: float = 1.0, epsilon_end: float = 0.1, alpha: float = 0.05) -> List[float]:
    # Initialize weights to zero
    weights: List[float] = [0.0, 0.0, 0.0, 0.0]
    for game_index in range(num_games):
        # Linearly decay epsilon
        t = game_index / max(1, num_games - 1)
        epsilon = epsilon_start * (1 - t) + epsilon_end * t
        play_game(weights, epsilon, alpha)
    return weights

def save_weights(weights, filename="weights.json"):
    with open(filename, "w") as f:
        json.dump(weights, f)
        
def load_weights(filename="weights.json"):
    with open(filename, "r") as f:
        return json.load(f)
    
def play_game_no_update(weights, epsilon: float = 0.0) -> float:

    board = game_logic.create_board()
    player = game_logic.BLACK

    while not game_logic.is_terminal(board):
        if player == game_logic.BLACK:
            move, _ = choose_action(board, player, weights, epsilon)
            if move is None:
                player = -player
                continue
            board = game_logic.apply_move(board, player, move)
        else:
            moves = game_logic.get_valid_moves(board, player)
            if moves:
                opp_move = random.choice(moves)
                board = game_logic.apply_move(board, player, opp_move)
        player = -player

    black_count, white_count = game_logic.count_pieces(board)
    if black_count > white_count:
        return 1.0
    elif black_count < white_count:
        return -1.0
    else:
        return 0.0


def evaluate_against_random(weights, num_games: int = 2000, epsilon: float = 0.0, seed: int = 0) -> dict:
    #Avalia o agente (sem treino) contra um adversário aleatório.
    #Retorna dicionário com wins/losses/draws e win_rate.
    
    wins = losses = draws = 0

    for i in range(num_games):
        random.seed(seed + i)
        r = play_game_no_update(weights, epsilon=epsilon)

        if r > 0:
            wins += 1
        elif r < 0:
            losses += 1
        else:
            draws += 1

    return {
        "games": num_games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / num_games
    }


def greedy_opponent_move(board, player):
    """
    Adversário greedy: escolhe a jogada que maximiza o ganho imediato de peças
    (diferença de contagem após a jogada).
    """
    moves = game_logic.get_valid_moves(board, player)
    if not moves:
        return None

    best_move = None
    best_score = -10**9

    for m in moves:
        new_board = game_logic.apply_move(board, player, m)
        black, white = game_logic.count_pieces(new_board)
        score = (black - white) if player == game_logic.BLACK else (white - black)

        if score > best_score:
            best_score = score
            best_move = m

    return best_move


def play_game_no_update_greedy(weights, epsilon: float = 0.0) -> float:
    
    board = game_logic.create_board()
    player = game_logic.BLACK

    while not game_logic.is_terminal(board):
        if player == game_logic.BLACK:
            move, _ = choose_action(board, player, weights, epsilon)
            if move is None:
                player = -player
                continue
            board = game_logic.apply_move(board, player, move)
        else:
            move = greedy_opponent_move(board, player)
            if move is not None:
                board = game_logic.apply_move(board, player, move)
        player = -player

    black_count, white_count = game_logic.count_pieces(board)
    if black_count > white_count:
        return 1.0
    elif black_count < white_count:
        return -1.0
    else:
        return 0.0


def evaluate_against_greedy(weights, num_games: int = 2000, epsilon: float = 0.0, seed: int = 0) -> dict:
    #Avalia o agente (sem treino) contra adversário greedy.
    wins = losses = draws = 0
    for i in range(num_games):
        # semente para reprodutibilidade
        random.seed(seed + i)
        r = play_game_no_update_greedy(weights, epsilon=epsilon)
        if r > 0:
            wins += 1
        elif r < 0:
            losses += 1
        else:
            draws += 1

    win_rate = wins / num_games
    return {
        "games": num_games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": win_rate
    }
