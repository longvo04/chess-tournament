"""
Chess AI Agents
"""
import random
import chess
import numpy as np
from typing import Optional


class Agent:
    """Base class for chess agents"""
    
    def __init__(self, name: str):
        self.name = name
    
    def get_move(self, board: chess.Board) -> chess.Move:
        """Get the next move for the given board state"""
        raise NotImplementedError


class RandomAgent(Agent):
    """Agent that makes random legal moves"""
    
    def __init__(self, name: str = "Random"):
        super().__init__(name)
    
    def get_move(self, board: chess.Board) -> chess.Move:
        legal_moves = list(board.legal_moves)
        return random.choice(legal_moves)


class MinimaxAgent(Agent):
    """Agent using Minimax algorithm with alpha-beta pruning"""
    
    # Piece values for evaluation
    PIECE_VALUES = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 20000
    }
    
    def __init__(self, name: str = "Minimax", depth: int = 3):
        super().__init__(name)
        self.depth = depth
    
    def get_move(self, board: chess.Board) -> chess.Move:
        # Determine if we're playing as white (maximizing) or black (minimizing)
        is_white = board.turn == chess.WHITE
        
        best_move = None
        if is_white:
            best_value = float('-inf')
        else:
            best_value = float('inf')
        
        alpha = float('-inf')
        beta = float('inf')
        shuffled_moves = list(board.legal_moves)
        random.shuffle(shuffled_moves)
        for move in shuffled_moves:
            board.push(move)
            # After our move, opponent's turn (flip maximizing)
            value = self._minimax(board, self.depth - 1, alpha, beta, not is_white)
            board.pop()
            
            # White maximizes, Black minimizes
            if is_white:
                if value > best_value:
                    best_value = value
                    best_move = move
                alpha = max(alpha, best_value)
            else:
                if value < best_value:
                    best_value = value
                    best_move = move
                beta = min(beta, best_value)
        
        return best_move if best_move else random.choice(list(board.legal_moves))
    
    def _minimax(self, board: chess.Board, depth: int, alpha: float, beta: float, maximizing: bool) -> float:
        """Minimax with alpha-beta pruning"""
        if depth == 0 or board.is_game_over():
            return self._evaluate_board(board)
        
        if maximizing:
            max_eval = float('-inf')
            for move in board.legal_moves:
                board.push(move)
                eval_score = self._minimax(board, depth - 1, alpha, beta, False)
                board.pop()
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in board.legal_moves:
                board.push(move)
                eval_score = self._minimax(board, depth - 1, alpha, beta, True)
                board.pop()
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval
    
    def _evaluate_board(self, board: chess.Board) -> float:
        """Evaluate board position from White's perspective"""
        if board.is_checkmate():
            # If it's white's turn and checkmate, white lost (negative)
            # If it's black's turn and checkmate, black lost (positive)
            return -20000 if board.turn == chess.WHITE else 20000
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
        
        score = 0
        for piece_type in self.PIECE_VALUES:
            # White pieces: positive contribution
            score += len(board.pieces(piece_type, chess.WHITE)) * self.PIECE_VALUES[piece_type]
            # Black pieces: negative contribution
            score -= len(board.pieces(piece_type, chess.BLACK)) * self.PIECE_VALUES[piece_type]
        
        # Always return from White's perspective (positive = good for white)
        return score


class MLAgent(Agent):
    """Machine Learning Agent using a simple neural network approach"""
    
    PIECE_VALUES = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0
    }
    
    def __init__(self, name: str = "ML"):
        super().__init__(name)
        # Simple evaluation with some randomness for variety
        self.temperature = 0.3
    
    def get_move(self, board: chess.Board) -> chess.Move:
        """Select move based on evaluation with exploration"""
        moves = list(board.legal_moves)
        move_scores = []
        
        for move in moves:
            board.push(move)
            score = self._evaluate_position(board)
            board.pop()
            move_scores.append(score)
        
        # Add some randomness (exploration)
        move_scores = np.array(move_scores)
        move_scores = move_scores + np.random.normal(0, self.temperature, len(move_scores))
        
        # White maximizes score, Black minimizes score
        if board.turn == chess.WHITE:
            best_idx = np.argmax(move_scores)
        else:
            best_idx = np.argmin(move_scores)
        
        return moves[best_idx]
    
    def _evaluate_position(self, board: chess.Board) -> float:
        """Evaluate position using various heuristics from White's perspective"""
        if board.is_checkmate():
            # If it's white's turn and checkmate, white lost (negative)
            # If it's black's turn and checkmate, black lost (positive)
            return -10000 if board.turn == chess.WHITE else 10000
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
        
        score = 0
        
        # Material count
        for piece_type in self.PIECE_VALUES:
            score += len(board.pieces(piece_type, chess.WHITE)) * self.PIECE_VALUES[piece_type]
            score -= len(board.pieces(piece_type, chess.BLACK)) * self.PIECE_VALUES[piece_type]
        
        # Mobility (number of legal moves for current player)
        mobility = len(list(board.legal_moves))
        # More mobility is good for the player whose turn it is
        if board.turn == chess.WHITE:
            score += mobility * 0.1
        else:
            score -= mobility * 0.1
        
        # Center control
        center_squares = [chess.E4, chess.E5, chess.D4, chess.D5]
        for square in center_squares:
            piece = board.piece_at(square)
            if piece:
                if piece.color == chess.WHITE:
                    score += 0.3
                else:
                    score -= 0.3
        
        # King safety (penalize exposed king in middlegame)
        if len(board.pieces(chess.QUEEN, chess.WHITE)) > 0 or len(board.pieces(chess.QUEEN, chess.BLACK)) > 0:
            white_king_square = board.king(chess.WHITE)
            black_king_square = board.king(chess.BLACK)
            
            # Check if king is in corner (safer)
            if white_king_square in [chess.A1, chess.H1, chess.A2, chess.B1, chess.G1, chess.H2]:
                score += 0.5
            if black_king_square in [chess.A8, chess.H8, chess.A7, chess.B8, chess.G8, chess.H7]:
                score -= 0.5
        
        # Always return from White's perspective (positive = good for white)
        return score


def create_agent(agent_type: str, name: str = None) -> Agent:
    """Factory function to create agents"""
    if agent_type.lower() == "random":
        return RandomAgent(name or "Random")
    elif agent_type.lower() == "minimax":
        return MinimaxAgent(name or "Minimax", depth=3)
    elif agent_type.lower() == "ml":
        return MLAgent(name or "ML")
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

