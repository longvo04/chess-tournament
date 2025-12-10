"""
Chess AI Agents
"""
import random
import chess
import numpy as np
import os
import pickle
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
    """
    Machine Learning Agent using Random Forest Regressor.
    Model được train từ self-play data để đánh giá vị trí bàn cờ.
    """
    
    MODEL_PATH = "ml_models/chess_rf_model.pkl"
    
    PIECE_VALUES = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0
    }
    
    def __init__(self, name: str = "ML", model_path: str = None):
        super().__init__(name)
        self.model_path = model_path or self.MODEL_PATH
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load pre-trained Random Forest model"""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                # Tắt verbose để tăng tốc
                if hasattr(self.model, 'verbose'):
                    self.model.verbose = 0
                if hasattr(self.model, 'n_jobs'):
                    self.model.n_jobs = 1  # Single thread cho prediction nhỏ
                print(f"ML Agent: Loaded model from {self.model_path}")
            except Exception as e:
                print(f"ML Agent: Failed to load model: {e}")
                self.model = None
        else:
            print(f"ML Agent: No pre-trained model found at {self.model_path}")
            print("ML Agent: Run 'python ml_training.py' to train the model first.")
            print("ML Agent: Using fallback heuristic evaluation.")
            self.model = None
    
    def _extract_features(self, board: chess.Board) -> np.ndarray:
        """
        Trích xuất đặc trưng từ bàn cờ.
        Phải giống hệt với function trong ml_training.py
        """
        features = []
        
        # 1. Material count - chênh lệch số quân (6 features)
        for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, 
                           chess.ROOK, chess.QUEEN, chess.KING]:
            white_count = len(board.pieces(piece_type, chess.WHITE))
            black_count = len(board.pieces(piece_type, chess.BLACK))
            features.append(white_count - black_count)
        
        # 2. Total material (2 features)
        white_material = sum(len(board.pieces(pt, chess.WHITE)) * self.PIECE_VALUES[pt] 
                            for pt in self.PIECE_VALUES)
        black_material = sum(len(board.pieces(pt, chess.BLACK)) * self.PIECE_VALUES[pt] 
                            for pt in self.PIECE_VALUES)
        features.append(white_material)
        features.append(black_material)
        
        # 3. Mobility (1 feature)
        current_mobility = len(list(board.legal_moves))
        features.append(current_mobility)
        
        # 4. Center control (2 features)
        center_squares = [chess.E4, chess.E5, chess.D4, chess.D5]
        extended_center = [chess.C3, chess.C4, chess.C5, chess.C6,
                           chess.D3, chess.D6, chess.E3, chess.E6,
                           chess.F3, chess.F4, chess.F5, chess.F6]
        
        white_center = sum(1 for sq in center_squares 
                          if board.piece_at(sq) and board.piece_at(sq).color == chess.WHITE)
        black_center = sum(1 for sq in center_squares 
                          if board.piece_at(sq) and board.piece_at(sq).color == chess.BLACK)
        features.append(white_center - black_center)
        
        white_ext_center = sum(1 for sq in extended_center 
                              if board.piece_at(sq) and board.piece_at(sq).color == chess.WHITE)
        black_ext_center = sum(1 for sq in extended_center 
                              if board.piece_at(sq) and board.piece_at(sq).color == chess.BLACK)
        features.append(white_ext_center - black_ext_center)
        
        # 5. Castling rights (4 features)
        features.append(1 if board.has_kingside_castling_rights(chess.WHITE) else 0)
        features.append(1 if board.has_queenside_castling_rights(chess.WHITE) else 0)
        features.append(1 if board.has_kingside_castling_rights(chess.BLACK) else 0)
        features.append(1 if board.has_queenside_castling_rights(chess.BLACK) else 0)
        
        # 6. King safety (2 features)
        white_king_sq = board.king(chess.WHITE)
        black_king_sq = board.king(chess.BLACK)
        
        white_king_safe = 1 if white_king_sq is not None and chess.square_rank(white_king_sq) == 0 else 0
        black_king_safe = 1 if black_king_sq is not None and chess.square_rank(black_king_sq) == 7 else 0
        features.append(white_king_safe)
        features.append(black_king_safe)
        
        # 7. Pawn structure (2 features)
        white_pawns = list(board.pieces(chess.PAWN, chess.WHITE))
        black_pawns = list(board.pieces(chess.PAWN, chess.BLACK))
        
        # Doubled pawns
        white_doubled = 0
        black_doubled = 0
        for file in range(8):
            white_in_file = sum(1 for sq in white_pawns if chess.square_file(sq) == file)
            black_in_file = sum(1 for sq in black_pawns if chess.square_file(sq) == file)
            if white_in_file > 1:
                white_doubled += white_in_file - 1
            if black_in_file > 1:
                black_doubled += black_in_file - 1
        features.append(black_doubled - white_doubled)
        
        # Passed pawns
        white_passed = 0
        black_passed = 0
        for sq in white_pawns:
            file = chess.square_file(sq)
            rank = chess.square_rank(sq)
            is_passed = True
            for enemy_sq in black_pawns:
                enemy_file = chess.square_file(enemy_sq)
                enemy_rank = chess.square_rank(enemy_sq)
                if abs(enemy_file - file) <= 1 and enemy_rank > rank:
                    is_passed = False
                    break
            if is_passed:
                white_passed += 1
        
        for sq in black_pawns:
            file = chess.square_file(sq)
            rank = chess.square_rank(sq)
            is_passed = True
            for enemy_sq in white_pawns:
                enemy_file = chess.square_file(enemy_sq)
                enemy_rank = chess.square_rank(enemy_sq)
                if abs(enemy_file - file) <= 1 and enemy_rank < rank:
                    is_passed = False
                    break
            if is_passed:
                black_passed += 1
        features.append(white_passed - black_passed)
        
        # 8. Is in check (1 feature)
        features.append(1 if board.is_check() else 0)
        
        # 9. Turn indicator (1 feature)
        features.append(1 if board.turn == chess.WHITE else -1)
        
        return np.array(features, dtype=np.float32).reshape(1, -1)
    
    def _extract_features_fast(self, board: chess.Board) -> np.ndarray:
        """
        Trích xuất đặc trưng từ bàn cờ - trả về 1D array với 21 features.
        Tối ưu cho batch prediction.
        """
        features = []
        
        # 1. Material count - chênh lệch số quân (6 features)
        for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, 
                           chess.ROOK, chess.QUEEN, chess.KING]:
            white_count = len(board.pieces(piece_type, chess.WHITE))
            black_count = len(board.pieces(piece_type, chess.BLACK))
            features.append(white_count - black_count)
        
        # 2. Total material (2 features)
        white_material = sum(len(board.pieces(pt, chess.WHITE)) * self.PIECE_VALUES[pt] 
                            for pt in self.PIECE_VALUES)
        black_material = sum(len(board.pieces(pt, chess.BLACK)) * self.PIECE_VALUES[pt] 
                            for pt in self.PIECE_VALUES)
        features.append(white_material)
        features.append(black_material)
        
        # 3. Mobility (1 feature)
        features.append(board.legal_moves.count())
        
        # 4. Center control (2 features)
        center_squares = [chess.E4, chess.E5, chess.D4, chess.D5]
        extended_center = [chess.C3, chess.C4, chess.C5, chess.C6,
                           chess.D3, chess.D6, chess.E3, chess.E6,
                           chess.F3, chess.F4, chess.F5, chess.F6]
        
        white_center = sum(1 for sq in center_squares 
                          if board.piece_at(sq) and board.piece_at(sq).color == chess.WHITE)
        black_center = sum(1 for sq in center_squares 
                          if board.piece_at(sq) and board.piece_at(sq).color == chess.BLACK)
        features.append(white_center - black_center)
        
        white_ext = sum(1 for sq in extended_center 
                       if board.piece_at(sq) and board.piece_at(sq).color == chess.WHITE)
        black_ext = sum(1 for sq in extended_center 
                       if board.piece_at(sq) and board.piece_at(sq).color == chess.BLACK)
        features.append(white_ext - black_ext)
        
        # 5. Castling rights (4 features)
        features.append(1 if board.has_kingside_castling_rights(chess.WHITE) else 0)
        features.append(1 if board.has_queenside_castling_rights(chess.WHITE) else 0)
        features.append(1 if board.has_kingside_castling_rights(chess.BLACK) else 0)
        features.append(1 if board.has_queenside_castling_rights(chess.BLACK) else 0)
        
        # 6. King safety (2 features)
        white_king_sq = board.king(chess.WHITE)
        black_king_sq = board.king(chess.BLACK)
        features.append(1 if white_king_sq and chess.square_rank(white_king_sq) == 0 else 0)
        features.append(1 if black_king_sq and chess.square_rank(black_king_sq) == 7 else 0)
        
        # 7. Pawn structure - simplified for speed (2 features)
        white_pawns = board.pieces(chess.PAWN, chess.WHITE)
        black_pawns = board.pieces(chess.PAWN, chess.BLACK)
        features.append(len(white_pawns) - len(black_pawns))  # Pawn difference
        features.append(0)  # Placeholder for passed pawns
        
        # 8. Is in check (1 feature)
        features.append(1 if board.is_check() else 0)
        
        # 9. Turn indicator (1 feature)
        features.append(1 if board.turn == chess.WHITE else -1)
        
        return np.array(features, dtype=np.float32)
    
    def _fallback_evaluate(self, board: chess.Board) -> float:
        """Fallback evaluation khi không có model"""
        if board.is_checkmate():
            return -10000 if board.turn == chess.WHITE else 10000
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
        
        score = 0
        for piece_type in self.PIECE_VALUES:
            score += len(board.pieces(piece_type, chess.WHITE)) * self.PIECE_VALUES[piece_type]
            score -= len(board.pieces(piece_type, chess.BLACK)) * self.PIECE_VALUES[piece_type]
        
        return score
    
    def get_move(self, board: chess.Board) -> chess.Move:
        """
        Chọn nước đi tốt nhất dựa trên Random Forest model.
        Sử dụng batch prediction để tăng tốc đáng kể.
        """
        moves = list(board.legal_moves)
        
        if not moves:
            return None
        
        # Nếu chỉ có 1 nước đi hợp lệ
        if len(moves) == 1:
            return moves[0]
        
        # Thu thập features cho tất cả nước đi cùng lúc
        valid_moves = []
        features_list = []
        
        for move in moves:
            board.push(move)
            
            # Ưu tiên checkmate ngay lập tức
            if board.is_checkmate():
                board.pop()
                return move  # Trả về ngay nếu tìm thấy checkmate
            
            # Bỏ qua stalemate (draw) trừ khi không có lựa chọn khác
            if not board.is_stalemate() and not board.is_insufficient_material():
                if self.model is not None:
                    features_list.append(self._extract_features_fast(board))
                else:
                    features_list.append(self._fallback_evaluate(board))
                valid_moves.append(move)
            
            board.pop()
        
        # Nếu không có nước đi hợp lệ (tất cả đều stalemate), chọn ngẫu nhiên
        if not valid_moves:
            return random.choice(moves)
        
        # Batch prediction - nhanh hơn nhiều so với predict từng cái
        if self.model is not None:
            features_array = np.array(features_list)
            scores = self.model.predict(features_array)
        else:
            scores = np.array(features_list)
        
        # Chọn nước đi tốt nhất
        if board.turn == chess.WHITE:
            best_idx = np.argmax(scores)
        else:
            best_idx = np.argmin(scores)
        
        return valid_moves[best_idx]


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

