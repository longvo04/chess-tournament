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
    """
    Machine Learning Agent sử dụng Neural Network
    Học từ Lichess database để đánh giá vị trí
    Kết hợp với 1-ply search để cải thiện hiệu suất
    """
    
    PIECE_VALUES = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0
    }
    
    def __init__(self, name: str = "ML", search_depth: int = 2):
        super().__init__(name)
        self.model = None
        self.use_torch = False
        self.input_size = 773
        self.model_loaded = False
        self.search_depth = search_depth  # Tìm trước n nước
        self._load_model()
    
    def _load_model(self):
        """Load trained model từ file"""
        import os
        
        pytorch_path = "ml/models/chess_model.pth"
        simple_path = "ml/models/simple_model.npz"
        
        # Thử load PyTorch model trước
        try:
            import torch
            from ml.model import ChessNet
            
            if os.path.exists(pytorch_path):
                checkpoint = torch.load(pytorch_path, map_location='cpu', weights_only=False)
                self.model = ChessNet(input_size=self.input_size)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.eval()
                self.use_torch = True
                self.model_loaded = True
                print(f"[{self.name}] ✓ Loaded PyTorch model")
                return
        except ImportError:
            pass
        except Exception as e:
            print(f"[{self.name}] PyTorch model error: {e}")
        
        # Fallback: SimpleModel
        if os.path.exists(simple_path):
            try:
                from ml.model import SimpleMLModel
                self.model = SimpleMLModel()
                self.model.load(simple_path)
                self.model_loaded = True
                print(f"[{self.name}] ✓ Loaded Simple model")
                return
            except Exception as e:
                print(f"[{self.name}] Simple model error: {e}")
        
        print(f"[{self.name}] ⚠ No trained model found, using heuristic evaluation")
        print(f"[{self.name}] Run 'python ml/train.py' to train a model")
    
    def _board_to_features(self, board: chess.Board) -> np.ndarray:
        """Chuyển board thành feature vector 773 chiều"""
        features = np.zeros(773, dtype=np.float32)
        
        # Piece positions (768 features)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                piece_idx = (piece.piece_type - 1)
                if piece.color == chess.BLACK:
                    piece_idx += 6
                feature_idx = piece_idx * 64 + square
                features[feature_idx] = 1.0
        
        # Turn
        features[768] = 1.0 if board.turn == chess.WHITE else 0.0
        
        # Castling rights
        features[769] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
        features[770] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
        features[771] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
        features[772] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
        
        return features
    
    def _order_moves(self, board: chess.Board, moves: list) -> list:
        """
        Sắp xếp nước đi theo độ ưu tiên để alpha-beta cắt tỉa tốt hơn
        Ưu tiên: captures > checks > others
        """
        scored_moves = []
        for move in moves:
            score = 0
            
            # Ưu tiên capture
            if board.is_capture(move):
                # MVV-LVA: Most Valuable Victim - Least Valuable Attacker
                victim = board.piece_at(move.to_square)
                attacker = board.piece_at(move.from_square)
                if victim:
                    score += 10 * self.PIECE_VALUES.get(victim.piece_type, 0)
                if attacker:
                    score -= self.PIECE_VALUES.get(attacker.piece_type, 0)
                score += 100  # Bonus for any capture
            
            # Ưu tiên check
            board.push(move)
            if board.is_check():
                score += 50
            # Checkmate là tốt nhất
            if board.is_checkmate():
                score += 10000
            board.pop()
            
            # Ưu tiên promotion
            if move.promotion:
                score += 80
            
            # Ưu tiên nước đi vào trung tâm
            to_sq = move.to_square
            if to_sq in [chess.E4, chess.D4, chess.E5, chess.D5]:
                score += 10
            elif to_sq in [chess.C3, chess.C4, chess.C5, chess.C6,
                          chess.D3, chess.D6, chess.E3, chess.E6,
                          chess.F3, chess.F4, chess.F5, chess.F6]:
                score += 5
            
            scored_moves.append((score, move))
        
        # Sắp xếp giảm dần theo score
        scored_moves.sort(key=lambda x: x[0], reverse=True)
        return [move for _, move in scored_moves]
    
    def get_move(self, board: chess.Board) -> chess.Move:
        """Chọn nước đi tốt nhất dựa trên minimax với ML evaluation"""
        moves = list(board.legal_moves)
        
        if len(moves) == 1:
            return moves[0]
        
        # Sắp xếp moves để tối ưu alpha-beta
        ordered_moves = self._order_moves(board, moves)
        
        is_white = board.turn == chess.WHITE
        best_move = ordered_moves[0]
        
        if is_white:
            best_value = float('-inf')
        else:
            best_value = float('inf')
        
        alpha = float('-inf')
        beta = float('inf')
        
        for move in ordered_moves:
            board.push(move)
            value = self._minimax_ml(board, self.search_depth - 1, alpha, beta, not is_white)
            board.pop()
            
            if is_white:
                if value > best_value:
                    best_value = value
                    best_move = move
                alpha = max(alpha, value)
            else:
                if value < best_value:
                    best_value = value
                    best_move = move
                beta = min(beta, value)
        
        return best_move
    
    def _minimax_ml(self, board: chess.Board, depth: int, alpha: float, beta: float, maximizing: bool) -> float:
        """Minimax search với ML evaluation ở leaf nodes"""
        # Terminal check
        if board.is_game_over():
            if board.is_checkmate():
                return 0.0 if board.turn == chess.WHITE else 1.0
            return 0.5  # Draw
        
        # Leaf node - use ML evaluation
        if depth <= 0:
            return self._evaluate_position(board)
        
        moves = list(board.legal_moves)
        
        # Quick move ordering cho internal nodes
        if depth > 1:
            moves = self._order_moves(board, moves)
        
        if maximizing:
            max_eval = float('-inf')
            for move in moves:
                board.push(move)
                eval_score = self._minimax_ml(board, depth - 1, alpha, beta, False)
                board.pop()
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in moves:
                board.push(move)
                eval_score = self._minimax_ml(board, depth - 1, alpha, beta, True)
                board.pop()
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval
    
    def _evaluate_position(self, board: chess.Board) -> float:
        """
        Đánh giá vị trí: trả về xác suất trắng thắng (0-1)
        0.0 = đen thắng chắc chắn
        0.5 = cân bằng/hòa
        1.0 = trắng thắng chắc chắn
        """
        # Terminal states
        if board.is_checkmate():
            return 0.0 if board.turn == chess.WHITE else 1.0
        if board.is_stalemate() or board.is_insufficient_material():
            return 0.5
        if board.can_claim_draw():
            return 0.5
        
        # Sử dụng ML model nếu có
        if self.model is not None and self.model_loaded:
            features = self._board_to_features(board)
            
            if self.use_torch:
                import torch
                with torch.no_grad():
                    x = torch.FloatTensor(features).unsqueeze(0)
                    ml_score = self.model(x).item()
            else:
                ml_score = self.model.predict(features.reshape(1, -1))[0]
            
            # Kết hợp ML score với material score để ổn định hơn
            material_score = self._get_material_score(board)
            
            # Weighted combination: 70% ML + 30% material
            combined_score = 0.7 * ml_score + 0.3 * material_score
            return float(combined_score)
        
        # Fallback: heuristic evaluation
        return self._heuristic_eval(board)
    
    def _get_material_score(self, board: chess.Board) -> float:
        """Tính điểm material, normalize về [0, 1]"""
        score = 0
        for piece_type, value in self.PIECE_VALUES.items():
            score += len(board.pieces(piece_type, chess.WHITE)) * value
            score -= len(board.pieces(piece_type, chess.BLACK)) * value
        
        # Normalize: score thường trong khoảng [-39, 39]
        # Map về [0, 1] dùng sigmoid
        return 1 / (1 + np.exp(-score / 10))
    
    def _heuristic_eval(self, board: chess.Board) -> float:
        """
        Heuristic evaluation khi không có ML model
        Trả về giá trị trong khoảng [0, 1]
        """
        score = 0.0
        
        # Material
        for piece_type, value in self.PIECE_VALUES.items():
            score += len(board.pieces(piece_type, chess.WHITE)) * value
            score -= len(board.pieces(piece_type, chess.BLACK)) * value
        
        # Mobility bonus
        mobility = len(list(board.legal_moves))
        if board.turn == chess.WHITE:
            score += mobility * 0.05
        else:
            score -= mobility * 0.05
        
        # Center control
        center_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
        for sq in center_squares:
            piece = board.piece_at(sq)
            if piece:
                if piece.color == chess.WHITE:
                    score += 0.3
                else:
                    score -= 0.3
        
        # Check bonus
        if board.is_check():
            if board.turn == chess.WHITE:
                score -= 0.5  # White is in check (bad for white)
            else:
                score += 0.5  # Black is in check (good for white)
        
        # Normalize to [0, 1] using sigmoid
        # score typically in range [-40, 40]
        return 1 / (1 + np.exp(-score / 10))


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

