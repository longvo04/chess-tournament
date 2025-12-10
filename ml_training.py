"""
Machine Learning Training Module for Chess AI
Uses Random Forest Regressor trained on self-play data
"""
import chess
import numpy as np
import random
import os
import pickle
from typing import List, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


# Đường dẫn lưu model
MODEL_PATH = "ml_models/chess_rf_model.pkl"


def extract_features(board: chess.Board) -> np.ndarray:
    """
    Trích xuất đặc trưng từ bàn cờ.
    Trả về vector 21 chiều - tối ưu cho tốc độ.
    """
    features = []
    
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0
    }
    
    # 1. Material count - chênh lệch số quân (6 features)
    for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, 
                       chess.ROOK, chess.QUEEN, chess.KING]:
        white_count = len(board.pieces(piece_type, chess.WHITE))
        black_count = len(board.pieces(piece_type, chess.BLACK))
        features.append(white_count - black_count)
    
    # 2. Total material (2 features)
    white_material = sum(len(board.pieces(pt, chess.WHITE)) * piece_values[pt] 
                        for pt in piece_values)
    black_material = sum(len(board.pieces(pt, chess.BLACK)) * piece_values[pt] 
                        for pt in piece_values)
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
    
    # 7. Pawn structure - simplified (2 features)
    white_pawns = board.pieces(chess.PAWN, chess.WHITE)
    black_pawns = board.pieces(chess.PAWN, chess.BLACK)
    features.append(len(white_pawns) - len(black_pawns))
    features.append(0)  # Placeholder
    
    # 8. Is in check (1 feature)
    features.append(1 if board.is_check() else 0)
    
    # 9. Turn indicator (1 feature)
    features.append(1 if board.turn == chess.WHITE else -1)
    
    return np.array(features, dtype=np.float32)


def generate_self_play_data(num_games: int = 2000, 
                            max_moves_per_game: int = 150,
                            verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tạo dữ liệu training từ self-play games.
    Mỗi game được chơi ngẫu nhiên và label được gán dựa trên kết quả.
    """
    X = []
    y = []
    
    wins = {"white": 0, "black": 0, "draw": 0}
    
    for game_idx in range(num_games):
        board = chess.Board()
        positions = []  # Lưu (features, turn) cho mỗi vị trí
        
        move_count = 0
        while not board.is_game_over() and move_count < max_moves_per_game:
            # Lưu vị trí hiện tại
            features = extract_features(board)
            positions.append((features, board.turn))
            
            # Chọn nước đi ngẫu nhiên
            move = random.choice(list(board.legal_moves))
            board.push(move)
            move_count += 1
        
        # Xác định kết quả
        if board.is_checkmate():
            # Người thua là người đang đi (không có nước đi)
            if board.turn == chess.WHITE:
                result = -1  # Black wins
                wins["black"] += 1
            else:
                result = 1   # White wins
                wins["white"] += 1
        else:
            result = 0  # Draw
            wins["draw"] += 1
        
        # Gán label cho tất cả các vị trí trong game
        for features, turn in positions:
            X.append(features)
            # Label từ góc nhìn của White
            # Nếu vị trí này là lượt White, result đã đúng
            # Nếu vị trí này là lượt Black, đảo ngược để đánh giá từ góc White
            y.append(result)
        
        if verbose and (game_idx + 1) % 200 == 0:
            print(f"Generated {game_idx + 1}/{num_games} games... "
                  f"(W: {wins['white']}, B: {wins['black']}, D: {wins['draw']})")
    
    if verbose:
        print(f"\nTotal positions collected: {len(X)}")
        print(f"Final results - White wins: {wins['white']}, "
              f"Black wins: {wins['black']}, Draws: {wins['draw']}")
    
    return np.array(X), np.array(y)


def generate_strategic_data(num_games: int = 1000, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tạo dữ liệu training với chiến lược thông minh hơn.
    Sử dụng heuristic đơn giản để chọn nước đi, giúp model học được các pattern tốt hơn.
    """
    X = []
    y = []
    
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0
    }
    
    def simple_eval(board: chess.Board) -> float:
        """Đánh giá đơn giản dựa trên material"""
        if board.is_checkmate():
            return -10000 if board.turn == chess.WHITE else 10000
        if board.is_stalemate():
            return 0
        
        score = 0
        for piece_type, value in piece_values.items():
            score += len(board.pieces(piece_type, chess.WHITE)) * value
            score -= len(board.pieces(piece_type, chess.BLACK)) * value
        return score
    
    def choose_move(board: chess.Board, randomness: float = 0.3) -> chess.Move:
        """Chọn nước đi với một ít ngẫu nhiên"""
        moves = list(board.legal_moves)
        
        if random.random() < randomness:
            return random.choice(moves)
        
        # Đánh giá từng nước đi
        best_move = None
        best_score = float('-inf') if board.turn == chess.WHITE else float('inf')
        
        for move in moves:
            board.push(move)
            score = simple_eval(board)
            board.pop()
            
            if board.turn == chess.WHITE:
                if score > best_score:
                    best_score = score
                    best_move = move
            else:
                if score < best_score:
                    best_score = score
                    best_move = move
        
        return best_move if best_move else random.choice(moves)
    
    for game_idx in range(num_games):
        board = chess.Board()
        positions = []
        
        move_count = 0
        while not board.is_game_over() and move_count < 150:
            features = extract_features(board)
            positions.append(features)
            
            move = choose_move(board)
            board.push(move)
            move_count += 1
        
        # Xác định kết quả
        if board.is_checkmate():
            result = -1 if board.turn == chess.WHITE else 1
        else:
            result = 0
        
        for features in positions:
            X.append(features)
            y.append(result)
        
        if verbose and (game_idx + 1) % 200 == 0:
            print(f"Generated {game_idx + 1}/{num_games} strategic games...")
    
    return np.array(X), np.array(y)


def train_model(X: np.ndarray, y: np.ndarray, 
                n_estimators: int = 200,
                max_depth: int = 15,
                verbose: bool = True) -> RandomForestRegressor:
    """
    Train Random Forest model.
    """
    if verbose:
        print(f"\nTraining Random Forest with {len(X)} samples...")
        print(f"Parameters: n_estimators={n_estimators}, max_depth={max_depth}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1,  # Sử dụng tất cả CPU cores
        random_state=42,
        verbose=1 if verbose else 0
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    if verbose:
        print(f"\nTraining R² score: {train_score:.4f}")
        print(f"Testing R² score: {test_score:.4f}")
    
    return model


def save_model(model: RandomForestRegressor, path: str = MODEL_PATH):
    """Lưu model ra file"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {path}")


def load_model(path: str = MODEL_PATH) -> RandomForestRegressor:
    """Load model từ file"""
    with open(path, 'rb') as f:
        return pickle.load(f)


def train_and_save(num_random_games: int = 3000, 
                   num_strategic_games: int = 2000,
                   n_estimators: int = 300,
                   max_depth: int = 20):
    """
    Pipeline hoàn chỉnh: Generate data -> Train -> Save
    """
    print("=" * 60)
    print("CHESS ML MODEL TRAINING")
    print("=" * 60)
    
    # Generate random self-play data
    print("\n[1/4] Generating random self-play data...")
    X1, y1 = generate_self_play_data(num_random_games)
    
    # Generate strategic data
    print("\n[2/4] Generating strategic self-play data...")
    X2, y2 = generate_strategic_data(num_strategic_games)
    
    # Combine data
    print("\n[3/4] Combining datasets...")
    X = np.vstack([X1, X2])
    y = np.concatenate([y1, y2])
    print(f"Total training samples: {len(X)}")
    
    # Train model
    print("\n[4/4] Training model...")
    model = train_model(X, y, n_estimators=n_estimators, max_depth=max_depth)
    
    # Save model
    save_model(model)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    
    return model


if __name__ == "__main__":
    # Chạy training với các tham số tối ưu
    train_and_save(
        num_random_games=3000,
        num_strategic_games=2000,
        n_estimators=300,
        max_depth=20
    )
