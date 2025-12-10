"""
PGN Data Processor - Xử lý file PGN từ Lichess Database
Hỗ trợ download, giải nén và xử lý dữ liệu
"""
import chess
import chess.pgn
import numpy as np
import os
import io
import requests
from typing import List, Tuple, Generator, Optional
from datetime import datetime


class PGNDataProcessor:
    """Xử lý dữ liệu PGN để training ML model"""
    
    def __init__(self):
        # Piece square tables cho feature engineering (optional)
        self.pawn_table = np.array([
            0,  0,  0,  0,  0,  0,  0,  0,
            50, 50, 50, 50, 50, 50, 50, 50,
            10, 10, 20, 30, 30, 20, 10, 10,
            5,  5, 10, 25, 25, 10,  5,  5,
            0,  0,  0, 20, 20,  0,  0,  0,
            5, -5,-10,  0,  0,-10, -5,  5,
            5, 10, 10,-20,-20, 10, 10,  5,
            0,  0,  0,  0,  0,  0,  0,  0
        ], dtype=np.float32) / 100.0
    
    def board_to_features(self, board: chess.Board) -> np.ndarray:
        """
        Chuyển board thành feature vector 773 chiều
        - 768: 12 piece types × 64 squares (one-hot encoding)
        - 1: lượt đi (1=trắng, 0=đen)
        - 4: quyền nhập thành
        """
        features = np.zeros(773, dtype=np.float32)
        
        # Piece positions (768 features)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                # piece_type: 1-6 (PAWN to KING)
                # color: WHITE=True, BLACK=False
                piece_idx = (piece.piece_type - 1)
                if piece.color == chess.BLACK:
                    piece_idx += 6
                feature_idx = piece_idx * 64 + square
                features[feature_idx] = 1.0
        
        # Turn (1 feature)
        features[768] = 1.0 if board.turn == chess.WHITE else 0.0
        
        # Castling rights (4 features)
        features[769] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
        features[770] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
        features[771] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
        features[772] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
        
        return features
    
    def board_to_extended_features(self, board: chess.Board) -> np.ndarray:
        """
        Feature vector mở rộng với thêm thông tin chiến thuật
        Output: 773 + 12 = 785 features
        """
        base_features = self.board_to_features(board)
        
        # Thêm features bổ sung
        extra = np.zeros(12, dtype=np.float32)
        
        # Material count normalized
        piece_values = {
            chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
            chess.ROOK: 5, chess.QUEEN: 9
        }
        
        white_material = 0
        black_material = 0
        for pt, val in piece_values.items():
            white_material += len(board.pieces(pt, chess.WHITE)) * val
            black_material += len(board.pieces(pt, chess.BLACK)) * val
        
        extra[0] = white_material / 39.0  # Normalized (max ~39)
        extra[1] = black_material / 39.0
        extra[2] = (white_material - black_material) / 39.0
        
        # Mobility
        mobility = len(list(board.legal_moves))
        extra[3] = mobility / 50.0  # Normalized
        
        # King safety (simplified)
        white_king_sq = board.king(chess.WHITE)
        black_king_sq = board.king(chess.BLACK)
        if white_king_sq:
            extra[4] = (white_king_sq % 8) / 7.0  # King file
            extra[5] = (white_king_sq // 8) / 7.0  # King rank
        if black_king_sq:
            extra[6] = (black_king_sq % 8) / 7.0
            extra[7] = (black_king_sq // 8) / 7.0
        
        # Check status
        extra[8] = 1.0 if board.is_check() else 0.0
        
        # Center control (e4, d4, e5, d5)
        center_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
        white_center = sum(1 for sq in center_squares 
                         if board.piece_at(sq) and board.piece_at(sq).color == chess.WHITE)
        black_center = sum(1 for sq in center_squares 
                         if board.piece_at(sq) and board.piece_at(sq).color == chess.BLACK)
        extra[9] = white_center / 4.0
        extra[10] = black_center / 4.0
        extra[11] = (white_center - black_center) / 4.0
        
        return np.concatenate([base_features, extra])
    
    def create_dataset(self, pgn_path: str, output_path: str,
                       max_games: int = 10000, 
                       max_positions_per_game: int = 50,
                       min_elo: int = 2000,
                       skip_opening_moves: int = 10,
                       use_extended_features: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tạo dataset từ file PGN
        
        Args:
            pgn_path: Đường dẫn file PGN
            output_path: Đường dẫn lưu dataset (.npz)
            max_games: Số games tối đa
            max_positions_per_game: Số positions tối đa mỗi game
            min_elo: Elo tối thiểu (0 để bỏ qua filter)
            skip_opening_moves: Bỏ qua n nước đi đầu
            use_extended_features: Sử dụng features mở rộng
        
        Returns:
            (X, y) arrays
        """
        print(f"{'='*60}")
        print(f"PROCESSING PGN FILE")
        print(f"{'='*60}")
        print(f"File: {pgn_path}")
        print(f"Max games: {max_games}")
        print(f"Min Elo: {min_elo}")
        print(f"Skip opening moves: {skip_opening_moves}")
        print(f"{'='*60}\n")
        
        all_features = []
        all_labels = []
        
        games_processed = 0
        games_skipped = 0
        
        with open(pgn_path, 'r', encoding='utf-8', errors='ignore') as pgn_file:
            while games_processed < max_games:
                try:
                    game = chess.pgn.read_game(pgn_file)
                    if game is None:
                        break
                    
                    # Lọc theo Elo nếu cần
                    headers = game.headers
                    if min_elo > 0:
                        try:
                            white_elo = int(headers.get('WhiteElo', 0))
                            black_elo = int(headers.get('BlackElo', 0))
                            if white_elo < min_elo or black_elo < min_elo:
                                games_skipped += 1
                                continue
                        except (ValueError, TypeError):
                            games_skipped += 1
                            continue
                    
                    # Lấy kết quả
                    result = headers.get('Result', '*')
                    if result == '1-0':
                        label = 1.0  # White wins
                    elif result == '0-1':
                        label = 0.0  # Black wins
                    elif result == '1/2-1/2':
                        label = 0.5  # Draw
                    else:
                        games_skipped += 1
                        continue
                    
                    # Xử lý moves
                    board = game.board()
                    positions_in_game = 0
                    move_number = 0
                    
                    for move in game.mainline_moves():
                        move_number += 1
                        
                        # Bỏ qua opening
                        if move_number > skip_opening_moves:
                            if positions_in_game < max_positions_per_game:
                                if use_extended_features:
                                    features = self.board_to_extended_features(board)
                                else:
                                    features = self.board_to_features(board)
                                
                                all_features.append(features)
                                all_labels.append(label)
                                positions_in_game += 1
                        
                        board.push(move)
                    
                    games_processed += 1
                    
                    if games_processed % 500 == 0:
                        print(f"  Processed: {games_processed} games, {len(all_features)} positions")
                        
                except Exception as e:
                    games_skipped += 1
                    continue
        
        X = np.array(all_features, dtype=np.float32)
        y = np.array(all_labels, dtype=np.float32)
        
        print(f"\n{'='*60}")
        print(f"DATASET STATISTICS")
        print(f"{'='*60}")
        print(f"Games processed: {games_processed}")
        print(f"Games skipped: {games_skipped}")
        print(f"Total positions: {len(X)}")
        print(f"Features shape: {X.shape}")
        print(f"White wins: {np.mean(y > 0.6):.1%}")
        print(f"Draws: {np.mean((y > 0.4) & (y < 0.6)):.1%}")
        print(f"Black wins: {np.mean(y < 0.4):.1%}")
        print(f"{'='*60}\n")
        
        # Lưu dataset
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.savez_compressed(output_path, X=X, y=y)
        print(f"Dataset saved to: {output_path}")
        
        return X, y


def decompress_zst_file(input_path: str, output_path: str) -> bool:
    """
    Giải nén file .zst bằng Python
    
    Args:
        input_path: Đường dẫn file .zst
        output_path: Đường dẫn file output
    
    Returns:
        True nếu thành công
    """
    try:
        import zstandard as zstd
        
        print(f"Đang giải nén: {input_path}")
        print(f"Output: {output_path}")
        
        # Lấy size file input
        input_size = os.path.getsize(input_path)
        print(f"Input size: {input_size / (1024*1024):.1f} MB")
        
        dctx = zstd.ZstdDecompressor()
        
        with open(input_path, 'rb') as ifh:
            with open(output_path, 'wb') as ofh:
                dctx.copy_stream(ifh, ofh)
        
        output_size = os.path.getsize(output_path)
        print(f"Output size: {output_size / (1024*1024):.1f} MB")
        print(f"Giải nén thành công!")
        
        return True
        
    except ImportError:
        print("ERROR: Cần cài đặt zstandard: pip install zstandard")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def download_lichess_elite(output_dir: str = "data/pgn", 
                           month: str = None) -> Optional[str]:
    """
    Download Lichess Elite Database
    
    Args:
        output_dir: Thư mục lưu file
        month: Tháng cần download (YYYY-MM), None = tháng mới nhất
    
    Returns:
        Đường dẫn file PGN đã giải nén, hoặc None nếu lỗi
    """
    import requests
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Nếu không chỉ định tháng, dùng tháng gần đây
    if month is None:
        # Thử các tháng gần đây
        from datetime import datetime, timedelta
        current = datetime.now()
        months_to_try = []
        for i in range(1, 6):  # Thử 5 tháng gần nhất
            d = current - timedelta(days=30*i)
            months_to_try.append(d.strftime("%Y-%m"))
    else:
        months_to_try = [month]
    
    base_url = "https://database.lichess.org/elite"
    
    for m in months_to_try:
        filename = f"lichess_elite_{m}.pgn.zst"
        url = f"{base_url}/{filename}"
        
        zst_path = os.path.join(output_dir, filename)
        pgn_path = os.path.join(output_dir, f"lichess_elite_{m}.pgn")
        
        # Nếu đã có file PGN, return luôn
        if os.path.exists(pgn_path):
            print(f"File PGN đã tồn tại: {pgn_path}")
            return pgn_path
        
        # Nếu đã có file ZST, chỉ cần giải nén
        if os.path.exists(zst_path):
            print(f"File ZST đã tồn tại, đang giải nén...")
            if decompress_zst_file(zst_path, pgn_path):
                return pgn_path
            continue
        
        # Download file
        print(f"Đang thử download: {url}")
        
        try:
            response = requests.head(url, timeout=10)
            if response.status_code != 200:
                print(f"  -> Không tìm thấy (status {response.status_code})")
                continue
            
            # File tồn tại, download
            print(f"Đang download {filename}...")
            
            response = requests.get(url, stream=True, timeout=30)
            total_size = int(response.headers.get('content-length', 0))
            
            downloaded = 0
            with open(zst_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            pct = downloaded / total_size * 100
                            print(f"\r  Progress: {pct:.1f}% ({downloaded/(1024*1024):.1f} MB)", end='')
            
            print(f"\n  Download hoàn tất: {zst_path}")
            
            # Giải nén
            if decompress_zst_file(zst_path, pgn_path):
                return pgn_path
                
        except requests.exceptions.RequestException as e:
            print(f"  -> Lỗi: {e}")
            continue
    
    print("\nKhông thể download file từ Lichess.")
    print("Vui lòng download thủ công từ: https://database.lichess.org/elite/")
    return None


def download_sample_games_api(output_path: str = "data/pgn/sample_games.pgn",
                              num_games: int = 1000) -> Optional[str]:
    """
    Download sample games từ Lichess API (players mạnh)
    Phương án backup khi không download được Elite database
    """
    import requests
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Top players trên Lichess
    top_players = [
        "DrNykterstein",  # Magnus Carlsen
        "Hikaru",         # Hikaru Nakamura
        "GMWSO",          # Wesley So
        "nihalsarin",
        "penguingm1",
        "Zhigalko_Sergei",
        "Oleksandr_Bortnyk",
        "chess-network",
        "Lance5500",
        "opperwezen",
    ]
    
    all_pgn = []
    games_per_player = num_games // len(top_players) + 1
    
    print(f"Downloading games từ {len(top_players)} top players...")
    
    for username in top_players:
        url = f"https://lichess.org/api/games/user/{username}"
        headers = {"Accept": "application/x-chess-pgn"}
        params = {
            "max": games_per_player,
            "rated": "true",
            "perfType": "rapid,classical,blitz",
            "moves": "true",
            "clocks": "false",
            "evals": "false",
            "opening": "false"
        }
        
        try:
            print(f"  {username}...", end=" ")
            response = requests.get(url, headers=headers, params=params, 
                                   timeout=60, stream=True)
            
            if response.status_code == 200:
                pgn_text = response.text
                all_pgn.append(pgn_text)
                
                # Đếm số games
                game_count = pgn_text.count('[Event ')
                print(f"{game_count} games")
            else:
                print(f"Error {response.status_code}")
                
        except Exception as e:
            print(f"Error: {e}")
    
    if all_pgn:
        # Ghi ra file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(all_pgn))
        
        total_games = sum(p.count('[Event ') for p in all_pgn)
        print(f"\nTổng: {total_games} games")
        print(f"Saved to: {output_path}")
        return output_path
    
    return None


def print_download_instructions():
    """In hướng dẫn download thủ công"""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║           HƯỚNG DẪN TẢI LICHESS DATABASE THỦ CÔNG                ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  1. Truy cập: https://database.lichess.org/elite/                ║
║                                                                  ║
║  2. Tải file .pgn.zst (ví dụ: lichess_elite_2024-10.pgn.zst)     ║
║     - File ~100-200MB                                            ║
║                                                                  ║
║  3. Đặt file vào: data/pgn/                                      ║
║                                                                  ║
║  4. Chạy lệnh giải nén:                                          ║
║     python -c "from ml.data_processor import decompress_zst_file;║
║                decompress_zst_file('data/pgn/FILE.pgn.zst',      ║
║                                    'data/pgn/lichess_elite.pgn')"║
║                                                                  ║
║  HOẶC sử dụng Lichess API (không cần download):                  ║
║     python ml/train.py --use-api --max-games 5000                ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    import sys
    
    # Tạo thư mục
    os.makedirs("data/pgn", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    # Kiểm tra args
    if len(sys.argv) > 1:
        if sys.argv[1] == "--download":
            # Download từ Lichess
            pgn_path = download_lichess_elite()
            if pgn_path:
                print(f"\nFile PGN: {pgn_path}")
        elif sys.argv[1] == "--api":
            # Download từ API
            pgn_path = download_sample_games_api()
        elif sys.argv[1] == "--decompress" and len(sys.argv) > 2:
            # Giải nén file cụ thể
            zst_file = sys.argv[2]
            pgn_file = zst_file.replace('.zst', '')
            decompress_zst_file(zst_file, pgn_file)
        else:
            print_download_instructions()
    else:
        print_download_instructions()
