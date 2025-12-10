"""
Training Script cho Chess ML Model
Hỗ trợ cả PyTorch và SimpleModel
Có thể download data từ Lichess hoặc sử dụng file PGN local
"""
import os
import sys
import numpy as np
import argparse
from datetime import datetime

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.data_processor import (
    PGNDataProcessor, 
    download_lichess_elite,
    download_sample_games_api,
    decompress_zst_file,
    print_download_instructions
)
from ml.model import SimpleMLModel, TORCH_AVAILABLE

if TORCH_AVAILABLE:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from ml.model import ChessNet, ChessResNet, save_pytorch_model


def train_simple_model(X_train, y_train, X_val, y_val, save_path: str):
    """Train SimpleMLModel (không cần PyTorch)"""
    print("\n" + "="*60)
    print("TRAINING SIMPLE MODEL (Logistic Regression)")
    print("="*60)
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Features: {X_train.shape[1]}")
    print("="*60 + "\n")
    
    model = SimpleMLModel(input_size=X_train.shape[1])
    
    history = model.fit(
        X_train, y_train,
        epochs=150,
        learning_rate=0.01,
        batch_size=512,
        verbose=True
    )
    
    # Validate
    val_pred = model.predict(X_val)
    val_loss = -np.mean(
        y_val * np.log(val_pred + 1e-7) + 
        (1 - y_val) * np.log(1 - val_pred + 1e-7)
    )
    val_acc = np.mean((val_pred > 0.5) == (y_val > 0.5))
    
    # More detailed accuracy
    white_win_acc = np.mean((val_pred[y_val > 0.6] > 0.5))
    black_win_acc = np.mean((val_pred[y_val < 0.4] < 0.5))
    
    print(f"\n{'='*60}")
    print("VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"Loss: {val_loss:.4f}")
    print(f"Overall Accuracy: {val_acc:.2%}")
    print(f"White win prediction accuracy: {white_win_acc:.2%}")
    print(f"Black win prediction accuracy: {black_win_acc:.2%}")
    print(f"{'='*60}\n")
    
    model.save(save_path)
    
    return model, history


def train_pytorch_model(X_train, y_train, X_val, y_val, save_path: str,
                        model_type: str = 'ChessNet',
                        epochs: int = 50,
                        batch_size: int = 256,
                        learning_rate: float = 0.001):
    """Train PyTorch model"""
    print("\n" + "="*60)
    print(f"TRAINING PYTORCH MODEL ({model_type})")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Features: {X_train.shape[1]}")
    print("="*60 + "\n")
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).unsqueeze(1).to(device)
    
    # DataLoader
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=0, pin_memory=True if device.type == 'cuda' else False)
    
    # Model - Deeper network cho performance tốt hơn
    input_size = X_train.shape[1]
    if model_type == 'ChessResNet':
        model = ChessResNet(input_size=input_size).to(device)
    else:
        model = ChessNet(input_size=input_size, 
                        hidden_sizes=[2048, 1024, 512, 256, 128],  # Deeper
                        dropout=0.25).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Loss & Optimizer - Sử dụng label smoothing
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Cosine annealing scheduler - tốt hơn cho convergence
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    # Training loop
    best_val_loss = float('inf')
    best_val_acc = 0
    patience_counter = 0
    max_patience = 20  # Tăng patience
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        scheduler.step()  # Cosine annealing step mỗi epoch
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_loss = criterion(val_outputs, y_val_t).item()
            val_acc = ((val_outputs > 0.5) == (y_val_t > 0.5)).float().mean().item()
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # Early stopping & save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            save_pytorch_model(model, save_path, optimizer, epoch, val_loss)
        else:
            patience_counter += 1
        
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:3d}: Train={train_loss:.4f}, Val={val_loss:.4f}, "
                  f"Acc={val_acc:.2%}, LR={current_lr:.6f}")
        
        if patience_counter >= max_patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Best Validation Accuracy: {best_val_acc:.2%}")
    print(f"{'='*60}\n")
    
    # Load best model
    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Train Chess ML Model')
    
    # Data source options
    data_group = parser.add_mutually_exclusive_group()
    data_group.add_argument('--pgn', type=str, 
                           help='Path to local PGN file')
    data_group.add_argument('--download', action='store_true',
                           help='Download Lichess Elite database')
    data_group.add_argument('--use-api', action='store_true',
                           help='Download games from Lichess API')
    data_group.add_argument('--skip-process', action='store_true',
                           help='Skip processing, use existing dataset')
    
    # Processing options
    parser.add_argument('--max-games', type=int, default=10000,
                        help='Maximum games to process (default: 10000)')
    parser.add_argument('--min-elo', type=int, default=2000,
                        help='Minimum Elo rating (default: 2000)')
    
    # Training options
    parser.add_argument('--epochs', type=int, default=50,
                        help='Training epochs (default: 50)')
    parser.add_argument('--model', type=str, default='all',
                        choices=['simple', 'pytorch', 'all'],
                        help='Model type to train (default: all)')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size (default: 256)')
    
    args = parser.parse_args()
    
    # Tạo thư mục
    os.makedirs("data/pgn", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("ml/models", exist_ok=True)
    
    dataset_path = "data/processed/chess_dataset.npz"
    pgn_path = None
    
    # ========== XÁC ĐỊNH NGUỒN DỮ LIỆU ==========
    
    if args.skip_process:
        # Sử dụng dataset có sẵn
        if not os.path.exists(dataset_path):
            print(f"[ERROR] Dataset không tồn tại: {dataset_path}")
            print("Chạy lại với --download hoặc --use-api để tạo dataset")
            return
        print(f"Using existing dataset: {dataset_path}")
        
    elif args.pgn:
        # Sử dụng file PGN local
        pgn_path = args.pgn
        
        # Kiểm tra nếu là file .zst
        if pgn_path.endswith('.zst'):
            output_pgn = pgn_path.replace('.zst', '')
            if not os.path.exists(output_pgn):
                print(f"Giải nén file: {pgn_path}")
                if not decompress_zst_file(pgn_path, output_pgn):
                    return
            pgn_path = output_pgn
        
        if not os.path.exists(pgn_path):
            print(f"[ERROR] File không tồn tại: {pgn_path}")
            return
            
    elif args.download:
        # Download từ Lichess Elite Database
        print("Đang download từ Lichess Elite Database...")
        pgn_path = download_lichess_elite("data/pgn")
        if pgn_path is None:
            print("\n[FALLBACK] Thử download từ API...")
            pgn_path = download_sample_games_api("data/pgn/lichess_api.pgn", 
                                                 num_games=min(args.max_games, 2000))
        if pgn_path is None:
            print_download_instructions()
            return
            
    elif args.use_api:
        # Download từ Lichess API
        print("Đang download games từ Lichess API...")
        pgn_path = download_sample_games_api(
            "data/pgn/lichess_api.pgn",
            num_games=min(args.max_games, 3000)
        )
        if pgn_path is None:
            print("[ERROR] Không thể download từ API")
            return
    
    else:
        # Không có option, kiểm tra file có sẵn
        default_paths = [
            "data/pgn/lichess_elite.pgn",
            "data/pgn/lichess_api.pgn",
        ]
        
        for p in default_paths:
            if os.path.exists(p):
                pgn_path = p
                print(f"Found existing PGN: {pgn_path}")
                break
        
        # Kiểm tra file .zst
        if pgn_path is None:
            import glob
            zst_files = glob.glob("data/pgn/*.pgn.zst")
            if zst_files:
                zst_file = zst_files[0]
                pgn_path = zst_file.replace('.zst', '')
                print(f"Found ZST file, decompressing: {zst_file}")
                if not decompress_zst_file(zst_file, pgn_path):
                    pgn_path = None
        
        if pgn_path is None and not os.path.exists(dataset_path):
            print("[INFO] Không tìm thấy dữ liệu, sẽ download từ API...")
            pgn_path = download_sample_games_api(
                "data/pgn/lichess_api.pgn",
                num_games=2000
            )
            if pgn_path is None:
                print_download_instructions()
                return
    
    # ========== XỬ LÝ DỮ LIỆU ==========
    
    if args.skip_process or (pgn_path is None and os.path.exists(dataset_path)):
        # Load dataset có sẵn
        print(f"\nLoading dataset: {dataset_path}")
        data = np.load(dataset_path)
        X, y = data['X'], data['y']
    else:
        # Xử lý PGN file
        processor = PGNDataProcessor()
        X, y = processor.create_dataset(
            pgn_path=pgn_path,
            output_path=dataset_path,
            max_games=args.max_games,
            max_positions_per_game=40,
            min_elo=args.min_elo,
            skip_opening_moves=8
        )
    
    print(f"\n{'='*60}")
    print("DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"Samples: {X.shape[0]:,}")
    print(f"Features: {X.shape[1]}")
    print(f"White wins: {np.mean(y > 0.6):.1%}")
    print(f"Draws: {np.mean((y > 0.4) & (y < 0.6)):.1%}")
    print(f"Black wins: {np.mean(y < 0.4):.1%}")
    print(f"{'='*60}\n")
    
    # ========== SPLIT DATA ==========
    
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    split = int(0.85 * len(X))
    train_idx, val_idx = indices[:split], indices[split:]
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    
    print(f"Train set: {len(X_train):,} samples")
    print(f"Validation set: {len(X_val):,} samples")
    
    # ========== TRAINING ==========
    
    if args.model in ['simple', 'all']:
        train_simple_model(
            X_train, y_train, X_val, y_val, 
            "ml/models/simple_model.npz"
        )
    
    if TORCH_AVAILABLE and args.model in ['pytorch', 'all']:
        train_pytorch_model(
            X_train, y_train, X_val, y_val,
            "ml/models/chess_model.pth",
            model_type='ChessNet',
            epochs=args.epochs,
            batch_size=args.batch_size
        )
    elif args.model == 'pytorch' and not TORCH_AVAILABLE:
        print("[WARNING] PyTorch không có sẵn. Chỉ train SimpleModel.")
    
    # ========== SUMMARY ==========
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print("\nModels saved:")
    
    if os.path.exists("ml/models/simple_model.npz"):
        print("  ✓ ml/models/simple_model.npz")
    if os.path.exists("ml/models/chess_model.pth"):
        print("  ✓ ml/models/chess_model.pth")
    
    print("\nDataset saved:")
    print(f"  ✓ {dataset_path}")
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("1. Chạy tournament để test ML agent:")
    print("   python main.py")
    print("\n2. Chạy 100 trận ML vs Random:")
    print("   - Chọn ML vs Random")
    print("   - Số trận: 100")
    print("   - Kiểm tra win rate >= 60%")
    print("="*60)


if __name__ == "__main__":
    main()
