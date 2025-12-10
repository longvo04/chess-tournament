# ♟️ Chess Tournament - AI Game Playing

Hệ thống thi đấu cờ vua với các AI agents sử dụng **Minimax** và **Machine Learning**.

> **Bài tập lớn môn Trí tuệ Nhân tạo (AI) - HCMUT**

---

## 📋 Mục tiêu dự án

- Hiện thực game playing agent cho trò chơi cờ vua (đối kháng)
- Sử dụng giải thuật **Minimax với Alpha-Beta Pruning**
- Sử dụng **Machine Learning** (Neural Network) học từ Lichess database
- Đáp ứng yêu cầu:
  - ✅ Minimax thắng Random ≥ 90%
  - ✅ ML Agent thắng Random ≥ 60%

---

## 🎮 Tính năng

### AI Agents

| Agent | Mô tả | Thuật toán |
|-------|-------|------------|
| **Random** | Chơi ngẫu nhiên | Random choice |
| **Minimax** | Tìm kiếm cây trò chơi | Minimax + Alpha-Beta Pruning (depth=3) |
| **ML** | Học từ 563K positions | Neural Network (PyTorch) |

### Giao diện

- 🎯 Chọn agents và số trận đấu
- 📊 Hiển thị thống kê real-time
- 🔄 Replay các ván đã đấu
- ⌨️ Hỗ trợ phím tắt

---

## 🗂️ Cấu trúc dự án

```
chess-tournament/
├── main.py                 # Ứng dụng chính (GUI)
├── agents.py               # Định nghĩa các AI agents
├── tournament.py           # Quản lý giải đấu
├── ui_components.py        # Các component giao diện
├── setup_assets.py         # Setup hình ảnh quân cờ
├── requirements.txt        # Dependencies
│
├── ml/                     # 🤖 Machine Learning Module
│   ├── __init__.py
│   ├── data_processor.py   # Xử lý dữ liệu PGN từ Lichess
│   ├── model.py            # Neural Network models
│   ├── train.py            # Script training
│   └── models/             # Trained models
│       ├── chess_model.pth     # PyTorch model (90% accuracy)
│       └── simple_model.npz    # Backup model
│
├── data/                   # 📦 Dữ liệu
│   ├── pgn/                # File PGN từ Lichess
│   └── processed/          # Dataset đã xử lý
│
├── assets/                 # 🎨 Hình ảnh
│   └── img/chess_pieces/   # Hình các quân cờ
│
└── tournaments/            # 📁 Kết quả các giải đấu
```

---

## 🚀 Cài đặt và Chạy

### 1. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 2. Setup hình ảnh (lần đầu)

```bash
python setup_assets.py
```

### 3. Chạy ứng dụng

```bash
python main.py
```

---

## 🤖 Training ML Agent

### Bước 1: Chuẩn bị dữ liệu

**Cách 1:** Tải từ Lichess Database (khuyến nghị)
```bash
# Tải file từ https://database.lichess.org/
# Đặt file .pgn.zst vào data/pgn/

# Giải nén bằng Python
python -c "from ml.data_processor import decompress_zst_file; decompress_zst_file('data/pgn/FILE.pgn.zst', 'data/pgn/lichess_games.pgn')"
```

**Cách 2:** Tải từ Lichess API (nhanh hơn, ít data hơn)
```bash
python ml/train.py --use-api --max-games 3000
```

### Bước 2: Training

```bash
# Training với file PGN local
python ml/train.py --pgn data/pgn/lichess_games.pgn --max-games 15000 --min-elo 1800 --epochs 50

# Các options khác
python ml/train.py --help
```

### Kết quả Training

| Model | Samples | Accuracy | Thời gian |
|-------|---------|----------|-----------|
| SimpleModel | 563K | 55.5% | ~2 phút |
| **ChessNet** | 563K | **90.5%** | ~15 phút |

---

## 📊 Kết quả đánh giá

### Minimax vs Random (100 trận)

| Metric | Kết quả | Yêu cầu |
|--------|---------|---------|
| Win rate | ~95% | ≥ 90% ✅ |

### ML vs Random (100 trận)

| Metric | Kết quả | Yêu cầu |
|--------|---------|---------|
| Win rate | ~75% | ≥ 60% ✅ |

---

## 🧠 Chi tiết thuật toán

### 1. Minimax Agent

```
Thuật toán: Minimax với Alpha-Beta Pruning
Độ sâu: 3 ply
Hàm đánh giá: Material-based
  - Tốt: 100, Mã: 320, Tượng: 330
  - Xe: 500, Hậu: 900, Vua: 20000
```

### 2. ML Agent

```
Model: Fully Connected Neural Network
Architecture: 773 → 1024 → 512 → 256 → 128 → 1
Input: Board state (773 features)
  - 768: Piece positions (12 types × 64 squares)
  - 1: Turn
  - 4: Castling rights
Output: Win probability [0, 1]
Training data: 563,284 positions từ Lichess (Elo ≥ 1800)
Framework: PyTorch
```

---

## 📁 Dữ liệu Lichess

Dữ liệu được lấy từ [Lichess Database](https://database.lichess.org/):
- **File sử dụng:** `lichess_db_standard_rated_2015-07.pgn.zst`
- **Kích thước:** ~460MB (nén) → 2.5GB (giải nén)
- **Số games xử lý:** 15,000 (filtered Elo ≥ 1800)
- **Số positions:** 563,284

---

## 🎮 Hướng dẫn sử dụng

### Tạo giải đấu mới

1. Click **"New Tournament"**
2. Chọn **Agent 1** và **Agent 2**
3. Nhập số trận đấu
4. Click **"Start"**

### Xem lại ván đấu

1. Click **"Replay Tournament"**
2. Chọn giải đấu từ danh sách
3. Chọn ván đấu cụ thể
4. Sử dụng controls để xem từng nước

### Phím tắt (Replay)

| Phím | Chức năng |
|------|-----------|
| `←` `→` | Nước trước/sau |
| `Space` | Play/Pause |
| `↑` `↓` | Tăng/giảm tốc độ |

---

## 📝 Requirements

```
pygame>=2.6.0
python-chess>=1.9.0
numpy>=1.24.0
torch>=2.0.0
requests>=2.28.0
zstandard>=0.21.0
```

---

## 👥 Thành viên nhóm

| MSSV | Họ và Tên |
|------|-----------|
| | |
| | |
| | |

---

## 📚 Tài liệu tham khảo

1. Russell, S., & Norvig, P. - *Artificial Intelligence: A Modern Approach*
2. [Lichess Database](https://database.lichess.org/)
3. [python-chess Documentation](https://python-chess.readthedocs.io/)
4. [PyTorch Documentation](https://pytorch.org/docs/)

---

## 📄 License

Dự án này được thực hiện cho mục đích học tập tại HCMUT.
