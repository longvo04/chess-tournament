# Chess Tournament - Game Playing AI

Hệ thống giải đấu cờ vua với các AI agents sử dụng thuật toán Minimax và Machine Learning.

> **Bài tập lớn môn Trí tuệ Nhân tạo - HCMUT**

## 📋 Mục tiêu dự án

- Hiện thực game playing agent cho trò chơi cờ vua (đối kháng)
- So sánh hiệu quả giữa thuật toán Minimax và Machine Learning
- Đáp ứng yêu cầu:
  - ✅ Agent chơi đúng luật cờ vua
  - ✅ Minimax thắng Random ≥ 90%
  - ✅ Machine Learning thắng Random ≥ 60%

## 🎮 Tính năng

### AI Agents

| Agent | Mô tả | Win Rate vs Random |
|-------|-------|-------------------|
| **Random** | Chọn nước đi ngẫu nhiên | - |
| **Minimax** | Alpha-beta pruning, depth 3 | ~100% |
| **ML (Random Forest)** | Machine Learning với 300 trees | ~70% |

### Giao diện đồ họa

- **Tournament Management**: Tạo và quản lý giải đấu
- **Live Statistics**: Theo dõi kết quả realtime
- **Replay System**: Xem lại các ván đấu với playback controls

## 🚀 Cài đặt

### Yêu cầu

- Python 3.8+
- Windows/Linux/macOS

### Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### Dependencies

```
chess==1.11.2
numpy==2.3.4
pygame==2.6.1
scikit-learn>=1.3.0
```

## 📖 Hướng dẫn sử dụng

### Chạy ứng dụng chính

```bash
python main.py
```

### Train lại ML Model (tuỳ chọn)

```bash
python ml_training.py
```

### Test tỉ lệ thắng

```bash
# Test với 100 games (mặc định)
python test_agents.py

# Test với số games tuỳ chọn
python test_agents.py 50
```

## 🧠 Chi tiết các AI Agents

### 1. Random Agent

Agent cơ bản nhất, chọn ngẫu nhiên từ các nước đi hợp lệ.

```python
def get_move(self, board):
    return random.choice(list(board.legal_moves))
```

### 2. Minimax Agent (75% điểm BTL)

Sử dụng thuật toán **Minimax với Alpha-Beta Pruning**:

- **Độ sâu**: 3 ply (có thể điều chỉnh)
- **Evaluation function**: Đánh giá từ góc nhìn White
- **Alpha-Beta Pruning**: Cắt tỉa để tăng tốc

**Bảng giá trị quân cờ:**

| Quân | Giá trị |
|------|---------|
| Pawn | 100 |
| Knight | 320 |
| Bishop | 330 |
| Rook | 500 |
| Queen | 900 |
| King | 20000 |

**Pseudocode:**

```
function minimax(board, depth, alpha, beta, maximizing):
    if depth == 0 or game_over:
        return evaluate(board)
    
    if maximizing:
        maxEval = -∞
        for each move:
            eval = minimax(board, depth-1, alpha, beta, false)
            maxEval = max(maxEval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break  # Cắt tỉa
        return maxEval
    else:
        minEval = +∞
        for each move:
            eval = minimax(board, depth-1, alpha, beta, true)
            minEval = min(minEval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break  # Cắt tỉa
        return minEval
```

### 3. ML Agent - Random Forest (25% điểm BTL)

Sử dụng **Random Forest Regressor** được train từ self-play data.

#### Feature Engineering (21 features)

| # | Feature | Mô tả |
|---|---------|-------|
| 1-6 | Material Difference | Chênh lệch số quân (Pawn, Knight, Bishop, Rook, Queen, King) |
| 7-8 | Total Material | Tổng giá trị quân cờ mỗi bên |
| 9 | Mobility | Số nước đi hợp lệ hiện tại |
| 10-11 | Center Control | Kiểm soát trung tâm (e4, e5, d4, d5) và vùng mở rộng |
| 12-15 | Castling Rights | Quyền nhập thành (4 features) |
| 16-17 | King Safety | Vị trí an toàn của vua |
| 18-19 | Pawn Structure | Cấu trúc tốt |
| 20 | Is Check | Đang bị chiếu? |
| 21 | Turn Indicator | Lượt đi (White = 1, Black = -1) |

#### Training Process

```
1. Generate Self-Play Data:
   - 3000 random games
   - 2000 strategic games (with simple heuristic)
   - Total: ~650,000 positions

2. Train Random Forest:
   - n_estimators: 300
   - max_depth: 20
   - R² score: ~0.37

3. Inference:
   - Batch prediction cho tất cả nước đi
   - Chọn nước có score cao nhất (White) hoặc thấp nhất (Black)
```

#### Tối ưu tốc độ

- **Batch Prediction**: Gom tất cả features và predict 1 lần
- **Feature tối ưu**: Loại bỏ tính toán phức tạp
- **Kết quả**: ~100x nhanh hơn so với predict từng nước

## 📊 Kết quả đánh giá

### Test với 100 games

| Agent | Wins | Losses | Draws | Win Rate | Yêu cầu | Status |
|-------|------|--------|-------|----------|---------|--------|
| **Minimax** vs Random | 100 | 0 | 0 | **100%** | ≥90% | ✅ PASS |
| **ML** vs Random | 70 | 2 | 28 | **70%** | ≥60% | ✅ PASS |

### Phân tích

- **Minimax**: Hiệu quả rất cao nhờ tìm kiếm có chiều sâu
- **ML Agent**: Học được patterns cơ bản từ self-play data, đủ để thắng Random một cách ổn định

## 📁 Cấu trúc dự án

```
chess-tournament/
├── agents.py           # Định nghĩa các AI agents
├── main.py             # Ứng dụng chính với GUI
├── tournament.py       # Quản lý giải đấu
├── ui_components.py    # UI components
├── ml_training.py      # Script train ML model
├── test_agents.py      # Script test tỉ lệ thắng
├── requirements.txt    # Dependencies
├── README.md           # File này
├── ml_models/
│   └── chess_rf_model.pkl  # Trained Random Forest model
├── assets/
│   └── img/            # Hình ảnh quân cờ và UI
└── tournaments/        # Lưu kết quả giải đấu
    └── <tournament_name>/
        ├── result.txt
        ├── game1_Minimax.txt
        └── ...
```

## 🎯 Độ phức tạp của trò chơi

Cờ vua đáp ứng yêu cầu BTL:

- **Hệ số nhánh trung bình**: ~35 nước đi/lượt
- **Độ sâu cây game**: 40-50 nước mỗi bên (~80-100 ply) > 30 ✅
- **Không gian trạng thái**: ~10^44 vị trí có thể

## 🔧 Cấu hình nâng cao

### Thay đổi độ sâu Minimax

Trong `agents.py`:

```python
def create_agent(agent_type: str, name: str = None) -> Agent:
    elif agent_type.lower() == "minimax":
        return MinimaxAgent(name or "Minimax", depth=4)  # Thay đổi depth
```

### Train lại ML Model với tham số khác

Trong `ml_training.py`:

```python
train_and_save(
    num_random_games=5000,      # Tăng số games
    num_strategic_games=3000,   
    n_estimators=500,           # Nhiều trees hơn
    max_depth=25                # Sâu hơn
)
```

## 📝 Ghi chú

- Model ML đã được train sẵn trong `ml_models/chess_rf_model.pkl`
- Nếu muốn train lại, chạy `python ml_training.py` (mất ~2-3 phút)
- Kết quả giải đấu được lưu tự động trong thư mục `tournaments/`

## 👥 Thành viên nhóm

- [Thêm thông tin thành viên ở đây]

## 📄 License

Dự án được phát triển cho mục đích học tập tại HCMUT.
