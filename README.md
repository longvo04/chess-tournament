# Chess Tournament Program

A Python-based chess tournament system with multiple AI agents, live tournament tracking, and comprehensive replay functionality.

## Features

- **Multiple AI Agents**: 
  - **Random Agent**: Makes random legal moves
  - **Minimax Agent**: Uses minimax algorithm with alpha-beta pruning (depth 3-5)
  - **ML Agent**: Machine learning-based agent with advanced position evaluation heuristics
  
- **Tournament Management**: 
  - Horizontal side-by-side agent selection interface
  - Customizable number of matches
  - UI remains responsive during tournaments using threading
  
- **Live Tournament Display**:
  - Real-time statistics showing both agents side-by-side
  - Live statistics panel showing:
    - Current match progress
    - Win/Loss/Draw counts for each agent
    - Win rates calculated in real-time
  
- **Tournament End Summary**:
  - Complete statistics for both agents
  - Time taken for tournament
  
- **Data Persistence**: 
  - Tournaments saved in organized folders
  - **Enhanced file naming**: `game1_winner.txt` format showing winner/draw
  - FEN notation AND UCI moves for every board state
  - Human-readable result summaries
  
- **Comprehensive Replay System**:
  - Browse all past tournaments
  - View tournament summaries with statistics
  - Select individual games for detailed replay
  - Playback controls with icon-based UI:
    - Previous/Next move navigation
    - Play/Pause button (dynamically switches)
    - Speed slider (0.5x to 5.0x) with real-time adjustment
    - **Keyboard shortcuts** with visual icon indicators

## Requirements

- Python 3.8 or higher
- pygame 2.6.1
- chess (python-chess) 1.11.2
- numpy 2.3.4

## Installation

### Quick Start

```bash
pip install -r requirements.txt
python main.py
```

## Usage Guide

### Starting a New Tournament

1. Click "New Tournament" from the main menu
2. Select **Agent 1** and **Agent 2** from the horizontal dropdown menus
   - Hover over dropdown options to see grey highlighting
3. Enter the number of matches (default: 10)
4. Optionally enter a tournament name
5. Click "Start" to begin

The tournament will run automatically in the background:
- **UI remains responsive** during execution (threaded)
- Live statistics displayed for both agents
- Progress through all matches
- "Running..." status indicator

### Viewing Tournament Results

After a tournament completes, you'll see:
- Total matches played
- Time taken
- Complete statistics for both agents (wins, win rate, losses)
- A colored progress bar showing win/draw/loss distribution

### Replaying Past Tournaments

1. Click "Replay Tournament" from the main menu
2. Select a tournament from the scrollable list
3. View the tournament summary and statistics
4. Select a specific game to replay (winner/draw shown in list)
5. Use the controls to navigate through the game:

**Mouse Controls:**
- **Speed Slider** (left): Adjust playback speed (0.5x - 5.0x)
- **◄ Prev**: Go to previous move
- **▶ Play** / **⏸ Pause**: Auto-play the game (button changes dynamically)
- **► Next**: Go to next move

**Keyboard Shortcuts** (with visual icons):
- **← →** : Previous/Next move
- **Space**: Play/Pause toggle
- **↑ ↓** (or **+** / **-**): Increase/Decrease speed

**Visual Features:**
- Move highlighting: Yellow overlay shows start/end squares and path
- Turn indicator: Colored text (blue for white, red for black)
- Game end display: "Checkmate!" (green) or "Draw!" (orange) at final move

## Tournament Data Structure

Tournaments are saved in the `tournaments/` directory with the following structure:

```
tournaments/
└── tournament_name_timestamp/
    ├── result.txt              # Tournament summary
    ├── game1_Minimax.txt      # First game (Minimax won)
    ├── game2_draw.txt         # Second game (draw)
    ├── game3_Random.txt       # Third game (Random won)
    └── ...
```

**File Naming:**
- Format: `game{number}_{winner}.txt`
- Winner is the agent name or "draw"
- Numbering starts from 1

**Each game file contains:**
- Result (winner or draw)
- White and Black player names
- **Moves (UCI)**: Standard chess notation (e.g., "e2e4", "g1f3")
- **Moves (FEN)**: Complete board states for each position

## AI Agent Details

### Random Agent
- Makes completely random moves from all legal options
- Fast execution
- Useful as a baseline for testing other agents

### Minimax Agent
- Implements minimax algorithm with **alpha-beta pruning**
- **Configurable depth** (default: 3 ply)
- **Proper evaluation** from White's perspective:
  - White maximizes score, Black minimizes score
  - Fixed alpha-beta pruning logic for optimal play
- Material-based evaluation:
  - Pawn: 100
  - Knight: 320
  - Bishop: 330
  - Rook: 500
  - Queen: 900
  - King: 20000

### ML Agent
- Advanced heuristic-based position evaluation
- Considers multiple strategic factors:
  - **Material count**: Traditional piece values
  - **Piece mobility**: Number of legal moves available
  - **Center control**: Bonus for pieces in central squares (e4, e5, d4, d5)
  - **King safety**: Evaluates king position in opening/middlegame
- **Exploration factor**: Adds randomness for variety (temperature = 0.3)
- **Proper perspective handling**: Always evaluates from White's viewpoint
- More sophisticated strategic play than pure minimax

## License

This project is provided as-is for educational purposes.
