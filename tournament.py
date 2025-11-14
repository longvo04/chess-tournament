"""
Tournament Management System
"""
import chess
import time
import os
import glob
from datetime import datetime
from typing import List, Tuple, Optional
from agents import Agent


class GameResult:
    """Stores result of a single game"""
    
    def __init__(self, winner: Optional[str], moves_fen: List[str], white_agent: str, black_agent: str, moves_uci: List[str] = None):
        self.winner = winner  # "white", "black", or "draw"
        self.moves_fen = moves_fen
        self.white_agent = white_agent
        self.black_agent = black_agent
        self.moves_uci = moves_uci or []  # UCI notation moves


class TournamentStats:
    """Tracks tournament statistics"""
    
    def __init__(self, agent1_name: str, agent2_name: str):
        self.agent1_name = agent1_name
        self.agent2_name = agent2_name
        self.agent1_wins = 0
        self.agent1_losses = 0
        self.agent2_wins = 0
        self.agent2_losses = 0
        self.draws = 0
        self.total_matches = 0
        self.time_taken = 0.0
    
    def update(self, result: str, agent1_color: str):
        """Update stats based on game result"""
        self.total_matches += 1
        
        if result == "draw":
            self.draws += 1
        elif (result == "white" and agent1_color == "white") or (result == "black" and agent1_color == "black"):
            self.agent1_wins += 1
            self.agent2_losses += 1
        else:
            self.agent2_wins += 1
            self.agent1_losses += 1
    
    def get_win_rate(self, agent_num: int) -> float:
        """Calculate win rate for an agent"""
        if self.total_matches == 0:
            return 0.0
        
        if agent_num == 1:
            return (self.agent1_wins / self.total_matches) * 100
        else:
            return (self.agent2_wins / self.total_matches) * 100
    
    def get_draw_rate(self) -> float:
        """Calculate draw rate"""
        if self.total_matches == 0:
            return 0.0
        return (self.draws / self.total_matches) * 100


class Tournament:
    """Manages a chess tournament between two agents"""
    
    def __init__(self, agent1: Agent, agent2: Agent, num_matches: int, name: str = None):
        self.agent1 = agent1
        self.agent2 = agent2
        self.num_matches = num_matches
        self.name = name or f"{agent1.name}_vs_{agent2.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.stats = TournamentStats(agent1.name, agent2.name)
        self.game_results: List[GameResult] = []
        self.current_match = 0
        self.start_time = None
    
    def play_game(self, white_agent: Agent, black_agent: Agent, max_moves: int = 500) -> GameResult:
        """Play a single game between two agents"""
        board = chess.Board()
        moves_fen = [board.fen()]
        moves_uci = []  # Store moves in UCI notation
        move_count = 0
        
        while not board.is_game_over() and move_count < max_moves:
            current_agent = white_agent if board.turn == chess.WHITE else black_agent
            
            try:
                move = current_agent.get_move(board)
                moves_uci.append(move.uci())  # Store move in UCI format
                board.push(move)
                moves_fen.append(board.fen())
                move_count += 1
            except Exception as e:
                print(f"Error getting move from {current_agent.name}: {e}")
                break
        
        # Determine winner
        if board.is_checkmate():
            winner = "white" if board.turn == chess.BLACK else "black"
        else:
            winner = "draw"
        
        return GameResult(winner, moves_fen, white_agent.name, black_agent.name, moves_uci)
    
    def run(self, callback=None):
        """Run the tournament with optional callback for progress updates"""
        self.start_time = time.time()
        self.game_results = []
        
        for i in range(self.num_matches):
            self.current_match = i + 1
            
            # Alternate colors
            if i % 2 == 0:
                white_agent = self.agent1
                black_agent = self.agent2
                agent1_color = "white"
            else:
                white_agent = self.agent2
                black_agent = self.agent1
                agent1_color = "black"
            
            result = self.play_game(white_agent, black_agent)
            self.game_results.append(result)
            self.stats.update(result.winner, agent1_color)
            
            if callback:
                callback(self.current_match, self.stats)
        
        self.stats.time_taken = time.time() - self.start_time
    
    def save_results(self):
        """Save tournament results to disk"""
        # Create tournaments directory if it doesn't exist
        os.makedirs("tournaments", exist_ok=True)
        
        # Create tournament folder
        tournament_dir = os.path.join("tournaments", self.name)
        os.makedirs(tournament_dir, exist_ok=True)
        
        # Save summary
        summary_path = os.path.join(tournament_dir, "result.txt")
        with open(summary_path, 'w') as f:
            f.write(f"Tournament: {self.name}\n")
            f.write(f"Total Matches: {self.stats.total_matches}\n")
            f.write(f"Time Taken: {self.stats.time_taken:.2f} seconds\n")
            f.write(f"\n{self.stats.agent1_name}:\n")
            f.write(f"  Wins: {self.stats.agent1_wins}\n")
            f.write(f"  Win Rate: {self.stats.get_win_rate(1):.2f}%\n")
            f.write(f"  Losses: {self.stats.agent1_losses}\n")
            f.write(f"\n{self.stats.agent2_name}:\n")
            f.write(f"  Wins: {self.stats.agent2_wins}\n")
            f.write(f"  Win Rate: {self.stats.get_win_rate(2):.2f}%\n")
            f.write(f"  Losses: {self.stats.agent2_losses}\n")
            f.write(f"\nDraws: {self.stats.draws}\n")
        
        # Save individual games
        for i, result in enumerate(self.game_results):
            # Determine winner name for filename
            if result.winner == "draw":
                winner_name = "draw"
            elif result.winner == "white":
                winner_name = result.white_agent
            else:  # black
                winner_name = result.black_agent
            
            # New format: game1_winner.txt (starting from 1)
            game_path = os.path.join(tournament_dir, f"game{i+1}_{winner_name}.txt")
            with open(game_path, 'w') as f:
                if result.winner == "draw":
                    f.write("Result: Draw\n")
                else:
                    f.write(f"Result: {result.winner.capitalize()} wins\n")
                f.write(f"White: {result.white_agent}\n")
                f.write(f"Black: {result.black_agent}\n")
                f.write("\nMoves (UCI):\n")
                for move_uci in result.moves_uci:
                    f.write(f"{move_uci}\n")
                f.write("\nMoves (FEN):\n")
                for fen in result.moves_fen:
                    f.write(f"{fen}\n")


def load_tournament(tournament_name: str) -> dict:
    """Load tournament data from disk"""
    tournament_dir = os.path.join("tournaments", tournament_name)
    
    if not os.path.exists(tournament_dir):
        return None
    
    # Load summary
    summary_path = os.path.join(tournament_dir, "result.txt")
    summary_data = {}
    
    with open(summary_path, 'r') as f:
        lines = f.readlines()
        summary_data['name'] = lines[0].split(': ')[1].strip()
        summary_data['total_matches'] = int(lines[1].split(': ')[1].strip())
        summary_data['time_taken'] = float(lines[2].split(': ')[1].strip().split()[0])
        
        # Parse agent stats
        summary_data['agent1'] = {}
        summary_data['agent1']['name'] = lines[4].strip().rstrip(':')
        summary_data['agent1']['wins'] = int(lines[5].split(': ')[1].strip())
        summary_data['agent1']['win_rate'] = float(lines[6].split(': ')[1].strip().rstrip('%'))
        summary_data['agent1']['losses'] = int(lines[7].split(': ')[1].strip())
        
        summary_data['agent2'] = {}
        summary_data['agent2']['name'] = lines[9].strip().rstrip(':')
        summary_data['agent2']['wins'] = int(lines[10].split(': ')[1].strip())
        summary_data['agent2']['win_rate'] = float(lines[11].split(': ')[1].strip().rstrip('%'))
        summary_data['agent2']['losses'] = int(lines[12].split(': ')[1].strip())
        
        summary_data['draws'] = int(lines[14].split(': ')[1].strip())
    
    # Load games - support both old (game0.txt) and new (game1_winner.txt) formats
    games = []
    game_files = sorted(glob.glob(os.path.join(tournament_dir, "game*.txt")))
    
    for game_path in game_files:
        # Extract winner from filename (new format: game1_winner.txt)
        filename = os.path.basename(game_path)
        winner_from_filename = None
        if '_' in filename:
            # New format: game1_winner.txt
            parts = filename.replace('.txt', '').split('_', 1)
            if len(parts) == 2:
                winner_from_filename = parts[1]
        
        with open(game_path, 'r') as f:
            lines = f.readlines()
            result = lines[0].split(': ')[1].strip()
            white = lines[1].split(': ')[1].strip()
            black = lines[2].split(': ')[1].strip()
            
            # Parse UCI moves and FEN positions
            moves_uci = []
            fens = []
            section = None
            
            for line in lines[3:]:  # Skip first 3 lines
                line = line.strip()
                if not line:
                    continue
                if line == "Moves (UCI):":
                    section = "uci"
                elif line == "Moves (FEN):":
                    section = "fen"
                elif section == "uci":
                    moves_uci.append(line)
                elif section == "fen":
                    fens.append(line)
            
            games.append({
                'result': result,
                'white': white,
                'black': black,
                'fens': fens,
                'moves_uci': moves_uci,
                'winner': winner_from_filename  # Store winner from filename
            })
    
    summary_data['games'] = games
    return summary_data


def list_tournaments() -> List[str]:
    """List all available tournaments"""
    tournaments_dir = "tournaments"
    if not os.path.exists(tournaments_dir):
        return []
    
    return [d for d in os.listdir(tournaments_dir) if os.path.isdir(os.path.join(tournaments_dir, d))]

