"""
Chess Tournament Main Application
"""
import pygame
import sys
import os
import chess
import threading
from typing import Optional
from agents import create_agent
from tournament import Tournament, TournamentStats, list_tournaments, load_tournament
from ui_components import (
    Button, Dropdown, TextInput, ChessBoardRenderer, Slider,
    draw_text, draw_progress_bar,
    WHITE, BLACK, GRAY, LIGHT_GRAY, GREEN, RED, BLUE, DARK_GREEN, DARK_RED
)


class GameState:
    MAIN_MENU = "main_menu"
    TOURNAMENT_SETUP = "tournament_setup"
    TOURNAMENT_RUNNING = "tournament_running"
    TOURNAMENT_ENDED = "tournament_ended"
    REPLAY_SELECT = "replay_select"
    REPLAY_SUMMARY = "replay_summary"
    REPLAY_GAME = "replay_game"


class ChessTournamentApp:
    """Main application class"""
    
    def __init__(self):
        pygame.init()
        
        # Screen setup - Fixed window size
        self.screen_width = 1200
        self.screen_height = 800
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Chess Tournament")
        
        # Fonts
        self.title_font = pygame.font.Font(None, 64)
        self.header_font = pygame.font.Font(None, 48)
        self.normal_font = pygame.font.Font(None, 32)
        self.small_font = pygame.font.Font(None, 24)
        self.piece_font = pygame.font.Font(None, 64)
        
        # Game state
        self.state = GameState.MAIN_MENU
        self.clock = pygame.time.Clock()
        self.running = True
        
        # Tournament data
        self.tournament: Optional[Tournament] = None
        self.current_board: Optional[chess.Board] = None
        
        # Replay data
        self.replay_data: Optional[dict] = None
        self.replay_game_idx: int = 0
        self.replay_move_idx: int = 0
        self.replay_playing: bool = False
        self.replay_speed: float = 1.0
        self.replay_timer: float = 0
        self.keyboard_icons: dict = {}  # Keyboard shortcut icons
        
        # Threading for tournament execution
        self.tournament_thread: Optional[threading.Thread] = None
        self.tournament_lock = threading.Lock()
        self.tournament_complete_flag = False
        self.tournament_stats_cache: Optional[TournamentStats] = None
        
        # Initialize screens
        self.init_main_menu()
        self.init_tournament_setup()
    
    def init_main_menu(self):
        """Initialize main menu"""
        center_x = self.screen_width // 2
        button_width = 300
        button_height = 60
        
        self.main_menu_buttons = {
            'new_tournament': Button(
                center_x - button_width // 2, 300, button_width, button_height,
                "New Tournament", BLUE
            ),
            'replay_tournament': Button(
                center_x - button_width // 2, 400, button_width, button_height,
                "Replay Tournament", GREEN
            ),
            'exit': Button(
                center_x - button_width // 2, 500, button_width, button_height,
                "Exit", RED
            )
        }
    
    def init_tournament_setup(self):
        """Initialize tournament setup screen"""
        self.agent_types = ["Random", "Minimax", "ML"]
        
        # Horizontal layout for agent selection
        self.setup_dropdowns = {
            'agent1': Dropdown(150, 250, 300, 50, self.agent_types, "Random"),
            'agent2': Dropdown(650, 250, 300, 50, self.agent_types, "Minimax")
        }
        
        # Center inputs (screen width 1200, input width 300)
        center_x = self.screen_width // 2
        input_width = 300
        
        self.setup_inputs = {
            'num_matches': TextInput(center_x - input_width // 2, 380, input_width, 50, "Matches (10)"),
            'tournament_name': TextInput(center_x - input_width // 2, 480, input_width, 50, "Name (optional)")
        }
        
        # Center buttons (total width: 140 + 20 spacing + 140 = 300)
        button_width = 140
        button_spacing = 20
        total_button_width = button_width * 2 + button_spacing
        start_x = center_x - total_button_width // 2
        
        self.setup_buttons = {
            'start': Button(start_x, 580, button_width, 50, "Start", GREEN),
            'back': Button(start_x + button_width + button_spacing, 580, button_width, 50, "Back", RED)
        }
    
    def init_tournament_running(self):
        """Initialize tournament running screen"""
        # Don't initialize board renderer for running tournament
        self.current_board = None
    
    def init_tournament_ended(self):
        """Initialize tournament ended screen"""
        self.ended_button = Button(
            20, 20, 150, 40,
            "Back to Menu", RED
        )
    
    def init_replay_select(self):
        """Initialize replay selection screen"""
        tournaments = list_tournaments()
        
        self.replay_scroll_offset = 0
        # Calculate max visible based on screen height with bottom padding
        available_height = self.screen_height - 150 - 100  # start_y - bottom padding
        self.replay_max_visible = max(3, min(8, available_height // 70))
        
        if not tournaments:
            self.replay_tournaments = []
            self.replay_buttons = []
        else:
            self.replay_tournaments = tournaments
            self.replay_buttons = []
            
            button_width = min(600, self.screen_width - 200)
            start_y = 150
            for i, tournament in enumerate(tournaments):
                button = Button(
                    self.screen_width // 2 - button_width // 2, start_y + i * 70,
                    button_width, 60, tournament, BLUE
                )
                self.replay_buttons.append(button)
        
        self.replay_back_button = Button(20, 20, 100, 40, "Back", RED, icon_path="assets/img/ui/back.png")
    
    def init_replay_summary(self):
        """Initialize replay summary screen"""
        self.replay_summary_buttons = []
        self.replay_summary_scroll_offset = 0
        # Calculate max visible based on available screen height
        # Account for header space (~450px) and bottom padding (150px)
        available_height = self.screen_height - 600
        self.replay_summary_max_visible = max(3, min(10, available_height // 50))
        
        if self.replay_data and 'games' in self.replay_data:
            start_y = 450
            for i, game in enumerate(self.replay_data['games']):
                # Get winner information for button text
                winner_text = ""
                if 'winner' in game and game['winner']:
                    # New format with winner in filename
                    winner_text = f" - {game['winner']}"
                elif 'result' in game:
                    # Fallback: extract from result text
                    result = game['result']
                    if result == "Draw":
                        winner_text = " - draw"
                    elif "White" in result:
                        winner_text = f" - {game['white']}"
                    elif "Black" in result:
                        winner_text = f" - {game['black']}"
                
                button = Button(
                    self.screen_width // 2 - 200, start_y + i * 50,
                    400, 40, f"Game {i + 1}{winner_text}", BLUE
                )
                self.replay_summary_buttons.append(button)
        
        self.replay_summary_back = Button(20, 20, 100, 40, "Back", RED, icon_path="assets/img/ui/back.png")
    
    def init_replay_game(self):
        """Initialize replay game screen"""
        # Calculate board size - 70-80% of screen with padding
        info_panel_width = int(self.screen_width * 0.25)  # 25% for info
        padding = 30
        available_width = self.screen_width - info_panel_width - padding * 2
        board_size = min(available_width, int(self.screen_height * 0.85))
        board_x = info_panel_width + padding
        board_y = max(70, (self.screen_height - board_size - 80) // 2)
        self.replay_board_renderer = ChessBoardRenderer(board_x, board_y, board_size)
        
        # Position slider and buttons at top right with right margin
        control_y = 20
        right_margin = 50
        slider_width = 200
        button_width = 220  # Total width of 3 buttons with spacing
        spacing = 50  # Space between slider and buttons (increased)
        
        # Calculate starting position (slider comes first)
        total_width = slider_width + spacing + button_width
        start_x = self.screen_width - total_width - right_margin
        
        # Position speed slider on the left (aligned with button height)
        slider_y = control_y + 15  # Center slider vertically with 40px tall buttons
        self.replay_speed_slider = Slider(start_x, slider_y, slider_width, 0.5, 5.0, 1.0)
        
        # Position buttons to the right of slider
        control_x = start_x + slider_width + spacing
        
        self.replay_game_buttons = {
            'back': Button(20, 20, 100, 40, "Back", RED, icon_path="assets/img/ui/back.png"),
            'prev': Button(control_x, control_y, 70, 40, "◄", BLUE, icon_path="assets/img/ui/prev.png"),
            'play': Button(control_x + 75, control_y, 70, 40, "▶", GREEN, icon_path="assets/img/ui/play.png"),
            'next': Button(control_x + 150, control_y, 70, 40, "►", BLUE, icon_path="assets/img/ui/next.png")
        }
        
        # Load keyboard shortcut icons
        self.keyboard_icons = {}
        icon_size = 25
        keyboard_icons_map = {
            'left': 'assets/img/ui/left.png',
            'right': 'assets/img/ui/right.png',
            'space': 'assets/img/ui/space.png',
            'up': 'assets/img/ui/up.png',
            'down': 'assets/img/ui/down.png'
        }
        
        for key, path in keyboard_icons_map.items():
            try:
                if os.path.exists(path):
                    icon = pygame.image.load(path)
                    self.keyboard_icons[key] = pygame.transform.smoothscale(icon, (icon_size, icon_size))
            except:
                pass  # If icon doesn't exist, we'll use text fallback
    
    def run(self):
        """Main game loop"""
        while self.running:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(60)
        
        pygame.quit()
        sys.exit()
    
    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            if self.state == GameState.MAIN_MENU:
                self.handle_main_menu_events(event)
            elif self.state == GameState.TOURNAMENT_SETUP:
                self.handle_tournament_setup_events(event)
            elif self.state == GameState.TOURNAMENT_ENDED:
                self.handle_tournament_ended_events(event)
            elif self.state == GameState.REPLAY_SELECT:
                self.handle_replay_select_events(event)
            elif self.state == GameState.REPLAY_SUMMARY:
                self.handle_replay_summary_events(event)
            elif self.state == GameState.REPLAY_GAME:
                self.handle_replay_game_events(event)
    
    def handle_main_menu_events(self, event):
        """Handle main menu events"""
        if self.main_menu_buttons['new_tournament'].handle_event(event):
            self.state = GameState.TOURNAMENT_SETUP
        elif self.main_menu_buttons['replay_tournament'].handle_event(event):
            self.init_replay_select()
            self.state = GameState.REPLAY_SELECT
        elif self.main_menu_buttons['exit'].handle_event(event):
            self.running = False
    
    def handle_tournament_setup_events(self, event):
        """Handle tournament setup events"""
        for dropdown in self.setup_dropdowns.values():
            dropdown.handle_event(event)
        
        for text_input in self.setup_inputs.values():
            text_input.handle_event(event)
        
        if self.setup_buttons['start'].handle_event(event):
            self.start_tournament()
        elif self.setup_buttons['back'].handle_event(event):
            self.state = GameState.MAIN_MENU
    
    def handle_tournament_ended_events(self, event):
        """Handle tournament ended events"""
        if self.ended_button.handle_event(event):
            self.state = GameState.MAIN_MENU
    
    def handle_replay_select_events(self, event):
        """Handle replay selection events"""
        # Handle scrolling - return early to prevent button clicks during scroll
        if event.type == pygame.MOUSEWHEEL:
            self.replay_scroll_offset -= event.y
            max_scroll = max(0, len(self.replay_buttons) - self.replay_max_visible)
            self.replay_scroll_offset = max(0, min(self.replay_scroll_offset, max_scroll))
            return  # Don't process button clicks during scroll
        
        # Handle back button
        if self.replay_back_button.handle_event(event):
            self.state = GameState.MAIN_MENU
            return
        
        # Handle tournament selection buttons - only for mouse clicks
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:  # Left click only
            # Calculate visible range
            start_idx = self.replay_scroll_offset
            end_idx = min(start_idx + self.replay_max_visible, len(self.replay_buttons))
            
            for i in range(start_idx, end_idx):
                button = self.replay_buttons[i]
                if button.handle_event(event):
                    self.load_replay(self.replay_tournaments[i])
                    break
    
    def handle_replay_summary_events(self, event):
        """Handle replay summary events"""
        # Handle scrolling - return early to prevent button clicks during scroll
        if event.type == pygame.MOUSEWHEEL:
            self.replay_summary_scroll_offset -= event.y
            max_scroll = max(0, len(self.replay_summary_buttons) - self.replay_summary_max_visible)
            self.replay_summary_scroll_offset = max(0, min(self.replay_summary_scroll_offset, max_scroll))
            return  # Don't process button clicks during scroll
        
        # Handle back button
        if self.replay_summary_back.handle_event(event):
            self.init_replay_select()
            self.state = GameState.REPLAY_SELECT
            return
        
        # Handle game selection buttons - only for mouse clicks
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:  # Left click only
            # Calculate visible range
            start_idx = self.replay_summary_scroll_offset
            end_idx = min(start_idx + self.replay_summary_max_visible, len(self.replay_summary_buttons))
            
            for i in range(start_idx, end_idx):
                button = self.replay_summary_buttons[i]
                if button.handle_event(event):
                    self.replay_game_idx = i
                    self.replay_move_idx = 0
                    self.replay_playing = False
                    self.init_replay_game()
                    self.state = GameState.REPLAY_GAME
                    break
    
    def handle_replay_game_events(self, event):
        """Handle replay game events"""
        # Handle keyboard shortcuts
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                self.replay_move_idx = max(0, self.replay_move_idx - 1)
            elif event.key == pygame.K_RIGHT:
                max_idx = len(self.replay_data['games'][self.replay_game_idx]['fens']) - 1
                self.replay_move_idx = min(max_idx, self.replay_move_idx + 1)
            elif event.key == pygame.K_SPACE:
                self.replay_playing = not self.replay_playing
                self._update_play_button()
            elif event.key == pygame.K_UP or event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                # Increase speed
                self.replay_speed = min(5.0, self.replay_speed + 0.5)
                self.replay_speed_slider.value = self.replay_speed
                self.replay_speed_slider.update_handle_pos()
            elif event.key == pygame.K_DOWN or event.key == pygame.K_MINUS:
                # Decrease speed
                self.replay_speed = max(0.5, self.replay_speed - 0.5)
                self.replay_speed_slider.value = self.replay_speed
                self.replay_speed_slider.update_handle_pos()
        
        # Handle button clicks
        if self.replay_game_buttons['prev'].handle_event(event):
            self.replay_move_idx = max(0, self.replay_move_idx - 1)
        elif self.replay_game_buttons['next'].handle_event(event):
            max_idx = len(self.replay_data['games'][self.replay_game_idx]['fens']) - 1
            self.replay_move_idx = min(max_idx, self.replay_move_idx + 1)
        elif self.replay_game_buttons['play'].handle_event(event):
            self.replay_playing = not self.replay_playing
            self._update_play_button()
        elif self.replay_game_buttons['back'].handle_event(event):
            self.replay_playing = False
            self.init_replay_summary()
            self.state = GameState.REPLAY_SUMMARY
        
        self.replay_speed_slider.handle_event(event)
        self.replay_speed = self.replay_speed_slider.value
    
    def _update_play_button(self):
        """Update play button icon and text based on playing state"""
        import os
        if self.replay_playing:
            self.replay_game_buttons['play'].text = "⏸"
            icon_path = "assets/img/ui/pause.png"
        else:
            self.replay_game_buttons['play'].text = "▶"
            icon_path = "assets/img/ui/play.png"
        
        # Update icon if it exists
        if os.path.exists(icon_path):
            try:
                icon_image = pygame.image.load(icon_path)
                icon_size = min(70 - 10, 40 - 10)
                self.replay_game_buttons['play'].icon = pygame.transform.smoothscale(icon_image, (icon_size, icon_size))
            except:
                pass
    
    def handle_resize(self, width: int, height: int):
        """Handle window resize event"""
        # Update screen dimensions
        self.screen_width = max(800, width)  # Minimum width
        self.screen_height = max(600, height)  # Minimum height
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height), pygame.RESIZABLE)
        
        # Reinitialize UI elements for current state
        if self.state == GameState.MAIN_MENU:
            self.init_main_menu()
        elif self.state == GameState.TOURNAMENT_SETUP:
            self.init_tournament_setup()
        elif self.state == GameState.TOURNAMENT_RUNNING:
            self.init_tournament_running()
        elif self.state == GameState.TOURNAMENT_ENDED:
            self.init_tournament_ended()
        elif self.state == GameState.REPLAY_SELECT:
            self.init_replay_select()
        elif self.state == GameState.REPLAY_SUMMARY:
            self.init_replay_summary()
        elif self.state == GameState.REPLAY_GAME:
            self.init_replay_game()
    
    def update(self):
        """Update game state"""
        if self.state == GameState.TOURNAMENT_RUNNING:
            # Check if tournament thread has completed
            if self.tournament_complete_flag:
                with self.tournament_lock:
                    self.tournament_complete_flag = False
                    # Copy final stats
                    if self.tournament:
                        self.tournament_stats_cache = self.tournament.stats
                
                # Move to ended screen
                self.init_tournament_ended()
                self.state = GameState.TOURNAMENT_ENDED
            else:
                # Update stats cache from running tournament (thread-safe)
                with self.tournament_lock:
                    if self.tournament and self.tournament.stats:
                        # Create a copy of stats to avoid race conditions
                        self.tournament_stats_cache = self.tournament.stats
        
        elif self.state == GameState.REPLAY_GAME and self.replay_playing:
            # Auto-advance replay
            self.replay_timer += self.clock.get_time() / 1000.0
            if self.replay_timer >= (1.0 / self.replay_speed):
                self.replay_timer = 0
                max_idx = len(self.replay_data['games'][self.replay_game_idx]['fens']) - 1
                self.replay_move_idx += 1
                if self.replay_move_idx > max_idx:
                    self.replay_move_idx = max_idx
                    self.replay_playing = False
                    self._update_play_button()
    
    def draw(self):
        """Draw current state"""
        self.screen.fill(WHITE)
        
        if self.state == GameState.MAIN_MENU:
            self.draw_main_menu()
        elif self.state == GameState.TOURNAMENT_SETUP:
            self.draw_tournament_setup()
        elif self.state == GameState.TOURNAMENT_RUNNING:
            self.draw_tournament_running()
        elif self.state == GameState.TOURNAMENT_ENDED:
            self.draw_tournament_ended()
        elif self.state == GameState.REPLAY_SELECT:
            self.draw_replay_select()
        elif self.state == GameState.REPLAY_SUMMARY:
            self.draw_replay_summary()
        elif self.state == GameState.REPLAY_GAME:
            self.draw_replay_game()
        
        pygame.display.flip()
    
    def draw_main_menu(self):
        """Draw main menu"""
        draw_text(self.screen, "Chess Tournament", self.screen_width // 2, 150,
                 self.title_font, BLACK, center=True)
        
        for button in self.main_menu_buttons.values():
            button.draw(self.screen, self.normal_font)
    
    def draw_tournament_setup(self):
        """Draw tournament setup screen"""
        draw_text(self.screen, "Tournament Setup", self.screen_width // 2, 100,
                 self.header_font, BLACK, center=True)
        
        # Agent selection header
        draw_text(self.screen, "Select Agents", self.screen_width // 2, 180,
                 self.normal_font, BLACK, center=True)
        
        # Labels for horizontal agent selection
        draw_text(self.screen, "Agent 1:", 150, 220, self.normal_font)
        draw_text(self.screen, "Agent 2:", 650, 220, self.normal_font)
        
        # Centered labels for inputs
        draw_text(self.screen, "Number of Matches:", self.screen_width // 2, 350, self.normal_font, BLACK, center=True)
        draw_text(self.screen, "Tournament Name:", self.screen_width // 2, 450, self.normal_font, BLACK, center=True)
        
        # Draw text inputs first
        for text_input in self.setup_inputs.values():
            text_input.draw(self.screen, self.normal_font)
        
        # Draw buttons
        for button in self.setup_buttons.values():
            button.draw(self.screen, self.normal_font)
        
        # Draw dropdowns LAST so they appear on top
        for dropdown in self.setup_dropdowns.values():
            dropdown.draw(self.screen, self.normal_font)
    
    def draw_tournament_running(self):
        """Draw tournament running screen"""
        if not self.tournament:
            return
        
        # Use cached stats to avoid race conditions with tournament thread
        stats = self.tournament_stats_cache
        if not stats:
            return
        
        # Center the stats on screen
        draw_text(self.screen, "Tournament in Progress", self.screen_width // 2, 80, self.header_font, BLACK, center=True)
        
        # Show status indicator
        status_text = "Running..." if self.tournament_thread and self.tournament_thread.is_alive() else "Finishing..."
        draw_text(self.screen, status_text, self.screen_width // 2, 130, self.small_font, GRAY, center=True)
        
        y_pos = 200
        draw_text(self.screen, f"Match: {self.tournament.current_match}/{self.tournament.num_matches}",
                 self.screen_width // 2, y_pos, self.title_font, BLACK, center=True)
        
        # Draw agents side by side
        y_pos += 150
        left_x = self.screen_width // 4
        right_x = 3 * self.screen_width // 4
        
        # Agent 1 on the left
        draw_text(self.screen, stats.agent1_name, left_x, y_pos, self.header_font, BLUE, center=True)
        y_temp = y_pos + 60
        draw_text(self.screen, f"Wins: {stats.agent1_wins}", left_x, y_temp, self.normal_font, BLACK, center=True)
        y_temp += 45
        draw_text(self.screen, f"Win Rate: {stats.get_win_rate(1):.1f}%", left_x, y_temp, self.normal_font, BLACK, center=True)
        y_temp += 45
        draw_text(self.screen, f"Losses: {stats.agent1_losses}", left_x, y_temp, self.normal_font, BLACK, center=True)
        
        # VS in the center
        draw_text(self.screen, "VS", self.screen_width // 2, y_pos + 90, self.title_font, GRAY, center=True)
        
        # Agent 2 on the right
        draw_text(self.screen, stats.agent2_name, right_x, y_pos, self.header_font, RED, center=True)
        y_temp = y_pos + 60
        draw_text(self.screen, f"Wins: {stats.agent2_wins}", right_x, y_temp, self.normal_font, BLACK, center=True)
        y_temp += 45
        draw_text(self.screen, f"Win Rate: {stats.get_win_rate(2):.1f}%", right_x, y_temp, self.normal_font, BLACK, center=True)
        y_temp += 45
        draw_text(self.screen, f"Losses: {stats.agent2_losses}", right_x, y_temp, self.normal_font, BLACK, center=True)
        
        # Draws centered below
        y_pos += 250
        draw_text(self.screen, f"Draws: {stats.draws}", self.screen_width // 2, y_pos, self.header_font, BLACK, center=True)
    
    def draw_tournament_ended(self):
        """Draw tournament ended screen"""
        if not self.tournament:
            return
        
        # Use cached stats (final results)
        stats = self.tournament_stats_cache
        if not stats:
            return
        
        draw_text(self.screen, "Tournament Complete!", self.screen_width // 2, 80,
                 self.header_font, BLACK, center=True)
        
        y_pos = 180
        draw_text(self.screen, f"Total Matches: {stats.total_matches}", 
                 self.screen_width // 2, y_pos, self.normal_font, BLACK, center=True)
        
        y_pos += 50
        draw_text(self.screen, f"Time Taken: {stats.time_taken:.2f} seconds",
                 self.screen_width // 2, y_pos, self.normal_font, BLACK, center=True)
        
        # Agent 1 stats
        y_pos += 80
        draw_text(self.screen, stats.agent1_name, self.screen_width // 2, y_pos,
                 self.header_font, BLUE, center=True)
        y_pos += 50
        draw_text(self.screen, f"Wins: {stats.agent1_wins}  |  Win Rate: {stats.get_win_rate(1):.1f}%  |  Losses: {stats.agent1_losses}",
                 self.screen_width // 2, y_pos, self.normal_font, BLACK, center=True)
        
        # Agent 2 stats
        y_pos += 80
        draw_text(self.screen, stats.agent2_name, self.screen_width // 2, y_pos,
                 self.header_font, RED, center=True)
        y_pos += 50
        draw_text(self.screen, f"Wins: {stats.agent2_wins}  |  Win Rate: {stats.get_win_rate(2):.1f}%  |  Losses: {stats.agent2_losses}",
                 self.screen_width // 2, y_pos, self.normal_font, BLACK, center=True)
        
        # Draws
        y_pos += 60
        draw_text(self.screen, f"Draws: {stats.draws}",
                 self.screen_width // 2, y_pos, self.normal_font, BLACK, center=True)
        
        # Progress bar
        y_pos += 80
        bar_width = min(800, self.screen_width - 100)
        bar_height = 60
        bar_x = (self.screen_width - bar_width) // 2
        
        agent1_win_percent = stats.get_win_rate(1)
        agent2_win_percent = stats.get_win_rate(2)
        draw_percent = stats.get_draw_rate()
        
        draw_progress_bar(
            self.screen, bar_x, y_pos, bar_width, bar_height,
            [agent1_win_percent, draw_percent, agent2_win_percent],
            [GREEN, GRAY, RED]
        )
        
        # Legend
        y_pos += 80
        legend_y = y_pos
        legend_spacing = bar_width // 3
        draw_text(self.screen, f"{stats.agent1_name} Wins ({agent1_win_percent:.1f}%)",
                 bar_x, legend_y, self.small_font, DARK_GREEN)
        draw_text(self.screen, f"Draws ({draw_percent:.1f}%)",
                 bar_x + legend_spacing, legend_y, self.small_font, GRAY)
        draw_text(self.screen, f"{stats.agent2_name} Wins ({agent2_win_percent:.1f}%)",
                 bar_x + legend_spacing * 2, legend_y, self.small_font, DARK_RED)
        
        self.ended_button.draw(self.screen, self.normal_font)
    
    def draw_replay_select(self):
        """Draw replay selection screen"""
        draw_text(self.screen, "Select Tournament to Replay", self.screen_width // 2, 100,
                 self.header_font, BLACK, center=True)
        
        if not self.replay_tournaments:
            draw_text(self.screen, "No tournaments available", self.screen_width // 2, 400,
                     self.normal_font, GRAY, center=True)
        else:
            # Calculate visible range
            start_idx = self.replay_scroll_offset
            end_idx = min(start_idx + self.replay_max_visible, len(self.replay_buttons))
            
            # Adjust button positions based on scroll offset
            start_y = 150
            list_height = self.replay_max_visible * 70
            
            for i in range(start_idx, end_idx):
                button = self.replay_buttons[i]
                # Update button position
                button.rect.y = start_y + (i - start_idx) * 70
                button.draw(self.screen, self.normal_font)
            
            # Draw scrollbar if needed
            if len(self.replay_buttons) > self.replay_max_visible:
                scrollbar_x = self.screen_width // 2 + 320
                scrollbar_y = start_y
                scrollbar_height = list_height
                scrollbar_width = 10
                
                # Draw scrollbar background
                pygame.draw.rect(self.screen, LIGHT_GRAY, 
                               (scrollbar_x, scrollbar_y, scrollbar_width, scrollbar_height))
                
                # Draw scrollbar handle
                handle_height = max(30, int(scrollbar_height * self.replay_max_visible / len(self.replay_buttons)))
                handle_y = scrollbar_y + int((scrollbar_height - handle_height) * self.replay_scroll_offset / 
                                            max(1, len(self.replay_buttons) - self.replay_max_visible))
                pygame.draw.rect(self.screen, GRAY, 
                               (scrollbar_x, handle_y, scrollbar_width, handle_height))
                
                # Scroll indicator text
                scroll_text = f"{start_idx + 1}-{end_idx} of {len(self.replay_buttons)}"
                draw_text(self.screen, scroll_text, self.screen_width // 2, self.screen_height - 50,
                         self.small_font, GRAY, center=True)
        
        self.replay_back_button.draw(self.screen, self.normal_font)
    
    def draw_replay_summary(self):
        """Draw replay summary screen"""
        if not self.replay_data:
            return
        
        draw_text(self.screen, f"Tournament: {self.replay_data['name']}", 
                 self.screen_width // 2, 80, self.header_font, BLACK, center=True)
        
        y_pos = 150
        draw_text(self.screen, f"Total Matches: {self.replay_data['total_matches']}",
                 self.screen_width // 2, y_pos, self.normal_font, BLACK, center=True)
        
        y_pos += 40
        draw_text(self.screen, f"Time Taken: {self.replay_data['time_taken']:.2f} seconds",
                 self.screen_width // 2, y_pos, self.normal_font, BLACK, center=True)
        
        # Agent stats
        y_pos += 60
        agent1 = self.replay_data['agent1']
        draw_text(self.screen, f"{agent1['name']}: {agent1['wins']} wins ({agent1['win_rate']:.1f}%) | {agent1['losses']} losses",
                 self.screen_width // 2, y_pos, self.normal_font, BLUE, center=True)
        
        y_pos += 40
        agent2 = self.replay_data['agent2']
        draw_text(self.screen, f"{agent2['name']}: {agent2['wins']} wins ({agent2['win_rate']:.1f}%) | {agent2['losses']} losses",
                 self.screen_width // 2, y_pos, self.normal_font, RED, center=True)
        
        y_pos += 40
        draw_text(self.screen, f"Draws: {self.replay_data['draws']}",
                 self.screen_width // 2, y_pos, self.normal_font, BLACK, center=True)
        
        # Progress bar
        y_pos += 60
        bar_width = min(800, self.screen_width - 100)
        bar_height = 60
        bar_x = (self.screen_width - bar_width) // 2
        
        draw_progress_bar(
            self.screen, bar_x, y_pos, bar_width, bar_height,
            [agent1['win_rate'], self.replay_data['draws'] / self.replay_data['total_matches'] * 100, agent2['win_rate']],
            [GREEN, GRAY, RED]
        )
        
        # Game selection
        y_pos += 120  # Increased spacing
        draw_text(self.screen, "Select a game to replay:", self.screen_width // 2, y_pos,
                 self.normal_font, BLACK, center=True)
        
        # Calculate visible range
        start_idx = self.replay_summary_scroll_offset
        end_idx = min(start_idx + self.replay_summary_max_visible, len(self.replay_summary_buttons))
        
        # Adjust button positions based on scroll offset
        start_y = y_pos + 60  # Add padding after header
        list_height = self.replay_summary_max_visible * 50
        
        for i in range(start_idx, end_idx):
            button = self.replay_summary_buttons[i]
            # Update button position
            button.rect.y = start_y + (i - start_idx) * 50
            button.draw(self.screen, self.small_font)
        
        # Draw scrollbar if needed
        if len(self.replay_summary_buttons) > self.replay_summary_max_visible:
            scrollbar_x = self.screen_width // 2 + 215  # Adjusted for 400px wide buttons
            scrollbar_y = start_y
            scrollbar_height = list_height
            scrollbar_width = 10
            
            # Draw scrollbar background
            pygame.draw.rect(self.screen, LIGHT_GRAY, 
                           (scrollbar_x, scrollbar_y, scrollbar_width, scrollbar_height))
            
            # Draw scrollbar handle
            handle_height = max(30, int(scrollbar_height * self.replay_summary_max_visible / len(self.replay_summary_buttons)))
            handle_y = scrollbar_y + int((scrollbar_height - handle_height) * self.replay_summary_scroll_offset / 
                                        max(1, len(self.replay_summary_buttons) - self.replay_summary_max_visible))
            pygame.draw.rect(self.screen, GRAY, 
                           (scrollbar_x, handle_y, scrollbar_width, handle_height))
        
        self.replay_summary_back.draw(self.screen, self.normal_font)
    
    def draw_replay_game(self):
        """Draw replay game screen"""
        if not self.replay_data or self.replay_game_idx >= len(self.replay_data['games']):
            return
        
        game = self.replay_data['games'][self.replay_game_idx]
        
        # Load current board state
        fen = game['fens'][self.replay_move_idx]
        board = chess.Board(fen)
        
        # Draw info panel on left (20-30% of screen) with padding
        info_panel_width = int(self.screen_width * 0.25)
        padding_left = 20
        panel_center = padding_left + (info_panel_width - padding_left) // 2
        
        y_pos = 100
        draw_text(self.screen, f"Game {self.replay_game_idx + 1}/{len(self.replay_data['games'])}",
                 panel_center, y_pos, self.header_font, BLACK, center=True)
        
        y_pos += 80
        draw_text(self.screen, f"Result:", panel_center, y_pos, self.normal_font, BLACK, center=True)
        y_pos += 40
        # Color result based on winner
        result_text = game['result']
        if "White" in result_text:
            result_color = BLUE
        elif "Black" in result_text:
            result_color = RED
        else:  # Draw
            result_color = BLACK
        draw_text(self.screen, f"{result_text}", panel_center, y_pos, self.normal_font, result_color, center=True)
        
        y_pos += 70
        draw_text(self.screen, f"White:", panel_center, y_pos, self.small_font, BLACK, center=True)
        y_pos += 35
        draw_text(self.screen, f"{game['white']}", panel_center, y_pos, self.normal_font, BLUE, center=True)
        
        y_pos += 60
        draw_text(self.screen, f"Black:", panel_center, y_pos, self.small_font, BLACK, center=True)
        y_pos += 35
        draw_text(self.screen, f"{game['black']}", panel_center, y_pos, self.normal_font, RED, center=True)
        
        y_pos += 70
        # Check if at last move to show game end status
        is_last_move = self.replay_move_idx == len(game['fens']) - 1
        if is_last_move and board.is_game_over():
            if board.is_checkmate():
                status_text = "Checkmate!"
                status_color = GREEN
            else:
                status_text = "Draw!"
                status_color = (255, 165, 0)  # Orange
            draw_text(self.screen, status_text, panel_center, y_pos, self.normal_font, status_color, center=True)
        else:
            turn_text = "White's Turn" if board.turn == chess.WHITE else "Black's Turn"
            turn_color = BLUE if board.turn == chess.WHITE else RED
            draw_text(self.screen, turn_text, panel_center, y_pos, self.normal_font, turn_color, center=True)
        
        y_pos += 60
        draw_text(self.screen, f"Move: {self.replay_move_idx}/{len(game['fens']) - 1}",
                 panel_center, y_pos, self.normal_font, BLACK, center=True)
        
        # Keyboard shortcuts help
        y_pos += 100
        draw_text(self.screen, "Keyboard Shortcuts:", panel_center, y_pos, self.small_font, GRAY, center=True)
        y_pos += 35
        
        # Draw keyboard shortcuts with icons
        self._draw_keyboard_shortcut(panel_center, y_pos, ['left', 'right'], "Prev/Next")
        y_pos += 30
        self._draw_keyboard_shortcut(panel_center, y_pos, ['space'], "Play/Pause")
        y_pos += 30
        self._draw_keyboard_shortcut(panel_center, y_pos, ['up', 'down'], "Speed +/-")
        
        # Draw board (70-80% of screen) with right padding
        # Get last move if not at starting position
        last_move = None
        if self.replay_move_idx > 0:
            try:
                # Get move from UCI stored moves
                if 'moves_uci' in game and game['moves_uci']:
                    move_uci = game['moves_uci'][self.replay_move_idx - 1]
                    last_move = chess.Move.from_uci(move_uci)
                else:
                    # Fallback: Reconstruct the move from previous FEN to current FEN
                    prev_fen = game['fens'][self.replay_move_idx - 1]
                    prev_board = chess.Board(prev_fen)
                    
                    # Find the move that was made
                    for move in prev_board.legal_moves:
                        prev_board.push(move)
                        if prev_board.fen() == fen:
                            last_move = move
                            break
                        prev_board.pop()
            except:
                pass
        
        self.replay_board_renderer.draw(self.screen, board, self.piece_font, last_move)
        
        # Draw controls at top right
        for button in self.replay_game_buttons.values():
            button.draw(self.screen, self.normal_font)
        
        # Draw speed slider and label at top right
        # Position "Speed:" label to the left of slider, vertically centered
        speed_label_x = self.replay_speed_slider.rect.x - 60
        # Center text vertically with the slider track (10px tall)
        speed_label_y = self.replay_speed_slider.rect.y + 5  # Center of 10px track
        draw_text(self.screen, "Speed:", speed_label_x, speed_label_y, self.small_font)
        self.replay_speed_slider.draw(self.screen, self.small_font)
    
    def _draw_keyboard_shortcut(self, center_x: int, y: int, keys: list, description: str):
        """Draw keyboard shortcut with icons and description"""
        icon_size = 25
        spacing = 5
        
        # Calculate total width of icons
        total_icon_width = len(keys) * icon_size + (len(keys) - 1) * spacing
        
        # Check if we have the icons
        has_icons = all(key in self.keyboard_icons for key in keys)
        
        if has_icons:
            # Draw icons
            start_x = center_x - (total_icon_width + 10 + self.small_font.size(description)[0]) // 2
            current_x = start_x
            
            for i, key in enumerate(keys):
                icon = self.keyboard_icons[key]
                self.screen.blit(icon, (current_x, y - icon_size // 2))
                current_x += icon_size + spacing
            
            # Draw " : description" text
            text_x = start_x + total_icon_width + 10
            draw_text(self.screen, f": {description}", text_x, y, self.small_font, GRAY)
        else:
            # Fallback to text if icons not available
            if len(keys) == 2:
                key_text = f"{keys[0][0].upper()} {keys[1][0].upper()}"
            else:
                key_text = keys[0].capitalize()
            draw_text(self.screen, f"{key_text} : {description}", center_x, y, self.small_font, GRAY, center=True)
    
    def start_tournament(self):
        """Start a new tournament"""
        try:
            # Get agent types
            agent1_type = self.setup_dropdowns['agent1'].selected
            agent2_type = self.setup_dropdowns['agent2'].selected
            
            # Create agent names with postfixes if same type
            if agent1_type == agent2_type:
                agent1_name = f"{agent1_type}-01"
                agent2_name = f"{agent2_type}-02"
            else:
                agent1_name = agent1_type
                agent2_name = agent2_type
            
            # Create agents
            agent1 = create_agent(agent1_type, agent1_name)
            agent2 = create_agent(agent2_type, agent2_name)
            
            # Get number of matches
            num_matches_text = self.setup_inputs['num_matches'].text
            if not num_matches_text or not num_matches_text.isdigit():
                num_matches = 10
            else:
                num_matches = int(num_matches_text)
            
            # Get tournament name
            tournament_name = self.setup_inputs['tournament_name'].text or None
            
            # Create tournament
            self.tournament = Tournament(agent1, agent2, num_matches, tournament_name)
            self.init_tournament_running()
            
            # Initialize stats cache
            self.tournament_stats_cache = self.tournament.stats
            
            # Reset completion flag
            self.tournament_complete_flag = False
            
            # Start tournament in background thread
            self.tournament_thread = threading.Thread(
                target=self._run_tournament_thread,
                daemon=True
            )
            self.tournament_thread.start()
            
            # Immediately switch to running state (UI stays responsive)
            self.state = GameState.TOURNAMENT_RUNNING
            
        except Exception as e:
            print(f"Error starting tournament: {e}")
            self.state = GameState.MAIN_MENU
    
    def _run_tournament_thread(self):
        """Run tournament in background thread (non-UI thread)"""
        try:
            # Run tournament with thread-safe callback
            def update_callback(match_num, stats):
                # Thread-safe stats update
                with self.tournament_lock:
                    self.tournament_stats_cache = stats
            
            self.tournament.run(update_callback)
            self.tournament.save_results()
            
            # Signal completion (thread-safe)
            with self.tournament_lock:
                self.tournament_complete_flag = True
                
        except Exception as e:
            print(f"Error in tournament thread: {e}")
            # Signal completion even on error
            with self.tournament_lock:
                self.tournament_complete_flag = True
    
    def load_replay(self, tournament_name: str):
        """Load a tournament for replay"""
        self.replay_data = load_tournament(tournament_name)
        if self.replay_data:
            self.init_replay_summary()
            self.state = GameState.REPLAY_SUMMARY
        else:
            self.state = GameState.REPLAY_SELECT


if __name__ == "__main__":
    app = ChessTournamentApp()
    app.run()

