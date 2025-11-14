"""
UI Components for Chess Tournament
"""
import pygame
import chess
from typing import Optional, Callable, Tuple


# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
LIGHT_GRAY = (200, 200, 200)
DARK_GRAY = (50, 50, 50)
GREEN = (34, 139, 34)
RED = (220, 20, 60)
BLUE = (70, 130, 180)
LIGHT_BLUE = (135, 206, 250)
DARK_GREEN = (0, 100, 0)
DARK_RED = (139, 0, 0)

# Chess board colors
BOARD_LIGHT = (241, 225, 207)  # #f1e1cf
BOARD_DARK = (182, 124, 105)   # #b67c69


class Button:
    """Simple button component"""
    
    def __init__(self, x: int, y: int, width: int, height: int, text: str, 
                 color: Tuple[int, int, int] = BLUE, text_color: Tuple[int, int, int] = WHITE,
                 icon_path: str = None):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.text_color = text_color
        self.hover_color = tuple(min(c + 30, 255) for c in color)
        self.is_hovered = False
        
        # Load icon if provided
        self.icon = None
        if icon_path:
            try:
                import os
                if os.path.exists(icon_path):
                    icon_image = pygame.image.load(icon_path)
                    # Scale icon to fit button with padding
                    icon_size = min(width - 10, height - 10)
                    self.icon = pygame.transform.smoothscale(icon_image, (icon_size, icon_size))
            except Exception as e:
                print(f"Could not load button icon {icon_path}: {e}")
    
    def draw(self, screen: pygame.Surface, font: pygame.font.Font):
        color = self.hover_color if self.is_hovered else self.color
        pygame.draw.rect(screen, color, self.rect, border_radius=5)
        pygame.draw.rect(screen, BLACK, self.rect, 2, border_radius=5)
        
        if self.icon:
            # Draw icon centered
            icon_x = self.rect.centerx - self.icon.get_width() // 2
            icon_y = self.rect.centery - self.icon.get_height() // 2
            screen.blit(self.icon, (icon_x, icon_y))
        else:
            # Draw text as fallback
            text_surface = font.render(self.text, True, self.text_color)
            text_rect = text_surface.get_rect(center=self.rect.center)
            screen.blit(text_surface, text_rect)
    
    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                return True
        return False


class Dropdown:
    """Dropdown menu component"""
    
    def __init__(self, x: int, y: int, width: int, height: int, options: list, default: str = None):
        self.rect = pygame.Rect(x, y, width, height)
        self.options = options
        self.selected = default if default in options else (options[0] if options else "")
        self.is_open = False
        self.option_rects = []
        self.hovered_option = None
    
    def draw(self, screen: pygame.Surface, font: pygame.font.Font):
        # Draw main box
        pygame.draw.rect(screen, WHITE, self.rect)
        pygame.draw.rect(screen, BLACK, self.rect, 2)
        
        text_surface = font.render(self.selected, True, BLACK)
        screen.blit(text_surface, (self.rect.x + 10, self.rect.y + 10))
        
        # Draw arrow
        arrow_x = self.rect.x + self.rect.width - 20
        arrow_y = self.rect.y + self.rect.height // 2
        pygame.draw.polygon(screen, BLACK, [
            (arrow_x, arrow_y - 5),
            (arrow_x + 10, arrow_y - 5),
            (arrow_x + 5, arrow_y + 5)
        ])
        
        # Draw options if open (with semi-transparent background overlay)
        if self.is_open:
            # Get mouse position for hover detection
            mouse_pos = pygame.mouse.get_pos()
            self.hovered_option = None
            
            self.option_rects = []
            for i, option in enumerate(self.options):
                opt_rect = pygame.Rect(
                    self.rect.x,
                    self.rect.y + self.rect.height + i * self.rect.height,
                    self.rect.width,
                    self.rect.height
                )
                self.option_rects.append((opt_rect, option))
                
                # Check if mouse is hovering over this option
                is_hovered = opt_rect.collidepoint(mouse_pos)
                if is_hovered:
                    self.hovered_option = option
                
                # Determine background color: grey for hover, light grey for selected, white for normal
                if is_hovered:
                    color = GRAY
                elif option == self.selected:
                    color = LIGHT_GRAY
                else:
                    color = WHITE
                
                pygame.draw.rect(screen, color, opt_rect)
                pygame.draw.rect(screen, BLACK, opt_rect, 1)
                
                text_surface = font.render(option, True, BLACK)
                screen.blit(text_surface, (opt_rect.x + 10, opt_rect.y + 10))
    
    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.is_open = not self.is_open
                return True
            elif self.is_open:
                for opt_rect, option in self.option_rects:
                    if opt_rect.collidepoint(event.pos):
                        self.selected = option
                        self.is_open = False
                        return True
                self.is_open = False
        return False


class TextInput:
    """Text input component"""
    
    def __init__(self, x: int, y: int, width: int, height: int, placeholder: str = ""):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = ""
        self.placeholder = placeholder
        self.active = False
    
    def draw(self, screen: pygame.Surface, font: pygame.font.Font):
        color = LIGHT_BLUE if self.active else WHITE
        pygame.draw.rect(screen, color, self.rect)
        pygame.draw.rect(screen, BLACK, self.rect, 2)
        
        display_text = self.text if self.text else self.placeholder
        text_color = BLACK if self.text else GRAY
        
        # Truncate text if too long
        text_surface = font.render(display_text, True, text_color)
        if text_surface.get_width() > self.rect.width - 20:
            # Truncate and add ellipsis
            while len(display_text) > 0 and text_surface.get_width() > self.rect.width - 20:
                display_text = display_text[:-1]
                text_surface = font.render(display_text + "...", True, text_color)
        
        screen.blit(text_surface, (self.rect.x + 10, self.rect.y + (self.rect.height - text_surface.get_height()) // 2))
    
    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.active = self.rect.collidepoint(event.pos)
            return self.active
        elif event.type == pygame.KEYDOWN and self.active:
            if event.key == pygame.K_RETURN:
                self.active = False
                return True
            elif event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            else:
                self.text += event.unicode
            return True
        return False


class ChessBoardRenderer:
    """Renders a chess board"""
    
    def __init__(self, x: int, y: int, size: int):
        self.x = x
        self.y = y
        self.size = size
        self.square_size = size // 8
        
        # Store original images (not scaled)
        self.original_images = {}
        self.piece_images = {}
        self.use_images = True
        
        try:
            import os
            # Piece mapping: piece symbol -> image filename
            self.piece_files = {
                'P': 'white_pawn.png',
                'N': 'white_knight.png',
                'B': 'white_bishop.png',
                'R': 'white_rook.png',
                'Q': 'white_queen.png',
                'K': 'white_king.png',
                'p': 'black_pawn.png',
                'n': 'black_knight.png',
                'b': 'black_bishop.png',
                'r': 'black_rook.png',
                'q': 'black_queen.png',
                'k': 'black_king.png'
            }
            
            # Load original images
            for symbol, filename in self.piece_files.items():
                image_path = os.path.join('assets/img/chess_pieces/', filename)
                if os.path.exists(image_path):
                    image = pygame.image.load(image_path)
                    self.original_images[symbol] = image
                else:
                    self.use_images = False
                    break
            
            # Scale images to current size
            if self.use_images:
                self._scale_images()
                
        except Exception as e:
            print(f"Could not load piece images: {e}")
            self.use_images = False
        
        # Fallback to Unicode symbols if images can't be loaded
        if not self.use_images:
            self.piece_symbols = {
                'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
                'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚'
            }
    
    def _scale_images(self):
        """Scale images to fit current square size"""
        piece_size = int(self.square_size * 0.8)
        for symbol, original_image in self.original_images.items():
            scaled_image = pygame.transform.smoothscale(
                original_image, 
                (piece_size, piece_size)
            )
            self.piece_images[symbol] = scaled_image
    
    def draw(self, screen: pygame.Surface, board: chess.Board, font: pygame.font.Font, last_move: chess.Move = None):
        # Draw squares
        for row in range(8):
            for col in range(8):
                color = BOARD_LIGHT if (row + col) % 2 == 0 else BOARD_DARK
                rect = pygame.Rect(
                    self.x + col * self.square_size,
                    self.y + row * self.square_size,
                    self.square_size,
                    self.square_size
                )
                pygame.draw.rect(screen, color, rect)
        
        # Draw last move highlight if provided
        if last_move:
            self._draw_move_highlight(screen, last_move, board)
        
        # Draw pieces
        for row in range(8):
            for col in range(8):
                square = chess.square(col, 7 - row)
                piece = board.piece_at(square)
                
                if piece:
                    piece_x = self.x + col * self.square_size
                    piece_y = self.y + row * self.square_size
                    
                    if self.use_images and piece.symbol() in self.piece_images:
                        # Draw image
                        image = self.piece_images[piece.symbol()]
                        # Center the image in the square
                        offset = (self.square_size - image.get_width()) // 2
                        screen.blit(image, (piece_x + offset, piece_y + offset))
                    else:
                        # Fallback to text
                        symbol = self.piece_symbols.get(piece.symbol(), piece.symbol())
                        text_surface = font.render(symbol, True, BLACK if piece.color == chess.WHITE else RED)
                        text_rect = text_surface.get_rect(center=(
                            piece_x + self.square_size // 2,
                            piece_y + self.square_size // 2
                        ))
                        screen.blit(text_surface, text_rect)
        
        # Draw border
        pygame.draw.rect(screen, BLACK, (self.x, self.y, self.size, self.size), 3)
    
    def _draw_move_highlight(self, screen: pygame.Surface, move: chess.Move, board: chess.Board):
        """Draw highlight for last move showing start, end, and path"""
        from_square = move.from_square
        to_square = move.to_square
        
        # Get file and rank for both squares
        from_col = chess.square_file(from_square)
        from_row = 7 - chess.square_rank(from_square)
        to_col = chess.square_file(to_square)
        to_row = 7 - chess.square_rank(to_square)
        
        # Highlight color (semi-transparent yellow)
        highlight_color = (255, 255, 0, 128)  # Yellow with alpha
        path_color = (255, 255, 150, 80)  # Lighter yellow for path
        
        # Create a temporary surface with alpha for transparency
        temp_surface = pygame.Surface((self.square_size, self.square_size), pygame.SRCALPHA)
        
        # Draw 'from' square highlight
        temp_surface.fill(highlight_color)
        screen.blit(temp_surface, (
            self.x + from_col * self.square_size,
            self.y + from_row * self.square_size
        ))
        
        # Draw 'to' square highlight
        temp_surface.fill(highlight_color)
        screen.blit(temp_surface, (
            self.x + to_col * self.square_size,
            self.y + to_row * self.square_size
        ))
        
        # Draw path if it's a straight line or diagonal (not knight move)
        col_diff = abs(to_col - from_col)
        row_diff = abs(to_row - from_row)
        
        # Check if it's a straight line or diagonal (not knight)
        is_straight = (col_diff == 0 or row_diff == 0 or col_diff == row_diff)
        is_knight = (col_diff == 2 and row_diff == 1) or (col_diff == 1 and row_diff == 2)
        
        if is_straight and not is_knight and (col_diff > 1 or row_diff > 1):
            # Calculate direction
            col_step = 0 if col_diff == 0 else (1 if to_col > from_col else -1)
            row_step = 0 if row_diff == 0 else (1 if to_row > from_row else -1)
            
            # Draw path squares
            current_col = from_col + col_step
            current_row = from_row + row_step
            temp_path_surface = pygame.Surface((self.square_size, self.square_size), pygame.SRCALPHA)
            temp_path_surface.fill(path_color)
            
            while current_col != to_col or current_row != to_row:
                screen.blit(temp_path_surface, (
                    self.x + current_col * self.square_size,
                    self.y + current_row * self.square_size
                ))
                current_col += col_step
                current_row += row_step


class Slider:
    """Slider component for speed control"""
    
    def __init__(self, x: int, y: int, width: int, min_val: float, max_val: float, initial: float):
        self.rect = pygame.Rect(x, y, width, 10)
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial
        self.dragging = False
        
        # Calculate handle position
        self.handle_radius = 8
        self.update_handle_pos()
    
    def update_handle_pos(self):
        ratio = (self.value - self.min_val) / (self.max_val - self.min_val)
        self.handle_x = self.rect.x + int(ratio * self.rect.width)
    
    def draw(self, screen: pygame.Surface, font: pygame.font.Font):
        # Draw track
        pygame.draw.rect(screen, GRAY, self.rect)
        pygame.draw.rect(screen, BLACK, self.rect, 2)
        
        # Draw handle
        pygame.draw.circle(screen, BLUE, (self.handle_x, self.rect.centery), self.handle_radius)
        pygame.draw.circle(screen, BLACK, (self.handle_x, self.rect.centery), self.handle_radius, 2)
        
        # Draw value
        text = f"{self.value:.1f}x"
        text_surface = font.render(text, True, BLACK)
        screen.blit(text_surface, (self.rect.x + self.rect.width + 10, self.rect.y - 5))
    
    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN:
            handle_rect = pygame.Rect(
                self.handle_x - self.handle_radius,
                self.rect.centery - self.handle_radius,
                self.handle_radius * 2,
                self.handle_radius * 2
            )
            if handle_rect.collidepoint(event.pos):
                self.dragging = True
                return True
        elif event.type == pygame.MOUSEBUTTONUP:
            if self.dragging:
                self.dragging = False
                return True
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            # Update value based on mouse position
            ratio = max(0, min(1, (event.pos[0] - self.rect.x) / self.rect.width))
            self.value = self.min_val + ratio * (self.max_val - self.min_val)
            self.update_handle_pos()
            return True
        return False


def draw_text(screen: pygame.Surface, text: str, x: int, y: int, 
              font: pygame.font.Font, color: Tuple[int, int, int] = BLACK, center: bool = False):
    """Helper function to draw text"""
    text_surface = font.render(text, True, color)
    if center:
        text_rect = text_surface.get_rect(center=(x, y))
        screen.blit(text_surface, text_rect)
    else:
        screen.blit(text_surface, (x, y))


def draw_progress_bar(screen: pygame.Surface, x: int, y: int, width: int, height: int, 
                     percentages: list, colors: list):
    """Draw a segmented progress bar"""
    # Draw border
    pygame.draw.rect(screen, BLACK, (x, y, width, height), 2)
    
    # Draw segments
    current_x = x
    for percent, color in zip(percentages, colors):
        segment_width = int((percent / 100) * width)
        if segment_width > 0:
            pygame.draw.rect(screen, color, (current_x, y, segment_width, height))
            current_x += segment_width

