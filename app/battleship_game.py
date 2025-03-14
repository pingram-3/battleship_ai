import numpy as np
import random

class CellState:
    """Represents possible states of a cell"""
    EMPTY = 0
    MISS = -1
    HIT = 1
    SHIP = 2

class BattleshipGame:
    """Handles random game state generation and game management"""

    def __init__(self, grid_size=9):
        self.grid_size = grid_size
        self.ships = [4, 4, 4, 3, 3, 3, 3, 3] # 3 ships of size 4, 5 ships of 3
        self.board = np.zeros((grid_size, grid_size), dtype=np.int8)
        self.board.fill(CellState.EMPTY)
        self.ship_objects = []
    
    def place_ships(self):
        """Places ships randomly on the board without overlap"""
        self.ship_objects = []  # Reset ship objects
        
        # Try different ship placement orders if we get stuck
        max_retries = 5
        for retry in range(max_retries):
            try:
                # Clear the board for a fresh attempt
                if retry > 0:
                    self.board.fill(CellState.EMPTY)
                    self.ship_objects = []
                
                # Randomize ship order to avoid getting stuck in the same pattern
                ship_sizes = self.ships.copy()
                random.shuffle(ship_sizes)
                
                # Place each ship
                for ship_size in ship_sizes:
                    placed = False
                    attempts = 0
                    max_attempts = 200  # Increased max attempts
                    
                    while not placed and attempts < max_attempts:
                        attempts += 1
                        
                        # Randomly choose orientation (0 = horizontal, 1 = vertical)
                        orientation = random.randint(0, 1)
                        
                        # Get random coordinates based on orientation and ship size
                        if orientation == 0:  # Horizontal
                            row = random.randint(0, self.grid_size - 1)
                            col = random.randint(0, self.grid_size - ship_size)
                        else:  # Vertical
                            row = random.randint(0, self.grid_size - ship_size)
                            col = random.randint(0, self.grid_size - 1)
                        
                        # Check if the placement is valid
                        if self._is_valid_placement(row, col, ship_size, orientation):
                            # Place the ship on the board
                            self._place_ship(row, col, ship_size, orientation)
                            
                            # Add ship to ship_objects list (can be used for game logic)
                            ship_info = {
                                'size': ship_size,
                                'orientation': orientation,
                                'row': row,
                                'col': col,
                                'hits': 0  # Track hits on this ship
                            }
                            self.ship_objects.append(ship_info)
                            
                            placed = True
                    
                    # If we couldn't place the ship after max attempts, try a new board layout
                    if not placed:
                        raise ValueError("Could not place ship, retrying with new layout")
                
                # If we get here, all ships were placed successfully
                return
                
            except ValueError:
                # If we failed to place all ships, we'll retry with a new board layout
                continue
                
        # If we get here, we failed all retries
        raise RuntimeError(f"Could not place all ships after {max_retries} board layout attempts")
                    
    def generate_random_state(self):
        """Generates a random board state"""
        self.board.fill(CellState.EMPTY)
        self.place_ships()

        # Randomly decide if we should add hits (50% chance)
        add_hits = random.choice([True, False])
        
        if add_hits:
            # Add random hits on ships (10-30% of ship cells)
            ship_cells = [(r, c) for r in range(self.grid_size) for c in range(self.grid_size) 
                         if self.board[r, c] == CellState.SHIP]
            num_hits = random.randint(max(1, len(ship_cells) // 10), max(2, len(ship_cells) // 3))
            
            for _ in range(min(num_hits, len(ship_cells))):
                if not ship_cells:
                    break
                idx = random.randint(0, len(ship_cells) - 1)
                r, c = ship_cells.pop(idx)
                self.board[r, c] = CellState.HIT
                
                # Update hit count in ship_objects
                for ship in self.ship_objects:
                    if (ship['orientation'] == 0 and ship['row'] == r and 
                        ship['col'] <= c < ship['col'] + ship['size']):
                        ship['hits'] += 1
                        break
                    elif (ship['orientation'] == 1 and ship['col'] == c and 
                          ship['row'] <= r < ship['row'] + ship['size']):
                        ship['hits'] += 1
                        break

        # Randomly place a few misses (10-15% of empty cells)
        empty_cells = [(r, c) for r in range(self.grid_size) for c in range(self.grid_size) 
                       if self.board[r, c] == CellState.EMPTY]
        num_misses = random.randint(len(empty_cells) // 10, len(empty_cells) // 7)
        
        for _ in range(min(num_misses, len(empty_cells))):
            if not empty_cells:
                break
            idx = random.randint(0, len(empty_cells) - 1)
            r, c = empty_cells.pop(idx)
            self.board[r, c] = CellState.MISS

    
    def _is_valid_placement(self, row, col, ship_size, orientation):
        """
        Check if a ship can be placed at the given position
        
        Args:
            row (int): Starting row
            col (int): Starting column
            ship_size (int): Size of the ship
            orientation (int): 0 for horizontal, 1 for vertical
            
        Returns:
            bool: True if placement is valid, False otherwise
        """
        # Check if placement is within bounds
        if orientation == 0:  # Horizontal
            if col + ship_size > self.grid_size:
                return False
        else:  # Vertical
            if row + ship_size > self.grid_size:
                return False
        
        # Define the area to check (ship area plus one cell in each direction)
        start_row = max(0, row - 1)
        end_row = min(self.grid_size, row + (1 if orientation == 0 else ship_size) + 1)
        
        start_col = max(0, col - 1)
        end_col = min(self.grid_size, col + (ship_size if orientation == 0 else 1) + 1)
        
        # Check if any cell in this area already has a ship
        for r in range(start_row, end_row):
            for c in range(start_col, end_col):
                if self.board[r, c] == CellState.SHIP:
                    return False
        
        return True
    
    def _place_ship(self, row, col, ship_size, orientation):
        """
        Place a ship on the board
        
        Args:
            row (int): Starting row
            col (int): Starting column
            ship_size (int): Size of the ship
            orientation (int): 0 for horizontal, 1 for vertical
        """
        if orientation == 0:  # Horizontal
            for c in range(col, col + ship_size):
                self.board[row, c] = CellState.SHIP
        else:  # Vertical
            for r in range(row, row + ship_size):
                self.board[r, col] = CellState.SHIP
                
    def get_board(self):
        """Returns the current game board"""
        return self.board
    
    def get_hidden_board(self):
        """
        Returns a board with ships hidden (for AI training)
        Only shows hits and misses, not unhit ships
        """
        hidden_board = self.board.copy()
        # Replace SHIP cells with EMPTY
        hidden_board[hidden_board == CellState.SHIP] = CellState.EMPTY
        return hidden_board

    def display_board(self, hide_ships=False):
        """
        Prints the board for debugging
        
        Args:
            hide_ships (bool): If True, hides unhit ships (for AI training)
        """
        board_to_display = self.get_hidden_board() if hide_ships else self.board
        board_symbols = {CellState.EMPTY: '.', CellState.MISS: "0", CellState.HIT: "X", CellState.SHIP: "S"}
        for row in board_to_display:
            print(" ".join(board_symbols[cell] for cell in row))

# Example Usage
if __name__ == "__main__":
    game = BattleshipGame()
    game.generate_random_state()
    print("Full board (with ships):")
    game.display_board()
    print("\nHidden board (for AI training):")
    game.display_board(hide_ships=True)
