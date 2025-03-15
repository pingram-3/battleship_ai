"""
Battleship Game - Random State Generator

This module provides a Battleship game implementation focused on generating random board states
for AI training. It handles ship placement following standard Battleship rules and can simulate
games in progress with hits and misses.

Key Features:
- Random ship placement with proper spacing (no ships touching, even diagonally)
- Option to generate boards with random hits and misses
- Ability to hide ship positions for AI training
- Configurable grid size
- Support for different ship sizes and quantities

Example:
    ```python
    # Create a new game with default 9x9 grid
    game = BattleshipGame()
    
    # Generate a random board state with ships, possibly some hits and misses
    game.generate_random_state()
    
    # Display the full board (including ships)
    game.display_board()
    
    # Display board with ships hidden (for AI)
    game.display_board(hide_ships=True)
    
    # Get the numpy array representing the board
    board = game.get_board()
    
    # Get board with ships hidden for AI training
    hidden_board = game.get_hidden_board()
    ```
"""

import numpy as np
import random

class CellState:
    """
    Represents possible states of a cell on the game board.
    
    Attributes:
        EMPTY (int): Represents an empty cell (0)
        MISS (int): Represents a missed shot (-1)
        HIT (int): Represents a hit on a ship (1)
        SHIP (int): Represents an undamaged ship cell (2)
    """
    EMPTY = 0
    MISS = -1
    HIT = 1
    SHIP = 2

class BattleshipGame:
    """
    Handles random game state generation and game management for Battleship.
    
    This class provides functionality to create and manage Battleship game boards,
    particularly focused on generating random states for AI training. It ensures
    proper ship placement according to standard Battleship rules where ships cannot
    touch (even diagonally).

    Attributes:
        grid_size (int): Size of the square game board (default: 9)
        ships (list): List of ship sizes to place (default: 3 ships of size 4, 5 ships of size 3)
        board (numpy.ndarray): The game board as a 2D numpy array
        ship_objects (list): List of dictionaries containing ship information

    The board uses the following cell states:
        0 (EMPTY): Empty cell
        -1 (MISS): Missed shot
        1 (HIT): Hit on a ship
        2 (SHIP): Undamaged ship cell
    """

    def __init__(self, grid_size=9):
        """
        Initialize a new Battleship game.

        Args:
            grid_size (int, optional): Size of the square game board. Defaults to 9.
        """
        self.grid_size = grid_size
        self.ships = [4, 4, 4, 3, 3, 3, 3, 3] # 3 ships of size 4, 5 ships of 3
        self.board = np.zeros((grid_size, grid_size), dtype=np.int8)
        self.board.fill(CellState.EMPTY)
        self.ship_objects = []
    
    def place_ships(self):
        """
        Places ships randomly on the board without overlap or adjacency.
        
        This method attempts to place all ships in the self.ships list randomly on the board.
        Ships cannot overlap or touch each other (even diagonally). If placement becomes
        impossible with the current layout, it will retry with a new board layout.
        
        Raises:
            RuntimeError: If unable to place all ships after maximum retries.
        """
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
        """
        Generates a random board state.
        
        This method:
        1. Places all ships randomly on the board
        2. Has a 50% chance to add random hits on ships (10-30% of ship cells)
        3. Adds random misses (10-15% of empty cells)
        
        The resulting board represents a game in progress, which can be used for AI training.
        """
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
        Check if a ship can be placed at the given position.
        
        This method ensures:
        1. The ship stays within the board boundaries
        2. The ship doesn't overlap with other ships
        3. The ship doesn't touch other ships (even diagonally)
        
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
        Place a ship on the board.
        
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
        """
        Returns the current game board.
        
        Returns:
            numpy.ndarray: 2D array representing the current board state
        """
        return self.board
    
    def get_hidden_board(self):
        """
        Returns a board with ships hidden (for AI training).
        
        This method returns a copy of the board where unhit ships are hidden
        (replaced with EMPTY). Only hits and misses remain visible.
        
        Returns:
            numpy.ndarray: 2D array representing the hidden board state
        """
        hidden_board = self.board.copy()
        
        # First mark sunken ships
        self._update_sunken_ships_display(hidden_board)
        
        # Replace non-sunken SHIP cells with EMPTY
        hidden_board[hidden_board == CellState.SHIP] = CellState.EMPTY
        
        return hidden_board

    def display_board(self, hide_ships=False):
        """
        Prints the board for debugging.
        
        Args:
            hide_ships (bool, optional): If True, hides unhit ships (for AI training).
                                      Defaults to False.
        
        Board symbols:
            '.' : Empty cell
            '0' : Miss
            'X' : Hit (on a ship that is not completely sunk)
            'S' : Ship (visible only if hide_ships is False) or hit on a sunken ship
        """
        # Get the board to display
        board_to_display = self.get_hidden_board().copy() if hide_ships else self.board.copy()
        
        # Update the display to show sunken ships with 'S' instead of 'X'
        self._update_sunken_ships_display(board_to_display)
        
        # Define board symbols (adding special state 3 for sunken ship hits)
        board_symbols = {CellState.EMPTY: '.', CellState.MISS: "0", CellState.HIT: "X", CellState.SHIP: "S", 3: "S"}
        
        for row in board_to_display:
            print(" ".join(board_symbols[cell] for cell in row))
    
    def _update_sunken_ships_display(self, board):
        """
        Update the board display to show sunken ships with 'S' markers.
        
        This method checks each ship to see if it's completely sunk (all cells hit).
        If a ship is sunk, it updates all its hit cells to a special state (3) that
        will be displayed as 'S'.
        
        Args:
            board (numpy.ndarray): The board to update
        """
        # Check each ship to see if it's sunk
        for ship in self.ship_objects:
            # A ship is sunk if the number of hits equals its size
            if ship['hits'] == ship['size']:
                # Update all cells of this sunken ship to the special state (3)
                if ship['orientation'] == 0:  # Horizontal
                    for c in range(ship['col'], ship['col'] + ship['size']):
                        # Only update cells that are hits (to avoid affecting empty cells)
                        if board[ship['row'], c] == CellState.HIT:
                            board[ship['row'], c] = 3  # Special state for sunken ship
                else:  # Vertical
                    for r in range(ship['row'], ship['row'] + ship['size']):
                        # Only update cells that are hits (to avoid affecting empty cells)
                        if board[r, ship['col']] == CellState.HIT:
                            board[r, ship['col']] = 3  # Special state for sunken ship

# Example Usage
if __name__ == "__main__":
    game = BattleshipGame()
    game.generate_random_state()
    print("Full board (with ships):")
    game.display_board()
    print("\nHidden board (for AI training):")
    game.display_board(hide_ships=True)
