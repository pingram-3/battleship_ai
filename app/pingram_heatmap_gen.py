import numpy as np
import torch
import random
from collections import defaultdict
from app.battleship_game import BattleshipGame, CellState

class HeatmapGenerator:
    """Pingram's personal algorithm for generating heatmaps"""

    def __init__(self, game):
        self.game = game
        self.grid_size = game.grid_size
        self.board = game.get_hidden_board()
        self.full_board = game.get_board()
        self.default_ships = [4, 4, 4, 3, 3, 3, 3]
        self.SUNKEN_SHIP = 3  # Special state for sunken ships
        self.remaining_ships = self._get_remaining_ships()
    
    def _get_remaining_ships(self):
        """
        Determine which ships are still available based on sunken ships
        
        Returns:
            list: List of ship sizes that remain"""
        
        # Start with default ship configuration
        remaining = self.default_ships.copy()
        
        # Find all confirmed sunken ships (marked with special value 3)
        sunken_patterns = []
        visited = set()
        
        # Scan for horizontal sunken ships
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if (r, c) not in visited and self.board[r, c] == self.SUNKEN_SHIP:
                    # Found start of a sunken ship
                    ship_size = 0
                    # Check horizontally
                    for dc in range(self.grid_size - c):
                        if c + dc < self.grid_size and self.board[r, c + dc] == self.SUNKEN_SHIP:
                            ship_size += 1
                            visited.add((r, c + dc))
                        else:
                            break
                            
                    if ship_size > 0:
                        sunken_patterns.append(ship_size)
        
        # Scan for vertical sunken ships (for any cells not already counted)
        visited = set()
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if (r, c) not in visited and self.board[r, c] == self.SUNKEN_SHIP:
                    # Found start of a sunken ship
                    ship_size = 0
                    # Check vertically
                    for dr in range(self.grid_size - r):
                        if r + dr < self.grid_size and self.board[r + dr, c] == self.SUNKEN_SHIP:
                            ship_size += 1
                            visited.add((r + dr, c))
                        else:
                            break
                            
                    if ship_size > 0:
                        sunken_patterns.append(ship_size)
        
        # Remove sunken ships from the remaining ships list
        for size in sunken_patterns:
            if size in remaining:
                remaining.remove(size)
        
        return remaining

    def _try_fit_ships_at_position(self, prob_grid, r, c):
        """Try fitting all possible remaining ships through position (r,c)"""
        
        if self.board[r, c] in [-1, 3]:  # Skip if position is a miss or sunken ship
            return prob_grid

        for ship_size in self.remaining_ships:
            # Try horizontal placement
            for dc in range(-(ship_size - 1), ship_size):
                if 0 <= c + dc < self.grid_size and 0 <= c + dc + ship_size - 1 < self.grid_size:
                    # Check if we can place the ship horizontally
                    valid = True
                    # Check ship cells and their adjacent cells
                    for i in range(ship_size):
                        cur_c = c + dc + i
                        
                        # Check the ship cell itself
                        if cur_c >= self.grid_size or self.board[r, cur_c] in [-1, 3]:
                            valid = False
                            break
                        
                        # Check cells above and below
                        if r > 0 and self.board[r-1, cur_c] in [1, 3]:  # Above
                            valid = False
                            break
                        if r < self.grid_size-1 and self.board[r+1, cur_c] in [1, 3]:  # Below
                            valid = False
                            break
                        
                        # For start of ship, check left side
                        if i == 0:
                            if cur_c > 0:
                                # Check left
                                if self.board[r, cur_c-1] in [1, 3]:
                                    valid = False
                                    break
                                # Check diagonal left-up
                                if r > 0 and self.board[r-1, cur_c-1] in [1, 3]:
                                    valid = False
                                    break
                                # Check diagonal left-down
                                if r < self.grid_size-1 and self.board[r+1, cur_c-1] in [1, 3]:
                                    valid = False
                                    break
                        
                        # For end of ship, check right side
                        if i == ship_size-1:
                            if cur_c < self.grid_size-1:
                                # Check right
                                if self.board[r, cur_c+1] in [1, 3]:
                                    valid = False
                                    break
                                # Check diagonal right-up
                                if r > 0 and self.board[r-1, cur_c+1] in [1, 3]:
                                    valid = False
                                    break
                                # Check diagonal right-down
                                if r < self.grid_size-1 and self.board[r+1, cur_c+1] in [1, 3]:
                                    valid = False
                                    break
                    if valid:
                        # Increment all cells the ship would occupy
                        for i in range(ship_size):
                            prob_grid[r, c + dc + i] += 1

            # Try vertical placement
            for dr in range(-(ship_size - 1), ship_size):
                if 0 <= r + dr < self.grid_size and 0 <= r + dr + ship_size - 1 < self.grid_size:
                    # Check if we can place the ship vertically
                    valid = True
                    # Check ship cells and their adjacent cells
                    for i in range(ship_size):
                        cur_r = r + dr + i
                        
                        # Check the ship cell itself
                        if cur_r >= self.grid_size or self.board[cur_r, c] in [-1, 3]:
                            valid = False
                            break
                        
                        # Check cells left and right
                        if c > 0 and self.board[cur_r, c-1] in [1, 3]:  # Left
                            valid = False
                            break
                        if c < self.grid_size-1 and self.board[cur_r, c+1] in [1, 3]:  # Right
                            valid = False
                            break
                        
                        # For start of ship, check top side
                        if i == 0:
                            if cur_r > 0:
                                # Check top
                                if self.board[cur_r-1, c] in [1, 3]:
                                    valid = False
                                    break
                                # Check diagonal top-left
                                if c > 0 and self.board[cur_r-1, c-1] in [1, 3]:
                                    valid = False
                                    break
                                # Check diagonal top-right
                                if c < self.grid_size-1 and self.board[cur_r-1, c+1] in [1, 3]:
                                    valid = False
                                    break
                        
                        # For end of ship, check bottom side
                        if i == ship_size-1:
                            if cur_r < self.grid_size-1:
                                # Check bottom
                                if self.board[cur_r+1, c] in [1, 3]:
                                    valid = False
                                    break
                                # Check diagonal bottom-left
                                if c > 0 and self.board[cur_r+1, c-1] in [1, 3]:
                                    valid = False
                                    break
                                # Check diagonal bottom-right
                                if c < self.grid_size-1 and self.board[cur_r+1, c+1] in [1, 3]:
                                    valid = False
                                    break
                    if valid:
                        # Increment all cells the ship would occupy
                        for i in range(ship_size):
                            prob_grid[r + dr + i, c] += 1

        return prob_grid
    
    def generate_heatmap(self):
        prob_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        # First pass: Checkerboard pattern
        start_offset = 1  # Start with 1 so first row starts with 0
        for r in range(self.grid_size):
            start_offset = 1 - start_offset  # Toggle at start of row
            for c in range(start_offset, self.grid_size, 2):  # Skip every other column
                prob_grid = self._try_fit_ships_at_position(prob_grid, r, c)

        # Second pass: Try fitting ships through hits
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if self.board[r, c] == 1:  # Hit
                    prob_grid = self._try_fit_ships_at_position(prob_grid, r, c)
        
        # Normalize probabilities to be between 0 and 1
        max_val = prob_grid.max()
        if max_val > 0:
            prob_grid /= max_val

        # Zero out probabilities for known hits, misses, and sunken ships since we don't need to shoot there
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if self.board[r, c] in [-1, 1, 3]:  # Miss, Hit, or Sunken ship
                    prob_grid[r, c] = 0
            
        return prob_grid
    

if __name__ == "__main__":
    game = BattleshipGame()
    game.generate_random_state()

    print("Game Board (AI's view - ships hidden):")
    game.display_board(hide_ships=True)
    
    heatmap_gen = HeatmapGenerator(game)
    heatmap = heatmap_gen.generate_heatmap()

    # Display the heatmap
    print("\nGenerated Probability Heat Map:")
    for r in range(game.grid_size):
        row = ""
        for c in range(game.grid_size):
            row += f"{heatmap[r,c]:.3f} "
        print(row)

    # Display actual ship positions
    print("\nActual Ship Positions:")
    game.display_board(hide_ships=False)
