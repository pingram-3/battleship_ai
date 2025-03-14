import numpy as np
import torch
import random
from collections import defaultdict
from app.battleship_game import BattleshipGame, CellState

class HeatmapGenerator:
    """Generates probability heat maps based only on hits and misses"""

    def __init__(self, game):
        """
        Initialize the heatmap generator with a game state.
        
        Args:
            game (BattleshipGame): Game instance to analyze
        """
        self.game = game
        self.grid_size = game.grid_size
        self.board = game.get_hidden_board()  # Get board with ships hidden (only hits/misses visible)
        self.full_board = game.get_board()    # Get full board with ships for validation
        # Default ship configuration if not provided from game
        self.default_ships = [4, 4, 4, 3, 3, 3, 3, 3]  # 3 ships of size 4, 5 ships of size 3
        
        # Special state for sunken ships from BattleshipGame
        self.SUNKEN_SHIP = 3
        
        # Get remaining ships by analyzing what's already on the board
        self.remaining_ships = self._get_remaining_ships()
        
        # Identify hit patterns on initialization
        self.hit_patterns = self._identify_hit_patterns()
        
    def _get_remaining_ships(self):
        """
        Determine which ships are still available to place based on sunken ships.
        
        Returns:
            list: List of ship sizes that remain to be placed
        """
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
    
    def _identify_hit_patterns(self):
        """
        Identify connected hit patterns on the board.
        
        Returns:
            list: List of hit patterns with format {'start': (row, col), 'orientation': 0|1, 
                                                  'length': int, 'sunk': bool}
        """
        patterns = []
        visited = set()
        
        # Helper function to check if a cell has a hit
        def is_hit(r, c):
            return 0 <= r < self.grid_size and 0 <= c < self.grid_size and (
                self.board[r, c] == CellState.HIT or self.board[r, c] == self.SUNKEN_SHIP)
        
        # Scan for horizontal hit patterns
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if (r, c) not in visited and is_hit(r, c):
                    # Found start of a hit pattern
                    length = 0
                    is_sunk = True  # Assume sunk until we find a non-sunken hit
                    
                    # Check horizontally
                    for dc in range(self.grid_size - c):
                        if is_hit(r, c + dc):
                            length += 1
                            visited.add((r, c + dc))
                            if self.board[r, c + dc] == CellState.HIT:
                                is_sunk = False
                        else:
                            break
                    
                    if length > 1:  # Only consider patterns of at least 2 hits
                        patterns.append({
                            'start': (r, c),
                            'orientation': 0,  # Horizontal
                            'length': length,
                            'sunk': is_sunk
                        })
        
        # Scan for vertical hit patterns
        visited = set()
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if (r, c) not in visited and is_hit(r, c):
                    # Found start of a hit pattern
                    length = 0
                    is_sunk = True  # Assume sunk until we find a non-sunken hit
                    
                    # Check vertically
                    for dr in range(self.grid_size - r):
                        if is_hit(r + dr, c):
                            length += 1
                            visited.add((r + dr, c))
                            if self.board[r + dr, c] == CellState.HIT:
                                is_sunk = False
                        else:
                            break
                    
                    if length > 1:  # Only consider patterns of at least 2 hits
                        patterns.append({
                            'start': (r, c),
                            'orientation': 1,  # Vertical
                            'length': length,
                            'sunk': is_sunk
                        })
        
        # Add isolated hits as separate patterns
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if is_hit(r, c) and not any((r, c) in visited for visited in [set(p) for p in visited]):
                    patterns.append({
                        'start': (r, c),
                        'orientation': None,  # Unknown orientation for isolated hits
                        'length': 1,
                        'sunk': self.board[r, c] == self.SUNKEN_SHIP
                    })
        
        return patterns
    
    def _is_valid_placement(self, row, col, ship_size, orientation):
        """
        Check if a ship of given size can be legally placed at the specified position.
        
        Args:
            row (int): Starting row
            col (int): Starting column
            ship_size (int): Size of the ship
            orientation (int): 0 for horizontal, 1 for vertical
            
        Returns:
            bool: True if placement is valid, False otherwise
        """
        # Check bounds
        if orientation == 0:  # Horizontal
            if col + ship_size > self.grid_size:
                return False
        else:  # Vertical
            if row + ship_size > self.grid_size:
                return False
                
        # Early rejection for placements near too many misses - using stricter threshold
        if self._calculate_miss_proximity(row, col, ship_size, orientation) > 0.25:
            return False  # Reject placements surrounded by too many misses
        
        # Check the ship area including one cell buffer for spacing rules
        start_row = max(0, row - 1)
        end_row = min(self.grid_size, row + (1 if orientation == 0 else ship_size) + 1)
        
        start_col = max(0, col - 1)
        end_col = min(self.grid_size, col + (ship_size if orientation == 0 else 1) + 1)
        
        # Check for conflicts with misses and other constraints
        for r in range(start_row, end_row):
            for c in range(start_col, end_col):
                # Skip checking the cells where the ship would be placed
                if (orientation == 0 and r == row and col <= c < col + ship_size) or \
                   (orientation == 1 and c == col and row <= r < row + ship_size):
                    # If there's a miss or sunken ship where we want to place our ship, it's invalid
                    if self.board[r, c] == CellState.MISS or self.board[r, c] == self.SUNKEN_SHIP:
                        return False
                    # If it's a hit, that's compatible with ship placement
                    continue
                
                # For buffer cells (around the ship), check there are no hits or sunken ships
                if self.board[r, c] == CellState.HIT or self.board[r, c] == self.SUNKEN_SHIP:
                    return False
        
        # Now check for consistency with hit patterns
        # A valid ship placement must cover all hits along its path
        if orientation == 0:  # Horizontal
            for c_offset in range(ship_size):
                c_pos = col + c_offset
                # If we're placing over a non-hit cell, that's fine
                if self.board[row, c_pos] != CellState.HIT:
                    continue
                # If we're placing over a hit, check if we cover all connected horizontal hits
                # If there's a horizontal hit pattern here, we must cover all of it
                for pattern in self.hit_patterns:
                    if pattern['orientation'] == 0:  # Horizontal pattern
                        p_row, p_col = pattern['start']
                        if p_row == row and p_col <= c_pos < p_col + pattern['length']:
                            # We're intersecting with this pattern, we must cover all of it
                            if p_col < col or p_col + pattern['length'] > col + ship_size:
                                return False
                            # If the pattern is marked as sunk, but this ship size doesn't match,
                            # this is an invalid placement
                            if pattern['sunk'] and pattern['length'] != ship_size:
                                return False
        else:  # Vertical
            for r_offset in range(ship_size):
                r_pos = row + r_offset
                # If we're placing over a non-hit cell, that's fine
                if self.board[r_pos, col] != CellState.HIT:
                    continue
                # If we're placing over a hit, check if we cover all connected vertical hits
                for pattern in self.hit_patterns:
                    if pattern['orientation'] == 1:  # Vertical pattern
                        p_row, p_col = pattern['start']
                        if p_col == col and p_row <= r_pos < p_row + pattern['length']:
                            # We're intersecting with this pattern, we must cover all of it
                            if p_row < row or p_row + pattern['length'] > row + ship_size:
                                return False
                            # If the pattern is marked as sunk, but this ship size doesn't match,
                            # this is an invalid placement
                            if pattern['sunk'] and pattern['length'] != ship_size:
                                return False
        
        return True
    
    def _calculate_miss_proximity(self, row, col, ship_size, orientation):
        """
        Calculate the ratio of MISS cells adjacent to a potential ship placement.
        
        Args:
            row (int): Starting row
            col (int): Starting column
            ship_size (int): Size of the ship
            orientation (int): 0 for horizontal, 1 for vertical
            
        Returns:
            float: Ratio of adjacent cells that are misses (0.0 to 1.0)
        """
        miss_count = 0
        total_cells = 0
        
        # Define cells to check - only the ones adjacent to the ship itself
        if orientation == 0:  # Horizontal
            # Check cells adjacent to a horizontal ship
            
            # Check one row above the ship
            if row > 0:
                for c in range(max(0, col-1), min(self.grid_size, col+ship_size+1)):
                    total_cells += 1
                    if self.board[row-1, c] == CellState.MISS:
                        miss_count += 1
            
            # Check one row below the ship
            if row < self.grid_size - 1:
                for c in range(max(0, col-1), min(self.grid_size, col+ship_size+1)):
                    total_cells += 1
                    if self.board[row+1, c] == CellState.MISS:
                        miss_count += 1
            
            # Check cell to left of ship
            if col > 0:
                total_cells += 1
                if self.board[row, col-1] == CellState.MISS:
                    miss_count += 1
            
            # Check cell to right of ship
            if col + ship_size < self.grid_size:
                total_cells += 1
                if self.board[row, col+ship_size] == CellState.MISS:
                    miss_count += 1
                    
        else:  # Vertical
            # Check cells adjacent to a vertical ship
            
            # Check one column to the left of the ship
            if col > 0:
                for r in range(max(0, row-1), min(self.grid_size, row+ship_size+1)):
                    total_cells += 1
                    if self.board[r, col-1] == CellState.MISS:
                        miss_count += 1
            
            # Check one column to the right of the ship
            if col < self.grid_size - 1:
                for r in range(max(0, row-1), min(self.grid_size, row+ship_size+1)):
                    total_cells += 1
                    if self.board[r, col+1] == CellState.MISS:
                        miss_count += 1
            
            # Check cell above ship
            if row > 0:
                total_cells += 1
                if self.board[row-1, col] == CellState.MISS:
                    miss_count += 1
            
            # Check cell below ship
            if row + ship_size < self.grid_size:
                total_cells += 1
                if self.board[row+ship_size, col] == CellState.MISS:
                    miss_count += 1
        
        # Calculate and return ratio of misses to total adjacent cells
        return miss_count / max(total_cells, 1)
    
    def generate_heatmap(self):
        """
        Generate a probability heatmap for ship placements.
        
        Returns:
            torch.Tensor: 2D tensor with probabilities for each cell
        """
        # Initialize probability grid
        prob_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        
        # Determine targeting mode based on game state
        targeting_mode = self._determine_targeting_mode()
        
        # Create exclusion zones around sunken ships
        exclusion_mask = self._create_exclusion_zones()
        
        # Track valid placements for each ship size to calculate weighted probabilities
        valid_placements_by_size = defaultdict(int)
        total_placements = 0
        
        # Create grids to track various cell states
        must_contain_ship = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        is_isolated_hit = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        hit_influence = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        
        # Identify cells with hits and isolate hits for special processing
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if self.board[r, c] == CellState.HIT or self.board[r, c] == self.SUNKEN_SHIP:
                    must_contain_ship[r, c] = True
                    
                    # Check if this is an isolated hit (not part of a multi-cell pattern)
                    isolated = True
                    for pattern in self.hit_patterns:
                        p_row, p_col = pattern['start']
                        if pattern['length'] > 1:
                            if (pattern['orientation'] == 0 and  # Horizontal
                                p_row == r and p_col <= c < p_col + pattern['length']):
                                isolated = False
                                break
                            elif (pattern['orientation'] == 1 and  # Vertical
                                  p_col == c and p_row <= r < p_row + pattern['length']):
                                isolated = False
                                break
                    
                    if isolated and self.board[r, c] == CellState.HIT:
                        is_isolated_hit[r, c] = True
        
        # Create influence map from hit cells (higher values closer to hits)
        self._create_hit_influence_map(hit_influence, is_isolated_hit)
        
        # Create constraint mask for potential ship placements
        constraint_mask = self._create_placement_constraint_mask()
        
        # Ship size weights - favor larger ships significantly more
        ship_size_weights = {
            4: 2.5,  # Much stronger preference for 4-cell ships
            3: 1.0   # Base weight for 3-cell ships
        }
        
        # Get remaining ships by size
        remaining_by_size = self._get_remaining_ships_by_size()
        
        # For each remaining ship size
        for ship_size, count in remaining_by_size.items():
            # Track valid placements for this specific ship size
            size_placements = 0
            size_grid = np.zeros_like(prob_grid)
            
            # For each possible starting position
            for r in range(self.grid_size):
                for c in range(self.grid_size):
                    # Try horizontal placement
                    if self._is_valid_placement(r, c, ship_size, 0):
                        size_placements += 1
                        total_placements += 1
                        valid_placements_by_size[ship_size] += 1
                        for dc in range(ship_size):
                            size_grid[r, c + dc] += 1
                    
                    # Try vertical placement
                    if self._is_valid_placement(r, c, ship_size, 1):
                        size_placements += 1
                        total_placements += 1
                        valid_placements_by_size[ship_size] += 1
                        for dr in range(ship_size):
                            size_grid[r + dr, c] += 1
            
            # Weight by relative frequency of ship size and normalize
            if size_placements > 0:
                # Apply more significant size-based weighting and consider quantity of each ship type
                weight = ship_size_weights.get(ship_size, 1.0) * (count / max(sum(remaining_by_size.values()), 1))
                size_grid = size_grid * weight / size_placements
                prob_grid += size_grid
        
        # Apply hit influence to the probability grid - stronger for targeting mode
        influence_weight = 1.0 if targeting_mode == "broad_hunting" else 2.0
        prob_grid = prob_grid * (1.0 + hit_influence * influence_weight)
        
        # Enhance cells with confirmed hits
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if must_contain_ship[r, c]:
                    # Give extra weight to cells with confirmed hits
                    prob_grid[r, c] *= 2.0
                
                # Apply constraint mask
                prob_grid[r, c] *= constraint_mask[r, c]
        
        # Special handling for isolated hits - enhance adjacent cells significantly
        self._apply_isolated_hit_bias(prob_grid, is_isolated_hit)
        
        # Zero out probability for cells that we know can't contain ships
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if self.board[r, c] == CellState.MISS:
                    prob_grid[r, c] = 0.0
        
        # Apply exclusion zones around sunken ships
        prob_grid *= exclusion_mask
        
        # Apply hunting mode specific enhancements
        if targeting_mode in ["broad_hunting", "focused_hunting"]:
            # Calculate information gain (how many placements would be eliminated by a miss)
            info_gain = self._calculate_information_gain(prob_grid)
            
            if targeting_mode == "broad_hunting":
                # Probabilistic search strategy for broad hunting - higher emphasis on info gain
                # This heavily favors cells that would clear large portions of the board if missed
                ship_density = np.sum(prob_grid) / (prob_grid > 0).sum() if np.any(prob_grid > 0) else 1.0
                
                # Dynamic alpha based on how clustered the probability map is
                # More clustered = higher ship_density = lower alpha (focus on high probability)
                # More spread out = lower ship_density = higher alpha (more exploration)
                alpha = min(0.6, max(0.3, 0.9 - ship_density))
                
                # Square the probability grid to make peaks more pronounced
                prob_grid = np.power(prob_grid, 2)
                
                # Blend with information gain using dynamic alpha
                prob_grid = (1 - alpha) * prob_grid + alpha * info_gain
                
            else:  # focused_hunting
                # In focused hunting, we're more selective and put less emphasis on information gain
                # Balance between targeting high probability cells and clearing areas
                alpha = 0.15  # Lower alpha = more focus on likely ship locations
                
                # Apply stronger power to prob_grid to emphasize peaks
                prob_grid = np.power(prob_grid, 1.5)
                
                # Blend probability with information gain
                prob_grid = (1 - alpha) * prob_grid + alpha * info_gain
        
        # Normalize the probabilities if there were valid placements
        if total_placements > 0:
            prob_grid = prob_grid / np.max(prob_grid)  # Normalize to [0, 1] range
        
        # Convert to torch tensor for compatibility with neural network models
        prob_tensor = torch.tensor(prob_grid, dtype=torch.float32)
        
        return prob_tensor
    
    def _create_hit_influence_map(self, influence_map, is_isolated_hit):
        """
        Create a map showing influence of hit cells on surrounding cells.
        
        Args:
            influence_map: numpy array to store influence values
            is_isolated_hit: boolean mask of isolated hit cells
        """
        # Process multi-hit patterns first to strongly enhance directional influence
        self._enhance_directional_patterns(influence_map)
        
        # Create influence based on regular hits (lower impact)
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if self.board[r, c] == CellState.HIT and not is_isolated_hit[r, c]:
                    # Apply influence to surrounding cells with distance decay
                    for dr in range(-2, 3):
                        for dc in range(-2, 3):
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                                # Calculate Manhattan distance
                                dist = abs(dr) + abs(dc)
                                if dist > 0:  # Don't influence the hit cell itself
                                    # Influence decreases with distance
                                    influence_map[nr, nc] += max(0, 0.5 - 0.2 * dist)
        
        # Create stronger influence for isolated hits (higher impact)
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if is_isolated_hit[r, c]:
                    # For isolated hits, strongly bias in cardinal directions
                    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, down, left, up
                    
                    for dr, dc in directions:
                        # Check up to 3 cells in each direction
                        for distance in range(1, 4):
                            nr, nc = r + dr * distance, c + dc * distance
                            if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                                if self.board[nr, nc] == CellState.MISS:
                                    # Stop at misses
                                    break
                                if self.board[nr, nc] == CellState.HIT or self.board[nr, nc] == self.SUNKEN_SHIP:
                                    # Stop at hits but don't add influence (already has a hit)
                                    break
                                
                                # Add strong influence that decreases with distance
                                influence_map[nr, nc] += max(0, 1.0 - 0.25 * distance)
    
    def _enhance_directional_patterns(self, influence_map):
        """
        Drastically increase probability in the direction of adjacent hits.
        
        Args:
            influence_map: numpy array to store influence values
        """
        # Process each hit pattern of length 2 or more that isn't sunk
        for pattern in self.hit_patterns:
            if pattern['length'] >= 2 and not pattern['sunk']:
                r, c = pattern['start']
                length = pattern['length']
                
                # Calculate pattern-based multiplier - longer patterns get stronger influence
                # This creates a multiplicative effect instead of just addition
                multiplier = 3.0 + length  # 3x-7x boost depending on length
                
                # Apply strong directional bias based on orientation
                if pattern['orientation'] == 0:  # Horizontal pattern
                    # Get cells at both ends of the pattern
                    # Apply high influence to the left
                    left_col = c - 1
                    if left_col >= 0 and self.board[r, left_col] == CellState.EMPTY:
                        # Very strong probability - set directly for multiplier effect
                        influence_map[r, left_col] = multiplier
                    
                    # Apply high influence to the right
                    right_col = c + length
                    if right_col < self.grid_size and self.board[r, right_col] == CellState.EMPTY:
                        # Very strong probability - set directly for multiplier effect
                        influence_map[r, right_col] = multiplier
                
                elif pattern['orientation'] == 1:  # Vertical pattern
                    # Apply high influence above
                    top_row = r - 1
                    if top_row >= 0 and self.board[top_row, c] == CellState.EMPTY:
                        # Very strong probability - set directly for multiplier effect
                        influence_map[top_row, c] = multiplier
                    
                    # Apply high influence below
                    bottom_row = r + length
                    if bottom_row < self.grid_size and self.board[bottom_row, c] == CellState.EMPTY:
                        # Very strong probability - set directly for multiplier effect
                        influence_map[bottom_row, c] = multiplier
    
    def _create_placement_constraint_mask(self):
        """
        Create a mask to constrain placements based on game rules and current state.
        
        Returns:
            numpy.ndarray: Grid with values 0.0-1.0 (0=impossible, 1=no constraint)
        """
        mask = np.ones((self.grid_size, self.grid_size), dtype=np.float32)
        
        # Apply pattern-based constraints
        for pattern in self.hit_patterns:
            if pattern['length'] > 1:
                r, c = pattern['start']
                
                if pattern['orientation'] == 0:  # Horizontal pattern
                    # Check cells directly above and below the pattern
                    for dc in range(pattern['length']):
                        # Check cell above
                        if 0 <= r-1 < self.grid_size and 0 <= c+dc < self.grid_size:
                            mask[r-1, c+dc] *= 0.2  # Reduce probability
                        
                        # Check cell below
                        if 0 <= r+1 < self.grid_size and 0 <= c+dc < self.grid_size:
                            mask[r+1, c+dc] *= 0.2  # Reduce probability
                    
                    # Reduce probability for cells diagonally adjacent to the pattern
                    if 0 <= r-1 < self.grid_size and 0 <= c-1 < self.grid_size:
                        mask[r-1, c-1] *= 0.2
                    if 0 <= r-1 < self.grid_size and 0 <= c+pattern['length'] < self.grid_size:
                        mask[r-1, c+pattern['length']] *= 0.2
                    if 0 <= r+1 < self.grid_size and 0 <= c-1 < self.grid_size:
                        mask[r+1, c-1] *= 0.2
                    if 0 <= r+1 < self.grid_size and 0 <= c+pattern['length'] < self.grid_size:
                        mask[r+1, c+pattern['length']] *= 0.2
                
                elif pattern['orientation'] == 1:  # Vertical pattern
                    # Check cells to the left and right of the pattern
                    for dr in range(pattern['length']):
                        # Check cell to the left
                        if 0 <= r+dr < self.grid_size and 0 <= c-1 < self.grid_size:
                            mask[r+dr, c-1] *= 0.2  # Reduce probability
                        
                        # Check cell to the right
                        if 0 <= r+dr < self.grid_size and 0 <= c+1 < self.grid_size:
                            mask[r+dr, c+1] *= 0.2  # Reduce probability
                    
                    # Reduce probability for cells diagonally adjacent to the pattern
                    if 0 <= r-1 < self.grid_size and 0 <= c-1 < self.grid_size:
                        mask[r-1, c-1] *= 0.2
                    if 0 <= r-1 < self.grid_size and 0 <= c+1 < self.grid_size:
                        mask[r-1, c+1] *= 0.2
                    if 0 <= r+pattern['length'] < self.grid_size and 0 <= c-1 < self.grid_size:
                        mask[r+pattern['length'], c-1] *= 0.2
                    if 0 <= r+pattern['length'] < self.grid_size and 0 <= c+1 < self.grid_size:
                        mask[r+pattern['length'], c+1] *= 0.2
        
        return mask
    
    def _apply_isolated_hit_bias(self, prob_grid, is_isolated_hit):
        """
        Apply special bias for isolated hits as they create stronger directional patterns.
        
        Args:
            prob_grid (numpy.ndarray): The probability grid to modify
            is_isolated_hit (numpy.ndarray): Boolean mask of isolated hits
        """
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if is_isolated_hit[r, c]:
                    # For isolated hits, check if there are directional patterns
                    horizontal_clear = True
                    vertical_clear = True
                    
                    # Check horizontal clearance (at least 2 cells in either direction)
                    hor_clear_count = 0
                    for dc in range(1, 4):
                        if c+dc < self.grid_size and self.board[r, c+dc] != CellState.MISS:
                            hor_clear_count += 1
                        else:
                            break
                    for dc in range(1, 4):
                        if c-dc >= 0 and self.board[r, c-dc] != CellState.MISS:
                            hor_clear_count += 1
                        else:
                            break
                    horizontal_clear = hor_clear_count >= 2
                    
                    # Check vertical clearance (at least 2 cells in either direction)
                    ver_clear_count = 0
                    for dr in range(1, 4):
                        if r+dr < self.grid_size and self.board[r+dr, c] != CellState.MISS:
                            ver_clear_count += 1
                        else:
                            break
                    for dr in range(1, 4):
                        if r-dr >= 0 and self.board[r-dr, c] != CellState.MISS:
                            ver_clear_count += 1
                        else:
                            break
                    vertical_clear = ver_clear_count >= 2
                    
                    # Apply directional bias
                    if horizontal_clear and not vertical_clear:
                        # Horizontal ship is more likely
                        for dc in [-1, 1]:
                            if 0 <= c+dc < self.grid_size and self.board[r, c+dc] == CellState.EMPTY:
                                prob_grid[r, c+dc] *= 2.0
                    elif vertical_clear and not horizontal_clear:
                        # Vertical ship is more likely
                        for dr in [-1, 1]:
                            if 0 <= r+dr < self.grid_size and self.board[r+dr, c] == CellState.EMPTY:
                                prob_grid[r+dr, c] *= 2.0
                    elif horizontal_clear and vertical_clear:
                        # Both directions possible, slight boost to all adjacent
                        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
                        for dr, dc in directions:
                            adj_r, adj_c = r+dr, c+dc
                            if (0 <= adj_r < self.grid_size and 0 <= adj_c < self.grid_size and 
                                    self.board[adj_r, adj_c] == CellState.EMPTY):
                                prob_grid[adj_r, adj_c] *= 1.5
    
    def compare_with_actual_board(self, heatmap, threshold=0.5):
        """
        Compare heatmap predictions with the actual ship positions.
        
        Args:
            heatmap (torch.Tensor): The predicted probability heatmap
            threshold (float, optional): Probability threshold to consider a prediction positive
            
        Returns:
            dict: Dictionary with precision, recall, and match percentage metrics
        """
        # Convert tensor to numpy if needed
        if isinstance(heatmap, torch.Tensor):
            heatmap_np = heatmap.detach().numpy()
        else:
            heatmap_np = heatmap
        
        # Create binary prediction based on threshold
        predicted_ships = (heatmap_np > threshold).astype(np.int8)
        
        # Get actual ship positions from the full board
        actual_ships = (self.full_board == CellState.SHIP).astype(np.int8)
        actual_ships += (self.full_board == CellState.HIT).astype(np.int8)  # Include hits as well
        
        # Calculate metrics
        true_positives = np.sum((predicted_ships == 1) & (actual_ships == 1))
        false_positives = np.sum((predicted_ships == 1) & (actual_ships == 0))
        false_negatives = np.sum((predicted_ships == 0) & (actual_ships == 1))
        
        # Avoid division by zero
        precision = true_positives / max(true_positives + false_positives, 1)
        recall = true_positives / max(true_positives + false_negatives, 1)
        f1_score = 2 * precision * recall / max(precision + recall, 1e-8)
        
        # Calculate match percentage (accuracy)
        total_matches = np.sum(predicted_ships == actual_ships)
        match_percentage = total_matches / (self.grid_size * self.grid_size)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'match_percentage': match_percentage,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }
    
    def evaluate_by_game_state(self, heatmap, threshold=0.5):
        """
        Evaluate heatmap performance based on specific game state characteristics.
        
        This method analyzes how well the heatmap performs given specific game conditions
        such as number of hits, misses, and remaining ships.
        
        Args:
            heatmap (torch.Tensor): The predicted probability heatmap
            threshold (float, optional): Probability threshold to consider a prediction positive
            
        Returns:
            dict: Dictionary with performance metrics categorized by game state characteristics
        """
        # Get basic comparison metrics
        base_metrics = self.compare_with_actual_board(heatmap, threshold)
        
        # Count hits, misses, and remaining ships
        hit_count = 0
        miss_count = 0
        sunken_ship_count = 0
        
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if self.board[r, c] == CellState.HIT:
                    hit_count += 1
                elif self.board[r, c] == CellState.MISS:
                    miss_count += 1
                elif self.board[r, c] == self.SUNKEN_SHIP:
                    sunken_ship_count += 1
        
        # Calculate total hit cells (including sunken ships)
        total_hit_cells = hit_count + sunken_ship_count
        
        # Get remaining ships count
        remaining_ships_count = len(self.remaining_ships)
        
        # Calculate game progress percentage (based on revealed cells)
        total_cells = self.grid_size * self.grid_size
        revealed_cells = hit_count + miss_count + sunken_ship_count
        game_progress = revealed_cells / total_cells
        
        # Create game state characteristics
        game_state = {
            'hit_count': hit_count,
            'miss_count': miss_count,
            'sunken_ship_count': sunken_ship_count,
            'total_hit_cells': total_hit_cells,
            'remaining_ships_count': remaining_ships_count,
            'game_progress': game_progress,
            'hit_to_miss_ratio': hit_count / max(miss_count, 1),  # Avoid division by zero
            'targeting_mode': self._determine_targeting_mode()
        }
        
        # Return combined metrics
        return {
            'base_metrics': base_metrics,
            'game_state': game_state
        }
    
    def analyze_performance_by_category(self, heatmaps_and_states, threshold=0.5):
        """
        Analyze heatmap performance across different game state categories.
        
        This method takes a collection of heatmaps and their corresponding game states,
        and analyzes performance metrics grouped by different characteristics.
        
        Args:
            heatmaps_and_states (list): List of tuples (heatmap, game) for different game states
            threshold (float, optional): Probability threshold to consider a prediction positive
            
        Returns:
            dict: Performance metrics grouped by game state categories
        """
        # Initialize data structures to store metrics by category
        metrics_by_hit_count = defaultdict(list)
        metrics_by_miss_count = defaultdict(list)
        metrics_by_remaining_ships = defaultdict(list)
        metrics_by_progress = defaultdict(list)
        metrics_by_targeting_mode = defaultdict(list)
        
        # Process each heatmap and game state
        for heatmap, game in heatmaps_and_states:
            # Create a heatmap generator for this game
            heatmap_gen = HeatmapGenerator(game)
            
            # Evaluate performance for this game state
            evaluation = heatmap_gen.evaluate_by_game_state(heatmap, threshold)
            
            # Extract metrics and game state
            metrics = evaluation['base_metrics']
            state = evaluation['game_state']
            
            # Categorize by hit count (binned)
            hit_bin = min(10, state['hit_count'] // 5)  # Bin by groups of 5 hits, max 10 bins
            metrics_by_hit_count[hit_bin].append(metrics)
            
            # Categorize by miss count (binned)
            miss_bin = min(10, state['miss_count'] // 10)  # Bin by groups of 10 misses, max 10 bins
            metrics_by_miss_count[miss_bin].append(metrics)
            
            # Categorize by remaining ships
            metrics_by_remaining_ships[state['remaining_ships_count']].append(metrics)
            
            # Categorize by game progress (binned into 10 progress levels)
            progress_bin = min(9, int(state['game_progress'] * 10))
            metrics_by_progress[progress_bin].append(metrics)
            
            # Categorize by targeting mode
            metrics_by_targeting_mode[state['targeting_mode']].append(metrics)
        
        # Calculate average metrics for each category
        results = {
            'by_hit_count': self._calculate_category_averages(metrics_by_hit_count),
            'by_miss_count': self._calculate_category_averages(metrics_by_miss_count),
            'by_remaining_ships': self._calculate_category_averages(metrics_by_remaining_ships),
            'by_game_progress': self._calculate_category_averages(metrics_by_progress),
            'by_targeting_mode': self._calculate_category_averages(metrics_by_targeting_mode)
        }
        
        return results
    
    def _calculate_category_averages(self, metrics_by_category):
        """
        Calculate average metrics for each category.
        
        Args:
            metrics_by_category (dict): Dictionary mapping categories to lists of metrics
            
        Returns:
            dict: Dictionary with average metrics for each category
        """
        category_averages = {}
        
        for category, metrics_list in metrics_by_category.items():
            if not metrics_list:
                continue
                
            # Initialize accumulators
            avg_metrics = {
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'match_percentage': 0.0,
                'sample_count': len(metrics_list)
            }
            
            # Sum metrics across all samples in this category
            for metrics in metrics_list:
                avg_metrics['precision'] += metrics['precision']
                avg_metrics['recall'] += metrics['recall']
                avg_metrics['f1_score'] += metrics['f1_score']
                avg_metrics['match_percentage'] += metrics['match_percentage']
            
            # Calculate averages
            count = len(metrics_list)
            avg_metrics['precision'] /= count
            avg_metrics['recall'] /= count
            avg_metrics['f1_score'] /= count
            avg_metrics['match_percentage'] /= count
            
            category_averages[category] = avg_metrics
        
        return category_averages

    def _get_remaining_ships_by_size(self):
        """
        Count remaining ships by size for more accurate probability calculation.
        
        Returns:
            dict: Dictionary with ship sizes as keys and counts as values
        """
        # Get the list of remaining ships
        remaining = self.remaining_ships
        
        # Count by size
        size_counts = defaultdict(int)
        for size in remaining:
            size_counts[size] += 1
        
        return size_counts
        
    def _determine_targeting_mode(self):
        """
        Determine whether to use hunting mode or targeting mode based on game state.
        
        Returns:
            str: "targeting", "focused_hunting", or "broad_hunting"
        """
        # Count unsunk hits (those marked as 'X' not 'S')
        unsunk_hit_count = 0
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if self.board[r, c] == CellState.HIT:
                    unsunk_hit_count += 1
        
        # Get count of remaining ships
        remaining_ship_count = len(self.remaining_ships)
        
        # If we have active hits, use targeting mode
        if unsunk_hit_count > 0:
            return "targeting"
        # If few ships remain, use focused hunting
        elif remaining_ship_count <= 3:
            return "focused_hunting"
        # Otherwise, use broad hunting mode to maximize information gain
        else:
            return "broad_hunting"
    
    def _create_exclusion_zones(self):
        """
        Create a mask with exclusion zones around sunken ships.
        
        Returns:
            numpy.ndarray: Grid mask with 0.0 for exclusion zones, 1.0 elsewhere
        """
        exclusion_mask = np.ones((self.grid_size, self.grid_size), dtype=np.float32)
        
        # Process each pattern marked as sunk
        for pattern in self.hit_patterns:
            if pattern['sunk']:
                r, c = pattern['start']
                length = pattern['length']
                orientation = pattern['orientation']
                
                # Create exclusion zone around the sunken ship
                buffer = 1  # Buffer distance around ship
                
                start_row = max(0, r - buffer)
                end_row = min(self.grid_size, r + (buffer if orientation == 0 else length) + buffer)
                
                start_col = max(0, c - buffer)
                end_col = min(self.grid_size, c + (length if orientation == 0 else buffer) + buffer)
                
                # Set exclusion zone - completely remove from probability consideration
                for er in range(start_row, end_row):
                    for ec in range(start_col, end_col):
                        # Keep the sunken ship cells as-is (for display purposes)
                        if not ((orientation == 0 and er == r and c <= ec < c + length) or
                                (orientation == 1 and ec == c and r <= er < r + length)):
                            exclusion_mask[er, ec] = 0.0
        
        return exclusion_mask
    
    def _calculate_information_gain(self, prob_grid):
        """
        Calculate the information gain potential for each cell.
        
        This estimates how much a miss at this location would reduce the search space,
        helping prioritize cells that would eliminate large areas if they're misses.
        
        Args:
            prob_grid: Current probability grid
            
        Returns:
            numpy.ndarray: Information gain grid
        """
        info_gain = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        
        # Skip cells that are already hits, misses, or have zero probability
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if self.board[r, c] != CellState.EMPTY or prob_grid[r, c] == 0:
                    continue
                
                # Estimate affected area if this is a miss
                affected_area = self._estimate_affected_area(r, c)
                
                # Calculate information gain as affected area × (1 - probability)
                # Higher value = more cells would be eliminated by a miss here, weighted by the
                # likelihood that this is actually a miss (1 - probability)
                info_gain[r, c] = affected_area * (1 - prob_grid[r, c])
        
        # Normalize
        if np.max(info_gain) > 0:
            info_gain = info_gain / np.max(info_gain)
        
        return info_gain
    
    def _estimate_affected_area(self, row, col):
        """
        Estimate how many cells would have reduced probability if this cell is a miss.
        
        Args:
            row (int): Row coordinate
            col (int): Column coordinate
            
        Returns:
            float: Estimate of affected area
        """
        affected_count = 0
        
        # For each remaining ship size
        for ship_size in set(self.remaining_ships):  # Use set to count each size once
            # Count nearby cells that could be part of a ship placement through this cell
            
            # Check horizontal placements that include this cell
            for offset in range(min(ship_size, col + 1)):
                start_col = col - offset
                
                # If this placement is potentially valid
                if start_col + ship_size <= self.grid_size:
                    # Quick check for obvious blockers
                    has_blocker = False
                    for c_pos in range(start_col, start_col + ship_size):
                        if c_pos != col and self.board[row, c_pos] == CellState.MISS:
                            has_blocker = True
                            break
                    
                    if not has_blocker:
                        # For each valid placement, add the ship size to affected count
                        affected_count += ship_size
            
            # Check vertical placements that include this cell
            for offset in range(min(ship_size, row + 1)):
                start_row = row - offset
                
                # If this placement is potentially valid
                if start_row + ship_size <= self.grid_size:
                    # Quick check for obvious blockers
                    has_blocker = False
                    for r_pos in range(start_row, start_row + ship_size):
                        if r_pos != row and self.board[r_pos, col] == CellState.MISS:
                            has_blocker = True
                            break
                    
                    if not has_blocker:
                        # For each valid placement, add the ship size to affected count
                        affected_count += ship_size
        
        return affected_count

if __name__ == "__main__":
    # Create a new game with random state
    game = BattleshipGame()
    game.generate_random_state()
    
    # Display the game board (with ships hidden)
    print("Game Board (AI's view - ships hidden):")
    game.display_board(hide_ships=True)
    
    # Create heatmap generator and generate heatmap
    heatmap_gen = HeatmapGenerator(game)
    heatmap = heatmap_gen.generate_heatmap()
    
    # Display the heatmap
    print("\nGenerated Probability Heat Map:")
    print(heatmap.numpy())
    
    # Compare with actual ship positions
    metrics = heatmap_gen.compare_with_actual_board(heatmap)
    match_percentage = metrics['match_percentage']
    
    print(f"\nMatch percentage (threshold 0.5): {match_percentage:.2%}")
    
    # Display actual ship positions
    print("\nActual Ship Positions:")
    game.display_board(hide_ships=False)
