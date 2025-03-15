import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from app.battleship_game import BattleshipGame
from app.pingram_heatmap_gen import HeatmapGenerator

class PingramMetricsCollector:
    """Simple metrics collector for Pingram's heatmap algorithm"""
    
    def __init__(self, output_dir="metrics/pingram"):
        """Initialize metrics collector with a specific output directory"""
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def run_simulations(self, num_games=1000):
        """Run battleship simulations and collect accuracy metrics"""
        metrics = {
            "match_percentages": [],
            "true_positives": [],
            "false_positives": [],
            "true_negatives": [],
            "false_negatives": []
        }
        
        print(f"Running {num_games} simulations...")
        successful_games = 0
        attempts = 0
        max_attempts = num_games * 2  # Allow for some failed board generations
        
        while successful_games < num_games and attempts < max_attempts:
            try:
                # Create a new game with random state
                game = BattleshipGame()
                game.generate_random_state()
            except RuntimeError:
                attempts += 1
                continue
                
            # Generate heatmap
            heatmap_gen = HeatmapGenerator(game)
            heatmap = heatmap_gen.generate_heatmap()
            
            # Get actual ship positions
            actual_ships = np.zeros((game.grid_size, game.grid_size), dtype=int)
            full_board = heatmap_gen.full_board
            for r in range(game.grid_size):
                for c in range(game.grid_size):
                    if full_board[r, c] == 2:  # Ship cell
                        actual_ships[r, c] = 1
            
            # Convert probabilities to predictions (threshold at 0.5)
            predictions = (heatmap > 0.5).astype(int)
            
            # Calculate metrics
            tp = np.sum((predictions == 1) & (actual_ships == 1))
            fp = np.sum((predictions == 1) & (actual_ships == 0))
            tn = np.sum((predictions == 0) & (actual_ships == 0))
            fn = np.sum((predictions == 0) & (actual_ships == 1))
            
            # Calculate match percentage
            total_cells = game.grid_size * game.grid_size
            matches = np.sum(predictions == actual_ships)
            match_percentage = matches / total_cells
            
            # Store metrics
            metrics["match_percentages"].append(float(match_percentage))
            metrics["true_positives"].append(int(tp))
            metrics["false_positives"].append(int(fp))
            metrics["true_negatives"].append(int(tn))
            metrics["false_negatives"].append(int(fn))
            
            successful_games += 1
            if successful_games % 100 == 0:
                print(f"Completed {successful_games} games")
        
        if successful_games == 0:
            raise RuntimeError("Failed to generate any valid games")
            
        # Calculate aggregate metrics
        metrics["avg_match_percentage"] = np.mean(metrics["match_percentages"])
        metrics["std_match_percentage"] = np.std(metrics["match_percentages"])
        
        avg_tp = np.mean(metrics["true_positives"])
        avg_fp = np.mean(metrics["false_positives"])
        avg_tn = np.mean(metrics["true_negatives"])
        avg_fn = np.mean(metrics["false_negatives"])
        
        # Calculate precision, recall, and F1
        precision = avg_tp / (avg_tp + avg_fp) if (avg_tp + avg_fp) > 0 else 0
        recall = avg_tp / (avg_tp + avg_fn) if (avg_tp + avg_fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics["avg_metrics"] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1_score)
        }
        
        return metrics
    
    def save_metrics(self, metrics):
        """Save metrics to a JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pingram_metrics_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        save_data = {
            "timestamp": datetime.now().isoformat(),
            "avg_match_percentage": metrics["avg_match_percentage"],
            "std_match_percentage": metrics["std_match_percentage"],
            "precision": metrics["avg_metrics"]["precision"],
            "recall": metrics["avg_metrics"]["recall"],
            "f1_score": metrics["avg_metrics"]["f1_score"],
            "sample_matches": metrics["match_percentages"][:10]  # Save first 10 for reference
        }
        
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"\nMetrics saved to {filepath}")
        print("\nSummary:")
        print(f"  Average Match Percentage: {metrics['avg_match_percentage']:.2%} (±{metrics['std_match_percentage']:.2%})")
        print(f"  Precision: {metrics['avg_metrics']['precision']:.3f}")
        print(f"  Recall: {metrics['avg_metrics']['recall']:.3f}")
        print(f"  F1 Score: {metrics['avg_metrics']['f1_score']:.3f}")
        
        return filepath
        
    def generate_plots(self, metrics):
        """Generate and save visualization plots"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Plot match percentage distribution
        plt.figure(figsize=(10, 6))
        plt.hist(metrics["match_percentages"], bins=20, alpha=0.7, color='blue')
        plt.axvline(metrics["avg_match_percentage"], color='red', linestyle='dashed', linewidth=2)
        plt.title('Distribution of Match Percentages')
        plt.xlabel('Match Percentage')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # Add text annotation for the average
        plt.text(0.95, 0.95, f"Avg: {metrics['avg_match_percentage']:.2%}",
                transform=plt.gca().transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Save plot
        plot_path = os.path.join(self.output_dir, f"match_percentage_dist_{timestamp}.png")
        plt.savefig(plot_path)
        plt.close()
        
        return plot_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect accuracy metrics for Pingram heatmap algorithm')
    parser.add_argument('--games', type=int, default=1000, help='Number of games to simulate')
    args = parser.parse_args()
    
    collector = PingramMetricsCollector()
    metrics = collector.run_simulations(num_games=args.games)
    collector.save_metrics(metrics)
    collector.generate_plots(metrics)
