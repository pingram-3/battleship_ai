import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from app.battleship_game import BattleshipGame
from app.pingram_heatmap_gen import HeatmapGenerator

class MetricsCollector:
    """Collects and saves metrics for evaluating heatmap performance"""
    
    def __init__(self, output_dir="metrics"):
        """
        Initialize metrics collector
        
        Args:
            output_dir (str): Directory to save metrics results
        """
        self.output_dir = output_dir
        # Create metrics directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
    def run_simulations(self, num_games=100000):
        """
        Run battleship game simulations and collect metrics
        
        Args:
            num_games (int): Number of games to simulate
            
        Returns:
            dict: Collected metrics
        """
        metrics = {
            "match_percentages": [],
            "ship_cell_avg_prob": [],
            "empty_cell_avg_prob": [],
            "hit_rate_high_prob": [],
            "board_configs": [],
            "game_states": []  # Store game state characteristics
        }
        
        # For category-based analysis
        heatmaps_and_games = []
        
        # Run simulations
        successful_games = 0
        attempts = 0
        max_attempts = num_games * 2  # Allow for some failed board generations
        
        while successful_games < num_games and attempts < max_attempts:
            attempts += 1
            try:
                # Create a new game with random state
                game = BattleshipGame()
                game.generate_random_state()
                
                # Create heatmap generator and generate heatmap
                heatmap_gen = HeatmapGenerator(game)
                heatmap = heatmap_gen.generate_heatmap()
                
                successful_games += 1
            except RuntimeError as e:
                # Skip this game if we can't generate a valid board layout
                print(f"Skipping game due to error: {e}")
                continue
            
            # Get the heatmap as a numpy array
            heatmap_np = heatmap.numpy()
            
            # Get detailed evaluation including game state characteristics
            evaluation = heatmap_gen.evaluate_by_game_state(heatmap)
            comparison = evaluation['base_metrics']
            game_state = evaluation['game_state']
            
            match_percentage = comparison['match_percentage']
            
            # Get actual ship positions directly from the full board
            full_board = heatmap_gen.full_board
            actual_ships = (full_board == 2).astype(np.int8)  # Ship cells
            actual_ships += (full_board == 1).astype(np.int8)  # Hit cells
            
            # Calculate additional metrics
            ship_cell_probs = []
            empty_cell_probs = []
            
            for r in range(game.grid_size):
                for c in range(game.grid_size):
                    if actual_ships[r, c] == 1:  # Ship cell
                        ship_cell_probs.append(heatmap_np[r, c])
                    else:  # Empty cell
                        empty_cell_probs.append(heatmap_np[r, c])
            
            ship_cell_avg_prob = np.mean(ship_cell_probs) if ship_cell_probs else 0
            empty_cell_avg_prob = np.mean(empty_cell_probs) if empty_cell_probs else 0
            
            # Calculate hit rate for high probability cells (>0.5)
            high_prob_cells = (heatmap_np > 0.5).astype(int)
            high_prob_hits = np.sum(high_prob_cells * actual_ships)
            high_prob_total = np.sum(high_prob_cells)
            hit_rate = high_prob_hits / high_prob_total if high_prob_total > 0 else 0
            
            # Save metrics for this game
            metrics["match_percentages"].append(float(match_percentage))
            metrics["ship_cell_avg_prob"].append(float(ship_cell_avg_prob))
            metrics["empty_cell_avg_prob"].append(float(empty_cell_avg_prob))
            metrics["hit_rate_high_prob"].append(float(hit_rate))
            metrics["game_states"].append(game_state)
            
            # Save board configuration
            board_config = {
                "ships": actual_ships.tolist(),
                "heatmap": heatmap_np.tolist()
            }
            metrics["board_configs"].append(board_config)
            
            # Store for category analysis
            heatmaps_and_games.append((heatmap, game))
            
            # Print progress
            if successful_games % 10 == 0:
                print(f"Processed {successful_games}/{num_games} games")
        
        # Calculate aggregate metrics
        metrics["avg_match_percentage"] = np.mean(metrics["match_percentages"])
        metrics["avg_ship_cell_prob"] = np.mean(metrics["ship_cell_avg_prob"])
        metrics["avg_empty_cell_prob"] = np.mean(metrics["empty_cell_avg_prob"])
        metrics["avg_hit_rate_high_prob"] = np.mean(metrics["hit_rate_high_prob"])
        
        # Analyze performance by categories
        if successful_games > 0:
            print("Analyzing performance by game state categories...")
            # Use the first heatmap generator to analyze all collected data
            category_analysis = HeatmapGenerator(heatmaps_and_games[0][1]).analyze_performance_by_category(heatmaps_and_games)
            metrics["category_analysis"] = category_analysis
        
        return metrics
    
    def save_metrics(self, metrics, filename=None):
        """
        Save metrics to a file and ensure only the most recent two metrics files are kept.
        
        Args:
            metrics (dict): Metrics to save
            filename (str, optional): Custom filename
            
        Returns:
            str: Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"heatmap_metrics_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Save only the aggregate metrics and some samples for file size management
        save_data = {
            "timestamp": datetime.now().isoformat(),
            "avg_match_percentage": metrics["avg_match_percentage"],
            "avg_ship_cell_prob": metrics["avg_ship_cell_prob"],
            "avg_empty_cell_prob": metrics["avg_empty_cell_prob"],
            "avg_hit_rate_high_prob": metrics["avg_hit_rate_high_prob"],
            "num_games": len(metrics["match_percentages"]),
            "sample_match_percentages": metrics["match_percentages"][:5],  # Just save the first 5 for reference
            "board_samples": metrics["board_configs"][:3]  # Save a few board examples
        }
        
        # Include category analysis if available
        if "category_analysis" in metrics:
            save_data["category_analysis"] = metrics["category_analysis"]
        
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        # Clean up old metrics files to keep only the most recent two
        self._cleanup_old_metrics_files()
        
        print(f"Metrics saved to {filepath}")
        return filepath
        
    def _cleanup_old_metrics_files(self):
        """
        Keep only the most recent two metrics files and their associated plots.
        """
        # Clean up metrics JSON files
        metrics_files = [f for f in os.listdir(self.output_dir) 
                        if f.startswith("heatmap_metrics_") and f.endswith(".json")]
        
        # If we have more than 2 files, delete the oldest ones
        if len(metrics_files) > 2:
            # Sort files by timestamp (assuming filename format)
            metrics_files.sort()
            
            # Get timestamps of files to keep (the two most recent)
            keep_timestamps = []
            for recent_file in metrics_files[-2:]:
                # Extract timestamp from filename (format: heatmap_metrics_YYYYMMDD_HHMMSS.json)
                timestamp = recent_file.replace("heatmap_metrics_", "").replace(".json", "")
                keep_timestamps.append(timestamp)
            
            # Remove all but the two most recent metric files
            for old_file in metrics_files[:-2]:
                old_path = os.path.join(self.output_dir, old_file)
                try:
                    os.remove(old_path)
                    print(f"Removed old metrics file: {old_path}")
                except OSError as e:
                    print(f"Error removing old metrics file {old_path}: {e}")
            
            # Clean up associated plot files and comparison files
            all_files = os.listdir(self.output_dir)
            for filename in all_files:
                # Check if file is a plot or comparison file but not one of the recent ones we want to keep
                is_plot = (filename.startswith("key_metrics_") or 
                          filename.startswith("match_percentage_dist_")) and filename.endswith(".png")
                is_comparison = filename.startswith("metrics_comparison_") and filename.endswith(".json")
                
                if (is_plot or is_comparison) and not any(timestamp in filename for timestamp in keep_timestamps):
                    file_path = os.path.join(self.output_dir, filename)
                    try:
                        os.remove(file_path)
                        print(f"Removed associated file: {file_path}")
                    except OSError as e:
                        print(f"Error removing file {file_path}: {e}")
    
    def generate_plots(self, metrics, save_dir=None):
        """
        Generate and save plots for metrics visualization
        
        Args:
            metrics (dict): Metrics to visualize
            save_dir (str, optional): Directory to save plots
            
        Returns:
            list: Paths to saved plot files
        """
        if save_dir is None:
            save_dir = self.output_dir
        
        os.makedirs(save_dir, exist_ok=True)
        saved_files = []
        
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
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(save_dir, f"match_percentage_dist_{timestamp}.png")
        plt.savefig(plot_path)
        saved_files.append(plot_path)
        plt.close()
        
        # Bar chart of key metrics
        plt.figure(figsize=(10, 6))
        metrics_names = ['Avg Match %', 'Avg Ship Cell Prob', 'Avg Empty Cell Prob', 'High Prob Hit Rate']
        metrics_values = [
            metrics["avg_match_percentage"],
            metrics["avg_ship_cell_prob"],
            metrics["avg_empty_cell_prob"],
            metrics["avg_hit_rate_high_prob"]
        ]
        
        plt.bar(metrics_names, metrics_values, color=['blue', 'green', 'red', 'purple'])
        plt.title('Key Performance Metrics')
        plt.ylabel('Value')
        plt.grid(True, axis='y', alpha=0.3)
        
        # Add value labels on top of the bars
        for i, v in enumerate(metrics_values):
            plt.text(i, v + 0.02, f"{v:.3f}", ha='center')
        
        # Save plot
        plot_path = os.path.join(save_dir, f"key_metrics_{timestamp}.png")
        plt.savefig(plot_path)
        saved_files.append(plot_path)
        plt.close()
        
        # Generate category-based plots if available
        if "category_analysis" in metrics:
            category_plots = self.generate_category_plots(metrics["category_analysis"], save_dir, timestamp)
            saved_files.extend(category_plots)
        
        print(f"Plots saved to {save_dir}")
        return saved_files
    
    def generate_category_plots(self, category_analysis, save_dir, timestamp=None):
        """
        Generate plots for category-based metrics analysis
        
        Args:
            category_analysis (dict): Category analysis data
            save_dir (str): Directory to save plots
            timestamp (str, optional): Timestamp for filenames
            
        Returns:
            list: Paths to saved plot files
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        saved_files = []
        
        # Plot performance by hit count
        if "by_hit_count" in category_analysis and category_analysis["by_hit_count"]:
            plt.figure(figsize=(12, 8))
            
            # Extract data - ensure we're using string keys consistently
            categories = sorted([int(k) for k in category_analysis["by_hit_count"].keys()])
            match_percentages = []
            precisions = []
            recalls = []
            f1_scores = []
            sample_counts = []
            
            for k in categories:
                k_str = str(k)
                if k_str in category_analysis["by_hit_count"]:
                    match_percentages.append(category_analysis["by_hit_count"][k_str]["match_percentage"])
                    precisions.append(category_analysis["by_hit_count"][k_str]["precision"])
                    recalls.append(category_analysis["by_hit_count"][k_str]["recall"])
                    f1_scores.append(category_analysis["by_hit_count"][k_str]["f1_score"])
                    sample_counts.append(category_analysis["by_hit_count"][k_str]["sample_count"])
            
            # Create x-axis labels (hit count bins)
            x_labels = [f"{k*5}-{(k+1)*5-1}" for k in categories]
            x = np.arange(len(x_labels))
            
            # Only plot if we have data
            if len(match_percentages) > 0:
                # Plot metrics
                bar_width = 0.2
                plt.bar(x - bar_width*1.5, match_percentages, bar_width, label='Match %', color='blue')
                plt.bar(x - bar_width*0.5, precisions, bar_width, label='Precision', color='green')
                plt.bar(x + bar_width*0.5, recalls, bar_width, label='Recall', color='red')
                plt.bar(x + bar_width*1.5, f1_scores, bar_width, label='F1 Score', color='purple')
                
                # Add sample count as text above bars
                for i, count in enumerate(sample_counts):
                    plt.text(x[i], max(match_percentages[i], precisions[i], recalls[i], f1_scores[i]) + 0.05, 
                             f"n={count}", ha='center', va='bottom', fontsize=8)
                
                plt.xlabel('Hit Count Range')
                plt.ylabel('Score')
                plt.title('Heatmap Performance by Hit Count')
                plt.xticks(x, x_labels)
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Save plot
                plot_path = os.path.join(save_dir, f"perf_by_hit_count_{timestamp}.png")
                plt.savefig(plot_path)
                saved_files.append(plot_path)
            else:
                print("No data available for hit count plot")
            
            plt.close()
        
        # Plot performance by miss count
        if "by_miss_count" in category_analysis and category_analysis["by_miss_count"]:
            plt.figure(figsize=(12, 8))
            
            # Extract data - ensure we're using string keys consistently
            categories = sorted([int(k) for k in category_analysis["by_miss_count"].keys()])
            match_percentages = []
            precisions = []
            recalls = []
            f1_scores = []
            sample_counts = []
            
            for k in categories:
                k_str = str(k)
                if k_str in category_analysis["by_miss_count"]:
                    match_percentages.append(category_analysis["by_miss_count"][k_str]["match_percentage"])
                    precisions.append(category_analysis["by_miss_count"][k_str]["precision"])
                    recalls.append(category_analysis["by_miss_count"][k_str]["recall"])
                    f1_scores.append(category_analysis["by_miss_count"][k_str]["f1_score"])
                    sample_counts.append(category_analysis["by_miss_count"][k_str]["sample_count"])
            
            # Create x-axis labels (miss count bins)
            x_labels = [f"{k*10}-{(k+1)*10-1}" for k in categories]
            x = np.arange(len(x_labels))
            
            # Only plot if we have data
            if len(match_percentages) > 0:
                # Plot metrics
                bar_width = 0.2
                plt.bar(x - bar_width*1.5, match_percentages, bar_width, label='Match %', color='blue')
                plt.bar(x - bar_width*0.5, precisions, bar_width, label='Precision', color='green')
                plt.bar(x + bar_width*0.5, recalls, bar_width, label='Recall', color='red')
                plt.bar(x + bar_width*1.5, f1_scores, bar_width, label='F1 Score', color='purple')
                
                # Add sample count as text above bars
                for i, count in enumerate(sample_counts):
                    plt.text(x[i], max(match_percentages[i], precisions[i], recalls[i], f1_scores[i]) + 0.05, 
                             f"n={count}", ha='center', va='bottom', fontsize=8)
                
                plt.xlabel('Miss Count Range')
                plt.ylabel('Score')
                plt.title('Heatmap Performance by Miss Count')
                plt.xticks(x, x_labels)
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Save plot
                plot_path = os.path.join(save_dir, f"perf_by_miss_count_{timestamp}.png")
                plt.savefig(plot_path)
                saved_files.append(plot_path)
            else:
                print("No data available for miss count plot")
            plt.close()
        
        # Plot performance by remaining ships
        if "by_remaining_ships" in category_analysis and category_analysis["by_remaining_ships"]:
            plt.figure(figsize=(12, 8))
            
            # Extract data - ensure we're using string keys consistently
            categories = sorted([int(k) for k in category_analysis["by_remaining_ships"].keys()])
            match_percentages = []
            precisions = []
            recalls = []
            f1_scores = []
            sample_counts = []
            
            for k in categories:
                k_str = str(k)
                if k_str in category_analysis["by_remaining_ships"]:
                    match_percentages.append(category_analysis["by_remaining_ships"][k_str]["match_percentage"])
                    precisions.append(category_analysis["by_remaining_ships"][k_str]["precision"])
                    recalls.append(category_analysis["by_remaining_ships"][k_str]["recall"])
                    f1_scores.append(category_analysis["by_remaining_ships"][k_str]["f1_score"])
                    sample_counts.append(category_analysis["by_remaining_ships"][k_str]["sample_count"])
            
            # Create x-axis labels
            x_labels = [str(k) for k in categories]
            x = np.arange(len(x_labels))
            
            # Only plot if we have data
            if len(match_percentages) > 0:
                # Plot metrics
                bar_width = 0.2
                plt.bar(x - bar_width*1.5, match_percentages, bar_width, label='Match %', color='blue')
                plt.bar(x - bar_width*0.5, precisions, bar_width, label='Precision', color='green')
                plt.bar(x + bar_width*0.5, recalls, bar_width, label='Recall', color='red')
                plt.bar(x + bar_width*1.5, f1_scores, bar_width, label='F1 Score', color='purple')
                
                # Add sample count as text above bars
                for i, count in enumerate(sample_counts):
                    plt.text(x[i], max(match_percentages[i], precisions[i], recalls[i], f1_scores[i]) + 0.05, 
                             f"n={count}", ha='center', va='bottom', fontsize=8)
                
                plt.xlabel('Remaining Ships')
                plt.ylabel('Score')
                plt.title('Heatmap Performance by Remaining Ships')
                plt.xticks(x, x_labels)
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Save plot
                plot_path = os.path.join(save_dir, f"perf_by_remaining_ships_{timestamp}.png")
                plt.savefig(plot_path)
                saved_files.append(plot_path)
            else:
                print("No data available for remaining ships plot")
            plt.close()
        
        # Plot performance by game progress
        if "by_game_progress" in category_analysis and category_analysis["by_game_progress"]:
            plt.figure(figsize=(12, 8))
            
            # Extract data - ensure we're using string keys consistently
            categories = sorted([int(k) for k in category_analysis["by_game_progress"].keys()])
            match_percentages = []
            precisions = []
            recalls = []
            f1_scores = []
            sample_counts = []
            
            for k in categories:
                k_str = str(k)
                if k_str in category_analysis["by_game_progress"]:
                    match_percentages.append(category_analysis["by_game_progress"][k_str]["match_percentage"])
                    precisions.append(category_analysis["by_game_progress"][k_str]["precision"])
                    recalls.append(category_analysis["by_game_progress"][k_str]["recall"])
                    f1_scores.append(category_analysis["by_game_progress"][k_str]["f1_score"])
                    sample_counts.append(category_analysis["by_game_progress"][k_str]["sample_count"])
            
            # Create x-axis labels (progress percentage bins)
            x_labels = [f"{k*10}%-{(k+1)*10}%" for k in categories]
            x = np.arange(len(x_labels))
            
            # Only plot if we have data
            if len(match_percentages) > 0:
                # Plot metrics
                bar_width = 0.2
                plt.bar(x - bar_width*1.5, match_percentages, bar_width, label='Match %', color='blue')
                plt.bar(x - bar_width*0.5, precisions, bar_width, label='Precision', color='green')
                plt.bar(x + bar_width*0.5, recalls, bar_width, label='Recall', color='red')
                plt.bar(x + bar_width*1.5, f1_scores, bar_width, label='F1 Score', color='purple')
                
                # Add sample count as text above bars
                for i, count in enumerate(sample_counts):
                    plt.text(x[i], max(match_percentages[i], precisions[i], recalls[i], f1_scores[i]) + 0.05, 
                             f"n={count}", ha='center', va='bottom', fontsize=8)
                
                plt.xlabel('Game Progress')
                plt.ylabel('Score')
                plt.title('Heatmap Performance by Game Progress')
                plt.xticks(x, x_labels)
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Save plot
                plot_path = os.path.join(save_dir, f"perf_by_game_progress_{timestamp}.png")
                plt.savefig(plot_path)
                saved_files.append(plot_path)
            else:
                print("No data available for game progress plot")
            plt.close()
        
        # Plot performance by targeting mode
        if "by_targeting_mode" in category_analysis and category_analysis["by_targeting_mode"]:
            plt.figure(figsize=(12, 8))
            
            # Extract data - ensure we're handling keys safely
            categories = list(category_analysis["by_targeting_mode"].keys())
            match_percentages = []
            precisions = []
            recalls = []
            f1_scores = []
            sample_counts = []
            
            for k in categories:
                if k in category_analysis["by_targeting_mode"]:
                    match_percentages.append(category_analysis["by_targeting_mode"][k]["match_percentage"])
                    precisions.append(category_analysis["by_targeting_mode"][k]["precision"])
                    recalls.append(category_analysis["by_targeting_mode"][k]["recall"])
                    f1_scores.append(category_analysis["by_targeting_mode"][k]["f1_score"])
                    sample_counts.append(category_analysis["by_targeting_mode"][k]["sample_count"])
            
            # Create x-axis labels
            x_labels = [k.replace("_", " ").title() for k in categories]
            x = np.arange(len(x_labels))
            
            # Only plot if we have data
            if len(match_percentages) > 0:
                # Plot metrics
                bar_width = 0.2
                plt.bar(x - bar_width*1.5, match_percentages, bar_width, label='Match %', color='blue')
                plt.bar(x - bar_width*0.5, precisions, bar_width, label='Precision', color='green')
                plt.bar(x + bar_width*0.5, recalls, bar_width, label='Recall', color='red')
                plt.bar(x + bar_width*1.5, f1_scores, bar_width, label='F1 Score', color='purple')
                
                # Add sample count as text above bars
                for i, count in enumerate(sample_counts):
                    plt.text(x[i], max(match_percentages[i], precisions[i], recalls[i], f1_scores[i]) + 0.05, 
                             f"n={count}", ha='center', va='bottom', fontsize=8)
                
                plt.xlabel('Targeting Mode')
                plt.ylabel('Score')
                plt.title('Heatmap Performance by Targeting Mode')
                plt.xticks(x, x_labels)
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Save plot
                plot_path = os.path.join(save_dir, f"perf_by_targeting_mode_{timestamp}.png")
                plt.savefig(plot_path)
                saved_files.append(plot_path)
            else:
                print("No data available for targeting mode plot")
            plt.close()
        
        return saved_files
    
    def compare_with_previous(self, current_metrics, previous_file):
        """
        Compare current metrics with previously saved metrics
        
        Args:
            current_metrics (dict): Current metrics
            previous_file (str): Path to previous metrics file
            
        Returns:
            dict: Comparison results
        """
        # Load previous metrics
        try:
            with open(previous_file, 'r') as f:
                previous_metrics = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading previous metrics: {e}")
            return None
        
        # Compare key metrics
        comparison = {
            "current_timestamp": datetime.now().isoformat(),
            "previous_timestamp": previous_metrics.get("timestamp", "Unknown"),
            "metrics_comparison": {}
        }
        
        for metric in ["avg_match_percentage", "avg_ship_cell_prob", "avg_empty_cell_prob", "avg_hit_rate_high_prob"]:
            current_value = current_metrics.get(metric, 0)
            previous_value = previous_metrics.get(metric, 0)
            
            # Calculate difference and percent change
            difference = current_value - previous_value
            percent_change = (difference / previous_value) * 100 if previous_value != 0 else float('inf')
            
            comparison["metrics_comparison"][metric] = {
                "current": current_value,
                "previous": previous_value,
                "difference": difference,
                "percent_change": percent_change
            }
        
        return comparison

    def save_comparison(self, comparison, filename=None):
        """
        Save comparison results to a file
        
        Args:
            comparison (dict): Comparison results
            filename (str, optional): Custom filename
            
        Returns:
            str: Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"metrics_comparison_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        print(f"Comparison saved to {filepath}")
        return filepath


if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Collect and analyze battleship heatmap metrics')
    parser.add_argument('--games', type=int, default=50, help='Number of games to simulate')
    parser.add_argument('--output-dir', type=str, default='metrics', help='Directory to save metrics and plots')
    parser.add_argument('--category-analysis', action='store_true', help='Enable detailed category-based analysis')
    parser.add_argument('--skip-plots', action='store_true', help='Skip generating plots')
    args = parser.parse_args()
    
    # Create metrics collector
    collector = MetricsCollector(output_dir=args.output_dir)
    
    # Run simulations
    print(f"Running battleship simulations with {args.games} games...")
    metrics = collector.run_simulations(num_games=args.games)
    
    # Save metrics
    metrics_file = collector.save_metrics(metrics)
    
    # Generate and save plots if not skipped
    if not args.skip_plots:
        print("Generating plots...")
        plot_files = collector.generate_plots(metrics)
    
    # Always compare the current metrics with the previous run if there's a previous metrics file
    # First, get the list of metrics files (excluding the one we just created)
    all_metrics_files = [f for f in os.listdir(collector.output_dir) 
                       if f.startswith("heatmap_metrics_") and f.endswith(".json") 
                       and os.path.join(collector.output_dir, f) != metrics_file]
    
    # If there's at least one previous metrics file, compare with the most recent one
    if all_metrics_files:
        # Sort files by timestamp (assuming filename format)
        all_metrics_files.sort()
        previous_file = os.path.join(collector.output_dir, all_metrics_files[-1])  # Most recent previous file
        
        print(f"Comparing with previous metrics: {previous_file}")
        comparison = collector.compare_with_previous(metrics, previous_file)
        if comparison:
            comparison_file = collector.save_comparison(comparison)
            
            # Print key comparisons with color-coded improvements/regressions
            print("\nPerformance Comparison:")
            for metric, values in comparison["metrics_comparison"].items():
                # Format the metric name for better readability
                metric_name = metric.replace("avg_", "").replace("_", " ").title()
                
                # Determine if the change is an improvement (depends on the metric)
                is_improvement = False
                if metric == "avg_match_percentage" or metric == "avg_ship_cell_prob" or metric == "avg_hit_rate_high_prob":
                    is_improvement = values["difference"] > 0
                elif metric == "avg_empty_cell_prob":  
                    is_improvement = values["difference"] < 0  # Lower is better for this one
                
                # Create change indicator
                change_indicator = "↑" if values["difference"] > 0 else "↓"
                status = "IMPROVED" if is_improvement else "REGRESSED"
                
                print(f"  {metric_name}: {values['current']:.4f} vs {values['previous']:.4f} " + 
                      f"({change_indicator} {abs(values['percent_change']):.2f}%) - {status}")
    
    # Print summary of category analysis if enabled
    if args.category_analysis and "category_analysis" in metrics:
        print("\nCategory Analysis Summary:")
        
        # Print targeting mode performance
        if "by_targeting_mode" in metrics["category_analysis"]:
            print("\n  Performance by Targeting Mode:")
            for mode, data in metrics["category_analysis"]["by_targeting_mode"].items():
                print(f"    {mode.replace('_', ' ').title()} (n={data['sample_count']}): " +
                      f"Match %: {data['match_percentage']:.2%}, " +
                      f"F1: {data['f1_score']:.3f}")
        
        # Print performance by hit count
        if "by_hit_count" in metrics["category_analysis"]:
            print("\n  Performance by Hit Count:")
            categories = sorted([int(k) for k in metrics["category_analysis"]["by_hit_count"].keys()])
            for k in categories:
                k_str = str(k)
                if k_str in metrics["category_analysis"]["by_hit_count"]:
                    data = metrics["category_analysis"]["by_hit_count"][k_str]
                    hit_range = f"{k*5}-{(k+1)*5-1}"
                    print(f"    {hit_range} hits (n={data['sample_count']}): " +
                          f"Match %: {data['match_percentage']:.2%}, " +
                          f"F1: {data['f1_score']:.3f}")
        
        # Print performance by remaining ships
        if "by_remaining_ships" in metrics["category_analysis"]:
            print("\n  Performance by Remaining Ships:")
            categories = sorted([int(k) for k in metrics["category_analysis"]["by_remaining_ships"].keys()])
            for k in categories:
                k_str = str(k)
                if k_str in metrics["category_analysis"]["by_remaining_ships"]:
                    data = metrics["category_analysis"]["by_remaining_ships"][k_str]
                    print(f"    {k} ships (n={data['sample_count']}): " +
                          f"Match %: {data['match_percentage']:.2%}, " +
                          f"F1: {data['f1_score']:.3f}")
    
    print("\nMetrics collection complete!")
    print(f"Summary of results:")
    print(f"  - Average match percentage: {metrics['avg_match_percentage']:.2%}")
    print(f"  - Average ship cell probability: {metrics['avg_ship_cell_prob']:.3f}")
    print(f"  - Average empty cell probability: {metrics['avg_empty_cell_prob']:.3f}")
    print(f"  - High probability hit rate: {metrics['avg_hit_rate_high_prob']:.3f}")
