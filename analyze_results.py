#!/usr/bin/env python3
"""
Analysis script for PULearningPC experiment results.
This script generates charts and performs statistical analysis on the experimental results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class ExperimentResultsAnalyzer:
    """Main class for analyzing experiment results and generating visualizations."""
    
    def __init__(self, results_dir: str = "experiment_results"):
        """
        Initialize the analyzer with the results directory.
        
        Args:
            results_dir: Path to directory containing experiment results
        """
        self.results_dir = Path(results_dir)
        self.summary_data = {}
        self.final_data = {}
        self.datasets = ['cora', 'citeseer', 'twitch', 'mnist']
        
        # Set up plotting style
        plt.style.use('default')
        
        # Color palette for datasets
        self.dataset_colors = {
            'cora': '#1f77b4',
            'citeseer': '#ff7f0e', 
            'twitch': '#2ca02c',
            'mnist': '#d62728'
        }
        
        # Line styles for different datasets
        self.dataset_linestyles = {
            'cora': '-',
            'citeseer': '--',
            'twitch': '-.',
            'mnist': ':'
        }
    
    def load_summary_results(self) -> Dict[str, pd.DataFrame]:
        """
        Load all summary results CSV files.
        
        Returns:
            Dictionary mapping dataset names to their summary DataFrames
        """
        print("Loading summary results...")
        
        # Define the expected column names for summary results
        column_names = [
            'num_particles', 'p_det', 'delta_v', 'delta_p', 'cluster_strategy',
            'positive_cluster_threshold', 'movement_strategy', 'initialization_strategy',
            'avg_node_pot_threshold', 'f1_score_mean', 'f1_score_std', 'f1_score_min',
            'f1_score_max', 'precision_mean', 'precision_std', 'recall_mean', 'recall_std',
            'num_reliable_negatives_mean', 'num_reliable_negatives_std', 'coverage_mean',
            'coverage_std', 'coverage_min', 'coverage_max', 'fallback_rule_used_sum',
            'fallback_rule_used_count', 'avg_threshold_criteria_sum', 'avg_threshold_criteria_count',
            'final_iteration_count_mean', 'final_iteration_count_std', 'final_iteration_count_min',
            'final_iteration_count_max'
        ]
        
        for dataset in self.datasets:
            file_path = self.results_dir / f"{dataset}_summary_results.csv"
            if file_path.exists():
                try:
                    # Read the CSV with proper header handling
                    # Skip first 3 rows: metric names, statistics, and parameter names
                    df = pd.read_csv(file_path, skiprows=3, header=None)
                    
                    # Set the column names
                    df.columns = column_names
                    
                    self.summary_data[dataset] = df
                    print(f"  ✓ Loaded {dataset}: {len(df)} parameter combinations")
                    print(f"    Columns: {list(df.columns)[:5]}...")  # Show first 5 columns
                except Exception as e:
                    print(f"  ✗ Error loading {dataset}: {e}")
            else:
                print(f"  ✗ File not found: {file_path}")
        
        return self.summary_data
    
    def load_final_results(self) -> Dict[str, pd.DataFrame]:
        """
        Load all final results CSV files.
        
        Returns:
            Dictionary mapping dataset names to their final DataFrames
        """
        print("Loading final results...")
        
        for dataset in self.datasets:
            file_path = self.results_dir / f"{dataset}_final_results.csv"
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path)
                    self.final_data[dataset] = df
                    print(f"  ✓ Loaded {dataset}: {len(df)} individual runs")
                except Exception as e:
                    print(f"  ✗ Error loading {dataset}: {e}")
            else:
                print(f"  ✗ File not found: {file_path}")
        
        return self.final_data
    
    def filter_data_by_threshold(self, threshold: float = 0.1) -> Dict[str, pd.DataFrame]:
        """
        Filter summary data to only include results with the specified positive_cluster_threshold.
        
        Args:
            threshold: The positive_cluster_threshold value to filter by
            
        Returns:
            Dictionary of filtered DataFrames
        """
        filtered_data = {}
        
        for dataset, df in self.summary_data.items():
            if 'positive_cluster_threshold' in df.columns:
                filtered_df = df[df['positive_cluster_threshold'] == threshold].copy()
                filtered_data[dataset] = filtered_df
                print(f"  {dataset}: {len(filtered_df)} combinations with threshold={threshold}")
            else:
                print(f"  {dataset}: No positive_cluster_threshold column found")
                filtered_data[dataset] = df
        
        return filtered_data
    
    def aggregate_data_by_particles(self, filtered_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Aggregate data by taking mean across avg_node_pot_threshold for each combination of
        num_particles, initialization_strategy, and movement_strategy.
        
        Args:
            filtered_data: Dictionary of filtered DataFrames
            
        Returns:
            Dictionary of aggregated DataFrames
        """
        aggregated_data = {}
        
        for dataset, df in filtered_data.items():
            if 'avg_node_pot_threshold' in df.columns:
                # Group by num_particles, initialization_strategy, movement_strategy
                # and take mean of f1_score_mean and other metrics
                agg_dict = {
                    'f1_score_mean': 'mean',
                    'f1_score_std': 'mean',  # Average the standard deviations
                    'precision_mean': 'mean',
                    'recall_mean': 'mean',
                    'coverage_mean': 'mean',
                    'num_reliable_negatives_mean': 'mean'
                }
                
                # Only include columns that exist in the DataFrame
                available_agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}
                
                aggregated_df = df.groupby(['num_particles', 'initialization_strategy', 'movement_strategy']).agg(available_agg_dict).reset_index()
                aggregated_data[dataset] = aggregated_df
                print(f"  {dataset}: Aggregated from {len(df)} to {len(aggregated_df)} combinations")
            else:
                aggregated_data[dataset] = df
        
        return aggregated_data
    
    def generate_f1_vs_particles_charts(self, 
                                      threshold: float = 0.1,
                                      save_dir: str = "analysis_charts",
                                      figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Generate line charts comparing F1 scores vs number of particles for different 
        parameter combinations.
        
        Args:
            threshold: Fixed positive_cluster_threshold value
            save_dir: Directory to save charts
            figsize: Figure size for charts
        """
        print(f"\nGenerating F1 vs Particles charts (threshold={threshold})...")
        
        # Create save directory
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        # Filter data by threshold
        filtered_data = self.filter_data_by_threshold(threshold)
        
        # Aggregate data to handle multiple avg_node_pot_threshold values
        print("Aggregating data across avg_node_pot_threshold values...")
        aggregated_data = self.aggregate_data_by_particles(filtered_data)
        
        # Get unique combinations of initialization_strategy and movement_strategy
        all_combinations = set()
        for df in aggregated_data.values():
            if 'initialization_strategy' in df.columns and 'movement_strategy' in df.columns:
                combinations = df[['initialization_strategy', 'movement_strategy']].drop_duplicates()
                for _, row in combinations.iterrows():
                    all_combinations.add((row['initialization_strategy'], row['movement_strategy']))

        print(f"Found {len(all_combinations)} parameter combinations:")
        for init_strat, move_strat in sorted(all_combinations):
            print(f"  - init: {init_strat}, move: {move_strat}")
        
        # Generate chart for each combination
        for init_strategy, movement_strategy in sorted(all_combinations):
            self._create_single_f1_chart(
                aggregated_data, 
                init_strategy, 
                movement_strategy, 
                threshold,
                save_path,
                figsize
            )
        
        print(f"\nCharts saved to: {save_path}")
    
    def _create_single_f1_chart(self, 
                               filtered_data: Dict[str, pd.DataFrame],
                               init_strategy: str,
                               movement_strategy: str,
                               threshold: float,
                               save_path: Path,
                               figsize: Tuple[int, int]) -> None:
        """
        Create a single F1 vs particles chart for a specific parameter combination.
        """
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        chart_title = f"F1 Score vs Number of Particles\n"
        chart_title += f"Init: {init_strategy}, Move: {movement_strategy}, Threshold: {threshold}"
        
        # Plot lines for each dataset
        for dataset in self.datasets:
            if dataset not in filtered_data:
                continue
                
            df = filtered_data[dataset]
            
            # Filter for this specific parameter combination
            mask = (
                (df['initialization_strategy'] == init_strategy) & 
                (df['movement_strategy'] == movement_strategy)
            )
            subset = df[mask].copy()
            
            if len(subset) == 0:
                print(f"  No data for {dataset} with init={init_strategy}, move={movement_strategy}")
                continue
            
            # Sort by num_particles for proper line plotting
            subset = subset.sort_values('num_particles')
            
            # Plot the line
            ax.plot(
                subset['num_particles'], 
                subset['f1_score_mean'],
                marker='o',
                linewidth=2,
                markersize=6,
                label=dataset.upper(),
                color=self.dataset_colors.get(dataset, 'black'),
                linestyle=self.dataset_linestyles.get(dataset, '-')
            )
            
            # Add error bars if std is available
            if 'f1_score_std' in subset.columns:
                ax.errorbar(
                    subset['num_particles'],
                    subset['f1_score_mean'],
                    yerr=subset['f1_score_std'],
                    fmt='none',
                    alpha=0.5,
                    color=self.dataset_colors.get(dataset, 'black')
                )
        
        # Customize the chart
        ax.set_xlabel('Number of Particles', fontsize=12, fontweight='bold')
        ax.set_ylabel('F1 Score (Mean)', fontsize=12, fontweight='bold')
        ax.set_title(chart_title, fontsize=14, fontweight='bold', pad=20)
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)
        
        # Set y-axis limits to better show differences
        ax.set_ylim(bottom=0)
        
        # Improve layout
        plt.tight_layout()
        
        # Save the chart
        filename = f"f1_vs_particles_init_{init_strategy}_move_{movement_strategy}_thresh_{threshold}.png"
        filepath = save_path / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {filename}")
    
    def generate_heatmap_charts(self,
                               threshold: float = 0.1,
                               save_dir: str = "analysis_charts",
                               figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Generate heatmap charts showing F1 score vs num_particles and avg_node_pot_threshold
        for different parameter combinations.
        
        Args:
            threshold: Fixed positive_cluster_threshold value
            save_dir: Directory to save charts
            figsize: Figure size for charts
        """
        print(f"\nGenerating heatmap charts (threshold={threshold})...")
        
        # Create save directory
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        # Filter data by threshold (don't aggregate for heatmaps)
        filtered_data = self.filter_data_by_threshold(threshold)
        
        # Get unique combinations of initialization_strategy and movement_strategy
        all_combinations = set()
        for df in filtered_data.values():
            if 'initialization_strategy' in df.columns and 'movement_strategy' in df.columns:
                combinations = df[['initialization_strategy', 'movement_strategy']].drop_duplicates()
                for _, row in combinations.iterrows():
                    all_combinations.add((row['initialization_strategy'], row['movement_strategy']))

        print(f"Found {len(all_combinations)} parameter combinations for heatmaps:")
        for init_strat, move_strat in sorted(all_combinations):
            print(f"  - init: {init_strat}, move: {move_strat}")
        
        # Generate heatmap for each combination
        for init_strategy, movement_strategy in sorted(all_combinations):
            self._create_single_heatmap(
                filtered_data, 
                init_strategy, 
                movement_strategy, 
                threshold,
                save_path,
                figsize
            )
        
        print(f"\nHeatmap charts saved to: {save_path}")
    
    def _create_single_heatmap(self,
                              filtered_data: Dict[str, pd.DataFrame],
                              init_strategy: str,
                              movement_strategy: str,
                              threshold: float,
                              save_path: Path,
                              figsize: Tuple[int, int]) -> None:
        """
        Create a single heatmap chart for a specific parameter combination.
        """
        # Create subplots for each dataset
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        chart_title = f"F1 Score Heatmap\n"
        chart_title += f"Init: {init_strategy}, Move: {movement_strategy}, Threshold: {threshold}"
        
        for idx, dataset in enumerate(self.datasets):
            if dataset not in filtered_data:
                continue
                
            ax = axes[idx]
            df = filtered_data[dataset]
            
            # Filter for this specific parameter combination
            mask = (
                (df['initialization_strategy'] == init_strategy) & 
                (df['movement_strategy'] == movement_strategy)
            )
            subset = df[mask].copy()
            
            if len(subset) == 0:
                ax.set_title(f'{dataset.upper()} - No Data')
                continue
            
            # Create pivot table for heatmap
            pivot_data = subset.pivot_table(
                values='f1_score_mean',
                index='avg_node_pot_threshold',
                columns='num_particles',
                aggfunc='mean'
            )
            
            # Create heatmap (reverse Y-axis to go from 0.2 to 0.9)
            im = ax.imshow(pivot_data.values, cmap='viridis', aspect='auto', origin='lower')
            
            # Set ticks and labels
            ax.set_xticks(range(len(pivot_data.columns)))
            ax.set_xticklabels(pivot_data.columns)
            ax.set_yticks(range(len(pivot_data.index)))
            ax.set_yticklabels([f'{x:.1f}' for x in pivot_data.index])
            
            # Add colorbar
            plt.colorbar(im, ax=ax, label='F1 Score')
            
            ax.set_title(f'{dataset.upper()} Dataset')
            ax.set_xlabel('Number of Particles')
            ax.set_ylabel('Avg Node Potential Threshold')
        
        # Hide empty subplots
        for idx in range(len(self.datasets), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle(chart_title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save the chart
        filename = f"heatmap_init_{init_strategy}_move_{movement_strategy}_thresh_{threshold}.png"
        filepath = save_path / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {filename}")
    
    def generate_parameter_comparison_chart(self,
                                          threshold: float = 0.1,
                                          save_dir: str = "analysis_charts",
                                          figsize: Tuple[int, int] = (15, 10)) -> None:
        """
        Generate a comprehensive chart comparing all parameter combinations.
        
        Args:
            threshold: Fixed positive_cluster_threshold value
            save_dir: Directory to save charts
            figsize: Figure size for charts
        """
        print(f"\nGenerating comprehensive parameter comparison chart...")
        
        # Create save directory
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        # Filter data by threshold
        filtered_data = self.filter_data_by_threshold(threshold)
        
        # Aggregate data to handle multiple avg_node_pot_threshold values
        print("Aggregating data across avg_node_pot_threshold values...")
        aggregated_data = self.aggregate_data_by_particles(filtered_data)
        
        # Create subplots for each dataset
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        # First pass: collect all F1 scores to determine global Y-axis range
        all_f1_scores = []
        for dataset in self.datasets:
            if dataset in aggregated_data:
                df = aggregated_data[dataset]
                all_f1_scores.extend(df['f1_score_mean'].tolist())
        
        # Set global Y-axis range with some padding
        if all_f1_scores:
            y_min = max(0, min(all_f1_scores) - 0.05)
            y_max = min(1, max(all_f1_scores) + 0.05)
        else:
            y_min, y_max = 0, 1
        
        for idx, dataset in enumerate(self.datasets):
            if dataset not in aggregated_data:
                continue
                
            ax = axes[idx]
            df = aggregated_data[dataset]
            
            # Get unique parameter combinations
            combinations = df[['initialization_strategy', 'movement_strategy']].drop_duplicates()
            
            # Plot lines for each parameter combination
            for _, row in combinations.iterrows():
                init_strat = row['initialization_strategy']
                move_strat = row['movement_strategy']
                
                mask = (
                    (df['initialization_strategy'] == init_strat) & 
                    (df['movement_strategy'] == move_strat)
                )
                subset = df[mask].sort_values('num_particles')
                
                if len(subset) == 0:
                    continue
                
                label = f"{init_strat}_{move_strat}"
                ax.plot(
                    subset['num_particles'],
                    subset['f1_score_mean'],
                    marker='o',
                    linewidth=2,
                    markersize=4,
                    label=label
                )
            
            ax.set_title(f'{dataset.upper()} Dataset', fontweight='bold')
            ax.set_xlabel('Number of Particles')
            ax.set_ylabel('F1 Score (Mean)')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(y_min, y_max)  # Use global Y-axis range
        
        # Hide empty subplots
        for idx in range(len(self.datasets), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle(f'F1 Score vs Number of Particles (Threshold={threshold})', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save the chart
        filename = f"comprehensive_f1_vs_particles_thresh_{threshold}.png"
        filepath = save_path / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {filename}")
    
    def generate_statistical_summary(self, 
                                   threshold: float = 0.1,
                                   save_dir: str = "analysis_charts") -> None:
        """
        Generate statistical summary of the results.
        
        Args:
            threshold: Fixed positive_cluster_threshold value
            save_dir: Directory to save summary
        """
        print(f"\nGenerating statistical summary...")
        
        # Create save directory
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        # Filter data by threshold
        filtered_data = self.filter_data_by_threshold(threshold)
        
        summary_stats = []
        
        for dataset, df in filtered_data.items():
            if 'f1_score_mean' not in df.columns:
                continue
                
            # Overall statistics for this dataset
            overall_stats = {
                'dataset': dataset,
                'parameter_combination': 'overall',
                'initialization_strategy': 'all',
                'movement_strategy': 'all',
                'num_combinations': len(df),
                'best_f1_score': df['f1_score_mean'].max(),
                'worst_f1_score': df['f1_score_mean'].min(),
                'mean_f1_score': df['f1_score_mean'].mean(),
                'std_f1_score': df['f1_score_mean'].std(),
                'best_num_particles': df.loc[df['f1_score_mean'].idxmax(), 'num_particles']
            }
            summary_stats.append(overall_stats)
            
            # Statistics by parameter combination
            if 'initialization_strategy' in df.columns and 'movement_strategy' in df.columns:
                combinations = df[['initialization_strategy', 'movement_strategy']].drop_duplicates()
                
                for _, row in combinations.iterrows():
                    init_strat = row['initialization_strategy']
                    move_strat = row['movement_strategy']
                    
                    mask = (
                        (df['initialization_strategy'] == init_strat) & 
                        (df['movement_strategy'] == move_strat)
                    )
                    subset = df[mask]
                    
                    if len(subset) == 0:
                        continue
                    
                    combo_stats = {
                        'dataset': dataset,
                        'parameter_combination': f"{init_strat}_{move_strat}",
                        'initialization_strategy': init_strat,
                        'movement_strategy': move_strat,
                        'num_combinations': len(subset),
                        'best_f1_score': subset['f1_score_mean'].max(),
                        'worst_f1_score': subset['f1_score_mean'].min(),
                        'mean_f1_score': subset['f1_score_mean'].mean(),
                        'std_f1_score': subset['f1_score_mean'].std(),
                        'best_num_particles': subset.loc[subset['f1_score_mean'].idxmax(), 'num_particles']
                    }
                    summary_stats.append(combo_stats)
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(summary_stats)
        
        # Save to CSV
        summary_file = save_path / f"statistical_summary_thresh_{threshold}.csv"
        summary_df.to_csv(summary_file, index=False)
        
        # Print summary
        print(f"\nStatistical Summary (Threshold={threshold}):")
        print("=" * 80)
        
        for dataset in self.datasets:
            if dataset not in filtered_data:
                continue
                
            dataset_stats = summary_df[summary_df['dataset'] == dataset]
            overall = dataset_stats[dataset_stats['parameter_combination'] == 'overall']
            
            if len(overall) > 0:
                stats = overall.iloc[0]
                print(f"\n{dataset.upper()} Dataset:")
                print(f"  Best F1 Score: {stats['best_f1_score']:.4f}")
                print(f"  Mean F1 Score: {stats['mean_f1_score']:.4f} ± {stats['std_f1_score']:.4f}")
                print(f"  Best # Particles: {stats['best_num_particles']}")
                
                # Show best parameter combination
                param_combos = dataset_stats[dataset_stats['parameter_combination'] != 'overall']
                if len(param_combos) > 0:
                    best_combo = param_combos.loc[param_combos['best_f1_score'].idxmax()]
                    print(f"  Best Parameters: {best_combo['initialization_strategy']} + {best_combo['movement_strategy']}")
        
        print(f"\nSummary saved to: {summary_file}")
    
    def run_full_analysis(self, 
                         threshold: float = 0.1,
                         save_dir: str = "analysis_charts") -> None:
        """
        Run the complete analysis pipeline.
        
        Args:
            threshold: Fixed positive_cluster_threshold value
            save_dir: Directory to save all outputs
        """
        print("=" * 80)
        print("PULearningPC Experiment Results Analysis")
        print("=" * 80)
        
        # Load data
        self.load_summary_results()
        
        if not self.summary_data:
            print("No summary data found! Please check the results directory.")
            return
        
        # Generate charts
        self.generate_f1_vs_particles_charts(threshold=threshold, save_dir=save_dir)
        self.generate_heatmap_charts(threshold=threshold, save_dir=save_dir)
        self.generate_parameter_comparison_chart(threshold=threshold, save_dir=save_dir)
        
        # Generate statistical summary
        self.generate_statistical_summary(threshold=threshold, save_dir=save_dir)
        
        print("\n" + "=" * 80)
        print("Analysis Complete!")
        print("=" * 80)
        print(f"All outputs saved to: {save_dir}/")
        print("\nGenerated files:")
        print("  - Individual F1 vs Particles charts for each parameter combination")
        print("  - Heatmap charts for each parameter combination")
        print("  - Comprehensive parameter comparison chart")
        print("  - Statistical summary CSV file")


def main():
    """Main function to run the analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze PULearningPC experiment results")
    parser.add_argument(
        '--results-dir', 
        default='experiment_results',
        help='Directory containing experiment results (default: experiment_results)'
    )
    parser.add_argument(
        '--threshold', 
        type=float, 
        default=0.1,
        help='Fixed positive_cluster_threshold value (default: 0.1)'
    )
    parser.add_argument(
        '--save-dir', 
        default='analysis_charts',
        help='Directory to save analysis outputs (default: analysis_charts)'
    )
    
    args = parser.parse_args()
    
    # Create analyzer and run analysis
    analyzer = ExperimentResultsAnalyzer(results_dir=args.results_dir)
    analyzer.run_full_analysis(threshold=args.threshold, save_dir=args.save_dir)


if __name__ == "__main__":
    main()
