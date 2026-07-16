#!/usr/bin/env python3
"""
Analysis script for PULearningPC experiment results.
This script generates charts and performs statistical analysis on the experimental results.

New functionality:
- Added filter_by_threshold parameter to control whether to filter by positive_cluster_threshold
- When filter_by_threshold=False, the script aggregates across all positive_cluster_threshold values
- This allows analysis of F1 scores vs num_particles with averaged results across thresholds
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
            'num_particles', 'cluster_strategy',
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
    
    def aggregate_data_by_particles(self, filtered_data: Dict[str, pd.DataFrame], 
                                   include_threshold_agg: bool = False,
                                   keep_avg_node_pot: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Aggregate data by taking mean across avg_node_pot_threshold for each combination of
        num_particles, initialization_strategy, and movement_strategy.
        Optionally also aggregate across positive_cluster_threshold.
        
        Args:
            filtered_data: Dictionary of filtered DataFrames
            include_threshold_agg: If True, also aggregate across positive_cluster_threshold
            keep_avg_node_pot: If True, keep avg_node_pot_threshold in grouping (for heatmaps)
            
        Returns:
            Dictionary of aggregated DataFrames
        """
        aggregated_data = {}
        
        for dataset, df in filtered_data.items():
            if 'avg_node_pot_threshold' in df.columns:
                # Define grouping columns
                group_cols = ['num_particles', 'initialization_strategy', 'movement_strategy']
                
                # Add avg_node_pot_threshold to grouping if keeping it (for heatmaps)
                if keep_avg_node_pot:
                    group_cols.append('avg_node_pot_threshold')
                
                # Add positive_cluster_threshold to grouping if not aggregating across it
                if not include_threshold_agg and 'positive_cluster_threshold' in df.columns:
                    group_cols.append('positive_cluster_threshold')
                
                # Group by the specified columns and take mean of f1_score_mean and other metrics
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
                
                aggregated_df = df.groupby(group_cols).agg(available_agg_dict).reset_index()
                aggregated_data[dataset] = aggregated_df
                print(f"  {dataset}: Aggregated from {len(df)} to {len(aggregated_df)} combinations")
            else:
                aggregated_data[dataset] = df
        
        return aggregated_data
    
    def generate_f1_vs_particles_charts(self, 
                                      threshold: float = 0.1,
                                      filter_by_threshold: bool = True,
                                      save_dir: str = "analysis_charts",
                                      figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Generate line charts comparing F1 scores vs number of particles for different 
        parameter combinations.
        
        Args:
            threshold: Fixed positive_cluster_threshold value (used only if filter_by_threshold=True)
            filter_by_threshold: If True, filter by threshold; if False, aggregate across all thresholds
            save_dir: Directory to save charts
            figsize: Figure size for charts
        """
        if filter_by_threshold:
            print(f"\nGenerating F1 vs Particles charts (threshold={threshold})...")
        else:
            print(f"\nGenerating F1 vs Particles charts (aggregated across all thresholds)...")
        
        # Create save directory
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        # Filter data by threshold or use all data
        if filter_by_threshold:
            filtered_data = self.filter_data_by_threshold(threshold)
        else:
            filtered_data = {}
            for dataset, df in self.summary_data.items():
                filtered_df = df.copy()
                filtered_data[dataset] = filtered_df

        # Aggregate data to handle multiple avg_node_pot_threshold values
        print("Aggregating data across avg_node_pot_threshold values...")
        aggregated_data = self.aggregate_data_by_particles(filtered_data, include_threshold_agg=not filter_by_threshold)
        
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
                filter_by_threshold,
                save_path,
                figsize
            )
        
        print(f"\nCharts saved to: {save_path}")
    
    def _create_single_f1_chart(self, 
                               filtered_data: Dict[str, pd.DataFrame],
                               init_strategy: str,
                               movement_strategy: str,
                               threshold: float,
                               filter_by_threshold: bool,
                               save_path: Path,
                               figsize: Tuple[int, int]) -> None:
        """
        Create a single F1 vs particles chart for a specific parameter combination.
        """
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        chart_title = f"F1 Score vs Number of Particles\n"
        chart_title += f"Init: {init_strategy}, Move: {movement_strategy}"
        if filter_by_threshold:
            chart_title += f", Threshold: {threshold}"
        else:
            chart_title += ", Aggregated across all thresholds"
        
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
                subset['num_particles'].values, 
                subset['f1_score_mean'].values,
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
                    subset['num_particles'].values,
                    subset['f1_score_mean'].values,
                    yerr=subset['f1_score_std'].values,
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
        if filter_by_threshold:
            filename = f"f1_vs_particles_init_{init_strategy}_move_{movement_strategy}_thresh_{threshold}.jpg"
        else:
            filename = f"f1_vs_particles_init_{init_strategy}_move_{movement_strategy}_aggregated.jpg"
        filepath = save_path / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {filename}")
    
    def generate_heatmap_charts(self,
                               threshold: float = 0.1,
                               filter_by_threshold: bool = True,
                               save_dir: str = "analysis_charts",
                               figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Generate heatmap charts showing F1 score vs num_particles and avg_node_pot_threshold
        for different parameter combinations.
        
        Args:
            threshold: Fixed positive_cluster_threshold value (used only if filter_by_threshold=True)
            filter_by_threshold: If True, filter by threshold; if False, aggregate across all thresholds
            save_dir: Directory to save charts
            figsize: Figure size for charts
        """
        if filter_by_threshold:
            print(f"\nGenerating heatmap charts (threshold={threshold})...")
        else:
            print(f"\nGenerating heatmap charts (aggregated across all thresholds)...")
        
        # Create save directory
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        # Filter data by threshold or use all data
        if filter_by_threshold:
            filtered_data = self.filter_data_by_threshold(threshold)
        else:
            # For heatmaps, we need to aggregate across positive_cluster_threshold
            # but keep avg_node_pot_threshold for the heatmap
            filtered_data = {}
            for dataset, df in self.summary_data.items():
                filtered_df = df.copy()
                filtered_data[dataset] = filtered_df
            
            # Aggregate across positive_cluster_threshold but keep avg_node_pot_threshold
            print("Aggregating data across positive_cluster_threshold values...")
            filtered_data = self.aggregate_data_by_particles(filtered_data, include_threshold_agg=True, keep_avg_node_pot=True)
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
                filter_by_threshold,
                save_path,
                figsize
            )
        
        print(f"\nHeatmap charts saved to: {save_path}")
    
    def _create_single_heatmap(self,
                              filtered_data: Dict[str, pd.DataFrame],
                              init_strategy: str,
                              movement_strategy: str,
                              threshold: float,
                              filter_by_threshold: bool,
                              save_path: Path,
                              figsize: Tuple[int, int]) -> None:
        """
        Create a single heatmap chart for a specific parameter combination.
        """
        # Create subplots for each dataset
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        chart_title = f"F1 Score Heatmap\n"
        chart_title += f"Init: {init_strategy}, Move: {movement_strategy}"
        if filter_by_threshold:
            chart_title += f", Threshold: {threshold}"
        else:
            chart_title += ", Aggregated across all thresholds"
        
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
        if filter_by_threshold:
            filename = f"heatmap_init_{init_strategy}_move_{movement_strategy}_thresh_{threshold}.jpg"
        else:
            filename = f"heatmap_init_{init_strategy}_move_{movement_strategy}_aggregated.jpg"
        filepath = save_path / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {filename}")
    
    def generate_parameter_comparison_chart(self,
                                          threshold: float = 0.1,
                                          filter_by_threshold: bool = True,
                                          save_dir: str = "analysis_charts",
                                          figsize: Tuple[int, int] = (15, 10)) -> None:
        """
        Generate a comprehensive chart comparing all parameter combinations.
        
        Args:
            threshold: Fixed positive_cluster_threshold value (used only if filter_by_threshold=True)
            filter_by_threshold: If True, filter by threshold; if False, aggregate across all thresholds
            save_dir: Directory to save charts
            figsize: Figure size for charts
        """
        if filter_by_threshold:
            print(f"\nGenerating comprehensive parameter comparison chart (threshold={threshold})...")
        else:
            print(f"\nGenerating comprehensive parameter comparison chart (aggregated across all thresholds)...")
        
        # Create save directory
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        # Filter data by threshold or use all data
        if filter_by_threshold:
            filtered_data = self.filter_data_by_threshold(threshold)
        else:
            filtered_data = {}
            for dataset, df in self.summary_data.items():
                filtered_df = df.copy()
                filtered_data[dataset] = filtered_df
        
        # Aggregate data to handle multiple avg_node_pot_threshold values
        print("Aggregating data across avg_node_pot_threshold values...")
        aggregated_data = self.aggregate_data_by_particles(filtered_data, include_threshold_agg=not filter_by_threshold)
        
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
                    subset['num_particles'].values,
                    subset['f1_score_mean'].values,
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
        
        if filter_by_threshold:
            plt.suptitle(f'F1 Score vs Number of Particles (Threshold={threshold})', 
                        fontsize=16, fontweight='bold')
        else:
            plt.suptitle('F1 Score vs Number of Particles (Aggregated across all thresholds)', 
                        fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save the chart
        if filter_by_threshold:
            filename = f"comprehensive_f1_vs_particles_thresh_{threshold}.jpg"
        else:
            filename = f"comprehensive_f1_vs_particles_aggregated.jpg"
        filepath = save_path / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {filename}")

    def generate_publication_grid_chart(self,
                                        save_dir: str = "analysis_charts",
                                        filename: str = "publication_f1_vs_particles_grid",
                                        figsize: Tuple[int, int] = (12, 14)) -> None:
        """
        Generate a publication-ready 3×2 grid: one subplot per dataset (5 datasets),
        strategies as lines, shared legend in the 6th cell.
        Saves both PDF (for LaTeX) and JPG versions.
        """
        print("\nGenerating publication grid chart (3×2, per-dataset, strategies as lines)...")

        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)

        # Support pubmed in addition to the default 4 datasets
        all_datasets = list(self.datasets) + [d for d in ['pubmed'] if d not in self.datasets]

        # Column positions that are stable regardless of total column count (CSV row 3 = header):
        #   0:percent_positive  1:num_particles  2:cluster_strategy  3:positive_cluster_threshold
        #   4:movement_strategy  5:initialization_strategy  6:avg_node_pot_threshold
        #   7:step1_f1_score_mean  8:step1_f1_score_std
        COLS = {
            'percent_positive': 0, 'num_particles': 1, 'cluster_strategy': 2,
            'positive_cluster_threshold': 3, 'movement_strategy': 4,
            'initialization_strategy': 5, 'avg_node_pot_threshold': 6,
            'f1_score_mean': 7, 'f1_score_std': 8,
        }

        def _load_summary(fp):
            raw = pd.read_csv(fp, skiprows=3, header=None)
            df = raw.iloc[:, list(COLS.values())].copy()
            df.columns = list(COLS.keys())
            for col in ('num_particles', 'f1_score_mean', 'f1_score_std',
                        'positive_cluster_threshold', 'avg_node_pot_threshold'):
                df[col] = pd.to_numeric(df[col], errors='coerce')
            return df

        for dataset in all_datasets:
            if dataset not in self.summary_data:
                fp = self.results_dir / f"{dataset}_summary_results.csv"
                if fp.exists():
                    try:
                        self.summary_data[dataset] = _load_summary(fp)
                        print(f"  ✓ Loaded {dataset} on demand")
                    except Exception as e:
                        print(f"  ✗ Could not load {dataset}: {e}")
            elif 'f1_score_mean' not in self.summary_data[dataset].columns:
                # Cached data has wrong columns (old loader) — reload
                fp = self.results_dir / f"{dataset}_summary_results.csv"
                if fp.exists():
                    try:
                        self.summary_data[dataset] = _load_summary(fp)
                        print(f"  ✓ Reloaded {dataset} (column fix)")
                    except Exception as e:
                        print(f"  ✗ Could not reload {dataset}: {e}")

        # Only plot datasets that have data
        datasets_to_plot = [d for d in all_datasets if d in self.summary_data]
        print(f"  Datasets in grid: {datasets_to_plot}")

        # Aggregate across all thresholds and avg_node_pot_threshold
        raw = {d: self.summary_data[d] for d in datasets_to_plot}
        aggregated = self.aggregate_data_by_particles(raw, include_threshold_agg=True)

        # Strategy definitions — color + linestyle + marker
        strategy_styles = {
            ('random',          'uniform'):          ('#0072B2', '-',  'o', 'Init: Random, Move: Uniform'),
            ('random',          'degree_weighted'):  ('#D55E00', '--', 's', 'Init: Random, Move: Degree-Weighted'),
            ('degree_weighted', 'uniform'):          ('#009E73', '-.', '^', 'Init: Degree-Weighted, Move: Uniform'),
            ('degree_weighted', 'degree_weighted'):  ('#CC79A7', ':',  'D', 'Init: Degree-Weighted, Move: Degree-Weighted'),
        }

        # Global y-axis range for comparability across panels
        all_f1 = []
        for df in aggregated.values():
            all_f1.extend(df['f1_score_mean'].dropna().tolist())
        y_min = max(0.0, np.percentile(all_f1, 2) - 0.03) if all_f1 else 0.0
        y_max = min(1.0, np.percentile(all_f1, 98) + 0.03) if all_f1 else 1.0

        # Build 2×3 grid (3 datasets top row, 2 + legend bottom row)
        n_datasets = len(datasets_to_plot)
        n_rows, n_cols = 2, 3
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes_flat = axes.flatten()

        legend_handles = []
        legend_labels  = []

        for idx, dataset in enumerate(datasets_to_plot):
            ax = axes_flat[idx]
            df = aggregated[dataset]

            for (init_s, move_s), (color, ls, marker, label) in strategy_styles.items():
                mask = (df['initialization_strategy'] == init_s) & (df['movement_strategy'] == move_s)
                subset = df[mask].sort_values('num_particles')
                if subset.empty:
                    continue

                line, = ax.plot(
                    subset['num_particles'].values,
                    subset['f1_score_mean'].values,
                    color=color, linestyle=ls, marker=marker,
                    linewidth=2, markersize=5,
                    label=label,
                )
                if idx == 0 and label not in legend_labels:
                    legend_handles.append(line)
                    legend_labels.append(label)

            ax.set_title(dataset.upper(), fontsize=13, fontweight='bold')
            ax.set_xlabel('Number of Particles', fontsize=10)
            ax.set_ylabel('F1 Score', fontsize=10)
            ax.set_ylim(y_min, y_max)
            ax.grid(True, alpha=0.25, linestyle='--')
            ax.tick_params(labelsize=9)

        # 6th cell: legend panel
        legend_ax = axes_flat[n_datasets]
        legend_ax.axis('off')
        legend_ax.legend(
            legend_handles, legend_labels,
            loc='center', fontsize=11,
            frameon=True, framealpha=0.9,
            title='Particle Strategy', title_fontsize=12,
            handlelength=3, handleheight=1.2,
        )

        # Hide any remaining empty axes beyond the legend
        for idx in range(n_datasets + 1, n_rows * n_cols):
            axes_flat[idx].set_visible(False)

        fig.suptitle(
            'F1 Score vs Number of Particles\n'
            '(Aggregated across all thresholds)',
            fontsize=14, fontweight='bold', y=1.01
        )
        plt.tight_layout()

        for ext in ('pdf', 'jpg'):
            out = save_path / f"{filename}.{ext}"
            dpi = 300 if ext == 'jpg' else None
            fig.savefig(out, dpi=dpi, bbox_inches='tight')
            print(f"  ✓ Saved: {out.name}")

        plt.close()

    def generate_publication_pot_threshold_grid(self,
                                                save_dir: str = "analysis_charts",
                                                filename: str = "publication_f1_vs_pot_threshold_grid",
                                                figsize: Tuple[int, int] = (12, 14)) -> None:
        """
        Publication-ready 3×2 grid: one subplot per dataset, lines = number of particles,
        x-axis = avg_node_pot_threshold, aggregated across all strategies.
        Saves both PDF (for LaTeX) and JPG versions.
        """
        print("\nGenerating publication pot-threshold grid chart...")

        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)

        all_datasets = list(self.datasets) + [d for d in ['pubmed'] if d not in self.datasets]

        # Column positions (stable across CSV versions)
        COLS = {
            'percent_positive': 0, 'num_particles': 1, 'cluster_strategy': 2,
            'positive_cluster_threshold': 3, 'movement_strategy': 4,
            'initialization_strategy': 5, 'avg_node_pot_threshold': 6,
            'f1_score_mean': 7, 'f1_score_std': 8,
        }

        def _load_summary(fp):
            raw = pd.read_csv(fp, skiprows=3, header=None)
            df = raw.iloc[:, list(COLS.values())].copy()
            df.columns = list(COLS.keys())
            for col in ('num_particles', 'f1_score_mean', 'f1_score_std',
                        'positive_cluster_threshold', 'avg_node_pot_threshold'):
                df[col] = pd.to_numeric(df[col], errors='coerce')
            return df

        for dataset in all_datasets:
            if dataset not in self.summary_data:
                fp = self.results_dir / f"{dataset}_summary_results.csv"
                if fp.exists():
                    try:
                        self.summary_data[dataset] = _load_summary(fp)
                        print(f"  ✓ Loaded {dataset} on demand")
                    except Exception as e:
                        print(f"  ✗ Could not load {dataset}: {e}")
            elif 'f1_score_mean' not in self.summary_data[dataset].columns:
                fp = self.results_dir / f"{dataset}_summary_results.csv"
                if fp.exists():
                    try:
                        self.summary_data[dataset] = _load_summary(fp)
                        print(f"  ✓ Reloaded {dataset} (column fix)")
                    except Exception as e:
                        print(f"  ✗ Could not reload {dataset}: {e}")

        datasets_to_plot = [d for d in all_datasets if d in self.summary_data]
        print(f"  Datasets in grid: {datasets_to_plot}")

        # Aggregate across all strategies and positive_cluster_threshold,
        # keeping num_particles and avg_node_pot_threshold as grouping dimensions
        aggregated = {}
        for dataset in datasets_to_plot:
            df = self.summary_data[dataset]
            grp = df.groupby(['num_particles', 'avg_node_pot_threshold']).agg(
                f1_score_mean=('f1_score_mean', 'mean'),
            ).reset_index()
            aggregated[dataset] = grp
            print(f"  {dataset}: {len(grp)} combinations")

        particle_counts = sorted(aggregated[datasets_to_plot[0]]['num_particles'].unique())

        # Sequential color palette + distinct linestyle + marker
        cmap = plt.cm.Blues
        n = len(particle_counts)
        colors     = [cmap(0.3 + 0.7 * i / (n - 1)) for i in range(n)]
        linestyles = ['-', '--', '-.', ':', '-', '--']
        markers    = ['o', 's', '^', 'D', 'v', 'P']

        particle_style = {
            int(p): (colors[i], linestyles[i], markers[i], f'{int(p)} particles')
            for i, p in enumerate(particle_counts)
        }

        # Global y-axis range
        all_f1 = []
        for df in aggregated.values():
            all_f1.extend(df['f1_score_mean'].dropna().tolist())
        y_min = max(0.0, np.percentile(all_f1, 2) - 0.03) if all_f1 else 0.0
        y_max = min(1.0, np.percentile(all_f1, 98) + 0.03) if all_f1 else 1.0

        # Build 2×3 grid (3 datasets top row, 2 + legend bottom row)
        n_datasets = len(datasets_to_plot)
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes_flat = axes.flatten()

        legend_handles = []
        legend_labels  = []

        for idx, dataset in enumerate(datasets_to_plot):
            ax = axes_flat[idx]
            df = aggregated[dataset]

            for p in particle_counts:
                color, ls, marker, label = particle_style[int(p)]
                subset = df[df['num_particles'] == p].sort_values('avg_node_pot_threshold')
                if subset.empty:
                    continue

                line, = ax.plot(
                    subset['avg_node_pot_threshold'].values,
                    subset['f1_score_mean'].values,
                    color=color, linestyle=ls, marker=marker,
                    linewidth=2, markersize=5, label=label,
                )
                if idx == 0 and label not in legend_labels:
                    legend_handles.append(line)
                    legend_labels.append(label)

            ax.set_title(dataset.upper(), fontsize=13, fontweight='bold')
            ax.set_xlabel('avg_pot_threshold', fontsize=10)
            ax.set_ylabel('F1 Score', fontsize=10)
            ax.set_ylim(y_min, y_max)
            ax.grid(True, alpha=0.25, linestyle='--')
            ax.tick_params(labelsize=9)

        # 6th cell: legend panel
        legend_ax = axes_flat[n_datasets]
        legend_ax.axis('off')
        legend_ax.legend(
            legend_handles, legend_labels,
            loc='center', fontsize=11,
            frameon=True, framealpha=0.9,
            title='Number of Particles', title_fontsize=12,
            handlelength=3, handleheight=1.2,
        )

        for idx in range(n_datasets + 1, 2 * 3):
            axes_flat[idx].set_visible(False)

        fig.suptitle(
            'F1 Score vs avg_pot_threshold\n'
            '(Aggregated across all strategies)',
            fontsize=14, fontweight='bold', y=1.01
        )
        plt.tight_layout()

        for ext in ('pdf', 'jpg'):
            out = save_path / f"{filename}.{ext}"
            dpi = 300 if ext == 'jpg' else None
            fig.savefig(out, dpi=dpi, bbox_inches='tight')
            print(f"  ✓ Saved: {out.name}")

        plt.close()

    def generate_statistical_summary(self,
                                   threshold: float = 0.1,
                                   filter_by_threshold: bool = True,
                                   save_dir: str = "analysis_charts") -> None:
        """
        Generate statistical summary of the results.
        
        Args:
            threshold: Fixed positive_cluster_threshold value (used only if filter_by_threshold=True)
            filter_by_threshold: If True, filter by threshold; if False, aggregate across all thresholds
            save_dir: Directory to save summary
        """
        if filter_by_threshold:
            print(f"\nGenerating statistical summary (threshold={threshold})...")
        else:
            print(f"\nGenerating statistical summary (aggregated across all thresholds)...")
        
        # Create save directory
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        # Filter data by threshold or use all data
        if filter_by_threshold:
            filtered_data = self.filter_data_by_threshold(threshold)
        else:
            filtered_data = {}
            for dataset, df in self.summary_data.items():
                filtered_df = df.copy()
                filtered_data[dataset] = filtered_df
            
            # Aggregate across positive_cluster_threshold
            print("Aggregating data across positive_cluster_threshold values...")
            filtered_data = self.aggregate_data_by_particles(filtered_data, include_threshold_agg=True)
        
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
        if filter_by_threshold:
            summary_file = save_path / f"statistical_summary_thresh_{threshold}.csv"
        else:
            summary_file = save_path / f"statistical_summary_aggregated.csv"
        summary_df.to_csv(summary_file, index=False)
        
        # Print summary
        if filter_by_threshold:
            print(f"\nStatistical Summary (Threshold={threshold}):")
        else:
            print(f"\nStatistical Summary (Aggregated across all thresholds):")
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
                         filter_by_threshold: bool = True,
                         save_dir: str = "analysis_charts") -> None:
        """
        Run the complete analysis pipeline.
        
        Args:
            threshold: Fixed positive_cluster_threshold value (used only if filter_by_threshold=True)
            filter_by_threshold: If True, filter by threshold; if False, aggregate across all thresholds
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
        self.generate_f1_vs_particles_charts(threshold=threshold, filter_by_threshold=filter_by_threshold, save_dir=save_dir)
        self.generate_heatmap_charts(threshold=threshold, filter_by_threshold=filter_by_threshold, save_dir=save_dir)
        self.generate_parameter_comparison_chart(threshold=threshold, filter_by_threshold=filter_by_threshold, save_dir=save_dir)
        self.generate_publication_grid_chart(save_dir=save_dir)
        self.generate_publication_pot_threshold_grid(save_dir=save_dir)

        # Generate statistical summary
        self.generate_statistical_summary(threshold=threshold, filter_by_threshold=filter_by_threshold, save_dir=save_dir)
        
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
        '--filter-by-threshold', 
        action='store_true',
        default=True,
        help='Filter by threshold (default: True)'
    )
    parser.add_argument(
        '--no-filter-by-threshold', 
        action='store_false',
        dest='filter_by_threshold',
        help='Aggregate across all thresholds instead of filtering'
    )
    parser.add_argument(
        '--save-dir', 
        default='analysis_charts',
        help='Directory to save analysis outputs (default: analysis_charts)'
    )
    
    args = parser.parse_args()
    
    # Create analyzer and run analysis
    analyzer = ExperimentResultsAnalyzer(results_dir=args.results_dir)
    analyzer.run_full_analysis(threshold=args.threshold, filter_by_threshold=args.filter_by_threshold, save_dir=args.save_dir)


if __name__ == "__main__":
    main()
