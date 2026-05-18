#!/usr/bin/env python3
"""
Command-line script to run parametric experiments for PULearningPC algorithm.
This script allows you to run experiments with different parameter combinations
and save results to CSV files.
"""

import argparse
import sys
import os
import multiprocessing as mp
from parametric_experiments import PULearningExperimentRunner
from experiment_config import (
    PARAMETER_RANGES, QUICK_TEST_PARAMS, DATASET_CONFIG, BASELINE_CONFIG
)

def main():
    parser = argparse.ArgumentParser(
        description="Run PULearningPC Parametric Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run experiments on Cora dataset with default parameters
  python run_experiments.py --dataset cora
  
  # Run experiments with custom parameter ranges
  python run_experiments.py --dataset cora --num-particles 100 200 387 --n-runs 5
  
  # Run experiments on multiple datasets
  python run_experiments.py --dataset cora citeseer --output-dir my_results
  
  # Quick test with reduced parameters
  python run_experiments.py --dataset cora --quick-test
        """
    )
    
    # Dataset and output options
    parser.add_argument(
        '--dataset', 
        nargs='+', 
        default=['cora'],
        choices=['cora', 'citeseer', 'pubmed', 'twitch', 'mnist'],
        help='Dataset(s) to run experiments on (default: cora)'
    )
    
    parser.add_argument(
        '--output-dir', 
        default='experiment_results',
        help='Output directory for results (default: experiment_results)'
    )
    
    parser.add_argument(
        '--n-runs', 
        type=int, 
        default=3,
        help='Number of runs per parameter combination (default: 3)'
    )
    
    parser.add_argument(
        '--random-seed', 
        type=int, 
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    # Parallelization options
    parser.add_argument(
        '--n-jobs', 
        type=int, 
        default=None,
        help='Number of parallel jobs (default: auto-detect, use --n-jobs 1 for sequential)'
    )
    
    # Quick test option
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Use reduced parameter ranges for quick testing'
    )
    
    # Individual parameter overrides
    parser.add_argument(
        '--num-particles',
        nargs='+',
        type=int,
        help='Number of particles to test (e.g., 100 200 387)'
    )
    
    parser.add_argument(
        '--cluster-strategy',
        nargs='+',
        choices=['majority', 'percentage'],
        help='Cluster strategy to test'
    )
    
    parser.add_argument(
        '--positive-cluster-threshold',
        nargs='+',
        type=float,
        help='Positive cluster threshold values to test'
    )
    
    parser.add_argument(
        '--movement-strategy',
        nargs='+',
        choices=['uniform', 'degree_weighted'],
        help='Movement strategy to test'
    )
    
    parser.add_argument(
        '--initialization-strategy',
        nargs='+',
        choices=['random', 'degree_weighted'],
        help='Initialization strategy to test'
    )
    
    parser.add_argument(
        '--avg-node-pot-threshold',
        nargs='+',
        type=float,
        help='Average node potential threshold values to test'
    )
    
    # Model parameters
    parser.add_argument(
        '--num-neg',
        type=int,
        default=200,
        help='Number of reliable negatives to select (default: 200)'
    )
    
    # Dataset-specific parameters
    parser.add_argument(
        '--k',
        type=int,
        default=3,
        help='k value for k-NN graph construction (default: 3)'
    )
    
    parser.add_argument(
        '--percent-positive',
        type=float,
        default=0.1,
        help='Percentage of positive examples to label (default: 0.1)'
    )
    
    parser.add_argument(
        '--use-original-edges',
        action='store_true',
        default=True,
        help='Use original edges from dataset (default: True)'
    )
    
    parser.add_argument(
        '--mst',
        action='store_true',
        help='Use minimum spanning tree for connectivity (default: False)'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set default parameter ranges from centralized config
    if args.quick_test:
        default_params = QUICK_TEST_PARAMS.copy()
    else:
        default_params = PARAMETER_RANGES.copy()
    
    # Override with command line arguments if provided
    if args.num_particles:
        default_params['num_particles'] = args.num_particles
    if args.cluster_strategy:
        default_params['cluster_strategy'] = args.cluster_strategy
    if args.positive_cluster_threshold:
        default_params['positive_cluster_threshold'] = args.positive_cluster_threshold
    if args.movement_strategy:
        default_params['movement_strategy'] = args.movement_strategy
    if args.initialization_strategy:
        default_params['initialization_strategy'] = args.initialization_strategy
    if args.avg_node_pot_threshold:
        default_params['avg_node_pot_threshold'] = args.avg_node_pot_threshold
    
    # Build dataset-specific parameter dictionaries from centralized config
    dataset_kwargs = {}
    dataset_num_neg = {name: BASELINE_CONFIG['num_neg'].get(name, args.num_neg)
                       for name in args.dataset}

    for dataset_name in args.dataset:
        base_params = DATASET_CONFIG.get(dataset_name, {}).copy()
        # Override with CLI arguments where provided
        base_params['percent_positive'] = args.percent_positive
        base_params['mst'] = args.mst
        if 'k' in base_params:
            base_params['k'] = args.k
        if 'use_original_edges' in base_params:
            base_params['use_original_edges'] = args.use_original_edges
        dataset_kwargs[dataset_name] = base_params

    print("="*80)
    print("PULearningPC Parametric Experiments")
    print("="*80)
    print(f"Datasets: {', '.join(args.dataset)}")
    print(f"Number of runs per combination: {args.n_runs}")
    print(f"Output directory: {args.output_dir}")
    print(f"Random seed: {args.random_seed}")
    print(f"Quick test mode: {args.quick_test}")
    print("="*80)
    
    # Calculate total experiments
    total_combinations = 1
    for param_values in default_params.values():
        total_combinations *= len(param_values)
    total_experiments = total_combinations * args.n_runs * len(args.dataset)
    
    print(f"Parameter combinations: {total_combinations}")
    print(f"Total experiments: {total_experiments}")
    print(f"Reliable negatives per experiment: {args.num_neg}")
    
    # Show parallelization info
    if args.n_jobs is None:
        n_jobs = min(mp.cpu_count(), 8)
        print(f"Parallelization: Auto-detected {n_jobs} processes")
    else:
        print(f"Parallelization: {args.n_jobs} processes")
    
    print("="*80)
    
    # Initialize experiment runner
    runner = PULearningExperimentRunner(
        n_runs=args.n_runs,
        output_dir=args.output_dir,
        random_seed=args.random_seed
    )
    
    # Set custom parameter ranges in the runner
    runner.set_custom_parameter_ranges(default_params)
    
    # Run experiments on each dataset
    all_results = {}
    
    for dataset_name in args.dataset:
        try:
            print(f"\n{'='*50}")
            print(f"Running experiments on {dataset_name.upper()} dataset")
            print(f"{'='*50}")
            
            # Get dataset-specific kwargs
            dataset_specific_kwargs = dataset_kwargs.get(dataset_name, {})

            # Use per-dataset num_neg, falling back to CLI arg
            num_neg = dataset_num_neg.get(dataset_name, args.num_neg)

            # Run experiments with parallelization
            results_df = runner.run_experiments(
                dataset_name,
                dataset_specific_kwargs,
                n_jobs=args.n_jobs,
                num_neg=num_neg
            )
            all_results[dataset_name] = results_df
            
            # Display summary for this dataset
            successful_results = results_df[results_df['status'] == 'success']
            if len(successful_results) > 0:
                print(f"\n{dataset_name.upper()} Results Summary:")
                print(f"  Total experiments: {len(results_df)}")
                print(f"  Successful runs: {len(successful_results)}")
                print(f"  Best Step1 F1: {successful_results['step1_f1_score'].max():.4f}")
                print(f"  Best Step2 F1: {successful_results['step2_f1'].max():.4f}")
                print(f"  Average Coverage: {successful_results['coverage'].mean():.4f}")

                # Show best parameters (by step2 F1 — end-to-end metric)
                best_run = successful_results.loc[successful_results['step2_f1'].idxmax()]
                print(f"  Best parameters: particles={best_run['num_particles']}, "
                      f"strategy={best_run['cluster_strategy']}, threshold={best_run['positive_cluster_threshold']}")
            
        except Exception as e:
            print(f"Error running experiments on {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Final summary
    print("\n" + "="*80)
    print("EXPERIMENTS COMPLETED!")
    print("="*80)
    
    for dataset_name, results_df in all_results.items():
        successful = results_df[results_df['status'] == 'success']
        print(f"{dataset_name.upper()}: {len(successful)}/{len(results_df)} successful runs")
    
    print(f"\nResults saved in: {args.output_dir}/")
    print("Files generated:")
    for dataset_name in args.dataset:
        print(f"  - {dataset_name}_final_results.csv (all results)")
        print(f"  - {dataset_name}_summary_results.csv (aggregated results)")
    
    print("\nTo analyze results, you can:")
    print("  - Load CSV files with pandas")
    print("  - Use the example_usage.py script")
    print("  - Create custom analysis scripts")

if __name__ == "__main__":
    main()