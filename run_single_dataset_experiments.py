#!/usr/bin/env python3
"""
Run multiple experiments for a single dataset.
Usage: python3 run_single_dataset_experiments.py <dataset_name> [n_samples] [percent_positive] [n_runs]
"""

import sys
from models_experiment import run_multiple_experiments, save_results_to_csv

def main():
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python3 run_single_dataset_experiments.py <dataset_name> [n_samples] [percent_positive] [n_runs]")
        print("Example: python3 run_single_dataset_experiments.py cora 1000 0.1 10")
        sys.exit(1)
    
    dataset_name = sys.argv[1]
    n_samples = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] != 'None' else None
    percent_positive = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
    n_runs = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    
    print(f"Running {n_runs} experiments for {dataset_name}")
    print(f"Parameters: n_samples={n_samples}, percent_positive={percent_positive*100}%")
    
    # Run experiments
    results_df, mean_results = run_multiple_experiments(
        dataset_name=dataset_name,
        n_samples=n_samples,
        percent_positive=percent_positive,
        n_runs=n_runs
    )
    
    # Save results
    detailed_file, summary_file = save_results_to_csv(
        results_df, mean_results, dataset_name, n_samples, percent_positive
    )
    
    print(f"\n✓ Experiments completed!")
    print(f"Results saved to:")
    print(f"  - {detailed_file}")
    print(f"  - {summary_file}")

if __name__ == "__main__":
    main()
