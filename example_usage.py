#!/usr/bin/env python3
"""
Example usage of the PULearningPC parametric experiment framework.
This script demonstrates different ways to run experiments.
"""

from parametric_experiments import PULearningExperimentRunner
from experiment_config import EXPERIMENT_CONFIG, DATASET_CONFIG, QUICK_TEST_PARAMS
import pandas as pd

def example_1_basic_usage():
    """Basic usage example - run experiments on Cora dataset."""
    print("="*60)
    print("EXAMPLE 1: Basic Usage")
    print("="*60)
    
    # Initialize runner with default settings
    runner = PULearningExperimentRunner(
        n_runs=3,  # Reduced for faster testing
        output_dir="example_results",
        random_seed=42
    )
    
    # Run experiments on Cora dataset
    results = runner.run_experiments('cora')
    
    print(f"Completed {len(results)} experiments")
    print(f"Results saved in: {runner.output_dir}/")
    
    return results

def example_2_custom_parameters():
    """Example with custom parameter ranges."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Custom Parameters")
    print("="*60)
    
    # Create custom parameter ranges
    custom_params = {
        'num_particles': [100, 200],           # Only test 2 values
        'p_det': [0.6],                       # Fixed as requested
        'delta_v': [0.2, 0.4],                # Only test 2 values
        'delta_p': [0.5, 0.8],                # Only test 2 values
        'cluster_strategy': ['majority'],      # Only test majority
        'positive_cluster_threshold': [0.1, 0.3],  # Only test 2 values
        'movement_strategy': ['uniform'],      # Only test uniform
        'initialization_strategy': ['random'], # Only test random
        'avg_node_pot_threshold': [0.8]       # Only test 0.8
    }
    
    # Override the parameter grid method
    runner = PULearningExperimentRunner(
        n_runs=2,  # Reduced for faster testing
        output_dir="custom_example_results",
        random_seed=42
    )
    
    # Override parameter generation
    runner.get_parameter_grid = lambda: [
        dict(zip(custom_params.keys(), values)) 
        for values in __import__('itertools').product(*custom_params.values())
    ]
    
    # Run experiments
    results = runner.run_experiments('cora')
    
    print(f"Completed {len(results)} experiments with custom parameters")
    return results

def example_3_multiple_datasets():
    """Example running experiments on multiple datasets."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Multiple Datasets")
    print("="*60)
    
    runner = PULearningExperimentRunner(
        n_runs=2,  # Reduced for faster testing
        output_dir="multi_dataset_results",
        random_seed=42
    )
    
    # Test on multiple datasets
    datasets = ['cora', 'citeseer']
    all_results = {}
    
    for dataset in datasets:
        print(f"\nRunning experiments on {dataset}...")
        try:
            results = runner.run_experiments(dataset)
            all_results[dataset] = results
            print(f"Completed {dataset} experiments")
        except Exception as e:
            print(f"Error with {dataset}: {e}")
    
    return all_results

def example_4_analyze_results():
    """Example of analyzing experiment results."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Analyzing Results")
    print("="*60)
    
    # Load results from a previous run
    try:
        results_df = pd.read_csv("example_results/cora_final_results.csv")
        
        print(f"Loaded {len(results_df)} experiment results")
        
        # Filter successful runs
        successful = results_df[results_df['status'] == 'success']
        print(f"Successful runs: {len(successful)}")
        
        if len(successful) > 0:
            # Show best parameters
            best_run = successful.loc[successful['f1_score'].idxmax()]
            print(f"\nBest F1 Score: {best_run['f1_score']:.4f}")
            
            # Parameter importance analysis
            print("\nParameter Analysis:")
            for param in ['num_particles', 'delta_v', 'delta_p', 'cluster_strategy']:
                if param in successful.columns:
                    param_values = successful[param].unique()
                    print(f"  {param}: {list(param_values)}")
            
            # Group by strategy and show performance
            if 'cluster_strategy' in successful.columns:
                strategy_performance = successful.groupby('cluster_strategy')['f1_score'].agg(['mean', 'std'])
                print(f"\nStrategy Performance:\n{strategy_performance}")
        
    except FileNotFoundError:
        print("No results file found. Run examples 1-3 first.")

def main():
    """Run all examples."""
    print("PULearningPC Parametric Experiment Examples")
    print("="*60)
    
    # Run examples
    try:
        # Example 1: Basic usage
        results1 = example_1_basic_usage()
        
        # Example 2: Custom parameters
        results2 = example_2_custom_parameters()
        
        # Example 3: Multiple datasets
        results3 = example_3_multiple_datasets()
        
        # Example 4: Analyze results
        example_4_analyze_results()
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 