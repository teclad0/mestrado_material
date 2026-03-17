#!/usr/bin/env python3
"""
Example usage of the updated run_multiple_experiments function
"""

from models_experiment import run_multiple_experiments, save_results_to_csv

def example_single_percent_positive():
    """Example with single percent_positive value (backward compatibility)"""
    print("=== Example: Single percent_positive value ===")
    
    results_df, summary_df = run_multiple_experiments(
        dataset_name="cora", 
        n_samples=100, 
        percent_positive=0.1,  # Single value
        n_runs=3,
        save_results=True  # Automatically saves results (default)
    )
    
    print(f"Results DataFrame shape: {results_df.shape}")
    print(f"Summary DataFrame shape: {summary_df.shape}")
    print(f"Unique percent_positive values: {results_df['percent_positive'].unique()}")
    
    return results_df, summary_df

def example_multiple_percent_positive():
    """Example with multiple percent_positive values"""
    print("\n=== Example: Multiple percent_positive values ===")
    
    results_df, summary_df = run_multiple_experiments(
        dataset_name="cora", 
        n_samples=100, 
        percent_positive=[0.05, 0.1, 0.2, 0.3],  # List of values
        n_runs=3,
        save_results=True  # Automatically saves results (default)
    )
    
    print(f"Results DataFrame shape: {results_df.shape}")
    print(f"Summary DataFrame shape: {summary_df.shape}")
    print(f"Unique percent_positive values: {results_df['percent_positive'].unique()}")
    
    # The summary_df will have one row per percent_positive value
    print("\nSummary by percent_positive:")
    for _, row in summary_df.iterrows():
        print(f"  {row['percent_positive']*100}%: PULearningPC={row['mean_PULearningPC_f1']:.4f}, "
              f"MCLS={row['mean_MCLS_f1']:.4f}, LP_PUL={row['mean_LP_PUL_f1']:.4f}")
    
    return results_df, summary_df

def example_no_saving():
    """Example with saving disabled"""
    print("\n=== Example: No automatic saving ===")
    
    results_df, summary_df = run_multiple_experiments(
        dataset_name="cora", 
        n_samples=100, 
        percent_positive=[0.1, 0.2],  # List of values
        n_runs=2,
        save_results=False  # Disable automatic saving
    )
    
    print(f"Results DataFrame shape: {results_df.shape}")
    print(f"Summary DataFrame shape: {summary_df.shape}")
    print("Results not automatically saved (save_results=False)")
    
    return results_df, summary_df

if __name__ == "__main__":
    print("Updated run_multiple_experiments function examples")
    print("=" * 60)
    
    # Example 1: Single percent_positive (backward compatibility)
    example_single_percent_positive()
    
    # Example 2: Multiple percent_positive values
    example_multiple_percent_positive()
    
    # Example 3: No automatic saving
    example_no_saving()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("\nKey features:")
    print("- Backward compatible: single percent_positive value works as before")
    print("- New functionality: pass a list of percent_positive values")
    print("- Separate summaries: each percent_positive gets its own summary row")
    print("- Automatic saving: results are automatically saved to CSV files")
    print("- Optional saving: set save_results=False to disable automatic saving")
    print("- Easy analysis: summary_df allows easy comparison across percent_positive values")