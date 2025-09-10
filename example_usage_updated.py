#!/usr/bin/env python3
"""
Example usage of the updated analyze_results.py script with the new filter_by_threshold option.
"""

from analyze_results import ExperimentResultsAnalyzer

def main():
    """Demonstrate both filtering and aggregation modes."""
    
    # Create analyzer
    analyzer = ExperimentResultsAnalyzer(results_dir="experiment_results")
    
    # Load data
    analyzer.load_summary_results()
    
    if not analyzer.summary_data:
        print("No summary data found! Please check the results directory.")
        return
    
    print("=" * 80)
    print("Example 1: Filter by threshold (default behavior)")
    print("=" * 80)
    
    # Example 1: Filter by threshold (original behavior)
    analyzer.generate_f1_vs_particles_charts(
        threshold=0.1,
        filter_by_threshold=True,
        save_dir="analysis_charts_filtered"
    )
    
    print("\n" + "=" * 80)
    print("Example 2: Aggregate across all thresholds (new behavior)")
    print("=" * 80)
    
    # Example 2: Aggregate across all thresholds (new behavior)
    analyzer.generate_f1_vs_particles_charts(
        threshold=0.1,  # This parameter is ignored when filter_by_threshold=False
        filter_by_threshold=False,
        save_dir="analysis_charts_aggregated"
    )
    
    print("\n" + "=" * 80)
    print("Example 3: Full analysis with aggregation")
    print("=" * 80)
    
    # Example 3: Run full analysis with aggregation
    analyzer.run_full_analysis(
        threshold=0.1,
        filter_by_threshold=False,
        save_dir="analysis_charts_full_aggregated"
    )
    
    print("\n" + "=" * 80)
    print("Examples completed!")
    print("=" * 80)
    print("Generated files:")
    print("  - analysis_charts_filtered/: Charts filtered by threshold=0.1")
    print("  - analysis_charts_aggregated/: Charts aggregated across all thresholds")
    print("  - analysis_charts_full_aggregated/: Complete analysis aggregated across all thresholds")

if __name__ == "__main__":
    main()
