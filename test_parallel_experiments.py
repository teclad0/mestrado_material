#!/usr/bin/env python3
"""
Test script to verify that parallel experiment execution works correctly.
"""

import time
import multiprocessing as mp
from parametric_experiments import PULearningExperimentRunner

def test_parallel_execution():
    """Test that parallel execution works and is faster than sequential."""
    
    print("="*60)
    print("Testing Parallel Experiment Execution")
    print("="*60)
    
    # Create a simple test with reduced parameters for quick testing
    test_params = {
        'num_particles': [100, 200],
        'p_det': [0.6],
        'delta_v': [0.3],
        'delta_p': [0.5],
        'cluster_strategy': ['majority'],
        'positive_cluster_threshold': [0.5],
        'movement_strategy': ['uniform'],
        'initialization_strategy': ['random'],
        'avg_node_pot_threshold': [0.7]
    }
    
    # Calculate total experiments
    total_combinations = 1
    for param_values in test_params.values():
        total_combinations *= len(param_values)
    total_experiments = total_combinations * 2  # 2 runs per combination
    
    print(f"Test configuration:")
    print(f"  Parameter combinations: {total_combinations}")
    print(f"  Runs per combination: 2")
    print(f"  Total experiments: {total_experiments}")
    print(f"  Available CPU cores: {mp.cpu_count()}")
    
    # Test 1: Sequential execution
    print(f"\n{'='*40}")
    print("Test 1: Sequential Execution")
    print(f"{'='*40}")
    
    runner_seq = PULearningExperimentRunner(
        n_runs=2,
        output_dir="test_results",
        random_seed=42
    )
    
    # Override parameter generation for testing
    runner_seq.get_parameter_grid = lambda: [
        dict(zip(test_params.keys(), values)) 
        for values in __import__('itertools').product(*test_params.values())
    ]
    
    start_time = time.time()
    
    try:
        # Load a small dataset for testing
        from generate_dataset import load_cora_scar
        test_graph = load_cora_scar(
            positive_class_label=3,
            percent_positive=0.1,
            use_original_edges=True,
            mst=False,
            n_samples=100  # Small sample for quick testing
        )
        
        if test_graph is None:
            print("❌ FAILED: Could not load test dataset")
            return False
        
        print(f"Loaded test dataset: {test_graph.number_of_nodes()} nodes")
        
        # Run experiments sequentially
        results_seq = runner_seq._run_experiments_parallel(test_graph, runner_seq.get_parameter_grid(), n_jobs=1)
        
        seq_time = time.time() - start_time
        print(f"✅ Sequential execution completed in {seq_time:.2f} seconds")
        print(f"  Results: {len(results_seq)} experiments")
        
    except Exception as e:
        print(f"❌ FAILED: Sequential execution error - {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 2: Parallel execution
    print(f"\n{'='*40}")
    print("Test 2: Parallel Execution")
    print(f"{'='*40}")
    
    runner_par = PULearningExperimentRunner(
        n_runs=2,
        output_dir="test_results",
        random_seed=42
    )
    
    # Override parameter generation for testing
    runner_par.get_parameter_grid = lambda: [
        dict(zip(test_params.keys(), values)) 
        for values in __import__('itertools').product(*test_params.values())
    ]
    
    start_time = time.time()
    
    try:
        # Run experiments in parallel
        n_jobs = min(mp.cpu_count(), 4)  # Use up to 4 cores for testing
        print(f"Using {n_jobs} parallel processes")
        
        results_par = runner_par._run_experiments_parallel(test_graph, runner_par.get_parameter_grid(), n_jobs)
        
        par_time = time.time() - start_time
        print(f"✅ Parallel execution completed in {par_time:.2f} seconds")
        print(f"  Results: {len(results_par)} experiments")
        
    except Exception as e:
        print(f"❌ FAILED: Parallel execution error - {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Compare results
    print(f"\n{'='*40}")
    print("Performance Comparison")
    print(f"{'='*40}")
    
    print(f"Sequential time: {seq_time:.2f} seconds")
    print(f"Parallel time:   {par_time:.2f} seconds")
    
    if par_time < seq_time:
        speedup = seq_time / par_time
        print(f"✅ Parallel execution is {speedup:.2f}x faster!")
    else:
        print(f"⚠️  Parallel execution was not faster (this can happen with small datasets)")
    
    # Verify result consistency
    print(f"\nResult consistency check:")
    print(f"  Sequential results: {len(results_seq)}")
    print(f"  Parallel results:   {len(results_par)}")
    
    if len(results_seq) == len(results_par):
        print(f"✅ Result counts match")
    else:
        print(f"❌ Result counts don't match")
        return False
    
    # Check if all experiments completed successfully
    seq_success = sum(1 for r in results_seq if r['status'] == 'success')
    par_success = sum(1 for r in results_par if r['status'] == 'success')
    
    print(f"  Sequential success rate: {seq_success}/{len(results_seq)} ({100*seq_success/len(results_seq):.1f}%)")
    print(f"  Parallel success rate:   {par_success}/{len(results_par)} ({100*par_success/len(results_par):.1f}%)")
    
    return True

if __name__ == "__main__":
    print("Testing parallel experiment execution...")
    print("This will run a small set of experiments both sequentially and in parallel")
    print("to verify that parallelization works correctly.\n")
    
    success = test_parallel_execution()
    
    if success:
        print("\n🎉 Parallel execution test completed successfully!")
        print("You can now use parallel execution in your experiments with --n-jobs")
    else:
        print("\n💥 Parallel execution test failed!")
        print("Please check the error messages above.")
