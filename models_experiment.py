#!/usr/bin/env python3
"""
Example: Using ultra_simple_dataset_system for consistent experiments
"""

from dataset_system import DatasetManager
from model import PULearningPC
from models import MCLS, LP_PUL
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
import os
import pandas as pd
import random
from typing import Union, List
from aux import dict_datasets_params_pulpc

def evaluate_f1_score(graph, reliable_negatives, true_label_key='true_label', target_negatives=None):
    """
    Evaluate F1 score for reliable negatives, handling gaps in node indices and invalid references.
    
    Args:
        graph: NetworkX graph with node attributes
        reliable_negatives: List of node indices that are predicted as negative
        true_label_key: Key for true labels in node attributes
        target_negatives: Target number of negatives (for reporting purposes)
    
    Returns:
        F1 score for negative class
    """
    if not reliable_negatives:
        return 0.0
    
    # Get node attributes dictionary for efficient lookup
    node_attributes = dict(graph.nodes(data=True))
    
    # Filter reliable negatives to only include nodes that exist in the graph
    valid_negatives = [node for node in reliable_negatives if node in graph.nodes()]
    
    if not valid_negatives:
        print(f"  Warning: No valid negatives found. {len(reliable_negatives)} provided, but none exist in graph.")
        return 0.0
    
    # Check for invalid nodes and report
    invalid_nodes = [node for node in reliable_negatives if node not in graph.nodes()]
    if invalid_nodes:
        print(f"  Warning: {len(invalid_nodes)} invalid node references found: {invalid_nodes[:5]}{'...' if len(invalid_nodes) > 5 else ''}")
    
    # Check if we have enough negatives compared to target
    if target_negatives is not None:
        actual_negatives = len(valid_negatives)
        if actual_negatives < target_negatives:
            print(f"  Warning: Found {actual_negatives} negatives (target: {target_negatives}) - returning 0 (failure)")
            return 0.0
    
    try:
        # Get true labels for valid negatives
        y_true = [node_attributes[node][true_label_key] for node in valid_negatives]
        y_pred = [0] * len(valid_negatives)  # All predicted as negative (0)
        
        return f1_score(y_true, y_pred, pos_label=0)
    
    except KeyError as e:
        print(f"  Error: Missing attribute '{e}' in node attributes")
        return 0.0
    except Exception as e:
        print(f"  Error in F1 evaluation: {e}")
        return 0.0

def run_models(dataset_name: str, n_samples: int = None, percent_positive: float = 0.1):
    """Run all three models on the same dataset."""
    
    if n_samples is not None:
        print(f"Running consistent experiment on {dataset_name} with {n_samples} samples, {percent_positive*100}% positive")
        filename = f"{dataset_name}_{n_samples}_samples.pkl"
    else:
        print(f"Running consistent experiment on {dataset_name} with all available samples, {percent_positive*100}% positive")
        filename = f"{dataset_name}_full.pkl"
    
    # Get dataset-specific parameters
    dataset_params = dict_datasets_params_pulpc().get(dataset_name, {})
    num_neg = dataset_params.get('num_neg', 200)
    rns_params = dataset_params.get('rns_params', {
        'cluster_strategy': 'percentage',
        'positive_cluster_threshold': 0.1
    })
    
    # Initialize manager
    manager = DatasetManager()
    
    # Check if dataset exists, if not generate it
    filepath = os.path.join(manager.datasets_dir, filename)
    if not os.path.exists(filepath):
        # Dataset parameters for generation
        dataset_params = {
            'cora': {
                'k': 3,
                'positive_class_label': 3,
                'use_original_edges': True,
                'mst': False
            },
            'citeseer': {
                'k': 3,
                'positive_class_label': 2,
                'use_original_edges': True,
                'mst': False
            },
            'twitch': {
                'mst': False
            },
            'mnist': {
                'k': 3,
                'mst': False,
                'n_samples': 3000
            }
        }
        
        params = dataset_params.get(dataset_name, {})
        manager.generate_and_save_dataset(dataset_name, params, n_samples, percent_positive)
    
    # Load dataset
    graph = manager.load_graph(filename)
    
    print(f"Dataset loaded: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    print(f"Using {num_neg} negatives, {percent_positive*100}% positive ratio")
    
    results = {}
    
    # Test PULearningPC
    print("\nTesting PULearningPC...")
    try:
        pulearning = PULearningPC(
            graph, 
            num_neg=num_neg,
            pcm_params={
                'num_particles': 100,
                'p_det': 0.6,
                'delta_p': 0.3,
                'delta_v': 0.4,
                'movement_strategy': 'uniform',
                'initialization_strategy': 'random',
                'average_node_potential_threshold': 0.6
            },
            rns_params=rns_params
        )
        pulearning.train()
        reliable_negatives = pulearning.select_reliable_negatives()
        
        # Use the new evaluation method
        f1 = evaluate_f1_score(graph, reliable_negatives, target_negatives=num_neg)
        results['PULearningPC'] = f1
        print(f"  PULearningPC F1: {f1:.4f} (found {len(reliable_negatives)} negatives)")
        
    except Exception as e:
        print(f"  PULearningPC failed: {e}")
        results['PULearningPC'] = 0
    
    # Test MCLS
    print("\nTesting MCLS...")
    try:
        mcls_data = manager.get_data_for_mcls(graph)
        
        # Debug MCLS data format
        print(f"  MCLS data debug:")
        print(f"    Features shape: {mcls_data.x.shape}")
        print(f"    Positives: {len(mcls_data.P)}")
        print(f"    Unlabeled: {len(mcls_data.U)}")
        print(f"    Positive indices: {mcls_data.P[:5]}...")  # Show first 5
        print(f"    Unlabeled indices: {mcls_data.U[:5]}...")  # Show first 5
        
        # Try different MCLS parameters
        print(f"    Trying MCLS with different parameters...")
        
        # Try with more clusters and different ratio
        mcls = MCLS(mcls_data, k=7, ratio=0.1)  # More clusters, lower ratio
        mcls.train()
        
        # Debug MCLS internal state
        print(f"    MCLS distance dictionary size: {len(mcls.distance)}")
        if len(mcls.distance) > 0:
            print(f"    Sample distances: {list(mcls.distance.items())[:3]}")
        else:
            print(f"    No negative clusters found! All clusters might be positive.")
        
        reliable_negatives = mcls.negative_inference(num_neg)
        
        print(f"    Reliable negatives found: {len(reliable_negatives)}")
        if len(reliable_negatives) > 0:
            print(f"    First few reliable negatives: {reliable_negatives[:5]}")
        else:
            print(f"    No reliable negatives found! Distance dict has {len(mcls.distance)} entries")
        
        # Convert indices to node IDs for evaluation
        node_list = list(graph.nodes())
        reliable_neg_nodes = [node_list[i] for i in reliable_negatives if i < len(node_list)]
        
        # Use the new evaluation method
        f1 = evaluate_f1_score(graph, reliable_neg_nodes, target_negatives=num_neg)
        results['MCLS'] = f1
        print(f"  MCLS F1: {f1:.4f} (found {len(reliable_neg_nodes)} negatives)")
        
    except Exception as e:
        print(f"  MCLS failed: {e}")
        results['MCLS'] = 0
    
    # Test LP_PUL
    print("\nTesting LP_PUL...")
    try:
        lp_pul_data = manager.get_data_for_lp_pul(graph)
        lp_pul = LP_PUL(lp_pul_data)
        lp_pul.train()
        reliable_negatives = lp_pul.negative_inference(num_neg)
        
        # Convert to list if tensor
        if hasattr(reliable_negatives, 'tolist'):
            reliable_negatives = reliable_negatives.tolist()
        
        # Convert indices to node IDs for evaluation
        node_list = list(graph.nodes())
        reliable_neg_nodes = [node_list[i] for i in reliable_negatives if i < len(node_list)]
        
        # Use the new evaluation method
        f1 = evaluate_f1_score(graph, reliable_neg_nodes, target_negatives=num_neg)
        results['LP_PUL'] = f1
        print(f"  LP_PUL F1: {f1:.4f} (found {len(reliable_neg_nodes)} negatives)")
        
    except Exception as e:
        print(f"  LP_PUL failed: {e}")
        results['LP_PUL'] = 0
    
    # Summary
    print(f"\nResults Summary:")
    for model, f1 in results.items():
        print(f"  {model}: {f1:.4f}")
    
    return results

def run_multiple_experiments(dataset_name: str, n_samples: int = None, percent_positive: Union[float, List[float]] = 0.1, n_runs: int = 10, save_results: bool = True):
    """
    Run multiple experiments with different random seeds and save results to CSV.
    
    Args:
        dataset_name: Name of the dataset
        n_samples: Number of samples to use (None for all)
        percent_positive: Percentage of positive samples (float) or list of percentages to test
        n_runs: Number of experimental runs
        save_results: Whether to automatically save results to CSV files (default: True)
    
    Returns:
        Tuple of (results_df, summary_df) where:
        - results_df: Detailed results for each run
        - summary_df: Summary statistics for each percent_positive value
    """
    # Convert single value to list for uniform processing
    if isinstance(percent_positive, (int, float)):
        percent_positives = [percent_positive]
    else:
        percent_positives = percent_positive
    
    print(f"Running {n_runs} experiments for {dataset_name} dataset...")
    print(f"Parameters: n_samples={n_samples}, percent_positives={[p*100 for p in percent_positives]}%")
    
    # Get dataset-specific parameters
    dataset_params = dict_datasets_params_pulpc().get(dataset_name, {})
    num_neg = dataset_params.get('num_neg', 200)
    rns_params = dataset_params.get('rns_params', {
        'cluster_strategy': 'percentage',
        'positive_cluster_threshold': 0.1
    })
    
    # Store results for each run and percent_positive
    all_results = []
    
    # Loop through each percent_positive value
    for percent_pos in percent_positives:
        print(f"\n{'='*60}")
        print(f"Testing percent_positive = {percent_pos*100}%")
        print(f"{'='*60}")
        
        for run in range(n_runs):
            print(f"\n--- Run {run + 1}/{n_runs} ---")
            
            # Set random seed for this run
            random_seed = 42 + run
            random.seed(random_seed)
            np.random.seed(random_seed)
            
            # Initialize manager with specific seed
            manager = DatasetManager(random_seed=random_seed)
            
            # Dataset parameters for generation
            dataset_gen_params = {
                'cora': {
                    'k': 3,
                    'positive_class_label': 3,
                    'use_original_edges': True,
                    'mst': False
                },
                'citeseer': {
                    'k': 3,
                    'positive_class_label': 2,
                    'use_original_edges': True,
                    'mst': False
                },
                'twitch': {
                    'mst': False
                },
                'mnist': {
                    'k': 3,
                    'mst': False,
                    'n_samples': 3000
                }
            }
            
            params = dataset_gen_params.get(dataset_name, {})
            # Generate dataset with run_number - this will create the appropriate filename
            filename = manager.generate_and_save_dataset(dataset_name, params, n_samples, percent_pos, run_number=run)
            
            # Load the generated dataset
            graph = manager.load_graph(filename)
            
            print(f"Dataset loaded: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
            
            # Run each model
            run_results = {
                'dataset': dataset_name,
                'n_samples': n_samples,
                'percent_positive': percent_pos,
                'num_neg': num_neg,
                'run': run + 1,
                'random_seed': random_seed,
                'n_nodes': graph.number_of_nodes(),
                'n_edges': graph.number_of_edges()
            }
            
            # Test PULearningPC
            print("  Testing PULearningPC...")
            try:
                pulearning = PULearningPC(
                    graph, 
                    num_neg=num_neg,
                    pcm_params={
                        'num_particles': 100,
                        'p_det': 0.6,
                        'delta_p': 0.3,
                        'delta_v': 0.4,
                        #'movement_strategy': 'uniform',
                        'movement_strategy': 'degree_weighted',

                        'initialization_strategy': 'random',
                        'average_node_potential_threshold': 0.6
                    },
                    rns_params=rns_params
                )
                pulearning.train()
                reliable_negatives = pulearning.select_reliable_negatives()
                
                f1 = evaluate_f1_score(graph, reliable_negatives, target_negatives=num_neg)
                run_results['PULearningPC_f1'] = f1
                run_results['PULearningPC_negatives'] = len(reliable_negatives)
                print(f"    PULearningPC F1: {f1:.4f} (found {len(reliable_negatives)} negatives)")
                
            except Exception as e:
                print(f"    PULearningPC failed: {e}")
                run_results['PULearningPC_f1'] = 0
                run_results['PULearningPC_negatives'] = 0
            
            # Test MCLS
            print("  Testing MCLS...")
            try:
                mcls_data = manager.get_data_for_mcls(graph)
                mcls = MCLS(mcls_data, k=7, ratio=0.1)
                mcls.train()
                reliable_negatives = mcls.negative_inference(num_neg)
                
                # Convert indices to node IDs for evaluation
                node_list = list(graph.nodes())
                reliable_neg_nodes = [node_list[i] for i in reliable_negatives if i < len(node_list)]
                
                f1 = evaluate_f1_score(graph, reliable_neg_nodes, target_negatives=num_neg)
                run_results['MCLS_f1'] = f1
                run_results['MCLS_negatives'] = len(reliable_neg_nodes)
                print(f"    MCLS F1: {f1:.4f} (found {len(reliable_neg_nodes)} negatives)")
                
            except Exception as e:
                print(f"    MCLS failed: {e}")
                run_results['MCLS_f1'] = 0
                run_results['MCLS_negatives'] = 0
            
            # Test LP_PUL
            print("  Testing LP_PUL...")
            try:
                lp_pul_data = manager.get_data_for_lp_pul(graph)
                lp_pul = LP_PUL(lp_pul_data)
                lp_pul.train()
                reliable_negatives = lp_pul.negative_inference(num_neg)
                
                # Convert to list if tensor
                if hasattr(reliable_negatives, 'tolist'):
                    reliable_negatives = reliable_negatives.tolist()
                
                # Convert indices to node IDs for evaluation
                node_list = list(graph.nodes())
                reliable_neg_nodes = [node_list[i] for i in reliable_negatives if i < len(node_list)]
                
                f1 = evaluate_f1_score(graph, reliable_neg_nodes, target_negatives=num_neg)
                run_results['LP_PUL_f1'] = f1
                run_results['LP_PUL_negatives'] = len(reliable_neg_nodes)
                print(f"    LP_PUL F1: {f1:.4f} (found {len(reliable_neg_nodes)} negatives)")
                
            except Exception as e:
                print(f"    LP_PUL failed: {e}")
                run_results['LP_PUL_f1'] = 0
                run_results['LP_PUL_negatives'] = 0
            
            all_results.append(run_results)
            
            # Clean up the temporary dataset file
            filepath = os.path.join(manager.datasets_dir, filename)
            if os.path.exists(filepath):
                os.remove(filepath)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Calculate summaries for each percent_positive value
    summaries = []
    
    for percent_pos in percent_positives:
        subset = results_df[results_df['percent_positive'] == percent_pos]
        if len(subset) > 0:
            summary = {
                'dataset': dataset_name,
                'n_samples': n_samples,
                'percent_positive': percent_pos,
                'num_neg': num_neg,
                'n_runs': len(subset),
                'mean_PULearningPC_f1': subset['PULearningPC_f1'].mean(),
                'std_PULearningPC_f1': subset['PULearningPC_f1'].std(),
                'mean_MCLS_f1': subset['MCLS_f1'].mean(),
                'std_MCLS_f1': subset['MCLS_f1'].std(),
                'mean_LP_PUL_f1': subset['LP_PUL_f1'].mean(),
                'std_LP_PUL_f1': subset['LP_PUL_f1'].std(),
                'mean_n_nodes': subset['n_nodes'].mean(),
                'mean_n_edges': subset['n_edges'].mean(),
                'PULearningPC_successful_runs': int((subset['PULearningPC_f1'] > 0).sum()),
                'MCLS_successful_runs': int((subset['MCLS_f1'] > 0).sum()),
                'LP_PUL_successful_runs': int((subset['LP_PUL_f1'] > 0).sum())
            }
            summaries.append(summary)
    
    # Print summary for each percent_positive value
    print(f"\n{'='*60}")
    print(f"EXPERIMENT SUMMARY - {dataset_name.upper()}")
    print(f"{'='*60}")
    print(f"Total runs: {len(results_df)} ({len(percent_positives)} percent_positive values × {n_runs} runs each)")
    print(f"Parameters: n_samples={n_samples}, percent_positives={[p*100 for p in percent_positives]}%")
    
    for summary in summaries:
        print(f"\n--- Results for {summary['percent_positive']*100}% positive samples ---")
        print(f"Runs: {summary['n_runs']}")
        print(f"Mean F1 Scores (successful runs in parentheses, 0=failure):")
        print(f"  PULearningPC: {summary['mean_PULearningPC_f1']:.4f} ± {summary['std_PULearningPC_f1']:.4f} ({summary['PULearningPC_successful_runs']}/{summary['n_runs']})")
        print(f"  MCLS:         {summary['mean_MCLS_f1']:.4f} ± {summary['std_MCLS_f1']:.4f} ({summary['MCLS_successful_runs']}/{summary['n_runs']})")
        print(f"  LP_PUL:       {summary['mean_LP_PUL_f1']:.4f} ± {summary['std_LP_PUL_f1']:.4f} ({summary['LP_PUL_successful_runs']}/{summary['n_runs']})")
    
    # Create a summary DataFrame for easy analysis
    summary_df = pd.DataFrame(summaries)
    
    # Automatically save results if requested
    if save_results:
        detailed_filename, summary_filename = save_results_to_csv(results_df, summary_df, dataset_name, n_samples, percent_positives)
        print(f"\nResults saved to:")
        print(f"  Detailed: {detailed_filename}")
        print(f"  Summary:  {summary_filename}")
    else:
        print(f"\nResults not saved (save_results=False)")
    
    return results_df, summary_df

def save_results_to_csv(results_df, summary_df, dataset_name, n_samples, percent_positives):
    """Save detailed results and summary to CSV files."""
    
    # Create results directory
    results_dir = "experiment_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Format percent positive as string for cleaner filenames
    if isinstance(percent_positives, (int, float)):
        percent_str = f"{int(percent_positives*100)}pct"
    else:
        percent_str = f"{len(percent_positives)}pct_values"
    
    # Save detailed results
    detailed_filename = f"{results_dir}/{dataset_name}_{n_samples or 'full'}_samples_{percent_str}_detailed_results.csv"
    results_df.to_csv(detailed_filename, index=False)
    print(f"Detailed results saved to: {detailed_filename}")
    
    # Save summary results (one row per percent_positive value)
    summary_filename = f"{results_dir}/{dataset_name}_{n_samples or 'full'}_samples_{percent_str}_summary_results.csv"
    summary_df.to_csv(summary_filename, index=False)
    print(f"Summary results saved to: {summary_filename}")
    
    return detailed_filename, summary_filename

if __name__ == "__main__":
    # Run multiple experiments with different parameters
    datasets = ['cora', 'citeseer', 'twitch']
    percent_positives = [0.05, 0.1, 0.2]
    n_samples_options = [1000, None]  # 1000 samples and full dataset
    
    all_summary_results = []
    
    for dataset in datasets:
        print(f"\n{'='*80}")
        print(f"RUNNING EXPERIMENTS FOR {dataset.upper()} DATASET")
        print(f"{'='*80}")
        
        for percent_pos in percent_positives:
            for n_samples in n_samples_options:
                print(f"\n{'='*60}")
                print(f"Testing {dataset} with {n_samples or 'all'} samples, {percent_pos*100}% positive")
                print(f"{'='*60}")
                
                # Run 10 experiments
                results_df, mean_results = run_multiple_experiments(
                    dataset_name=dataset,
                    n_samples=n_samples,
                    percent_positive=percent_pos,
                    n_runs=10
                )
                
                # Save results to CSV
                detailed_file, summary_file = save_results_to_csv(
                    results_df, mean_results, dataset, n_samples, percent_pos
                )
                
                # Store summary for overall results
                all_summary_results.append(mean_results)
    
    # Save overall summary
    if all_summary_results:
        overall_summary_df = pd.DataFrame(all_summary_results)
        overall_summary_file = "experiment_results/overall_summary_results.csv"
        overall_summary_df.to_csv(overall_summary_file, index=False)
        print(f"\n{'='*80}")
        print(f"OVERALL SUMMARY SAVED TO: {overall_summary_file}")
        print(f"{'='*80}")
        print(overall_summary_df[['dataset', 'n_samples', 'percent_positive', 
                                 'mean_PULearningPC_f1', 'mean_MCLS_f1', 'mean_LP_PUL_f1']].to_string(index=False))