#!/usr/bin/env python3
"""
Example script demonstrating consistent dataset usage across all models.

This script shows how to:
1. Generate datasets with consistent labeling
2. Load the same dataset for different model types
3. Run experiments ensuring all models use identical data

Usage:
    python example_consistent_experiments.py --dataset cora --n-samples 1000
"""

import argparse
import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any
import random

from dataset_generator import DatasetGenerator
from dataset_loader import DatasetLoader, load_dataset_for_model
from parametric_experiments import PULearningExperimentRunner
from models import MCLS, LP_PUL
from model import PULearningPC

def run_consistent_experiments(dataset_name: str, 
                              n_samples: int = None,
                              percent_positive: float = 0.1,
                              datasets_dir: str = "datasets",
                              results_dir: str = "consistent_results",
                              random_seed: int = 42):
    """
    Run experiments with consistent datasets across all models.
    
    Args:
        dataset_name: Name of the dataset to use
        n_samples: Number of samples to use
        percent_positive: Percentage of positive examples to label
        datasets_dir: Directory for generated datasets
        results_dir: Directory for results
        random_seed: Random seed for reproducibility
    """
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    print("="*80)
    print("CONSISTENT DATASET EXPERIMENTS")
    print("="*80)
    print(f"Dataset: {dataset_name}")
    print(f"Samples: {n_samples if n_samples else 'all'}")
    print(f"Percent positive: {percent_positive}")
    print(f"Random seed: {random_seed}")
    print("="*80)
    
    # Step 1: Generate dataset if it doesn't exist
    dataset_filename = f"{dataset_name}_{n_samples}_samples.json" if n_samples else f"{dataset_name}_full.json"
    dataset_path = os.path.join(datasets_dir, dataset_filename)
    
    if not os.path.exists(dataset_path):
        print(f"\nGenerating dataset: {dataset_filename}")
        generator = DatasetGenerator(datasets_dir, random_seed)
        
        # Dataset-specific parameters
        dataset_params = {
            'cora': {
                'k': 3,
                'positive_class_label': 3,
                'percent_positive': percent_positive,
                'use_original_edges': True,
                'mst': False
            },
            'citeseer': {
                'k': 3,
                'positive_class_label': 2,
                'percent_positive': percent_positive,
                'use_original_edges': True,
                'mst': False
            },
            'twitch': {
                'percent_positive': percent_positive,
                'mst': False
            },
            'mnist': {
                'k': 3,
                'percent_positive': percent_positive,
                'mst': False,
                'n_samples': 3000
            }
        }
        
        params = dataset_params.get(dataset_name, {})
        dataset_info = generator.generate_dataset(dataset_name, params, n_samples)
        generator.save_dataset(dataset_info, dataset_filename)
    else:
        print(f"\nUsing existing dataset: {dataset_filename}")
    
    # Step 2: Load dataset for different model types
    print(f"\nLoading dataset for different model types...")
    
    # Load for graph-based models (PULearningPC, LP_PUL)
    graph_data = load_dataset_for_model(dataset_filename, 'graph', datasets_dir)
    networkx_graph = load_dataset_for_model(dataset_filename, 'networkx', datasets_dir)
    
    # Load for feature-based models (MCLS)
    features, true_labels, observed_labels, pos_indices, unlab_indices = load_dataset_for_model(
        dataset_filename, 'feature', datasets_dir
    )
    
    print(f"Dataset loaded successfully:")
    print(f"  Nodes: {graph_data.x.shape[0]}")
    print(f"  Features: {graph_data.x.shape[1]}")
    print(f"  Edges: {graph_data.edge_index.shape[1]}")
    print(f"  Positives: {len(graph_data.P)}")
    print(f"  Unlabeled: {len(graph_data.U)}")
    
    # Step 3: Run experiments with PULearningPC (graph-based)
    print(f"\n{'='*50}")
    print("RUNNING PULearningPC EXPERIMENTS")
    print(f"{'='*50}")
    
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize experiment runner with saved datasets
    runner = PULearningExperimentRunner(
        n_runs=3,
        output_dir=os.path.join(results_dir, "pulearningpc"),
        random_seed=random_seed,
        use_saved_datasets=True,
        datasets_dir=datasets_dir
    )
    
    # Set reduced parameter ranges for demonstration
    runner.set_custom_parameter_ranges({
        'num_particles': [100, 200],
        'p_det': [0.6],
        'delta_v': [0.4],
        'delta_p': [0.3],
        'cluster_strategy': ['percentage'],
        'positive_cluster_threshold': [0.1, 0.2],
        'movement_strategy': ['uniform', 'degree_weighted'],
        'initialization_strategy': ['random', 'degree_weighted'],
        'avg_node_pot_threshold': [0.7, 0.8]
    })
    
    # Run PULearningPC experiments
    pulearning_results = runner.run_experiments(
        dataset_name=dataset_name,
        dataset_filename=dataset_filename,
        n_jobs=2,  # Use fewer jobs for demonstration
        num_neg=100
    )
    
    print(f"PULearningPC experiments completed: {len(pulearning_results)} runs")
    
    # Step 4: Test MCLS model (feature-based)
    print(f"\n{'='*50}")
    print("TESTING MCLS MODEL")
    print(f"{'='*50}")
    
    # Create a simple data object for MCLS
    class SimpleData:
        def __init__(self, x, P, U):
            self.x = torch.tensor(x, dtype=torch.float)
            self.P = P
            self.U = U
    
    mcls_data = SimpleData(features, pos_indices, unlab_indices)
    
    # Test MCLS with different parameters
    mcls_results = []
    for k in [5, 7, 10]:
        for ratio in [0.3, 0.5]:
            try:
                print(f"Testing MCLS: k={k}, ratio={ratio}")
                mcls = MCLS(mcls_data, k=k, ratio=ratio)
                mcls.train()
                
                # Get reliable negatives
                num_neg = min(100, len(unlab_indices) // 2)
                reliable_negatives = mcls.negative_inference(num_neg)
                
                # Evaluate
                y_true = [true_labels[i] for i in reliable_negatives]
                y_pred = [0] * len(reliable_negatives)
                
                from sklearn.metrics import f1_score, precision_score, recall_score
                f1 = f1_score(y_true, y_pred, pos_label=0)
                precision = precision_score(y_true, y_pred, pos_label=0)
                recall = recall_score(y_true, y_pred, pos_label=0)
                
                mcls_results.append({
                    'model': 'MCLS',
                    'k': k,
                    'ratio': ratio,
                    'f1_score': f1,
                    'precision': precision,
                    'recall': recall,
                    'num_negatives': len(reliable_negatives)
                })
                
                print(f"  F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
                
            except Exception as e:
                print(f"  Error: {e}")
                continue
    
    # Step 5: Test LP_PUL model (graph-based)
    print(f"\n{'='*50}")
    print("TESTING LP_PUL MODEL")
    print(f"{'='*50}")
    
    try:
        print("Testing LP_PUL...")
        lp_pul = LP_PUL(graph_data)
        lp_pul.train()
        
        # Get reliable negatives
        num_neg = min(100, len(graph_data.U))
        reliable_negatives = lp_pul.negative_inference(num_neg)
        
        # Evaluate
        y_true = [true_labels[i] for i in reliable_negatives]
        y_pred = [0] * len(reliable_negatives)
        
        from sklearn.metrics import f1_score, precision_score, recall_score
        f1 = f1_score(y_true, y_pred, pos_label=0)
        precision = precision_score(y_true, y_pred, pos_label=0)
        recall = recall_score(y_true, y_pred, pos_label=0)
        
        lp_pul_results = [{
            'model': 'LP_PUL',
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'num_negatives': len(reliable_negatives)
        }]
        
        print(f"  F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        
    except Exception as e:
        print(f"  Error: {e}")
        lp_pul_results = []
    
    # Step 6: Save and summarize results
    print(f"\n{'='*50}")
    print("RESULTS SUMMARY")
    print(f"{'='*50}")
    
    # Save MCLS results
    if mcls_results:
        mcls_df = pd.DataFrame(mcls_results)
        mcls_df.to_csv(os.path.join(results_dir, f"{dataset_name}_mcls_results.csv"), index=False)
        print(f"MCLS results saved: {len(mcls_results)} configurations tested")
        print(f"Best MCLS F1: {mcls_df['f1_score'].max():.4f}")
    
    # Save LP_PUL results
    if lp_pul_results:
        lp_pul_df = pd.DataFrame(lp_pul_results)
        lp_pul_df.to_csv(os.path.join(results_dir, f"{dataset_name}_lp_pul_results.csv"), index=False)
        print(f"LP_PUL results saved: {len(lp_pul_results)} configurations tested")
        print(f"LP_PUL F1: {lp_pul_df['f1_score'].iloc[0]:.4f}")
    
    # PULearningPC results summary
    successful_pulearning = pulearning_results[pulearning_results['status'] == 'success']
    if len(successful_pulearning) > 0:
        print(f"PULearningPC results: {len(successful_pulearning)} successful runs")
        print(f"Best PULearningPC F1: {successful_pulearning['f1_score'].max():.4f}")
    
    print(f"\nAll results saved in: {results_dir}/")
    print("Dataset used consistently across all models:")
    print(f"  File: {dataset_filename}")
    print(f"  Nodes: {graph_data.x.shape[0]}")
    print(f"  Positives: {len(graph_data.P)}")
    print(f"  Unlabeled: {len(graph_data.U)}")

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Run consistent experiments across all models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run experiments on Cora with 1000 samples
  python example_consistent_experiments.py --dataset cora --n-samples 1000
  
  # Run experiments on all available samples
  python example_consistent_experiments.py --dataset citeseer
  
  # Run with custom parameters
  python example_consistent_experiments.py --dataset twitch --percent-positive 0.2 --random-seed 123
        """
    )
    
    parser.add_argument(
        '--dataset',
        choices=['cora', 'citeseer', 'twitch', 'mnist', 'ionosphere'],
        required=True,
        help='Dataset to use for experiments'
    )
    
    parser.add_argument(
        '--n-samples',
        type=int,
        help='Number of samples to use (if not provided, use all available)'
    )
    
    parser.add_argument(
        '--percent-positive',
        type=float,
        default=0.1,
        help='Percentage of positive examples to label (default: 0.1)'
    )
    
    parser.add_argument(
        '--datasets-dir',
        default='datasets',
        help='Directory for generated datasets (default: datasets)'
    )
    
    parser.add_argument(
        '--results-dir',
        default='consistent_results',
        help='Directory for results (default: consistent_results)'
    )
    
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Run experiments
    run_consistent_experiments(
        dataset_name=args.dataset,
        n_samples=args.n_samples,
        percent_positive=args.percent_positive,
        datasets_dir=args.datasets_dir,
        results_dir=args.results_dir,
        random_seed=args.random_seed
    )

if __name__ == "__main__":
    main()
