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
from aux import dict_datasets_params_pulpc

def evaluate_f1_score(graph, reliable_negatives, true_label_key='true_label'):
    """
    Evaluate F1 score for reliable negatives, handling gaps in node indices and invalid references.
    
    Args:
        graph: NetworkX graph with node attributes
        reliable_negatives: List of node indices that are predicted as negative
        true_label_key: Key for true labels in node attributes
    
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
        f1 = evaluate_f1_score(graph, reliable_negatives)
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
        f1 = evaluate_f1_score(graph, reliable_neg_nodes)
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
        f1 = evaluate_f1_score(graph, reliable_neg_nodes)
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

if __name__ == "__main__":
    # Test different percent positive values
    percent_positives = [0.05, 0.1, 0.2]
    
    for percent_pos in percent_positives:
        print(f"\n{'='*60}")
        print(f"Testing with {percent_pos*100}% positive samples")
        print(f"{'='*60}")
        
        # Run experiments with specific number of samples
        print(f"\n--- Running with 1000 samples ---")
        results = run_models("cora", 1000, percent_pos)
        
        # Run experiments with all available samples
        print(f"\n--- Running with all available samples ---")
        results = run_models("cora", None, percent_pos)