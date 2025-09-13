#!/usr/bin/env python3

import argparse
import os
import pickle
import numpy as np
import networkx as nx
import torch
from typing import Dict, List, Any, Optional, Tuple
import random
from torch_geometric.data import Data
from models import MCLS


from generate_dataset import (
    load_cora_scar, load_citeseer_scar, load_twitch_scar, 
    load_mnist_scar
)

class DatasetManager:    
    def __init__(self, datasets_dir: str = "datasets", random_seed: int = 42):
        self.datasets_dir = datasets_dir
        self.random_seed = random_seed
        os.makedirs(datasets_dir, exist_ok=True)
        
        # Set random seeds
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
    
    def generate_and_save_dataset(self, dataset_name: str, dataset_params: Dict, n_samples: Optional[int] = None, percent_positive: float = 0.1) -> str:
        """Generate and save a dataset as a NetworkX graph."""
        print(f"Generating {dataset_name} dataset...")
        
        # Update dataset_params with percent_positive
        dataset_params = dataset_params.copy()
        dataset_params['percent_positive'] = percent_positive
        
        # Load dataset using existing functions
        if dataset_name == 'cora':
            graph = load_cora_scar(**dataset_params)
        elif dataset_name == 'citeseer':
            graph = load_citeseer_scar(**dataset_params)
        elif dataset_name == 'twitch':
            graph = load_twitch_scar(**dataset_params)
        elif dataset_name == 'mnist':
            graph = load_mnist_scar(**dataset_params)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        if graph is None:
            raise ValueError(f"Failed to load {dataset_name} dataset")
        
      
        # Save the graph directly
        filename = f"{dataset_name}_{n_samples}_samples.pkl" if n_samples else f"{dataset_name}_full.pkl"
        filepath = os.path.join(self.datasets_dir, filename)
        
        with open(filepath, 'wb') as f:
            pickle.dump(graph, f)
        
        print(f"Dataset saved to: {filepath}")
        print(f"  Nodes: {graph.number_of_nodes()}")
        print(f"  Edges: {graph.number_of_edges()}")
        
        # Get features count safely
        first_node = list(graph.nodes())[0]
        features_count = len(graph.nodes[first_node]['features'])
        print(f"  Features: {features_count}")
        
        return filepath
    
    def load_graph(self, filename: str) -> nx.Graph:
        """Load a NetworkX graph from file."""
        filepath = os.path.join(self.datasets_dir, filename)
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def get_data_for_mcls(self, graph: nx.Graph):
        """Get data in the format MCLS expects."""
        node_list = sorted(list(graph.nodes()))
        features = np.array([graph.nodes[node]['features'] for node in node_list])
        true_labels = np.array([graph.nodes[node]['true_label'] for node in node_list])
        observed_labels = np.array([graph.nodes[node]['observed_label'] for node in node_list])
        
        # Get positive and unlabeled indices
        positive_indices = np.where(observed_labels == 1)[0].tolist()
        unlabeled_indices = np.where(observed_labels == 0)[0].tolist()
        
        # Create simple data object for MCLS
        class SimpleData:
            def __init__(self, x, P, U):
                self.x = torch.tensor(x, dtype=torch.float)
                self.P = P
                self.U = U
        
        return SimpleData(features, positive_indices, unlabeled_indices)
    
    def get_data_for_lp_pul(self, graph: nx.Graph):
        """Get data in the format LP_PUL expects (PyTorch Geometric Data)."""
        node_list = sorted(list(graph.nodes()))
        node_to_idx = {node: i for i, node in enumerate(node_list)}
        
        features = np.array([graph.nodes[node]['features'] for node in node_list])
        true_labels = np.array([graph.nodes[node]['true_label'] for node in node_list])
        observed_labels = np.array([graph.nodes[node]['observed_label'] for node in node_list])
        
        # Get positive and unlabeled indices
        positive_indices = np.where(observed_labels == 1)[0]
        unlabeled_indices = np.where(observed_labels == 0)[0]
        
        # Create edge index from graph edges
        edge_list = []
        for edge in graph.edges():
            u, v = edge
            if u in node_to_idx and v in node_to_idx:
                edge_list.append([node_to_idx[u], node_to_idx[v]])
                edge_list.append([node_to_idx[v], node_to_idx[u]])  # Add reverse edge for undirected graph
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous() if edge_list else torch.empty((2, 0), dtype=torch.long)
        
        return Data(
            x=torch.tensor(features, dtype=torch.float),
            y=torch.tensor(true_labels, dtype=torch.long),
            P=torch.tensor(positive_indices, dtype=torch.long),
            U=torch.tensor(unlabeled_indices, dtype=torch.long),
            observed_labels=torch.tensor(observed_labels, dtype=torch.long),
            edge_index=edge_index
        )

def run_experiment(dataset_name: str, n_samples: int = None, percent_positive: float = 0.1):
    """Run a simple experiment demonstrating the ultra-simple system."""
    
    # Dataset parameters
    dataset_params = {
        'cora': {
            'k': 3,
            'positive_class_label': 3,
            'percent_positive': 0.1,
            'use_original_edges': True,
            'mst': False
        },
        'citeseer': {
            'k': 3,
            'positive_class_label': 2,
            'percent_positive': 0.1,
            'use_original_edges': True,
            'mst': False
        },
        'twitch': {
            'percent_positive': 0.1,
            'mst': False
        },
        'mnist': {
            'k': 3,
            'percent_positive': 0.1,
            'mst': False,
            'n_samples': 3000
        }
    }
    
    manager = DatasetManager()
    
    # Generate dataset
    filename = f"{dataset_name}_{n_samples}_samples.pkl" if n_samples else f"{dataset_name}_full.pkl"
    filepath = os.path.join(manager.datasets_dir, filename)
    
    if not os.path.exists(filepath):
        params = dataset_params[dataset_name]
        manager.generate_and_save_dataset(dataset_name, params, n_samples, percent_positive)
    else:
        print(f"Using existing dataset: {filepath}")
    
    # Load the same dataset for different model types
    print(f"\nLoading dataset for different model types...")
    
    # Load as NetworkX graph (for PULearningPC)
    graph = manager.load_graph(filename)
    print(f"✓ NetworkX graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    # Convert for MCLS
    mcls_data = manager.get_data_for_mcls(graph)
    print(f"✓ MCLS data: {mcls_data.x.shape[0]} samples, {mcls_data.x.shape[1]} features")
    print(f"  Positives: {len(mcls_data.P)}, Unlabeled: {len(mcls_data.U)}")
    
    # Convert for LP_PUL
    lp_pul_data = manager.get_data_for_lp_pul(graph)
    print(f"✓ LP_PUL data: {lp_pul_data.x.shape[0]} nodes, {lp_pul_data.x.shape[1]} features")
    print(f"  Positives: {len(lp_pul_data.P)}, Unlabeled: {len(lp_pul_data.U)}")
    
    print(f"\n✅ All models can use the same consistent dataset!")
    print(f"Dataset file: {filepath}")
    
    # Test with actual models
    print(f"\nTesting with actual models...")
    
    try:
        # Test MCLS
        from models import MCLS
        mcls = MCLS(mcls_data, k=5, ratio=0.3)
        print("✓ MCLS initialized successfully")
        
        # Test LP_PUL
        from models import LP_PUL
        lp_pul = LP_PUL(lp_pul_data)
        print("✓ LP_PUL initialized successfully")
        
        # Test PULearningPC (takes NetworkX graph directly)
        from model import PULearningPC
        pulearning = PULearningPC(graph, num_neg=20)
        print("✓ PULearningPC initialized successfully")
        
    except Exception as e:
        print(f"✗ Model initialization failed: {e}")
    
    return filepath

def main():
    parser = argparse.ArgumentParser(description="Ultra simple dataset system")
    parser.add_argument('--dataset', choices=['cora', 'citeseer', 'twitch', 'mnist', 'ionosphere'], 
                       required=True, help='Dataset to generate')
    parser.add_argument('--n-samples', type=int, help='Number of samples to use')
    parser.add_argument('--datasets-dir', default='datasets', help='Directory for datasets')
    
    args = parser.parse_args()
    
    run_experiment(args.dataset, args.n_samples)

if __name__ == "__main__":
    main()
