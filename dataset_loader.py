#!/usr/bin/env python3
"""
Dataset Loader for PULearning Experiments

This script provides functions to load pre-generated datasets for different model types.
It ensures that all models use the same dataset with consistent labeling.

Usage:
    from dataset_loader import load_dataset_for_model
    data = load_dataset_for_model('cora_1000_samples.json', model_type='graph')
"""

import json
import os
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
from typing import Dict, List, Any, Optional, Union, Tuple
from sklearn.preprocessing import StandardScaler

class DatasetLoader:
    """
    Loads pre-generated datasets for different model types.
    """
    
    def __init__(self, datasets_dir: str = "datasets"):
        """
        Initialize the dataset loader.
        
        Args:
            datasets_dir: Directory containing the generated dataset files
        """
        self.datasets_dir = datasets_dir
    
    def load_dataset_info(self, filename: str) -> Dict[str, Any]:
        """
        Load dataset information from a JSON file.
        
        Args:
            filename: Name of the dataset file (with or without .json extension)
            
        Returns:
            Dictionary containing the dataset information
        """
        if not filename.endswith('.json'):
            filename += '.json'
        
        filepath = os.path.join(self.datasets_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            dataset_info = json.load(f)
        
        return dataset_info
    
    def load_dataset_for_graph_models(self, filename: str) -> Data:
        """
        Load dataset for graph-based models (PULearningPC, LP_PUL).
        
        Args:
            filename: Name of the dataset file
            
        Returns:
            PyTorch Geometric Data object
        """
        # Load the NetworkX graph first
        graph = self.load_dataset_as_networkx(filename)
        
        # Convert to PyTorch Geometric format
        node_list = sorted(list(graph.nodes()))
        features = np.array([graph.nodes[node]['features'] for node in node_list])
        true_labels = np.array([graph.nodes[node]['true_label'] for node in node_list])
        observed_labels = np.array([graph.nodes[node]['observed_label'] for node in node_list])
        
        # Get positive and unlabeled indices
        positive_indices = np.where(observed_labels == 1)[0]
        unlabeled_indices = np.where(observed_labels == 0)[0]
        
        # Create edge index
        edge_list = []
        for edge in graph.edges():
            node1_idx = node_list.index(edge[0])
            node2_idx = node_list.index(edge[1])
            edge_list.append([node1_idx, node2_idx])
        
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        
        # Create PyTorch Geometric data object
        data = Data(
            x=torch.tensor(features, dtype=torch.float),
            edge_index=edge_index,
            y=torch.tensor(true_labels, dtype=torch.long),
            P=torch.tensor(positive_indices, dtype=torch.long),
            U=torch.tensor(unlabeled_indices, dtype=torch.long),
            observed_labels=torch.tensor(observed_labels, dtype=torch.long)
        )
        
        return data
    
    def load_dataset_for_feature_models(self, filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int], List[int]]:
        """
        Load dataset for feature-based models (MCLS).
        
        Args:
            filename: Name of the dataset file
            
        Returns:
            Tuple of (features, true_labels, observed_labels, positive_indices, unlabeled_indices)
        """
        # Load the NetworkX graph first
        graph = self.load_dataset_as_networkx(filename)
        
        # Convert to numpy arrays
        node_list = sorted(list(graph.nodes()))
        features = np.array([graph.nodes[node]['features'] for node in node_list])
        true_labels = np.array([graph.nodes[node]['true_label'] for node in node_list])
        observed_labels = np.array([graph.nodes[node]['observed_label'] for node in node_list])
        
        # Get positive and unlabeled indices
        positive_indices = np.where(observed_labels == 1)[0].tolist()
        unlabeled_indices = np.where(observed_labels == 0)[0].tolist()
        
        return features, true_labels, observed_labels, positive_indices, unlabeled_indices
    
    def load_dataset_as_networkx(self, filename: str) -> nx.Graph:
        """
        Load dataset as a NetworkX graph.
        
        Args:
            filename: Name of the dataset file
            
        Returns:
            NetworkX graph with node attributes
        """
        dataset_info = self.load_dataset_info(filename)
        
        # Reconstruct the NetworkX graph from saved data
        G = nx.Graph()
        
        # Add nodes with attributes
        graph_data = dataset_info['graph_data']
        for node in graph_data['nodes']:
            attrs = graph_data['node_attributes'][node]
            G.add_node(node, 
                      features=np.array(attrs['features']),
                      true_label=attrs['true_label'],
                      observed_label=attrs['observed_label'])
        
        # Add edges
        for edge in graph_data['edges']:
            G.add_edge(edge[0], edge[1])
        
        return G
    
    def get_dataset_summary(self, filename: str) -> Dict[str, Any]:
        """
        Get a summary of the dataset without loading the full data.
        
        Args:
            filename: Name of the dataset file
            
        Returns:
            Dictionary with dataset summary information
        """
        dataset_info = self.load_dataset_info(filename)
        
        summary = {
            'dataset_name': dataset_info['dataset_name'],
            'n_nodes': dataset_info['n_nodes'],
            'n_edges': dataset_info['n_edges'],
            'n_features': dataset_info['n_features'],
            'n_positives': dataset_info['n_positives'],
            'n_unlabeled': dataset_info['n_unlabeled'],
            'positive_ratio': dataset_info['positive_ratio'],
            'is_connected': dataset_info['is_connected'],
            'graph_density': dataset_info['graph_density'],
            'random_seed': dataset_info['random_seed'],
            'dataset_params': dataset_info['dataset_params']
        }
        
        return summary
    
    def list_available_datasets(self) -> List[str]:
        """
        List all available dataset files in the datasets directory.
        
        Returns:
            List of dataset filenames
        """
        if not os.path.exists(self.datasets_dir):
            return []
        
        files = [f for f in os.listdir(self.datasets_dir) if f.endswith('.json')]
        return sorted(files)

# Convenience functions for easy usage
def load_dataset_for_model(filename: str, 
                          model_type: str = 'graph',
                          datasets_dir: str = "datasets") -> Union[Data, Tuple, nx.Graph]:
    """
    Load dataset for a specific model type.
    
    Args:
        filename: Name of the dataset file
        model_type: Type of model ('graph', 'feature', 'networkx')
        datasets_dir: Directory containing dataset files
        
    Returns:
        Dataset in the appropriate format for the model type
    """
    loader = DatasetLoader(datasets_dir)
    
    if model_type == 'graph':
        return loader.load_dataset_for_graph_models(filename)
    elif model_type == 'feature':
        return loader.load_dataset_for_feature_models(filename)
    elif model_type == 'networkx':
        return loader.load_dataset_as_networkx(filename)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose from 'graph', 'feature', 'networkx'")

def get_dataset_info(filename: str, datasets_dir: str = "datasets") -> Dict[str, Any]:
    """
    Get dataset information without loading the full data.
    
    Args:
        filename: Name of the dataset file
        datasets_dir: Directory containing dataset files
        
    Returns:
        Dictionary with dataset information
    """
    loader = DatasetLoader(datasets_dir)
    return loader.get_dataset_summary(filename)

def list_datasets(datasets_dir: str = "datasets") -> List[str]:
    """
    List all available datasets.
    
    Args:
        datasets_dir: Directory containing dataset files
        
    Returns:
        List of dataset filenames
    """
    loader = DatasetLoader(datasets_dir)
    return loader.list_available_datasets()

# Example usage and testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test dataset loading functionality")
    parser.add_argument('--filename', required=True, help='Dataset filename to load')
    parser.add_argument('--model-type', choices=['graph', 'feature', 'networkx'], 
                       default='graph', help='Model type to load for')
    parser.add_argument('--datasets-dir', default='datasets', help='Datasets directory')
    
    args = parser.parse_args()
    
    try:
        # Load dataset
        data = load_dataset_for_model(args.filename, args.model_type, args.datasets_dir)
        
        print(f"Successfully loaded dataset: {args.filename}")
        print(f"Model type: {args.model_type}")
        
        if args.model_type == 'graph':
            print(f"Graph data - Nodes: {data.x.shape[0]}, Features: {data.x.shape[1]}, Edges: {data.edge_index.shape[1]}")
            print(f"Positives: {len(data.P)}, Unlabeled: {len(data.U)}")
        elif args.model_type == 'feature':
            features, true_labels, observed_labels, pos_indices, unlab_indices = data
            print(f"Feature data - Samples: {features.shape[0]}, Features: {features.shape[1]}")
            print(f"Positives: {len(pos_indices)}, Unlabeled: {len(unlab_indices)}")
        elif args.model_type == 'networkx':
            print(f"NetworkX graph - Nodes: {data.number_of_nodes()}, Edges: {data.number_of_edges()}")
        
        # Get dataset info
        info = get_dataset_info(args.filename, args.datasets_dir)
        print(f"\nDataset info:")
        for key, value in info.items():
            print(f"  {key}: {value}")
            
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
