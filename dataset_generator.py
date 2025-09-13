#!/usr/bin/env python3
"""
Dataset Generator for Consistent PULearning Experiments

This script generates datasets with consistent labeling and saves them to JSON files
to ensure reproducibility across different model experiments. The generated datasets
can be loaded by different model types (graph-based and feature-based models).

Usage:
    python dataset_generator.py --dataset cora --output-dir datasets/
    python dataset_generator.py --dataset all --output-dir datasets/ --n-samples 1000
"""

import argparse
import json
import os
import numpy as np
import networkx as nx
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import random
from sklearn.preprocessing import StandardScaler
import torch
from torch_geometric.data import Data

from generate_dataset import (
    load_cora_scar, load_citeseer_scar, load_twitch_scar, 
    load_mnist_scar, load_ionosphere_scar
)

class DatasetGenerator:
    """
    Generates and saves datasets with consistent labeling for reproducible experiments.
    """
    
    def __init__(self, output_dir: str = "datasets", random_seed: int = 42):
        """
        Initialize the dataset generator.
        
        Args:
            output_dir: Directory to save generated datasets
            random_seed: Random seed for reproducibility
        """
        self.output_dir = output_dir
        self.random_seed = random_seed
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Set random seeds for reproducibility
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        
    def generate_dataset(self, 
                        dataset_name: str, 
                        dataset_params: Dict[str, Any],
                        n_samples: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate a dataset with consistent labeling.
        
        Args:
            dataset_name: Name of the dataset to generate
            dataset_params: Parameters for dataset generation
            n_samples: Number of samples to use (if None, use all available)
            
        Returns:
            Dictionary containing the generated dataset data
        """
        print(f"Generating {dataset_name} dataset...")
        
        # Load the dataset using the existing functions
        if dataset_name == 'cora':
            graph = load_cora_scar(**dataset_params)
        elif dataset_name == 'citeseer':
            graph = load_citeseer_scar(**dataset_params)
        elif dataset_name == 'twitch':
            graph = load_twitch_scar(**dataset_params)
        elif dataset_name == 'mnist':
            graph = load_mnist_scar(**dataset_params)
        elif dataset_name == 'ionosphere':
            graph = load_ionosphere_scar(**dataset_params)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        if graph is None:
            raise ValueError(f"Failed to load {dataset_name} dataset")
        
        # Apply sampling if requested
        if n_samples and n_samples < graph.number_of_nodes():
            print(f"Sampling {n_samples} nodes from {graph.number_of_nodes()} total nodes...")
            nodes = list(graph.nodes())
            sampled_nodes = random.sample(nodes, n_samples)
            graph = graph.subgraph(sampled_nodes).copy()
        
        # Extract data from the graph
        node_list = sorted(list(graph.nodes()))
        features = np.array([graph.nodes[node]['features'] for node in node_list])
        true_labels = np.array([graph.nodes[node]['true_label'] for node in node_list])
        observed_labels = np.array([graph.nodes[node]['observed_label'] for node in node_list])
        
        # Get positive and unlabeled indices
        positive_indices = np.where(observed_labels == 1)[0].tolist()
        unlabeled_indices = np.where(observed_labels == 0)[0].tolist()
        
        # Save the NetworkX graph directly - much simpler!
        # We'll convert to other formats when loading
        
        # Prepare dataset information - much simpler!
        dataset_info = {
            'dataset_name': dataset_name,
            'dataset_params': dataset_params,
            'n_samples': n_samples,
            'n_nodes': graph.number_of_nodes(),
            'n_edges': graph.number_of_edges(),
            'n_features': features.shape[1],
            'n_positives': len(positive_indices),
            'n_unlabeled': len(unlabeled_indices),
            'positive_ratio': len(positive_indices) / len(node_list),
            'random_seed': self.random_seed,
            'is_connected': nx.is_connected(graph),
            'graph_density': nx.density(graph),
            # Save the graph data for easy reconstruction
            'graph_data': {
                'nodes': list(graph.nodes()),
                'edges': list(graph.edges()),
                'node_attributes': {
                    node: {
                        'features': graph.nodes[node]['features'].tolist(),
                        'true_label': int(graph.nodes[node]['true_label']),
                        'observed_label': int(graph.nodes[node]['observed_label'])
                    }
                    for node in graph.nodes()
                }
            }
        }
        
        print(f"Generated {dataset_name} dataset:")
        print(f"  Nodes: {dataset_info['n_nodes']}")
        print(f"  Edges: {dataset_info['n_edges']}")
        print(f"  Features: {dataset_info['n_features']}")
        print(f"  Positives: {dataset_info['n_positives']}")
        print(f"  Unlabeled: {dataset_info['n_unlabeled']}")
        print(f"  Connected: {dataset_info['is_connected']}")
        print(f"  Density: {dataset_info['graph_density']:.4f}")
        
        return dataset_info
    
    def save_dataset(self, dataset_info: Dict[str, Any], filename: Optional[str] = None) -> str:
        """
        Save dataset information to a JSON file.
        
        Args:
            dataset_info: Dataset information dictionary
            filename: Optional custom filename
            
        Returns:
            Path to the saved file
        """
        if filename is None:
            dataset_name = dataset_info['dataset_name']
            n_samples = dataset_info['n_samples']
            if n_samples:
                filename = f"{dataset_name}_{n_samples}_samples.json"
            else:
                filename = f"{dataset_name}_full.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        print(f"Dataset saved to: {filepath}")
        return filepath
    
    def generate_all_datasets(self, 
                             n_samples: Optional[int] = None,
                             percent_positive: float = 0.1) -> Dict[str, str]:
        """
        Generate all available datasets with consistent parameters.
        
        Args:
            n_samples: Number of samples to use for each dataset
            percent_positive: Percentage of positive examples to label
            
        Returns:
            Dictionary mapping dataset names to file paths
        """
        # Default parameters for each dataset
        dataset_configs = {
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
                'n_samples': 3000  # MNIST is large, so we limit it
            }
        }
        
        saved_files = {}
        
        for dataset_name, params in dataset_configs.items():
            try:
                # Override n_samples if provided
                if n_samples and 'n_samples' not in params:
                    params['n_samples'] = n_samples
                
                # Generate dataset
                dataset_info = self.generate_dataset(dataset_name, params, n_samples)
                
                # Save dataset
                filepath = self.save_dataset(dataset_info)
                saved_files[dataset_name] = filepath
                
            except Exception as e:
                print(f"Error generating {dataset_name} dataset: {e}")
                continue
        
        return saved_files

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Generate datasets with consistent labeling for PULearning experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Generate Cora dataset
            python dataset_generator.py --dataset cora
            
            # Generate all datasets with 1000 samples each
            python dataset_generator.py --dataset all --n-samples 1000
            
            # Generate with custom parameters
            python dataset_generator.py --dataset cora --percent-positive 0.2 --output-dir my_datasets/
                    """
    )
    
    parser.add_argument(
        '--dataset',
        choices=['cora', 'citeseer', 'twitch', 'mnist', 'ionosphere', 'all'],
        default='all',
        help='Dataset(s) to generate (default: all)'
    )
    
    parser.add_argument(
        '--output-dir',
        default='datasets',
        help='Output directory for generated datasets (default: datasets)'
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
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = DatasetGenerator(
        output_dir=args.output_dir,
        random_seed=args.random_seed
    )
    
    # Generate datasets
    if args.dataset == 'all':
        print("Generating all datasets...")
        saved_files = generator.generate_all_datasets(
            n_samples=args.n_samples,
            percent_positive=args.percent_positive
        )
        
        print(f"\nGenerated {len(saved_files)} datasets:")
        for dataset_name, filepath in saved_files.items():
            print(f"  {dataset_name}: {filepath}")
    
    else:
        # Generate single dataset
        dataset_configs = {
            'cora': {
                'k': 3,
                'positive_class_label': 3,
                'percent_positive': args.percent_positive,
                'use_original_edges': True,
                'mst': False
            },
            'citeseer': {
                'k': 3,
                'positive_class_label': 2,
                'percent_positive': args.percent_positive,
                'use_original_edges': True,
                'mst': False
            },
            'twitch': {
                'percent_positive': args.percent_positive,
                'mst': False
            },
            'mnist': {
                'k': 3,
                'percent_positive': args.percent_positive,
                'mst': False,
                'n_samples': 3000
            },
            'ionosphere': {
                'k': 3,
                'percent_positive': args.percent_positive,
                'mst': False
            }
        }
        
        params = dataset_configs[args.dataset]
        if args.n_samples and 'n_samples' not in params:
            params['n_samples'] = args.n_samples
        
        dataset_info = generator.generate_dataset(args.dataset, params, args.n_samples)
        filepath = generator.save_dataset(dataset_info)
        print(f"\nGenerated {args.dataset} dataset: {filepath}")

if __name__ == "__main__":
    main()
