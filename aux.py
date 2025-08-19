
import networkx as nx
import random
import numpy as np
from typing import Tuple

def prepare_pu_graph_data(
    graph: nx.Graph, 
    feature_class: str,
    percentage_labeled: float, 
    positive_class: str,
    random_seed: int = None
) -> Tuple[nx.Graph, np.ndarray]:
    """
    Prepare graph data for PU learning under SCAR assumption.
    
    Args:
        G: Input graph with node attributes containing class information
        percentage_labeled: Fraction of positive nodes to be labeled as positive (0-1)
        positive_class: String indicating which class is considered positive
        random_seed: Seed for reproducibility
        
    Returns:
        Tuple of (modified graph with PU labels, array of true labels)
    """
    G = graph.copy()
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    # Get true labels (1 for positive class, 0 otherwise)
    true_labels = np.array([
        1 if G.nodes[n][feature_class] == positive_class else 0 
        for n in G.nodes()
    ])
    
    # Get indices of positive nodes
    positive_indices = np.where(true_labels == 1)[0]
    num_positives = len(positive_indices)
    num_labeled_positives = int(percentage_labeled * num_positives)
    
    # Randomly select which positives will be labeled
    labeled_positives = set(
        random.sample(list(positive_indices), num_labeled_positives))
    
    # Create observed labels (1 for labeled positives, 0 otherwise)
    observed_labels = np.array([
        1 if i in labeled_positives else 0 
        for i in range(len(G.nodes()))
    ])
    
    # Add labels to graph nodes
    for i, n in enumerate(G.nodes()):
        G.nodes[n]['true_label'] = true_labels[i]
        G.nodes[n]['observed_label'] = observed_labels[i]
    
    return G

def get_percentages_labels_graph(graph: nx.Graph, feature_class: str, positive_class: str) -> Tuple[float, float]:   
    """
    Calculate the percentage of positive and negative nodes in the graph.
    
    Args:
        graph: Input graph with node attributes containing class information
        feature_class: String indicating which attribute contains class information
        positive_class: String indicating which class is considered positive
    Returns:
        Tuple of (percentage of positive nodes, percentage of negative nodes)
    """
    total_nodes = len(graph.nodes())
    if total_nodes == 0:
        return 0.0, 0.0
    
    positive_count = sum(
        1 for n in graph.nodes() if graph.nodes[n][feature_class] == positive_class)
    
    negative_count = total_nodes - positive_count
    
    percentage_positive = positive_count / total_nodes
    percentage_negative = negative_count / total_nodes
    
    return percentage_positive, percentage_negative