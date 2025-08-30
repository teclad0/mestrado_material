#!/usr/bin/env python3
"""
Test script to verify that fallback rule tracking is now working correctly.
"""

import networkx as nx
import numpy as np
from model import PULearningPC

def test_fallback_tracking():
    """Test that fallback rule usage is properly tracked and propagated."""
    
    # Create a simple test graph
    G = nx.Graph()
    
    # Add nodes with true labels (0 = negative, 1 = positive)
    # Create a scenario where no clusters will be positive by default
    for i in range(20):
        true_label = 1 if i < 2 else 0  # Only 2 positive nodes out of 20
        G.add_node(i, true_label=true_label, observed_label=1 if i < 2 else 0)
    
    # Add some edges to make it connected
    for i in range(19):
        G.add_edge(i, i+1)
    
    print("Test Graph:")
    print(f"Nodes: {G.number_of_nodes()}")
    
    # Use safer node access since graph nodes may have gaps after processing
    node_attributes = dict(G.nodes(data=True))
    print(f"Positive nodes (true_label=1): {sum(1 for n in G.nodes() if node_attributes[n]['true_label'] == 1)}")
    print(f"Observed positive nodes: {sum(1 for n in G.nodes() if node_attributes[n]['observed_label'] == 1)}")
    
    # Test with very strict threshold that should trigger fallback rule
    print("\nTesting with strict threshold (should trigger fallback rule):")
    
    pcm_params = {
        'num_particles': 3,
        'p_det': 0.6,
        'delta_v': 0.3,
        'delta_p': 0.4,
        'movement_strategy': 'uniform',
        'initialization_strategy': 'random',
        'average_node_potential_threshold': 0.6
    }
    
    rns_params = {
        'cluster_strategy': 'percentage',
        'positive_cluster_threshold': 0.95  # Extremely strict - 95% positive required
    }
    
    model = PULearningPC(
        graph=G,
        num_neg=5,
        pcm_params=pcm_params,
        rns_params=rns_params
    )
    
    # Train the model
    model.train()
    
    # Check if fallback rule was used in the labeled_graph
    fallback_used = model.labeled_graph.graph.get('fallback_rule_used', False)
    print(f"Fallback rule used (from labeled_graph): {fallback_used}")
    
    # Also check if it's in the original graph
    fallback_original = G.graph.get('fallback_rule_used', False)
    print(f"Fallback rule used (from original graph): {fallback_original}")
    
    # Check if the attribute exists at all
    print(f"Labeled graph attributes: {list(model.labeled_graph.graph.keys())}")
    print(f"Original graph attributes: {list(G.graph.keys())}")
    
    # Let's also check what the actual cluster composition was
    print(f"\nDebugging cluster composition:")
    print(f"Owner groups: {model.pcm.owner_groups}")
    print(f"Cluster sizes: {model.pcm.cluster_sizes}")
    print(f"Cluster positive counts: {model.pcm.cluster_positive_counts}")
    
    # Check if any clusters would meet the 95% threshold
    threshold = 0.95
    for owner, nodes in model.pcm.owner_groups.items():
        cluster_size = model.pcm.cluster_sizes.get(owner, 0)
        positive_count = model.pcm.cluster_positive_counts.get(owner, 0)
        if cluster_size >= 3:  # Only consider clusters with 3+ nodes
            ratio = positive_count / cluster_size
            print(f"Cluster {owner}: size={cluster_size}, positives={positive_count}, ratio={ratio:.3f}, meets_threshold={ratio >= threshold}")
    
    if fallback_used:
        print("✅ Test PASSED: Fallback rule tracking is working correctly!")
    else:
        print("❌ Test FAILED: Fallback rule tracking is still not working.")

if __name__ == "__main__":
    test_fallback_tracking() 