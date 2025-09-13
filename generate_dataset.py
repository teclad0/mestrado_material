import networkx as nx
import numpy as np
import pandas as pd
import os
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.datasets import fetch_openml
from sklearn.model_selection import StratifiedShuffleSplit

# --- PyTorch Geometric Dependency ---
try:
    from torch_geometric.datasets import Planetoid
    from torch_geometric.utils import to_networkx
except ImportError:
    print("PyTorch Geometric not found. Please install it with 'pip install torch torch-geometric'")
    to_networkx = None

# --- Helper Functions ---

def _stratified_sample(features: np.ndarray, labels: np.ndarray, n_samples: int):
    """
    Performs stratified sampling on features and labels to maintain class distribution.
    """
    if n_samples >= len(labels):
        return np.arange(len(labels)) # Return all indices if n_samples is too large

    splitter = StratifiedShuffleSplit(n_splits=1, train_size=n_samples, random_state=42)
    try:
        train_idx, _ = next(splitter.split(features, labels))
        return train_idx
    except ValueError:
        print("Warning: Stratified sampling failed. Using random sampling instead.")
        return np.random.choice(len(labels), n_samples, replace=False)


def _handle_graph_connectivity(G: nx.Graph, mst: bool, features: np.ndarray = None) -> nx.Graph:
    """
    Ensures the graph is connected, either by extracting the largest component or by adding MST edges.
    """
    if nx.is_connected(G):
        return G

    if not mst:
        print("Graph is not connected. Extracting the largest connected component.")
        largest_cc = max(nx.connected_components(G), key=len)
        return G.subgraph(largest_cc).copy()
    else:
        print("Graph is not connected. Connecting components using Minimum Spanning Tree.")
        components = list(nx.connected_components(G))
        G_connected = G.copy()
        
        component_graph = nx.Graph()
        representatives = [min(comp) for comp in components]

        for i in range(len(components)):
            for j in range(i + 1, len(components)):
                rep_i, rep_j = representatives[i], representatives[j]
                distance = np.linalg.norm(features[rep_i] - features[rep_j]) if features is not None else 1.0
                component_graph.add_edge(i, j, weight=distance)
        
        mst_edges = nx.minimum_spanning_tree(component_graph)

        for u, v in mst_edges.edges():
            rep_u, rep_v = representatives[u], representatives[v]
            G_connected.add_edge(rep_u, rep_v)
            
        return G_connected


def apply_scar_labeling(labels: np.ndarray, positive_class_label: int, percent_positive: float) -> np.ndarray:
    """
    Applies the SCAR (Selected Completely at Random) labeling assumption.
    """
    positive_indices = np.where(labels == positive_class_label)[0]
    num_to_label = int(len(positive_indices) * percent_positive)
    labeled_indices = np.random.choice(positive_indices, size=num_to_label, replace=False) if num_to_label > 0 else []
    
    observed_labels = np.zeros_like(labels)
    if len(labeled_indices) > 0:
        observed_labels[labeled_indices] = 1
    return observed_labels


def binarize_labels(labels: np.ndarray, positive_class_label: int) -> np.ndarray:
    """
    Binarizes labels based on the positive class.
    """
    return (labels == positive_class_label).astype(int)


def build_knn_graph_from_features(features: np.ndarray, k: int = 10) -> nx.Graph:
    """
    Constructs a k-NN graph from feature vectors.
    """
    adjacency_matrix = kneighbors_graph(features, n_neighbors=k, mode='connectivity', include_self=False)
    adjacency_matrix = ((adjacency_matrix + adjacency_matrix.T) > 0).astype(int)
    return nx.from_scipy_sparse_array(adjacency_matrix)


# --- Dataset Loaders ---

def _load_planetoid_dataset(
    name: str,
    positive_class_label: int,
    percent_positive: float,
    use_original_edges: bool,
    k: int,
    mst: bool,
    n_samples: int,
    stratified_sampling: bool,
    data_dir: str
) -> nx.Graph:
    """
    Generic loader for Planetoid datasets (Cora, CiteSeer) with robust attribute handling.
    """
    if to_networkx is None:
        raise ImportError("PyTorch Geometric is required for this function.")
        
    print(f"Loading {name} dataset...")
    dataset = Planetoid(root=data_dir, name=name)
    data = dataset[0]

    G = to_networkx(data, to_undirected=True)
    all_features = data.x.numpy()
    all_labels = data.y.numpy()

    # --- Step 1: Handle Connectivity First ---
    if use_original_edges:
        if not nx.is_connected(G):
            G = _handle_graph_connectivity(G, mst=mst, features=all_features)
    else:
        # For k-NN, we build it on the full dataset first
        G_knn = build_knn_graph_from_features(all_features, k=k)
        if not nx.is_connected(G_knn):
            G = _handle_graph_connectivity(G_knn, mst=mst, features=all_features)
        else:
            G = G_knn
    
    # --- Step 2: Then, perform sampling on the connected graph ---
    connected_nodes = sorted(list(G.nodes()))
    if n_samples and n_samples < len(connected_nodes):
        if stratified_sampling:
            print(f"Sampling {n_samples} nodes with stratified sampling from the largest component...")
            # Filter features/labels to only the connected component for stratified sampling
            lcc_features = all_features[connected_nodes]
            lcc_labels = all_labels[connected_nodes]
            sample_indices_in_lcc = _stratified_sample(lcc_features, lcc_labels, n_samples)
            # Map back to original node IDs
            sampled_node_ids = [connected_nodes[i] for i in sample_indices_in_lcc]
        else:
            print(f"Sampling the first {n_samples} nodes from the largest component...")
            sampled_node_ids = connected_nodes[:n_samples]
        
        G = G.subgraph(sampled_node_ids).copy()

    final_node_ids = sorted(list(G.nodes()))
    
    final_features = all_features[final_node_ids]
    final_labels = all_labels[final_node_ids]

    final_labels_binary = binarize_labels(final_labels, positive_class_label)
    final_observed_labels = apply_scar_labeling(final_labels_binary, 1, percent_positive)
    
    node_attributes = {
        node_id: {
            'features': final_features[i],
            'true_label': final_labels_binary[i],
            'observed_label': final_observed_labels[i]
        }
        for i, node_id in enumerate(final_node_ids)
    }
    nx.set_node_attributes(G, node_attributes)
    
    print(f"{name} graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    return G


def load_cora_scar(
    positive_class_label: int,
    percent_positive: float,
    use_original_edges: bool = True,
    k: int = 10,
    mst: bool = True,
    n_samples: int = None,
    stratified_sampling: bool = True,
    data_dir: str = '/tmp/Cora'
) -> nx.Graph:
    return _load_planetoid_dataset("Cora", positive_class_label, percent_positive, use_original_edges, k, mst, n_samples, stratified_sampling, data_dir)


def load_citeseer_scar(
    positive_class_label: int,
    percent_positive: float,
    use_original_edges: bool = True,
    k: int = 10,
    mst: bool = True,
    n_samples: int = None,
    stratified_sampling: bool = True,
    data_dir: str = '/tmp/CiteSeer'
) -> nx.Graph:
    return _load_planetoid_dataset("CiteSeer", positive_class_label, percent_positive, use_original_edges, k, mst, n_samples, stratified_sampling, data_dir)


def load_twitch_scar(
    percent_positive: float,
    mst: bool = False,
    data_dir: str = 'data/twitch_gamers'
) -> nx.Graph:
    """
    Load Twitch dataset from local files, filtering to only include Portuguese (PT) users.
    """
    print("Loading Twitch dataset from local files, filtering to PT users...")
    
    try:
        # Load features and edges
        features_path = os.path.join(data_dir, 'large_twitch_features.csv')
        edges_path = os.path.join(data_dir, 'large_twitch_edges.csv')
        
        features_df = pd.read_csv(features_path)
        edges_df = pd.read_csv(edges_path)
        
        print(f"Loaded features: {len(features_df)} users")
        print(f"Loaded edges: {len(edges_df)} connections")
        
        # Filter to only PT users
        pt_users = features_df[features_df['language'] == 'PT']
        print(f"Found {len(pt_users)} PT users")
        
        if len(pt_users) == 0:
            print("Error: No PT users found. Check if 'name' column contains 'PT' values.")
            return None
        
        # Get the numeric IDs of PT users
        pt_user_ids = set(pt_users['numeric_id'].values)
        
        # Filter edges to only include connections between PT users
        pt_edges = edges_df[
            (edges_df['numeric_id_1'].isin(pt_user_ids)) & 
            (edges_df['numeric_id_2'].isin(pt_user_ids))
        ]
        
        print(f"Found {len(pt_edges)} connections between PT users")
        
        # Create graph from PT edges
        G = nx.from_pandas_edgelist(
            pt_edges, 
            source='numeric_id_1', 
            target='numeric_id_2', 
            create_using=nx.Graph()
        )
        
        # Filter features to only PT users
        pt_features_df = features_df[features_df['numeric_id'].isin(G.nodes())]
        
        # Rename affiliate column to true_label
        pt_features_df = pt_features_df.copy()
        pt_features_df['true_label'] = pt_features_df['affiliate']
        
        # Handle connectivity if needed
        if not nx.is_connected(G):
            print("Graph is not connected. Handling connectivity...")
            # Get features for connectivity handling - only use numeric columns
            feature_columns = ['views', 'mature', 'life_time', 'dead_account']
            all_features = pt_features_df[feature_columns].values.astype(np.float64)
            G = _handle_graph_connectivity(G, mst=mst, features=all_features)
        
        # Get final node list after connectivity handling
        final_nodes = sorted(list(G.nodes()))
        final_features_df = pt_features_df[pt_features_df['numeric_id'].isin(final_nodes)]
        
        # Extract features and labels - only use numeric columns
        feature_columns = ['views', 'mature', 'life_time', 'dead_account']
        final_features = final_features_df[feature_columns].values.astype(np.float64)
        true_labels_binary = final_features_df['true_label'].values
        
        # Apply SCAR labeling
        observed_labels = apply_scar_labeling(true_labels_binary, 1, percent_positive)
        
        # Set node attributes
        node_attributes = {}
        for i, node in enumerate(final_nodes):
            node_attributes[node] = {
                'features': final_features[i],
                'true_label': true_labels_binary[i],
                'observed_label': observed_labels[i]
            }
        
        nx.set_node_attributes(G, node_attributes)
        
        print(f"Twitch-PT graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
        return G
        
    except Exception as e:
        print(f"Error loading Twitch dataset: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_mnist_scar(
    percent_positive: float,
    k: int = 10,
    mst: bool = True,
    n_samples: int = None,
    stratified_sampling: bool = False
) -> nx.Graph:
    print("Loading MNIST dataset...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    
    features = mnist.data[:n_samples]
    labels = mnist.target.astype(int)[:n_samples]

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    G = build_knn_graph_from_features(features_scaled, k=k)
    
    if not nx.is_connected(G):
        G = _handle_graph_connectivity(G, mst=mst, features=features_scaled)
    
    connected_nodes = sorted(list(G.nodes()))

    # if n_samples and n_samples < len(connected_nodes):
    #     lcc_features = features[connected_nodes]
    #     lcc_labels = labels[connected_nodes]
    #     if stratified_sampling:
    #         print(f"Sampling {n_samples} nodes with stratified sampling from the largest component...")
    #         sample_indices_in_lcc = _stratified_sample(lcc_features, lcc_labels, n_samples)
    #         sampled_node_ids = [connected_nodes[i] for i in sample_indices_in_lcc]
    #     else:
    #         print(f"Sampling the first {n_samples} nodes from the largest component...")
    #         sampled_node_ids = connected_nodes[:n_samples]
        
    #     G = G.subgraph(sampled_node_ids).copy()

    final_node_ids = sorted(list(G.nodes()))

    final_features = features[final_node_ids]
    final_labels = labels[final_node_ids]

    positive_classes = [0, 2, 4, 6, 8]
    final_labels_binary = np.isin(final_labels, positive_classes).astype(int)
    final_observed_labels = apply_scar_labeling(final_labels_binary, 1, percent_positive)

    final_features_scaled = scaler.fit_transform(final_features)

    node_attributes = {
        node_id: {
            'features': final_features_scaled[i],
            'true_label': final_labels_binary[i],
            'observed_label': final_observed_labels[i]
        }
        for i, node_id in enumerate(final_node_ids)
    }
    nx.set_node_attributes(G, node_attributes)

    print(f"MNIST graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    return G


def load_ionosphere_scar(
    percent_positive: float,
    k: int = 10,
    mst: bool = True,
    n_samples: int = None,
    stratified_sampling: bool = True
) -> nx.Graph:
    print("Loading Ionosphere dataset...")
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data'
    df = pd.read_csv(url, header=None)
    
    features = df.iloc[:, :-1].values
    labels_str = df.iloc[:, -1].values
    
    le = LabelEncoder()
    labels = le.fit_transform(labels_str)

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    G = build_knn_graph_from_features(features_scaled, k=k)
    
    if not nx.is_connected(G):
        G = _handle_graph_connectivity(G, mst=mst, features=features_scaled)
    
    connected_nodes = sorted(list(G.nodes()))

    if n_samples and n_samples < len(connected_nodes):
        lcc_features = features[connected_nodes]
        lcc_labels = labels[connected_nodes]
        if stratified_sampling:
            print(f"Sampling {n_samples} nodes with stratified sampling from the largest component...")
            sample_indices_in_lcc = _stratified_sample(lcc_features, lcc_labels, n_samples)
            sampled_node_ids = [connected_nodes[i] for i in sample_indices_in_lcc]
        else:
            print(f"Sampling the first {n_samples} nodes from the largest component...")
            sampled_node_ids = connected_nodes[:n_samples]
        
        G = G.subgraph(sampled_node_ids).copy()

    final_node_ids = sorted(list(G.nodes()))

    final_features = features[final_node_ids]
    final_labels = labels[final_node_ids]

    positive_class_label = le.transform(['g'])[0]
    final_labels_binary = binarize_labels(final_labels, positive_class_label)
    final_observed_labels = apply_scar_labeling(final_labels_binary, 1, percent_positive)
    
    final_features_scaled = scaler.fit_transform(final_features)

    node_attributes = {
        node_id: {
            'features': final_features_scaled[i],
            'true_label': final_labels_binary[i],
            'observed_label': final_observed_labels[i]
        }
        for i, node_id in enumerate(final_node_ids)
    }
    nx.set_node_attributes(G, node_attributes)
    
    print(f"Ionosphere graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    return G


# --- Example Usage ---
if __name__ == '__main__':
    print("--- Running Example: Cora (LCC then First-N Sampling) ---")
    cora_graph_strat = load_cora_scar(
        positive_class_label=2, 
        percent_positive=0.1, 
        use_original_edges=True,
        mst=False,
        n_samples=1000,
        stratified_sampling=False
    )
    if cora_graph_strat: 
        print(f"Graph is connected: {nx.is_connected(cora_graph_strat)}")
        print(f"Final node count: {cora_graph_strat.number_of_nodes()}")
    print("-" * 20)

    print("\n--- Running Example: MNIST (LCC then Stratified Sampling) ---")
    mnist_graph_first_n = load_mnist_scar(
        percent_positive=0.2, 
        k=15, 
        mst=False,
        n_samples=500,
        stratified_sampling=True
    )
    if mnist_graph_first_n: 
        print(f"Graph is connected: {nx.is_connected(mnist_graph_first_n)}")
        print(f"Final node count: {mnist_graph_first_n.number_of_nodes()}")
    print("-" * 20)

