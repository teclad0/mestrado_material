import networkx as nx
import numpy as np
import pandas as pd
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics.pairwise import rbf_kernel, euclidean_distances
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.datasets import fetch_openml
import os
import requests
import zipfile
import io

# --- New Dependencies ---
# You will need to install torch and torch_geometric
# pip install torch torch-geometric
try:
    from torch_geometric.datasets import Planetoid, Twitch
    from torch_geometric.utils import to_networkx
except ImportError:
    print("PyTorch Geometric not found. Please install it with 'pip install torch torch-geometric'")
    to_networkx = None # Define to avoid NameError on script load

# --- Helper Functions ---

def apply_scar_labeling(labels: np.ndarray, positive_class_label: int, percent_positive: float) -> np.ndarray:
    """
    Applies the SCAR (Selected Completely at Random) labeling assumption.

    Args:
        labels (np.ndarray): The array of true labels.
        positive_class_label (int): The label value representing the positive class.
        percent_positive (float): The percentage (0.0 to 1.0) of positive nodes to be labeled.

    Returns:
        np.ndarray: The array of observed labels (1 for labeled positive, 0 for unlabeled).
    """
    if not (0.0 <= percent_positive <= 1.0):
        raise ValueError("percent_positive must be between 0.0 and 1.0.")

    positive_indices = np.where(labels == positive_class_label)[0]
    
    # Ensure we don't try to sample more than available
    num_to_label = min(len(positive_indices), int(len(positive_indices) * percent_positive))
    
    # Randomly select a subset of positive nodes to be labeled
    if num_to_label > 0:
        labeled_indices = np.random.choice(positive_indices, size=num_to_label, replace=False)
    else:
        labeled_indices = []

    observed_labels = np.zeros_like(labels)
    if len(labeled_indices) > 0:
        observed_labels[labeled_indices] = 1
    
    return observed_labels

def binarize_labels(labels: np.ndarray, positive_class_label: int) -> np.ndarray:
    """
    Binarizes labels based on the positive class.
    
    Args:
        labels (np.ndarray): The array of original labels.
        positive_class_label (int): The label value representing the positive class.
    
    Returns:
        np.ndarray: Binary labels (1 for positive class, 0 for others).
    """
    return (labels == positive_class_label).astype(int)

def build_knn_graph_from_features(
    features: np.ndarray, 
    k: int = 10
) -> nx.Graph:
    """
    Constructs a k-NN graph from feature vectors.

    Args:
        features (np.ndarray): A (n_samples, n_features) matrix of features.
        k (int): The number of neighbors for the k-NN graph.

    Returns:
        nx.Graph: The constructed NetworkX graph.
    """
    # k-NN graph construction
    adjacency_matrix = kneighbors_graph(features, n_neighbors=k, mode='connectivity', include_self=False)
    # Symmetrize the matrix
    adjacency_matrix = ((adjacency_matrix + adjacency_matrix.T) > 0).astype(int)
        
    return nx.from_scipy_sparse_array(adjacency_matrix)

def make_graph_connected_with_mst(G: nx.Graph, features: np.ndarray = None) -> nx.Graph:
    """
    Makes a graph fully connected by adding MST edges between disconnected components.
    
    Args:
        G (nx.Graph): Input graph (possibly disconnected).
        features (np.ndarray): Optional feature matrix for computing distances between components.
                              If None, uses graph distance.
    
    Returns:
        nx.Graph: Fully connected graph with original edges plus MST edges.
    """
    # Identify connected components
    components = list(nx.connected_components(G))
    
    if len(components) == 1:
        # Graph is already connected
        return G.copy()
    
    print(f"Graph has {len(components)} connected components. Connecting them using MST...")
    
    # Create a copy of the graph to modify
    G_connected = G.copy()
    
    # Create a complete graph between component representatives
    component_graph = nx.Graph()
    representatives = []  # One node from each component
    
    for i, comp in enumerate(components):
        # Choose a representative node from each component (e.g., the one with minimum index)
        rep = min(comp)
        representatives.append(rep)
        component_graph.add_node(i, representative=rep, component=comp)
    
    # Add edges between all pairs of components with weights based on distance
    for i in range(len(components)):
        for j in range(i + 1, len(components)):
            rep_i = representatives[i]
            rep_j = representatives[j]
            
            if features is not None:
                # Use feature distance
                distance = np.linalg.norm(features[rep_i] - features[rep_j])
            else:
                # Use a default distance (could be customized)
                distance = 1.0
            
            component_graph.add_edge(i, j, weight=distance)
    
    # Compute MST on the component graph
    mst = nx.minimum_spanning_tree(component_graph)
    
    # Add edges to connect components based on MST
    for edge in mst.edges():
        comp_i, comp_j = edge
        rep_i = component_graph.nodes[comp_i]['representative']
        rep_j = component_graph.nodes[comp_j]['representative']
        
        # Find the closest pair of nodes between the two components
        if features is not None:
            comp_i_nodes = list(component_graph.nodes[comp_i]['component'])
            comp_j_nodes = list(component_graph.nodes[comp_j]['component'])
            
            min_dist = float('inf')
            best_pair = (rep_i, rep_j)
            
            # Find the closest pair of nodes between components
            for node_i in comp_i_nodes[:min(10, len(comp_i_nodes))]:  # Limit search for efficiency
                for node_j in comp_j_nodes[:min(10, len(comp_j_nodes))]:
                    dist = np.linalg.norm(features[node_i] - features[node_j])
                    if dist < min_dist:
                        min_dist = dist
                        best_pair = (node_i, node_j)
            
            G_connected.add_edge(best_pair[0], best_pair[1])
        else:
            # Just connect the representatives
            G_connected.add_edge(rep_i, rep_j)
    
    # Verify connectivity
    if nx.is_connected(G_connected):
        print(f"Graph is now fully connected with {G_connected.number_of_nodes()} nodes and {G_connected.number_of_edges()} edges.")
    else:
        print("Warning: Graph is still not fully connected after MST operation.")
    
    return G_connected

# --- Dataset Loaders ---

def _load_twitch_data_manual(name: str, data_dir: str):
    """
    Manually downloads and loads Twitch data from a reliable source.
    """
    url = f"https://raw.githubusercontent.com/shchur/gnn-benchmark/master/data/twitch_gamers/{name}.npz"
    filepath = os.path.join(data_dir, f"{name}.npz")
    
    if not os.path.exists(filepath):
        os.makedirs(data_dir, exist_ok=True)
        print(f"Downloading Twitch-{name} data from {url}...")
        response = requests.get(url)
        response.raise_for_status()
        with open(filepath, 'wb') as f:
            f.write(response.content)
            
    data = np.load(filepath)
    return data['features'], data['labels'], data['adj_matrix']

def load_twitch_scar(
    name: str,
    percent_positive: float,
    use_original_edges: bool = True,
    k: int = 10,
    data_dir: str = '/tmp/Twitch'
) -> nx.Graph:
    """
    Loads a Twitch gamer dataset, builds a graph, and applies SCAR labeling.
    The positive class is 1 (bot). Graph is made fully connected using MST.
    """
    print(f"Loading Twitch-{name} dataset...")
    features, true_labels, adj_sparse = _load_twitch_data_manual(name, data_dir)
    
    num_nodes = features.shape[0]
    positive_class_label = 1 # In Twitch datasets, 1 usually means 'bot'
    
    # Binarize true labels (already binary in Twitch, but keeping for consistency)
    true_labels_binary = binarize_labels(true_labels, positive_class_label)

    if use_original_edges:
        print("Using original graph structure.")
        G = nx.from_scipy_sparse_array(adj_sparse)
    else:
        print(f"Constructing k-NN graph from features with k={k}")
        G = build_knn_graph_from_features(features, k=k)

    # Make graph fully connected using MST
    G = make_graph_connected_with_mst(G, features)
    
    # Apply SCAR labeling on binary labels
    observed_labels = apply_scar_labeling(true_labels_binary, 1, percent_positive)
    
    node_attributes = {
        i: {
            'features': features[i],
            'true_label': true_labels_binary[i],
            'observed_label': observed_labels[i]
        }
        for i in range(num_nodes)
    }
    nx.set_node_attributes(G, node_attributes)
    
    print(f"Twitch-{name} graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    return G

def load_cora_scar(
    positive_class_label: int,
    percent_positive: float,
    use_original_edges: bool = True,
    k: int = 10,
    data_dir: str = '/tmp/Cora'
) -> nx.Graph:
    """
    Loads the Cora dataset using PyG, builds a graph, and applies SCAR labeling.
    Graph is made fully connected using MST.
    """
    if to_networkx is None:
        raise ImportError("PyTorch Geometric is required for this function.")
        
    print("Loading Cora dataset using PyTorch Geometric...")
    dataset = Planetoid(root=data_dir, name='Cora')
    data = dataset[0]

    features = data.x.numpy()
    true_labels = data.y.numpy()
    num_nodes = features.shape[0]
    
    # Binarize true labels based on positive class
    true_labels_binary = binarize_labels(true_labels, positive_class_label)

    if use_original_edges:
        print("Using original graph structure.")
        # Convert the PyG Data object to a NetworkX graph
        G = to_networkx(data, to_undirected=True)
    else:
        print(f"Constructing k-NN graph from features with k={k}")
        G = build_knn_graph_from_features(features, k=k)

    # Make graph fully connected using MST
    G = make_graph_connected_with_mst(G, features)

    # Apply SCAR labeling on binary labels
    observed_labels = apply_scar_labeling(true_labels_binary, 1, percent_positive)
    
    # Add attributes to all nodes
    node_attributes = {
        i: {
            'features': features[i],
            'true_label': true_labels_binary[i],
            'observed_label': observed_labels[i]
        }
        for i in range(num_nodes)
    }
    nx.set_node_attributes(G, node_attributes)
    
    print(f"Cora graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    return G

def load_citeseer_scar(
    positive_class_label: int,
    percent_positive: float,
    use_original_edges: bool = True,
    k: int = 10,
    data_dir: str = '/tmp/CiteSeer'
) -> nx.Graph:
    """
    Loads the CiteSeer dataset using PyG, builds a graph, and applies SCAR labeling.
    Graph is made fully connected using MST.
    """
    if to_networkx is None:
        raise ImportError("PyTorch Geometric is required for this function.")

    print("Loading CiteSeer dataset using PyTorch Geometric...")
    dataset = Planetoid(root=data_dir, name='CiteSeer')
    data = dataset[0]

    features = data.x.numpy()
    true_labels = data.y.numpy()
    num_nodes = features.shape[0]
    
    # Binarize true labels based on positive class
    true_labels_binary = binarize_labels(true_labels, positive_class_label)

    if use_original_edges:
        print("Using original graph structure.")
        G = to_networkx(data, to_undirected=True)
    else:
        print(f"Constructing k-NN graph from features with k={k}")
        G = build_knn_graph_from_features(features, k=k)

    # Make graph fully connected using MST
    G = make_graph_connected_with_mst(G, features)

    # Apply SCAR labeling on binary labels
    observed_labels = apply_scar_labeling(true_labels_binary, 1, percent_positive)
    
    node_attributes = {
        i: {
            'features': features[i],
            'true_label': true_labels_binary[i],
            'observed_label': observed_labels[i]
        }
        for i in range(num_nodes)
    }
    nx.set_node_attributes(G, node_attributes)
    
    print(f"CiteSeer graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    return G

def load_mnist_scar(
    percent_positive: float,
    k: int = 10,
    num_samples: int = 5000
) -> nx.Graph:
    """
    Loads the MNIST dataset, converts to binary (even vs. odd),
    constructs a k-NN graph, and applies SCAR labeling.
    Graph is made fully connected using MST.
    """
    print("Loading MNIST dataset...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    
    # Subsample the dataset for manageable graph sizes
    #indices = np.random.choice(len(mnist.data), num_samples, replace=False)
    features = mnist.data
    original_labels = mnist.target

    # Convert to binary: 0, 2, 4, 6, 8 are positives (1), others are negatives (0)
    positive_classes = [0, 2, 4, 6, 8]
    true_labels_binary = np.isin(original_labels, positive_classes).astype(int)

    # Normalize features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    print(f"Building k-NN graph from features with k={k}")
    G = build_knn_graph_from_features(features, k=k)
    
    # Make graph fully connected using MST
    G = make_graph_connected_with_mst(G, features)
    
    # Apply SCAR labeling on binary labels
    observed_labels = apply_scar_labeling(true_labels_binary, 1, percent_positive)

    # Add attributes to nodes
    node_attributes = {
        i: {
            'features': features[i],
            'true_label': true_labels_binary[i],
            'observed_label': observed_labels[i]
        }
        for i in range(num_samples)
    }
    nx.set_node_attributes(G, node_attributes)

    print(f"MNIST graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    return G

def load_ionosphere_scar(
    percent_positive: float,
    k: int = 10
) -> nx.Graph:
    """
    Loads the Ionosphere dataset, constructs a k-NN graph, and applies SCAR labeling.
    The positive class is 'g' (good). Graph is made fully connected using MST.
    """
    print("Loading Ionosphere dataset...")
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data'
    df = pd.read_csv(url, header=None)
    
    features = df.iloc[:, :-1].values
    labels_str = df.iloc[:, -1].values
    
    # Encode labels: 'g' -> 1 (positive), 'b' -> 0 (negative)
    le = LabelEncoder()
    true_labels = le.fit_transform(labels_str)
    positive_class_label = le.transform(['g'])[0]
    
    # Binarize true labels (already binary, but keeping for consistency)
    true_labels_binary = binarize_labels(true_labels, positive_class_label)
    
    # Normalize features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    
    print(f"Building k-NN graph from features with k={k}")
    G = build_knn_graph_from_features(features, k=k)
    
    # Make graph fully connected using MST
    G = make_graph_connected_with_mst(G, features)
    
    # Apply SCAR labeling on binary labels
    observed_labels = apply_scar_labeling(true_labels_binary, 1, percent_positive)
    
    # Add attributes to nodes
    node_attributes = {
        i: {
            'features': features[i],
            'true_label': true_labels_binary[i],
            'observed_label': observed_labels[i]
        }
        for i in range(features.shape[0])
    }
    nx.set_node_attributes(G, node_attributes)
    
    print(f"Ionosphere graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    return G

if __name__ == '__main__':
    print("--- Running Example: Cora ---")
    # Example: Load Cora, treat class 2 as positive, label 10% of them.
    # Use the original graph edges.
    cora_graph = load_cora_scar(
        positive_class_label=2, 
        percent_positive=0.1, 
        use_original_edges=True
    )
    # Check a node's attributes
    if 10 in cora_graph.nodes:
        node_10_data = cora_graph.nodes[10]
        print(f"Node 10 Data: True Label={node_10_data['true_label']}, Observed Label={node_10_data['observed_label']}")
        print(f"Graph is connected: {nx.is_connected(cora_graph)}")
    else:
        print("Node 10 not found in graph.")
    print("-" * 20)

    print("\n--- Running Example: MNIST ---")
    # Example: Load MNIST, treat even digits as positive, label 20% of them.
    # Build a 15-NN graph.
    mnist_graph = load_mnist_scar(
        percent_positive=0.2, 
        k=15, 
        num_samples=1000 # Use a smaller sample for a quick test
    )
    # Check a node's attributes
    node_20_data = mnist_graph.nodes[20]
    print(f"Node 20 Data: True Label={node_20_data['true_label']}, Observed Label={node_20_data['observed_label']}")
    print(f"Graph is connected: {nx.is_connected(mnist_graph)}")
    print("-" * 20)
    
    print("\n--- Running Example: Ionosphere ---")
    # Example: Load Ionosphere, label 50% of the positive class ('g').
    # Build a k-NN graph with k=10.
    ionosphere_graph = load_ionosphere_scar(
        percent_positive=0.5, 
        k=10
    )
    # Check a node's attributes
    node_30_data = ionosphere_graph.nodes[30]
    print(f"Node 30 Data: True Label={node_30_data['true_label']}, Observed Label={node_30_data['observed_label']}")
    print(f"Graph is connected: {nx.is_connected(ionosphere_graph)}")
    print("-" * 20)