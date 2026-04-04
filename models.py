
from sklearn.cluster import KMeans
import numpy as np
from networkit.nxadapter import nk2nx
from torch_geometric.utils import to_networkit
import networkit as nk
import networkx as nx
import copy
import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch.nn
from sklearn.svm import OneClassSVM
from sklearn.neighbors import kneighbors_graph
import scipy.sparse as sp

def _to_long_tensor(indices):
    """Convert list/array/tensor indices to a 1-D long tensor."""
    if isinstance(indices, torch.Tensor):
        return indices.long().flatten()
    return torch.tensor(list(indices), dtype=torch.long)

def _edge_index_to_dense_adjacency(edge_index, num_nodes):
    """
    Build a dense adjacency matrix from PyG edge_index.
    Keeps self-loops at 0 and symmetrizes for undirected usage.
    """
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float64)
    if edge_index is None or edge_index.numel() == 0:
        return adj.numpy()
    src = edge_index[0].long()
    dst = edge_index[1].long()
    adj[src, dst] = 1.0
    # Ensure symmetry for methods that assume undirected graphs.
    adj = torch.maximum(adj, adj.t())
    return adj.numpy()

def _prepare_pu_lp_input(data):
    """
    Normalize PU_LP input from either:
    - PyG Data (with x, edge_index, P, U)
    - NetworkX graph (with node attribute 'observed_label')
    """
    # Case 1: NetworkX graph
    if isinstance(data, nx.Graph):
        node_list = sorted(list(data.nodes()))
        node_to_idx = {node: idx for idx, node in enumerate(node_list)}
        adj = nx.to_numpy_array(data, nodelist=node_list, dtype=np.float64)

        observed = []
        for node in node_list:
            attrs = data.nodes[node]
            if 'observed_label' not in attrs:
                raise ValueError("NetworkX graph nodes must include 'observed_label' for PU_LP.")
            observed.append(attrs['observed_label'])
        observed = np.array(observed)
        positives = torch.tensor(np.where(observed == 1)[0], dtype=torch.long)
        unlabeled = torch.tensor(np.where(observed == 0)[0], dtype=torch.long)
        return adj, positives, unlabeled, node_list, node_to_idx

    # Case 2: PyG-like Data
    if not hasattr(data, "edge_index") or not hasattr(data, "x"):
        raise ValueError("PU_LP input must be a NetworkX graph or PyG Data with edge_index/x.")
    adj = _edge_index_to_dense_adjacency(data.edge_index, data.x.shape[0])
    positives = _to_long_tensor(data.P)
    unlabeled = _to_long_tensor(data.U)
    return adj, positives, unlabeled, None, None

class PU_LP:
    '''
    Class to implement the PU_LP algorithm
    '''
    def __init__(self, data, alpha=0.1, m = 3, l = 1):
        self.data_graph, self.positives, self.unlabeled, self._idx_to_node, self._node_to_idx = _prepare_pu_lp_input(data)
        self.alpha = alpha
        self.m = m
        self.l = l

    def _restore_original_node_ids(self, idx_tensor):
        """Map internal contiguous indices back to original graph node ids when needed."""
        if self._idx_to_node is None:
            return idx_tensor
        mapped = [self._idx_to_node[int(i)] for i in idx_tensor.tolist()]
        if all(isinstance(x, (int, np.integer)) for x in mapped):
            return torch.tensor(mapped, dtype=torch.long)
        return mapped
        
    def train(self):
        '''
        Method to train the model

        Main changes: Considering the fact that the model automatically returns the number of reliable negative elements, we change the number of elements to be returned. This action is justified by the fact that the benchmark needs to be more fair possible.
        '''

        # Setting the dinamic parameters
        self.RP = torch.tensor([], dtype=torch.long)
        P_line = self.positives.clone()
        U_line = self.unlabeled.clone()

        # Computing W = (I - alpha*A)**-1 - I
        I = np.eye(len(self.data_graph))
        W = np.linalg.inv(I - self.alpha * self.data_graph) - I
        W = torch.tensor(W, dtype=torch.float64)

        # Ranking the most similar elements based on the selected positives
        for k in range(self.m):
            rank_dict = dict()
            for vi in U_line.tolist():
                S_vi = 0
                for vj in P_line.tolist():
                    S_vi += W[vi, vj]
                S_vi /= len(P_line)
                rank_dict[vi] = S_vi
            rank_dict = sorted(rank_dict.items(), key=lambda x:x[1], reverse=True)
            rank_dict = [tupla[0] for tupla in rank_dict]

            # Updating sets each iteration
            num_to_promote = int((self.l / self.m) * len(self.positives))
            if num_to_promote <= 0:
                continue
            RP_line = torch.tensor(rank_dict[:num_to_promote], dtype=torch.long)
            if RP_line.numel() == 0:
                continue
            P_line = torch.cat([P_line, RP_line], dim=0)
            U_line = torch.tensor(list(set(U_line.tolist()) - set(RP_line.tolist())), dtype=torch.long)
            self.RP = torch.cat([self.RP, RP_line], dim=0)

        # Classifying the reliable negatives
        rank_dict = dict()
        current_pos = torch.cat([self.positives, self.RP], dim=0)
        remaining_unlabeled = list(set(self.unlabeled.tolist()) - set(self.RP.tolist()))
        for vi in remaining_unlabeled:
            S_vi = 0
            for vj in current_pos.tolist():
                S_vi += W[vi, int(vj)]
            S_vi /= len(P_line)
            rank_dict[vi] = S_vi
        rank_dict = sorted(rank_dict.items(), key=lambda x:x[1])
        rank_dict = [tupla[0] for tupla in rank_dict]
        self.RN = torch.tensor(rank_dict, dtype=torch.long)

    def negative_inference(self, num_neg = None):
        '''
        Method that returns the reliable negative examples. The num_neg value can be set to none and the algorithm will return the number of negative elements based on the proposed method.

        Parameters:
        num_neg (int): Number of reliable negative elements to return

        Returns:
        torch.tensor: tensor of negative elements.
        '''
        #return self.RN[-len(self.positives + self.RP):][:num_neg]
        if num_neg is None:
            num_neg = len(self.RN)
        num_neg = min(num_neg, len(self.RN))
        if num_neg == 0:
            return torch.tensor([], dtype=torch.long)
        return self._restore_original_node_ids(self.RN[:num_neg])
    
    
    def positive_inference(self):
        '''
        Method that returns the positive elements (computed at the training phase)
        '''
        return self._restore_original_node_ids(self.RP)

def mst_graph(features):
    """
    Create a minimum spanning tree graph from features.
    
    Args:
        features: Feature matrix (tensor or numpy array)
    
    Returns:
        Sparse adjacency matrix of the MST
    """
    if hasattr(features, 'detach'):
        features = features.detach().numpy()
    
    # Create k-NN graph first
    knn_graph = kneighbors_graph(features, n_neighbors=min(5, len(features)-1), 
                                mode='connectivity', include_self=False)
    
    # Convert to NetworkX for MST computation
    import networkx as nx
    G = nx.from_scipy_sparse_array(knn_graph)
    
    # Compute MST
    mst = nx.minimum_spanning_tree(G)
    
    # Convert back to sparse matrix
    return nx.to_scipy_sparse_array(mst)

def cluster_signal_ratio(cluster: list, positives: list, ratio: float = 0.5)-> int:
    '''
    Compute the signal of a cluster. If more than 0.5 of the cluster has positive labels, then return 1. Else, return 0

    Parameters:
    cluster: The cluster to determine if it's positive or negative
    positives: The list of positive elements (indexes)
    ratio: The ratio that determines if a cluster is positive or negative

    Returns:
    0 if the cluster is negative, 1 if it's positive
    '''
    pos = 0
    for i in cluster:
        if i in positives:
            pos += 1
    if pos > ratio * len(cluster):
        return 1
    else:
        return 0

def cluster_signal_ratio(cluster: list, positives: list, ratio: float = 0.5)-> int:
    '''
    Compute the signal of a cluster. If more than 0.5 of the cluster has positive labels, then return 1. Else, return 0

    Parameters:
    cluster: The cluster to determine if it's positive or negative
    positives: The list of positive elements (indexes)
    ratio: The ratio that determines if a cluster is positive or negative

    Returns:
    0 if the cluster is negative, 1 if it's positive
    '''
    pos = 0
    for i in cluster:
        if i in positives:
            pos += 1
    if pos > ratio * len(cluster):
        return 1
    else:
        return 0
    
def cluster_signal_abs(cluster: list, positives: list, k: int):

    '''
    Compute the signal of a cluster based in how many positive elements they have. If there is more than num_positives/k positive elements, than the cluster is positive. Else, is negative.

    Parameters:
    cluster: The cluster to determine if it's positive or negative
    positives: The list of positive elements (indexes)
    k: the number of clusters used in the MCLS algorithm

    Returns:
    0 if the cluster is negative, 1 if it's positive   
    '''
    pos = 0
    for i in cluster:
        if i in positives:
            pos += 1
    if pos > len(positives) / k:
        return 1
    else:
        return 0

def euclidean_distance(tensor1: torch.tensor, tensor2: torch.tensor) -> torch.tensor:
    '''
    Compute the distance between two torch tensors

    Parameters:
    tensor1: the first input tensor
    tensor2: the second input tensor

    Returns:
    The euclidean distance between tensor1 and tensor2
    '''
    return torch.sqrt(torch.sum((tensor1 - tensor2) ** 2))

class MCLS:
    '''
    Class that implements the MCLS algorithm.

    Main changes: Considering there is no waranty that if it's possible to correct classify a cluster (the algorithm can classify all clusters as negative), we change the way of compute if a cluster is negative or positive. If the algorithm classify all clusters as negatives and there is more than num_positive/k elements positives in a cluster, then, this cluster is positive.

    This change guarantees that the algorithm can be run.
    '''
    def __init__(self, data, k = 7, ratio = 0.3):
        self.positives = data.P
        self.k = k
        self.ratio = ratio
        self.data = data.x
        self.distance = dict()
   
    def train(self):
        '''
        Method that trains the model
        '''

        # Start the kmeans algorithm
        kmeans = KMeans(n_clusters=self.k)
        kmeans.fit(self.data.detach().numpy())
        clusters_labels = kmeans.labels_

        # Dictionary to fill with the clusters
        clusters = {}

        # filling the clusters
        for indice, rotulo in enumerate(clusters_labels):
            if rotulo not in clusters:
                clusters[rotulo] = []
            clusters[rotulo].append(indice)

        # dictionary to fill with cluster labels
        cluster_signals = {}

        # labeling clusters. If there is no clusters with ratio = 0.5, the algorithm uses the num_positives/k value to determine if a cluster is positive
        for cluster in clusters:
            sig = cluster_signal_ratio(clusters[cluster], self.positives,ratio = self.ratio)
            cluster_signals[cluster] = sig
        if np.sum(list(cluster_signals.values())) == 0:
            cluster_signals = {}
            for cluster in clusters:
                sig = cluster_signal_abs(clusters[cluster], self.positives, self.k)
                cluster_signals[cluster] = sig

        # Saving centroids
        cluster_centroids = {}
        centroids = kmeans.cluster_centers_

        print('number of centroids in MCLS', len(centroids))

        for i, center in enumerate(centroids):
            cluster_centroids[i] = center
        
        # list to save the elements of positive and negative clusters
        positive_clusters = [cluster for cluster in clusters if cluster_signals[cluster] == 1]
        negative_clusters = [cluster for cluster in clusters if cluster_signals[cluster] == 0]
        #import ipdb; ipdb.set_trace()

        # computing positive centroids
        positive_centroids = torch.tensor(np.array([cluster_centroids[i] for i in positive_clusters]))

        # filling the distance dictionary with the most distant elements
        for cluster in negative_clusters:
            for element in clusters[cluster]:
                distances = [euclidean_distance(self.data[element], centroid) for centroid in positive_centroids]
                mean_distance = torch.mean(torch.stack(distances))
                self.distance[element] = mean_distance

    def negative_inference(self, num_neg):
        '''
        Method to sort the elements from farest to nearest to return the most distant

        Parameters:
        num_neg: number of elements to classify as reliable negatives

        Returns:
        list: list with num_neg elements classify as reliable negatives
        '''
        RN = sorted(self.distance, key=self.distance.get, reverse=True)
        RN = RN[:num_neg]
        return RN

class LP_PUL:
    '''
    Class that implement LP_PUL algorithm.

    Main changes: If the graph is not connected, then it's impossible to compute the breadth first search length of every node to every node. To avoid this, we add the following criteria: If the graph is not connected then we create the Minimum Spanning Tree (MST) if the graph and add all the vertex that are not previously on the graph.
    '''
    def __init__(self, data):
        self.graph = to_networkit(data.edge_index, directed=False)
        self.data = data.x
        self.positives = data.P
        self.unlabeled = data.U

        
    def train(self):
        '''
        Method to train the model
        '''

        # Verify if it's conneted. If not, connetc the graph with the MST
        is_connected = nk.components.ConnectedComponents(self.graph).run().numberOfComponents() == 1
        if not is_connected:
            aux_g = nk2nx(self.graph)
            adj = nx.to_scipy_sparse_array(aux_g)  # Convert to sparse matrix
            adj_aux = mst_graph(self.data).toarray()
            
            rows, cols = np.where((adj.toarray() == 0) & (adj_aux == 1))
            
            for i, j in zip(rows, cols):
                self.graph.addEdge(i, j)

        # Distance vector
        self.a = np.zeros(len(self.unlabeled))

        # Compute the minimum paths with MultiTargetBFS function
        for p in self.positives:
            d = nk.distance.MultiTargetBFS(self.graph, p, self.unlabeled).run().getDistances()
            self.a += d
        self.a = self.a / len(self.positives)

    def negative_inference(self, num_neg):
        '''
        Method to classify the reliable negative examples (vertex)

        Parameter:
        num_neg (int): number of reliable negatives to returns

        Return:
        list: list of reliable negatives (indexes)
        '''
        RN = torch.stack([x for _, x in sorted(zip(self.a, self.unlabeled), reverse=True)][:num_neg])
        return RN