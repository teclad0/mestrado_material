
from sklearn.cluster import KMeans
import numpy as np
from networkit.nxadapter import nk2nx
from torch_geometric.utils import to_networkit
import networkit as nk
import networkx as nx
import copy
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch.nn
from sklearn.svm import OneClassSVM
from sklearn.neighbors import kneighbors_graph
import scipy.sparse as sp

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