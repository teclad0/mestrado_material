import networkx as nx
import matplotlib.pyplot as plt
import random
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas
from collections import defaultdict, Counter
from core import Particle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

def is_cluster_positive(
    graph: nx.Graph, 
    nodes_in_cluster: List[Any], 
    strategy: str,
    threshold: float = 0.5,
    positive_label: int = 1
) -> bool:
    """
    Determines if a cluster of nodes is 'positive' based on a given strategy.

    Args:
        graph: The NetworkX graph containing node attributes.
        nodes_in_cluster: A list of node IDs belonging to the cluster.
        strategy: The strategy to use ('majority' or 'percentage').
        threshold: The percentage threshold required for the 'percentage' strategy (e.g., 0.3 for 30%).
        positive_label: The label value representing the positive class (default: 1).

    Returns:
        True if the cluster is considered positive, False otherwise.
    """
    if not nodes_in_cluster:
        return False

    observed_labels = [graph.nodes[n]['observed_label'] for n in nodes_in_cluster]

    if strategy == 'majority':
        label_counts = Counter(observed_labels)
        # Handles ties by preferring the higher label value (1 over 0)
        majority_label = max(label_counts, key=lambda k: (label_counts.get(k, 0), k))
        return majority_label == positive_label

    elif strategy == 'percentage':
        count_positives = observed_labels.count(positive_label)
        percentage_positives = count_positives / len(nodes_in_cluster)
        return percentage_positives >= threshold

    else:
        raise ValueError(f"Unknown strategy: '{strategy}'. Choose 'majority' or 'percentage'.")


class ParticleCompetitionModel:
    """Main class implementing the particle competition algorithm"""
    
    def __init__(
        self, 
        graph: nx.Graph, 
        num_particles: int,
        p_det: float = 0.6,
        delta_p: float = 0.4,
        delta_v: float = 0.3,
        cluster_strategy: str = 'majority',
        positive_cluster_threshold: float = 0.5, 
        movement_strategy: str = 'uniform' ,
        initialization_strategy: str = 'random',
        average_node_potential_threshold: float = 0.7,
        coverage_graph_threshold: float = 0.8
    ):
        self.graph = graph
        self.degrees = dict(graph.degree())
        self.neighbors_dict = {node: list(graph.neighbors(node)) for node in graph.nodes}
        self.neighbor_degrees = {}
        self.num_particles = num_particles        
        self.p_det = p_det
        self.delta_p = delta_p
        self.delta_v = delta_v
        self.particles: List[Particle] = []        
        self.cluster_strategy = cluster_strategy
        self.positive_cluster_threshold = positive_cluster_threshold
        self.movement_strategy = movement_strategy
        self.initialization_strategy = initialization_strategy
        self.average_node_potential_threshold = average_node_potential_threshold
        self.coverage_graph_threshold = coverage_graph_threshold

        # Initialize graph nodes
        for node in self.graph.nodes:
            self.graph.nodes[node]['owner'] = None
            self.graph.nodes[node]['potential'] = 0.05
            nbrs = self.neighbors_dict[node]
            self.neighbor_degrees[node] = tuple(self.degrees[n] for n in nbrs)
        
        # Initialize particles
        self.initialize_particles()

        if self.initialization_strategy == 'degree_weighted':
            # Create a candidate pool at least as large as the number of particles
            nodes_sorted_by_degree = sorted(
                self.graph.nodes,
                key=lambda n: self.degrees[n],
                reverse=True
            )
            # Ensure we have enough unique nodes if the graph is large enough
            pool_size = min(len(nodes_sorted_by_degree), self.num_particles)
            self._candidate_start_nodes = nodes_sorted_by_degree[:pool_size]
            # Shuffle to randomize assignment among the top nodes
            random.shuffle(self._candidate_start_nodes)

        # Run the simulation
        #self.graph_populated = self._run_simulation()
        if self.movement_strategy not in ['uniform', 'degree_weighted']:
            raise ValueError("movement_strategy must be 'uniform' or 'degree_weighted'")
        
        if self.initialization_strategy not in ['random', 'degree_weighted']:
            raise ValueError("initialization_strategy must be 'random' or 'degree_weighted'")
        
        #self._precompute_network_data()

    def _precompute_network_data(self):
        """One-time precomputation for efficiency"""
        self.degrees = dict(self.graph.degree())
        
        # Only compute these if using degree_weighted strategy
        if self.movement_strategy == 'degree_weighted':
            self.neighbors_dict = {}
            self.neighbor_degrees = {}
            
            for node in self.graph.nodes:
                neighbors = list(self.graph.neighbors(node))
                self.neighbors_dict[node] = neighbors
                self.neighbor_degrees[node] = [self.degrees[n] for n in neighbors]

    def _degree_weighted_choice(self, neighbors, current_node):
        """Degree-weighted random selection from neighbors"""
        # Use precomputed degrees if available
        if hasattr(self, 'neighbor_degrees'):
            degrees = self.neighbor_degrees[current_node]
        else:
            # Compute on the fly if not precomputed
            degrees = [self.graph.degree[n] for n in neighbors]
        
        total = sum(degrees)
        if total == 0:
            return random.choice(neighbors)
        
        r = random.random() * total
        cumulative = 0
        for i, deg in enumerate(degrees):
            cumulative += deg
            if cumulative >= r:
                return neighbors[i]
        return neighbors[-1]
    
    def _get_distinct_start_node(self, particle: Particle) -> Any:
        """
        Get a unique high-degree start node for the particle.
        Ensures each particle gets a different node from the pre-computed
        candidate pool, preventing collisions.
        """
        # The list is already created and shuffled in __init__
        # Use the modulo operator only as a safeguard in case num_particles > pool_size,
        # though ideally, the pool should be large enough.
        return self._candidate_start_nodes[particle.id % len(self._candidate_start_nodes)]
    
    def initialize_particles(self):
        """Create particle instances"""
        self.particles = [Particle(i) for i in range(self.num_particles)]

    def move_particle(self,
        particle: Particle, 
    ) -> Any:
        '''
        The particle can visit a node it already visited or a random
        neighbor(position is from last visited from the last iteration)

        Args:
        particle (Particle): particle
        graph (Graph): graph
        '''
    # Handle uninitialized particles
        if not particle.visited_nodes:
            if self.initialization_strategy == 'degree_weighted':
                # Choose a distinct start node based on degree
                return self._get_distinct_start_node(particle)
            else:          
                return random.choice(list(self.graph.nodes))
        
        current_node = particle.current_position
        neighbors = self.neighbors_dict.get(current_node, []).copy()
        
        # Deterministic movement: owned neighbors
        if random.random() < self.p_det:
            # remove current node if particle is visiting again
            owned_nodes = [n for n in particle.visited_nodes if n != current_node] 

            if owned_nodes:
                if particle.node_visited_last_iteration in owned_nodes and \
                    self.degrees.get(particle.node_visited_last_iteration, 0) != 1:
                        owned_nodes.remove(particle.node_visited_last_iteration)
                # check if there the node_visitet_last_iteration wasn't the only one visited
                if owned_nodes:
                    return random.choice(owned_nodes)
        # Random movement: select based on strategy
        if self.movement_strategy == 'degree_weighted':
            return self._degree_weighted_choice(neighbors, current_node)
        else:  
            if particle.node_visited_last_iteration in neighbors and \
                    self.degrees.get(particle.node_visited_last_iteration, 0) != 1:
                        neighbors.remove(particle.node_visited_last_iteration)
                        # if there are no neighbors left, it's okay to return the last visited node
                        if not neighbors:
                            return particle.node_visited_last_iteration
            try:
                return random.choice(neighbors)
            except IndexError:
                import ipdb; ipdb.set_trace()
        
    def update_particle(self,
        particle: Particle, 
        node: Any,
    ) -> None:
        '''
        Dynamics of particles and dynamics of nodes

        Args:
        particle (Particle): particle
        node(Node): node the particle selected
        '''
        current_owner = self.graph.nodes[node]['owner']
        current_potential = self.graph.nodes[node]['potential']

        if current_owner is None:
            # Case 1: Node is free
            particle.visited_nodes.add(node)
            self.graph.nodes[node]['owner'] = particle.id
            self.graph.nodes[node]['potential'] = particle.potential
            
        elif current_owner == particle.id:
            # Case 2: Node owned by current particle
            particle.potential = min(1, particle.potential + (1 - particle.potential) * self.delta_p)
            self.graph.nodes[node]['potential'] = particle.potential
            
        else:
            # Case 3: Node owned by another particle
            particle.potential = particle.potential - (particle.potential - 0.05) * self.delta_p
            self.graph.nodes[node]['potential'] = current_potential * (1 - self.delta_v)
            
            # Remove from particle's owned nodes if present
            if node in particle.visited_nodes:
                particle.visited_nodes.remove(node)
                
            # Reset particle if potential too low
            if particle.potential <= 0.05:
                particle.potential = 0.05
                if free_node := self.get_free_node():
                    particle.visited_nodes.add(free_node)
                    self.graph.nodes[free_node]['owner'] = particle.id

            # Free node if potential too low
            if self.graph.nodes[node]['potential'] <= 0.05:
                self.graph.nodes[node]['owner'] = None

    def get_free_node(self) -> Optional[Any]:
        return next(
            (n for n in self.graph.nodes if self.graph.nodes[n]['owner'] is None),
            None
        )
    
    def check_positive_cluster_existence(self) -> bool:
        """
        Check if the graph has at least one positive cluster using the specified strategy.
        """
        owner_groups = defaultdict(list)
        for node in self.graph.nodes:
            if (owner := self.graph.nodes[node]['owner']) is not None:
                owner_groups[owner].append(node)
        
        return any(
            len(nodes) >= 3 and is_cluster_positive(
                self.graph,
                nodes,
                self.cluster_strategy,
                self.positive_cluster_threshold
            )
            for nodes in owner_groups.values()
        )
    

    def check_average_node_potential(self, 
                                    threshold: float = 0.9) -> bool:
        '''
        Check if the average potential of the nodes is greater than threshold
        '''
        potentials = [data['potential'] for _, data in self.graph.nodes(data=True)]
        avg = np.mean(potentials)
        return avg, avg <= threshold

    def has_unowned_nodes(self) -> bool:
        '''
        Check if the graph has any node with an owner
        '''
        return any(data['owner'] is None for _, data in self.graph.nodes(data=True))

    def graph_without_owners(self) -> bool:
        '''
        Check if the graph has any node with an owner
        '''
        self.get_dict_nodes_owner()
        return any(value is None for value in self.dict_node_owner.values())

    def get_dict_nodes_owner(self) -> Dict[Any, Optional[int]]:
        '''
        Get a dictionary with the nodes and their owners
        '''
        self.dict_node_owner: Dict[Any, Optional[int]] = {}
        for node in self.graph.nodes():
            self.dict_node_owner[node] = self.graph.nodes[node]['data'].owner        
    
    def get_average_node_potential(self) -> float:
        """Calculates the average potential of all nodes in the graph."""
        potentials = [data['potential'] for _, data in self.graph.nodes(data=True)]
        return np.mean(potentials) if potentials else 0.0

    def get_graph_coverage(self) -> Tuple[int, float]:
        """Calculates the number of owned nodes and the coverage percentage."""
        total_nodes = self.graph.number_of_nodes()
        if total_nodes == 0:
            return 0, 0.0
        
        num_owned_nodes = sum(1 for _, data in self.graph.nodes(data=True) if data['owner'] is not None)
        coverage = num_owned_nodes / total_nodes
        return num_owned_nodes, coverage

    def run_simulation(self, max_iterations=2000) -> nx.Graph:
        """Run the main simulation loop with clear stopping criteria."""
        total_nodes = self.graph.number_of_nodes()

        for iteration_count in range(1, max_iterations + 1):
            # --- Particle Movement and Update Phase ---
            for particle in self.particles:
                node = self.move_particle(particle)
                self.update_particle(particle, node)

            # --- Check Stopping Criteria at the end of the iteration ---
            # 1. Calculate current state metrics
            avg_potential = self.get_average_node_potential()
            num_owned_nodes, coverage = self.get_graph_coverage()
            positive_cluster_found = self.check_positive_cluster_existence()

            # 2. Check if thresholds are met
            potential_threshold_met = avg_potential >= self.average_node_potential_threshold
            coverage_threshold_met = coverage >= self.coverage_graph_threshold

            # --- DEBUGGING ---
            print(
                f"\rIter {iteration_count}: "
                f"Avg Potential: {avg_potential:.4f} (Goal: {self.average_node_potential_threshold}), "
                f"Coverage: {coverage:.2%} (Goal: {self.coverage_graph_threshold:.0%}), "
                f"Positive Cluster Found? {positive_cluster_found}",
                end=""
            )

            # 3. Apply stopping logic
            # The simulation can only stop IF a positive cluster has been found.
            if positive_cluster_found and potential_threshold_met and coverage_threshold_met:
                print("\n\nStopping criteria met: Positive cluster found, and potential and coverage thresholds reached.")
                break
        
        # This message will be printed if the loop finishes due to max_iterations
        else:
            print(f"\n\nStopping simulation: Maximum iterations ({max_iterations}) reached.")

        return self.graph

   # def run_simulation(self, max_iterations = 200) -> Tuple[nx.Graph, List[Particle]]:
        """Run the main simulation loop."""
        iteration_count = 0
        #max_iterations = 200 # Safety break to prevent accidental infinite loops

        while True:
            # Step 1: Evaluate each condition and store its state in a variable.
            # This makes the logic clear and easy to debug.
            
            # Condition A: Continue if the average potential is still below the threshold.
            potential_is_low = self.check_average_node_potential()[1]
            
            # Condition B: Continue if a stable positive cluster has NOT been found yet.
            no_positive_cluster = not self.check_positive_cluster_existence()
            
            # Condition C: Continue if there are still nodes that don't have an owner.
            #unowned_nodes_exist = self.has_unowned_nodes()
            # --- DEBUGGING ---
            # This print statement is now incredibly useful for seeing the state each iteration.
            print(
                f"Iter {iteration_count}: "
                f"Potential Low? {potential_is_low}, "
                f"Potential Avg: {self.check_average_node_potential()[0]}, "
                f"No Positive Cluster? {no_positive_cluster}, "
              #  f"Unowned Nodes? {unowned_nodes_exist}",
                f"Num owned nodes: {sum([1 if n[1]['owner'] else 0 for n in self.graph.nodes(data=True)])}"
            )

            # Step 2: Check if the loop should stop.
            # The loop stops ONLY when ALL THREE conditions are False.
            # Therefore, it continues if AT LEAST ONE is True.
         #   if not (potential_is_low or no_positive_cluster or unowned_nodes_exist):
          #      print("All conditions met. Stopping simulation.")
           #     break
                
            if iteration_count < max_iterations:
                for particle in self.particles:
                    node = self.move_particle(particle)
                    self.update_particle(particle, node)
                    

                iteration_count += 1
            else: 
                break         
 

        return self.graph
     
    def visualize_communities(self, G: nx.Graph) -> None:
        color_map = [self.graph.nodes[n]['owner'] for n in self.graph.nodes]
        nx.draw(self.graph, node_color=color_map, with_labels=True)
        plt.show()

# Step 2: MCLS for Reliable Negative Selection 
class MCLS:
    def __init__(
        self, 
        G: nx.Graph,
        cluster_strategy: str = 'majority',
        positive_cluster_threshold: float = 0.5,
        dissimilarity_strategy: str = 'shortest_path'
    ):
        self.graph = G
        self.cluster_strategy = cluster_strategy
        self.positive_cluster_threshold = positive_cluster_threshold
        self.dissimilarity_strategy = dissimilarity_strategy

    def assign_cluster_label(self):
        owner_groups = defaultdict(list)
        for node in self.graph.nodes:
            owner = self.graph.nodes[node]['owner']
            if owner is not None:
                owner_groups[owner].append(node)
        for owner, nodes in owner_groups.items():
            is_positive = is_cluster_positive(
                self.graph, nodes, self.cluster_strategy, self.positive_cluster_threshold
            )
            cluster_value = 1 if is_positive else 0
            for node in nodes:
                self.graph.nodes[node]['cluster_positive'] = cluster_value

    def calculate_dissimilarity(self):
        label_clusters = defaultdict(list)
        for node, data in self.graph.nodes(data=True):
            cluster_label = data.get('cluster_positive')
            if cluster_label is not None:
                label_clusters[cluster_label].append(node)
        
        positive_nodes = label_clusters.get(1, [])
        if not positive_nodes:
            print("Warning: No positive clusters found. Cannot calculate dissimilarity.")
            return

        for node_negative in label_clusters.get(0, []):
            if self.dissimilarity_strategy == 'shortest_path':
                distances = [
                    nx.shortest_path_length(self.graph, source=node_negative, target=node_positive)
                    for node_positive in positive_nodes
                ]
                if distances:
                    # The dissimilarity is the distance to the *closest* positive node
                    self.graph.nodes[node_negative]['dissimilarity'] = min(distances)
            elif self.dissimilarity_strategy == 'jaccard':
                # Jaccard similarity is |N(u) intersect N(v)| / |N(u) union N(v)|
                # Dissimilarity is 1 - similarity. We average it over all positive nodes.
                neg_neighbors = set(self.graph.neighbors(node_negative))
                dissimilarities = []
                for node_positive in positive_nodes:
                    pos_neighbors = set(self.graph.neighbors(node_positive))
                    intersection_len = len(neg_neighbors.intersection(pos_neighbors))
                    union_len = len(neg_neighbors.union(pos_neighbors))
                    if union_len == 0:
                        jaccard_sim = 0
                    else:
                        jaccard_sim = intersection_len / union_len
                    dissimilarities.append(1 - jaccard_sim)
                if dissimilarities:
                    self.graph.nodes[node_negative]['dissimilarity'] = np.mean(dissimilarities)

    def rank_nodes_dissimilarity(self, num_neg: int = 10):
        # Filter for nodes in negative clusters that have a dissimilarity score
        data = [
            (node, data.get('dissimilarity'))
            for node, data in self.graph.nodes(data=True)
            if data.get('cluster_positive') == 0 and data.get('dissimilarity') is not None
        ]
        # Rank by dissimilarity (higher is better for reliable negatives)
        return sorted(data, key=lambda x: x[1], reverse=True)[:num_neg]


# Step 3: Simple MLP Classifier
class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size=64, output_size=1):
        super(SimpleMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.layers(x)
    
class DijkstraClustering:
    """
    A baseline clustering method that assigns each node to the nearest 
    positive seed node based on shortest path distance.
    """
    def __init__(self, graph: nx.Graph):
        self.graph = graph.copy()

    def run_simulation(self) -> nx.Graph:
        """
        Executes the Dijkstra-based clustering. The "owner" of each node
        will be the ID of the closest positive node.
        """
        # Find all initial positive nodes
        positive_nodes = [
            n for n, d in self.graph.nodes(data=True) if d.get('observed_label') == 1
        ]
        
        if not positive_nodes:
            print("Warning: No positive nodes found in the graph for DijkstraClustering.")
            return self.graph

        # For each node in the graph, find the closest positive node
        for node in self.graph.nodes():
            min_dist = float('inf')
            closest_positive_node = None
            
            for pos_node in positive_nodes:
                try:
                    dist = nx.shortest_path_length(self.graph, source=node, target=pos_node)
                    if dist < min_dist:
                        min_dist = dist
                        closest_positive_node = pos_node
                except nx.NetworkXNoPath:
                    continue # Node is not reachable from this positive node
            
            # Assign the closest positive node as the "owner"
            self.graph.nodes[node]['owner'] = closest_positive_node

        return self.graph


# Main Wrapper Class for the PU Learning Algorithm
class PULearningPCC:
    def __init__(
        self,
        graph: nx.Graph,
        feature_names: List[str],
        num_neg: int = 20,
        dissimilarity_strategy: str = 'shortest_path',
        initialization_method: str = 'particle_competition',
        # Pass-through parameters for ParticleCompetitionModel
        pcm_params: Dict[str, Any] = None,
        # Pass-through parameters for MCLS
        mcls_params: Dict[str, Any] = None,
        # Parameters for the MLP classifier
        mlp_params: Dict[str, Any] = None
        
    ):
        self.graph = graph.copy()
        self.feature_names = feature_names
        self.num_neg = num_neg
        self.dissimilarity_strategy = dissimilarity_strategy
        self.initialization_method = initialization_method
        
        # Set default parameters if not provided
        self.pcm_params = pcm_params if pcm_params is not None else {}
        self.mcls_params = mcls_params if mcls_params is not None else {}
        self.mlp_params = mlp_params if mlp_params is not None else {}

    def _step1_label_initialization(self) -> nx.Graph:
        print(f"--- Step 1: Running Label Initialization using '{self.initialization_method}' method ---")
        
        if self.initialization_method == 'particle_competition':
            max_iterations = self.pcm_params.pop('max_iterations', 2000)
            pcm = ParticleCompetitionModel(self.graph, **self.pcm_params)
            return pcm.run_simulation(max_iterations=max_iterations)
        
        elif self.initialization_method == 'dijkstra':
            dijkstra_clusterer = DijkstraClustering(self.graph)
            return dijkstra_clusterer.run_simulation()
            
        else:
            raise ValueError(f"Unknown initialization_method: '{self.initialization_method}'. "
                             "Choose 'particle_competition' or 'dijkstra'.")

    def _step2_select_reliable_negatives(self, labeled_graph: nx.Graph) -> Tuple[nx.Graph, List[Any]]:
        print("\n--- Step 2: Selecting Reliable Negatives using MCLS ---")
        mcls = MCLS(labeled_graph, dissimilarity_strategy=self.dissimilarity_strategy, **self.mcls_params)
        mcls.assign_cluster_label()
        mcls.calculate_dissimilarity()
        
        ranked_nodes = mcls.rank_nodes_dissimilarity(num_neg=self.num_neg)
        reliable_negatives = [node for node, _ in ranked_nodes]
        
        print(f"Identified {len(reliable_negatives)} reliable negatives.")
        return labeled_graph, reliable_negatives
    def _step3_classification(self, graph_with_rn: nx.Graph, reliable_negatives: List[Any]) -> nx.Graph:
        print("\n--- Step 3: Training Classifier and Predicting Labels ---")
        
        # 1. Prepare unique sets of nodes for training
        positives = {n for n, d in graph_with_rn.nodes(data=True) if d['observed_label'] == 1}
        # Ensure reliable negatives are not also in the positive set
        reliable_negatives_set = {n for n in reliable_negatives if n not in positives}
        
        # Create a final, unique, and sorted list of nodes for training
        train_nodes = sorted(list(positives.union(reliable_negatives_set)))
        
        # Handle edge case where no reliable negatives were found
        if not reliable_negatives_set:
            print("Warning: No reliable negatives were identified. The classifier will be trained on positive examples only.")
        
        # Handle edge case where there is no data to train on at all
        if not train_nodes:
             print("CRITICAL WARNING: No training data (positives or reliable negatives) found. Cannot train classifier.")
             # Return the graph with only the initial positive labels marked.
             final_graph = graph_with_rn.copy()
             for node in final_graph.nodes():
                 final_graph.nodes[node]['final_label'] = final_graph.nodes[node]['observed_label']
             return final_graph

        # 2. Build feature and label lists with a single, ordered loop
        X_train_list, y_train_list = [], []
        for node in train_nodes:
            node_data = graph_with_rn.nodes[node]
            features = [node_data.get(feat, 0) for feat in self.feature_names]
            X_train_list.append(features)
            
            label = 1.0 if node in positives else 0.0
            y_train_list.append([label])

        # 3. Create Tensors and add a sanity check
        X_train = torch.tensor(X_train_list, dtype=torch.float32)
        y_train = torch.tensor(y_train_list, dtype=torch.float32)

        # THIS IS THE KEY SANITY CHECK
        if X_train.shape[0] != y_train.shape[0]:
            raise RuntimeError(
                f"FATAL: Mismatch in training data size. "
                f"Features has {X_train.shape[0]} samples, but Labels has {y_train.shape[0]} samples. "
                "This should not happen with the current logic. Please check the data."
            )

        # 4. Prepare data for prediction
        unlabeled = [
            n for n, d in graph_with_rn.nodes(data=True) 
            if n not in train_nodes
        ]
        
        # 5. Train and predict
        input_size = len(self.feature_names)
        model = SimpleMLP(input_size, **self.mlp_params)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        
        dataset = TensorDataset(X_train, y_train)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        epochs = 100
        for epoch in range(epochs):
            for inputs, labels in loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        print("Classifier training complete.")

        # 6. Assign final labels
        final_graph = graph_with_rn.copy()
        # Assign labels to the training nodes first
        for node in train_nodes:
            final_graph.nodes[node]['final_label'] = 1 if node in positives else 0
        
        # Predict on the remaining unlabeled nodes
        if unlabeled:
            X_predict = torch.tensor(
                [[d.get(feat, 0) for feat in self.feature_names] for n, d in graph_with_rn.nodes(data=True) if n in unlabeled],
                dtype=torch.float32
            )
            with torch.no_grad():
                predictions = model(X_predict)
                predicted_labels = (predictions > 0.5).float().squeeze().numpy()

            if len(unlabeled) == 1 and isinstance(predicted_labels, np.ndarray) and predicted_labels.ndim == 0:
                final_graph.nodes[unlabeled[0]]['final_label'] = int(predicted_labels.item())
            elif len(unlabeled) > 0:
                for i, node in enumerate(unlabeled):
                    final_graph.nodes[node]['final_label'] = int(predicted_labels[i])
            
        print("Final labels assigned to all nodes.")
        return final_graph
    def fit_predict(self) -> nx.Graph:
        """
        Executes the full PU learning pipeline.
        
        Returns:
            A NetworkX graph with a new 'final_label' attribute for every node.
        """
        # Step 1
        graph_labeled_by_pcm = self._step1_label_initialization()
        
        # Step 2
        graph_with_rn, reliable_negatives = self._step2_select_reliable_negatives(graph_labeled_by_pcm)
        
        # Step 3
        final_labeled_graph = self._step3_classification(graph_with_rn, reliable_negatives)
        
        return final_labeled_graph

