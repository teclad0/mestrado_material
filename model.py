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
        initialization_strategy: str = 'random'
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

    def run_simulation(self, max_iterations = 200) -> Tuple[nx.Graph, List[Particle]]:
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

class MCLS:
    """
    Class to handle the MCLS algorithm for clustering nodes in a graph.
    """
    
    def __init__(
        self, 
        G: nx.Graph,
        cluster_strategy: str = 'majority',
        positive_cluster_threshold: float = 0.5
    ):
        self.graph = G
        self.cluster_strategy = cluster_strategy
        self.positive_cluster_threshold = positive_cluster_threshold

    def assign_cluster_label(self):
        '''
        Assign a feature based on majority observed_label within nodes sharing the same owner
        
        Args:
            G: Input graph with node attributes
            feature_name: Name of the new feature to create
        
        Returns:
            Modified graph with owner-cluster-based features
        '''
        # Step 1: Group nodes by their owner
        owner_groups = defaultdict(list)
        for node in self.graph.nodes:
            owner = self.graph.nodes[node]['data'].owner
            owner_groups[owner].append(node)
        # Process each owner cluster
        for owner, nodes in owner_groups.items():
            # --- USE THE HELPER FUNCTION ---
            is_positive = is_cluster_positive(
                self.graph, 
                nodes, 
                self.cluster_strategy, 
                self.positive_cluster_threshold
            )
            
            cluster_value = 1 if is_positive else 0
            
            # Assign feature to all nodes in this owner cluster
            for node in nodes:
                self.graph.nodes[node]['cluster_positive'] = cluster_value


    def calculate_dissimilarity(self) -> nx.Graph:
        '''
        Assign a feature based on majority observed_label within nodes sharing the same owner
        
        Args:
            G: Input graph with node attributes
        
        Returns:
            Modified graph with owner-cluster-based features
        '''
        # Get the nodes from positive clusters
        label_clusters = defaultdict(list)
        for node, cluster in list(self.graph.nodes(data='cluster_positive')):
            label_clusters[cluster].append(node)
        #import ipdb; ipdb.set_trace()
        for node_negative in label_clusters[0]:
            self.graph.nodes[node_negative]['dissimilarity'] = []
            for node_positive in label_clusters[1]:
                distance = nx.shortest_path_length(self.graph, source=node_negative, target=node_positive)
                self.graph.nodes[node_negative]['dissimilarity'].append(distance)
            self.graph.nodes[node_negative]['dissimilarity'] = max(self.graph.nodes[node_negative]['dissimilarity']) 
       

    def rank_nodes_dissimilarity(self, num_reg: int = 10):
        '''
        Rank nodes based on dissimilarity
        '''
        data = list(self.graph.nodes(data='dissimilarity'))
        return sorted(data, key=lambda x: float('-inf') if x[1] is None else x[1], reverse=True)[:num_reg]








def check_cluster_label(arr):
    '''
    Check if cluster is positive
    '''
    count_ones = np.count_nonzero(arr == 1)
    count_zeros = np.count_nonzero(arr == 0)
    return 1 if count_ones > count_zeros else 0


