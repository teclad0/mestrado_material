import networkx as nx
import matplotlib.pyplot as plt
import random
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas
from collections import defaultdict, Counter
from core import Particle, Node

class ParticleCompetitionModel:
    """Main class implementing the particle competition algorithm"""
    
    def __init__(
        self, 
        graph: nx.Graph, 
        num_particles: int,
        p_det: float = 0.6,
        delta_p: float = 0.4,
        delta_v: float = 0.3
    ):
        self.graph = graph
        self.num_particles = num_particles
        self.p_det = p_det
        self.delta_p = delta_p
        self.delta_v = delta_v
        self.particles: List[Particle] = []
        
        # Initialize graph nodes
        for node in self.graph.nodes:
            self.graph.nodes[node]['data'] = Node()
        
        # Initialize particles
        self.initialize_particles()

        # Run the simulation
        #self.graph_populated = self._run_simulation()
    
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
        if len(particle.visited_nodes) == 0: # Checks if it's the first iteration of the particle
            return random.choice(list(self.graph.nodes))
        if random.random() < self.p_det:  # Deterministic movement
            # Prefer to visit owned nodes
            owned_nodes = [n for n in particle.visited_nodes if self.graph.has_node(n)]
            if owned_nodes:
                return random.choice(owned_nodes)
        else:
            # Random movement - select from current neighbors
            current_node = particle.visited_nodes.get_last()
            neighbors = list(self.graph.neighbors(current_node))
            return random.choice(neighbors)

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
        if self.graph.nodes[node]['data'].owner == None:
            # Case 1: node is free
            particle.visited_nodes.add(node) # add to particle's owned nodes
            self.graph.nodes[node]['data'].owner = particle.id # add particle to the node's current owner
            self.graph.nodes[node]['data'].potential = particle.potential # unowned node receives the new owner potential
        elif self.graph.nodes[node]['data'].owner == particle.id:
            # Case 2: node it's owned by the current particle
            particle.potential = particle.potential + ((1 - particle.potential) * self.delta_p)  # increase potential
            particle.potential = min(1, particle.potential)
            self.graph.nodes[node]['data'].potential = particle.potential # node receives the node's potential

        else:
            # Case 3: node it's owned by another particle
            particle.potential = particle.potential - ((particle.potential - 0.05) * self.delta_p)  # decrease potential
            self.graph.nodes[node]['data'].potential = self.graph.nodes[node]['data'].potential - self.delta_v  # decrease node potential
            if node in particle.visited_nodes:
                particle.visited_nodes.remove_last() # pops the particle from the node's visited nodes if the node got owned by another particle

            if particle.potential < 0.05:
                #If particle is reduced below min, it is reset to a randomly chosen node and 
                # its potential is setto the minimum level, min.
                particle.potential = 0.05
                free_node = self.get_free_node(self.graph)
                particle.visited_nodes = [free_node]
                self.graph.nodes[free_node]['data'].owner = particle.id

            if self.graph.nodes[node]['data'].potential < 0.05:
                self.graph.nodes[node]['data'].owner = None  # node becomes unowned

    def get_free_node(self) -> Optional[Any]:
        '''
        Get a node that is free for a reseted particle
        '''
        for n in self.graph.nodes():
            if self.graph[n]['data'].owner == None:
                return n
        return None
    
    #[] TODO: rever essa estrategia
    def check_positive_cluster_existence(self) -> bool:
        '''
        Check if the graph has at least one positive cluster
        '''
        owner_groups = defaultdict(list)
        for node in self.graph.nodes:
            owner = self.graph.nodes[node]['data'].owner
            owner_groups[owner].append(node)
        
        # Step 2: Process each owner cluster
        for owner, nodes in owner_groups.items():
            if len(nodes) >=3: # Only consider clusters with at least 3 nodes
                # Collect observed labels for this owner's nodes
                observed_labels = [self.graph.nodes[n]['observed_label'] for n in nodes]
            
                # Calculate majority label (prefer 1 in case of tie)
                label_counts = Counter(observed_labels)
                majority_label = max(label_counts, 
                                key=lambda k: (label_counts[k], k))  # (count, label value)
            
                # Assign feature to all nodes in this owner cluster
                if majority_label == 1:
                    # If a positive cluster is found, return True
                    return True

        # If no positive cluster is found, return False
        return False
    

    def check_average_node_potential(self, 
                                    threshold: float = 0.9) -> bool:
        '''
        Check if the average potential of the nodes is greater than 0.5
        '''
        avg_potential = []
        for i in range(len(self.graph.nodes())):
            node_potential = self.graph.nodes(data=True)[i]['data'].potential
            avg_potential.append(node_potential)
        avg_potential = np.mean(avg_potential)
        return avg_potential, avg_potential <= threshold

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
            unowned_nodes_exist = self.graph_without_owners()
            # --- DEBUGGING ---
            # This print statement is now incredibly useful for seeing the state each iteration.
            # print(
            #     f"Iter {iteration_count}: "
            #     f"Potential Low? {potential_is_low}, "
            #     f"No Positive Cluster? {no_positive_cluster}, "
            #     f"Unowned Nodes? {unowned_nodes_exist}"
            # )

            # Step 2: Check if the loop should stop.
            # The loop stops ONLY when ALL THREE conditions are False.
            # Therefore, it continues if AT LEAST ONE is True.
            if not (potential_is_low or no_positive_cluster or unowned_nodes_exist):
                print("All conditions met. Stopping simulation.")
                break
                
            # Safety break
            # if iteration_count >= max_iterations:
            #     print(f"Max iterations ({max_iterations}) reached. Stopping.")
            #     break

            # Step 3: If not stopping, run the simulation logic for one iteration.
            for particle in self.particles:
                node = self.move_particle(particle)
                self.update_particle(particle, node)
            
            iteration_count += 1

        return self.graph



    # def run_simulation(self) -> Tuple[nx.Graph, List[Particle]]:
    #     """Run the main simulation loop"""
    #     print("teste", self.check_average_node_potential()[1], self._check_positive_cluster_existence(), self.graph_without_owners())

    #     while self.check_average_node_potential()[1] and \
    #         not self._check_positive_cluster_existence() and \
    #             self.graph_without_owners():

    #         print("teste2", self.check_average_node_potential()[1], self._check_positive_cluster_existence())
    #         for particle in self.particles:
    #             node = self._move_particle(particle)
    #             self._update_particle(particle, node)

    #     return self.graph

    
     
    def visualize_communities(self, G: nx.Graph) -> None:
        color_map: List[Optional[int]] = []
        for node in G.nodes:
            color_map.append(G.nodes[node]['data'].owner)  # Color by owner
        nx.draw(G, node_color=color_map, with_labels=True)
        plt.show()


# def get_optimal_K( G: nx.Graph, 
#     K: List[int], 
#     p_det: float = 0.6,
#     delta_p: float = 0.4,
#     delta_v: float = 0.3):

#     potential_list = []
#     for k in tqdm(range(2, K, 2)):
#         G, _ = run_simulation(G, k, p_det=p_det, delta_p=delta_p, delta_v=delta_v)
#         potential_list.append(check_average_node_potential(G)[0])
#     return potential_list


class MCLS:
    """
    Class to handle the MCLS algorithm for clustering nodes in a graph.
    """
    
    def __init__(self, G: nx.Graph):
        self.graph = G

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
        # Step 2: Process each owner cluster
        for owner, nodes in owner_groups.items():
            # Collect observed labels for this owner's nodes
            observed_labels = [self.graph.nodes[n]['observed_label'] for n in nodes]
            # Calculate majority label (prefer 1 in case of tie)
            label_counts = Counter(observed_labels)
            majority_label = max(label_counts, 
                            key=lambda k: (label_counts[k], k))  # (count, label value)
            
            # Assign feature to all nodes in this owner cluster
            cluster_value = 1 if majority_label == 1 else 0
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


