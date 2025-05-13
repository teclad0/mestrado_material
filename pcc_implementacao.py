import networkx as nx
import matplotlib.pyplot as plt
import random
from typing import Any, Dict, List, Optional, Tuple

class OrderedSet:
    def __init__(self) -> None:
        self.items: List[Any] = []
        self.set: set = set()  # Tracks uniqueness

    def add(self, value: Any) -> None:
        """Adds a unique value while maintaining order."""
        if value not in self.set:
            self.items.append(value)
            self.set.add(value)

    def remove_last(self) -> Any:
        """Removes and returns the last item."""
        if self.items:
            value = self.items.pop()  # Remove from the list
            self.set.remove(value)  # Remove from the set
            return value
        raise IndexError("remove_last() called on empty OrderedSet")

    def get_last(self) -> Any:
        """Returns the last item without removing it."""
        if self.items:
            return self.items[-1]
        raise IndexError("get_last() called on empty OrderedSet")

    def __contains__(self, value: Any) -> bool:
        """Check if the value exists in the set."""
        return value in self.set

    def __iter__(self):
        """Allows iteration over the items."""
        return iter(self.items)

    def __len__(self) -> int:
        """Returns the number of items."""
        return len(self.items)

    def __repr__(self) -> str:
        """String representation of the OrderedSet."""
        return f"OrderedSet({self.items})"


class Particle:
    def __init__(self, id: int) -> None:
        self.id: int = id
        self.potential: float = 0.05  # min potential
        self.visited_nodes: OrderedSet = OrderedSet() # owned nodes

class Node:
    def __init__(self) -> None:
        self.owner: Optional[int] = None  # Owner particle (0 if free)
        self.potential: float = 0.05  # Initial potential (min)

def move_particle(
    particle: Particle, 
    graph: nx.Graph, 
    p_det: float
) -> Any:
    '''
    The particle can visit a node it already visited or a random
    neighbor(position is from last visited from the last iteration)

    Args:
      particle (Particle): particle
      graph (Graph): graph
    '''
    if len(particle.visited_nodes) == 0: # Checks if it's the first iteration of the particle
        return random.choice(list(graph.nodes))
    if random.random() < p_det:  # Deterministic movement
        # Prefer to visit owned nodes
        owned_nodes = [n for n in particle.visited_nodes if graph.has_node(n)]
        if owned_nodes:
            return random.choice(owned_nodes)
    else:
        # Random movement - select from current neighbors
        current_node = particle.visited_nodes.get_last()
        neighbors = list(graph.neighbors(current_node))
        return random.choice(neighbors)

def update_particle(
    G: nx.Graph, 
    particle: Particle, 
    node: Any, 
    delta_v: float = 0.3, 
    delta_p: float = 0.4
) -> None:
    '''
    Dynamics of particles and dynamics of nodes

    Args:
      G (Graph): graph
      particle (Particle): particle
      node(Node): node the particle selected
      delta_v (float): potential changing rate of the node
      delta_p (float): potential changing rate of the particle
    '''
    if G.nodes[node]['data'].owner == None:
        # Case 1: node is free
        particle.visited_nodes.add(node) # add to particle's owned nodes
        G.nodes[node]['data'].owner = particle.id # add particle to the node's current owner
        G.nodes[node]['data'].potential = particle.potential # unowned node receives the new owner potential
    elif G.nodes[node]['data'].owner == particle.id:
        # Case 2: node it's owned by the current particle
        particle.potential = particle.potential + ((1 - particle.potential) * delta_p)  # increase potential
        particle.potential = min(1, particle.potential)
        G.nodes[node]['data'].potential = particle.potential # node receives the node's potential

    else:
        # Case 3: node it's owned by another particle
        particle.potential = particle.potential - ((particle.potential - 0.05) * delta_p)  # decrease potential
        G.nodes[node]['data'].potential = G.nodes[node]['data'].potential - delta_v  # decrease node potential
        if node in particle.visited_nodes:
            particle.visited_nodes.remove_last() # pops the particle from the node's visited nodes if the node got owned by another particle

        if particle.potential < 0.05:
            #If particle is reduced below min, it isreset to a randomly chosen node and its potential is setto the minimum level, min.
            particle.potential = 0.05
            free_node = get_free_node(G)
            particle.visited_nodes = [free_node]
            G.nodes[free_node]['data'].owner = particle.id

        if G.nodes[node]['data'].potential < 0.05:
            G.nodes[node]['data'].owner = None  # node becomes unowned

def get_free_node(G: nx.Graph) -> Optional[Any]:
    '''
    Get a node that is free for a reseted particle
    '''
    for n in G.nodes():
        if G.nodes[n]['data'].owner == None:
            return n
    return None

def run_simulation(
    G: nx.Graph, 
    num_particles: int, 
    iterations: int, 
    p_det: float = 0.6
) -> Tuple[nx.Graph, List[Particle]]:
    particles: List[Particle] = [Particle(i) for i in range(num_particles)]
    for node in G.nodes():
        G.nodes[node]['data'] = Node()  # add the custom Node object as a node attribute

    for _ in range(iterations):
        for particle in particles:
            node = move_particle(particle, G, p_det)
            update_particle(G, particle, node)

    return G, particles

def get_if_graph_without_owners(G: nx.Graph) -> bool:
    '''
    Check if the graph has any node with an owner
    '''
    for node in G.nodes():
        if G.nodes[node]['data'].owner != None:
            return False
    return True

def visualize_communities(G: nx.Graph) -> None:
    color_map: List[Optional[int]] = []
    for node in G.nodes:
        color_map.append(G.nodes[node]['data'].owner)  # Color by owner
    nx.draw(G, node_color=color_map, with_labels=True)
    plt.show()

def get_dict_nodes_owner(G: nx.Graph) -> Dict[Any, Optional[int]]:
    '''
    Get a dictionary with the nodes and their owners
    '''
    dict_nodes: Dict[Any, Optional[int]] = {}
    for node in G.nodes():
        dict_nodes[node] = G.nodes[node]['data'].owner
    return dict_nodes

def pseudo_labeling() -> None:
    '''
    1 - para cada cluster
     1 - atribuir: positivo ou negativo 
    2 - para cada ponto em um cluster negativo
     1 - calcular a distancia para os centroides de todos os clusters negativos
     2 - quantifique a dissimilaridade, distancia minima ou distancia media
    3 - rankear todos os pontos nao-rotulados pela dissimilaridade
    4 - selecionar os top num_reg
    '''