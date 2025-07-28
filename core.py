import networkx as nx
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict, Counter

class OrderedSet:
    def __init__(self) -> None:
        self.items: List[Any] = []
        self.item_set: set = set()

    def add(self, value: Any) -> None:
        if value not in self.item_set:
            self.items.append(value)
            self.item_set.add(value)

    def remove(self, value: Any) -> None:
        if value in self.item_set:
            self.items.remove(value)
            self.item_set.remove(value)

    def get_last(self) -> Any:
        return self.items[-1] if self.items else None

    def __contains__(self, value: Any) -> bool:
        return value in self.item_set

    def __iter__(self):
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def __repr__(self) -> str:
        return f"OrderedSet({self.items})"


class Particle:
    def __init__(self, id: int) -> None:
        self.id: int = id
        self.potential: float = 0.05
        self.visited_nodes: OrderedSet = OrderedSet()

    @property
    def current_position(self) -> Any:
        """Current position is always the last visited node"""
        return self.visited_nodes.get_last()
    
    @property
    def node_visited_last_iteration(self):
        if len(self.visited_nodes) > 1:
            return self.visited_nodes.items[-2]
        # returns a node that doesn't exist 
        return -1
