import networkx as nx
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict, Counter

class OrderedSet:
    """Maintains insertion order while ensuring uniqueness"""
    def __init__(self) -> None:
        self.items: List[Any] = []
        self.set: set = set()

    def add(self, value: Any) -> None:
        if value not in self.set:
            self.items.append(value)
            self.set.add(value)

    def remove_last(self) -> Any:
        if not self.items:
            raise IndexError("remove_last() called on empty OrderedSet")
        value = self.items.pop()
        self.set.remove(value)
        return value

    def get_last(self) -> Any:
        if not self.items:
            raise IndexError("get_last() called on empty OrderedSet")
        return self.items[-1]

    def __contains__(self, value: Any) -> bool:
        return value in self.set

    def __iter__(self):
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def __repr__(self) -> str:
        return f"OrderedSet({self.items})"


class Particle:
    """Represents a particle in the competition model"""
    def __init__(self, id: int) -> None:
        self.id: int = id
        self.potential: float = 0.05  # min potential
        self.visited_nodes: OrderedSet = OrderedSet()  # owned nodes


class Node:
    """Represents a node in the graph"""
    def __init__(self) -> None:
        self.owner: Optional[int] = None  # Owner particle
        self.potential: float = 0.05  # Initial potential