"""
Graph utilities for reading Sedgewick & Wayne format graph files
and handling vertex cover operations.
"""

import random
from typing import List, Set, Tuple, Dict


class Graph:
    """Graph representation for Minimum Weight Vertex Cover problem."""

    def __init__(self, n_vertices: int):
        """
        Initialize graph with n vertices.

        Args:
            n_vertices: Number of vertices in the graph
        """
        self.n_vertices = n_vertices
        self.edges: List[Tuple[int, int]] = []
        self.adj_list: Dict[int, Set[int]] = {i: set() for i in range(n_vertices)}
        self.vertex_weights: List[float] = [1.0] * n_vertices  # Default weight 1.0
        self.is_directed = False
        self.has_edge_weights = False

    def add_edge(self, u: int, v: int, weight: float = None):
        """
        Add an edge to the graph.

        Args:
            u: First vertex
            v: Second vertex
            weight: Edge weight (optional, not used for vertex cover)
        """
        # Skip self-loops as per requirements
        if u == v:
            return

        # Add edge to list if not already present
        edge = (min(u, v), max(u, v)) if not self.is_directed else (u, v)
        if edge not in self.edges:
            self.edges.append(edge)

        # Update adjacency list
        self.adj_list[u].add(v)
        if not self.is_directed:
            self.adj_list[v].add(u)

    def set_vertex_weights(self, weights: List[float]):
        """Set vertex weights."""
        if len(weights) == self.n_vertices:
            self.vertex_weights = weights.copy()

    def is_vertex_cover(self, cover: Set[int]) -> bool:
        """
        Check if a set of vertices is a valid vertex cover.

        Args:
            cover: Set of vertex indices

        Returns:
            True if cover is valid, False otherwise
        """
        for u, v in self.edges:
            if u not in cover and v not in cover:
                return False
        return True

    def get_cover_weight(self, cover: Set[int]) -> float:
        """
        Calculate total weight of a vertex cover.

        Args:
            cover: Set of vertex indices

        Returns:
            Sum of weights of vertices in cover
        """
        return sum(self.vertex_weights[v] for v in cover)

    def get_uncovered_edges(self, cover: Set[int]) -> List[Tuple[int, int]]:
        """
        Get list of edges not covered by the given vertex set.

        Args:
            cover: Set of vertex indices

        Returns:
            List of uncovered edges
        """
        uncovered = []
        for u, v in self.edges:
            if u not in cover and v not in cover:
                uncovered.append((u, v))
        return uncovered


def read_graph_file(filename: str) -> Graph:
    """
    Read graph from Sedgewick & Wayne format file.

    Format:
        Line 1: 0/1 (is directed?)
        Line 2: 0/1 (has edge weights?)
        Line 3: number of vertices
        Line 4: number of edges
        Line 5+: vertex_i vertex_j [weight]

    Args:
        filename: Path to graph file

    Returns:
        Graph object
    """
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    is_directed = int(lines[0]) == 1
    has_edge_weights = int(lines[1]) == 1
    n_vertices = int(lines[2])
    n_edges = int(lines[3])

    graph = Graph(n_vertices)
    graph.is_directed = is_directed
    graph.has_edge_weights = has_edge_weights

    # Read edges
    for i in range(4, min(4 + n_edges, len(lines))):
        parts = lines[i].split()
        if len(parts) >= 2:
            u = int(parts[0])
            v = int(parts[1])
            edge_weight = float(parts[2]) if len(parts) > 2 and has_edge_weights else None
            graph.add_edge(u, v, edge_weight)

    return graph


def generate_random_weights(graph: Graph, min_weight: float = 1.0, max_weight: float = 10.0, seed: int = None):
    """
    Generate random weights for vertices.

    Args:
        graph: Graph object
        min_weight: Minimum vertex weight
        max_weight: Maximum vertex weight
        seed: Random seed for reproducibility
    """
    if seed is not None:
        random.seed(seed)

    weights = [random.uniform(min_weight, max_weight) for _ in range(graph.n_vertices)]
    graph.set_vertex_weights(weights)


def greedy_vertex_cover(graph: Graph) -> Set[int]:
    """
    Simple greedy algorithm to get an initial vertex cover.

    Args:
        graph: Graph object

    Returns:
        Set of vertices forming a cover
    """
    cover = set()
    uncovered_edges = graph.edges.copy()

    while uncovered_edges:
        # Find vertex covering most uncovered edges with best weight ratio
        best_vertex = None
        best_ratio = float('inf')

        for v in range(graph.n_vertices):
            if v in cover:
                continue

            # Count how many uncovered edges this vertex covers
            covered_count = sum(1 for u, w in uncovered_edges if v == u or v == w)

            if covered_count > 0:
                ratio = graph.vertex_weights[v] / covered_count
                if ratio < best_ratio:
                    best_ratio = ratio
                    best_vertex = v

        if best_vertex is not None:
            cover.add(best_vertex)
            # Remove covered edges
            uncovered_edges = [(u, v) for u, v in uncovered_edges
                             if u != best_vertex and v != best_vertex]
        else:
            break

    return cover
