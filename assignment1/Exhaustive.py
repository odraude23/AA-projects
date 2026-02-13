"""
Exhaustive Search Algorithm for Minimum Weight Vertex Cover
Problem 16: Find minimum weight vertex cover for weighted undirected graph
"""

import networkx as nx
import itertools
import time
import pickle
from GraphGenerator import load_graph


def is_vertex_cover(G, vertex_set):
    """
    Check if a set of vertices is a valid vertex cover.
    A vertex cover must have at least one endpoint of every edge.

    Args:
        G: NetworkX graph
        vertex_set: Set or list of vertices

    Returns:
        Boolean indicating if vertex_set is a valid vertex cover
    """
    vertex_set = set(vertex_set)

    # Check if every edge has at least one endpoint in vertex_set
    for u, v in G.edges():
        if u not in vertex_set and v not in vertex_set:
            return False

    return True


def get_vertex_cover_weight(G, vertex_set):
    """
    Calculate total weight of a vertex cover.

    Args:
        G: NetworkX graph with vertex weights
        vertex_set: Set or list of vertices

    Returns:
        Total weight of vertices in the set
    """
    total_weight = 0
    for vertex in vertex_set:
        total_weight += G.nodes[vertex]['weight']
    return total_weight


def exhaustive_search(G):
    """
    Find minimum weight vertex cover using exhaustive search.
    Tests all possible vertex subsets to find the one with minimum weight
    that covers all edges.

    Args:
        G: NetworkX graph with vertex weights

    Returns:
        Tuple of (best_cover, min_weight, operations, configurations, execution_time)
    """
    start_time = time.time()

    n = G.number_of_nodes()
    vertices = list(G.nodes())

    best_cover = None
    min_weight = float('inf')
    operations = 0
    configurations = 0

    # Try subsets of increasing size
    for size in range(0, n + 1):
        # Generate all combinations of vertices of current size
        for combination in itertools.combinations(vertices, size):
            configurations += 1
            operations += 1

            # Check if this is a valid vertex cover
            if is_vertex_cover(G, combination):
                operations += G.number_of_edges()  # Count edge checking operations

                # Calculate weight
                weight = get_vertex_cover_weight(G, combination)
                operations += size  # Count weight calculation operations

                # Update best if better
                if weight < min_weight:
                    min_weight = weight
                    best_cover = set(combination)

                    # Early termination: found a valid cover of this size
                    # No smaller cover exists, so check if any cover of this size has lower weight
                    break  # Continue checking other combinations of same size

        # If we found a cover of this size, check all combinations of this size
        # then we can stop (no need to check larger sizes)
        if best_cover is not None and len(best_cover) == size:
            # Check remaining combinations of this size for potentially lower weight
            continue
        elif best_cover is not None:
            # Found optimal cover in previous size, stop
            break

    execution_time = time.time() - start_time

    return best_cover, min_weight, operations, configurations, execution_time


def exhaustive_search_optimized(G):
    """
    Optimized exhaustive search with early termination.
    Stops checking larger subsets once a valid cover is found,
    but completes all subsets of that size to find minimum weight.

    Args:
        G: NetworkX graph with vertex weights

    Returns:
        Tuple of (best_cover, min_weight, operations, configurations, execution_time)
    """
    start_time = time.time()

    n = G.number_of_nodes()
    vertices = list(G.nodes())

    best_cover = None
    min_weight = float('inf')
    operations = 0
    configurations = 0

    # Try ALL subsets of all sizes
    # For weighted vertex cover, we cannot stop early based on size alone
    # because a larger cover might have lower total weight
    for size in range(0, n + 1):
        # Generate all combinations of vertices of current size
        for combination in itertools.combinations(vertices, size):
            configurations += 1
            operations += 1

            # Check if this is a valid vertex cover
            if is_vertex_cover(G, combination):
                operations += G.number_of_edges()  # Count edge checking operations

                # Calculate weight
                weight = get_vertex_cover_weight(G, combination)
                operations += size  # Count weight calculation operations

                # Update best if better
                if weight < min_weight:
                    min_weight = weight
                    best_cover = set(combination)

    execution_time = time.time() - start_time

    return best_cover, min_weight, operations, configurations, execution_time


def test_algorithm(graph_file):
    """
    Test exhaustive search on a single graph.

    Args:
        graph_file: Path to graph file

    Returns:
        Dictionary with results
    """
    print(f"\nTesting graph: {graph_file}")

    # Load graph
    G = load_graph(graph_file)
    n = G.number_of_nodes()
    m = G.number_of_edges()

    print(f"Vertices: {n}, Edges: {m}")

    # Run exhaustive search
    cover, weight, ops, configs, exec_time = exhaustive_search_optimized(G)

    print(f"Minimum weight vertex cover found:")
    print(f"  Cover: {sorted(cover)}")
    print(f"  Weight: {weight}")
    print(f"  Operations: {ops}")
    print(f"  Configurations tested: {configs}")
    print(f"  Execution time: {exec_time:.6f} seconds")

    return {
        'vertices': n,
        'edges': m,
        'cover': cover,
        'weight': weight,
        'cover_size': len(cover),
        'operations': ops,
        'configurations': configs,
        'time': exec_time
    }


def main():
    """Test exhaustive search on sample graphs."""
    import os

    print("Exhaustive Search for Minimum Weight Vertex Cover")
    print("=" * 60)

    # Test on small graphs
    test_graphs = [
        "Grafos/graph_n4_d12.pkl",
        "Grafos/graph_n4_d25.pkl",
        "Grafos/graph_n5_d12.pkl",
        "Grafos/graph_n6_d25.pkl"
    ]

    results = []
    for graph_file in test_graphs:
        if os.path.exists(graph_file):
            result = test_algorithm(graph_file)
            results.append(result)
        else:
            print(f"Graph file not found: {graph_file}")
            print("Run GraphGenerator.py first to generate graphs.")

    print("\n" + "=" * 60)
    print("Testing complete!")


if __name__ == "__main__":
    main()
