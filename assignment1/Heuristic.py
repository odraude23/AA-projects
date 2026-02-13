"""
Greedy Heuristic Algorithm for Minimum Weight Vertex Cover
Problem 16: Find approximate minimum weight vertex cover using greedy approach
"""

import networkx as nx
import time
import pickle
from GraphGenerator import load_graph


def is_vertex_cover(G, vertex_set):
    """
    Check if a set of vertices is a valid vertex cover.

    Args:
        G: NetworkX graph
        vertex_set: Set or list of vertices

    Returns:
        Boolean indicating if vertex_set is a valid vertex cover
    """
    vertex_set = set(vertex_set)

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


def greedy_vertex_cover(G):
    """
    Find approximate minimum weight vertex cover using greedy heuristic.
    Strategy: Repeatedly select the vertex with best weight-to-uncovered-edges ratio.

    Args:
        G: NetworkX graph with vertex weights

    Returns:
        Tuple of (cover, weight, operations, execution_time)
    """
    start_time = time.time()

    cover = set()
    uncovered_edges = set(G.edges())
    operations = 0

    while uncovered_edges:
        operations += 1

        # Find vertex that covers most uncovered edges with best weight ratio
        best_vertex = None
        best_ratio = float('inf')

        for vertex in G.nodes():
            if vertex in cover:
                continue

            # Count how many uncovered edges this vertex would cover
            edges_covered = 0
            for u, v in uncovered_edges:
                if u == vertex or v == vertex:
                    edges_covered += 1

            operations += len(uncovered_edges)  # Count edge checking operations

            if edges_covered == 0:
                continue

            # Calculate weight-to-coverage ratio
            vertex_weight = G.nodes[vertex]['weight']
            ratio = vertex_weight / edges_covered

            if ratio < best_ratio:
                best_ratio = ratio
                best_vertex = vertex

        # If no vertex found (shouldn't happen), break
        if best_vertex is None:
            break

        # Add best vertex to cover
        cover.add(best_vertex)
        operations += 1

        # Remove covered edges
        edges_to_remove = set()
        for u, v in uncovered_edges:
            if u == best_vertex or v == best_vertex:
                edges_to_remove.add((u, v))

        uncovered_edges -= edges_to_remove
        operations += len(edges_to_remove)

    weight = get_vertex_cover_weight(G, cover)
    execution_time = time.time() - start_time

    return cover, weight, operations, execution_time


def greedy_vertex_cover_simple(G):
    """
    Simpler greedy heuristic: Select vertex with maximum degree / weight ratio.

    Args:
        G: NetworkX graph with vertex weights

    Returns:
        Tuple of (cover, weight, operations, execution_time)
    """
    start_time = time.time()

    cover = set()
    uncovered_edges = set(G.edges())
    operations = 0

    while uncovered_edges:
        operations += 1

        # Find vertex with best degree/weight ratio among vertices incident to uncovered edges
        best_vertex = None
        best_score = -1

        # Get vertices incident to uncovered edges
        incident_vertices = set()
        for u, v in uncovered_edges:
            incident_vertices.add(u)
            incident_vertices.add(v)

        for vertex in incident_vertices:
            if vertex in cover:
                continue

            # Count degree in uncovered edges
            degree_in_uncovered = 0
            for u, v in uncovered_edges:
                if u == vertex or v == vertex:
                    degree_in_uncovered += 1

            vertex_weight = G.nodes[vertex]['weight']
            score = degree_in_uncovered / vertex_weight  # Higher is better

            operations += len(uncovered_edges)

            if score > best_score:
                best_score = score
                best_vertex = vertex

        if best_vertex is None:
            break

        # Add to cover
        cover.add(best_vertex)

        # Remove covered edges
        edges_to_remove = set()
        for u, v in uncovered_edges:
            if u == best_vertex or v == best_vertex:
                edges_to_remove.add((u, v))

        uncovered_edges -= edges_to_remove
        operations += len(edges_to_remove)

    weight = get_vertex_cover_weight(G, cover)
    execution_time = time.time() - start_time

    return cover, weight, operations, execution_time


def greedy_edge_selection(G):
    """
    Edge-based greedy heuristic: Select both endpoints of edge with minimum total weight.

    Args:
        G: NetworkX graph with vertex weights

    Returns:
        Tuple of (cover, weight, operations, execution_time)
    """
    start_time = time.time()

    cover = set()
    uncovered_edges = set(G.edges())
    operations = 0

    while uncovered_edges:
        operations += 1

        # Find edge with minimum total weight of endpoints (not already in cover)
        best_edge = None
        best_weight = float('inf')

        for u, v in uncovered_edges:
            # Calculate weight of adding both vertices (if not already in cover)
            edge_weight = 0
            if u not in cover:
                edge_weight += G.nodes[u]['weight']
            if v not in cover:
                edge_weight += G.nodes[v]['weight']

            operations += 1

            if edge_weight < best_weight:
                best_weight = edge_weight
                best_edge = (u, v)

        if best_edge is None:
            break

        # Add both endpoints to cover
        u, v = best_edge
        cover.add(u)
        cover.add(v)

        # Remove all covered edges
        edges_to_remove = set()
        for u_edge, v_edge in uncovered_edges:
            if u_edge in cover or v_edge in cover:
                edges_to_remove.add((u_edge, v_edge))

        uncovered_edges -= edges_to_remove
        operations += len(edges_to_remove)

    weight = get_vertex_cover_weight(G, cover)
    execution_time = time.time() - start_time

    return cover, weight, operations, execution_time


def test_algorithm(graph_file):
    """
    Test all greedy heuristics on a single graph.

    Args:
        graph_file: Path to graph file

    Returns:
        Dictionary with results from all heuristics
    """
    print(f"\nTesting graph: {graph_file}")

    # Load graph
    G = load_graph(graph_file)
    n = G.number_of_nodes()
    m = G.number_of_edges()

    print(f"Vertices: {n}, Edges: {m}")

    results = {}

    # Test greedy weight-to-coverage ratio
    print("\n  Greedy (weight/coverage ratio):")
    cover1, weight1, ops1, time1 = greedy_vertex_cover(G)
    print(f"    Cover size: {len(cover1)}, Weight: {weight1}")
    print(f"    Operations: {ops1}, Time: {time1:.6f}s")
    results['greedy_ratio'] = {
        'cover': cover1,
        'weight': weight1,
        'cover_size': len(cover1),
        'operations': ops1,
        'time': time1
    }

    # Test greedy degree/weight
    print("\n  Greedy (degree/weight):")
    cover2, weight2, ops2, time2 = greedy_vertex_cover_simple(G)
    print(f"    Cover size: {len(cover2)}, Weight: {weight2}")
    print(f"    Operations: {ops2}, Time: {time2:.6f}s")
    results['greedy_degree'] = {
        'cover': cover2,
        'weight': weight2,
        'cover_size': len(cover2),
        'operations': ops2,
        'time': time2
    }

    # Test edge-based greedy
    print("\n  Greedy (edge selection):")
    cover3, weight3, ops3, time3 = greedy_edge_selection(G)
    print(f"    Cover size: {len(cover3)}, Weight: {weight3}")
    print(f"    Operations: {ops3}, Time: {time3:.6f}s")
    results['greedy_edge'] = {
        'cover': cover3,
        'weight': weight3,
        'cover_size': len(cover3),
        'operations': ops3,
        'time': time3
    }

    # Find best heuristic
    best_weight = min(weight1, weight2, weight3)
    print(f"\n  Best heuristic weight: {best_weight}")

    return {
        'vertices': n,
        'edges': m,
        'results': results,
        'best_weight': best_weight
    }


def main():
    """Test greedy heuristics on sample graphs."""
    import os

    print("Greedy Heuristics for Minimum Weight Vertex Cover")
    print("=" * 60)

    # Test on sample graphs
    test_graphs = [
        "Grafos/graph_n4_d12.pkl",
        "Grafos/graph_n4_d25.pkl",
        "Grafos/graph_n5_d12.pkl",
        "Grafos/graph_n6_d25.pkl"
    ]

    for graph_file in test_graphs:
        if os.path.exists(graph_file):
            test_algorithm(graph_file)
        else:
            print(f"Graph file not found: {graph_file}")
            print("Run GraphGenerator.py first to generate graphs.")

    print("\n" + "=" * 60)
    print("Testing complete!")


if __name__ == "__main__":
    main()
