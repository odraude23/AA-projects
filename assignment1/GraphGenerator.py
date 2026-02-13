"""
Graph Generator for Minimum Weight Vertex Cover Problem
Generates random graphs with specified vertex counts and edge densities
"""

import networkx as nx
import random
import pickle
import os


STUDENT_NUMBER = 103070


def generate_random_coordinates(n, min_coord=1, max_coord=500, min_distance=10):
    """
    Generate random 2D coordinates for n vertices.
    Ensures no vertices are coincident or too close.

    Args:
        n: Number of vertices
        min_coord: Minimum coordinate value
        max_coord: Maximum coordinate value
        min_distance: Minimum distance between vertices

    Returns:
        Dictionary mapping vertex IDs to (x, y) coordinates
    """
    coordinates = {}
    attempts = 0
    max_attempts = 10000

    for vertex in range(n):
        while attempts < max_attempts:
            x = random.randint(min_coord, max_coord)
            y = random.randint(min_coord, max_coord)

            # Check if too close to existing vertices
            too_close = False
            for existing_coord in coordinates.values():
                distance = ((x - existing_coord[0])**2 + (y - existing_coord[1])**2)**0.5
                if distance < min_distance:
                    too_close = True
                    break

            if not too_close:
                coordinates[vertex] = (x, y)
                break

            attempts += 1

        if attempts >= max_attempts:
            raise ValueError(f"Could not generate valid coordinates for {n} vertices")

    return coordinates


def generate_random_weights(n, min_weight=1, max_weight=100):
    """
    Generate random positive weights for n vertices.

    Args:
        n: Number of vertices
        min_weight: Minimum vertex weight
        max_weight: Maximum vertex weight

    Returns:
        Dictionary mapping vertex IDs to weights
    """
    weights = {}
    for vertex in range(n):
        weights[vertex] = random.randint(min_weight, max_weight)
    return weights


def generate_graph(n, edge_density, seed=None):
    """
    Generate a random undirected graph with weighted vertices.

    Args:
        n: Number of vertices
        edge_density: Percentage of maximum possible edges (e.g., 0.125, 0.25, 0.5, 0.75)
        seed: Random seed for reproducibility

    Returns:
        NetworkX Graph with vertex weights and coordinates
    """
    if seed is not None:
        random.seed(seed)

    # Calculate probability for edge creation
    # For n vertices, max edges = n(n-1)/2
    # We want edge_density * max_edges
    # Using Erdős-Rényi model: G(n, p)
    p = edge_density

    # Generate random graph
    G = nx.fast_gnp_random_graph(n, p, seed=seed)

    # Generate and assign vertex weights
    weights = generate_random_weights(n)
    nx.set_node_attributes(G, weights, 'weight')

    # Generate and assign coordinates
    coordinates = generate_random_coordinates(n)
    nx.set_node_attributes(G, coordinates, 'pos')

    return G


def save_graph(G, filename):
    """
    Save graph to file using pickle.

    Args:
        G: NetworkX graph
        filename: Output filename
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(G, f)


def load_graph(filename):
    """
    Load graph from file.

    Args:
        filename: Input filename

    Returns:
        NetworkX graph
    """
    with open(filename, 'rb') as f:
        return pickle.load(f)


def save_graph_text(G, filename):
    """
    Save graph to text file in a human-readable format.

    Args:
        G: NetworkX graph
        filename: Output filename
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        # Write metadata
        f.write(f"0\n")  # Undirected
        f.write(f"1\n")  # Weighted vertices
        f.write(f"{G.number_of_nodes()}\n")
        f.write(f"{G.number_of_edges()}\n")

        # Write vertex weights
        f.write("# Vertex weights\n")
        for node in G.nodes():
            weight = G.nodes[node]['weight']
            f.write(f"{node} {weight}\n")

        # Write edges
        f.write("# Edges\n")
        for u, v in G.edges():
            f.write(f"{u} {v}\n")


def generate_all_graphs(min_vertices=4, max_vertices=15, densities=None, seed=STUDENT_NUMBER):
    """
    Generate all graph instances for experiments.

    Args:
        min_vertices: Minimum number of vertices
        max_vertices: Maximum number of vertices
        densities: List of edge densities (default: [0.125, 0.25, 0.5, 0.75])
        seed: Random seed

    Returns:
        Dictionary of generated graphs
    """
    if densities is None:
        densities = [0.125, 0.25, 0.5, 0.75]

    graphs = {}

    for n in range(min_vertices, max_vertices + 1):
        for density in densities:
            # Create unique seed for each graph
            graph_seed = seed + n * 1000 + int(density * 100)

            G = generate_graph(n, density, seed=graph_seed)

            # Store in dictionary
            key = (n, density)
            graphs[key] = G

            # Save to files
            pickle_filename = f"Grafos/graph_n{n}_d{int(density*100)}.pkl"
            text_filename = f"Grafos/graph_n{n}_d{int(density*100)}.txt"

            save_graph(G, pickle_filename)
            save_graph_text(G, text_filename)

            print(f"Generated graph: n={n}, density={density}, edges={G.number_of_edges()}")

    return graphs


def main():
    """Generate all test graphs."""
    print("Generating graphs for Minimum Weight Vertex Cover experiments...")
    print(f"Random seed: {STUDENT_NUMBER}")
    print()

    graphs = generate_all_graphs(
        min_vertices=4,
        max_vertices=15,  # Start conservative for exhaustive search
        densities=[0.125, 0.25, 0.5, 0.75],
        seed=STUDENT_NUMBER
    )

    print()
    print(f"Generated {len(graphs)} graphs total")
    print("Graphs saved to Grafos/ directory")


if __name__ == "__main__":
    main()
