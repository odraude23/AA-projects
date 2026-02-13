"""
Convert directed EWD graphs to undirected graphs.
Removes duplicate edges that occur due to bidirectional edges in directed graphs.
"""

import os
from pathlib import Path


def convert_to_undirected(input_file: str, output_file: str):
    """
    Convert a directed graph file to undirected.

    Args:
        input_file: Path to input directed graph
        output_file: Path to output undirected graph
    """
    with open(input_file, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    # Parse header
    is_directed = int(lines[0])
    has_edge_weights = int(lines[1])
    n_vertices = int(lines[2])
    n_edges = int(lines[3])

    print(f"Converting {input_file}:")
    print(f"  Original: {n_vertices} vertices, {n_edges} edges, directed={is_directed}")

    # Read edges and remove duplicates for undirected graph
    edges_set = set()
    edges_list = []

    for i in range(4, min(4 + n_edges, len(lines))):
        parts = lines[i].split()
        if len(parts) >= 2:
            u = int(parts[0])
            v = int(parts[1])
            weight = parts[2] if len(parts) > 2 and has_edge_weights else None

            # Skip self-loops
            if u == v:
                continue

            # For undirected, treat (u,v) and (v,u) as same edge
            edge_key = (min(u, v), max(u, v))

            if edge_key not in edges_set:
                edges_set.add(edge_key)
                edges_list.append((u, v, weight))

    # Write output file
    with open(output_file, 'w') as f:
        # Header: make it undirected (0)
        f.write("0\n")  # NOT directed
        f.write(f"{has_edge_weights}\n")
        f.write(f"{n_vertices}\n")
        f.write(f"{len(edges_list)}\n")

        # Write edges
        for u, v, weight in edges_list:
            if weight is not None:
                f.write(f"{u} {v} {weight}\n")
            else:
                f.write(f"{u} {v}\n")

    print(f"  Converted: {n_vertices} vertices, {len(edges_list)} edges (undirected)")
    print(f"  Saved to: {output_file}")
    print(f"  Removed {n_edges - len(edges_list)} duplicate/self-loop edges\n")


def main():
    """Convert all EWD files to undirected versions."""

    ewd_files = [
        "SW_ALGUNS_GRAFOS/SWtinyEWD.txt",
        "SW_ALGUNS_GRAFOS/SWmediumEWD.txt",
        "SW_ALGUNS_GRAFOS/SW1000EWD.txt",
        "SW_ALGUNS_GRAFOS/SW10000EWD.txt"
    ]

    print("="*70)
    print("Converting EWD (Directed) Graphs to Undirected Graphs")
    print("="*70)
    print()

    for input_file in ewd_files:
        if not os.path.exists(input_file):
            print(f"File not found: {input_file}")
            continue

        # Create output filename
        base_name = os.path.basename(input_file)
        # Replace EWD with EWG (Edge-Weighted Graph, undirected)
        output_name = base_name.replace("EWD", "EWG")
        output_file = os.path.join(os.path.dirname(input_file), output_name)

        convert_to_undirected(input_file, output_file)

    print("="*70)
    print("Conversion Complete!")
    print("="*70)
    print("\nNew undirected graph files created:")
    for input_file in ewd_files:
        output_name = os.path.basename(input_file).replace("EWD", "EWG")
        output_path = os.path.join(os.path.dirname(input_file), output_name)
        if os.path.exists(output_path):
            print(f"  ✓ {output_path}")


if __name__ == "__main__":
    main()
