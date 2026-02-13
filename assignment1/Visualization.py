"""
Visualization Module for Minimum Weight Vertex Cover
Generates plots and visualizations for graphs and results
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import pickle
import os
import numpy as np
from GraphGenerator import load_graph


def visualize_graph_with_cover(G, vertex_cover=None, title="Graph", output_file=None):
    """
    Visualize a graph with optional vertex cover highlighted.

    Args:
        G: NetworkX graph with 'pos' and 'weight' attributes
        vertex_cover: Set of vertices in the cover (optional)
        title: Plot title
        output_file: Save to file if provided
    """
    plt.figure(figsize=(12, 8))

    # Get positions
    pos = nx.get_node_attributes(G, 'pos')

    # If no positions, generate layout
    if not pos:
        pos = nx.spring_layout(G, seed=42)

    # Get weights for labels
    weights = nx.get_node_attributes(G, 'weight')

    # Color nodes based on whether they're in cover
    if vertex_cover:
        node_colors = ['red' if node in vertex_cover else 'lightblue' for node in G.nodes()]
    else:
        node_colors = ['lightblue' for _ in G.nodes()]

    # Draw graph
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=2)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=700, alpha=0.9)

    # Draw labels with weights
    labels = {node: f"{node}\n({weights[node]})" for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=10)

    plt.title(title, fontsize=14, fontweight='bold')

    # Add legend
    if vertex_cover:
        cover_weight = sum(weights[v] for v in vertex_cover)
        red_patch = mpatches.Patch(color='red', label=f'Vertex Cover (n={len(vertex_cover)}, w={cover_weight})')
        blue_patch = mpatches.Patch(color='lightblue', label='Not in Cover')
        plt.legend(handles=[red_patch, blue_patch], loc='upper right')

    plt.axis('off')
    plt.tight_layout()

    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {output_file}")

    plt.close()


def plot_performance_comparison(results, output_dir=None):
    """
    Plot execution time comparison between exhaustive and heuristic algorithms.
    Creates separate images for each density.

    Args:
        results: List of experiment results
        output_dir: Directory to save files if provided
    """
    densities = [0.125, 0.25, 0.5, 0.75]

    for density in densities:
        fig, ax = plt.subplots(figsize=(10, 7))

        # Filter results for this density
        density_results = [r for r in results if r['density'] == density]

        if not density_results:
            plt.close()
            continue

        # Extract data
        vertices = [r['vertices'] for r in density_results]

        # Exhaustive times
        ex_times = []
        for r in density_results:
            if 'exhaustive' in r and 'time' in r['exhaustive']:
                ex_times.append(r['exhaustive']['time'])
            else:
                ex_times.append(None)

        # Heuristic times (use best heuristic)
        h_times = []
        for r in density_results:
            best_time = float('inf')
            for h_key in ['greedy_ratio', 'greedy_degree', 'greedy_edge']:
                if h_key in r and 'time' in r[h_key]:
                    best_time = min(best_time, r[h_key]['time'])
            h_times.append(best_time if best_time < float('inf') else None)

        # Plot
        if any(t is not None for t in ex_times):
            valid_vertices = [v for v, t in zip(vertices, ex_times) if t is not None]
            valid_ex_times = [t for t in ex_times if t is not None]
            ax.plot(valid_vertices, valid_ex_times, 'o-', label='Exhaustive', linewidth=2, markersize=8)

        if any(t is not None for t in h_times):
            valid_vertices_h = [v for v, t in zip(vertices, h_times) if t is not None]
            valid_h_times = [t for t in h_times if t is not None]
            ax.plot(valid_vertices_h, valid_h_times, 's-', label='Best Heuristic', linewidth=2, markersize=8)

        ax.set_xlabel('Number of Vertices', fontsize=12)
        ax.set_ylabel('Execution Time (seconds)', fontsize=12)
        ax.set_title(f'Execution Time Comparison - Density = {density}', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        plt.tight_layout()

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_file = f"{output_dir}/performance_comparison_d{int(density*1000)}.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved performance plot to {output_file}")

        plt.close()


def plot_configurations_vs_vertices(results, output_dir=None):
    """
    Plot number of configurations tested vs number of vertices.
    Creates separate images for each density.

    Args:
        results: List of experiment results
        output_dir: Directory to save files if provided
    """
    densities = [0.125, 0.25, 0.5, 0.75]

    for density in densities:
        fig, ax = plt.subplots(figsize=(10, 7))

        # Filter results for this density
        density_results = [r for r in results if r['density'] == density]

        if not density_results:
            plt.close()
            continue

        # Extract data
        vertices = []
        configurations = []

        for r in density_results:
            if 'exhaustive' in r and 'configurations' in r['exhaustive']:
                vertices.append(r['vertices'])
                configurations.append(r['exhaustive']['configurations'])

        if vertices:
            ax.plot(vertices, configurations, 'o-', linewidth=2, markersize=8, color='darkblue')

        ax.set_xlabel('Number of Vertices', fontsize=12)
        ax.set_ylabel('Configurations Tested', fontsize=12)
        ax.set_title(f'Configurations Tested - Density = {density}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        plt.tight_layout()

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_file = f"{output_dir}/configurations_vs_vertices_d{int(density*1000)}.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved configurations plot to {output_file}")

        plt.close()


def plot_heuristic_precision(results, output_dir=None):
    """
    Plot heuristic precision (optimal/heuristic * 100).
    Creates separate images for each density.

    Args:
        results: List of experiment results
        output_dir: Directory to save files if provided
    """
    densities = [0.125, 0.25, 0.5, 0.75]
    heuristic_names = ['greedy_ratio', 'greedy_degree', 'greedy_edge']
    labels = ['Weight/Coverage', 'Degree/Weight', 'Edge Selection']
    colors = ['red', 'green', 'blue']

    # Use a scale that positions 100% higher on the y-axis
    # This emphasizes precision values relative to optimal
    y_min = 0
    y_max = 120  # Makes 100% appear at ~83% height of the graph

    for density in densities:
        fig, ax = plt.subplots(figsize=(10, 7))

        # Filter results for this density
        density_results = [r for r in results if r['density'] == density]

        if not density_results:
            plt.close()
            continue

        # Plot each heuristic
        for h_name, label, color in zip(heuristic_names, labels, colors):
            vertices = []
            precisions = []

            for r in density_results:
                if h_name in r and 'precision' in r[h_name]:
                    vertices.append(r['vertices'])
                    precision = r[h_name]['precision']
                    if precision != float('inf'):
                        precisions.append(precision)
                    else:
                        precisions.append(None)  # Skip inf values

            if vertices:
                ax.plot(vertices, precisions, 'o-', label=label, linewidth=2,
                       markersize=8, color=color, alpha=0.7)

        ax.set_xlabel('Number of Vertices', fontsize=12)
        ax.set_ylabel('Precision (%)', fontsize=12)
        ax.set_title(f'Heuristic Precision - Density = {density}', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([y_min, y_max])
        ax.axhline(y=100, color='black', linestyle='--', linewidth=2, alpha=0.5, label='Optimal (100%)')

        plt.tight_layout()

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_file = f"{output_dir}/heuristic_precision_d{int(density*1000)}.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved precision plot to {output_file}")

        plt.close()


def plot_operations_vs_time(results, output_dir=None):
    """
    Plot operations count vs execution time to verify complexity.
    Creates separate images for exhaustive and heuristic algorithms.

    Args:
        results: List of experiment results
        output_dir: Directory to save files if provided
    """
    # Exhaustive search
    ex_ops = []
    ex_times = []

    for r in results:
        if 'exhaustive' in r and 'operations' in r['exhaustive'] and 'time' in r['exhaustive']:
            ex_ops.append(r['exhaustive']['operations'])
            ex_times.append(r['exhaustive']['time'])

    if ex_ops:
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.scatter(ex_ops, ex_times, alpha=0.6, s=100)
        ax.set_xlabel('Operations', fontsize=12)
        ax.set_ylabel('Execution Time (seconds)', fontsize=12)
        ax.set_title('Exhaustive Search: Operations vs Time', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        ax.set_yscale('log')

        plt.tight_layout()

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_file = f"{output_dir}/operations_vs_time_exhaustive.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved operations vs time plot (exhaustive) to {output_file}")

        plt.close()

    # Best heuristic
    h_ops = []
    h_times = []

    for r in results:
        best_ops = None
        best_time = None
        for h_key in ['greedy_ratio', 'greedy_degree', 'greedy_edge']:
            if h_key in r and 'operations' in r[h_key] and 'time' in r[h_key]:
                if best_ops is None or r[h_key]['operations'] < best_ops:
                    best_ops = r[h_key]['operations']
                    best_time = r[h_key]['time']

        if best_ops is not None:
            h_ops.append(best_ops)
            h_times.append(best_time)

    if h_ops:
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.scatter(h_ops, h_times, alpha=0.6, s=100, color='orange')
        ax.set_xlabel('Operations', fontsize=12)
        ax.set_ylabel('Execution Time (seconds)', fontsize=12)
        ax.set_title('Best Heuristic: Operations vs Time', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        ax.set_yscale('log')

        plt.tight_layout()

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_file = f"{output_dir}/operations_vs_time_heuristic.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved operations vs time plot (heuristic) to {output_file}")

        plt.close()


def generate_all_visualizations(results_file, output_dir="Outputs/Figures"):
    """
    Generate all visualizations from experiment results.

    Args:
        results_file: Path to results pickle file
        output_dir: Output directory for figures
    """
    print("Generating visualizations...")

    # Load results
    with open(results_file, 'rb') as f:
        results = pickle.load(f)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Generate performance comparison (creates 4 separate images)
    print("Generating performance comparison plots...")
    plot_performance_comparison(results, output_dir)

    # Generate configurations plot (creates 4 separate images)
    print("Generating configurations vs vertices plots...")
    plot_configurations_vs_vertices(results, output_dir)

    # Generate precision plot (creates 4 separate images)
    print("Generating heuristic precision plots...")
    plot_heuristic_precision(results, output_dir)

    # Generate operations vs time (creates 2 separate images)
    print("Generating operations vs time plots...")
    plot_operations_vs_time(results, output_dir)

    # Visualize sample graphs with solutions
    print("\nGenerating sample graph visualizations...")
    sample_results = results[:6]  # First 6 graphs

    for idx, result in enumerate(sample_results):
        G = load_graph(result['graph_file'])

        # Visualize with exhaustive solution
        if 'exhaustive' in result and 'cover' in result['exhaustive']:
            cover = result['exhaustive']['cover']
            n = result['vertices']
            d = result['density']
            visualize_graph_with_cover(
                G, cover,
                title=f"Optimal Cover (n={n}, density={d})",
                output_file=f"{output_dir}/graph_n{n}_d{int(d*100)}_optimal.png"
            )

    print("\nAll visualizations generated!")


def main():
    """Generate visualizations from experiment results."""
    import sys

    if len(sys.argv) > 1:
        results_file = sys.argv[1]
    else:
        # Try to find most recent results file
        results_files = [f"Outputs/{f}" for f in os.listdir("Outputs")
                        if f.startswith("Results_") and f.endswith(".pkl")]

        if not results_files:
            print("No results files found in Outputs/")
            print("Run Experiments.py first to generate results.")
            return

        results_file = max(results_files, key=os.path.getmtime)
        print(f"Using results file: {results_file}")

    generate_all_visualizations(results_file)


if __name__ == "__main__":
    main()
