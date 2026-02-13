"""
Visualization tools for Assignment 2 - Randomized Algorithms for Vertex Cover
Generates charts and graphs for performance analysis and report
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import networkx as nx
from graph_utils import read_graph_file


# Output directory
OUTPUT_DIR = Path("Outputs")
FIGURES_DIR = OUTPUT_DIR / "Figures"

# Create directories if they don't exist
OUTPUT_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# Plot styling
plt.style.use('seaborn-v0_8-darkgrid')
COLORS = {
    'SA': '#1f77b4',  # Blue
    'GA': '#ff7f0e',  # Orange
    'GRASP': '#2ca02c'  # Green (if needed)
}


def visualize_graph_solution(graph_file: str, solution: List[int], algorithm_name: str,
                            weight: float, output_file: str):
    """
    Visualize a graph with its vertex cover solution.
    Similar to assignment1's graph_nX_dY_optimal.png

    Args:
        graph_file: Path to graph file
        solution: List of vertex indices in the cover
        algorithm_name: Name of algorithm used
        weight: Weight of the solution
        output_file: Output file path
    """
    from graph_utils import read_graph_file

    # Read graph
    graph = read_graph_file(graph_file)

    # Create NetworkX graph for visualization
    G = nx.Graph()
    G.add_nodes_from(range(graph.n_vertices))
    G.add_edges_from(graph.edges)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Layout
    pos = nx.spring_layout(G, seed=42, k=2, iterations=50)

    # Node colors (red for cover, lightblue for not in cover)
    node_colors = ['#ff6b6b' if i in solution else '#a8d8f0' for i in range(graph.n_vertices)]

    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=2, ax=ax)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                          node_size=800, alpha=0.9, ax=ax)

    # Draw labels with weights
    labels = {i: f"{i}\n({graph.vertex_weights[i]:.1f})" for i in range(graph.n_vertices)}
    nx.draw_networkx_labels(G, pos, labels, font_size=9, font_weight='bold', ax=ax)

    # Title and legend
    graph_name = Path(graph_file).stem
    ax.set_title(f'{graph_name} - {algorithm_name} Solution\n'
                f'Vertices: {graph.n_vertices}, Edges: {len(graph.edges)}\n'
                f'Cover Size: {len(solution)}, Weight: {weight:.2f}',
                fontsize=14, fontweight='bold')

    # Legend
    red_patch = mpatches.Patch(color='#ff6b6b', label='In Vertex Cover')
    blue_patch = mpatches.Patch(color='#a8d8f0', label='Not in Cover')
    ax.legend(handles=[red_patch, blue_patch], loc='upper right', fontsize=11)

    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_file}")


def plot_time_vs_vertices(results_by_graph: Dict[str, Dict[str, Any]],
                         algorithms: List[str] = ['SA', 'GA']):
    """
    Plot execution time vs number of vertices for each algorithm.
    Shows scalability.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    for alg in algorithms:
        # Extract data
        data_points = []

        for graph_name, results in results_by_graph.items():
            if alg in results:
                n_vertices = results[alg].get('vertices', 0)
                time = results[alg].get('execution_time', 0)

                if n_vertices > 0:
                    data_points.append((n_vertices, time))

        if data_points:
            # Sort by vertices to ensure proper line connection
            data_points.sort(key=lambda x: x[0])
            vertices = [x[0] for x in data_points]
            times = [x[1] for x in data_points]

            # Plot
            ax.plot(vertices, times, 'o-', label=alg, color=COLORS[alg],
                   linewidth=2, markersize=8, alpha=0.8)

    ax.set_xlabel('Number of Vertices', fontsize=12, fontweight='bold')
    ax.set_ylabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Execution Time vs Graph Size', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')  # Log scale for better visualization

    plt.tight_layout()
    output_file = FIGURES_DIR / 'time_vs_vertices.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_file}")


def plot_operations_vs_vertices(results_by_graph: Dict[str, Dict[str, Any]],
                               algorithms: List[str] = ['SA', 'GA']):
    """
    Plot basic operations vs number of vertices.
    Validates computational complexity.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    for alg in algorithms:
        data_points = []

        for graph_name, results in results_by_graph.items():
            if alg in results:
                n_vertices = results[alg].get('vertices', 0)
                ops = results[alg].get('basic_operations', 0)

                if n_vertices > 0:
                    data_points.append((n_vertices, ops))

        if data_points:
            # Sort by vertices
            data_points.sort(key=lambda x: x[0])
            vertices = [x[0] for x in data_points]
            operations = [x[1] for x in data_points]

            ax.plot(vertices, operations, 'o-', label=alg, color=COLORS[alg],
                   linewidth=2, markersize=8, alpha=0.8)

    ax.set_xlabel('Number of Vertices', fontsize=12, fontweight='bold')
    ax.set_ylabel('Basic Operations', fontsize=12, fontweight='bold')
    ax.set_title('Basic Operations vs Graph Size', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()
    output_file = FIGURES_DIR / 'operations_vs_vertices.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_file}")


def plot_solutions_tested_vs_vertices(results_by_graph: Dict[str, Dict[str, Any]],
                                      algorithms: List[str] = ['SA', 'GA']):
    """
    Plot number of unique solutions tested vs vertices.
    Shows exploration strategy.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    for alg in algorithms:
        data_points = []

        for graph_name, results in results_by_graph.items():
            if alg in results:
                n_vertices = results[alg].get('vertices', 0)
                sols = results[alg].get('solutions_tested', 0)

                if n_vertices > 0:
                    data_points.append((n_vertices, sols))

        if data_points:
            # Sort by vertices
            data_points.sort(key=lambda x: x[0])
            vertices = [x[0] for x in data_points]
            solutions = [x[1] for x in data_points]

            ax.plot(vertices, solutions, 'o-', label=alg, color=COLORS[alg],
                   linewidth=2, markersize=8, alpha=0.8)

    ax.set_xlabel('Number of Vertices', fontsize=12, fontweight='bold')
    ax.set_ylabel('Solutions Tested (unique)', fontsize=12, fontweight='bold')
    ax.set_title('Solution Space Exploration vs Graph Size', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = FIGURES_DIR / 'solutions_tested_vs_vertices.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_file}")


def plot_operations_vs_time(results_by_graph: Dict[str, Dict[str, Any]],
                           algorithms: List[str] = ['SA', 'GA']):
    """
    Plot operations vs time to show algorithm efficiency.
    Higher is better (more work per second).
    Creates separate plots for each algorithm.
    """
    for alg in algorithms:
        fig, ax = plt.subplots(figsize=(10, 8))

        data_points = []

        for graph_name, results in results_by_graph.items():
            if alg in results:
                time = results[alg].get('execution_time', 0)
                ops = results[alg].get('basic_operations', 0)
                n_vertices = results[alg].get('vertices', 0)

                if time > 0 and ops > 0:
                    data_points.append((time, ops, n_vertices))

        # Sort by vertices for consistent coloring
        data_points.sort(key=lambda x: x[2])
        times = [x[0] for x in data_points]
        operations = [x[1] for x in data_points]
        vertices_list = [x[2] for x in data_points]

        if times:
            scatter = ax.scatter(times, operations, c=vertices_list,
                               cmap='viridis', s=150, alpha=0.7)
            ax.set_xlabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Basic Operations', fontsize=12, fontweight='bold')
            ax.set_title(f'{alg}: Operations vs Time', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xscale('log')
            ax.set_yscale('log')

            # Colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Vertices', fontsize=11)

            plt.tight_layout()
            output_file = FIGURES_DIR / f'operations_vs_time_{alg}.png'
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"  Saved: {output_file}")


def plot_performance_comparison(results_by_graph: Dict[str, Dict[str, Any]],
                               algorithms: List[str] = ['SA', 'GA']):
    """
    Side-by-side comparison of key metrics.
    Creates 4 separate plots.
    """
    # Prepare data - collect all graphs that have both algorithms
    graph_data = []

    for graph_name, results in results_by_graph.items():
        # Only include if both algorithms have results
        if all(alg in results for alg in algorithms):
            # Get vertex count for sorting
            n_vertices = results[algorithms[0]].get('vertices', 0)
            graph_data.append((n_vertices, graph_name, results))

    # Sort by vertex count
    graph_data.sort(key=lambda x: x[0])

    # Extract sorted data
    graph_names = []
    graph_display_names = []
    data = {alg: {'time': [], 'weight': [], 'size': []} for alg in algorithms}

    for n_vertices, graph_name, results in graph_data:
        graph_names.append(graph_name)
        # Extract just the filename without path
        display_name = Path(graph_name).stem
        graph_display_names.append(display_name)

        for alg in algorithms:
            data[alg]['time'].append(results[alg].get('execution_time', 0))
            data[alg]['weight'].append(results[alg].get('solution_weight', 0))
            data[alg]['size'].append(results[alg].get('solution_size', 0))

    if not graph_names:
        print("  No common graphs for comparison")
        return

    x = np.arange(len(graph_display_names))
    width = 0.35

    # 1. Execution Time Comparison
    fig, ax = plt.subplots(figsize=(12, 8))
    for i, alg in enumerate(algorithms):
        ax.bar(x + i*width, data[alg]['time'], width,
              label=alg, color=COLORS[alg], alpha=0.8)
    ax.set_ylabel('Time (s)', fontsize=12, fontweight='bold')
    ax.set_title('Execution Time Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width/2)
    ax.set_xticklabels(graph_display_names, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    output_file = FIGURES_DIR / 'comparison_execution_time.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_file}")

    # 2. Solution Weight Comparison
    fig, ax = plt.subplots(figsize=(12, 8))
    for i, alg in enumerate(algorithms):
        ax.bar(x + i*width, data[alg]['weight'], width,
              label=alg, color=COLORS[alg], alpha=0.8)
    ax.set_ylabel('Solution Weight', fontsize=12, fontweight='bold')
    ax.set_title('Solution Quality Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width/2)
    ax.set_xticklabels(graph_display_names, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    output_file = FIGURES_DIR / 'comparison_solution_quality.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_file}")

    # 3. Solution Size Comparison
    fig, ax = plt.subplots(figsize=(12, 8))
    for i, alg in enumerate(algorithms):
        ax.bar(x + i*width, data[alg]['size'], width,
              label=alg, color=COLORS[alg], alpha=0.8)
    ax.set_ylabel('Cover Size (vertices)', fontsize=12, fontweight='bold')
    ax.set_title('Solution Size Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width/2)
    ax.set_xticklabels(graph_display_names, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    output_file = FIGURES_DIR / 'comparison_solution_size.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_file}")

    # 4. Quality Gap Comparison
    fig, ax = plt.subplots(figsize=(12, 8))
    best_weights = [min(data[alg]['weight'][i] for alg in algorithms)
                   for i in range(len(graph_display_names))]

    for i, alg in enumerate(algorithms):
        quality_gap = [(data[alg]['weight'][j] - best_weights[j]) / best_weights[j] * 100
                      if best_weights[j] > 0 else 0
                      for j in range(len(graph_display_names))]
        ax.bar(x + i*width, quality_gap, width,
              label=alg, color=COLORS[alg], alpha=0.8)
    ax.set_ylabel('Gap from Best (%)', fontsize=12, fontweight='bold')
    ax.set_title('Solution Quality Gap', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width/2)
    ax.set_xticklabels(graph_display_names, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    plt.tight_layout()
    output_file = FIGURES_DIR / 'comparison_quality_gap.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_file}")


def plot_complexity_validation(results_by_graph: Dict[str, Dict[str, Any]],
                               algorithms: List[str] = ['SA', 'GA']):
    """
    Compare experimental complexity with theoretical predictions.
    For SA: Expected O(I * n) where I = iterations
    For GA: Expected O(G * P * n) where G = generations, P = population
    Creates separate plots for each algorithm.
    """
    for alg in algorithms:
        fig, ax = plt.subplots(figsize=(10, 8))

        data_points = []

        for graph_name, results in results_by_graph.items():
            if alg in results:
                n = results[alg].get('vertices', 0)
                ops = results[alg].get('basic_operations', 0)
                iters = results[alg].get('iterations', 0)

                if n > 0 and ops > 0 and iters > 0:
                    data_points.append((n, ops, iters))

        # Sort by vertices
        data_points.sort(key=lambda x: x[0])
        vertices = [x[0] for x in data_points]
        operations = [x[1] for x in data_points]
        iterations = [x[2] for x in data_points]

        if vertices:
            # Theoretical prediction
            if alg == 'SA':
                # O(I * n) - operations per iteration * iterations
                # SA does neighbor evaluation for each iteration
                theoretical = [vertices[i] * iterations[i] for i in range(len(vertices))]
            else:  # GA
                # O(G * P * n) where G = generations, P = population
                # GA evaluates population for each generation
                # Assuming population size ~ 100
                theoretical = [vertices[i] * iterations[i] * 100 for i in range(len(vertices))]

            # Plot both on same scale (no normalization)
            ax.plot(vertices, operations, 'o-', label=f'{alg} Experimental',
                   color=COLORS[alg], linewidth=2.5, markersize=10)
            ax.plot(vertices, theoretical, 's--', label=f'{alg} Theoretical',
                   color=COLORS[alg], alpha=0.6, linewidth=2, markersize=8)

            ax.set_xlabel('Number of Vertices', fontsize=12, fontweight='bold')
            ax.set_ylabel('Operations', fontsize=12, fontweight='bold')
            ax.set_title(f'{alg}: Theoretical vs Experimental Complexity',
                        fontsize=14, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')

            plt.tight_layout()
            output_file = FIGURES_DIR / f'complexity_validation_{alg}.png'
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"  Saved: {output_file}")


def plot_algorithm_efficiency(results_by_graph: Dict[str, Dict[str, Any]],
                             algorithms: List[str] = ['SA', 'GA']):
    """
    Plot operations per second for each algorithm.
    Shows computational efficiency.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    for alg in algorithms:
        data_points = []

        for graph_name, results in results_by_graph.items():
            if alg in results:
                n = results[alg].get('vertices', 0)
                ops = results[alg].get('basic_operations', 0)
                time = results[alg].get('execution_time', 0)

                if time > 0 and n > 0:
                    ops_per_sec = ops / time
                    data_points.append((n, ops_per_sec))

        if data_points:
            # Sort by vertices
            data_points.sort(key=lambda x: x[0])
            vertices = [x[0] for x in data_points]
            efficiency = [x[1] for x in data_points]

            ax.plot(vertices, efficiency, 'o-', label=alg, color=COLORS[alg],
                   linewidth=2, markersize=8, alpha=0.8)

    ax.set_xlabel('Number of Vertices', fontsize=12, fontweight='bold')
    ax.set_ylabel('Operations per Second', fontsize=12, fontweight='bold')
    ax.set_title('Algorithm Efficiency (Higher is Better)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()
    output_file = FIGURES_DIR / 'algorithm_efficiency.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_file}")


def generate_all_visualizations(results_file: str, graph_files: List[str] = None):
    """
    Generate all visualizations from results file.

    Args:
        results_file: Path to JSON results file
        graph_files: Optional list of specific graphs to visualize
    """
    print("="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)

    # Load results
    print(f"\nLoading results from {results_file}...")
    with open(results_file, 'r') as f:
        all_results = json.load(f)

    # Organize results by graph
    results_by_graph = {}

    for result in all_results:
        graph_name = result.get('graph_file', 'unknown')
        alg_name = result.get('algorithm', 'unknown')

        # Map algorithm names
        if 'Simulated Annealing' in alg_name:
            alg_key = 'SA'
        elif 'Genetic' in alg_name:
            alg_key = 'GA'
        elif 'GRASP' in alg_name:
            alg_key = 'GRASP'
        else:
            alg_key = alg_name

        if graph_name not in results_by_graph:
            results_by_graph[graph_name] = {}

        results_by_graph[graph_name][alg_key] = result

    print(f"Loaded results for {len(results_by_graph)} graphs")

    # Generate aggregate charts
    print("\nGenerating aggregate performance charts...")
    algorithms = ['SA', 'GA']  # Exclude GRASP

    plot_time_vs_vertices(results_by_graph, algorithms)
    plot_operations_vs_vertices(results_by_graph, algorithms)
    plot_solutions_tested_vs_vertices(results_by_graph, algorithms)
    plot_operations_vs_time(results_by_graph, algorithms)
    plot_performance_comparison(results_by_graph, algorithms)
    plot_complexity_validation(results_by_graph, algorithms)
    plot_algorithm_efficiency(results_by_graph, algorithms)

    # Generate individual graph visualizations
    if graph_files:
        print("\nGenerating individual graph visualizations...")
        for graph_file in graph_files:
            graph_name = Path(graph_file).stem

            if graph_name in results_by_graph or graph_file in results_by_graph:
                key = graph_name if graph_name in results_by_graph else graph_file

                # Generate for each algorithm
                for alg_key in ['SA', 'GA']:
                    if alg_key in results_by_graph[key]:
                        result = results_by_graph[key][alg_key]
                        solution = result.get('solution', [])
                        weight = result.get('solution_weight', 0)

                        output_file = FIGURES_DIR / f'{graph_name}_{alg_key}_solution.png'
                        visualize_graph_solution(graph_file, solution, alg_key,
                                               weight, str(output_file))

    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE")
    print("="*70)
    print(f"\nAll figures saved to: {FIGURES_DIR.absolute()}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python3 visualization.py <results.json> [graph_file1 graph_file2 ...]")
        print("\nExample:")
        print("  python3 visualization.py results.json")
        print("  python3 visualization.py results.json SW_ALGUNS_GRAFOS/SWtinyEWG.txt")
        sys.exit(1)

    results_file = sys.argv[1]
    graph_files = sys.argv[2:] if len(sys.argv) > 2 else None

    generate_all_visualizations(results_file, graph_files)
