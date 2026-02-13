"""
Experimental Runner for Minimum Weight Vertex Cover
Runs both exhaustive and heuristic algorithms on all generated graphs
Collects and analyzes performance metrics
"""

import os
import time
import pickle
from GraphGenerator import load_graph
from Exhaustive import exhaustive_search_optimized, is_vertex_cover, get_vertex_cover_weight
from Heuristic import greedy_vertex_cover, greedy_vertex_cover_simple, greedy_edge_selection


STUDENT_NUMBER = 103070


def run_experiments(min_vertices=4, max_vertices=15, densities=None, timeout=300):
    """
    Run all experiments systematically.

    Args:
        min_vertices: Minimum number of vertices
        max_vertices: Maximum number of vertices
        densities: List of edge densities
        timeout: Maximum time per experiment in seconds

    Returns:
        List of experiment results
    """
    if densities is None:
        densities = [0.125, 0.25, 0.5, 0.75]

    results = []

    print("Running Experiments for Minimum Weight Vertex Cover")
    print("=" * 70)
    print()

    for n in range(min_vertices, max_vertices + 1):
        for density in densities:
            # Load graph
            graph_file = f"Grafos/graph_n{n}_d{int(density*100)}.pkl"

            if not os.path.exists(graph_file):
                print(f"Graph not found: {graph_file}")
                continue

            G = load_graph(graph_file)
            m = G.number_of_edges()

            print(f"Graph: n={n}, density={density}, edges={m}")

            # Initialize result dictionary
            result = {
                'vertices': n,
                'edges': m,
                'density': density,
                'graph_file': graph_file
            }

            # Run exhaustive search (with timeout protection)
            print("  Running exhaustive search...")
            try:
                start_time = time.time()
                cover_ex, weight_ex, ops_ex, configs_ex, time_ex = exhaustive_search_optimized(G)

                if time.time() - start_time > timeout:
                    print(f"    TIMEOUT (>{timeout}s)")
                    result['exhaustive'] = {'timeout': True}
                else:
                    result['exhaustive'] = {
                        'cover': cover_ex,
                        'weight': weight_ex,
                        'cover_size': len(cover_ex),
                        'operations': ops_ex,
                        'configurations': configs_ex,
                        'time': time_ex,
                        'timeout': False
                    }
                    print(f"    Weight: {weight_ex}, Size: {len(cover_ex)}, " +
                          f"Configs: {configs_ex}, Time: {time_ex:.6f}s")

            except Exception as e:
                print(f"    ERROR: {e}")
                result['exhaustive'] = {'error': str(e)}

            # Run greedy heuristics
            heuristics = [
                ('greedy_ratio', greedy_vertex_cover, 'Weight/Coverage Ratio'),
                ('greedy_degree', greedy_vertex_cover_simple, 'Degree/Weight'),
                ('greedy_edge', greedy_edge_selection, 'Edge Selection')
            ]

            for heuristic_name, heuristic_func, description in heuristics:
                print(f"  Running {description} heuristic...")
                try:
                    cover_h, weight_h, ops_h, time_h = heuristic_func(G)

                    result[heuristic_name] = {
                        'cover': cover_h,
                        'weight': weight_h,
                        'cover_size': len(cover_h),
                        'operations': ops_h,
                        'time': time_h
                    }
                    print(f"    Weight: {weight_h}, Size: {len(cover_h)}, Time: {time_h:.6f}s")

                    # Calculate precision if exhaustive succeeded
                    if 'exhaustive' in result and 'weight' in result['exhaustive']:
                        optimal_weight = result['exhaustive']['weight']
                        # Precision: 100% = optimal, <100% = worse (quality metric)
                        precision = (optimal_weight / weight_h) * 100 if weight_h > 0 else 0
                        # Approximation ratio: 1.0 = optimal, >1.0 = worse
                        approximation_ratio = weight_h / optimal_weight if optimal_weight > 0 else float('inf')
                        result[heuristic_name]['precision'] = precision
                        result[heuristic_name]['approximation_ratio'] = approximation_ratio
                        print(f"    Precision: {precision:.2f}%, Ratio: {approximation_ratio:.2f}")

                except Exception as e:
                    print(f"    ERROR: {e}")
                    result[heuristic_name] = {'error': str(e)}

            results.append(result)
            print()

    return results


def save_results(results, filename):
    """
    Save experimental results to file.

    Args:
        results: List of result dictionaries
        filename: Output filename
    """
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)

    # Save as pickle for later analysis
    pickle_file = filename.replace('.txt', '.pkl')
    with open(pickle_file, 'wb') as f:
        pickle.dump(results, f)

    # Save as text file for human reading
    with open(filename, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("Minimum Weight Vertex Cover - Experimental Results\n")
        f.write(f"Student Number: {STUDENT_NUMBER}\n")
        f.write("=" * 70 + "\n\n")

        for result in results:
            f.write(f"Graph: n={result['vertices']}, m={result['edges']}, " +
                   f"density={result['density']}\n")
            f.write("-" * 70 + "\n")

            # Exhaustive search results
            if 'exhaustive' in result:
                ex = result['exhaustive']
                if 'timeout' in ex and ex['timeout']:
                    f.write(f"Exhaustive Search: TIMEOUT\n")
                elif 'error' in ex:
                    f.write(f"Exhaustive Search: ERROR - {ex['error']}\n")
                else:
                    f.write(f"Exhaustive Search:\n")
                    f.write(f"  Weight: {ex['weight']}\n")
                    f.write(f"  Cover Size: {ex['cover_size']}\n")
                    f.write(f"  Operations: {ex['operations']}\n")
                    f.write(f"  Configurations: {ex['configurations']}\n")
                    f.write(f"  Time: {ex['time']:.6f} seconds\n")
                    f.write(f"  Cover: {sorted(ex['cover'])}\n")

            # Heuristic results
            heuristic_names = [
                ('greedy_ratio', 'Greedy (Weight/Coverage)'),
                ('greedy_degree', 'Greedy (Degree/Weight)'),
                ('greedy_edge', 'Greedy (Edge Selection)')
            ]

            for h_key, h_name in heuristic_names:
                if h_key in result:
                    h = result[h_key]
                    if 'error' in h:
                        f.write(f"{h_name}: ERROR - {h['error']}\n")
                    else:
                        f.write(f"\n{h_name}:\n")
                        f.write(f"  Weight: {h['weight']}\n")
                        f.write(f"  Cover Size: {h['cover_size']}\n")
                        f.write(f"  Operations: {h['operations']}\n")
                        f.write(f"  Time: {h['time']:.6f} seconds\n")
                        if 'precision' in h:
                            f.write(f"  Precision: {h['precision']:.2f}%\n")
                            f.write(f"  Approximation Ratio: {h['approximation_ratio']:.2f}\n")
                        f.write(f"  Cover: {sorted(h['cover'])}\n")

            f.write("\n" + "=" * 70 + "\n\n")

    print(f"Results saved to {filename} and {pickle_file}")


def generate_summary_table(results):
    """
    Generate summary statistics table.

    Args:
        results: List of result dictionaries

    Returns:
        Formatted summary string
    """
    summary = []
    summary.append("=" * 100)
    summary.append("Summary Table: Exhaustive Search Performance")
    summary.append("=" * 100)
    summary.append(f"{'n':<4} {'Density':<8} {'Edges':<6} {'Weight':<8} {'Size':<6} " +
                  f"{'Operations':<12} {'Configs':<12} {'Time (s)':<10}")
    summary.append("-" * 100)

    for result in results:
        if 'exhaustive' in result and 'weight' in result['exhaustive']:
            ex = result['exhaustive']
            n = result['vertices']
            density = result['density']
            edges = result['edges']

            summary.append(f"{n:<4} {density:<8.3f} {edges:<6} {ex['weight']:<8} " +
                          f"{ex['cover_size']:<6} {ex['operations']:<12} " +
                          f"{ex['configurations']:<12} {ex['time']:<10.6f}")

    summary.append("=" * 100)
    summary.append("")
    summary.append("Summary Table: Best Heuristic Performance")
    summary.append("=" * 100)
    summary.append(f"{'n':<4} {'Density':<8} {'Best Weight':<12} {'Optimal':<8} " +
                  f"{'Ratio':<8} {'Precision':<10} {'Time (s)':<10}")
    summary.append("-" * 100)

    for result in results:
        if 'exhaustive' in result and 'weight' in result['exhaustive']:
            optimal = result['exhaustive']['weight']

            # Find best heuristic
            best_weight = float('inf')
            best_time = 0
            for h_key in ['greedy_ratio', 'greedy_degree', 'greedy_edge']:
                if h_key in result and 'weight' in result[h_key]:
                    if result[h_key]['weight'] < best_weight:
                        best_weight = result[h_key]['weight']
                        best_time = result[h_key]['time']

            if best_weight < float('inf') and optimal > 0:
                ratio = best_weight / optimal
                precision = (optimal / best_weight) * 100

                n = result['vertices']
                density = result['density']

                summary.append(f"{n:<4} {density:<8.3f} {best_weight:<12} {optimal:<8} " +
                              f"{ratio:<8.2f} {precision:<10.2f} {best_time:<10.6f}")

    summary.append("=" * 100)

    return "\n".join(summary)


def main():
    """Main experimental runner."""
    print("Starting Experimental Analysis")
    print()

    # Check if graphs exist
    if not os.path.exists("Grafos"):
        print("Grafos directory not found!")
        print("Run GraphGenerator.py first to generate test graphs.")
        return

    # Run experiments
    # Start with small graphs, increase max_vertices if time permits
    results = run_experiments(
        min_vertices=4,
        max_vertices=12,  # Conservative start - increase if experiments run quickly
        densities=[0.125, 0.25, 0.5, 0.75],
        timeout=300  # 5 minutes per experiment
    )

    # Save results
    output_file = f"Outputs/Results_{STUDENT_NUMBER}.txt"
    save_results(results, output_file)

    # Generate and print summary
    summary = generate_summary_table(results)
    print("\n" + summary)

    # Save summary
    summary_file = f"Outputs/Summary_{STUDENT_NUMBER}.txt"
    with open(summary_file, 'w') as f:
        f.write(summary)

    print(f"\nSummary saved to {summary_file}")
    print("\nExperiments complete!")


if __name__ == "__main__":
    main()
