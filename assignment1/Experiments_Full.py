"""
Full Experimental Runner for Minimum Weight Vertex Cover
Tests all graphs from n=4 to n=25 to demonstrate exponential growth
No timeout - let it run until complete
"""

import os
import time
import pickle
from GraphGenerator import load_graph
from Exhaustive import exhaustive_search_optimized
from Heuristic import greedy_vertex_cover, greedy_vertex_cover_simple, greedy_edge_selection
from Experiments import save_results, generate_summary_table

STUDENT_NUMBER = 103070


def run_full_experiments(min_vertices=4, max_vertices=24, densities=None):
    """
    Run complete experiments from n=4 to n=25.
    No timeout - designed to run until completion.

    Args:
        min_vertices: Minimum number of vertices
        max_vertices: Maximum number of vertices
        densities: List of edge densities

    Returns:
        List of experiment results
    """
    if densities is None:
        densities = [0.125, 0.25, 0.5, 0.75]

    results = []
    total_graphs = (max_vertices - min_vertices + 1) * len(densities)
    current = 0

    print("=" * 80)
    print("FULL EXPERIMENTAL RUN: n=4 to n=25")
    print("Testing ALL graphs to demonstrate complete exponential growth")
    print("This will take several minutes - progress will be shown")
    print("=" * 80)
    print()

    for n in range(min_vertices, max_vertices + 1):
        for density in densities:
            current += 1

            # Load graph
            graph_file = f"Grafos/graph_n{n}_d{int(density*100)}.pkl"

            if not os.path.exists(graph_file):
                print(f"[{current}/{total_graphs}] Graph not found: {graph_file}")
                continue

            G = load_graph(graph_file)
            m = G.number_of_edges()

            print(f"[{current}/{total_graphs}] Testing: n={n}, density={density}, edges={m}")

            # Initialize result dictionary
            result = {
                'vertices': n,
                'edges': m,
                'density': density,
                'graph_file': graph_file
            }

            # Run exhaustive search
            print(f"  → Exhaustive search...", end=' ', flush=True)
            try:
                start_time = time.time()
                cover_ex, weight_ex, ops_ex, configs_ex, time_ex = exhaustive_search_optimized(G)

                result['exhaustive'] = {
                    'cover': cover_ex,
                    'weight': weight_ex,
                    'cover_size': len(cover_ex),
                    'operations': ops_ex,
                    'configurations': configs_ex,
                    'time': time_ex,
                    'timeout': False
                }

                if time_ex < 0.001:
                    print(f"✓ {time_ex*1000:.2f}ms, {configs_ex:,} configs")
                elif time_ex < 1:
                    print(f"✓ {time_ex:.3f}s, {configs_ex:,} configs")
                else:
                    print(f"✓ {time_ex:.2f}s, {configs_ex:,} configs")

            except KeyboardInterrupt:
                print("\n\nINTERRUPTED by user")
                result['exhaustive'] = {'interrupted': True}
                results.append(result)
                break
            except Exception as e:
                print(f"✗ ERROR: {e}")
                result['exhaustive'] = {'error': str(e)}

            # Run heuristics (fast)
            heuristics = [
                ('greedy_ratio', greedy_vertex_cover, 'Ratio'),
                ('greedy_degree', greedy_vertex_cover_simple, 'Degree'),
                ('greedy_edge', greedy_edge_selection, 'Edge')
            ]

            heuristic_times = []
            for heuristic_name, heuristic_func, short_name in heuristics:
                try:
                    cover_h, weight_h, ops_h, time_h = heuristic_func(G)

                    result[heuristic_name] = {
                        'cover': cover_h,
                        'weight': weight_h,
                        'cover_size': len(cover_h),
                        'operations': ops_h,
                        'time': time_h
                    }

                    heuristic_times.append(time_h)

                    # Calculate precision if exhaustive succeeded
                    if 'exhaustive' in result and 'weight' in result['exhaustive']:
                        optimal_weight = result['exhaustive']['weight']
                        # Precision: 100% = optimal, <100% = worse (quality metric)
                        precision = (optimal_weight / weight_h) * 100 if weight_h > 0 else 0
                        # Approximation ratio: 1.0 = optimal, >1.0 = worse
                        approximation_ratio = weight_h / optimal_weight if optimal_weight > 0 else float('inf')
                        result[heuristic_name]['precision'] = precision
                        result[heuristic_name]['approximation_ratio'] = approximation_ratio

                except Exception as e:
                    result[heuristic_name] = {'error': str(e)}

            # Report heuristic performance
            if heuristic_times and 'exhaustive' in result and 'time' in result['exhaustive']:
                best_h_time = min(heuristic_times)
                speedup = result['exhaustive']['time'] / best_h_time if best_h_time > 0 else 0
                best_precision = max([result[h]['precision'] for h in ['greedy_ratio', 'greedy_degree', 'greedy_edge']
                                     if h in result and 'precision' in result[h]])
                print(f"  → Heuristics: {best_h_time*1000:.2f}ms, {speedup:.0f}× faster, {best_precision:.1f}% precision")

            results.append(result)
            print()

    return results


def main():
    """Main full experimental runner."""
    import datetime

    start_overall = time.time()

    print("\n" + "=" * 80)
    print("STARTING FULL EXPERIMENTAL RUN")
    print(f"Start time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print()

    # Check if graphs exist for n=16-25
    print("Checking if all graphs exist...")
    missing = []
    for n in range(4, 26):
        for density in [0.125, 0.25, 0.5, 0.75]:
            graph_file = f"Grafos/graph_n{n}_d{int(density*100)}.pkl"
            if not os.path.exists(graph_file):
                missing.append(f"n={n}, d={density}")

    if missing:
        print(f"\nWARNING: {len(missing)} graph files missing!")
        print("Missing graphs:", missing[:5], "..." if len(missing) > 5 else "")
        print("\nGenerating missing graphs...")
        from GraphGenerator import generate_all_graphs
        generate_all_graphs(min_vertices=4, max_vertices=25, seed=STUDENT_NUMBER)
        print("✓ All graphs generated\n")
    else:
        print("✓ All graph files present\n")

    # Run experiments
    results = run_full_experiments(
        min_vertices=4,
        max_vertices=25,
        densities=[0.125, 0.25, 0.5, 0.75]
    )

    # Save results
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    output_file = f"Outputs/Results_{STUDENT_NUMBER}_Full.txt"
    save_results(results, output_file)
    print(f"✓ Saved to {output_file}")

    # Generate and print summary
    summary = generate_summary_table(results)
    print("\n" + summary)

    # Save summary
    summary_file = f"Outputs/Summary_{STUDENT_NUMBER}_Full.txt"
    with open(summary_file, 'w') as f:
        f.write(summary)
    print(f"\n✓ Summary saved to {summary_file}")

    # Print detailed timing analysis
    print("\n" + "=" * 80)
    print("EXPONENTIAL GROWTH ANALYSIS")
    print("=" * 80)

    for density in [0.125, 0.25, 0.5, 0.75]:
        print(f"\nDensity = {density}:")
        print(f"{'n':<4} {'Edges':<8} {'Configs':<15} {'Time':<12} {'Growth':<10}")
        print("-" * 55)

        prev_time = None
        for result in results:
            if result['density'] == density and 'exhaustive' in result:
                ex = result['exhaustive']
                if 'time' in ex and 'configurations' in ex:
                    n = result['vertices']
                    edges = result['edges']
                    configs = ex['configurations']
                    curr_time = ex['time']

                    growth = f"{curr_time/prev_time:.2f}×" if prev_time and prev_time > 0 else "-"

                    if curr_time < 1:
                        time_str = f"{curr_time:.4f}s"
                    else:
                        time_str = f"{curr_time:.2f}s"

                    print(f"{n:<4} {edges:<8} {configs:<15,} {time_str:<12} {growth:<10}")
                    prev_time = curr_time

    # Final statistics
    elapsed_overall = time.time() - start_overall

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    print(f"End time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total duration: {elapsed_overall/60:.1f} minutes")
    print(f"Graphs tested: {len(results)}")

    # Count successful exhaustive runs
    successful = sum(1 for r in results if 'exhaustive' in r and 'time' in r['exhaustive'])
    print(f"Successful exhaustive searches: {successful}/{len(results)}")

    # Find largest graph
    largest_time = 0
    largest_n = 0
    for r in results:
        if 'exhaustive' in r and 'time' in r['exhaustive']:
            if r['exhaustive']['time'] > largest_time:
                largest_time = r['exhaustive']['time']
                largest_n = r['vertices']

    print(f"Largest graph processed: n={largest_n} in {largest_time:.2f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
