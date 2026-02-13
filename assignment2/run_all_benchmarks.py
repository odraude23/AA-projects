#!/usr/bin/env python3
"""
Run SA and GA algorithms on all benchmark graphs.
Generates comprehensive results for visualization and analysis.
"""

import os
import json
import sys
import time
from pathlib import Path
from graph_utils import read_graph_file, generate_random_weights
from simulated_annealing import simulated_annealing
from genetic_algorithm import genetic_algorithm


def run_benchmark_suite():
    """Run all algorithms on all graphs in SW_ALGUNS_GRAFOS."""

    print("="*80)
    print("COMPREHENSIVE BENCHMARK TESTING")
    print("Algorithms: SA (Simulated Annealing) + GA (Genetic Algorithm)")
    print("="*80)

    # Find all graph files
    graph_dir = Path("SW_ALGUNS_GRAFOS")
    graph_files = sorted([f for f in graph_dir.glob("*.txt") if f.name != "README.txt"])

    print(f"\nFound {len(graph_files)} graph files:")
    for gf in graph_files:
        print(f"  - {gf.name}")

    # Configuration
    config = {
        'max_time': 120,  # 2 minutes per algorithm per graph
        'seed': 42,
        'weight_seed': 100,
        'weight_min': 1.0,
        'weight_max': 100.0
    }

    print(f"\nConfiguration:")
    print(f"  Max time per algorithm: {config['max_time']} seconds")
    print(f"  Random seed: {config['seed']}")
    print(f"  Weight seed: {config['weight_seed']}")
    print(f"  Vertex weight range: [{config['weight_min']}, {config['weight_max']}]")
    print()

    results = []
    total_graphs = len(graph_files)

    for idx, graph_file in enumerate(graph_files, 1):
        print("="*80)
        print(f"Graph {idx}/{total_graphs}: {graph_file.name}")
        print("="*80)

        try:
            # Read graph
            print(f"Loading graph...")
            graph = read_graph_file(str(graph_file))

            print(f"  Vertices: {graph.n_vertices}")
            print(f"  Edges: {len(graph.edges)}")

            # Generate vertex weights
            generate_random_weights(
                graph,
                min_weight=config['weight_min'],
                max_weight=config['weight_max'],
                seed=config['weight_seed']
            )

            # Determine appropriate time limit based on graph size
            if graph.n_vertices < 100:
                time_limit = 30
            elif graph.n_vertices < 500:
                time_limit = 60
            elif graph.n_vertices < 1000:
                time_limit = 120
            else:
                time_limit = 300  # 5 minutes for very large graphs

            print(f"  Time limit: {time_limit} seconds per algorithm")
            print()

            # Run Simulated Annealing
            print(f"[1/2] Running Simulated Annealing...")
            sa_start = time.time()

            try:
                sa_solution, sa_stats = simulated_annealing(
                    graph,
                    max_time=time_limit,
                    seed=config['seed']
                )
                sa_time = time.time() - sa_start

                # Calculate solution weight
                sa_weight = graph.get_cover_weight(sa_solution)
                sa_is_valid = graph.is_vertex_cover(sa_solution)

                # Store result
                result_sa = {
                    'graph_file': str(graph_file),
                    'graph_name': graph_file.name,
                    'algorithm': 'Simulated Annealing',
                    'vertices': graph.n_vertices,
                    'edges': len(graph.edges),
                    'solution': list(sa_solution),
                    'solution_size': len(sa_solution),
                    'solution_weight': sa_weight,
                    'is_valid': sa_is_valid,
                    'execution_time': sa_time,
                    'basic_operations': sa_stats.basic_operations,
                    'solutions_tested': sa_stats.solutions_tested,
                    'iterations': sa_stats.iterations
                }
                results.append(result_sa)

                print(f"  ✓ Complete in {sa_time:.2f}s")
                print(f"    Solution: {len(sa_solution)} vertices, weight={sa_weight:.2f}")
                print(f"    Valid: {sa_is_valid}")
                print(f"    Operations: {sa_stats.basic_operations}")

            except Exception as e:
                print(f"  ✗ Error: {str(e)}")
                sa_time = time.time() - sa_start

            print()

            # Run Genetic Algorithm
            print(f"[2/2] Running Genetic Algorithm...")
            ga_start = time.time()

            try:
                ga_solution, ga_stats = genetic_algorithm(
                    graph,
                    max_time=time_limit,
                    seed=config['seed']
                )
                ga_time = time.time() - ga_start

                # Calculate solution weight
                ga_weight = graph.get_cover_weight(ga_solution)
                ga_is_valid = graph.is_vertex_cover(ga_solution)

                # Store result
                result_ga = {
                    'graph_file': str(graph_file),
                    'graph_name': graph_file.name,
                    'algorithm': 'Genetic Algorithm',
                    'vertices': graph.n_vertices,
                    'edges': len(graph.edges),
                    'solution': list(ga_solution),
                    'solution_size': len(ga_solution),
                    'solution_weight': ga_weight,
                    'is_valid': ga_is_valid,
                    'execution_time': ga_time,
                    'basic_operations': ga_stats.basic_operations,
                    'solutions_tested': ga_stats.solutions_tested,
                    'iterations': ga_stats.generations
                }
                results.append(result_ga)

                print(f"  ✓ Complete in {ga_time:.2f}s")
                print(f"    Solution: {len(ga_solution)} vertices, weight={ga_weight:.2f}")
                print(f"    Valid: {ga_is_valid}")
                print(f"    Operations: {ga_stats.basic_operations}")

            except Exception as e:
                print(f"  ✗ Error: {str(e)}")
                ga_time = time.time() - ga_start

            print()

            # Quick comparison
            if 'sa_weight' in locals() and 'ga_weight' in locals():
                print("Quick Comparison:")
                print(f"  Time:   SA={sa_time:.2f}s  vs  GA={ga_time:.2f}s")
                print(f"  Weight: SA={sa_weight:.2f}  vs  GA={ga_weight:.2f}")
                if sa_weight < ga_weight:
                    print(f"  Winner: SA (by {ga_weight - sa_weight:.2f})")
                elif ga_weight < sa_weight:
                    print(f"  Winner: GA (by {sa_weight - ga_weight:.2f})")
                else:
                    print(f"  Winner: Tie!")

            print()

        except Exception as e:
            print(f"✗ Failed to process graph: {str(e)}")
            import traceback
            traceback.print_exc()
            print()
            continue

    # Save results
    print("="*80)
    print("SAVING RESULTS")
    print("="*80)

    output_dir = Path("Outputs")
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "results_all_benchmarks.json"

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {output_file}")
    print(f"  Total results: {len(results)}")
    print(f"  Graphs tested: {len(graph_files)}")
    print(f"  Algorithms: 2 (SA, GA)")

    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    sa_results = [r for r in results if 'Simulated Annealing' in r['algorithm']]
    ga_results = [r for r in results if 'Genetic' in r['algorithm']]

    if sa_results:
        print("\nSimulated Annealing:")
        print(f"  Tests run: {len(sa_results)}")
        print(f"  Avg time: {sum(r['execution_time'] for r in sa_results) / len(sa_results):.2f}s")
        print(f"  Avg weight: {sum(r['solution_weight'] for r in sa_results) / len(sa_results):.2f}")
        print(f"  Avg operations: {sum(r['basic_operations'] for r in sa_results) / len(sa_results):.0f}")

    if ga_results:
        print("\nGenetic Algorithm:")
        print(f"  Tests run: {len(ga_results)}")
        print(f"  Avg time: {sum(r['execution_time'] for r in ga_results) / len(ga_results):.2f}s")
        print(f"  Avg weight: {sum(r['solution_weight'] for r in ga_results) / len(ga_results):.2f}")
        print(f"  Avg operations: {sum(r['basic_operations'] for r in ga_results) / len(ga_results):.0f}")

    print("\n" + "="*80)
    print("BENCHMARK TESTING COMPLETE")
    print("="*80)
    print(f"\nNext step: Generate visualizations")
    print(f"  python3 visualization.py {output_file}")
    print()

    return str(output_file)


if __name__ == "__main__":
    results_file = run_benchmark_suite()
    sys.exit(0)
