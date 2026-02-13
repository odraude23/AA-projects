"""
Main script to run and compare randomized algorithms for Minimum Weight Vertex Cover.

This script runs all three implemented algorithms and generates performance reports
as required by the project specifications.
"""

import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any
import json

from graph_utils import Graph, read_graph_file, generate_random_weights
from simulated_annealing import simulated_annealing, simulated_annealing_adaptive
from genetic_algorithm import genetic_algorithm


def run_algorithm(
    algorithm_name: str,
    graph: Graph,
    max_time: float = 60.0,
    seed: int = 42,
    **kwargs
) -> Dict[str, Any]:
    """
    Run a specific algorithm and return results.

    Args:
        algorithm_name: Name of the algorithm to run
        graph: Graph object
        max_time: Maximum execution time in seconds
        seed: Random seed
        **kwargs: Additional algorithm-specific parameters

    Returns:
        Dictionary with results
    """
    print(f"\nRunning {algorithm_name}...")

    if algorithm_name == "Simulated Annealing":
        solution, stats = simulated_annealing(
            graph,
            max_time=max_time,
            seed=seed,
            **kwargs
        )
    elif algorithm_name == "Simulated Annealing (Adaptive)":
        solution, stats = simulated_annealing_adaptive(
            graph,
            max_time=max_time,
            seed=seed,
            **kwargs
        )
    elif algorithm_name == "Genetic Algorithm":
        solution, stats = genetic_algorithm(
            graph,
            max_time=max_time,
            seed=seed,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")

    # Verify solution
    is_valid = graph.is_vertex_cover(solution)
    weight = graph.get_cover_weight(solution)

    results = {
        "algorithm": algorithm_name,
        "valid": is_valid,
        "solution_size": len(solution),
        "solution_weight": weight,
        "execution_time": stats.get_execution_time(),
        "basic_operations": stats.basic_operations,
        "solutions_tested": stats.solutions_tested,
        "iterations": getattr(stats, 'iterations', getattr(stats, 'generations', 0)),
        "solution": sorted(list(solution))
    }

    return results


def print_results(results: Dict[str, Any]):
    """Print algorithm results in a formatted way."""
    print(f"\n{'='*70}")
    print(f"Algorithm: {results['algorithm']}")
    print(f"{'='*70}")
    print(f"Valid Solution: {results['valid']}")
    print(f"Solution Size: {results['solution_size']} vertices")
    print(f"Solution Weight: {results['solution_weight']:.4f}")
    print(f"Execution Time: {results['execution_time']:.4f} seconds")
    print(f"Basic Operations: {results['basic_operations']:,}")
    print(f"Solutions Tested: {results['solutions_tested']:,}")
    print(f"Iterations/Generations: {results['iterations']}")
    print(f"{'='*70}")


def compare_results(all_results: List[Dict[str, Any]]):
    """Print comparison of all algorithms."""
    print("\n" + "="*70)
    print("COMPARISON OF ALL ALGORITHMS")
    print("="*70)

    # Sort by solution weight
    valid_results = [r for r in all_results if r['valid']]

    if not valid_results:
        print("No valid solutions found!")
        return

    valid_results.sort(key=lambda x: x['solution_weight'])

    print(f"\n{'Algorithm':<35} {'Weight':<12} {'Time(s)':<10} {'Ops':<12} {'Sols':<8}")
    print("-" * 70)

    for r in valid_results:
        print(f"{r['algorithm']:<35} "
              f"{r['solution_weight']:<12.4f} "
              f"{r['execution_time']:<10.4f} "
              f"{r['basic_operations']:<12,} "
              f"{r['solutions_tested']:<8,}")

    print("\n" + "="*70)
    print(f"Best Solution: {valid_results[0]['algorithm']}")
    print(f"Best Weight: {valid_results[0]['solution_weight']:.4f}")
    print(f"Best Solution Size: {valid_results[0]['solution_size']} vertices")
    print("="*70)


def save_results(all_results: List[Dict[str, Any]], output_file: str):
    """Save results to JSON file."""
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_file}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Run randomized algorithms for Minimum Weight Vertex Cover'
    )
    parser.add_argument(
        'graph_file',
        type=str,
        help='Path to graph file in Sedgewick & Wayne format'
    )
    parser.add_argument(
        '--algorithms',
        type=str,
        nargs='+',
        choices=['SA', 'SA-Adaptive', 'GRASP', 'GRASP-PR', 'GA', 'all'],
        default=['all'],
        help='Algorithms to run (default: all)'
    )
    parser.add_argument(
        '--max-time',
        type=float,
        default=60.0,
        help='Maximum execution time per algorithm in seconds (default: 60)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--weight-seed',
        type=int,
        default=None,
        help='Random seed for vertex weights (default: None = use graph vertex index as weight)'
    )
    parser.add_argument(
        '--min-weight',
        type=float,
        default=1.0,
        help='Minimum vertex weight (default: 1.0)'
    )
    parser.add_argument(
        '--max-weight',
        type=float,
        default=10.0,
        help='Maximum vertex weight (default: 10.0)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output JSON file for results (default: None)'
    )

    args = parser.parse_args()

    # Read graph
    print(f"Reading graph from {args.graph_file}...")
    graph = read_graph_file(args.graph_file)

    print(f"Graph loaded:")
    print(f"  Vertices: {graph.n_vertices}")
    print(f"  Edges: {len(graph.edges)}")
    print(f"  Directed: {graph.is_directed}")

    # Generate or set vertex weights
    if args.weight_seed is not None:
        print(f"\nGenerating random vertex weights (seed={args.weight_seed})...")
        generate_random_weights(graph, args.min_weight, args.max_weight, args.weight_seed)
    else:
        # Use vertex index + 1 as weight
        print("\nUsing vertex index + 1 as weights...")
        graph.set_vertex_weights([float(i + 1) for i in range(graph.n_vertices)])

    # Determine which algorithms to run
    algorithm_map = {
        'SA': ('Simulated Annealing', {'max_iterations': 10000}),
        'SA-Adaptive': ('Simulated Annealing (Adaptive)', {'max_iterations': 10000}),
        'GRASP': ('GRASP', {'max_iterations': 1000, 'alpha': 0.3}),
        'GRASP-PR': ('GRASP with Path Relinking', {'max_iterations': 500, 'alpha': 0.3}),
        'GA': ('Genetic Algorithm', {'population_size': 100, 'max_generations': 500})
    }

    if 'all' in args.algorithms:
        algorithms_to_run = list(algorithm_map.keys())
    else:
        algorithms_to_run = args.algorithms

    # Run algorithms
    all_results = []

    for alg_key in algorithms_to_run:
        alg_name, alg_params = algorithm_map[alg_key]

        try:
            results = run_algorithm(
                alg_name,
                graph,
                max_time=args.max_time,
                seed=args.seed,
                **alg_params
            )
            all_results.append(results)
            print_results(results)

        except Exception as e:
            print(f"\nError running {alg_name}: {e}")
            import traceback
            traceback.print_exc()

    # Compare results
    if len(all_results) > 1:
        compare_results(all_results)

    # Save results if requested
    if args.output:
        save_results(all_results, args.output)


if __name__ == '__main__':
    main()
