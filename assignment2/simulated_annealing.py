"""
Simulated Annealing algorithm for Minimum Weight Vertex Cover.
"""

import random
import math
import time
from typing import Set, Tuple
from graph_utils import Graph, greedy_vertex_cover


class SimulatedAnnealingStats:
    """Statistics tracking for Simulated Annealing."""

    def __init__(self):
        self.basic_operations = 0  # Number of basic operations
        self.solutions_tested = 0  # Number of unique solutions tested
        self.solutions_seen = set()  # Hash of solutions to avoid duplicates
        self.start_time = None
        self.end_time = None
        self.best_weight = float('inf')
        self.best_solution = None
        self.iterations = 0

    def get_execution_time(self) -> float:
        """Get execution time in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0


def simulated_annealing(
    graph: Graph,
    initial_temp: float = 1000.0,
    cooling_rate: float = 0.95,
    min_temp: float = 0.01,
    max_iterations: int = 10000,
    max_time: float = 60.0,
    seed: int = None
) -> Tuple[Set[int], SimulatedAnnealingStats]:
    """
    Solve Minimum Weight Vertex Cover using Simulated Annealing.

    Algorithm:
    1. Start with a greedy initial solution
    2. At each iteration, generate a neighbor solution
    3. Accept better solutions always
    4. Accept worse solutions with probability exp(-delta/temperature)
    5. Cool down temperature gradually

    Args:
        graph: Graph object
        initial_temp: Starting temperature
        cooling_rate: Temperature reduction factor (0 < rate < 1)
        min_temp: Minimum temperature (stopping criterion)
        max_iterations: Maximum number of iterations
        max_time: Maximum execution time in seconds
        seed: Random seed for reproducibility

    Returns:
        Tuple of (best solution set, statistics)
    """
    if seed is not None:
        random.seed(seed)

    stats = SimulatedAnnealingStats()
    stats.start_time = time.time()

    # Initialize with greedy solution
    current_solution = greedy_vertex_cover(graph)
    current_weight = graph.get_cover_weight(current_solution)
    stats.basic_operations += 1

    # Track best solution
    best_solution = current_solution.copy()
    best_weight = current_weight
    stats.best_solution = best_solution
    stats.best_weight = best_weight

    # Add to seen solutions
    solution_hash = tuple(sorted(current_solution))
    stats.solutions_seen.add(solution_hash)
    stats.solutions_tested = 1

    temperature = initial_temp
    iteration = 0

    while temperature > min_temp and iteration < max_iterations:
        # Check time limit
        if time.time() - stats.start_time > max_time:
            break

        iteration += 1
        stats.iterations = iteration

        # Generate neighbor solution
        neighbor = _get_neighbor(graph, current_solution)
        stats.basic_operations += 1

        # Check if solution is new
        neighbor_hash = tuple(sorted(neighbor))
        if neighbor_hash not in stats.solutions_seen:
            stats.solutions_seen.add(neighbor_hash)
            stats.solutions_tested += 1

        # Evaluate neighbor
        if graph.is_vertex_cover(neighbor):
            neighbor_weight = graph.get_cover_weight(neighbor)
            stats.basic_operations += 1

            # Calculate acceptance probability
            delta = neighbor_weight - current_weight

            if delta < 0:
                # Better solution - always accept
                current_solution = neighbor
                current_weight = neighbor_weight
                stats.basic_operations += 1

                # Update best if necessary
                if current_weight < best_weight:
                    best_solution = current_solution.copy()
                    best_weight = current_weight
                    stats.best_solution = best_solution
                    stats.best_weight = best_weight
                    stats.basic_operations += 1
            else:
                # Worse solution - accept with probability
                acceptance_prob = math.exp(-delta / temperature)
                stats.basic_operations += 1

                if random.random() < acceptance_prob:
                    current_solution = neighbor
                    current_weight = neighbor_weight
                    stats.basic_operations += 1

        # Cool down
        temperature *= cooling_rate
        stats.basic_operations += 1

    stats.end_time = time.time()
    return best_solution, stats


def _get_neighbor(graph: Graph, solution: Set[int]) -> Set[int]:
    """
    Generate a neighbor solution using various neighborhood operations.

    Operations:
    1. Add a random vertex not in solution
    2. Remove a random vertex from solution
    3. Swap: remove one, add another
    4. Add vertex covering uncovered edges (if any)

    Args:
        graph: Graph object
        solution: Current solution

    Returns:
        Neighbor solution
    """
    neighbor = solution.copy()
    operation = random.randint(0, 3)

    if operation == 0:
        # Add a random vertex not in solution
        not_in_solution = [v for v in range(graph.n_vertices) if v not in solution]
        if not_in_solution:
            vertex = random.choice(not_in_solution)
            neighbor.add(vertex)

    elif operation == 1:
        # Remove a random vertex from solution
        if len(neighbor) > 0:
            # Try to remove a vertex that doesn't break the cover
            vertices_list = list(neighbor)
            random.shuffle(vertices_list)

            for vertex in vertices_list:
                test_solution = neighbor.copy()
                test_solution.remove(vertex)
                if graph.is_vertex_cover(test_solution):
                    neighbor = test_solution
                    break
            else:
                # If no safe removal, just remove random one (may create invalid cover)
                vertex = random.choice(vertices_list)
                neighbor.remove(vertex)

    elif operation == 2:
        # Swap: remove one vertex, add another
        if len(neighbor) > 0:
            to_remove = random.choice(list(neighbor))
            neighbor.remove(to_remove)

            not_in_solution = [v for v in range(graph.n_vertices) if v not in neighbor]
            if not_in_solution:
                to_add = random.choice(not_in_solution)
                neighbor.add(to_add)

    else:
        # Add vertex covering uncovered edges
        uncovered = graph.get_uncovered_edges(neighbor)
        if uncovered:
            # Pick a random uncovered edge and add one of its vertices
            edge = random.choice(uncovered)
            vertex = random.choice(edge)
            neighbor.add(vertex)

    return neighbor


def simulated_annealing_adaptive(
    graph: Graph,
    max_iterations: int = 10000,
    max_time: float = 60.0,
    seed: int = None
) -> Tuple[Set[int], SimulatedAnnealingStats]:
    """
    Simulated Annealing with adaptive temperature schedule.

    The temperature is adjusted based on acceptance rate:
    - If acceptance rate is too high, decrease temperature faster
    - If acceptance rate is too low, decrease temperature slower

    Args:
        graph: Graph object
        max_iterations: Maximum number of iterations
        max_time: Maximum execution time in seconds
        seed: Random seed

    Returns:
        Tuple of (best solution set, statistics)
    """
    # Calculate initial temperature based on graph size
    initial_temp = graph.n_vertices * 10.0
    # Adaptive cooling rate
    cooling_rate = 0.95

    return simulated_annealing(
        graph,
        initial_temp=initial_temp,
        cooling_rate=cooling_rate,
        min_temp=0.01,
        max_iterations=max_iterations,
        max_time=max_time,
        seed=seed
    )
