"""
Genetic Algorithm for Minimum Weight Vertex Cover.
"""

import random
import time
from typing import Set, List, Tuple
from graph_utils import Graph, greedy_vertex_cover


class GeneticAlgorithmStats:
    """Statistics tracking for Genetic Algorithm."""

    def __init__(self):
        self.basic_operations = 0
        self.solutions_tested = 0
        self.solutions_seen = set()
        self.start_time = None
        self.end_time = None
        self.best_weight = float('inf')
        self.best_solution = None
        self.generations = 0

    def get_execution_time(self) -> float:
        """Get execution time in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0


class Individual:
    """Represents an individual in the population."""

    def __init__(self, vertices: Set[int], graph: Graph):
        self.vertices = vertices
        self.fitness = self._calculate_fitness(graph)
        self.is_valid = graph.is_vertex_cover(vertices)

    def _calculate_fitness(self, graph: Graph) -> float:
        """
        Calculate fitness of the individual.
        Lower is better (minimization problem).

        Fitness = weight + penalty for uncovered edges
        """
        weight = graph.get_cover_weight(self.vertices)
        uncovered = len(graph.get_uncovered_edges(self.vertices))

        # Heavy penalty for invalid solutions
        penalty = uncovered * sum(graph.vertex_weights) * 10

        return weight + penalty


def genetic_algorithm(
    graph: Graph,
    population_size: int = 100,
    max_generations: int = 500,
    mutation_rate: float = 0.1,
    crossover_rate: float = 0.8,
    elitism_count: int = 2,
    max_time: float = 60.0,
    seed: int = None
) -> Tuple[Set[int], GeneticAlgorithmStats]:
    """
    Solve Minimum Weight Vertex Cover using Genetic Algorithm.

    Algorithm:
    1. Initialize population with random solutions
    2. For each generation:
       - Evaluate fitness
       - Select parents
       - Apply crossover
       - Apply mutation
       - Replace population (with elitism)

    Args:
        graph: Graph object
        population_size: Number of individuals in population
        max_generations: Maximum number of generations
        mutation_rate: Probability of mutation
        crossover_rate: Probability of crossover
        elitism_count: Number of best individuals to preserve
        max_time: Maximum execution time in seconds
        seed: Random seed

    Returns:
        Tuple of (best solution set, statistics)
    """
    if seed is not None:
        random.seed(seed)

    stats = GeneticAlgorithmStats()
    stats.start_time = time.time()

    # Initialize population
    population = _initialize_population(graph, population_size, stats)

    best_individual = None
    best_fitness = float('inf')

    for generation in range(max_generations):
        # Check time limit
        if time.time() - stats.start_time > max_time:
            break

        stats.generations = generation + 1

        # Evaluate and sort population
        population.sort(key=lambda ind: ind.fitness)
        stats.basic_operations += len(population)

        # Update best solution
        if population[0].fitness < best_fitness:
            best_individual = population[0]
            best_fitness = population[0].fitness
            stats.best_solution = best_individual.vertices.copy()
            stats.best_weight = graph.get_cover_weight(best_individual.vertices)
            stats.basic_operations += 1

        # Check convergence (optional)
        if best_individual and best_individual.is_valid:
            # If we have a valid solution and population is converging
            unique_fitnesses = len(set(ind.fitness for ind in population[:10]))
            if unique_fitnesses == 1:
                # Population converged, could restart or continue
                pass

        # Create new population
        new_population = []

        # Elitism: preserve best individuals
        for i in range(min(elitism_count, len(population))):
            new_population.append(population[i])

        # Generate offspring
        while len(new_population) < population_size:
            # Selection
            parent1 = _tournament_selection(population, 3)
            parent2 = _tournament_selection(population, 3)
            stats.basic_operations += 2

            # Crossover
            if random.random() < crossover_rate:
                child1, child2 = _crossover(parent1, parent2, graph, stats)
                stats.basic_operations += 1
            else:
                child1, child2 = parent1, parent2

            # Mutation
            if random.random() < mutation_rate:
                child1 = _mutate(child1, graph, stats)
                stats.basic_operations += 1

            if random.random() < mutation_rate:
                child2 = _mutate(child2, graph, stats)
                stats.basic_operations += 1

            # Repair if needed (greedy repair for invalid solutions)
            child1 = _repair_solution(child1, graph, stats)
            child2 = _repair_solution(child2, graph, stats)

            new_population.append(child1)
            if len(new_population) < population_size:
                new_population.append(child2)

        population = new_population

    stats.end_time = time.time()

    # Return best valid solution found
    if best_individual and best_individual.is_valid:
        return best_individual.vertices, stats
    else:
        # Fallback to greedy if no valid solution
        fallback = greedy_vertex_cover(graph)
        stats.best_solution = fallback
        stats.best_weight = graph.get_cover_weight(fallback)
        return fallback, stats


def _initialize_population(
    graph: Graph,
    population_size: int,
    stats: GeneticAlgorithmStats
) -> List[Individual]:
    """
    Initialize population with diverse solutions.

    Mix of:
    - Random solutions
    - Greedy solutions with randomization
    - Partial random solutions
    """
    population = []

    # Add greedy solution
    greedy_sol = greedy_vertex_cover(graph)
    population.append(Individual(greedy_sol, graph))
    solution_hash = tuple(sorted(greedy_sol))
    stats.solutions_seen.add(solution_hash)
    stats.solutions_tested += 1

    # Add random solutions
    for _ in range(population_size - 1):
        # Random solution: include each vertex with some probability
        vertices = set()

        for v in range(graph.n_vertices):
            if random.random() < 0.5:
                vertices.add(v)

        # Ensure it's a valid cover by adding vertices for uncovered edges
        uncovered = graph.get_uncovered_edges(vertices)
        while uncovered:
            edge = random.choice(uncovered)
            vertex = random.choice(edge)
            vertices.add(vertex)
            uncovered = graph.get_uncovered_edges(vertices)
            stats.basic_operations += 1

        individual = Individual(vertices, graph)
        population.append(individual)

        solution_hash = tuple(sorted(vertices))
        if solution_hash not in stats.solutions_seen:
            stats.solutions_seen.add(solution_hash)
            stats.solutions_tested += 1

    return population


def _tournament_selection(population: List[Individual], tournament_size: int) -> Individual:
    """Select individual using tournament selection."""
    tournament = random.sample(population, min(tournament_size, len(population)))
    return min(tournament, key=lambda ind: ind.fitness)


def _crossover(
    parent1: Individual,
    parent2: Individual,
    graph: Graph,
    stats: GeneticAlgorithmStats
) -> Tuple[Individual, Individual]:
    """
    Perform crossover between two parents.

    Uses uniform crossover: each vertex is inherited from either parent.
    """
    child1_vertices = set()
    child2_vertices = set()

    for v in range(graph.n_vertices):
        if random.random() < 0.5:
            if v in parent1.vertices:
                child1_vertices.add(v)
            if v in parent2.vertices:
                child2_vertices.add(v)
        else:
            if v in parent2.vertices:
                child1_vertices.add(v)
            if v in parent1.vertices:
                child2_vertices.add(v)

    child1 = Individual(child1_vertices, graph)
    child2 = Individual(child2_vertices, graph)

    # Track new solutions
    for vertices in [child1_vertices, child2_vertices]:
        solution_hash = tuple(sorted(vertices))
        if solution_hash not in stats.solutions_seen:
            stats.solutions_seen.add(solution_hash)
            stats.solutions_tested += 1

    return child1, child2


def _mutate(individual: Individual, graph: Graph, stats: GeneticAlgorithmStats) -> Individual:
    """
    Mutate an individual by randomly adding/removing vertices.

    Mutation types:
    1. Flip: add if not in, remove if in
    2. Add random vertex
    3. Remove random vertex
    """
    vertices = individual.vertices.copy()
    mutation_type = random.randint(0, 2)

    if mutation_type == 0:
        # Flip a random vertex
        vertex = random.randint(0, graph.n_vertices - 1)
        if vertex in vertices:
            vertices.remove(vertex)
        else:
            vertices.add(vertex)

    elif mutation_type == 1:
        # Add a random vertex not in solution
        not_in = [v for v in range(graph.n_vertices) if v not in vertices]
        if not_in:
            vertices.add(random.choice(not_in))

    else:
        # Remove a random vertex from solution
        if vertices:
            vertices.remove(random.choice(list(vertices)))

    mutated = Individual(vertices, graph)

    solution_hash = tuple(sorted(vertices))
    if solution_hash not in stats.solutions_seen:
        stats.solutions_seen.add(solution_hash)
        stats.solutions_tested += 1

    return mutated


def _repair_solution(
    individual: Individual,
    graph: Graph,
    stats: GeneticAlgorithmStats
) -> Individual:
    """
    Repair invalid solution using randomized greedy approach.

    Add vertices to cover uncovered edges, preferring low-weight vertices.
    """
    if individual.is_valid:
        return individual

    vertices = individual.vertices.copy()
    uncovered = graph.get_uncovered_edges(vertices)

    while uncovered:
        # Get vertices that can cover uncovered edges
        candidates = set()
        for u, v in uncovered:
            candidates.add(u)
            candidates.add(v)

        candidates = list(candidates - vertices)

        if not candidates:
            break

        # Select vertex with probability inversely proportional to weight
        weights = [graph.vertex_weights[v] for v in candidates]
        min_weight = min(weights)

        # Randomized greedy: bias towards low weight
        probs = [min_weight / w for w in weights]
        total = sum(probs)
        probs = [p / total for p in probs]

        selected = random.choices(candidates, weights=probs)[0]
        vertices.add(selected)

        uncovered = graph.get_uncovered_edges(vertices)
        stats.basic_operations += 1

    repaired = Individual(vertices, graph)

    solution_hash = tuple(sorted(vertices))
    if solution_hash not in stats.solutions_seen:
        stats.solutions_seen.add(solution_hash)
        stats.solutions_tested += 1

    return repaired
