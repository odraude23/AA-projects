"""
Decreasing Probability Counter Implementation
Uses probability 1/sqrt(2)^k for incrementing.
"""

import random
import math
import time
import sys
from collections import defaultdict
from typing import Dict, List, Tuple
from data_utils import parse_cast_from_csv


class ProbabilisticCounter:
    """
    Probabilistic counter with decreasing probability 1/sqrt(2)^k.
    Based on Morris' approximate counting algorithm.
    """

    def __init__(self, seed: int = None):
        """
        Initialize probabilistic counter.

        Args:
            seed: Random seed for reproducibility
        """
        self.counters = defaultdict(int)  # Stores counter values (not estimates)
        self.seed = seed
        if seed is not None:
            random.seed(seed)
        self.memory_bytes = 0
        self.execution_time = 0

    def increment(self, item: str):
        """
        Probabilistically increment counter for an item.
        Increments with probability 1/sqrt(2)^k where k is current counter value.

        Args:
            item: Item to increment
        """
        k = self.counters[item]

        # Probability of incrementing: 1 / sqrt(2)^k
        probability = 1.0 / (math.sqrt(2) ** k)

        # Increment with this probability
        if random.random() < probability:
            self.counters[item] += 1

    def estimate_count(self, item: str) -> float:
        """
        Estimate the actual count from the counter value.

        For decreasing probability counter with p = 1/sqrt(2)^k,
        the estimate is: (sqrt(2)^k - 1) / (sqrt(2) - 1)

        This is the sum of geometric series: 1 + sqrt(2) + sqrt(2)^2 + ... + sqrt(2)^(k-1)
        which represents the expected number of items processed to reach counter value k.

        Args:
            item: Item to estimate count for

        Returns:
            Estimated count
        """
        k = self.counters.get(item, 0)

        if k == 0:
            return 0.0

        # Estimate based on geometric series sum
        sqrt_2 = math.sqrt(2)
        estimate = (sqrt_2 ** k - 1) / (sqrt_2 - 1)

        return estimate

    def count(self, items: list):
        """
        Count items using probabilistic counter.

        Args:
            items: List of items to count
        """
        start_time = time.perf_counter()

        for item in items:
            self.increment(item)

        end_time = time.perf_counter()
        self.execution_time = end_time - start_time

        # Calculate memory
        self.memory_bytes = self._calculate_memory()

    def _calculate_memory(self) -> int:
        """Calculate approximate memory usage in bytes."""
        memory = sys.getsizeof(self.counters)

        for key, value in self.counters.items():
            memory += sys.getsizeof(key) + sys.getsizeof(value)

        return memory

    def get_all_estimates(self) -> Dict[str, float]:
        """Get estimated counts for all items."""
        return {item: self.estimate_count(item) for item in self.counters}

    def get_top_k(self, k: int) -> List[Tuple[str, float]]:
        """
        Get top k items by estimated count.

        Args:
            k: Number of top items

        Returns:
            List of (item, estimated_count) tuples
        """
        estimates = self.get_all_estimates()
        sorted_items = sorted(estimates.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:k]

    def get_statistics(self) -> dict:
        """Get statistics about the counter."""
        return {
            'total_unique_items': len(self.counters),
            'memory_bytes': self.memory_bytes,
            'memory_kb': self.memory_bytes / 1024,
            'memory_mb': self.memory_bytes / (1024 * 1024),
            'execution_time_seconds': self.execution_time,
            'execution_time_ms': self.execution_time * 1000,
            'seed': self.seed
        }


def run_probabilistic_counter_trials(csv_file: str, num_trials: int = 20,
                                     verbose: bool = True) -> Tuple[List[ProbabilisticCounter], dict]:
    """
    Run multiple trials of probabilistic counter.

    Args:
        csv_file: Path to CSV file
        num_trials: Number of independent trials to run
        verbose: Print results if True

    Returns:
        Tuple of (list of counter instances, aggregate statistics)
    """
    if verbose:
        print("=" * 60)
        print(f"PROBABILISTIC COUNTER - {num_trials} Trials")
        print("=" * 60)

    # Parse data once
    if verbose:
        print("\nParsing dataset...")
    cast_members = parse_cast_from_csv(csv_file)

    if verbose:
        print(f"Total cast occurrences: {len(cast_members)}")

    # Run multiple trials
    counters = []
    all_stats = []

    if verbose:
        print(f"\nRunning {num_trials} independent trials...")

    for trial in range(num_trials):
        counter = ProbabilisticCounter(seed=trial)
        counter.count(cast_members)
        counters.append(counter)
        all_stats.append(counter.get_statistics())

        if verbose and (trial + 1) % 5 == 0:
            print(f"  Completed trial {trial + 1}/{num_trials}")

    # Aggregate statistics
    avg_memory = sum(s['memory_bytes'] for s in all_stats) / num_trials
    avg_time = sum(s['execution_time_seconds'] for s in all_stats) / num_trials

    aggregate_stats = {
        'num_trials': num_trials,
        'avg_memory_bytes': avg_memory,
        'avg_memory_kb': avg_memory / 1024,
        'avg_memory_mb': avg_memory / (1024 * 1024),
        'avg_execution_time_seconds': avg_time,
        'avg_execution_time_ms': avg_time * 1000,
        'all_trial_stats': all_stats
    }

    if verbose:
        print(f"\nAggregate Results:")
        print(f"  Average memory used: {aggregate_stats['avg_memory_kb']:.2f} KB")
        print(f"  Average execution time: {aggregate_stats['avg_execution_time_ms']:.2f} ms")

        # Show top 10 from first trial as example
        print(f"\nTop 10 from Trial 1 (example):")
        for i, (actor, est_count) in enumerate(counters[0].get_top_k(10), 1):
            print(f"  {i:2d}. {actor:40s} : {est_count:8.2f}")

    return counters, aggregate_stats


if __name__ == "__main__":
    csv_file = "amazon_prime_titles.csv"

    # Run multiple trials
    counters, stats = run_probabilistic_counter_trials(csv_file, num_trials=20)

    # Save results
    print("\nSaving probabilistic counter results...")
    import json

    # Save estimates from all trials for the top 100 items (based on trial 0)
    top_100_items = [item for item, _ in counters[0].get_top_k(100)]

    results = {
        'statistics': stats,
        'trials_estimates': {}
    }

    # For each trial, save estimates for the top 100 items
    for trial_idx, counter in enumerate(counters):
        trial_estimates = {item: counter.estimate_count(item) for item in top_100_items}
        results['trials_estimates'][f'trial_{trial_idx}'] = trial_estimates

    with open('probabilistic_counts.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("Probabilistic counter results saved to probabilistic_counts.json")
