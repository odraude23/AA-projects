"""
Exact Counter Implementation
Uses hash table (Python dict/Counter) to count exact occurrences.
"""

import time
import sys
from collections import Counter
from typing import Dict, Tuple
from data_utils import parse_cast_from_csv


class ExactCounter:
    """
    Exact counter using hash table for precise counting.
    """

    def __init__(self):
        self.counter = Counter()
        self.memory_bytes = 0
        self.execution_time = 0

    def count(self, items: list) -> Counter:
        """
        Count exact occurrences of all items.

        Args:
            items: List of items to count

        Returns:
            Counter object with exact counts
        """
        start_time = time.perf_counter()

        # Count all items
        self.counter = Counter(items)

        end_time = time.perf_counter()
        self.execution_time = end_time - start_time

        # Calculate memory usage
        self.memory_bytes = self._calculate_memory()

        return self.counter

    def _calculate_memory(self) -> int:
        """Calculate approximate memory usage in bytes."""
        # Counter object size + all keys and values
        memory = sys.getsizeof(self.counter)

        # Add memory for all stored items
        for key, value in self.counter.items():
            memory += sys.getsizeof(key) + sys.getsizeof(value)

        return memory

    def get_top_k(self, k: int) -> list:
        """
        Get top k most frequent items.

        Args:
            k: Number of top items to return

        Returns:
            List of (item, count) tuples
        """
        return self.counter.most_common(k)

    def get_bottom_k(self, k: int) -> list:
        """
        Get k least frequent items.

        Args:
            k: Number of bottom items to return

        Returns:
            List of (item, count) tuples
        """
        return self.counter.most_common()[:-k-1:-1]

    def get_count(self, item: str) -> int:
        """Get exact count for a specific item."""
        return self.counter.get(item, 0)

    def get_statistics(self) -> dict:
        """
        Get statistics about the counting process.

        Returns:
            Dictionary with statistics
        """
        return {
            'total_unique_items': len(self.counter),
            'total_occurrences': sum(self.counter.values()),
            'memory_bytes': self.memory_bytes,
            'memory_kb': self.memory_bytes / 1024,
            'memory_mb': self.memory_bytes / (1024 * 1024),
            'execution_time_seconds': self.execution_time,
            'execution_time_ms': self.execution_time * 1000
        }


def run_exact_counter(csv_file: str, verbose: bool = True) -> Tuple[ExactCounter, dict]:
    """
    Run exact counter on the dataset.

    Args:
        csv_file: Path to CSV file
        verbose: Print results if True

    Returns:
        Tuple of (ExactCounter instance, statistics dict)
    """
    if verbose:
        print("=" * 60)
        print("EXACT COUNTER")
        print("=" * 60)

    # Parse data
    if verbose:
        print("\nParsing dataset...")
    cast_members = parse_cast_from_csv(csv_file)

    if verbose:
        print(f"Total cast occurrences: {len(cast_members)}")

    # Count
    if verbose:
        print("\nCounting exact occurrences...")
    exact = ExactCounter()
    exact.count(cast_members)

    # Get statistics
    stats = exact.get_statistics()

    if verbose:
        print(f"\nResults:")
        print(f"  Unique cast members: {stats['total_unique_items']}")
        print(f"  Total occurrences: {stats['total_occurrences']}")
        print(f"  Memory used: {stats['memory_kb']:.2f} KB ({stats['memory_mb']:.4f} MB)")
        print(f"  Execution time: {stats['execution_time_ms']:.2f} ms")

        print(f"\nTop 10 most frequent cast members:")
        for i, (actor, count) in enumerate(exact.get_top_k(10), 1):
            print(f"  {i:2d}. {actor:40s} : {count:5d}")

    return exact, stats


if __name__ == "__main__":
    csv_file = "amazon_prime_titles.csv"
    exact_counter, stats = run_exact_counter(csv_file)

    # Save results
    print("\nSaving exact counts for later comparison...")
    import json

    # Save top 100 for later analysis
    top_100 = exact_counter.get_top_k(100)
    results = {
        'statistics': stats,
        'top_100': [(actor, count) for actor, count in top_100]
    }

    with open('exact_counts.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("Exact counts saved to exact_counts.json")
