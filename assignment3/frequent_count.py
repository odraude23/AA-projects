"""
Frequent-Count Algorithm Implementation (Misra-Gries)
Finds the k most frequent items in a data stream.
"""

import time
import sys
from typing import Dict, List, Tuple
from data_utils import stream_cast_from_csv


class FrequentCount:
    """
    Frequent-Count algorithm (Misra-Gries) for finding frequent items in streams.

    Guarantees: Any item with frequency > n/k will be in the output.
    Uses O(k) space to track up to k-1 counters.
    """

    def __init__(self, k: int):
        """
        Initialize Frequent-Count algorithm.

        Args:
            k: Parameter controlling memory usage (maintains k-1 counters)
                Items appearing more than n/k times are guaranteed to be found
        """
        self.k = k
        self.counters = {}  # Current counters
        self.n = 0  # Total items processed
        self.memory_bytes = 0
        self.execution_time = 0

    def process_item(self, item: str):
        """
        Process a single item from the stream.

        Args:
            item: Item to process
        """
        self.n += 1

        if item in self.counters:
            # Item already being tracked, increment
            self.counters[item] += 1
        elif len(self.counters) < self.k - 1:
            # We have space, add new item
            self.counters[item] = 1
        else:
            to_remove = []

            for tracked_item in self.counters:
                self.counters[tracked_item] -= 1
                if self.counters[tracked_item] == 0:
                    to_remove.append(tracked_item)

            # Remove items with count 0
            for tracked_item in to_remove:
                del self.counters[tracked_item]

    def process_stream(self, items):
        """
        Process all items from a stream.

        Args:
            items: Iterable of items (can be list or generator)
        """
        start_time = time.perf_counter()

        for item in items:
            self.process_item(item)

        end_time = time.perf_counter()
        self.execution_time = end_time - start_time

        # Calculate memory
        self.memory_bytes = self._calculate_memory()

    def _calculate_memory(self) -> int:
        """Calculate approximate memory usage in bytes."""
        memory = sys.getsizeof(self.counters)
        memory += sys.getsizeof(self.k)
        memory += sys.getsizeof(self.n)

        for key, value in self.counters.items():
            memory += sys.getsizeof(key) + sys.getsizeof(value)

        return memory

    def get_frequent_items(self) -> List[Tuple[str, int]]:
        """
        Get all frequent items found by the algorithm.

        Returns:
            List of (item, approximate_count) tuples, sorted by count
        """
        sorted_items = sorted(self.counters.items(), key=lambda x: x[1], reverse=True)
        return sorted_items

    def get_top_k(self, k: int = None) -> List[Tuple[str, int]]:
        """
        Get top k frequent items.

        Args:
            k: Number of items to return (if None, use self.k)

        Returns:
            List of (item, approximate_count) tuples
        """
        if k is None:
            k = self.k

        frequent = self.get_frequent_items()
        return frequent[:k]

    def get_statistics(self) -> dict:
        """Get statistics about the algorithm."""
        return {
            'k_parameter': self.k,
            'items_processed': self.n,
            'items_tracked': len(self.counters),
            'max_possible_tracked': self.k - 1,
            'memory_bytes': self.memory_bytes,
            'memory_kb': self.memory_bytes / 1024,
            'memory_mb': self.memory_bytes / (1024 * 1024),
            'execution_time_seconds': self.execution_time,
            'execution_time_ms': self.execution_time * 1000
        }


def run_frequent_count(csv_file: str, n_values: List[int] = None,
                       verbose: bool = True) -> Dict[int, Tuple[FrequentCount, dict]]:
    """
    Run Frequent-Count algorithm with different n values (top-n to find).

    Args:
        csv_file: Path to CSV file
        n_values: List of n values (top-n items) to test (if None, use default)
        verbose: Print results if True

    Returns:
        Dictionary mapping n -> (FrequentCount instance, statistics)
    """
    if n_values is None:
        n_values = [5, 10, 15, 20, 25, 30]

    if verbose:
        print("=" * 60)
        print("FREQUENT-COUNT ALGORITHM (Misra-Gries)")
        print("=" * 60)

    results = {}

    for n in n_values:
        # Choose k parameter: for Misra-Gries, items with freq > total/(k+1) are found
        k = max(2000, n * 100)  # Large k necessary for low-frequency items

        if verbose:
            print(f"\n{'='*60}")
            print(f"Running to find top-{n} items (using k={k})")
            print(f"{'='*60}")

        # Create algorithm instance
        freq_count = FrequentCount(k)

        # Process stream
        if verbose:
            print(f"Processing stream...")

        # Use streaming to simulate real stream processing
        stream = stream_cast_from_csv(csv_file)
        freq_count.process_stream(stream)

        # Get statistics
        stats = freq_count.get_statistics()

        if verbose:
            print(f"\nResults for n={n} (k={k}):")
            print(f"  Items processed: {stats['items_processed']}")
            print(f"  Items tracked: {stats['items_tracked']}/{stats['max_possible_tracked']}")
            print(f"  Memory used: {stats['memory_kb']:.2f} KB")
            print(f"  Execution time: {stats['execution_time_ms']:.2f} ms")

            print(f"\nTop {min(10, n)} frequent items found:")
            for i, (actor, count) in enumerate(freq_count.get_top_k(min(10, n)), 1):
                print(f"  {i:2d}. {actor:40s} : {count:5d} (approx)")

        results[n] = (freq_count, stats)

    return results


if __name__ == "__main__":
    csv_file = "amazon_prime_titles.csv"

    # Run with different n values from 5 to 100 (step 5)
    n_values = list(range(5, 101, 5))
    results = run_frequent_count(csv_file, n_values=n_values)

    # Save results
    print("\n" + "="*60)
    print("Saving Frequent-Count results...")
    import json

    save_results = {}

    for n, (freq_count, stats) in results.items():
        top_items = freq_count.get_top_k(min(100, n))
        save_results[f'n_{n}'] = {
            'statistics': stats,
            'top_items': [(actor, count) for actor, count in top_items]
        }

    with open('frequent_count_results.json', 'w') as f:
        json.dump(save_results, f, indent=2)

    print("Frequent-Count results saved to frequent_count_results.json")
