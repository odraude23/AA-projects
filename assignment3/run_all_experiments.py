import json
import time
import numpy as np
from collections import Counter
from exact_counter import ExactCounter, run_exact_counter
from probabilistic_counter import ProbabilisticCounter
from frequent_count import FrequentCount
from metrics import MetricsCalculator, aggregate_trial_metrics, print_metrics_summary
from data_utils import parse_cast_from_csv, stream_cast_from_csv, get_dataset_stats


def convert_to_python_types(obj):
    """
    Convert numpy types to Python native types for JSON serialization.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_python_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_python_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_python_types(item) for item in obj)
    else:
        return obj


def requirement_a_exact_counts(csv_file: str):
    """
    Requirement A: Compute the exact number of occurrences of each item.
    """
    print("\n" + "="*80)
    print("REQUIREMENT A: EXACT NUMBER OF OCCURRENCES")
    print("="*80)

    exact_counter, exact_stats = run_exact_counter(csv_file, verbose=True)

    # Additional analysis
    print(f"\nDistribution Analysis:")
    counts = list(exact_counter.counter.values())
    counts.sort(reverse=True)

    print(f"  Top 1 count: {counts[0]}")
    print(f"  Top 10 average: {sum(counts[:10])/10:.2f}")
    print(f"  Top 100 average: {sum(counts[:100])/100:.2f}")
    print(f"  Median count: {counts[len(counts)//2]}")
    print(f"  Items appearing once: {sum(1 for c in counts if c == 1)}")

    return exact_counter, exact_stats


def requirement_b_approximate_counts(csv_file: str, num_trials: int = 20):
    """
    Requirement B: Estimate occurrences using approximate counters.
    Perform multiple trials to assess variability.
    """
    print("\n" + "="*80)
    print(f"REQUIREMENT B: APPROXIMATE COUNTS ({num_trials} trials)")
    print("="*80)

    print(f"\nParsing dataset...")
    cast_members = parse_cast_from_csv(csv_file)
    print(f"Total items to process: {len(cast_members)}")

    all_counters = []
    all_stats = []

    print(f"\nRunning {num_trials} independent trials...")
    start_overall = time.time()

    for trial in range(num_trials):
        trial_start = time.time()

        # Create counter with different seed for each trial
        counter = ProbabilisticCounter(seed=trial)
        counter.count(cast_members)

        all_counters.append(counter)
        stats = counter.get_statistics()
        all_stats.append(stats)

        trial_time = time.time() - trial_start

        if (trial + 1) % 5 == 0:
            print(f"  Trial {trial + 1:2d}/{num_trials} completed in {trial_time:.3f}s")

    total_time = time.time() - start_overall

    print(f"\nAll trials completed in {total_time:.2f}s")
    print(f"Average time per trial: {total_time/num_trials:.3f}s")

    # Analyze variance across trials
    print(f"\nVariance Analysis:")

    # Pick a sample of items to analyze
    sample_items = list(all_counters[0].counters.keys())[:20]

    for item_idx, item in enumerate(sample_items[:5]):  # Show 5 examples
        estimates = [c.estimate_count(item) for c in all_counters]
        import numpy as np
        print(f"\n  Item: {item[:40]}")
        print(f"    Estimates range: [{min(estimates):.1f}, {max(estimates):.1f}]")
        print(f"    Mean: {np.mean(estimates):.2f} ± {np.std(estimates):.2f}")
        print(f"    Coefficient of variation: {np.std(estimates)/np.mean(estimates)*100:.1f}%")

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
        'total_experiment_time': total_time,
        'all_trial_stats': all_stats
    }

    print(f"\nAggregate Statistics:")
    print(f"  Average memory: {aggregate_stats['avg_memory_kb']:.2f} KB")
    print(f"  Average execution time: {aggregate_stats['avg_execution_time_ms']:.2f} ms")

    return all_counters, aggregate_stats


def requirement_c_frequent_items(csv_file: str, n_values: list):
    """
    Requirement C: Estimate the n most frequent items using data stream algorithm.
    Test with n = 5, 10, 15, 20, ..., 100
    """
    print("\n" + "="*80)
    print("REQUIREMENT C: MOST FREQUENT ITEMS (Data Stream Algorithm)")
    print("="*80)

    print(f"\nTesting with n values: {n_values}")
    print(f"Total configurations to test: {len(n_values)}")

    results = {}

    for i, n in enumerate(n_values, 1):
        print(f"\n{'-'*80}")
        print(f"Configuration {i}/{len(n_values)}: n = {n}")
        print(f"{'-'*80}")

        # Create Frequent-Count instance
        k = max(2000, n * 100) 

        freq_count = FrequentCount(k)

        # Process stream
        print(f"Processing stream with k={k}...")
        stream = stream_cast_from_csv(csv_file)
        freq_count.process_stream(stream)

        stats = freq_count.get_statistics()

        print(f"  Items processed: {stats['items_processed']}")
        print(f"  Items tracked: {stats['items_tracked']}/{stats['max_possible_tracked']}")
        print(f"  Memory: {stats['memory_kb']:.2f} KB")
        print(f"  Time: {stats['execution_time_ms']:.2f} ms")

        # Get top n items
        top_n = freq_count.get_top_k(n)

        print(f"\n  Top {min(5, n)} items found:")
        for rank, (actor, count) in enumerate(top_n[:5], 1):
            print(f"    {rank}. {actor[:40]:40s} : {count:5d}")

        results[n] = {
            'instance': freq_count,
            'statistics': stats,
            'k_parameter': k,
            'top_n': top_n
        }

    print(f"\n{'='*80}")
    print("Requirement C completed!")
    print(f"Tested {len(n_values)} different n values")
    print(f"{'='*80}")

    return results


def requirement_d_compare_performance(exact_counter, prob_counters, freq_results,
                                     exact_stats, prob_stats, n_values):
    """
    Requirement D: Compare performance of all algorithms.
    """
    print("\n" + "="*80)
    print("REQUIREMENT D: PERFORMANCE COMPARISON")
    print("="*80)

    # Get exact counts as ground truth
    exact_counts_dict = dict(exact_counter.counter)
    metrics_calc = MetricsCalculator(exact_counts_dict)

    print("\n" + "-"*80)
    print("D.1: PROBABILISTIC COUNTER EVALUATION")
    print("-"*80)

    prob_trial_metrics = []

    print(f"\nEvaluating {len(prob_counters)} trials...")
    for trial_idx, counter in enumerate(prob_counters):
        estimates = counter.get_all_estimates()
        top_k = counter.get_top_k(100)

        trial_metrics = metrics_calc.comprehensive_evaluation(
            estimates, top_k, k_values=[5, 10, 20, 50, 100]
        )
        prob_trial_metrics.append(trial_metrics)

    # Aggregate across trials
    prob_aggregated = aggregate_trial_metrics(prob_trial_metrics)

    print(f"\n--- Probabilistic Counter Results (averaged over {len(prob_counters)} trials) ---")
    print(f"\nError Metrics:")
    print(f"  Absolute Error:")
    print(f"    Mean: {prob_aggregated['absolute_error_mean']['mean']:.2f} ± {prob_aggregated['absolute_error_mean']['std']:.2f}")
    print(f"    Range: [{prob_aggregated['absolute_error_mean']['min']:.2f}, {prob_aggregated['absolute_error_mean']['max']:.2f}]")
    print(f"\n  Relative Error:")
    print(f"    Mean: {prob_aggregated['relative_error_mean']['mean']*100:.2f}% ± {prob_aggregated['relative_error_mean']['std']*100:.2f}%")
    print(f"    Range: [{prob_aggregated['relative_error_mean']['min']*100:.2f}%, {prob_aggregated['relative_error_mean']['max']*100:.2f}%]")

    print(f"\nRank Correlation:")
    print(f"  Spearman: {prob_aggregated['spearman_correlation']['mean']:.4f} ± {prob_aggregated['spearman_correlation']['std']:.4f}")

    if 'top_10_precision' in prob_aggregated:
        print(f"\nTop-10 Identification:")
        print(f"  Precision: {prob_aggregated['top_10_precision']['mean']:.4f} ± {prob_aggregated['top_10_precision']['std']:.4f}")
        print(f"  Recall: {prob_aggregated['top_10_recall']['mean']:.4f} ± {prob_aggregated['top_10_recall']['std']:.4f}")
        print(f"  F1-Score: {prob_aggregated['top_10_f1']['mean']:.4f} ± {prob_aggregated['top_10_f1']['std']:.4f}")

    print("\n" + "-"*80)
    print("D.2: FREQUENT-COUNT EVALUATION")
    print("-"*80)

    freq_evaluations = {}

    for n in n_values:
        result = freq_results[n]
        freq_counter = result['instance']

        # Get counts and top items
        freq_counts_dict = dict(freq_counter.counters)
        freq_top_n = result['top_n']

        # Evaluate
        freq_metrics = metrics_calc.comprehensive_evaluation(
            freq_counts_dict, freq_top_n, k_values=[min(n, k) for k in [5, 10, 20, 50] if k <= n]
        )

        freq_evaluations[n] = freq_metrics

    # Print sample results for key n values
    sample_n_values = [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    for n in sample_n_values:
        if n in freq_evaluations and n in n_values:
            metrics = freq_evaluations[n]
            print(f"\n--- Frequent-Count Results (n={n}) ---")

            if 'errors' in metrics and metrics['errors']['num_items_compared'] > 0:
                print(f"  Absolute Error: {metrics['errors']['absolute_error_mean']:.2f}")
                print(f"  Relative Error: {metrics['errors']['relative_error_mean']*100:.2f}%")

            if 'rank_correlation' in metrics:
                print(f"  Spearman Correlation: {metrics['rank_correlation']['spearman_correlation']:.4f}")

            # Top-k metrics
            k_key = f'k_{min(n, 10)}'  # Show top-10 or top-n if n < 10
            if 'top_k_metrics' in metrics and k_key in metrics['top_k_metrics']:
                tk = metrics['top_k_metrics'][k_key]
                print(f"  Top-{tk['k']} Precision: {tk['precision']:.4f}")
                print(f"  Top-{tk['k']} Recall: {tk['recall']:.4f}")

    print("\n" + "-"*80)
    print("D.3: MEMORY USAGE COMPARISON")
    print("-"*80)

    print(f"\nAlgorithm                        Memory (KB)    Memory (MB)")
    print(f"{'-'*60}")
    print(f"Exact Counter                    {exact_stats['memory_kb']:10.2f}    {exact_stats['memory_mb']:10.4f}")
    print(f"Probabilistic Counter (avg)      {prob_stats['avg_memory_kb']:10.2f}    {prob_stats['avg_memory_mb']:10.4f}")

    print(f"\nFrequent-Count (various n values):")
    display_n_values = [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for n in display_n_values:
        if n in freq_results:
            stats = freq_results[n]['statistics']
            print(f"  n={n:3d}                          {stats['memory_kb']:10.2f}    {stats['memory_mb']:10.4f}")

    print("\n" + "-"*80)
    print("D.4: EXECUTION TIME COMPARISON")
    print("-"*80)

    print(f"\nAlgorithm                        Time (ms)      Time (s)")
    print(f"{'-'*60}")
    print(f"Exact Counter                    {exact_stats['execution_time_ms']:10.2f}    {exact_stats['execution_time_seconds']:10.4f}")
    print(f"Probabilistic Counter (avg)      {prob_stats['avg_execution_time_ms']:10.2f}    {prob_stats['avg_execution_time_seconds']:10.4f}")

    print(f"\nFrequent-Count (various n values):")
    display_n_values = [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for n in display_n_values:
        if n in freq_results:
            stats = freq_results[n]['statistics']
            print(f"  n={n:3d}                          {stats['execution_time_ms']:10.2f}    {stats['execution_time_seconds']:10.4f}")

    print("\n" + "-"*80)
    print("D.5: TOP-10 ITEMS COMPARISON")
    print("-"*80)

    exact_top_10 = exact_counter.get_top_k(10)

    # For probabilistic counter, use AVERAGE estimates across all trials (more representative than single trial)
    from collections import defaultdict
    avg_estimates = defaultdict(float)
    for counter in prob_counters:
        estimates = counter.get_all_estimates()
        for item, est in estimates.items():
            avg_estimates[item] += est / len(prob_counters)

    # Sort by average estimate and get top-10
    prob_top_10 = sorted(avg_estimates.items(), key=lambda x: x[1], reverse=True)[:10]

    # Use the best n value (highest tested) for most accurate Frequent-Count results
    best_n = max(n_values)
    freq_top_10 = freq_results[best_n]['top_n'][:10]  # Get top-10 from best configuration

    num_trials = len(prob_counters)
    print(f"\n{'Rank':<6}{'Exact':<45}{f'Probabilistic (Avg of {num_trials} trials)':<45}{f'Frequent-Count (n={best_n})':<45}")
    print(f"{'-'*140}")

    # Get exact top-10 names for comparison
    exact_top_10_names = {item[0] for item in exact_top_10}

    for i in range(10):
        exact_item = f"{exact_top_10[i][0][:35]} ({exact_top_10[i][1]})" if i < len(exact_top_10) else "-"
        prob_item = f"{prob_top_10[i][0][:25]} ({prob_top_10[i][1]:.1f})" if i < len(prob_top_10) else "-"
        freq_item = f"{freq_top_10[i][0][:25]} ({freq_top_10[i][1]})" if i < len(freq_top_10) else "-"

        print(f"{i+1:<6}{exact_item:<45}{prob_item:<45}{freq_item:<45}")

    # Show overlap statistics
    prob_top_10_names = {item[0] for item in prob_top_10}
    freq_top_10_names = {item[0] for item in freq_top_10}

    prob_overlap = len(exact_top_10_names & prob_top_10_names)
    freq_overlap = len(exact_top_10_names & freq_top_10_names)

    print(f"\nTop-10 Accuracy:")
    print(f"  Probabilistic: {prob_overlap}/10 correct items ({prob_overlap*10}%)")
    print(f"  Frequent-Count: {freq_overlap}/10 correct items ({freq_overlap*10}%)")

    print("\n" + "-"*80)
    print("D.6: SUMMARY STATISTICS")
    print("-"*80)

    print(f"\nDataset:")
    print(f"  Total unique items (exact): {len(exact_counts_dict)}")
    print(f"  Total occurrences: {sum(exact_counts_dict.values())}")

    print(f"\nProbabilistic Counter:")
    print(f"  Avg items tracked: {sum(len(c.counters) for c in prob_counters) / len(prob_counters):.0f}")
    print(f"  Memory reduction: {(1 - prob_stats['avg_memory_kb']/exact_stats['memory_kb'])*100:.1f}%")
    print(f"  Time overhead: {(prob_stats['avg_execution_time_ms']/exact_stats['execution_time_ms'] - 1)*100:+.1f}%")

    print(f"\nFrequent-Count (n=50 as example):")
    if 50 in freq_results:
        freq_50 = freq_results[50]
        print(f"  Items tracked: {freq_50['statistics']['items_tracked']}")
        print(f"  Memory reduction: {(1 - freq_50['statistics']['memory_kb']/exact_stats['memory_kb'])*100:.1f}%")
        print(f"  Time overhead: {(freq_50['statistics']['execution_time_ms']/exact_stats['execution_time_ms'] - 1)*100:+.1f}%")

    return {
        'probabilistic_metrics': prob_trial_metrics,
        'probabilistic_aggregated': prob_aggregated,
        'frequent_count_metrics': freq_evaluations
    }


def main():
    """
    Main experiment runner following all assignment requirements.
    """
    csv_file = "amazon_prime_titles.csv"

    print("="*80)
    print(" " * 15 + "ADVANCED ALGORITHMS - ASSIGNMENT 3")
    print(" " * 10 + "Comprehensive Experiments - All Requirements")
    print("="*80)

    # Dataset info
    print("\n" + "="*80)
    print("DATASET INFORMATION")
    print("="*80)
    stats = get_dataset_stats(csv_file)
    for key, value in stats.items():
        print(f"  {key}: {value}")


    exact_counter, exact_stats = requirement_a_exact_counts(csv_file)

    num_trials = 10
    prob_counters, prob_stats = requirement_b_approximate_counts(csv_file, num_trials)

    n_values = list(range(5, 101, 5))  
    freq_results = requirement_c_frequent_items(csv_file, n_values)

    comparison_results = requirement_d_compare_performance(
        exact_counter, prob_counters, freq_results,
        exact_stats, prob_stats, n_values
    )

    # Save all results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)

    all_results = {
        'dataset_statistics': stats,
        'requirement_a_exact': {
            'statistics': exact_stats,
            'top_100': [(item, count) for item, count in exact_counter.get_top_k(100)]
        },
        'requirement_b_probabilistic': {
            'num_trials': num_trials,
            'statistics': prob_stats,
            'aggregated_metrics': comparison_results['probabilistic_aggregated'],
        },
        'requirement_c_frequent_count': {
            'n_values': n_values,
            'results': {}
        },
        'requirement_d_comparison': {
            'probabilistic_metrics': comparison_results['probabilistic_metrics'],
            'frequent_count_metrics': comparison_results['frequent_count_metrics']
        }
    }

    # Save Frequent-Count results for all n values
    for n in n_values:
        result = freq_results[n]
        all_results['requirement_c_frequent_count']['results'][f'n_{n}'] = {
            'k_parameter': result['k_parameter'],
            'statistics': result['statistics'],
            'top_items': [(item, count) for item, count in result['top_n']]
        }

    # Convert all numpy types to Python native types for JSON serialization
    all_results_cleaned = convert_to_python_types(all_results)

    with open('Outputs/all_experiments_results.json', 'w') as f:
        json.dump(all_results_cleaned, f, indent=2)

    print("\nResults saved to: Outputs/all_experiments_results.json")

    # Save detailed comparison report
    with open('Outputs/DETAILED_COMPARISON.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("ADVANCED ALGORITHMS - ASSIGNMENT 3\n")
        f.write("Detailed Performance Comparison\n")
        f.write("="*80 + "\n\n")

        f.write("REQUIREMENT A: EXACT COUNTER\n")
        f.write("-"*80 + "\n")
        f.write(f"Unique items: {exact_stats['total_unique_items']}\n")
        f.write(f"Total occurrences: {exact_stats['total_occurrences']}\n")
        f.write(f"Memory: {exact_stats['memory_kb']:.2f} KB\n")
        f.write(f"Time: {exact_stats['execution_time_ms']:.2f} ms\n\n")

        f.write("REQUIREMENT B: PROBABILISTIC COUNTER (20 trials)\n")
        f.write("-"*80 + "\n")
        agg = comparison_results['probabilistic_aggregated']
        f.write(f"Absolute Error (mean): {agg['absolute_error_mean']['mean']:.2f} ± {agg['absolute_error_mean']['std']:.2f}\n")
        f.write(f"Relative Error (mean): {agg['relative_error_mean']['mean']*100:.2f}% ± {agg['relative_error_mean']['std']*100:.2f}%\n")
        f.write(f"Spearman Correlation: {agg['spearman_correlation']['mean']:.4f} ± {agg['spearman_correlation']['std']:.4f}\n")
        f.write(f"Memory (avg): {prob_stats['avg_memory_kb']:.2f} KB\n")
        f.write(f"Time (avg): {prob_stats['avg_execution_time_ms']:.2f} ms\n\n")

        f.write("REQUIREMENT C: FREQUENT-COUNT ALGORITHM\n")
        f.write("-"*80 + "\n")
        f.write(f"n values tested: {n_values}\n\n")

        for n in n_values:
            result = freq_results[n]
            metrics = comparison_results['frequent_count_metrics'][n]
            f.write(f"\nn = {n}:\n")
            f.write(f"  k parameter: {result['k_parameter']}\n")
            f.write(f"  Items tracked: {result['statistics']['items_tracked']}\n")
            f.write(f"  Memory: {result['statistics']['memory_kb']:.2f} KB\n")
            f.write(f"  Time: {result['statistics']['execution_time_ms']:.2f} ms\n")

            if 'errors' in metrics and metrics['errors']['num_items_compared'] > 0:
                f.write(f"  Absolute Error: {metrics['errors']['absolute_error_mean']:.2f}\n")
                f.write(f"  Relative Error: {metrics['errors']['relative_error_mean']*100:.2f}%\n")

            if 'rank_correlation' in metrics:
                f.write(f"  Spearman Correlation: {metrics['rank_correlation']['spearman_correlation']:.4f}\n")

    print("Detailed comparison saved to: Outputs/DETAILED_COMPARISON.txt")

    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"\nResults Summary:")
    print(f"  - Exact counting: ✓")
    print(f"  - Probabilistic counter: {num_trials} trials ✓")
    print(f"  - Frequent-Count: {len(n_values)} different n values (5 to 100, step 5) ✓")
    print(f"  - Performance comparison: ✓")
    print(f"\nOutput files:")
    print(f"  - Outputs/all_experiments_results.json")
    print(f"  - Outputs/DETAILED_COMPARISON.txt")


if __name__ == "__main__":
    main()
