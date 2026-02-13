"""
Main script to run all experiments and comparisons.
"""

import json
from exact_counter import ExactCounter, run_exact_counter
from probabilistic_counter import run_probabilistic_counter_trials
from frequent_count import run_frequent_count
from metrics import MetricsCalculator, aggregate_trial_metrics, print_metrics_summary
from data_utils import get_dataset_stats


def main():
    """Run all experiments and generate comprehensive results."""

    csv_file = "amazon_prime_titles.csv"

    print("="*80)
    print(" " * 20 + "ADVANCED ALGORITHMS - ASSIGNMENT 3")
    print(" " * 15 + "Frequent Items Counting - Performance Analysis")
    print("="*80)

    # Dataset statistics
    print("\n" + "="*80)
    print("DATASET STATISTICS")
    print("="*80)
    stats = get_dataset_stats(csv_file)
    print(f"Total rows in dataset: {stats['total_rows']}")
    print(f"Total cast occurrences: {stats['total_cast_occurrences']}")
    print(f"Unique cast members: {stats['unique_cast_members']}")
    print(f"Empty cast rows: {stats['empty_cast_rows']}")
    print(f"Average cast per title: {stats['avg_cast_per_title']:.2f}")

    # 1. Run Exact Counter
    print("\n" + "="*80)
    print("STEP 1: EXACT COUNTING (Baseline)")
    print("="*80)
    exact_counter, exact_stats = run_exact_counter(csv_file, verbose=True)

    # Save exact counts
    exact_counts_dict = dict(exact_counter.counter)

    # 2. Run Probabilistic Counter
    print("\n" + "="*80)
    print("STEP 2: PROBABILISTIC COUNTER (1/sqrt(2)^k)")
    print("="*80)
    num_trials = 20
    prob_counters, prob_stats = run_probabilistic_counter_trials(
        csv_file, num_trials=num_trials, verbose=True
    )

    # 3. Run Frequent-Count
    print("\n" + "="*80)
    print("STEP 3: FREQUENT-COUNT ALGORITHM")
    print("="*80)
    # n values from 5 to 100, increasing by 5 each time
    n_values = list(range(5, 101, 5))
    freq_results = run_frequent_count(csv_file, n_values=n_values, verbose=True)

    # 4. Evaluate Probabilistic Counter
    print("\n" + "="*80)
    print("STEP 4: EVALUATING PROBABILISTIC COUNTER")
    print("="*80)

    metrics_calc = MetricsCalculator(exact_counts_dict)
    prob_trial_metrics = []

    for trial_idx, counter in enumerate(prob_counters):
        estimates = counter.get_all_estimates()
        top_k = counter.get_top_k(100)

        trial_metrics = metrics_calc.comprehensive_evaluation(
            estimates, top_k, k_values=[5, 10, 20, 50, 100]
        )
        prob_trial_metrics.append(trial_metrics)

    # Aggregate across trials
    prob_aggregated = aggregate_trial_metrics(prob_trial_metrics)

    print("\nProbabilistic Counter - Aggregated Results:")
    print(f"Number of trials: {prob_aggregated['num_trials']}")
    print(f"\nAbsolute Error (mean across trials):")
    print(f"  Mean: {prob_aggregated['absolute_error_mean']['mean']:.2f} ± {prob_aggregated['absolute_error_mean']['std']:.2f}")
    print(f"  Range: [{prob_aggregated['absolute_error_mean']['min']:.2f}, {prob_aggregated['absolute_error_mean']['max']:.2f}]")
    print(f"\nRelative Error (mean across trials):")
    print(f"  Mean: {prob_aggregated['relative_error_mean']['mean']:.4f} ± {prob_aggregated['relative_error_mean']['std']:.4f}")
    print(f"  Percentage: {prob_aggregated['relative_error_mean']['mean']*100:.2f}% ± {prob_aggregated['relative_error_mean']['std']*100:.2f}%")
    print(f"\nSpearman Correlation:")
    print(f"  Mean: {prob_aggregated['spearman_correlation']['mean']:.4f} ± {prob_aggregated['spearman_correlation']['std']:.4f}")

    if 'top_10_precision' in prob_aggregated:
        print(f"\nTop-10 Precision: {prob_aggregated['top_10_precision']['mean']:.4f} ± {prob_aggregated['top_10_precision']['std']:.4f}")
        print(f"Top-10 Recall: {prob_aggregated['top_10_recall']['mean']:.4f} ± {prob_aggregated['top_10_recall']['std']:.4f}")

    # 5. Evaluate Frequent-Count
    print("\n" + "="*80)
    print("STEP 5: EVALUATING FREQUENT-COUNT ALGORITHM")
    print("="*80)

    freq_evaluations = {}

    for k, (freq_counter, freq_stats) in freq_results.items():
        print(f"\n--- Evaluating k={k} ---")

        # Get counts from frequent-count
        freq_counts_dict = dict(freq_counter.counters)
        freq_top_k = freq_counter.get_top_k(min(100, k))

        # Evaluate
        freq_metrics = metrics_calc.comprehensive_evaluation(
            freq_counts_dict, freq_top_k, k_values=[5, 10, 20, min(k, 50)]
        )

        print_metrics_summary(freq_metrics, f"Frequent-Count (k={k})")

        freq_evaluations[k] = freq_metrics

    # 6. Compare all methods
    print("\n" + "="*80)
    print("STEP 6: COMPREHENSIVE COMPARISON")
    print("="*80)

    print("\nMemory Usage Comparison:")
    print(f"  Exact Counter:          {exact_stats['memory_kb']:8.2f} KB")
    print(f"  Probabilistic Counter:  {prob_stats['avg_memory_kb']:8.2f} KB")
    for k in [10, 20, 50, 100]:
        if k in freq_results:
            print(f"  Frequent-Count (k={k:3d}): {freq_results[k][1]['memory_kb']:8.2f} KB")

    print("\nExecution Time Comparison:")
    print(f"  Exact Counter:          {exact_stats['execution_time_ms']:8.2f} ms")
    print(f"  Probabilistic Counter:  {prob_stats['avg_execution_time_ms']:8.2f} ms (avg)")
    for k in [10, 20, 50, 100]:
        if k in freq_results:
            print(f"  Frequent-Count (k={k:3d}): {freq_results[k][1]['execution_time_ms']:8.2f} ms")

    # 7. Save all results
    print("\n" + "="*80)
    print("STEP 7: SAVING RESULTS")
    print("="*80)

    all_results = {
        'dataset_stats': stats,
        'exact_counter': {
            'statistics': exact_stats,
            'top_100': [(item, count) for item, count in exact_counter.get_top_k(100)]
        },
        'probabilistic_counter': {
            'statistics': prob_stats,
            'aggregated_metrics': prob_aggregated,
            'individual_trial_metrics': prob_trial_metrics
        },
        'frequent_count': {
            'n_values': n_values,
            'results': {}
        }
    }

    for k, (freq_counter, freq_stats) in freq_results.items():
        all_results['frequent_count']['results'][f'k_{k}'] = {
            'statistics': freq_stats,
            'metrics': freq_evaluations[k],
            'top_items': [(item, count) for item, count in freq_counter.get_top_k(min(100, k))]
        }

    # Save to JSON
    with open('all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\nAll results saved to: all_results.json")

    # Save summary statistics
    with open('EXPERIMENT_STATISTICS.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("ADVANCED ALGORITHMS - ASSIGNMENT 3\n")
        f.write("Frequent Items Counting - Experiment Statistics\n")
        f.write("="*80 + "\n\n")

        f.write("DATASET STATISTICS\n")
        f.write("-"*80 + "\n")
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")

        f.write("\n\nALGORITHM STATISTICS\n")
        f.write("-"*80 + "\n")

        f.write("\n1. Exact Counter:\n")
        f.write(f"   Memory: {exact_stats['memory_kb']:.2f} KB\n")
        f.write(f"   Time: {exact_stats['execution_time_ms']:.2f} ms\n")
        f.write(f"   Unique items: {exact_stats['total_unique_items']}\n")

        f.write("\n2. Probabilistic Counter (averaged over 20 trials):\n")
        f.write(f"   Memory: {prob_stats['avg_memory_kb']:.2f} KB\n")
        f.write(f"   Time: {prob_stats['avg_execution_time_ms']:.2f} ms\n")
        f.write(f"   Absolute Error (mean): {prob_aggregated['absolute_error_mean']['mean']:.2f}\n")
        f.write(f"   Relative Error (mean): {prob_aggregated['relative_error_mean']['mean']*100:.2f}%\n")
        f.write(f"   Spearman Correlation: {prob_aggregated['spearman_correlation']['mean']:.4f}\n")

        f.write("\n3. Frequent-Count Algorithm:\n")
        for k in k_values:
            if k in freq_results:
                fstats = freq_results[k][1]
                f.write(f"\n   k={k}:\n")
                f.write(f"     Memory: {fstats['memory_kb']:.2f} KB\n")
                f.write(f"     Time: {fstats['execution_time_ms']:.2f} ms\n")
                f.write(f"     Items tracked: {fstats['items_tracked']}/{fstats['max_possible_tracked']}\n")

    print("Summary statistics saved to: EXPERIMENT_STATISTICS.txt")

    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("="*80)


if __name__ == "__main__":
    main()
