"""
Visualization script for algorithm comparison.
Generates plots for report and analysis.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Set style for academic papers
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# Create output directory
OUTPUT_DIR = Path("Outputs/Figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_results():
    """Load experiment results from JSON file."""
    with open('Outputs/all_experiments_results.json', 'r') as f:
        return json.load(f)


def plot_memory_comparison(results):
    """
    Plot 1: Frequent-Count memory scaling.
    """
    print("Generating Plot 1: Frequent-Count Memory Scaling...")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Get data
    exact_mem = results['requirement_a_exact']['statistics']['memory_kb']

    freq_data = results['requirement_c_frequent_count']['results']
    n_values = sorted([int(k.split('_')[1]) for k in freq_data.keys()])
    freq_mem = [freq_data[f'n_{n}']['statistics']['memory_kb'] for n in n_values]

    # Frequent-Count memory scaling
    ax.plot(n_values, freq_mem, marker='o', linewidth=2.5,
             markersize=8, color='#F18F01', label='Frequent-Count')
    ax.axhline(y=exact_mem, color='#2E86AB', linestyle='--',
                linewidth=2, label=f'Exact Counter ({exact_mem:.1f} KB)')

    ax.set_xlabel('n (Top-n Items to Find)', fontweight='bold')
    ax.set_ylabel('Memory Usage (KB)', fontweight='bold')
    ax.set_title('Memory Scaling: Frequent-Count Algorithm', fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')

    # Add memory reduction annotation
    max_freq_mem = max(freq_mem)
    reduction = (1 - max_freq_mem / exact_mem) * 100
    ax.text(0.98, 0.95, f'Max reduction:\n{reduction:.1f}%',
             transform=ax.transAxes, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '01_memory_scaling_frequent_count.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {OUTPUT_DIR / '01_memory_scaling_frequent_count.png'}")


def plot_execution_time_comparison(results):
    """
    Plot 2: Execution time comparison.
    """
    print("Generating Plot 2: Execution Time Comparison...")

    fig, ax = plt.subplots(figsize=(12, 6))

    # Get data
    exact_time = results['requirement_a_exact']['statistics']['execution_time_ms']
    prob_time = results['requirement_b_probabilistic']['statistics']['avg_execution_time_ms']

    freq_data = results['requirement_c_frequent_count']['results']
    n_values = sorted([int(k.split('_')[1]) for k in freq_data.keys()])
    freq_times = [freq_data[f'n_{n}']['statistics']['execution_time_ms'] for n in n_values]

    # Create grouped bar chart
    x = np.arange(len(n_values))
    width = 0.35

    # Plot Frequent-Count times
    bars1 = ax.bar(x, freq_times, width, label='Frequent-Count',
                   color='#F18F01', alpha=0.8, edgecolor='black')

    # Add horizontal lines for exact and probabilistic
    ax.axhline(y=exact_time, color='#2E86AB', linestyle='--',
               linewidth=2.5, label=f'Exact Counter ({exact_time:.2f} ms)')
    ax.axhline(y=prob_time, color='#A23B72', linestyle=':',
               linewidth=2.5, label=f'Probabilistic Counter ({prob_time:.2f} ms)')

    ax.set_xlabel('n (Top-n Items to Find)', fontweight='bold')
    ax.set_ylabel('Execution Time (ms)', fontweight='bold')
    ax.set_title('Execution Time Comparison Across Algorithms', fontweight='bold')
    ax.set_xticks(x[::2])  # Show every other n value
    ax.set_xticklabels([str(n) for n in n_values[::2]])
    ax.legend(loc='best')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '02_execution_time_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {OUTPUT_DIR / '02_execution_time_comparison.png'}")


def plot_prob_absolute_error(results):
    """
    Plot 3a: Absolute error across trials for probabilistic counter.
    """
    print("Generating Plot 3a: Absolute Error Across Trials...")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Get trial metrics
    trial_metrics = results['requirement_d_comparison']['probabilistic_metrics']
    abs_errors_mean = [m['errors']['absolute_error_mean'] for m in trial_metrics]
    trials = list(range(1, len(trial_metrics) + 1))

    # Plot Absolute Error (mean) across trials
    ax.plot(trials, abs_errors_mean, marker='o', linewidth=2, markersize=6, color='#E63946')
    ax.axhline(y=np.mean(abs_errors_mean), color='black', linestyle='--',
                linewidth=1.5, label=f'Average: {np.mean(abs_errors_mean):.2f}')
    ax.fill_between(trials,
                     np.mean(abs_errors_mean) - np.std(abs_errors_mean),
                     np.mean(abs_errors_mean) + np.std(abs_errors_mean),
                     alpha=0.2, color='gray')
    ax.set_xlabel('Trial Number', fontweight='bold')
    ax.set_ylabel('Mean Absolute Error', fontweight='bold')
    ax.set_title('Absolute Error Across Trials', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '03a_prob_absolute_error.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {OUTPUT_DIR / '03a_prob_absolute_error.png'}")


def plot_prob_relative_error(results):
    """
    Plot 3b: Relative error across trials for probabilistic counter.
    """
    print("Generating Plot 3b: Relative Error Across Trials...")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Get trial metrics
    trial_metrics = results['requirement_d_comparison']['probabilistic_metrics']
    rel_errors_mean = [m['errors']['relative_error_mean'] for m in trial_metrics]
    trials = list(range(1, len(trial_metrics) + 1))

    # Plot Relative Error (mean) across trials
    rel_errors_pct = [r * 100 for r in rel_errors_mean]
    ax.plot(trials, rel_errors_pct, marker='s', linewidth=2, markersize=6, color='#F77F00')
    ax.axhline(y=np.mean(rel_errors_pct), color='black', linestyle='--',
                linewidth=1.5, label=f'Average: {np.mean(rel_errors_pct):.2f}%')
    ax.fill_between(trials,
                     np.mean(rel_errors_pct) - np.std(rel_errors_pct),
                     np.mean(rel_errors_pct) + np.std(rel_errors_pct),
                     alpha=0.2, color='gray')
    ax.set_xlabel('Trial Number', fontweight='bold')
    ax.set_ylabel('Mean Relative Error (%)', fontweight='bold')
    ax.set_title('Relative Error Across Trials', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '03b_prob_relative_error.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {OUTPUT_DIR / '03b_prob_relative_error.png'}")


def plot_prob_rank_correlation(results):
    """
    Plot 3c: Rank correlation across trials for probabilistic counter.
    """
    print("Generating Plot 3c: Rank Correlation Across Trials...")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Get trial metrics
    trial_metrics = results['requirement_d_comparison']['probabilistic_metrics']
    spearman_corrs = [m['rank_correlation']['spearman_correlation'] for m in trial_metrics]
    trials = list(range(1, len(trial_metrics) + 1))

    # Plot Spearman Correlation across trials
    ax.plot(trials, spearman_corrs, marker='^', linewidth=2, markersize=6, color='#06A77D')
    ax.axhline(y=np.mean(spearman_corrs), color='black', linestyle='--',
                linewidth=1.5, label=f'Average: {np.mean(spearman_corrs):.4f}')
    ax.fill_between(trials,
                     np.mean(spearman_corrs) - np.std(spearman_corrs),
                     np.mean(spearman_corrs) + np.std(spearman_corrs),
                     alpha=0.2, color='gray')
    ax.set_xlabel('Trial Number', fontweight='bold')
    ax.set_ylabel('Spearman Correlation', fontweight='bold')
    ax.set_title('Rank Correlation Across Trials', fontweight='bold')
    ax.set_ylim([min(spearman_corrs) - 0.01, 1.0])
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '03c_prob_rank_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {OUTPUT_DIR / '03c_prob_rank_correlation.png'}")


def plot_prob_error_distribution(results):
    """
    Plot 3d: Error distribution (box plot) for probabilistic counter.
    """
    print("Generating Plot 3d: Error Metrics Distribution...")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Get trial metrics
    trial_metrics = results['requirement_d_comparison']['probabilistic_metrics']
    abs_errors_mean = [m['errors']['absolute_error_mean'] for m in trial_metrics]
    rel_errors_mean = [m['errors']['relative_error_mean'] for m in trial_metrics]
    spearman_corrs = [m['rank_correlation']['spearman_correlation'] for m in trial_metrics]

    rel_errors_pct = [r * 100 for r in rel_errors_mean]

    # Error distribution (box plot)
    ax.boxplot([abs_errors_mean, rel_errors_pct,
                 [s * 100 for s in spearman_corrs]],
                labels=['Abs Error', 'Rel Error (%)', 'Correlation (×100)'],
                patch_artist=True,
                boxprops=dict(facecolor='lightblue', alpha=0.7),
                medianprops=dict(color='red', linewidth=2))
    ax.set_ylabel('Value', fontweight='bold')
    ax.set_title('Error Metrics Distribution (20 Trials)', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '03d_prob_error_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {OUTPUT_DIR / '03d_prob_error_distribution.png'}")


def plot_freq_precision_recall(results):
    """
    Plot 4a: Frequent-Count precision, recall, and F1-score.
    """
    print("Generating Plot 4a: Frequent-Count Precision/Recall/F1-Score...")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Get data
    freq_metrics = results['requirement_d_comparison']['frequent_count_metrics']
    n_values = sorted([int(k) for k in freq_metrics.keys()])

    precisions = []
    recalls = []
    f1_scores = []

    for n in n_values:
        metrics = freq_metrics[str(n)]
        # Get top-k metrics for this n (try k=n, k=10, k=5)
        for k_key in [f'k_{n}', 'k_10', 'k_5']:
            if k_key in metrics.get('top_k_metrics', {}):
                tk = metrics['top_k_metrics'][k_key]
                precisions.append(tk['precision'])
                recalls.append(tk['recall'])
                f1_scores.append(tk['f1_score'])
                break
        else:
            precisions.append(0)
            recalls.append(0)
            f1_scores.append(0)

    # Plot Precision and Recall vs n
    ax.plot(n_values, precisions, marker='o', linewidth=2.5, markersize=8,
             color='#2A9D8F', label='Precision')
    ax.plot(n_values, recalls, marker='s', linewidth=2.5, markersize=8,
             color='#E76F51', label='Recall')
    ax.plot(n_values, f1_scores, marker='^', linewidth=2.5, markersize=8,
             color='#F4A261', label='F1-Score')

    ax.set_xlabel('n (Top-n Items to Find)', fontweight='bold')
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('Frequent-Count: Precision, Recall & F1-Score', fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([-0.05, 1.05])

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '04a_freq_precision_recall.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {OUTPUT_DIR / '04a_freq_precision_recall.png'}")


def plot_freq_error_analysis(results):
    """
    Plot 4b: Frequent-Count error analysis.
    """
    print("Generating Plot 4b: Frequent-Count Error Analysis...")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Get data
    freq_metrics = results['requirement_d_comparison']['frequent_count_metrics']
    n_values = sorted([int(k) for k in freq_metrics.keys()])

    abs_errors = []
    rel_errors = []

    for n in n_values:
        metrics = freq_metrics[str(n)]
        if 'errors' in metrics and metrics['errors']['num_items_compared'] > 0:
            abs_errors.append(metrics['errors']['absolute_error_mean'])
            rel_errors.append(metrics['errors']['relative_error_mean'] * 100)
        else:
            abs_errors.append(None)
            rel_errors.append(None)

    # Filter out None values
    valid_indices = [i for i, e in enumerate(abs_errors) if e is not None]
    valid_n = [n_values[i] for i in valid_indices]
    valid_abs = [abs_errors[i] for i in valid_indices]
    valid_rel = [rel_errors[i] for i in valid_indices]

    ax_twin = ax.twinx()

    line1 = ax.plot(valid_n, valid_abs, marker='o', linewidth=2.5, markersize=8,
                     color='#E63946', label='Absolute Error')
    line2 = ax_twin.plot(valid_n, valid_rel, marker='s', linewidth=2.5, markersize=8,
                          color='#457B9D', label='Relative Error (%)')

    ax.set_xlabel('n (Top-n Items to Find)', fontweight='bold')
    ax.set_ylabel('Mean Absolute Error', fontweight='bold', color='#E63946')
    ax_twin.set_ylabel('Mean Relative Error (%)', fontweight='bold', color='#457B9D')
    ax.set_title('Frequent-Count: Error Analysis', fontweight='bold')

    ax.tick_params(axis='y', labelcolor='#E63946')
    ax_twin.tick_params(axis='y', labelcolor='#457B9D')

    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='best')

    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '04b_freq_error_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {OUTPUT_DIR / '04b_freq_error_analysis.png'}")


def plot_memory_vs_accuracy_tradeoff(results):
    """
    Plot 5: Memory vs Accuracy trade-off.
    """
    print("Generating Plot 5: Memory vs Accuracy Trade-off...")

    fig, ax = plt.subplots(figsize=(10, 8))

    # Exact counter (baseline)
    exact_mem = results['requirement_a_exact']['statistics']['memory_kb']
    exact_acc = 100.0  # 100% accurate

    # Probabilistic counter
    prob_mem = results['requirement_b_probabilistic']['statistics']['avg_memory_kb']
    prob_agg = results['requirement_b_probabilistic']['aggregated_metrics']
    prob_corr = prob_agg['spearman_correlation']['mean'] * 100  # Convert to percentage

    # Frequent-Count (sample some n values)
    freq_data = results['requirement_c_frequent_count']['results']
    freq_metrics = results['requirement_d_comparison']['frequent_count_metrics']

    sample_n = [5, 10, 20, 30, 50, 75, 100]
    freq_mems = []
    freq_accs = []

    for n in sample_n:
        if f'n_{n}' in freq_data:
            freq_mems.append(freq_data[f'n_{n}']['statistics']['memory_kb'])

            # Use precision as accuracy measure
            metrics = freq_metrics[str(n)]
            for k_key in [f'k_{n}', 'k_10', 'k_5']:
                if k_key in metrics.get('top_k_metrics', {}):
                    freq_accs.append(metrics['top_k_metrics'][k_key]['precision'] * 100)
                    break
            else:
                freq_accs.append(0)

    # Scatter plot
    ax.scatter(exact_mem, exact_acc, s=300, marker='*', color='gold',
               edgecolors='black', linewidth=2, label='Exact Counter', zorder=5)
    ax.scatter(prob_mem, prob_corr, s=200, marker='D', color='#A23B72',
               edgecolors='black', linewidth=1.5, label='Probabilistic Counter', zorder=4)

    scatter = ax.scatter(freq_mems, freq_accs, s=150, c=sample_n, cmap='viridis',
                        marker='o', edgecolors='black', linewidth=1,
                        label='Frequent-Count', zorder=3)

    # Add colorbar for n values
    cbar = plt.colorbar(scatter, ax=ax, label='n (top-n items)')

    # Annotate points
    ax.annotate('Perfect\nAccuracy', xy=(exact_mem, exact_acc),
                xytext=(exact_mem + 200, exact_acc - 10),
                arrowprops=dict(arrowstyle='->', lw=1.5),
                fontsize=10, fontweight='bold')

    ax.annotate(f'{prob_corr:.1f}%\nCorrelation', xy=(prob_mem, prob_corr),
                xytext=(prob_mem - 800, prob_corr - 15),
                arrowprops=dict(arrowstyle='->', lw=1.5),
                fontsize=10, fontweight='bold')

    # Add diagonal "Pareto frontier" guide line
    ax.plot([0, exact_mem], [0, exact_acc], 'k--', alpha=0.3, linewidth=1, zorder=1)

    ax.set_xlabel('Memory Usage (KB)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Accuracy/Correlation (%)', fontweight='bold', fontsize=12)
    ax.set_title('Memory vs Accuracy Trade-off\n(Higher Right = Better, Lower Left = More Efficient)',
                 fontweight='bold', fontsize=14)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim([-100, exact_mem + 500])
    ax.set_ylim([-5, 105])

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '05_memory_accuracy_tradeoff.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {OUTPUT_DIR / '05_memory_accuracy_tradeoff.png'}")


def plot_top10_comparison(results):
    """
    Plot 6: Top-10 items comparison across algorithms.
    """
    print("Generating Plot 6: Top-10 Items Comparison...")

    fig, ax = plt.subplots(figsize=(12, 8))

    # Get top 10 from each algorithm
    exact_top10 = results['requirement_a_exact']['top_100'][:10]

    items = [item for item, count in exact_top10]
    exact_counts = [count for item, count in exact_top10]

    y_pos = np.arange(len(items))

    bars = ax.barh(y_pos, exact_counts, color='#2E86AB', alpha=0.8, edgecolor='black')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(items)
    ax.invert_yaxis()
    ax.set_xlabel('Exact Count (Occurrences)', fontweight='bold')
    ax.set_title('Top 10 Most Frequent Cast Members (Exact Count)', fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    # Add value labels
    for i, (bar, count) in enumerate(zip(bars, exact_counts)):
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2.,
                f'{count}',
                ha='left', va='center', fontweight='bold')

    # Add annotation about "1"
    ax.text(0.98, 0.98, 'Note: "1" is test/placeholder data\nin the dataset',
            transform=ax.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
            fontsize=10)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '06_top10_exact_counts.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {OUTPUT_DIR / '06_top10_exact_counts.png'}")


def plot_algorithm_efficiency(results):
    """
    Plot 7: Overall algorithm efficiency (composite score).
    """
    print("Generating Plot 7: Algorithm Efficiency Comparison...")

    fig, ax = plt.subplots(figsize=(12, 7))

    # Normalize metrics (0-100 scale)
    exact_mem = results['requirement_a_exact']['statistics']['memory_kb']
    exact_time = results['requirement_a_exact']['statistics']['execution_time_ms']

    prob_mem = results['requirement_b_probabilistic']['statistics']['avg_memory_kb']
    prob_time = results['requirement_b_probabilistic']['statistics']['avg_execution_time_ms']
    prob_corr = results['requirement_b_probabilistic']['aggregated_metrics']['spearman_correlation']['mean']

    # Frequent-Count n=20 as example
    freq_20 = results['requirement_c_frequent_count']['results']['n_20']
    freq_mem = freq_20['statistics']['memory_kb']
    freq_time = freq_20['statistics']['execution_time_ms']

    freq_metrics_20 = results['requirement_d_comparison']['frequent_count_metrics']['20']
    freq_prec = 0
    for k_key in ['k_20', 'k_10']:
        if k_key in freq_metrics_20.get('top_k_metrics', {}):
            freq_prec = freq_metrics_20['top_k_metrics'][k_key]['precision']
            break

    # Create metrics
    algorithms = ['Exact\nCounter', 'Probabilistic\nCounter', 'Frequent-Count\n(n=20)']

    # Accuracy: 100%, correlation*100, precision*100
    accuracy = [100, prob_corr * 100, freq_prec * 100]

    # Memory efficiency: 100 - (mem/max_mem * 100)
    max_mem = max(exact_mem, prob_mem, freq_mem)
    memory_eff = [100 - (m/max_mem * 100) for m in [exact_mem, prob_mem, freq_mem]]

    # Speed efficiency: 100 - (time/max_time * 100)
    max_time = max(exact_time, prob_time, freq_time)
    speed_eff = [100 - (t/max_time * 100) for t in [exact_time, prob_time, freq_time]]

    x = np.arange(len(algorithms))
    width = 0.25

    bars1 = ax.bar(x - width, accuracy, width, label='Accuracy/Precision',
                   color='#2A9D8F', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x, memory_eff, width, label='Memory Efficiency',
                   color='#E76F51', alpha=0.8, edgecolor='black')
    bars3 = ax.bar(x + width, speed_eff, width, label='Speed Efficiency',
                   color='#F4A261', alpha=0.8, edgecolor='black')

    ax.set_ylabel('Score (0-100, higher is better)', fontweight='bold')
    ax.set_title('Algorithm Efficiency Comparison\n(Higher = Better)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms)
    ax.legend(loc='best')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim([0, 105])

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '07_algorithm_efficiency.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {OUTPUT_DIR / '07_algorithm_efficiency.png'}")


def plot_freq_memory_scaling_k(results):
    """
    Plot 8a: Frequent-Count memory scaling with k parameter.
    """
    print("Generating Plot 8a: Frequent-Count Memory Scaling with k...")

    fig, ax = plt.subplots(figsize=(10, 6))

    freq_data = results['requirement_c_frequent_count']['results']
    n_values = sorted([int(k.split('_')[1]) for k in freq_data.keys()])

    k_params = []
    memories = []

    for n in n_values:
        stats = freq_data[f'n_{n}']['statistics']
        k_params.append(stats['k_parameter'])
        memories.append(stats['memory_kb'])

    # Plot Memory vs k parameter
    ax.scatter(k_params, memories, s=100, alpha=0.7, c=n_values, cmap='plasma', edgecolors='black')
    ax.plot(k_params, memories, 'k--', alpha=0.3, linewidth=1)

    ax.set_xlabel('k Parameter (Max Counters)', fontweight='bold')
    ax.set_ylabel('Memory Usage (KB)', fontweight='bold')
    ax.set_title('Memory Scaling with k Parameter', fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '08a_freq_memory_scaling_k.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {OUTPUT_DIR / '08a_freq_memory_scaling_k.png'}")


def plot_freq_items_tracked_k(results):
    """
    Plot 8b: Items tracked vs k parameter for Frequent-Count.
    """
    print("Generating Plot 8b: Items Tracked vs k Parameter...")

    fig, ax = plt.subplots(figsize=(10, 6))

    freq_data = results['requirement_c_frequent_count']['results']
    n_values = sorted([int(k.split('_')[1]) for k in freq_data.keys()])

    k_params = []
    items_tracked = []

    for n in n_values:
        stats = freq_data[f'n_{n}']['statistics']
        k_params.append(stats['k_parameter'])
        items_tracked.append(stats['items_tracked'])

    # Plot Items tracked vs k parameter
    ax.scatter(k_params, items_tracked, s=100, alpha=0.7, c=n_values, cmap='plasma', edgecolors='black')
    ax.plot(k_params, items_tracked, 'k--', alpha=0.3, linewidth=1)
    ax.plot(k_params, [k-1 for k in k_params], 'r--', linewidth=2, label='Theoretical Max (k-1)')

    ax.set_xlabel('k Parameter (Max Counters)', fontweight='bold')
    ax.set_ylabel('Actual Items Tracked', fontweight='bold')
    ax.set_title('Items Tracked vs k Parameter', fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '08b_freq_items_tracked_k.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {OUTPUT_DIR / '08b_freq_items_tracked_k.png'}")


def plot_correlation_heatmap(results):
    """
    Plot 9: Correlation matrix/heatmap for algorithm comparison.
    """
    print("Generating Plot 9: Algorithm Performance Heatmap...")

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create performance matrix (normalized 0-1)
    metrics_names = ['Accuracy', 'Memory\nEfficiency', 'Speed', 'Top-10\nPrecision']
    algorithms = ['Exact', 'Probabilistic', 'Frequent\n(n=20)']

    # Normalize data
    exact_mem = results['requirement_a_exact']['statistics']['memory_kb']
    prob_mem = results['requirement_b_probabilistic']['statistics']['avg_memory_kb']
    freq_20_mem = results['requirement_c_frequent_count']['results']['n_20']['statistics']['memory_kb']

    exact_time = results['requirement_a_exact']['statistics']['execution_time_ms']
    prob_time = results['requirement_b_probabilistic']['statistics']['avg_execution_time_ms']
    freq_20_time = results['requirement_c_frequent_count']['results']['n_20']['statistics']['execution_time_ms']

    prob_corr = results['requirement_b_probabilistic']['aggregated_metrics']['spearman_correlation']['mean']

    freq_metrics_20 = results['requirement_d_comparison']['frequent_count_metrics']['20']
    freq_prec = 0
    for k_key in ['k_20', 'k_10']:
        if k_key in freq_metrics_20.get('top_k_metrics', {}):
            freq_prec = freq_metrics_20['top_k_metrics'][k_key]['precision']
            break

    # Create matrix (rows: algorithms, cols: metrics)
    max_mem = max(exact_mem, prob_mem, freq_20_mem)
    max_time = max(exact_time, prob_time, freq_20_time)

    performance = np.array([
        [1.0, 1 - exact_mem/max_mem, 1 - exact_time/max_time, 1.0],  # Exact
        [prob_corr, 1 - prob_mem/max_mem, 1 - prob_time/max_time, prob_corr],  # Probabilistic
        [freq_prec, 1 - freq_20_mem/max_mem, 1 - freq_20_time/max_time, freq_prec]  # Frequent
    ])

    im = ax.imshow(performance, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    # Set ticks
    ax.set_xticks(np.arange(len(metrics_names)))
    ax.set_yticks(np.arange(len(algorithms)))
    ax.set_xticklabels(metrics_names)
    ax.set_yticklabels(algorithms)

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    for i in range(len(algorithms)):
        for j in range(len(metrics_names)):
            text = ax.text(j, i, f'{performance[i, j]:.2f}',
                          ha="center", va="center", color="black", fontweight='bold', fontsize=11)

    ax.set_title('Algorithm Performance Heatmap\n(1.0 = Best, 0.0 = Worst)', fontweight='bold', fontsize=14)
    fig.colorbar(im, ax=ax, label='Normalized Performance Score')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '09_performance_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {OUTPUT_DIR / '09_performance_heatmap.png'}")


def plot_cumulative_distribution(results):
    """
    Plot 10: Cumulative distribution of cast member frequencies.
    """
    print("Generating Plot 10: Cumulative Distribution...")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Load exact counts from CSV (need all items, not just top-100)
    from data_utils import parse_cast_from_csv
    from collections import Counter

    cast_members = parse_cast_from_csv('amazon_prime_titles.csv')
    exact_counts = Counter(cast_members)
    frequencies = sorted(exact_counts.values(), reverse=True)

    # Calculate cumulative sum
    total = sum(frequencies)
    cumulative = np.cumsum(frequencies) / total * 100

    # Plot
    x = np.arange(1, len(frequencies) + 1)
    ax.plot(x, cumulative, linewidth=2, color='#2E86AB')
    ax.axhline(y=50, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='50%')
    ax.axhline(y=80, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='80%')
    ax.axhline(y=95, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='95%')

    # Find percentage markers
    items_for_50 = np.argmax(cumulative >= 50) + 1
    items_for_80 = np.argmax(cumulative >= 80) + 1

    ax.set_xlabel('Number of Cast Members (ranked by frequency)', fontweight='bold')
    ax.set_ylabel('Cumulative Percentage of Occurrences (%)', fontweight='bold')
    ax.set_title('Cumulative Distribution of Cast Member Frequencies\n(Shows concentration in top items)', fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, linestyle='--')

    # Add annotation
    ax.text(0.05, 0.95, f'Top {items_for_50} items = 50% of data\nTop {items_for_80} items = 80% of data',
            transform=ax.transAxes, ha='left', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=10, fontweight='bold')

    # Use log scale for x-axis if many items
    if len(frequencies) > 100:
        ax.set_xscale('log')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '10_cumulative_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {OUTPUT_DIR / '10_cumulative_distribution.png'}")


def plot_confidence_intervals_prob(results):
    """
    Plot 11: Confidence intervals for probabilistic counter metrics across trials.
    """
    print("Generating Plot 11: Confidence Intervals...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Get aggregated metrics
    agg = results['requirement_b_probabilistic']['aggregated_metrics']

    # Metric names and values
    metrics = ['Absolute\nError', 'Relative\nError (%)', 'Spearman\nCorrelation']

    abs_error = agg['absolute_error_mean']
    rel_error = agg['relative_error_mean']
    spear_corr = agg['spearman_correlation']

    means = [abs_error['mean'], rel_error['mean'] * 100, spear_corr['mean']]
    ci_lowers = [abs_error['ci_95_lower'], rel_error['ci_95_lower'] * 100, spear_corr['ci_95_lower']]
    ci_uppers = [abs_error['ci_95_upper'], rel_error['ci_95_upper'] * 100, spear_corr['ci_95_upper']]

    # Calculate error bars
    yerr_lower = [means[i] - ci_lowers[i] for i in range(3)]
    yerr_upper = [ci_uppers[i] - means[i] for i in range(3)]

    # Plot 11a: Confidence intervals
    x = np.arange(len(metrics))
    ax1.errorbar(x, means, yerr=[yerr_lower, yerr_upper], fmt='o', markersize=10,
                 capsize=10, capthick=2, linewidth=2, color='#2E86AB',
                 ecolor='#A23B72', label='95% CI')
    ax1.scatter(x, means, s=100, color='#F18F01', zorder=5, edgecolors='black', linewidth=1.5)

    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.set_ylabel('Value', fontweight='bold')
    ax1.set_title('Probabilistic Counter: 95% Confidence Intervals\n(20 trials)', fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # Plot 11b: Mean ± MAD
    mads = [abs_error['mad'], rel_error['mad'] * 100, spear_corr['mad']]

    ax2.bar(x, means, alpha=0.6, color='#2E86AB', edgecolor='black', label='Mean')
    ax2.errorbar(x, means, yerr=mads, fmt='none', capsize=10, capthick=2,
                 linewidth=2, ecolor='#E63946', label='±MAD (robust)')

    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics)
    ax2.set_ylabel('Value', fontweight='bold')
    ax2.set_title('Probabilistic Counter: Mean ± MAD\n(Median Absolute Deviation)', fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '11_confidence_intervals_prob.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {OUTPUT_DIR / '11_confidence_intervals_prob.png'}")


def plot_top_k_overlap_analysis(results):
    """
    Plot 12: Top-K overlap analysis for Frequent-Count algorithm.
    """
    print("Generating Plot 12: Top-K Overlap Analysis...")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Get overlap data for different k values
    freq_metrics = results['requirement_d_comparison']['frequent_count_metrics']
    n_values = sorted([int(k) for k in freq_metrics.keys()])

    jaccard_similarities = []
    overlap_percentages = []

    for n in n_values:
        metrics = freq_metrics[str(n)]
        # Try to find top-k overlap for k=n
        for k_key in [f'k_{n}', 'k_10', 'k_5']:
            if k_key in metrics.get('top_k_overlap', {}):
                overlap = metrics['top_k_overlap'][k_key]
                jaccard_similarities.append(overlap['jaccard_similarity'])
                overlap_percentages.append(overlap['overlap_pct'])
                break
        else:
            jaccard_similarities.append(0)
            overlap_percentages.append(0)

    # Plot both metrics
    ax.plot(n_values, jaccard_similarities, marker='o', linewidth=2.5, markersize=8,
            color='#2A9D8F', label='Jaccard Similarity')
    ax.plot(n_values, [op/100 for op in overlap_percentages], marker='s', linewidth=2.5, markersize=8,
            color='#E76F51', label='Overlap (fraction)')

    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(y=0.9, color='gray', linestyle=':', linewidth=1, alpha=0.5, label='90% threshold')

    ax.set_xlabel('n (Top-n Items to Find)', fontweight='bold')
    ax.set_ylabel('Similarity / Overlap Score', fontweight='bold')
    ax.set_title('Frequent-Count: Top-K Overlap with Exact Results', fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([-0.05, 1.05])

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '12_top_k_overlap_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {OUTPUT_DIR / '12_top_k_overlap_analysis.png'}")


def generate_all_plots():
    """
    Generate all visualization plots.
    """
    print("\n" + "="*70)
    print(" " * 20 + "GENERATING VISUALIZATIONS")
    print("="*70 + "\n")

    # Load results
    print("Loading experiment results...")
    results = load_results()
    print("  ✓ Results loaded successfully\n")

    # Generate all plots
    plot_memory_comparison(results)  # Plot 1: Frequent-Count memory scaling
    plot_execution_time_comparison(results)  # Plot 2: Execution time

    # Probabilistic counter plots (3a-3d)
    plot_prob_absolute_error(results)
    plot_prob_relative_error(results)
    plot_prob_rank_correlation(results)
    plot_prob_error_distribution(results)

    # Frequent-Count precision/recall plots (4a-4b)
    plot_freq_precision_recall(results)
    plot_freq_error_analysis(results)

    plot_memory_vs_accuracy_tradeoff(results)  # Plot 5
    plot_top10_comparison(results)  # Plot 6
    plot_algorithm_efficiency(results)  # Plot 7

    # Frequent-Count k parameter plots (8a-8b)
    plot_freq_memory_scaling_k(results)
    plot_freq_items_tracked_k(results)

    plot_correlation_heatmap(results)  # Plot 9

    plot_cumulative_distribution(results)  # Plot 10
    plot_confidence_intervals_prob(results)  # Plot 11
    plot_top_k_overlap_analysis(results)  # Plot 12

    print("\n" + "="*70)
    print("ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
    print("="*70)
    print(f"\nPlots saved to: {OUTPUT_DIR.absolute()}/")
    print("\nGenerated plots:")
    plot_files = sorted(OUTPUT_DIR.glob('*.png'))
    for i, plot_file in enumerate(plot_files, 1):
        print(f"  {i}. {plot_file.name}")
    print("\nThese plots are ready to include in your report!")


if __name__ == "__main__":
    generate_all_plots()
