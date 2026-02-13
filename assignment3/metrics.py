"""
Metrics for evaluating and comparing counting algorithms.
"""

import numpy as np
from typing import Dict, List, Tuple
from scipy.stats import spearmanr, kendalltau


class MetricsCalculator:
    """
    Calculate various metrics to compare approximate algorithms against exact counts.
    """

    def __init__(self, exact_counts: Dict[str, int]):
        """
        Initialize metrics calculator with exact counts as ground truth.

        Args:
            exact_counts: Dictionary of item -> exact_count
        """
        self.exact_counts = exact_counts

    def calculate_errors(self, approximate_counts: Dict[str, float]) -> dict:
        """
        Calculate absolute and relative errors for approximate counts.

        Args:
            approximate_counts: Dictionary of item -> approximate_count

        Returns:
            Dictionary with error statistics
        """
        absolute_errors = []
        relative_errors = []

        # Calculate errors for items in approximate counts
        for item, approx_count in approximate_counts.items():
            exact_count = self.exact_counts.get(item, 0)

            if exact_count > 0:
                abs_error = abs(approx_count - exact_count)
                rel_error = abs_error / exact_count

                absolute_errors.append(abs_error)
                relative_errors.append(rel_error)

        if not absolute_errors:
            return {
                'absolute_error_min': 0,
                'absolute_error_max': 0,
                'absolute_error_mean': 0,
                'absolute_error_median': 0,
                'absolute_error_std': 0,
                'relative_error_min': 0,
                'relative_error_max': 0,
                'relative_error_mean': 0,
                'relative_error_median': 0,
                'relative_error_std': 0,
                'num_items_compared': 0
            }

        return {
            'absolute_error_min': float(np.min(absolute_errors)),
            'absolute_error_max': float(np.max(absolute_errors)),
            'absolute_error_mean': float(np.mean(absolute_errors)),
            'absolute_error_median': float(np.median(absolute_errors)),
            'absolute_error_std': float(np.std(absolute_errors)),
            'relative_error_min': float(np.min(relative_errors)),
            'relative_error_max': float(np.max(relative_errors)),
            'relative_error_mean': float(np.mean(relative_errors)),
            'relative_error_median': float(np.median(relative_errors)),
            'relative_error_std': float(np.std(relative_errors)),
            'num_items_compared': len(absolute_errors)
        }

    def calculate_top_k_metrics(self, approximate_top_k: List[Tuple[str, float]],
                                k: int) -> dict:
        """
        Calculate metrics for top-k identification.

        Args:
            approximate_top_k: List of (item, count) from approximate algorithm
            k: Number of top items

        Returns:
            Dictionary with top-k metrics
        """
        # Get exact top-k
        exact_top_k = sorted(self.exact_counts.items(), key=lambda x: x[1], reverse=True)[:k]
        exact_top_k_items = set(item for item, _ in exact_top_k)

        # Get approximate top-k items
        approx_top_k_items = set(item for item, _ in approximate_top_k[:k])

        # Calculate precision, recall, F1
        true_positives = len(exact_top_k_items & approx_top_k_items)
        false_positives = len(approx_top_k_items - exact_top_k_items)
        false_negatives = len(exact_top_k_items - approx_top_k_items)

        precision = true_positives / k if k > 0 else 0
        recall = true_positives / k if k > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # Jaccard similarity
        jaccard = len(exact_top_k_items & approx_top_k_items) / len(exact_top_k_items | approx_top_k_items)

        return {
            'k': k,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'jaccard_similarity': jaccard,
            'items_matched': true_positives,
            'items_total': k
        }

    def calculate_median_absolute_deviation(self, values: List[float]) -> float:
        """
        Calculate Median Absolute Deviation (MAD) - robust measure of variability.

        Args:
            values: List of numeric values

        Returns:
            Median absolute deviation
        """
        if not values:
            return 0.0

        median = np.median(values)
        abs_deviations = [abs(v - median) for v in values]
        return float(np.median(abs_deviations))

    def calculate_value_accuracy(self, approximate_counts: Dict[str, float]) -> dict:
        """
        Calculate percentage of items with exact count match.

        Args:
            approximate_counts: Dictionary of item -> approximate_count

        Returns:
            Dictionary with value accuracy metrics
        """
        if not approximate_counts:
            return {
                'value_accuracy_pct': 0.0,
                'exact_matches': 0,
                'total_items': 0
            }

        exact_matches = 0
        total_items = 0

        for item, approx_count in approximate_counts.items():
            exact_count = self.exact_counts.get(item, 0)
            total_items += 1

            # Allow small tolerance for floating point comparison
            if abs(approx_count - exact_count) < 1e-6:
                exact_matches += 1

        accuracy_pct = (exact_matches / total_items * 100) if total_items > 0 else 0.0

        return {
            'value_accuracy_pct': float(accuracy_pct),
            'exact_matches': exact_matches,
            'total_items': total_items
        }

    def calculate_top_k_overlap(self, approximate_top_k: List[Tuple[str, float]], k: int) -> dict:
        """
        Calculate Jaccard similarity and overlap metrics for top-k sets.

        Args:
            approximate_top_k: List of (item, count) from approximate algorithm
            k: Number of top items

        Returns:
            Dictionary with overlap metrics
        """
        # Get exact top-k
        exact_top_k = sorted(self.exact_counts.items(), key=lambda x: x[1], reverse=True)[:k]
        exact_top_k_items = set(item for item, _ in exact_top_k)

        # Get approximate top-k items
        approx_top_k_items = set(item for item, _ in approximate_top_k[:k])

        # Calculate set operations
        intersection = exact_top_k_items & approx_top_k_items
        union = exact_top_k_items | approx_top_k_items

        jaccard = len(intersection) / len(union) if len(union) > 0 else 0.0
        overlap_pct = len(intersection) / k * 100 if k > 0 else 0.0

        return {
            'jaccard_similarity': float(jaccard),
            'overlap_count': len(intersection),
            'overlap_pct': float(overlap_pct),
            'exact_only': len(exact_top_k_items - approx_top_k_items),
            'approx_only': len(approx_top_k_items - exact_top_k_items)
        }

    def calculate_mean_squared_error(self, approximate_counts: Dict[str, float]) -> float:
        """
        Calculate Mean Squared Error between approximate and exact counts.

        Args:
            approximate_counts: Dictionary of item -> approximate_count

        Returns:
            Mean squared error value
        """
        squared_errors = []

        # Calculate MSE for items in both dictionaries
        all_items = set(self.exact_counts.keys()) | set(approximate_counts.keys())

        for item in all_items:
            exact_count = self.exact_counts.get(item, 0)
            approx_count = approximate_counts.get(item, 0)
            squared_errors.append((approx_count - exact_count) ** 2)

        return float(np.mean(squared_errors)) if squared_errors else 0.0

    def calculate_confidence_intervals(self, values: List[float], confidence: float = 0.95) -> dict:
        """
        Calculate confidence intervals for a list of values.

        Args:
            values: List of numeric values
            confidence: Confidence level (default 0.95 for 95% CI)

        Returns:
            Dictionary with confidence interval bounds
        """
        from scipy import stats

        if not values or len(values) < 2:
            return {
                'mean': 0.0,
                'std_error': 0.0,
                'ci_lower': 0.0,
                'ci_upper': 0.0,
                'confidence_level': confidence
            }

        mean = np.mean(values)
        std_error = stats.sem(values)  # Standard error of the mean

        # Calculate confidence interval
        ci = stats.t.interval(confidence, len(values) - 1, loc=mean, scale=std_error)

        return {
            'mean': float(mean),
            'std_error': float(std_error),
            'ci_lower': float(ci[0]),
            'ci_upper': float(ci[1]),
            'confidence_level': confidence,
            'margin_of_error': float(ci[1] - mean)
        }

    def calculate_rank_correlation(self, approximate_counts: Dict[str, float]) -> dict:
        """
        Calculate rank correlation between exact and approximate counts.

        Args:
            approximate_counts: Dictionary of item -> approximate_count

        Returns:
            Dictionary with correlation metrics
        """
        # Get common items
        common_items = set(self.exact_counts.keys()) & set(approximate_counts.keys())

        if len(common_items) < 2:
            return {
                'spearman_correlation': 0,
                'spearman_pvalue': 1,
                'kendall_tau': 0,
                'kendall_pvalue': 1,
                'num_items': len(common_items)
            }

        # Create aligned lists
        exact_ranks = []
        approx_ranks = []

        for item in common_items:
            exact_ranks.append(self.exact_counts[item])
            approx_ranks.append(approximate_counts[item])

        # Calculate correlations
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            spearman_corr, spearman_p = spearmanr(exact_ranks, approx_ranks)
            kendall_corr, kendall_p = kendalltau(exact_ranks, approx_ranks)

        # Handle NaN values (when input is constant)
        if np.isnan(spearman_corr):
            spearman_corr = 0.0
            spearman_p = 1.0
        if np.isnan(kendall_corr):
            kendall_corr = 0.0
            kendall_p = 1.0

        return {
            'spearman_correlation': float(spearman_corr),
            'spearman_pvalue': float(spearman_p),
            'kendall_tau': float(kendall_corr),
            'kendall_pvalue': float(kendall_p),
            'num_items': len(common_items)
        }

    def compare_rankings(self, approximate_top_k: List[Tuple[str, float]], k: int) -> dict:
        """
        Compare the ordering of top-k items between exact and approximate.

        Args:
            approximate_top_k: List of (item, count) from approximate algorithm
            k: Number of items to compare

        Returns:
            Dictionary with ranking comparison metrics
        """
        # Get exact top-k
        exact_top_k = sorted(self.exact_counts.items(), key=lambda x: x[1], reverse=True)[:k]

        # Items in exact top-k
        exact_items = [item for item, _ in exact_top_k]
        approx_items = [item for item, _ in approximate_top_k[:k]]

        # Check if same items in same order
        same_items = exact_items == approx_items

        # Count position differences for matching items
        position_diffs = []
        for exact_pos, item in enumerate(exact_items):
            if item in approx_items:
                approx_pos = approx_items.index(item)
                position_diffs.append(abs(exact_pos - approx_pos))

        avg_position_diff = np.mean(position_diffs) if position_diffs else k

        return {
            'same_items_same_order': same_items,
            'avg_position_difference': float(avg_position_diff),
            'max_position_difference': float(np.max(position_diffs)) if position_diffs else k,
            'items_in_correct_position': sum(1 for d in position_diffs if d == 0)
        }

    def comprehensive_evaluation(self, approximate_counts: Dict[str, float],
                                 approximate_top_k: List[Tuple[str, float]],
                                 k_values: List[int] = None) -> dict:
        """
        Perform comprehensive evaluation with all metrics.

        Args:
            approximate_counts: Dictionary of item -> approximate_count
            approximate_top_k: List of (item, count) tuples
            k_values: List of k values to evaluate top-k metrics

        Returns:
            Dictionary with all metrics
        """
        if k_values is None:
            k_values = [5, 10, 20, 50, 100]

        results = {
            'errors': self.calculate_errors(approximate_counts),
            'rank_correlation': self.calculate_rank_correlation(approximate_counts),
            'value_accuracy': self.calculate_value_accuracy(approximate_counts),
            'mean_squared_error': self.calculate_mean_squared_error(approximate_counts),
            'top_k_metrics': {},
            'ranking_comparison': {},
            'top_k_overlap': {}
        }

        # Calculate top-k metrics for different k values
        for k in k_values:
            if len(approximate_top_k) >= k:
                results['top_k_metrics'][f'k_{k}'] = self.calculate_top_k_metrics(approximate_top_k, k)
                results['ranking_comparison'][f'k_{k}'] = self.compare_rankings(approximate_top_k, k)
                results['top_k_overlap'][f'k_{k}'] = self.calculate_top_k_overlap(approximate_top_k, k)

        return results


def aggregate_trial_metrics(trial_metrics: List[dict]) -> dict:
    """
    Aggregate metrics from multiple trials (for probabilistic counter).

    Args:
        trial_metrics: List of metrics dictionaries from individual trials

    Returns:
        Aggregated metrics with mean, std, CI, and MAD
    """
    from scipy import stats

    if not trial_metrics:
        return {}

    # Collect all error values
    abs_errors_mean = [m['errors']['absolute_error_mean'] for m in trial_metrics]
    rel_errors_mean = [m['errors']['relative_error_mean'] for m in trial_metrics]

    # Collect top-k precision/recall for k=10
    precisions = []
    recalls = []
    f1_scores = []

    for m in trial_metrics:
        if 'k_10' in m.get('top_k_metrics', {}):
            precisions.append(m['top_k_metrics']['k_10']['precision'])
            recalls.append(m['top_k_metrics']['k_10']['recall'])
            f1_scores.append(m['top_k_metrics']['k_10']['f1_score'])

    # Collect correlations
    spearman_corrs = [m['rank_correlation']['spearman_correlation'] for m in trial_metrics]

    # Helper function to calculate CI and MAD
    def calc_stats(values):
        mean_val = float(np.mean(values))
        std_val = float(np.std(values))
        median_val = float(np.median(values))

        # MAD
        mad_val = float(np.median([abs(v - median_val) for v in values]))

        # 95% Confidence Interval
        if len(values) >= 2:
            std_error = stats.sem(values)
            ci = stats.t.interval(0.95, len(values) - 1, loc=mean_val, scale=std_error)
            ci_lower = float(ci[0])
            ci_upper = float(ci[1])
        else:
            ci_lower = mean_val
            ci_upper = mean_val

        return {
            'mean': mean_val,
            'std': std_val,
            'median': median_val,
            'mad': mad_val,
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'ci_95_lower': ci_lower,
            'ci_95_upper': ci_upper
        }

    aggregated = {
        'num_trials': len(trial_metrics),
        'absolute_error_mean': calc_stats(abs_errors_mean),
        'relative_error_mean': calc_stats(rel_errors_mean),
        'spearman_correlation': calc_stats(spearman_corrs)
    }

    if precisions:
        aggregated['top_10_precision'] = calc_stats(precisions)
        aggregated['top_10_recall'] = calc_stats(recalls)
        aggregated['top_10_f1'] = calc_stats(f1_scores)

    return aggregated


def print_metrics_summary(metrics: dict, algorithm_name: str):
    """
    Print a formatted summary of metrics.

    Args:
        metrics: Metrics dictionary
        algorithm_name: Name of the algorithm
    """
    print(f"\n{'='*60}")
    print(f"METRICS SUMMARY: {algorithm_name}")
    print(f"{'='*60}")

    if 'errors' in metrics:
        print("\nError Statistics:")
        print(f"  Absolute Error:")
        print(f"    Mean:   {metrics['errors']['absolute_error_mean']:.2f}")
        print(f"    Median: {metrics['errors']['absolute_error_median']:.2f}")
        print(f"    Std:    {metrics['errors']['absolute_error_std']:.2f}")
        print(f"    Range:  [{metrics['errors']['absolute_error_min']:.2f}, {metrics['errors']['absolute_error_max']:.2f}]")
        print(f"\n  Relative Error:")
        print(f"    Mean:   {metrics['errors']['relative_error_mean']:.4f} ({metrics['errors']['relative_error_mean']*100:.2f}%)")
        print(f"    Median: {metrics['errors']['relative_error_median']:.4f} ({metrics['errors']['relative_error_median']*100:.2f}%)")
        print(f"    Std:    {metrics['errors']['relative_error_std']:.4f}")

    if 'rank_correlation' in metrics:
        print("\nRank Correlation:")
        print(f"  Spearman: {metrics['rank_correlation']['spearman_correlation']:.4f}")
        print(f"  Kendall:  {metrics['rank_correlation']['kendall_tau']:.4f}")

    if 'top_k_metrics' in metrics and 'k_10' in metrics['top_k_metrics']:
        print("\nTop-10 Identification:")
        tk = metrics['top_k_metrics']['k_10']
        print(f"  Precision: {tk['precision']:.2f}")
        print(f"  Recall:    {tk['recall']:.2f}")
        print(f"  F1 Score:  {tk['f1_score']:.2f}")
        print(f"  Matched:   {tk['items_matched']}/{tk['items_total']}")
