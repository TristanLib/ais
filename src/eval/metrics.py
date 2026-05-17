"""
Evaluation metrics for ship trajectory prediction and collision avoidance.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Union
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class TrajectoryMetrics:
    """Metrics for trajectory prediction evaluation."""

    def __init__(self, config: Dict):
        """Initialize trajectory metrics."""
        self.config = config
        eval_config = config.get('evaluation', {}).get('prediction', {})

        self.metrics = eval_config.get('metrics', ['ade', 'fde', 'rmse', 'mae'])
        self.horizons = eval_config.get('horizons', [5, 10, 20, 30])  # minutes
        self.confidence_intervals = eval_config.get('confidence_intervals', [0.95])

        # Geographic conversion
        self.meters_per_degree_lat = 111000  # approximately
        self.nmi_per_degree = 60  # nautical miles per degree

    def calculate_ade(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        Calculate Average Displacement Error (ADE).

        Args:
            predictions: Predicted trajectories (batch_size, time_steps, 2)
            targets: Ground truth trajectories (batch_size, time_steps, 2)

        Returns:
            ADE in meters
        """
        # Calculate displacement at each time step
        displacements = np.linalg.norm(predictions - targets, axis=-1)  # (batch_size, time_steps)

        # Average over time steps and batch
        ade = np.mean(displacements)

        return ade

    def calculate_fde(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        Calculate Final Displacement Error (FDE).

        Args:
            predictions: Predicted trajectories (batch_size, time_steps, 2)
            targets: Ground truth trajectories (batch_size, time_steps, 2)

        Returns:
            FDE in meters
        """
        # Final positions
        final_pred = predictions[:, -1, :]  # (batch_size, 2)
        final_true = targets[:, -1, :]      # (batch_size, 2)

        # Calculate final displacement
        final_displacements = np.linalg.norm(final_pred - final_true, axis=-1)  # (batch_size,)

        # Average over batch
        fde = np.mean(final_displacements)

        return fde

    def calculate_rmse(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Calculate Root Mean Square Error."""
        mse = mean_squared_error(targets.reshape(-1), predictions.reshape(-1))
        return np.sqrt(mse)

    def calculate_mae(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Calculate Mean Absolute Error."""
        return mean_absolute_error(targets.reshape(-1), predictions.reshape(-1))

    def calculate_hit_rate(self, predictions: np.ndarray, targets: np.ndarray,
                          threshold: float = 100.0) -> float:
        """
        Calculate hit rate (percentage of predictions within threshold).

        Args:
            predictions: Predicted trajectories
            targets: Ground truth trajectories
            threshold: Distance threshold in meters

        Returns:
            Hit rate (0-1)
        """
        displacements = np.linalg.norm(predictions - targets, axis=-1)
        hits = displacements <= threshold
        return np.mean(hits)

    def calculate_horizon_metrics(self, predictions: np.ndarray, targets: np.ndarray,
                                dt_minutes: int = 1) -> Dict[int, Dict[str, float]]:
        """
        Calculate metrics at different prediction horizons.

        Args:
            predictions: Predicted trajectories (batch_size, time_steps, 2)
            targets: Ground truth trajectories (batch_size, time_steps, 2)
            dt_minutes: Time step in minutes

        Returns:
            Dictionary of metrics by horizon
        """
        horizon_metrics = {}

        for horizon_minutes in self.horizons:
            horizon_step = min(horizon_minutes // dt_minutes, predictions.shape[1])

            if horizon_step <= 0:
                continue

            # Extract predictions and targets up to horizon
            pred_horizon = predictions[:, :horizon_step, :]
            target_horizon = targets[:, :horizon_step, :]

            # Calculate metrics
            metrics = {}
            if 'ade' in self.metrics:
                metrics['ade'] = self.calculate_ade(pred_horizon, target_horizon)
            if 'fde' in self.metrics:
                metrics['fde'] = self.calculate_fde(pred_horizon, target_horizon)
            if 'rmse' in self.metrics:
                metrics['rmse'] = self.calculate_rmse(pred_horizon, target_horizon)
            if 'mae' in self.metrics:
                metrics['mae'] = self.calculate_mae(pred_horizon, target_horizon)

            horizon_metrics[horizon_minutes] = metrics

        return horizon_metrics

    def calculate_confidence_intervals(self, values: np.ndarray,
                                     confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence intervals for metric values."""
        alpha = 1 - confidence
        lower = np.percentile(values, 100 * alpha / 2)
        upper = np.percentile(values, 100 * (1 - alpha / 2))
        return lower, upper

    def evaluate_model(self, predictions: np.ndarray, targets: np.ndarray,
                      dt_minutes: int = 1) -> Dict:
        """
        Comprehensive model evaluation.

        Args:
            predictions: Model predictions
            targets: Ground truth
            dt_minutes: Time step in minutes

        Returns:
            Complete evaluation results
        """
        results = {
            'overall_metrics': {},
            'horizon_metrics': {},
            'confidence_intervals': {}
        }

        # Overall metrics
        if 'ade' in self.metrics:
            results['overall_metrics']['ade'] = self.calculate_ade(predictions, targets)
        if 'fde' in self.metrics:
            results['overall_metrics']['fde'] = self.calculate_fde(predictions, targets)
        if 'rmse' in self.metrics:
            results['overall_metrics']['rmse'] = self.calculate_rmse(predictions, targets)
        if 'mae' in self.metrics:
            results['overall_metrics']['mae'] = self.calculate_mae(predictions, targets)

        # Horizon-specific metrics
        results['horizon_metrics'] = self.calculate_horizon_metrics(
            predictions, targets, dt_minutes
        )

        # Calculate confidence intervals for key metrics
        batch_size = predictions.shape[0]
        for confidence in self.confidence_intervals:
            ci_results = {}

            # Calculate per-sample ADE for confidence intervals
            if 'ade' in self.metrics:
                sample_ades = []
                for i in range(batch_size):
                    ade = self.calculate_ade(predictions[i:i+1], targets[i:i+1])
                    sample_ades.append(ade)

                ci_lower, ci_upper = self.calculate_confidence_intervals(
                    np.array(sample_ades), confidence
                )
                ci_results['ade'] = (ci_lower, ci_upper)

            results['confidence_intervals'][confidence] = ci_results

        return results


class CollisionAvoidanceMetrics:
    """Metrics for collision avoidance evaluation."""

    def __init__(self, config: Dict):
        """Initialize collision avoidance metrics."""
        self.config = config
        eval_config = config.get('evaluation', {}).get('avoidance', {})

        self.collision_threshold = eval_config.get('collision_threshold', 0.1)  # nautical miles
        self.near_miss_threshold = eval_config.get('near_miss_threshold', 0.2)  # nautical miles

        success_criteria = eval_config.get('success_criteria', {})
        self.max_collision_rate = success_criteria.get('max_collision_rate', 0.02)
        self.min_avg_cpa = success_criteria.get('min_avg_cpa', 0.2)

    def evaluate_scenario_batch(self, simulation_results: List[Dict]) -> Dict:
        """
        Evaluate batch of simulation scenarios.

        Args:
            simulation_results: List of simulation result dictionaries

        Returns:
            Aggregated evaluation metrics
        """
        n_scenarios = len(simulation_results)
        if n_scenarios == 0:
            return {}

        # Collision statistics
        collisions = sum(1 for result in simulation_results if result['collision_occurred'])
        collision_rate = collisions / n_scenarios

        # CPA statistics
        min_cpas = [result['min_cpa_achieved'] for result in simulation_results]
        avg_cpa = np.mean(min_cpas)
        median_cpa = np.median(min_cpas)
        std_cpa = np.std(min_cpas)

        # Near miss statistics
        near_misses = sum(1 for cpa in min_cpas
                         if self.collision_threshold < cpa <= self.near_miss_threshold)
        near_miss_rate = near_misses / n_scenarios

        # Safe encounters
        safe_encounters = sum(1 for cpa in min_cpas if cpa > self.near_miss_threshold)
        safe_encounter_rate = safe_encounters / n_scenarios

        # Success criteria
        meets_collision_criteria = collision_rate <= self.max_collision_rate
        meets_cpa_criteria = avg_cpa >= self.min_avg_cpa
        overall_success = meets_collision_criteria and meets_cpa_criteria

        return {
            'n_scenarios': n_scenarios,
            'collision_rate': collision_rate,
            'near_miss_rate': near_miss_rate,
            'safe_encounter_rate': safe_encounter_rate,
            'cpa_statistics': {
                'mean': avg_cpa,
                'median': median_cpa,
                'std': std_cpa,
                'min': np.min(min_cpas),
                'max': np.max(min_cpas)
            },
            'success_criteria': {
                'collision_rate_ok': meets_collision_criteria,
                'avg_cpa_ok': meets_cpa_criteria,
                'overall_success': overall_success
            }
        }

    def analyze_avoidance_actions(self, avoidance_results: List[Dict]) -> Dict:
        """Analyze effectiveness of avoidance actions."""
        if not avoidance_results:
            return {}

        # Action type distribution
        action_types = {}
        improvements = []
        costs = []
        success_rates = {}

        for result in avoidance_results:
            action_type = result['best_action'].action_type.value
            action_types[action_type] = action_types.get(action_type, 0) + 1

            improvements.append(result['improvement'])
            costs.append(result['cost'])

            if action_type not in success_rates:
                success_rates[action_type] = []
            success_rates[action_type].append(result['success'])

        # Calculate success rates by action type
        for action_type in success_rates:
            success_rates[action_type] = np.mean(success_rates[action_type])

        return {
            'action_distribution': action_types,
            'improvement_statistics': {
                'mean': np.mean(improvements),
                'std': np.std(improvements),
                'min': np.min(improvements),
                'max': np.max(improvements)
            },
            'cost_statistics': {
                'mean': np.mean(costs),
                'std': np.std(costs),
                'min': np.min(costs),
                'max': np.max(costs)
            },
            'success_rates_by_action': success_rates,
            'overall_success_rate': np.mean([r['success'] for r in avoidance_results])
        }


class StatisticalTesting:
    """Statistical significance testing for model comparisons."""

    def __init__(self, config: Dict):
        """Initialize statistical testing."""
        self.config = config
        stat_config = config.get('evaluation', {}).get('statistical', {})

        self.test_type = stat_config.get('test_type', 'paired_t')
        self.alpha = stat_config.get('alpha', 0.05)
        self.bootstrap_samples = stat_config.get('bootstrap_samples', 1000)

    def paired_t_test(self, metric_a: np.ndarray, metric_b: np.ndarray) -> Dict:
        """
        Perform paired t-test between two model metrics.

        Args:
            metric_a: Metrics from model A
            metric_b: Metrics from model B

        Returns:
            Test results dictionary
        """
        if len(metric_a) != len(metric_b):
            raise ValueError("Metrics arrays must have same length for paired test")

        # Perform paired t-test
        statistic, p_value = stats.ttest_rel(metric_a, metric_b)

        # Effect size (Cohen's d for paired samples)
        differences = metric_a - metric_b
        cohen_d = np.mean(differences) / np.std(differences)

        # Confidence interval for the difference
        diff_mean = np.mean(differences)
        diff_se = stats.sem(differences)
        ci_lower, ci_upper = stats.t.interval(
            1 - self.alpha, len(differences) - 1, diff_mean, diff_se
        )

        return {
            'test_type': 'paired_t_test',
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < self.alpha,
            'effect_size_cohen_d': cohen_d,
            'mean_difference': diff_mean,
            'confidence_interval': (ci_lower, ci_upper),
            'alpha': self.alpha
        }

    def bootstrap_test(self, metric_a: np.ndarray, metric_b: np.ndarray) -> Dict:
        """
        Perform bootstrap test for difference in means.

        Args:
            metric_a: Metrics from model A
            metric_b: Metrics from model B

        Returns:
            Bootstrap test results
        """
        # Observed difference
        observed_diff = np.mean(metric_a) - np.mean(metric_b)

        # Bootstrap resampling
        bootstrap_diffs = []
        for _ in range(self.bootstrap_samples):
            # Resample with replacement
            sample_a = np.random.choice(metric_a, size=len(metric_a), replace=True)
            sample_b = np.random.choice(metric_b, size=len(metric_b), replace=True)

            diff = np.mean(sample_a) - np.mean(sample_b)
            bootstrap_diffs.append(diff)

        bootstrap_diffs = np.array(bootstrap_diffs)

        # P-value (two-tailed test)
        p_value = 2 * min(
            np.mean(bootstrap_diffs <= 0),
            np.mean(bootstrap_diffs >= 0)
        )

        # Confidence interval
        ci_lower = np.percentile(bootstrap_diffs, 100 * self.alpha / 2)
        ci_upper = np.percentile(bootstrap_diffs, 100 * (1 - self.alpha / 2))

        return {
            'test_type': 'bootstrap_test',
            'observed_difference': observed_diff,
            'p_value': p_value,
            'significant': p_value < self.alpha,
            'confidence_interval': (ci_lower, ci_upper),
            'bootstrap_samples': self.bootstrap_samples,
            'alpha': self.alpha
        }

    def compare_models(self, metrics_a: Dict[str, np.ndarray],
                      metrics_b: Dict[str, np.ndarray],
                      model_names: Tuple[str, str] = ("Model A", "Model B")) -> Dict:
        """
        Compare two models across multiple metrics.

        Args:
            metrics_a: Dictionary of metrics for model A
            metrics_b: Dictionary of metrics for model B
            model_names: Names of the models being compared

        Returns:
            Comprehensive comparison results
        """
        results = {
            'model_names': model_names,
            'metric_comparisons': {},
            'summary': {}
        }

        significant_metrics = []

        for metric_name in metrics_a:
            if metric_name not in metrics_b:
                continue

            metric_a = metrics_a[metric_name]
            metric_b = metrics_b[metric_name]

            # Perform statistical test
            if self.test_type == 'paired_t':
                test_result = self.paired_t_test(metric_a, metric_b)
            elif self.test_type == 'bootstrap':
                test_result = self.bootstrap_test(metric_a, metric_b)
            else:
                logger.warning(f"Unknown test type: {self.test_type}")
                continue

            results['metric_comparisons'][metric_name] = test_result

            if test_result['significant']:
                significant_metrics.append(metric_name)

        # Summary
        results['summary'] = {
            'total_metrics_compared': len(results['metric_comparisons']),
            'significant_differences': len(significant_metrics),
            'significant_metrics': significant_metrics,
            'alpha_level': self.alpha
        }

        return results


def create_evaluation_report(trajectory_results: Dict, avoidance_results: Dict,
                           statistical_results: Dict = None) -> str:
    """Create comprehensive evaluation report."""
    report_lines = []

    report_lines.append("# Ship Trajectory Prediction and Collision Avoidance Evaluation Report")
    report_lines.append("")

    # Trajectory prediction results
    if trajectory_results:
        report_lines.append("## Trajectory Prediction Results")
        report_lines.append("")

        overall = trajectory_results.get('overall_metrics', {})
        for metric, value in overall.items():
            report_lines.append(f"- **{metric.upper()}**: {value:.4f}")

        report_lines.append("")

        # Horizon results
        horizon_metrics = trajectory_results.get('horizon_metrics', {})
        if horizon_metrics:
            report_lines.append("### Performance by Prediction Horizon")
            report_lines.append("")
            report_lines.append("| Horizon (min) | ADE | FDE | RMSE | MAE |")
            report_lines.append("|---------------|-----|-----|------|-----|")

            for horizon in sorted(horizon_metrics.keys()):
                metrics = horizon_metrics[horizon]
                ade = metrics.get('ade', 0)
                fde = metrics.get('fde', 0)
                rmse = metrics.get('rmse', 0)
                mae = metrics.get('mae', 0)
                report_lines.append(f"| {horizon:2d} | {ade:.3f} | {fde:.3f} | {rmse:.3f} | {mae:.3f} |")

        report_lines.append("")

    # Collision avoidance results
    if avoidance_results:
        report_lines.append("## Collision Avoidance Results")
        report_lines.append("")

        collision_rate = avoidance_results.get('collision_rate', 0)
        near_miss_rate = avoidance_results.get('near_miss_rate', 0)
        safe_rate = avoidance_results.get('safe_encounter_rate', 0)

        report_lines.append(f"- **Collision Rate**: {collision_rate:.1%}")
        report_lines.append(f"- **Near Miss Rate**: {near_miss_rate:.1%}")
        report_lines.append(f"- **Safe Encounter Rate**: {safe_rate:.1%}")

        cpa_stats = avoidance_results.get('cpa_statistics', {})
        if cpa_stats:
            report_lines.append("")
            report_lines.append("### CPA Statistics")
            report_lines.append(f"- **Mean CPA**: {cpa_stats.get('mean', 0):.3f} nmi")
            report_lines.append(f"- **Median CPA**: {cpa_stats.get('median', 0):.3f} nmi")
            report_lines.append(f"- **Min CPA**: {cpa_stats.get('min', 0):.3f} nmi")

        report_lines.append("")

    # Statistical significance
    if statistical_results:
        report_lines.append("## Statistical Significance Testing")
        report_lines.append("")

        summary = statistical_results.get('summary', {})
        significant_metrics = summary.get('significant_metrics', [])

        if significant_metrics:
            report_lines.append(f"**Significant improvements found in**: {', '.join(significant_metrics)}")
        else:
            report_lines.append("**No statistically significant differences found**")

        report_lines.append("")

    return "\n".join(report_lines)


if __name__ == "__main__":
    # Example usage
    config = {
        'evaluation': {
            'prediction': {
                'metrics': ['ade', 'fde', 'rmse', 'mae'],
                'horizons': [5, 10, 20, 30],
                'confidence_intervals': [0.95]
            },
            'avoidance': {
                'collision_threshold': 0.1,
                'near_miss_threshold': 0.2,
                'success_criteria': {
                    'max_collision_rate': 0.02,
                    'min_avg_cpa': 0.2
                }
            },
            'statistical': {
                'test_type': 'paired_t',
                'alpha': 0.05,
                'bootstrap_samples': 1000
            }
        }
    }

    # Test trajectory metrics
    traj_metrics = TrajectoryMetrics(config)

    # Sample data
    batch_size, time_steps, coords = 100, 30, 2
    predictions = np.random.randn(batch_size, time_steps, coords)
    targets = np.random.randn(batch_size, time_steps, coords)

    results = traj_metrics.evaluate_model(predictions, targets)
    print(f"ADE: {results['overall_metrics']['ade']:.4f}")
    print(f"FDE: {results['overall_metrics']['fde']:.4f}")

    # Test statistical comparison
    stat_test = StatisticalTesting(config)

    metric_a = np.random.normal(1.0, 0.2, 100)
    metric_b = np.random.normal(0.8, 0.2, 100)

    comparison = stat_test.compare_models(
        {'ade': metric_a}, {'ade': metric_b},
        ("Baseline", "Transformer")
    )

    print(f"Statistical significance: {comparison['metric_comparisons']['ade']['significant']}")
    print(f"P-value: {comparison['metric_comparisons']['ade']['p_value']:.4f}")
