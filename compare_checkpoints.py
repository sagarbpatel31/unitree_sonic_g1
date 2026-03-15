#!/usr/bin/env python3
"""
Multi-checkpoint comparison tool for Unitree G1 controllers.

This script compares multiple policy checkpoints across all test suites,
generating comprehensive comparison reports and visualizations.
"""

import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from omegaconf import OmegaConf
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from evaluate_policy import PolicyEvaluator
from sonic_g1.eval.metrics import MetricsTracker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CheckpointComparator:
    """
    Tool for comparing multiple policy checkpoints across test suites.
    """

    def __init__(self, config: OmegaConf):
        """
        Initialize checkpoint comparator.

        Args:
            config: Comparison configuration
        """
        self.config = config
        self.evaluator = PolicyEvaluator(config)
        self.metrics_tracker = MetricsTracker(config.metrics)

        # Results storage
        self.comparison_results = {}

        logger.info("Initialized CheckpointComparator")

    def compare_checkpoints(self,
                          checkpoint_configs: List[Dict[str, str]],
                          output_dir: str) -> Dict[str, Any]:
        """
        Compare multiple checkpoints across all test suites.

        Args:
            checkpoint_configs: List of dicts with 'name', 'path', 'description' keys
            output_dir: Directory to save comparison results

        Returns:
            Comprehensive comparison results
        """
        logger.info(f"Comparing {len(checkpoint_configs)} checkpoints")

        # Evaluate each checkpoint
        all_results = {}
        for checkpoint_config in checkpoint_configs:
            checkpoint_name = checkpoint_config['name']
            checkpoint_path = checkpoint_config['path']

            logger.info(f"Evaluating checkpoint: {checkpoint_name}")

            # Run evaluation
            checkpoint_results = self.evaluator.run_comprehensive_evaluation(
                checkpoint_path=checkpoint_path,
                output_dir=f"{output_dir}/{checkpoint_name}",
                policy_name=checkpoint_name
            )

            all_results[checkpoint_name] = checkpoint_results

        # Generate comparison analysis
        comparison = self._analyze_comparison(all_results, checkpoint_configs)

        # Save comparison results
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save detailed comparison
        comparison_file = output_path / "checkpoint_comparison.json"
        self._save_comparison_results(comparison, comparison_file)

        # Generate comparison report
        report_file = output_path / "comparison_report.txt"
        self._generate_comparison_report(comparison, report_file)

        # Generate comparison plots
        self._generate_comparison_plots(comparison, output_path)

        # Generate summary table
        summary_file = output_path / "comparison_summary.csv"
        self._generate_summary_table(comparison, summary_file)

        logger.info(f"Comparison completed. Results saved to: {output_path}")

        return comparison

    def _analyze_comparison(self,
                          all_results: Dict[str, Any],
                          checkpoint_configs: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Analyze comparison across checkpoints.

        Args:
            all_results: Results for all checkpoints
            checkpoint_configs: Checkpoint configuration information

        Returns:
            Comparison analysis
        """
        comparison = {
            'checkpoints': {config['name']: config for config in checkpoint_configs},
            'results': all_results,
            'metrics_comparison': {},
            'rankings': {},
            'statistical_analysis': {}
        }

        # Extract metrics for comparison
        test_suites = set()
        for results in all_results.values():
            test_suites.update(results.keys())

        # Compare metrics across test suites
        for test_suite in test_suites:
            suite_comparison = {}
            checkpoint_names = []

            for checkpoint_name, results in all_results.items():
                if test_suite in results:
                    suite_metrics = results[test_suite]['metrics']
                    suite_comparison[checkpoint_name] = suite_metrics
                    checkpoint_names.append(checkpoint_name)

            if suite_comparison:
                comparison['metrics_comparison'][test_suite] = suite_comparison
                comparison['rankings'][test_suite] = self._rank_checkpoints(
                    suite_comparison
                )

        # Overall rankings
        comparison['overall_rankings'] = self._compute_overall_rankings(
            comparison['metrics_comparison']
        )

        # Statistical analysis
        comparison['statistical_analysis'] = self._perform_statistical_analysis(
            comparison['metrics_comparison']
        )

        return comparison

    def _rank_checkpoints(self, suite_metrics: Dict[str, Dict[str, float]]) -> Dict[str, List[str]]:
        """
        Rank checkpoints for a test suite across different metrics.

        Args:
            suite_metrics: Metrics for each checkpoint in test suite

        Returns:
            Rankings for each metric
        """
        rankings = {}

        # Define metrics and their optimization direction
        metric_directions = {
            'success_rate': 'higher',
            'fall_rate': 'lower',
            'root_tracking_error': 'lower',
            'joint_tracking_error': 'lower',
            'action_smoothness': 'lower',
            'energy_usage': 'lower',
            'command_tracking_quality': 'higher',
            'overall_score': 'higher'
        }

        # Rank each metric
        for metric, direction in metric_directions.items():
            metric_values = {}
            for checkpoint, metrics in suite_metrics.items():
                if metric in metrics:
                    metric_values[checkpoint] = metrics[metric]

            if metric_values:
                if direction == 'higher':
                    sorted_checkpoints = sorted(metric_values.items(),
                                              key=lambda x: x[1], reverse=True)
                else:
                    sorted_checkpoints = sorted(metric_values.items(),
                                              key=lambda x: x[1])

                rankings[metric] = [checkpoint for checkpoint, _ in sorted_checkpoints]

        return rankings

    def _compute_overall_rankings(self, metrics_comparison: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute overall rankings across all test suites.

        Args:
            metrics_comparison: Metrics comparison across test suites

        Returns:
            Overall scores for each checkpoint
        """
        checkpoint_scores = {}

        # Collect all checkpoints
        all_checkpoints = set()
        for suite_metrics in metrics_comparison.values():
            all_checkpoints.update(suite_metrics.keys())

        # Compute overall scores
        for checkpoint in all_checkpoints:
            scores = []

            for test_suite, suite_metrics in metrics_comparison.items():
                if checkpoint in suite_metrics:
                    overall_score = suite_metrics[checkpoint].get('overall_score', 0)
                    scores.append(overall_score)

            checkpoint_scores[checkpoint] = np.mean(scores) if scores else 0.0

        # Sort by overall score
        sorted_scores = sorted(checkpoint_scores.items(), key=lambda x: x[1], reverse=True)

        return {checkpoint: score for checkpoint, score in sorted_scores}

    def _perform_statistical_analysis(self, metrics_comparison: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform statistical analysis on checkpoint comparison.

        Args:
            metrics_comparison: Metrics comparison across test suites

        Returns:
            Statistical analysis results
        """
        analysis = {}

        # Compute statistical significance tests (simplified)
        try:
            from scipy import stats

            # Compare success rates across checkpoints
            for test_suite, suite_metrics in metrics_comparison.items():
                if len(suite_metrics) >= 2:
                    success_rates = [metrics.get('success_rate', 0)
                                   for metrics in suite_metrics.values()]

                    # Variance and basic statistics
                    analysis[f'{test_suite}_success_variance'] = np.var(success_rates)
                    analysis[f'{test_suite}_success_range'] = np.max(success_rates) - np.min(success_rates)

                    # Tracking error comparison
                    tracking_errors = [metrics.get('root_tracking_error', 0)
                                     for metrics in suite_metrics.values()
                                     if 'root_tracking_error' in metrics]

                    if len(tracking_errors) >= 2:
                        analysis[f'{test_suite}_tracking_variance'] = np.var(tracking_errors)

        except ImportError:
            logger.warning("scipy not available for statistical analysis")

        return analysis

    def _save_comparison_results(self, comparison: Dict[str, Any], file_path: Path):
        """Save comparison results to JSON file."""
        # Make JSON serializable
        serializable_comparison = self._make_json_serializable(comparison)

        with open(file_path, 'w') as f:
            json.dump(serializable_comparison, f, indent=2, default=str)

        logger.info(f"Comparison results saved to: {file_path}")

    def _make_json_serializable(self, obj):
        """Convert objects for JSON serialization."""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64, np.float16)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64, np.int16)):
            return int(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj

    def _generate_comparison_report(self, comparison: Dict[str, Any], file_path: Path):
        """Generate comprehensive comparison report."""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("CHECKPOINT COMPARISON REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")

        # Checkpoint information
        report_lines.append("CHECKPOINTS:")
        report_lines.append("-" * 40)
        for name, config in comparison['checkpoints'].items():
            report_lines.append(f"{name}:")
            report_lines.append(f"  Path: {config['path']}")
            if 'description' in config:
                report_lines.append(f"  Description: {config['description']}")
        report_lines.append("")

        # Overall rankings
        report_lines.append("OVERALL RANKINGS:")
        report_lines.append("-" * 40)
        for rank, (checkpoint, score) in enumerate(comparison['overall_rankings'].items(), 1):
            report_lines.append(f"{rank}. {checkpoint}: {score:.4f}")
        report_lines.append("")

        # Test suite comparisons
        for test_suite, suite_metrics in comparison['metrics_comparison'].items():
            report_lines.append(f"{test_suite.upper()} TEST SUITE:")
            report_lines.append("-" * 40)

            # Create comparison table
            metric_names = ['success_rate', 'fall_rate', 'root_tracking_error',
                          'joint_tracking_error', 'action_smoothness', 'energy_usage']

            # Header
            header = f"{'Checkpoint':<20}"
            for metric in metric_names:
                header += f"{metric.replace('_', ' ').title():<15}"
            report_lines.append(header)
            report_lines.append("-" * len(header))

            # Data rows
            for checkpoint, metrics in suite_metrics.items():
                row = f"{checkpoint:<20}"
                for metric in metric_names:
                    value = metrics.get(metric, 0)
                    row += f"{value:<15.4f}"
                report_lines.append(row)

            report_lines.append("")

            # Rankings
            if test_suite in comparison['rankings']:
                rankings = comparison['rankings'][test_suite]
                report_lines.append("Rankings:")
                for metric, ranked_list in rankings.items():
                    if ranked_list:
                        report_lines.append(f"  {metric.replace('_', ' ').title()}: {' > '.join(ranked_list)}")

            report_lines.append("")

        # Statistical analysis
        if comparison['statistical_analysis']:
            report_lines.append("STATISTICAL ANALYSIS:")
            report_lines.append("-" * 40)
            for metric, value in comparison['statistical_analysis'].items():
                report_lines.append(f"{metric}: {value:.6f}")
            report_lines.append("")

        report_lines.append("=" * 80)

        # Write report
        with open(file_path, 'w') as f:
            f.write('\n'.join(report_lines))

        logger.info(f"Comparison report saved to: {file_path}")

    def _generate_comparison_plots(self, comparison: Dict[str, Any], output_dir: Path):
        """Generate comparison visualization plots."""
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)

        # 1. Overall rankings bar chart
        self._plot_overall_rankings(comparison['overall_rankings'],
                                   plots_dir / "overall_rankings.png")

        # 2. Success rate comparison across test suites
        self._plot_success_rates(comparison['metrics_comparison'],
                               plots_dir / "success_rates.png")

        # 3. Tracking error comparison
        self._plot_tracking_errors(comparison['metrics_comparison'],
                                 plots_dir / "tracking_errors.png")

        # 4. Performance radar chart
        self._plot_performance_radar(comparison['metrics_comparison'],
                                   plots_dir / "performance_radar.png")

        # 5. Comprehensive dashboard
        self._plot_comparison_dashboard(comparison,
                                      plots_dir / "comparison_dashboard.png")

        logger.info(f"Comparison plots saved to: {plots_dir}")

    def _plot_overall_rankings(self, rankings: Dict[str, float], save_path: Path):
        """Plot overall rankings bar chart."""
        fig, ax = plt.subplots(figsize=(12, 6))

        checkpoints = list(rankings.keys())
        scores = list(rankings.values())

        bars = ax.bar(checkpoints, scores, alpha=0.8, color='skyblue')

        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

        ax.set_xlabel('Checkpoint')
        ax.set_ylabel('Overall Score')
        ax.set_title('Overall Performance Rankings')
        ax.set_ylim(0, 1.1)
        plt.xticks(rotation=45, ha='right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_success_rates(self, metrics_comparison: Dict[str, Any], save_path: Path):
        """Plot success rates across test suites."""
        fig, ax = plt.subplots(figsize=(14, 8))

        test_suites = list(metrics_comparison.keys())
        checkpoints = set()
        for suite_metrics in metrics_comparison.values():
            checkpoints.update(suite_metrics.keys())
        checkpoints = list(checkpoints)

        # Create data matrix
        data = []
        for checkpoint in checkpoints:
            checkpoint_data = []
            for test_suite in test_suites:
                if (test_suite in metrics_comparison and
                    checkpoint in metrics_comparison[test_suite]):
                    success_rate = metrics_comparison[test_suite][checkpoint].get('success_rate', 0)
                    checkpoint_data.append(success_rate)
                else:
                    checkpoint_data.append(0)
            data.append(checkpoint_data)

        # Create heatmap
        im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

        # Set ticks and labels
        ax.set_xticks(range(len(test_suites)))
        ax.set_xticklabels(test_suites, rotation=45, ha='right')
        ax.set_yticks(range(len(checkpoints)))
        ax.set_yticklabels(checkpoints)

        # Add text annotations
        for i in range(len(checkpoints)):
            for j in range(len(test_suites)):
                text = ax.text(j, i, f'{data[i][j]:.2f}',
                             ha="center", va="center", color="black", fontweight='bold')

        ax.set_title('Success Rate Comparison Across Test Suites')
        plt.colorbar(im, ax=ax, label='Success Rate')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_tracking_errors(self, metrics_comparison: Dict[str, Any], save_path: Path):
        """Plot tracking error comparison."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        test_suites = list(metrics_comparison.keys())
        checkpoints = set()
        for suite_metrics in metrics_comparison.values():
            checkpoints.update(suite_metrics.keys())
        checkpoints = list(checkpoints)

        # Root tracking error
        x = np.arange(len(test_suites))
        width = 0.8 / len(checkpoints)

        for i, checkpoint in enumerate(checkpoints):
            root_errors = []
            for test_suite in test_suites:
                if (test_suite in metrics_comparison and
                    checkpoint in metrics_comparison[test_suite]):
                    error = metrics_comparison[test_suite][checkpoint].get('root_tracking_error', 0)
                    root_errors.append(error)
                else:
                    root_errors.append(0)

            ax1.bar(x + i * width, root_errors, width, label=checkpoint, alpha=0.8)

        ax1.set_xlabel('Test Suite')
        ax1.set_ylabel('Root Tracking Error')
        ax1.set_title('Root Tracking Error Comparison')
        ax1.set_xticks(x + width * (len(checkpoints) - 1) / 2)
        ax1.set_xticklabels(test_suites, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Joint tracking error
        for i, checkpoint in enumerate(checkpoints):
            joint_errors = []
            for test_suite in test_suites:
                if (test_suite in metrics_comparison and
                    checkpoint in metrics_comparison[test_suite]):
                    error = metrics_comparison[test_suite][checkpoint].get('joint_tracking_error', 0)
                    joint_errors.append(error)
                else:
                    joint_errors.append(0)

            ax2.bar(x + i * width, joint_errors, width, label=checkpoint, alpha=0.8)

        ax2.set_xlabel('Test Suite')
        ax2.set_ylabel('Joint Tracking Error')
        ax2.set_title('Joint Tracking Error Comparison')
        ax2.set_xticks(x + width * (len(checkpoints) - 1) / 2)
        ax2.set_xticklabels(test_suites, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_performance_radar(self, metrics_comparison: Dict[str, Any], save_path: Path):
        """Plot performance radar chart."""
        # Get representative test suite (first available)
        if not metrics_comparison:
            return

        test_suite = list(metrics_comparison.keys())[0]
        suite_metrics = metrics_comparison[test_suite]

        checkpoints = list(suite_metrics.keys())
        if len(checkpoints) < 2:
            return

        # Define metrics for radar chart
        metrics = ['success_rate', 'root_tracking_error', 'joint_tracking_error',
                  'action_smoothness', 'energy_usage']

        # Normalize metrics (convert to 0-1 scale where 1 is better)
        normalized_data = {}
        for checkpoint in checkpoints:
            checkpoint_metrics = suite_metrics[checkpoint]
            normalized = []

            for metric in metrics:
                value = checkpoint_metrics.get(metric, 0)

                if metric == 'success_rate':
                    # Higher is better
                    normalized.append(value)
                else:
                    # Lower is better - invert and normalize
                    if value > 0:
                        normalized.append(1.0 / (1.0 + value))
                    else:
                        normalized.append(1.0)

            normalized_data[checkpoint] = normalized

        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        for checkpoint, values in normalized_data.items():
            values += values[:1]  # Complete the circle
            ax.plot(angles, values, linewidth=2, label=checkpoint)
            ax.fill(angles, values, alpha=0.25)

        # Add metric labels
        metric_labels = [m.replace('_', ' ').title() for m in metrics]
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels)

        ax.set_ylim(0, 1)
        ax.set_title(f'Performance Radar Chart - {test_suite.title()}', size=16, y=1.1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
        ax.grid(True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_comparison_dashboard(self, comparison: Dict[str, Any], save_path: Path):
        """Plot comprehensive comparison dashboard."""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Checkpoint Comparison Dashboard', fontsize=16)

        # 1. Overall rankings
        ax = axes[0, 0]
        rankings = comparison['overall_rankings']
        checkpoints = list(rankings.keys())
        scores = list(rankings.values())

        bars = ax.bar(checkpoints, scores, alpha=0.8)
        ax.set_title('Overall Rankings')
        ax.set_ylabel('Overall Score')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        # 2. Success rate heatmap
        ax = axes[0, 1]
        self._create_metric_heatmap(ax, comparison['metrics_comparison'], 'success_rate', 'Success Rate')

        # 3. Fall rate heatmap
        ax = axes[0, 2]
        self._create_metric_heatmap(ax, comparison['metrics_comparison'], 'fall_rate', 'Fall Rate')

        # 4. Root tracking error
        ax = axes[1, 0]
        self._create_metric_heatmap(ax, comparison['metrics_comparison'], 'root_tracking_error', 'Root Tracking Error')

        # 5. Action smoothness
        ax = axes[1, 1]
        self._create_metric_heatmap(ax, comparison['metrics_comparison'], 'action_smoothness', 'Action Smoothness')

        # 6. Energy usage
        ax = axes[1, 2]
        self._create_metric_heatmap(ax, comparison['metrics_comparison'], 'energy_usage', 'Energy Usage')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _create_metric_heatmap(self, ax, metrics_comparison: Dict[str, Any],
                              metric_name: str, title: str):
        """Create heatmap for specific metric."""
        test_suites = list(metrics_comparison.keys())
        checkpoints = set()
        for suite_metrics in metrics_comparison.values():
            checkpoints.update(suite_metrics.keys())
        checkpoints = list(checkpoints)

        # Create data matrix
        data = []
        for checkpoint in checkpoints:
            checkpoint_data = []
            for test_suite in test_suites:
                if (test_suite in metrics_comparison and
                    checkpoint in metrics_comparison[test_suite]):
                    value = metrics_comparison[test_suite][checkpoint].get(metric_name, 0)
                    checkpoint_data.append(value)
                else:
                    checkpoint_data.append(0)
            data.append(checkpoint_data)

        # Choose colormap based on metric
        if metric_name in ['success_rate']:
            cmap = 'RdYlGn'
            vmin, vmax = 0, 1
        else:
            cmap = 'RdYlGn_r'  # Reversed for "lower is better" metrics
            vmin, vmax = None, None

        im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)

        ax.set_xticks(range(len(test_suites)))
        ax.set_xticklabels(test_suites, rotation=45, ha='right')
        ax.set_yticks(range(len(checkpoints)))
        ax.set_yticklabels(checkpoints)
        ax.set_title(title)

        # Add text annotations for small heatmaps
        if len(checkpoints) <= 5 and len(test_suites) <= 6:
            for i in range(len(checkpoints)):
                for j in range(len(test_suites)):
                    text = ax.text(j, i, f'{data[i][j]:.3f}',
                                 ha="center", va="center", color="black", fontsize=8)

    def _generate_summary_table(self, comparison: Dict[str, Any], file_path: Path):
        """Generate CSV summary table."""
        summary_data = []

        for test_suite, suite_metrics in comparison['metrics_comparison'].items():
            for checkpoint, metrics in suite_metrics.items():
                row = {
                    'Checkpoint': checkpoint,
                    'Test_Suite': test_suite,
                    'Success_Rate': metrics.get('success_rate', 0),
                    'Fall_Rate': metrics.get('fall_rate', 0),
                    'Root_Tracking_Error': metrics.get('root_tracking_error', 0),
                    'Joint_Tracking_Error': metrics.get('joint_tracking_error', 0),
                    'Action_Smoothness': metrics.get('action_smoothness', 0),
                    'Energy_Usage': metrics.get('energy_usage', 0),
                    'Overall_Score': metrics.get('overall_score', 0)
                }

                if 'command_tracking_quality' in metrics:
                    row['Command_Tracking_Quality'] = metrics['command_tracking_quality']

                summary_data.append(row)

        # Create and save DataFrame
        df = pd.DataFrame(summary_data)
        df.to_csv(file_path, index=False)

        logger.info(f"Summary table saved to: {file_path}")

    def quick_compare(self, checkpoint_paths: List[str], output_dir: str):
        """
        Quick comparison with minimal configuration.

        Args:
            checkpoint_paths: List of checkpoint paths
            output_dir: Output directory
        """
        # Create simple configurations
        checkpoint_configs = []
        for i, path in enumerate(checkpoint_paths):
            name = Path(path).stem
            checkpoint_configs.append({
                'name': name,
                'path': path,
                'description': f'Checkpoint {i+1}'
            })

        # Run comparison
        return self.compare_checkpoints(checkpoint_configs, output_dir)


def main():
    """Main comparison function."""
    parser = argparse.ArgumentParser(description="Compare G1 policy checkpoints")
    parser.add_argument("--checkpoints", type=str, nargs='+', required=True,
                       help="Paths to policy checkpoints")
    parser.add_argument("--names", type=str, nargs='+',
                       help="Names for checkpoints (optional)")
    parser.add_argument("--config", type=str,
                       default="configs/evaluation/eval_config.yaml",
                       help="Path to evaluation configuration")
    parser.add_argument("--output_dir", type=str, default="comparison_results",
                       help="Output directory for results")
    parser.add_argument("--episodes", type=int, default=10,
                       help="Number of episodes per test suite")

    args = parser.parse_args()

    # Load configuration
    try:
        config = OmegaConf.load(args.config)
    except FileNotFoundError:
        logger.warning(f"Config file not found: {args.config}. Using default config.")
        config = OmegaConf.create({
            'use_cuda': True,
            'max_episode_steps': 1000,
            'episodes_per_suite': 10,
            'record_videos': False,
            'environment': {
                'model_path': 'assets/unitree_g1/g1.xml',
                'frame_skip': 10
            },
            'metrics': {},
            'test_suites': {
                'default_suites': ['stand', 'walk', 'turn', 'stop']
            }
        })

    # Override episodes
    config.episodes_per_suite = args.episodes

    # Create checkpoint configurations
    checkpoint_configs = []
    for i, checkpoint_path in enumerate(args.checkpoints):
        name = args.names[i] if args.names and i < len(args.names) else Path(checkpoint_path).stem
        checkpoint_configs.append({
            'name': name,
            'path': checkpoint_path,
            'description': f'Checkpoint from {checkpoint_path}'
        })

    # Initialize comparator and run comparison
    comparator = CheckpointComparator(config)
    results = comparator.compare_checkpoints(checkpoint_configs, args.output_dir)

    # Print summary
    print(f"\nComparison completed for {len(checkpoint_configs)} checkpoints")
    print(f"Results saved to: {args.output_dir}")
    print("\nOverall Rankings:")
    for rank, (checkpoint, score) in enumerate(results['overall_rankings'].items(), 1):
        print(f"{rank}. {checkpoint}: {score:.4f}")


if __name__ == "__main__":
    main()