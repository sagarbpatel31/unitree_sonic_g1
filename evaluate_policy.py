#!/usr/bin/env python3
"""
Policy evaluation tool for Unitree G1 MuJoCo SONIC-inspired controller.

This script provides comprehensive rollout evaluation with detailed metrics,
video recording, and support for named test suites including standing,
walking, turning, recovery, and more.
"""

import sys
import logging
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
from omegaconf import OmegaConf
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from sonic_g1.models.policy import G1Policy
from sonic_g1.envs.g1_env import G1Env
from sonic_g1.eval.metrics import EvaluationMetrics, MetricsTracker
from sonic_g1.eval.video import VideoRecorder
from sonic_g1.eval.test_suites import TestSuiteRunner
from sonic_g1.utils.checkpoints import load_checkpoint

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PolicyEvaluator:
    """
    Comprehensive policy evaluator for G1 controllers.

    Supports rollout evaluation with detailed metrics tracking,
    video recording, and named test suites.
    """

    def __init__(self, config: OmegaConf):
        """
        Initialize policy evaluator.

        Args:
            config: Evaluation configuration
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.use_cuda else 'cpu')

        # Initialize components
        self.metrics_tracker = MetricsTracker(config.metrics)
        self.test_suite_runner = TestSuiteRunner(config.test_suites)

        # Video recording (optional)
        self.video_recorder = None
        if config.get('record_videos', False):
            self.video_recorder = VideoRecorder(config.video)

        # Environment
        self.env = None

        # Results storage
        self.evaluation_results = {}

        logger.info("Initialized PolicyEvaluator")

    def load_policy(self, checkpoint_path: str) -> torch.nn.Module:
        """
        Load policy from checkpoint.

        Args:
            checkpoint_path: Path to policy checkpoint

        Returns:
            Loaded policy
        """
        logger.info(f"Loading policy from: {checkpoint_path}")

        checkpoint = load_checkpoint(checkpoint_path, self.device)

        # Extract model info
        obs_dim = checkpoint['obs_dim']
        action_dim = checkpoint['action_dim']
        policy_config = checkpoint.get('policy_config', {})

        # Create and load policy
        policy = G1Policy(obs_dim, action_dim, policy_config).to(self.device)
        policy.load_state_dict(checkpoint['policy_state_dict'])
        policy.eval()

        logger.info(f"Loaded policy: obs_dim={obs_dim}, action_dim={action_dim}")
        return policy

    def create_environment(self, test_suite: Optional[str] = None) -> G1Env:
        """
        Create evaluation environment.

        Args:
            test_suite: Optional test suite name for specialized setup

        Returns:
            Configured environment
        """
        env_config = self.config.environment.copy()

        # Modify environment based on test suite
        if test_suite:
            suite_config = self.test_suite_runner.get_suite_config(test_suite)
            if suite_config:
                # Override environment settings for specific test
                env_config.update(suite_config.get('env_overrides', {}))

        env = G1Env(
            model_path=env_config.model_path,
            config=env_config
        )

        logger.info(f"Created environment for test suite: {test_suite}")
        return env

    def evaluate_policy(self,
                       policy: torch.nn.Module,
                       policy_name: str,
                       test_suites: Optional[List[str]] = None,
                       num_episodes: Optional[int] = None) -> Dict[str, Any]:
        """
        Evaluate policy with comprehensive metrics.

        Args:
            policy: Policy to evaluate
            policy_name: Name for identification
            test_suites: List of test suites to run (None for all)
            num_episodes: Number of episodes per test suite

        Returns:
            Evaluation results
        """
        logger.info(f"Evaluating policy: {policy_name}")

        if test_suites is None:
            test_suites = self.config.test_suites.get('default_suites',
                                                    ['stand', 'walk', 'turn', 'stop'])

        if num_episodes is None:
            num_episodes = self.config.get('episodes_per_suite', 10)

        policy_results = {}

        for test_suite in test_suites:
            logger.info(f"Running test suite: {test_suite}")

            # Create environment for this test suite
            self.env = self.create_environment(test_suite)

            # Run episodes
            suite_results = self._run_test_suite(
                policy, test_suite, num_episodes, policy_name
            )

            policy_results[test_suite] = suite_results

        return policy_results

    def _run_test_suite(self,
                       policy: torch.nn.Module,
                       test_suite: str,
                       num_episodes: int,
                       policy_name: str) -> Dict[str, Any]:
        """
        Run a specific test suite.

        Args:
            policy: Policy to evaluate
            test_suite: Test suite name
            num_episodes: Number of episodes to run
            policy_name: Policy name for video naming

        Returns:
            Test suite results
        """
        episode_results = []

        # Get test suite configuration
        suite_config = self.test_suite_runner.get_suite_config(test_suite)

        for episode in range(num_episodes):
            logger.info(f"Running {test_suite} episode {episode + 1}/{num_episodes}")

            # Setup video recording if enabled
            video_path = None
            if self.video_recorder:
                video_path = self.video_recorder.start_recording(
                    policy_name, test_suite, episode
                )

            # Run episode
            episode_result = self._run_episode(
                policy, test_suite, suite_config, episode, video_path
            )

            episode_results.append(episode_result)

            # Stop video recording
            if self.video_recorder and video_path:
                self.video_recorder.stop_recording()

        # Aggregate results
        suite_metrics = self.metrics_tracker.aggregate_episodes(
            episode_results, test_suite
        )

        return {
            'episodes': episode_results,
            'metrics': suite_metrics,
            'test_suite': test_suite,
            'num_episodes': num_episodes
        }

    def _run_episode(self,
                    policy: torch.nn.Module,
                    test_suite: str,
                    suite_config: Dict[str, Any],
                    episode_num: int,
                    video_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Run a single evaluation episode.

        Args:
            policy: Policy to evaluate
            test_suite: Test suite name
            suite_config: Test suite configuration
            episode_num: Episode number
            video_path: Optional video recording path

        Returns:
            Episode results
        """
        # Reset environment
        obs = self.env.reset()

        # Initialize episode tracking
        episode_data = {
            'test_suite': test_suite,
            'episode': episode_num,
            'observations': [],
            'actions': [],
            'rewards': [],
            'reference_poses': [],
            'root_positions': [],
            'root_orientations': [],
            'joint_positions': [],
            'joint_velocities': [],
            'timestamps': [],
            'info_data': []
        }

        # Test suite specific initialization
        self.test_suite_runner.initialize_episode(self.env, test_suite, suite_config)

        done = False
        step = 0
        total_reward = 0.0
        start_time = time.time()

        # Episode loop
        while not done and step < self.config.get('max_episode_steps', 1000):
            # Get action from policy
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                action_dist = policy.get_distribution(obs_tensor)
                action = action_dist.mean.cpu().numpy()[0]  # Deterministic evaluation

            # Execute action
            obs_next, reward, done, info = self.env.step(action)

            # Record data
            episode_data['observations'].append(obs.copy())
            episode_data['actions'].append(action.copy())
            episode_data['rewards'].append(reward)
            episode_data['timestamps'].append(step * self.env.dt)
            episode_data['info_data'].append(info.copy())

            # Extract state information
            self._extract_state_data(episode_data, info)

            # Update test suite state
            done = self.test_suite_runner.update_episode(
                self.env, test_suite, suite_config, step, info, done
            )

            # Record video frame
            if self.video_recorder and video_path:
                self.video_recorder.capture_frame(self.env)

            total_reward += reward
            step += 1
            obs = obs_next

        # Episode completed
        episode_duration = time.time() - start_time

        # Compute episode-level metrics
        episode_metrics = self.metrics_tracker.compute_episode_metrics(
            episode_data, test_suite
        )

        return {
            'episode_data': episode_data,
            'metrics': episode_metrics,
            'total_reward': total_reward,
            'episode_length': step,
            'duration': episode_duration,
            'success': episode_metrics.get('success', False),
            'test_suite': test_suite,
            'episode_num': episode_num
        }

    def _extract_state_data(self, episode_data: Dict[str, Any], info: Dict[str, Any]):
        """Extract state data from environment info."""
        # Root pose
        if 'root_position' in info:
            episode_data['root_positions'].append(info['root_position'].copy())

        if 'root_orientation' in info:
            episode_data['root_orientations'].append(info['root_orientation'].copy())

        # Joint states
        if 'joint_positions' in info:
            episode_data['joint_positions'].append(info['joint_positions'].copy())

        if 'joint_velocities' in info:
            episode_data['joint_velocities'].append(info['joint_velocities'].copy())

        # Reference poses
        if 'reference_pose' in info:
            episode_data['reference_poses'].append(info['reference_pose'].copy())

    def run_comprehensive_evaluation(self,
                                   checkpoint_path: str,
                                   output_dir: str,
                                   policy_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Run comprehensive evaluation with all test suites.

        Args:
            checkpoint_path: Path to policy checkpoint
            output_dir: Directory to save results
            policy_name: Optional policy name (defaults to checkpoint name)

        Returns:
            Complete evaluation results
        """
        if policy_name is None:
            policy_name = Path(checkpoint_path).stem

        # Load policy
        policy = self.load_policy(checkpoint_path)

        # Run evaluation
        results = self.evaluate_policy(policy, policy_name)

        # Save results
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save detailed results
        results_file = output_path / f"{policy_name}_results.json"
        self._save_results(results, results_file)

        # Generate summary report
        report_file = output_path / f"{policy_name}_report.txt"
        self._generate_report(results, report_file, policy_name)

        # Generate metrics summary
        metrics_file = output_path / f"{policy_name}_metrics.json"
        summary_metrics = self._extract_summary_metrics(results)
        with open(metrics_file, 'w') as f:
            json.dump(summary_metrics, f, indent=2, default=str)

        logger.info(f"Evaluation completed. Results saved to: {output_path}")

        return results

    def _save_results(self, results: Dict[str, Any], file_path: Path):
        """Save detailed results to JSON file."""
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = self._make_json_serializable(results)

        with open(file_path, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)

        logger.info(f"Results saved to: {file_path}")

    def _make_json_serializable(self, obj):
        """Convert numpy arrays and other objects for JSON serialization."""
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

    def _generate_report(self, results: Dict[str, Any], file_path: Path, policy_name: str):
        """Generate comprehensive evaluation report."""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append(f"POLICY EVALUATION REPORT: {policy_name}")
        report_lines.append("=" * 80)
        report_lines.append("")

        for test_suite, suite_results in results.items():
            metrics = suite_results['metrics']
            num_episodes = suite_results['num_episodes']

            report_lines.append(f"{test_suite.upper()} TEST SUITE:")
            report_lines.append("-" * 40)
            report_lines.append(f"Episodes: {num_episodes}")
            report_lines.append("")

            # Key metrics
            report_lines.append("Key Metrics:")
            report_lines.append(f"  Success Rate:         {metrics.get('success_rate', 0):.3f}")
            report_lines.append(f"  Fall Rate:            {metrics.get('fall_rate', 0):.3f}")
            report_lines.append(f"  Avg Episode Length:   {metrics.get('avg_episode_length', 0):.1f}")
            report_lines.append(f"  Root Tracking Error:  {metrics.get('root_tracking_error', 0):.4f}")
            report_lines.append(f"  Joint Tracking Error: {metrics.get('joint_tracking_error', 0):.4f}")
            report_lines.append(f"  Action Smoothness:    {metrics.get('action_smoothness', 0):.4f}")
            report_lines.append(f"  Energy Usage:         {metrics.get('energy_usage', 0):.4f}")

            if 'command_tracking_quality' in metrics:
                report_lines.append(f"  Command Tracking:     {metrics['command_tracking_quality']:.4f}")

            report_lines.append("")

        report_lines.append("=" * 80)

        # Write report
        with open(file_path, 'w') as f:
            f.write('\n'.join(report_lines))

        logger.info(f"Report saved to: {file_path}")

    def _extract_summary_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key summary metrics across all test suites."""
        summary = {}

        for test_suite, suite_results in results.items():
            metrics = suite_results['metrics']

            summary[test_suite] = {
                'success_rate': metrics.get('success_rate', 0),
                'fall_rate': metrics.get('fall_rate', 0),
                'avg_episode_length': metrics.get('avg_episode_length', 0),
                'root_tracking_error': metrics.get('root_tracking_error', 0),
                'joint_tracking_error': metrics.get('joint_tracking_error', 0),
                'action_smoothness': metrics.get('action_smoothness', 0),
                'energy_usage': metrics.get('energy_usage', 0)
            }

        # Overall summary
        all_success_rates = [s['success_rate'] for s in summary.values()]
        all_fall_rates = [s['fall_rate'] for s in summary.values()]

        summary['overall'] = {
            'avg_success_rate': np.mean(all_success_rates),
            'avg_fall_rate': np.mean(all_fall_rates),
            'num_test_suites': len(summary),
            'test_suites': list(results.keys())
        }

        return summary


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate G1 policy")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to policy checkpoint")
    parser.add_argument("--config", type=str,
                       default="configs/evaluation/eval_config.yaml",
                       help="Path to evaluation configuration")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                       help="Output directory for results")
    parser.add_argument("--policy_name", type=str,
                       help="Name for the policy (defaults to checkpoint filename)")
    parser.add_argument("--test_suites", type=str, nargs='+',
                       choices=['stand', 'walk', 'turn', 'stop', 'recovery_from_push', 'crouch'],
                       help="Specific test suites to run")
    parser.add_argument("--episodes", type=int, default=10,
                       help="Number of episodes per test suite")
    parser.add_argument("--record_videos", action='store_true',
                       help="Record videos of evaluation")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")

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
            }
        })

    # Override config with command line arguments
    if args.test_suites:
        config.test_suites.default_suites = args.test_suites

    config.episodes_per_suite = args.episodes
    config.record_videos = args.record_videos

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Initialize evaluator
    evaluator = PolicyEvaluator(config)

    # Run evaluation
    results = evaluator.run_comprehensive_evaluation(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        policy_name=args.policy_name
    )

    # Print summary
    policy_name = args.policy_name or Path(args.checkpoint).stem
    print(f"\nEvaluation completed for: {policy_name}")
    print(f"Results saved to: {args.output_dir}")

    # Quick summary
    for test_suite, suite_results in results.items():
        metrics = suite_results['metrics']
        success_rate = metrics.get('success_rate', 0)
        fall_rate = metrics.get('fall_rate', 0)
        print(f"{test_suite}: Success={success_rate:.3f}, Fall={fall_rate:.3f}")


if __name__ == "__main__":
    main()