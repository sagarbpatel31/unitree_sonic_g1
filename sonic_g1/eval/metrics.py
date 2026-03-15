"""
Comprehensive metrics tracking for G1 policy evaluation.

This module implements detailed metrics calculation for evaluating
G1 controller performance across different test scenarios.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple
from scipy.spatial.transform import Rotation as R
from scipy import signal
import logging

logger = logging.getLogger(__name__)


class EvaluationMetrics:
    """
    Core metrics calculator for policy evaluation.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize metrics calculator.

        Args:
            config: Metrics configuration
        """
        self.config = config

        # Metric weights and thresholds
        self.fall_height_threshold = config.get('fall_height_threshold', 0.3)
        self.fall_angle_threshold = config.get('fall_angle_threshold', np.pi/3)
        self.tracking_error_scale = config.get('tracking_error_scale', 1.0)
        self.energy_scale = config.get('energy_scale', 1.0)

    def compute_episode_length(self, episode_data: Dict[str, Any]) -> float:
        """Compute episode length in steps."""
        return len(episode_data['actions'])

    def compute_fall_rate(self, episode_results: List[Dict[str, Any]]) -> float:
        """
        Compute fall rate across episodes.

        Args:
            episode_results: List of episode result dictionaries

        Returns:
            Fall rate (0-1)
        """
        total_episodes = len(episode_results)
        if total_episodes == 0:
            return 0.0

        fallen_episodes = 0
        for episode in episode_results:
            if self._detect_fall(episode['episode_data']):
                fallen_episodes += 1

        return fallen_episodes / total_episodes

    def _detect_fall(self, episode_data: Dict[str, Any]) -> bool:
        """
        Detect if robot fell during episode.

        Args:
            episode_data: Episode data dictionary

        Returns:
            True if fall detected
        """
        # Check root height
        if 'root_positions' in episode_data and episode_data['root_positions']:
            min_height = min([pos[2] for pos in episode_data['root_positions']])
            if min_height < self.fall_height_threshold:
                return True

        # Check root orientation
        if 'root_orientations' in episode_data and episode_data['root_orientations']:
            for quat in episode_data['root_orientations']:
                # Convert quaternion to Euler angles
                rotation = R.from_quat(quat)
                euler = rotation.as_euler('xyz')
                roll, pitch = euler[0], euler[1]

                if abs(roll) > self.fall_angle_threshold or abs(pitch) > self.fall_angle_threshold:
                    return True

        # Check info data for explicit fall flag
        if 'info_data' in episode_data:
            for info in episode_data['info_data']:
                if info.get('fell', False):
                    return True

        return False

    def compute_root_tracking_error(self, episode_data: Dict[str, Any]) -> float:
        """
        Compute root tracking error.

        Args:
            episode_data: Episode data dictionary

        Returns:
            Average root tracking error
        """
        if ('root_positions' not in episode_data or
            'reference_poses' not in episode_data or
            not episode_data['root_positions'] or
            not episode_data['reference_poses']):
            return 0.0

        errors = []
        min_length = min(len(episode_data['root_positions']),
                        len(episode_data['reference_poses']))

        for i in range(min_length):
            actual_pos = np.array(episode_data['root_positions'][i])
            ref_pose = episode_data['reference_poses'][i]

            # Extract reference position (assuming it's in the reference pose)
            if isinstance(ref_pose, dict) and 'root_position' in ref_pose:
                ref_pos = np.array(ref_pose['root_position'])
            else:
                # Assume first 3 elements are position
                ref_pos = np.array(ref_pose[:3])

            # Compute position error
            pos_error = np.linalg.norm(actual_pos - ref_pos)
            errors.append(pos_error)

        return np.mean(errors) if errors else 0.0

    def compute_joint_tracking_error(self, episode_data: Dict[str, Any]) -> float:
        """
        Compute joint tracking error.

        Args:
            episode_data: Episode data dictionary

        Returns:
            Average joint tracking error
        """
        if ('joint_positions' not in episode_data or
            'reference_poses' not in episode_data or
            not episode_data['joint_positions'] or
            not episode_data['reference_poses']):
            return 0.0

        errors = []
        min_length = min(len(episode_data['joint_positions']),
                        len(episode_data['reference_poses']))

        for i in range(min_length):
            actual_joints = np.array(episode_data['joint_positions'][i])
            ref_pose = episode_data['reference_poses'][i]

            # Extract reference joint positions
            if isinstance(ref_pose, dict) and 'joint_positions' in ref_pose:
                ref_joints = np.array(ref_pose['joint_positions'])
            else:
                # Assume joint positions are after root pose (7 DOF: 3 pos + 4 quat)
                ref_joints = np.array(ref_pose[7:])

            # Ensure same dimensionality
            min_joints = min(len(actual_joints), len(ref_joints))
            if min_joints > 0:
                joint_error = np.linalg.norm(actual_joints[:min_joints] - ref_joints[:min_joints])
                errors.append(joint_error)

        return np.mean(errors) if errors else 0.0

    def compute_action_smoothness(self, episode_data: Dict[str, Any]) -> float:
        """
        Compute action smoothness metric.

        Args:
            episode_data: Episode data dictionary

        Returns:
            Action smoothness score (lower is smoother)
        """
        actions = episode_data.get('actions', [])
        if len(actions) < 2:
            return 0.0

        actions_array = np.array(actions)

        # Compute action derivatives (velocity and acceleration)
        action_vel = np.diff(actions_array, axis=0)
        action_accel = np.diff(action_vel, axis=0)

        # RMS of acceleration as smoothness metric
        smoothness = np.sqrt(np.mean(action_accel**2))

        return smoothness

    def compute_energy_usage_proxy(self, episode_data: Dict[str, Any]) -> float:
        """
        Compute energy usage proxy based on actions and joint velocities.

        Args:
            episode_data: Episode data dictionary

        Returns:
            Energy usage proxy
        """
        actions = episode_data.get('actions', [])
        joint_vels = episode_data.get('joint_velocities', [])

        if not actions:
            return 0.0

        actions_array = np.array(actions)

        # Simple energy proxy: sum of squared actions
        action_energy = np.sum(actions_array**2)

        # If joint velocities available, use torque * velocity approximation
        if joint_vels and len(joint_vels) == len(actions):
            joint_vels_array = np.array(joint_vels)
            # Approximate torque from actions (assume actions are target positions)
            # Energy = torque * velocity ≈ action_magnitude * velocity
            power = np.sum(np.abs(actions_array) * np.abs(joint_vels_array), axis=1)
            total_energy = np.sum(power)
        else:
            total_energy = action_energy

        return total_energy * self.energy_scale

    def compute_command_tracking_quality(self, episode_data: Dict[str, Any],
                                       test_suite: str) -> float:
        """
        Compute command tracking quality for applicable test suites.

        Args:
            episode_data: Episode data dictionary
            test_suite: Test suite name

        Returns:
            Command tracking quality (0-1, higher is better)
        """
        if test_suite not in ['walk', 'turn', 'stop']:
            return 1.0  # N/A for non-command tests

        info_data = episode_data.get('info_data', [])
        if not info_data:
            return 0.0

        # Extract command tracking errors from info
        tracking_errors = []
        for info in info_data:
            if 'command_tracking_error' in info:
                tracking_errors.append(info['command_tracking_error'])
            elif test_suite == 'walk' and 'speed_error' in info:
                tracking_errors.append(info['speed_error'])
            elif test_suite == 'turn' and 'turn_rate_error' in info:
                tracking_errors.append(info['turn_rate_error'])

        if not tracking_errors:
            return 1.0

        # Convert error to quality score
        avg_error = np.mean(tracking_errors)
        quality = np.exp(-avg_error)  # Exponential decay of quality with error

        return quality

    def compute_stability_metrics(self, episode_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute various stability metrics.

        Args:
            episode_data: Episode data dictionary

        Returns:
            Dictionary of stability metrics
        """
        metrics = {}

        # Root position stability (variance in x, y)
        if 'root_positions' in episode_data and episode_data['root_positions']:
            positions = np.array(episode_data['root_positions'])
            if len(positions) > 1:
                x_var = np.var(positions[:, 0])
                y_var = np.var(positions[:, 1])
                metrics['root_xy_stability'] = 1.0 / (1.0 + x_var + y_var)

        # Root orientation stability
        if 'root_orientations' in episode_data and episode_data['root_orientations']:
            orientations = np.array(episode_data['root_orientations'])
            if len(orientations) > 1:
                # Compute orientation variance (simplified)
                quat_var = np.var(orientations, axis=0)
                metrics['root_orientation_stability'] = 1.0 / (1.0 + np.sum(quat_var))

        # Joint velocity stability
        if 'joint_velocities' in episode_data and episode_data['joint_velocities']:
            joint_vels = np.array(episode_data['joint_velocities'])
            if len(joint_vels) > 1:
                vel_var = np.var(joint_vels)
                metrics['joint_velocity_stability'] = 1.0 / (1.0 + vel_var)

        return metrics

    def compute_success_criteria(self, episode_data: Dict[str, Any],
                               test_suite: str) -> bool:
        """
        Compute test-specific success criteria.

        Args:
            episode_data: Episode data dictionary
            test_suite: Test suite name

        Returns:
            True if episode was successful
        """
        # Basic criterion: no fall
        if self._detect_fall(episode_data):
            return False

        # Test-suite specific criteria
        episode_length = len(episode_data['actions'])

        if test_suite == 'stand':
            # Success: maintain standing position for minimum duration
            min_duration = self.config.get('stand_min_duration', 300)  # steps
            return episode_length >= min_duration

        elif test_suite == 'walk':
            # Success: walk for minimum distance/duration without falling
            min_duration = self.config.get('walk_min_duration', 500)
            return episode_length >= min_duration

        elif test_suite == 'turn':
            # Success: complete turning command
            min_duration = self.config.get('turn_min_duration', 200)
            return episode_length >= min_duration

        elif test_suite == 'stop':
            # Success: come to stable stop
            return self._check_stop_success(episode_data)

        elif test_suite == 'recovery_from_push':
            # Success: recover stability after disturbance
            return self._check_recovery_success(episode_data)

        elif test_suite == 'crouch':
            # Success: achieve and maintain crouch position
            return self._check_crouch_success(episode_data)

        else:
            # Default: episode completion without fall
            min_duration = self.config.get('default_min_duration', 200)
            return episode_length >= min_duration

    def _check_stop_success(self, episode_data: Dict[str, Any]) -> bool:
        """Check if stop test was successful."""
        # Look for final velocity near zero
        if 'joint_velocities' in episode_data and episode_data['joint_velocities']:
            final_vels = episode_data['joint_velocities'][-10:]  # Last 10 steps
            avg_final_vel = np.mean([np.linalg.norm(v) for v in final_vels])
            return avg_final_vel < self.config.get('stop_velocity_threshold', 0.1)

        # Fallback: check root velocity from info
        if 'info_data' in episode_data:
            final_info = episode_data['info_data'][-10:]
            for info in final_info:
                if 'root_velocity' in info:
                    root_vel = np.linalg.norm(info['root_velocity'][:2])  # x, y velocity
                    if root_vel > self.config.get('stop_velocity_threshold', 0.1):
                        return False
            return True

        return True

    def _check_recovery_success(self, episode_data: Dict[str, Any]) -> bool:
        """Check if recovery from push was successful."""
        # Look for stability in final portion of episode
        if len(episode_data['actions']) < 100:
            return False

        # Check if robot maintained stability in last quarter of episode
        final_quarter_start = len(episode_data['actions']) * 3 // 4
        final_data = {
            key: value[final_quarter_start:] if isinstance(value, list) else value
            for key, value in episode_data.items()
        }

        # No fall in final quarter
        return not self._detect_fall(final_data)

    def _check_crouch_success(self, episode_data: Dict[str, Any]) -> bool:
        """Check if crouch test was successful."""
        # Check if root height was lowered appropriately
        if 'root_positions' in episode_data and episode_data['root_positions']:
            positions = episode_data['root_positions']
            if len(positions) > 50:
                # Compare initial height to final height
                initial_height = np.mean([pos[2] for pos in positions[:10]])
                final_height = np.mean([pos[2] for pos in positions[-10:]])

                height_reduction = initial_height - final_height
                min_crouch = self.config.get('crouch_min_height_reduction', 0.2)

                return height_reduction >= min_crouch

        return False


class MetricsTracker:
    """
    High-level metrics tracking and aggregation.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize metrics tracker.

        Args:
            config: Metrics configuration
        """
        self.config = config
        self.metrics_calculator = EvaluationMetrics(config)

    def compute_episode_metrics(self, episode_data: Dict[str, Any],
                              test_suite: str) -> Dict[str, float]:
        """
        Compute all metrics for a single episode.

        Args:
            episode_data: Episode data dictionary
            test_suite: Test suite name

        Returns:
            Dictionary of computed metrics
        """
        metrics = {}

        # Basic metrics
        metrics['episode_length'] = self.metrics_calculator.compute_episode_length(episode_data)
        metrics['fell'] = self.metrics_calculator._detect_fall(episode_data)
        metrics['success'] = self.metrics_calculator.compute_success_criteria(episode_data, test_suite)

        # Tracking errors
        metrics['root_tracking_error'] = self.metrics_calculator.compute_root_tracking_error(episode_data)
        metrics['joint_tracking_error'] = self.metrics_calculator.compute_joint_tracking_error(episode_data)

        # Motion quality metrics
        metrics['action_smoothness'] = self.metrics_calculator.compute_action_smoothness(episode_data)
        metrics['energy_usage'] = self.metrics_calculator.compute_energy_usage_proxy(episode_data)

        # Command tracking (if applicable)
        if test_suite in ['walk', 'turn', 'stop']:
            metrics['command_tracking_quality'] = self.metrics_calculator.compute_command_tracking_quality(
                episode_data, test_suite
            )

        # Stability metrics
        stability_metrics = self.metrics_calculator.compute_stability_metrics(episode_data)
        metrics.update(stability_metrics)

        return metrics

    def aggregate_episodes(self, episode_results: List[Dict[str, Any]],
                          test_suite: str) -> Dict[str, float]:
        """
        Aggregate metrics across multiple episodes.

        Args:
            episode_results: List of episode result dictionaries
            test_suite: Test suite name

        Returns:
            Aggregated metrics
        """
        if not episode_results:
            return {}

        aggregated = {}
        num_episodes = len(episode_results)

        # Extract metrics from all episodes
        all_metrics = [result['metrics'] for result in episode_results]

        # Basic statistics
        aggregated['num_episodes'] = num_episodes
        aggregated['success_rate'] = np.mean([m.get('success', False) for m in all_metrics])
        aggregated['fall_rate'] = np.mean([m.get('fell', False) for m in all_metrics])

        # Episode lengths
        episode_lengths = [m.get('episode_length', 0) for m in all_metrics]
        aggregated['avg_episode_length'] = np.mean(episode_lengths)
        aggregated['std_episode_length'] = np.std(episode_lengths)
        aggregated['min_episode_length'] = np.min(episode_lengths)
        aggregated['max_episode_length'] = np.max(episode_lengths)

        # Tracking errors
        root_errors = [m.get('root_tracking_error', 0) for m in all_metrics if 'root_tracking_error' in m]
        if root_errors:
            aggregated['root_tracking_error'] = np.mean(root_errors)
            aggregated['root_tracking_error_std'] = np.std(root_errors)

        joint_errors = [m.get('joint_tracking_error', 0) for m in all_metrics if 'joint_tracking_error' in m]
        if joint_errors:
            aggregated['joint_tracking_error'] = np.mean(joint_errors)
            aggregated['joint_tracking_error_std'] = np.std(joint_errors)

        # Motion quality
        smoothness_values = [m.get('action_smoothness', 0) for m in all_metrics if 'action_smoothness' in m]
        if smoothness_values:
            aggregated['action_smoothness'] = np.mean(smoothness_values)
            aggregated['action_smoothness_std'] = np.std(smoothness_values)

        energy_values = [m.get('energy_usage', 0) for m in all_metrics if 'energy_usage' in m]
        if energy_values:
            aggregated['energy_usage'] = np.mean(energy_values)
            aggregated['energy_usage_std'] = np.std(energy_values)

        # Command tracking (if applicable)
        if test_suite in ['walk', 'turn', 'stop']:
            cmd_values = [m.get('command_tracking_quality', 0) for m in all_metrics
                         if 'command_tracking_quality' in m]
            if cmd_values:
                aggregated['command_tracking_quality'] = np.mean(cmd_values)
                aggregated['command_tracking_quality_std'] = np.std(cmd_values)

        # Stability metrics
        for stability_metric in ['root_xy_stability', 'root_orientation_stability', 'joint_velocity_stability']:
            values = [m.get(stability_metric, 0) for m in all_metrics if stability_metric in m]
            if values:
                aggregated[stability_metric] = np.mean(values)
                aggregated[f'{stability_metric}_std'] = np.std(values)

        # Overall performance score
        aggregated['overall_score'] = self._compute_overall_score(aggregated, test_suite)

        return aggregated

    def _compute_overall_score(self, metrics: Dict[str, float], test_suite: str) -> float:
        """
        Compute overall performance score for test suite.

        Args:
            metrics: Aggregated metrics
            test_suite: Test suite name

        Returns:
            Overall score (0-1, higher is better)
        """
        # Base score from success rate
        score = metrics.get('success_rate', 0) * 0.4

        # Bonus for low fall rate
        score += (1 - metrics.get('fall_rate', 1)) * 0.2

        # Bonus for good tracking (if available)
        if 'root_tracking_error' in metrics:
            tracking_score = np.exp(-metrics['root_tracking_error'])
            score += tracking_score * 0.2

        # Bonus for smoothness (lower is better)
        if 'action_smoothness' in metrics:
            smoothness_score = 1.0 / (1.0 + metrics['action_smoothness'])
            score += smoothness_score * 0.1

        # Bonus for command tracking (if applicable)
        if 'command_tracking_quality' in metrics:
            score += metrics['command_tracking_quality'] * 0.1

        return min(score, 1.0)

    def create_metrics_summary(self, results: Dict[str, Any]) -> str:
        """
        Create text summary of metrics.

        Args:
            results: Evaluation results across test suites

        Returns:
            Formatted metrics summary
        """
        summary_lines = []
        summary_lines.append("METRICS SUMMARY")
        summary_lines.append("=" * 50)

        for test_suite, suite_results in results.items():
            metrics = suite_results['metrics']

            summary_lines.append(f"\n{test_suite.upper()}:")
            summary_lines.append("-" * 30)
            summary_lines.append(f"Episodes:               {metrics.get('num_episodes', 0)}")
            summary_lines.append(f"Success Rate:           {metrics.get('success_rate', 0):.3f}")
            summary_lines.append(f"Fall Rate:              {metrics.get('fall_rate', 0):.3f}")
            summary_lines.append(f"Avg Episode Length:     {metrics.get('avg_episode_length', 0):.1f}")
            summary_lines.append(f"Root Tracking Error:    {metrics.get('root_tracking_error', 0):.4f}")
            summary_lines.append(f"Joint Tracking Error:   {metrics.get('joint_tracking_error', 0):.4f}")
            summary_lines.append(f"Action Smoothness:      {metrics.get('action_smoothness', 0):.4f}")
            summary_lines.append(f"Energy Usage:           {metrics.get('energy_usage', 0):.2f}")

            if 'command_tracking_quality' in metrics:
                summary_lines.append(f"Command Tracking:       {metrics['command_tracking_quality']:.3f}")

            summary_lines.append(f"Overall Score:          {metrics.get('overall_score', 0):.3f}")

        return '\n'.join(summary_lines)