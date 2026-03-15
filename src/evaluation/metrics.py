"""
Evaluation metrics for model assessment.
Provides comprehensive metrics for tracking performance, robustness, and efficiency.
"""

import numpy as np
from typing import Dict, List, Any
from abc import ABC, abstractmethod


class BaseMetrics(ABC):
    """Base class for evaluation metrics."""

    @abstractmethod
    def compute_metrics(self, trajectories: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute metrics from trajectory data."""
        pass


class TrackingMetrics(BaseMetrics):
    """Metrics for motion tracking performance."""

    def compute_metrics(self, trajectories: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute tracking performance metrics."""
        if not trajectories:
            return {}

        metrics = {}

        # Collect tracking errors from all trajectories
        all_tracking_errors = []
        all_joint_errors = []

        for traj in trajectories:
            infos = traj.get("infos", [])
            for info in infos:
                if "tracking_error" in info:
                    all_tracking_errors.append(info["tracking_error"])

                if "joint_tracking_errors" in info:
                    all_joint_errors.extend(info["joint_tracking_errors"])

        # Overall tracking statistics
        if all_tracking_errors:
            metrics["tracking_error_mean"] = np.mean(all_tracking_errors)
            metrics["tracking_error_std"] = np.std(all_tracking_errors)
            metrics["tracking_error_max"] = np.max(all_tracking_errors)
            metrics["tracking_error_p95"] = np.percentile(all_tracking_errors, 95)

        # Joint-wise tracking statistics
        if all_joint_errors:
            metrics["joint_tracking_mean"] = np.mean(all_joint_errors)
            metrics["joint_tracking_std"] = np.std(all_joint_errors)

        # Success rate
        successful_episodes = sum(1 for traj in trajectories if traj.get("success", False))
        metrics["success_rate"] = successful_episodes / len(trajectories) if trajectories else 0.0

        # Episode completion rate
        target_length = 1000  # Expected episode length
        completion_rates = []
        for traj in trajectories:
            completion_rate = min(1.0, traj.get("episode_length", 0) / target_length)
            completion_rates.append(completion_rate)

        metrics["completion_rate_mean"] = np.mean(completion_rates) if completion_rates else 0.0

        return metrics


class RobustnessMetrics(BaseMetrics):
    """Metrics for robustness assessment."""

    def compute_metrics(self, trajectories: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute robustness metrics."""
        if not trajectories:
            return {}

        metrics = {}

        # Recovery time analysis
        recovery_times = []
        perturbation_responses = []

        for traj in trajectories:
            infos = traj.get("infos", [])

            for info in infos:
                if "recovery_time" in info and info["recovery_time"] > 0:
                    recovery_times.append(info["recovery_time"])

                if "perturbation_response" in info:
                    perturbation_responses.append(info["perturbation_response"])

        if recovery_times:
            metrics["recovery_time_mean"] = np.mean(recovery_times)
            metrics["recovery_time_std"] = np.std(recovery_times)
            metrics["recovery_time_max"] = np.max(recovery_times)

        # Stability under perturbations
        if perturbation_responses:
            metrics["perturbation_stability"] = np.mean(perturbation_responses)

        # Fall rate
        falls = sum(1 for traj in trajectories if self._detect_fall(traj))
        metrics["fall_rate"] = falls / len(trajectories) if trajectories else 0.0

        # Variance in performance across conditions
        episode_rewards = [traj.get("episode_reward", 0) for traj in trajectories]
        if episode_rewards:
            metrics["performance_variance"] = np.var(episode_rewards)
            metrics["performance_cv"] = np.std(episode_rewards) / max(np.mean(episode_rewards), 1e-6)

        return metrics

    def _detect_fall(self, trajectory: Dict[str, Any]) -> bool:
        """Detect if robot fell during episode."""
        infos = trajectory.get("infos", [])

        for info in infos:
            # Check base height
            if "base_height" in info and info["base_height"] < 0.3:
                return True

            # Check episode termination reason
            if info.get("termination_reason") == "fall":
                return True

        # Check if episode ended prematurely
        episode_length = trajectory.get("episode_length", 0)
        if episode_length < 200:  # Very short episode likely indicates fall
            return True

        return False


class EfficiencyMetrics(BaseMetrics):
    """Metrics for energy efficiency and control effort."""

    def compute_metrics(self, trajectories: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute efficiency metrics."""
        if not trajectories:
            return {}

        metrics = {}

        # Energy consumption analysis
        total_energy = []
        energy_per_step = []
        action_magnitudes = []

        for traj in trajectories:
            actions = traj.get("actions", [])
            episode_length = traj.get("episode_length", len(actions))
            episode_reward = traj.get("episode_reward", 0)

            if actions:
                # Compute energy metrics
                action_norms = [np.linalg.norm(action) for action in actions]
                episode_energy = np.sum(np.square(action_norms))

                total_energy.append(episode_energy)
                action_magnitudes.extend(action_norms)

                if episode_length > 0:
                    energy_per_step.append(episode_energy / episode_length)

                # Energy efficiency (reward per unit energy)
                if episode_energy > 0:
                    efficiency = episode_reward / episode_energy
                    if "energy_efficiency" not in metrics:
                        metrics["energy_efficiency"] = []
                    metrics.setdefault("energy_efficiency", []).append(efficiency)

        # Aggregate energy metrics
        if total_energy:
            metrics["total_energy_mean"] = np.mean(total_energy)
            metrics["total_energy_std"] = np.std(total_energy)

        if energy_per_step:
            metrics["energy_per_step_mean"] = np.mean(energy_per_step)
            metrics["energy_per_step_std"] = np.std(energy_per_step)

        if action_magnitudes:
            metrics["action_magnitude_mean"] = np.mean(action_magnitudes)
            metrics["action_magnitude_std"] = np.std(action_magnitudes)

        # Control smoothness
        control_smoothness = []
        for traj in trajectories:
            actions = traj.get("actions", [])
            if len(actions) > 1:
                action_diffs = np.diff(actions, axis=0)
                smoothness = np.mean([np.linalg.norm(diff) for diff in action_diffs])
                control_smoothness.append(smoothness)

        if control_smoothness:
            metrics["control_smoothness_mean"] = np.mean(control_smoothness)
            metrics["control_smoothness_std"] = np.std(control_smoothness)

        # Energy efficiency statistics
        if isinstance(metrics.get("energy_efficiency"), list):
            efficiency_values = metrics["energy_efficiency"]
            metrics["energy_efficiency_mean"] = np.mean(efficiency_values)
            metrics["energy_efficiency_std"] = np.std(efficiency_values)
            del metrics["energy_efficiency"]  # Remove list, keep statistics

        return metrics