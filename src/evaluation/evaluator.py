"""
Comprehensive model evaluation system.
Supports multiple evaluation scenarios and metrics.
"""

import numpy as np
import torch
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
from tqdm import tqdm

from ..core.config import Config
from ..core.logging import Logger, MetricsTracker
from ..envs import create_g1_environment
from ..models.transformer_policy import TransformerPolicy
from .metrics import TrackingMetrics, RobustnessMetrics, EfficiencyMetrics


class ModelEvaluator:
    """
    Comprehensive model evaluator for G1 policies.

    This evaluator runs multiple evaluation scenarios and computes
    detailed metrics for model performance assessment.
    """

    def __init__(self, model: TransformerPolicy, config: Config, logger: Logger):
        self.model = model
        self.config = config
        self.logger = logger
        self.eval_config = config.get("evaluation", {})

        # Setup device
        self.device = torch.device(config.get("device", "auto"))
        if self.device.type == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = self.model.to(self.device)
        self.model.eval()

        # Metrics trackers
        self.tracking_metrics = TrackingMetrics()
        self.robustness_metrics = RobustnessMetrics()
        self.efficiency_metrics = EfficiencyMetrics()

        # Evaluation scenarios
        self.scenarios = self._load_evaluation_scenarios()

        self.logger.info(f"Loaded {len(self.scenarios)} evaluation scenarios")

    def _load_evaluation_scenarios(self) -> List[Dict[str, Any]]:
        """Load evaluation scenarios from configuration."""
        scenarios = self.config.get("scenarios", [])

        # Add default scenarios if none specified
        if not scenarios:
            scenarios = [
                {
                    "name": "baseline",
                    "description": "Baseline performance",
                    "episodes": 100,
                    "metrics": ["tracking_error", "success_rate"],
                }
            ]

        return scenarios

    def evaluate_all_scenarios(self) -> Dict[str, Dict[str, Any]]:
        """Run evaluation on all configured scenarios."""
        results = {}

        for scenario in self.scenarios:
            self.logger.info(f"Evaluating scenario: {scenario['name']}")
            scenario_results = self.evaluate_scenario(scenario)
            results[scenario["name"]] = scenario_results

        # Compute aggregate metrics
        results["summary"] = self._compute_summary_metrics(results)

        return results

    def evaluate_scenario(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate model on a specific scenario."""
        scenario_name = scenario["name"]
        num_episodes = scenario.get("episodes", 50)
        requested_metrics = scenario.get("metrics", ["tracking_error", "success_rate"])

        # Create environment for scenario
        env = self._create_scenario_environment(scenario)

        # Reset metrics trackers
        episode_metrics = MetricsTracker()
        episode_trajectories = []

        # Run episodes
        for episode in tqdm(range(num_episodes), desc=f"Scenario: {scenario_name}"):
            trajectory = self._run_episode(env, episode)
            episode_trajectories.append(trajectory)

            # Compute episode metrics
            ep_metrics = self._compute_episode_metrics(trajectory, scenario)
            episode_metrics.update(ep_metrics)

        # Aggregate results
        avg_metrics = episode_metrics.get_averages()

        # Compute detailed metrics
        detailed_metrics = {}
        if "tracking_error" in requested_metrics:
            detailed_metrics.update(self.tracking_metrics.compute_metrics(episode_trajectories))

        if "robustness" in requested_metrics:
            detailed_metrics.update(self.robustness_metrics.compute_metrics(episode_trajectories))

        if "efficiency" in requested_metrics:
            detailed_metrics.update(self.efficiency_metrics.compute_metrics(episode_trajectories))

        results = {
            "scenario": scenario,
            "num_episodes": num_episodes,
            "average_metrics": avg_metrics,
            "detailed_metrics": detailed_metrics,
            "trajectories": episode_trajectories if self.eval_config.get("save_trajectories", False) else None,
        }

        return results

    def _create_scenario_environment(self, scenario: Dict[str, Any]) -> Any:
        """Create environment configured for specific scenario."""
        # Create base environment
        env = create_g1_environment(self.config)

        # Apply scenario-specific perturbations
        perturbations = scenario.get("perturbations", [])
        self._apply_perturbations(env, perturbations)

        return env

    def _apply_perturbations(self, env, perturbations: List[str]):
        """Apply perturbations to environment for robustness testing."""
        for perturbation in perturbations:
            if perturbation == "mass_random":
                if hasattr(env, "set_mass_multiplier"):
                    multiplier = np.random.uniform(0.7, 1.3)
                    env.set_mass_multiplier(multiplier)

            elif perturbation == "friction_random":
                if hasattr(env, "set_friction_multiplier"):
                    multiplier = np.random.uniform(0.3, 2.0)
                    env.set_friction_multiplier(multiplier)

            elif perturbation == "force_random":
                if hasattr(env, "enable_external_forces"):
                    env.enable_external_forces(max_force=100.0)

            elif perturbation == "noise_high":
                if hasattr(env, "set_observation_noise"):
                    env.set_observation_noise(0.2)

    def _run_episode(self, env, episode_idx: int) -> Dict[str, Any]:
        """Run a single evaluation episode."""
        obs, info = env.reset(seed=self.config.get("seed", 0) + episode_idx)
        done = False
        truncated = False

        trajectory = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "infos": [],
            "episode_length": 0,
            "episode_reward": 0.0,
            "success": False,
        }

        step_count = 0
        while not (done or truncated) and step_count < self.eval_config.get("max_steps", 1000):
            # Get observation tensor
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

            # Get action from model
            with torch.no_grad():
                action = self.model.get_action(obs_tensor, deterministic=True)
                action = action.cpu().numpy()

            # Step environment
            next_obs, reward, done, truncated, info = env.step(action)

            # Store trajectory data
            trajectory["observations"].append(obs.copy())
            trajectory["actions"].append(action.copy())
            trajectory["rewards"].append(reward)
            trajectory["infos"].append(info.copy())

            trajectory["episode_reward"] += reward
            step_count += 1

            obs = next_obs

        trajectory["episode_length"] = step_count
        trajectory["success"] = self._determine_episode_success(trajectory, info)

        return trajectory

    def _determine_episode_success(self, trajectory: Dict[str, Any], final_info: Dict[str, Any]) -> bool:
        """Determine if episode was successful."""
        # Simple success criteria - could be more sophisticated
        episode_length = trajectory["episode_length"]
        min_length = self.eval_config.get("min_success_length", 500)

        # Episode must be reasonably long
        length_success = episode_length >= min_length

        # Check tracking performance
        tracking_errors = [info.get("tracking_error", 1.0) for info in trajectory["infos"]]
        if tracking_errors:
            avg_tracking_error = np.mean(tracking_errors)
            tracking_success = avg_tracking_error < 0.15  # radians
        else:
            tracking_success = True

        return length_success and tracking_success

    def _compute_episode_metrics(self, trajectory: Dict[str, Any], scenario: Dict[str, Any]) -> Dict[str, float]:
        """Compute metrics for a single episode."""
        metrics = {}

        # Basic metrics
        metrics["episode_length"] = trajectory["episode_length"]
        metrics["episode_reward"] = trajectory["episode_reward"]
        metrics["success"] = 1.0 if trajectory["success"] else 0.0

        # Tracking metrics
        tracking_errors = [info.get("tracking_error", 0.0) for info in trajectory["infos"]]
        if tracking_errors:
            metrics["tracking_error_mean"] = np.mean(tracking_errors)
            metrics["tracking_error_max"] = np.max(tracking_errors)
            metrics["tracking_error_std"] = np.std(tracking_errors)

        # Energy metrics
        actions = trajectory["actions"]
        if actions:
            action_magnitudes = [np.linalg.norm(action) for action in actions]
            metrics["energy_consumption"] = np.sum(np.square(action_magnitudes))
            metrics["energy_efficiency"] = metrics["episode_reward"] / max(metrics["energy_consumption"], 1e-6)

        # Robustness metrics
        if "recovery_time" in [info.keys() for info in trajectory["infos"]][0]:
            recovery_times = [info.get("recovery_time", 0.0) for info in trajectory["infos"] if "recovery_time" in info]
            if recovery_times:
                metrics["recovery_time_mean"] = np.mean(recovery_times)

        return metrics

    def _compute_summary_metrics(self, scenario_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Compute summary metrics across all scenarios."""
        summary = {
            "total_scenarios": len(scenario_results) - 1,  # Exclude summary itself
            "overall_performance": {},
        }

        # Aggregate metrics across scenarios
        all_metrics = {}
        for scenario_name, results in scenario_results.items():
            if scenario_name == "summary":
                continue

            avg_metrics = results.get("average_metrics", {})
            for metric_name, value in avg_metrics.items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(value)

        # Compute overall statistics
        for metric_name, values in all_metrics.items():
            summary["overall_performance"][metric_name] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
            }

        return summary

    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save evaluation results to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert numpy arrays to lists for JSON serialization
        serializable_results = self._make_serializable(results)

        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        self.logger.info(f"Evaluation results saved to: {output_path}")

    def _make_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to JSON-compatible format."""
        if isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj

    def generate_report(self, results: Dict[str, Any], output_dir: str):
        """Generate comprehensive evaluation report."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save detailed results
        self.save_results(results, output_dir / "detailed_results.json")

        # Generate summary report
        self._generate_summary_report(results, output_dir / "summary.txt")

        # Generate plots if visualization is available
        try:
            from ..utils.visualization import create_evaluation_plots
            create_evaluation_plots(results, output_dir / "plots")
        except ImportError:
            self.logger.warning("Visualization tools not available, skipping plots")

        self.logger.info(f"Evaluation report generated in: {output_dir}")

    def _generate_summary_report(self, results: Dict[str, Any], output_path: Path):
        """Generate text summary report."""
        with open(output_path, 'w') as f:
            f.write("Unitree G1 Model Evaluation Report\n")
            f.write("=" * 40 + "\n\n")

            # Overall summary
            summary = results.get("summary", {})
            f.write(f"Total Scenarios: {summary.get('total_scenarios', 0)}\n\n")

            # Scenario results
            for scenario_name, scenario_results in results.items():
                if scenario_name == "summary":
                    continue

                f.write(f"Scenario: {scenario_name}\n")
                f.write("-" * 20 + "\n")

                scenario_info = scenario_results.get("scenario", {})
                f.write(f"Description: {scenario_info.get('description', 'N/A')}\n")
                f.write(f"Episodes: {scenario_results.get('num_episodes', 0)}\n\n")

                avg_metrics = scenario_results.get("average_metrics", {})
                for metric_name, value in avg_metrics.items():
                    f.write(f"  {metric_name}: {value:.4f}\n")

                f.write("\n")

            # Overall performance
            f.write("Overall Performance Summary\n")
            f.write("-" * 30 + "\n")

            overall_perf = summary.get("overall_performance", {})
            for metric_name, stats in overall_perf.items():
                f.write(f"{metric_name}:\n")
                f.write(f"  Mean: {stats.get('mean', 0):.4f}\n")
                f.write(f"  Std:  {stats.get('std', 0):.4f}\n")
                f.write(f"  Range: [{stats.get('min', 0):.4f}, {stats.get('max', 0):.4f}]\n\n")