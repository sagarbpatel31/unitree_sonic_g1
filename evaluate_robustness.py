#!/usr/bin/env python3
"""
Comprehensive evaluation and comparison script for robustness fine-tuning.

This script evaluates and compares different policies (pretrained, direct fine-tuned,
residual fine-tuned) across various robustness metrics including fall rate,
tracking error, and command following quality.
"""

import sys
import logging
import time
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

from sonic_g1.models.policy import G1Policy
from sonic_g1.envs.robust_env import RobustG1Env, DisturbanceConfig
from sonic_g1.eval.metrics import RobustnessMetrics, DetailedMetricsTracker
from sonic_g1.utils.checkpoints import load_checkpoint
from finetune_residual import ResidualPolicy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RobustnessEvaluator:
    """
    Comprehensive evaluator for policy robustness across different disturbance conditions.
    """

    def __init__(self, config: OmegaConf):
        """
        Initialize robustness evaluator.

        Args:
            config: Evaluation configuration
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.use_cuda else 'cpu')

        # Initialize metrics tracker
        self.metrics_tracker = DetailedMetricsTracker(config.metrics)

        # Initialize plot styling
        self._setup_plotting()

        # Results storage
        self.evaluation_results = {}

        logger.info("Initialized RobustnessEvaluator")

    def _setup_plotting(self):
        """Setup matplotlib and seaborn styling."""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12

    def load_policy(self, checkpoint_path: str, policy_type: str = "pretrained") -> torch.nn.Module:
        """
        Load a policy from checkpoint.

        Args:
            checkpoint_path: Path to policy checkpoint
            policy_type: Type of policy ("pretrained", "direct", "residual")

        Returns:
            Loaded policy
        """
        logger.info(f"Loading {policy_type} policy from: {checkpoint_path}")

        checkpoint = load_checkpoint(checkpoint_path, self.device)

        obs_dim = checkpoint['obs_dim']
        action_dim = checkpoint['action_dim']
        policy_config = checkpoint.get('policy_config', {})

        if policy_type == "residual":
            # Load base policy + residual policy
            base_policy = G1Policy(obs_dim, action_dim, policy_config).to(self.device)
            base_policy.load_state_dict(checkpoint['base_policy_state_dict'])
            base_policy.eval()

            residual_policy = ResidualPolicy(obs_dim, action_dim, checkpoint.get('residual_config', {})).to(self.device)
            residual_policy.load_state_dict(checkpoint['policy_state_dict'])
            residual_policy.eval()

            # Create combined policy wrapper
            return ResidualPolicyWrapper(base_policy, residual_policy)

        else:
            # Load regular policy (pretrained or direct fine-tuned)
            policy = G1Policy(obs_dim, action_dim, policy_config).to(self.device)
            policy.load_state_dict(checkpoint['policy_state_dict'])
            policy.eval()
            return policy

    def create_environment(self, disturbance_level: str) -> RobustG1Env:
        """
        Create evaluation environment with specified disturbance level.

        Args:
            disturbance_level: Disturbance intensity ("none", "low", "medium", "high")

        Returns:
            Configured environment
        """
        # Define disturbance parameters based on level
        if disturbance_level == "none":
            disturbance_config = DisturbanceConfig(
                enable_pushes=False,
                friction_range=[1.0, 1.0],
                mass_range=[1.0, 1.0],
                motor_strength_range=[1.0, 1.0],
                obs_noise_std=0.0,
                action_delay_steps=0,
                enable_terrain=False,
                enable_commands=True,
                speed_command_range=[1.0, 1.0],
                turn_command_range=[0.0, 0.0]
            )
        elif disturbance_level == "low":
            disturbance_config = DisturbanceConfig(
                enable_pushes=True,
                push_force_range=[25, 75],
                push_frequency=0.005,
                friction_range=[0.8, 1.2],
                mass_range=[0.95, 1.05],
                motor_strength_range=[0.9, 1.1],
                obs_noise_std=0.005,
                action_delay_steps=0,
                enable_terrain=False,
                enable_commands=True,
                speed_command_range=[0.8, 1.2],
                turn_command_range=[-0.3, 0.3]
            )
        elif disturbance_level == "medium":
            disturbance_config = DisturbanceConfig(
                enable_pushes=True,
                push_force_range=[50, 150],
                push_frequency=0.02,
                friction_range=[0.6, 1.4],
                mass_range=[0.85, 1.15],
                motor_strength_range=[0.8, 1.2],
                obs_noise_std=0.01,
                action_delay_steps=1,
                enable_terrain=True,
                terrain_roughness=0.03,
                enable_commands=True,
                speed_command_range=[0.5, 1.5],
                turn_command_range=[-0.7, 0.7]
            )
        else:  # high
            disturbance_config = DisturbanceConfig(
                enable_pushes=True,
                push_force_range=[100, 250],
                push_frequency=0.04,
                friction_range=[0.4, 1.8],
                mass_range=[0.7, 1.3],
                motor_strength_range=[0.6, 1.4],
                obs_noise_std=0.02,
                action_delay_steps=2,
                enable_terrain=True,
                terrain_roughness=0.08,
                enable_commands=True,
                speed_command_range=[0.2, 2.0],
                turn_command_range=[-1.2, 1.2]
            )

        env = RobustG1Env(
            model_path=self.config.env.model_path,
            disturbance_config=disturbance_config,
            env_config=self.config.env
        )

        logger.info(f"Created environment with {disturbance_level} disturbances")
        return env

    def evaluate_policy(self,
                       policy: torch.nn.Module,
                       policy_name: str,
                       disturbance_levels: List[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate a policy across different disturbance conditions.

        Args:
            policy: Policy to evaluate
            policy_name: Name for logging
            disturbance_levels: List of disturbance levels to test

        Returns:
            Dictionary of results by disturbance level
        """
        if disturbance_levels is None:
            disturbance_levels = ["none", "low", "medium", "high"]

        logger.info(f"Evaluating policy: {policy_name}")

        policy_results = {}

        for disturbance_level in disturbance_levels:
            logger.info(f"Testing {policy_name} with {disturbance_level} disturbances")

            # Create environment for this disturbance level
            env = self.create_environment(disturbance_level)

            # Run evaluation episodes
            episode_results = []

            for episode in range(self.config.num_episodes):
                result = self._run_episode(policy, env, episode)
                episode_results.append(result)

            # Aggregate results for this disturbance level
            level_results = self.metrics_tracker.aggregate_episodes(episode_results)
            policy_results[disturbance_level] = level_results

            logger.info(f"{policy_name} - {disturbance_level}: "
                       f"Success Rate: {level_results['success_rate']:.3f}, "
                       f"Fall Rate: {level_results['fall_rate']:.3f}")

        return policy_results

    def _run_episode(self, policy: torch.nn.Module, env: RobustG1Env, episode_num: int) -> Dict[str, Any]:
        """
        Run a single evaluation episode.

        Args:
            policy: Policy to evaluate
            env: Environment
            episode_num: Episode number

        Returns:
            Episode results
        """
        obs = env.reset()
        done = False
        step = 0

        # Episode tracking
        episode_data = {
            'rewards': [],
            'tracking_errors': [],
            'command_following_errors': [],
            'energy_consumption': [],
            'actions': [],
            'observations': [],
            'fell': False,
            'fall_step': None,
            'command_speeds': [],
            'actual_speeds': [],
            'command_turns': [],
            'actual_turns': []
        }

        total_reward = 0.0

        while not done and step < self.config.max_episode_steps:
            # Get action from policy
            with torch.no_grad():
                if hasattr(policy, 'get_action'):
                    # For residual policies
                    action = policy.get_action(obs, deterministic=True)
                else:
                    # For regular policies
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                    action_dist = policy.get_distribution(obs_tensor)
                    action = action_dist.mean.cpu().numpy()[0]

            # Execute action
            obs_next, reward, done, info = env.step(action)

            # Track episode data
            episode_data['rewards'].append(reward)
            episode_data['actions'].append(action.copy())
            episode_data['observations'].append(obs.copy())

            # Extract info metrics
            tracking_error = info.get('tracking_error', 0.0)
            command_error = info.get('command_following_error', 0.0)
            energy = info.get('energy_consumption', 0.0)

            episode_data['tracking_errors'].append(tracking_error)
            episode_data['command_following_errors'].append(command_error)
            episode_data['energy_consumption'].append(energy)

            # Command tracking
            command_speed = info.get('command_speed', 0.0)
            actual_speed = info.get('actual_speed', 0.0)
            command_turn = info.get('command_turn', 0.0)
            actual_turn = info.get('actual_turn', 0.0)

            episode_data['command_speeds'].append(command_speed)
            episode_data['actual_speeds'].append(actual_speed)
            episode_data['command_turns'].append(command_turn)
            episode_data['actual_turns'].append(actual_turn)

            # Check for fall
            if info.get('fell', False) and not episode_data['fell']:
                episode_data['fell'] = True
                episode_data['fall_step'] = step

            total_reward += reward
            step += 1
            obs = obs_next

        # Compute episode-level metrics
        episode_results = {
            'episode_reward': total_reward,
            'episode_length': step,
            'success': not episode_data['fell'] and step >= self.config.min_success_steps,
            'fell': episode_data['fell'],
            'fall_step': episode_data['fall_step'],

            # Tracking metrics
            'avg_tracking_error': np.mean(episode_data['tracking_errors']),
            'max_tracking_error': np.max(episode_data['tracking_errors']),
            'tracking_error_std': np.std(episode_data['tracking_errors']),

            # Command following metrics
            'avg_command_error': np.mean(episode_data['command_following_errors']),
            'speed_tracking_rmse': np.sqrt(np.mean((np.array(episode_data['command_speeds']) -
                                                   np.array(episode_data['actual_speeds'])) ** 2)),
            'turn_tracking_rmse': np.sqrt(np.mean((np.array(episode_data['command_turns']) -
                                                  np.array(episode_data['actual_turns'])) ** 2)),

            # Energy efficiency
            'total_energy': np.sum(episode_data['energy_consumption']),
            'energy_per_step': np.mean(episode_data['energy_consumption']),

            # Raw data for detailed analysis
            'episode_data': episode_data
        }

        return episode_results

    def compare_policies(self, policy_configs: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Compare multiple policies across robustness metrics.

        Args:
            policy_configs: List of dicts with 'name', 'path', 'type' keys

        Returns:
            Comparison results
        """
        logger.info(f"Comparing {len(policy_configs)} policies")

        comparison_results = {}
        disturbance_levels = ["none", "low", "medium", "high"]

        for policy_config in policy_configs:
            policy_name = policy_config['name']
            policy_path = policy_config['path']
            policy_type = policy_config.get('type', 'pretrained')

            # Load policy
            policy = self.load_policy(policy_path, policy_type)

            # Evaluate policy
            policy_results = self.evaluate_policy(policy, policy_name, disturbance_levels)
            comparison_results[policy_name] = policy_results

        # Store results
        self.evaluation_results = comparison_results

        return comparison_results

    def generate_plots(self, results: Dict[str, Any], output_dir: str):
        """
        Generate comprehensive plots for robustness evaluation.

        Args:
            results: Evaluation results
            output_dir: Directory to save plots
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Generating plots in: {output_path}")

        # 1. Fall Rate Comparison
        self._plot_fall_rates(results, output_path / "fall_rates.png")

        # 2. Tracking Error Comparison
        self._plot_tracking_errors(results, output_path / "tracking_errors.png")

        # 3. Command Following Quality
        self._plot_command_following(results, output_path / "command_following.png")

        # 4. Success Rate by Disturbance Level
        self._plot_success_rates(results, output_path / "success_rates.png")

        # 5. Energy Efficiency
        self._plot_energy_efficiency(results, output_path / "energy_efficiency.png")

        # 6. Robustness Summary Dashboard
        self._plot_robustness_dashboard(results, output_path / "robustness_dashboard.png")

        logger.info("Plot generation completed")

    def _plot_fall_rates(self, results: Dict[str, Any], save_path: Path):
        """Plot fall rate comparison across policies and disturbance levels."""
        fig, ax = plt.subplots(figsize=(12, 6))

        policies = list(results.keys())
        disturbance_levels = ["none", "low", "medium", "high"]

        x = np.arange(len(disturbance_levels))
        width = 0.8 / len(policies)

        for i, policy in enumerate(policies):
            fall_rates = [results[policy][level]['fall_rate'] for level in disturbance_levels]
            ax.bar(x + i * width, fall_rates, width, label=policy, alpha=0.8)

        ax.set_xlabel('Disturbance Level')
        ax.set_ylabel('Fall Rate')
        ax.set_title('Fall Rate Comparison Across Policies')
        ax.set_xticks(x + width * (len(policies) - 1) / 2)
        ax.set_xticklabels(disturbance_levels)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_tracking_errors(self, results: Dict[str, Any], save_path: Path):
        """Plot tracking error comparison."""
        fig, ax = plt.subplots(figsize=(12, 6))

        policies = list(results.keys())
        disturbance_levels = ["none", "low", "medium", "high"]

        x = np.arange(len(disturbance_levels))
        width = 0.8 / len(policies)

        for i, policy in enumerate(policies):
            tracking_errors = [results[policy][level]['avg_tracking_error'] for level in disturbance_levels]
            ax.bar(x + i * width, tracking_errors, width, label=policy, alpha=0.8)

        ax.set_xlabel('Disturbance Level')
        ax.set_ylabel('Average Tracking Error')
        ax.set_title('Motion Tracking Error Comparison')
        ax.set_xticks(x + width * (len(policies) - 1) / 2)
        ax.set_xticklabels(disturbance_levels)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_command_following(self, results: Dict[str, Any], save_path: Path):
        """Plot command following quality."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        policies = list(results.keys())
        disturbance_levels = ["none", "low", "medium", "high"]

        x = np.arange(len(disturbance_levels))
        width = 0.8 / len(policies)

        # Speed tracking
        for i, policy in enumerate(policies):
            speed_rmse = [results[policy][level]['speed_tracking_rmse'] for level in disturbance_levels]
            ax1.bar(x + i * width, speed_rmse, width, label=policy, alpha=0.8)

        ax1.set_xlabel('Disturbance Level')
        ax1.set_ylabel('Speed Tracking RMSE (m/s)')
        ax1.set_title('Speed Command Following')
        ax1.set_xticks(x + width * (len(policies) - 1) / 2)
        ax1.set_xticklabels(disturbance_levels)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Turn tracking
        for i, policy in enumerate(policies):
            turn_rmse = [results[policy][level]['turn_tracking_rmse'] for level in disturbance_levels]
            ax2.bar(x + i * width, turn_rmse, width, label=policy, alpha=0.8)

        ax2.set_xlabel('Disturbance Level')
        ax2.set_ylabel('Turn Tracking RMSE (rad/s)')
        ax2.set_title('Turn Command Following')
        ax2.set_xticks(x + width * (len(policies) - 1) / 2)
        ax2.set_xticklabels(disturbance_levels)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_success_rates(self, results: Dict[str, Any], save_path: Path):
        """Plot success rates across conditions."""
        fig, ax = plt.subplots(figsize=(12, 6))

        policies = list(results.keys())
        disturbance_levels = ["none", "low", "medium", "high"]

        for policy in policies:
            success_rates = [results[policy][level]['success_rate'] for level in disturbance_levels]
            ax.plot(disturbance_levels, success_rates, marker='o', linewidth=2, markersize=8, label=policy)

        ax.set_xlabel('Disturbance Level')
        ax.set_ylabel('Success Rate')
        ax.set_title('Success Rate vs Disturbance Intensity')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_energy_efficiency(self, results: Dict[str, Any], save_path: Path):
        """Plot energy efficiency comparison."""
        fig, ax = plt.subplots(figsize=(12, 6))

        policies = list(results.keys())
        disturbance_levels = ["none", "low", "medium", "high"]

        x = np.arange(len(disturbance_levels))
        width = 0.8 / len(policies)

        for i, policy in enumerate(policies):
            energy_per_step = [results[policy][level]['avg_energy_per_step'] for level in disturbance_levels]
            ax.bar(x + i * width, energy_per_step, width, label=policy, alpha=0.8)

        ax.set_xlabel('Disturbance Level')
        ax.set_ylabel('Energy per Step')
        ax.set_title('Energy Efficiency Comparison')
        ax.set_xticks(x + width * (len(policies) - 1) / 2)
        ax.set_xticklabels(disturbance_levels)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_robustness_dashboard(self, results: Dict[str, Any], save_path: Path):
        """Create comprehensive robustness dashboard."""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Robustness Evaluation Dashboard', fontsize=16)

        policies = list(results.keys())
        disturbance_levels = ["none", "low", "medium", "high"]

        # 1. Success Rate Heatmap
        ax = axes[0, 0]
        success_data = []
        for policy in policies:
            policy_success = [results[policy][level]['success_rate'] for level in disturbance_levels]
            success_data.append(policy_success)

        im = ax.imshow(success_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        ax.set_xticks(range(len(disturbance_levels)))
        ax.set_xticklabels(disturbance_levels)
        ax.set_yticks(range(len(policies)))
        ax.set_yticklabels(policies)
        ax.set_title('Success Rate Heatmap')
        plt.colorbar(im, ax=ax)

        # Add text annotations
        for i in range(len(policies)):
            for j in range(len(disturbance_levels)):
                text = ax.text(j, i, f'{success_data[i][j]:.2f}',
                             ha="center", va="center", color="black", fontweight='bold')

        # 2. Fall Rate Line Plot
        ax = axes[0, 1]
        for policy in policies:
            fall_rates = [results[policy][level]['fall_rate'] for level in disturbance_levels]
            ax.plot(disturbance_levels, fall_rates, marker='o', linewidth=2, label=policy)
        ax.set_title('Fall Rate vs Disturbance Level')
        ax.set_ylabel('Fall Rate')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Tracking Error Box Plot
        ax = axes[0, 2]
        tracking_data = []
        labels = []
        for policy in policies:
            for level in disturbance_levels:
                # Get episode-level tracking errors for box plot
                episodes = results[policy][level].get('episodes', [])
                if episodes:
                    errors = [ep.get('avg_tracking_error', 0) for ep in episodes]
                    tracking_data.append(errors)
                    labels.append(f'{policy}\n{level}')

        if tracking_data:
            ax.boxplot(tracking_data, labels=labels)
            ax.set_title('Tracking Error Distribution')
            ax.set_ylabel('Tracking Error')
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        # 4. Command Following Comparison
        ax = axes[1, 0]
        x = np.arange(len(disturbance_levels))
        width = 0.35

        for i, metric in enumerate(['speed_tracking_rmse', 'turn_tracking_rmse']):
            offset = (i - 0.5) * width
            for j, policy in enumerate(policies):
                values = [results[policy][level][metric] for level in disturbance_levels]
                ax.bar(x + offset + j * width / len(policies), values,
                      width / len(policies), label=f'{policy} {metric.split("_")[0]}',
                      alpha=0.7)

        ax.set_xlabel('Disturbance Level')
        ax.set_ylabel('RMSE')
        ax.set_title('Command Following Quality')
        ax.set_xticks(x)
        ax.set_xticklabels(disturbance_levels)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # 5. Energy Efficiency
        ax = axes[1, 1]
        for policy in policies:
            energy_values = [results[policy][level]['avg_energy_per_step'] for level in disturbance_levels]
            ax.plot(disturbance_levels, energy_values, marker='s', linewidth=2, label=policy)
        ax.set_title('Energy Efficiency')
        ax.set_ylabel('Energy per Step')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 6. Overall Robustness Score
        ax = axes[1, 2]
        robustness_scores = []
        for policy in policies:
            # Compute weighted robustness score
            score = 0
            for level in disturbance_levels:
                success_rate = results[policy][level]['success_rate']
                fall_rate = results[policy][level]['fall_rate']
                tracking_error = results[policy][level]['avg_tracking_error']

                # Simple robustness metric (can be improved)
                level_score = success_rate * (1 - fall_rate) * (1 / (1 + tracking_error))
                score += level_score

            robustness_scores.append(score / len(disturbance_levels))

        ax.bar(policies, robustness_scores, alpha=0.8)
        ax.set_title('Overall Robustness Score')
        ax.set_ylabel('Robustness Score')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save evaluation results to JSON file."""
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = self._make_json_serializable(results)

        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        logger.info(f"Results saved to: {output_path}")

    def _make_json_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects for JSON."""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        else:
            return obj

    def print_summary(self, results: Dict[str, Any]):
        """Print summary of evaluation results."""
        print("\n" + "="*80)
        print("ROBUSTNESS EVALUATION SUMMARY")
        print("="*80)

        disturbance_levels = ["none", "low", "medium", "high"]

        for policy_name, policy_results in results.items():
            print(f"\n{policy_name.upper()}:")
            print("-" * 40)

            for level in disturbance_levels:
                metrics = policy_results[level]
                print(f"{level.upper():>8}: Success={metrics['success_rate']:.3f}, "
                      f"Fall={metrics['fall_rate']:.3f}, "
                      f"Tracking={metrics['avg_tracking_error']:.3f}, "
                      f"Speed RMSE={metrics['speed_tracking_rmse']:.3f}")

        print("\n" + "="*80)


class ResidualPolicyWrapper:
    """Wrapper for combining base and residual policies."""

    def __init__(self, base_policy: torch.nn.Module, residual_policy: torch.nn.Module):
        self.base_policy = base_policy
        self.residual_policy = residual_policy

    def get_action(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Get combined action from base + residual policies."""
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)

        with torch.no_grad():
            # Base action
            base_action_dist = self.base_policy.get_distribution(obs_tensor)
            base_action = base_action_dist.mean if deterministic else base_action_dist.sample()

            # Residual action
            residual_action = self.residual_policy(obs_tensor)

            # Combined action
            combined_action = base_action + residual_action

        return combined_action.cpu().numpy()[0]


def main():
    """Main evaluation function."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate robustness of G1 policies")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to evaluation configuration")
    parser.add_argument("--policies", type=str, nargs='+', required=True,
                       help="Paths to policy checkpoints")
    parser.add_argument("--policy_names", type=str, nargs='+',
                       help="Names for policies (optional)")
    parser.add_argument("--policy_types", type=str, nargs='+',
                       choices=['pretrained', 'direct', 'residual'],
                       help="Types of policies (optional)")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                       help="Output directory for results and plots")

    args = parser.parse_args()

    # Load configuration
    config = OmegaConf.load(args.config)

    # Prepare policy configurations
    policy_configs = []
    for i, policy_path in enumerate(args.policies):
        policy_name = args.policy_names[i] if args.policy_names and i < len(args.policy_names) else f"Policy_{i+1}"
        policy_type = args.policy_types[i] if args.policy_types and i < len(args.policy_types) else "pretrained"

        policy_configs.append({
            'name': policy_name,
            'path': policy_path,
            'type': policy_type
        })

    # Initialize evaluator
    evaluator = RobustnessEvaluator(config)

    # Run comparison
    results = evaluator.compare_policies(policy_configs)

    # Generate outputs
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save results
    evaluator.save_results(results, str(output_path / "evaluation_results.json"))

    # Generate plots
    evaluator.generate_plots(results, str(output_path / "plots"))

    # Print summary
    evaluator.print_summary(results)

    print(f"\nEvaluation completed. Results saved to: {output_path}")


if __name__ == "__main__":
    main()