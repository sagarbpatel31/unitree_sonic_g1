#!/usr/bin/env python3
"""
Fine-tuning script for pretrained Unitree G1 whole-body controller.

This script fine-tunes a pretrained policy with robustness improvements including:
- Random pushes and disturbances
- Physical parameter variations
- Observation noise and action delays
- Command-conditioned control
- Terrain variations

Supports both direct policy fine-tuning and residual RL approaches.
"""

import sys
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
import wandb
from tensorboardX import SummaryWriter

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from sonic_g1.models.policy import G1Policy
from sonic_g1.envs.robust_env import RobustG1Env, DisturbanceConfig
from sonic_g1.train.ppo import PPOTrainer
from sonic_g1.train.residual_trainer import ResidualTrainer
from sonic_g1.eval.metrics import RobustnessMetrics
from sonic_g1.utils.checkpoints import load_checkpoint, save_checkpoint

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PolicyFinetuner:
    """
    Fine-tuner for pretrained G1 policies with robustness improvements.

    Supports both direct policy fine-tuning and residual RL approaches
    while preserving motion priors.
    """

    def __init__(self, config: OmegaConf):
        """
        Initialize policy fine-tuner.

        Args:
            config: Fine-tuning configuration
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.training.use_cuda else 'cpu')

        # Initialize logging
        self._setup_logging()

        # Load pretrained policy
        self.base_policy = self._load_pretrained_policy()

        # Create robust environment
        self.env = self._create_robust_environment()

        # Initialize trainer based on approach
        if config.approach == "residual":
            self.trainer = self._create_residual_trainer()
        else:
            self.trainer = self._create_direct_trainer()

        # Initialize metrics tracking
        self.metrics = RobustnessMetrics(config.eval)

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_success_rate = 0.0

        logger.info(f"Initialized PolicyFinetuner with approach: {config.approach}")

    def _setup_logging(self):
        """Setup logging backends."""
        self.log_dir = Path(self.config.logging.log_dir) / self.config.experiment.name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard
        if self.config.logging.use_tensorboard:
            self.tb_writer = SummaryWriter(str(self.log_dir / "tensorboard"))
        else:
            self.tb_writer = None

        # Weights & Biases
        if self.config.logging.use_wandb:
            wandb.init(
                project=self.config.logging.wandb_project,
                entity=self.config.logging.wandb_entity,
                name=self.config.experiment.name,
                config=OmegaConf.to_container(self.config, resolve=True),
                tags=self.config.experiment.tags
            )
            self.use_wandb = True
        else:
            self.use_wandb = False

    def _load_pretrained_policy(self) -> nn.Module:
        """Load pretrained policy checkpoint."""
        checkpoint_path = self.config.model.pretrained_checkpoint
        logger.info(f"Loading pretrained policy from: {checkpoint_path}")

        checkpoint = load_checkpoint(checkpoint_path, self.device)

        # Extract model architecture info
        obs_dim = checkpoint['obs_dim']
        action_dim = checkpoint['action_dim']
        policy_config = checkpoint.get('policy_config', self.config.model.policy)

        # Create and load policy
        policy = G1Policy(obs_dim, action_dim, policy_config).to(self.device)
        policy.load_state_dict(checkpoint['policy_state_dict'])

        # Freeze if using residual approach
        if self.config.approach == "residual":
            for param in policy.parameters():
                param.requires_grad = False
            logger.info("Frozen pretrained policy for residual learning")

        return policy

    def _create_robust_environment(self) -> RobustG1Env:
        """Create robustness environment with disturbances."""
        disturbance_config = DisturbanceConfig(
            # Physical disturbances
            enable_pushes=self.config.disturbances.enable_pushes,
            push_force_range=self.config.disturbances.push_force_range,
            push_frequency=self.config.disturbances.push_frequency,

            # Parameter variations
            friction_range=self.config.disturbances.friction_range,
            mass_range=self.config.disturbances.mass_range,
            motor_strength_range=self.config.disturbances.motor_strength_range,

            # Sensor noise
            obs_noise_std=self.config.disturbances.obs_noise_std,
            action_delay_steps=self.config.disturbances.action_delay_steps,

            # Terrain
            enable_terrain=self.config.disturbances.enable_terrain,
            terrain_roughness=self.config.disturbances.terrain_roughness,

            # Commands
            enable_commands=self.config.disturbances.enable_commands,
            speed_command_range=self.config.disturbances.speed_command_range,
            turn_command_range=self.config.disturbances.turn_command_range
        )

        env = RobustG1Env(
            model_path=self.config.env.model_path,
            disturbance_config=disturbance_config,
            env_config=self.config.env
        )

        logger.info(f"Created robust environment with disturbances: {disturbance_config}")
        return env

    def _create_direct_trainer(self) -> PPOTrainer:
        """Create PPO trainer for direct policy fine-tuning."""
        trainer_config = self.config.training.copy()
        trainer_config.approach = "direct"

        trainer = PPOTrainer(
            policy=self.base_policy,
            env=self.env,
            config=trainer_config,
            device=self.device
        )

        # Use lower learning rate for fine-tuning
        for param_group in trainer.optimizer.param_groups:
            param_group['lr'] *= self.config.training.finetune_lr_scale

        logger.info("Created direct policy fine-tuning trainer")
        return trainer

    def _create_residual_trainer(self) -> ResidualTrainer:
        """Create residual trainer for residual RL."""
        trainer = ResidualTrainer(
            base_policy=self.base_policy,
            env=self.env,
            config=self.config.training,
            device=self.device
        )

        logger.info("Created residual RL trainer")
        return trainer

    def finetune(self):
        """Main fine-tuning loop."""
        logger.info("Starting fine-tuning...")

        start_time = time.time()

        for epoch in range(self.config.training.max_epochs):
            self.epoch = epoch

            # Training step
            train_metrics = self._train_epoch()

            # Evaluation step
            if (epoch + 1) % self.config.eval.eval_interval == 0:
                eval_metrics = self._evaluate()

                # Check for improvement
                success_rate = eval_metrics.get('success_rate', 0.0)
                if success_rate > self.best_success_rate:
                    self.best_success_rate = success_rate
                    self._save_checkpoint(is_best=True)
                    logger.info(f"New best success rate: {success_rate:.3f}")

                # Log metrics
                self._log_metrics(train_metrics, eval_metrics, epoch)

                # Early stopping check
                if self._should_early_stop(eval_metrics):
                    logger.info("Early stopping triggered")
                    break

            # Regular checkpointing
            if (epoch + 1) % self.config.training.checkpoint_interval == 0:
                self._save_checkpoint()

        total_time = time.time() - start_time
        logger.info(f"Fine-tuning completed in {total_time:.2f} seconds")

        # Final evaluation
        final_metrics = self._final_evaluation()
        self._log_final_results(final_metrics)

    def _train_epoch(self) -> Dict[str, float]:
        """Execute one training epoch."""
        epoch_metrics = {}

        # Collect rollouts and train
        if self.config.approach == "residual":
            metrics = self.trainer.train_step()
        else:
            # Direct PPO training
            rollouts = self.trainer.collect_rollouts()
            metrics = self.trainer.update_policy(rollouts)

        # Update global step
        self.global_step += self.config.training.steps_per_epoch

        # Log training metrics
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                epoch_metrics[f"train/{key}"] = value

        return epoch_metrics

    def _evaluate(self) -> Dict[str, float]:
        """Evaluate current policy."""
        logger.info(f"Evaluating at epoch {self.epoch}")

        # Run evaluation episodes
        eval_results = []

        for episode in range(self.config.eval.num_episodes):
            result = self._run_evaluation_episode()
            eval_results.append(result)

        # Aggregate metrics
        eval_metrics = self.metrics.aggregate_results(eval_results)

        logger.info(f"Evaluation - Success Rate: {eval_metrics['success_rate']:.3f}, "
                   f"Fall Rate: {eval_metrics['fall_rate']:.3f}")

        return eval_metrics

    def _run_evaluation_episode(self) -> Dict[str, Any]:
        """Run a single evaluation episode."""
        obs = self.env.reset()
        done = False
        step = 0
        episode_reward = 0.0

        # Episode tracking
        tracking_errors = []
        command_following_errors = []
        fall_occurred = False

        while not done and step < self.config.eval.max_episode_steps:
            # Get action from policy
            if self.config.approach == "residual":
                action = self.trainer.get_action(obs)
            else:
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                    action_dist = self.base_policy.get_distribution(obs_tensor)
                    action = action_dist.sample().cpu().numpy()[0]

            # Execute action
            obs_next, reward, done, info = self.env.step(action)

            episode_reward += reward
            step += 1

            # Track metrics
            tracking_error = info.get('tracking_error', 0.0)
            command_error = info.get('command_following_error', 0.0)

            tracking_errors.append(tracking_error)
            command_following_errors.append(command_error)

            if info.get('fell', False):
                fall_occurred = True

            obs = obs_next

        return {
            'episode_reward': episode_reward,
            'episode_length': step,
            'tracking_errors': tracking_errors,
            'command_following_errors': command_following_errors,
            'fell': fall_occurred,
            'success': not fall_occurred and step >= self.config.eval.min_success_steps
        }

    def _final_evaluation(self) -> Dict[str, Any]:
        """Run comprehensive final evaluation."""
        logger.info("Running final evaluation...")

        # Test different disturbance levels
        disturbance_levels = ['low', 'medium', 'high']
        final_results = {}

        for level in disturbance_levels:
            logger.info(f"Testing disturbance level: {level}")

            # Update environment disturbances
            self._set_disturbance_level(level)

            # Run evaluation
            eval_results = []
            for episode in range(self.config.eval.final_episodes):
                result = self._run_evaluation_episode()
                eval_results.append(result)

            # Aggregate results
            level_metrics = self.metrics.aggregate_results(eval_results)
            final_results[level] = level_metrics

        return final_results

    def _set_disturbance_level(self, level: str):
        """Adjust environment disturbance intensity."""
        if level == 'low':
            scale = 0.5
        elif level == 'medium':
            scale = 1.0
        else:  # high
            scale = 1.5

        self.env.set_disturbance_scale(scale)

    def _log_metrics(self, train_metrics: Dict, eval_metrics: Dict, epoch: int):
        """Log training and evaluation metrics."""
        all_metrics = {**train_metrics, **eval_metrics}

        # TensorBoard logging
        if self.tb_writer:
            for key, value in all_metrics.items():
                if isinstance(value, (int, float)):
                    self.tb_writer.add_scalar(key, value, epoch)

        # Weights & Biases logging
        if self.use_wandb:
            wandb.log(all_metrics, step=epoch)

        # Console logging
        logger.info(f"Epoch {epoch}: {all_metrics}")

    def _log_final_results(self, final_metrics: Dict[str, Any]):
        """Log final evaluation results."""
        logger.info("Final Evaluation Results:")

        for level, metrics in final_metrics.items():
            logger.info(f"  {level.upper()} disturbances:")
            logger.info(f"    Success Rate: {metrics['success_rate']:.3f}")
            logger.info(f"    Fall Rate: {metrics['fall_rate']:.3f}")
            logger.info(f"    Avg Tracking Error: {metrics['avg_tracking_error']:.3f}")
            logger.info(f"    Avg Command Error: {metrics['avg_command_error']:.3f}")

        # Log to W&B
        if self.use_wandb:
            for level, metrics in final_metrics.items():
                wandb.log({f"final_{level}_{k}": v for k, v in metrics.items()})

    def _save_checkpoint(self, is_best: bool = False):
        """Save training checkpoint."""
        checkpoint_dir = Path(self.config.training.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if self.config.approach == "residual":
            policy_state_dict = self.trainer.residual_policy.state_dict()
            extra_data = {
                'base_policy_state_dict': self.base_policy.state_dict(),
                'approach': 'residual'
            }
        else:
            policy_state_dict = self.base_policy.state_dict()
            extra_data = {'approach': 'direct'}

        checkpoint_data = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'policy_state_dict': policy_state_dict,
            'config': self.config,
            'best_success_rate': self.best_success_rate,
            **extra_data
        }

        # Save regular checkpoint
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{self.epoch}.pth"
        save_checkpoint(checkpoint_data, checkpoint_path)

        # Save best checkpoint
        if is_best:
            best_path = checkpoint_dir / "best_checkpoint.pth"
            save_checkpoint(checkpoint_data, best_path)
            logger.info(f"Saved best checkpoint: {best_path}")

    def _should_early_stop(self, eval_metrics: Dict[str, float]) -> bool:
        """Check if early stopping should be triggered."""
        if not hasattr(self, '_patience_counter'):
            self._patience_counter = 0
            self._best_metric = 0.0

        current_metric = eval_metrics.get('success_rate', 0.0)

        if current_metric > self._best_metric:
            self._best_metric = current_metric
            self._patience_counter = 0
        else:
            self._patience_counter += 1

        patience = self.config.training.get('early_stopping_patience', 20)
        return self._patience_counter >= patience


def main():
    """Main function for policy fine-tuning."""
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune pretrained G1 policy")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to fine-tuning configuration file")
    parser.add_argument("--approach", type=str, choices=['direct', 'residual'],
                       default='residual', help="Fine-tuning approach")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to pretrained policy checkpoint")
    parser.add_argument("--output_dir", type=str, default="checkpoints/finetuned",
                       help="Output directory for checkpoints")
    parser.add_argument("--wandb", action='store_true',
                       help="Enable Weights & Biases logging")
    parser.add_argument("--debug", action='store_true',
                       help="Enable debug mode")

    args = parser.parse_args()

    # Load configuration
    config = OmegaConf.load(args.config)

    # Override config with command line arguments
    config.approach = args.approach
    config.model.pretrained_checkpoint = args.checkpoint
    config.training.checkpoint_dir = args.output_dir
    config.logging.use_wandb = args.wandb

    if args.debug:
        config.training.max_epochs = 5
        config.eval.num_episodes = 2
        config.eval.final_episodes = 3

    # Set random seeds
    torch.manual_seed(config.training.seed)
    np.random.seed(config.training.seed)

    # Initialize and run fine-tuning
    finetuner = PolicyFinetuner(config)
    finetuner.finetune()


if __name__ == "__main__":
    main()