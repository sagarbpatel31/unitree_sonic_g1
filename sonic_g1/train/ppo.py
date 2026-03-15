"""
Proximal Policy Optimization (PPO) implementation for Unitree G1 motion imitation.

This module provides a robust PPO trainer with:
- Clipped policy objective
- Value function learning with optional clipping
- Entropy regularization
- Gradient clipping and mixed precision support
- Comprehensive logging and metrics
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, List, Optional, Any, Tuple
from omegaconf import DictConfig
import logging

logger = logging.getLogger(__name__)


class PPOTrainer:
    """PPO trainer for motion imitation with G1 robot."""

    def __init__(self,
                 policy: nn.Module,
                 critic: nn.Module,
                 optimizer_policy: torch.optim.Optimizer,
                 optimizer_critic: torch.optim.Optimizer,
                 config: DictConfig,
                 device: torch.device):
        """
        Initialize PPO trainer.

        Args:
            policy: Policy network
            critic: Value function network
            optimizer_policy: Policy optimizer
            optimizer_critic: Critic optimizer
            config: PPO configuration
            device: Torch device
        """
        self.policy = policy
        self.critic = critic
        self.optimizer_policy = optimizer_policy
        self.optimizer_critic = optimizer_critic
        self.config = config
        self.device = device

        # PPO hyperparameters
        self.clip_ratio = config.clip_ratio
        self.clip_critic = config.get('clip_critic', False)
        self.critic_clip_ratio = config.get('critic_clip_ratio', 0.2)
        self.entropy_coeff = config.entropy_coeff
        self.value_loss_coeff = config.value_loss_coeff
        self.max_grad_norm = config.get('max_grad_norm', 0.5)

        # Training settings
        self.epochs = config.epochs
        self.minibatch_size = config.minibatch_size
        self.target_kl = config.get('target_kl', 0.01)
        self.use_mixed_precision = config.get('use_mixed_precision', False)

        # Initialize mixed precision scaler
        self.scaler_policy = GradScaler(enabled=self.use_mixed_precision)
        self.scaler_critic = GradScaler(enabled=self.use_mixed_precision)

        # Metrics tracking
        self.training_step = 0
        self.policy_updates = 0
        self.critic_updates = 0

        logger.info(f"Initialized PPO trainer with clip_ratio={self.clip_ratio}, "
                   f"epochs={self.epochs}, minibatch_size={self.minibatch_size}")

    def compute_policy_loss(self,
                           observations: torch.Tensor,
                           actions: torch.Tensor,
                           old_log_probs: torch.Tensor,
                           advantages: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute PPO policy loss with clipping.

        Args:
            observations: Observation tensor (batch_size, obs_dim)
            actions: Action tensor (batch_size, action_dim)
            old_log_probs: Old log probabilities (batch_size,)
            advantages: Advantage estimates (batch_size,)

        Returns:
            Tuple of (loss, metrics_dict)
        """
        # Get current policy outputs
        action_dist = self.policy.get_distribution(observations)
        new_log_probs = action_dist.log_prob(actions)
        entropy = action_dist.entropy().mean()

        # Compute probability ratios
        ratio = torch.exp(new_log_probs - old_log_probs)

        # Normalize advantages
        advantages_normalized = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Compute surrogate losses
        surr1 = ratio * advantages_normalized
        surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages_normalized

        # PPO loss
        policy_loss = -torch.min(surr1, surr2).mean()

        # Entropy loss
        entropy_loss = -self.entropy_coeff * entropy

        # Total loss
        total_loss = policy_loss + entropy_loss

        # Compute metrics
        with torch.no_grad():
            # Approximate KL divergence
            approx_kl = (old_log_probs - new_log_probs).mean().item()

            # Fraction of clipped ratios
            clip_fraction = ((ratio - 1.0).abs() > self.clip_ratio).float().mean().item()

            # Policy gradient variance
            pg_variance = (ratio * advantages_normalized).var().item()

        metrics = {
            'policy_loss': policy_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'entropy': entropy.item(),
            'approx_kl': approx_kl,
            'clip_fraction': clip_fraction,
            'pg_variance': pg_variance
        }

        return total_loss, metrics

    def compute_critic_loss(self,
                           observations: torch.Tensor,
                           returns: torch.Tensor,
                           old_values: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute value function loss with optional clipping.

        Args:
            observations: Observation tensor (batch_size, obs_dim)
            returns: Return estimates (batch_size,)
            old_values: Old value predictions (batch_size,)

        Returns:
            Tuple of (loss, metrics_dict)
        """
        # Get current value predictions
        new_values = self.critic(observations).squeeze(-1)

        if self.clip_critic:
            # Clipped value loss (similar to policy clipping)
            value_diff = new_values - old_values
            clipped_values = old_values + torch.clamp(
                value_diff, -self.critic_clip_ratio, self.critic_clip_ratio
            )

            loss1 = F.mse_loss(new_values, returns)
            loss2 = F.mse_loss(clipped_values, returns)
            value_loss = torch.max(loss1, loss2)
        else:
            # Standard MSE loss
            value_loss = F.mse_loss(new_values, returns)

        # Compute metrics
        with torch.no_grad():
            # Explained variance
            returns_var = returns.var()
            residual_var = (returns - new_values).var()
            explained_variance = 1 - (residual_var / (returns_var + 1e-8))
            explained_variance = explained_variance.item()

            # Value prediction statistics
            value_mean = new_values.mean().item()
            value_std = new_values.std().item()
            returns_mean = returns.mean().item()
            returns_std = returns.std().item()

        metrics = {
            'value_loss': value_loss.item(),
            'explained_variance': explained_variance,
            'value_mean': value_mean,
            'value_std': value_std,
            'returns_mean': returns_mean,
            'returns_std': returns_std
        }

        return value_loss, metrics

    def update_policy(self, rollout_buffer) -> Dict[str, float]:
        """
        Update policy using PPO with multiple epochs and minibatches.

        Args:
            rollout_buffer: Rollout buffer containing training data

        Returns:
            Dictionary of training metrics
        """
        all_metrics = {
            'policy_loss': [],
            'entropy_loss': [],
            'entropy': [],
            'approx_kl': [],
            'clip_fraction': [],
            'pg_variance': []
        }

        # Get data from buffer
        observations, actions, old_log_probs, advantages, _ = rollout_buffer.get_policy_data()

        # Training loop
        for epoch in range(self.epochs):
            # Generate random minibatches
            indices = torch.randperm(len(observations), device=self.device)

            for start in range(0, len(observations), self.minibatch_size):
                end = start + self.minibatch_size
                mb_indices = indices[start:end]

                mb_observations = observations[mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_advantages = advantages[mb_indices]

                # Zero gradients
                self.optimizer_policy.zero_grad()

                # Forward pass with mixed precision
                with autocast(enabled=self.use_mixed_precision):
                    loss, metrics = self.compute_policy_loss(
                        mb_observations, mb_actions, mb_old_log_probs, mb_advantages
                    )

                # Backward pass
                if self.use_mixed_precision:
                    self.scaler_policy.scale(loss).backward()
                    # Gradient clipping
                    if self.max_grad_norm > 0:
                        self.scaler_policy.unscale_(self.optimizer_policy)
                        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.scaler_policy.step(self.optimizer_policy)
                    self.scaler_policy.update()
                else:
                    loss.backward()
                    # Gradient clipping
                    if self.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.optimizer_policy.step()

                # Store metrics
                for key, value in metrics.items():
                    all_metrics[key].append(value)

                self.policy_updates += 1

                # Early stopping if KL divergence is too large
                if metrics['approx_kl'] > self.target_kl * 1.5:
                    logger.debug(f"Early stopping at epoch {epoch} due to high KL: {metrics['approx_kl']:.4f}")
                    break

            # Early stopping check for epoch level
            if len(all_metrics['approx_kl']) > 0 and all_metrics['approx_kl'][-1] > self.target_kl * 1.5:
                break

        # Average metrics
        avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}

        return avg_metrics

    def update_critic(self, rollout_buffer) -> Dict[str, float]:
        """
        Update value function using multiple epochs and minibatches.

        Args:
            rollout_buffer: Rollout buffer containing training data

        Returns:
            Dictionary of training metrics
        """
        all_metrics = {
            'value_loss': [],
            'explained_variance': [],
            'value_mean': [],
            'value_std': [],
            'returns_mean': [],
            'returns_std': []
        }

        # Get data from buffer
        observations, returns, old_values = rollout_buffer.get_critic_data()

        # Training loop
        for epoch in range(self.epochs):
            # Generate random minibatches
            indices = torch.randperm(len(observations), device=self.device)

            for start in range(0, len(observations), self.minibatch_size):
                end = start + self.minibatch_size
                mb_indices = indices[start:end]

                mb_observations = observations[mb_indices]
                mb_returns = returns[mb_indices]
                mb_old_values = old_values[mb_indices]

                # Zero gradients
                self.optimizer_critic.zero_grad()

                # Forward pass with mixed precision
                with autocast(enabled=self.use_mixed_precision):
                    loss, metrics = self.compute_critic_loss(
                        mb_observations, mb_returns, mb_old_values
                    )
                    loss = loss * self.value_loss_coeff

                # Backward pass
                if self.use_mixed_precision:
                    self.scaler_critic.scale(loss).backward()
                    # Gradient clipping
                    if self.max_grad_norm > 0:
                        self.scaler_critic.unscale_(self.optimizer_critic)
                        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                    self.scaler_critic.step(self.optimizer_critic)
                    self.scaler_critic.update()
                else:
                    loss.backward()
                    # Gradient clipping
                    if self.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                    self.optimizer_critic.step()

                # Store metrics
                for key, value in metrics.items():
                    all_metrics[key].append(value)

                self.critic_updates += 1

        # Average metrics
        avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}

        return avg_metrics

    def update(self, rollout_buffer) -> Dict[str, float]:
        """
        Perform one PPO update using the rollout buffer.

        Args:
            rollout_buffer: Buffer containing collected experience

        Returns:
            Dictionary of training metrics
        """
        # Ensure models are in training mode
        self.policy.train()
        self.critic.train()

        logger.debug(f"Starting PPO update {self.training_step}")

        # Update policy
        policy_metrics = self.update_policy(rollout_buffer)

        # Update critic
        critic_metrics = self.update_critic(rollout_buffer)

        # Combine metrics
        combined_metrics = {}
        combined_metrics.update(policy_metrics)
        combined_metrics.update(critic_metrics)

        self.training_step += 1

        logger.debug(f"PPO update {self.training_step} complete. "
                    f"Policy loss: {policy_metrics['policy_loss']:.4f}, "
                    f"Value loss: {critic_metrics['value_loss']:.4f}")

        return combined_metrics

    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        return {
            'training_step': self.training_step,
            'policy_updates': self.policy_updates,
            'critic_updates': self.critic_updates,
            'clip_ratio': self.clip_ratio,
            'entropy_coeff': self.entropy_coeff,
            'value_loss_coeff': self.value_loss_coeff
        }

    def set_learning_rates(self, policy_lr: float, critic_lr: float):
        """Update learning rates for both optimizers."""
        for param_group in self.optimizer_policy.param_groups:
            param_group['lr'] = policy_lr

        for param_group in self.optimizer_critic.param_groups:
            param_group['lr'] = critic_lr

        logger.info(f"Updated learning rates: policy={policy_lr:.6f}, critic={critic_lr:.6f}")

    def get_learning_rates(self) -> Tuple[float, float]:
        """Get current learning rates."""
        policy_lr = self.optimizer_policy.param_groups[0]['lr']
        critic_lr = self.optimizer_critic.param_groups[0]['lr']
        return policy_lr, critic_lr


class PPOLearningRateScheduler:
    """Learning rate scheduler for PPO training."""

    def __init__(self,
                 ppo_trainer: PPOTrainer,
                 initial_policy_lr: float,
                 initial_critic_lr: float,
                 schedule_type: str = "linear",
                 total_steps: int = 1000000,
                 min_lr_ratio: float = 0.1):
        """
        Initialize learning rate scheduler.

        Args:
            ppo_trainer: PPO trainer instance
            initial_policy_lr: Initial policy learning rate
            initial_critic_lr: Initial critic learning rate
            schedule_type: Type of schedule ("linear", "cosine", "constant")
            total_steps: Total training steps
            min_lr_ratio: Minimum learning rate as ratio of initial
        """
        self.ppo_trainer = ppo_trainer
        self.initial_policy_lr = initial_policy_lr
        self.initial_critic_lr = initial_critic_lr
        self.schedule_type = schedule_type
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio

    def step(self, current_step: int):
        """Update learning rates based on current step."""
        if self.schedule_type == "constant":
            return

        progress = min(current_step / self.total_steps, 1.0)

        if self.schedule_type == "linear":
            lr_mult = max(1.0 - progress, self.min_lr_ratio)
        elif self.schedule_type == "cosine":
            lr_mult = self.min_lr_ratio + (1.0 - self.min_lr_ratio) * 0.5 * (1.0 + np.cos(np.pi * progress))
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")

        policy_lr = self.initial_policy_lr * lr_mult
        critic_lr = self.initial_critic_lr * lr_mult

        self.ppo_trainer.set_learning_rates(policy_lr, critic_lr)