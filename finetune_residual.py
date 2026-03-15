#!/usr/bin/env python3
"""
Residual RL implementation for fine-tuning pretrained G1 controllers.

This module implements residual reinforcement learning where a learnable
residual policy is added to a frozen pretrained policy. This approach
preserves the motion prior while learning robustness adaptations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent
from typing import Dict, Tuple, Optional, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ResidualPolicy(nn.Module):
    """
    Residual policy that learns corrections to a base policy.

    The residual policy outputs action deltas that are added to the
    base policy's actions, allowing fine-grained adaptation while
    preserving the original motion patterns.
    """

    def __init__(self,
                 obs_dim: int,
                 action_dim: int,
                 config: Dict[str, Any]):
        """
        Initialize residual policy.

        Args:
            obs_dim: Observation dimensionality
            action_dim: Action dimensionality
            config: Residual policy configuration
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.config = config

        # Network architecture
        hidden_dims = config.get('hidden_dims', [256, 256])
        activation = config.get('activation', 'ReLU')

        # Build network layers
        layers = []
        prev_dim = obs_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                getattr(nn, activation)(),
                nn.LayerNorm(hidden_dim) if config.get('use_layer_norm', True) else nn.Identity(),
                nn.Dropout(config.get('dropout_rate', 0.0)) if config.get('dropout_rate', 0.0) > 0 else nn.Identity()
            ])
            prev_dim = hidden_dim

        # Output layer for residual actions
        layers.append(nn.Linear(prev_dim, action_dim))

        self.network = nn.Sequential(*layers)

        # Output scaling and clipping
        self.residual_scale = config.get('residual_scale', 0.1)
        self.max_residual = config.get('max_residual', 0.5)

        # Optional learned variance for stochastic residuals
        self.learn_std = config.get('learn_std', False)
        if self.learn_std:
            self.log_std = nn.Parameter(torch.zeros(action_dim))
            self.min_log_std = config.get('min_log_std', -5.0)
            self.max_log_std = config.get('max_log_std', 0.0)

        # Initialize weights
        self._initialize_weights()

        logger.info(f"Created ResidualPolicy: obs_dim={obs_dim}, action_dim={action_dim}, "
                   f"hidden_dims={hidden_dims}, residual_scale={self.residual_scale}")

    def _initialize_weights(self):
        """Initialize network weights for small initial residuals."""
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                # Small initial weights
                nn.init.normal_(layer.weight, mean=0.0, std=0.01)
                nn.init.zeros_(layer.bias)

        # Final layer gets even smaller initialization
        if isinstance(self.network[-1], nn.Linear):
            nn.init.normal_(self.network[-1].weight, mean=0.0, std=0.001)
            nn.init.zeros_(self.network[-1].bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute residual actions.

        Args:
            obs: Observations (batch_size, obs_dim)

        Returns:
            Residual actions (batch_size, action_dim)
        """
        raw_residual = self.network(obs)

        # Scale and clip residual
        residual = raw_residual * self.residual_scale
        residual = torch.clamp(residual, -self.max_residual, self.max_residual)

        return residual

    def get_distribution(self, obs: torch.Tensor) -> torch.distributions.Distribution:
        """
        Get residual action distribution.

        Args:
            obs: Observations (batch_size, obs_dim)

        Returns:
            Action distribution
        """
        mean_residual = self.forward(obs)

        if self.learn_std:
            # Learned standard deviation
            log_std = torch.clamp(self.log_std, self.min_log_std, self.max_log_std)
            std = torch.exp(log_std).expand_as(mean_residual)
        else:
            # Fixed small standard deviation
            std = torch.ones_like(mean_residual) * 0.01

        return Independent(Normal(mean_residual, std), 1)

    def get_stats(self) -> Dict[str, float]:
        """Get policy statistics for monitoring."""
        with torch.no_grad():
            # Compute statistics on recent forward passes
            stats = {
                'residual_scale': self.residual_scale,
                'max_residual': self.max_residual
            }

            if self.learn_std:
                stats['avg_log_std'] = self.log_std.mean().item()
                stats['std_range'] = (self.log_std.min().item(), self.log_std.max().item())

        return stats


class ResidualTrainer:
    """
    Trainer for residual reinforcement learning.

    Combines a frozen base policy with a learnable residual policy,
    training the residual to improve robustness while preserving
    the original motion patterns.
    """

    def __init__(self,
                 base_policy: nn.Module,
                 env: Any,
                 config: Dict[str, Any],
                 device: torch.device):
        """
        Initialize residual trainer.

        Args:
            base_policy: Frozen pretrained policy
            env: Training environment
            config: Training configuration
            device: PyTorch device
        """
        self.base_policy = base_policy
        self.env = env
        self.config = config
        self.device = device

        # Create residual policy
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        self.residual_policy = ResidualPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            config=config.residual_policy
        ).to(device)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.residual_policy.parameters(),
            lr=config.optimizer.lr,
            weight_decay=config.optimizer.get('weight_decay', 1e-4)
        )

        # PPO-specific parameters
        self.clip_ratio = config.get('clip_ratio', 0.2)
        self.value_clip = config.get('value_clip', True)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.value_coef = config.get('value_coef', 0.5)
        self.max_grad_norm = config.get('max_grad_norm', 1.0)

        # Value function (optional - can reuse base policy's value function)
        if config.get('learn_value_function', True):
            self.value_function = self._create_value_function(obs_dim)
            self.value_optimizer = torch.optim.Adam(
                self.value_function.parameters(),
                lr=config.optimizer.lr
            )
        else:
            self.value_function = None

        # Residual-specific losses
        self.residual_regularization = config.get('residual_regularization', 0.01)
        self.base_tracking_weight = config.get('base_tracking_weight', 0.1)

        # Experience buffer
        self.buffer_size = config.get('buffer_size', 2048)
        self.batch_size = config.get('batch_size', 64)
        self.ppo_epochs = config.get('ppo_epochs', 10)

        logger.info(f"Initialized ResidualTrainer with buffer_size={self.buffer_size}")

    def _create_value_function(self, obs_dim: int) -> nn.Module:
        """Create value function network."""
        hidden_dims = self.config.value_function.get('hidden_dims', [256, 256])

        layers = []
        prev_dim = obs_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))

        return nn.Sequential(*layers).to(self.device)

    def get_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Get combined action from base + residual policies.

        Args:
            obs: Observation
            deterministic: Whether to use deterministic action

        Returns:
            Combined action
        """
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Base policy action
            base_action_dist = self.base_policy.get_distribution(obs_tensor)
            if deterministic:
                base_action = base_action_dist.mean
            else:
                base_action = base_action_dist.sample()

            # Residual action
            residual_dist = self.residual_policy.get_distribution(obs_tensor)
            if deterministic:
                residual_action = residual_dist.mean
            else:
                residual_action = residual_dist.sample()

            # Combined action
            combined_action = base_action + residual_action

        return combined_action.cpu().numpy()[0]

    def collect_rollouts(self) -> Dict[str, torch.Tensor]:
        """
        Collect rollouts for training.

        Returns:
            Dictionary containing rollout data
        """
        observations = []
        actions = []
        base_actions = []
        residual_actions = []
        rewards = []
        values = []
        log_probs = []
        dones = []

        obs = self.env.reset()
        episode_reward = 0
        episode_length = 0

        for _ in range(self.buffer_size):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

            with torch.no_grad():
                # Base policy action
                base_action_dist = self.base_policy.get_distribution(obs_tensor)
                base_action = base_action_dist.sample()

                # Residual action
                residual_dist = self.residual_policy.get_distribution(obs_tensor)
                residual_action = residual_dist.sample()
                residual_log_prob = residual_dist.log_prob(residual_action)

                # Combined action
                combined_action = base_action + residual_action

                # Value estimate
                if self.value_function:
                    value = self.value_function(obs_tensor)
                else:
                    value = torch.zeros(1, 1, device=self.device)

            # Execute action
            action_np = combined_action.cpu().numpy()[0]
            obs_next, reward, done, info = self.env.step(action_np)

            # Store experience
            observations.append(obs)
            actions.append(combined_action.squeeze(0))
            base_actions.append(base_action.squeeze(0))
            residual_actions.append(residual_action.squeeze(0))
            rewards.append(reward)
            values.append(value.squeeze())
            log_probs.append(residual_log_prob.squeeze(0))
            dones.append(done)

            episode_reward += reward
            episode_length += 1

            if done:
                obs = self.env.reset()
                episode_reward = 0
                episode_length = 0
            else:
                obs = obs_next

        # Convert to tensors
        rollout_data = {
            'observations': torch.FloatTensor(np.array(observations)).to(self.device),
            'actions': torch.stack(actions),
            'base_actions': torch.stack(base_actions),
            'residual_actions': torch.stack(residual_actions),
            'rewards': torch.FloatTensor(rewards).to(self.device),
            'values': torch.stack(values),
            'log_probs': torch.stack(log_probs),
            'dones': torch.BoolTensor(dones).to(self.device)
        }

        return rollout_data

    def compute_advantages(self, rollout_data: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute advantages and returns using GAE.

        Args:
            rollout_data: Rollout data

        Returns:
            Tuple of (advantages, returns)
        """
        rewards = rollout_data['rewards']
        values = rollout_data['values']
        dones = rollout_data['dones']

        gamma = self.config.get('gamma', 0.99)
        lambda_gae = self.config.get('lambda_gae', 0.95)

        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)

        last_advantage = 0
        last_return = values[-1] if len(values) > 0 else 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
                next_non_terminal = 1
            else:
                next_value = values[t + 1]
                next_non_terminal = 1 - dones[t + 1].float()

            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            advantages[t] = last_advantage = delta + gamma * lambda_gae * next_non_terminal * last_advantage

            returns[t] = last_return = rewards[t] + gamma * next_non_terminal * last_return

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def train_step(self) -> Dict[str, float]:
        """
        Perform one training step.

        Returns:
            Dictionary of training metrics
        """
        # Collect rollouts
        rollout_data = self.collect_rollouts()

        # Compute advantages and returns
        advantages, returns = self.compute_advantages(rollout_data)

        # Training metrics
        metrics = {}

        # PPO updates
        for epoch in range(self.ppo_epochs):
            epoch_metrics = self._ppo_update(rollout_data, advantages, returns)

            for key, value in epoch_metrics.items():
                if key not in metrics:
                    metrics[key] = []
                metrics[key].append(value)

        # Average metrics across epochs
        for key, values in metrics.items():
            metrics[key] = np.mean(values)

        # Add residual-specific metrics
        residual_stats = self.residual_policy.get_stats()
        metrics.update({f"residual/{k}": v for k, v in residual_stats.items()})

        return metrics

    def _ppo_update(self, rollout_data: Dict[str, torch.Tensor],
                   advantages: torch.Tensor, returns: torch.Tensor) -> Dict[str, float]:
        """Perform PPO policy update."""
        observations = rollout_data['observations']
        actions = rollout_data['actions']
        base_actions = rollout_data['base_actions']
        residual_actions = rollout_data['residual_actions']
        old_log_probs = rollout_data['log_probs']
        values = rollout_data['values']

        # Shuffle data
        indices = torch.randperm(len(observations))
        obs_batch = observations[indices]
        actions_batch = actions[indices]
        base_actions_batch = base_actions[indices]
        residual_actions_batch = residual_actions[indices]
        old_log_probs_batch = old_log_probs[indices]
        advantages_batch = advantages[indices]
        returns_batch = returns[indices]
        values_batch = values[indices]

        # Mini-batch updates
        batch_size = min(self.batch_size, len(observations))
        total_loss = 0
        policy_loss = 0
        value_loss = 0
        entropy_loss = 0

        for start_idx in range(0, len(observations), batch_size):
            end_idx = start_idx + batch_size

            # Mini-batch data
            mb_obs = obs_batch[start_idx:end_idx]
            mb_actions = actions_batch[start_idx:end_idx]
            mb_base_actions = base_actions_batch[start_idx:end_idx]
            mb_residual_actions = residual_actions_batch[start_idx:end_idx]
            mb_old_log_probs = old_log_probs_batch[start_idx:end_idx]
            mb_advantages = advantages_batch[start_idx:end_idx]
            mb_returns = returns_batch[start_idx:end_idx]
            mb_values = values_batch[start_idx:end_idx]

            # Get current residual policy predictions
            residual_dist = self.residual_policy.get_distribution(mb_obs)
            new_log_probs = residual_dist.log_prob(mb_residual_actions)
            entropy = residual_dist.entropy().mean()

            # Compute ratio and clipped objective
            ratio = torch.exp(new_log_probs - mb_old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)

            policy_objective = torch.min(
                ratio * mb_advantages,
                clipped_ratio * mb_advantages
            ).mean()

            # Value function loss
            if self.value_function:
                new_values = self.value_function(mb_obs).squeeze()
                if self.value_clip:
                    clipped_values = mb_values + torch.clamp(
                        new_values - mb_values, -self.clip_ratio, self.clip_ratio
                    )
                    value_loss_1 = (new_values - mb_returns) ** 2
                    value_loss_2 = (clipped_values - mb_returns) ** 2
                    value_objective = -torch.max(value_loss_1, value_loss_2).mean()
                else:
                    value_objective = -F.mse_loss(new_values, mb_returns)
            else:
                value_objective = torch.tensor(0.0, device=self.device)

            # Residual regularization (encourage small residuals)
            residual_reg = -torch.mean(mb_residual_actions ** 2) * self.residual_regularization

            # Base tracking loss (encourage residuals that don't deviate too much)
            base_tracking_loss = -F.mse_loss(mb_actions, mb_base_actions) * self.base_tracking_weight

            # Total loss
            loss = -(policy_objective +
                    self.value_coef * value_objective +
                    self.entropy_coef * entropy +
                    residual_reg +
                    base_tracking_loss)

            # Backward pass
            self.optimizer.zero_grad()
            if self.value_function and self.value_optimizer:
                self.value_optimizer.zero_grad()

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.residual_policy.parameters(), self.max_grad_norm)
            if self.value_function:
                torch.nn.utils.clip_grad_norm_(self.value_function.parameters(), self.max_grad_norm)

            self.optimizer.step()
            if self.value_function and self.value_optimizer:
                self.value_optimizer.step()

            total_loss += loss.item()
            policy_loss += policy_objective.item()
            value_loss += value_objective.item() if isinstance(value_objective, torch.Tensor) else 0
            entropy_loss += entropy.item()

        num_batches = (len(observations) + batch_size - 1) // batch_size

        return {
            'total_loss': total_loss / num_batches,
            'policy_loss': policy_loss / num_batches,
            'value_loss': value_loss / num_batches,
            'entropy': entropy_loss / num_batches
        }


def create_residual_trainer(base_policy: nn.Module,
                          env: Any,
                          config: Dict[str, Any],
                          device: torch.device) -> ResidualTrainer:
    """
    Factory function to create a residual trainer.

    Args:
        base_policy: Pretrained base policy
        env: Training environment
        config: Configuration dictionary
        device: PyTorch device

    Returns:
        Configured ResidualTrainer
    """
    return ResidualTrainer(
        base_policy=base_policy,
        env=env,
        config=config,
        device=device
    )