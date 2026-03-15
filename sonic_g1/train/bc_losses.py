"""
Loss functions for behavior cloning training.

This module provides various loss functions for supervised learning
of robot policies from demonstration data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent
from typing import Dict, Optional, Any
from omegaconf import DictConfig
import numpy as np
import logging

logger = logging.getLogger(__name__)


class BCLossCollection:
    """
    Collection of loss functions for behavior cloning.

    Combines multiple loss components with configurable weights for
    comprehensive policy learning.
    """

    def __init__(self, config: DictConfig, device: torch.device):
        """
        Initialize BC loss collection.

        Args:
            config: Loss configuration
            device: Torch device
        """
        self.config = config
        self.device = device

        # Loss weights
        self.mse_weight = config.get('mse_weight', 1.0)
        self.mae_weight = config.get('mae_weight', 0.0)
        self.regularization_weight = config.get('regularization_weight', 0.0)
        self.velocity_consistency_weight = config.get('velocity_consistency_weight', 0.0)
        self.action_smoothness_weight = config.get('action_smoothness_weight', 0.0)
        self.entropy_penalty_weight = config.get('entropy_penalty_weight', 0.0)

        # Loss function configurations
        self.mse_reduction = config.get('mse_reduction', 'mean')
        self.mae_reduction = config.get('mae_reduction', 'mean')
        self.use_huber_loss = config.get('use_huber_loss', False)
        self.huber_delta = config.get('huber_delta', 1.0)

        # Regularization types
        self.regularization_type = config.get('regularization_type', 'l2')  # 'l1', 'l2', 'elastic'
        self.l1_ratio = config.get('l1_ratio', 0.5)  # For elastic net

        # Action-specific configurations
        self.per_joint_weights = config.get('per_joint_weights', None)
        self.critical_joint_multiplier = config.get('critical_joint_multiplier', 1.0)

        # Advanced loss features
        self.adaptive_weights = config.get('adaptive_weights', False)
        self.focal_loss_gamma = config.get('focal_loss_gamma', 0.0)

        logger.info(f"Initialized BCLossCollection with weights: "
                   f"MSE={self.mse_weight}, MAE={self.mae_weight}, "
                   f"Reg={self.regularization_weight}")

    def compute_losses(self,
                      predicted_actions: torch.Tensor,
                      target_actions: torch.Tensor,
                      action_distribution: Optional[Any] = None,
                      states: Optional[torch.Tensor] = None,
                      state_derivatives: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute all configured loss components.

        Args:
            predicted_actions: Predicted actions (batch_size, action_dim)
            target_actions: Target actions (batch_size, action_dim)
            action_distribution: Optional action distribution from policy
            states: Optional state inputs
            state_derivatives: Optional state derivatives for velocity consistency

        Returns:
            Dictionary of loss components and total loss
        """
        losses = {}

        # Primary action prediction losses
        if self.mse_weight > 0:
            mse_loss = self._compute_mse_loss(predicted_actions, target_actions)
            losses['mse_loss'] = mse_loss * self.mse_weight

        if self.mae_weight > 0:
            mae_loss = self._compute_mae_loss(predicted_actions, target_actions)
            losses['mae_loss'] = mae_loss * self.mae_weight

        # Regularization losses
        if self.regularization_weight > 0 and action_distribution is not None:
            reg_loss = self._compute_regularization_loss(action_distribution)
            losses['regularization_loss'] = reg_loss * self.regularization_weight

        # Velocity consistency loss
        if self.velocity_consistency_weight > 0 and state_derivatives is not None:
            vel_loss = self._compute_velocity_consistency_loss(
                predicted_actions, target_actions, state_derivatives
            )
            losses['velocity_consistency_loss'] = vel_loss * self.velocity_consistency_weight

        # Action smoothness loss
        if self.action_smoothness_weight > 0:
            smoothness_loss = self._compute_action_smoothness_loss(predicted_actions)
            losses['action_smoothness_loss'] = smoothness_loss * self.action_smoothness_weight

        # Entropy penalty (for stochastic policies)
        if self.entropy_penalty_weight > 0 and action_distribution is not None:
            entropy_loss = self._compute_entropy_penalty(action_distribution)
            losses['entropy_penalty'] = entropy_loss * self.entropy_penalty_weight

        # Compute total loss
        total_loss = sum(losses.values())
        losses['total_loss'] = total_loss

        return losses

    def _compute_mse_loss(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute mean squared error loss."""
        if self.use_huber_loss:
            # Huber loss (smooth L1 loss)
            loss = F.smooth_l1_loss(predicted, target, reduction='none', beta=self.huber_delta)
        else:
            # Standard MSE
            loss = F.mse_loss(predicted, target, reduction='none')

        # Apply per-joint weights if specified
        if self.per_joint_weights is not None:
            weights = torch.tensor(self.per_joint_weights, device=predicted.device)
            loss = loss * weights.unsqueeze(0)

        # Apply focal loss weighting if enabled
        if self.focal_loss_gamma > 0:
            focal_weights = (1 + loss.detach()) ** self.focal_loss_gamma
            loss = loss * focal_weights

        # Reduction
        if self.mse_reduction == 'mean':
            return loss.mean()
        elif self.mse_reduction == 'sum':
            return loss.sum()
        else:
            return loss

    def _compute_mae_loss(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute mean absolute error loss."""
        loss = F.l1_loss(predicted, target, reduction='none')

        # Apply per-joint weights if specified
        if self.per_joint_weights is not None:
            weights = torch.tensor(self.per_joint_weights, device=predicted.device)
            loss = loss * weights.unsqueeze(0)

        # Reduction
        if self.mae_reduction == 'mean':
            return loss.mean()
        elif self.mae_reduction == 'sum':
            return loss.sum()
        else:
            return loss

    def _compute_regularization_loss(self, action_distribution) -> torch.Tensor:
        """Compute regularization loss on policy parameters."""
        if hasattr(action_distribution, 'scale'):
            # For Gaussian distributions, regularize the standard deviation
            if self.regularization_type == 'l2':
                # Encourage moderate exploration
                target_std = torch.ones_like(action_distribution.scale) * 0.1
                reg_loss = F.mse_loss(action_distribution.scale, target_std)
            elif self.regularization_type == 'l1':
                # Sparsity on exploration
                reg_loss = torch.mean(torch.abs(action_distribution.scale))
            else:
                reg_loss = torch.tensor(0.0, device=self.device)
        else:
            reg_loss = torch.tensor(0.0, device=self.device)

        return reg_loss

    def _compute_velocity_consistency_loss(self,
                                         predicted_actions: torch.Tensor,
                                         target_actions: torch.Tensor,
                                         state_derivatives: torch.Tensor) -> torch.Tensor:
        """
        Compute loss for consistency between actions and state changes.

        This encourages the policy to predict actions that are consistent
        with the observed state transitions.
        """
        # Compute implied velocities from actions
        # This assumes actions represent joint position targets
        action_velocities = predicted_actions - state_derivatives  # Simplified

        # Compare with target action velocities
        target_velocities = target_actions - state_derivatives

        # MSE loss on velocity consistency
        velocity_loss = F.mse_loss(action_velocities, target_velocities)

        return velocity_loss

    def _compute_action_smoothness_loss(self, predicted_actions: torch.Tensor) -> torch.Tensor:
        """
        Compute loss to encourage smooth action sequences.

        This is useful when training on sequence data.
        """
        if len(predicted_actions.shape) < 3:
            # No sequence dimension, cannot compute smoothness
            return torch.tensor(0.0, device=self.device)

        # Compute second derivatives (acceleration) and penalize large values
        if predicted_actions.shape[1] > 2:  # Need at least 3 timesteps
            # First derivative (velocity)
            first_diff = torch.diff(predicted_actions, dim=1)

            # Second derivative (acceleration)
            second_diff = torch.diff(first_diff, dim=1)

            # L2 penalty on acceleration
            smoothness_loss = torch.mean(second_diff ** 2)
        else:
            # Just penalize first derivatives
            first_diff = torch.diff(predicted_actions, dim=1)
            smoothness_loss = torch.mean(first_diff ** 2)

        return smoothness_loss

    def _compute_entropy_penalty(self, action_distribution) -> torch.Tensor:
        """Compute entropy penalty for stochastic policies."""
        if hasattr(action_distribution, 'entropy'):
            # Penalize high entropy (encourage deterministic behavior)
            entropy = action_distribution.entropy()
            entropy_penalty = torch.mean(entropy)
        else:
            entropy_penalty = torch.tensor(0.0, device=self.device)

        return entropy_penalty

    def update_adaptive_weights(self, losses: Dict[str, torch.Tensor], step: int):
        """Update loss weights adaptively based on training progress."""
        if not self.adaptive_weights:
            return

        # Simple adaptive weighting: reduce regularization over time
        decay_factor = max(0.1, 1.0 - step / 100000)  # Decay over 100k steps

        self.regularization_weight = self.config.get('regularization_weight', 0.0) * decay_factor
        self.entropy_penalty_weight = self.config.get('entropy_penalty_weight', 0.0) * decay_factor

        logger.debug(f"Updated adaptive weights at step {step}: "
                    f"reg={self.regularization_weight:.4f}, "
                    f"entropy={self.entropy_penalty_weight:.4f}")


class MultiTaskBCLoss:
    """
    Multi-task behavior cloning loss for learning multiple objectives.

    Useful when training on diverse motion types (walking, dancing, etc.)
    with different importance weights.
    """

    def __init__(self, config: DictConfig, device: torch.device):
        """
        Initialize multi-task BC loss.

        Args:
            config: Multi-task loss configuration
            device: Torch device
        """
        self.config = config
        self.device = device

        # Task-specific loss collections
        self.task_losses = {}
        for task_name, task_config in config.tasks.items():
            self.task_losses[task_name] = BCLossCollection(task_config, device)

        # Task balancing
        self.task_weights = config.get('task_weights', {})
        self.use_uncertainty_weighting = config.get('use_uncertainty_weighting', False)

        # Learnable task weights (uncertainty weighting)
        if self.use_uncertainty_weighting:
            num_tasks = len(self.task_losses)
            self.log_task_vars = nn.Parameter(torch.zeros(num_tasks, device=device))

    def compute_losses(self,
                      predicted_actions: torch.Tensor,
                      target_actions: torch.Tensor,
                      task_labels: torch.Tensor,
                      **kwargs) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task losses.

        Args:
            predicted_actions: Predicted actions
            target_actions: Target actions
            task_labels: Task labels for each sample
            **kwargs: Additional arguments for loss computation

        Returns:
            Dictionary of task-specific and total losses
        """
        task_losses = {}
        total_loss = torch.tensor(0.0, device=self.device)

        for task_idx, (task_name, task_loss_fn) in enumerate(self.task_losses.items()):
            # Get samples for this task
            task_mask = task_labels == task_idx
            if not task_mask.any():
                continue

            # Compute task-specific loss
            task_pred = predicted_actions[task_mask]
            task_target = target_actions[task_mask]

            # Filter kwargs for this task
            task_kwargs = {}
            for key, value in kwargs.items():
                if value is not None:
                    task_kwargs[key] = value[task_mask]

            task_loss_dict = task_loss_fn.compute_losses(
                task_pred, task_target, **task_kwargs
            )

            # Store task losses
            for loss_name, loss_value in task_loss_dict.items():
                task_losses[f"{task_name}_{loss_name}"] = loss_value

            # Add to total loss with weighting
            if self.use_uncertainty_weighting:
                # Uncertainty weighting (Kendall et al.)
                precision = torch.exp(-self.log_task_vars[task_idx])
                weighted_loss = precision * task_loss_dict['total_loss'] + self.log_task_vars[task_idx]
            else:
                # Fixed weighting
                task_weight = self.task_weights.get(task_name, 1.0)
                weighted_loss = task_weight * task_loss_dict['total_loss']

            total_loss = total_loss + weighted_loss

        task_losses['total_loss'] = total_loss

        return task_losses


class DistributionMatchingLoss:
    """
    Loss for matching action distributions rather than just point estimates.

    Useful for learning stochastic policies that capture the variability
    in expert demonstrations.
    """

    def __init__(self, config: DictConfig, device: torch.device):
        """
        Initialize distribution matching loss.

        Args:
            config: Distribution loss configuration
            device: Torch device
        """
        self.config = config
        self.device = device

        self.kl_weight = config.get('kl_weight', 1.0)
        self.wasserstein_weight = config.get('wasserstein_weight', 0.0)
        self.mmd_weight = config.get('mmd_weight', 0.0)

    def compute_loss(self,
                    predicted_dist,
                    target_actions: torch.Tensor) -> torch.Tensor:
        """
        Compute distribution matching loss.

        Args:
            predicted_dist: Predicted action distribution
            target_actions: Target action samples

        Returns:
            Distribution matching loss
        """
        loss = torch.tensor(0.0, device=self.device)

        # KL divergence (if target distribution is available)
        if self.kl_weight > 0:
            # Approximate target distribution as Gaussian
            target_mean = torch.mean(target_actions, dim=0, keepdim=True)
            target_std = torch.std(target_actions, dim=0, keepdim=True) + 1e-6

            target_dist = Independent(Normal(target_mean, target_std), 1)

            # KL divergence from predicted to target
            kl_loss = torch.distributions.kl_divergence(predicted_dist, target_dist).mean()
            loss = loss + self.kl_weight * kl_loss

        # Maximum Mean Discrepancy (MMD)
        if self.mmd_weight > 0:
            mmd_loss = self._compute_mmd_loss(predicted_dist, target_actions)
            loss = loss + self.mmd_weight * mmd_loss

        return loss

    def _compute_mmd_loss(self, predicted_dist, target_actions: torch.Tensor) -> torch.Tensor:
        """Compute Maximum Mean Discrepancy loss."""
        # Sample from predicted distribution
        predicted_samples = predicted_dist.sample((target_actions.shape[0],))

        # Simple RBF kernel MMD
        sigma = 1.0

        def rbf_kernel(x, y):
            return torch.exp(-torch.norm(x - y, dim=-1) ** 2 / (2 * sigma ** 2))

        # MMD computation (simplified)
        xx = rbf_kernel(predicted_samples.unsqueeze(1), predicted_samples.unsqueeze(0))
        yy = rbf_kernel(target_actions.unsqueeze(1), target_actions.unsqueeze(0))
        xy = rbf_kernel(predicted_samples.unsqueeze(1), target_actions.unsqueeze(0))

        mmd = xx.mean() + yy.mean() - 2 * xy.mean()

        return mmd