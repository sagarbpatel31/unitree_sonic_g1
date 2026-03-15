"""
Rollout buffer for storing and managing experience data for PPO training.

This module provides efficient storage and retrieval of experience tuples,
GAE computation, and data batching for PPO updates.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class RolloutBuffer:
    """
    Buffer for storing rollout data for PPO training.

    Stores observations, actions, rewards, values, log probabilities, and dones.
    Supports GAE (Generalized Advantage Estimation) computation and efficient
    data retrieval for policy updates.
    """

    def __init__(self,
                 buffer_size: int,
                 obs_dim: int,
                 action_dim: int,
                 num_envs: int,
                 device: torch.device,
                 dtype: torch.dtype = torch.float32):
        """
        Initialize rollout buffer.

        Args:
            buffer_size: Number of steps to store per environment
            obs_dim: Observation space dimension
            action_dim: Action space dimension
            num_envs: Number of parallel environments
            device: Torch device for tensors
            dtype: Data type for tensors
        """
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_envs = num_envs
        self.device = device
        self.dtype = dtype

        # Current buffer position
        self.pos = 0
        self.full = False

        # Initialize storage tensors
        self._init_storage()

        logger.info(f"Initialized rollout buffer: size={buffer_size}, "
                   f"envs={num_envs}, obs_dim={obs_dim}, action_dim={action_dim}")

    def _init_storage(self):
        """Initialize storage tensors."""
        # Core experience data
        self.observations = torch.zeros(
            (self.buffer_size, self.num_envs, self.obs_dim),
            dtype=self.dtype, device=self.device
        )
        self.actions = torch.zeros(
            (self.buffer_size, self.num_envs, self.action_dim),
            dtype=self.dtype, device=self.device
        )
        self.rewards = torch.zeros(
            (self.buffer_size, self.num_envs),
            dtype=self.dtype, device=self.device
        )
        self.dones = torch.zeros(
            (self.buffer_size, self.num_envs),
            dtype=torch.bool, device=self.device
        )

        # Policy-related data
        self.values = torch.zeros(
            (self.buffer_size, self.num_envs),
            dtype=self.dtype, device=self.device
        )
        self.log_probs = torch.zeros(
            (self.buffer_size, self.num_envs),
            dtype=self.dtype, device=self.device
        )

        # GAE computation results
        self.advantages = torch.zeros(
            (self.buffer_size, self.num_envs),
            dtype=self.dtype, device=self.device
        )
        self.returns = torch.zeros(
            (self.buffer_size, self.num_envs),
            dtype=self.dtype, device=self.device
        )

    def add(self,
            observations: np.ndarray,
            actions: np.ndarray,
            rewards: np.ndarray,
            dones: np.ndarray,
            values: np.ndarray,
            log_probs: np.ndarray):
        """
        Add a step of experience to the buffer.

        Args:
            observations: Observations (num_envs, obs_dim)
            actions: Actions (num_envs, action_dim)
            rewards: Rewards (num_envs,)
            dones: Done flags (num_envs,)
            values: Value predictions (num_envs,)
            log_probs: Log probabilities (num_envs,)
        """
        if self.pos >= self.buffer_size:
            raise RuntimeError(f"Buffer overflow! Position {self.pos} >= size {self.buffer_size}")

        # Convert numpy arrays to tensors if needed
        if isinstance(observations, np.ndarray):
            observations = torch.from_numpy(observations).to(dtype=self.dtype, device=self.device)
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).to(dtype=self.dtype, device=self.device)
        if isinstance(rewards, np.ndarray):
            rewards = torch.from_numpy(rewards).to(dtype=self.dtype, device=self.device)
        if isinstance(dones, np.ndarray):
            dones = torch.from_numpy(dones).to(dtype=torch.bool, device=self.device)
        if isinstance(values, np.ndarray):
            values = torch.from_numpy(values).to(dtype=self.dtype, device=self.device)
        if isinstance(log_probs, np.ndarray):
            log_probs = torch.from_numpy(log_probs).to(dtype=self.dtype, device=self.device)

        # Store data
        self.observations[self.pos] = observations
        self.actions[self.pos] = actions
        self.rewards[self.pos] = rewards
        self.dones[self.pos] = dones
        self.values[self.pos] = values
        self.log_probs[self.pos] = log_probs

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def compute_returns(self,
                       next_values: np.ndarray,
                       gamma: float = 0.99,
                       gae_lambda: float = 0.95):
        """
        Compute returns and advantages using GAE.

        Args:
            next_values: Value predictions for next states (num_envs,)
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
        """
        if not self.full and self.pos == 0:
            raise RuntimeError("Cannot compute returns on empty buffer")

        # Convert next_values to tensor
        if isinstance(next_values, np.ndarray):
            next_values = torch.from_numpy(next_values).to(dtype=self.dtype, device=self.device)

        # Number of steps to process
        num_steps = self.buffer_size if self.full else self.pos

        # Initialize last values and advantages
        last_values = next_values
        last_gae_lam = 0

        # Compute advantages backwards
        for step in reversed(range(num_steps)):
            if step == num_steps - 1:
                next_non_terminal = ~self.dones[step]
                next_values_step = last_values
            else:
                next_non_terminal = ~self.dones[step]
                next_values_step = self.values[step + 1]

            # TD error
            delta = (self.rewards[step] +
                    gamma * next_values_step * next_non_terminal -
                    self.values[step])

            # GAE computation
            self.advantages[step] = last_gae_lam = (
                delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            )

        # Compute returns
        self.returns[:num_steps] = self.advantages[:num_steps] + self.values[:num_steps]

        logger.debug(f"Computed returns and advantages for {num_steps} steps")

    def get_policy_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get flattened data for policy updates.

        Returns:
            Tuple of (observations, actions, log_probs, advantages, returns)
        """
        num_steps = self.buffer_size if self.full else self.pos

        # Flatten data (steps * envs, ...)
        observations = self.observations[:num_steps].view(-1, self.obs_dim)
        actions = self.actions[:num_steps].view(-1, self.action_dim)
        log_probs = self.log_probs[:num_steps].view(-1)
        advantages = self.advantages[:num_steps].view(-1)
        returns = self.returns[:num_steps].view(-1)

        return observations, actions, log_probs, advantages, returns

    def get_critic_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get flattened data for critic updates.

        Returns:
            Tuple of (observations, returns, old_values)
        """
        num_steps = self.buffer_size if self.full else self.pos

        # Flatten data (steps * envs, ...)
        observations = self.observations[:num_steps].view(-1, self.obs_dim)
        returns = self.returns[:num_steps].view(-1)
        old_values = self.values[:num_steps].view(-1)

        return observations, returns, old_values

    def get_batch_data(self) -> Dict[str, torch.Tensor]:
        """
        Get all data as flattened tensors in a dictionary.

        Returns:
            Dictionary containing all flattened buffer data
        """
        num_steps = self.buffer_size if self.full else self.pos

        return {
            'observations': self.observations[:num_steps].view(-1, self.obs_dim),
            'actions': self.actions[:num_steps].view(-1, self.action_dim),
            'rewards': self.rewards[:num_steps].view(-1),
            'dones': self.dones[:num_steps].view(-1),
            'values': self.values[:num_steps].view(-1),
            'log_probs': self.log_probs[:num_steps].view(-1),
            'advantages': self.advantages[:num_steps].view(-1),
            'returns': self.returns[:num_steps].view(-1)
        }

    def clear(self):
        """Clear the buffer and reset position."""
        self.pos = 0
        self.full = False

        # Zero out tensors for clean state
        self.observations.zero_()
        self.actions.zero_()
        self.rewards.zero_()
        self.dones.zero_()
        self.values.zero_()
        self.log_probs.zero_()
        self.advantages.zero_()
        self.returns.zero_()

    def size(self) -> int:
        """Return the current number of stored steps."""
        return self.buffer_size if self.full else self.pos

    def is_full(self) -> bool:
        """Check if buffer is full."""
        return self.full

    def get_statistics(self) -> Dict[str, float]:
        """Get buffer statistics for debugging and monitoring."""
        if self.size() == 0:
            return {}

        num_steps = self.buffer_size if self.full else self.pos

        with torch.no_grad():
            stats = {
                'buffer_size': self.size(),
                'mean_reward': self.rewards[:num_steps].mean().item(),
                'std_reward': self.rewards[:num_steps].std().item(),
                'mean_value': self.values[:num_steps].mean().item(),
                'std_value': self.values[:num_steps].std().item(),
                'mean_advantage': self.advantages[:num_steps].mean().item(),
                'std_advantage': self.advantages[:num_steps].std().item(),
                'mean_return': self.returns[:num_steps].mean().item(),
                'std_return': self.returns[:num_steps].std().item(),
                'done_rate': self.dones[:num_steps].float().mean().item()
            }

        return stats


class VectorizedRolloutBuffer:
    """
    Alternative implementation optimized for vectorized environments.

    This version assumes all environments step synchronously and stores
    data in a more vectorized format for improved performance.
    """

    def __init__(self,
                 buffer_size: int,
                 obs_shape: Tuple[int, ...],
                 action_shape: Tuple[int, ...],
                 num_envs: int,
                 device: torch.device,
                 dtype: torch.dtype = torch.float32):
        """
        Initialize vectorized rollout buffer.

        Args:
            buffer_size: Number of steps to store
            obs_shape: Shape of observation space (without batch dimension)
            action_shape: Shape of action space (without batch dimension)
            num_envs: Number of parallel environments
            device: Torch device for tensors
            dtype: Data type for tensors
        """
        self.buffer_size = buffer_size
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.num_envs = num_envs
        self.device = device
        self.dtype = dtype

        # Total storage size
        self.total_size = buffer_size * num_envs

        # Current position
        self.pos = 0

        # Initialize storage
        self._init_vectorized_storage()

        logger.info(f"Initialized vectorized rollout buffer: "
                   f"total_size={self.total_size}, obs_shape={obs_shape}, action_shape={action_shape}")

    def _init_vectorized_storage(self):
        """Initialize vectorized storage tensors."""
        # Flatten all data across time and environment dimensions
        self.observations = torch.zeros(
            (self.total_size, *self.obs_shape),
            dtype=self.dtype, device=self.device
        )
        self.actions = torch.zeros(
            (self.total_size, *self.action_shape),
            dtype=self.dtype, device=self.device
        )
        self.rewards = torch.zeros(
            self.total_size, dtype=self.dtype, device=self.device
        )
        self.dones = torch.zeros(
            self.total_size, dtype=torch.bool, device=self.device
        )
        self.values = torch.zeros(
            self.total_size, dtype=self.dtype, device=self.device
        )
        self.log_probs = torch.zeros(
            self.total_size, dtype=self.dtype, device=self.device
        )
        self.advantages = torch.zeros(
            self.total_size, dtype=self.dtype, device=self.device
        )
        self.returns = torch.zeros(
            self.total_size, dtype=self.dtype, device=self.device
        )

    def add_batch(self,
                  observations: torch.Tensor,
                  actions: torch.Tensor,
                  rewards: torch.Tensor,
                  dones: torch.Tensor,
                  values: torch.Tensor,
                  log_probs: torch.Tensor):
        """
        Add a batch of experiences from all environments.

        Args:
            observations: Batch observations (num_envs, *obs_shape)
            actions: Batch actions (num_envs, *action_shape)
            rewards: Batch rewards (num_envs,)
            dones: Batch done flags (num_envs,)
            values: Batch values (num_envs,)
            log_probs: Batch log probabilities (num_envs,)
        """
        if self.pos + self.num_envs > self.total_size:
            raise RuntimeError("Buffer overflow!")

        end_pos = self.pos + self.num_envs

        self.observations[self.pos:end_pos] = observations
        self.actions[self.pos:end_pos] = actions
        self.rewards[self.pos:end_pos] = rewards
        self.dones[self.pos:end_pos] = dones
        self.values[self.pos:end_pos] = values
        self.log_probs[self.pos:end_pos] = log_probs

        self.pos = end_pos

    def compute_returns_vectorized(self,
                                  next_values: torch.Tensor,
                                  gamma: float = 0.99,
                                  gae_lambda: float = 0.95):
        """
        Compute returns using vectorized GAE.

        Args:
            next_values: Next values for all environments (num_envs,)
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
        """
        if self.pos == 0:
            raise RuntimeError("Cannot compute returns on empty buffer")

        # Reshape data back to (steps, envs) format
        steps = self.pos // self.num_envs
        rewards = self.rewards[:self.pos].view(steps, self.num_envs)
        values = self.values[:self.pos].view(steps, self.num_envs)
        dones = self.dones[:self.pos].view(steps, self.num_envs)

        # Initialize storage for vectorized computation
        advantages = torch.zeros_like(rewards)
        last_gae_lam = torch.zeros(self.num_envs, device=self.device, dtype=self.dtype)

        # Vectorized GAE computation
        for step in reversed(range(steps)):
            if step == steps - 1:
                next_non_terminal = ~dones[step]
                next_values_step = next_values
            else:
                next_non_terminal = ~dones[step]
                next_values_step = values[step + 1]

            delta = rewards[step] + gamma * next_values_step * next_non_terminal - values[step]
            advantages[step] = last_gae_lam = (
                delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            )

        # Store results back in flattened format
        self.advantages[:self.pos] = advantages.view(-1)
        returns = advantages + values
        self.returns[:self.pos] = returns.view(-1)

    def get_training_data(self) -> Dict[str, torch.Tensor]:
        """Get all data for training."""
        return {
            'observations': self.observations[:self.pos],
            'actions': self.actions[:self.pos],
            'rewards': self.rewards[:self.pos],
            'dones': self.dones[:self.pos],
            'values': self.values[:self.pos],
            'log_probs': self.log_probs[:self.pos],
            'advantages': self.advantages[:self.pos],
            'returns': self.returns[:self.pos]
        }

    def clear(self):
        """Clear the buffer."""
        self.pos = 0

    def size(self) -> int:
        """Return current buffer size."""
        return self.pos