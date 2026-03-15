"""
Critic (Value Function) network implementation for Unitree G1 motion imitation.

This module provides value function networks for estimating state values
in PPO-based reinforcement learning for humanoid robot control.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
from omegaconf import DictConfig
import numpy as np
import logging

logger = logging.getLogger(__name__)


class G1Critic(nn.Module):
    """
    Value function network for Unitree G1 robot motion imitation.

    Features:
    - Multi-layer perceptron with configurable architecture
    - Optional layer normalization and dropout
    - Configurable activation functions
    - Value normalization options
    - Ensemble support for improved stability
    """

    def __init__(self,
                 obs_dim: int,
                 config: DictConfig):
        """
        Initialize G1 critic network.

        Args:
            obs_dim: Observation space dimension
            config: Critic configuration
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.config = config

        # Network architecture parameters
        self.hidden_dims = config.hidden_dims
        self.activation = getattr(torch.nn, config.get('activation', 'ReLU'))
        self.use_layer_norm = config.get('use_layer_norm', False)
        self.dropout_rate = config.get('dropout_rate', 0.0)

        # Value normalization
        self.use_value_norm = config.get('use_value_norm', False)
        self.value_norm_eps = config.get('value_norm_eps', 1e-8)

        # Build network
        self._build_network()

        # Initialize weights
        self._init_weights()

        # Value normalization statistics
        if self.use_value_norm:
            self.value_normalizer = ValueNormalizer(eps=self.value_norm_eps)

        logger.info(f"Initialized G1Critic: obs_dim={obs_dim}, "
                   f"hidden_dims={self.hidden_dims}, "
                   f"use_value_norm={self.use_value_norm}")

    def _build_network(self):
        """Build the critic network architecture."""
        layers = []
        input_dim = self.obs_dim

        # Hidden layers
        for i, hidden_dim in enumerate(self.hidden_dims):
            # Linear layer
            layers.append(nn.Linear(input_dim, hidden_dim))

            # Layer normalization
            if self.use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))

            # Activation
            layers.append(self.activation())

            # Dropout
            if self.dropout_rate > 0:
                layers.append(nn.Dropout(self.dropout_rate))

            input_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)

        # Value head (single output)
        self.value_head = nn.Linear(input_dim, 1)

    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier uniform initialization for linear layers
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Initialize value head with smaller weights for stability
        nn.init.xavier_uniform_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to estimate state values.

        Args:
            observations: Observation tensor (batch_size, obs_dim)

        Returns:
            State values (batch_size, 1)
        """
        # Extract features
        features = self.backbone(observations)

        # Value prediction
        values = self.value_head(features)

        # Apply value normalization if enabled
        if self.use_value_norm and hasattr(self, 'value_normalizer'):
            values = self.value_normalizer.normalize(values)

        return values

    def update_value_norm(self, values: torch.Tensor):
        """Update value normalization statistics."""
        if self.use_value_norm and hasattr(self, 'value_normalizer'):
            self.value_normalizer.update(values)

    def get_value_statistics(self, observations: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get value prediction statistics.

        Args:
            observations: Observation tensor (batch_size, obs_dim)

        Returns:
            Dictionary of statistics
        """
        with torch.no_grad():
            values = self.forward(observations)

            stats = {
                'mean_value': values.mean(),
                'std_value': values.std(),
                'min_value': values.min(),
                'max_value': values.max(),
            }

            if self.use_value_norm and hasattr(self, 'value_normalizer'):
                norm_stats = self.value_normalizer.get_statistics()
                stats.update(norm_stats)

        return stats


class G1CriticEnsemble(nn.Module):
    """
    Ensemble of critic networks for improved value estimation stability.

    Uses multiple independent critic networks and combines their predictions
    to reduce overestimation bias and improve training stability.
    """

    def __init__(self,
                 obs_dim: int,
                 config: DictConfig,
                 num_critics: int = 2):
        """
        Initialize critic ensemble.

        Args:
            obs_dim: Observation space dimension
            config: Critic configuration
            num_critics: Number of critics in ensemble
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.config = config
        self.num_critics = num_critics

        # Create ensemble of critics
        self.critics = nn.ModuleList([
            G1Critic(obs_dim, config) for _ in range(num_critics)
        ])

        # Ensemble combination method
        self.combination_method = config.get('combination_method', 'mean')  # 'mean', 'min', 'median'

        logger.info(f"Initialized G1CriticEnsemble: num_critics={num_critics}, "
                   f"combination_method={self.combination_method}")

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ensemble.

        Args:
            observations: Observation tensor (batch_size, obs_dim)

        Returns:
            Combined value estimates (batch_size, 1)
        """
        # Get predictions from all critics
        values_list = [critic(observations) for critic in self.critics]
        values = torch.stack(values_list, dim=-1)  # (batch_size, 1, num_critics)

        # Combine predictions
        if self.combination_method == 'mean':
            combined_values = values.mean(dim=-1)
        elif self.combination_method == 'min':
            combined_values = values.min(dim=-1).values
        elif self.combination_method == 'median':
            combined_values = values.median(dim=-1).values
        else:
            raise ValueError(f"Unknown combination method: {self.combination_method}")

        return combined_values

    def forward_all(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning all critic predictions.

        Args:
            observations: Observation tensor (batch_size, obs_dim)

        Returns:
            All value estimates (batch_size, num_critics)
        """
        values_list = [critic(observations).squeeze(-1) for critic in self.critics]
        return torch.stack(values_list, dim=-1)

    def update_value_norm(self, values: torch.Tensor):
        """Update value normalization for all critics."""
        for critic in self.critics:
            critic.update_value_norm(values)


class G1CriticLSTM(nn.Module):
    """
    LSTM-based critic network for temporal value estimation.

    Useful for tasks requiring memory and temporal modeling of value functions.
    """

    def __init__(self,
                 obs_dim: int,
                 config: DictConfig):
        """
        Initialize LSTM-based critic.

        Args:
            obs_dim: Observation space dimension
            config: Critic configuration
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.config = config

        # Architecture parameters
        self.hidden_dim = config.get('hidden_dim', 256)
        self.num_layers = config.get('num_layers', 2)
        self.dropout_rate = config.get('dropout_rate', 0.0)

        # Value normalization
        self.use_value_norm = config.get('use_value_norm', False)
        self.value_norm_eps = config.get('value_norm_eps', 1e-8)

        # Build network
        self._build_lstm_network()
        self._init_weights()

        # Hidden state management
        self.hidden_state = None
        self.cell_state = None

        # Value normalization
        if self.use_value_norm:
            self.value_normalizer = ValueNormalizer(eps=self.value_norm_eps)

        logger.info(f"Initialized G1CriticLSTM: obs_dim={obs_dim}, "
                   f"hidden_dim={self.hidden_dim}, num_layers={self.num_layers}")

    def _build_lstm_network(self):
        """Build LSTM-based network architecture."""
        # Input processing
        self.input_layer = nn.Linear(self.obs_dim, self.hidden_dim)

        # LSTM core
        self.lstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout_rate if self.num_layers > 1 else 0,
            batch_first=True
        )

        # Value head
        self.value_head = nn.Linear(self.hidden_dim, 1)

    def _init_weights(self):
        """Initialize network weights."""
        # Initialize linear layers
        nn.init.xavier_uniform_(self.input_layer.weight)
        nn.init.zeros_(self.input_layer.bias)

        nn.init.xavier_uniform_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)

        # Initialize LSTM weights
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)

    def forward(self,
                observations: torch.Tensor,
                hidden_state: Optional[torch.Tensor] = None,
                cell_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with LSTM.

        Args:
            observations: Observation tensor (batch_size, seq_len, obs_dim) or (batch_size, obs_dim)
            hidden_state: Previous hidden state
            cell_state: Previous cell state

        Returns:
            Tuple of (values, (new_hidden_state, new_cell_state))
        """
        batch_size = observations.shape[0]

        # Handle single timestep input
        if len(observations.shape) == 2:
            observations = observations.unsqueeze(1)  # Add sequence dimension

        # Input processing
        features = F.relu(self.input_layer(observations))

        # LSTM forward pass
        if hidden_state is None or cell_state is None:
            lstm_out, (new_hidden, new_cell) = self.lstm(features)
        else:
            lstm_out, (new_hidden, new_cell) = self.lstm(features, (hidden_state, cell_state))

        # Take last timestep output
        last_output = lstm_out[:, -1, :]

        # Value prediction
        values = self.value_head(last_output)

        # Apply value normalization if enabled
        if self.use_value_norm and hasattr(self, 'value_normalizer'):
            values = self.value_normalizer.normalize(values)

        return values, (new_hidden, new_cell)

    def reset_hidden_state(self, batch_size: int = 1):
        """Reset hidden states for new episodes."""
        device = next(self.parameters()).device
        self.hidden_state = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        self.cell_state = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)


class ValueNormalizer(nn.Module):
    """
    Running normalization for value function outputs.

    Maintains running statistics of value predictions and normalizes
    them to have zero mean and unit variance for training stability.
    """

    def __init__(self, eps: float = 1e-8, momentum: float = 0.99):
        """
        Initialize value normalizer.

        Args:
            eps: Small constant for numerical stability
            momentum: Momentum for running statistics update
        """
        super().__init__()

        self.eps = eps
        self.momentum = momentum

        # Running statistics
        self.register_buffer('running_mean', torch.zeros(1))
        self.register_buffer('running_var', torch.ones(1))
        self.register_buffer('count', torch.zeros(1))

    def update(self, values: torch.Tensor):
        """
        Update running statistics with new values.

        Args:
            values: Value tensor to update statistics with
        """
        if self.training:
            with torch.no_grad():
                batch_mean = values.mean()
                batch_var = values.var()
                batch_count = values.numel()

                if self.count == 0:
                    # First update
                    self.running_mean.copy_(batch_mean)
                    self.running_var.copy_(batch_var)
                else:
                    # Update with momentum
                    self.running_mean.mul_(self.momentum).add_(batch_mean, alpha=1 - self.momentum)
                    self.running_var.mul_(self.momentum).add_(batch_var, alpha=1 - self.momentum)

                self.count.add_(batch_count)

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        """
        Normalize values using running statistics.

        Args:
            values: Values to normalize

        Returns:
            Normalized values
        """
        # Update statistics if in training mode
        if self.training:
            self.update(values)

        # Normalize
        normalized = (values - self.running_mean) / torch.sqrt(self.running_var + self.eps)

        return normalized

    def denormalize(self, normalized_values: torch.Tensor) -> torch.Tensor:
        """
        Denormalize values back to original scale.

        Args:
            normalized_values: Normalized values

        Returns:
            Denormalized values
        """
        return normalized_values * torch.sqrt(self.running_var + self.eps) + self.running_mean

    def get_statistics(self) -> Dict[str, torch.Tensor]:
        """Get current normalization statistics."""
        return {
            'value_norm_mean': self.running_mean,
            'value_norm_var': self.running_var,
            'value_norm_std': torch.sqrt(self.running_var + self.eps),
            'value_norm_count': self.count
        }