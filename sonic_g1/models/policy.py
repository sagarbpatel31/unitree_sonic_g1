"""
Policy network implementation for Unitree G1 motion imitation.

This module provides policy networks optimized for humanoid robot control,
with support for both continuous action spaces and deterministic/stochastic
action sampling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent
from typing import Tuple, Optional, Dict, Any
from omegaconf import DictConfig
import numpy as np
import logging

logger = logging.getLogger(__name__)


class G1Policy(nn.Module):
    """
    Policy network for Unitree G1 robot motion imitation.

    Features:
    - Multi-layer perceptron with configurable architecture
    - Gaussian action distribution for continuous control
    - Optional layer normalization and dropout
    - Configurable activation functions
    - Action scaling and clipping
    """

    def __init__(self,
                 obs_dim: int,
                 action_dim: int,
                 config: DictConfig):
        """
        Initialize G1 policy network.

        Args:
            obs_dim: Observation space dimension
            action_dim: Action space dimension
            config: Policy configuration
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.config = config

        # Network architecture parameters
        self.hidden_dims = config.hidden_dims
        self.activation = getattr(torch.nn, config.get('activation', 'ReLU'))
        self.use_layer_norm = config.get('use_layer_norm', False)
        self.dropout_rate = config.get('dropout_rate', 0.0)

        # Action distribution parameters
        self.log_std_type = config.get('log_std_type', 'learned')  # 'learned', 'fixed', 'state_dependent'
        self.initial_log_std = config.get('initial_log_std', 0.0)
        self.min_log_std = config.get('min_log_std', -10.0)
        self.max_log_std = config.get('max_log_std', 2.0)

        # Action scaling
        self.action_scale = config.get('action_scale', 1.0)
        self.action_clip = config.get('action_clip', None)

        # Build network
        self._build_network()

        # Initialize weights
        self._init_weights()

        logger.info(f"Initialized G1Policy: obs_dim={obs_dim}, action_dim={action_dim}, "
                   f"hidden_dims={self.hidden_dims}, log_std_type={self.log_std_type}")

    def _build_network(self):
        """Build the policy network architecture."""
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

        # Action mean head
        self.action_mean = nn.Linear(input_dim, self.action_dim)

        # Action log standard deviation
        if self.log_std_type == 'learned':
            # Single learnable parameter for all actions
            self.log_std = nn.Parameter(torch.ones(self.action_dim) * self.initial_log_std)
        elif self.log_std_type == 'state_dependent':
            # State-dependent log std
            self.log_std_net = nn.Linear(input_dim, self.action_dim)
        elif self.log_std_type == 'fixed':
            # Fixed log std (not learnable)
            self.register_buffer('log_std', torch.ones(self.action_dim) * self.initial_log_std)
        else:
            raise ValueError(f"Unknown log_std_type: {self.log_std_type}")

    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier uniform initialization for linear layers
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Initialize action mean head with smaller weights for stability
        nn.init.xavier_uniform_(self.action_mean.weight, gain=0.01)
        nn.init.zeros_(self.action_mean.bias)

        # Initialize log std network if state-dependent
        if self.log_std_type == 'state_dependent':
            nn.init.xavier_uniform_(self.log_std_net.weight, gain=0.01)
            nn.init.constant_(self.log_std_net.bias, self.initial_log_std)

    def forward(self, observations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to get action mean and log std.

        Args:
            observations: Observation tensor (batch_size, obs_dim)

        Returns:
            Tuple of (action_mean, log_std)
        """
        # Extract features
        features = self.backbone(observations)

        # Action mean
        action_mean = self.action_mean(features)

        # Action log standard deviation
        if self.log_std_type == 'learned' or self.log_std_type == 'fixed':
            # Broadcast log_std to match batch size
            log_std = self.log_std.expand_as(action_mean)
        elif self.log_std_type == 'state_dependent':
            log_std = self.log_std_net(features)
        else:
            raise ValueError(f"Unknown log_std_type: {self.log_std_type}")

        # Clamp log std to reasonable range
        log_std = torch.clamp(log_std, self.min_log_std, self.max_log_std)

        return action_mean, log_std

    def get_distribution(self, observations: torch.Tensor) -> Independent:
        """
        Get action distribution for given observations.

        Args:
            observations: Observation tensor (batch_size, obs_dim)

        Returns:
            Independent Gaussian distribution
        """
        action_mean, log_std = self.forward(observations)
        std = torch.exp(log_std)

        # Create diagonal Gaussian distribution
        distribution = Independent(Normal(action_mean, std), 1)

        return distribution

    def act(self,
            observations: torch.Tensor,
            deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample actions from the policy.

        Args:
            observations: Observation tensor (batch_size, obs_dim)
            deterministic: If True, return mean actions

        Returns:
            Tuple of (actions, log_probabilities)
        """
        with torch.no_grad():
            action_mean, log_std = self.forward(observations)

            if deterministic:
                actions = action_mean
                # For deterministic actions, log prob is not meaningful
                # Return zeros with same shape
                log_probs = torch.zeros(action_mean.shape[0], device=observations.device)
            else:
                std = torch.exp(log_std)
                distribution = Independent(Normal(action_mean, std), 1)
                actions = distribution.sample()
                log_probs = distribution.log_prob(actions)

            # Apply action scaling and clipping
            actions = self._process_actions(actions)

        return actions, log_probs

    def get_action_log_prob(self,
                           observations: torch.Tensor,
                           actions: torch.Tensor) -> torch.Tensor:
        """
        Get log probabilities for given observation-action pairs.

        Args:
            observations: Observation tensor (batch_size, obs_dim)
            actions: Action tensor (batch_size, action_dim)

        Returns:
            Log probabilities (batch_size,)
        """
        # Reverse action processing
        processed_actions = self._unprocess_actions(actions)

        # Get distribution
        distribution = self.get_distribution(observations)

        # Compute log probabilities
        log_probs = distribution.log_prob(processed_actions)

        return log_probs

    def _process_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Apply action scaling and clipping."""
        # Scale actions
        if self.action_scale != 1.0:
            actions = actions * self.action_scale

        # Clip actions if specified
        if self.action_clip is not None:
            actions = torch.clamp(actions, -self.action_clip, self.action_clip)

        return actions

    def _unprocess_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Reverse action processing for log prob computation."""
        # Unclip (no exact reverse, but clamp to original range)
        if self.action_clip is not None:
            actions = torch.clamp(actions, -self.action_clip, self.action_clip)

        # Unscale actions
        if self.action_scale != 1.0:
            actions = actions / self.action_scale

        return actions

    def entropy(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy of the action distribution.

        Args:
            observations: Observation tensor (batch_size, obs_dim)

        Returns:
            Entropy values (batch_size,)
        """
        distribution = self.get_distribution(observations)
        return distribution.entropy()

    def get_action_statistics(self, observations: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get action distribution statistics.

        Args:
            observations: Observation tensor (batch_size, obs_dim)

        Returns:
            Dictionary of statistics
        """
        action_mean, log_std = self.forward(observations)
        std = torch.exp(log_std)

        return {
            'action_mean': action_mean,
            'action_std': std,
            'action_log_std': log_std,
            'mean_action_mean': action_mean.mean(dim=0),
            'mean_action_std': std.mean(dim=0),
            'min_action_std': std.min(dim=0).values,
            'max_action_std': std.max(dim=0).values
        }


class G1PolicyLSTM(nn.Module):
    """
    LSTM-based policy network for G1 robot with temporal modeling.

    Useful for motion imitation tasks that require temporal coherence
    and memory of previous states.
    """

    def __init__(self,
                 obs_dim: int,
                 action_dim: int,
                 config: DictConfig):
        """
        Initialize LSTM-based G1 policy.

        Args:
            obs_dim: Observation space dimension
            action_dim: Action space dimension
            config: Policy configuration
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.config = config

        # Architecture parameters
        self.hidden_dim = config.get('hidden_dim', 256)
        self.num_layers = config.get('num_layers', 2)
        self.dropout_rate = config.get('dropout_rate', 0.0)

        # Action distribution parameters
        self.initial_log_std = config.get('initial_log_std', 0.0)
        self.min_log_std = config.get('min_log_std', -10.0)
        self.max_log_std = config.get('max_log_std', 2.0)

        # Action processing
        self.action_scale = config.get('action_scale', 1.0)
        self.action_clip = config.get('action_clip', None)

        # Build network
        self._build_lstm_network()
        self._init_weights()

        # Hidden state management
        self.hidden_state = None
        self.cell_state = None

        logger.info(f"Initialized G1PolicyLSTM: obs_dim={obs_dim}, action_dim={action_dim}, "
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

        # Output heads
        self.action_mean = nn.Linear(self.hidden_dim, self.action_dim)
        self.log_std = nn.Parameter(torch.ones(self.action_dim) * self.initial_log_std)

    def _init_weights(self):
        """Initialize network weights."""
        # Initialize linear layers
        nn.init.xavier_uniform_(self.input_layer.weight)
        nn.init.zeros_(self.input_layer.bias)

        nn.init.xavier_uniform_(self.action_mean.weight, gain=0.01)
        nn.init.zeros_(self.action_mean.bias)

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
                cell_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with LSTM.

        Args:
            observations: Observation tensor (batch_size, seq_len, obs_dim) or (batch_size, obs_dim)
            hidden_state: Previous hidden state
            cell_state: Previous cell state

        Returns:
            Tuple of (action_mean, log_std, (new_hidden_state, new_cell_state))
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

        # Action outputs
        action_mean = self.action_mean(last_output)
        log_std = self.log_std.expand_as(action_mean)
        log_std = torch.clamp(log_std, self.min_log_std, self.max_log_std)

        return action_mean, log_std, (new_hidden, new_cell)

    def act(self,
            observations: torch.Tensor,
            deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample actions using LSTM policy.

        Args:
            observations: Observation tensor
            deterministic: If True, return mean actions

        Returns:
            Tuple of (actions, log_probabilities)
        """
        with torch.no_grad():
            action_mean, log_std, (new_hidden, new_cell) = self.forward(
                observations, self.hidden_state, self.cell_state
            )

            # Update internal state
            self.hidden_state = new_hidden
            self.cell_state = new_cell

            if deterministic:
                actions = action_mean
                log_probs = torch.zeros(action_mean.shape[0], device=observations.device)
            else:
                std = torch.exp(log_std)
                distribution = Independent(Normal(action_mean, std), 1)
                actions = distribution.sample()
                log_probs = distribution.log_prob(actions)

            # Process actions
            actions = self._process_actions(actions)

        return actions, log_probs

    def reset_hidden_state(self, batch_size: int = 1):
        """Reset hidden states for new episodes."""
        device = next(self.parameters()).device
        self.hidden_state = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        self.cell_state = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)

    def _process_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Apply action scaling and clipping."""
        if self.action_scale != 1.0:
            actions = actions * self.action_scale

        if self.action_clip is not None:
            actions = torch.clamp(actions, -self.action_clip, self.action_clip)

        return actions