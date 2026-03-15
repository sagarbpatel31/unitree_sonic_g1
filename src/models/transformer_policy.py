"""
Transformer-based policy network for motion imitation and control.
Supports both behavior cloning and reinforcement learning training.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any
import numpy as np

from ..core.config import Config


class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional relative position encoding."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1,
                 relative_position: bool = False, max_relative_distance: int = 32):
        super().__init__()
        assert hidden_dim % num_heads == 0

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Linear projections
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.output = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)

        # Relative position encoding
        self.relative_position = relative_position
        if relative_position:
            self.max_relative_distance = max_relative_distance
            self.relative_position_embeddings = nn.Embedding(
                2 * max_relative_distance + 1, self.head_dim
            )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = x.shape

        # Linear projections
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # Add relative position encoding
        if self.relative_position:
            rel_pos_scores = self._compute_relative_position_scores(Q, seq_len)
            scores = scores + rel_pos_scores

        # Apply mask
        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)

        # Softmax and dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        output = torch.matmul(attention_weights, V)

        # Reshape and apply output projection
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)
        output = self.output(output)

        return output

    def _compute_relative_position_scores(self, Q: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Compute relative position attention scores."""
        batch_size, num_heads, _, head_dim = Q.shape

        # Create relative position indices
        positions = torch.arange(seq_len, device=Q.device)
        relative_positions = positions[:, None] - positions[None, :]
        relative_positions = torch.clamp(
            relative_positions, -self.max_relative_distance, self.max_relative_distance
        )
        relative_positions = relative_positions + self.max_relative_distance

        # Get relative position embeddings
        rel_embeddings = self.relative_position_embeddings(relative_positions)

        # Compute scores
        scores = torch.einsum('bhqd,qkd->bhqk', Q, rel_embeddings)

        return scores


class TransformerBlock(nn.Module):
    """Transformer block with attention and feed-forward layers."""

    def __init__(self, hidden_dim: int, num_heads: int, feedforward_dim: int,
                 dropout: float = 0.1, prenorm: bool = True, **attention_kwargs):
        super().__init__()

        self.prenorm = prenorm
        self.attention = MultiHeadAttention(hidden_dim, num_heads, dropout, **attention_kwargs)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_dim, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, hidden_dim),
            nn.Dropout(dropout)
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.prenorm:
            # Pre-LayerNorm
            x = x + self.attention(self.norm1(x), mask)
            x = x + self.feedforward(self.norm2(x))
        else:
            # Post-LayerNorm
            x = self.norm1(x + self.attention(x, mask))
            x = self.norm2(x + self.feedforward(x))

        return x


class PositionalEncoding(nn.Module):
    """Learnable or sinusoidal positional encoding."""

    def __init__(self, hidden_dim: int, max_sequence_length: int = 64, encoding_type: str = "learned"):
        super().__init__()

        self.encoding_type = encoding_type
        self.hidden_dim = hidden_dim

        if encoding_type == "learned":
            self.position_embeddings = nn.Embedding(max_sequence_length, hidden_dim)
        elif encoding_type == "sinusoidal":
            self.register_buffer("position_embeddings", self._create_sinusoidal_embeddings(
                max_sequence_length, hidden_dim
            ))
        else:
            raise ValueError(f"Unknown encoding type: {encoding_type}")

    def _create_sinusoidal_embeddings(self, max_len: int, hidden_dim: int) -> torch.Tensor:
        """Create sinusoidal position embeddings."""
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()

        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() *
                           -(math.log(10000.0) / hidden_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        if self.encoding_type == "learned":
            positions = torch.arange(seq_len, device=x.device)
            pos_embeddings = self.position_embeddings(positions)
        else:
            pos_embeddings = self.position_embeddings[:seq_len]

        return x + pos_embeddings.unsqueeze(0).expand(batch_size, -1, -1)


class TransformerPolicy(nn.Module):
    """
    Transformer-based policy network for humanoid control.

    This network processes sequences of observations and reference motions
    to produce control actions for motion imitation and robust control.
    """

    def __init__(self, config: Config, obs_dim: int, action_dim: int):
        super().__init__()

        self.config = config
        model_config = config.model
        transformer_config = config.get("models.transformer", {})

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = model_config.hidden_dim
        self.sequence_length = model_config.get("sequence_length", 32)

        # Input encoders
        input_config = transformer_config.get("input", {})
        obs_encoder_config = input_config.get("obs_encoder", {})
        ref_encoder_config = input_config.get("ref_encoder", {})

        # Observation encoder
        obs_encoder_layers = obs_encoder_config.get("layers", [512, 256])
        self.obs_encoder = self._build_mlp(
            obs_dim, obs_encoder_layers + [self.hidden_dim],
            obs_encoder_config
        )

        # Reference motion encoder (if using reference motions)
        ref_dim = obs_dim // 2  # Approximate reference motion dimension
        ref_encoder_layers = ref_encoder_config.get("layers", [256, 256])
        self.ref_encoder = self._build_mlp(
            ref_dim, ref_encoder_layers + [self.hidden_dim],
            ref_encoder_config
        )

        # Positional encoding
        pos_config = input_config.get("positional_encoding", {})
        self.positional_encoding = PositionalEncoding(
            self.hidden_dim,
            max_sequence_length=pos_config.get("max_sequence_length", 64),
            encoding_type=pos_config.get("type", "learned")
        )

        # Transformer layers
        tf_config = transformer_config.get("transformer", {})
        self.num_layers = model_config.num_layers
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                hidden_dim=self.hidden_dim,
                num_heads=model_config.num_heads,
                feedforward_dim=tf_config.get("feedforward_dim", 2048),
                dropout=model_config.dropout,
                prenorm=tf_config.get("layer", {}).get("prenorm", True),
                relative_position=tf_config.get("attention", {}).get("relative_position", True),
                max_relative_distance=tf_config.get("attention", {}).get("max_relative_distance", 32)
            )
            for _ in range(self.num_layers)
        ])

        # Output decoders
        output_config = transformer_config.get("output", {})
        action_decoder_config = output_config.get("action_decoder", {})

        # Action decoder
        action_decoder_layers = action_decoder_config.get("layers", [256, 128])
        self.action_decoder = self._build_mlp(
            self.hidden_dim, action_decoder_layers + [action_dim],
            action_decoder_config
        )

        # Value function head (for RL)
        if config.training.algorithm != "BC":
            value_decoder_config = output_config.get("value_decoder", {})
            value_decoder_layers = value_decoder_config.get("layers", [256, 128, 1])
            self.value_decoder = self._build_mlp(
                self.hidden_dim, value_decoder_layers,
                value_decoder_config
            )
        else:
            self.value_decoder = None

        # Initialize weights
        self._initialize_weights()

    def _build_mlp(self, input_dim: int, layer_dims: list, config: Dict[str, Any]) -> nn.Module:
        """Build MLP with specified configuration."""
        layers = []
        prev_dim = input_dim

        activation_name = config.get("activation", "relu")
        activation_fn = getattr(nn, activation_name.upper())() if hasattr(nn, activation_name.upper()) else nn.ReLU()

        dropout = config.get("dropout", 0.1)
        layer_norm = config.get("layer_norm", False)

        for i, dim in enumerate(layer_dims[:-1]):
            layers.append(nn.Linear(prev_dim, dim))

            if layer_norm:
                layers.append(nn.LayerNorm(dim))

            layers.append(activation_fn)

            if dropout > 0 and i < len(layer_dims) - 2:  # No dropout before last layer
                layers.append(nn.Dropout(dropout))

            prev_dim = dim

        # Output layer
        layers.append(nn.Linear(prev_dim, layer_dims[-1]))

        # Output activation
        output_activation = config.get("output_activation")
        if output_activation:
            if output_activation == "tanh":
                layers.append(nn.Tanh())
            elif output_activation == "sigmoid":
                layers.append(nn.Sigmoid())

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)

        # Special initialization for output layers
        if hasattr(self.action_decoder, "0"):
            nn.init.zeros_(self.action_decoder[-1].weight)
            nn.init.zeros_(self.action_decoder[-1].bias)

    def forward(self, observations: torch.Tensor,
                return_value: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the policy network.

        Args:
            observations: Observation sequence [batch, seq_len, obs_dim]
            return_value: Whether to return value estimate

        Returns:
            Dictionary containing actions and optionally values
        """
        batch_size, seq_len, obs_dim = observations.shape

        # Split observations into proprioceptive and reference
        # This is a simplified split - in practice would be more sophisticated
        prop_obs = observations[..., :obs_dim//2]
        ref_obs = observations[..., obs_dim//2:]

        # Encode observations
        prop_encoded = self.obs_encoder(prop_obs)  # [batch, seq_len, hidden_dim]
        ref_encoded = self.ref_encoder(ref_obs)    # [batch, seq_len, hidden_dim]

        # Combine encodings (simple addition - could be more sophisticated)
        hidden = prop_encoded + ref_encoded

        # Add positional encoding
        hidden = self.positional_encoding(hidden)

        # Apply transformer blocks
        for transformer_block in self.transformer_blocks:
            hidden = transformer_block(hidden)

        # Decode actions from last timestep
        last_hidden = hidden[:, -1]  # [batch, hidden_dim]
        actions = self.action_decoder(last_hidden)

        outputs = {"actions": actions}

        # Compute value if needed
        if return_value and self.value_decoder is not None:
            values = self.value_decoder(last_hidden)
            outputs["values"] = values.squeeze(-1)

        return outputs

    def get_action(self, observation: torch.Tensor,
                   deterministic: bool = True) -> torch.Tensor:
        """
        Get action for a single observation.

        Args:
            observation: Single observation [obs_dim] or [1, obs_dim]
            deterministic: Whether to use deterministic action

        Returns:
            Action tensor
        """
        if observation.dim() == 1:
            observation = observation.unsqueeze(0)

        if observation.dim() == 2:
            observation = observation.unsqueeze(1)  # Add sequence dimension

        with torch.no_grad():
            outputs = self.forward(observation)
            action = outputs["actions"]

        return action.squeeze(0) if action.shape[0] == 1 else action

    def compute_loss(self, observations: torch.Tensor,
                     actions: torch.Tensor,
                     values: Optional[torch.Tensor] = None,
                     **kwargs) -> Dict[str, torch.Tensor]:
        """
        Compute training loss.

        Args:
            observations: Observation sequences
            actions: Target actions
            values: Target values (for RL)

        Returns:
            Dictionary of losses
        """
        outputs = self.forward(observations, return_value=(values is not None))

        # Action loss (MSE for continuous control)
        action_loss = F.mse_loss(outputs["actions"], actions)

        losses = {"action_loss": action_loss}

        # Value loss (if training with RL)
        if values is not None and "values" in outputs:
            value_loss = F.mse_loss(outputs["values"], values)
            losses["value_loss"] = value_loss

        # Total loss
        total_loss = action_loss
        if "value_loss" in losses:
            vf_coef = self.config.training.get("vf_coef", 0.5)
            total_loss += vf_coef * losses["value_loss"]

        losses["total_loss"] = total_loss

        return losses