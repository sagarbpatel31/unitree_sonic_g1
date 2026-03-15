#!/usr/bin/env python3
"""
Enhanced Transformer Policy for Sonic G1 Motion Imitation.
Larger, more sophisticated architecture for better performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer input."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        x = x + self.pe[:x.size(0)].transpose(0, 1)
        return self.dropout(x)


class MultiHeadSelfAttention(nn.Module):
    """Enhanced multi-head self attention with residual connections."""

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        assert d_model % nhead == 0

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.size()

        # Multi-head attention
        Q = self.query(x).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )

        # Output projection and residual connection
        output = self.out_proj(attn_output)
        return self.layer_norm(x + self.dropout(output))


class FeedForwardNetwork(nn.Module):
    """Enhanced feedforward network with GELU activation."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.gelu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return self.layer_norm(residual + x)


class TransformerBlock(nn.Module):
    """Enhanced transformer block with modern techniques."""

    def __init__(self, d_model: int, nhead: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadSelfAttention(d_model, nhead, dropout)
        self.feedforward = FeedForwardNetwork(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attention(x)
        x = self.feedforward(x)
        return x


class EnhancedTransformerPolicy(nn.Module):
    """
    Enhanced Transformer Policy for Motion Imitation.

    Features:
    - Larger architecture (512 hidden, 8 layers)
    - Positional encoding
    - Modern attention mechanisms
    - GELU activation
    - Layer normalization
    - Dropout for regularization
    - MuJoCo-compatible output format
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 256
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.d_model = d_model

        # Input projection with layer norm
        self.input_proj = nn.Sequential(
            nn.Linear(obs_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_seq_len)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, nhead, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Output projection with multiple layers
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, action_dim),
            nn.Tanh()  # Bound actions to [-1, 1] for MuJoCo
        )

        # Action scaling parameters (learnable)
        self.action_scale = nn.Parameter(torch.ones(action_dim))
        self.action_bias = nn.Parameter(torch.zeros(action_dim))

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the transformer.

        Args:
            observations: [batch_size, seq_len, obs_dim]

        Returns:
            actions: [batch_size, seq_len, action_dim] bounded to [-1, 1]
        """
        batch_size, seq_len, _ = observations.shape

        # Input projection
        x = self.input_proj(observations)

        # Add positional encoding
        x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)

        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Output projection
        actions = self.output_proj(x)

        # Apply learnable scaling and bias
        actions = actions * self.action_scale + self.action_bias

        return actions

    def get_action(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Get single action for MuJoCo simulation.

        Args:
            observation: [obs_dim] single observation

        Returns:
            action: [action_dim] single action
        """
        if observation.dim() == 1:
            observation = observation.unsqueeze(0).unsqueeze(0)  # [1, 1, obs_dim]
        elif observation.dim() == 2:
            observation = observation.unsqueeze(1)  # [batch, 1, obs_dim]

        with torch.no_grad():
            actions = self.forward(observation)
            return actions.squeeze()  # Remove batch/sequence dimensions

    def get_model_info(self) -> dict:
        """Get model information for logging."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "model_name": "EnhancedTransformerPolicy",
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "d_model": self.d_model,
            "num_layers": len(self.transformer_blocks),
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "parameter_size_mb": total_params * 4 / (1024 * 1024)  # Assuming float32
        }