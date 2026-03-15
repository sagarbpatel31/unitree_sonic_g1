#!/usr/bin/env python3
"""
Final Enhanced Training Script - Fixed and optimized for production use.
"""

import os
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR


class FinalCSVDataset(Dataset):
    """Optimized CSV dataset for production training."""

    def __init__(self, csv_path: str, sequence_length: int = 32, train_split: bool = True):
        self.sequence_length = sequence_length

        # Load and process data
        df = pd.read_csv(csv_path)
        split_idx = int(0.8 * len(df))

        if train_split:
            self.df = df.iloc[:split_idx]
        else:
            self.df = df.iloc[split_idx:]

        # Extract actions (columns starting with 'act_')
        action_cols = [col for col in self.df.columns if col.startswith('act_')]
        self.actions = self.df[action_cols].values.astype(np.float32)
        self.observations = self.actions.copy()  # Use actions as observations

        # Normalize data
        self.action_mean = np.mean(self.actions, axis=0)
        self.action_std = np.std(self.actions, axis=0) + 1e-8
        self.actions = (self.actions - self.action_mean) / self.action_std
        self.observations = self.actions.copy()

        # Generate windows
        step_size = max(1, sequence_length // 2)
        self.windows = []
        for start in range(0, len(self.df) - sequence_length + 1, step_size):
            self.windows.append((start, start + sequence_length))

        print(f"Dataset: {len(self.df)} rows, {len(self.windows)} windows, {self.actions.shape[1]} action dims")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        start, end = self.windows[idx]
        return {
            "observations": torch.FloatTensor(self.observations[start:end]),
            "actions": torch.FloatTensor(self.actions[start:end])
        }


class OptimizedTransformerPolicy(nn.Module):
    """Optimized transformer policy with fixed positional encoding."""

    def __init__(self, obs_dim: int, action_dim: int, d_model: int = 256, nhead: int = 8, num_layers: int = 4):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(obs_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, action_dim),
            nn.Tanh()  # Bound outputs for MuJoCo compatibility
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, obs_dim]
        x = self.input_proj(x)  # [batch, seq_len, d_model]
        x = self.transformer(x)  # [batch, seq_len, d_model]
        return self.output_proj(x)  # [batch, seq_len, action_dim]

    def get_action(self, observation: torch.Tensor) -> torch.Tensor:
        """Get single action for MuJoCo simulation."""
        if observation.dim() == 1:
            observation = observation.unsqueeze(0).unsqueeze(0)
        elif observation.dim() == 2:
            observation = observation.unsqueeze(1)

        with torch.no_grad():
            actions = self.forward(observation)
            return actions.squeeze()

    def get_model_info(self) -> Dict:
        total_params = sum(p.numel() for p in self.parameters())
        return {
            "model_name": "OptimizedTransformerPolicy",
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "d_model": self.d_model,
            "total_parameters": total_params,
            "parameter_size_mb": total_params * 4 / (1024 * 1024)
        }


class ProductionTrainer:
    """Production-ready trainer with essential features."""

    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                 config: Dict, save_dir: str):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config.get("weight_decay", 1e-4)
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config["num_epochs"],
            eta_min=config["learning_rate"] * 0.1
        )

        self.criterion = nn.MSELoss()
        self.best_val_loss = float('inf')
        self.epoch = 0

    def save_checkpoint(self, is_best: bool = False):
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'model_info': self.model.get_model_info()
        }

        torch.save(checkpoint, self.save_dir / "latest_checkpoint.pth")
        if is_best:
            torch.save(checkpoint, self.save_dir / "best_checkpoint.pth")
            print(f"🏆 New best model! Val loss: {self.best_val_loss:.6f}")

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            observations = batch['observations'].to(self.device)
            actions = batch['actions'].to(self.device)

            self.optimizer.zero_grad()
            predicted_actions = self.model(observations)
            loss = self.criterion(predicted_actions, actions)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if (batch_idx + 1) % 10 == 0:
                print(f"   Batch {batch_idx+1}/{len(self.train_loader)}, Loss: {loss.item():.6f}")

        return total_loss / num_batches

    def validate(self) -> float:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                observations = batch['observations'].to(self.device)
                actions = batch['actions'].to(self.device)

                predicted_actions = self.model(observations)
                loss = self.criterion(predicted_actions, actions)

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches if num_batches > 0 else float('inf')

    def train(self):
        print(f"🚀 Training for {self.config['num_epochs']} epochs...")
        model_info = self.model.get_model_info()
        print(f"Model: {model_info['total_parameters']:,} parameters ({model_info['parameter_size_mb']:.1f} MB)")

        start_time = time.time()

        for epoch in range(self.config['num_epochs']):
            self.epoch = epoch
            print(f"\nEpoch {epoch+1}/{self.config['num_epochs']}")

            # Train
            train_loss = self.train_epoch()

            # Validate
            val_loss = self.validate()

            # Update scheduler
            self.scheduler.step()

            # Log
            lr = self.optimizer.param_groups[0]['lr']
            print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {lr:.8f}")

            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            self.save_checkpoint(is_best)

        training_time = (time.time() - start_time) / 3600
        print(f"\n🎉 Training completed in {training_time:.1f} hours!")
        print(f"Best validation loss: {self.best_val_loss:.6f}")

        # Save final statistics
        stats = {
            "best_val_loss": self.best_val_loss,
            "epochs": self.epoch + 1,
            "training_time_hours": training_time,
            "model_info": model_info,
            "config": self.config
        }

        with open(self.save_dir / "training_stats.json", 'w') as f:
            json.dump(stats, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/lightwheel_bevorg_frames.csv")
    parser.add_argument("--save_dir", type=str, default="checkpoints/final_model")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--layers", type=int, default=6)

    args = parser.parse_args()

    config = {
        "num_epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": 1e-4,
        "sequence_length": 32
    }

    print("=" * 60)
    print("🚀 Final Enhanced Training")
    print("=" * 60)
    print(f"Config: {json.dumps(config, indent=2)}")

    # Create datasets
    train_dataset = FinalCSVDataset(args.data, config["sequence_length"], True)
    val_dataset = FinalCSVDataset(args.data, config["sequence_length"], False)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    # Get dimensions
    sample = train_dataset[0]
    obs_dim = sample['observations'].shape[-1]
    action_dim = sample['actions'].shape[-1]

    print(f"Data: {len(train_dataset)} train, {len(val_dataset)} val sequences")
    print(f"Dimensions: {obs_dim} obs → {action_dim} actions")

    # Create model
    model = OptimizedTransformerPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        d_model=args.d_model,
        nhead=8,
        num_layers=args.layers
    )

    # Train
    trainer = ProductionTrainer(model, train_loader, val_loader, config, args.save_dir)
    trainer.train()

    print(f"✅ Training complete! Check: {args.save_dir}")


if __name__ == "__main__":
    main()