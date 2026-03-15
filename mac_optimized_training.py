#!/usr/bin/env python3
"""
Mac-Optimized Training Script for Enhanced Transformer Policy
Optimized for macOS with CPU training and 16GB RAM.
"""

import os
import time
import json
import argparse
from pathlib import Path
from typing import Dict

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from final_enhanced_training import OptimizedTransformerPolicy, FinalCSVDataset


class MacOptimizedTrainer:
    """Mac-optimized trainer for CPU-only training with memory efficiency."""

    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                 config: Dict, save_dir: str):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Force CPU for Mac (more stable than MPS for training)
        self.device = torch.device("cpu")
        print(f"🖥️  Using device: {self.device}")

        # Enable CPU optimizations
        torch.set_num_threads(4)  # Optimize for Mac CPU cores

        self.model.to(self.device)

        # Optimizer optimized for CPU training
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config.get("weight_decay", 1e-4),
            eps=1e-7  # More stable for CPU
        )

        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config["num_epochs"],
            eta_min=config["learning_rate"] * 0.01
        )

        self.criterion = nn.MSELoss()
        self.best_val_loss = float('inf')
        self.epoch = 0

    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
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
        """Train for one epoch with Mac optimizations."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            observations = batch['observations'].to(self.device)
            actions = batch['actions'].to(self.device)

            # Clear gradients
            self.optimizer.zero_grad()

            # Forward pass
            predicted_actions = self.model(observations)
            loss = self.criterion(predicted_actions, actions)

            # Backward pass
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            # Optimizer step
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Progress reporting (every 5 batches to reduce overhead)
            if (batch_idx + 1) % 5 == 0:
                print(f"   Batch {batch_idx+1}/{len(self.train_loader)}, Loss: {loss.item():.6f}")

        return total_loss / num_batches

    def validate(self) -> float:
        """Validate model performance."""
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
        """Main training loop optimized for Mac."""
        print(f"🚀 Mac-optimized training for {self.config['num_epochs']} epochs...")
        model_info = self.model.get_model_info()
        print(f"Model: {model_info['total_parameters']:,} parameters ({model_info['parameter_size_mb']:.1f} MB)")
        print(f"Device: {self.device}")
        print(f"CPU threads: {torch.get_num_threads()}")

        start_time = time.time()

        for epoch in range(self.config['num_epochs']):
            self.epoch = epoch
            epoch_start = time.time()

            print(f"\nEpoch {epoch+1}/{self.config['num_epochs']}")

            # Train
            train_loss = self.train_epoch()

            # Validate
            val_loss = self.validate()

            # Update scheduler
            self.scheduler.step()

            # Calculate timing
            epoch_time = time.time() - epoch_start
            estimated_total = epoch_time * self.config['num_epochs']
            elapsed_total = time.time() - start_time
            remaining = estimated_total - elapsed_total

            # Log results
            lr = self.optimizer.param_groups[0]['lr']
            print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            print(f"LR: {lr:.8f}, Epoch time: {epoch_time:.1f}s, Est. remaining: {remaining/60:.1f}min")

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
            "config": self.config,
            "platform": "macOS_CPU"
        }

        with open(self.save_dir / "mac_training_stats.json", 'w') as f:
            json.dump(stats, f, indent=2)


def create_mac_optimized_model(obs_dim: int, action_dim: int, d_model: int = 128, num_layers: int = 3) -> OptimizedTransformerPolicy:
    """Create Mac-optimized model with configurable architecture for CPU training."""
    # Calculate number of attention heads based on d_model
    nhead = max(4, d_model // 32)  # Ensure divisibility and reasonable head count

    return OptimizedTransformerPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers
    )


def main():
    """Main Mac training script."""
    parser = argparse.ArgumentParser(description="Mac-Optimized Training")
    parser.add_argument("--data", type=str, default="data/lightwheel_bevorg_frames.csv")
    parser.add_argument("--save_dir", type=str, default="checkpoints/mac_training")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)  # Smaller for CPU
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d_model", type=int, default=128)  # Model dimension
    parser.add_argument("--num_layers", type=int, default=3)  # Number of transformer layers
    parser.add_argument("--sequence_length", type=int, default=16)  # Shorter sequences

    args = parser.parse_args()

    # Mac-optimized configuration
    config = {
        "num_epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": 1e-4,
        "sequence_length": args.sequence_length
    }

    print("=" * 60)
    print("🍎 Mac-Optimized Training for Enhanced Transformer Policy")
    print("=" * 60)
    print(f"Platform: macOS CPU")
    print(f"RAM optimization: Enabled")
    print(f"CPU threads: {torch.get_num_threads()}")
    print(f"Config: {json.dumps(config, indent=2)}")

    # Create datasets with Mac optimizations
    print(f"\n📊 Loading data from: {args.data}")

    train_dataset = FinalCSVDataset(args.data, config["sequence_length"], True)
    val_dataset = FinalCSVDataset(args.data, config["sequence_length"], False)

    # Smaller batch size and no multiprocessing for Mac stability
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0  # Important: No multiprocessing on Mac
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0
    )

    # Get dimensions
    sample = train_dataset[0]
    obs_dim = sample['observations'].shape[-1]
    action_dim = sample['actions'].shape[-1]

    print(f"Data: {len(train_dataset)} train, {len(val_dataset)} val sequences")
    print(f"Dimensions: {obs_dim} obs → {action_dim} actions")

    # Create Mac-optimized model
    model = create_mac_optimized_model(obs_dim, action_dim, args.d_model, args.num_layers)

    model_info = model.get_model_info()
    print(f"\n🧠 Model: {model_info['total_parameters']:,} parameters ({model_info['parameter_size_mb']:.1f} MB)")

    # Train with Mac optimizations
    trainer = MacOptimizedTrainer(model, train_loader, val_loader, config, args.save_dir)
    trainer.train()

    print(f"\n✅ Mac training complete! Check: {args.save_dir}")
    print(f"🎮 Test with: python3 mujoco_demo.py --checkpoint {args.save_dir}/best_checkpoint.pth")


if __name__ == "__main__":
    main()