#!/usr/bin/env python3
"""
Advanced training script for Enhanced Transformer Policy.
Features: Large model, proper checkpointing, advanced training techniques, MuJoCo compatibility.
"""

import os
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from enhanced_transformer_policy import EnhancedTransformerPolicy


class AdvancedCSVMotionDataset(Dataset):
    """Advanced CSV dataset with data augmentation and normalization."""

    def __init__(
        self,
        csv_path: str,
        sequence_length: int = 64,  # Increased for better temporal modeling
        train_split: bool = True,
        split_ratio: float = 0.8,
        normalize: bool = True,
        augment: bool = True,
        overlap: float = 0.5
    ):
        self.sequence_length = sequence_length
        self.train_split = train_split
        self.normalize = normalize
        self.augment = augment

        print(f"Loading CSV from: {csv_path}")
        self.df = pd.read_csv(csv_path)
        print(f"Raw CSV shape: {self.df.shape}")

        # Split data
        split_idx = int(split_ratio * len(self.df))
        if train_split:
            self.df = self.df.iloc[:split_idx]
            print(f"Training split: {len(self.df)} rows")
        else:
            self.df = self.df.iloc[split_idx:]
            print(f"Validation split: {len(self.df)} rows")

        # Extract and normalize features
        self._prepare_data()

        # Generate sequence windows
        self._generate_windows(overlap)

    def _prepare_data(self):
        """Extract and normalize observations and actions."""
        # Extract action and observation columns
        action_cols = [col for col in self.df.columns if col.startswith('act_')]
        obs_cols = [col for col in self.df.columns
                   if col not in action_cols and col != '_id' and
                   self.df[col].dtype in ['float64', 'int64', 'float32', 'int32']]

        print(f"Action columns: {len(action_cols)}")
        print(f"Observation columns: {len(obs_cols)}")

        # Convert to numpy
        self.raw_actions = self.df[action_cols].values.astype(np.float32)
        self.raw_observations = (self.df[obs_cols].values.astype(np.float32)
                                if obs_cols else self.raw_actions.copy())

        print(f"Raw actions shape: {self.raw_actions.shape}")
        print(f"Raw observations shape: {self.raw_observations.shape}")

        # Normalize data if requested
        if self.normalize:
            self.action_mean = np.mean(self.raw_actions, axis=0)
            self.action_std = np.std(self.raw_actions, axis=0) + 1e-8

            self.obs_mean = np.mean(self.raw_observations, axis=0)
            self.obs_std = np.std(self.raw_observations, axis=0) + 1e-8

            self.actions = (self.raw_actions - self.action_mean) / self.action_std
            self.observations = (self.raw_observations - self.obs_mean) / self.obs_std

            print("Applied normalization:")
            print(f"  Action range: [{self.actions.min():.3f}, {self.actions.max():.3f}]")
            print(f"  Observation range: [{self.observations.min():.3f}, {self.observations.max():.3f}]")
        else:
            self.actions = self.raw_actions
            self.observations = self.raw_observations

    def _generate_windows(self, overlap: float):
        """Generate sliding windows for sequence training."""
        step_size = max(1, int(self.sequence_length * (1 - overlap)))
        self.windows = []

        for start_idx in range(0, len(self.df) - self.sequence_length + 1, step_size):
            end_idx = start_idx + self.sequence_length
            self.windows.append((start_idx, end_idx))

        print(f"Generated {len(self.windows)} sequence windows")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        start_idx, end_idx = self.windows[idx]

        obs_seq = self.observations[start_idx:end_idx].copy()
        action_seq = self.actions[start_idx:end_idx].copy()

        # Data augmentation for training
        if self.augment and self.train_split:
            obs_seq, action_seq = self._augment_sequence(obs_seq, action_seq)

        return {
            "observations": torch.FloatTensor(obs_seq),
            "actions": torch.FloatTensor(action_seq)
        }

    def _augment_sequence(self, obs: np.ndarray, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply data augmentation techniques."""
        # Add small random noise (10% probability)
        if np.random.random() < 0.1:
            noise_scale = 0.01
            obs += np.random.normal(0, noise_scale, obs.shape).astype(np.float32)
            actions += np.random.normal(0, noise_scale * 0.5, actions.shape).astype(np.float32)

        # Time scaling (5% probability)
        if np.random.random() < 0.05:
            # Simple temporal subsampling (every other frame)
            if len(obs) > self.sequence_length // 2:
                indices = np.linspace(0, len(obs) - 1, self.sequence_length, dtype=int)
                obs = obs[indices]
                actions = actions[indices]

        return obs, actions

    def get_normalization_stats(self) -> Dict:
        """Get normalization statistics for model inference."""
        if self.normalize:
            return {
                "action_mean": self.action_mean,
                "action_std": self.action_std,
                "obs_mean": self.obs_mean,
                "obs_std": self.obs_std
            }
        return {}


class AdvancedTrainer:
    """Advanced trainer with checkpointing, logging, and model management."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        save_dir: str = "checkpoints"
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)

        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"Using device: {self.device}")

        # Optimizer with advanced settings
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config.get("weight_decay", 1e-4),
            betas=(0.9, 0.999),
            eps=1e-8
        )

        # Learning rate scheduler
        if config.get("scheduler") == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=config["num_epochs"],
                eta_min=config["learning_rate"] * 0.01
            )
        else:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10,
                verbose=True
            )

        # Loss function
        self.criterion = nn.MSELoss()

        # Logging
        self.writer = SummaryWriter(log_dir=str(self.save_dir / "logs"))

        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []

    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config,
            'model_info': self.model.get_model_info()
        }

        # Save latest checkpoint
        checkpoint_path = self.save_dir / "latest_checkpoint.pth"
        torch.save(checkpoint, checkpoint_path)

        # Save best checkpoint
        if is_best:
            best_path = self.save_dir / "best_checkpoint.pth"
            torch.save(checkpoint, best_path)
            print(f"🏆 New best model saved! Validation loss: {self.best_val_loss:.6f}")

        # Save periodic checkpoints
        if self.epoch % 20 == 0:
            epoch_path = self.save_dir / f"checkpoint_epoch_{self.epoch}.pth"
            torch.save(checkpoint, epoch_path)

    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """Load model checkpoint."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            self.epoch = checkpoint['epoch']
            self.best_val_loss = checkpoint['best_val_loss']
            self.train_losses = checkpoint.get('train_losses', [])
            self.val_losses = checkpoint.get('val_losses', [])

            print(f"✅ Checkpoint loaded from epoch {self.epoch}")
            return True
        except Exception as e:
            print(f"❌ Failed to load checkpoint: {e}")
            return False

    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            observations = batch['observations'].to(self.device)
            actions = batch['actions'].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            predicted_actions = self.model(observations)

            # Compute loss
            loss = self.criterion(predicted_actions, actions)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Log progress
            if (batch_idx + 1) % 100 == 0:
                print(f"   Batch {batch_idx+1}/{len(self.train_loader)}, Loss: {loss.item():.6f}")

        avg_loss = total_loss / num_batches
        return avg_loss

    def validate(self) -> float:
        """Validate the model."""
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

        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        return avg_loss

    def train(self):
        """Main training loop."""
        print(f"🚀 Starting training for {self.config['num_epochs']} epochs...")

        model_info = self.model.get_model_info()
        print(f"Model: {model_info['model_name']}")
        print(f"Parameters: {model_info['total_parameters']:,} ({model_info['parameter_size_mb']:.1f} MB)")

        start_time = time.time()

        for epoch in range(self.epoch, self.config['num_epochs']):
            self.epoch = epoch

            print(f"\nEpoch {epoch+1}/{self.config['num_epochs']}")
            print("-" * 50)

            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)

            # Validate
            val_loss = self.validate()
            self.val_losses.append(val_loss)

            # Update learning rate
            if isinstance(self.scheduler, CosineAnnealingLR):
                self.scheduler.step()
            else:
                self.scheduler.step(val_loss)

            # Log metrics
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Train Loss: {train_loss:.6f}")
            print(f"Val Loss: {val_loss:.6f}")
            print(f"Learning Rate: {current_lr:.8f}")

            # TensorBoard logging
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Validation', val_loss, epoch)
            self.writer.add_scalar('LearningRate', current_lr, epoch)

            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss

            self.save_checkpoint(is_best)

            # Early stopping check (optional)
            if len(self.val_losses) > 20:
                recent_losses = self.val_losses[-10:]
                if all(loss > self.best_val_loss * 1.1 for loss in recent_losses):
                    print("Early stopping triggered - validation loss not improving")
                    break

        total_time = time.time() - start_time
        print(f"\n🎉 Training completed in {total_time/3600:.1f} hours!")
        print(f"Best validation loss: {self.best_val_loss:.6f}")

        # Save final stats
        stats = {
            "best_val_loss": self.best_val_loss,
            "total_epochs": self.epoch + 1,
            "training_time_hours": total_time / 3600,
            "model_info": model_info,
            "config": self.config
        }

        with open(self.save_dir / "training_stats.json", 'w') as f:
            json.dump(stats, f, indent=2)

        self.writer.close()


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Advanced Transformer Policy Training")
    parser.add_argument("--data", type=str, default="data/lightwheel_bevorg_frames.csv",
                       help="Path to CSV data file")
    parser.add_argument("--save_dir", type=str, default="checkpoints/enhanced_model",
                       help="Directory to save checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--epochs", type=int, default=200,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")

    args = parser.parse_args()

    # Training configuration
    config = {
        "num_epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": 1e-4,
        "scheduler": "cosine",
        "sequence_length": 64,
        "normalize_data": True,
        "augment_data": True
    }

    print("=" * 60)
    print("🚀 Enhanced Transformer Policy Training")
    print("=" * 60)
    print(f"Configuration: {json.dumps(config, indent=2)}")

    # Create datasets
    print("\n1. Loading datasets...")
    train_dataset = AdvancedCSVMotionDataset(
        csv_path=args.data,
        sequence_length=config["sequence_length"],
        train_split=True,
        normalize=config["normalize_data"],
        augment=config["augment_data"]
    )

    val_dataset = AdvancedCSVMotionDataset(
        csv_path=args.data,
        sequence_length=config["sequence_length"],
        train_split=False,
        normalize=config["normalize_data"],
        augment=False
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")

    # Get dimensions from data
    sample = train_dataset[0]
    obs_dim = sample['observations'].shape[-1]
    action_dim = sample['actions'].shape[-1]

    print(f"\n2. Model configuration:")
    print(f"Observation dim: {obs_dim}")
    print(f"Action dim: {action_dim}")

    # Create enhanced model
    model = EnhancedTransformerPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        d_model=512,           # Large model
        nhead=8,
        num_layers=8,          # Deep network
        d_ff=2048,
        dropout=0.1
    )

    # Create trainer
    trainer = AdvancedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        save_dir=args.save_dir
    )

    # Resume from checkpoint if specified
    if args.resume and Path(args.resume).exists():
        trainer.load_checkpoint(args.resume)

    # Save normalization stats for inference
    norm_stats = train_dataset.get_normalization_stats()
    if norm_stats:
        with open(Path(args.save_dir) / "normalization_stats.json", 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_stats = {}
            for key, value in norm_stats.items():
                json_stats[key] = value.tolist()
            json.dump(json_stats, f, indent=2)

    # Start training
    trainer.train()

    print(f"\n✅ Training completed! Check results in: {args.save_dir}")


if __name__ == "__main__":
    main()