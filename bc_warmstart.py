#!/usr/bin/env python3
"""
Behavior Cloning Warm-start Training for Unitree G1 Motion Imitation.

This script implements supervised pretraining of the policy network using
expert demonstrations (retargeted human motion data) before PPO fine-tuning.
"""

import os
import sys
import argparse
import logging
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import wandb
from omegaconf import OmegaConf, DictConfig
from tensorboardX import SummaryWriter
import time
from collections import defaultdict

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.data import MotionNormalizer
from sonic_g1.models.policy import G1Policy
from sonic_g1.data.bc_dataset import G1MotionDataset, create_bc_dataloaders
from sonic_g1.train.bc_losses import BCLossCollection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BCTrainer:
    """Behavior cloning trainer for G1 policy warm-start."""

    def __init__(self, config: DictConfig, device: torch.device):
        """
        Initialize BC trainer.

        Args:
            config: Training configuration
            device: Torch device
        """
        self.config = config
        self.device = device

        # Training state
        self.step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # Initialize model
        self.policy = self._create_policy()

        # Initialize optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        # Loss collection
        self.loss_fn = BCLossCollection(config.losses, device)

        # Data normalizers
        self.state_normalizer = None
        self.action_normalizer = None
        self._load_normalizers()

        # Logging
        self.writers = self._setup_logging()

        # Metrics tracking
        self.train_metrics = defaultdict(list)
        self.val_metrics = defaultdict(list)

        logger.info(f"Initialized BC trainer with {sum(p.numel() for p in self.policy.parameters())} parameters")

    def _create_policy(self) -> G1Policy:
        """Create and initialize policy network."""
        # Get dimensions from config or sample data
        obs_dim = self.config.model.obs_dim
        action_dim = self.config.model.action_dim

        policy = G1Policy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            config=self.config.model.policy
        ).to(self.device)

        logger.info(f"Created policy: obs_dim={obs_dim}, action_dim={action_dim}")
        return policy

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer."""
        optimizer_type = self.config.training.optimizer.type
        lr = self.config.training.optimizer.lr
        weight_decay = self.config.training.optimizer.get('weight_decay', 0.0)

        if optimizer_type == 'adam':
            return torch.optim.Adam(
                self.policy.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                eps=self.config.training.optimizer.get('eps', 1e-8)
            )
        elif optimizer_type == 'adamw':
            return torch.optim.AdamW(
                self.policy.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_type == 'sgd':
            return torch.optim.SGD(
                self.policy.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                momentum=self.config.training.optimizer.get('momentum', 0.9)
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler."""
        if not self.config.training.get('use_scheduler', False):
            return None

        scheduler_type = self.config.training.scheduler.type

        if scheduler_type == 'step':
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.training.scheduler.step_size,
                gamma=self.config.training.scheduler.gamma
            )
        elif scheduler_type == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.max_epochs,
                eta_min=self.config.training.scheduler.get('min_lr', 0)
            )
        elif scheduler_type == 'reduce_on_plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.config.training.scheduler.factor,
                patience=self.config.training.scheduler.patience,
                verbose=True
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    def _load_normalizers(self):
        """Load state and action normalizers if available."""
        if self.config.data.get('state_normalizer_path'):
            try:
                self.state_normalizer = MotionNormalizer()
                self.state_normalizer.load_statistics(self.config.data.state_normalizer_path)
                logger.info("Loaded state normalizer")
            except Exception as e:
                logger.warning(f"Failed to load state normalizer: {e}")

        if self.config.data.get('action_normalizer_path'):
            try:
                self.action_normalizer = MotionNormalizer()
                self.action_normalizer.load_statistics(self.config.data.action_normalizer_path)
                logger.info("Loaded action normalizer")
            except Exception as e:
                logger.warning(f"Failed to load action normalizer: {e}")

    def _setup_logging(self) -> Dict:
        """Setup logging backends."""
        writers = {}

        # TensorBoard
        if self.config.logging.use_tensorboard:
            log_dir = Path(self.config.logging.log_dir) / "bc_warmstart" / "tensorboard"
            log_dir.mkdir(parents=True, exist_ok=True)
            writers['tensorboard'] = SummaryWriter(str(log_dir))

        # Weights & Biases
        if self.config.logging.get('use_wandb', False):
            wandb.init(
                project=self.config.logging.wandb_project,
                name=f"{self.config.experiment.name}_bc_warmstart",
                config=OmegaConf.to_container(self.config, resolve=True),
                tags=['behavior_cloning', 'warmstart'] + self.config.experiment.get('tags', [])
            )
            writers['wandb'] = wandb

        return writers

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.policy.train()
        epoch_metrics = defaultdict(list)

        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            states = batch['states'].to(self.device)
            actions = batch['actions'].to(self.device)

            # Optional additional data
            state_derivatives = batch.get('state_derivatives', None)
            if state_derivatives is not None:
                state_derivatives = state_derivatives.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()

            # Get policy distribution
            action_dist = self.policy.get_distribution(states)
            predicted_actions = action_dist.mean

            # Compute losses
            losses = self.loss_fn.compute_losses(
                predicted_actions=predicted_actions,
                target_actions=actions,
                action_distribution=action_dist,
                states=states,
                state_derivatives=state_derivatives
            )

            total_loss = losses['total_loss']

            # Backward pass
            total_loss.backward()

            # Gradient clipping
            if self.config.training.get('clip_grad_norm'):
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    self.config.training.clip_grad_norm
                )

            self.optimizer.step()

            # Track metrics
            for key, value in losses.items():
                epoch_metrics[key].append(value.item())

            # Additional metrics
            with torch.no_grad():
                action_mse = F.mse_loss(predicted_actions, actions)
                action_mae = F.l1_loss(predicted_actions, actions)
                epoch_metrics['action_mse'].append(action_mse.item())
                epoch_metrics['action_mae'].append(action_mae.item())

                # Policy statistics
                policy_std = action_dist.scale.mean()
                epoch_metrics['policy_std'].append(policy_std.item())

            self.step += 1

            # Log training progress
            if batch_idx % self.config.logging.log_interval == 0:
                logger.info(
                    f"Epoch {self.epoch}, Batch {batch_idx}/{len(train_loader)}, "
                    f"Loss: {total_loss.item():.6f}, "
                    f"Action MSE: {action_mse.item():.6f}"
                )

        # Average metrics
        avg_metrics = {key: np.mean(values) for key, values in epoch_metrics.items()}
        return avg_metrics

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate on validation set."""
        self.policy.eval()
        val_metrics = defaultdict(list)

        with torch.no_grad():
            for batch in val_loader:
                states = batch['states'].to(self.device)
                actions = batch['actions'].to(self.device)

                state_derivatives = batch.get('state_derivatives', None)
                if state_derivatives is not None:
                    state_derivatives = state_derivatives.to(self.device)

                # Forward pass
                action_dist = self.policy.get_distribution(states)
                predicted_actions = action_dist.mean

                # Compute losses
                losses = self.loss_fn.compute_losses(
                    predicted_actions=predicted_actions,
                    target_actions=actions,
                    action_distribution=action_dist,
                    states=states,
                    state_derivatives=state_derivatives
                )

                for key, value in losses.items():
                    val_metrics[key].append(value.item())

                # Additional metrics
                action_mse = F.mse_loss(predicted_actions, actions)
                action_mae = F.l1_loss(predicted_actions, actions)
                val_metrics['action_mse'].append(action_mse.item())
                val_metrics['action_mae'].append(action_mae.item())

                policy_std = action_dist.scale.mean()
                val_metrics['policy_std'].append(policy_std.item())

        # Average metrics
        avg_metrics = {key: np.mean(values) for key, values in val_metrics.items()}
        return avg_metrics

    def evaluate_rollout(self, eval_env, num_episodes: int = 5) -> Dict[str, float]:
        """Evaluate policy with environment rollouts."""
        self.policy.eval()

        episode_rewards = []
        episode_lengths = []
        tracking_errors = []

        with torch.no_grad():
            for episode in range(num_episodes):
                obs, info = eval_env.reset()
                episode_reward = 0
                episode_length = 0
                episode_tracking_error = 0
                done = False

                while not done and episode_length < self.config.eval.max_episode_length:
                    # Normalize observation if normalizer available
                    if self.state_normalizer:
                        obs_norm = self.state_normalizer.normalize({'observations': obs.reshape(1, -1)})
                        obs_tensor = torch.FloatTensor(obs_norm['observations']).to(self.device)
                    else:
                        obs_tensor = torch.FloatTensor(obs.reshape(1, -1)).to(self.device)

                    # Get action from policy (deterministic)
                    action, _ = self.policy.act(obs_tensor, deterministic=True)
                    action = action.cpu().numpy().flatten()

                    # Denormalize action if normalizer available
                    if self.action_normalizer:
                        action = self.action_normalizer.denormalize({'actions': action.reshape(1, -1)})['actions'].flatten()

                    # Step environment
                    obs, reward, terminated, truncated, info = eval_env.step(action)
                    done = terminated or truncated

                    episode_reward += reward
                    episode_length += 1

                    # Track motion tracking error if available
                    if 'tracking_error' in info:
                        episode_tracking_error += info['tracking_error']

                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                tracking_errors.append(episode_tracking_error / episode_length if episode_length > 0 else 0)

        return {
            'eval_mean_reward': np.mean(episode_rewards),
            'eval_std_reward': np.std(episode_rewards),
            'eval_mean_length': np.mean(episode_lengths),
            'eval_mean_tracking_error': np.mean(tracking_errors)
        }

    def log_metrics(self, metrics: Dict[str, float], prefix: str):
        """Log metrics to all configured backends."""
        # Console logging
        metrics_str = ", ".join([f"{k}: {v:.6f}" for k, v in metrics.items()])
        logger.info(f"{prefix} - Epoch {self.epoch}: {metrics_str}")

        # TensorBoard
        if 'tensorboard' in self.writers:
            for key, value in metrics.items():
                self.writers['tensorboard'].add_scalar(f"{prefix}/{key}", value, self.step)

        # Weights & Biases
        if 'wandb' in self.writers:
            log_dict = {f"{prefix}/{k}": v for k, v in metrics.items()}
            log_dict['epoch'] = self.epoch
            log_dict['step'] = self.step
            self.writers['wandb'].log(log_dict)

    def save_checkpoint(self, filepath: Path, is_best: bool = False):
        """Save training checkpoint."""
        filepath.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'epoch': self.epoch,
            'step': self.step,
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': OmegaConf.to_container(self.config, resolve=True),
            'train_metrics': dict(self.train_metrics),
            'val_metrics': dict(self.val_metrics)
        }

        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(checkpoint, filepath)

        if is_best:
            best_path = filepath.parent / 'best_model.pth'
            torch.save(checkpoint, best_path)

        logger.info(f"Saved checkpoint: {filepath}")

    def load_checkpoint(self, filepath: Path):
        """Load training checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        self.best_val_loss = checkpoint['best_val_loss']

        self.policy.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.train_metrics = defaultdict(list, checkpoint.get('train_metrics', {}))
        self.val_metrics = defaultdict(list, checkpoint.get('val_metrics', {}))

        logger.info(f"Loaded checkpoint from epoch {self.epoch}")

    def export_for_ppo(self, export_path: Path):
        """Export trained policy for PPO initialization."""
        export_path.parent.mkdir(parents=True, exist_ok=True)

        # Save only the policy state dict and config
        export_data = {
            'policy_state_dict': self.policy.state_dict(),
            'obs_dim': self.config.model.obs_dim,
            'action_dim': self.config.model.action_dim,
            'policy_config': OmegaConf.to_container(self.config.model.policy, resolve=True),
            'training_info': {
                'bc_epochs': self.epoch,
                'bc_steps': self.step,
                'final_val_loss': self.best_val_loss
            },
            'normalizers': {
                'state_normalizer_path': self.config.data.get('state_normalizer_path'),
                'action_normalizer_path': self.config.data.get('action_normalizer_path')
            }
        }

        torch.save(export_data, export_path)
        logger.info(f"Exported BC policy for PPO: {export_path}")

    def train(self, train_loader: DataLoader, val_loader: DataLoader, eval_env=None):
        """Main training loop."""
        logger.info(f"Starting BC training for {self.config.training.max_epochs} epochs")

        for epoch in range(self.epoch, self.config.training.max_epochs):
            self.epoch = epoch
            epoch_start_time = time.time()

            # Training
            train_metrics = self.train_epoch(train_loader)
            self.log_metrics(train_metrics, 'train')

            # Validation
            val_metrics = self.validate(val_loader)
            self.log_metrics(val_metrics, 'val')

            # Environment evaluation
            if eval_env is not None and epoch % self.config.eval.eval_interval == 0:
                eval_metrics = self.evaluate_rollout(eval_env, self.config.eval.num_episodes)
                self.log_metrics(eval_metrics, 'eval')

            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['total_loss'])
                else:
                    self.scheduler.step()

            # Model selection
            val_loss = val_metrics['total_loss']
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # Checkpointing
            if epoch % self.config.training.checkpoint_interval == 0:
                checkpoint_path = Path(self.config.training.checkpoint_dir) / f'checkpoint_epoch_{epoch}.pth'
                self.save_checkpoint(checkpoint_path, is_best)

            # Early stopping
            if (self.config.training.get('early_stopping_patience') and
                self.patience_counter >= self.config.training.early_stopping_patience):
                logger.info(f"Early stopping after {self.patience_counter} epochs without improvement")
                break

            # Epoch timing
            epoch_time = time.time() - epoch_start_time
            logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s")

        # Final checkpoint and export
        final_checkpoint = Path(self.config.training.checkpoint_dir) / 'final_checkpoint.pth'
        self.save_checkpoint(final_checkpoint, False)

        # Export for PPO
        ppo_export_path = Path(self.config.training.checkpoint_dir) / 'bc_policy_for_ppo.pth'
        self.export_for_ppo(ppo_export_path)

        logger.info("BC training completed successfully")


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description="Behavior Cloning Warm-start for Unitree G1")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--export-only", action="store_true", help="Export existing checkpoint for PPO")
    parser.add_argument("--seed", type=int, help="Random seed override")

    args = parser.parse_args()

    # Load configuration
    config = OmegaConf.load(args.config)

    # Override seed if provided
    if args.seed is not None:
        config.training.seed = args.seed

    # Set random seed
    set_seed(config.training.seed)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() and config.training.use_cuda else "cpu")
    logger.info(f"Using device: {device}")

    # Initialize trainer
    trainer = BCTrainer(config, device)

    # Load checkpoint if resuming
    if args.resume:
        trainer.load_checkpoint(Path(args.resume))

    # Export only mode
    if args.export_only:
        if not args.resume:
            raise ValueError("Must specify --resume when using --export-only")

        export_path = Path(config.training.checkpoint_dir) / 'bc_policy_for_ppo.pth'
        trainer.export_for_ppo(export_path)
        logger.info("Export completed")
        return

    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader, dataset_info = create_bc_dataloaders(config)

    logger.info(f"Dataset info: {dataset_info}")
    logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.info(f"Validation samples: {len(val_loader.dataset)}")

    # Optional: Create evaluation environment
    eval_env = None
    if config.eval.get('use_env_eval', False):
        try:
            from src.envs.g1 import create_g1_env
            eval_env = create_g1_env(
                model_path=config.eval.env_config.model_path,
                config=config.eval.env_config
            )
            logger.info("Created evaluation environment")
        except Exception as e:
            logger.warning(f"Failed to create evaluation environment: {e}")

    # Start training
    trainer.train(train_loader, val_loader, eval_env)

    # Close logging
    if 'tensorboard' in trainer.writers:
        trainer.writers['tensorboard'].close()
    if 'wandb' in trainer.writers:
        trainer.writers['wandb'].finish()


if __name__ == "__main__":
    main()