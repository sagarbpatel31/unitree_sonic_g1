"""
Behavior Cloning trainer for motion imitation learning.
Implements supervised learning from reference motion data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
import numpy as np
from tqdm import tqdm

from ..core.config import Config
from ..core.logging import Logger
from ..core.utils import Timer, MovingAverage, count_parameters
from ..models.transformer_policy import TransformerPolicy


class BehaviorCloningTrainer:
    """
    Behavior Cloning trainer for learning from demonstration data.

    This trainer implements supervised learning to train a policy network
    to imitate expert demonstrations from motion capture data.
    """

    def __init__(self, model: TransformerPolicy, config: Config, logger: Logger):
        self.model = model
        self.config = config
        self.logger = logger
        self.training_config = config.training

        # Setup device
        self.device = torch.device(config.get("device", "auto"))
        if self.device.type == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = self.model.to(self.device)

        # Log model information
        param_count = count_parameters(self.model)
        self.logger.info(f"Model parameters: {param_count}")

        # Setup optimizer
        self.optimizer = self._setup_optimizer()

        # Setup learning rate scheduler
        self.scheduler = self._setup_scheduler()

        # Training state
        self.step = 0
        self.epoch = 0
        self.best_loss = float('inf')

        # Metrics tracking
        self.train_metrics = MovingAverage(decay=0.95)
        self.eval_metrics = {}

        # Timers
        self.train_timer = Timer("Training")
        self.eval_timer = Timer("Evaluation")

    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer based on configuration."""
        optimizer_config = self.training_config

        optimizer_type = optimizer_config.get("optimizer", "AdamW")
        learning_rate = optimizer_config.learning_rate
        weight_decay = optimizer_config.get("weight_decay", 1e-5)

        if optimizer_type == "AdamW":
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif optimizer_type == "Adam":
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_type == "SGD":
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")

        return optimizer

    def _setup_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Setup learning rate scheduler."""
        scheduler_type = self.training_config.get("scheduler")

        if scheduler_type == "CosineAnnealingLR":
            T_max = self.training_config.total_steps
            min_lr = self.training_config.learning_rate * 0.1
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=T_max, eta_min=min_lr
            )
        elif scheduler_type == "StepLR":
            step_size = self.training_config.get("scheduler_step_size", 100000)
            gamma = self.training_config.get("scheduler_gamma", 0.1)
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=step_size, gamma=gamma
            )
        elif scheduler_type == "ExponentialLR":
            gamma = self.training_config.get("scheduler_gamma", 0.99995)
            scheduler = optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=gamma
            )
        else:
            scheduler = None

        return scheduler

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_metrics = {"loss": 0.0, "action_loss": 0.0, "samples": 0}

        with self.train_timer:
            for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {self.epoch}")):
                # Move batch to device
                observations = batch["observations"].to(self.device)
                actions = batch["actions"].to(self.device)

                # Forward pass
                losses = self.model.compute_loss(observations, actions)

                # Backward pass
                self.optimizer.zero_grad()
                losses["total_loss"].backward()

                # Gradient clipping
                grad_clip = self.training_config.get("grad_clip_norm")
                if grad_clip:
                    nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

                self.optimizer.step()

                if self.scheduler:
                    self.scheduler.step()

                # Update metrics
                batch_size = observations.shape[0]
                epoch_metrics["loss"] += losses["total_loss"].item() * batch_size
                epoch_metrics["action_loss"] += losses["action_loss"].item() * batch_size
                epoch_metrics["samples"] += batch_size

                # Log training metrics
                if self.step % self.config.logging.log_frequency == 0:
                    self._log_training_step(losses, batch_size)

                self.step += 1

                # Early stopping check
                if self.step >= self.training_config.total_steps:
                    break

        # Compute average metrics
        for key in ["loss", "action_loss"]:
            epoch_metrics[key] /= epoch_metrics["samples"]

        return epoch_metrics

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model on validation data."""
        self.model.eval()
        eval_metrics = {"loss": 0.0, "action_loss": 0.0, "samples": 0}

        with torch.no_grad():
            with self.eval_timer:
                for batch in tqdm(dataloader, desc="Evaluating"):
                    observations = batch["observations"].to(self.device)
                    actions = batch["actions"].to(self.device)

                    losses = self.model.compute_loss(observations, actions)

                    batch_size = observations.shape[0]
                    eval_metrics["loss"] += losses["total_loss"].item() * batch_size
                    eval_metrics["action_loss"] += losses["action_loss"].item() * batch_size
                    eval_metrics["samples"] += batch_size

        # Compute averages
        for key in ["loss", "action_loss"]:
            eval_metrics[key] /= eval_metrics["samples"]

        return eval_metrics

    def train(self, train_dataloader: DataLoader,
              eval_dataloader: Optional[DataLoader] = None) -> None:
        """Main training loop."""
        self.logger.info("Starting behavior cloning training")
        self.logger.info(f"Total steps: {self.training_config.total_steps}")
        self.logger.info(f"Batch size: {self.training_config.batch_size}")

        total_epochs = max(1, self.training_config.total_steps // len(train_dataloader))

        for epoch in range(total_epochs):
            self.epoch = epoch

            # Training
            train_metrics = self.train_epoch(train_dataloader)

            # Evaluation
            if eval_dataloader is not None:
                eval_metrics = self.evaluate(eval_dataloader)
                self._log_evaluation(eval_metrics)

                # Save best model
                if eval_metrics["loss"] < self.best_loss:
                    self.best_loss = eval_metrics["loss"]
                    self._save_checkpoint("best_model")

            # Save periodic checkpoint
            if self.step % self.config.logging.save_frequency == 0:
                self._save_checkpoint(f"checkpoint_step_{self.step}")

            # Early stopping
            if self.step >= self.training_config.total_steps:
                break

        # Save final model
        self._save_checkpoint("final_model")
        self.logger.info("Training completed")

    def _log_training_step(self, losses: Dict[str, torch.Tensor], batch_size: int):
        """Log training step metrics."""
        # Update moving average
        for key, loss in losses.items():
            self.train_metrics.update(loss.item())

        # Log to tensorboard/wandb
        metrics = {
            f"train/{key}": loss.item()
            for key, loss in losses.items()
        }
        metrics["train/learning_rate"] = self.optimizer.param_groups[0]["lr"]
        metrics["train/step"] = self.step

        self.logger.log_scalars(metrics, self.step)

        # Log to console
        if self.step % (self.config.logging.log_frequency * 10) == 0:
            avg_loss = self.train_metrics.get()
            self.logger.info(
                f"Step {self.step}: Loss={avg_loss:.4f}, "
                f"LR={self.optimizer.param_groups[0]['lr']:.6f}"
            )

    def _log_evaluation(self, eval_metrics: Dict[str, float]):
        """Log evaluation metrics."""
        # Log to tensorboard/wandb
        metrics = {f"eval/{key}": value for key, value in eval_metrics.items()}
        self.logger.log_scalars(metrics, self.step)

        # Log to console
        self.logger.info(f"Evaluation - Loss: {eval_metrics['loss']:.4f}")

    def _save_checkpoint(self, name: str) -> str:
        """Save model checkpoint."""
        checkpoint_data = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "step": self.step,
            "epoch": self.epoch,
            "best_loss": self.best_loss,
            "config": self.config.to_dict(),
        }

        checkpoint_path = self.logger.save_checkpoint(checkpoint_data, name)
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.step = checkpoint.get("step", 0)
        self.epoch = checkpoint.get("epoch", 0)
        self.best_loss = checkpoint.get("best_loss", float('inf'))

        self.logger.info(f"Loaded checkpoint from step {self.step}")

    def get_model(self) -> TransformerPolicy:
        """Get trained model."""
        return self.model