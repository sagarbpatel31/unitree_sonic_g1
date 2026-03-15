"""
Logging and monitoring utilities for training and evaluation.
Integrates with TensorBoard, Weights & Biases, and standard logging.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union
import time
from datetime import datetime

import torch
import wandb
from torch.utils.tensorboard import SummaryWriter

from .utils import get_device


class Logger:
    """Unified logger for experiments with multiple backends."""

    def __init__(
        self,
        experiment_name: str,
        log_dir: Union[str, Path],
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_config: Optional[Dict[str, Any]] = None,
        log_level: str = "INFO",
    ):
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Setup Python logging
        self._setup_python_logging(log_level)

        # Setup TensorBoard
        self.tensorboard_writer = None
        if use_tensorboard:
            tb_dir = self.log_dir / "tensorboard"
            tb_dir.mkdir(exist_ok=True)
            self.tensorboard_writer = SummaryWriter(tb_dir)

        # Setup Weights & Biases
        self.use_wandb = use_wandb
        if use_wandb and wandb_config:
            wandb.init(
                project=wandb_config.get("project", "unitree-g1"),
                entity=wandb_config.get("entity"),
                name=experiment_name,
                config=wandb_config.get("config", {}),
                tags=wandb_config.get("tags", []),
                dir=str(self.log_dir),
            )

        # Metrics storage
        self.step = 0
        self.metrics_buffer = {}
        self.start_time = time.time()

        self.info(f"Logger initialized for experiment: {experiment_name}")
        self.info(f"Log directory: {self.log_dir}")
        self.info(f"Device: {get_device()}")

    def _setup_python_logging(self, log_level: str):
        """Setup Python logging with file and console handlers."""
        # Create logger
        self.logger = logging.getLogger(self.experiment_name)
        self.logger.setLevel(getattr(logging, log_level.upper()))

        # Clear existing handlers
        self.logger.handlers.clear()

        # File handler
        log_file = self.log_dir / "experiment.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def log_scalar(self, tag: str, value: float, step: Optional[int] = None):
        """Log a scalar value."""
        if step is None:
            step = self.step

        # TensorBoard
        if self.tensorboard_writer:
            self.tensorboard_writer.add_scalar(tag, value, step)

        # Weights & Biases
        if self.use_wandb:
            wandb.log({tag: value}, step=step)

        # Store in buffer
        self.metrics_buffer[tag] = value

    def log_scalars(self, scalars: Dict[str, float], step: Optional[int] = None):
        """Log multiple scalar values."""
        for tag, value in scalars.items():
            self.log_scalar(tag, value, step)

    def log_histogram(self, tag: str, values: torch.Tensor, step: Optional[int] = None):
        """Log histogram of tensor values."""
        if step is None:
            step = self.step

        if self.tensorboard_writer:
            self.tensorboard_writer.add_histogram(tag, values, step)

    def log_image(self, tag: str, image: torch.Tensor, step: Optional[int] = None):
        """Log an image."""
        if step is None:
            step = self.step

        if self.tensorboard_writer:
            self.tensorboard_writer.add_image(tag, image, step)

        if self.use_wandb:
            wandb.log({tag: wandb.Image(image)}, step=step)

    def log_video(self, tag: str, video: torch.Tensor, step: Optional[int] = None, fps: int = 30):
        """Log a video."""
        if step is None:
            step = self.step

        if self.tensorboard_writer:
            self.tensorboard_writer.add_video(tag, video, step, fps=fps)

        if self.use_wandb:
            # Convert to numpy for wandb
            video_np = video.cpu().numpy()
            wandb.log({tag: wandb.Video(video_np, fps=fps)}, step=step)

    def log_model(self, model: torch.nn.Module, input_data: torch.Tensor):
        """Log model graph."""
        if self.tensorboard_writer:
            try:
                self.tensorboard_writer.add_graph(model, input_data)
            except Exception as e:
                self.warning(f"Failed to log model graph: {e}")

        if self.use_wandb:
            wandb.watch(model, log="all", log_freq=1000)

    def log_hyperparameters(self, hparams: Dict[str, Any], metrics: Dict[str, float]):
        """Log hyperparameters and corresponding metrics."""
        if self.tensorboard_writer:
            self.tensorboard_writer.add_hparams(hparams, metrics)

        if self.use_wandb:
            wandb.config.update(hparams)

    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)

    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)

    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)

    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)

    def step_increment(self, steps: int = 1):
        """Increment step counter."""
        self.step += steps

    def get_step(self) -> int:
        """Get current step."""
        return self.step

    def get_elapsed_time(self) -> float:
        """Get elapsed time since logger creation."""
        return time.time() - self.start_time

    def save_checkpoint(self, checkpoint_data: Dict[str, Any], name: str = "checkpoint"):
        """Save checkpoint data."""
        checkpoint_path = self.log_dir / "checkpoints"
        checkpoint_path.mkdir(exist_ok=True)

        filename = f"{name}_step_{self.step}.pt"
        filepath = checkpoint_path / filename

        torch.save(checkpoint_data, filepath)
        self.info(f"Checkpoint saved: {filepath}")

        return filepath

    def close(self):
        """Close all logging resources."""
        if self.tensorboard_writer:
            self.tensorboard_writer.close()

        if self.use_wandb:
            wandb.finish()

        # Close file handlers
        for handler in self.logger.handlers:
            handler.close()

        self.info(f"Logger closed. Total time: {self.get_elapsed_time():.2f}s")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class MetricsTracker:
    """Simple metrics tracking for evaluation."""

    def __init__(self):
        self.metrics = {}
        self.counts = {}

    def update(self, metrics: Dict[str, float]):
        """Update metrics with new values."""
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = 0.0
                self.counts[key] = 0

            self.metrics[key] += value
            self.counts[key] += 1

    def get_averages(self) -> Dict[str, float]:
        """Get average values for all metrics."""
        return {
            key: self.metrics[key] / self.counts[key]
            for key in self.metrics.keys()
            if self.counts[key] > 0
        }

    def reset(self):
        """Reset all metrics."""
        self.metrics.clear()
        self.counts.clear()


def setup_logging(
    experiment_name: str,
    log_dir: Union[str, Path],
    config: Optional[Dict[str, Any]] = None,
) -> Logger:
    """
    Setup logging for an experiment.

    Args:
        experiment_name: Name of the experiment
        log_dir: Directory for log files
        config: Configuration dictionary

    Returns:
        Logger instance
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_name = f"{experiment_name}_{timestamp}"

    log_config = config.get("logging", {}) if config else {}
    wandb_config = log_config.get("wandb", {})

    logger = Logger(
        experiment_name=full_name,
        log_dir=Path(log_dir) / full_name,
        use_tensorboard=log_config.get("use_tensorboard", True),
        use_wandb=wandb_config.get("enabled", False),
        wandb_config=wandb_config,
        log_level=log_config.get("level", "INFO"),
    )

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a standard Python logger."""
    return logging.getLogger(name)