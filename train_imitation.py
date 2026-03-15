#!/usr/bin/env python3
"""
Main script for training motion imitation models using behavior cloning.
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
from torch.utils.data import DataLoader

from src.core.config import load_config
from src.core.logging import setup_logging
from src.core.utils import set_seed, get_device
from src.envs import create_g1_environment
from src.models.transformer_policy import TransformerPolicy
from src.training.bc_trainer import BehaviorCloningTrainer
from src.training.data_loader import MotionDataLoader


def main():
    parser = argparse.ArgumentParser(description="Train G1 motion imitation model")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to configuration file")
    parser.add_argument("--overrides", type=str, nargs="*", default=[],
                       help="Configuration overrides (e.g., training.batch_size=128)")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--eval-only", action="store_true",
                       help="Run evaluation only")

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config, overrides=args.overrides)

    # Setup logging
    logger = setup_logging(
        experiment_name=config.experiment.name,
        log_dir="logs",
        config=config.to_dict()
    )

    try:
        # Set random seed
        set_seed(config.seed, config.get("deterministic", False))

        # Log system information
        from src.core.utils import get_system_info
        system_info = get_system_info()
        logger.info(f"System info: {system_info}")

        # Create environment (for observation/action space info)
        env = create_g1_environment(config)
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        logger.info(f"Environment: {config.env.name}")
        logger.info(f"Observation dim: {obs_dim}")
        logger.info(f"Action dim: {action_dim}")

        # Create model
        model = TransformerPolicy(config, obs_dim, action_dim)
        logger.info(f"Model: {model.__class__.__name__}")

        # Log model architecture
        logger.log_model(model, torch.randn(1, config.model.sequence_length, obs_dim))

        # Create trainer
        trainer = BehaviorCloningTrainer(model, config, logger)

        # Load checkpoint if resuming
        if args.resume:
            trainer.load_checkpoint(args.resume)

        # Create data loaders
        logger.info("Loading training data...")
        data_loader = MotionDataLoader(config)

        train_dataset = data_loader.get_training_dataset()
        eval_dataset = data_loader.get_evaluation_dataset()

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.training.get("num_workers", 4),
            pin_memory=True,
            drop_last=True
        )

        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.training.get("num_workers", 4),
            pin_memory=True
        ) if eval_dataset else None

        logger.info(f"Training samples: {len(train_dataset)}")
        if eval_dataset:
            logger.info(f"Evaluation samples: {len(eval_dataset)}")

        # Run training or evaluation
        if args.eval_only:
            if eval_dataloader is None:
                raise ValueError("No evaluation data available")

            logger.info("Running evaluation only...")
            eval_metrics = trainer.evaluate(eval_dataloader)
            logger.info(f"Evaluation results: {eval_metrics}")
        else:
            # Start training
            trainer.train(train_dataloader, eval_dataloader)

        logger.info("Training completed successfully")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        logger.close()


if __name__ == "__main__":
    main()