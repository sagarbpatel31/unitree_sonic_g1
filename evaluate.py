#!/usr/bin/env python3
"""
Main evaluation script for trained G1 models.
Runs comprehensive evaluation across multiple scenarios.
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch

from src.core.config import load_config
from src.core.logging import setup_logging
from src.core.utils import set_seed, get_device
from src.envs import create_g1_environment
from src.models.transformer_policy import TransformerPolicy
from src.evaluation.evaluator import ModelEvaluator


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained G1 models")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to evaluation configuration file")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default="results/evaluation",
                       help="Output directory for results")
    parser.add_argument("--scenarios", type=str, nargs="*",
                       help="Specific scenarios to run (default: all)")
    parser.add_argument("--overrides", type=str, nargs="*", default=[],
                       help="Configuration overrides")

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config, overrides=args.overrides)

    # Setup logging
    logger = setup_logging(
        experiment_name=f"eval_{config.experiment.name}",
        log_dir="logs/evaluation",
        config=config.to_dict()
    )

    try:
        # Set random seed
        set_seed(config.seed, config.get("deterministic", True))

        # Load checkpoint
        logger.info(f"Loading checkpoint: {args.checkpoint}")
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        device = get_device(config.get("device", "auto"))
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Create environment for observation/action dimensions
        env = create_g1_environment(config)
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        logger.info(f"Environment: {config.env.name}")
        logger.info(f"Observation dim: {obs_dim}")
        logger.info(f"Action dim: {action_dim}")

        # Create and load model
        if "config" in checkpoint:
            # Use config from checkpoint
            from src.core.config import Config
            model_config = Config(checkpoint["config"])
            model = TransformerPolicy(model_config, obs_dim, action_dim)
        else:
            # Use provided config
            model = TransformerPolicy(config, obs_dim, action_dim)

        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)

        logger.info(f"Loaded model from step {checkpoint.get('step', 'unknown')}")

        # Create evaluator
        evaluator = ModelEvaluator(model, config, logger)

        # Filter scenarios if specified
        if args.scenarios:
            evaluator.scenarios = [
                scenario for scenario in evaluator.scenarios
                if scenario["name"] in args.scenarios
            ]
            logger.info(f"Running specific scenarios: {args.scenarios}")

        # Run evaluation
        logger.info("Starting evaluation...")
        results = evaluator.evaluate_all_scenarios()

        # Generate report
        output_dir = Path(args.output)
        evaluator.generate_report(results, output_dir)

        # Log summary
        summary = results.get("summary", {})
        logger.info("Evaluation completed successfully")
        logger.info(f"Results saved to: {output_dir}")

        # Print key metrics
        overall_perf = summary.get("overall_performance", {})
        if "success_rate" in overall_perf:
            success_stats = overall_perf["success_rate"]
            logger.info(f"Overall success rate: {success_stats.get('mean', 0):.2%} ± {success_stats.get('std', 0):.2%}")

        if "tracking_error_mean" in overall_perf:
            tracking_stats = overall_perf["tracking_error_mean"]
            logger.info(f"Tracking error: {tracking_stats.get('mean', 0):.4f} ± {tracking_stats.get('std', 0):.4f}")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        logger.close()

    return 0


if __name__ == "__main__":
    exit(main())