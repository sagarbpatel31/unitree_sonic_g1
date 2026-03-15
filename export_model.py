#!/usr/bin/env python3
"""
Export trained models for deployment.
Supports ONNX and TorchScript formats.
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch

from src.core.config import load_config
from src.core.utils import get_device
from src.models.transformer_policy import TransformerPolicy
from src.utils.export_model import create_deployment_package


def main():
    parser = argparse.ArgumentParser(description="Export G1 trained models")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--config", type=str,
                       help="Path to config file (if not in checkpoint)")
    parser.add_argument("--output", type=str, default="models/deployed",
                       help="Output directory for exported models")
    parser.add_argument("--formats", type=str, nargs="+",
                       default=["onnx", "torchscript"],
                       choices=["onnx", "torchscript"],
                       help="Export formats")
    parser.add_argument("--optimize", action="store_true", default=True,
                       help="Optimize exported models")

    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")

    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint file not found: {checkpoint_path}")
        return 1

    device = get_device("cpu")  # Use CPU for export
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load configuration
    if args.config:
        config = load_config(args.config)
    elif "config" in checkpoint:
        from src.core.config import Config
        config = Config(checkpoint["config"])
    else:
        print("Error: No configuration found. Please provide --config")
        return 1

    print(f"Model type: {config.model.name}")
    print(f"Export formats: {args.formats}")

    try:
        # Create environment to get observation/action dimensions
        from src.envs import create_g1_environment
        env = create_g1_environment(config)
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        print(f"Observation dim: {obs_dim}")
        print(f"Action dim: {action_dim}")

        # Create model
        model = TransformerPolicy(config, obs_dim, action_dim)

        # Load model weights
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        print(f"Loaded model from step {checkpoint.get('step', 'unknown')}")

        # Create deployment package
        exported_files = create_deployment_package(
            model=model,
            config=config,
            output_dir=args.output,
            formats=args.formats
        )

        print("\nExported files:")
        for format_name, file_path in exported_files.items():
            file_size = Path(file_path).stat().st_size / (1024 * 1024)  # MB
            print(f"  {format_name}: {file_path} ({file_size:.2f} MB)")

        print(f"\nDeployment package ready: {args.output}")

    except Exception as e:
        print(f"Export failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())