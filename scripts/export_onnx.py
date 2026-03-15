#!/usr/bin/env python3
"""
Model export script for G1 policies.

This script exports trained PyTorch policies to ONNX format for deployment.
"""

import argparse
import logging
from pathlib import Path

import torch
from omegaconf import OmegaConf

from sonic_g1.deploy import PolicyExporter

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def main():
    """Main export function."""
    parser = argparse.ArgumentParser(description='Export G1 policy to ONNX')
    parser.add_argument('--checkpoint', required=True, help='Path to policy checkpoint')
    parser.add_argument('--output', required=True, help='Output ONNX file path')
    parser.add_argument('--config', help='Export configuration file')
    parser.add_argument('--validate', action='store_true', help='Validate exported model')
    parser.add_argument('--optimize', action='store_true', help='Optimize ONNX model')
    parser.add_argument('--tensorrt', action='store_true', help='Create TensorRT engine')
    parser.add_argument('--log-level', default='INFO', help='Logging level')

    args = parser.parse_args()

    setup_logging(args.log_level)
    logger.info("Starting policy export")

    # Validate paths
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load export configuration
    export_config = {}
    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            export_config = OmegaConf.load(config_path)
        else:
            logger.warning(f"Config file not found: {config_path}")

    try:
        # Initialize exporter
        exporter = PolicyExporter(export_config)

        # Export to ONNX
        logger.info(f"Exporting {checkpoint_path} to {output_path}")
        export_info = exporter.export_policy_to_onnx(
            checkpoint_path=str(checkpoint_path),
            output_path=str(output_path),
            validate=args.validate
        )

        logger.info("Export completed successfully")
        logger.info(f"Model size: {export_info['model_size_mb']:.2f} MB")

        if args.validate and 'validation' in export_info:
            validation = export_info['validation']
            if validation['validation_passed']:
                logger.info("✓ Model validation PASSED")
                logger.info(f"  Max absolute error: {validation['max_absolute_error']:.2e}")
                logger.info(f"  Mean absolute error: {validation['mean_absolute_error']:.2e}")
            else:
                logger.warning("✗ Model validation FAILED")
                logger.warning(f"  Max absolute error: {validation['max_absolute_error']:.2e}")

        # Optimize if requested
        if args.optimize:
            logger.info("Optimizing ONNX model")
            optimized_path = str(output_path).replace('.onnx', '_optimized.onnx')
            exporter.optimize_onnx_model(str(output_path), optimized_path)
            logger.info(f"Optimized model saved to: {optimized_path}")

        # Create TensorRT engine if requested
        if args.tensorrt:
            logger.info("Creating TensorRT engine")
            tensorrt_path = str(output_path).replace('.onnx', '.trt')
            success = exporter.create_tensorrt_engine(str(output_path), tensorrt_path)
            if success:
                logger.info(f"TensorRT engine saved to: {tensorrt_path}")
            else:
                logger.warning("TensorRT engine creation failed")

        # Print export summary
        logger.info("\nExport Summary:")
        logger.info(f"  Input checkpoint: {checkpoint_path}")
        logger.info(f"  Output ONNX: {output_path}")
        logger.info(f"  Input shape: {export_info['input_shape']}")
        logger.info(f"  Observation dim: {export_info['obs_dim']}")
        logger.info(f"  Action dim: {export_info['action_dim']}")
        logger.info(f"  Model size: {export_info['model_size_mb']:.2f} MB")

        return 0

    except Exception as e:
        logger.error(f"Export failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())