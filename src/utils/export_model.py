"""
Model export utilities for deployment.
Supports ONNX and TorchScript export with optimization.
"""

import torch
import torch.onnx
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np

from ..core.config import Config
from ..core.logging import get_logger
from ..models.transformer_policy import TransformerPolicy


logger = get_logger(__name__)


def export_to_onnx(
    model: TransformerPolicy,
    dummy_input: torch.Tensor,
    output_path: str,
    config: Optional[Config] = None,
    optimize: bool = True,
    verify: bool = True,
) -> bool:
    """
    Export model to ONNX format.

    Args:
        model: Trained model to export
        dummy_input: Sample input for tracing
        output_path: Path to save ONNX model
        config: Model configuration
        optimize: Whether to optimize the exported model
        verify: Whether to verify the exported model

    Returns:
        Success status
    """
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Set model to evaluation mode
        model.eval()

        # Get export configuration
        export_config = config.get("models.transformer.export.onnx", {}) if config else {}
        opset_version = export_config.get("opset_version", 17)
        dynamic_axes = export_config.get("dynamic_axes", {})

        # Default dynamic axes for batch and sequence dimensions
        if not dynamic_axes:
            dynamic_axes = {
                "input": {0: "batch_size", 1: "sequence_length"},
                "output": {0: "batch_size"}
            }

        logger.info(f"Exporting model to ONNX: {output_path}")
        logger.info(f"Input shape: {dummy_input.shape}")
        logger.info(f"Opset version: {opset_version}")

        # Export to ONNX
        with torch.no_grad():
            torch.onnx.export(
                model,
                dummy_input,
                str(output_path),
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=["observation_sequence"],
                output_names=["actions"],
                dynamic_axes=dynamic_axes,
                verbose=False,
            )

        # Optimize exported model
        if optimize:
            _optimize_onnx_model(output_path)

        # Verify exported model
        if verify:
            success = _verify_onnx_model(model, dummy_input, output_path)
            if not success:
                logger.error("ONNX model verification failed")
                return False

        # Log model information
        _log_onnx_model_info(output_path)

        logger.info(f"Successfully exported model to: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to export model to ONNX: {e}")
        return False


def export_to_torchscript(
    model: TransformerPolicy,
    dummy_input: torch.Tensor,
    output_path: str,
    config: Optional[Config] = None,
    method: str = "trace",
) -> bool:
    """
    Export model to TorchScript format.

    Args:
        model: Trained model to export
        dummy_input: Sample input for tracing
        output_path: Path to save TorchScript model
        config: Model configuration
        method: Export method ("trace" or "script")

    Returns:
        Success status
    """
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Set model to evaluation mode
        model.eval()

        logger.info(f"Exporting model to TorchScript: {output_path}")
        logger.info(f"Export method: {method}")

        with torch.no_grad():
            if method == "trace":
                # Trace the model
                traced_model = torch.jit.trace(model, dummy_input)
            elif method == "script":
                # Script the model
                traced_model = torch.jit.script(model)
            else:
                raise ValueError(f"Unknown export method: {method}")

            # Optimize for inference
            export_config = config.get("models.transformer.export.torchscript", {}) if config else {}
            if export_config.get("optimize_for_inference", True):
                traced_model = torch.jit.optimize_for_inference(traced_model)

            # Save model
            traced_model.save(str(output_path))

        # Verify exported model
        success = _verify_torchscript_model(model, dummy_input, output_path)
        if not success:
            logger.error("TorchScript model verification failed")
            return False

        logger.info(f"Successfully exported model to: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to export model to TorchScript: {e}")
        return False


def _optimize_onnx_model(model_path: Path) -> None:
    """Optimize ONNX model for inference."""
    try:
        import onnx
        import onnxruntime as ort
        from onnxruntime.tools import onnx_model_utils

        logger.info("Optimizing ONNX model...")

        # Load model
        model = onnx.load(str(model_path))

        # Basic optimization
        optimized_model = onnx_model_utils.optimize_model(
            str(model_path),
            model_type="transformer",
            num_heads=0,  # Auto-detect
            hidden_size=0,  # Auto-detect
        )

        # Save optimized model
        if optimized_model:
            onnx.save(optimized_model, str(model_path))
            logger.info("ONNX model optimization completed")

    except ImportError:
        logger.warning("ONNX optimization tools not available, skipping optimization")
    except Exception as e:
        logger.warning(f"ONNX optimization failed: {e}")


def _verify_onnx_model(
    original_model: TransformerPolicy,
    dummy_input: torch.Tensor,
    onnx_path: Path,
    tolerance: float = 1e-5,
) -> bool:
    """Verify ONNX model produces same outputs as original."""
    try:
        import onnxruntime as ort

        # Get original output
        original_model.eval()
        with torch.no_grad():
            original_output = original_model.forward(dummy_input)["actions"]

        # Create ONNX inference session
        session = ort.InferenceSession(str(onnx_path))

        # Run ONNX inference
        onnx_input = {session.get_inputs()[0].name: dummy_input.cpu().numpy()}
        onnx_output = session.run(None, onnx_input)[0]

        # Compare outputs
        original_np = original_output.cpu().numpy()
        max_diff = np.max(np.abs(original_np - onnx_output))

        if max_diff < tolerance:
            logger.info(f"ONNX verification passed (max diff: {max_diff:.2e})")
            return True
        else:
            logger.error(f"ONNX verification failed (max diff: {max_diff:.2e})")
            return False

    except ImportError:
        logger.warning("ONNX Runtime not available, skipping verification")
        return True
    except Exception as e:
        logger.error(f"ONNX verification failed: {e}")
        return False


def _verify_torchscript_model(
    original_model: TransformerPolicy,
    dummy_input: torch.Tensor,
    ts_path: Path,
    tolerance: float = 1e-5,
) -> bool:
    """Verify TorchScript model produces same outputs as original."""
    try:
        # Get original output
        original_model.eval()
        with torch.no_grad():
            original_output = original_model.forward(dummy_input)["actions"]

        # Load TorchScript model
        ts_model = torch.jit.load(str(ts_path))

        # Run TorchScript inference
        with torch.no_grad():
            ts_output = ts_model.forward(dummy_input)["actions"]

        # Compare outputs
        max_diff = torch.max(torch.abs(original_output - ts_output)).item()

        if max_diff < tolerance:
            logger.info(f"TorchScript verification passed (max diff: {max_diff:.2e})")
            return True
        else:
            logger.error(f"TorchScript verification failed (max diff: {max_diff:.2e})")
            return False

    except Exception as e:
        logger.error(f"TorchScript verification failed: {e}")
        return False


def _log_onnx_model_info(model_path: Path) -> None:
    """Log information about exported ONNX model."""
    try:
        import onnx

        model = onnx.load(str(model_path))

        # Get model information
        file_size = model_path.stat().st_size / (1024 * 1024)  # MB

        # Count parameters
        total_params = 0
        for initializer in model.graph.initializer:
            param_size = 1
            for dim in initializer.dims:
                param_size *= dim
            total_params += param_size

        logger.info(f"ONNX Model Info:")
        logger.info(f"  File size: {file_size:.2f} MB")
        logger.info(f"  Parameters: {total_params:,}")
        logger.info(f"  Opset version: {model.opset_import[0].version}")

        # Log input/output shapes
        for input_tensor in model.graph.input:
            shape = [dim.dim_value if dim.dim_value > 0 else "dynamic"
                    for dim in input_tensor.type.tensor_type.shape.dim]
            logger.info(f"  Input '{input_tensor.name}': {shape}")

        for output_tensor in model.graph.output:
            shape = [dim.dim_value if dim.dim_value > 0 else "dynamic"
                    for dim in output_tensor.type.tensor_type.shape.dim]
            logger.info(f"  Output '{output_tensor.name}': {shape}")

    except ImportError:
        logger.warning("ONNX not available for model info logging")
    except Exception as e:
        logger.warning(f"Failed to log ONNX model info: {e}")


def create_deployment_package(
    model: TransformerPolicy,
    config: Config,
    output_dir: str,
    formats: list = ["onnx", "torchscript"],
) -> Dict[str, str]:
    """
    Create complete deployment package with multiple model formats.

    Args:
        model: Trained model
        config: Model configuration
        output_dir: Output directory
        formats: Export formats to include

    Returns:
        Dictionary mapping format names to file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create dummy input
    obs_dim = model.obs_dim
    sequence_length = model.sequence_length
    dummy_input = torch.randn(1, sequence_length, obs_dim)

    exported_files = {}

    # Export in requested formats
    if "onnx" in formats:
        onnx_path = output_dir / "model.onnx"
        if export_to_onnx(model, dummy_input, str(onnx_path), config):
            exported_files["onnx"] = str(onnx_path)

    if "torchscript" in formats:
        ts_path = output_dir / "model.pt"
        if export_to_torchscript(model, dummy_input, str(ts_path), config):
            exported_files["torchscript"] = str(ts_path)

    # Save configuration
    config_path = output_dir / "config.yaml"
    config.save(config_path)
    exported_files["config"] = str(config_path)

    # Save model metadata
    metadata = {
        "model_type": model.__class__.__name__,
        "input_dim": obs_dim,
        "output_dim": model.action_dim,
        "sequence_length": sequence_length,
        "hidden_dim": model.hidden_dim,
        "num_layers": model.num_layers,
        "num_heads": getattr(model, 'num_heads', None),
    }

    metadata_path = output_dir / "metadata.json"
    from ..core.utils import save_json
    save_json(metadata, metadata_path)
    exported_files["metadata"] = str(metadata_path)

    logger.info(f"Deployment package created in: {output_dir}")
    logger.info(f"Exported formats: {list(exported_files.keys())}")

    return exported_files