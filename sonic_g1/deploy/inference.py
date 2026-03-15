"""
Runtime inference engine for deployed G1 controllers.

This module provides a production-ready inference wrapper with normalization,
action clamping, filtering, and safety features for real-time control.
"""

import time
import threading
from typing import Dict, Any, Optional, Tuple, Union, List
import numpy as np
import logging
from pathlib import Path
import json
from abc import ABC, abstractmethod

try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    import torch
except ImportError:
    torch = None

logger = logging.getLogger(__name__)


class InferenceEngine(ABC):
    """Abstract base class for inference engines."""

    @abstractmethod
    def predict(self, observation: np.ndarray) -> np.ndarray:
        """Run inference on observation."""
        pass

    @abstractmethod
    def warmup(self, num_iterations: int = 100):
        """Warmup the inference engine."""
        pass


class ONNXInferenceEngine(InferenceEngine):
    """ONNX Runtime inference engine."""

    def __init__(self, model_path: str, providers: Optional[List[str]] = None):
        """
        Initialize ONNX inference engine.

        Args:
            model_path: Path to ONNX model
            providers: List of execution providers
        """
        if ort is None:
            raise ImportError("onnxruntime is required for ONNX inference")

        # Default providers
        if providers is None:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

        # Create session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(model_path, sess_options, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        logger.info(f"Initialized ONNX inference engine with providers: {self.session.get_providers()}")

    def predict(self, observation: np.ndarray) -> np.ndarray:
        """Run inference on observation."""
        if observation.ndim == 1:
            observation = observation.reshape(1, -1)

        inputs = {self.input_name: observation}
        outputs = self.session.run([self.output_name], inputs)
        return outputs[0]

    def warmup(self, num_iterations: int = 100):
        """Warmup the inference engine."""
        # Get input shape from model
        input_shape = self.session.get_inputs()[0].shape
        if input_shape[0] == 'batch_size' or input_shape[0] is None:
            input_shape[0] = 1

        dummy_input = np.random.randn(*input_shape).astype(np.float32)

        logger.info(f"Warming up ONNX engine with {num_iterations} iterations")
        start_time = time.time()

        for _ in range(num_iterations):
            self.predict(dummy_input)

        warmup_time = time.time() - start_time
        avg_latency = (warmup_time / num_iterations) * 1000

        logger.info(f"Warmup completed. Average latency: {avg_latency:.2f}ms")


class TorchScriptInferenceEngine(InferenceEngine):
    """TorchScript inference engine."""

    def __init__(self, model_path: str, device: str = 'auto'):
        """
        Initialize TorchScript inference engine.

        Args:
            model_path: Path to TorchScript model
            device: Device to run inference on
        """
        if torch is None:
            raise ImportError("torch is required for TorchScript inference")

        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.device = torch.device(device)
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()

        logger.info(f"Initialized TorchScript inference engine on device: {device}")

    def predict(self, observation: np.ndarray) -> np.ndarray:
        """Run inference on observation."""
        if observation.ndim == 1:
            observation = observation.reshape(1, -1)

        input_tensor = torch.from_numpy(observation).float().to(self.device)

        with torch.no_grad():
            output_tensor = self.model(input_tensor)

        return output_tensor.cpu().numpy()

    def warmup(self, num_iterations: int = 100):
        """Warmup the inference engine."""
        # Use dummy input
        dummy_input = torch.randn(1, 100, device=self.device)  # Placeholder size

        logger.info(f"Warming up TorchScript engine with {num_iterations} iterations")
        start_time = time.time()

        with torch.no_grad():
            for _ in range(num_iterations):
                _ = self.model(dummy_input)

        warmup_time = time.time() - start_time
        avg_latency = (warmup_time / num_iterations) * 1000

        logger.info(f"Warmup completed. Average latency: {avg_latency:.2f}ms")


class RuntimeInferenceEngine:
    """
    Production-ready inference wrapper with safety features.

    Provides normalization, action clamping, filtering, watchdog timeout,
    and configurable control rate for real-time control applications.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize runtime inference engine.

        Args:
            config: Runtime configuration
        """
        self.config = config

        # Load model
        self.inference_engine = self._create_inference_engine()

        # Normalization
        self.state_normalizer = self._load_normalizer('state_normalizer')
        self.action_normalizer = self._load_normalizer('action_normalizer')

        # Action processing
        self.action_limits = np.array(config.get('action_limits', [-1.0, 1.0]))
        self.action_scale = config.get('action_scale', 1.0)

        # Low-pass filter for actions
        self.use_action_filter = config.get('use_action_filter', True)
        self.filter_cutoff = config.get('filter_cutoff_hz', 10.0)
        self.control_frequency = config.get('control_frequency', 100.0)
        self._init_action_filter()

        # Watchdog timer
        self.watchdog_timeout = config.get('watchdog_timeout_ms', 100.0) / 1000.0
        self.last_inference_time = 0.0
        self.watchdog_enabled = config.get('enable_watchdog', True)

        # Control rate management
        self.target_dt = 1.0 / self.control_frequency
        self.last_control_time = 0.0

        # Statistics
        self.inference_count = 0
        self.total_inference_time = 0.0
        self.max_inference_time = 0.0
        self.timeout_count = 0

        # Thread safety
        self._lock = threading.Lock()

        # Warmup
        self.inference_engine.warmup(config.get('warmup_iterations', 100))

        logger.info("Initialized RuntimeInferenceEngine")

    def _create_inference_engine(self) -> InferenceEngine:
        """Create appropriate inference engine based on model format."""
        model_config = self.config['model']
        model_path = model_config['path']
        model_format = model_config.get('format', 'auto')

        # Auto-detect format
        if model_format == 'auto':
            if model_path.endswith('.onnx'):
                model_format = 'onnx'
            elif model_path.endswith('.pt') or model_path.endswith('.pth'):
                model_format = 'torchscript'
            else:
                raise ValueError(f"Cannot auto-detect format for {model_path}")

        # Create engine
        if model_format == 'onnx':
            providers = model_config.get('providers', None)
            return ONNXInferenceEngine(model_path, providers)
        elif model_format == 'torchscript':
            device = model_config.get('device', 'auto')
            return TorchScriptInferenceEngine(model_path, device)
        else:
            raise ValueError(f"Unsupported model format: {model_format}")

    def _load_normalizer(self, normalizer_key: str) -> Optional[object]:
        """Load observation/action normalizer."""
        normalizer_config = self.config.get(normalizer_key)
        if not normalizer_config:
            return None

        normalizer_path = normalizer_config.get('path')
        if not normalizer_path or not Path(normalizer_path).exists():
            logger.warning(f"Normalizer not found: {normalizer_path}")
            return None

        try:
            # Load normalizer statistics
            with open(normalizer_path, 'r') as f:
                stats = json.load(f)

            normalizer = SimpleNormalizer(
                mean=np.array(stats['mean']),
                std=np.array(stats['std']),
                clip_range=normalizer_config.get('clip_range')
            )

            logger.info(f"Loaded {normalizer_key} from {normalizer_path}")
            return normalizer

        except Exception as e:
            logger.error(f"Failed to load {normalizer_key}: {e}")
            return None

    def _init_action_filter(self):
        """Initialize low-pass filter for action smoothing."""
        if not self.use_action_filter:
            self.action_filter = None
            return

        # Simple first-order low-pass filter
        dt = 1.0 / self.control_frequency
        tau = 1.0 / (2.0 * np.pi * self.filter_cutoff)
        self.filter_alpha = dt / (dt + tau)

        self.prev_filtered_action = None
        logger.info(f"Initialized action filter: cutoff={self.filter_cutoff}Hz, alpha={self.filter_alpha:.3f}")

    def predict(self, observation: np.ndarray,
               enable_safety: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Run inference with safety checks and filtering.

        Args:
            observation: Raw observation from robot
            enable_safety: Whether to apply safety checks

        Returns:
            Tuple of (processed_action, inference_info)
        """
        start_time = time.time()
        info = {}

        with self._lock:
            try:
                # Check watchdog timeout
                if enable_safety and self.watchdog_enabled:
                    time_since_last = start_time - self.last_inference_time
                    if self.last_inference_time > 0 and time_since_last > self.watchdog_timeout:
                        self.timeout_count += 1
                        logger.warning(f"Watchdog timeout: {time_since_last*1000:.1f}ms > {self.watchdog_timeout*1000:.1f}ms")
                        info['timeout'] = True

                # Normalize observation
                normalized_obs = self._normalize_observation(observation)

                # Run inference
                raw_action = self.inference_engine.predict(normalized_obs)

                # Post-process action
                processed_action = self._post_process_action(raw_action)

                # Update timing
                self.last_inference_time = start_time
                inference_time = time.time() - start_time
                self._update_statistics(inference_time)

                info.update({
                    'inference_time_ms': inference_time * 1000,
                    'normalized_obs_shape': normalized_obs.shape,
                    'raw_action_shape': raw_action.shape,
                    'timeout': False
                })

                return processed_action.flatten(), info

            except Exception as e:
                logger.error(f"Inference failed: {e}")
                # Return safe zero action
                safe_action = np.zeros(22)  # G1 has 22 DOF
                info['error'] = str(e)
                return safe_action, info

    def _normalize_observation(self, observation: np.ndarray) -> np.ndarray:
        """Normalize observation using loaded statistics."""
        if self.state_normalizer:
            return self.state_normalizer.normalize(observation)
        return observation

    def _post_process_action(self, raw_action: np.ndarray) -> np.ndarray:
        """Post-process raw network output."""
        action = raw_action.copy()

        # Denormalize if needed
        if self.action_normalizer:
            action = self.action_normalizer.denormalize(action)

        # Scale actions
        action = action * self.action_scale

        # Apply low-pass filter
        if self.use_action_filter:
            action = self._apply_action_filter(action)

        # Clamp to limits
        action = np.clip(action, self.action_limits[0], self.action_limits[1])

        return action

    def _apply_action_filter(self, action: np.ndarray) -> np.ndarray:
        """Apply low-pass filter to action."""
        if self.prev_filtered_action is None:
            self.prev_filtered_action = action.copy()
            return action

        # First-order low-pass filter: y[n] = α*x[n] + (1-α)*y[n-1]
        filtered_action = (self.filter_alpha * action +
                          (1.0 - self.filter_alpha) * self.prev_filtered_action)

        self.prev_filtered_action = filtered_action.copy()
        return filtered_action

    def _update_statistics(self, inference_time: float):
        """Update inference statistics."""
        self.inference_count += 1
        self.total_inference_time += inference_time
        self.max_inference_time = max(self.max_inference_time, inference_time)

    def get_statistics(self) -> Dict[str, float]:
        """Get inference performance statistics."""
        if self.inference_count == 0:
            return {}

        avg_time = self.total_inference_time / self.inference_count

        return {
            'inference_count': self.inference_count,
            'avg_inference_time_ms': avg_time * 1000,
            'max_inference_time_ms': self.max_inference_time * 1000,
            'avg_frequency_hz': 1.0 / avg_time if avg_time > 0 else 0.0,
            'timeout_count': self.timeout_count,
            'timeout_rate': self.timeout_count / self.inference_count
        }

    def reset_statistics(self):
        """Reset performance statistics."""
        with self._lock:
            self.inference_count = 0
            self.total_inference_time = 0.0
            self.max_inference_time = 0.0
            self.timeout_count = 0

    def set_control_frequency(self, frequency: float):
        """Update control frequency and reconfigure filter."""
        self.control_frequency = frequency
        self.target_dt = 1.0 / frequency

        # Reconfigure filter
        if self.use_action_filter:
            self._init_action_filter()

        logger.info(f"Updated control frequency to {frequency}Hz")

    def emergency_action(self) -> np.ndarray:
        """Return emergency safe action."""
        # Return zero action clamped to limits
        safe_action = np.zeros(22)  # G1 DOF
        return np.clip(safe_action, self.action_limits[0], self.action_limits[1])

    def health_check(self) -> Dict[str, bool]:
        """Perform health check of inference system."""
        health = {
            'inference_engine_ready': self.inference_engine is not None,
            'within_watchdog_timeout': True,
            'filter_initialized': True
        }

        # Check watchdog
        if self.watchdog_enabled and self.last_inference_time > 0:
            time_since_last = time.time() - self.last_inference_time
            health['within_watchdog_timeout'] = time_since_last < self.watchdog_timeout

        # Check filter
        if self.use_action_filter:
            health['filter_initialized'] = self.prev_filtered_action is not None

        return health

    def shutdown(self):
        """Shutdown inference engine and cleanup resources."""
        logger.info("Shutting down RuntimeInferenceEngine")

        with self._lock:
            # Print final statistics
            stats = self.get_statistics()
            if stats:
                logger.info(f"Final statistics: {stats}")

            # Cleanup resources
            self.inference_engine = None

        logger.info("RuntimeInferenceEngine shutdown complete")


class SimpleNormalizer:
    """Simple normalizer using mean and standard deviation."""

    def __init__(self, mean: np.ndarray, std: np.ndarray,
                 clip_range: Optional[Tuple[float, float]] = None):
        """
        Initialize normalizer.

        Args:
            mean: Mean values for normalization
            std: Standard deviation values for normalization
            clip_range: Optional clipping range after normalization
        """
        self.mean = mean
        self.std = std
        self.clip_range = clip_range

    def normalize(self, data: np.ndarray) -> np.ndarray:
        """Normalize data."""
        normalized = (data - self.mean) / (self.std + 1e-8)

        if self.clip_range:
            normalized = np.clip(normalized, self.clip_range[0], self.clip_range[1])

        return normalized

    def denormalize(self, normalized_data: np.ndarray) -> np.ndarray:
        """Denormalize data."""
        return normalized_data * self.std + self.mean


def create_inference_config_template() -> Dict[str, Any]:
    """Create template configuration for inference engine."""
    return {
        'model': {
            'path': 'path/to/model.onnx',
            'format': 'onnx',  # 'onnx', 'torchscript', 'auto'
            'providers': ['CUDAExecutionProvider', 'CPUExecutionProvider'],
            'device': 'auto'
        },
        'state_normalizer': {
            'path': 'path/to/state_stats.json',
            'clip_range': [-5.0, 5.0]
        },
        'action_normalizer': {
            'path': 'path/to/action_stats.json'
        },
        'action_limits': [-1.0, 1.0],
        'action_scale': 1.0,
        'use_action_filter': True,
        'filter_cutoff_hz': 10.0,
        'control_frequency': 100.0,
        'enable_watchdog': True,
        'watchdog_timeout_ms': 100.0,
        'warmup_iterations': 100
    }