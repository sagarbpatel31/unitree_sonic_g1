"""
Utility functions for the training stack.
Common utilities for device handling, seeding, timing, etc.
"""

import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch
import json


def set_seed(seed: int, deterministic: bool = False):
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value
        deterministic: Whether to use deterministic algorithms (slower but reproducible)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Set deterministic algorithms
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True


def get_device(device_str: str = "auto") -> torch.device:
    """
    Get torch device based on string specification.

    Args:
        device_str: Device specification ("auto", "cpu", "cuda", "cuda:0", etc.)

    Returns:
        torch.device object
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)

    # Validate device
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available, falling back to CPU")
        device = torch.device("cpu")

    return device


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count model parameters.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total": total_params,
        "trainable": trainable_params,
        "non_trainable": total_params - trainable_params,
    }


def save_json(data: Union[Dict, List], filepath: Union[str, Path]):
    """Save data to JSON file."""
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(filepath: Union[str, Path]) -> Union[Dict, List]:
    """Load data from JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def ensure_dir(dirpath: Union[str, Path]) -> Path:
    """Ensure directory exists, create if necessary."""
    dirpath = Path(dirpath)
    dirpath.mkdir(parents=True, exist_ok=True)
    return dirpath


def get_git_commit() -> Optional[str]:
    """Get current git commit hash."""
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except:
        pass
    return None


def get_system_info() -> Dict[str, Any]:
    """Get system information for reproducibility."""
    info = {
        "python_version": os.sys.version,
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "git_commit": get_git_commit(),
    }

    if torch.cuda.is_available():
        info.update({
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "gpu_count": torch.cuda.device_count(),
            "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        })

    return info


class Timer:
    """Simple timer utility."""

    def __init__(self, name: str = "Timer"):
        self.name = name
        self.start_time = None
        self.elapsed_times = []

    def start(self):
        """Start timing."""
        self.start_time = time.time()

    def stop(self) -> float:
        """Stop timing and return elapsed time."""
        if self.start_time is None:
            raise ValueError("Timer not started")

        elapsed = time.time() - self.start_time
        self.elapsed_times.append(elapsed)
        self.start_time = None
        return elapsed

    def get_average(self) -> float:
        """Get average elapsed time."""
        if not self.elapsed_times:
            return 0.0
        return sum(self.elapsed_times) / len(self.elapsed_times)

    def get_total(self) -> float:
        """Get total elapsed time."""
        return sum(self.elapsed_times)

    def reset(self):
        """Reset timer."""
        self.start_time = None
        self.elapsed_times.clear()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = self.stop()
        print(f"{self.name}: {elapsed:.4f}s")


class MovingAverage:
    """Exponential moving average."""

    def __init__(self, decay: float = 0.99):
        self.decay = decay
        self.value = None
        self.count = 0

    def update(self, value: float):
        """Update moving average."""
        if self.value is None:
            self.value = value
        else:
            self.value = self.decay * self.value + (1 - self.decay) * value
        self.count += 1

    def get(self) -> float:
        """Get current average value."""
        if self.value is None:
            return 0.0
        # Bias correction
        return self.value / (1 - self.decay ** self.count)

    def reset(self):
        """Reset average."""
        self.value = None
        self.count = 0


class AttrDict(dict):
    """Dictionary that supports attribute-style access."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'AttrDict' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'AttrDict' object has no attribute '{key}'")


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi]."""
    return ((angle + np.pi) % (2 * np.pi)) - np.pi


def angle_difference(a1: float, a2: float) -> float:
    """Compute shortest angular difference between two angles."""
    return normalize_angle(a2 - a1)


def quaternion_to_euler(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion [w, x, y, z] to Euler angles [roll, pitch, yaw]."""
    w, x, y, z = quat

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw])


def euler_to_quaternion(euler: np.ndarray) -> np.ndarray:
    """Convert Euler angles [roll, pitch, yaw] to quaternion [w, x, y, z]."""
    roll, pitch, yaw = euler

    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return np.array([w, x, y, z])


def safe_mean(values: List[float]) -> float:
    """Compute mean, handling empty lists."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def interpolate_poses(pose1: np.ndarray, pose2: np.ndarray, alpha: float) -> np.ndarray:
    """
    Interpolate between two poses (positions + quaternions).

    Args:
        pose1: [x, y, z, qw, qx, qy, qz]
        pose2: [x, y, z, qw, qx, qy, qz]
        alpha: Interpolation factor [0, 1]

    Returns:
        Interpolated pose
    """
    # Linear interpolation for position
    pos = (1 - alpha) * pose1[:3] + alpha * pose2[:3]

    # SLERP for quaternion
    q1, q2 = pose1[3:], pose2[3:]
    dot = np.dot(q1, q2)

    # Handle negative dot product
    if dot < 0:
        q2 = -q2
        dot = -dot

    # Use linear interpolation for very close quaternions
    if dot > 0.9995:
        quat = (1 - alpha) * q1 + alpha * q2
        quat = quat / np.linalg.norm(quat)
    else:
        # SLERP
        theta = np.arccos(np.abs(dot))
        sin_theta = np.sin(theta)
        w1 = np.sin((1 - alpha) * theta) / sin_theta
        w2 = np.sin(alpha * theta) / sin_theta
        quat = w1 * q1 + w2 * q2

    return np.concatenate([pos, quat])