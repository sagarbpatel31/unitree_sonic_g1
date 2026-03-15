"""
Data loading utilities for motion imitation training.
Handles loading and preprocessing of motion capture data.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json

from ..core.config import Config
from ..core.logging import get_logger


logger = get_logger(__name__)


class MotionSequenceDataset(Dataset):
    """Dataset for motion imitation sequences."""

    def __init__(
        self,
        sequences: List[Dict[str, np.ndarray]],
        sequence_length: int = 32,
        overlap: float = 0.5,
        augment: bool = True,
    ):
        self.sequences = sequences
        self.sequence_length = sequence_length
        self.overlap = overlap
        self.augment = augment

        # Generate sequence windows
        self.sequence_windows = self._generate_windows()

        logger.info(f"Created dataset with {len(self.sequence_windows)} sequence windows")

    def _generate_windows(self) -> List[Tuple[int, int, int]]:
        """Generate sliding windows over sequences."""
        windows = []

        for seq_idx, sequence in enumerate(self.sequences):
            seq_len = len(sequence["observations"])

            if seq_len < self.sequence_length:
                continue

            # Calculate step size based on overlap
            step_size = max(1, int(self.sequence_length * (1 - self.overlap)))

            # Generate windows
            for start_idx in range(0, seq_len - self.sequence_length + 1, step_size):
                end_idx = start_idx + self.sequence_length
                windows.append((seq_idx, start_idx, end_idx))

        return windows

    def __len__(self) -> int:
        return len(self.sequence_windows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq_idx, start_idx, end_idx = self.sequence_windows[idx]
        sequence = self.sequences[seq_idx]

        # Extract sequence window
        observations = sequence["observations"][start_idx:end_idx]
        actions = sequence["actions"][start_idx:end_idx]

        # Convert to tensors
        obs_tensor = torch.FloatTensor(observations)
        action_tensor = torch.FloatTensor(actions)

        # Data augmentation
        if self.augment and np.random.random() < 0.5:
            obs_tensor, action_tensor = self._augment_sequence(obs_tensor, action_tensor)

        return {
            "observations": obs_tensor,
            "actions": action_tensor,
        }

    def _augment_sequence(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply data augmentation to sequence."""
        # Add noise
        if np.random.random() < 0.3:
            noise_scale = 0.01
            obs_noise = torch.normal(0, noise_scale, observations.shape)
            observations = observations + obs_noise

        # Time scaling (speed up/down)
        if np.random.random() < 0.2:
            # Simple temporal subsampling/upsampling
            scale_factor = np.random.uniform(0.8, 1.2)
            if scale_factor != 1.0:
                # This is a simplified implementation
                # In practice, would need more sophisticated temporal scaling
                pass

        return observations, actions


class MotionDataLoader:
    """Main data loader for motion imitation training."""

    def __init__(self, config: Config):
        self.config = config
        self.data_config = config.env.task

        # Data paths
        self.data_path = Path(self.data_config.reference_data_path)

        # Load motion data
        self.motion_sequences = self._load_motion_data()

        # Split data
        self.train_sequences, self.eval_sequences = self._split_data()

    def _load_motion_data(self) -> List[Dict[str, np.ndarray]]:
        """Load motion data from files."""
        if not self.data_path.exists():
            logger.warning(f"Data path does not exist: {self.data_path}")
            logger.info("Using synthetic data for demonstration")
            return self._generate_synthetic_data()

        motion_sequences = []

        # Look for motion files
        motion_files = list(self.data_path.glob("*.npz")) + list(self.data_path.glob("*.json"))

        if not motion_files:
            logger.warning("No motion files found, using synthetic data")
            return self._generate_synthetic_data()

        for motion_file in motion_files:
            try:
                sequence = self._load_motion_file(motion_file)
                if sequence is not None:
                    motion_sequences.append(sequence)
            except Exception as e:
                logger.warning(f"Failed to load {motion_file}: {e}")

        logger.info(f"Loaded {len(motion_sequences)} motion sequences")
        return motion_sequences

    def _load_motion_file(self, file_path: Path) -> Optional[Dict[str, np.ndarray]]:
        """Load a single motion file."""
        if file_path.suffix == ".npz":
            data = np.load(file_path)
            return {
                "observations": data["observations"],
                "actions": data["actions"],
                "metadata": data.get("metadata", {}),
            }

        elif file_path.suffix == ".json":
            with open(file_path, "r") as f:
                data = json.load(f)

            return {
                "observations": np.array(data["observations"]),
                "actions": np.array(data["actions"]),
                "metadata": data.get("metadata", {}),
            }

        return None

    def _generate_synthetic_data(self) -> List[Dict[str, np.ndarray]]:
        """Generate synthetic motion data for demonstration."""
        sequences = []

        # Generate a few example sequences
        for seq_idx in range(10):
            sequence_length = 1000
            obs_dim = 64  # Placeholder observation dimension
            action_dim = 22  # Placeholder action dimension

            # Generate synthetic walking motion
            t = np.linspace(0, 10, sequence_length)

            # Create observations (simplified)
            observations = np.zeros((sequence_length, obs_dim))
            actions = np.zeros((sequence_length, action_dim))

            # Simple walking pattern
            for i in range(sequence_length):
                # Joint positions (simplified walking)
                phase = (t[i] * 2) % (2 * np.pi)
                hip_angle = 0.3 * np.sin(phase)
                knee_angle = max(0, 0.6 * np.sin(phase))

                # Fill in some observations and actions
                if action_dim > 0:
                    actions[i, 0] = hip_angle  # Left hip
                    actions[i, 6] = -hip_angle  # Right hip
                    actions[i, 3] = knee_angle  # Left knee
                    actions[i, 9] = knee_angle  # Right knee

                # Add some observation data
                if obs_dim > action_dim:
                    observations[i, :action_dim] = actions[i]  # Include current actions
                    observations[i, action_dim:action_dim+4] = [1, 0, 0, 0]  # Base orientation
                    observations[i, action_dim+4:action_dim+7] = [0, 0, 0.75]  # Base position

            sequences.append({
                "observations": observations,
                "actions": actions,
                "metadata": {"sequence_id": seq_idx, "type": "synthetic_walk"},
            })

        return sequences

    def _split_data(self) -> Tuple[List[Dict[str, np.ndarray]], List[Dict[str, np.ndarray]]]:
        """Split data into training and evaluation sets."""
        if not self.motion_sequences:
            return [], []

        # Simple 80/20 split
        split_idx = int(0.8 * len(self.motion_sequences))

        train_sequences = self.motion_sequences[:split_idx]
        eval_sequences = self.motion_sequences[split_idx:]

        logger.info(f"Data split: {len(train_sequences)} train, {len(eval_sequences)} eval")

        return train_sequences, eval_sequences

    def get_training_dataset(self) -> MotionSequenceDataset:
        """Get training dataset."""
        sequence_length = self.config.model.get("sequence_length", 32)

        return MotionSequenceDataset(
            sequences=self.train_sequences,
            sequence_length=sequence_length,
            overlap=0.5,
            augment=True,
        )

    def get_evaluation_dataset(self) -> Optional[MotionSequenceDataset]:
        """Get evaluation dataset."""
        if not self.eval_sequences:
            return None

        sequence_length = self.config.model.get("sequence_length", 32)

        return MotionSequenceDataset(
            sequences=self.eval_sequences,
            sequence_length=sequence_length,
            overlap=0.5,
            augment=False,
        )