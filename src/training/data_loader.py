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
import pandas as pd

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


class CSVMotionDataset(Dataset):
    """Dataset for CSV-based motion imitation data (for Deep Lake exports)."""

    def __init__(
        self,
        csv_path: str,
        sequence_length: int = 32,
        overlap: float = 0.5,
        augment: bool = True,
        train_split: bool = True,
        split_ratio: float = 0.8,
    ):
        self.csv_path = Path(csv_path)
        self.sequence_length = sequence_length
        self.overlap = overlap
        self.augment = augment
        self.train_split = train_split
        self.split_ratio = split_ratio

        # Load CSV data
        logger.info(f"Loading CSV data from: {csv_path}")
        self.df = pd.read_csv(csv_path)
        logger.info(f"Loaded CSV with shape: {self.df.shape}")

        # Split data
        self._split_data()

        # Prepare features
        self._prepare_data()

        # Generate sequence windows
        self.sequence_windows = self._generate_windows()

        logger.info(f"Created CSV dataset with {len(self.sequence_windows)} sequence windows")

    def _split_data(self):
        """Split data into train/validation sets."""
        split_idx = int(self.split_ratio * len(self.df))

        if self.train_split:
            self.df = self.df.iloc[:split_idx]
            logger.info(f"Using training split: {len(self.df)} rows")
        else:
            self.df = self.df.iloc[split_idx:]
            logger.info(f"Using validation split: {len(self.df)} rows")

    def _prepare_data(self):
        """Extract observations and actions from CSV data."""
        # Action columns (those starting with 'act_')
        action_cols = [col for col in self.df.columns if col.startswith('act_')]

        # Observation columns (remaining numeric columns, excluding _id)
        obs_cols = [col for col in self.df.columns
                   if col not in action_cols and col != '_id' and
                   self.df[col].dtype in ['float64', 'int64', 'float32', 'int32']]

        logger.info(f"Found {len(action_cols)} action columns")
        logger.info(f"Found {len(obs_cols)} observation columns")

        # Extract data
        if action_cols:
            self.actions = self.df[action_cols].values.astype(np.float32)
        else:
            logger.warning("No action columns found, using random data")
            self.actions = np.random.randn(len(self.df), 22).astype(np.float32)

        if obs_cols:
            self.observations = self.df[obs_cols].values.astype(np.float32)
        else:
            logger.warning("No observation columns found, using action data")
            self.observations = self.actions

        logger.info(f"Actions shape: {self.actions.shape}")
        logger.info(f"Observations shape: {self.observations.shape}")

    def _generate_windows(self) -> List[Tuple[int, int]]:
        """Generate sliding windows over the data."""
        windows = []

        # Calculate step size based on overlap
        step_size = max(1, int(self.sequence_length * (1 - self.overlap)))

        # Generate windows
        for start_idx in range(0, len(self.df) - self.sequence_length + 1, step_size):
            end_idx = start_idx + self.sequence_length
            windows.append((start_idx, end_idx))

        return windows

    def __len__(self) -> int:
        return len(self.sequence_windows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start_idx, end_idx = self.sequence_windows[idx]

        # Extract sequence window
        observations = self.observations[start_idx:end_idx]
        actions = self.actions[start_idx:end_idx]

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

        return observations, actions


class MotionDataLoader:
    """Main data loader for motion imitation training."""

    def __init__(self, config: Config):
        self.config = config
        self.data_config = config.env.task

        # Check if using CSV data source (for Deep Lake)
        self.use_csv = getattr(config.env.task, 'use_csv_data', False)

        if self.use_csv:
            self.csv_path = getattr(config.env.task, 'csv_data_path', 'data/lightwheel_bevorg_frames.csv')
            logger.info(f"Using CSV data source: {self.csv_path}")
        else:
            # Data paths for traditional motion files
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

    def get_training_dataset(self) -> Dataset:
        """Get training dataset."""
        sequence_length = self.config.model.get("sequence_length", 32)

        if self.use_csv:
            return CSVMotionDataset(
                csv_path=self.csv_path,
                sequence_length=sequence_length,
                overlap=0.5,
                augment=True,
                train_split=True,
            )
        else:
            return MotionSequenceDataset(
                sequences=self.train_sequences,
                sequence_length=sequence_length,
                overlap=0.5,
                augment=True,
            )

    def get_evaluation_dataset(self) -> Optional[Dataset]:
        """Get evaluation dataset."""
        sequence_length = self.config.model.get("sequence_length", 32)

        if self.use_csv:
            return CSVMotionDataset(
                csv_path=self.csv_path,
                sequence_length=sequence_length,
                overlap=0.5,
                augment=False,
                train_split=False,
            )
        else:
            if not self.eval_sequences:
                return None

            return MotionSequenceDataset(
                sequences=self.eval_sequences,
                sequence_length=sequence_length,
                overlap=0.5,
                augment=False,
            )