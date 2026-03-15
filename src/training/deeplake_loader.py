"""
Deep Lake data loader for motion imitation training.
Loads data from Deep Lake datasets for training Sonic G1 motion models.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import deeplake

from ..core.config import Config
from ..core.logging import get_logger
from .data_loader import MotionSequenceDataset


logger = get_logger(__name__)


class DeepLakeMotionDataset(Dataset):
    """Dataset for motion imitation sequences from Deep Lake."""

    def __init__(
        self,
        deeplake_url: str,
        sequence_length: int = 32,
        overlap: float = 0.5,
        augment: bool = True,
        train_split: bool = True,
        split_ratio: float = 0.8,
    ):
        self.deeplake_url = deeplake_url
        self.sequence_length = sequence_length
        self.overlap = overlap
        self.augment = augment
        self.train_split = train_split
        self.split_ratio = split_ratio

        # Load Deep Lake dataset using v3.x API
        logger.info(f"Loading Deep Lake dataset from: {deeplake_url}")
        self.ds = deeplake.load(deeplake_url)

        # Get data split indices
        self.indices = self._get_split_indices()

        # Generate sequence windows
        self.sequence_windows = self._generate_windows()

        logger.info(f"Created Deep Lake dataset with {len(self.sequence_windows)} sequence windows")

    def _get_split_indices(self) -> List[int]:
        """Get indices for train/eval split."""
        total_samples = len(self.ds)
        split_idx = int(self.split_ratio * total_samples)

        if self.train_split:
            return list(range(split_idx))
        else:
            return list(range(split_idx, total_samples))

    def _generate_windows(self) -> List[Tuple[int, int, int]]:
        """Generate sliding windows over sequences."""
        windows = []

        # Group consecutive frames into sequences based on metadata
        current_sequence = []
        sequences = []

        for idx in self.indices:
            # Check if this frame belongs to the same sequence
            if len(current_sequence) == 0:
                current_sequence = [idx]
            else:
                # For now, assume consecutive indices are part of same sequence
                # In practice, you might check metadata/timestamps
                if idx == current_sequence[-1] + 1:
                    current_sequence.append(idx)
                else:
                    if len(current_sequence) >= self.sequence_length:
                        sequences.append(current_sequence)
                    current_sequence = [idx]

        # Add the last sequence if valid
        if len(current_sequence) >= self.sequence_length:
            sequences.append(current_sequence)

        # Generate sliding windows within each sequence
        for seq in sequences:
            seq_len = len(seq)
            step_size = max(1, int(self.sequence_length * (1 - self.overlap)))

            for start_idx in range(0, seq_len - self.sequence_length + 1, step_size):
                end_idx = start_idx + self.sequence_length
                # Store original indices from the sequence
                windows.append((seq, start_idx, end_idx))

        return windows

    def __len__(self) -> int:
        return len(self.sequence_windows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq_indices, start_idx, end_idx = self.sequence_windows[idx]

        # Get the actual Deep Lake indices for this window
        window_indices = seq_indices[start_idx:end_idx]

        # Extract data from Deep Lake
        observations = []
        actions = []

        for dl_idx in window_indices:
            # Extract frame data from Deep Lake
            frame_data = self.ds[dl_idx]

            # Extract observations and actions
            # Adjust these field names based on your Deep Lake dataset schema
            if hasattr(frame_data, 'observations') or 'observations' in frame_data:
                obs = np.array(frame_data['observations'])
            elif hasattr(frame_data, 'state') or 'state' in frame_data:
                obs = np.array(frame_data['state'])
            else:
                # Try to infer observations from available data
                obs = self._extract_observations(frame_data)

            if hasattr(frame_data, 'actions') or 'actions' in frame_data:
                action = np.array(frame_data['actions'])
            elif hasattr(frame_data, 'joints') or 'joints' in frame_data:
                action = np.array(frame_data['joints'])
            else:
                # Try to infer actions from available data
                action = self._extract_actions(frame_data)

            observations.append(obs)
            actions.append(action)

        # Stack into sequence tensors
        observations = np.stack(observations)
        actions = np.stack(actions)

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

    def _extract_observations(self, frame_data) -> np.ndarray:
        """Extract observation data from frame based on available fields."""
        # This is a fallback method to extract observations
        # You may need to customize this based on your data structure
        obs_fields = []

        # Look for common observation fields
        possible_obs_fields = [
            'joint_positions', 'joint_velocities', 'base_position',
            'base_orientation', 'base_linear_velocity', 'base_angular_velocity',
            'contact_forces', 'imu_data'
        ]

        for field in possible_obs_fields:
            if hasattr(frame_data, field) or field in frame_data:
                data = np.array(frame_data[field]).flatten()
                obs_fields.append(data)

        if obs_fields:
            return np.concatenate(obs_fields)
        else:
            logger.warning("Could not extract observations, using zeros")
            return np.zeros(64)  # Default size

    def _extract_actions(self, frame_data) -> np.ndarray:
        """Extract action data from frame based on available fields."""
        # This is a fallback method to extract actions
        # You may need to customize this based on your data structure

        # Look for common action fields
        possible_action_fields = [
            'joint_targets', 'joint_commands', 'motor_commands',
            'joint_positions'  # Sometimes actions are stored as positions
        ]

        for field in possible_action_fields:
            if hasattr(frame_data, field) or field in frame_data:
                return np.array(frame_data[field]).flatten()

        logger.warning("Could not extract actions, using zeros")
        return np.zeros(22)  # Default G1 action size

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


class DeepLakeMotionDataLoader:
    """Main data loader for motion imitation training using Deep Lake."""

    def __init__(self, config: Config, deeplake_url: str):
        self.config = config
        self.deeplake_url = deeplake_url
        self.data_config = config.env.task

        logger.info(f"Initializing Deep Lake data loader with URL: {deeplake_url}")

    def get_training_dataset(self) -> DeepLakeMotionDataset:
        """Get training dataset."""
        sequence_length = self.config.model.get("sequence_length", 32)

        return DeepLakeMotionDataset(
            deeplake_url=self.deeplake_url,
            sequence_length=sequence_length,
            overlap=0.5,
            augment=True,
            train_split=True,
            split_ratio=0.8,
        )

    def get_evaluation_dataset(self) -> DeepLakeMotionDataset:
        """Get evaluation dataset."""
        sequence_length = self.config.model.get("sequence_length", 32)

        return DeepLakeMotionDataset(
            deeplake_url=self.deeplake_url,
            sequence_length=sequence_length,
            overlap=0.5,
            augment=False,
            train_split=False,
            split_ratio=0.8,
        )

    def explore_dataset(self) -> Dict[str, Any]:
        """Explore the Deep Lake dataset structure and return info."""
        try:
            ds = deeplake.load(self.deeplake_url)

            info = {
                "dataset_size": len(ds),
                "tensors": list(ds.tensors.keys()) if hasattr(ds, 'tensors') else [],
                "schema": {}
            }

            # Get schema information
            for tensor_name in info["tensors"]:
                tensor = ds[tensor_name]
                info["schema"][tensor_name] = {
                    "shape": tensor.shape if hasattr(tensor, 'shape') else "Unknown",
                    "dtype": str(tensor.dtype) if hasattr(tensor, 'dtype') else "Unknown"
                }

            # Sample first item to understand structure
            if len(ds) > 0:
                sample = ds[0]
                info["sample_keys"] = list(sample.keys()) if hasattr(sample, 'keys') else []

            logger.info(f"Dataset exploration complete: {info}")
            return info

        except Exception as e:
            logger.error(f"Error exploring dataset: {e}")
            return {"error": str(e)}