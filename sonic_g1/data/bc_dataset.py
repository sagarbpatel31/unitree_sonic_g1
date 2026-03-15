"""
Dataset and data loading utilities for behavior cloning training.

This module provides efficient data loading for state-action pairs extracted
from retargeted motion clips for supervised learning.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import logging
from omegaconf import DictConfig
import pickle
import random

from src.data import load_g1_trajectory_from_npz, MotionNormalizer
from .state_action_extractor import StateActionExtractor

logger = logging.getLogger(__name__)


class G1MotionDataset(Dataset):
    """
    Dataset for behavior cloning from G1 motion trajectories.

    Loads retargeted motion clips and extracts state-action pairs for
    supervised learning of the policy.
    """

    def __init__(self,
                 trajectory_files: List[str],
                 extractor: StateActionExtractor,
                 config: DictConfig,
                 state_normalizer: Optional[MotionNormalizer] = None,
                 action_normalizer: Optional[MotionNormalizer] = None):
        """
        Initialize BC dataset.

        Args:
            trajectory_files: List of paths to NPZ trajectory files
            extractor: State-action extractor
            config: Dataset configuration
            state_normalizer: Optional state normalizer
            action_normalizer: Optional action normalizer
        """
        self.trajectory_files = trajectory_files
        self.extractor = extractor
        self.config = config
        self.state_normalizer = state_normalizer
        self.action_normalizer = action_normalizer

        # Dataset configuration
        self.sequence_length = config.get('sequence_length', 1)
        self.skip_frames = config.get('skip_frames', 1)
        self.min_trajectory_length = config.get('min_trajectory_length', 10)
        self.max_trajectory_length = config.get('max_trajectory_length', 1000)
        self.augment_data = config.get('augment_data', False)
        self.filter_quality = config.get('filter_quality', False)
        self.quality_threshold = config.get('quality_threshold', 0.5)

        # Load and process data
        self.data_samples = []
        self._load_data()

        logger.info(f"Created BC dataset with {len(self.data_samples)} samples "
                   f"from {len(trajectory_files)} trajectory files")

    def _load_data(self):
        """Load and process all trajectory data."""
        total_trajectories = 0
        successful_trajectories = 0
        total_samples = 0

        for traj_file in self.trajectory_files:
            try:
                # Load trajectory
                trajectory = load_g1_trajectory_from_npz(traj_file)
                total_trajectories += 1

                # Filter by quality if enabled
                if self.filter_quality:
                    quality_score = trajectory.metadata.get('overall_quality_score', 1.0)
                    if quality_score < self.quality_threshold:
                        logger.debug(f"Skipping low-quality trajectory: {traj_file} (score: {quality_score:.3f})")
                        continue

                # Filter by length
                traj_length = len(trajectory.timestamps)
                if (traj_length < self.min_trajectory_length or
                    traj_length > self.max_trajectory_length):
                    logger.debug(f"Skipping trajectory due to length: {traj_file} (length: {traj_length})")
                    continue

                # Extract state-action pairs
                samples = self._extract_samples_from_trajectory(trajectory, traj_file)
                self.data_samples.extend(samples)
                total_samples += len(samples)
                successful_trajectories += 1

            except Exception as e:
                logger.warning(f"Failed to load trajectory {traj_file}: {e}")
                continue

        logger.info(f"Loaded {successful_trajectories}/{total_trajectories} trajectories, "
                   f"extracted {total_samples} samples")

        if len(self.data_samples) == 0:
            raise ValueError("No valid samples found in dataset")

    def _extract_samples_from_trajectory(self,
                                       trajectory,
                                       trajectory_file: str) -> List[Dict[str, Any]]:
        """Extract training samples from a single trajectory."""
        samples = []

        # Extract states and actions using the extractor
        states, actions, metadata = self.extractor.extract_from_trajectory(trajectory)

        # Apply frame skipping
        if self.skip_frames > 1:
            indices = range(0, len(states), self.skip_frames)
            states = states[indices]
            actions = actions[indices]

        # Create samples with sequence support
        for i in range(len(states) - self.sequence_length + 1):
            sample = {
                'states': states[i:i+self.sequence_length],
                'actions': actions[i:i+self.sequence_length],
                'trajectory_file': trajectory_file,
                'sample_index': i,
                'metadata': metadata
            }

            # Add state derivatives if available
            if i < len(states) - 1:
                state_derivative = states[i+1] - states[i]
                sample['state_derivatives'] = state_derivative

            samples.append(sample)

        return samples

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.data_samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample."""
        sample = self.data_samples[idx]

        # Get states and actions
        states = sample['states'].copy()
        actions = sample['actions'].copy()

        # Apply data augmentation if enabled
        if self.augment_data and random.random() < 0.5:
            states, actions = self._augment_sample(states, actions)

        # Normalize if normalizers are available
        if self.state_normalizer:
            if self.sequence_length > 1:
                # Normalize each timestep
                normalized_states = []
                for t in range(self.sequence_length):
                    norm_state = self.state_normalizer.normalize({'states': states[t:t+1]})['states']
                    normalized_states.append(norm_state)
                states = np.concatenate(normalized_states, axis=0)
            else:
                states = self.state_normalizer.normalize({'states': states})['states']

        if self.action_normalizer:
            if self.sequence_length > 1:
                # Normalize each timestep
                normalized_actions = []
                for t in range(self.sequence_length):
                    norm_action = self.action_normalizer.normalize({'actions': actions[t:t+1]})['actions']
                    normalized_actions.append(norm_action)
                actions = np.concatenate(normalized_actions, axis=0)
            else:
                actions = self.action_normalizer.normalize({'actions': actions})['actions']

        # Convert to tensors
        result = {
            'states': torch.FloatTensor(states),
            'actions': torch.FloatTensor(actions)
        }

        # Add state derivatives if available
        if 'state_derivatives' in sample:
            state_derivatives = sample['state_derivatives'].copy()
            if self.state_normalizer:
                # Use the same normalization as states (derivatives should have similar scale)
                state_derivatives = self.state_normalizer.normalize({'states': state_derivatives.reshape(1, -1)})['states'].flatten()
            result['state_derivatives'] = torch.FloatTensor(state_derivatives)

        # Handle sequence length
        if self.sequence_length == 1:
            # Squeeze sequence dimension for single timestep
            result['states'] = result['states'].squeeze(0)
            result['actions'] = result['actions'].squeeze(0)

        return result

    def _augment_sample(self,
                       states: np.ndarray,
                       actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply data augmentation to a sample."""
        # Simple noise augmentation
        state_noise_std = self.config.get('state_noise_std', 0.01)
        action_noise_std = self.config.get('action_noise_std', 0.005)

        if state_noise_std > 0:
            states = states + np.random.normal(0, state_noise_std, states.shape)

        if action_noise_std > 0:
            actions = actions + np.random.normal(0, action_noise_std, actions.shape)

        return states, actions

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        if not self.data_samples:
            return {}

        # Collect all states and actions
        all_states = []
        all_actions = []

        for sample in self.data_samples[:1000]:  # Sample subset for efficiency
            states = sample['states']
            actions = sample['actions']

            if self.sequence_length > 1:
                states = states.reshape(-1, states.shape[-1])
                actions = actions.reshape(-1, actions.shape[-1])

            all_states.append(states)
            all_actions.append(actions)

        all_states = np.concatenate(all_states, axis=0)
        all_actions = np.concatenate(all_actions, axis=0)

        return {
            'num_samples': len(self.data_samples),
            'num_trajectories': len(self.trajectory_files),
            'state_dim': all_states.shape[1],
            'action_dim': all_actions.shape[1],
            'state_mean': np.mean(all_states, axis=0),
            'state_std': np.std(all_states, axis=0),
            'action_mean': np.mean(all_actions, axis=0),
            'action_std': np.std(all_actions, axis=0),
            'state_range': {
                'min': np.min(all_states, axis=0),
                'max': np.max(all_states, axis=0)
            },
            'action_range': {
                'min': np.min(all_actions, axis=0),
                'max': np.max(all_actions, axis=0)
            }
        }


class SequenceDataset(G1MotionDataset):
    """
    Dataset variant that returns sequences of states/actions for RNN training.
    """

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sequence sample."""
        sample = self.data_samples[idx]

        states = sample['states'].copy()
        actions = sample['actions'].copy()

        # Apply augmentation
        if self.augment_data and random.random() < 0.5:
            # Apply same noise to entire sequence
            state_noise_std = self.config.get('state_noise_std', 0.01)
            action_noise_std = self.config.get('action_noise_std', 0.005)

            if state_noise_std > 0:
                noise_shape = (1,) + states.shape[1:]  # Broadcast across sequence
                states = states + np.random.normal(0, state_noise_std, noise_shape)

            if action_noise_std > 0:
                noise_shape = (1,) + actions.shape[1:]
                actions = actions + np.random.normal(0, action_noise_std, noise_shape)

        # Normalize sequences
        if self.state_normalizer:
            normalized_states = []
            for t in range(len(states)):
                norm_state = self.state_normalizer.normalize({'states': states[t:t+1]})['states']
                normalized_states.append(norm_state)
            states = np.concatenate(normalized_states, axis=0)

        if self.action_normalizer:
            normalized_actions = []
            for t in range(len(actions)):
                norm_action = self.action_normalizer.normalize({'actions': actions[t:t+1]})['actions']
                normalized_actions.append(norm_action)
            actions = np.concatenate(normalized_actions, axis=0)

        return {
            'states': torch.FloatTensor(states),
            'actions': torch.FloatTensor(actions)
        }


def create_bc_dataloaders(config: DictConfig) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
    """
    Create training and validation data loaders for behavior cloning.

    Args:
        config: Configuration containing data and training parameters

    Returns:
        Tuple of (train_loader, val_loader, dataset_info)
    """
    # Find trajectory files
    data_dir = Path(config.data.trajectory_dir)
    trajectory_files = list(data_dir.glob("**/*.npz"))

    if not trajectory_files:
        raise ValueError(f"No trajectory files found in {data_dir}")

    trajectory_files = [str(f) for f in trajectory_files]
    logger.info(f"Found {len(trajectory_files)} trajectory files")

    # Load normalizers if specified
    state_normalizer = None
    action_normalizer = None

    if config.data.get('state_normalizer_path'):
        try:
            state_normalizer = MotionNormalizer()
            state_normalizer.load_statistics(config.data.state_normalizer_path)
            logger.info("Loaded state normalizer")
        except Exception as e:
            logger.warning(f"Failed to load state normalizer: {e}")

    if config.data.get('action_normalizer_path'):
        try:
            action_normalizer = MotionNormalizer()
            action_normalizer.load_statistics(config.data.action_normalizer_path)
            logger.info("Loaded action normalizer")
        except Exception as e:
            logger.warning(f"Failed to load action normalizer: {e}")

    # Create state-action extractor
    extractor = StateActionExtractor(config.data.extraction)

    # Split files into train/val
    random.shuffle(trajectory_files)
    val_split = config.data.get('val_split', 0.2)
    val_size = int(len(trajectory_files) * val_split)
    val_files = trajectory_files[:val_size]
    train_files = trajectory_files[val_size:]

    logger.info(f"Train files: {len(train_files)}, Val files: {len(val_files)}")

    # Create datasets
    dataset_class = SequenceDataset if config.data.get('use_sequences', False) else G1MotionDataset

    train_dataset = dataset_class(
        trajectory_files=train_files,
        extractor=extractor,
        config=config.data,
        state_normalizer=state_normalizer,
        action_normalizer=action_normalizer
    )

    val_dataset = dataset_class(
        trajectory_files=val_files,
        extractor=extractor,
        config=config.data,
        state_normalizer=state_normalizer,
        action_normalizer=action_normalizer
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.get('num_workers', 4),
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.get('num_workers', 4),
        pin_memory=True
    )

    # Collect dataset info
    train_stats = train_dataset.get_statistics()
    val_stats = val_dataset.get_statistics()

    dataset_info = {
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'train_files': len(train_files),
        'val_files': len(val_files),
        'state_dim': train_stats.get('state_dim', 0),
        'action_dim': train_stats.get('action_dim', 0),
        'train_stats': train_stats,
        'val_stats': val_stats
    }

    return train_loader, val_loader, dataset_info


def save_dataset_statistics(dataset: G1MotionDataset, save_path: str):
    """Save dataset statistics for later analysis."""
    stats = dataset.get_statistics()

    with open(save_path, 'wb') as f:
        pickle.dump(stats, f)

    logger.info(f"Saved dataset statistics to {save_path}")


def load_dataset_statistics(load_path: str) -> Dict[str, Any]:
    """Load previously saved dataset statistics."""
    with open(load_path, 'rb') as f:
        stats = pickle.load(f)

    logger.info(f"Loaded dataset statistics from {load_path}")
    return stats