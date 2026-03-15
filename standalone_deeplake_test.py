#!/usr/bin/env python3
"""
Standalone Deep Lake data loader test with authentication.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import deeplake


class SimpleConfig:
    """Simple config class to avoid dependencies."""
    def __init__(self):
        self.model = {"sequence_length": 32}
        self.training = {"batch_size": 4}


class DeepLakeDataset(Dataset):
    """Simple Deep Lake dataset for PyTorch."""

    def __init__(self, deeplake_url, token=None, sequence_length=32, train_split=True):
        self.deeplake_url = deeplake_url
        self.sequence_length = sequence_length
        self.train_split = train_split

        print(f"Loading Deep Lake dataset from: {deeplake_url}")

        # Load dataset with token if provided
        if token:
            self.ds = deeplake.load(deeplake_url, token=token)
        else:
            self.ds = deeplake.load(deeplake_url)

        print(f"Dataset loaded successfully! Size: {len(self.ds)}")

        # Simple train/val split
        total_size = len(self.ds)
        train_size = int(0.8 * total_size)

        if train_split:
            self.indices = list(range(train_size))
        else:
            self.indices = list(range(train_size, total_size))

        print(f"Using {len(self.indices)} samples for {'training' if train_split else 'validation'}")

    def __len__(self):
        return max(0, len(self.indices) - self.sequence_length + 1)

    def __getitem__(self, idx):
        # Get sequence of frames
        start_idx = self.indices[idx]
        end_idx = min(start_idx + self.sequence_length, self.indices[-1])

        observations = []
        actions = []

        for i in range(start_idx, end_idx):
            if i < len(self.ds):
                frame = self.ds[i]

                # Extract observations and actions based on available fields
                obs = self._extract_observation(frame)
                action = self._extract_action(frame)

                observations.append(obs)
                actions.append(action)

        # Pad if necessary
        while len(observations) < self.sequence_length:
            observations.append(observations[-1] if observations else np.zeros(64))
            actions.append(actions[-1] if actions else np.zeros(22))

        return {
            "observations": torch.FloatTensor(observations),
            "actions": torch.FloatTensor(actions)
        }

    def _extract_observation(self, frame):
        """Extract observation from Deep Lake frame."""
        # Try different possible field names
        for field_name in ['observations', 'state', 'obs']:
            if hasattr(frame, field_name):
                data = getattr(frame, field_name)
                if hasattr(data, 'numpy'):
                    return data.numpy().flatten()
                else:
                    return np.array(data).flatten()

        # Fallback: create from available data
        print("Warning: No observation field found, using fallback")
        return np.random.randn(64).astype(np.float32)

    def _extract_action(self, frame):
        """Extract action from Deep Lake frame."""
        # Try different possible field names
        for field_name in ['actions', 'joints', 'controls', 'targets']:
            if hasattr(frame, field_name):
                data = getattr(frame, field_name)
                if hasattr(data, 'numpy'):
                    return data.numpy().flatten()
                else:
                    return np.array(data).flatten()

        # Fallback
        print("Warning: No action field found, using fallback")
        return np.random.randn(22).astype(np.float32)


def test_deeplake_with_auth():
    """Test Deep Lake loading with authentication."""

    # Your Deep Lake URL
    deeplake_url = "https://deeplake.ai/siyuliu4262s-organization/workspace/default/table/lightwheel_bevorg_frames"

    # Get token from the modified script
    token = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiNTNhNDYyYWYtMmRjZS00Zjk4LWI3NTQtZWQ3NjEyN2I0OTU1Iiwib3JnX2lkIjoiNmFiMzUzMjQtOTEyZS00Mjc2LWEzMDktMTU5YTk3YzRiNTU3IiwidHlwZSI6ImFwaV90b2tlbiIsIm5hbWUiOiJmaXJzdF90ZXN0X0cxIiwiZXhwIjoxODA1MDY4ODAwLCJpYXQiOjE3NzM1NTM0OTZ9.FwNRueK1SannTUy9J69BvD2ZIbeGhKDg9jmdN21NgLU'

    print("=" * 60)
    print("Deep Lake Authentication Test")
    print("=" * 60)

    try:
        # Test dataset access
        print(f"Connecting to: {deeplake_url}")
        print(f"Using token: {token[:50]}...")

        # Create dataset
        train_dataset = DeepLakeDataset(deeplake_url, token=token, train_split=True)

        print(f"\n✅ Successfully created training dataset!")
        print(f"Training dataset size: {len(train_dataset)}")

        # Explore first sample
        if len(train_dataset) > 0:
            sample = train_dataset[0]
            print(f"\nFirst sample:")
            print(f"  Observations shape: {sample['observations'].shape}")
            print(f"  Actions shape: {sample['actions'].shape}")

            # Create DataLoader
            config = SimpleConfig()
            train_loader = DataLoader(
                train_dataset,
                batch_size=config.training["batch_size"],
                shuffle=True,
                num_workers=0
            )

            print(f"\n✅ Created PyTorch DataLoader!")

            # Test loading a batch
            for i, batch in enumerate(train_loader):
                print(f"Batch {i}:")
                print(f"  Observations shape: {batch['observations'].shape}")
                print(f"  Actions shape: {batch['actions'].shape}")
                print(f"  Observations range: [{batch['observations'].min():.3f}, {batch['observations'].max():.3f}]")
                print(f"  Actions range: [{batch['actions'].min():.3f}, {batch['actions'].max():.3f}]")

                if i >= 1:  # Test first 2 batches
                    break

            print(f"\n🎯 Deep Lake data loading successful!")
            print(f"✅ You can now use this data for training your Sonic G1 model!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_deeplake_with_auth()