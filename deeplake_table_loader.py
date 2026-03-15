#!/usr/bin/env python3
"""
Deep Lake table data loader for Sonic G1 training.
Works with Deep Lake table format data.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import deeplake
import pandas as pd


class DeepLakeTableDataset(Dataset):
    """PyTorch dataset for Deep Lake table data."""

    def __init__(self, table_url, token=None, sequence_length=32, train_split=True):
        self.table_url = table_url
        self.token = token
        self.sequence_length = sequence_length
        self.train_split = train_split

        print(f"Loading Deep Lake table from: {table_url}")

        # Load the table data
        if token:
            self.ds = deeplake.load(table_url, token=token)
        else:
            self.ds = deeplake.load(table_url)

        print(f"✅ Table loaded successfully! Total rows: {len(self.ds)}")

        # Convert to pandas for easier manipulation
        print("Converting to pandas DataFrame...")
        self.df = self.ds.pandas()
        print(f"DataFrame shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")

        # Split data for train/validation
        split_idx = int(0.8 * len(self.df))
        if train_split:
            self.df = self.df.iloc[:split_idx]
            print(f"Using training split: {len(self.df)} rows")
        else:
            self.df = self.df.iloc[split_idx:]
            print(f"Using validation split: {len(self.df)} rows")

        # Extract and organize data
        self._prepare_data()

    def _prepare_data(self):
        """Extract observations and actions from the dataframe."""

        # Identify action columns (those starting with 'act_')
        action_cols = [col for col in self.df.columns if col.startswith('act_')]

        # Identify state/observation columns
        obs_cols = [col for col in self.df.columns if not col.startswith('act_') and col != '_id']

        print(f"Found {len(action_cols)} action columns: {action_cols[:5]}...")
        print(f"Found {len(obs_cols)} observation columns: {obs_cols[:5]}...")

        # Extract numeric data
        self.actions = self.df[action_cols].values.astype(np.float32)
        self.observations = self.df[obs_cols].values.astype(np.float32) if obs_cols else self.actions  # Fallback

        print(f"Actions shape: {self.actions.shape}")
        print(f"Observations shape: {self.observations.shape}")

    def __len__(self):
        return max(0, len(self.df) - self.sequence_length + 1)

    def __getitem__(self, idx):
        # Get sequence of consecutive frames
        end_idx = idx + self.sequence_length

        obs_sequence = self.observations[idx:end_idx]
        action_sequence = self.actions[idx:end_idx]

        # Convert to tensors
        return {
            "observations": torch.FloatTensor(obs_sequence),
            "actions": torch.FloatTensor(action_sequence)
        }


def test_table_loading():
    """Test loading the Deep Lake table data."""

    # The table URL from your screenshot
    table_url = "https://deeplake.ai/siyuliu4262s-organization/workspace/default/table/lightwheel_bevorg_frames"

    # Your authentication token
    token = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiNTNhNDYyYWYtMmRjZS00Zjk4LWI3NTQtZWQ3NjEyN2I0OTU1Iiwib3JnX2lkIjoiNmFiMzUzMjQtOTEyZS00Mjc2LWEzMDktMTU5YTk3YzRiNTU3IiwidHlwZSI6ImFwaV90b2tlbiIsIm5hbWUiOiJmaXJzdF90ZXN0X0cxIiwiZXhwIjoxODA1MDY4ODAwLCJpYXQiOjE3NzM1NTM0OTZ9.FwNRueK1SannTUy9J69BvD2ZIbeGhKDg9jmdN21NgLU'

    print("=" * 60)
    print("Deep Lake Table Data Loader Test")
    print("=" * 60)

    try:
        # Create training dataset
        print("Creating training dataset...")
        train_dataset = DeepLakeTableDataset(
            table_url,
            token=token,
            sequence_length=32,
            train_split=True
        )

        print(f"\n✅ Training dataset created!")
        print(f"Dataset size: {len(train_dataset)} sequences")

        # Create validation dataset
        print("\nCreating validation dataset...")
        val_dataset = DeepLakeTableDataset(
            table_url,
            token=token,
            sequence_length=32,
            train_split=False
        )

        print(f"Validation dataset size: {len(val_dataset)} sequences")

        # Test data loading
        print(f"\n🔥 Testing data loading...")

        if len(train_dataset) > 0:
            sample = train_dataset[0]
            print(f"Sample data:")
            print(f"  Observations shape: {sample['observations'].shape}")
            print(f"  Actions shape: {sample['actions'].shape}")
            print(f"  Observations range: [{sample['observations'].min():.3f}, {sample['observations'].max():.3f}]")
            print(f"  Actions range: [{sample['actions'].min():.3f}, {sample['actions'].max():.3f}]")

            # Create DataLoader
            train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)

            print(f"\n⚡ Testing batch loading...")
            for i, batch in enumerate(train_loader):
                print(f"Batch {i}:")
                print(f"  Observations: {batch['observations'].shape}")
                print(f"  Actions: {batch['actions'].shape}")

                if i >= 1:  # Test first 2 batches
                    break

            print(f"\n🚀 SUCCESS! Your data is ready for training!")
            print(f"✅ {len(train_dataset)} training sequences available")
            print(f"✅ {len(val_dataset)} validation sequences available")
            print(f"✅ Data format: observations({train_dataset.observations.shape[1]}) -> actions({train_dataset.actions.shape[1]})")

        return train_dataset, val_dataset

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    test_table_loading()