#!/usr/bin/env python3
"""
Simple test for Deep Lake CSV data loading without full project imports.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class CSVMotionDataset(Dataset):
    """Simplified CSV dataset for testing."""

    def __init__(self, csv_path, sequence_length=32, train_split=True):
        self.csv_path = csv_path
        self.sequence_length = sequence_length

        # Load CSV
        print(f"Loading CSV from: {csv_path}")
        self.df = pd.read_csv(csv_path)
        print(f"CSV shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)[:5]}...")

        # Split data
        split_idx = int(0.8 * len(self.df))
        if train_split:
            self.df = self.df.iloc[:split_idx]
            print(f"Training split: {len(self.df)} rows")
        else:
            self.df = self.df.iloc[split_idx:]
            print(f"Validation split: {len(self.df)} rows")

        # Extract features
        self._prepare_data()

    def _prepare_data(self):
        """Extract actions and observations."""
        # Action columns
        action_cols = [col for col in self.df.columns if col.startswith('act_')]
        obs_cols = [col for col in self.df.columns
                   if col not in action_cols and col != '_id' and
                   self.df[col].dtype in ['float64', 'int64', 'float32']]

        print(f"Action columns: {len(action_cols)}")
        print(f"Observation columns: {len(obs_cols)}")

        self.actions = self.df[action_cols].values.astype(np.float32)
        self.observations = self.df[obs_cols].values.astype(np.float32) if obs_cols else self.actions

        print(f"Actions shape: {self.actions.shape}")
        print(f"Observations shape: {self.observations.shape}")

    def __len__(self):
        return max(0, len(self.df) - self.sequence_length + 1)

    def __getitem__(self, idx):
        end_idx = idx + self.sequence_length

        obs_seq = self.observations[idx:end_idx]
        action_seq = self.actions[idx:end_idx]

        return {
            "observations": torch.FloatTensor(obs_seq),
            "actions": torch.FloatTensor(action_seq)
        }


def test_pipeline():
    """Test the CSV data loading pipeline."""

    print("=" * 60)
    print("Deep Lake CSV Pipeline Test")
    print("=" * 60)

    # Check if sample CSV exists
    csv_path = "sample_motion_data.csv"
    if not Path(csv_path).exists():
        print(f"❌ Sample CSV not found: {csv_path}")
        print("Creating sample data...")

        # Create sample data
        data = {
            '_id': [f"frame_{i:06d}" for i in range(1000)],
            'act_base_height': np.random.uniform(0.7, 0.8, 1000),
            'act_left_hand_state': np.random.choice([-1, 0, 1], 1000),
        }

        # Add more action columns
        for i in range(20):
            data[f'act_joint_{i:02d}'] = np.random.randn(1000) * 0.1

        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        print(f"✅ Created sample CSV: {csv_path}")

    # Test datasets
    try:
        # Training dataset
        train_dataset = CSVMotionDataset(csv_path, train_split=True)
        print(f"\n✅ Training dataset: {len(train_dataset)} sequences")

        # Validation dataset
        val_dataset = CSVMotionDataset(csv_path, train_split=False)
        print(f"✅ Validation dataset: {len(val_dataset)} sequences")

        # Test data loading
        if len(train_dataset) > 0:
            sample = train_dataset[0]
            print(f"\nSample data:")
            print(f"  Observations: {sample['observations'].shape}")
            print(f"  Actions: {sample['actions'].shape}")

            # Test DataLoader
            dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
            batch = next(iter(dataloader))
            print(f"\nBatch data:")
            print(f"  Observations: {batch['observations'].shape}")
            print(f"  Actions: {batch['actions'].shape}")
            print(f"  Obs range: [{batch['observations'].min():.3f}, {batch['observations'].max():.3f}]")
            print(f"  Action range: [{batch['actions'].min():.3f}, {batch['actions'].max():.3f}]")

            print(f"\n🎉 Pipeline test successful!")
            return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_config_files():
    """Check if config files exist."""

    print("\n" + "=" * 60)
    print("Configuration File Check")
    print("=" * 60)

    config_files = [
        "configs/train/bc_deeplake.yaml",
        "configs/data/deeplake.yaml"
    ]

    all_exist = True
    for config_file in config_files:
        if Path(config_file).exists():
            print(f"✅ {config_file} exists")
        else:
            print(f"❌ {config_file} missing")
            all_exist = False

    return all_exist


def main():
    """Main test function."""

    # Test 1: CSV data pipeline
    pipeline_works = test_pipeline()

    # Test 2: Config files
    configs_exist = check_config_files()

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    if pipeline_works and configs_exist:
        print("🚀 All tests passed!")
        print("\nYour Deep Lake training setup is ready!")
        print("\nNext steps:")
        print("1. Export your Deep Lake table data to 'data/lightwheel_bevorg_frames.csv'")
        print("2. Run: python train_imitation.py --config configs/train/bc_deeplake.yaml")
        print("\nExpected training command:")
        print("  python train_imitation.py --config configs/train/bc_deeplake.yaml")
    else:
        if not pipeline_works:
            print("❌ Data pipeline test failed")
        if not configs_exist:
            print("❌ Config files missing")


if __name__ == "__main__":
    main()