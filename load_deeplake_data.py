#!/usr/bin/env python3
"""
Script to load data from Deep Lake for Sonic G1 training.
"""

import sys
from pathlib import Path
import os

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

import deeplake
from src.training.deeplake_loader import DeepLakeMotionDataLoader
from src.core.config import Config
import torch
from torch.utils.data import DataLoader


def setup_deeplake_auth():
    """Setup Deep Lake authentication if needed."""
    # Check if user has authentication token
    token = os.environ.get('eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiNTNhNDYyYWYtMmRjZS00Zjk4LWI3NTQtZWQ3NjEyN2I0OTU1Iiwib3JnX2lkIjoiNmFiMzUzMjQtOTEyZS00Mjc2LWEzMDktMTU5YTk3YzRiNTU3IiwidHlwZSI6ImFwaV90b2tlbiIsIm5hbWUiOiJmaXJzdF90ZXN0X0cxIiwiZXhwIjoxODA1MDY4ODAwLCJpYXQiOjE3NzM1NTM0OTZ9.FwNRueK1SannTUy9J69BvD2ZIbeGhKDg9jmdN21NgLU')

    if token:
        print(f"Using Deep Lake token from environment")
        return token
    else:
        print("No Deep Lake token found in environment.")
        print("To use private datasets, set DEEPLAKE_TOKEN environment variable:")
        print("export DEEPLAKE_TOKEN='your_token_here'")
        return None


def test_dataset_access(deeplake_url, token=None):
    """Test if we can access the dataset."""
    try:
        if token:
            ds = deeplake.load(deeplake_url, token=token)
        else:
            ds = deeplake.load(deeplake_url)

        print(f"✅ Dataset accessible! Size: {len(ds)}")
        return ds
    except Exception as e:
        print(f"❌ Cannot access dataset: {e}")
        return None


def create_sample_dataset(local_path="./sample_deeplake_data"):
    """Create a sample Deep Lake dataset for testing."""
    print(f"Creating sample dataset at: {local_path}")

    try:
        # Create a sample dataset
        ds = deeplake.empty(local_path, overwrite=True)

        # Define schema
        ds.create_tensor('observations', htype='generic')
        ds.create_tensor('actions', htype='generic')
        ds.create_tensor('timestamps', htype='generic')

        # Add some sample data
        import numpy as np

        for i in range(100):
            # Sample motion data
            obs = np.random.randn(64).astype(np.float32)  # 64-dim observation
            action = np.random.randn(22).astype(np.float32)  # 22-dim action (G1 joints)
            timestamp = float(i * 0.02)  # 50Hz

            ds.observations.append(obs)
            ds.actions.append(action)
            ds.timestamps.append(timestamp)

        print(f"✅ Sample dataset created with {len(ds)} frames")
        return local_path

    except Exception as e:
        print(f"❌ Error creating sample dataset: {e}")
        return None


def main():
    """Main function to load Deep Lake data."""

    # Your Deep Lake URL
    deeplake_url = "https://deeplake.ai/siyuliu4262s-organization/workspace/default/table/lightwheel_bevorg_frames"

    print("=" * 60)
    print("Deep Lake Data Loader for Sonic G1 Training")
    print("=" * 60)

    # Setup authentication
    token = setup_deeplake_auth()

    # Test dataset access
    print(f"\nTesting access to: {deeplake_url}")
    ds = test_dataset_access(deeplake_url, token)

    if ds is None:
        print("\n⚠️  Could not access remote dataset. Creating sample dataset for testing...")
        sample_path = create_sample_dataset()
        if sample_path:
            deeplake_url = sample_path
            print(f"Using sample dataset: {deeplake_url}")
        else:
            print("❌ Could not create sample dataset either. Exiting.")
            return

    # Create config
    config = Config({
        'model': {'sequence_length': 32},
        'env': {'task': {}},
        'training': {'batch_size': 4}
    })

    print(f"\n📊 Creating Deep Lake data loader...")

    try:
        # Create data loader
        dl_loader = DeepLakeMotionDataLoader(config, deeplake_url)

        # Explore dataset
        print("\n🔍 Exploring dataset structure...")
        dataset_info = dl_loader.explore_dataset()
        print("Dataset Information:")
        for key, value in dataset_info.items():
            print(f"  {key}: {value}")

        # Create training dataset
        print("\n🏃 Creating training dataset...")
        train_dataset = dl_loader.get_training_dataset()
        print(f"Training dataset size: {len(train_dataset)}")

        # Create PyTorch DataLoader
        print("\n⚡ Creating PyTorch DataLoader...")
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=0
        )

        # Test loading batches
        print("\n🎯 Testing batch loading...")
        for i, batch in enumerate(train_loader):
            print(f"Batch {i}:")
            print(f"  Observations shape: {batch['observations'].shape}")
            print(f"  Actions shape: {batch['actions'].shape}")
            print(f"  Observations range: [{batch['observations'].min():.3f}, {batch['observations'].max():.3f}]")
            print(f"  Actions range: [{batch['actions'].min():.3f}, {batch['actions'].max():.3f}]")

            if i >= 2:  # Test first 3 batches
                break

        # Create evaluation dataset
        print("\n📈 Creating evaluation dataset...")
        eval_dataset = dl_loader.get_evaluation_dataset()
        print(f"Evaluation dataset size: {len(eval_dataset)}")

        print("\n✅ Deep Lake data loader setup complete!")
        print("\nYou can now use this data loader in your training scripts:")
        print("  from src.training.deeplake_loader import DeepLakeMotionDataLoader")
        print(f"  loader = DeepLakeMotionDataLoader(config, '{deeplake_url}')")
        print("  train_dataset = loader.get_training_dataset()")

    except Exception as e:
        print(f"\n❌ Error during data loading: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()