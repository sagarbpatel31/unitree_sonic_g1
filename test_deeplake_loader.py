#!/usr/bin/env python3
"""
Test script for Deep Lake data loader.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.training.deeplake_loader import DeepLakeMotionDataLoader
from src.core.config import Config
import torch
from torch.utils.data import DataLoader


def test_deeplake_loader():
    """Test the Deep Lake data loader with your dataset."""

    # Your Deep Lake URL
    deeplake_url = "https://deeplake.ai/siyuliu4262s-organization/workspace/default/table/lightwheel_bevorg_frames"

    # Create a minimal config
    config = Config({
        'model': {'sequence_length': 32},
        'env': {'task': {}},
        'training': {'batch_size': 4}
    })

    print(f"Testing Deep Lake loader with URL: {deeplake_url}")

    try:
        # Create data loader
        dl_loader = DeepLakeMotionDataLoader(config, deeplake_url)

        # Explore dataset structure
        print("\nExploring dataset structure...")
        dataset_info = dl_loader.explore_dataset()
        print("Dataset info:")
        for key, value in dataset_info.items():
            print(f"  {key}: {value}")

        # Try to create training dataset
        print("\nCreating training dataset...")
        train_dataset = dl_loader.get_training_dataset()
        print(f"Training dataset size: {len(train_dataset)}")

        # Create PyTorch DataLoader
        print("\nCreating PyTorch DataLoader...")
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=0  # Use 0 for debugging
        )

        # Test loading a batch
        print("\nTesting batch loading...")
        for i, batch in enumerate(train_loader):
            print(f"Batch {i}:")
            print(f"  Observations shape: {batch['observations'].shape}")
            print(f"  Actions shape: {batch['actions'].shape}")
            print(f"  Observations dtype: {batch['observations'].dtype}")
            print(f"  Actions dtype: {batch['actions'].dtype}")

            # Only test first batch
            break

        print("\n✅ Deep Lake loader test successful!")

        # Create evaluation dataset
        print("\nCreating evaluation dataset...")
        eval_dataset = dl_loader.get_evaluation_dataset()
        print(f"Evaluation dataset size: {len(eval_dataset)}")

    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_deeplake_loader()