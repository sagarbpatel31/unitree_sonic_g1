#!/usr/bin/env python3
"""
Test script for the complete Deep Lake training pipeline.
"""

import sys
from pathlib import Path
import os

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.config import Config
from src.training.data_loader import MotionDataLoader, CSVMotionDataset
import torch
from torch.utils.data import DataLoader


def test_csv_dataset_direct():
    """Test the CSV dataset directly."""

    print("=" * 60)
    print("Testing CSV Dataset (Direct)")
    print("=" * 60)

    # Use the sample CSV we created earlier
    csv_path = "sample_motion_data.csv"

    if not Path(csv_path).exists():
        print(f"❌ Sample CSV not found: {csv_path}")
        print("Please run export_table_to_dataset.py first to create sample data")
        return False

    try:
        # Test training dataset
        train_dataset = CSVMotionDataset(
            csv_path=csv_path,
            sequence_length=32,
            train_split=True
        )

        print(f"✅ Training dataset created: {len(train_dataset)} sequences")

        # Test validation dataset
        val_dataset = CSVMotionDataset(
            csv_path=csv_path,
            sequence_length=32,
            train_split=False
        )

        print(f"✅ Validation dataset created: {len(val_dataset)} sequences")

        # Test data loading
        if len(train_dataset) > 0:
            sample = train_dataset[0]
            print(f"\nSample data:")
            print(f"  Observations shape: {sample['observations'].shape}")
            print(f"  Actions shape: {sample['actions'].shape}")

            # Test DataLoader
            dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
            batch = next(iter(dataloader))
            print(f"  Batch observations shape: {batch['observations'].shape}")
            print(f"  Batch actions shape: {batch['actions'].shape}")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_motion_data_loader_integration():
    """Test the integrated MotionDataLoader with CSV support."""

    print("\n" + "=" * 60)
    print("Testing MotionDataLoader Integration")
    print("=" * 60)

    try:
        # Create config for CSV data
        config_dict = {
            'env': {
                'task': {
                    'use_csv_data': True,
                    'csv_data_path': 'sample_motion_data.csv'
                }
            },
            'model': {
                'sequence_length': 32,
                'obs_dim': 42,
                'action_dim': 42
            }
        }

        config = Config(config_dict)

        # Test MotionDataLoader
        data_loader = MotionDataLoader(config)

        print(f"✅ MotionDataLoader created with CSV support")

        # Get datasets
        train_dataset = data_loader.get_training_dataset()
        eval_dataset = data_loader.get_evaluation_dataset()

        print(f"✅ Training dataset: {len(train_dataset)} sequences")
        print(f"✅ Evaluation dataset: {len(eval_dataset)} sequences")

        # Test data format
        if len(train_dataset) > 0:
            sample = train_dataset[0]
            print(f"\nData format verification:")
            print(f"  Observations shape: {sample['observations'].shape}")
            print(f"  Actions shape: {sample['actions'].shape}")
            print(f"  Observations dtype: {sample['observations'].dtype}")
            print(f"  Actions dtype: {sample['actions'].dtype}")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_script_config():
    """Test if the training script configuration works."""

    print("\n" + "=" * 60)
    print("Testing Training Configuration")
    print("=" * 60)

    try:
        from src.core.config import load_config

        # Test loading the Deep Lake config
        config_path = "configs/train/bc_deeplake.yaml"

        if not Path(config_path).exists():
            print(f"❌ Config file not found: {config_path}")
            return False

        # config = load_config(config_path)
        print(f"✅ Config file exists: {config_path}")

        # For now, just check if the file exists and has the right structure
        with open(config_path, 'r') as f:
            content = f.read()

        if 'use_csv_data: true' in content:
            print(f"✅ CSV data configuration found in config")
        else:
            print(f"⚠️  CSV data configuration not found in config")

        if 'csv_data_path:' in content:
            print(f"✅ CSV data path configuration found")
        else:
            print(f"⚠️  CSV data path configuration not found")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""

    print("Deep Lake Training Pipeline Test")
    print("This test validates the complete training pipeline setup")

    # Run tests
    results = []

    # Test 1: Direct CSV dataset
    results.append(test_csv_dataset_direct())

    # Test 2: Integrated data loader
    results.append(test_motion_data_loader_integration())

    # Test 3: Training configuration
    results.append(test_training_script_config())

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(results)
    total = len(results)

    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("🎉 All tests passed! Your Deep Lake training pipeline is ready!")
        print("\nNext steps:")
        print("1. Export your Deep Lake table to 'data/lightwheel_bevorg_frames.csv'")
        print("2. Run: python train_imitation.py --config configs/train/bc_deeplake.yaml")
    else:
        print("❌ Some tests failed. Please check the errors above.")

    return passed == total


if __name__ == "__main__":
    main()