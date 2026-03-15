#!/usr/bin/env python3
"""
Create or load Deep Lake dataset for Sonic G1 training.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import deeplake


def try_different_urls(base_org_name="siyuliu4262s-organization", dataset_name="lightwheel_bevorg_frames"):
    """Try different URL formats for Deep Lake."""

    possible_urls = [
        f"hub://{base_org_name}/{dataset_name}",
        f"https://deeplake.ai/{base_org_name}/{dataset_name}",
        f"https://deeplake.ai/{base_org_name}/workspace/default/table/{dataset_name}",
        f"hub://siyuliu4262s-organization/default/{dataset_name}",
        f"hub://siyuliu4262s/{dataset_name}",
    ]

    token = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiNTNhNDYyYWYtMmRjZS00Zjk4LWI3NTQtZWQ3NjEyN2I0OTU1Iiwib3JnX2lkIjoiNmFiMzUzMjQtOTEyZS00Mjc2LWEzMDktMTU5YTk3YzRiNTU3IiwidHlwZSI6ImFwaV90b2tlbiIsIm5hbWUiOiJmaXJzdF90ZXN0X0cxIiwiZXhwIjoxODA1MDY4ODAwLCJpYXQiOjE3NzM1NTM0OTZ9.FwNRueK1SannTUy9J69BvD2ZIbeGhKDg9jmdN21NgLU'

    print("Testing different Deep Lake URL formats...")

    for url in possible_urls:
        print(f"\nTrying: {url}")
        try:
            ds = deeplake.load(url, token=token)
            print(f"✅ SUCCESS! Dataset found at: {url}")
            print(f"   Dataset size: {len(ds)}")
            if hasattr(ds, 'tensors'):
                print(f"   Available tensors: {list(ds.tensors.keys())}")
            return url, ds
        except Exception as e:
            print(f"❌ Failed: {e}")

    print("\n⚠️  No existing dataset found with any URL format.")
    return None, None


def create_sample_dataset_remote(token):
    """Create a sample dataset in Deep Lake cloud."""

    # Try to create a new dataset in the user's account
    dataset_name = "sonic_g1_motion_data"
    url = f"hub://siyuliu4262s-organization/{dataset_name}"

    print(f"Creating new dataset at: {url}")

    try:
        # Create empty dataset
        ds = deeplake.empty(url, token=token, overwrite=True)

        # Define schema for motion data
        ds.create_tensor('observations', htype='generic')
        ds.create_tensor('actions', htype='generic')
        ds.create_tensor('timestamps', htype='generic')
        ds.create_tensor('episode_id', htype='generic')

        print("✅ Dataset schema created successfully!")

        # Add some sample motion data
        print("Adding sample motion data...")

        for episode in range(3):  # 3 episodes
            episode_length = np.random.randint(50, 200)  # Variable episode lengths

            for step in range(episode_length):
                # Generate realistic motion data
                t = step * 0.02  # 50Hz

                # Sample observation (joint positions, velocities, IMU, etc.)
                joint_pos = np.sin(t * 2 * np.pi * 0.5) * 0.5  # Walking frequency
                obs = np.random.randn(64).astype(np.float32) * 0.1
                obs[0] = joint_pos  # Hip position
                obs[1] = max(0, np.sin(t * 4 * np.pi * 0.5)) * 0.3  # Knee

                # Sample action (joint targets)
                action = np.random.randn(22).astype(np.float32) * 0.1
                action[0] = joint_pos + np.random.randn() * 0.05
                action[1] = obs[1] + np.random.randn() * 0.05

                # Append to dataset
                ds.observations.append(obs)
                ds.actions.append(action)
                ds.timestamps.append(t)
                ds.episode_id.append(episode)

        print(f"✅ Added {len(ds)} samples across 3 episodes")
        return url, ds

    except Exception as e:
        print(f"❌ Failed to create remote dataset: {e}")
        return None, None


def create_local_dataset():
    """Create a local dataset for testing."""
    local_path = "./sonic_g1_motion_data"

    print(f"Creating local dataset at: {local_path}")

    try:
        # Create local dataset
        ds = deeplake.empty(local_path, overwrite=True)

        # Define schema
        ds.create_tensor('observations', htype='generic')
        ds.create_tensor('actions', htype='generic')
        ds.create_tensor('timestamps', htype='generic')

        # Add sample data (same as above but simpler)
        for i in range(100):
            t = i * 0.02
            obs = np.random.randn(64).astype(np.float32)
            action = np.random.randn(22).astype(np.float32)

            ds.observations.append(obs)
            ds.actions.append(action)
            ds.timestamps.append(t)

        print(f"✅ Local dataset created with {len(ds)} samples")
        return local_path, ds

    except Exception as e:
        print(f"❌ Failed to create local dataset: {e}")
        return None, None


class MotionDataset(Dataset):
    """PyTorch dataset for motion data."""

    def __init__(self, dataset_url, token=None, sequence_length=32):
        self.sequence_length = sequence_length

        if token:
            self.ds = deeplake.load(dataset_url, token=token)
        else:
            self.ds = deeplake.load(dataset_url)

        print(f"Loaded dataset with {len(self.ds)} samples")

    def __len__(self):
        return max(0, len(self.ds) - self.sequence_length + 1)

    def __getitem__(self, idx):
        observations = []
        actions = []

        for i in range(idx, idx + self.sequence_length):
            obs = self.ds.observations[i].numpy()
            action = self.ds.actions[i].numpy()
            observations.append(obs)
            actions.append(action)

        return {
            "observations": torch.FloatTensor(observations),
            "actions": torch.FloatTensor(actions)
        }


def main():
    """Main function."""
    print("=" * 60)
    print("Deep Lake Dataset Setup for Sonic G1 Training")
    print("=" * 60)

    token = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiNTNhNDYyYWYtMmRjZS00Zjk4LWI3NTQtZWQ3NjEyN2I0OTU1Iiwib3JnX2lkIjoiNmFiMzUzMjQtOTEyZS00Mjc2LWEzMDktMTU5YTk3YzRiNTU3IiwidHlwZSI6ImFwaV90b2tlbiIsIm5hbWUiOiJmaXJzdF90ZXN0X0cxIiwiZXhwIjoxODA1MDY4ODAwLCJpYXQiOjE3NzM1NTM0OTZ9.FwNRueK1SannTUy9J69BvD2ZIbeGhKDg9jmdN21NgLU'

    # Step 1: Try to find existing dataset
    dataset_url, ds = try_different_urls()

    if dataset_url is None:
        print("\n📝 No existing dataset found. Creating new dataset...")

        # Step 2: Try to create remote dataset
        dataset_url, ds = create_sample_dataset_remote(token)

        if dataset_url is None:
            print("\n💽 Creating local dataset instead...")
            # Step 3: Fallback to local dataset
            dataset_url, ds = create_local_dataset()

    if dataset_url and ds:
        print(f"\n✅ Dataset ready at: {dataset_url}")

        # Step 4: Create PyTorch dataset
        print("\n🔥 Creating PyTorch dataset...")
        motion_dataset = MotionDataset(dataset_url, token=token if 'hub://' in dataset_url else None)

        # Step 5: Create DataLoader
        dataloader = DataLoader(motion_dataset, batch_size=4, shuffle=True)

        print(f"PyTorch dataset size: {len(motion_dataset)}")

        # Step 6: Test loading
        print("\n🎯 Testing data loading...")
        for i, batch in enumerate(dataloader):
            print(f"Batch {i}:")
            print(f"  Observations: {batch['observations'].shape}")
            print(f"  Actions: {batch['actions'].shape}")
            print(f"  Obs range: [{batch['observations'].min():.3f}, {batch['observations'].max():.3f}]")
            print(f"  Action range: [{batch['actions'].min():.3f}, {batch['actions'].max():.3f}]")

            if i >= 1:  # Test first 2 batches
                break

        print(f"\n🚀 SUCCESS! Your Deep Lake dataset is ready for training!")
        print(f"Dataset URL: {dataset_url}")
        print(f"\nNext steps:")
        print(f"1. Use this dataset URL in your training scripts")
        print(f"2. Modify the data loading code to match your actual data schema")
        print(f"3. Add your real motion capture data to the dataset")

    else:
        print("❌ Could not create or access any dataset.")


if __name__ == "__main__":
    main()