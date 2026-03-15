#!/usr/bin/env python3
"""
Simple test script for Deep Lake data loading.
"""

import deeplake
import numpy as np
import torch


def test_deeplake_connection():
    """Test connection to Deep Lake dataset and explore structure."""

    deeplake_url = "https://deeplake.ai/siyuliu4262s-organization/workspace/default/table/lightwheel_bevorg_frames"

    print(f"Connecting to Deep Lake dataset: {deeplake_url}")

    try:
        # Load the dataset using v3.x API
        ds = deeplake.load(deeplake_url)

        print(f"✅ Successfully connected to Deep Lake dataset!")
        print(f"Dataset size: {len(ds)}")

        # Explore dataset structure
        if hasattr(ds, 'tensors'):
            print(f"Available tensors: {list(ds.tensors.keys())}")

            # Get tensor info
            for tensor_name in ds.tensors.keys():
                tensor = ds.tensors[tensor_name]
                print(f"  {tensor_name}: shape={tensor.shape}, dtype={tensor.dtype}")

        # Try to access first few samples
        print("\nSampling first few items:")
        for i in range(min(3, len(ds))):
            print(f"\nSample {i}:")
            sample = ds[i]

            if hasattr(sample, 'keys'):
                print(f"  Keys: {list(sample.keys())}")
                for key in sample.keys():
                    try:
                        value = sample[key]
                        if hasattr(value, 'shape'):
                            print(f"    {key}: shape={value.shape}, dtype={value.dtype}")
                        else:
                            print(f"    {key}: {type(value)}")
                    except Exception as e:
                        print(f"    {key}: Error accessing - {e}")
            else:
                print(f"  Type: {type(sample)}")
                print(f"  Content: {sample}")

        # Test creating a simple data loader
        print("\nTesting data extraction:")

        if len(ds) > 0:
            sample = ds[0]
            print(f"Sample keys: {list(sample.keys()) if hasattr(sample, 'keys') else 'No keys'}")

            # Try to extract some data
            for key in (sample.keys() if hasattr(sample, 'keys') else []):
                try:
                    data = np.array(sample[key])
                    print(f"  {key}: converted to numpy shape {data.shape}")
                except Exception as e:
                    print(f"  {key}: Error converting to numpy - {e}")

        print("\n✅ Deep Lake dataset exploration completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Error accessing Deep Lake dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_deeplake_connection()