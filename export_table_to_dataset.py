#!/usr/bin/env python3
"""
Script to export Deep Lake table data to a proper Deep Lake dataset for training.
"""

import requests
import json
import numpy as np
import pandas as pd
import deeplake
import torch
from torch.utils.data import Dataset, DataLoader


def query_deeplake_table(table_url, token, limit=1000, offset=0):
    """Query Deep Lake table using SQL-like syntax."""

    # Extract organization and table name from URL
    parts = table_url.split('/')
    org_name = parts[3]  # siyuliu4262s-organization
    table_name = parts[-1]  # lightwheel_bevorg_frames

    # Deep Lake API endpoint for querying
    api_url = f"https://api.deeplake.ai/v1/organizations/{org_name}/workspaces/default/tables/{table_name}/query"

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    # SQL query to get data
    query_data = {
        "sql": f"SELECT * FROM {table_name} LIMIT {limit} OFFSET {offset}",
        "format": "json"
    }

    try:
        response = requests.post(api_url, headers=headers, json=query_data)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"API Error {response.status_code}: {response.text}")
            return None
    except Exception as e:
        print(f"Request failed: {e}")
        return None


def export_table_to_dataset():
    """Export table data to a Deep Lake dataset."""

    table_url = "https://deeplake.ai/siyuliu4262s-organization/workspace/default/table/lightwheel_bevorg_frames"
    token = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiNTNhNDYyYWYtMmRjZS00Zjk4LWI3NTQtZWQ3NjEyN2I0OTU1Iiwib3JnX2lkIjoiNmFiMzUzMjQtOTEyZS00Mjc2LWEzMDktMTU5YTk3YzRiNTU3IiwidHlwZSI6ImFwaV90b2tlbiIsIm5hbWUiOiJmaXJzdF90ZXN0X0cxIiwiZXhwIjoxODA1MDY4ODAwLCJpYXQiOjE3NzM1NTM0OTZ9.FwNRueK1SannTUy9J69BvD2ZIbeGhKDg9jmdN21NgLU'

    print("=" * 60)
    print("Deep Lake Table Export to Dataset")
    print("=" * 60)

    # Query some sample data first
    print("Querying sample data from table...")
    sample_data = query_deeplake_table(table_url, token, limit=100)

    if sample_data:
        print(f"✅ Successfully queried sample data!")
        print(f"Sample structure: {list(sample_data.keys()) if isinstance(sample_data, dict) else type(sample_data)}")

        # Process the data to understand structure
        if isinstance(sample_data, dict) and 'data' in sample_data:
            df = pd.DataFrame(sample_data['data'])
            print(f"DataFrame shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            print(f"First few rows:\n{df.head()}")

        return sample_data
    else:
        print("❌ Failed to query table data.")
        return None


class LocalCSVDataset(Dataset):
    """PyTorch dataset for local CSV data as fallback."""

    def __init__(self, csv_path, sequence_length=32, train_split=True):
        self.sequence_length = sequence_length

        # Load CSV data
        self.df = pd.read_csv(csv_path)
        print(f"Loaded CSV with shape: {self.df.shape}")

        # Split data
        split_idx = int(0.8 * len(self.df))
        if train_split:
            self.df = self.df.iloc[:split_idx]
        else:
            self.df = self.df.iloc[split_idx:]

        # Extract features
        self._prepare_data()

    def _prepare_data(self):
        """Prepare observation and action data."""
        # Action columns (those starting with 'act_')
        action_cols = [col for col in self.df.columns if col.startswith('act_')]

        # Observation columns (remaining numeric columns)
        obs_cols = [col for col in self.df.columns
                   if col not in action_cols and col != '_id' and self.df[col].dtype in ['float64', 'int64']]

        # Extract data
        if action_cols:
            self.actions = self.df[action_cols].values.astype(np.float32)
        else:
            self.actions = np.random.randn(len(self.df), 22).astype(np.float32)

        if obs_cols:
            self.observations = self.df[obs_cols].values.astype(np.float32)
        else:
            self.observations = self.actions  # Fallback

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


def create_sample_csv():
    """Create a sample CSV file based on the table structure we saw."""

    print("\n📝 Creating sample CSV data based on your table structure...")

    # Based on the screenshot, create similar data structure
    n_samples = 1000

    data = {
        '_id': [f"sample_{i:06d}" for i in range(n_samples)],
        'act_base_height_cm': np.random.uniform(0.7, 0.8, n_samples),
        'act_left_hand_state': np.random.choice([-1, 0, 1], n_samples),
        'act_left_wrist_qw': np.random.uniform(0.99, 1.0, n_samples),
        'act_left_wrist_qx': np.random.uniform(-0.02, 0.02, n_samples),
        'act_left_wrist_qy': np.random.uniform(-0.02, 0.02, n_samples),
        'act_left_wrist_qz': np.random.uniform(0.19, 0.21, n_samples),
        'act_left_wrist_x': np.random.uniform(0.19, 0.21, n_samples),
    }

    # Add more action columns to match typical humanoid robot
    for joint in ['right_wrist', 'left_elbow', 'right_elbow', 'left_shoulder', 'right_shoulder']:
        for coord in ['qw', 'qx', 'qy', 'qz', 'x', 'y', 'z']:
            if coord.startswith('q'):
                data[f'act_{joint}_{coord}'] = np.random.uniform(-0.02, 0.02, n_samples)
            else:
                data[f'act_{joint}_{coord}'] = np.random.uniform(-0.1, 0.1, n_samples)

    # Create DataFrame and save
    df = pd.DataFrame(data)
    csv_path = "sample_motion_data.csv"
    df.to_csv(csv_path, index=False)

    print(f"✅ Created sample CSV: {csv_path}")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)[:5]}...")

    return csv_path


def main():
    """Main function."""

    # Try to export from Deep Lake table
    table_data = export_table_to_dataset()

    if not table_data:
        print("\n💡 Since we can't access the table directly, let's create a working example...")

        # Create sample CSV
        csv_path = create_sample_csv()

        # Test with CSV dataset
        print(f"\n🔥 Testing with sample CSV dataset...")

        train_dataset = LocalCSVDataset(csv_path, sequence_length=32, train_split=True)
        val_dataset = LocalCSVDataset(csv_path, sequence_length=32, train_split=False)

        print(f"Training dataset: {len(train_dataset)} sequences")
        print(f"Validation dataset: {len(val_dataset)} sequences")

        # Test data loading
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

        for i, batch in enumerate(train_loader):
            print(f"\nBatch {i}:")
            print(f"  Observations: {batch['observations'].shape}")
            print(f"  Actions: {batch['actions'].shape}")

            if i >= 1:
                break

        print(f"\n✅ Sample data loader working!")
        print(f"\n📋 Next steps to use your actual data:")
        print(f"1. Export your Deep Lake table to CSV format")
        print(f"2. Replace the sample CSV with your actual data")
        print(f"3. Use the LocalCSVDataset class to load your data")
        print(f"4. Alternatively, convert your table to a proper Deep Lake dataset")


if __name__ == "__main__":
    main()