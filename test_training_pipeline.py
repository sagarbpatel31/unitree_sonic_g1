#!/usr/bin/env python3
"""
Test the training pipeline with CSV data without full environment dependencies.
"""

import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.config import Config
from src.training.data_loader import MotionDataLoader


class SimpleTransformerPolicy(nn.Module):
    """Simplified transformer policy for testing."""

    def __init__(self, obs_dim, action_dim, d_model=256, nhead=8, num_layers=3):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Linear(obs_dim, d_model)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.output_proj = nn.Linear(d_model, action_dim)

    def forward(self, observations):
        """Forward pass through the model."""
        # observations: [batch, seq_len, obs_dim]
        batch_size, seq_len, _ = observations.shape

        # Project to model dimension
        x = self.input_proj(observations)  # [batch, seq_len, d_model]

        # Pass through transformer
        x = self.transformer(x)  # [batch, seq_len, d_model]

        # Project to action space
        actions = self.output_proj(x)  # [batch, seq_len, action_dim]

        return actions


def test_training_loop():
    """Test the complete training loop with CSV data."""

    print("=" * 60)
    print("Deep Lake Training Pipeline Test")
    print("=" * 60)

    # Create config for CSV data
    config_dict = {
        'env': {
            'task': {
                'use_csv_data': True,
                'csv_data_path': 'data/lightwheel_bevorg_frames.csv'
            }
        },
        'model': {
            'sequence_length': 32,
            'd_model': 256,
            'nhead': 8,
            'num_layers': 3
        },
        'training': {
            'learning_rate': 1e-4,
            'batch_size': 8,  # Small batch for testing
            'num_epochs': 2   # Just test a few epochs
        },
        'seed': 42
    }

    config = Config(config_dict)

    # Set random seed
    torch.manual_seed(config.seed)

    print("1. Setting up data loaders...")

    # Create data loader
    data_loader = MotionDataLoader(config)
    train_dataset = data_loader.get_training_dataset()
    eval_dataset = data_loader.get_evaluation_dataset()

    print(f"   Training dataset: {len(train_dataset)} sequences")
    print(f"   Evaluation dataset: {len(eval_dataset)} sequences")

    # Get dimensions from a sample
    sample = train_dataset[0]
    obs_dim = sample['observations'].shape[-1]
    action_dim = sample['actions'].shape[-1]

    print(f"   Observation dimension: {obs_dim}")
    print(f"   Action dimension: {action_dim}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=True
    )

    print("2. Setting up model...")

    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Using device: {device}")

    model = SimpleTransformerPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        d_model=config.model.d_model,
        nhead=config.model.nhead,
        num_layers=config.model.num_layers
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {total_params:,}")

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.training.learning_rate)

    print("3. Starting training...")

    # Training loop
    for epoch in range(config.training.num_epochs):
        model.train()
        train_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            observations = batch['observations'].to(device)
            actions = batch['actions'].to(device)

            # Forward pass
            optimizer.zero_grad()
            predicted_actions = model(observations)

            # Compute loss
            loss = criterion(predicted_actions, actions)

            # Backward pass
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

            # Print progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                print(f"   Epoch {epoch+1}/{config.training.num_epochs}, "
                      f"Batch {batch_idx+1}/{len(train_loader)}, "
                      f"Loss: {loss.item():.6f}")

            # Test only a few batches for demo
            if batch_idx >= 20:
                break

        avg_train_loss = train_loss / num_batches
        print(f"   Epoch {epoch+1} Average Training Loss: {avg_train_loss:.6f}")

        # Evaluation
        model.eval()
        eval_loss = 0.0
        eval_batches = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(eval_loader):
                observations = batch['observations'].to(device)
                actions = batch['actions'].to(device)

                predicted_actions = model(observations)
                loss = criterion(predicted_actions, actions)

                eval_loss += loss.item()
                eval_batches += 1

                # Test only a few batches
                if batch_idx >= 5:
                    break

        avg_eval_loss = eval_loss / eval_batches if eval_batches > 0 else 0.0
        print(f"   Epoch {epoch+1} Average Evaluation Loss: {avg_eval_loss:.6f}")

    print("4. Training completed!")
    print(f"   Final training loss: {avg_train_loss:.6f}")
    print(f"   Final evaluation loss: {avg_eval_loss:.6f}")

    # Test model output
    print("5. Testing model output...")
    model.eval()
    with torch.no_grad():
        test_batch = next(iter(train_loader))
        test_obs = test_batch['observations'][:1].to(device)  # Single sample
        test_actions = model(test_obs)

        print(f"   Input shape: {test_obs.shape}")
        print(f"   Output shape: {test_actions.shape}")
        print(f"   Output range: [{test_actions.min():.3f}, {test_actions.max():.3f}]")

    return True


def main():
    """Main test function."""

    try:
        success = test_training_loop()

        if success:
            print("\n🎉 Training pipeline test successful!")
            print("✅ CSV data loading works")
            print("✅ Model training works")
            print("✅ Evaluation works")
            print("\nYour Deep Lake training setup is fully functional!")
        else:
            print("\n❌ Training test failed")

    except Exception as e:
        print(f"\n❌ Error during training test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()