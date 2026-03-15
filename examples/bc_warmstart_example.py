#!/usr/bin/env python3
"""
Complete example demonstrating behavior cloning warm-start for Unitree G1.

This script shows the full workflow from data preparation to policy export
for PPO initialization.
"""

import sys
import logging
from pathlib import Path
import torch
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from omegaconf import OmegaConf
from src.data import load_g1_trajectory_from_npz
from sonic_g1.data import create_bc_dataloaders, StateActionExtractor
from sonic_g1.models.policy import G1Policy
from sonic_g1.train.bc_losses import BCLossCollection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_data_inspection():
    """
    Example 1: Inspect retargeted trajectory data for BC training.
    """
    print("\n" + "="*60)
    print("EXAMPLE 1: Data Inspection")
    print("="*60)

    # Load a sample trajectory (replace with actual path)
    trajectory_path = "data/g1_trajectories/sample_trajectory.npz"

    try:
        trajectory = load_g1_trajectory_from_npz(trajectory_path)

        print(f"Trajectory metadata:")
        print(f"  Duration: {trajectory.metadata.get('duration', 'Unknown'):.2f}s")
        print(f"  Frames: {len(trajectory.timestamps)}")
        print(f"  Joint positions shape: {trajectory.joint_positions.shape}")
        print(f"  Root positions shape: {trajectory.root_positions.shape}")
        print(f"  Quality score: {trajectory.metadata.get('overall_quality_score', 'Unknown')}")

        # Create state-action extractor
        extraction_config = OmegaConf.create({
            'state': {
                'include_joint_pos': True,
                'include_joint_vel': True,
                'include_root_pos': True,
                'include_root_orient': True,
                'include_reference_features': True,
                'reference_horizon': 10
            },
            'action': {
                'type': 'joint_positions'
            },
            'action_lookahead': 1
        })

        extractor = StateActionExtractor(extraction_config)
        states, actions, metadata = extractor.extract_from_trajectory(trajectory)

        print(f"\nExtracted data:")
        print(f"  State dimension: {states.shape[1]}")
        print(f"  Action dimension: {actions.shape[1]}")
        print(f"  Number of samples: {states.shape[0]}")
        print(f"  State range: [{states.min():.3f}, {states.max():.3f}]")
        print(f"  Action range: [{actions.min():.3f}, {actions.max():.3f}]")

    except Exception as e:
        print(f"Could not load sample trajectory: {e}")
        print("Note: Replace 'trajectory_path' with actual trajectory file")


def example_dataset_creation():
    """
    Example 2: Create BC dataset and data loaders.
    """
    print("\n" + "="*60)
    print("EXAMPLE 2: Dataset Creation")
    print("="*60)

    # Create minimal BC config
    config = OmegaConf.create({
        'data': {
            'trajectory_dir': 'data/g1_trajectories',
            'val_split': 0.2,
            'sequence_length': 1,
            'skip_frames': 1,
            'min_trajectory_length': 50,
            'filter_quality': True,
            'quality_threshold': 0.6,
            'augment_data': False,
            'num_workers': 2,
            'extraction': {
                'state': {
                    'include_joint_pos': True,
                    'include_joint_vel': True,
                    'include_root_pos': True,
                    'include_root_orient': True,
                    'include_reference_features': False  # Simplified for example
                },
                'action': {
                    'type': 'joint_positions'
                },
                'action_lookahead': 1
            }
        },
        'training': {
            'batch_size': 64
        }
    })

    try:
        # Create data loaders
        train_loader, val_loader, dataset_info = create_bc_dataloaders(config)

        print(f"Dataset statistics:")
        print(f"  Training samples: {dataset_info['train_samples']}")
        print(f"  Validation samples: {dataset_info['val_samples']}")
        print(f"  State dimension: {dataset_info['state_dim']}")
        print(f"  Action dimension: {dataset_info['action_dim']}")

        # Inspect a batch
        sample_batch = next(iter(train_loader))
        print(f"\nSample batch:")
        print(f"  States shape: {sample_batch['states'].shape}")
        print(f"  Actions shape: {sample_batch['actions'].shape}")
        print(f"  States range: [{sample_batch['states'].min():.3f}, {sample_batch['states'].max():.3f}]")
        print(f"  Actions range: [{sample_batch['actions'].min():.3f}, {sample_batch['actions'].max():.3f}]")

    except Exception as e:
        print(f"Could not create dataset: {e}")
        print("Note: Ensure 'data/g1_trajectories' exists with trajectory files")


def example_model_training():
    """
    Example 3: Simple BC model training loop.
    """
    print("\n" + "="*60)
    print("EXAMPLE 3: Model Training")
    print("="*60)

    # Model configuration
    obs_dim = 100  # Example dimension
    action_dim = 22  # G1 DOF

    policy_config = OmegaConf.create({
        'hidden_dims': [256, 256],
        'activation': 'ReLU',
        'log_std_type': 'learned',
        'initial_log_std': -2.0,
        'action_scale': 1.0
    })

    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy = G1Policy(obs_dim, action_dim, policy_config).to(device)

    print(f"Created policy with {sum(p.numel() for p in policy.parameters())} parameters")

    # Create loss function
    loss_config = OmegaConf.create({
        'mse_weight': 1.0,
        'mae_weight': 0.1,
        'regularization_weight': 0.001
    })

    loss_fn = BCLossCollection(loss_config, device)

    # Create optimizer
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    # Simulate training with dummy data
    print("\nSimulating training with dummy data...")

    for epoch in range(5):
        # Create dummy batch
        batch_size = 32
        states = torch.randn(batch_size, obs_dim, device=device)
        target_actions = torch.randn(batch_size, action_dim, device=device)

        # Forward pass
        optimizer.zero_grad()
        action_dist = policy.get_distribution(states)
        predicted_actions = action_dist.mean

        # Compute loss
        losses = loss_fn.compute_losses(
            predicted_actions=predicted_actions,
            target_actions=target_actions,
            action_distribution=action_dist
        )

        total_loss = losses['total_loss']

        # Backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        print(f"Epoch {epoch+1}: Loss = {total_loss.item():.6f}")

    print("Training simulation completed")


def example_policy_export():
    """
    Example 4: Export BC policy for PPO initialization.
    """
    print("\n" + "="*60)
    print("EXAMPLE 4: Policy Export")
    print("="*60)

    # Create dummy trained policy
    obs_dim = 100
    action_dim = 22

    policy_config = OmegaConf.create({
        'hidden_dims': [512, 512, 256],
        'activation': 'ReLU',
        'log_std_type': 'learned',
        'initial_log_std': -1.0
    })

    policy = G1Policy(obs_dim, action_dim, policy_config)

    # Export for PPO
    export_path = Path("checkpoints/bc_warmstart/bc_policy_for_ppo.pth")
    export_path.parent.mkdir(parents=True, exist_ok=True)

    export_data = {
        'policy_state_dict': policy.state_dict(),
        'obs_dim': obs_dim,
        'action_dim': action_dim,
        'policy_config': policy_config,
        'training_info': {
            'bc_epochs': 100,
            'bc_steps': 50000,
            'final_val_loss': 0.008
        },
        'normalizers': {
            'state_normalizer_path': 'data/normalization_stats.pkl',
            'action_normalizer_path': None
        }
    }

    torch.save(export_data, export_path)
    print(f"Exported BC policy to: {export_path}")

    # Show how to load for PPO
    print("\nTo load in PPO training:")
    print(f"checkpoint = torch.load('{export_path}')")
    print("ppo_policy.load_state_dict(checkpoint['policy_state_dict'])")


def example_complete_workflow():
    """
    Example 5: Complete BC warm-start workflow.
    """
    print("\n" + "="*60)
    print("EXAMPLE 5: Complete Workflow")
    print("="*60)

    print("1. Data Preparation:")
    print("   - Retarget human motion clips using motion retargeting pipeline")
    print("   - Save trajectories as NPZ files in data/g1_trajectories/")
    print("   - Generate normalization statistics if needed")

    print("\n2. BC Training:")
    print("   python bc_warmstart.py --config configs/training/bc_warmstart.yaml")

    print("\n3. Monitor Training:")
    print("   - Check TensorBoard logs in logs/bc_warmstart/")
    print("   - Monitor validation loss convergence")
    print("   - Evaluate policy rollouts in environment")

    print("\n4. Export for PPO:")
    print("   python bc_warmstart.py --config configs/training/bc_warmstart.yaml \\")
    print("     --resume checkpoints/bc_warmstart/best_model.pth --export-only")

    print("\n5. Initialize PPO:")
    print("   # In PPO training script:")
    print("   bc_checkpoint = torch.load('checkpoints/bc_warmstart/bc_policy_for_ppo.pth')")
    print("   ppo_policy.load_state_dict(bc_checkpoint['policy_state_dict'])")
    print("   # Then continue with PPO training")

    print("\nExpected BC Performance:")
    print("   - Final MSE loss < 0.01")
    print("   - Validation loss converging")
    print("   - Environment rollouts showing reasonable motion tracking")
    print("   - Policy should provide good warm-start for PPO")


def main():
    """Run all BC warm-start examples."""
    print("Behavior Cloning Warm-start Examples for Unitree G1")
    print("="*60)

    # Run examples
    example_data_inspection()
    example_dataset_creation()
    example_model_training()
    example_policy_export()
    example_complete_workflow()

    print("\n" + "="*60)
    print("Examples completed!")
    print("="*60)
    print("\nNext steps:")
    print("1. Prepare retargeted motion data")
    print("2. Run BC training with: python bc_warmstart.py --config configs/training/bc_warmstart.yaml")
    print("3. Export policy and use for PPO initialization")


if __name__ == "__main__":
    main()