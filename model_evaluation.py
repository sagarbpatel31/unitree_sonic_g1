#!/usr/bin/env python3
"""
Model evaluation utilities for Enhanced Transformer Policy.
Comprehensive testing and analysis tools.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

from enhanced_transformer_policy import EnhancedTransformerPolicy
from advanced_training import AdvancedCSVMotionDataset


class PolicyEvaluator:
    """Comprehensive policy evaluation toolkit."""

    def __init__(
        self,
        checkpoint_path: str,
        data_path: str,
        normalization_stats: Optional[str] = None,
        device: str = "auto"
    ):
        self.checkpoint_path = checkpoint_path
        self.data_path = data_path

        # Device setup
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load model and data
        self.model = self._load_model()
        self.test_dataset = self._load_test_data()

        # Load normalization stats
        self.norm_stats = None
        if normalization_stats and Path(normalization_stats).exists():
            with open(normalization_stats, 'r') as f:
                self.norm_stats = json.load(f)

        print(f"✅ Evaluator initialized")
        print(f"   Device: {self.device}")
        print(f"   Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"   Test sequences: {len(self.test_dataset)}")

    def _load_model(self) -> EnhancedTransformerPolicy:
        """Load trained model from checkpoint."""
        print(f"Loading model from: {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        model_info = checkpoint.get('model_info', {})
        obs_dim = model_info.get('obs_dim', 42)
        action_dim = model_info.get('action_dim', 42)

        model = EnhancedTransformerPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            d_model=512,
            nhead=8,
            num_layers=8,
            d_ff=2048,
            dropout=0.0  # No dropout for evaluation
        )

        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()

        return model

    def _load_test_data(self) -> AdvancedCSVMotionDataset:
        """Load test dataset."""
        return AdvancedCSVMotionDataset(
            csv_path=self.data_path,
            sequence_length=64,
            train_split=False,  # Use validation split for testing
            normalize=True,
            augment=False  # No augmentation for testing
        )

    def compute_prediction_accuracy(self, num_samples: int = 100) -> Dict:
        """Compute prediction accuracy metrics."""
        print("🧪 Computing prediction accuracy...")

        test_loader = DataLoader(
            self.test_dataset,
            batch_size=16,
            shuffle=False,
            num_workers=0
        )

        total_mse = 0.0
        total_mae = 0.0
        max_error = 0.0
        predictions = []
        targets = []
        samples_processed = 0

        with torch.no_grad():
            for batch in test_loader:
                if samples_processed >= num_samples:
                    break

                observations = batch['observations'].to(self.device)
                actions = batch['actions'].to(self.device)

                # Forward pass
                predicted_actions = self.model(observations)

                # Compute metrics
                mse = nn.MSELoss()(predicted_actions, actions)
                mae = nn.L1Loss()(predicted_actions, actions)

                total_mse += mse.item()
                total_mae += mae.item()

                # Track max error
                batch_max_error = torch.max(torch.abs(predicted_actions - actions)).item()
                max_error = max(max_error, batch_max_error)

                # Store predictions for detailed analysis
                predictions.append(predicted_actions.cpu().numpy())
                targets.append(actions.cpu().numpy())

                samples_processed += observations.size(0)

        # Calculate average metrics
        num_batches = len([p for p in predictions])
        avg_mse = total_mse / num_batches
        avg_mae = total_mae / num_batches
        rmse = np.sqrt(avg_mse)

        # Detailed analysis
        all_predictions = np.concatenate(predictions, axis=0)
        all_targets = np.concatenate(targets, axis=0)

        # Per-dimension analysis
        per_dim_mse = np.mean((all_predictions - all_targets) ** 2, axis=(0, 1))
        per_dim_mae = np.mean(np.abs(all_predictions - all_targets), axis=(0, 1))

        results = {
            "overall_metrics": {
                "mse": avg_mse,
                "mae": avg_mae,
                "rmse": rmse,
                "max_error": max_error,
                "samples_evaluated": samples_processed
            },
            "per_dimension": {
                "mse": per_dim_mse.tolist(),
                "mae": per_dim_mae.tolist()
            },
            "predictions_shape": all_predictions.shape,
            "targets_shape": all_targets.shape
        }

        print(f"   MSE: {avg_mse:.6f}")
        print(f"   MAE: {avg_mae:.6f}")
        print(f"   RMSE: {rmse:.6f}")
        print(f"   Max Error: {max_error:.6f}")

        return results

    def analyze_temporal_consistency(self, num_sequences: int = 20) -> Dict:
        """Analyze temporal consistency of predictions."""
        print("⏰ Analyzing temporal consistency...")

        temporal_errors = []
        velocity_errors = []

        test_loader = DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0
        )

        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                if i >= num_sequences:
                    break

                observations = batch['observations'].to(self.device)  # [1, seq_len, obs_dim]
                actions = batch['actions'].to(self.device)  # [1, seq_len, action_dim]

                predicted_actions = self.model(observations)

                # Convert to numpy for analysis
                pred_seq = predicted_actions.squeeze().cpu().numpy()  # [seq_len, action_dim]
                true_seq = actions.squeeze().cpu().numpy()

                # Compute temporal differences (velocities)
                pred_vel = np.diff(pred_seq, axis=0)
                true_vel = np.diff(true_seq, axis=0)

                # Temporal smoothness error
                temporal_error = np.mean(np.abs(pred_vel - true_vel))
                temporal_errors.append(temporal_error)

                # Action velocity magnitude comparison
                pred_vel_mag = np.mean(np.linalg.norm(pred_vel, axis=1))
                true_vel_mag = np.mean(np.linalg.norm(true_vel, axis=1))
                velocity_errors.append(abs(pred_vel_mag - true_vel_mag))

        results = {
            "temporal_smoothness_error": {
                "mean": np.mean(temporal_errors),
                "std": np.std(temporal_errors),
                "max": np.max(temporal_errors),
                "min": np.min(temporal_errors)
            },
            "velocity_magnitude_error": {
                "mean": np.mean(velocity_errors),
                "std": np.std(velocity_errors),
                "max": np.max(velocity_errors),
                "min": np.min(velocity_errors)
            },
            "sequences_analyzed": len(temporal_errors)
        }

        print(f"   Temporal smoothness error: {np.mean(temporal_errors):.6f}")
        print(f"   Velocity magnitude error: {np.mean(velocity_errors):.6f}")

        return results

    def test_single_step_prediction(self) -> Dict:
        """Test single-step prediction accuracy."""
        print("🎯 Testing single-step prediction...")

        # Use model's get_action method for single predictions
        single_errors = []
        sequence_errors = []

        test_loader = DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=0
        )

        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                if i >= 50:  # Test on 50 samples
                    break

                observations = batch['observations'].to(self.device)  # [1, seq_len, obs_dim]
                actions = batch['actions'].to(self.device)  # [1, seq_len, action_dim]

                # Test single observation (last timestep)
                single_obs = observations[0, -1]  # [obs_dim]
                true_action = actions[0, -1]  # [action_dim]

                predicted_action = self.model.get_action(single_obs)
                single_error = torch.mean(torch.abs(predicted_action - true_action)).item()
                single_errors.append(single_error)

                # Test full sequence
                predicted_sequence = self.model(observations)
                sequence_error = torch.mean(torch.abs(predicted_sequence - actions)).item()
                sequence_errors.append(sequence_error)

        results = {
            "single_step": {
                "mean_error": np.mean(single_errors),
                "std_error": np.std(single_errors),
                "max_error": np.max(single_errors)
            },
            "sequence_prediction": {
                "mean_error": np.mean(sequence_errors),
                "std_error": np.std(sequence_errors),
                "max_error": np.max(sequence_errors)
            },
            "samples_tested": len(single_errors)
        }

        print(f"   Single-step MAE: {np.mean(single_errors):.6f}")
        print(f"   Sequence MAE: {np.mean(sequence_errors):.6f}")

        return results

    def visualize_predictions(self, save_dir: str = "evaluation_plots") -> None:
        """Create visualization plots of model predictions."""
        print("📊 Creating prediction visualizations...")

        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)

        # Get a sample sequence for visualization
        test_loader = DataLoader(self.test_dataset, batch_size=1, shuffle=True)
        sample_batch = next(iter(test_loader))

        observations = sample_batch['observations'].to(self.device)
        true_actions = sample_batch['actions'].to(self.device)

        with torch.no_grad():
            predicted_actions = self.model(observations)

        # Convert to numpy
        pred_actions = predicted_actions.squeeze().cpu().numpy()  # [seq_len, action_dim]
        true_actions = true_actions.squeeze().cpu().numpy()

        # Plot first 6 action dimensions
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for i in range(min(6, pred_actions.shape[1])):
            axes[i].plot(true_actions[:, i], label='True', linewidth=2)
            axes[i].plot(pred_actions[:, i], label='Predicted', linewidth=2, alpha=0.8)
            axes[i].set_title(f'Action Dimension {i}')
            axes[i].set_xlabel('Time Step')
            axes[i].set_ylabel('Action Value')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path / "action_predictions.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Error distribution plot
        errors = np.abs(pred_actions - true_actions)
        plt.figure(figsize=(10, 6))
        plt.hist(errors.flatten(), bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Absolute Error')
        plt.ylabel('Frequency')
        plt.title('Prediction Error Distribution')
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path / "error_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Per-dimension error boxplot
        plt.figure(figsize=(12, 6))
        error_per_dim = [errors[:, i] for i in range(min(20, errors.shape[1]))]
        plt.boxplot(error_per_dim, labels=[f'Dim {i}' for i in range(len(error_per_dim))])
        plt.xlabel('Action Dimension')
        plt.ylabel('Absolute Error')
        plt.title('Prediction Error by Action Dimension')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path / "error_by_dimension.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"   Plots saved to: {save_path}")

    def run_comprehensive_evaluation(self, save_dir: str = "evaluation_results") -> Dict:
        """Run complete evaluation suite."""
        print("🔍 Running comprehensive evaluation...")

        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)

        # Run all tests
        accuracy_results = self.compute_prediction_accuracy()
        temporal_results = self.analyze_temporal_consistency()
        single_step_results = self.test_single_step_prediction()

        # Create visualizations
        self.visualize_predictions(str(save_path / "plots"))

        # Combine all results
        full_results = {
            "model_info": self.model.get_model_info(),
            "evaluation_date": str(Path().cwd()),
            "accuracy_metrics": accuracy_results,
            "temporal_analysis": temporal_results,
            "single_step_analysis": single_step_results
        }

        # Save results
        with open(save_path / "evaluation_results.json", 'w') as f:
            json.dump(full_results, f, indent=2)

        # Print summary
        print("\n📋 Evaluation Summary:")
        print(f"   Model: {full_results['model_info']['model_name']}")
        print(f"   Parameters: {full_results['model_info']['total_parameters']:,}")
        print(f"   MSE: {accuracy_results['overall_metrics']['mse']:.6f}")
        print(f"   MAE: {accuracy_results['overall_metrics']['mae']:.6f}")
        print(f"   Temporal Error: {temporal_results['temporal_smoothness_error']['mean']:.6f}")
        print(f"   Results saved to: {save_path}")

        return full_results


def main():
    """Main evaluation script."""
    parser = argparse.ArgumentParser(description="Evaluate Enhanced Transformer Policy")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--data", type=str, default="data/lightwheel_bevorg_frames.csv",
                       help="Path to test data")
    parser.add_argument("--norm_stats", type=str, default=None,
                       help="Path to normalization statistics")
    parser.add_argument("--save_dir", type=str, default="evaluation_results",
                       help="Directory to save results")

    args = parser.parse_args()

    if not Path(args.checkpoint).exists():
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        return

    if not Path(args.data).exists():
        print(f"❌ Data file not found: {args.data}")
        return

    # Run evaluation
    evaluator = PolicyEvaluator(
        checkpoint_path=args.checkpoint,
        data_path=args.data,
        normalization_stats=args.norm_stats
    )

    results = evaluator.run_comprehensive_evaluation(args.save_dir)

    print("\n🎉 Evaluation completed successfully!")


if __name__ == "__main__":
    main()