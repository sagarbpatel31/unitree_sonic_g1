#!/usr/bin/env python3
"""
Final Model Evaluation - Test the OptimizedTransformerPolicy from final_enhanced_training.py
Compatible with the latest model architecture and training pipeline.
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

from final_enhanced_training import OptimizedTransformerPolicy, FinalCSVDataset


class FinalModelEvaluator:
    """Evaluation toolkit for the final optimized transformer policy."""

    def __init__(
        self,
        checkpoint_path: str,
        data_path: str,
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

        print(f"✅ Final Model Evaluator initialized")
        print(f"   Device: {self.device}")
        print(f"   Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"   Test sequences: {len(self.test_dataset)}")

    def _load_model(self) -> OptimizedTransformerPolicy:
        """Load trained model from final checkpoint."""
        print(f"Loading final model from: {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        model_info = checkpoint.get('model_info', {})
        obs_dim = model_info.get('obs_dim', 42)
        action_dim = model_info.get('action_dim', 42)
        d_model = model_info.get('d_model', 256)

        # Create model with same architecture as final training
        model = OptimizedTransformerPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            d_model=d_model,
            nhead=8,
            num_layers=4
        )

        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()

        print(f"   Architecture: {d_model}d, {obs_dim}→{action_dim}")
        return model

    def _load_test_data(self) -> FinalCSVDataset:
        """Load test dataset using same format as training."""
        return FinalCSVDataset(
            csv_path=self.data_path,
            sequence_length=32,  # Match training
            train_split=False    # Use validation split for testing
        )

    def compute_accuracy_metrics(self, num_samples: int = 100) -> Dict:
        """Compute comprehensive accuracy metrics."""
        print("🧪 Computing accuracy metrics...")

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

        # Calculate metrics
        num_batches = len(predictions)
        avg_mse = total_mse / num_batches
        avg_mae = total_mae / num_batches
        rmse = np.sqrt(avg_mse)

        # Per-dimension analysis
        all_predictions = np.concatenate(predictions, axis=0)
        all_targets = np.concatenate(targets, axis=0)
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
            "data_shapes": {
                "predictions": all_predictions.shape,
                "targets": all_targets.shape
            }
        }

        print(f"   MSE: {avg_mse:.6f}")
        print(f"   MAE: {avg_mae:.6f}")
        print(f"   RMSE: {rmse:.6f}")
        print(f"   Max Error: {max_error:.6f}")

        return results

    def test_single_action_prediction(self) -> Dict:
        """Test single action prediction using get_action method."""
        print("🎯 Testing single action prediction...")

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
                if i >= 50:  # Test 50 samples
                    break

                observations = batch['observations'].to(self.device)
                actions = batch['actions'].to(self.device)

                # Test single observation prediction
                single_obs = observations[0, -1]  # Last timestep
                true_action = actions[0, -1]

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

    def analyze_action_distribution(self) -> Dict:
        """Analyze the distribution of predicted actions."""
        print("📊 Analyzing action distributions...")

        test_loader = DataLoader(self.test_dataset, batch_size=32, shuffle=False)

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                if i >= 10:  # Analyze 10 batches
                    break

                observations = batch['observations'].to(self.device)
                actions = batch['actions'].to(self.device)

                predicted_actions = self.model(observations)

                all_predictions.append(predicted_actions.cpu().numpy())
                all_targets.append(actions.cpu().numpy())

        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)

        # Flatten for distribution analysis
        pred_flat = predictions.reshape(-1, predictions.shape[-1])
        target_flat = targets.reshape(-1, targets.shape[-1])

        results = {
            "prediction_stats": {
                "mean": np.mean(pred_flat, axis=0).tolist(),
                "std": np.std(pred_flat, axis=0).tolist(),
                "min": np.min(pred_flat, axis=0).tolist(),
                "max": np.max(pred_flat, axis=0).tolist()
            },
            "target_stats": {
                "mean": np.mean(target_flat, axis=0).tolist(),
                "std": np.std(target_flat, axis=0).tolist(),
                "min": np.min(target_flat, axis=0).tolist(),
                "max": np.max(target_flat, axis=0).tolist()
            },
            "distribution_match": {
                "mean_diff": np.mean(np.abs(np.mean(pred_flat, axis=0) - np.mean(target_flat, axis=0))),
                "std_diff": np.mean(np.abs(np.std(pred_flat, axis=0) - np.std(target_flat, axis=0)))
            }
        }

        print(f"   Mean distribution difference: {results['distribution_match']['mean_diff']:.6f}")
        print(f"   Std distribution difference: {results['distribution_match']['std_diff']:.6f}")

        return results

    def create_visualizations(self, save_dir: str = "evaluation_results") -> None:
        """Create evaluation visualizations."""
        print("📈 Creating visualizations...")

        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)

        # Get sample for plotting
        test_loader = DataLoader(self.test_dataset, batch_size=1, shuffle=True)
        sample_batch = next(iter(test_loader))

        observations = sample_batch['observations'].to(self.device)
        true_actions = sample_batch['actions'].to(self.device)

        with torch.no_grad():
            predicted_actions = self.model(observations)

        pred_actions = predicted_actions.squeeze().cpu().numpy()
        true_actions = true_actions.squeeze().cpu().numpy()

        # Plot action predictions for first 6 dimensions
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for i in range(min(6, pred_actions.shape[1])):
            axes[i].plot(true_actions[:, i], label='Ground Truth', linewidth=2)
            axes[i].plot(pred_actions[:, i], label='Predicted', linewidth=2, alpha=0.8)
            axes[i].set_title(f'Action Dimension {i}')
            axes[i].set_xlabel('Time Step')
            axes[i].set_ylabel('Action Value')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path / "action_predictions.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Error distribution
        errors = np.abs(pred_actions - true_actions)
        plt.figure(figsize=(10, 6))
        plt.hist(errors.flatten(), bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Absolute Error')
        plt.ylabel('Frequency')
        plt.title('Prediction Error Distribution')
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path / "error_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"   Visualizations saved to: {save_path}")

    def run_comprehensive_evaluation(self, save_dir: str = "final_evaluation_results") -> Dict:
        """Run complete evaluation suite."""
        print("🔍 Running comprehensive evaluation of final model...")

        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)

        # Load training stats for context
        training_stats = {}
        stats_path = Path(self.checkpoint_path).parent / "training_stats.json"
        if stats_path.exists():
            with open(stats_path, 'r') as f:
                training_stats = json.load(f)

        # Run evaluations
        accuracy_results = self.compute_accuracy_metrics()
        single_step_results = self.test_single_action_prediction()
        distribution_results = self.analyze_action_distribution()

        # Create visualizations
        self.create_visualizations(str(save_path))

        # Combine results
        full_results = {
            "model_info": self.model.get_model_info(),
            "training_stats": training_stats,
            "evaluation_metrics": {
                "accuracy": accuracy_results,
                "single_step": single_step_results,
                "distributions": distribution_results
            },
            "checkpoint_path": self.checkpoint_path,
            "data_path": self.data_path
        }

        # Convert numpy types to Python native types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj

        # Save results
        with open(save_path / "final_evaluation_results.json", 'w') as f:
            json.dump(convert_numpy_types(full_results), f, indent=2)

        # Print summary
        print("\n📋 Final Model Evaluation Summary:")
        print(f"   Model: {full_results['model_info']['model_name']}")
        print(f"   Parameters: {full_results['model_info']['total_parameters']:,}")
        print(f"   Size: {full_results['model_info']['parameter_size_mb']:.1f} MB")
        if training_stats:
            print(f"   Training epochs: {training_stats.get('epochs', 'Unknown')}")
            print(f"   Best validation loss: {training_stats.get('best_val_loss', 'Unknown'):.6f}")
        print(f"   Test MSE: {accuracy_results['overall_metrics']['mse']:.6f}")
        print(f"   Test MAE: {accuracy_results['overall_metrics']['mae']:.6f}")
        print(f"   Single-step MAE: {single_step_results['single_step']['mean_error']:.6f}")
        print(f"   Results saved to: {save_path}")

        return full_results


def main():
    """Main evaluation script."""
    parser = argparse.ArgumentParser(description="Evaluate Final Optimized Transformer Policy")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/final_model/best_checkpoint.pth",
                       help="Path to model checkpoint")
    parser.add_argument("--data", type=str, default="data/lightwheel_bevorg_frames.csv",
                       help="Path to test data")
    parser.add_argument("--save_dir", type=str, default="final_evaluation_results",
                       help="Directory to save results")

    args = parser.parse_args()

    if not Path(args.checkpoint).exists():
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        return

    if not Path(args.data).exists():
        print(f"❌ Data file not found: {args.data}")
        print("Using dummy data for evaluation...")

    # Run evaluation
    evaluator = FinalModelEvaluator(
        checkpoint_path=args.checkpoint,
        data_path=args.data
    )

    results = evaluator.run_comprehensive_evaluation(args.save_dir)

    print("\n🎉 Final model evaluation completed successfully!")


if __name__ == "__main__":
    main()