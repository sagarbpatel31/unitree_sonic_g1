#!/usr/bin/env python3
"""
Quick Mac Test Script - Easy way to test your latest trained model
"""

import sys
from pathlib import Path
from mujoco_mac_demo import MacMuJoCoPolicyInterface, create_mac_mujoco_simulation

def find_latest_checkpoint():
    """Find the most recently trained checkpoint."""
    checkpoint_dirs = [
        "checkpoints/enhanced_mac_model",
        "checkpoints/mac_training",
        "checkpoints/final_model"
    ]

    for checkpoint_dir in checkpoint_dirs:
        checkpoint_path = Path(checkpoint_dir) / "best_checkpoint.pth"
        if checkpoint_path.exists():
            return str(checkpoint_path)

    return None

def quick_test(checkpoint_path=None):
    """Run a quick test of the trained model."""
    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint()

    if checkpoint_path is None:
        print("❌ No trained model found!")
        print("\n🚀 Train a model first:")
        print("   python3 mac_optimized_training.py --epochs 30")
        return

    print(f"🎯 Testing model: {checkpoint_path}")

    try:
        # Load and test the model
        policy = MacMuJoCoPolicyInterface(checkpoint_path)

        # Quick performance test
        import numpy as np
        import time

        obs = np.random.randn(policy.obs_dim).astype(np.float32)

        # Warm up
        for _ in range(5):
            policy.get_action(obs)

        # Time inference
        times = []
        for _ in range(100):
            start = time.time()
            action = policy.get_action(obs)
            times.append(time.time() - start)

        avg_time = np.mean(times) * 1000

        print(f"\n📊 Quick Performance Test:")
        print(f"   ✅ Model loaded successfully")
        print(f"   ✅ Inference time: {avg_time:.2f}ms")
        print(f"   ✅ FPS capability: {1000/avg_time:.0f}")
        print(f"   ✅ Action range: [{action.min():.3f}, {action.max():.3f}]")

        print(f"\n🎮 Quick test completed successfully!")
        print(f"   Run full demo: python3 mujoco_mac_demo.py")
        print(f"   Train larger model: python3 mac_optimized_training.py --epochs 50 --d_model 512")

    except Exception as e:
        print(f"❌ Error testing model: {e}")

if __name__ == "__main__":
    # Check if checkpoint path provided
    checkpoint = sys.argv[1] if len(sys.argv) > 1 else None
    quick_test(checkpoint)