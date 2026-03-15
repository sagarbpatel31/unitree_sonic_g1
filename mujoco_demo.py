#!/usr/bin/env python3
"""
MuJoCo Simulation Demo Script
Demonstrates how to use the trained model with MuJoCo simulation.
"""

import json
import torch
import numpy as np
from pathlib import Path

# Import our trained model
from final_enhanced_training import OptimizedTransformerPolicy


class MuJoCoPolicyInterface:
    """Interface to use trained policy with MuJoCo."""

    def __init__(self, checkpoint_path: str):
        """Load trained model from checkpoint."""
        print(f"Loading model from: {checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model_info = checkpoint['model_info']

        # Create model with same architecture
        self.model = OptimizedTransformerPolicy(
            obs_dim=model_info['obs_dim'],
            action_dim=model_info['action_dim'],
            d_model=256,  # Match training config
            nhead=8,
            num_layers=4
        )

        # Load trained weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Store dimensions
        self.obs_dim = model_info['obs_dim']
        self.action_dim = model_info['action_dim']

        # History for sequence-based policy
        self.obs_history = []
        self.max_history = 32  # Match training sequence length

        print(f"✅ Model loaded successfully!")
        print(f"   Parameters: {model_info['total_parameters']:,}")
        print(f"   Obs dim: {self.obs_dim}, Action dim: {self.action_dim}")

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """
        Get action from policy given current observation.

        Args:
            observation: Current observation [obs_dim]

        Returns:
            action: Action to take [action_dim]
        """
        # Ensure observation is correct shape
        if len(observation) != self.obs_dim:
            # Pad or truncate as needed
            if len(observation) > self.obs_dim:
                observation = observation[:self.obs_dim]
            else:
                padded_obs = np.zeros(self.obs_dim)
                padded_obs[:len(observation)] = observation
                observation = padded_obs

        # Add to history
        self.obs_history.append(observation.astype(np.float32))

        # Maintain sequence length
        if len(self.obs_history) > self.max_history:
            self.obs_history.pop(0)

        # Pad history if needed
        if len(self.obs_history) < self.max_history:
            padding_obs = [self.obs_history[0]] * (self.max_history - len(self.obs_history))
            obs_sequence = np.array(padding_obs + self.obs_history)
        else:
            obs_sequence = np.array(self.obs_history)

        # Convert to tensor and get action
        obs_tensor = torch.FloatTensor(obs_sequence).unsqueeze(0)  # [1, seq_len, obs_dim]

        with torch.no_grad():
            action_tensor = self.model(obs_tensor)
            # Take the last timestep's action
            action = action_tensor[0, -1].cpu().numpy()  # [action_dim]

        # Clip action to safe range for MuJoCo
        action = np.clip(action, -1.0, 1.0)

        return action

    def reset(self):
        """Reset the policy state."""
        self.obs_history.clear()

    def get_model_info(self) -> dict:
        """Get model information."""
        return {
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "model_type": "OptimizedTransformerPolicy",
            "sequence_length": self.max_history
        }


def create_dummy_mujoco_simulation():
    """
    Demo function showing how to integrate with MuJoCo.
    Replace this with actual MuJoCo simulation code.
    """
    checkpoint_path = "checkpoints/final_model/best_checkpoint.pth"

    if not Path(checkpoint_path).exists():
        print("❌ Model checkpoint not found!")
        print("Train the model first with: python3 final_enhanced_training.py")
        return

    # Create policy interface
    policy = MuJoCoPolicyInterface(checkpoint_path)

    print("\n🎮 Running simulation demo...")

    # Simulate episode
    policy.reset()
    total_reward = 0.0

    for step in range(100):
        # Get dummy observation (replace with actual MuJoCo observation)
        observation = np.random.randn(policy.obs_dim).astype(np.float32)

        # Get action from policy
        action = policy.get_action(observation)

        # Apply action to MuJoCo simulation (replace with actual MuJoCo step)
        # reward, done = mujoco_env.step(action)

        # Dummy reward calculation
        reward = 1.0 - np.sum(np.abs(action)) * 0.1  # Encourage smaller actions
        total_reward += reward

        # Print progress
        if step % 20 == 0:
            print(f"   Step {step}: Action range [{action.min():.3f}, {action.max():.3f}], Reward: {reward:.3f}")

        # Dummy termination condition
        if step > 80 and np.random.random() < 0.1:
            break

    print(f"\n📊 Episode completed!")
    print(f"   Steps: {step + 1}")
    print(f"   Total reward: {total_reward:.2f}")
    print(f"   Average reward: {total_reward / (step + 1):.3f}")

    return total_reward


def mujoco_integration_example():
    """
    Example of how to integrate with actual MuJoCo simulation.
    """
    print("\n" + "=" * 60)
    print("MuJoCo Integration Example")
    print("=" * 60)

    print("""
To use this model with MuJoCo simulation:

1. Install MuJoCo:
   pip install mujoco

2. Create your robot XML/URDF file

3. Use the MuJoCoPolicyInterface like this:

```python
import mujoco
import mujoco.viewer

# Load your robot model
model = mujoco.MjModel.from_xml_path("your_robot.xml")
data = mujoco.MjData(model)

# Load trained policy
policy = MuJoCoPolicyInterface("checkpoints/final_model/best_checkpoint.pth")

# Create viewer (optional)
viewer = mujoco.viewer.launch_passive(model, data)

# Simulation loop
policy.reset()
for step in range(1000):
    # Extract observation from MuJoCo state
    observation = np.concatenate([
        data.qpos[7:],  # joint positions (skip root)
        data.qvel[6:],  # joint velocities (skip root)
        # Add other relevant observations
    ])

    # Get action from policy
    action = policy.get_action(observation)

    # Apply action to MuJoCo
    data.ctrl[:] = action

    # Step simulation
    mujoco.mj_step(model, data)

    # Update viewer
    if viewer:
        viewer.sync()
```

4. Key considerations:
   - Match observation dimension (currently {obs_dim})
   - Match action dimension (currently {action_dim})
   - Actions are bounded to [-1, 1]
   - Policy expects sequence of observations
   - Reset policy state between episodes
""".format(obs_dim=42, action_dim=42))


def main():
    """Main demo function."""
    print("🚀 MuJoCo Integration Demo")

    # Run dummy simulation
    create_dummy_mujoco_simulation()

    # Show integration example
    mujoco_integration_example()

    print("\n✅ Demo completed!")
    print("\nYour trained model is ready for MuJoCo simulation!")


if __name__ == "__main__":
    main()