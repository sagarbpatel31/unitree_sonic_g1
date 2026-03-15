#!/usr/bin/env python3
"""
MuJoCo Interface for Enhanced Transformer Policy.
Allows trained models to be tested in MuJoCo simulation.
"""

import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import torch

try:
    import mujoco
    import mujoco.viewer
    MUJOCO_AVAILABLE = True
except ImportError:
    print("MuJoCo not installed. Install with: pip install mujoco")
    MUJOCO_AVAILABLE = False

from enhanced_transformer_policy import EnhancedTransformerPolicy


class MuJoCoSimulator:
    """MuJoCo simulator interface for policy evaluation."""

    def __init__(
        self,
        model_xml_path: str,
        policy_checkpoint: str,
        normalization_stats: Optional[str] = None,
        use_viewer: bool = True
    ):
        if not MUJOCO_AVAILABLE:
            raise ImportError("MuJoCo is required for simulation")

        self.use_viewer = use_viewer

        # Load MuJoCo model
        print(f"Loading MuJoCo model from: {model_xml_path}")
        self.model = mujoco.MjModel.from_xml_path(model_xml_path)
        self.data = mujoco.MjData(self.model)

        # Initialize viewer if requested
        if self.use_viewer:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        else:
            self.viewer = None

        # Load policy
        print(f"Loading policy from: {policy_checkpoint}")
        self.policy = self._load_policy(policy_checkpoint)

        # Load normalization stats
        self.norm_stats = None
        if normalization_stats and Path(normalization_stats).exists():
            with open(normalization_stats, 'r') as f:
                self.norm_stats = json.load(f)
            print("Loaded normalization statistics")

        # Get model dimensions
        self.obs_dim = self._get_obs_dim()
        self.action_dim = self.model.nu  # Number of actuators

        print(f"Observation dimension: {self.obs_dim}")
        print(f"Action dimension: {self.action_dim}")

        # History for sequence-based policies
        self.obs_history = []
        self.max_history = 64  # Match training sequence length

    def _load_policy(self, checkpoint_path: str) -> EnhancedTransformerPolicy:
        """Load the trained policy from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        model_info = checkpoint.get('model_info', {})
        obs_dim = model_info.get('obs_dim', 42)
        action_dim = model_info.get('action_dim', 42)

        # Create policy with same architecture as training
        policy = EnhancedTransformerPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            d_model=512,
            nhead=8,
            num_layers=8,
            d_ff=2048,
            dropout=0.0  # No dropout during inference
        )

        policy.load_state_dict(checkpoint['model_state_dict'])
        policy.eval()

        print(f"Loaded policy with {sum(p.numel() for p in policy.parameters()):,} parameters")
        return policy

    def _get_obs_dim(self) -> int:
        """Calculate observation dimension based on MuJoCo model."""
        # Standard observations for humanoid robots
        obs_components = []

        # Joint positions (exclude root joints for floating base)
        if self.model.nq > 0:
            obs_components.append(self.model.nq - 7)  # Exclude root position and quaternion

        # Joint velocities
        if self.model.nv > 0:
            obs_components.append(self.model.nv - 6)  # Exclude root linear/angular velocity

        # Body positions and orientations
        obs_components.append(self.model.nbody * 3)  # Positions
        obs_components.append(self.model.nbody * 4)  # Quaternions

        # Actuator forces
        obs_components.append(self.model.nu)

        # Contact forces (simplified)
        obs_components.append(self.model.nbody * 3)

        total_obs_dim = sum(obs_components)
        print(f"Calculated observation dimension: {total_obs_dim}")
        return total_obs_dim

    def _get_observation(self) -> np.ndarray:
        """Extract observation from current MuJoCo state."""
        obs = []

        # Joint positions (skip root for floating base)
        if self.model.nq > 7:
            obs.extend(self.data.qpos[7:])  # Skip root position and quaternion

        # Joint velocities (skip root)
        if self.model.nv > 6:
            obs.extend(self.data.qvel[6:])  # Skip root linear and angular velocity

        # Body positions
        obs.extend(self.data.xpos.flatten())

        # Body orientations (quaternions)
        obs.extend(self.data.xquat.flatten())

        # Actuator forces
        obs.extend(self.data.actuator_force)

        # Contact forces (simplified - body contact forces)
        for i in range(self.model.nbody):
            contact_force = np.zeros(3)
            for j in range(self.data.ncon):
                contact = self.data.contact[j]
                if contact.geom1 // self.model.ngeom_body == i or contact.geom2 // self.model.ngeom_body == i:
                    contact_force += contact.f[:3]
            obs.extend(contact_force)

        # Pad or truncate to expected dimension
        obs_array = np.array(obs, dtype=np.float32)
        if len(obs_array) > self.obs_dim:
            obs_array = obs_array[:self.obs_dim]
        elif len(obs_array) < self.obs_dim:
            padding = np.zeros(self.obs_dim - len(obs_array), dtype=np.float32)
            obs_array = np.concatenate([obs_array, padding])

        return obs_array

    def _normalize_observation(self, obs: np.ndarray) -> np.ndarray:
        """Apply normalization to observation if available."""
        if self.norm_stats is None:
            return obs

        obs_mean = np.array(self.norm_stats['obs_mean'], dtype=np.float32)
        obs_std = np.array(self.norm_stats['obs_std'], dtype=np.float32)

        return (obs - obs_mean) / obs_std

    def _denormalize_action(self, action: np.ndarray) -> np.ndarray:
        """Remove normalization from action if applied during training."""
        if self.norm_stats is None:
            return action

        action_mean = np.array(self.norm_stats['action_mean'], dtype=np.float32)
        action_std = np.array(self.norm_stats['action_std'], dtype=np.float32)

        return action * action_std + action_mean

    def get_policy_action(self) -> np.ndarray:
        """Get action from policy based on current state."""
        # Get current observation
        obs = self._get_observation()
        obs = self._normalize_observation(obs)

        # Maintain observation history for sequence-based policy
        self.obs_history.append(obs)
        if len(self.obs_history) > self.max_history:
            self.obs_history.pop(0)

        # Pad history if needed
        if len(self.obs_history) < self.max_history:
            # Repeat first observation for initial padding
            padding_obs = [self.obs_history[0]] * (self.max_history - len(self.obs_history))
            obs_sequence = np.array(padding_obs + self.obs_history)
        else:
            obs_sequence = np.array(self.obs_history)

        # Convert to torch tensor and get action
        obs_tensor = torch.FloatTensor(obs_sequence).unsqueeze(0)  # [1, seq_len, obs_dim]

        with torch.no_grad():
            action_tensor = self.policy(obs_tensor)
            action = action_tensor.squeeze().cpu().numpy()  # Get last timestep action

        # Take the last timestep's action
        if action.ndim > 1:
            action = action[-1]  # Last timestep

        # Denormalize action
        action = self._denormalize_action(action)

        # Clip action to valid range for MuJoCo
        action = np.clip(action, -1.0, 1.0)

        return action

    def reset(self) -> np.ndarray:
        """Reset simulation to initial state."""
        mujoco.mj_resetData(self.model, self.data)
        self.obs_history.clear()

        # Add small random noise to initial state for diversity
        if self.model.nq > 0:
            noise_scale = 0.01
            self.data.qpos += np.random.normal(0, noise_scale, self.data.qpos.shape)

        # Forward kinematics to update derived quantities
        mujoco.mj_forward(self.model, self.data)

        return self._get_observation()

    def step(self, action: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float, bool, Dict]:
        """Step simulation forward one timestep."""
        if action is None:
            action = self.get_policy_action()

        # Ensure action is correct size
        if len(action) != self.action_dim:
            if len(action) > self.action_dim:
                action = action[:self.action_dim]
            else:
                # Pad with zeros
                padded_action = np.zeros(self.action_dim)
                padded_action[:len(action)] = action
                action = padded_action

        # Apply action
        self.data.ctrl[:] = action

        # Step simulation
        mujoco.mj_step(self.model, self.data)

        # Update viewer
        if self.viewer is not None:
            self.viewer.sync()

        # Get new observation
        obs = self._get_observation()

        # Simple reward (staying upright)
        reward = 1.0 if self.data.qpos[2] > 0.5 else 0.0  # Height check

        # Check if simulation should terminate
        done = self.data.time > 10.0 or self.data.qpos[2] < 0.3  # Time limit or fall

        info = {
            'time': self.data.time,
            'height': self.data.qpos[2],
            'action': action.copy()
        }

        return obs, reward, done, info

    def run_episode(self, max_steps: int = 1000, render_fps: int = 30) -> Dict:
        """Run a complete episode with policy control."""
        obs = self.reset()
        total_reward = 0.0
        steps = 0

        episode_data = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'total_reward': 0.0,
            'steps': 0,
            'success': False
        }

        print("🚀 Starting episode...")

        for step in range(max_steps):
            obs, reward, done, info = self.step()

            episode_data['observations'].append(obs.copy())
            episode_data['actions'].append(info['action'])
            episode_data['rewards'].append(reward)

            total_reward += reward
            steps += 1

            # Control rendering frequency
            if self.viewer is not None and step % (60 // render_fps) == 0:
                time.sleep(1.0 / render_fps)

            if done:
                break

        episode_data['total_reward'] = total_reward
        episode_data['steps'] = steps
        episode_data['success'] = steps >= max_steps * 0.5  # Success if ran for reasonable time

        print(f"Episode completed: {steps} steps, reward: {total_reward:.2f}")
        return episode_data

    def close(self):
        """Close viewer and cleanup."""
        if self.viewer is not None:
            self.viewer.close()


def demo_simulation():
    """Demo function showing how to use the MuJoCo interface."""
    # Example usage - you'll need to provide actual paths
    model_xml = "path/to/your/robot.xml"  # Your robot URDF/XML
    checkpoint = "checkpoints/enhanced_model/best_checkpoint.pth"
    norm_stats = "checkpoints/enhanced_model/normalization_stats.json"

    if not Path(checkpoint).exists():
        print("❌ Checkpoint not found. Train the model first!")
        return

    try:
        # Create simulator
        sim = MuJoCoSimulator(
            model_xml_path=model_xml,
            policy_checkpoint=checkpoint,
            normalization_stats=norm_stats,
            use_viewer=True
        )

        # Run multiple episodes
        for episode in range(3):
            print(f"\n🎮 Episode {episode + 1}")
            result = sim.run_episode(max_steps=1000)

            print(f"Result: {result['steps']} steps, reward: {result['total_reward']:.2f}")

            if not result['success']:
                print("Episode ended early - robot may have fallen")

        sim.close()

    except Exception as e:
        print(f"❌ Simulation error: {e}")


if __name__ == "__main__":
    demo_simulation()