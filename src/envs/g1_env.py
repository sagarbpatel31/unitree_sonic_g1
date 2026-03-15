"""
Base MuJoCo environment for Unitree G1 humanoid robot.
Provides core simulation functionality and interfaces.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

import mujoco
import gymnasium as gym
from gymnasium import spaces

from ..core.config import Config
from ..core.utils import normalize_angle, quaternion_to_euler


class G1Environment(gym.Env):
    """
    Base MuJoCo environment for Unitree G1 humanoid robot.

    This class provides the core simulation functionality:
    - MuJoCo physics simulation
    - Robot state management
    - Basic observation/action spaces
    - Safety monitoring
    """

    def __init__(self, config: Config):
        super().__init__()

        self.config = config
        self.env_config = config.env
        self.robot_config = config.env.robot

        # Load MuJoCo model
        self.model_path = Path(self.robot_config.model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"MuJoCo model not found: {self.model_path}")

        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)

        # Environment parameters
        self.dt = self.model.opt.timestep
        self.frame_skip = self.env_config.get("frame_skip", 10)
        self.max_episode_steps = self.env_config.get("max_episode_steps", 1000)

        # Robot configuration
        self.init_qpos = self.data.qpos.copy()
        self.init_qvel = self.data.qvel.copy()

        # Get joint information
        self._setup_joint_info()

        # Define spaces
        self._setup_spaces()

        # State tracking
        self.step_count = 0
        self.episode_reward = 0.0

        # Safety monitoring
        self.safety_config = self.env_config.get("safety", {})

    def _setup_joint_info(self):
        """Setup joint information and control mappings."""
        # Get position-controlled joints
        controlled_joints = self.robot_config.get("joints", {}).get("position_controlled", [])

        self.actuated_joint_ids = []
        self.actuated_joint_names = []

        for joint_name in controlled_joints:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id >= 0:
                self.actuated_joint_ids.append(joint_id)
                self.actuated_joint_names.append(joint_name)
            else:
                print(f"Warning: Joint '{joint_name}' not found in model")

        self.num_actuated_joints = len(self.actuated_joint_ids)

        # Joint limits
        self.joint_pos_limits = []
        self.joint_vel_limits = []

        for joint_id in self.actuated_joint_ids:
            # Position limits
            pos_range = self.model.jnt_range[joint_id]
            if np.allclose(pos_range, 0):
                # No limits specified, use default
                pos_range = [-np.pi, np.pi]
            self.joint_pos_limits.append(pos_range)

            # Velocity limits
            vel_limit = self.robot_config.get("joints", {}).get("velocity_limits", {}).get("default", [-10, 10])
            self.joint_vel_limits.append(vel_limit)

        self.joint_pos_limits = np.array(self.joint_pos_limits)
        self.joint_vel_limits = np.array(self.joint_vel_limits)

        # Control parameters
        control_config = self.robot_config.get("joints", {})
        self.kp = control_config.get("control_gains", {}).get("kp", 100.0)
        self.kd = control_config.get("control_gains", {}).get("kd", 10.0)
        self.max_torque = control_config.get("max_torque", 100.0)

    def _setup_spaces(self):
        """Setup observation and action spaces."""
        # Action space: normalized joint positions [-1, 1]
        action_dim = self.num_actuated_joints
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32
        )

        # Observation space will be defined by subclasses
        # Base observations include:
        obs_dim = (
            self.num_actuated_joints * 2  # joint pos + vel
            + 4  # base orientation (quaternion)
            + 3  # base angular velocity
            + 3  # base linear acceleration
        )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state."""
        super().reset(seed=seed)

        # Reset simulation
        mujoco.mj_resetData(self.model, self.data)

        # Set initial pose with noise
        init_noise = self.robot_config.get("init_noise", 0.1)
        noise_scale = self.np_random.uniform(-init_noise, init_noise, size=self.data.qpos.shape)

        self.data.qpos[:] = self.init_qpos + noise_scale
        self.data.qvel[:] = self.init_qvel

        # Ensure joint limits
        for i, joint_id in enumerate(self.actuated_joint_ids):
            qpos_idx = self.model.jnt_qposadr[joint_id]
            self.data.qpos[qpos_idx] = np.clip(
                self.data.qpos[qpos_idx],
                self.joint_pos_limits[i, 0],
                self.joint_pos_limits[i, 1]
            )

        # Forward dynamics
        mujoco.mj_forward(self.model, self.data)

        # Reset counters
        self.step_count = 0
        self.episode_reward = 0.0

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one environment step."""
        # Process action
        target_positions = self._process_action(action)

        # Apply control
        for _ in range(self.frame_skip):
            self._apply_control(target_positions)
            mujoco.mj_step(self.model, self.data)

        # Get observation
        observation = self._get_observation()

        # Compute reward
        reward = self._compute_reward(action)

        # Check termination
        terminated = self._is_terminated()
        truncated = self.step_count >= self.max_episode_steps

        # Update counters
        self.step_count += 1
        self.episode_reward += reward

        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _process_action(self, action: np.ndarray) -> np.ndarray:
        """Process normalized action to target joint positions."""
        # Clip action
        action = np.clip(action, -1.0, 1.0)

        # Scale to joint limits
        action_range = self.joint_pos_limits[:, 1] - self.joint_pos_limits[:, 0]
        action_center = (self.joint_pos_limits[:, 1] + self.joint_pos_limits[:, 0]) / 2

        target_positions = action_center + action * action_range / 2

        return target_positions

    def _apply_control(self, target_positions: np.ndarray):
        """Apply PD control to reach target joint positions."""
        for i, joint_id in enumerate(self.actuated_joint_ids):
            qpos_idx = self.model.jnt_qposadr[joint_id]
            qvel_idx = self.model.jnt_dofadr[joint_id]

            # Current state
            current_pos = self.data.qpos[qpos_idx]
            current_vel = self.data.qvel[qvel_idx]

            # PD control
            pos_error = target_positions[i] - current_pos
            vel_error = 0.0 - current_vel  # Target velocity is 0

            torque = self.kp * pos_error + self.kd * vel_error
            torque = np.clip(torque, -self.max_torque, self.max_torque)

            # Apply torque
            actuator_id = i  # Assuming actuators are in same order as joints
            if actuator_id < self.model.nu:
                self.data.ctrl[actuator_id] = torque

    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        obs_parts = []

        # Joint positions and velocities
        joint_positions = []
        joint_velocities = []

        for joint_id in self.actuated_joint_ids:
            qpos_idx = self.model.jnt_qposadr[joint_id]
            qvel_idx = self.model.jnt_dofadr[joint_id]

            joint_positions.append(self.data.qpos[qpos_idx])
            joint_velocities.append(self.data.qvel[qvel_idx])

        obs_parts.extend(joint_positions)
        obs_parts.extend(joint_velocities)

        # Base orientation (quaternion)
        base_quat = self.data.qpos[3:7]  # Assuming floating base
        obs_parts.extend(base_quat)

        # Base angular velocity
        base_angvel = self.data.qvel[3:6]  # Assuming floating base
        obs_parts.extend(base_angvel)

        # Base linear acceleration (from IMU)
        base_linacc = self.data.qacc[:3]  # Assuming floating base
        obs_parts.extend(base_linacc)

        observation = np.array(obs_parts, dtype=np.float32)

        # Add noise if specified
        obs_noise = self.env_config.get("observation_noise", 0.0)
        if obs_noise > 0:
            noise = self.np_random.normal(0, obs_noise, size=observation.shape)
            observation += noise

        return observation

    def _compute_reward(self, action: np.ndarray) -> float:
        """Compute reward (to be implemented by subclasses)."""
        # Base reward: small alive bonus
        reward = 0.1

        # Energy penalty
        energy_penalty = np.sum(np.square(action)) * 0.001
        reward -= energy_penalty

        return reward

    def _is_terminated(self) -> bool:
        """Check if episode should be terminated."""
        # Check base height
        base_height = self.data.qpos[2]  # Assuming floating base
        min_height = self.safety_config.get("termination", {}).get("base_height_min", 0.3)
        max_height = self.safety_config.get("termination", {}).get("base_height_max", 1.5)

        if base_height < min_height or base_height > max_height:
            return True

        # Check for self-collision (if enabled)
        if self.robot_config.get("self_collision", False):
            if self.data.ncon > 0:
                for i in range(self.data.ncon):
                    contact = self.data.contact[i]
                    # Check if both bodies belong to robot (simple check)
                    if contact.geom1 != contact.geom2:  # Different geoms in contact
                        return True

        return False

    def _get_info(self) -> Dict[str, Any]:
        """Get additional information."""
        base_pos = self.data.qpos[:3]
        base_vel = self.data.qvel[:3]

        return {
            "step_count": self.step_count,
            "episode_reward": self.episode_reward,
            "base_position": base_pos.copy(),
            "base_velocity": base_vel.copy(),
            "base_height": self.data.qpos[2],
        }

    def get_joint_positions(self) -> np.ndarray:
        """Get current joint positions."""
        positions = []
        for joint_id in self.actuated_joint_ids:
            qpos_idx = self.model.jnt_qposadr[joint_id]
            positions.append(self.data.qpos[qpos_idx])
        return np.array(positions)

    def get_joint_velocities(self) -> np.ndarray:
        """Get current joint velocities."""
        velocities = []
        for joint_id in self.actuated_joint_ids:
            qvel_idx = self.model.jnt_dofadr[joint_id]
            velocities.append(self.data.qvel[qvel_idx])
        return np.array(velocities)

    def get_base_pose(self) -> np.ndarray:
        """Get base position and orientation."""
        position = self.data.qpos[:3]
        quaternion = self.data.qpos[3:7]
        return np.concatenate([position, quaternion])

    def get_base_velocity(self) -> np.ndarray:
        """Get base linear and angular velocity."""
        linear_vel = self.data.qvel[:3]
        angular_vel = self.data.qvel[3:6]
        return np.concatenate([linear_vel, angular_vel])

    def render(self, mode: str = "rgb_array", width: int = 480, height: int = 480, camera_id: Optional[int] = None) -> Optional[np.ndarray]:
        """Render environment."""
        if mode == "rgb_array":
            # Create renderer if not exists
            if not hasattr(self, '_renderer'):
                self._renderer = mujoco.Renderer(self.model, width=width, height=height)

            # Update renderer data
            self._renderer.update_scene(self.data, camera=camera_id)

            # Render
            return self._renderer.render()

        elif mode == "human":
            # TODO: Implement interactive viewer
            raise NotImplementedError("Human rendering mode not implemented yet")

        else:
            raise ValueError(f"Unsupported render mode: {mode}")

    def close(self):
        """Clean up resources."""
        if hasattr(self, '_renderer'):
            self._renderer.close()


def create_g1_environment(config: Config) -> G1Environment:
    """Factory function to create G1 environment."""
    env_name = config.env.name

    if env_name == "G1MotionImitation":
        from .motion_imitation import MotionImitationEnv
        return MotionImitationEnv(config)
    elif env_name == "G1RobustTraining":
        from .robust_training import RobustTrainingEnv
        return RobustTrainingEnv(config)
    elif env_name == "G1Evaluation":
        # Use motion imitation env for evaluation
        from .motion_imitation import MotionImitationEnv
        return MotionImitationEnv(config)
    else:
        # Default to base environment
        return G1Environment(config)