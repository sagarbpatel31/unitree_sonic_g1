"""
Robust G1 environment with configurable disturbances for fine-tuning.

This environment extends the base G1 environment with various robustness
challenges including random pushes, parameter variations, sensor noise,
and command-conditioned control.
"""

import numpy as np
import mujoco
from typing import Dict, Tuple, Optional, Any, List
from dataclasses import dataclass
import logging

try:
    import gym
    from gym import spaces
except ImportError:
    import gymnasium as gym
    from gymnasium import spaces

logger = logging.getLogger(__name__)


@dataclass
class DisturbanceConfig:
    """Configuration for environment disturbances."""

    # Physical disturbances
    enable_pushes: bool = True
    push_force_range: Tuple[float, float] = (50.0, 200.0)  # Newtons
    push_frequency: float = 0.02  # Probability per step
    push_duration_range: Tuple[float, float] = (0.1, 0.3)  # Seconds
    push_direction_range: Tuple[float, float] = (0.0, 2 * np.pi)  # Radians

    # Parameter variations
    friction_range: Tuple[float, float] = (0.5, 1.5)  # Multiplier
    mass_range: Tuple[float, float] = (0.8, 1.3)  # Multiplier
    motor_strength_range: Tuple[float, float] = (0.7, 1.2)  # Multiplier

    # Sensor noise and delays
    obs_noise_std: float = 0.01  # Gaussian noise std
    action_delay_steps: int = 0  # Action delay (0-3 steps)

    # Terrain variations
    enable_terrain: bool = False
    terrain_roughness: float = 0.03  # Height variation (meters)
    terrain_frequency: float = 2.0  # Spatial frequency

    # Command variations
    enable_commands: bool = True
    speed_command_range: Tuple[float, float] = (0.5, 2.0)  # m/s
    turn_command_range: Tuple[float, float] = (-1.0, 1.0)  # rad/s
    command_change_frequency: float = 0.005  # Probability per step


class RobustG1Env(gym.Env):
    """
    Robust G1 environment with configurable disturbances.

    This environment adds various robustness challenges to the base G1
    simulation including pushes, parameter variations, noise, and
    command-conditioned control.
    """

    def __init__(self,
                 model_path: str,
                 disturbance_config: DisturbanceConfig,
                 env_config: Dict[str, Any]):
        """
        Initialize robust G1 environment.

        Args:
            model_path: Path to MuJoCo model file
            disturbance_config: Disturbance configuration
            env_config: Environment configuration
        """
        super().__init__()

        self.model_path = model_path
        self.disturbance_config = disturbance_config
        self.env_config = env_config

        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # Store original model parameters for resetting
        self._original_friction = self.model.geom_friction.copy()
        self._original_mass = self.model.body_mass.copy()
        self._original_actuator_gear = self.model.actuator_gear.copy()

        # Environment parameters
        self.dt = self.model.opt.timestep * env_config.get('frame_skip', 10)
        self.frame_skip = env_config.get('frame_skip', 10)
        self.max_episode_steps = env_config.get('max_episode_steps', 1000)

        # Action and observation spaces
        self.action_dim = self.model.nu
        self.obs_dim = self._get_obs_dim()

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )

        # State variables
        self.step_count = 0
        self.episode_count = 0

        # Disturbance state
        self._push_remaining_steps = 0
        self._push_force = np.zeros(3)
        self._current_friction_multipliers = np.ones(len(self._original_friction))
        self._current_mass_multipliers = np.ones(len(self._original_mass))
        self._current_motor_multipliers = np.ones(len(self._original_actuator_gear))

        # Action delay buffer
        self._action_buffer = []
        max_delay = max(3, disturbance_config.action_delay_steps + 1)
        for _ in range(max_delay):
            self._action_buffer.append(np.zeros(self.action_dim))

        # Command state
        self._current_speed_command = 1.0
        self._current_turn_command = 0.0

        # Reference motion (placeholder - should be loaded from trajectory)
        self._reference_trajectory = None
        self._reference_index = 0

        # Terrain heightfield (if enabled)
        if disturbance_config.enable_terrain:
            self._setup_terrain()

        # Metrics tracking
        self._episode_metrics = {
            'tracking_errors': [],
            'command_following_errors': [],
            'energy_consumption': [],
            'falls': 0,
            'push_events': 0
        }

        # Root body ID for applying pushes
        self._root_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        if self._root_body_id == -1:
            # Try alternative names
            for name in ["trunk", "torso", "base", "pelvis"]:
                self._root_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
                if self._root_body_id != -1:
                    break

        logger.info(f"Initialized RobustG1Env with root_body_id: {self._root_body_id}")

    def _get_obs_dim(self) -> int:
        """Calculate observation dimensionality."""
        obs = self._get_obs()
        return len(obs)

    def _setup_terrain(self):
        """Setup terrain heightfield for uneven surfaces."""
        # Create heightfield data
        hfield_size = 100  # Grid size
        roughness = self.disturbance_config.terrain_roughness
        frequency = self.disturbance_config.terrain_frequency

        x = np.linspace(-5, 5, hfield_size)
        y = np.linspace(-5, 5, hfield_size)
        X, Y = np.meshgrid(x, y)

        # Generate terrain using Perlin-like noise
        height_data = (
            roughness * np.sin(frequency * X) * np.cos(frequency * Y) +
            0.5 * roughness * np.sin(2 * frequency * X + np.pi/4) +
            0.25 * roughness * np.random.normal(0, 1, (hfield_size, hfield_size))
        )

        # Flatten and normalize
        height_data = height_data.flatten()
        height_data = (height_data - height_data.min()) / (height_data.max() - height_data.min())
        height_data = height_data * roughness

        # Apply to model (if heightfield exists)
        if hasattr(self.model, 'hfield_data') and len(self.model.hfield_data) > 0:
            self.model.hfield_data[:len(height_data)] = height_data

    def reset(self, **kwargs) -> np.ndarray:
        """Reset environment to initial state."""
        # Reset MuJoCo simulation
        mujoco.mj_resetData(self.model, self.data)

        # Reset counters
        self.step_count = 0
        self.episode_count += 1

        # Randomize model parameters
        self._randomize_parameters()

        # Reset disturbance state
        self._push_remaining_steps = 0
        self._push_force = np.zeros(3)

        # Reset action buffer
        for i in range(len(self._action_buffer)):
            self._action_buffer[i] = np.zeros(self.action_dim)

        # Reset commands
        if self.disturbance_config.enable_commands:
            self._randomize_commands()

        # Reset reference trajectory
        self._reference_index = 0

        # Reset metrics
        self._episode_metrics = {
            'tracking_errors': [],
            'command_following_errors': [],
            'energy_consumption': [],
            'falls': 0,
            'push_events': 0
        }

        # Set initial pose with noise
        self._set_initial_pose()

        # Forward simulation to stabilize
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        return self._get_obs()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute one environment step."""
        self.step_count += 1

        # Apply action delay
        self._action_buffer.append(action.copy())
        if len(self._action_buffer) > self.disturbance_config.action_delay_steps + 1:
            self._action_buffer.pop(0)

        delayed_action = self._action_buffer[0]

        # Apply action
        self._apply_action(delayed_action)

        # Apply disturbances
        self._apply_disturbances()

        # Step simulation
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        # Get observation
        obs = self._get_obs()

        # Compute reward
        reward, reward_info = self._compute_reward(action, obs)

        # Check termination
        done, done_info = self._check_termination()

        # Update commands
        if self.disturbance_config.enable_commands:
            self._update_commands()

        # Collect metrics
        info = self._collect_info(reward_info, done_info)

        return obs, reward, done, info

    def _randomize_parameters(self):
        """Randomize physical parameters for robustness training."""
        # Randomize friction
        friction_multipliers = np.random.uniform(
            self.disturbance_config.friction_range[0],
            self.disturbance_config.friction_range[1],
            len(self._original_friction)
        )
        self._current_friction_multipliers = friction_multipliers
        self.model.geom_friction[:] = self._original_friction * friction_multipliers.reshape(-1, 1)

        # Randomize masses
        mass_multipliers = np.random.uniform(
            self.disturbance_config.mass_range[0],
            self.disturbance_config.mass_range[1],
            len(self._original_mass)
        )
        self._current_mass_multipliers = mass_multipliers
        self.model.body_mass[:] = self._original_mass * mass_multipliers

        # Randomize motor strength
        motor_multipliers = np.random.uniform(
            self.disturbance_config.motor_strength_range[0],
            self.disturbance_config.motor_strength_range[1],
            len(self._original_actuator_gear)
        )
        self._current_motor_multipliers = motor_multipliers
        self.model.actuator_gear[:] = self._original_actuator_gear * motor_multipliers.reshape(-1, 1)

    def _randomize_commands(self):
        """Randomize motion commands."""
        self._current_speed_command = np.random.uniform(
            self.disturbance_config.speed_command_range[0],
            self.disturbance_config.speed_command_range[1]
        )
        self._current_turn_command = np.random.uniform(
            self.disturbance_config.turn_command_range[0],
            self.disturbance_config.turn_command_range[1]
        )

    def _update_commands(self):
        """Update commands during episode."""
        if np.random.random() < self.disturbance_config.command_change_frequency:
            self._randomize_commands()

    def _set_initial_pose(self):
        """Set initial pose with random noise."""
        # Set to reference pose if available
        if self.env_config.get('resets', {}).get('use_reference_pose', False):
            # Use reference trajectory initial pose (placeholder)
            pass

        # Add noise to initial pose
        noise_scale = self.env_config.get('resets', {}).get('pose_noise_scale', 0.02)

        if noise_scale > 0:
            # Add noise to joint positions
            joint_noise = np.random.normal(0, noise_scale, self.model.nq)
            self.data.qpos[:] += joint_noise

            # Add noise to joint velocities
            vel_noise = np.random.normal(0, noise_scale * 0.1, self.model.nv)
            self.data.qvel[:] += vel_noise

    def _apply_action(self, action: np.ndarray):
        """Apply action to the robot."""
        # Clip action
        action = np.clip(action, -1.0, 1.0)

        # Scale action based on configuration
        action_scale = self.env_config.get('action_scale', 0.1)
        scaled_action = action * action_scale

        # Apply action based on type
        action_type = self.env_config.get('action_type', 'position_delta')

        if action_type == 'position_delta':
            # Position control with deltas
            target_pos = self.data.qpos[7:] + scaled_action  # Skip free joint
            self.data.ctrl[:] = target_pos

        elif action_type == 'position_absolute':
            # Direct position control
            self.data.ctrl[:] = scaled_action

        elif action_type == 'torque':
            # Direct torque control
            torque_scale = self.env_config.get('torque_scale', 100.0)
            self.data.ctrl[:] = action * torque_scale

        else:
            raise ValueError(f"Unknown action type: {action_type}")

    def _apply_disturbances(self):
        """Apply various disturbances to the environment."""
        # Random pushes
        if self.disturbance_config.enable_pushes:
            self._apply_random_pushes()

    def _apply_random_pushes(self):
        """Apply random external pushes."""
        # Start new push
        if (self._push_remaining_steps <= 0 and
            np.random.random() < self.disturbance_config.push_frequency):

            # Generate push parameters
            force_magnitude = np.random.uniform(
                self.disturbance_config.push_force_range[0],
                self.disturbance_config.push_force_range[1]
            )

            direction = np.random.uniform(
                self.disturbance_config.push_direction_range[0],
                self.disturbance_config.push_direction_range[1]
            )

            # Convert to 3D force
            self._push_force[0] = force_magnitude * np.cos(direction)
            self._push_force[1] = force_magnitude * np.sin(direction)
            self._push_force[2] = 0  # No vertical push

            # Set duration
            duration = np.random.uniform(
                self.disturbance_config.push_duration_range[0],
                self.disturbance_config.push_duration_range[1]
            )
            self._push_remaining_steps = int(duration / self.dt)

            self._episode_metrics['push_events'] += 1

        # Apply current push
        if self._push_remaining_steps > 0:
            if self._root_body_id >= 0:
                # Apply external force to root body
                self.data.xfrc_applied[self._root_body_id][:3] = self._push_force
            self._push_remaining_steps -= 1
        else:
            # Clear force
            if self._root_body_id >= 0:
                self.data.xfrc_applied[self._root_body_id][:3] = 0

    def _get_obs(self) -> np.ndarray:
        """Get current observation."""
        obs_components = []

        obs_config = self.env_config.get('observations', {})

        # Joint positions
        if obs_config.get('include_joint_pos', True):
            joint_pos = self.data.qpos[7:].copy()  # Skip free joint
            obs_components.append(joint_pos)

        # Joint velocities
        if obs_config.get('include_joint_vel', True):
            joint_vel = self.data.qvel[6:].copy()  # Skip free joint
            obs_components.append(joint_vel)

        # Root orientation (quaternion)
        if obs_config.get('include_root_orientation', True):
            root_quat = self.data.qpos[3:7].copy()
            obs_components.append(root_quat)

        # Root linear velocity
        if obs_config.get('include_root_linear_vel', True):
            root_lin_vel = self.data.qvel[:3].copy()
            obs_components.append(root_lin_vel)

        # Root angular velocity
        if obs_config.get('include_root_angular_vel', True):
            root_ang_vel = self.data.qvel[3:6].copy()
            obs_components.append(root_ang_vel)

        # Reference motion features
        if obs_config.get('include_reference_motion', True):
            ref_features = self._get_reference_features()
            if len(ref_features) > 0:
                obs_components.append(ref_features)

        # Command features
        if self.disturbance_config.enable_commands:
            command_features = np.array([self._current_speed_command, self._current_turn_command])
            obs_components.append(command_features)

        # Concatenate all components
        obs = np.concatenate(obs_components)

        # Add observation noise
        if self.disturbance_config.obs_noise_std > 0:
            noise = np.random.normal(0, self.disturbance_config.obs_noise_std, obs.shape)
            obs += noise

        return obs.astype(np.float32)

    def _get_reference_features(self) -> np.ndarray:
        """Get reference motion features for future timesteps."""
        if self._reference_trajectory is None:
            # Placeholder reference features
            horizon = self.env_config.get('observations', {}).get('reference_horizon', 10)
            ref_dim = 22 + 3  # Joint positions + root position
            return np.zeros(horizon * ref_dim)

        # Extract reference features from trajectory
        horizon = self.env_config.get('observations', {}).get('reference_horizon', 10)
        ref_features = []

        for i in range(1, horizon + 1):
            ref_idx = min(self._reference_index + i, len(self._reference_trajectory) - 1)
            # Extract joint positions and root position from reference
            joint_pos = self._reference_trajectory[ref_idx]['joint_positions']
            root_pos = self._reference_trajectory[ref_idx]['root_position']
            ref_features.extend(joint_pos)
            ref_features.extend(root_pos)

        return np.array(ref_features)

    def _compute_reward(self, action: np.ndarray, obs: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """Compute reward for current state and action."""
        reward_config = self.env_config.get('rewards', {})
        reward_components = {}

        total_reward = 0.0

        # Joint position tracking reward
        if reward_config.get('joint_pos_weight', 0) > 0:
            joint_pos_reward = self._compute_joint_tracking_reward()
            reward_components['joint_pos'] = joint_pos_reward
            total_reward += reward_config['joint_pos_weight'] * joint_pos_reward

        # Root position tracking reward
        if reward_config.get('root_pos_weight', 0) > 0:
            root_pos_reward = self._compute_root_position_reward()
            reward_components['root_pos'] = root_pos_reward
            total_reward += reward_config['root_pos_weight'] * root_pos_reward

        # Root orientation reward
        if reward_config.get('root_orient_weight', 0) > 0:
            root_orient_reward = self._compute_root_orientation_reward()
            reward_components['root_orient'] = root_orient_reward
            total_reward += reward_config['root_orient_weight'] * root_orient_reward

        # Stability reward
        if reward_config.get('stability_weight', 0) > 0:
            stability_reward = self._compute_stability_reward()
            reward_components['stability'] = stability_reward
            total_reward += reward_config['stability_weight'] * stability_reward

        # Command following reward
        if reward_config.get('command_following_weight', 0) > 0:
            command_reward = self._compute_command_following_reward()
            reward_components['command_following'] = command_reward
            total_reward += reward_config['command_following_weight'] * command_reward

        # Energy efficiency reward
        if reward_config.get('energy_efficiency_weight', 0) > 0:
            energy_reward = self._compute_energy_efficiency_reward(action)
            reward_components['energy_efficiency'] = energy_reward
            total_reward += reward_config['energy_efficiency_weight'] * energy_reward

        # Fall penalty
        if self._check_fall():
            fall_penalty = reward_config.get('fall_penalty', -10.0)
            reward_components['fall_penalty'] = fall_penalty
            total_reward += fall_penalty

        return total_reward, reward_components

    def _compute_joint_tracking_reward(self) -> float:
        """Compute reward for tracking reference joint positions."""
        if self._reference_trajectory is None:
            return 0.0

        current_joint_pos = self.data.qpos[7:]  # Skip free joint
        ref_idx = min(self._reference_index, len(self._reference_trajectory) - 1)
        ref_joint_pos = self._reference_trajectory[ref_idx]['joint_positions']

        # L2 tracking error
        tracking_error = np.linalg.norm(current_joint_pos - ref_joint_pos)
        tracking_reward = np.exp(-tracking_error)

        # Store for metrics
        self._episode_metrics['tracking_errors'].append(tracking_error)

        return tracking_reward

    def _compute_root_position_reward(self) -> float:
        """Compute reward for root position tracking."""
        if self._reference_trajectory is None:
            return 0.0

        current_root_pos = self.data.qpos[:3]
        ref_idx = min(self._reference_index, len(self._reference_trajectory) - 1)
        ref_root_pos = self._reference_trajectory[ref_idx]['root_position']

        # Horizontal tracking (ignore Z for now)
        pos_error = np.linalg.norm(current_root_pos[:2] - ref_root_pos[:2])
        pos_reward = np.exp(-pos_error * 2.0)

        return pos_reward

    def _compute_root_orientation_reward(self) -> float:
        """Compute reward for maintaining proper root orientation."""
        # Target upright orientation
        target_quat = np.array([1, 0, 0, 0])  # Identity quaternion
        current_quat = self.data.qpos[3:7]

        # Quaternion distance
        quat_error = 1 - np.abs(np.dot(current_quat, target_quat))
        orient_reward = np.exp(-quat_error * 10.0)

        return orient_reward

    def _compute_stability_reward(self) -> float:
        """Compute reward for maintaining stability."""
        # Base on root angular velocity and acceleration
        root_ang_vel = self.data.qvel[3:6]
        ang_vel_penalty = np.sum(root_ang_vel ** 2)

        stability_reward = np.exp(-ang_vel_penalty * 0.1)

        return stability_reward

    def _compute_command_following_reward(self) -> float:
        """Compute reward for following speed and turn commands."""
        if not self.disturbance_config.enable_commands:
            return 0.0

        # Get current velocity
        current_lin_vel = self.data.qvel[:3]
        current_speed = np.linalg.norm(current_lin_vel[:2])
        current_turn = self.data.qvel[5]  # Z angular velocity

        # Compute errors
        speed_error = abs(current_speed - self._current_speed_command)
        turn_error = abs(current_turn - self._current_turn_command)

        # Combined error
        command_error = speed_error + turn_error
        command_reward = np.exp(-command_error * 2.0)

        # Store for metrics
        self._episode_metrics['command_following_errors'].append(command_error)

        return command_reward

    def _compute_energy_efficiency_reward(self, action: np.ndarray) -> float:
        """Compute reward for energy efficiency."""
        # Simple energy model based on action magnitude
        action_magnitude = np.linalg.norm(action)
        energy = action_magnitude ** 2

        energy_reward = np.exp(-energy * 0.1)

        # Store for metrics
        self._episode_metrics['energy_consumption'].append(energy)

        return energy_reward

    def _check_termination(self) -> Tuple[bool, Dict[str, Any]]:
        """Check if episode should terminate."""
        done = False
        info = {}

        # Check for fall
        if self._check_fall():
            done = True
            info['fell'] = True
            self._episode_metrics['falls'] += 1

        # Check episode length
        if self.step_count >= self.max_episode_steps:
            done = True
            info['timeout'] = True

        return done, info

    def _check_fall(self) -> bool:
        """Check if robot has fallen."""
        # Check root height
        root_height = self.data.qpos[2]
        if root_height < 0.3:  # Threshold height
            return True

        # Check root orientation
        root_quat = self.data.qpos[3:7]
        # Convert to roll, pitch, yaw
        w, x, y, z = root_quat
        roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
        pitch = np.arcsin(2*(w*y - z*x))

        # Check if tilted too much
        if abs(roll) > np.pi/4 or abs(pitch) > np.pi/4:
            return True

        return False

    def _collect_info(self, reward_info: Dict[str, float], done_info: Dict[str, Any]) -> Dict[str, Any]:
        """Collect information dictionary for step."""
        info = {**reward_info, **done_info}

        # Add current metrics
        if len(self._episode_metrics['tracking_errors']) > 0:
            info['tracking_error'] = self._episode_metrics['tracking_errors'][-1]

        if len(self._episode_metrics['command_following_errors']) > 0:
            info['command_following_error'] = self._episode_metrics['command_following_errors'][-1]

        if len(self._episode_metrics['energy_consumption']) > 0:
            info['energy_consumption'] = self._episode_metrics['energy_consumption'][-1]

        # Add command information
        info['command_speed'] = self._current_speed_command
        info['command_turn'] = self._current_turn_command

        # Add actual motion information
        actual_lin_vel = self.data.qvel[:3]
        info['actual_speed'] = np.linalg.norm(actual_lin_vel[:2])
        info['actual_turn'] = self.data.qvel[5]

        return info

    def set_disturbance_scale(self, scale: float):
        """Scale disturbance intensity for evaluation."""
        # Scale push forces
        original_push_range = self.disturbance_config.push_force_range
        self.disturbance_config.push_force_range = (
            original_push_range[0] * scale,
            original_push_range[1] * scale
        )

        # Scale push frequency
        original_push_freq = self.disturbance_config.push_frequency
        self.disturbance_config.push_frequency = original_push_freq * scale

        # Scale observation noise
        original_obs_noise = self.disturbance_config.obs_noise_std
        self.disturbance_config.obs_noise_std = original_obs_noise * scale

        logger.info(f"Set disturbance scale to: {scale}")

    def get_episode_metrics(self) -> Dict[str, Any]:
        """Get metrics for completed episode."""
        if len(self._episode_metrics['tracking_errors']) == 0:
            return {}

        return {
            'avg_tracking_error': np.mean(self._episode_metrics['tracking_errors']),
            'max_tracking_error': np.max(self._episode_metrics['tracking_errors']),
            'avg_command_following_error': np.mean(self._episode_metrics['command_following_errors'])
                if len(self._episode_metrics['command_following_errors']) > 0 else 0,
            'total_energy': np.sum(self._episode_metrics['energy_consumption']),
            'avg_energy_per_step': np.mean(self._episode_metrics['energy_consumption']),
            'num_falls': self._episode_metrics['falls'],
            'num_pushes': self._episode_metrics['push_events'],
            'episode_length': self.step_count
        }

    def render(self, mode: str = 'human'):
        """Render the environment."""
        # Placeholder for rendering
        pass

    def close(self):
        """Clean up environment resources."""
        pass