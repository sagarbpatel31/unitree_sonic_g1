"""
MuJoCo-based Unitree G1 humanoid environment for whole-body motion imitation.

This module implements a high-fidelity simulation environment for the Unitree G1
humanoid robot with support for motion tracking, domain randomization, and
command-conditioned behaviors.
"""

import numpy as np
import mujoco
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import warnings

import gymnasium as gym
from gymnasium import spaces

from .observations import ObservationManager
from .rewards import RewardManager
from .resets import ResetManager
from .randomization import DomainRandomizer
from .commands import CommandManager


class G1Environment(gym.Env):
    """
    Unitree G1 humanoid environment for whole-body motion imitation.

    This environment provides:
    - High-fidelity MuJoCo physics simulation
    - Configurable observation and action spaces
    - Multi-term reward functions for motion tracking
    - Domain randomization for robustness
    - Command-conditioned behaviors
    - Safety monitoring and termination
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        frame_skip: int = 10,
        observation_config: Optional[Dict[str, Any]] = None,
        reward_config: Optional[Dict[str, Any]] = None,
        reset_config: Optional[Dict[str, Any]] = None,
        randomization_config: Optional[Dict[str, Any]] = None,
        command_config: Optional[Dict[str, Any]] = None,
        action_type: str = "position_delta",
        action_scale: float = 0.1,
        safety_config: Optional[Dict[str, Any]] = None,
        render_mode: Optional[str] = None,
    ):
        """
        Initialize the G1 environment.

        Args:
            model_path: Path to G1 MuJoCo model file (.xml)
            frame_skip: Number of physics steps per environment step
            observation_config: Configuration for observation space
            reward_config: Configuration for reward computation
            reset_config: Configuration for environment resets
            randomization_config: Configuration for domain randomization
            command_config: Configuration for command conditioning
            action_type: Type of action space ("position_delta" or "position_absolute")
            action_scale: Scaling factor for actions
            safety_config: Configuration for safety monitoring
            render_mode: Rendering mode ("human", "rgb_array", None)
        """
        self.model_path = Path(model_path)
        self.frame_skip = frame_skip
        self.action_type = action_type
        self.action_scale = action_scale
        self.render_mode = render_mode

        # Load MuJoCo model
        self._load_model()

        # Initialize managers
        self.obs_manager = ObservationManager(
            self.model, self.data, observation_config or {}
        )
        self.reward_manager = RewardManager(
            self.model, self.data, reward_config or {}
        )
        self.reset_manager = ResetManager(
            self.model, self.data, reset_config or {}
        )
        self.randomizer = DomainRandomizer(
            self.model, self.data, randomization_config or {}
        )
        self.command_manager = CommandManager(
            self.model, self.data, command_config or {}
        )

        # Safety configuration
        self.safety_config = safety_config or {}
        self.fall_height_threshold = self.safety_config.get("fall_height", 0.3)
        self.fall_angle_threshold = self.safety_config.get("fall_angle", 1.0)
        self.joint_vel_threshold = self.safety_config.get("max_joint_vel", 50.0)

        # Environment state
        self.step_count = 0
        self.episode_length = 1000
        self.last_action = None
        self.reference_motion = None
        self.current_command = None

        # Define spaces
        self._setup_spaces()

        # Initialize renderer
        self.renderer = None
        if self.render_mode is not None:
            self._setup_renderer()

    def _load_model(self):
        """Load the MuJoCo model and initialize simulation data."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        try:
            self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
            self.data = mujoco.MjData(self.model)
        except Exception as e:
            raise RuntimeError(f"Failed to load MuJoCo model: {e}")

        # Store initial state
        self.init_qpos = self.data.qpos.copy()
        self.init_qvel = self.data.qvel.copy()

        # Get robot information
        self._get_robot_info()

    def _get_robot_info(self):
        """Extract robot-specific information from the model."""
        # Find actuated joints
        self.actuated_joints = []
        for i in range(self.model.nu):
            actuator_id = i
            joint_id = self.model.actuator_trnid[actuator_id, 0]
            joint_name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id
            )
            self.actuated_joints.append({
                'id': joint_id,
                'name': joint_name,
                'actuator_id': actuator_id
            })

        self.num_actuated_joints = len(self.actuated_joints)

        # Get joint limits
        self.joint_limits = np.zeros((self.num_actuated_joints, 2))
        for i, joint_info in enumerate(self.actuated_joints):
            joint_id = joint_info['id']
            self.joint_limits[i] = self.model.jnt_range[joint_id]

        # Find important body IDs
        self.torso_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link"
        )
        if self.torso_id == -1:
            self.torso_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, "torso"
            )

        # Find foot bodies
        foot_names = ["left_foot", "right_foot", "left_ankle", "right_ankle"]
        self.foot_ids = []
        for name in foot_names:
            foot_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, name
            )
            if foot_id != -1:
                self.foot_ids.append(foot_id)

        # Find end effector bodies
        ee_names = ["left_hand", "right_hand", "left_gripper", "right_gripper"]
        self.ee_ids = []
        for name in ee_names:
            ee_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, name
            )
            if ee_id != -1:
                self.ee_ids.append(ee_id)

    def _setup_spaces(self):
        """Setup observation and action spaces."""
        # Action space
        if self.action_type == "position_delta":
            # Joint position deltas
            action_low = np.full(self.num_actuated_joints, -self.action_scale)
            action_high = np.full(self.num_actuated_joints, self.action_scale)
        elif self.action_type == "position_absolute":
            # Normalized joint positions [-1, 1]
            action_low = np.full(self.num_actuated_joints, -1.0)
            action_high = np.full(self.num_actuated_joints, 1.0)
        else:
            raise ValueError(f"Unknown action type: {self.action_type}")

        self.action_space = spaces.Box(
            low=action_low, high=action_high, dtype=np.float32
        )

        # Observation space (set by observation manager)
        obs_dim = self.obs_manager.get_observation_dim()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

    def _setup_renderer(self):
        """Setup MuJoCo renderer."""
        if self.render_mode == "rgb_array":
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        elif self.render_mode == "human":
            # For human rendering, we'll use the rgb_array renderer and display
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.

        Args:
            seed: Random seed for reproducibility
            options: Additional options for reset

        Returns:
            observation: Initial observation
            info: Additional information
        """
        super().reset(seed=seed)

        if options is None:
            options = {}

        # Reset environment state
        self.step_count = 0
        self.last_action = np.zeros(self.num_actuated_joints)

        # Apply domain randomization
        if options.get("randomize", True):
            self.randomizer.randomize()

        # Reset to initial pose with optional noise
        self.reset_manager.reset(options)

        # Generate new command if using command conditioning
        if self.command_manager.enabled:
            self.current_command = self.command_manager.sample_command()

        # Forward simulation
        mujoco.mj_forward(self.model, self.data)

        # Get initial observation
        observation = self.obs_manager.get_observation(
            last_action=self.last_action,
            reference_motion=self.reference_motion,
            command=self.current_command
        )

        # Initialize reward manager
        self.reward_manager.reset()

        info = {
            "step_count": self.step_count,
            "command": self.current_command,
            "randomization": self.randomizer.get_current_params()
        }

        return observation, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one environment step.

        Args:
            action: Action to execute

        Returns:
            observation: New observation
            reward: Reward for this step
            terminated: Whether episode terminated
            truncated: Whether episode was truncated
            info: Additional information
        """
        # Validate action
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Convert action to joint targets
        joint_targets = self._process_action(action)

        # Apply action with potential latency
        if self.randomizer.control_latency > 0:
            # Simple latency simulation - use delayed action
            if hasattr(self, '_action_buffer'):
                actual_targets = self._action_buffer
                self._action_buffer = joint_targets
            else:
                actual_targets = joint_targets
                self._action_buffer = joint_targets
        else:
            actual_targets = joint_targets

        # Execute action for frame_skip steps
        for _ in range(self.frame_skip):
            # Set control signals
            for i, joint_info in enumerate(self.actuated_joints):
                self.data.ctrl[joint_info['actuator_id']] = actual_targets[i]

            # Apply external perturbations
            if self.randomizer.should_apply_push():
                self.randomizer.apply_external_push()

            # Step simulation
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1

        # Get observation
        observation = self.obs_manager.get_observation(
            last_action=action,
            reference_motion=self.reference_motion,
            command=self.current_command
        )

        # Compute reward
        reward_info = self.reward_manager.compute_reward(
            action=action,
            reference_motion=self.reference_motion,
            command=self.current_command
        )
        reward = reward_info["total_reward"]

        # Check termination conditions
        terminated = self._check_termination()
        truncated = self.step_count >= self.episode_length

        # Update last action
        self.last_action = action.copy()

        # Compile info
        info = {
            "step_count": self.step_count,
            "reward_terms": reward_info,
            "command": self.current_command,
            "termination_reason": self._get_termination_reason() if terminated else None,
            "joint_targets": joint_targets,
            "joint_positions": self._get_joint_positions(),
            "root_position": self.data.qpos[:3].copy(),
            "root_orientation": self.data.qpos[3:7].copy(),
        }

        return observation, reward, terminated, truncated, info

    def _process_action(self, action: np.ndarray) -> np.ndarray:
        """
        Process action into joint targets.

        Args:
            action: Raw action from policy

        Returns:
            joint_targets: Target joint positions
        """
        if self.action_type == "position_delta":
            # Add delta to current positions
            current_positions = self._get_joint_positions()
            joint_targets = current_positions + action * self.action_scale
        elif self.action_type == "position_absolute":
            # Scale normalized actions to joint limits
            joint_ranges = self.joint_limits[:, 1] - self.joint_limits[:, 0]
            joint_centers = (self.joint_limits[:, 1] + self.joint_limits[:, 0]) / 2
            joint_targets = joint_centers + action * joint_ranges / 2
        else:
            raise ValueError(f"Unknown action type: {self.action_type}")

        # Clip to joint limits
        joint_targets = np.clip(
            joint_targets, self.joint_limits[:, 0], self.joint_limits[:, 1]
        )

        return joint_targets

    def _get_joint_positions(self) -> np.ndarray:
        """Get current joint positions for actuated joints."""
        positions = []
        for joint_info in self.actuated_joints:
            joint_id = joint_info['id']
            positions.append(self.data.qpos[joint_id])
        return np.array(positions)

    def _get_joint_velocities(self) -> np.ndarray:
        """Get current joint velocities for actuated joints."""
        velocities = []
        for joint_info in self.actuated_joints:
            joint_id = joint_info['id']
            velocities.append(self.data.qvel[joint_id])
        return np.array(velocities)

    def _check_termination(self) -> bool:
        """Check if episode should terminate."""
        # Check for NaN values
        if np.any(np.isnan(self.data.qpos)) or np.any(np.isnan(self.data.qvel)):
            return True

        # Check if robot fell
        if self._check_fall():
            return True

        # Check joint velocity limits
        joint_velocities = self._get_joint_velocities()
        if np.any(np.abs(joint_velocities) > self.joint_vel_threshold):
            return True

        return False

    def _check_fall(self) -> bool:
        """Check if the robot has fallen."""
        if self.torso_id == -1:
            return False

        # Check height
        torso_pos = self.data.xpos[self.torso_id]
        if torso_pos[2] < self.fall_height_threshold:
            return True

        # Check orientation
        torso_quat = self.data.xquat[self.torso_id]
        # Convert quaternion to rotation matrix to get up vector
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, torso_quat)
        R = R.reshape(3, 3)
        up_vector = R[:, 2]  # z-axis in body frame

        # Check if robot is too tilted
        if up_vector[2] < np.cos(self.fall_angle_threshold):
            return True

        return False

    def _get_termination_reason(self) -> str:
        """Get the reason for termination."""
        if np.any(np.isnan(self.data.qpos)) or np.any(np.isnan(self.data.qvel)):
            return "nan_state"

        if self._check_fall():
            return "fall"

        joint_velocities = self._get_joint_velocities()
        if np.any(np.abs(joint_velocities) > self.joint_vel_threshold):
            return "excessive_velocity"

        return "unknown"

    def render(self) -> Optional[np.ndarray]:
        """Render the environment."""
        if self.render_mode is None:
            return None

        if self.renderer is None:
            self._setup_renderer()

        # Update renderer
        self.renderer.update_scene(self.data)

        if self.render_mode == "rgb_array":
            return self.renderer.render()
        elif self.render_mode == "human":
            # For human mode, we could display the image
            # For now, just return the array
            return self.renderer.render()

    def close(self):
        """Clean up resources."""
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def set_reference_motion(self, reference_motion: Dict[str, np.ndarray]):
        """
        Set reference motion for imitation.

        Args:
            reference_motion: Dictionary containing reference trajectory data
        """
        self.reference_motion = reference_motion

    def set_command(self, command: Dict[str, Any]):
        """
        Set command for command-conditioned behavior.

        Args:
            command: Command specification
        """
        self.current_command = command

    def get_robot_state(self) -> Dict[str, np.ndarray]:
        """Get current robot state."""
        return {
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "joint_positions": self._get_joint_positions(),
            "joint_velocities": self._get_joint_velocities(),
            "root_position": self.data.qpos[:3].copy(),
            "root_orientation": self.data.qpos[3:7].copy(),
            "root_linear_velocity": self.data.qvel[:3].copy(),
            "root_angular_velocity": self.data.qvel[3:6].copy(),
        }

    def set_robot_state(self, state: Dict[str, np.ndarray]):
        """Set robot state."""
        if "qpos" in state:
            self.data.qpos[:] = state["qpos"]
        if "qvel" in state:
            self.data.qvel[:] = state["qvel"]
        mujoco.mj_forward(self.model, self.data)


def create_g1_env(
    model_path: Union[str, Path],
    config: Optional[Dict[str, Any]] = None
) -> G1Environment:
    """
    Factory function to create G1 environment.

    Args:
        model_path: Path to G1 MuJoCo model
        config: Environment configuration

    Returns:
        G1Environment instance
    """
    if config is None:
        config = {}

    return G1Environment(
        model_path=model_path,
        frame_skip=config.get("frame_skip", 10),
        observation_config=config.get("observations", {}),
        reward_config=config.get("rewards", {}),
        reset_config=config.get("resets", {}),
        randomization_config=config.get("randomization", {}),
        command_config=config.get("commands", {}),
        action_type=config.get("action_type", "position_delta"),
        action_scale=config.get("action_scale", 0.1),
        safety_config=config.get("safety", {}),
        render_mode=config.get("render_mode", None),
    )