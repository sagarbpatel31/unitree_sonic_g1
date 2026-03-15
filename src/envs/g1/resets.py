"""
Reset management for Unitree G1 environment.

This module handles environment resets including pose initialization,
noise injection, and reference motion synchronization.
"""

import numpy as np
import mujoco
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class ResetConfig:
    """Configuration for environment resets."""
    # Initial pose options
    use_reference_pose: bool = True
    pose_noise_scale: float = 0.1
    velocity_noise_scale: float = 0.1

    # Reset types
    reset_to_reference: bool = True
    reset_to_default: bool = False
    reset_random: bool = False

    # Reference motion sampling
    random_reference_time: bool = True
    reference_time_range: tuple = (0.0, 1.0)  # Fraction of motion

    # Joint-specific noise
    joint_pos_noise: float = 0.05  # radians
    joint_vel_noise: float = 0.1   # rad/s

    # Root pose noise
    root_pos_noise: float = 0.02   # meters
    root_orient_noise: float = 0.1  # radians

    # Safety constraints during reset
    max_joint_deviation: float = 0.5  # radians from reference
    min_height: float = 0.5  # meters
    max_height: float = 1.2  # meters


class ResetManager:
    """
    Manages environment resets for the G1 environment.

    This class handles:
    - Initialization to reference poses from motion data
    - Noise injection for robust initialization
    - Safety constraint enforcement during reset
    - Synchronization with reference motion timing
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        config: Dict[str, Any]
    ):
        """
        Initialize reset manager.

        Args:
            model: MuJoCo model
            data: MuJoCo data
            config: Configuration dictionary
        """
        self.model = model
        self.data = data

        # Parse configuration
        self.config = ResetConfig(**config)

        # Get robot information
        self._get_robot_info()

        # Store default pose
        self.default_qpos = self.data.qpos.copy()
        self.default_qvel = self.data.qvel.copy()

        # Initialize reset statistics
        self.reset_count = 0

        print(f"ResetManager initialized with {len(self.reset_strategies)} reset strategies")

    def _get_robot_info(self):
        """Extract robot-specific information from model."""
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

        self.num_joints = len(self.actuated_joints)

        # Get joint limits for safety
        self.joint_limits = np.zeros((self.num_joints, 2))
        for i, joint_info in enumerate(self.actuated_joints):
            joint_id = joint_info['id']
            self.joint_limits[i] = self.model.jnt_range[joint_id]

        # Find torso body
        self.torso_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link"
        )
        if self.torso_id == -1:
            self.torso_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, "torso"
            )

        # Define reset strategies
        self.reset_strategies = []
        if self.config.reset_to_reference:
            self.reset_strategies.append("reference")
        if self.config.reset_to_default:
            self.reset_strategies.append("default")
        if self.config.reset_random:
            self.reset_strategies.append("random")

        if not self.reset_strategies:
            self.reset_strategies = ["default"]

    def reset(
        self,
        options: Optional[Dict[str, Any]] = None,
        reference_motion: Optional[Dict[str, Any]] = None
    ):
        """
        Reset the environment.

        Args:
            options: Reset options
            reference_motion: Reference motion data for pose initialization
        """
        if options is None:
            options = {}

        # Select reset strategy
        strategy = options.get("reset_strategy", None)
        if strategy is None:
            strategy = np.random.choice(self.reset_strategies)

        # Perform reset based on strategy
        if strategy == "reference" and reference_motion is not None:
            self._reset_to_reference(reference_motion, options)
        elif strategy == "random":
            self._reset_random(options)
        else:
            self._reset_to_default(options)

        # Apply additional noise if requested
        noise_scale = options.get("noise_scale", 1.0)
        if noise_scale > 0:
            self._apply_reset_noise(noise_scale)

        # Ensure safety constraints
        self._enforce_safety_constraints()

        # Forward simulation to update derived quantities
        mujoco.mj_forward(self.model, self.data)

        self.reset_count += 1

    def _reset_to_reference(
        self,
        reference_motion: Dict[str, Any],
        options: Dict[str, Any]
    ):
        """Reset to a pose from reference motion."""
        # Select time in reference motion
        if self.config.random_reference_time:
            motion_times = reference_motion.get("times", [0.0])
            if len(motion_times) > 1:
                time_range = self.config.reference_time_range
                start_idx = int(time_range[0] * len(motion_times))
                end_idx = int(time_range[1] * len(motion_times))
                time_idx = np.random.randint(start_idx, min(end_idx, len(motion_times)))
                target_time = motion_times[time_idx]
            else:
                target_time = motion_times[0]
                time_idx = 0
        else:
            target_time = options.get("reference_time", 0.0)
            motion_times = reference_motion.get("times", [0.0])
            time_idx = np.argmin(np.abs(np.array(motion_times) - target_time))

        # Extract reference state
        ref_joint_pos = self._get_reference_joint_positions(reference_motion, time_idx)
        ref_joint_vel = self._get_reference_joint_velocities(reference_motion, time_idx)
        ref_root_pos = self._get_reference_root_position(reference_motion, time_idx)
        ref_root_quat = self._get_reference_root_orientation(reference_motion, time_idx)
        ref_root_vel = self._get_reference_root_velocity(reference_motion, time_idx)

        # Set root position and orientation
        if ref_root_pos is not None:
            self.data.qpos[:3] = ref_root_pos
        else:
            self.data.qpos[:3] = self.default_qpos[:3]

        if ref_root_quat is not None:
            self.data.qpos[3:7] = ref_root_quat / np.linalg.norm(ref_root_quat)
        else:
            self.data.qpos[3:7] = self.default_qpos[3:7]

        # Set joint positions
        if ref_joint_pos is not None:
            self._set_joint_positions(ref_joint_pos)
        else:
            self._set_joint_positions(self._get_default_joint_positions())

        # Set velocities
        if ref_root_vel is not None:
            if len(ref_root_vel) >= 6:
                self.data.qvel[:6] = ref_root_vel[:6]
            elif len(ref_root_vel) >= 3:
                self.data.qvel[:3] = ref_root_vel[:3]
                self.data.qvel[3:6] = 0.0
        else:
            self.data.qvel[:6] = 0.0

        if ref_joint_vel is not None:
            self._set_joint_velocities(ref_joint_vel)
        else:
            self._set_joint_velocities(np.zeros(self.num_joints))

    def _reset_to_default(self, options: Dict[str, Any]):
        """Reset to default pose."""
        self.data.qpos[:] = self.default_qpos
        self.data.qvel[:] = self.default_qvel

    def _reset_random(self, options: Dict[str, Any]):
        """Reset to random valid pose."""
        # Random root position (small deviation from default)
        root_noise = np.random.normal(0, 0.1, 3)
        root_noise[2] = np.abs(root_noise[2])  # Keep above ground
        self.data.qpos[:3] = self.default_qpos[:3] + root_noise

        # Random root orientation (small deviation from upright)
        euler_noise = np.random.normal(0, 0.1, 3)
        quat_noise = self._euler_to_quaternion(euler_noise)
        default_quat = self.default_qpos[3:7]
        new_quat = self._quaternion_multiply(default_quat, quat_noise)
        self.data.qpos[3:7] = new_quat / np.linalg.norm(new_quat)

        # Random joint positions within limits
        joint_positions = []
        for i, joint_info in enumerate(self.actuated_joints):
            joint_id = joint_info['id']
            joint_range = self.joint_limits[i]

            if np.allclose(joint_range, 0):
                # No limits specified, use small noise around default
                default_pos = self.default_qpos[joint_id]
                noise = np.random.normal(0, 0.1)
                joint_positions.append(default_pos + noise)
            else:
                # Random within limits (biased toward center)
                center = (joint_range[0] + joint_range[1]) / 2
                range_size = (joint_range[1] - joint_range[0]) / 4  # Use 1/4 of range
                noise = np.random.normal(0, range_size)
                position = np.clip(center + noise, joint_range[0], joint_range[1])
                joint_positions.append(position)

        self._set_joint_positions(np.array(joint_positions))

        # Small random velocities
        self.data.qvel[:] = np.random.normal(0, 0.1, len(self.data.qvel))

    def _apply_reset_noise(self, noise_scale: float):
        """Apply noise to current pose."""
        # Joint position noise
        if self.config.joint_pos_noise > 0:
            joint_noise = np.random.normal(
                0, self.config.joint_pos_noise * noise_scale, self.num_joints
            )
            current_joint_pos = self._get_joint_positions()
            new_joint_pos = current_joint_pos + joint_noise
            self._set_joint_positions(new_joint_pos)

        # Joint velocity noise
        if self.config.joint_vel_noise > 0:
            vel_noise = np.random.normal(
                0, self.config.joint_vel_noise * noise_scale, self.num_joints
            )
            current_joint_vel = self._get_joint_velocities()
            new_joint_vel = current_joint_vel + vel_noise
            self._set_joint_velocities(new_joint_vel)

        # Root position noise
        if self.config.root_pos_noise > 0:
            pos_noise = np.random.normal(
                0, self.config.root_pos_noise * noise_scale, 3
            )
            self.data.qpos[:3] += pos_noise

        # Root orientation noise
        if self.config.root_orient_noise > 0:
            orient_noise = np.random.normal(
                0, self.config.root_orient_noise * noise_scale, 3
            )
            quat_noise = self._euler_to_quaternion(orient_noise)
            current_quat = self.data.qpos[3:7]
            new_quat = self._quaternion_multiply(current_quat, quat_noise)
            self.data.qpos[3:7] = new_quat / np.linalg.norm(new_quat)

    def _enforce_safety_constraints(self):
        """Ensure reset pose satisfies safety constraints."""
        # Enforce joint limits
        for i, joint_info in enumerate(self.actuated_joints):
            joint_id = joint_info['id']
            joint_range = self.joint_limits[i]

            if not np.allclose(joint_range, 0):
                self.data.qpos[joint_id] = np.clip(
                    self.data.qpos[joint_id], joint_range[0], joint_range[1]
                )

        # Enforce height constraints
        if self.torso_id != -1:
            current_height = self.data.qpos[2]  # Assume root z is height
            if current_height < self.config.min_height:
                self.data.qpos[2] = self.config.min_height
            elif current_height > self.config.max_height:
                self.data.qpos[2] = self.config.max_height

        # Limit joint velocities
        max_init_vel = 5.0  # rad/s
        joint_velocities = self._get_joint_velocities()
        clipped_velocities = np.clip(joint_velocities, -max_init_vel, max_init_vel)
        self._set_joint_velocities(clipped_velocities)

    def _get_joint_positions(self) -> np.ndarray:
        """Get current joint positions."""
        positions = []
        for joint_info in self.actuated_joints:
            positions.append(self.data.qpos[joint_info['id']])
        return np.array(positions)

    def _get_joint_velocities(self) -> np.ndarray:
        """Get current joint velocities."""
        velocities = []
        for joint_info in self.actuated_joints:
            velocities.append(self.data.qvel[joint_info['id']])
        return np.array(velocities)

    def _set_joint_positions(self, positions: np.ndarray):
        """Set joint positions."""
        for i, joint_info in enumerate(self.actuated_joints):
            if i < len(positions):
                self.data.qpos[joint_info['id']] = positions[i]

    def _set_joint_velocities(self, velocities: np.ndarray):
        """Set joint velocities."""
        for i, joint_info in enumerate(self.actuated_joints):
            if i < len(velocities):
                self.data.qvel[joint_info['id']] = velocities[i]

    def _get_default_joint_positions(self) -> np.ndarray:
        """Get default joint positions."""
        positions = []
        for joint_info in self.actuated_joints:
            positions.append(self.default_qpos[joint_info['id']])
        return np.array(positions)

    def _get_reference_joint_positions(
        self,
        reference_motion: Dict[str, Any],
        time_idx: int
    ) -> Optional[np.ndarray]:
        """Extract joint positions from reference motion at given time."""
        joint_positions = reference_motion.get("joint_positions", [])
        if time_idx < len(joint_positions):
            return np.array(joint_positions[time_idx])
        return None

    def _get_reference_joint_velocities(
        self,
        reference_motion: Dict[str, Any],
        time_idx: int
    ) -> Optional[np.ndarray]:
        """Extract joint velocities from reference motion at given time."""
        joint_velocities = reference_motion.get("joint_velocities", [])
        if time_idx < len(joint_velocities):
            return np.array(joint_velocities[time_idx])
        return None

    def _get_reference_root_position(
        self,
        reference_motion: Dict[str, Any],
        time_idx: int
    ) -> Optional[np.ndarray]:
        """Extract root position from reference motion at given time."""
        root_positions = reference_motion.get("root_positions", [])
        if time_idx < len(root_positions):
            return np.array(root_positions[time_idx])
        return None

    def _get_reference_root_orientation(
        self,
        reference_motion: Dict[str, Any],
        time_idx: int
    ) -> Optional[np.ndarray]:
        """Extract root orientation from reference motion at given time."""
        root_orientations = reference_motion.get("root_orientations", [])
        if time_idx < len(root_orientations):
            return np.array(root_orientations[time_idx])
        return None

    def _get_reference_root_velocity(
        self,
        reference_motion: Dict[str, Any],
        time_idx: int
    ) -> Optional[np.ndarray]:
        """Extract root velocity from reference motion at given time."""
        root_velocities = reference_motion.get("root_velocities", [])
        if time_idx < len(root_velocities):
            return np.array(root_velocities[time_idx])

        # Fallback: try separate linear and angular velocities
        lin_vels = reference_motion.get("root_linear_velocities", [])
        ang_vels = reference_motion.get("root_angular_velocities", [])

        if time_idx < len(lin_vels) and time_idx < len(ang_vels):
            lin_vel = np.array(lin_vels[time_idx])
            ang_vel = np.array(ang_vels[time_idx])
            return np.concatenate([lin_vel, ang_vel])

        return None

    def _euler_to_quaternion(self, euler: np.ndarray) -> np.ndarray:
        """Convert Euler angles to quaternion."""
        roll, pitch, yaw = euler

        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return np.array([w, x, y, z])

    def _quaternion_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return np.array([w, x, y, z])

    def get_reset_statistics(self) -> Dict[str, Any]:
        """Get reset statistics."""
        return {
            "total_resets": self.reset_count,
            "strategies": self.reset_strategies,
            "default_pose": self.default_qpos.copy(),
        }