"""
Observation management for Unitree G1 environment.

This module handles the construction and management of observations including
proprioceptive information, reference motion features, and optional command
conditioning features.
"""

import numpy as np
import mujoco
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class ObservationConfig:
    """Configuration for observations."""
    # Proprioceptive observations
    include_joint_pos: bool = True
    include_joint_vel: bool = True
    include_root_orientation: bool = True
    include_root_linear_vel: bool = True
    include_root_angular_vel: bool = True
    include_previous_action: bool = True

    # Reference motion features
    include_reference_motion: bool = True
    reference_horizon: int = 10  # Number of future timesteps

    # Command features
    include_commands: bool = False

    # Additional features
    include_foot_contacts: bool = True
    include_imu: bool = True
    include_height_scan: bool = False
    height_scan_points: int = 17

    # Noise settings
    joint_pos_noise: float = 0.01
    joint_vel_noise: float = 0.1
    imu_noise: float = 0.05

    # Normalization
    normalize_observations: bool = True


class ObservationManager:
    """
    Manages observation construction for the G1 environment.

    This class handles:
    - Proprioceptive observations (joints, IMU, contacts)
    - Reference motion features for imitation
    - Command conditioning features
    - Observation normalization and noise injection
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        config: Dict[str, Any]
    ):
        """
        Initialize observation manager.

        Args:
            model: MuJoCo model
            data: MuJoCo data
            config: Configuration dictionary
        """
        self.model = model
        self.data = data

        # Parse configuration
        self.config = ObservationConfig(**config)

        # Get robot information
        self._get_robot_info()

        # Calculate observation dimensions
        self._calculate_obs_dims()

        # Initialize normalization parameters
        self._init_normalization()

        print(f"ObservationManager initialized with {self.obs_dim} dimensions")

    def _get_robot_info(self):
        """Extract robot-specific information from model."""
        # Find actuated joints
        self.actuated_joints = []
        for i in range(self.model.nu):
            actuator_id = i
            joint_id = self.model.actuator_trnid[actuator_id, 0]
            self.actuated_joints.append(joint_id)

        self.num_joints = len(self.actuated_joints)

        # Find torso body for IMU
        self.torso_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link"
        )
        if self.torso_id == -1:
            self.torso_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, "torso"
            )

        # Find foot contact sites/bodies
        foot_names = ["left_foot", "right_foot", "left_ankle", "right_ankle"]
        self.foot_contact_ids = []
        for name in foot_names:
            # Try to find as geom first
            geom_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, name
            )
            if geom_id != -1:
                self.foot_contact_ids.append(("geom", geom_id))
                continue

            # Try to find as body
            body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, name
            )
            if body_id != -1:
                self.foot_contact_ids.append(("body", body_id))

    def _calculate_obs_dims(self):
        """Calculate total observation dimensions."""
        dims = 0

        # Proprioceptive observations
        if self.config.include_joint_pos:
            dims += self.num_joints
        if self.config.include_joint_vel:
            dims += self.num_joints
        if self.config.include_root_orientation:
            dims += 4  # quaternion
        if self.config.include_root_linear_vel:
            dims += 3
        if self.config.include_root_angular_vel:
            dims += 3
        if self.config.include_previous_action:
            dims += self.num_joints

        # Contact information
        if self.config.include_foot_contacts:
            dims += len(self.foot_contact_ids)

        # IMU information
        if self.config.include_imu:
            dims += 3  # linear acceleration

        # Height scan
        if self.config.include_height_scan:
            dims += self.config.height_scan_points

        # Reference motion features
        if self.config.include_reference_motion:
            # Joint positions + root pose + velocities for reference horizon
            ref_dims = (self.num_joints + 7 + 6) * self.config.reference_horizon
            dims += ref_dims

        # Command features
        if self.config.include_commands:
            dims += 3  # [forward_vel, lateral_vel, yaw_rate]

        self.obs_dim = dims

    def _init_normalization(self):
        """Initialize observation normalization parameters."""
        if not self.config.normalize_observations:
            return

        # Joint position limits
        self.joint_pos_mean = np.zeros(self.num_joints)
        self.joint_pos_scale = np.ones(self.num_joints)

        for i, joint_id in enumerate(self.actuated_joints):
            joint_range = self.model.jnt_range[joint_id]
            if not np.allclose(joint_range, 0):
                self.joint_pos_mean[i] = (joint_range[0] + joint_range[1]) / 2
                self.joint_pos_scale[i] = (joint_range[1] - joint_range[0]) / 2

        # Velocity normalization (heuristic)
        self.joint_vel_scale = np.full(self.num_joints, 10.0)

        # Root velocity normalization
        self.root_vel_scale = 5.0  # m/s
        self.root_angvel_scale = 10.0  # rad/s

    def get_observation_dim(self) -> int:
        """Get total observation dimension."""
        return self.obs_dim

    def get_observation(
        self,
        last_action: Optional[np.ndarray] = None,
        reference_motion: Optional[Dict[str, Any]] = None,
        command: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Get current observation.

        Args:
            last_action: Previous action taken
            reference_motion: Reference motion data
            command: Current command

        Returns:
            Observation vector
        """
        obs_parts = []

        # Proprioceptive observations
        if self.config.include_joint_pos:
            joint_pos = self._get_joint_positions()
            if self.config.normalize_observations:
                joint_pos = (joint_pos - self.joint_pos_mean) / self.joint_pos_scale
            # Add noise
            if self.config.joint_pos_noise > 0:
                joint_pos += np.random.normal(0, self.config.joint_pos_noise, joint_pos.shape)
            obs_parts.append(joint_pos)

        if self.config.include_joint_vel:
            joint_vel = self._get_joint_velocities()
            if self.config.normalize_observations:
                joint_vel = joint_vel / self.joint_vel_scale
            # Add noise
            if self.config.joint_vel_noise > 0:
                joint_vel += np.random.normal(0, self.config.joint_vel_noise, joint_vel.shape)
            obs_parts.append(joint_vel)

        if self.config.include_root_orientation:
            root_quat = self._get_root_orientation()
            obs_parts.append(root_quat)

        if self.config.include_root_linear_vel:
            root_vel = self._get_root_linear_velocity()
            if self.config.normalize_observations:
                root_vel = root_vel / self.root_vel_scale
            obs_parts.append(root_vel)

        if self.config.include_root_angular_vel:
            root_angvel = self._get_root_angular_velocity()
            if self.config.normalize_observations:
                root_angvel = root_angvel / self.root_angvel_scale
            obs_parts.append(root_angvel)

        if self.config.include_previous_action:
            if last_action is not None:
                obs_parts.append(last_action)
            else:
                obs_parts.append(np.zeros(self.num_joints))

        # Contact observations
        if self.config.include_foot_contacts:
            contacts = self._get_foot_contacts()
            obs_parts.append(contacts)

        # IMU observations
        if self.config.include_imu:
            imu_data = self._get_imu_data()
            if self.config.imu_noise > 0:
                imu_data += np.random.normal(0, self.config.imu_noise, imu_data.shape)
            obs_parts.append(imu_data)

        # Height scan
        if self.config.include_height_scan:
            height_scan = self._get_height_scan()
            obs_parts.append(height_scan)

        # Reference motion features
        if self.config.include_reference_motion:
            ref_features = self._get_reference_features(reference_motion)
            obs_parts.append(ref_features)

        # Command features
        if self.config.include_commands:
            cmd_features = self._get_command_features(command)
            obs_parts.append(cmd_features)

        # Concatenate all observation parts
        observation = np.concatenate(obs_parts)

        # Final safety check
        observation = np.clip(observation, -10, 10)  # Prevent extreme values

        return observation.astype(np.float32)

    def _get_joint_positions(self) -> np.ndarray:
        """Get current joint positions."""
        positions = []
        for joint_id in self.actuated_joints:
            positions.append(self.data.qpos[joint_id])
        return np.array(positions)

    def _get_joint_velocities(self) -> np.ndarray:
        """Get current joint velocities."""
        velocities = []
        for joint_id in self.actuated_joints:
            velocities.append(self.data.qvel[joint_id])
        return np.array(velocities)

    def _get_root_orientation(self) -> np.ndarray:
        """Get root orientation as quaternion."""
        return self.data.qpos[3:7].copy()

    def _get_root_linear_velocity(self) -> np.ndarray:
        """Get root linear velocity."""
        return self.data.qvel[:3].copy()

    def _get_root_angular_velocity(self) -> np.ndarray:
        """Get root angular velocity."""
        return self.data.qvel[3:6].copy()

    def _get_foot_contacts(self) -> np.ndarray:
        """Get binary foot contact information."""
        contacts = []

        for contact_type, contact_id in self.foot_contact_ids:
            in_contact = False

            # Check all contacts
            for i in range(self.data.ncon):
                contact = self.data.contact[i]

                if contact_type == "geom":
                    if contact.geom1 == contact_id or contact.geom2 == contact_id:
                        in_contact = True
                        break
                elif contact_type == "body":
                    # Convert body ID to geom IDs and check
                    geom1_body = self.model.geom_bodyid[contact.geom1]
                    geom2_body = self.model.geom_bodyid[contact.geom2]
                    if geom1_body == contact_id or geom2_body == contact_id:
                        in_contact = True
                        break

            contacts.append(float(in_contact))

        return np.array(contacts)

    def _get_imu_data(self) -> np.ndarray:
        """Get IMU data (linear acceleration in world frame)."""
        if self.torso_id == -1:
            return np.zeros(3)

        # Get linear acceleration
        # Note: In MuJoCo, acceleration is not directly available
        # We approximate it as the time derivative of velocity
        # For proper IMU simulation, you'd want to use sensors

        # For now, return gravitational acceleration modified by orientation
        torso_quat = self.data.xquat[self.torso_id]
        gravity_world = np.array([0, 0, -9.81])

        # Transform gravity to body frame
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, torso_quat)
        R = R.reshape(3, 3)

        # Gravity in body frame (what IMU would measure when static)
        gravity_body = R.T @ gravity_world

        # Add linear acceleration (simplified)
        # In practice, you'd differentiate velocity or use MuJoCo sensors
        acc_body = gravity_body  # + linear_acceleration_body

        return acc_body

    def _get_height_scan(self) -> np.ndarray:
        """Get height scan around robot."""
        if self.torso_id == -1:
            return np.zeros(self.config.height_scan_points)

        # Get robot position
        robot_pos = self.data.xpos[self.torso_id]

        # Generate scan points in a circle around robot
        angles = np.linspace(0, 2*np.pi, self.config.height_scan_points, endpoint=False)
        scan_radius = 0.5  # meters

        heights = []
        for angle in angles:
            # Calculate scan point
            scan_x = robot_pos[0] + scan_radius * np.cos(angle)
            scan_y = robot_pos[1] + scan_radius * np.sin(angle)

            # Raycast down to find ground height
            # For simplicity, assume flat ground at z=0
            ground_height = 0.0
            relative_height = ground_height - robot_pos[2]
            heights.append(relative_height)

        return np.array(heights)

    def _get_reference_features(
        self,
        reference_motion: Optional[Dict[str, Any]]
    ) -> np.ndarray:
        """Get reference motion features."""
        ref_dim = (self.num_joints + 7 + 6) * self.config.reference_horizon

        if reference_motion is None:
            return np.zeros(ref_dim)

        # Extract reference features for future timesteps
        ref_features = []

        current_time = reference_motion.get("current_time", 0.0)
        dt = reference_motion.get("dt", 0.02)

        for i in range(self.config.reference_horizon):
            future_time = current_time + (i + 1) * dt

            # Get reference state at future time
            ref_joint_pos = self._interpolate_reference_data(
                reference_motion.get("joint_positions", []),
                reference_motion.get("times", []),
                future_time
            )

            ref_root_pos = self._interpolate_reference_data(
                reference_motion.get("root_positions", []),
                reference_motion.get("times", []),
                future_time
            )

            ref_root_quat = self._interpolate_reference_data(
                reference_motion.get("root_orientations", []),
                reference_motion.get("times", []),
                future_time
            )

            ref_root_vel = self._interpolate_reference_data(
                reference_motion.get("root_velocities", []),
                reference_motion.get("times", []),
                future_time
            )

            ref_root_angvel = self._interpolate_reference_data(
                reference_motion.get("root_angular_velocities", []),
                reference_motion.get("times", []),
                future_time
            )

            # Normalize if needed
            if self.config.normalize_observations:
                if len(ref_joint_pos) > 0:
                    ref_joint_pos = (ref_joint_pos - self.joint_pos_mean) / self.joint_pos_scale
                if len(ref_root_vel) > 0:
                    ref_root_vel = ref_root_vel / self.root_vel_scale
                if len(ref_root_angvel) > 0:
                    ref_root_angvel = ref_root_angvel / self.root_angvel_scale

            # Concatenate reference features for this timestep
            timestep_features = np.concatenate([
                ref_joint_pos,
                ref_root_pos,
                ref_root_quat,
                ref_root_vel,
                ref_root_angvel
            ])

            ref_features.append(timestep_features)

        return np.concatenate(ref_features)

    def _get_command_features(
        self,
        command: Optional[Dict[str, Any]]
    ) -> np.ndarray:
        """Get command features."""
        if command is None:
            return np.zeros(3)

        return np.array([
            command.get("forward_vel", 0.0),
            command.get("lateral_vel", 0.0),
            command.get("yaw_rate", 0.0)
        ])

    def _interpolate_reference_data(
        self,
        data_array: List[np.ndarray],
        times: List[float],
        target_time: float
    ) -> np.ndarray:
        """Interpolate reference data to target time."""
        if not data_array or not times:
            return np.zeros(self.num_joints)  # Default size

        data_array = np.array(data_array)
        times = np.array(times)

        if len(data_array) == 0:
            return np.zeros(data_array.shape[-1] if len(data_array.shape) > 1 else self.num_joints)

        # Find surrounding time indices
        if target_time <= times[0]:
            return data_array[0]
        elif target_time >= times[-1]:
            return data_array[-1]
        else:
            # Linear interpolation
            idx = np.searchsorted(times, target_time)
            t0, t1 = times[idx-1], times[idx]
            alpha = (target_time - t0) / (t1 - t0)
            return (1 - alpha) * data_array[idx-1] + alpha * data_array[idx]