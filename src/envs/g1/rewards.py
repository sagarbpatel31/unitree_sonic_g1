"""
Reward computation for Unitree G1 environment.

This module implements comprehensive reward functions for motion imitation
including joint tracking, root pose tracking, contact consistency, and
various penalty terms for natural and stable motion.
"""

import numpy as np
import mujoco
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class RewardConfig:
    """Configuration for reward terms."""
    # Joint tracking
    joint_pos_weight: float = 1.0
    joint_vel_weight: float = 0.1

    # Root tracking
    root_pos_weight: float = 1.0
    root_orient_weight: float = 1.0
    root_vel_weight: float = 0.5
    root_angvel_weight: float = 0.1

    # End effector tracking
    end_effector_weight: float = 0.5

    # Contact consistency
    foot_contact_weight: float = 0.2
    foot_slip_weight: float = 0.1

    # Stability and naturalness
    upright_weight: float = 0.1
    action_smoothness_weight: float = 0.05
    torque_penalty_weight: float = 0.001
    energy_penalty_weight: float = 0.001

    # Alive bonus
    alive_bonus: float = 0.1

    # Task-specific rewards
    command_tracking_weight: float = 0.5

    # Scaling factors
    joint_pos_scale: float = 10.0
    joint_vel_scale: float = 1.0
    root_pos_scale: float = 20.0
    root_orient_scale: float = 5.0


class RewardManager:
    """
    Manages reward computation for the G1 environment.

    This class implements various reward terms:
    - Motion tracking rewards for imitation
    - Stability and naturalness rewards
    - Contact consistency rewards
    - Energy and effort penalties
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        config: Dict[str, Any]
    ):
        """
        Initialize reward manager.

        Args:
            model: MuJoCo model
            data: MuJoCo data
            config: Configuration dictionary
        """
        self.model = model
        self.data = data

        # Parse configuration
        self.config = RewardConfig(**config)

        # Get robot information
        self._get_robot_info()

        # Initialize tracking variables
        self.prev_action = None
        self.prev_joint_pos = None
        self.prev_torques = None

        print(f"RewardManager initialized with {len(self._get_reward_terms())} reward terms")

    def _get_robot_info(self):
        """Extract robot-specific information from model."""
        # Find actuated joints
        self.actuated_joints = []
        for i in range(self.model.nu):
            actuator_id = i
            joint_id = self.model.actuator_trnid[actuator_id, 0]
            self.actuated_joints.append({
                'id': joint_id,
                'actuator_id': actuator_id
            })

        self.num_joints = len(self.actuated_joints)

        # Find torso body
        self.torso_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link"
        )
        if self.torso_id == -1:
            self.torso_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, "torso"
            )

        # Find foot bodies/geoms
        foot_names = ["left_foot", "right_foot", "left_ankle", "right_ankle"]
        self.foot_ids = []
        for name in foot_names:
            # Try body first
            body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, name
            )
            if body_id != -1:
                self.foot_ids.append(("body", body_id))
                continue

            # Try geom
            geom_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, name
            )
            if geom_id != -1:
                self.foot_ids.append(("geom", geom_id))

        # Find end effector bodies
        ee_names = ["left_hand", "right_hand", "left_gripper", "right_gripper"]
        self.ee_ids = []
        for name in ee_names:
            body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, name
            )
            if body_id != -1:
                self.ee_ids.append(body_id)

    def reset(self):
        """Reset reward manager for new episode."""
        self.prev_action = None
        self.prev_joint_pos = None
        self.prev_torques = None

    def compute_reward(
        self,
        action: np.ndarray,
        reference_motion: Optional[Dict[str, Any]] = None,
        command: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Compute total reward and individual components.

        Args:
            action: Action taken this step
            reference_motion: Reference motion data
            command: Current command

        Returns:
            Dictionary with total reward and individual components
        """
        reward_terms = {}

        # Joint tracking rewards
        if reference_motion is not None:
            joint_pos_reward = self._compute_joint_position_reward(reference_motion)
            joint_vel_reward = self._compute_joint_velocity_reward(reference_motion)

            reward_terms["joint_pos"] = joint_pos_reward
            reward_terms["joint_vel"] = joint_vel_reward
        else:
            reward_terms["joint_pos"] = 0.0
            reward_terms["joint_vel"] = 0.0

        # Root tracking rewards
        if reference_motion is not None:
            root_pos_reward = self._compute_root_position_reward(reference_motion)
            root_orient_reward = self._compute_root_orientation_reward(reference_motion)
            root_vel_reward = self._compute_root_velocity_reward(reference_motion)

            reward_terms["root_pos"] = root_pos_reward
            reward_terms["root_orient"] = root_orient_reward
            reward_terms["root_vel"] = root_vel_reward
        else:
            reward_terms["root_pos"] = 0.0
            reward_terms["root_orient"] = 0.0
            reward_terms["root_vel"] = 0.0

        # End effector tracking
        if reference_motion is not None and self.ee_ids:
            ee_reward = self._compute_end_effector_reward(reference_motion)
            reward_terms["end_effector"] = ee_reward
        else:
            reward_terms["end_effector"] = 0.0

        # Contact rewards
        if reference_motion is not None:
            contact_reward = self._compute_contact_reward(reference_motion)
            slip_penalty = self._compute_foot_slip_penalty()

            reward_terms["foot_contact"] = contact_reward
            reward_terms["foot_slip"] = -slip_penalty
        else:
            reward_terms["foot_contact"] = 0.0
            reward_terms["foot_slip"] = 0.0

        # Stability rewards
        upright_reward = self._compute_upright_reward()
        reward_terms["upright"] = upright_reward

        # Action smoothness
        smoothness_penalty = self._compute_action_smoothness_penalty(action)
        reward_terms["action_smoothness"] = -smoothness_penalty

        # Energy penalties
        torque_penalty = self._compute_torque_penalty()
        energy_penalty = self._compute_energy_penalty()

        reward_terms["torque_penalty"] = -torque_penalty
        reward_terms["energy_penalty"] = -energy_penalty

        # Alive bonus
        reward_terms["alive"] = self.config.alive_bonus

        # Command tracking reward
        if command is not None:
            command_reward = self._compute_command_tracking_reward(command)
            reward_terms["command_tracking"] = command_reward
        else:
            reward_terms["command_tracking"] = 0.0

        # Compute total reward
        total_reward = self._compute_total_reward(reward_terms)
        reward_terms["total_reward"] = total_reward

        # Update tracking variables
        self.prev_action = action.copy()
        self.prev_joint_pos = self._get_joint_positions()
        self.prev_torques = self._get_joint_torques()

        return reward_terms

    def _compute_total_reward(self, reward_terms: Dict[str, float]) -> float:
        """Compute weighted sum of all reward terms."""
        total = 0.0

        total += self.config.joint_pos_weight * reward_terms["joint_pos"]
        total += self.config.joint_vel_weight * reward_terms["joint_vel"]
        total += self.config.root_pos_weight * reward_terms["root_pos"]
        total += self.config.root_orient_weight * reward_terms["root_orient"]
        total += self.config.root_vel_weight * reward_terms["root_vel"]
        total += self.config.end_effector_weight * reward_terms["end_effector"]
        total += self.config.foot_contact_weight * reward_terms["foot_contact"]
        total += self.config.foot_slip_weight * reward_terms["foot_slip"]
        total += self.config.upright_weight * reward_terms["upright"]
        total += self.config.action_smoothness_weight * reward_terms["action_smoothness"]
        total += self.config.torque_penalty_weight * reward_terms["torque_penalty"]
        total += self.config.energy_penalty_weight * reward_terms["energy_penalty"]
        total += reward_terms["alive"]
        total += self.config.command_tracking_weight * reward_terms["command_tracking"]

        return total

    def _compute_joint_position_reward(self, reference_motion: Dict[str, Any]) -> float:
        """Compute reward for joint position tracking."""
        current_joint_pos = self._get_joint_positions()
        ref_joint_pos = self._get_reference_joint_positions(reference_motion)

        if ref_joint_pos is None or len(ref_joint_pos) != len(current_joint_pos):
            return 0.0

        # Compute position error
        pos_error = np.linalg.norm(current_joint_pos - ref_joint_pos)

        # Exponential reward
        reward = np.exp(-self.config.joint_pos_scale * pos_error)

        return reward

    def _compute_joint_velocity_reward(self, reference_motion: Dict[str, Any]) -> float:
        """Compute reward for joint velocity tracking."""
        current_joint_vel = self._get_joint_velocities()
        ref_joint_vel = self._get_reference_joint_velocities(reference_motion)

        if ref_joint_vel is None or len(ref_joint_vel) != len(current_joint_vel):
            return 0.0

        # Compute velocity error
        vel_error = np.linalg.norm(current_joint_vel - ref_joint_vel)

        # Exponential reward
        reward = np.exp(-self.config.joint_vel_scale * vel_error)

        return reward

    def _compute_root_position_reward(self, reference_motion: Dict[str, Any]) -> float:
        """Compute reward for root position tracking."""
        current_root_pos = self.data.qpos[:3]
        ref_root_pos = self._get_reference_root_position(reference_motion)

        if ref_root_pos is None:
            return 0.0

        # Compute position error (only x, y - ignore z for height tolerance)
        pos_error = np.linalg.norm(current_root_pos[:2] - ref_root_pos[:2])

        # Exponential reward
        reward = np.exp(-self.config.root_pos_scale * pos_error)

        return reward

    def _compute_root_orientation_reward(self, reference_motion: Dict[str, Any]) -> float:
        """Compute reward for root orientation tracking."""
        current_root_quat = self.data.qpos[3:7]
        ref_root_quat = self._get_reference_root_orientation(reference_motion)

        if ref_root_quat is None:
            return 0.0

        # Compute quaternion difference
        quat_diff = self._quaternion_distance(current_root_quat, ref_root_quat)

        # Exponential reward
        reward = np.exp(-self.config.root_orient_scale * quat_diff)

        return reward

    def _compute_root_velocity_reward(self, reference_motion: Dict[str, Any]) -> float:
        """Compute reward for root velocity tracking."""
        current_root_vel = self.data.qvel[:6]  # linear + angular
        ref_root_vel = self._get_reference_root_velocity(reference_motion)

        if ref_root_vel is None:
            return 0.0

        # Compute velocity error
        vel_error = np.linalg.norm(current_root_vel - ref_root_vel)

        # Exponential reward
        reward = np.exp(-2.0 * vel_error)

        return reward

    def _compute_end_effector_reward(self, reference_motion: Dict[str, Any]) -> float:
        """Compute reward for end effector tracking."""
        if not self.ee_ids:
            return 0.0

        total_reward = 0.0
        count = 0

        ref_ee_poses = self._get_reference_end_effector_poses(reference_motion)
        if ref_ee_poses is None:
            return 0.0

        for i, ee_id in enumerate(self.ee_ids):
            if i >= len(ref_ee_poses):
                continue

            current_pos = self.data.xpos[ee_id]
            ref_pos = ref_ee_poses[i]

            if ref_pos is not None:
                pos_error = np.linalg.norm(current_pos - ref_pos)
                reward = np.exp(-10.0 * pos_error)
                total_reward += reward
                count += 1

        return total_reward / max(count, 1)

    def _compute_contact_reward(self, reference_motion: Dict[str, Any]) -> float:
        """Compute reward for foot contact consistency."""
        current_contacts = self._get_foot_contacts()
        ref_contacts = self._get_reference_foot_contacts(reference_motion)

        if ref_contacts is None:
            return 0.0

        # Compute contact agreement
        contact_agreement = 0.0
        for i in range(min(len(current_contacts), len(ref_contacts))):
            if current_contacts[i] == ref_contacts[i]:
                contact_agreement += 1.0

        return contact_agreement / max(len(current_contacts), 1)

    def _compute_foot_slip_penalty(self) -> float:
        """Compute penalty for foot slipping."""
        slip_penalty = 0.0

        for foot_type, foot_id in self.foot_ids:
            if self._is_foot_in_contact(foot_type, foot_id):
                # Check foot velocity
                if foot_type == "body":
                    foot_vel = self._get_body_velocity(foot_id)
                else:
                    # For geom, approximate using nearby body
                    body_id = self.model.geom_bodyid[foot_id]
                    foot_vel = self._get_body_velocity(body_id)

                # Penalty for horizontal velocity when in contact
                horizontal_vel = np.linalg.norm(foot_vel[:2])
                slip_penalty += horizontal_vel

        return slip_penalty

    def _compute_upright_reward(self) -> float:
        """Compute reward for staying upright."""
        if self.torso_id == -1:
            return 0.0

        # Get torso orientation
        torso_quat = self.data.xquat[self.torso_id]

        # Convert to rotation matrix
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, torso_quat)
        R = R.reshape(3, 3)

        # Get z-axis (up vector) in world frame
        up_vector = R @ np.array([0, 0, 1])

        # Reward for being upright (z-component close to 1)
        upright_reward = up_vector[2]

        return max(0.0, upright_reward)

    def _compute_action_smoothness_penalty(self, action: np.ndarray) -> float:
        """Compute penalty for non-smooth actions."""
        if self.prev_action is None:
            return 0.0

        # L2 norm of action difference
        action_diff = action - self.prev_action
        smoothness_penalty = np.linalg.norm(action_diff)

        return smoothness_penalty

    def _compute_torque_penalty(self) -> float:
        """Compute penalty for high joint torques."""
        joint_torques = self._get_joint_torques()
        torque_penalty = np.sum(np.square(joint_torques))

        return torque_penalty

    def _compute_energy_penalty(self) -> float:
        """Compute penalty for high energy consumption."""
        joint_torques = self._get_joint_torques()
        joint_velocities = self._get_joint_velocities()

        # Power = torque * velocity
        power = np.abs(joint_torques * joint_velocities)
        energy_penalty = np.sum(power)

        return energy_penalty

    def _compute_command_tracking_reward(self, command: Dict[str, Any]) -> float:
        """Compute reward for following velocity commands."""
        if self.torso_id == -1:
            return 0.0

        # Get current root velocity
        current_vel = self.data.qvel[:3]  # linear velocity

        # Get desired velocity from command
        desired_vel = np.array([
            command.get("forward_vel", 0.0),
            command.get("lateral_vel", 0.0),
            0.0  # No vertical velocity command
        ])

        # Compute velocity error
        vel_error = np.linalg.norm(current_vel - desired_vel)

        # Exponential reward
        reward = np.exp(-5.0 * vel_error)

        return reward

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

    def _get_joint_torques(self) -> np.ndarray:
        """Get current joint torques."""
        torques = []
        for joint_info in self.actuated_joints:
            torques.append(self.data.qfrc_actuator[joint_info['actuator_id']])
        return np.array(torques)

    def _get_foot_contacts(self) -> List[bool]:
        """Get current foot contact states."""
        contacts = []
        for foot_type, foot_id in self.foot_ids:
            in_contact = self._is_foot_in_contact(foot_type, foot_id)
            contacts.append(in_contact)
        return contacts

    def _is_foot_in_contact(self, foot_type: str, foot_id: int) -> bool:
        """Check if foot is in contact with ground."""
        for i in range(self.data.ncon):
            contact = self.data.contact[i]

            if foot_type == "geom":
                if contact.geom1 == foot_id or contact.geom2 == foot_id:
                    return True
            elif foot_type == "body":
                geom1_body = self.model.geom_bodyid[contact.geom1]
                geom2_body = self.model.geom_bodyid[contact.geom2]
                if geom1_body == foot_id or geom2_body == foot_id:
                    return True

        return False

    def _get_body_velocity(self, body_id: int) -> np.ndarray:
        """Get 6D velocity of a body (linear + angular)."""
        if body_id >= self.model.nbody:
            return np.zeros(6)

        # Get body velocity using MuJoCo function
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_BODY,
            body_id, vel, 0
        )
        return vel

    def _quaternion_distance(self, q1: np.ndarray, q2: np.ndarray) -> float:
        """Compute angular distance between two quaternions."""
        # Ensure quaternions are normalized
        q1 = q1 / np.linalg.norm(q1)
        q2 = q2 / np.linalg.norm(q2)

        # Compute dot product
        dot = np.abs(np.dot(q1, q2))

        # Clamp to avoid numerical issues
        dot = np.clip(dot, 0.0, 1.0)

        # Angular distance
        angle = 2 * np.arccos(dot)

        return angle

    def _get_reference_joint_positions(self, reference_motion: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract reference joint positions from motion data."""
        return reference_motion.get("joint_positions")

    def _get_reference_joint_velocities(self, reference_motion: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract reference joint velocities from motion data."""
        return reference_motion.get("joint_velocities")

    def _get_reference_root_position(self, reference_motion: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract reference root position from motion data."""
        return reference_motion.get("root_position")

    def _get_reference_root_orientation(self, reference_motion: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract reference root orientation from motion data."""
        return reference_motion.get("root_orientation")

    def _get_reference_root_velocity(self, reference_motion: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract reference root velocity from motion data."""
        root_lin_vel = reference_motion.get("root_linear_velocity")
        root_ang_vel = reference_motion.get("root_angular_velocity")

        if root_lin_vel is not None and root_ang_vel is not None:
            return np.concatenate([root_lin_vel, root_ang_vel])

        return None

    def _get_reference_end_effector_poses(self, reference_motion: Dict[str, Any]) -> Optional[List[np.ndarray]]:
        """Extract reference end effector poses from motion data."""
        return reference_motion.get("end_effector_positions")

    def _get_reference_foot_contacts(self, reference_motion: Dict[str, Any]) -> Optional[List[bool]]:
        """Extract reference foot contacts from motion data."""
        return reference_motion.get("foot_contacts")

    def _get_reward_terms(self) -> List[str]:
        """Get list of all reward terms."""
        return [
            "joint_pos", "joint_vel", "root_pos", "root_orient", "root_vel",
            "end_effector", "foot_contact", "foot_slip", "upright",
            "action_smoothness", "torque_penalty", "energy_penalty",
            "alive", "command_tracking"
        ]