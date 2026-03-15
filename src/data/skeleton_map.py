"""
Skeleton mapping for motion retargeting.

This module defines joint mappings between human skeletons (AMASS, CMU, etc.)
and the Unitree G1 robot, handling joint name translation, DOF mapping,
and kinematic chain correspondence.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum


class JointType(Enum):
    """Types of joints for mapping."""
    REVOLUTE = "revolute"
    PRISMATIC = "prismatic"
    SPHERICAL = "spherical"
    FIXED = "fixed"


@dataclass
class JointInfo:
    """Information about a joint in the skeleton."""
    name: str
    type: JointType
    dof: int
    parent: Optional[str] = None
    children: List[str] = None
    axis: Optional[np.ndarray] = None
    limits: Optional[Tuple[float, float]] = None
    rest_position: float = 0.0

    def __post_init__(self):
        if self.children is None:
            self.children = []


# ROBOT-SPECIFIC ASSUMPTION: Unitree G1 joint configuration
# This mapping assumes the G1 has the following joint structure:
# - 6 DOF per leg (hip_yaw, hip_roll, hip_pitch, knee_pitch, ankle_pitch, ankle_roll)
# - 2 DOF torso (torso_pitch, torso_roll)
# - 3 DOF per arm (shoulder_pitch, shoulder_roll, elbow_pitch)
# - 2 DOF head (head_yaw, head_pitch)
# Total: 22 DOF
G1_JOINT_MAP = {
    # Left leg (6 DOF)
    "left_hip_yaw": JointInfo("left_hip_yaw", JointType.REVOLUTE, 1,
                              limits=(-1.57, 1.57), axis=np.array([0, 0, 1])),
    "left_hip_roll": JointInfo("left_hip_roll", JointType.REVOLUTE, 1,
                               limits=(-0.52, 0.52), axis=np.array([1, 0, 0])),
    "left_hip_pitch": JointInfo("left_hip_pitch", JointType.REVOLUTE, 1,
                                limits=(-1.57, 1.57), axis=np.array([0, 1, 0])),
    "left_knee_pitch": JointInfo("left_knee_pitch", JointType.REVOLUTE, 1,
                                 limits=(0.0, 2.35), axis=np.array([0, 1, 0])),
    "left_ankle_pitch": JointInfo("left_ankle_pitch", JointType.REVOLUTE, 1,
                                  limits=(-0.87, 0.87), axis=np.array([0, 1, 0])),
    "left_ankle_roll": JointInfo("left_ankle_roll", JointType.REVOLUTE, 1,
                                 limits=(-0.52, 0.52), axis=np.array([1, 0, 0])),

    # Right leg (6 DOF)
    "right_hip_yaw": JointInfo("right_hip_yaw", JointType.REVOLUTE, 1,
                               limits=(-1.57, 1.57), axis=np.array([0, 0, 1])),
    "right_hip_roll": JointInfo("right_hip_roll", JointType.REVOLUTE, 1,
                                limits=(-0.52, 0.52), axis=np.array([1, 0, 0])),
    "right_hip_pitch": JointInfo("right_hip_pitch", JointType.REVOLUTE, 1,
                                 limits=(-1.57, 1.57), axis=np.array([0, 1, 0])),
    "right_knee_pitch": JointInfo("right_knee_pitch", JointType.REVOLUTE, 1,
                                  limits=(0.0, 2.35), axis=np.array([0, 1, 0])),
    "right_ankle_pitch": JointInfo("right_ankle_pitch", JointType.REVOLUTE, 1,
                                   limits=(-0.87, 0.87), axis=np.array([0, 1, 0])),
    "right_ankle_roll": JointInfo("right_ankle_roll", JointType.REVOLUTE, 1,
                                  limits=(-0.52, 0.52), axis=np.array([1, 0, 0])),

    # Torso (2 DOF)
    "torso_pitch": JointInfo("torso_pitch", JointType.REVOLUTE, 1,
                             limits=(-0.26, 0.26), axis=np.array([0, 1, 0])),
    "torso_roll": JointInfo("torso_roll", JointType.REVOLUTE, 1,
                            limits=(-0.26, 0.26), axis=np.array([1, 0, 0])),

    # Left arm (3 DOF)
    "left_shoulder_pitch": JointInfo("left_shoulder_pitch", JointType.REVOLUTE, 1,
                                     limits=(-1.57, 1.57), axis=np.array([0, 1, 0])),
    "left_shoulder_roll": JointInfo("left_shoulder_roll", JointType.REVOLUTE, 1,
                                    limits=(-1.57, 1.57), axis=np.array([1, 0, 0])),
    "left_elbow_pitch": JointInfo("left_elbow_pitch", JointType.REVOLUTE, 1,
                                  limits=(-2.09, 2.09), axis=np.array([0, 1, 0])),

    # Right arm (3 DOF)
    "right_shoulder_pitch": JointInfo("right_shoulder_pitch", JointType.REVOLUTE, 1,
                                      limits=(-1.57, 1.57), axis=np.array([0, 1, 0])),
    "right_shoulder_roll": JointInfo("right_shoulder_roll", JointType.REVOLUTE, 1,
                                     limits=(-1.57, 1.57), axis=np.array([1, 0, 0])),
    "right_elbow_pitch": JointInfo("right_elbow_pitch", JointType.REVOLUTE, 1,
                                   limits=(-2.09, 2.09), axis=np.array([0, 1, 0])),

    # Head (2 DOF)
    "head_yaw": JointInfo("head_yaw", JointType.REVOLUTE, 1,
                          limits=(-1.57, 1.57), axis=np.array([0, 0, 1])),
    "head_pitch": JointInfo("head_pitch", JointType.REVOLUTE, 1,
                            limits=(-0.52, 0.52), axis=np.array([0, 1, 0])),
}


# ROBOT-SPECIFIC ASSUMPTION: Joint ordering for G1
# This defines the expected order of joints in the control vector
G1_JOINT_ORDER = [
    # Left leg
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch",
    "left_knee_pitch", "left_ankle_pitch", "left_ankle_roll",
    # Right leg
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch",
    "right_knee_pitch", "right_ankle_pitch", "right_ankle_roll",
    # Torso
    "torso_pitch", "torso_roll",
    # Left arm
    "left_shoulder_pitch", "left_shoulder_roll", "left_elbow_pitch",
    # Right arm
    "right_shoulder_pitch", "right_shoulder_roll", "right_elbow_pitch",
    # Head
    "head_yaw", "head_pitch",
]


# Human skeleton mappings for common motion capture formats
# ASSUMPTION: Human skeleton follows standard naming conventions
HUMAN_SKELETON_MAP = {
    # AMASS/SMPL skeleton mapping
    "amass": {
        # Root
        "root": "Pelvis",

        # Left leg
        "left_hip": "L_Hip",
        "left_knee": "L_Knee",
        "left_ankle": "L_Ankle",
        "left_foot": "L_Foot",

        # Right leg
        "right_hip": "R_Hip",
        "right_knee": "R_Knee",
        "right_ankle": "R_Ankle",
        "right_foot": "R_Foot",

        # Spine
        "spine1": "Spine1",
        "spine2": "Spine2",
        "spine3": "Spine3",
        "neck": "Neck",
        "head": "Head",

        # Left arm
        "left_collar": "L_Collar",
        "left_shoulder": "L_Shoulder",
        "left_elbow": "L_Elbow",
        "left_wrist": "L_Wrist",
        "left_hand": "L_Hand",

        # Right arm
        "right_collar": "R_Collar",
        "right_shoulder": "R_Shoulder",
        "right_elbow": "R_Elbow",
        "right_wrist": "R_Wrist",
        "right_hand": "R_Hand",
    },

    # CMU mocap skeleton mapping
    "cmu": {
        "root": "root",

        # Left leg
        "left_hip": "lhipjoint",
        "left_knee": "lknee",
        "left_ankle": "lankle",
        "left_foot": "lfoot",

        # Right leg
        "right_hip": "rhipjoint",
        "right_knee": "rknee",
        "right_ankle": "rankle",
        "right_foot": "rfoot",

        # Spine
        "spine1": "lowerback",
        "spine2": "upperback",
        "spine3": "thorax",
        "neck": "lowerneck",
        "head": "upperneck",

        # Left arm
        "left_collar": "lclavicle",
        "left_shoulder": "lhumerus",
        "left_elbow": "lradius",
        "left_wrist": "lwrist",
        "left_hand": "lhand",

        # Right arm
        "right_collar": "rclavicle",
        "right_shoulder": "rhumerus",
        "right_elbow": "rradius",
        "right_wrist": "rwrist",
        "right_hand": "rhand",
    }
}


# Joint mapping from human skeleton to G1 robot
# ROBOT-SPECIFIC ASSUMPTION: This mapping defines how human joint rotations
# are transferred to G1 joints. Some joints may be averaged, ignored, or
# computed from multiple human joints.
HUMAN_TO_G1_MAPPING = {
    # Left leg mapping
    "left_hip_yaw": {
        "source_joints": ["left_hip"],
        "component": "yaw",  # Extract yaw component from hip rotation
        "scale": 0.8,  # Scale factor to account for different ranges
        "offset": 0.0,
    },
    "left_hip_roll": {
        "source_joints": ["left_hip"],
        "component": "roll",
        "scale": 1.0,
        "offset": 0.0,
    },
    "left_hip_pitch": {
        "source_joints": ["left_hip"],
        "component": "pitch",
        "scale": 1.0,
        "offset": 0.0,
    },
    "left_knee_pitch": {
        "source_joints": ["left_knee"],
        "component": "pitch",
        "scale": 1.0,
        "offset": 0.0,
    },
    "left_ankle_pitch": {
        "source_joints": ["left_ankle"],
        "component": "pitch",
        "scale": 0.8,
        "offset": 0.0,
    },
    "left_ankle_roll": {
        "source_joints": ["left_ankle"],
        "component": "roll",
        "scale": 0.8,
        "offset": 0.0,
    },

    # Right leg mapping (mirror of left)
    "right_hip_yaw": {
        "source_joints": ["right_hip"],
        "component": "yaw",
        "scale": 0.8,
        "offset": 0.0,
    },
    "right_hip_roll": {
        "source_joints": ["right_hip"],
        "component": "roll",
        "scale": -1.0,  # Mirror for right side
        "offset": 0.0,
    },
    "right_hip_pitch": {
        "source_joints": ["right_hip"],
        "component": "pitch",
        "scale": 1.0,
        "offset": 0.0,
    },
    "right_knee_pitch": {
        "source_joints": ["right_knee"],
        "component": "pitch",
        "scale": 1.0,
        "offset": 0.0,
    },
    "right_ankle_pitch": {
        "source_joints": ["right_ankle"],
        "component": "pitch",
        "scale": 0.8,
        "offset": 0.0,
    },
    "right_ankle_roll": {
        "source_joints": ["right_ankle"],
        "component": "roll",
        "scale": -0.8,  # Mirror for right side
        "offset": 0.0,
    },

    # Torso mapping
    "torso_pitch": {
        "source_joints": ["spine2", "spine3"],
        "component": "pitch",
        "scale": 0.5,  # Average and scale down
        "offset": 0.0,
    },
    "torso_roll": {
        "source_joints": ["spine2", "spine3"],
        "component": "roll",
        "scale": 0.5,
        "offset": 0.0,
    },

    # Left arm mapping
    "left_shoulder_pitch": {
        "source_joints": ["left_shoulder"],
        "component": "pitch",
        "scale": 0.8,
        "offset": 0.0,
    },
    "left_shoulder_roll": {
        "source_joints": ["left_shoulder"],
        "component": "roll",
        "scale": 0.8,
        "offset": 0.0,
    },
    "left_elbow_pitch": {
        "source_joints": ["left_elbow"],
        "component": "pitch",
        "scale": 1.0,
        "offset": 0.0,
    },

    # Right arm mapping
    "right_shoulder_pitch": {
        "source_joints": ["right_shoulder"],
        "component": "pitch",
        "scale": 0.8,
        "offset": 0.0,
    },
    "right_shoulder_roll": {
        "source_joints": ["right_shoulder"],
        "component": "roll",
        "scale": -0.8,  # Mirror for right side
        "offset": 0.0,
    },
    "right_elbow_pitch": {
        "source_joints": ["right_elbow"],
        "component": "pitch",
        "scale": 1.0,
        "offset": 0.0,
    },

    # Head mapping
    "head_yaw": {
        "source_joints": ["head", "neck"],
        "component": "yaw",
        "scale": 0.6,
        "offset": 0.0,
    },
    "head_pitch": {
        "source_joints": ["head", "neck"],
        "component": "pitch",
        "scale": 0.6,
        "offset": 0.0,
    },
}


class SkeletonMapper:
    """
    Handles mapping between human skeleton and robot joint configurations.

    This class provides functionality to:
    - Map joint names between different skeleton formats
    - Extract robot joint angles from human joint rotations
    - Apply scaling and offset transformations
    - Enforce joint limits
    """

    def __init__(
        self,
        robot_joint_map: Dict[str, JointInfo] = None,
        human_skeleton_format: str = "amass",
        mapping_config: Dict[str, Any] = None
    ):
        """
        Initialize skeleton mapper.

        Args:
            robot_joint_map: Robot joint configuration
            human_skeleton_format: Format of human skeleton data
            mapping_config: Custom mapping configuration
        """
        self.robot_joint_map = robot_joint_map or G1_JOINT_MAP
        self.human_skeleton_format = human_skeleton_format
        self.human_joint_names = HUMAN_SKELETON_MAP.get(human_skeleton_format, {})
        self.mapping_config = mapping_config or HUMAN_TO_G1_MAPPING

        # Create robot joint order and limits
        self.robot_joint_order = G1_JOINT_ORDER
        self.robot_joint_limits = self._extract_joint_limits()

        # Validate mapping
        self._validate_mapping()

    def _extract_joint_limits(self) -> np.ndarray:
        """Extract joint limits for robot joints."""
        limits = []
        for joint_name in self.robot_joint_order:
            joint_info = self.robot_joint_map[joint_name]
            if joint_info.limits:
                limits.append(joint_info.limits)
            else:
                limits.append((-np.pi, np.pi))  # Default limits
        return np.array(limits)

    def _validate_mapping(self):
        """Validate the joint mapping configuration."""
        # Check that all robot joints have mapping
        for joint_name in self.robot_joint_order:
            if joint_name not in self.mapping_config:
                print(f"Warning: No mapping found for robot joint '{joint_name}'")

    def map_joint_names(self, human_joint_names: List[str]) -> Dict[str, str]:
        """
        Map human joint names to standardized names.

        Args:
            human_joint_names: List of joint names from human skeleton

        Returns:
            Dictionary mapping human joint names to standard names
        """
        name_mapping = {}

        for human_name in human_joint_names:
            # Find matching standard name
            for standard_name, mapped_name in self.human_joint_names.items():
                if mapped_name.lower() == human_name.lower():
                    name_mapping[human_name] = standard_name
                    break

        return name_mapping

    def extract_euler_angles(self, rotation_matrix: np.ndarray) -> Tuple[float, float, float]:
        """
        Extract Euler angles (roll, pitch, yaw) from rotation matrix.

        Args:
            rotation_matrix: 3x3 rotation matrix

        Returns:
            Tuple of (roll, pitch, yaw) in radians
        """
        # Extract Euler angles using ZYX convention
        sy = np.sqrt(rotation_matrix[0, 0] * rotation_matrix[0, 0] +
                     rotation_matrix[1, 0] * rotation_matrix[1, 0])

        singular = sy < 1e-6

        if not singular:
            x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])  # roll
            y = np.arctan2(-rotation_matrix[2, 0], sy)                    # pitch
            z = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])  # yaw
        else:
            x = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
            y = np.arctan2(-rotation_matrix[2, 0], sy)
            z = 0

        return x, y, z  # roll, pitch, yaw

    def quaternion_to_euler(self, quaternion: np.ndarray) -> Tuple[float, float, float]:
        """
        Convert quaternion to Euler angles.

        Args:
            quaternion: Quaternion as [w, x, y, z]

        Returns:
            Tuple of (roll, pitch, yaw) in radians
        """
        w, x, y, z = quaternion

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        pitch = np.arcsin(np.clip(sinp, -1, 1))

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    def map_human_to_robot_joints(
        self,
        human_joint_rotations: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Map human joint rotations to robot joint angles.

        Args:
            human_joint_rotations: Dictionary of human joint rotations
                                 (can be rotation matrices, quaternions, or euler angles)

        Returns:
            Robot joint angles as numpy array in robot joint order
        """
        robot_joint_angles = np.zeros(len(self.robot_joint_order))

        for i, robot_joint in enumerate(self.robot_joint_order):
            if robot_joint not in self.mapping_config:
                continue

            mapping = self.mapping_config[robot_joint]
            source_joints = mapping["source_joints"]
            component = mapping["component"]
            scale = mapping.get("scale", 1.0)
            offset = mapping.get("offset", 0.0)

            # Extract angles from source joints
            joint_values = []
            for source_joint in source_joints:
                if source_joint not in human_joint_rotations:
                    continue

                rotation_data = human_joint_rotations[source_joint]

                # Handle different rotation representations
                if rotation_data.shape == (3, 3):
                    # Rotation matrix
                    roll, pitch, yaw = self.extract_euler_angles(rotation_data)
                elif rotation_data.shape == (4,):
                    # Quaternion
                    roll, pitch, yaw = self.quaternion_to_euler(rotation_data)
                elif rotation_data.shape == (3,):
                    # Euler angles
                    roll, pitch, yaw = rotation_data
                else:
                    print(f"Warning: Unknown rotation format for joint {source_joint}")
                    continue

                # Extract requested component
                if component == "roll":
                    joint_values.append(roll)
                elif component == "pitch":
                    joint_values.append(pitch)
                elif component == "yaw":
                    joint_values.append(yaw)

            # Combine values (average if multiple sources)
            if joint_values:
                combined_value = np.mean(joint_values)
                robot_joint_angles[i] = scale * combined_value + offset

        # Enforce joint limits
        robot_joint_angles = self.enforce_joint_limits(robot_joint_angles)

        return robot_joint_angles

    def enforce_joint_limits(self, joint_angles: np.ndarray) -> np.ndarray:
        """
        Enforce joint limits on robot joint angles.

        Args:
            joint_angles: Robot joint angles

        Returns:
            Joint angles clipped to limits
        """
        limited_angles = joint_angles.copy()

        for i, (min_limit, max_limit) in enumerate(self.robot_joint_limits):
            limited_angles[i] = np.clip(limited_angles[i], min_limit, max_limit)

        return limited_angles

    def get_joint_info(self) -> Dict[str, Any]:
        """Get information about the joint mapping."""
        return {
            "robot_joint_count": len(self.robot_joint_order),
            "robot_joint_names": self.robot_joint_order,
            "robot_joint_limits": self.robot_joint_limits.tolist(),
            "human_skeleton_format": self.human_skeleton_format,
            "mapping_config": self.mapping_config,
        }