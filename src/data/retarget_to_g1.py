"""
Motion retargeting module for converting human motion clips to Unitree G1 trajectories.

This module handles the core retargeting logic, transforming human mocap data
into robot-compatible joint trajectories with proper scaling, filtering,
and constraint handling.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import logging

from .skeleton_map import SkeletonMapper, G1_JOINT_MAP, HUMAN_TO_G1_MAPPING

logger = logging.getLogger(__name__)


@dataclass
class MotionClipData:
    """Container for input human motion data."""
    timestamps: np.ndarray  # (T,)
    joint_positions: Dict[str, np.ndarray]  # joint_name -> (T, 3) positions
    joint_rotations: Dict[str, np.ndarray]  # joint_name -> (T, 4) quaternions
    root_positions: np.ndarray  # (T, 3)
    root_orientations: np.ndarray  # (T, 4) quaternions
    skeleton_type: str = "amass"  # "amass", "cmu", etc.


@dataclass
class G1TrajectoryData:
    """Container for output G1 trajectory data."""
    timestamps: np.ndarray  # (T,) at fixed framerate
    joint_positions: np.ndarray  # (T, 22) G1 joint angles
    joint_velocities: np.ndarray  # (T, 22) joint velocities
    joint_accelerations: np.ndarray  # (T, 22) joint accelerations
    root_positions: np.ndarray  # (T, 3) root position
    root_orientations: np.ndarray  # (T, 4) root orientation (quaternion)
    root_linear_velocities: np.ndarray  # (T, 3) root linear velocity
    root_angular_velocities: np.ndarray  # (T, 3) root angular velocity
    foot_contacts: np.ndarray  # (T, 2) left/right foot contact states
    metadata: Dict  # Additional info about the trajectory


class MotionRetargeter:
    """Main class for retargeting human motion to G1 robot."""

    def __init__(self,
                 target_fps: float = 50.0,
                 smoothing_window: int = 5,
                 position_scale: float = 1.0,
                 orientation_scale: float = 1.0,
                 apply_joint_limits: bool = True,
                 filter_velocities: bool = True):
        """
        Initialize motion retargeter.

        Args:
            target_fps: Target framerate for output trajectories
            smoothing_window: Window size for Savitzky-Golay filtering
            position_scale: Scale factor for root positions
            orientation_scale: Scale factor for joint rotations
            apply_joint_limits: Whether to enforce G1 joint limits
            filter_velocities: Whether to smooth velocity profiles
        """
        self.target_fps = target_fps
        self.target_dt = 1.0 / target_fps
        self.smoothing_window = max(3, smoothing_window | 1)  # Ensure odd
        self.position_scale = position_scale
        self.orientation_scale = orientation_scale
        self.apply_joint_limits = apply_joint_limits
        self.filter_velocities = filter_velocities

        self.skeleton_mapper = SkeletonMapper()
        self.joint_names = list(G1_JOINT_MAP.keys())
        self.n_joints = len(self.joint_names)

        # Joint limit caching
        self._joint_limits = self._build_joint_limits()

        logger.info(f"Initialized MotionRetargeter (fps={target_fps}, joints={self.n_joints})")

    def _build_joint_limits(self) -> Dict[str, Tuple[float, float]]:
        """Build joint limits dictionary for efficient access."""
        limits = {}
        for joint_name in self.joint_names:
            joint_info = G1_JOINT_MAP[joint_name]
            limits[joint_name] = joint_info.limits
        return limits

    def retarget_clip(self, motion_clip: MotionClipData) -> G1TrajectoryData:
        """
        Retarget a single motion clip to G1 format.

        Args:
            motion_clip: Input human motion data

        Returns:
            G1TrajectoryData: Retargeted trajectory
        """
        logger.info(f"Retargeting motion clip: {len(motion_clip.timestamps)} frames, "
                   f"duration={motion_clip.timestamps[-1]:.2f}s")

        # Step 1: Temporal resampling to fixed framerate
        target_timestamps = self._resample_timestamps(motion_clip.timestamps)

        # Step 2: Resample all motion data to target timestamps
        resampled_data = self._resample_motion_data(motion_clip, target_timestamps)

        # Step 3: Map human skeleton to G1 joints
        g1_joint_angles = self._map_to_g1_joints(resampled_data)

        # Step 4: Apply joint limits and smoothing
        if self.apply_joint_limits:
            g1_joint_angles = self._enforce_joint_limits(g1_joint_angles)

        g1_joint_angles = self._smooth_trajectories(g1_joint_angles, target_timestamps)

        # Step 5: Compute derivatives
        joint_velocities = self._compute_velocities(g1_joint_angles, target_timestamps)
        joint_accelerations = self._compute_accelerations(joint_velocities, target_timestamps)

        # Step 6: Process root motion
        root_positions, root_orientations = self._process_root_motion(resampled_data)
        root_lin_vel = self._compute_velocities(root_positions, target_timestamps)
        root_ang_vel = self._compute_angular_velocities(root_orientations, target_timestamps)

        # Step 7: Estimate foot contacts (placeholder - will be implemented in contact_estimation.py)
        foot_contacts = np.zeros((len(target_timestamps), 2))

        # Step 8: Build metadata
        metadata = {
            "source_skeleton": motion_clip.skeleton_type,
            "original_fps": 1.0 / np.mean(np.diff(motion_clip.timestamps)),
            "target_fps": self.target_fps,
            "duration": target_timestamps[-1],
            "n_frames": len(target_timestamps),
            "joint_mapping": HUMAN_TO_G1_MAPPING.get(motion_clip.skeleton_type, {}),
            "processing": {
                "smoothing_window": self.smoothing_window,
                "position_scale": self.position_scale,
                "orientation_scale": self.orientation_scale,
                "joint_limits_applied": self.apply_joint_limits
            }
        }

        return G1TrajectoryData(
            timestamps=target_timestamps,
            joint_positions=g1_joint_angles,
            joint_velocities=joint_velocities,
            joint_accelerations=joint_accelerations,
            root_positions=root_positions,
            root_orientations=root_orientations,
            root_linear_velocities=root_lin_vel,
            root_angular_velocities=root_ang_vel,
            foot_contacts=foot_contacts,
            metadata=metadata
        )

    def _resample_timestamps(self, original_timestamps: np.ndarray) -> np.ndarray:
        """Create fixed-rate timestamp array."""
        start_time = original_timestamps[0]
        end_time = original_timestamps[-1]
        duration = end_time - start_time
        n_samples = int(duration * self.target_fps) + 1
        return np.linspace(start_time, end_time, n_samples)

    def _resample_motion_data(self, motion_clip: MotionClipData,
                            target_timestamps: np.ndarray) -> MotionClipData:
        """Resample all motion data to target timestamps."""
        original_times = motion_clip.timestamps

        # Resample joint rotations
        resampled_joint_rotations = {}
        for joint_name, rotations in motion_clip.joint_rotations.items():
            if len(rotations.shape) == 2 and rotations.shape[1] == 4:
                # Quaternion SLERP interpolation
                resampled_joint_rotations[joint_name] = self._slerp_quaternions(
                    original_times, rotations, target_timestamps
                )
            else:
                logger.warning(f"Unexpected rotation shape for {joint_name}: {rotations.shape}")
                continue

        # Resample joint positions (linear interpolation)
        resampled_joint_positions = {}
        for joint_name, positions in motion_clip.joint_positions.items():
            if len(positions.shape) == 2 and positions.shape[1] == 3:
                interp_func = interp1d(original_times, positions, axis=0,
                                     kind='linear', bounds_error=False,
                                     fill_value='extrapolate')
                resampled_joint_positions[joint_name] = interp_func(target_timestamps)
            else:
                logger.warning(f"Unexpected position shape for {joint_name}: {positions.shape}")
                continue

        # Resample root motion
        root_pos_interp = interp1d(original_times, motion_clip.root_positions,
                                  axis=0, kind='linear', bounds_error=False,
                                  fill_value='extrapolate')
        resampled_root_positions = root_pos_interp(target_timestamps) * self.position_scale

        resampled_root_orientations = self._slerp_quaternions(
            original_times, motion_clip.root_orientations, target_timestamps
        )

        return MotionClipData(
            timestamps=target_timestamps,
            joint_positions=resampled_joint_positions,
            joint_rotations=resampled_joint_rotations,
            root_positions=resampled_root_positions,
            root_orientations=resampled_root_orientations,
            skeleton_type=motion_clip.skeleton_type
        )

    def _slerp_quaternions(self, original_times: np.ndarray,
                          quaternions: np.ndarray,
                          target_times: np.ndarray) -> np.ndarray:
        """Spherical linear interpolation for quaternions."""
        if len(quaternions) == 0:
            return np.zeros((len(target_times), 4))

        # Normalize quaternions
        quaternions = quaternions / np.linalg.norm(quaternions, axis=1, keepdims=True)

        # Use scipy's Rotation for SLERP
        rotations = R.from_quat(quaternions)

        # Handle single frame case
        if len(rotations) == 1:
            return np.tile(quaternions[0], (len(target_times), 1))

        # Create interpolator
        try:
            slerp = R.from_quat(quaternions).as_rotvec()
            interp_func = interp1d(original_times, slerp, axis=0, kind='linear',
                                 bounds_error=False, fill_value='extrapolate')
            interp_rotvec = interp_func(target_times)
            return R.from_rotvec(interp_rotvec).as_quat()
        except Exception as e:
            logger.warning(f"SLERP failed, using linear interpolation: {e}")
            interp_func = interp1d(original_times, quaternions, axis=0, kind='linear',
                                 bounds_error=False, fill_value='extrapolate')
            result = interp_func(target_times)
            return result / np.linalg.norm(result, axis=1, keepdims=True)

    def _map_to_g1_joints(self, motion_data: MotionClipData) -> np.ndarray:
        """Map human joint rotations to G1 joint angles."""
        n_frames = len(motion_data.timestamps)
        g1_joint_angles = np.zeros((n_frames, self.n_joints))

        # Get mapping for this skeleton type
        mapping = HUMAN_TO_G1_MAPPING.get(motion_data.skeleton_type, {})

        for g1_idx, g1_joint_name in enumerate(self.joint_names):
            if g1_joint_name in mapping:
                human_joint_name, transform_info = mapping[g1_joint_name]

                if human_joint_name in motion_data.joint_rotations:
                    human_rotations = motion_data.joint_rotations[human_joint_name]

                    # Apply transformation based on transform_info
                    g1_angles = self._transform_joint_rotation(
                        human_rotations, transform_info
                    )

                    g1_joint_angles[:, g1_idx] = g1_angles * self.orientation_scale
                else:
                    logger.warning(f"Human joint '{human_joint_name}' not found for G1 joint '{g1_joint_name}'")
            else:
                # Use default pose for unmapped joints
                joint_info = G1_JOINT_MAP[g1_joint_name]
                default_angle = (joint_info.limits[0] + joint_info.limits[1]) / 2
                g1_joint_angles[:, g1_idx] = default_angle

        return g1_joint_angles

    def _transform_joint_rotation(self, quaternions: np.ndarray,
                                transform_info: Dict) -> np.ndarray:
        """Transform human joint rotation to G1 joint angle."""
        # Convert quaternions to rotation matrices
        rotations = R.from_quat(quaternions)

        # Extract specified axis rotation
        axis = transform_info.get("axis", "y")
        sign = transform_info.get("sign", 1)
        offset = transform_info.get("offset", 0.0)

        # Convert to Euler angles and extract specified axis
        if axis == "x":
            angles = rotations.as_euler('xyz')[:, 0]
        elif axis == "y":
            angles = rotations.as_euler('xyz')[:, 1]
        elif axis == "z":
            angles = rotations.as_euler('xyz')[:, 2]
        else:
            logger.warning(f"Unknown axis '{axis}', using y")
            angles = rotations.as_euler('xyz')[:, 1]

        return sign * angles + offset

    def _enforce_joint_limits(self, joint_angles: np.ndarray) -> np.ndarray:
        """Enforce G1 joint limits by clamping."""
        clamped_angles = joint_angles.copy()

        for i, joint_name in enumerate(self.joint_names):
            limits = self._joint_limits[joint_name]
            clamped_angles[:, i] = np.clip(joint_angles[:, i], limits[0], limits[1])

            # Log violations
            violations = np.sum((joint_angles[:, i] < limits[0]) |
                              (joint_angles[:, i] > limits[1]))
            if violations > 0:
                logger.debug(f"Joint '{joint_name}': {violations} limit violations corrected")

        return clamped_angles

    def _smooth_trajectories(self, joint_angles: np.ndarray,
                           timestamps: np.ndarray) -> np.ndarray:
        """Apply Savitzky-Golay smoothing to joint trajectories."""
        if len(timestamps) < self.smoothing_window:
            logger.warning("Trajectory too short for smoothing")
            return joint_angles

        smoothed_angles = np.zeros_like(joint_angles)

        for i in range(self.n_joints):
            smoothed_angles[:, i] = savgol_filter(
                joint_angles[:, i],
                window_length=self.smoothing_window,
                polyorder=min(3, self.smoothing_window - 1)
            )

        return smoothed_angles

    def _compute_velocities(self, positions: np.ndarray,
                          timestamps: np.ndarray) -> np.ndarray:
        """Compute velocities using finite differences."""
        velocities = np.zeros_like(positions)
        dt = np.diff(timestamps)

        # Forward difference for first frame
        if len(positions) > 1:
            velocities[0] = (positions[1] - positions[0]) / dt[0]

        # Central difference for middle frames
        for i in range(1, len(positions) - 1):
            velocities[i] = (positions[i + 1] - positions[i - 1]) / (dt[i-1] + dt[i])

        # Backward difference for last frame
        if len(positions) > 1:
            velocities[-1] = (positions[-1] - positions[-2]) / dt[-1]

        # Optional smoothing
        if self.filter_velocities and len(positions) >= self.smoothing_window:
            for i in range(velocities.shape[1]):
                velocities[:, i] = savgol_filter(
                    velocities[:, i],
                    window_length=self.smoothing_window,
                    polyorder=min(2, self.smoothing_window - 1)
                )

        return velocities

    def _compute_accelerations(self, velocities: np.ndarray,
                             timestamps: np.ndarray) -> np.ndarray:
        """Compute accelerations from velocities."""
        return self._compute_velocities(velocities, timestamps)

    def _compute_angular_velocities(self, quaternions: np.ndarray,
                                  timestamps: np.ndarray) -> np.ndarray:
        """Compute angular velocities from quaternions."""
        if len(quaternions) <= 1:
            return np.zeros((len(quaternions), 3))

        angular_velocities = np.zeros((len(quaternions), 3))

        for i in range(len(quaternions) - 1):
            dt = timestamps[i + 1] - timestamps[i]

            # Compute relative rotation
            q1 = R.from_quat(quaternions[i])
            q2 = R.from_quat(quaternions[i + 1])
            rel_rot = q1.inv() * q2

            # Convert to axis-angle and scale by time
            axis_angle = rel_rot.as_rotvec()
            angular_velocities[i] = axis_angle / dt

        # Copy last velocity
        angular_velocities[-1] = angular_velocities[-2] if len(angular_velocities) > 1 else 0

        return angular_velocities

    def _process_root_motion(self, motion_data: MotionClipData) -> Tuple[np.ndarray, np.ndarray]:
        """Process root position and orientation for G1."""
        # Root positions are already scaled during resampling
        root_positions = motion_data.root_positions.copy()

        # Ensure root orientations are normalized
        root_orientations = motion_data.root_orientations.copy()
        root_orientations = root_orientations / np.linalg.norm(root_orientations, axis=1, keepdims=True)

        return root_positions, root_orientations


def retarget_motion_clip(motion_clip: MotionClipData,
                        target_fps: float = 50.0,
                        **kwargs) -> G1TrajectoryData:
    """
    Convenience function for retargeting a single motion clip.

    Args:
        motion_clip: Input human motion data
        target_fps: Target output framerate
        **kwargs: Additional arguments passed to MotionRetargeter

    Returns:
        G1TrajectoryData: Retargeted trajectory
    """
    retargeter = MotionRetargeter(target_fps=target_fps, **kwargs)
    return retargeter.retarget_clip(motion_clip)


def load_motion_clip_from_npz(file_path: str) -> MotionClipData:
    """
    Load motion clip from NPZ file.

    Expected format:
    - timestamps: (T,) array
    - joint_positions: dict of joint_name -> (T, 3) arrays
    - joint_rotations: dict of joint_name -> (T, 4) quaternion arrays
    - root_positions: (T, 3) array
    - root_orientations: (T, 4) quaternion array
    - skeleton_type: string

    Args:
        file_path: Path to NPZ file

    Returns:
        MotionClipData: Loaded motion data
    """
    data = np.load(file_path, allow_pickle=True)

    # Extract basic arrays
    timestamps = data['timestamps']
    root_positions = data['root_positions']
    root_orientations = data['root_orientations']
    skeleton_type = str(data.get('skeleton_type', 'unknown'))

    # Extract joint data dictionaries
    joint_positions = data['joint_positions'].item()
    joint_rotations = data['joint_rotations'].item()

    return MotionClipData(
        timestamps=timestamps,
        joint_positions=joint_positions,
        joint_rotations=joint_rotations,
        root_positions=root_positions,
        root_orientations=root_orientations,
        skeleton_type=skeleton_type
    )


def save_g1_trajectory_to_npz(trajectory: G1TrajectoryData,
                             file_path: str) -> None:
    """
    Save G1 trajectory to NPZ file.

    Args:
        trajectory: G1 trajectory data
        file_path: Output NPZ file path
    """
    np.savez_compressed(
        file_path,
        timestamps=trajectory.timestamps,
        joint_positions=trajectory.joint_positions,
        joint_velocities=trajectory.joint_velocities,
        joint_accelerations=trajectory.joint_accelerations,
        root_positions=trajectory.root_positions,
        root_orientations=trajectory.root_orientations,
        root_linear_velocities=trajectory.root_linear_velocities,
        root_angular_velocities=trajectory.root_angular_velocities,
        foot_contacts=trajectory.foot_contacts,
        metadata=trajectory.metadata
    )

    logger.info(f"Saved G1 trajectory to {file_path}")