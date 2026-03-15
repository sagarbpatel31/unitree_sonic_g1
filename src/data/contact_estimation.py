"""
Contact estimation module for detecting foot-ground contacts in human motion.

This module provides algorithms for estimating when feet are in contact with
the ground based on motion data, which is crucial for locomotion analysis
and robot control.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from scipy.signal import find_peaks, savgol_filter
import logging

logger = logging.getLogger(__name__)


@dataclass
class ContactInfo:
    """Information about a contact event."""
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    duration: float
    foot: str  # "left" or "right"
    confidence: float  # 0.0 to 1.0


@dataclass
class ContactEstimationConfig:
    """Configuration for contact estimation algorithms."""
    velocity_threshold: float = 0.02  # m/s
    height_threshold: float = 0.05    # m
    acceleration_threshold: float = 2.0  # m/s²
    min_contact_duration: float = 0.1   # seconds
    smoothing_window: int = 5
    use_velocity: bool = True
    use_height: bool = True
    use_acceleration: bool = True
    combine_method: str = "voting"  # "voting", "weighted", "all"


class ContactEstimator:
    """Foot contact estimation for human motion data."""

    def __init__(self, config: Optional[ContactEstimationConfig] = None):
        """
        Initialize contact estimator.

        Args:
            config: Configuration parameters for contact estimation
        """
        self.config = config or ContactEstimationConfig()
        self.foot_joint_names = {
            "left": ["left_ankle", "left_foot", "left_toe"],
            "right": ["right_ankle", "right_foot", "right_toe"]
        }
        logger.info("Initialized ContactEstimator")

    def estimate_contacts(self,
                         timestamps: np.ndarray,
                         joint_positions: Dict[str, np.ndarray],
                         joint_velocities: Optional[Dict[str, np.ndarray]] = None,
                         root_height: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Estimate foot contacts for a motion sequence.

        Args:
            timestamps: Time array (T,)
            joint_positions: Joint positions dict {joint_name: (T, 3)}
            joint_velocities: Joint velocities dict {joint_name: (T, 3)}
            root_height: Root height array (T,)

        Returns:
            Dict with "left" and "right" contact arrays (T,) with values 0.0-1.0
        """
        contacts = {}

        for foot_name in ["left", "right"]:
            foot_contact = self._estimate_foot_contact(
                foot_name, timestamps, joint_positions, joint_velocities, root_height
            )
            contacts[foot_name] = foot_contact

        # Post-process contacts
        contacts = self._post_process_contacts(contacts, timestamps)

        logger.info(f"Estimated contacts: left={np.sum(contacts['left'] > 0.5)} frames, "
                   f"right={np.sum(contacts['right'] > 0.5)} frames")

        return contacts

    def _estimate_foot_contact(self,
                              foot_name: str,
                              timestamps: np.ndarray,
                              joint_positions: Dict[str, np.ndarray],
                              joint_velocities: Optional[Dict[str, np.ndarray]],
                              root_height: Optional[np.ndarray]) -> np.ndarray:
        """Estimate contact for a single foot."""
        T = len(timestamps)
        contact_scores = []

        # Method 1: Velocity-based detection
        if self.config.use_velocity:
            velocity_score = self._velocity_based_contact(
                foot_name, timestamps, joint_positions, joint_velocities
            )
            contact_scores.append(velocity_score)

        # Method 2: Height-based detection
        if self.config.use_height:
            height_score = self._height_based_contact(
                foot_name, timestamps, joint_positions, root_height
            )
            contact_scores.append(height_score)

        # Method 3: Acceleration-based detection
        if self.config.use_acceleration:
            accel_score = self._acceleration_based_contact(
                foot_name, timestamps, joint_positions, joint_velocities
            )
            contact_scores.append(accel_score)

        # Combine methods
        if len(contact_scores) == 0:
            logger.warning("No contact detection methods enabled")
            return np.zeros(T)

        combined_score = self._combine_contact_scores(contact_scores)
        return combined_score

    def _velocity_based_contact(self,
                               foot_name: str,
                               timestamps: np.ndarray,
                               joint_positions: Dict[str, np.ndarray],
                               joint_velocities: Optional[Dict[str, np.ndarray]]) -> np.ndarray:
        """Detect contacts based on foot velocity."""
        T = len(timestamps)

        # Find relevant foot joint
        foot_joint = self._get_foot_joint(foot_name, joint_positions)
        if foot_joint is None:
            logger.warning(f"No foot joint found for {foot_name}")
            return np.zeros(T)

        # Compute velocities if not provided
        if joint_velocities is None or foot_joint not in joint_velocities:
            foot_positions = joint_positions[foot_joint]
            foot_velocities = self._compute_velocities(foot_positions, timestamps)
        else:
            foot_velocities = joint_velocities[foot_joint]

        # Compute speed (magnitude of velocity)
        foot_speeds = np.linalg.norm(foot_velocities, axis=1)

        # Smooth speeds
        if len(foot_speeds) >= self.config.smoothing_window:
            foot_speeds = savgol_filter(
                foot_speeds,
                window_length=self.config.smoothing_window,
                polyorder=min(2, self.config.smoothing_window - 1)
            )

        # Contact when speed is below threshold
        contact_score = 1.0 - np.clip(foot_speeds / self.config.velocity_threshold, 0, 1)

        return contact_score

    def _height_based_contact(self,
                             foot_name: str,
                             timestamps: np.ndarray,
                             joint_positions: Dict[str, np.ndarray],
                             root_height: Optional[np.ndarray]) -> np.ndarray:
        """Detect contacts based on foot height."""
        T = len(timestamps)

        # Find relevant foot joint
        foot_joint = self._get_foot_joint(foot_name, joint_positions)
        if foot_joint is None:
            logger.warning(f"No foot joint found for {foot_name}")
            return np.zeros(T)

        foot_positions = joint_positions[foot_joint]
        foot_heights = foot_positions[:, 2]  # Assume Z is up

        # Relative to minimum height if no root height provided
        if root_height is None:
            min_height = np.min(foot_heights)
            relative_heights = foot_heights - min_height
        else:
            # Use ground-relative height
            relative_heights = foot_heights - (root_height - 1.0)  # Assume 1m standing height

        # Smooth heights
        if len(relative_heights) >= self.config.smoothing_window:
            relative_heights = savgol_filter(
                relative_heights,
                window_length=self.config.smoothing_window,
                polyorder=min(2, self.config.smoothing_window - 1)
            )

        # Contact when height is below threshold
        contact_score = 1.0 - np.clip(relative_heights / self.config.height_threshold, 0, 1)

        return contact_score

    def _acceleration_based_contact(self,
                                   foot_name: str,
                                   timestamps: np.ndarray,
                                   joint_positions: Dict[str, np.ndarray],
                                   joint_velocities: Optional[Dict[str, np.ndarray]]) -> np.ndarray:
        """Detect contacts based on foot acceleration."""
        T = len(timestamps)

        # Find relevant foot joint
        foot_joint = self._get_foot_joint(foot_name, joint_positions)
        if foot_joint is None:
            logger.warning(f"No foot joint found for {foot_name}")
            return np.zeros(T)

        # Compute velocities if not provided
        if joint_velocities is None or foot_joint not in joint_velocities:
            foot_positions = joint_positions[foot_joint]
            foot_velocities = self._compute_velocities(foot_positions, timestamps)
        else:
            foot_velocities = joint_velocities[foot_joint]

        # Compute accelerations
        foot_accelerations = self._compute_velocities(foot_velocities, timestamps)
        acceleration_magnitudes = np.linalg.norm(foot_accelerations, axis=1)

        # Smooth accelerations
        if len(acceleration_magnitudes) >= self.config.smoothing_window:
            acceleration_magnitudes = savgol_filter(
                acceleration_magnitudes,
                window_length=self.config.smoothing_window,
                polyorder=min(2, self.config.smoothing_window - 1)
            )

        # Contact when acceleration is low (stable contact)
        contact_score = 1.0 - np.clip(acceleration_magnitudes / self.config.acceleration_threshold, 0, 1)

        return contact_score

    def _get_foot_joint(self, foot_name: str,
                       joint_positions: Dict[str, np.ndarray]) -> Optional[str]:
        """Find the best available foot joint for the given foot."""
        possible_joints = self.foot_joint_names[foot_name]

        for joint_name in possible_joints:
            if joint_name in joint_positions:
                return joint_name

        # Try alternative naming conventions
        if foot_name == "left":
            alternatives = ["LeftAnkle", "LeftFoot", "LeftToe", "l_ankle", "l_foot"]
        else:
            alternatives = ["RightAnkle", "RightFoot", "RightToe", "r_ankle", "r_foot"]

        for alt_name in alternatives:
            if alt_name in joint_positions:
                return alt_name

        return None

    def _compute_velocities(self, positions: np.ndarray,
                          timestamps: np.ndarray) -> np.ndarray:
        """Compute velocities using finite differences."""
        velocities = np.zeros_like(positions)

        if len(positions) <= 1:
            return velocities

        dt = np.diff(timestamps)

        # Forward difference for first frame
        velocities[0] = (positions[1] - positions[0]) / dt[0]

        # Central difference for middle frames
        for i in range(1, len(positions) - 1):
            velocities[i] = (positions[i + 1] - positions[i - 1]) / (dt[i-1] + dt[i])

        # Backward difference for last frame
        velocities[-1] = (positions[-1] - positions[-2]) / dt[-1]

        return velocities

    def _combine_contact_scores(self, contact_scores: List[np.ndarray]) -> np.ndarray:
        """Combine multiple contact scores into a single score."""
        if len(contact_scores) == 1:
            return contact_scores[0]

        contact_scores = np.array(contact_scores)

        if self.config.combine_method == "voting":
            # Simple majority voting (threshold at 0.5)
            votes = (contact_scores > 0.5).astype(float)
            combined = np.mean(votes, axis=0)

        elif self.config.combine_method == "weighted":
            # Weighted average (velocity gets highest weight)
            weights = np.array([0.5, 0.3, 0.2])[:len(contact_scores)]
            weights = weights / np.sum(weights)
            combined = np.average(contact_scores, axis=0, weights=weights)

        elif self.config.combine_method == "all":
            # All methods must agree (conservative)
            combined = np.min(contact_scores, axis=0)

        else:
            # Default: simple average
            combined = np.mean(contact_scores, axis=0)

        return combined

    def _post_process_contacts(self, contacts: Dict[str, np.ndarray],
                              timestamps: np.ndarray) -> Dict[str, np.ndarray]:
        """Post-process contact signals to remove noise and enforce constraints."""
        processed_contacts = {}

        for foot_name, contact_signal in contacts.items():
            # Apply minimum duration constraint
            processed_signal = self._enforce_min_duration(
                contact_signal, timestamps, self.config.min_contact_duration
            )

            # Apply final smoothing
            if len(processed_signal) >= self.config.smoothing_window:
                processed_signal = savgol_filter(
                    processed_signal,
                    window_length=self.config.smoothing_window,
                    polyorder=1
                )

            # Ensure values are in [0, 1]
            processed_signal = np.clip(processed_signal, 0, 1)

            processed_contacts[foot_name] = processed_signal

        return processed_contacts

    def _enforce_min_duration(self, contact_signal: np.ndarray,
                             timestamps: np.ndarray,
                             min_duration: float) -> np.ndarray:
        """Remove contact events shorter than minimum duration."""
        if min_duration <= 0:
            return contact_signal

        # Threshold to binary
        binary_contacts = (contact_signal > 0.5).astype(int)

        # Find contact segments
        diff = np.diff(np.concatenate(([0], binary_contacts, [0])))
        contact_starts = np.where(diff == 1)[0]
        contact_ends = np.where(diff == -1)[0]

        # Filter short contacts
        for start, end in zip(contact_starts, contact_ends):
            duration = timestamps[end-1] - timestamps[start]
            if duration < min_duration:
                binary_contacts[start:end] = 0

        # Apply back to original signal
        processed_signal = contact_signal * binary_contacts

        return processed_signal

    def extract_contact_events(self, contacts: Dict[str, np.ndarray],
                              timestamps: np.ndarray,
                              threshold: float = 0.5) -> List[ContactInfo]:
        """
        Extract discrete contact events from contact signals.

        Args:
            contacts: Contact signals dict
            timestamps: Time array
            threshold: Contact threshold (0.0-1.0)

        Returns:
            List of ContactInfo objects
        """
        events = []

        for foot_name, contact_signal in contacts.items():
            # Threshold to binary
            binary_contacts = (contact_signal > threshold).astype(int)

            # Find contact segments
            diff = np.diff(np.concatenate(([0], binary_contacts, [0])))
            contact_starts = np.where(diff == 1)[0]
            contact_ends = np.where(diff == -1)[0]

            # Create contact events
            for start, end in zip(contact_starts, contact_ends):
                if end > start:  # Valid contact
                    start_time = timestamps[start]
                    end_time = timestamps[end-1] if end-1 < len(timestamps) else timestamps[-1]
                    duration = end_time - start_time

                    # Compute confidence as average contact score
                    confidence = np.mean(contact_signal[start:end])

                    event = ContactInfo(
                        start_frame=start,
                        end_frame=end-1,
                        start_time=start_time,
                        end_time=end_time,
                        duration=duration,
                        foot=foot_name,
                        confidence=confidence
                    )
                    events.append(event)

        # Sort by start time
        events.sort(key=lambda x: x.start_time)

        logger.info(f"Extracted {len(events)} contact events")
        return events


def estimate_foot_contacts(timestamps: np.ndarray,
                          joint_positions: Dict[str, np.ndarray],
                          joint_velocities: Optional[Dict[str, np.ndarray]] = None,
                          config: Optional[ContactEstimationConfig] = None) -> Dict[str, np.ndarray]:
    """
    Convenience function for estimating foot contacts.

    Args:
        timestamps: Time array (T,)
        joint_positions: Joint positions dict {joint_name: (T, 3)}
        joint_velocities: Optional joint velocities dict
        config: Optional configuration

    Returns:
        Dict with "left" and "right" contact arrays (T,)
    """
    estimator = ContactEstimator(config)
    return estimator.estimate_contacts(timestamps, joint_positions, joint_velocities)


def validate_contact_estimation(contacts: Dict[str, np.ndarray],
                               timestamps: np.ndarray) -> Dict[str, float]:
    """
    Compute validation metrics for contact estimation.

    Args:
        contacts: Estimated contact signals
        timestamps: Time array

    Returns:
        Dict of validation metrics
    """
    metrics = {}

    for foot_name, contact_signal in contacts.items():
        # Contact ratio
        contact_ratio = np.mean(contact_signal > 0.5)

        # Average contact duration
        binary_contacts = (contact_signal > 0.5).astype(int)
        diff = np.diff(np.concatenate(([0], binary_contacts, [0])))
        contact_starts = np.where(diff == 1)[0]
        contact_ends = np.where(diff == -1)[0]

        durations = []
        for start, end in zip(contact_starts, contact_ends):
            if end > start and end-1 < len(timestamps):
                duration = timestamps[end-1] - timestamps[start]
                durations.append(duration)

        avg_duration = np.mean(durations) if durations else 0.0

        # Number of contact events
        n_contacts = len(durations)

        # Signal smoothness (low values = smooth)
        smoothness = np.mean(np.abs(np.diff(contact_signal)))

        metrics[f"{foot_name}_contact_ratio"] = contact_ratio
        metrics[f"{foot_name}_avg_duration"] = avg_duration
        metrics[f"{foot_name}_n_contacts"] = n_contacts
        metrics[f"{foot_name}_smoothness"] = smoothness

    # Overall metrics
    total_contacts = sum(v for k, v in metrics.items() if k.endswith('_n_contacts'))
    overall_duration = sum(v for k, v in metrics.items() if k.endswith('_avg_duration')) / 2

    metrics["total_contact_events"] = total_contacts
    metrics["overall_avg_duration"] = overall_duration
    metrics["sequence_duration"] = timestamps[-1] - timestamps[0]

    return metrics