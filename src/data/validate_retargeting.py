"""
Validation module for motion retargeting quality assessment.

This module provides comprehensive validation tools to assess the quality
of retargeted motion data and identify potential issues before training.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
import matplotlib.pyplot as plt
from pathlib import Path

from .retarget_to_g1 import G1TrajectoryData, MotionClipData
from .skeleton_map import G1_JOINT_MAP
from .contact_estimation import ContactInfo

logger = logging.getLogger(__name__)


@dataclass
class ValidationMetrics:
    """Container for validation metrics."""
    # Overall metrics
    sequence_duration: float
    n_frames: int
    framerate: float

    # Joint metrics
    joint_limit_violations: Dict[str, int]
    joint_velocity_violations: Dict[str, int]
    joint_acceleration_violations: Dict[str, int]
    joint_continuity_score: float

    # Root motion metrics
    root_position_range: Dict[str, Tuple[float, float]]
    root_velocity_range: Dict[str, Tuple[float, float]]
    root_trajectory_smoothness: float

    # Contact metrics
    contact_consistency_score: float
    foot_penetration_frames: int
    contact_transition_smoothness: float

    # Kinematic metrics
    kinematic_feasibility_score: float
    energy_efficiency_score: float
    naturalness_score: float

    # Quality flags
    has_critical_issues: bool
    has_warnings: bool
    overall_quality_score: float


@dataclass
class ValidationConfig:
    """Configuration for retargeting validation."""
    # Joint limit checking
    joint_limit_tolerance: float = 0.1  # radians beyond limit
    max_joint_velocity: float = 20.0    # rad/s
    max_joint_acceleration: float = 100.0  # rad/s²

    # Root motion bounds
    max_root_displacement: float = 10.0  # meters per frame
    max_root_velocity: float = 5.0       # m/s
    max_root_acceleration: float = 20.0  # m/s²

    # Contact validation
    min_contact_duration: float = 0.05   # seconds
    max_foot_penetration: float = 0.01   # meters
    contact_transition_smoothness_threshold: float = 0.5

    # Quality thresholds
    critical_quality_threshold: float = 0.3
    warning_quality_threshold: float = 0.7

    # Plotting
    generate_plots: bool = True
    plot_output_dir: Optional[str] = None


class RetargetingValidator:
    """Comprehensive validation for retargeted motion data."""

    def __init__(self, config: Optional[ValidationConfig] = None):
        """
        Initialize retargeting validator.

        Args:
            config: Validation configuration
        """
        self.config = config or ValidationConfig()
        self.joint_limits = self._build_joint_limits()

        logger.info("Initialized RetargetingValidator")

    def _build_joint_limits(self) -> Dict[str, Tuple[float, float]]:
        """Build joint limits dictionary."""
        limits = {}
        for joint_name, joint_info in G1_JOINT_MAP.items():
            limits[joint_name] = joint_info.limits
        return limits

    def validate_trajectory(self, trajectory: G1TrajectoryData,
                          original_motion: Optional[MotionClipData] = None,
                          contact_events: Optional[List[ContactInfo]] = None) -> ValidationMetrics:
        """
        Perform comprehensive validation of a retargeted trajectory.

        Args:
            trajectory: Retargeted G1 trajectory
            original_motion: Optional original human motion for comparison
            contact_events: Optional contact events for validation

        Returns:
            ValidationMetrics object
        """
        logger.info(f"Validating trajectory: {trajectory.metadata.get('duration', 0):.2f}s, "
                   f"{len(trajectory.timestamps)} frames")

        # Basic metrics
        sequence_duration = trajectory.timestamps[-1] - trajectory.timestamps[0]
        n_frames = len(trajectory.timestamps)
        framerate = n_frames / sequence_duration if sequence_duration > 0 else 0

        # Joint validation
        joint_limit_violations = self._check_joint_limits(trajectory)
        joint_velocity_violations = self._check_joint_velocities(trajectory)
        joint_acceleration_violations = self._check_joint_accelerations(trajectory)
        joint_continuity_score = self._compute_joint_continuity(trajectory)

        # Root motion validation
        root_position_range = self._compute_root_position_range(trajectory)
        root_velocity_range = self._compute_root_velocity_range(trajectory)
        root_trajectory_smoothness = self._compute_root_smoothness(trajectory)

        # Contact validation
        contact_consistency_score = self._compute_contact_consistency(trajectory, contact_events)
        foot_penetration_frames = self._check_foot_penetration(trajectory)
        contact_transition_smoothness = self._compute_contact_transition_smoothness(trajectory)

        # Kinematic validation
        kinematic_feasibility_score = self._compute_kinematic_feasibility(trajectory)
        energy_efficiency_score = self._compute_energy_efficiency(trajectory)
        naturalness_score = self._compute_naturalness_score(trajectory)

        # Overall quality assessment
        quality_scores = [
            joint_continuity_score,
            root_trajectory_smoothness,
            contact_consistency_score,
            kinematic_feasibility_score,
            energy_efficiency_score,
            naturalness_score
        ]
        overall_quality_score = np.mean(quality_scores)

        # Quality flags
        has_critical_issues = overall_quality_score < self.config.critical_quality_threshold
        has_warnings = overall_quality_score < self.config.warning_quality_threshold

        # Log critical issues
        if has_critical_issues:
            logger.warning(f"Critical quality issues detected (score: {overall_quality_score:.3f})")

        metrics = ValidationMetrics(
            sequence_duration=sequence_duration,
            n_frames=n_frames,
            framerate=framerate,
            joint_limit_violations=joint_limit_violations,
            joint_velocity_violations=joint_velocity_violations,
            joint_acceleration_violations=joint_acceleration_violations,
            joint_continuity_score=joint_continuity_score,
            root_position_range=root_position_range,
            root_velocity_range=root_velocity_range,
            root_trajectory_smoothness=root_trajectory_smoothness,
            contact_consistency_score=contact_consistency_score,
            foot_penetration_frames=foot_penetration_frames,
            contact_transition_smoothness=contact_transition_smoothness,
            kinematic_feasibility_score=kinematic_feasibility_score,
            energy_efficiency_score=energy_efficiency_score,
            naturalness_score=naturalness_score,
            has_critical_issues=has_critical_issues,
            has_warnings=has_warnings,
            overall_quality_score=overall_quality_score
        )

        # Generate plots if requested
        if self.config.generate_plots:
            self._generate_validation_plots(trajectory, metrics)

        return metrics

    def _check_joint_limits(self, trajectory: G1TrajectoryData) -> Dict[str, int]:
        """Check for joint limit violations."""
        violations = {}
        joint_names = trajectory.metadata.get("joint_names", list(G1_JOINT_MAP.keys()))

        for i, joint_name in enumerate(joint_names):
            if joint_name in self.joint_limits:
                limits = self.joint_limits[joint_name]
                joint_angles = trajectory.joint_positions[:, i]

                # Count violations with tolerance
                lower_violations = np.sum(joint_angles < (limits[0] - self.config.joint_limit_tolerance))
                upper_violations = np.sum(joint_angles > (limits[1] + self.config.joint_limit_tolerance))

                violations[joint_name] = lower_violations + upper_violations

        return violations

    def _check_joint_velocities(self, trajectory: G1TrajectoryData) -> Dict[str, int]:
        """Check for excessive joint velocities."""
        violations = {}
        joint_names = trajectory.metadata.get("joint_names", list(G1_JOINT_MAP.keys()))

        joint_speeds = np.abs(trajectory.joint_velocities)

        for i, joint_name in enumerate(joint_names):
            violations[joint_name] = np.sum(joint_speeds[:, i] > self.config.max_joint_velocity)

        return violations

    def _check_joint_accelerations(self, trajectory: G1TrajectoryData) -> Dict[str, int]:
        """Check for excessive joint accelerations."""
        violations = {}
        joint_names = trajectory.metadata.get("joint_names", list(G1_JOINT_MAP.keys()))

        joint_acc_magnitudes = np.abs(trajectory.joint_accelerations)

        for i, joint_name in enumerate(joint_names):
            violations[joint_name] = np.sum(joint_acc_magnitudes[:, i] > self.config.max_joint_acceleration)

        return violations

    def _compute_joint_continuity(self, trajectory: G1TrajectoryData) -> float:
        """Compute joint trajectory continuity score."""
        # Measure smoothness using second derivatives
        joint_positions = trajectory.joint_positions

        if len(joint_positions) < 3:
            return 1.0  # Too short to measure

        # Compute second derivatives (discrete)
        second_derivatives = np.diff(joint_positions, n=2, axis=0)

        # Continuity score based on smoothness (lower second derivatives = higher score)
        smoothness_scores = 1.0 / (1.0 + np.mean(np.abs(second_derivatives), axis=0))

        return np.mean(smoothness_scores)

    def _compute_root_position_range(self, trajectory: G1TrajectoryData) -> Dict[str, Tuple[float, float]]:
        """Compute root position ranges."""
        root_positions = trajectory.root_positions

        return {
            "x": (np.min(root_positions[:, 0]), np.max(root_positions[:, 0])),
            "y": (np.min(root_positions[:, 1]), np.max(root_positions[:, 1])),
            "z": (np.min(root_positions[:, 2]), np.max(root_positions[:, 2]))
        }

    def _compute_root_velocity_range(self, trajectory: G1TrajectoryData) -> Dict[str, Tuple[float, float]]:
        """Compute root velocity ranges."""
        root_velocities = trajectory.root_linear_velocities

        return {
            "linear": (np.min(np.linalg.norm(root_velocities, axis=1)),
                      np.max(np.linalg.norm(root_velocities, axis=1))),
            "angular": (np.min(np.linalg.norm(trajectory.root_angular_velocities, axis=1)),
                       np.max(np.linalg.norm(trajectory.root_angular_velocities, axis=1)))
        }

    def _compute_root_smoothness(self, trajectory: G1TrajectoryData) -> float:
        """Compute root trajectory smoothness."""
        # Measure jerk (third derivative) for smoothness
        root_positions = trajectory.root_positions

        if len(root_positions) < 4:
            return 1.0

        # Compute third derivatives
        third_derivatives = np.diff(root_positions, n=3, axis=0)
        jerk_magnitude = np.mean(np.linalg.norm(third_derivatives, axis=1))

        # Convert to score (lower jerk = higher score)
        smoothness_score = 1.0 / (1.0 + jerk_magnitude)

        return smoothness_score

    def _compute_contact_consistency(self, trajectory: G1TrajectoryData,
                                   contact_events: Optional[List[ContactInfo]]) -> float:
        """Compute contact consistency score."""
        if contact_events is None:
            # Basic consistency check using foot contacts array
            foot_contacts = trajectory.foot_contacts

            # Check for rapid contact changes (inconsistency)
            contact_changes = np.sum(np.abs(np.diff(foot_contacts, axis=0)))
            max_possible_changes = len(foot_contacts) * 2  # 2 feet

            consistency_score = 1.0 - (contact_changes / max_possible_changes)
            return max(0, consistency_score)

        # More detailed analysis with contact events
        total_duration = trajectory.timestamps[-1] - trajectory.timestamps[0]
        contact_duration = sum(event.duration for event in contact_events)

        # Reasonable contact ratio (50-90% of time in contact is reasonable for walking)
        contact_ratio = contact_duration / total_duration
        optimal_range = (0.5, 0.9)

        if optimal_range[0] <= contact_ratio <= optimal_range[1]:
            return 1.0
        else:
            return max(0, 1.0 - abs(contact_ratio - 0.7) / 0.3)

    def _check_foot_penetration(self, trajectory: G1TrajectoryData) -> int:
        """Check for foot ground penetration."""
        # Estimate foot heights (simplified - assumes lowest point is foot)
        root_heights = trajectory.root_positions[:, 2]
        min_expected_foot_height = root_heights - 1.0  # Approximate leg length

        # Count frames where estimated foot height is below ground + tolerance
        penetration_threshold = -self.config.max_foot_penetration
        penetration_frames = np.sum(min_expected_foot_height < penetration_threshold)

        return penetration_frames

    def _compute_contact_transition_smoothness(self, trajectory: G1TrajectoryData) -> float:
        """Compute smoothness of contact transitions."""
        foot_contacts = trajectory.foot_contacts

        # Compute gradient of contact signals
        contact_gradients = np.abs(np.diff(foot_contacts, axis=0))

        # Smooth transitions have small gradients
        transition_smoothness = 1.0 - np.mean(contact_gradients)

        return max(0, transition_smoothness)

    def _compute_kinematic_feasibility(self, trajectory: G1TrajectoryData) -> float:
        """Compute kinematic feasibility score."""
        # Check if joint velocities and accelerations are within reasonable bounds
        joint_velocities = trajectory.joint_velocities
        joint_accelerations = trajectory.joint_accelerations

        # Feasibility based on velocity and acceleration magnitudes
        velocity_feasibility = np.mean(
            np.linalg.norm(joint_velocities, axis=1) < self.config.max_joint_velocity
        )

        acceleration_feasibility = np.mean(
            np.linalg.norm(joint_accelerations, axis=1) < self.config.max_joint_acceleration
        )

        return (velocity_feasibility + acceleration_feasibility) / 2

    def _compute_energy_efficiency(self, trajectory: G1TrajectoryData) -> float:
        """Compute energy efficiency score based on motion smoothness."""
        # Energy efficiency approximated by low accelerations and smooth motion
        joint_accelerations = trajectory.joint_accelerations

        # Lower accelerations suggest more efficient motion
        mean_acceleration = np.mean(np.linalg.norm(joint_accelerations, axis=1))
        efficiency_score = 1.0 / (1.0 + mean_acceleration / 10.0)  # Normalize

        return efficiency_score

    def _compute_naturalness_score(self, trajectory: G1TrajectoryData) -> float:
        """Compute naturalness score based on motion patterns."""
        # Naturalness based on periodicity and symmetry (for walking motions)
        joint_positions = trajectory.joint_positions

        # Simple naturalness metric: consistency of joint ranges
        joint_ranges = np.max(joint_positions, axis=0) - np.min(joint_positions, axis=0)
        range_consistency = 1.0 - np.std(joint_ranges) / (np.mean(joint_ranges) + 1e-6)

        return max(0, range_consistency)

    def _generate_validation_plots(self, trajectory: G1TrajectoryData,
                                 metrics: ValidationMetrics) -> None:
        """Generate validation plots."""
        if self.config.plot_output_dir is None:
            return

        output_dir = Path(self.config.plot_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Plot 1: Joint trajectories
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Joint Validation Plots')

            # Joint positions
            axes[0, 0].plot(trajectory.timestamps, trajectory.joint_positions)
            axes[0, 0].set_title('Joint Positions')
            axes[0, 0].set_xlabel('Time (s)')
            axes[0, 0].set_ylabel('Angle (rad)')

            # Joint velocities
            axes[0, 1].plot(trajectory.timestamps, trajectory.joint_velocities)
            axes[0, 1].set_title('Joint Velocities')
            axes[0, 1].set_xlabel('Time (s)')
            axes[0, 1].set_ylabel('Angular Velocity (rad/s)')

            # Root position
            axes[1, 0].plot(trajectory.timestamps, trajectory.root_positions)
            axes[1, 0].set_title('Root Position')
            axes[1, 0].set_xlabel('Time (s)')
            axes[1, 0].set_ylabel('Position (m)')
            axes[1, 0].legend(['X', 'Y', 'Z'])

            # Foot contacts
            axes[1, 1].plot(trajectory.timestamps, trajectory.foot_contacts)
            axes[1, 1].set_title('Foot Contacts')
            axes[1, 1].set_xlabel('Time (s)')
            axes[1, 1].set_ylabel('Contact Probability')
            axes[1, 1].legend(['Left', 'Right'])

            plt.tight_layout()
            plt.savefig(output_dir / 'joint_validation.png')
            plt.close()

            # Plot 2: Quality metrics
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))

            quality_metrics = [
                ('Joint Continuity', metrics.joint_continuity_score),
                ('Root Smoothness', metrics.root_trajectory_smoothness),
                ('Contact Consistency', metrics.contact_consistency_score),
                ('Kinematic Feasibility', metrics.kinematic_feasibility_score),
                ('Energy Efficiency', metrics.energy_efficiency_score),
                ('Naturalness', metrics.naturalness_score),
                ('Overall Quality', metrics.overall_quality_score)
            ]

            metric_names, metric_values = zip(*quality_metrics)
            bars = ax.bar(metric_names, metric_values)

            # Color bars based on quality
            for i, (bar, value) in enumerate(zip(bars, metric_values)):
                if value < self.config.critical_quality_threshold:
                    bar.set_color('red')
                elif value < self.config.warning_quality_threshold:
                    bar.set_color('orange')
                else:
                    bar.set_color('green')

            ax.set_title('Quality Metrics')
            ax.set_ylabel('Score')
            ax.set_ylim(0, 1)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(output_dir / 'quality_metrics.png')
            plt.close()

            logger.info(f"Validation plots saved to {output_dir}")

        except Exception as e:
            logger.warning(f"Failed to generate validation plots: {e}")

    def validate_batch(self, trajectories: List[G1TrajectoryData],
                      output_summary: bool = True) -> List[ValidationMetrics]:
        """
        Validate a batch of trajectories.

        Args:
            trajectories: List of trajectories to validate
            output_summary: Whether to output summary statistics

        Returns:
            List of ValidationMetrics
        """
        logger.info(f"Validating batch of {len(trajectories)} trajectories")

        all_metrics = []
        for i, trajectory in enumerate(trajectories):
            try:
                metrics = self.validate_trajectory(trajectory)
                all_metrics.append(metrics)
            except Exception as e:
                logger.error(f"Failed to validate trajectory {i}: {e}")
                continue

        if output_summary and all_metrics:
            self._print_batch_summary(all_metrics)

        return all_metrics

    def _print_batch_summary(self, all_metrics: List[ValidationMetrics]) -> None:
        """Print summary statistics for batch validation."""
        print("\n" + "="*60)
        print("BATCH VALIDATION SUMMARY")
        print("="*60)

        n_trajectories = len(all_metrics)
        n_critical = sum(m.has_critical_issues for m in all_metrics)
        n_warnings = sum(m.has_warnings for m in all_metrics)

        print(f"Total trajectories validated: {n_trajectories}")
        print(f"Critical issues: {n_critical} ({100*n_critical/n_trajectories:.1f}%)")
        print(f"Warnings: {n_warnings} ({100*n_warnings/n_trajectories:.1f}%)")

        # Quality score distribution
        quality_scores = [m.overall_quality_score for m in all_metrics]
        print(f"\nQuality Score Distribution:")
        print(f"  Mean: {np.mean(quality_scores):.3f}")
        print(f"  Std:  {np.std(quality_scores):.3f}")
        print(f"  Min:  {np.min(quality_scores):.3f}")
        print(f"  Max:  {np.max(quality_scores):.3f}")

        # Joint limit violations summary
        all_violations = {}
        for metrics in all_metrics:
            for joint, count in metrics.joint_limit_violations.items():
                if joint not in all_violations:
                    all_violations[joint] = []
                all_violations[joint].append(count)

        print(f"\nJoint Limit Violations (mean per trajectory):")
        for joint, violations in all_violations.items():
            if np.mean(violations) > 0:
                print(f"  {joint}: {np.mean(violations):.1f}")

        print("="*60)


def validate_retargeted_motion(trajectory: G1TrajectoryData,
                              config: Optional[ValidationConfig] = None) -> ValidationMetrics:
    """
    Convenience function for validating a single retargeted trajectory.

    Args:
        trajectory: G1 trajectory to validate
        config: Optional validation configuration

    Returns:
        ValidationMetrics object
    """
    validator = RetargetingValidator(config)
    return validator.validate_trajectory(trajectory)