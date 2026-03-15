"""
Safety filtering system for real hardware deployment.
Monitors robot state and filters dangerous commands.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time

from ..core.config import Config
from ..core.logging import get_logger


logger = get_logger(__name__)


class SafetyLevel(Enum):
    """Safety alert levels."""
    SAFE = 0
    WARNING = 1
    CRITICAL = 2
    EMERGENCY = 3


@dataclass
class SafetyViolation:
    """Record of a safety violation."""
    timestamp: float
    level: SafetyLevel
    violation_type: str
    description: str
    suggested_action: str
    data: Dict[str, Any]


class SafetyFilter:
    """
    Real-time safety filter for Unitree G1.

    This filter monitors robot state and commands to prevent:
    - Joint limit violations
    - Excessive velocities and accelerations
    - Self-collision risks
    - Fall detection and prevention
    - Emergency stop conditions
    """

    def __init__(self, config: Config):
        self.config = config
        self.safety_config = config.get("hardware.safety", {})

        # Safety limits
        self.joint_limits = self._load_joint_limits()
        self.velocity_limits = self._load_velocity_limits()
        self.acceleration_limits = self._load_acceleration_limits()

        # Fall detection parameters
        self.fall_detection_config = self.safety_config.get("fall_detection", {})
        self.min_base_height = self.fall_detection_config.get("min_height", 0.3)
        self.max_base_tilt = self.fall_detection_config.get("max_tilt", 0.5)  # radians

        # Emergency stop conditions
        self.emergency_config = self.safety_config.get("emergency_stop", {})
        self.max_joint_velocity = self.emergency_config.get("max_joint_velocity", 20.0)
        self.max_contact_force = self.emergency_config.get("max_contact_force", 500.0)

        # State tracking
        self.previous_state = None
        self.violation_history: List[SafetyViolation] = []
        self.emergency_stop_triggered = False

        # Safety monitoring
        self.monitoring_enabled = True

        logger.info("Safety filter initialized")
        logger.info(f"Joint limits: {len(self.joint_limits)} joints")
        logger.info(f"Fall detection enabled: min_height={self.min_base_height}m")

    def _load_joint_limits(self) -> List[Tuple[float, float]]:
        """Load joint position limits."""
        # TODO: Load from robot URDF/config
        # For now, use safe default limits for G1
        default_limits = [
            (-1.57, 1.57),   # Hip yaw
            (-0.52, 0.52),   # Hip roll
            (-1.57, 1.57),   # Hip pitch
            (0.0, 2.35),     # Knee
            (-0.87, 0.87),   # Ankle pitch
            (-0.52, 0.52),   # Ankle roll
        ]

        # Repeat for both legs and arms
        joint_limits = []
        for _ in range(2):  # Left and right leg
            joint_limits.extend(default_limits)

        # Add torso limits
        joint_limits.extend([(-0.26, 0.26), (-0.26, 0.26)])  # Torso pitch, roll

        # Add arm limits (simplified)
        arm_limits = [(-1.57, 1.57), (-1.57, 1.57), (-2.09, 2.09)]  # Shoulder pitch, roll, elbow
        for _ in range(2):  # Left and right arm
            joint_limits.extend(arm_limits)

        # Add head limits
        joint_limits.extend([(-1.57, 1.57), (-0.52, 0.52)])  # Head yaw, pitch

        return joint_limits

    def _load_velocity_limits(self) -> List[float]:
        """Load joint velocity limits."""
        # Default safe velocity limits (rad/s)
        default_velocity = 5.0
        return [default_velocity] * len(self.joint_limits)

    def _load_acceleration_limits(self) -> List[float]:
        """Load joint acceleration limits."""
        # Default safe acceleration limits (rad/s^2)
        default_acceleration = 50.0
        return [default_acceleration] * len(self.joint_limits)

    def filter_commands(
        self,
        desired_commands: np.ndarray,
        current_state: Dict[str, Any],
        dt: float = 0.01
    ) -> Tuple[np.ndarray, List[SafetyViolation]]:
        """
        Filter robot commands for safety.

        Args:
            desired_commands: Desired joint commands
            current_state: Current robot state
            dt: Time step

        Returns:
            Filtered commands and list of violations
        """
        if not self.monitoring_enabled:
            return desired_commands, []

        violations = []
        filtered_commands = desired_commands.copy()

        # Check emergency stop conditions
        emergency_violations = self._check_emergency_conditions(current_state)
        violations.extend(emergency_violations)

        if emergency_violations:
            # Emergency stop: return zero commands
            self.emergency_stop_triggered = True
            filtered_commands = np.zeros_like(desired_commands)
            logger.error("EMERGENCY STOP TRIGGERED")
            return filtered_commands, violations

        # Joint limit checking and enforcement
        joint_violations = self._check_joint_limits(filtered_commands, current_state)
        violations.extend(joint_violations)
        filtered_commands = self._enforce_joint_limits(filtered_commands)

        # Velocity limit checking
        velocity_violations = self._check_velocity_limits(filtered_commands, current_state, dt)
        violations.extend(velocity_violations)
        filtered_commands = self._enforce_velocity_limits(filtered_commands, current_state, dt)

        # Acceleration limit checking
        accel_violations = self._check_acceleration_limits(filtered_commands, current_state, dt)
        violations.extend(accel_violations)
        filtered_commands = self._enforce_acceleration_limits(filtered_commands, current_state, dt)

        # Fall detection
        fall_violations = self._check_fall_conditions(current_state)
        violations.extend(fall_violations)

        # Self-collision detection (simplified)
        collision_violations = self._check_collision_risk(filtered_commands, current_state)
        violations.extend(collision_violations)

        # Update state history
        self.previous_state = current_state.copy()

        # Log violations
        for violation in violations:
            self.violation_history.append(violation)
            if violation.level == SafetyLevel.CRITICAL:
                logger.error(f"CRITICAL SAFETY VIOLATION: {violation.description}")
            elif violation.level == SafetyLevel.WARNING:
                logger.warning(f"Safety warning: {violation.description}")

        return filtered_commands, violations

    def _check_emergency_conditions(self, state: Dict[str, Any]) -> List[SafetyViolation]:
        """Check for emergency stop conditions."""
        violations = []
        timestamp = time.time()

        # Check joint velocities
        joint_velocities = state.get("joint_velocities", np.array([]))
        if len(joint_velocities) > 0:
            max_velocity = np.max(np.abs(joint_velocities))
            if max_velocity > self.max_joint_velocity:
                violations.append(SafetyViolation(
                    timestamp=timestamp,
                    level=SafetyLevel.EMERGENCY,
                    violation_type="excessive_velocity",
                    description=f"Joint velocity {max_velocity:.2f} exceeds limit {self.max_joint_velocity}",
                    suggested_action="Emergency stop",
                    data={"max_velocity": max_velocity, "joint_velocities": joint_velocities}
                ))

        # Check contact forces
        contact_forces = state.get("contact_forces", np.array([]))
        if len(contact_forces) > 0:
            max_force = np.max(contact_forces)
            if max_force > self.max_contact_force:
                violations.append(SafetyViolation(
                    timestamp=timestamp,
                    level=SafetyLevel.EMERGENCY,
                    violation_type="excessive_force",
                    description=f"Contact force {max_force:.1f}N exceeds limit {self.max_contact_force}N",
                    suggested_action="Emergency stop",
                    data={"max_force": max_force, "contact_forces": contact_forces}
                ))

        return violations

    def _check_joint_limits(self, commands: np.ndarray, state: Dict[str, Any]) -> List[SafetyViolation]:
        """Check for joint limit violations."""
        violations = []
        timestamp = time.time()

        joint_positions = state.get("joint_positions", np.array([]))
        if len(joint_positions) != len(self.joint_limits):
            return violations

        for i, (pos, (min_limit, max_limit)) in enumerate(zip(joint_positions, self.joint_limits)):
            margin = 0.1  # Safety margin in radians

            if pos <= min_limit + margin:
                violations.append(SafetyViolation(
                    timestamp=timestamp,
                    level=SafetyLevel.WARNING,
                    violation_type="joint_limit_min",
                    description=f"Joint {i} near minimum limit: {pos:.3f} <= {min_limit + margin:.3f}",
                    suggested_action="Limit command",
                    data={"joint_id": i, "position": pos, "limit": min_limit}
                ))

            elif pos >= max_limit - margin:
                violations.append(SafetyViolation(
                    timestamp=timestamp,
                    level=SafetyLevel.WARNING,
                    violation_type="joint_limit_max",
                    description=f"Joint {i} near maximum limit: {pos:.3f} >= {max_limit - margin:.3f}",
                    suggested_action="Limit command",
                    data={"joint_id": i, "position": pos, "limit": max_limit}
                ))

        return violations

    def _check_fall_conditions(self, state: Dict[str, Any]) -> List[SafetyViolation]:
        """Check for fall conditions."""
        violations = []
        timestamp = time.time()

        # Check base height
        base_position = state.get("base_position", np.array([0, 0, 0]))
        base_height = base_position[2] if len(base_position) > 2 else 0.0

        if base_height < self.min_base_height:
            violations.append(SafetyViolation(
                timestamp=timestamp,
                level=SafetyLevel.CRITICAL,
                violation_type="low_base_height",
                description=f"Base height {base_height:.3f}m below minimum {self.min_base_height}m",
                suggested_action="Attempt recovery or emergency stop",
                data={"base_height": base_height, "min_height": self.min_base_height}
            ))

        # Check base orientation
        base_orientation = state.get("base_orientation", np.array([1, 0, 0, 0]))  # quaternion
        if len(base_orientation) >= 4:
            # Convert quaternion to euler angles (simplified)
            from ..core.utils import quaternion_to_euler
            euler = quaternion_to_euler(base_orientation)
            roll, pitch = euler[0], euler[1]

            if abs(roll) > self.max_base_tilt or abs(pitch) > self.max_base_tilt:
                violations.append(SafetyViolation(
                    timestamp=timestamp,
                    level=SafetyLevel.CRITICAL,
                    violation_type="excessive_tilt",
                    description=f"Base tilt ({roll:.3f}, {pitch:.3f}) exceeds limit {self.max_base_tilt:.3f}",
                    suggested_action="Stabilize base orientation",
                    data={"roll": roll, "pitch": pitch, "max_tilt": self.max_base_tilt}
                ))

        return violations

    def _check_velocity_limits(
        self,
        commands: np.ndarray,
        state: Dict[str, Any],
        dt: float
    ) -> List[SafetyViolation]:
        """Check velocity limit violations."""
        violations = []
        timestamp = time.time()

        current_positions = state.get("joint_positions", np.array([]))
        current_velocities = state.get("joint_velocities", np.array([]))

        if len(current_velocities) != len(self.velocity_limits):
            return violations

        for i, (velocity, limit) in enumerate(zip(current_velocities, self.velocity_limits)):
            if abs(velocity) > limit * 0.9:  # 90% of limit as warning
                level = SafetyLevel.WARNING if abs(velocity) < limit else SafetyLevel.CRITICAL

                violations.append(SafetyViolation(
                    timestamp=timestamp,
                    level=level,
                    violation_type="velocity_limit",
                    description=f"Joint {i} velocity {velocity:.3f} near/exceeds limit {limit:.3f}",
                    suggested_action="Limit velocity command",
                    data={"joint_id": i, "velocity": velocity, "limit": limit}
                ))

        return violations

    def _check_acceleration_limits(
        self,
        commands: np.ndarray,
        state: Dict[str, Any],
        dt: float
    ) -> List[SafetyViolation]:
        """Check acceleration limit violations."""
        violations = []
        if self.previous_state is None or dt <= 0:
            return violations

        timestamp = time.time()

        current_velocities = state.get("joint_velocities", np.array([]))
        previous_velocities = self.previous_state.get("joint_velocities", np.array([]))

        if len(current_velocities) != len(previous_velocities) or len(current_velocities) != len(self.acceleration_limits):
            return violations

        # Compute accelerations
        accelerations = (current_velocities - previous_velocities) / dt

        for i, (accel, limit) in enumerate(zip(accelerations, self.acceleration_limits)):
            if abs(accel) > limit * 0.9:  # 90% of limit as warning
                level = SafetyLevel.WARNING if abs(accel) < limit else SafetyLevel.CRITICAL

                violations.append(SafetyViolation(
                    timestamp=timestamp,
                    level=level,
                    violation_type="acceleration_limit",
                    description=f"Joint {i} acceleration {accel:.1f} near/exceeds limit {limit:.1f}",
                    suggested_action="Smooth command trajectory",
                    data={"joint_id": i, "acceleration": accel, "limit": limit}
                ))

        return violations

    def _check_collision_risk(self, commands: np.ndarray, state: Dict[str, Any]) -> List[SafetyViolation]:
        """Check for self-collision risk (simplified implementation)."""
        violations = []
        timestamp = time.time()

        # Simple collision check based on joint configurations
        # In practice, this would use detailed collision detection

        joint_positions = state.get("joint_positions", np.array([]))
        if len(joint_positions) < 6:  # Need at least leg joints
            return violations

        # Check for problematic configurations
        # Example: both knees fully bent
        left_knee = joint_positions[3] if len(joint_positions) > 3 else 0
        right_knee = joint_positions[9] if len(joint_positions) > 9 else 0

        if left_knee > 2.0 and right_knee > 2.0:
            violations.append(SafetyViolation(
                timestamp=timestamp,
                level=SafetyLevel.WARNING,
                violation_type="collision_risk",
                description="Both knees highly flexed - collision risk",
                suggested_action="Adjust leg configuration",
                data={"left_knee": left_knee, "right_knee": right_knee}
            ))

        return violations

    def _enforce_joint_limits(self, commands: np.ndarray) -> np.ndarray:
        """Enforce joint limits on commands."""
        filtered = commands.copy()

        for i, (command, (min_limit, max_limit)) in enumerate(zip(commands, self.joint_limits)):
            if i < len(filtered):
                margin = 0.05  # Safety margin
                filtered[i] = np.clip(command, min_limit + margin, max_limit - margin)

        return filtered

    def _enforce_velocity_limits(
        self,
        commands: np.ndarray,
        state: Dict[str, Any],
        dt: float
    ) -> np.ndarray:
        """Enforce velocity limits on commands."""
        if self.previous_state is None or dt <= 0:
            return commands

        filtered = commands.copy()
        current_positions = state.get("joint_positions", np.array([]))
        previous_positions = self.previous_state.get("joint_positions", current_positions)

        if len(current_positions) != len(previous_positions):
            return commands

        for i, limit in enumerate(self.velocity_limits):
            if i < len(filtered) and i < len(current_positions):
                # Estimate required velocity
                required_velocity = (commands[i] - current_positions[i]) / dt
                max_velocity = limit * 0.8  # Conservative limit

                if abs(required_velocity) > max_velocity:
                    # Scale command to respect velocity limit
                    sign = np.sign(required_velocity)
                    max_position_change = sign * max_velocity * dt
                    filtered[i] = current_positions[i] + max_position_change

        return filtered

    def _enforce_acceleration_limits(
        self,
        commands: np.ndarray,
        state: Dict[str, Any],
        dt: float
    ) -> np.ndarray:
        """Enforce acceleration limits on commands."""
        # Simplified implementation - in practice would use more sophisticated trajectory smoothing
        return commands

    def reset_emergency_stop(self):
        """Reset emergency stop condition (manual override)."""
        self.emergency_stop_triggered = False
        logger.info("Emergency stop reset")

    def get_safety_status(self) -> Dict[str, Any]:
        """Get current safety status."""
        recent_violations = [v for v in self.violation_history if time.time() - v.timestamp < 10.0]

        return {
            "monitoring_enabled": self.monitoring_enabled,
            "emergency_stop_triggered": self.emergency_stop_triggered,
            "recent_violations": len(recent_violations),
            "total_violations": len(self.violation_history),
            "last_violation": self.violation_history[-1].__dict__ if self.violation_history else None,
        }