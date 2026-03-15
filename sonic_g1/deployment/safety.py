"""
Safety filter system for G1 robot action validation and filtering.

This module implements comprehensive safety checks including joint limits,
velocity limits, action rate limiting, and emergency stop conditions.
"""

import numpy as np
import time
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class SafetyLimits:
    """Safety limits configuration."""
    joint_position_min: np.ndarray
    joint_position_max: np.ndarray
    joint_velocity_max: np.ndarray
    joint_torque_max: np.ndarray
    action_rate_limit: float
    emergency_action: np.ndarray


@dataclass
class SafetyViolation:
    """Safety violation information."""
    violation_type: str
    joint_indices: List[int]
    severity: str  # 'warning', 'critical'
    values: np.ndarray
    limits: np.ndarray
    timestamp: float


class SafetyCheck(ABC):
    """Abstract base class for safety checks."""

    @abstractmethod
    def check(self, action: np.ndarray, observation: Any,
             previous_action: Optional[np.ndarray] = None) -> Tuple[bool, Optional[SafetyViolation]]:
        """
        Perform safety check.

        Args:
            action: Proposed action
            observation: Current robot observation
            previous_action: Previous action for rate checking

        Returns:
            Tuple of (is_safe, violation_info)
        """
        pass

    @abstractmethod
    def correct_action(self, action: np.ndarray, violation: SafetyViolation) -> np.ndarray:
        """
        Correct unsafe action.

        Args:
            action: Original unsafe action
            violation: Safety violation information

        Returns:
            Corrected safe action
        """
        pass


class JointLimitsCheck(SafetyCheck):
    """Joint position and velocity limits safety check."""

    def __init__(self, limits: SafetyLimits, prediction_horizon: float = 0.01):
        """
        Initialize joint limits checker.

        Args:
            limits: Safety limits configuration
            prediction_horizon: Time horizon for velocity prediction (seconds)
        """
        self.limits = limits
        self.prediction_horizon = prediction_horizon

    def check(self, action: np.ndarray, observation: Any,
             previous_action: Optional[np.ndarray] = None) -> Tuple[bool, Optional[SafetyViolation]]:
        """Check joint position and velocity limits."""

        # Extract joint positions and velocities from observation
        joint_pos = observation.joint_positions
        joint_vel = observation.joint_velocities

        # Check current position limits
        pos_violations = []
        pos_violation_indices = []

        min_violations = joint_pos < self.limits.joint_position_min
        max_violations = joint_pos > self.limits.joint_position_max

        if np.any(min_violations):
            pos_violation_indices.extend(np.where(min_violations)[0].tolist())
            pos_violations.extend((joint_pos - self.limits.joint_position_min)[min_violations])

        if np.any(max_violations):
            pos_violation_indices.extend(np.where(max_violations)[0].tolist())
            pos_violations.extend((joint_pos - self.limits.joint_position_max)[max_violations])

        if pos_violations:
            violation = SafetyViolation(
                violation_type="joint_position_limit",
                joint_indices=pos_violation_indices,
                severity="critical",
                values=joint_pos[pos_violation_indices],
                limits=np.concatenate([
                    self.limits.joint_position_min[pos_violation_indices],
                    self.limits.joint_position_max[pos_violation_indices]
                ]),
                timestamp=time.time()
            )
            return False, violation

        # Check velocity limits
        vel_violations = np.abs(joint_vel) > self.limits.joint_velocity_max
        if np.any(vel_violations):
            vel_violation_indices = np.where(vel_violations)[0].tolist()
            violation = SafetyViolation(
                violation_type="joint_velocity_limit",
                joint_indices=vel_violation_indices,
                severity="warning",
                values=joint_vel[vel_violation_indices],
                limits=self.limits.joint_velocity_max[vel_violation_indices],
                timestamp=time.time()
            )
            return False, violation

        # Predict future position based on action
        predicted_pos = joint_pos + joint_vel * self.prediction_horizon

        future_min_violations = predicted_pos < self.limits.joint_position_min
        future_max_violations = predicted_pos > self.limits.joint_position_max

        if np.any(future_min_violations) or np.any(future_max_violations):
            future_violation_indices = []
            if np.any(future_min_violations):
                future_violation_indices.extend(np.where(future_min_violations)[0].tolist())
            if np.any(future_max_violations):
                future_violation_indices.extend(np.where(future_max_violations)[0].tolist())

            violation = SafetyViolation(
                violation_type="predicted_position_limit",
                joint_indices=future_violation_indices,
                severity="warning",
                values=predicted_pos[future_violation_indices],
                limits=np.concatenate([
                    self.limits.joint_position_min[future_violation_indices],
                    self.limits.joint_position_max[future_violation_indices]
                ]),
                timestamp=time.time()
            )
            return False, violation

        return True, None

    def correct_action(self, action: np.ndarray, violation: SafetyViolation) -> np.ndarray:
        """Correct action by clamping to safe limits."""
        corrected_action = action.copy()

        if violation.violation_type == "joint_position_limit":
            # Reduce action magnitude for violating joints
            for idx in violation.joint_indices:
                corrected_action[idx] *= 0.1  # Reduce action to 10%

        elif violation.violation_type == "joint_velocity_limit":
            # Clamp action for high velocity joints
            for idx in violation.joint_indices:
                corrected_action[idx] *= 0.5  # Reduce action to 50%

        elif violation.violation_type == "predicted_position_limit":
            # Reduce action for joints approaching limits
            for idx in violation.joint_indices:
                corrected_action[idx] *= 0.3  # Reduce action to 30%

        return corrected_action


class ActionRateCheck(SafetyCheck):
    """Action rate limiting safety check."""

    def __init__(self, rate_limit: float):
        """
        Initialize action rate checker.

        Args:
            rate_limit: Maximum allowed change in action per timestep
        """
        self.rate_limit = rate_limit

    def check(self, action: np.ndarray, observation: Any,
             previous_action: Optional[np.ndarray] = None) -> Tuple[bool, Optional[SafetyViolation]]:
        """Check action rate limits."""

        if previous_action is None:
            return True, None

        action_diff = np.abs(action - previous_action)
        max_diff = np.max(action_diff)

        if max_diff > self.rate_limit:
            violation_indices = np.where(action_diff > self.rate_limit)[0].tolist()
            violation = SafetyViolation(
                violation_type="action_rate_limit",
                joint_indices=violation_indices,
                severity="warning",
                values=action_diff[violation_indices],
                limits=np.full(len(violation_indices), self.rate_limit),
                timestamp=time.time()
            )
            return False, violation

        return True, None

    def correct_action(self, action: np.ndarray, violation: SafetyViolation) -> np.ndarray:
        """Correct action by rate limiting."""
        if violation.violation_type != "action_rate_limit":
            return action

        # Implementation would need previous_action - this is a simplified version
        corrected_action = np.clip(action, -self.rate_limit, self.rate_limit)
        return corrected_action


class TorqueLimitCheck(SafetyCheck):
    """Joint torque limits safety check."""

    def __init__(self, limits: SafetyLimits):
        """Initialize torque limit checker."""
        self.limits = limits

    def check(self, action: np.ndarray, observation: Any,
             previous_action: Optional[np.ndarray] = None) -> Tuple[bool, Optional[SafetyViolation]]:
        """Check torque limits (assuming action represents torque commands)."""

        torque_violations = np.abs(action) > self.limits.joint_torque_max

        if np.any(torque_violations):
            violation_indices = np.where(torque_violations)[0].tolist()
            violation = SafetyViolation(
                violation_type="joint_torque_limit",
                joint_indices=violation_indices,
                severity="critical",
                values=action[violation_indices],
                limits=self.limits.joint_torque_max[violation_indices],
                timestamp=time.time()
            )
            return False, violation

        return True, None

    def correct_action(self, action: np.ndarray, violation: SafetyViolation) -> np.ndarray:
        """Correct action by clamping torques."""
        corrected_action = np.clip(action, -self.limits.joint_torque_max,
                                 self.limits.joint_torque_max)
        return corrected_action


class WorkspaceCheck(SafetyCheck):
    """End-effector workspace safety check."""

    def __init__(self, workspace_limits: Dict[str, Any]):
        """
        Initialize workspace checker.

        Args:
            workspace_limits: Dictionary containing workspace boundaries
        """
        self.workspace_limits = workspace_limits

    def check(self, action: np.ndarray, observation: Any,
             previous_action: Optional[np.ndarray] = None) -> Tuple[bool, Optional[SafetyViolation]]:
        """Check end-effector workspace limits."""

        # PLACEHOLDER: This would require forward kinematics
        # For now, return safe
        return True, None

    def correct_action(self, action: np.ndarray, violation: SafetyViolation) -> np.ndarray:
        """Correct action to keep within workspace."""
        # PLACEHOLDER: Implementation would depend on inverse kinematics
        return action


class SafetyFilter:
    """
    Comprehensive safety filter for G1 robot actions.

    Applies multiple safety checks and corrections to ensure safe operation.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize safety filter.

        Args:
            config: Safety configuration dictionary
        """
        self.config = config

        # Create safety limits
        self.limits = self._create_safety_limits(config)

        # Initialize safety checks
        self.safety_checks: List[SafetyCheck] = []

        if config.get('enable_joint_limits', True):
            self.safety_checks.append(JointLimitsCheck(self.limits))

        if config.get('enable_action_rate_limit', True):
            rate_limit = config.get('action_rate_limit', 0.1)
            self.safety_checks.append(ActionRateCheck(rate_limit))

        if config.get('enable_torque_limits', True):
            self.safety_checks.append(TorqueLimitCheck(self.limits))

        if config.get('enable_workspace_check', False):
            workspace_limits = config.get('workspace_limits', {})
            self.safety_checks.append(WorkspaceCheck(workspace_limits))

        # Safety state
        self.previous_action: Optional[np.ndarray] = None
        self.violation_history: List[SafetyViolation] = []
        self.max_history_length = config.get('max_violation_history', 100)

        # Emergency stop conditions
        self.max_critical_violations = config.get('max_critical_violations', 3)
        self.critical_violation_window = config.get('critical_violation_window', 1.0)  # seconds

        logger.info("Initialized SafetyFilter")

    def _create_safety_limits(self, config: Dict[str, Any]) -> SafetyLimits:
        """Create safety limits from configuration."""

        # Default G1 joint limits (22 DOF)
        default_pos_min = np.array([-2.0] * 22)
        default_pos_max = np.array([2.0] * 22)
        default_vel_max = np.array([10.0] * 22)
        default_torque_max = np.array([100.0] * 22)

        limits = SafetyLimits(
            joint_position_min=np.array(config.get('joint_position_limits', {}).get('min', default_pos_min)),
            joint_position_max=np.array(config.get('joint_position_limits', {}).get('max', default_pos_max)),
            joint_velocity_max=np.array(config.get('joint_velocity_limits', {}).get('max', default_vel_max)),
            joint_torque_max=np.array(config.get('joint_torque_limits', {}).get('max', default_torque_max)),
            action_rate_limit=config.get('action_rate_limit', 0.1),
            emergency_action=np.zeros(22)
        )

        return limits

    def filter_action(self, action: np.ndarray, observation: Any,
                     inference_info: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply safety filter to action.

        Args:
            action: Raw action from policy
            observation: Current robot observation
            inference_info: Information from inference engine

        Returns:
            Tuple of (filtered_action, safety_info)
        """
        safety_info = {
            'violation': False,
            'violation_type': None,
            'correction_applied': False,
            'emergency_stop': False
        }

        # Check for inference errors/timeouts
        if inference_info.get('error') or inference_info.get('timeout'):
            logger.warning("Inference error/timeout detected, using emergency action")
            safety_info.update({
                'violation': True,
                'violation_type': 'inference_failure',
                'emergency_stop': True
            })
            return self.limits.emergency_action.copy(), safety_info

        filtered_action = action.copy()
        violations_detected = []

        # Apply all safety checks
        for safety_check in self.safety_checks:
            try:
                is_safe, violation = safety_check.check(
                    filtered_action, observation, self.previous_action
                )

                if not is_safe and violation is not None:
                    violations_detected.append(violation)

                    # Apply correction
                    corrected_action = safety_check.correct_action(filtered_action, violation)
                    filtered_action = corrected_action

                    safety_info.update({
                        'violation': True,
                        'violation_type': violation.violation_type,
                        'correction_applied': True
                    })

                    # Log violation
                    logger.warning(f"Safety violation: {violation.violation_type} "
                                 f"on joints {violation.joint_indices}")

                    # Store violation in history
                    self._record_violation(violation)

            except Exception as e:
                logger.error(f"Safety check failed: {e}")
                # Use emergency action if safety check fails
                filtered_action = self.limits.emergency_action.copy()
                safety_info.update({
                    'violation': True,
                    'violation_type': 'safety_check_failure',
                    'emergency_stop': True
                })
                break

        # Check for emergency stop conditions
        if self._should_emergency_stop():
            logger.error("Emergency stop condition triggered")
            filtered_action = self.limits.emergency_action.copy()
            safety_info.update({
                'violation': True,
                'violation_type': 'emergency_stop_condition',
                'emergency_stop': True
            })

        # Update previous action
        self.previous_action = filtered_action.copy()

        # Add violation details to safety info
        if violations_detected:
            safety_info['violations'] = [
                {
                    'type': v.violation_type,
                    'joints': v.joint_indices,
                    'severity': v.severity
                }
                for v in violations_detected
            ]

        return filtered_action, safety_info

    def _record_violation(self, violation: SafetyViolation):
        """Record safety violation in history."""
        self.violation_history.append(violation)

        # Trim history if too long
        if len(self.violation_history) > self.max_history_length:
            self.violation_history.pop(0)

    def _should_emergency_stop(self) -> bool:
        """Check if emergency stop conditions are met."""
        current_time = time.time()

        # Count critical violations in recent window
        critical_violations = [
            v for v in self.violation_history
            if (v.severity == 'critical' and
                current_time - v.timestamp <= self.critical_violation_window)
        ]

        return len(critical_violations) >= self.max_critical_violations

    def reset(self):
        """Reset safety filter state."""
        self.previous_action = None
        self.violation_history.clear()
        logger.info("Safety filter reset")

    def get_violation_statistics(self) -> Dict[str, Any]:
        """Get safety violation statistics."""
        if not self.violation_history:
            return {}

        violation_types = {}
        critical_count = 0
        warning_count = 0

        for violation in self.violation_history:
            vtype = violation.violation_type
            if vtype not in violation_types:
                violation_types[vtype] = 0
            violation_types[vtype] += 1

            if violation.severity == 'critical':
                critical_count += 1
            elif violation.severity == 'warning':
                warning_count += 1

        return {
            'total_violations': len(self.violation_history),
            'critical_violations': critical_count,
            'warning_violations': warning_count,
            'violation_types': violation_types,
            'recent_critical_violations': len([
                v for v in self.violation_history
                if (v.severity == 'critical' and
                    time.time() - v.timestamp <= self.critical_violation_window)
            ])
        }

    def health_check(self) -> bool:
        """Perform health check of safety filter."""
        try:
            # Check that safety checks are initialized
            if not self.safety_checks:
                logger.error("No safety checks configured")
                return False

            # Check for recent critical violations
            if self._should_emergency_stop():
                logger.error("Too many recent critical violations")
                return False

            logger.info("Safety filter health check passed")
            return True

        except Exception as e:
            logger.error(f"Safety filter health check failed: {e}")
            return False


def create_safety_config_template() -> Dict[str, Any]:
    """Create template safety configuration."""
    return {
        'enable_joint_limits': True,
        'enable_action_rate_limit': True,
        'enable_torque_limits': True,
        'enable_workspace_check': False,

        'joint_position_limits': {
            'min': [-2.0] * 22,  # G1 joint limits (radians)
            'max': [2.0] * 22
        },
        'joint_velocity_limits': {
            'max': [10.0] * 22  # rad/s
        },
        'joint_torque_limits': {
            'max': [100.0] * 22  # Nm
        },
        'action_rate_limit': 0.1,  # max change per timestep

        'workspace_limits': {
            'x_range': [-1.0, 1.0],
            'y_range': [-1.0, 1.0],
            'z_range': [0.0, 2.0]
        },

        'max_critical_violations': 3,
        'critical_violation_window': 1.0,  # seconds
        'max_violation_history': 100
    }