"""
Test suite definitions and execution for G1 policy evaluation.

This module defines the various test scenarios including stand, walk,
turn, stop, recovery, and crouch tests with their specific initialization
and success criteria.
"""

import numpy as np
import mujoco
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class TestSuite(ABC):
    """
    Abstract base class for test suites.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize test suite.

        Args:
            config: Test suite configuration
        """
        self.config = config
        self.name = self.__class__.__name__.replace('TestSuite', '').lower()
        self.duration = config.get('duration', 30.0)
        self.success_criteria = config.get('success_criteria', [])

    @abstractmethod
    def initialize_episode(self, env: Any) -> Dict[str, Any]:
        """
        Initialize environment for this test suite.

        Args:
            env: MuJoCo environment

        Returns:
            Initialization info
        """
        pass

    @abstractmethod
    def update_episode(self, env: Any, step: int, info: Dict[str, Any]) -> bool:
        """
        Update episode state and check for early termination.

        Args:
            env: MuJoCo environment
            step: Current step number
            info: Step info dictionary

        Returns:
            True if episode should terminate
        """
        pass

    def get_success_criteria(self) -> List[str]:
        """Get success criteria for this test suite."""
        return self.success_criteria

    def get_description(self) -> str:
        """Get test suite description."""
        return self.config.get('description', f"{self.name} test suite")


class StandTestSuite(TestSuite):
    """
    Standing balance test - maintain upright position.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.target_height = config.get('target_height', 1.0)
        self.position_tolerance = config.get('position_tolerance', 0.1)
        self.initial_position = None

    def initialize_episode(self, env: Any) -> Dict[str, Any]:
        """Initialize standing test."""
        # Reset to standing position
        if hasattr(env, 'set_standing_pose'):
            env.set_standing_pose()
        else:
            # Default standing initialization
            if hasattr(env.data, 'qpos'):
                # Set root position
                env.data.qpos[2] = self.target_height  # Z height
                # Set joint positions to neutral standing
                if len(env.data.qpos) > 7:  # Has joint angles
                    env.data.qpos[7:] = 0.0  # Neutral joint angles

        # Store initial position
        self.initial_position = env.data.qpos[:3].copy()

        return {
            'test_type': 'stand',
            'target_height': self.target_height,
            'initial_position': self.initial_position
        }

    def update_episode(self, env: Any, step: int, info: Dict[str, Any]) -> bool:
        """Update standing test."""
        # Check if robot moved too far from initial position
        if self.initial_position is not None:
            current_position = env.data.qpos[:3]
            displacement = np.linalg.norm(current_position[:2] - self.initial_position[:2])

            # Add displacement info
            info['position_displacement'] = displacement

            # Terminate if moved too far (optional)
            max_displacement = self.config.get('max_displacement', float('inf'))
            if displacement > max_displacement:
                info['terminated_reason'] = 'excessive_displacement'
                return True

        return False  # Continue episode


class WalkTestSuite(TestSuite):
    """
    Walking test with speed commands.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.target_speed = config.get('target_speed', 1.0)
        self.speed_variations = config.get('speed_variations', [1.0])
        self.current_speed_target = self.target_speed
        self.speed_change_interval = config.get('speed_change_interval', 5.0)  # seconds
        self.last_speed_change = 0

    def initialize_episode(self, env: Any) -> Dict[str, Any]:
        """Initialize walking test."""
        # Set initial walking pose
        if hasattr(env, 'set_walking_pose'):
            env.set_walking_pose()

        # Randomize target speed if variations provided
        if self.speed_variations:
            self.current_speed_target = np.random.choice(self.speed_variations)

        return {
            'test_type': 'walk',
            'target_speed': self.current_speed_target,
            'speed_variations': self.speed_variations
        }

    def update_episode(self, env: Any, step: int, info: Dict[str, Any]) -> bool:
        """Update walking test."""
        dt = getattr(env, 'dt', 0.02)
        current_time = step * dt

        # Change speed target periodically
        if (current_time - self.last_speed_change) >= self.speed_change_interval:
            if self.speed_variations:
                self.current_speed_target = np.random.choice(self.speed_variations)
                self.last_speed_change = current_time

        # Calculate current speed
        if hasattr(env.data, 'qvel'):
            current_velocity = env.data.qvel[:3]
            current_speed = np.linalg.norm(current_velocity[:2])  # Horizontal speed
        else:
            current_speed = 0.0

        # Add speed tracking info
        info['target_speed'] = self.current_speed_target
        info['current_speed'] = current_speed
        info['speed_error'] = abs(current_speed - self.current_speed_target)
        info['command_tracking_error'] = info['speed_error']

        return False  # Continue episode


class TurnTestSuite(TestSuite):
    """
    Turning test with angular velocity commands.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.target_turn_rate = config.get('target_turn_rate', 0.5)
        self.turn_variations = config.get('turn_variations', [0.5])
        self.current_turn_target = self.target_turn_rate
        self.turn_change_interval = config.get('turn_change_interval', 3.0)  # seconds
        self.last_turn_change = 0

    def initialize_episode(self, env: Any) -> Dict[str, Any]:
        """Initialize turning test."""
        # Set initial pose
        if hasattr(env, 'set_walking_pose'):
            env.set_walking_pose()

        # Randomize turn rate
        if self.turn_variations:
            self.current_turn_target = np.random.choice(self.turn_variations)

        return {
            'test_type': 'turn',
            'target_turn_rate': self.current_turn_target,
            'turn_variations': self.turn_variations
        }

    def update_episode(self, env: Any, step: int, info: Dict[str, Any]) -> bool:
        """Update turning test."""
        dt = getattr(env, 'dt', 0.02)
        current_time = step * dt

        # Change turn target periodically
        if (current_time - self.last_turn_change) >= self.turn_change_interval:
            if self.turn_variations:
                self.current_turn_target = np.random.choice(self.turn_variations)
                self.last_turn_change = current_time

        # Calculate current angular velocity
        if hasattr(env.data, 'qvel'):
            current_angular_vel = env.data.qvel[5]  # Z angular velocity
        else:
            current_angular_vel = 0.0

        # Add turn tracking info
        info['target_turn_rate'] = self.current_turn_target
        info['current_turn_rate'] = current_angular_vel
        info['turn_rate_error'] = abs(current_angular_vel - self.current_turn_target)
        info['command_tracking_error'] = info['turn_rate_error']

        return False  # Continue episode


class StopTestSuite(TestSuite):
    """
    Stopping test - decelerate to complete stop.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.initial_speed = config.get('initial_speed', 1.0)
        self.deceleration_time = config.get('deceleration_time', 5.0)
        self.stop_threshold = config.get('stop_threshold', 0.1)
        self.deceleration_started = False
        self.start_deceleration_step = 0

    def initialize_episode(self, env: Any) -> Dict[str, Any]:
        """Initialize stopping test."""
        # Set initial walking pose with forward velocity
        if hasattr(env, 'set_walking_pose'):
            env.set_walking_pose()

        # Set initial forward velocity
        if hasattr(env.data, 'qvel'):
            env.data.qvel[0] = self.initial_speed  # Forward velocity

        return {
            'test_type': 'stop',
            'initial_speed': self.initial_speed,
            'deceleration_time': self.deceleration_time
        }

    def update_episode(self, env: Any, step: int, info: Dict[str, Any]) -> bool:
        """Update stopping test."""
        dt = getattr(env, 'dt', 0.02)
        current_time = step * dt

        # Start deceleration after initial period
        if not self.deceleration_started and current_time > 2.0:  # Wait 2 seconds
            self.deceleration_started = True
            self.start_deceleration_step = step

        # Calculate target speed based on deceleration profile
        if self.deceleration_started:
            elapsed_decel = (step - self.start_deceleration_step) * dt
            decel_progress = min(elapsed_decel / self.deceleration_time, 1.0)
            target_speed = self.initial_speed * (1.0 - decel_progress)
        else:
            target_speed = self.initial_speed

        # Calculate current speed
        if hasattr(env.data, 'qvel'):
            current_velocity = env.data.qvel[:3]
            current_speed = np.linalg.norm(current_velocity[:2])
        else:
            current_speed = 0.0

        # Add stopping info
        info['target_speed'] = target_speed
        info['current_speed'] = current_speed
        info['speed_error'] = abs(current_speed - target_speed)
        info['command_tracking_error'] = info['speed_error']
        info['deceleration_phase'] = self.deceleration_started

        # Check if successfully stopped
        if self.deceleration_started and target_speed < 0.01:
            info['stop_complete'] = current_speed < self.stop_threshold

        return False  # Continue episode


class RecoveryFromPushTestSuite(TestSuite):
    """
    Recovery test - maintain balance after external disturbance.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.push_timing = config.get('push_timing', 2.0)  # seconds
        self.push_force = config.get('push_force', 150.0)  # Newtons
        self.push_direction = config.get('push_direction', 0.0)  # radians
        self.push_duration = config.get('push_duration', 0.2)  # seconds
        self.push_applied = False
        self.push_start_step = -1
        self.push_end_step = -1

    def initialize_episode(self, env: Any) -> Dict[str, Any]:
        """Initialize recovery test."""
        # Set stable standing pose
        if hasattr(env, 'set_standing_pose'):
            env.set_standing_pose()

        self.push_applied = False
        self.push_start_step = -1
        self.push_end_step = -1

        return {
            'test_type': 'recovery_from_push',
            'push_timing': self.push_timing,
            'push_force': self.push_force,
            'push_direction': self.push_direction
        }

    def update_episode(self, env: Any, step: int, info: Dict[str, Any]) -> bool:
        """Update recovery test."""
        dt = getattr(env, 'dt', 0.02)
        current_time = step * dt

        # Apply push at specified time
        if not self.push_applied and current_time >= self.push_timing:
            self._apply_push(env)
            self.push_applied = True
            self.push_start_step = step
            self.push_end_step = step + int(self.push_duration / dt)

            info['push_applied'] = True
            info['push_start_time'] = current_time

        # Remove push after duration
        if (self.push_applied and step >= self.push_end_step and
            hasattr(env, 'data') and hasattr(env.data, 'xfrc_applied')):
            # Clear external forces
            env.data.xfrc_applied[:] = 0
            info['push_ended'] = True

        # Track recovery phase
        if self.push_applied:
            recovery_time = current_time - (self.push_start_step * dt)
            info['recovery_time'] = recovery_time
            info['push_phase'] = step < self.push_end_step

        return False  # Continue episode

    def _apply_push(self, env: Any):
        """Apply external push force to robot."""
        if not hasattr(env, 'data') or not hasattr(env.data, 'xfrc_applied'):
            logger.warning("Environment does not support external forces")
            return

        try:
            # Find robot root body
            root_body_id = self._find_root_body_id(env)
            if root_body_id < 0:
                logger.warning("Could not find robot root body")
                return

            # Calculate force vector
            force_x = self.push_force * np.cos(self.push_direction)
            force_y = self.push_force * np.sin(self.push_direction)

            # Apply external force
            env.data.xfrc_applied[root_body_id][:3] = [force_x, force_y, 0.0]

            logger.info(f"Applied push force: {self.push_force}N at {self.push_direction:.2f}rad")

        except Exception as e:
            logger.error(f"Failed to apply push: {e}")

    def _find_root_body_id(self, env: Any) -> int:
        """Find the root body ID for applying forces."""
        if not hasattr(env, 'model'):
            return -1

        # Try common root body names
        for name in ['base_link', 'trunk', 'torso', 'base', 'pelvis']:
            body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id >= 0:
                return body_id

        # Fallback to body 1 (often the root)
        return 1 if env.model.nbody > 1 else -1


class CrouchTestSuite(TestSuite):
    """
    Crouching test - lower body position and maintain.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.target_pose = config.get('target_pose', 'crouched')
        self.height_reduction = config.get('height_reduction', 0.3)
        self.crouch_duration = config.get('crouch_duration', 10.0)
        self.initial_height = None
        self.target_height = None
        self.crouch_started = False
        self.crouch_start_step = -1

    def initialize_episode(self, env: Any) -> Dict[str, Any]:
        """Initialize crouching test."""
        # Start in standing position
        if hasattr(env, 'set_standing_pose'):
            env.set_standing_pose()

        # Record initial height
        if hasattr(env.data, 'qpos'):
            self.initial_height = env.data.qpos[2]
            self.target_height = self.initial_height - self.height_reduction

        self.crouch_started = False
        self.crouch_start_step = -1

        return {
            'test_type': 'crouch',
            'initial_height': self.initial_height,
            'target_height': self.target_height,
            'height_reduction': self.height_reduction
        }

    def update_episode(self, env: Any, step: int, info: Dict[str, Any]) -> bool:
        """Update crouching test."""
        dt = getattr(env, 'dt', 0.02)
        current_time = step * dt

        # Start crouching after initial stabilization
        if not self.crouch_started and current_time > 3.0:
            self.crouch_started = True
            self.crouch_start_step = step

        # Get current height
        current_height = env.data.qpos[2] if hasattr(env.data, 'qpos') else 0.0

        # Calculate target height based on phase
        if self.crouch_started:
            crouch_time = (step - self.crouch_start_step) * dt
            if crouch_time < 3.0:  # Transition period
                progress = crouch_time / 3.0
                target_height = self.initial_height - (self.height_reduction * progress)
            else:  # Maintain crouched position
                target_height = self.target_height
        else:
            target_height = self.initial_height

        # Add crouching info
        info['target_height'] = target_height
        info['current_height'] = current_height
        info['height_error'] = abs(current_height - target_height)
        info['crouch_phase'] = self.crouch_started

        if self.crouch_started:
            height_reduction_achieved = self.initial_height - current_height
            info['height_reduction_achieved'] = height_reduction_achieved
            info['crouch_complete'] = height_reduction_achieved >= (self.height_reduction * 0.8)

        return False  # Continue episode


class TestSuiteRunner:
    """
    Runner for managing test suites during evaluation.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize test suite runner.

        Args:
            config: Test suites configuration
        """
        self.config = config

        # Available test suites
        self.test_suite_classes = {
            'stand': StandTestSuite,
            'walk': WalkTestSuite,
            'turn': TurnTestSuite,
            'stop': StopTestSuite,
            'recovery_from_push': RecoveryFromPushTestSuite,
            'crouch': CrouchTestSuite
        }

        # Loaded test suite instances
        self.test_suites = {}
        self._load_test_suites()

        logger.info(f"Initialized TestSuiteRunner with {len(self.test_suites)} test suites")

    def _load_test_suites(self):
        """Load test suite configurations."""
        suites_config = self.config.get('suites', {})

        for suite_name, suite_class in self.test_suite_classes.items():
            if suite_name in suites_config:
                suite_config = suites_config[suite_name]
            else:
                # Use default configuration
                suite_config = {'description': f'{suite_name} test suite'}

            self.test_suites[suite_name] = suite_class(suite_config)

    def get_suite_config(self, test_suite: str) -> Optional[Dict[str, Any]]:
        """
        Get configuration for a test suite.

        Args:
            test_suite: Test suite name

        Returns:
            Test suite configuration or None if not found
        """
        if test_suite in self.test_suites:
            return self.test_suites[test_suite].config
        return None

    def initialize_episode(self, env: Any, test_suite: str, suite_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Initialize episode for specific test suite.

        Args:
            env: Environment
            test_suite: Test suite name
            suite_config: Suite configuration

        Returns:
            Initialization info
        """
        if test_suite in self.test_suites:
            return self.test_suites[test_suite].initialize_episode(env)
        else:
            logger.warning(f"Unknown test suite: {test_suite}")
            return {'test_type': test_suite}

    def update_episode(self, env: Any, test_suite: str, suite_config: Dict[str, Any],
                      step: int, info: Dict[str, Any], done: bool) -> bool:
        """
        Update episode for specific test suite.

        Args:
            env: Environment
            test_suite: Test suite name
            suite_config: Suite configuration
            step: Current step
            info: Step info
            done: Current done flag

        Returns:
            Updated done flag
        """
        if test_suite in self.test_suites:
            suite_done = self.test_suites[test_suite].update_episode(env, step, info)
            return done or suite_done
        else:
            return done

    def get_available_suites(self) -> List[str]:
        """Get list of available test suites."""
        return list(self.test_suites.keys())

    def get_suite_description(self, test_suite: str) -> str:
        """Get description for a test suite."""
        if test_suite in self.test_suites:
            return self.test_suites[test_suite].get_description()
        return f"Unknown test suite: {test_suite}"

    def get_success_criteria(self, test_suite: str) -> List[str]:
        """Get success criteria for a test suite."""
        if test_suite in self.test_suites:
            return self.test_suites[test_suite].get_success_criteria()
        return []

    def validate_suite_config(self, test_suite: str) -> bool:
        """Validate configuration for a test suite."""
        if test_suite not in self.test_suites:
            logger.error(f"Test suite '{test_suite}' not available")
            return False

        # Additional validation could be added here
        return True