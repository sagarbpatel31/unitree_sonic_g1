"""
Command management for Unitree G1 environment.

This module implements command-conditioned behaviors for the G1 robot
including walking, turning, and stopping commands.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class CommandType(Enum):
    """Types of commands for the robot."""
    STOP = "stop"
    WALK_FORWARD = "walk_forward"
    WALK_BACKWARD = "walk_backward"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    STRAFE_LEFT = "strafe_left"
    STRAFE_RIGHT = "strafe_right"
    CUSTOM = "custom"


@dataclass
class CommandConfig:
    """Configuration for command generation."""
    # Enable command conditioning
    enabled: bool = True

    # Velocity ranges for different commands
    forward_vel_range: tuple = (0.0, 2.0)  # m/s
    backward_vel_range: tuple = (-2.0, 0.0)  # m/s
    lateral_vel_range: tuple = (-1.0, 1.0)  # m/s
    yaw_rate_range: tuple = (-2.0, 2.0)  # rad/s

    # Command probabilities
    command_probabilities: dict = None  # Will be set in __post_init__

    # Command duration
    command_duration_range: tuple = (2.0, 10.0)  # seconds
    resample_interval: int = 100  # steps

    # Smoothing
    smooth_commands: bool = True
    smoothing_factor: float = 0.9

    def __post_init__(self):
        if self.command_probabilities is None:
            self.command_probabilities = {
                CommandType.STOP: 0.1,
                CommandType.WALK_FORWARD: 0.4,
                CommandType.WALK_BACKWARD: 0.1,
                CommandType.TURN_LEFT: 0.1,
                CommandType.TURN_RIGHT: 0.1,
                CommandType.STRAFE_LEFT: 0.1,
                CommandType.STRAFE_RIGHT: 0.1,
            }


class CommandManager:
    """
    Manages command generation and conditioning for the G1 environment.

    This class handles:
    - Random command sampling for diverse behavior training
    - Command interpolation and smoothing
    - Velocity command mapping to robot actions
    - Command curriculum for progressive training
    """

    def __init__(
        self,
        model,  # mujoco.MjModel
        data,   # mujoco.MjData
        config: Dict[str, Any]
    ):
        """
        Initialize command manager.

        Args:
            model: MuJoCo model (not used directly but kept for consistency)
            data: MuJoCo data (not used directly but kept for consistency)
            config: Configuration dictionary
        """
        self.model = model
        self.data = data

        # Parse configuration
        self.config = CommandConfig(**config)

        # Command state
        self.current_command = None
        self.target_command = None
        self.command_duration = 0
        self.steps_since_command = 0
        self.smoothed_command = None

        # Command history for analysis
        self.command_history = []
        self.max_history = 1000

        print(f"CommandManager initialized with {len(self.config.command_probabilities)} command types")

    @property
    def enabled(self) -> bool:
        """Check if command conditioning is enabled."""
        return self.config.enabled

    def sample_command(self) -> Dict[str, Any]:
        """
        Sample a new random command.

        Returns:
            Dictionary containing command information
        """
        if not self.enabled:
            return self._create_zero_command()

        # Sample command type
        command_type = self._sample_command_type()

        # Generate command parameters
        command = self._generate_command(command_type)

        # Set command duration
        self.command_duration = np.random.uniform(*self.config.command_duration_range)
        self.steps_since_command = 0

        # Initialize smoothing
        if self.smoothed_command is None:
            self.smoothed_command = command.copy()

        # Store in history
        self._add_to_history(command)

        self.current_command = command
        self.target_command = command.copy()

        return command

    def update_command(self, dt: float = 0.02) -> Dict[str, Any]:
        """
        Update current command with smoothing.

        Args:
            dt: Time step in seconds

        Returns:
            Updated command
        """
        if not self.enabled or self.current_command is None:
            return self._create_zero_command()

        self.steps_since_command += 1

        # Check if we should resample command
        if self.steps_since_command >= self.config.resample_interval:
            time_elapsed = self.steps_since_command * dt
            if time_elapsed >= self.command_duration:
                return self.sample_command()

        # Apply smoothing if enabled
        if self.config.smooth_commands and self.smoothed_command is not None:
            alpha = self.config.smoothing_factor
            for key in ['forward_vel', 'lateral_vel', 'yaw_rate']:
                if key in self.smoothed_command and key in self.target_command:
                    self.smoothed_command[key] = (
                        alpha * self.smoothed_command[key] +
                        (1 - alpha) * self.target_command[key]
                    )

            return self.smoothed_command.copy()

        return self.current_command

    def set_command(self, command: Dict[str, Any]):
        """
        Set a specific command manually.

        Args:
            command: Command specification
        """
        self.current_command = command
        self.target_command = command.copy()
        self.command_duration = float('inf')  # Manual commands don't expire
        self.steps_since_command = 0

        if self.config.smooth_commands:
            self.smoothed_command = command.copy()

        self._add_to_history(command)

    def get_current_command(self) -> Optional[Dict[str, Any]]:
        """Get the current command."""
        return self.current_command

    def reset(self):
        """Reset command manager for new episode."""
        self.current_command = None
        self.target_command = None
        self.smoothed_command = None
        self.command_duration = 0
        self.steps_since_command = 0

    def _sample_command_type(self) -> CommandType:
        """Sample command type based on probabilities."""
        types = list(self.config.command_probabilities.keys())
        probs = list(self.config.command_probabilities.values())

        # Normalize probabilities
        probs = np.array(probs)
        probs = probs / probs.sum()

        return np.random.choice(types, p=probs)

    def _generate_command(self, command_type: CommandType) -> Dict[str, Any]:
        """
        Generate command parameters for given command type.

        Args:
            command_type: Type of command to generate

        Returns:
            Command dictionary
        """
        if command_type == CommandType.STOP:
            return self._create_stop_command()

        elif command_type == CommandType.WALK_FORWARD:
            return self._create_forward_command()

        elif command_type == CommandType.WALK_BACKWARD:
            return self._create_backward_command()

        elif command_type == CommandType.TURN_LEFT:
            return self._create_turn_left_command()

        elif command_type == CommandType.TURN_RIGHT:
            return self._create_turn_right_command()

        elif command_type == CommandType.STRAFE_LEFT:
            return self._create_strafe_left_command()

        elif command_type == CommandType.STRAFE_RIGHT:
            return self._create_strafe_right_command()

        else:
            return self._create_zero_command()

    def _create_stop_command(self) -> Dict[str, Any]:
        """Create stop command."""
        return {
            'type': CommandType.STOP.value,
            'forward_vel': 0.0,
            'lateral_vel': 0.0,
            'yaw_rate': 0.0,
            'description': "Stop in place"
        }

    def _create_forward_command(self) -> Dict[str, Any]:
        """Create forward walking command."""
        forward_vel = np.random.uniform(*self.config.forward_vel_range)

        return {
            'type': CommandType.WALK_FORWARD.value,
            'forward_vel': forward_vel,
            'lateral_vel': 0.0,
            'yaw_rate': 0.0,
            'description': f"Walk forward at {forward_vel:.2f} m/s"
        }

    def _create_backward_command(self) -> Dict[str, Any]:
        """Create backward walking command."""
        backward_vel = np.random.uniform(*self.config.backward_vel_range)

        return {
            'type': CommandType.WALK_BACKWARD.value,
            'forward_vel': backward_vel,
            'lateral_vel': 0.0,
            'yaw_rate': 0.0,
            'description': f"Walk backward at {abs(backward_vel):.2f} m/s"
        }

    def _create_turn_left_command(self) -> Dict[str, Any]:
        """Create left turn command."""
        yaw_rate = np.random.uniform(0.5, self.config.yaw_rate_range[1])
        forward_vel = np.random.uniform(0.0, 1.0)  # Slow forward while turning

        return {
            'type': CommandType.TURN_LEFT.value,
            'forward_vel': forward_vel,
            'lateral_vel': 0.0,
            'yaw_rate': yaw_rate,
            'description': f"Turn left at {yaw_rate:.2f} rad/s"
        }

    def _create_turn_right_command(self) -> Dict[str, Any]:
        """Create right turn command."""
        yaw_rate = np.random.uniform(self.config.yaw_rate_range[0], -0.5)
        forward_vel = np.random.uniform(0.0, 1.0)  # Slow forward while turning

        return {
            'type': CommandType.TURN_RIGHT.value,
            'forward_vel': forward_vel,
            'lateral_vel': 0.0,
            'yaw_rate': yaw_rate,
            'description': f"Turn right at {abs(yaw_rate):.2f} rad/s"
        }

    def _create_strafe_left_command(self) -> Dict[str, Any]:
        """Create left strafe command."""
        lateral_vel = np.random.uniform(0.2, self.config.lateral_vel_range[1])

        return {
            'type': CommandType.STRAFE_LEFT.value,
            'forward_vel': 0.0,
            'lateral_vel': lateral_vel,
            'yaw_rate': 0.0,
            'description': f"Strafe left at {lateral_vel:.2f} m/s"
        }

    def _create_strafe_right_command(self) -> Dict[str, Any]:
        """Create right strafe command."""
        lateral_vel = np.random.uniform(self.config.lateral_vel_range[0], -0.2)

        return {
            'type': CommandType.STRAFE_RIGHT.value,
            'forward_vel': 0.0,
            'lateral_vel': lateral_vel,
            'yaw_rate': 0.0,
            'description': f"Strafe right at {abs(lateral_vel):.2f} m/s"
        }

    def _create_zero_command(self) -> Dict[str, Any]:
        """Create zero velocity command."""
        return {
            'type': 'zero',
            'forward_vel': 0.0,
            'lateral_vel': 0.0,
            'yaw_rate': 0.0,
            'description': "Zero velocity"
        }

    def _add_to_history(self, command: Dict[str, Any]):
        """Add command to history."""
        self.command_history.append(command.copy())

        # Keep history bounded
        if len(self.command_history) > self.max_history:
            self.command_history.pop(0)

    def create_command_curriculum(
        self,
        training_step: int,
        total_steps: int
    ) -> Dict[str, Any]:
        """
        Create curriculum-based command probabilities.

        Args:
            training_step: Current training step
            total_steps: Total training steps

        Returns:
            Curriculum-adjusted command probabilities
        """
        progress = min(1.0, training_step / total_steps)

        # Start with simple commands, gradually add complexity
        if progress < 0.25:
            # Early training: mostly stop and simple forward
            curriculum_probs = {
                CommandType.STOP: 0.3,
                CommandType.WALK_FORWARD: 0.7,
                CommandType.WALK_BACKWARD: 0.0,
                CommandType.TURN_LEFT: 0.0,
                CommandType.TURN_RIGHT: 0.0,
                CommandType.STRAFE_LEFT: 0.0,
                CommandType.STRAFE_RIGHT: 0.0,
            }
        elif progress < 0.5:
            # Mid training: add turns
            curriculum_probs = {
                CommandType.STOP: 0.2,
                CommandType.WALK_FORWARD: 0.5,
                CommandType.WALK_BACKWARD: 0.1,
                CommandType.TURN_LEFT: 0.1,
                CommandType.TURN_RIGHT: 0.1,
                CommandType.STRAFE_LEFT: 0.0,
                CommandType.STRAFE_RIGHT: 0.0,
            }
        elif progress < 0.75:
            # Late training: add strafing
            curriculum_probs = {
                CommandType.STOP: 0.15,
                CommandType.WALK_FORWARD: 0.4,
                CommandType.WALK_BACKWARD: 0.15,
                CommandType.TURN_LEFT: 0.1,
                CommandType.TURN_RIGHT: 0.1,
                CommandType.STRAFE_LEFT: 0.05,
                CommandType.STRAFE_RIGHT: 0.05,
            }
        else:
            # Final training: full complexity
            curriculum_probs = self.config.command_probabilities

        return curriculum_probs

    def set_command_probabilities(self, probabilities: Dict[CommandType, float]):
        """Set command probabilities."""
        # Normalize probabilities
        total = sum(probabilities.values())
        if total > 0:
            normalized = {k: v / total for k, v in probabilities.items()}
            self.config.command_probabilities = normalized

    def get_command_statistics(self) -> Dict[str, Any]:
        """Get statistics about generated commands."""
        if not self.command_history:
            return {}

        # Count command types
        type_counts = {}
        forward_vels = []
        lateral_vels = []
        yaw_rates = []

        for cmd in self.command_history:
            cmd_type = cmd.get('type', 'unknown')
            type_counts[cmd_type] = type_counts.get(cmd_type, 0) + 1

            forward_vels.append(cmd.get('forward_vel', 0.0))
            lateral_vels.append(cmd.get('lateral_vel', 0.0))
            yaw_rates.append(cmd.get('yaw_rate', 0.0))

        return {
            'total_commands': len(self.command_history),
            'type_distribution': type_counts,
            'forward_vel_stats': {
                'mean': np.mean(forward_vels),
                'std': np.std(forward_vels),
                'range': [np.min(forward_vels), np.max(forward_vels)]
            },
            'lateral_vel_stats': {
                'mean': np.mean(lateral_vels),
                'std': np.std(lateral_vels),
                'range': [np.min(lateral_vels), np.max(lateral_vels)]
            },
            'yaw_rate_stats': {
                'mean': np.mean(yaw_rates),
                'std': np.std(yaw_rates),
                'range': [np.min(yaw_rates), np.max(yaw_rates)]
            }
        }

    def visualize_command(self, command: Dict[str, Any]) -> str:
        """
        Create ASCII visualization of command.

        Args:
            command: Command to visualize

        Returns:
            ASCII string representation
        """
        if command is None:
            return "No command"

        forward = command.get('forward_vel', 0.0)
        lateral = command.get('lateral_vel', 0.0)
        yaw = command.get('yaw_rate', 0.0)

        # Create simple arrow representation
        arrow = "○"  # Center

        if forward > 0.1:
            arrow = "↑"
        elif forward < -0.1:
            arrow = "↓"

        if lateral > 0.1:
            arrow = "←" if arrow == "○" else "↖" if forward > 0 else "↙"
        elif lateral < -0.1:
            arrow = "→" if arrow == "○" else "↗" if forward > 0 else "↘"

        if abs(yaw) > 0.1:
            if yaw > 0:
                arrow += " ↻"  # Counter-clockwise
            else:
                arrow += " ↺"  # Clockwise

        desc = command.get('description', 'Custom command')
        return f"{arrow} {desc}"