"""
Domain randomization for Unitree G1 environment.

This module implements comprehensive domain randomization including
physics parameters, sensor characteristics, and external disturbances
to improve sim-to-real transfer.
"""

import numpy as np
import mujoco
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import time


@dataclass
class RandomizationConfig:
    """Configuration for domain randomization."""
    # Enable/disable randomization
    enabled: bool = True

    # Friction randomization
    friction_enabled: bool = True
    friction_range: tuple = (0.5, 1.5)  # Multiplier range

    # Mass randomization
    mass_enabled: bool = True
    mass_range: tuple = (0.8, 1.2)  # Multiplier range

    # Motor strength randomization
    motor_enabled: bool = True
    motor_range: tuple = (0.9, 1.1)  # Multiplier range

    # Damping randomization
    damping_enabled: bool = True
    damping_range: tuple = (0.7, 1.3)  # Multiplier range

    # Sensor noise
    sensor_noise_enabled: bool = True
    joint_pos_noise_range: tuple = (0.001, 0.01)  # radians
    joint_vel_noise_range: tuple = (0.01, 0.1)    # rad/s
    imu_noise_range: tuple = (0.01, 0.1)          # m/s^2

    # Control latency
    latency_enabled: bool = True
    latency_range: tuple = (0, 2)  # timesteps

    # External pushes
    push_enabled: bool = True
    push_probability: float = 0.002  # per timestep
    push_force_range: tuple = (50.0, 200.0)  # Newtons
    push_duration_range: tuple = (5, 20)  # timesteps

    # Ground properties
    ground_enabled: bool = True
    ground_friction_range: tuple = (0.3, 2.0)
    ground_restitution_range: tuple = (0.0, 0.3)

    # Actuator properties
    actuator_enabled: bool = True
    actuator_gear_range: tuple = (0.9, 1.1)
    actuator_bias_range: tuple = (-0.1, 0.1)  # Nm

    # Joint properties
    joint_enabled: bool = True
    joint_stiffness_range: tuple = (0.8, 1.2)
    joint_armature_range: tuple = (0.5, 1.5)


class DomainRandomizer:
    """
    Handles domain randomization for the G1 environment.

    This class implements various randomization strategies:
    - Physics parameter randomization
    - Sensor noise injection
    - Control latency simulation
    - External disturbance application
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        config: Dict[str, Any]
    ):
        """
        Initialize domain randomizer.

        Args:
            model: MuJoCo model
            data: MuJoCo data
            config: Configuration dictionary
        """
        self.model = model
        self.data = data

        # Parse configuration
        self.config = RandomizationConfig(**config)

        # Store original model parameters
        self._store_original_parameters()

        # Initialize randomization state
        self.current_params = {}
        self.control_latency = 0
        self.external_push = {
            'active': False,
            'force': np.zeros(3),
            'duration': 0,
            'remaining': 0,
            'body_id': -1
        }

        print(f"DomainRandomizer initialized with {self._count_randomization_types()} randomization types")

    def _store_original_parameters(self):
        """Store original model parameters for restoration."""
        self.original_params = {
            'body_mass': self.model.body_mass.copy(),
            'geom_friction': self.model.geom_friction.copy(),
            'dof_damping': self.model.dof_damping.copy(),
            'actuator_gear': self.model.actuator_gear.copy(),
            'actuator_bias': self.model.actuator_bias.copy(),
            'jnt_stiffness': self.model.jnt_stiffness.copy(),
            'dof_armature': self.model.dof_armature.copy(),
        }

        # Find torso body for external forces
        self.torso_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link"
        )
        if self.torso_id == -1:
            self.torso_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, "torso"
            )

    def _count_randomization_types(self) -> int:
        """Count number of enabled randomization types."""
        count = 0
        if self.config.friction_enabled: count += 1
        if self.config.mass_enabled: count += 1
        if self.config.motor_enabled: count += 1
        if self.config.damping_enabled: count += 1
        if self.config.sensor_noise_enabled: count += 1
        if self.config.latency_enabled: count += 1
        if self.config.push_enabled: count += 1
        if self.config.ground_enabled: count += 1
        if self.config.actuator_enabled: count += 1
        if self.config.joint_enabled: count += 1
        return count

    def randomize(self, force: bool = False) -> Dict[str, Any]:
        """
        Apply domain randomization.

        Args:
            force: Force randomization even if disabled

        Returns:
            Dictionary of applied randomization parameters
        """
        if not self.config.enabled and not force:
            return {}

        self.current_params = {}

        # Reset to original parameters first
        self._reset_to_original()

        # Apply various randomizations
        if self.config.friction_enabled:
            self._randomize_friction()

        if self.config.mass_enabled:
            self._randomize_mass()

        if self.config.motor_enabled:
            self._randomize_motor_strength()

        if self.config.damping_enabled:
            self._randomize_damping()

        if self.config.ground_enabled:
            self._randomize_ground_properties()

        if self.config.actuator_enabled:
            self._randomize_actuator_properties()

        if self.config.joint_enabled:
            self._randomize_joint_properties()

        if self.config.latency_enabled:
            self._randomize_control_latency()

        # Sensor noise is applied during observation, not here

        return self.current_params

    def _reset_to_original(self):
        """Reset model parameters to original values."""
        self.model.body_mass[:] = self.original_params['body_mass']
        self.model.geom_friction[:] = self.original_params['geom_friction']
        self.model.dof_damping[:] = self.original_params['dof_damping']
        self.model.actuator_gear[:] = self.original_params['actuator_gear']
        self.model.actuator_bias[:] = self.original_params['actuator_bias']
        self.model.jnt_stiffness[:] = self.original_params['jnt_stiffness']
        self.model.dof_armature[:] = self.original_params['dof_armature']

    def _randomize_friction(self):
        """Randomize friction coefficients."""
        friction_multiplier = np.random.uniform(*self.config.friction_range)

        # Apply to all geoms (sliding friction)
        self.model.geom_friction[:, 0] *= friction_multiplier

        self.current_params['friction_multiplier'] = friction_multiplier

    def _randomize_mass(self):
        """Randomize body masses."""
        mass_multipliers = []

        for i in range(self.model.nbody):
            if i == 0:  # Skip world body
                continue

            # Different randomization for different body types
            if "torso" in mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i) or \
               "base" in mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i):
                # Less variation for torso
                multiplier = np.random.uniform(0.9, 1.1)
            else:
                # More variation for limbs
                multiplier = np.random.uniform(*self.config.mass_range)

            self.model.body_mass[i] *= multiplier
            mass_multipliers.append(multiplier)

        self.current_params['mass_multipliers'] = mass_multipliers

    def _randomize_motor_strength(self):
        """Randomize motor strength/gain."""
        motor_multiplier = np.random.uniform(*self.config.motor_range)

        # Apply to actuator gains (gear ratios)
        self.model.actuator_gear[:, 0] *= motor_multiplier

        self.current_params['motor_multiplier'] = motor_multiplier

    def _randomize_damping(self):
        """Randomize joint damping."""
        damping_multiplier = np.random.uniform(*self.config.damping_range)

        # Apply to all DOF damping
        self.model.dof_damping[:] *= damping_multiplier

        self.current_params['damping_multiplier'] = damping_multiplier

    def _randomize_ground_properties(self):
        """Randomize ground friction and restitution."""
        # Find ground geoms (typically have "floor" or "ground" in name)
        ground_friction = np.random.uniform(*self.config.ground_friction_range)
        ground_restitution = np.random.uniform(*self.config.ground_restitution_range)

        for i in range(self.model.ngeom):
            geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if geom_name and ("floor" in geom_name.lower() or "ground" in geom_name.lower()):
                self.model.geom_friction[i, 0] = ground_friction
                # Note: MuJoCo doesn't have direct restitution, but we could modify solimp

        self.current_params['ground_friction'] = ground_friction
        self.current_params['ground_restitution'] = ground_restitution

    def _randomize_actuator_properties(self):
        """Randomize actuator properties."""
        # Actuator gear randomization (already done in motor strength)

        # Actuator bias (constant torque offset)
        bias_values = []
        for i in range(self.model.nu):
            bias = np.random.uniform(*self.config.actuator_bias_range)
            self.model.actuator_bias[i] = bias
            bias_values.append(bias)

        self.current_params['actuator_biases'] = bias_values

    def _randomize_joint_properties(self):
        """Randomize joint properties."""
        # Joint stiffness
        stiffness_multiplier = np.random.uniform(*self.config.joint_stiffness_range)
        self.model.jnt_stiffness[:] *= stiffness_multiplier

        # Joint armature (rotor inertia)
        armature_multiplier = np.random.uniform(*self.config.joint_armature_range)
        self.model.dof_armature[:] *= armature_multiplier

        self.current_params['joint_stiffness_multiplier'] = stiffness_multiplier
        self.current_params['joint_armature_multiplier'] = armature_multiplier

    def _randomize_control_latency(self):
        """Randomize control latency."""
        self.control_latency = np.random.randint(*self.config.latency_range)
        self.current_params['control_latency'] = self.control_latency

    def should_apply_push(self) -> bool:
        """Check if external push should be applied this timestep."""
        if not self.config.push_enabled:
            return False

        if self.external_push['active']:
            return False  # Don't start new push while one is active

        return np.random.random() < self.config.push_probability

    def apply_external_push(self):
        """Apply external push force."""
        if self.torso_id == -1:
            return

        if not self.external_push['active']:
            # Start new push
            force_magnitude = np.random.uniform(*self.config.push_force_range)
            force_direction = np.random.normal(0, 1, 3)
            force_direction[2] = 0  # No vertical force
            force_direction = force_direction / (np.linalg.norm(force_direction) + 1e-6)

            self.external_push['active'] = True
            self.external_push['force'] = force_direction * force_magnitude
            self.external_push['duration'] = np.random.randint(*self.config.push_duration_range)
            self.external_push['remaining'] = self.external_push['duration']
            self.external_push['body_id'] = self.torso_id

        # Apply force
        if self.external_push['active'] and self.external_push['remaining'] > 0:
            body_id = self.external_push['body_id']
            force = self.external_push['force']

            # Apply force to body
            if body_id < self.model.nbody:
                self.data.xfrc_applied[body_id, :3] = force

            self.external_push['remaining'] -= 1

            if self.external_push['remaining'] <= 0:
                self.external_push['active'] = False
                self.data.xfrc_applied[body_id, :3] = 0.0

    def get_sensor_noise_scales(self) -> Dict[str, float]:
        """Get current sensor noise scales."""
        if not self.config.sensor_noise_enabled:
            return {
                'joint_pos_noise': 0.0,
                'joint_vel_noise': 0.0,
                'imu_noise': 0.0
            }

        return {
            'joint_pos_noise': np.random.uniform(*self.config.joint_pos_noise_range),
            'joint_vel_noise': np.random.uniform(*self.config.joint_vel_noise_range),
            'imu_noise': np.random.uniform(*self.config.imu_noise_range)
        }

    def get_current_params(self) -> Dict[str, Any]:
        """Get current randomization parameters."""
        params = self.current_params.copy()

        # Add runtime parameters
        params['control_latency'] = self.control_latency
        params['external_push_active'] = self.external_push['active']

        if self.external_push['active']:
            params['external_push_force'] = self.external_push['force'].copy()
            params['external_push_remaining'] = self.external_push['remaining']

        return params

    def reset_external_forces(self):
        """Reset all external forces."""
        self.data.xfrc_applied[:, :] = 0.0
        self.external_push['active'] = False

    def set_randomization_seed(self, seed: int):
        """Set random seed for reproducible randomization."""
        np.random.seed(seed)

    def enable_randomization(self, enabled: bool = True):
        """Enable or disable randomization."""
        self.config.enabled = enabled

    def enable_specific_randomization(self, **kwargs):
        """Enable/disable specific randomization types."""
        for key, value in kwargs.items():
            if hasattr(self.config, f"{key}_enabled"):
                setattr(self.config, f"{key}_enabled", value)

    def get_randomization_info(self) -> Dict[str, Any]:
        """Get information about randomization configuration."""
        return {
            'enabled': self.config.enabled,
            'types': {
                'friction': self.config.friction_enabled,
                'mass': self.config.mass_enabled,
                'motor': self.config.motor_enabled,
                'damping': self.config.damping_enabled,
                'sensor_noise': self.config.sensor_noise_enabled,
                'latency': self.config.latency_enabled,
                'push': self.config.push_enabled,
                'ground': self.config.ground_enabled,
                'actuator': self.config.actuator_enabled,
                'joint': self.config.joint_enabled,
            },
            'current_params': self.get_current_params()
        }

    def create_randomization_curriculum(
        self,
        training_step: int,
        total_steps: int
    ) -> Dict[str, Any]:
        """
        Create curriculum-based randomization parameters.

        Args:
            training_step: Current training step
            total_steps: Total training steps

        Returns:
            Curriculum-adjusted randomization config
        """
        # Linear curriculum from 0% to 100% randomization strength
        progress = min(1.0, training_step / total_steps)

        # Start with smaller ranges and gradually increase
        curriculum_config = {
            'friction_range': self._interpolate_range((0.9, 1.1), self.config.friction_range, progress),
            'mass_range': self._interpolate_range((0.95, 1.05), self.config.mass_range, progress),
            'motor_range': self._interpolate_range((0.98, 1.02), self.config.motor_range, progress),
            'damping_range': self._interpolate_range((0.9, 1.1), self.config.damping_range, progress),
            'joint_pos_noise_range': self._interpolate_range((0.0, 0.001), self.config.joint_pos_noise_range, progress),
            'joint_vel_noise_range': self._interpolate_range((0.0, 0.01), self.config.joint_vel_noise_range, progress),
            'imu_noise_range': self._interpolate_range((0.0, 0.01), self.config.imu_noise_range, progress),
            'push_probability': self.config.push_probability * progress,
        }

        return curriculum_config

    def _interpolate_range(
        self,
        start_range: Tuple[float, float],
        end_range: Tuple[float, float],
        progress: float
    ) -> Tuple[float, float]:
        """Interpolate between two ranges based on progress."""
        low = start_range[0] + progress * (end_range[0] - start_range[0])
        high = start_range[1] + progress * (end_range[1] - start_range[1])
        return (low, high)