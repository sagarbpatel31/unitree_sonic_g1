"""
Robust training environment with domain randomization and perturbations.
Used for fine-tuning policies for real-world deployment robustness.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
import random

from .motion_imitation import MotionImitationEnv
from ..core.config import Config


class RobustTrainingEnv(MotionImitationEnv):
    """
    Environment for robustness training with domain randomization.

    This environment extends motion imitation with:
    - Dynamic parameter randomization
    - External force perturbations
    - Variable motion difficulty
    - Curriculum learning
    """

    def __init__(self, config: Config):
        super().__init__(config)

        # Domain randomization configuration
        self.domain_rand = config.env.robot.get("mass_multiplier", [1.0, 1.0])
        self.friction_rand = config.env.robot.get("friction_multiplier", [1.0, 1.0])
        self.damping_rand = config.env.robot.get("damping_multiplier", [1.0, 1.0])
        self.actuator_rand = config.env.robot.get("actuator_strength", [1.0, 1.0])

        # Perturbation configuration
        self.perturbation_config = config.env.task.get("external_forces", {})
        self.perturbation_enabled = self.perturbation_config.get("enabled", False)
        self.max_force = self.perturbation_config.get("max_force", 100.0)
        self.force_frequency = self.perturbation_config.get("frequency", 0.1)

        # Curriculum learning
        self.curriculum_config = config.training.get("curriculum", {})
        self.curriculum_enabled = self.curriculum_config.get("enabled", False)
        self.current_difficulty = 0.3  # Start easy

        # Randomization state
        self.current_mass_multiplier = 1.0
        self.current_friction_multiplier = 1.0
        self.current_damping_multiplier = 1.0
        self.current_actuator_strength = 1.0

        # Perturbation state
        self.applied_force = np.zeros(3)
        self.force_countdown = 0

        # Enhanced observation space for robustness information
        self._update_robust_observation_space()

    def _update_robust_observation_space(self):
        """Update observation space to include robustness-related information."""
        # Add domain randomization parameters as observations
        base_obs_dim = self.observation_space.shape[0]
        robust_obs_dim = 4  # mass, friction, damping, actuator strength multipliers

        from gymnasium import spaces
        total_obs_dim = base_obs_dim + robust_obs_dim

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_obs_dim,), dtype=np.float32
        )

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment with domain randomization."""
        # Update curriculum difficulty
        if self.curriculum_enabled:
            self._update_curriculum_difficulty()

        # Apply domain randomization
        self._randomize_dynamics()

        # Reset base environment
        obs, info = super().reset(seed=seed, options=options)

        # Reset perturbation state
        self.applied_force = np.zeros(3)
        self.force_countdown = 0

        # Add domain randomization info
        info.update({
            "mass_multiplier": self.current_mass_multiplier,
            "friction_multiplier": self.current_friction_multiplier,
            "damping_multiplier": self.current_damping_multiplier,
            "actuator_strength": self.current_actuator_strength,
            "curriculum_difficulty": self.current_difficulty,
        })

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute step with perturbations and domain randomization."""
        # Apply external perturbations
        if self.perturbation_enabled:
            self._apply_external_forces()

        # Scale actions by actuator strength
        scaled_action = action * self.current_actuator_strength

        # Execute base step
        observation, reward, terminated, truncated, info = super().step(scaled_action)

        # Enhanced reward with robustness bonuses
        robust_reward = self._compute_robustness_reward(action, info)
        reward += robust_reward

        info.update({
            "robustness_reward": robust_reward,
            "applied_force_magnitude": np.linalg.norm(self.applied_force),
        })

        return observation, reward, terminated, truncated, info

    def _update_curriculum_difficulty(self):
        """Update curriculum difficulty based on training progress."""
        if not self.curriculum_enabled:
            return

        # Get training step from environment (simplified - in practice would come from trainer)
        training_step = getattr(self, '_training_step', 0)

        # Linear curriculum progression
        start_difficulty = self.curriculum_config.get("start_difficulty", 0.3)
        end_difficulty = self.curriculum_config.get("end_difficulty", 1.0)
        steps_to_full = self.curriculum_config.get("steps_to_full", 2_000_000)

        progress = min(1.0, training_step / steps_to_full)
        self.current_difficulty = start_difficulty + progress * (end_difficulty - start_difficulty)

    def _randomize_dynamics(self):
        """Apply domain randomization to physics parameters."""
        difficulty = self.current_difficulty

        # Mass randomization
        mass_range = self.domain_rand
        mass_spread = (mass_range[1] - mass_range[0]) * difficulty
        mass_center = (mass_range[1] + mass_range[0]) / 2
        self.current_mass_multiplier = self.np_random.uniform(
            mass_center - mass_spread/2,
            mass_center + mass_spread/2
        )

        # Friction randomization
        friction_range = self.friction_rand
        friction_spread = (friction_range[1] - friction_range[0]) * difficulty
        friction_center = (friction_range[1] + friction_range[0]) / 2
        self.current_friction_multiplier = self.np_random.uniform(
            friction_center - friction_spread/2,
            friction_center + friction_spread/2
        )

        # Damping randomization
        damping_range = self.damping_rand
        damping_spread = (damping_range[1] - damping_range[0]) * difficulty
        damping_center = (damping_range[1] + damping_range[0]) / 2
        self.current_damping_multiplier = self.np_random.uniform(
            damping_center - damping_spread/2,
            damping_center + damping_spread/2
        )

        # Actuator strength randomization
        actuator_range = self.actuator_rand
        actuator_spread = (actuator_range[1] - actuator_range[0]) * difficulty
        actuator_center = (actuator_range[1] + actuator_range[0]) / 2
        self.current_actuator_strength = self.np_random.uniform(
            actuator_center - actuator_spread/2,
            actuator_center + actuator_spread/2
        )

        # Apply randomization to MuJoCo model
        self._apply_physics_randomization()

    def _apply_physics_randomization(self):
        """Apply randomized parameters to MuJoCo model."""
        # Mass randomization
        for i in range(self.model.nbody):
            if i > 0:  # Skip world body
                self.model.body_mass[i] *= self.current_mass_multiplier

        # Friction randomization
        for i in range(self.model.ngeom):
            self.model.geom_friction[i, 0] *= self.current_friction_multiplier

        # Damping randomization
        for i in range(self.model.njnt):
            self.model.dof_damping[i] *= self.current_damping_multiplier

    def _apply_external_forces(self):
        """Apply external force perturbations."""
        if self.force_countdown <= 0:
            # Decide whether to apply new perturbation
            if self.np_random.random() < self.force_frequency:
                # Generate random force
                force_magnitude = self.np_random.uniform(0, self.max_force * self.current_difficulty)
                force_direction = self.np_random.uniform(-1, 1, size=3)
                force_direction /= np.linalg.norm(force_direction)

                self.applied_force = force_direction * force_magnitude
                self.force_countdown = self.np_random.randint(5, 20)  # Apply for 5-20 steps
            else:
                self.applied_force = np.zeros(3)
        else:
            self.force_countdown -= 1

        # Apply force to robot base
        if np.linalg.norm(self.applied_force) > 0:
            # Add external force to base body (simplified - would be more sophisticated)
            base_body_id = 1  # Assuming base body is body 1
            if base_body_id < self.model.nbody:
                self.data.xfrc_applied[base_body_id, :3] = self.applied_force

    def _get_observation(self) -> np.ndarray:
        """Get observation including domain randomization parameters."""
        # Get base observation
        base_obs = super()._get_observation()

        # Add domain randomization parameters
        robust_obs = np.array([
            self.current_mass_multiplier,
            self.current_friction_multiplier,
            self.current_damping_multiplier,
            self.current_actuator_strength,
        ], dtype=np.float32)

        # Combine observations
        observation = np.concatenate([base_obs, robust_obs])

        return observation

    def _compute_robustness_reward(self, action: np.ndarray, info: Dict[str, Any]) -> float:
        """Compute additional rewards for robustness training."""
        reward = 0.0

        # Stability reward - bonus for maintaining good tracking under perturbations
        if np.linalg.norm(self.applied_force) > 0:
            # Reward for good tracking despite external forces
            tracking_error = info.get("tracking_error", 1.0)
            stability_bonus = np.exp(-5 * tracking_error) * 0.5
            reward += stability_bonus

        # Recovery reward - bonus for quick recovery from perturbations
        if hasattr(self, '_previous_tracking_error'):
            tracking_error = info.get("tracking_error", 1.0)
            error_improvement = self._previous_tracking_error - tracking_error
            if error_improvement > 0 and self._previous_tracking_error > 0.1:
                recovery_bonus = error_improvement * 1.0
                reward += recovery_bonus

        self._previous_tracking_error = info.get("tracking_error", 1.0)

        # Efficiency reward - bonus for achieving good performance with less effort
        action_magnitude = np.linalg.norm(action)
        efficiency_bonus = max(0, 1.0 - action_magnitude) * 0.1
        reward += efficiency_bonus

        return reward

    def set_training_step(self, step: int):
        """Set current training step for curriculum learning."""
        self._training_step = step
        if self.curriculum_enabled:
            self._update_curriculum_difficulty()

    def get_domain_randomization_info(self) -> Dict[str, float]:
        """Get current domain randomization parameters."""
        return {
            "mass_multiplier": self.current_mass_multiplier,
            "friction_multiplier": self.current_friction_multiplier,
            "damping_multiplier": self.current_damping_multiplier,
            "actuator_strength": self.current_actuator_strength,
            "difficulty": self.current_difficulty,
        }