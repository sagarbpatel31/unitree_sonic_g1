#!/usr/bin/env python3
"""
Test script for the Unitree G1 MuJoCo environment.

This script demonstrates the usage of the G1 environment with various
configurations and provides a comprehensive test of all features.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import time
from typing import Dict, Any

from envs.g1 import (
    create_g1_env,
    ObservationConfig,
    RewardConfig,
    ResetConfig,
    RandomizationConfig,
    CommandConfig,
    CommandType
)


def create_test_config() -> Dict[str, Any]:
    """Create a test configuration for the G1 environment."""
    return {
        "frame_skip": 10,
        "action_type": "position_delta",  # or "position_absolute"
        "action_scale": 0.1,
        "render_mode": None,  # "rgb_array" for headless, "human" for display

        "observations": {
            "include_joint_pos": True,
            "include_joint_vel": True,
            "include_root_orientation": True,
            "include_root_linear_vel": True,
            "include_root_angular_vel": True,
            "include_previous_action": True,
            "include_reference_motion": True,
            "reference_horizon": 10,
            "include_commands": True,
            "include_foot_contacts": True,
            "include_imu": True,
            "include_height_scan": False,
            "joint_pos_noise": 0.01,
            "joint_vel_noise": 0.1,
            "imu_noise": 0.05,
            "normalize_observations": True,
        },

        "rewards": {
            "joint_pos_weight": 1.0,
            "joint_vel_weight": 0.1,
            "root_pos_weight": 1.0,
            "root_orient_weight": 1.0,
            "root_vel_weight": 0.5,
            "end_effector_weight": 0.5,
            "foot_contact_weight": 0.2,
            "foot_slip_weight": 0.1,
            "upright_weight": 0.1,
            "action_smoothness_weight": 0.05,
            "torque_penalty_weight": 0.001,
            "energy_penalty_weight": 0.001,
            "alive_bonus": 0.1,
            "command_tracking_weight": 0.5,
        },

        "resets": {
            "use_reference_pose": True,
            "pose_noise_scale": 0.1,
            "velocity_noise_scale": 0.1,
            "reset_to_reference": True,
            "reset_to_default": True,
            "reset_random": False,
            "random_reference_time": True,
            "reference_time_range": (0.0, 1.0),
            "joint_pos_noise": 0.05,
            "joint_vel_noise": 0.1,
            "root_pos_noise": 0.02,
            "root_orient_noise": 0.1,
        },

        "randomization": {
            "enabled": True,
            "friction_enabled": True,
            "friction_range": (0.5, 1.5),
            "mass_enabled": True,
            "mass_range": (0.8, 1.2),
            "motor_enabled": True,
            "motor_range": (0.9, 1.1),
            "damping_enabled": True,
            "damping_range": (0.7, 1.3),
            "sensor_noise_enabled": True,
            "joint_pos_noise_range": (0.001, 0.01),
            "joint_vel_noise_range": (0.01, 0.1),
            "imu_noise_range": (0.01, 0.1),
            "latency_enabled": False,  # Disabled for initial testing
            "latency_range": (0, 2),
            "push_enabled": True,
            "push_probability": 0.001,  # Low probability for testing
            "push_force_range": (50.0, 200.0),
            "push_duration_range": (5, 20),
        },

        "commands": {
            "enabled": True,
            "forward_vel_range": (0.0, 2.0),
            "lateral_vel_range": (-1.0, 1.0),
            "yaw_rate_range": (-2.0, 2.0),
            "command_duration_range": (2.0, 10.0),
            "resample_interval": 100,
            "smooth_commands": True,
            "smoothing_factor": 0.9,
        },

        "safety": {
            "fall_height": 0.3,
            "fall_angle": 1.0,
            "max_joint_vel": 50.0,
        }
    }


def create_dummy_reference_motion() -> Dict[str, Any]:
    """Create dummy reference motion for testing."""
    # Create a simple walking motion
    duration = 2.0  # seconds
    dt = 0.02  # 50 Hz
    num_frames = int(duration / dt)

    times = np.linspace(0, duration, num_frames)

    # Joint positions (22 joints for G1)
    num_joints = 22
    joint_positions = []
    joint_velocities = []

    for i, t in enumerate(times):
        # Simple sinusoidal walking pattern
        hip_angle = 0.3 * np.sin(2 * np.pi * t)
        knee_angle = max(0, 0.6 * np.sin(2 * np.pi * t))

        # Create joint position vector
        joint_pos = np.zeros(num_joints)
        if num_joints > 10:
            joint_pos[0] = hip_angle    # Left hip pitch
            joint_pos[3] = knee_angle   # Left knee
            joint_pos[6] = -hip_angle   # Right hip pitch
            joint_pos[9] = knee_angle   # Right knee

        joint_positions.append(joint_pos)

        # Simple velocity (derivative of position)
        joint_vel = np.zeros(num_joints)
        if i > 0:
            joint_vel = (joint_pos - joint_positions[i-1]) / dt
        joint_velocities.append(joint_vel)

    # Root motion
    root_positions = []
    root_orientations = []
    root_velocities = []

    for i, t in enumerate(times):
        # Root moves forward
        root_pos = np.array([0.5 * t, 0.0, 0.75])  # 0.5 m/s forward
        root_quat = np.array([1.0, 0.0, 0.0, 0.0])  # Upright

        root_positions.append(root_pos)
        root_orientations.append(root_quat)

        # Root velocity
        root_lin_vel = np.array([0.5, 0.0, 0.0])  # Constant forward velocity
        root_ang_vel = np.array([0.0, 0.0, 0.0])   # No rotation
        root_vel = np.concatenate([root_lin_vel, root_ang_vel])
        root_velocities.append(root_vel)

    # Foot contacts (alternating)
    foot_contacts = []
    for t in times:
        phase = (t * 2) % 2  # 2 Hz step frequency
        if phase < 1:
            contacts = [True, False]  # Left foot down
        else:
            contacts = [False, True]  # Right foot down
        foot_contacts.append(contacts)

    return {
        "times": times.tolist(),
        "joint_positions": joint_positions,
        "joint_velocities": joint_velocities,
        "root_positions": root_positions,
        "root_orientations": root_orientations,
        "root_velocities": root_velocities,
        "root_linear_velocities": [rv[:3] for rv in root_velocities],
        "root_angular_velocities": [rv[3:] for rv in root_velocities],
        "foot_contacts": foot_contacts,
        "current_time": 0.0,
        "dt": dt
    }


def test_basic_functionality(env, reference_motion):
    """Test basic environment functionality."""
    print("\n=== Testing Basic Functionality ===")

    # Test reset
    print("Testing reset...")
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Info keys: {list(info.keys())}")

    # Test steps
    print("Testing steps...")
    total_reward = 0
    for step in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if step % 20 == 0:
            print(f"Step {step}: reward={reward:.3f}, terminated={terminated}")

        if terminated or truncated:
            print(f"Episode ended at step {step}")
            break

    print(f"Total reward: {total_reward:.3f}")


def test_reference_motion(env, reference_motion):
    """Test reference motion functionality."""
    print("\n=== Testing Reference Motion ===")

    # Set reference motion
    env.set_reference_motion(reference_motion)

    # Reset with reference
    obs, info = env.reset()
    print("Reset with reference motion")

    # Run with reference
    for step in range(50):
        # Update reference time
        reference_motion["current_time"] = step * 0.02

        action = env.action_space.sample() * 0.1  # Smaller actions for stability
        obs, reward, terminated, truncated, info = env.step(action)

        if step % 10 == 0:
            reward_terms = info.get("reward_terms", {})
            print(f"Step {step}: total_reward={reward_terms.get('total_reward', 0):.3f}, "
                  f"joint_tracking={reward_terms.get('joint_pos', 0):.3f}")

        if terminated or truncated:
            break


def test_command_conditioning(env):
    """Test command-conditioned behavior."""
    print("\n=== Testing Command Conditioning ===")

    # Test different commands
    commands = [
        {"type": "stop", "forward_vel": 0.0, "lateral_vel": 0.0, "yaw_rate": 0.0},
        {"type": "walk_forward", "forward_vel": 1.0, "lateral_vel": 0.0, "yaw_rate": 0.0},
        {"type": "turn_left", "forward_vel": 0.5, "lateral_vel": 0.0, "yaw_rate": 1.0},
    ]

    for i, command in enumerate(commands):
        print(f"\nTesting command {i+1}: {command['type']}")

        env.set_command(command)
        obs, info = env.reset()

        for step in range(30):
            action = np.zeros(env.action_space.shape[0])  # Zero action to test command tracking
            obs, reward, terminated, truncated, info = env.step(action)

            if step % 10 == 0:
                reward_terms = info.get("reward_terms", {})
                cmd_reward = reward_terms.get("command_tracking", 0)
                print(f"  Step {step}: command_tracking_reward={cmd_reward:.3f}")

            if terminated or truncated:
                break


def test_domain_randomization(env):
    """Test domain randomization."""
    print("\n=== Testing Domain Randomization ===")

    for trial in range(3):
        print(f"\nRandomization trial {trial + 1}")

        # Reset with randomization
        obs, info = env.reset(options={"randomize": True})

        # Print randomization parameters
        rand_params = info.get("randomization", {})
        print(f"Randomization parameters: {rand_params}")

        # Run a few steps
        for step in range(20):
            action = env.action_space.sample() * 0.05  # Small actions
            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                print(f"Episode terminated at step {step}")
                break


def test_safety_features(env):
    """Test safety and termination conditions."""
    print("\n=== Testing Safety Features ===")

    # Test with extreme actions
    obs, info = env.reset()
    print("Testing with extreme actions...")

    for step in range(50):
        # Large random actions to test safety
        action = env.action_space.sample() * 2.0  # Large actions
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated:
            termination_reason = info.get("termination_reason")
            print(f"Safety termination at step {step}: {termination_reason}")
            break


def create_dummy_g1_model():
    """Create a minimal dummy G1 model file for testing."""
    model_xml = """
    <mujoco model="unitree_g1">
        <compiler angle="radian" coordinate="local" inertiafromgeom="true"/>

        <default>
            <joint armature="0.01" damping="0.1" limited="true"/>
            <geom friction="0.8 0.1 0.1" rgba="0.7 0.7 0.7 1"/>
        </default>

        <worldbody>
            <geom name="floor" pos="0 0 0" size="10 10 0.1" type="plane" material="floor"/>
            <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>

            <body name="base_link" pos="0 0 0.75">
                <freejoint name="root"/>
                <inertial pos="0 0 0" mass="10" diaginertia="0.1 0.1 0.1"/>
                <geom name="torso" size="0.15 0.1 0.3" type="box"/>

                <!-- Left leg -->
                <body name="left_hip" pos="0 0.1 -0.1">
                    <joint name="left_hip_yaw" axis="0 0 1" range="-1.57 1.57"/>
                    <joint name="left_hip_roll" axis="1 0 0" range="-0.52 0.52"/>
                    <joint name="left_hip_pitch" axis="0 1 0" range="-1.57 1.57"/>
                    <inertial pos="0 0 -0.1" mass="2" diaginertia="0.01 0.01 0.01"/>
                    <geom name="left_thigh" size="0.05" fromto="0 0 0 0 0 -0.3" type="capsule"/>

                    <body name="left_knee" pos="0 0 -0.3">
                        <joint name="left_knee_pitch" axis="0 1 0" range="0 2.35"/>
                        <inertial pos="0 0 -0.15" mass="1.5" diaginertia="0.01 0.01 0.01"/>
                        <geom name="left_shin" size="0.03" fromto="0 0 0 0 0 -0.3" type="capsule"/>

                        <body name="left_foot" pos="0 0 -0.3">
                            <joint name="left_ankle_pitch" axis="0 1 0" range="-0.87 0.87"/>
                            <joint name="left_ankle_roll" axis="1 0 0" range="-0.52 0.52"/>
                            <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
                            <geom name="left_foot" size="0.1 0.05 0.02" type="box" rgba="0.9 0.2 0.2 1"/>
                        </body>
                    </body>
                </body>

                <!-- Right leg -->
                <body name="right_hip" pos="0 -0.1 -0.1">
                    <joint name="right_hip_yaw" axis="0 0 1" range="-1.57 1.57"/>
                    <joint name="right_hip_roll" axis="1 0 0" range="-0.52 0.52"/>
                    <joint name="right_hip_pitch" axis="0 1 0" range="-1.57 1.57"/>
                    <inertial pos="0 0 -0.1" mass="2" diaginertia="0.01 0.01 0.01"/>
                    <geom name="right_thigh" size="0.05" fromto="0 0 0 0 0 -0.3" type="capsule"/>

                    <body name="right_knee" pos="0 0 -0.3">
                        <joint name="right_knee_pitch" axis="0 1 0" range="0 2.35"/>
                        <inertial pos="0 0 -0.15" mass="1.5" diaginertia="0.01 0.01 0.01"/>
                        <geom name="right_shin" size="0.03" fromto="0 0 0 0 0 -0.3" type="capsule"/>

                        <body name="right_foot" pos="0 0 -0.3">
                            <joint name="right_ankle_pitch" axis="0 1 0" range="-0.87 0.87"/>
                            <joint name="right_ankle_roll" axis="1 0 0" range="-0.52 0.52"/>
                            <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
                            <geom name="right_foot" size="0.1 0.05 0.02" type="box" rgba="0.2 0.9 0.2 1"/>
                        </body>
                    </body>
                </body>

                <!-- Torso joints -->
                <body name="torso_pitch" pos="0 0 0.1">
                    <joint name="torso_pitch" axis="0 1 0" range="-0.26 0.26"/>
                    <body name="torso_roll" pos="0 0 0">
                        <joint name="torso_roll" axis="1 0 0" range="-0.26 0.26"/>

                        <!-- Left arm -->
                        <body name="left_shoulder" pos="0 0.2 0.1">
                            <joint name="left_shoulder_pitch" axis="0 1 0" range="-1.57 1.57"/>
                            <joint name="left_shoulder_roll" axis="1 0 0" range="-1.57 1.57"/>
                            <inertial pos="0 0 -0.1" mass="1" diaginertia="0.01 0.01 0.01"/>
                            <geom name="left_upper_arm" size="0.03" fromto="0 0 0 0 0 -0.25" type="capsule"/>

                            <body name="left_elbow" pos="0 0 -0.25">
                                <joint name="left_elbow_pitch" axis="0 1 0" range="-2.09 2.09"/>
                                <inertial pos="0 0 -0.1" mass="0.5" diaginertia="0.01 0.01 0.01"/>
                                <geom name="left_forearm" size="0.025" fromto="0 0 0 0 0 -0.2" type="capsule"/>

                                <body name="left_hand" pos="0 0 -0.2">
                                    <inertial pos="0 0 0" mass="0.3" diaginertia="0.01 0.01 0.01"/>
                                    <geom name="left_hand" size="0.05 0.03 0.02" type="box" rgba="0.9 0.9 0.2 1"/>
                                </body>
                            </body>
                        </body>

                        <!-- Right arm -->
                        <body name="right_shoulder" pos="0 -0.2 0.1">
                            <joint name="right_shoulder_pitch" axis="0 1 0" range="-1.57 1.57"/>
                            <joint name="right_shoulder_roll" axis="1 0 0" range="-1.57 1.57"/>
                            <inertial pos="0 0 -0.1" mass="1" diaginertia="0.01 0.01 0.01"/>
                            <geom name="right_upper_arm" size="0.03" fromto="0 0 0 0 0 -0.25" type="capsule"/>

                            <body name="right_elbow" pos="0 0 -0.25">
                                <joint name="right_elbow_pitch" axis="0 1 0" range="-2.09 2.09"/>
                                <inertial pos="0 0 -0.1" mass="0.5" diaginertia="0.01 0.01 0.01"/>
                                <geom name="right_forearm" size="0.025" fromto="0 0 0 0 0 -0.2" type="capsule"/>

                                <body name="right_hand" pos="0 0 -0.2">
                                    <inertial pos="0 0 0" mass="0.3" diaginertia="0.01 0.01 0.01"/>
                                    <geom name="right_hand" size="0.05 0.03 0.02" type="box" rgba="0.2 0.9 0.9 1"/>
                                </body>
                            </body>
                        </body>

                        <!-- Head -->
                        <body name="head" pos="0 0 0.2">
                            <joint name="head_yaw" axis="0 0 1" range="-1.57 1.57"/>
                            <joint name="head_pitch" axis="0 1 0" range="-0.52 0.52"/>
                            <inertial pos="0 0 0.05" mass="0.5" diaginertia="0.01 0.01 0.01"/>
                            <geom name="head" size="0.08" pos="0 0 0.05" type="sphere" rgba="0.9 0.9 0.9 1"/>
                        </body>
                    </body>
                </body>
            </body>
        </worldbody>

        <actuator>
            <!-- Leg actuators -->
            <position name="left_hip_yaw_act" joint="left_hip_yaw" kp="100" kd="10"/>
            <position name="left_hip_roll_act" joint="left_hip_roll" kp="100" kd="10"/>
            <position name="left_hip_pitch_act" joint="left_hip_pitch" kp="100" kd="10"/>
            <position name="left_knee_pitch_act" joint="left_knee_pitch" kp="100" kd="10"/>
            <position name="left_ankle_pitch_act" joint="left_ankle_pitch" kp="50" kd="5"/>
            <position name="left_ankle_roll_act" joint="left_ankle_roll" kp="50" kd="5"/>

            <position name="right_hip_yaw_act" joint="right_hip_yaw" kp="100" kd="10"/>
            <position name="right_hip_roll_act" joint="right_hip_roll" kp="100" kd="10"/>
            <position name="right_hip_pitch_act" joint="right_hip_pitch" kp="100" kd="10"/>
            <position name="right_knee_pitch_act" joint="right_knee_pitch" kp="100" kd="10"/>
            <position name="right_ankle_pitch_act" joint="right_ankle_pitch" kp="50" kd="5"/>
            <position name="right_ankle_roll_act" joint="right_ankle_roll" kp="50" kd="5"/>

            <!-- Torso actuators -->
            <position name="torso_pitch_act" joint="torso_pitch" kp="200" kd="20"/>
            <position name="torso_roll_act" joint="torso_roll" kp="200" kd="20"/>

            <!-- Arm actuators -->
            <position name="left_shoulder_pitch_act" joint="left_shoulder_pitch" kp="50" kd="5"/>
            <position name="left_shoulder_roll_act" joint="left_shoulder_roll" kp="50" kd="5"/>
            <position name="left_elbow_pitch_act" joint="left_elbow_pitch" kp="30" kd="3"/>

            <position name="right_shoulder_pitch_act" joint="right_shoulder_pitch" kp="50" kd="5"/>
            <position name="right_shoulder_roll_act" joint="right_shoulder_roll" kp="50" kd="5"/>
            <position name="right_elbow_pitch_act" joint="right_elbow_pitch" kp="30" kd="3"/>

            <!-- Head actuators -->
            <position name="head_yaw_act" joint="head_yaw" kp="20" kd="2"/>
            <position name="head_pitch_act" joint="head_pitch" kp="20" kd="2"/>
        </actuator>

        <asset>
            <material name="floor" rgba="0.8 0.9 0.8 1"/>
        </asset>
    </mujoco>
    """

    return model_xml


def main():
    """Main test function."""
    print("Unitree G1 Environment Test")
    print("=" * 50)

    # Create dummy model
    model_xml = create_dummy_g1_model()

    # Save dummy model
    model_path = Path(__file__).parent / "dummy_g1.xml"
    with open(model_path, 'w') as f:
        f.write(model_xml)

    try:
        # Create test configuration
        config = create_test_config()

        # Create environment
        print(f"Creating environment with model: {model_path}")
        env = create_g1_env(model_path=str(model_path), config=config)

        print(f"Environment created successfully!")
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")

        # Create reference motion
        reference_motion = create_dummy_reference_motion()

        # Run tests
        test_basic_functionality(env, reference_motion)
        test_reference_motion(env, reference_motion)
        test_command_conditioning(env)
        test_domain_randomization(env)
        test_safety_features(env)

        print("\n=== All Tests Completed Successfully! ===")

        # Clean up
        env.close()

    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up dummy model file
        if model_path.exists():
            model_path.unlink()


if __name__ == "__main__":
    main()