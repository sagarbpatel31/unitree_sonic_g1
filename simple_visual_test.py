#!/usr/bin/env python3
"""
Simple Visual MuJoCo Test - Quick test of visual simulation
"""

import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path

from mujoco_mac_demo import MacMuJoCoPolicyInterface

# Simple robot XML
SIMPLE_ROBOT_XML = """
<mujoco model="test_robot">
    <option timestep="0.002"/>

    <asset>
        <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0"/>
        <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3" rgb2=".2 .3 .4"/>
        <material name="grid" texture="grid" texrepeat="1 1" reflectance=".2"/>
    </asset>

    <worldbody>
        <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
        <geom name="floor" size="0 0 .05" type="plane" material="grid"/>

        <body name="robot" pos="0 0 1">
            <joint name="root" type="free"/>
            <geom name="torso" type="box" size="0.1 0.1 0.3" rgba="0.8 0.2 0.2 1"/>

            <body name="arm1" pos="0.15 0 0.2">
                <joint name="shoulder1" type="hinge" axis="0 1 0" range="-90 90"/>
                <geom name="arm1" type="capsule" size="0.05 0.2" rgba="0.2 0.8 0.2 1"/>

                <body name="arm2" pos="0.25 0 0">
                    <joint name="elbow1" type="hinge" axis="0 1 0" range="-120 0"/>
                    <geom name="arm2" type="capsule" size="0.04 0.15" rgba="0.2 0.2 0.8 1"/>
                </body>
            </body>

            <body name="leg1" pos="0.05 0 -0.35">
                <joint name="hip1" type="hinge" axis="0 1 0" range="-30 30"/>
                <geom name="leg1" type="capsule" size="0.04 0.15" rgba="0.6 0.6 0.2 1"/>

                <body name="foot1" pos="0 0 -0.2">
                    <joint name="knee1" type="hinge" axis="0 1 0" range="-90 0"/>
                    <geom name="foot1" type="capsule" size="0.03 0.1" rgba="0.4 0.4 0.8 1"/>
                </body>
            </body>

            <body name="leg2" pos="-0.05 0 -0.35">
                <joint name="hip2" type="hinge" axis="0 1 0" range="-30 30"/>
                <geom name="leg2" type="capsule" size="0.04 0.15" rgba="0.6 0.6 0.2 1"/>

                <body name="foot2" pos="0 0 -0.2">
                    <joint name="knee2" type="hinge" axis="0 1 0" range="-90 0"/>
                    <geom name="foot2" type="capsule" size="0.03 0.1" rgba="0.4 0.4 0.8 1"/>
                </body>
            </body>
        </body>
    </worldbody>

    <actuator>
        <motor name="shoulder1_motor" joint="shoulder1" gear="50"/>
        <motor name="elbow1_motor" joint="elbow1" gear="30"/>
        <motor name="hip1_motor" joint="hip1" gear="80"/>
        <motor name="knee1_motor" joint="knee1" gear="60"/>
        <motor name="hip2_motor" joint="hip2" gear="80"/>
        <motor name="knee2_motor" joint="knee2" gear="60"/>
    </actuator>
</mujoco>
"""

def test_visual_simulation():
    """Test visual simulation with your trained model."""
    print("🤖 Testing Visual MuJoCo Simulation")
    print("=" * 40)

    # Find checkpoint
    checkpoint_candidates = [
        "checkpoints/production_large_model/best_checkpoint.pth",
        "checkpoints/enhanced_mac_model/best_checkpoint.pth",
        "checkpoints/mac_training/best_checkpoint.pth"
    ]

    checkpoint_path = None
    for candidate in checkpoint_candidates:
        if Path(candidate).exists():
            checkpoint_path = candidate
            break

    if not checkpoint_path:
        print("❌ No trained model found!")
        return

    print(f"🎯 Using model: {checkpoint_path}")

    # Create robot file
    robot_file = "test_robot.xml"
    with open(robot_file, 'w') as f:
        f.write(SIMPLE_ROBOT_XML)

    try:
        # Load policy
        policy = MacMuJoCoPolicyInterface(checkpoint_path)
        print(f"✅ Policy loaded: {policy.model.get_model_info()['total_parameters']:,} params")

        # Load MuJoCo model
        model = mujoco.MjModel.from_xml_path(robot_file)
        data = mujoco.MjData(model)
        print(f"✅ Robot loaded: {model.nu} actuators")

        # Test without viewer first
        print("\n🧪 Testing simulation (no viewer)...")
        policy.reset()

        for step in range(50):
            # Get observation
            obs = np.concatenate([
                data.qpos[7:],  # joint positions
                data.qvel[6:],  # joint velocities
                np.zeros(max(0, policy.obs_dim - len(data.qpos[7:]) - len(data.qvel[6:])))
            ])[:policy.obs_dim]

            # Get action
            action = policy.get_action(obs)

            # Apply action
            data.ctrl[:len(action)] = action[:model.nu]

            # Step
            mujoco.mj_step(model, data)

            if step % 10 == 0:
                height = data.qpos[2] if len(data.qpos) > 2 else 1.0
                print(f"   Step {step}: Height = {height:.3f}, Action range = [{action.min():.3f}, {action.max():.3f}]")

        print("✅ Simulation test successful!")

        # Test with viewer
        print("\n🎮 Testing with visual viewer...")
        print("   (Close the viewer window to continue)")

        with mujoco.viewer.launch_passive(model, data) as viewer:
            policy.reset()
            mujoco.mj_resetData(model, data)

            for step in range(500):
                # Get observation
                obs = np.concatenate([
                    data.qpos[7:],  # joint positions
                    data.qvel[6:],  # joint velocities
                    np.zeros(max(0, policy.obs_dim - len(data.qpos[7:]) - len(data.qvel[6:])))
                ])[:policy.obs_dim]

                # Get action from your trained model
                action = policy.get_action(obs)

                # Apply to robot
                data.ctrl[:len(action)] = action[:model.nu]

                # Step simulation
                mujoco.mj_step(model, data)

                # Update visualization
                viewer.sync()

                # Reset if robot falls
                if len(data.qpos) > 2 and data.qpos[2] < 0.3:
                    mujoco.mj_resetData(model, data)
                    policy.reset()
                    print(f"   Reset at step {step}")

        print("✅ Visual simulation test completed!")

    except Exception as e:
        print(f"❌ Error: {e}")

    finally:
        # Clean up
        Path(robot_file).unlink(missing_ok=True)

if __name__ == "__main__":
    test_visual_simulation()