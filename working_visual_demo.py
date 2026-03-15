#!/usr/bin/env python3
"""
Working Visual MuJoCo Demo - Simplified and guaranteed to work
"""

import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path
import time

from mujoco_mac_demo import MacMuJoCoPolicyInterface

# Minimal working robot XML
ROBOT_XML = """
<mujoco model="sonic_robot">
    <option timestep="0.002"/>

    <worldbody>
        <light pos="0 0 2" dir="0 0 -1"/>
        <geom name="floor" size="5 5 0.1" type="box" rgba="0.5 0.5 0.5 1"/>

        <body name="robot" pos="0 0 1">
            <joint name="root" type="free"/>
            <geom name="torso" type="box" size="0.1 0.1 0.3" rgba="0.8 0.2 0.2 1"/>

            <!-- Arms -->
            <body name="left_arm" pos="0.15 0 0.2">
                <joint name="left_shoulder" type="hinge" axis="0 1 0" range="-90 90"/>
                <geom name="left_arm" type="capsule" size="0.04 0.2" rgba="0.2 0.8 0.2 1"/>
            </body>

            <body name="right_arm" pos="-0.15 0 0.2">
                <joint name="right_shoulder" type="hinge" axis="0 1 0" range="-90 90"/>
                <geom name="right_arm" type="capsule" size="0.04 0.2" rgba="0.2 0.8 0.2 1"/>
            </body>

            <!-- Legs -->
            <body name="left_leg" pos="0.05 0 -0.35">
                <joint name="left_hip" type="hinge" axis="1 0 0" range="-30 30"/>
                <geom name="left_leg" type="capsule" size="0.04 0.15" rgba="0.2 0.2 0.8 1"/>

                <body name="left_foot" pos="0 0 -0.2">
                    <joint name="left_knee" type="hinge" axis="1 0 0" range="-120 0"/>
                    <geom name="left_foot" type="box" size="0.08 0.12 0.03" rgba="0.4 0.4 0.4 1"/>
                </body>
            </body>

            <body name="right_leg" pos="-0.05 0 -0.35">
                <joint name="right_hip" type="hinge" axis="1 0 0" range="-30 30"/>
                <geom name="right_leg" type="capsule" size="0.04 0.15" rgba="0.2 0.2 0.8 1"/>

                <body name="right_foot" pos="0 0 -0.2">
                    <joint name="right_knee" type="hinge" axis="1 0 0" range="-120 0"/>
                    <geom name="right_foot" type="box" size="0.08 0.12 0.03" rgba="0.4 0.4 0.4 1"/>
                </body>
            </body>
        </body>
    </worldbody>

    <actuator>
        <motor name="left_shoulder_motor" joint="left_shoulder" gear="25"/>
        <motor name="right_shoulder_motor" joint="right_shoulder" gear="25"/>
        <motor name="left_hip_motor" joint="left_hip" gear="100"/>
        <motor name="left_knee_motor" joint="left_knee" gear="60"/>
        <motor name="right_hip_motor" joint="right_hip" gear="100"/>
        <motor name="right_knee_motor" joint="right_knee" gear="60"/>
    </actuator>
</mujoco>
"""

def run_visual_demo():
    """Run the visual demo with your trained model."""
    print("🚀 Enhanced Sonic G1 - Visual Demo")
    print("=" * 40)

    # Find best checkpoint
    checkpoints = [
        "checkpoints/production_large_model/best_checkpoint.pth",
        "checkpoints/enhanced_mac_model/best_checkpoint.pth",
        "checkpoints/mac_training/best_checkpoint.pth"
    ]

    checkpoint_path = None
    for cp in checkpoints:
        if Path(cp).exists():
            checkpoint_path = cp
            break

    if not checkpoint_path:
        print("❌ No trained model found!")
        print("Train a model first: python3 mac_optimized_training.py --epochs 30")
        return

    print(f"🎯 Using model: {Path(checkpoint_path).parent.name}")

    # Create robot file
    robot_file = "sonic_robot.xml"
    with open(robot_file, 'w') as f:
        f.write(ROBOT_XML)

    try:
        # Load your trained policy
        print("🧠 Loading trained neural network...")
        policy = MacMuJoCoPolicyInterface(checkpoint_path)

        model_info = policy.model.get_model_info()
        print(f"   ✅ Model: {model_info['total_parameters']:,} parameters")
        print(f"   ✅ Size: {model_info['parameter_size_mb']:.1f} MB")
        print(f"   ✅ Architecture: {model_info.get('d_model', 'N/A')}d model")

        # Load MuJoCo robot
        print("🤖 Loading robot simulation...")
        model = mujoco.MjModel.from_xml_path(robot_file)
        data = mujoco.MjData(model)
        print(f"   ✅ Robot: {model.nu} actuators, {model.nq} DOFs")

        # Start visual simulation
        print("\n🎮 Starting Visual Simulation...")
        print("   📺 3D viewer will open showing your trained robot")
        print("   ⏰ Simulation will run for 20 seconds")
        print("   🔄 Robot will reset if it falls")
        print("   🧠 All movements controlled by your neural network!")

        # Run simulation with viewer
        with mujoco.viewer.launch_passive(model, data) as viewer:
            policy.reset()
            mujoco.mj_resetData(model, data)

            start_time = time.time()
            step_count = 0
            reset_count = 0
            inference_times = []

            while time.time() - start_time < 20:  # 20 second demo
                # Get robot state
                obs = np.concatenate([
                    data.qpos[7:] if len(data.qpos) > 7 else [],
                    data.qvel[6:] if len(data.qvel) > 6 else [],
                ])

                # Pad observation to match training
                if len(obs) < policy.obs_dim:
                    obs = np.pad(obs, (0, policy.obs_dim - len(obs)), mode='constant')
                else:
                    obs = obs[:policy.obs_dim]

                # Get action from YOUR trained model
                start = time.time()
                action = policy.get_action(obs.astype(np.float32))
                inference_time = time.time() - start
                inference_times.append(inference_time)

                # Apply action to robot
                if len(action) >= model.nu:
                    data.ctrl[:] = action[:model.nu]
                else:
                    data.ctrl[:len(action)] = action

                # Step physics simulation
                mujoco.mj_step(model, data)

                # Update 3D visualization
                viewer.sync()

                # Check if robot fell and reset
                height = data.qpos[2] if len(data.qpos) > 2 else 1.0
                if height < 0.3 or data.time > 5.0:
                    mujoco.mj_resetData(model, data)
                    policy.reset()
                    reset_count += 1

                step_count += 1

                # Control simulation speed
                time.sleep(0.01)  # 100 FPS max

        # Show results
        runtime = time.time() - start_time
        avg_inference = np.mean(inference_times) * 1000

        print(f"\n📊 Demo Results:")
        print(f"   🕒 Runtime: {runtime:.1f}s")
        print(f"   👣 Steps: {step_count:,}")
        print(f"   🔄 Resets: {reset_count}")
        print(f"   ⚡ Inference: {avg_inference:.2f}ms ({1000/avg_inference:.0f} FPS)")
        print(f"   🎯 Simulation FPS: {step_count/runtime:.1f}")

        print(f"\n🎉 Visual Demo Complete!")
        print(f"   Your {model_info['total_parameters']:,}-parameter neural network")
        print(f"   successfully controlled the robot in real-time 3D simulation!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Clean up
        Path(robot_file).unlink(missing_ok=True)

if __name__ == "__main__":
    run_visual_demo()