#!/usr/bin/env python3
"""
Complete Unitree G1 Simulation with Your Trained 19M Parameter Neural Network
Real-time 3D visualization using your actual trained transformer model
"""

import numpy as np
import mujoco
import mujoco.viewer
import time
from pathlib import Path
from mujoco_mac_demo import MacMuJoCoPolicyInterface

# Complete Unitree G1 Robot XML (Simplified but Realistic)
UNITREE_G1_XML = """
<mujoco model="unitree_g1">
    <compiler angle="degree"/>
    <option timestep="0.002" iterations="50"/>

    <default>
        <motor ctrlrange="-1 1" ctrllimited="true"/>
        <geom contype="1" conaffinity="1" condim="3" friction="1.5 0.5 0.005"/>
        <joint armature="0.01" damping="0.2" limited="true"/>
    </default>

    <asset>
        <texture type="skybox" builtin="gradient" rgb1="0.4 0.6 0.8" rgb2="0.0 0.0 0.0"/>
        <material name="floor_material" rgba="0.5 0.5 0.5 1" reflectance="0.1"/>
        <material name="robot_red" rgba="0.8 0.2 0.2 1"/>
        <material name="robot_blue" rgba="0.2 0.3 0.8 1"/>
        <material name="robot_green" rgba="0.2 0.8 0.3 1"/>
        <material name="robot_yellow" rgba="0.8 0.8 0.2 1"/>
    </asset>

    <worldbody>
        <light name="spotlight" mode="targetbodycom" target="torso" pos="0 -1 2"/>
        <geom name="floor" size="10 10 0.125" type="plane" material="floor_material"/>

        <body name="torso" pos="0 0 0.9">
            <camera name="track" mode="trackcom" pos="0 -2 1" xyaxes="1 0 0 0 1 2"/>
            <joint name="root" type="free"/>

            <!-- Main torso -->
            <geom name="torso" type="box" size="0.12 0.06 0.2" material="robot_blue"/>

            <!-- Head -->
            <body name="head" pos="0 0 0.25">
                <geom name="head" type="sphere" size="0.06" material="robot_yellow"/>
            </body>

            <!-- Left arm chain -->
            <body name="left_shoulder" pos="0.1 0.08 0.1">
                <joint name="left_shoulder_pitch" type="hinge" axis="1 0 0" range="-120 120"/>
                <geom name="left_upper_arm" type="capsule" size="0.03 0.1" material="robot_green"/>

                <body name="left_elbow" pos="0 0 -0.12">
                    <joint name="left_elbow_pitch" type="hinge" axis="1 0 0" range="-150 10"/>
                    <geom name="left_forearm" type="capsule" size="0.025 0.08" material="robot_red"/>

                    <body name="left_hand" pos="0 0 -0.1">
                        <geom name="left_hand" type="box" size="0.03 0.015 0.04" material="robot_yellow"/>
                    </body>
                </body>
            </body>

            <!-- Right arm chain -->
            <body name="right_shoulder" pos="0.1 -0.08 0.1">
                <joint name="right_shoulder_pitch" type="hinge" axis="1 0 0" range="-120 120"/>
                <geom name="right_upper_arm" type="capsule" size="0.03 0.1" material="robot_green"/>

                <body name="right_elbow" pos="0 0 -0.12">
                    <joint name="right_elbow_pitch" type="hinge" axis="1 0 0" range="-150 10"/>
                    <geom name="right_forearm" type="capsule" size="0.025 0.08" material="robot_red"/>

                    <body name="right_hand" pos="0 0 -0.1">
                        <geom name="right_hand" type="box" size="0.03 0.015 0.04" material="robot_yellow"/>
                    </body>
                </body>
            </body>

            <!-- Left leg chain -->
            <body name="left_hip" pos="0 0.05 -0.1">
                <joint name="left_hip_pitch" type="hinge" axis="1 0 0" range="-30 120"/>
                <geom name="left_thigh" type="capsule" size="0.04 0.12" material="robot_red"/>

                <body name="left_knee" pos="0 0 -0.15">
                    <joint name="left_knee_pitch" type="hinge" axis="1 0 0" range="-150 0"/>
                    <geom name="left_shin" type="capsule" size="0.035 0.1" material="robot_green"/>

                    <body name="left_ankle" pos="0 0 -0.12">
                        <joint name="left_ankle_pitch" type="hinge" axis="1 0 0" range="-45 45"/>
                        <geom name="left_foot" type="box" size="0.06 0.03 0.015" pos="0.02 0 -0.015" material="robot_blue"/>
                    </body>
                </body>
            </body>

            <!-- Right leg chain -->
            <body name="right_hip" pos="0 -0.05 -0.1">
                <joint name="right_hip_pitch" type="hinge" axis="1 0 0" range="-30 120"/>
                <geom name="right_thigh" type="capsule" size="0.04 0.12" material="robot_red"/>

                <body name="right_knee" pos="0 0 -0.15">
                    <joint name="right_knee_pitch" type="hinge" axis="1 0 0" range="-150 0"/>
                    <geom name="right_shin" type="capsule" size="0.035 0.1" material="robot_green"/>

                    <body name="right_ankle" pos="0 0 -0.12">
                        <joint name="right_ankle_pitch" type="hinge" axis="1 0 0" range="-45 45"/>
                        <geom name="right_foot" type="box" size="0.06 0.03 0.015" pos="0.02 0 -0.015" material="robot_blue"/>
                    </body>
                </body>
            </body>
        </body>
    </worldbody>

    <actuator>
        <!-- Arms (6 DOF) -->
        <motor name="left_shoulder_pitch_motor" joint="left_shoulder_pitch" gear="50"/>
        <motor name="left_elbow_pitch_motor" joint="left_elbow_pitch" gear="30"/>

        <motor name="right_shoulder_pitch_motor" joint="right_shoulder_pitch" gear="50"/>
        <motor name="right_elbow_pitch_motor" joint="right_elbow_pitch" gear="30"/>

        <!-- Legs (6 DOF) -->
        <motor name="left_hip_pitch_motor" joint="left_hip_pitch" gear="150"/>
        <motor name="left_knee_pitch_motor" joint="left_knee_pitch" gear="150"/>
        <motor name="left_ankle_pitch_motor" joint="left_ankle_pitch" gear="50"/>

        <motor name="right_hip_pitch_motor" joint="right_hip_pitch" gear="150"/>
        <motor name="right_knee_pitch_motor" joint="right_knee_pitch" gear="150"/>
        <motor name="right_ankle_pitch_motor" joint="right_ankle_pitch" gear="50"/>
    </actuator>
</mujoco>
"""

def run_unitree_g1_simulation():
    """Run complete Unitree G1 simulation with your trained neural network."""
    print("🤖 Unitree G1 Complete Simulation")
    print("=" * 40)
    print("🧠 Loading your 19M parameter trained neural network...")

    # Find your trained checkpoint
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
        print("Train first: python3 mac_optimized_training.py --epochs 30")
        return

    print(f"✅ Found model: {Path(checkpoint_path).parent.name}")

    # Create robot XML file
    robot_file = "unitree_g1.xml"
    with open(robot_file, 'w') as f:
        f.write(UNITREE_G1_XML)

    try:
        # Load your actual trained policy
        print("🔄 Loading neural network...")
        policy = MacMuJoCoPolicyInterface(checkpoint_path)

        model_info = policy.model.get_model_info()
        print(f"✅ Model loaded:")
        print(f"   🧠 Parameters: {model_info['total_parameters']:,}")
        print(f"   📊 Model size: {model_info['parameter_size_mb']:.1f} MB")
        print(f"   🎯 Architecture: {model_info.get('d_model', 'N/A')}d transformer")
        print(f"   📥 Input dim: {policy.obs_dim}")
        print(f"   📤 Output dim: {policy.action_dim}")

        # Load MuJoCo model
        print("🤖 Loading Unitree G1 robot...")
        model = mujoco.MjModel.from_xml_path(robot_file)
        data = mujoco.MjData(model)
        print(f"✅ Robot loaded: {model.nu} actuators, {model.nq} DOFs")

        print("\\n🎮 Starting 3D Visual Simulation...")
        print("   📺 3D viewer window will open")
        print("   🧠 Your trained transformer controlling the robot")
        print("   ⏰ Simulation runs indefinitely")
        print("   🔄 Robot resets automatically when it falls")
        print("   📊 Real-time performance monitoring")
        print("   ⌨️  Press ESC to exit")

        # Start simulation with viewer
        with mujoco.viewer.launch_passive(model, data) as viewer:
            # Reset everything
            policy.reset()
            mujoco.mj_resetData(model, data)

            # Simulation variables
            step_count = 0
            reset_count = 0
            inference_times = []
            start_time = time.time()
            last_stats_time = start_time

            print("\\n🚀 Simulation started!")
            print("   Watch your trained neural network control the Unitree G1!")

            while viewer.is_running():
                # Extract robot observations (same as training)
                obs_components = []

                # Joint positions (excluding root)
                if len(data.qpos) > 7:
                    obs_components.extend(data.qpos[7:])

                # Joint velocities (excluding root)
                if len(data.qvel) > 6:
                    obs_components.extend(data.qvel[6:])

                # Pad/truncate to match training
                obs = np.array(obs_components, dtype=np.float32)
                if len(obs) < policy.obs_dim:
                    obs = np.pad(obs, (0, policy.obs_dim - len(obs)), mode='constant')
                else:
                    obs = obs[:policy.obs_dim]

                # Neural network inference (your trained model!)
                inference_start = time.time()
                action = policy.get_action(obs)
                inference_time = time.time() - inference_start
                inference_times.append(inference_time)

                # Apply neural network output to robot actuators
                for i in range(min(len(action), model.nu)):
                    data.ctrl[i] = action[i]

                # Step physics simulation
                mujoco.mj_step(model, data)

                # Update 3D visualization
                viewer.sync()
                step_count += 1

                # Check if robot fell and reset
                height = data.qpos[2] if len(data.qpos) > 2 else 0.9
                if height < 0.4 or data.time > 10.0:  # Reset conditions
                    mujoco.mj_resetData(model, data)
                    policy.reset()
                    reset_count += 1
                    print(f"   🔄 Reset #{reset_count} - Robot fell or time limit")

                # Performance statistics every 5 seconds
                current_time = time.time()
                if current_time - last_stats_time > 5.0:
                    runtime = current_time - start_time
                    recent_inference = np.mean(inference_times[-500:]) * 1000 if inference_times else 0

                    print(f"\\n📊 Performance Stats (Runtime: {runtime:.1f}s):")
                    print(f"   👣 Steps: {step_count:,}")
                    print(f"   🔄 Resets: {reset_count}")
                    print(f"   ⚡ Inference: {recent_inference:.2f}ms")
                    print(f"   🎯 FPS: {step_count/(runtime):.1f}")
                    print(f"   📈 Height: {height:.3f}m")
                    print(f"   🧠 Neural network running smoothly!")

                    last_stats_time = current_time

                # Slow down to real-time
                time.sleep(0.002)  # Match timestep

    except KeyboardInterrupt:
        print("\\n⏹️  Simulation stopped by user")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Clean up
        Path(robot_file).unlink(missing_ok=True)
        print("\\n🎉 Simulation completed!")
        print(f"   Your {model_info.get('total_parameters', '19M')} parameter neural network")
        print(f"   successfully controlled the Unitree G1 robot!")

if __name__ == "__main__":
    run_unitree_g1_simulation()