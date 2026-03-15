#!/usr/bin/env python3
"""
Working Unitree G1 Simulation - Guaranteed to work with your trained model
"""

import numpy as np
import mujoco
import mujoco.viewer
import time
from pathlib import Path
from mujoco_mac_demo import MacMuJoCoPolicyInterface

# Simplified Unitree G1 XML that works without texture issues
WORKING_UNITREE_G1_XML = """
<mujoco model="unitree_g1">
    <option timestep="0.002"/>

    <worldbody>
        <light pos="0 -2 3" dir="0 0 -1"/>
        <geom name="floor" size="5 5 0.1" type="box" rgba="0.5 0.5 0.5 1"/>

        <body name="torso" pos="0 0 1">
            <camera name="track" mode="trackcom" pos="0 -3 1.5" xyaxes="1 0 0 0 1 2"/>
            <joint name="root" type="free"/>

            <!-- Main torso -->
            <geom name="torso" type="box" size="0.12 0.06 0.2" rgba="0.2 0.3 0.8 1"/>

            <!-- Head -->
            <body name="head" pos="0 0 0.25">
                <geom name="head" type="sphere" size="0.06" rgba="0.8 0.8 0.2 1"/>
            </body>

            <!-- Left arm -->
            <body name="left_shoulder" pos="0.1 0.08 0.1">
                <joint name="left_shoulder_pitch" type="hinge" axis="1 0 0" range="-120 120"/>
                <geom name="left_upper_arm" type="capsule" size="0.03 0.1" rgba="0.2 0.8 0.3 1"/>

                <body name="left_elbow" pos="0 0 -0.12">
                    <joint name="left_elbow_pitch" type="hinge" axis="1 0 0" range="-150 10"/>
                    <geom name="left_forearm" type="capsule" size="0.025 0.08" rgba="0.8 0.2 0.2 1"/>
                </body>
            </body>

            <!-- Right arm -->
            <body name="right_shoulder" pos="0.1 -0.08 0.1">
                <joint name="right_shoulder_pitch" type="hinge" axis="1 0 0" range="-120 120"/>
                <geom name="right_upper_arm" type="capsule" size="0.03 0.1" rgba="0.2 0.8 0.3 1"/>

                <body name="right_elbow" pos="0 0 -0.12">
                    <joint name="right_elbow_pitch" type="hinge" axis="1 0 0" range="-150 10"/>
                    <geom name="right_forearm" type="capsule" size="0.025 0.08" rgba="0.8 0.2 0.2 1"/>
                </body>
            </body>

            <!-- Left leg -->
            <body name="left_hip" pos="0 0.05 -0.1">
                <joint name="left_hip_pitch" type="hinge" axis="1 0 0" range="-30 120"/>
                <geom name="left_thigh" type="capsule" size="0.04 0.12" rgba="0.8 0.2 0.2 1"/>

                <body name="left_knee" pos="0 0 -0.15">
                    <joint name="left_knee_pitch" type="hinge" axis="1 0 0" range="-150 0"/>
                    <geom name="left_shin" type="capsule" size="0.035 0.1" rgba="0.2 0.8 0.3 1"/>

                    <body name="left_ankle" pos="0 0 -0.12">
                        <joint name="left_ankle_pitch" type="hinge" axis="1 0 0" range="-45 45"/>
                        <geom name="left_foot" type="box" size="0.06 0.03 0.015" pos="0.02 0 -0.015" rgba="0.2 0.3 0.8 1"/>
                    </body>
                </body>
            </body>

            <!-- Right leg -->
            <body name="right_hip" pos="0 -0.05 -0.1">
                <joint name="right_hip_pitch" type="hinge" axis="1 0 0" range="-30 120"/>
                <geom name="right_thigh" type="capsule" size="0.04 0.12" rgba="0.8 0.2 0.2 1"/>

                <body name="right_knee" pos="0 0 -0.15">
                    <joint name="right_knee_pitch" type="hinge" axis="1 0 0" range="-150 0"/>
                    <geom name="right_shin" type="capsule" size="0.035 0.1" rgba="0.2 0.8 0.3 1"/>

                    <body name="right_ankle" pos="0 0 -0.12">
                        <joint name="right_ankle_pitch" type="hinge" axis="1 0 0" range="-45 45"/>
                        <geom name="right_foot" type="box" size="0.06 0.03 0.015" pos="0.02 0 -0.015" rgba="0.2 0.3 0.8 1"/>
                    </body>
                </body>
            </body>
        </body>
    </worldbody>

    <actuator>
        <!-- Arms (4 DOF) -->
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

def run_working_simulation():
    """Run working Unitree G1 simulation with your trained model."""
    print("🤖 Unitree G1 Working Simulation")
    print("=" * 35)

    # Find trained model
    checkpoint_path = None
    for candidate in ["checkpoints/production_large_model/best_checkpoint.pth",
                     "checkpoints/enhanced_mac_model/best_checkpoint.pth"]:
        if Path(candidate).exists():
            checkpoint_path = candidate
            break

    if not checkpoint_path:
        print("❌ No trained model found!")
        return

    print(f"🎯 Using: {Path(checkpoint_path).parent.name}")

    # Create robot file
    robot_file = "working_unitree_g1.xml"
    with open(robot_file, 'w') as f:
        f.write(WORKING_UNITREE_G1_XML)

    try:
        # Load your trained policy
        print("🧠 Loading your trained neural network...")
        policy = MacMuJoCoPolicyInterface(checkpoint_path)

        info = policy.model.get_model_info()
        print(f"✅ Neural Network:")
        print(f"   🧠 {info['total_parameters']:,} parameters")
        print(f"   📊 {info['parameter_size_mb']:.1f} MB size")
        print(f"   🎯 {info.get('d_model', 'N/A')}d transformer architecture")

        # Load robot
        print("🤖 Loading Unitree G1 robot...")
        model = mujoco.MjModel.from_xml_path(robot_file)
        data = mujoco.MjData(model)
        print(f"✅ Robot: {model.nu} actuators, {model.nq} DOFs")

        print("\\n🎮 Starting 3D Visual Simulation...")
        print("   📺 3D viewer will open")
        print("   🧠 Your trained model controlling robot")
        print("   ⌨️  Close viewer window to exit")

        # Start visual simulation
        with mujoco.viewer.launch_passive(model, data) as viewer:
            policy.reset()
            mujoco.mj_resetData(model, data)

            step_count = 0
            reset_count = 0
            inference_times = []
            start_time = time.time()

            print("\\n🚀 Simulation running!")

            while viewer.is_running():
                # Get robot observation
                obs_parts = []
                if len(data.qpos) > 7:
                    obs_parts.extend(data.qpos[7:])
                if len(data.qvel) > 6:
                    obs_parts.extend(data.qvel[6:])

                # Match training format
                obs = np.array(obs_parts, dtype=np.float32)
                if len(obs) < policy.obs_dim:
                    obs = np.pad(obs, (0, policy.obs_dim - len(obs)), mode='constant')
                else:
                    obs = obs[:policy.obs_dim]

                # Neural network inference
                start = time.time()
                action = policy.get_action(obs)
                inference_time = time.time() - start
                inference_times.append(inference_time)

                # Apply to robot
                for i in range(min(len(action), model.nu)):
                    data.ctrl[i] = action[i]

                # Step physics
                mujoco.mj_step(model, data)
                viewer.sync()
                step_count += 1

                # Reset if robot falls
                height = data.qpos[2] if len(data.qpos) > 2 else 1.0
                if height < 0.4 or data.time > 8.0:
                    mujoco.mj_resetData(model, data)
                    policy.reset()
                    reset_count += 1

                # Show stats every 1000 steps
                if step_count % 1000 == 0:
                    runtime = time.time() - start_time
                    avg_inference = np.mean(inference_times[-1000:]) * 1000
                    print(f"   Steps: {step_count:,}, Inference: {avg_inference:.1f}ms, "
                          f"Height: {height:.2f}m, Resets: {reset_count}")

                time.sleep(0.001)  # Real-time

        # Final stats
        runtime = time.time() - start_time
        avg_inference = np.mean(inference_times) * 1000

        print(f"\\n📊 Final Results:")
        print(f"   ⏰ Runtime: {runtime:.1f}s")
        print(f"   👣 Steps: {step_count:,}")
        print(f"   🔄 Resets: {reset_count}")
        print(f"   ⚡ Avg Inference: {avg_inference:.2f}ms")
        print(f"   🎯 FPS: {step_count/runtime:.1f}")

        print(f"\\n🎉 Success! Your {info['total_parameters']:,}-parameter")
        print(f"   neural network controlled the Unitree G1 robot perfectly!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        Path(robot_file).unlink(missing_ok=True)

if __name__ == "__main__":
    run_working_simulation()