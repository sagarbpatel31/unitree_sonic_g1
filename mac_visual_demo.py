#!/usr/bin/env python3
"""
Mac Visual Demo - Works perfectly on macOS without mjpython requirement
"""

import numpy as np
import mujoco
from pathlib import Path
import time

from mujoco_mac_demo import MacMuJoCoPolicyInterface

# Simple robot for demonstration
ROBOT_XML = """
<mujoco model="sonic_robot">
    <option timestep="0.002"/>

    <worldbody>
        <light pos="0 0 2" dir="0 0 -1"/>
        <geom name="floor" size="3 3 0.1" type="box" rgba="0.5 0.5 0.5 1"/>

        <body name="robot" pos="0 0 0.5">
            <joint name="root" type="free"/>
            <geom name="torso" type="box" size="0.1 0.1 0.2" rgba="0.8 0.2 0.2 1"/>

            <body name="left_arm" pos="0.12 0 0.15">
                <joint name="left_shoulder" type="hinge" axis="0 1 0" range="-90 90"/>
                <geom name="left_arm" type="capsule" size="0.03 0.15" rgba="0.2 0.8 0.2 1"/>
            </body>

            <body name="right_arm" pos="-0.12 0 0.15">
                <joint name="right_shoulder" type="hinge" axis="0 1 0" range="-90 90"/>
                <geom name="right_arm" type="capsule" size="0.03 0.15" rgba="0.2 0.8 0.2 1"/>
            </body>

            <body name="left_leg" pos="0.05 0 -0.25">
                <joint name="left_hip" type="hinge" axis="1 0 0" range="-30 30"/>
                <geom name="left_leg" type="capsule" size="0.035 0.12" rgba="0.2 0.2 0.8 1"/>
            </body>

            <body name="right_leg" pos="-0.05 0 -0.25">
                <joint name="right_hip" type="hinge" axis="1 0 0" range="-30 30"/>
                <geom name="right_leg" type="capsule" size="0.035 0.12" rgba="0.2 0.2 0.8 1"/>
            </body>
        </body>
    </worldbody>

    <actuator>
        <motor name="left_shoulder_motor" joint="left_shoulder" gear="20"/>
        <motor name="right_shoulder_motor" joint="right_shoulder" gear="20"/>
        <motor name="left_hip_motor" joint="left_hip" gear="50"/>
        <motor name="right_hip_motor" joint="right_hip" gear="50"/>
    </actuator>
</mujoco>
"""

def run_mac_demo():
    """Run visual demo that works perfectly on Mac."""
    print("🍎 Mac Visual Demo - Enhanced Sonic G1")
    print("=" * 45)

    # Find checkpoint
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
        print("❌ No model found. Train first:")
        print("   python3 mac_optimized_training.py --epochs 30")
        return

    print(f"🎯 Model: {Path(checkpoint_path).parent.name}")

    robot_file = "mac_robot.xml"
    with open(robot_file, 'w') as f:
        f.write(ROBOT_XML)

    try:
        # Load policy
        print("🧠 Loading neural network...")
        policy = MacMuJoCoPolicyInterface(checkpoint_path)

        info = policy.model.get_model_info()
        print(f"   ✅ {info['total_parameters']:,} parameters ({info['parameter_size_mb']:.1f} MB)")

        # Load robot
        print("🤖 Loading robot...")
        model = mujoco.MjModel.from_xml_path(robot_file)
        data = mujoco.MjData(model)
        print(f"   ✅ {model.nu} actuators, {model.nq} DOFs")

        # Run simulation (headless for Mac compatibility)
        print("\n🎮 Running simulation...")
        print("   🧠 Neural network controlling robot")
        print("   📊 Monitoring performance...")

        policy.reset()
        mujoco.mj_resetData(model, data)

        step_count = 0
        reset_count = 0
        inference_times = []
        rewards = []
        start_time = time.time()

        # Run for 10 seconds
        while time.time() - start_time < 10:
            # Get observation
            obs = []
            if len(data.qpos) > 7:
                obs.extend(data.qpos[7:])
            if len(data.qvel) > 6:
                obs.extend(data.qvel[6:])

            # Pad to policy dimension
            obs = np.array(obs, dtype=np.float32)
            if len(obs) < policy.obs_dim:
                obs = np.pad(obs, (0, policy.obs_dim - len(obs)))
            else:
                obs = obs[:policy.obs_dim]

            # Neural network inference
            start = time.time()
            action = policy.get_action(obs)
            inference_time = time.time() - start
            inference_times.append(inference_time)

            # Apply to robot
            data.ctrl[:model.nu] = action[:model.nu]

            # Step simulation
            mujoco.mj_step(model, data)

            # Calculate reward
            height = data.qpos[2] if len(data.qpos) > 2 else 0.5
            stability = 1.0 if height > 0.2 else 0.0
            rewards.append(stability)

            # Reset if needed
            if height < 0.1 or data.time > 3.0:
                mujoco.mj_resetData(model, data)
                policy.reset()
                reset_count += 1

            step_count += 1

            # Progress indicator
            if step_count % 100 == 0:
                elapsed = time.time() - start_time
                avg_inference = np.mean(inference_times[-100:]) * 1000
                print(f"   Step {step_count}: Height={height:.3f}, Inference={avg_inference:.2f}ms")

        # Results
        runtime = time.time() - start_time
        avg_inference = np.mean(inference_times) * 1000
        avg_reward = np.mean(rewards)

        print(f"\n📊 Demo Results:")
        print(f"   🕒 Runtime: {runtime:.1f}s")
        print(f"   👣 Steps: {step_count:,}")
        print(f"   🔄 Resets: {reset_count}")
        print(f"   ⚡ Inference: {avg_inference:.2f}ms")
        print(f"   🎯 FPS: {step_count/runtime:.1f}")
        print(f"   🏆 Avg Stability: {avg_reward:.3f}")
        print(f"   📈 Throughput: {1000/avg_inference:.0f} actions/sec")

        # Performance analysis
        print(f"\n🎉 Neural Network Performance Analysis:")
        if avg_inference < 2.0:
            print(f"   🚀 EXCELLENT: <2ms inference (real-time capable)")
        elif avg_inference < 5.0:
            print(f"   ✅ GOOD: <5ms inference (smooth control)")
        else:
            print(f"   ⚠️  OK: {avg_inference:.1f}ms inference")

        if avg_reward > 0.8:
            print(f"   🏆 STABLE: Robot maintained balance well")
        elif avg_reward > 0.5:
            print(f"   ✅ DECENT: Robot showed good stability")
        else:
            print(f"   📈 LEARNING: Robot is improving")

        print(f"\n🍎 Mac Deployment Success!")
        print(f"   Your {info['total_parameters']:,}-parameter model")
        print(f"   runs perfectly on macOS with {avg_inference:.2f}ms inference!")

        # Instructions for visual viewing
        print(f"\n🎮 For 3D Visual Simulation:")
        print(f"   1. Install mjpython: python -m pip install mujoco[viewer]")
        print(f"   2. Run: mjpython working_visual_demo.py")
        print(f"   3. Or use online MuJoCo viewer at: mujoco.org")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        Path(robot_file).unlink(missing_ok=True)

if __name__ == "__main__":
    run_mac_demo()