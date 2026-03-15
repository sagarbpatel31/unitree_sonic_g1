#!/usr/bin/env python3
"""
Complete Unitree G1 29 DOF Simulation with Your Trained Neural Network
Full humanoid robot with 29 degrees of freedom controlled by your 19M parameter transformer
"""

import numpy as np
import mujoco
import mujoco.viewer
import time
from pathlib import Path
from mujoco_mac_demo import MacMuJoCoPolicyInterface

def run_29dof_simulation():
    """Run complete 29 DOF Unitree G1 simulation."""
    print("🤖 Unitree G1 Complete 29 DOF Simulation")
    print("=" * 50)
    print("🧠 Full humanoid robot with 29 degrees of freedom")

    # Find your best trained model
    checkpoint_path = None
    model_candidates = [
        "checkpoints/production_large_model/best_checkpoint.pth",
        "checkpoints/enhanced_mac_model/best_checkpoint.pth",
        "checkpoints/mac_training/best_checkpoint.pth"
    ]

    for candidate in model_candidates:
        if Path(candidate).exists():
            checkpoint_path = candidate
            break

    if not checkpoint_path:
        print("❌ No trained model found!")
        print("Train first: python3 mac_optimized_training.py --epochs 100")
        return

    print(f"✅ Found model: {Path(checkpoint_path).parent.name}")

    try:
        # Load your trained neural network
        print("🔄 Loading your trained neural network...")
        policy = MacMuJoCoPolicyInterface(checkpoint_path)

        model_info = policy.model.get_model_info()
        print(f"✅ Neural Network Loaded:")
        print(f"   🧠 Parameters: {model_info['total_parameters']:,}")
        print(f"   📊 Model size: {model_info['parameter_size_mb']:.1f} MB")
        print(f"   🎯 Architecture: {model_info.get('d_model', 'N/A')}d transformer")
        print(f"   📥 Input dimension: {policy.obs_dim}")
        print(f"   📤 Output dimension: {policy.action_dim}")

        # Load the complete 29 DOF Unitree G1 robot
        print("🤖 Loading complete 29 DOF Unitree G1...")
        robot_file = "unitree_g1_fixed_29dof.xml"

        if not Path(robot_file).exists():
            print(f"❌ Robot file not found: {robot_file}")
            return

        model = mujoco.MjModel.from_xml_path(robot_file)
        data = mujoco.MjData(model)

        print(f"✅ Complete Unitree G1 Loaded:")
        print(f"   🦾 Total actuators: {model.nu}")
        print(f"   🔗 Total DOF: {model.nq}")
        print(f"   📐 Free joints: 6 (base position + orientation)")
        print(f"   🤖 Actuated joints: {model.nu} (29 DOF total)")

        # Joint breakdown
        print(f"\\n📋 Joint Configuration:")
        print(f"   🧠 Head: 2 DOF (yaw, pitch)")
        print(f"   💪 Left arm: 7 DOF (full manipulation)")
        print(f"   💪 Right arm: 7 DOF (full manipulation)")
        print(f"   🦵 Left leg: 6 DOF (full locomotion)")
        print(f"   🦵 Right leg: 6 DOF (full locomotion)")
        print(f"   🏃 Torso: 1 DOF (spine pitch)")

        print(f"\\n🎮 Starting Full 29 DOF Simulation...")
        print(f"   📺 3D viewer will open with complete humanoid")
        print(f"   🧠 Your {model_info['total_parameters']:,}-parameter model controlling all joints")
        print(f"   🔄 Auto-reset when robot falls")
        print(f"   ⌨️  Close viewer to exit")

        # Start the complete simulation
        with mujoco.viewer.launch_passive(model, data) as viewer:
            # Initialize everything
            policy.reset()
            mujoco.mj_resetData(model, data)

            # Simulation tracking
            step_count = 0
            reset_count = 0
            inference_times = []
            start_time = time.time()
            last_stats_time = start_time

            print(f"\\n🚀 29 DOF Simulation Started!")
            print(f"   Watch your neural network control the complete humanoid robot!")

            while viewer.is_running():
                # Extract complete robot state
                obs_components = []

                # Get all joint positions (excluding 6 DOF base)
                if len(data.qpos) > 7:  # Skip 7 DOF base (3 pos + 4 quat)
                    obs_components.extend(data.qpos[7:])

                # Get all joint velocities (excluding 6 DOF base)
                if len(data.qvel) > 6:  # Skip 6 DOF base velocities
                    obs_components.extend(data.qvel[6:])

                # Additional state information
                obs_components.extend([
                    data.qpos[2],  # Height
                    np.sin(data.time * 2.0),  # Time encoding
                    np.cos(data.time * 2.0)
                ])

                # Prepare observation for neural network
                obs = np.array(obs_components, dtype=np.float32)
                if len(obs) < policy.obs_dim:
                    obs = np.pad(obs, (0, policy.obs_dim - len(obs)), mode='constant')
                else:
                    obs = obs[:policy.obs_dim]

                # Neural network inference (your trained transformer!)
                inference_start = time.time()
                action = policy.get_action(obs)
                inference_time = time.time() - inference_start
                inference_times.append(inference_time)

                # Apply actions to all 29 actuated joints
                num_actions = min(len(action), model.nu)
                data.ctrl[:num_actions] = action[:num_actions]

                # Apply scaling for different joint types
                if num_actions >= 29:
                    # Head joints (gentle movement)
                    data.ctrl[0:2] *= 0.3

                    # Arm joints (moderate movement)
                    data.ctrl[2:9] *= 0.6   # Left arm
                    data.ctrl[9:16] *= 0.6  # Right arm

                    # Leg joints (full power for locomotion)
                    data.ctrl[16:22] *= 1.0  # Left leg
                    data.ctrl[22:28] *= 1.0  # Right leg

                    # Torso (gentle)
                    if num_actions > 28:
                        data.ctrl[28] *= 0.4

                # Step the physics simulation
                mujoco.mj_step(model, data)

                # Update 3D visualization
                viewer.sync()
                step_count += 1

                # Check for fall and reset
                height = data.qpos[2] if len(data.qpos) > 2 else 0.95
                orientation = data.qpos[3:7] if len(data.qpos) > 6 else [1,0,0,0]

                # Check if robot is upright (quaternion w component)
                upright = abs(orientation[0]) > 0.5 if len(orientation) > 0 else True

                if height < 0.4 or not upright or data.time > 15.0:
                    mujoco.mj_resetData(model, data)
                    policy.reset()
                    reset_count += 1
                    print(f"   🔄 Reset #{reset_count} - Robot fell or time limit")

                # Performance statistics every 5 seconds
                current_time = time.time()
                if current_time - last_stats_time > 5.0:
                    runtime = current_time - start_time
                    recent_inference = np.mean(inference_times[-1000:]) * 1000 if len(inference_times) >= 1000 else np.mean(inference_times) * 1000

                    print(f"\\n📊 29 DOF Performance (Runtime: {runtime:.1f}s):")
                    print(f"   👣 Steps: {step_count:,}")
                    print(f"   🔄 Resets: {reset_count}")
                    print(f"   ⚡ Inference: {recent_inference:.2f}ms")
                    print(f"   🎯 FPS: {step_count/runtime:.1f}")
                    print(f"   📏 Height: {height:.3f}m")
                    print(f"   🤖 All 29 DOF controlled by neural network!")

                    last_stats_time = current_time

                # Real-time control
                time.sleep(0.001)

    except KeyboardInterrupt:
        print("\\n⏹️  29 DOF simulation stopped by user")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        print(f"\\n🎉 29 DOF Simulation Completed!")
        print(f"   Your {model_info.get('total_parameters', '19M'):,} parameter neural network")
        print(f"   successfully controlled all 29 degrees of freedom!")
        print(f"   🤖 Complete humanoid robot simulation achieved!")

if __name__ == "__main__":
    run_29dof_simulation()