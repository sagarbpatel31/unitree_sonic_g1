#!/usr/bin/env python3
"""
Visual MuJoCo Demo with Interactive UI
Complete visual simulation showing your trained robot model in action.
"""

import json
import time
import numpy as np
from pathlib import Path
import threading
import tkinter as tk
from tkinter import ttk, messagebox

import mujoco
import mujoco.viewer

from mujoco_mac_demo import MacMuJoCoPolicyInterface


# Simple humanoid robot XML for visualization
ROBOT_XML = """
<mujoco model="enhanced_humanoid">
    <option timestep="0.002"/>

    <asset>
        <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="512"/>
        <texture name="body" type="cube" builtin="flat" mark="cross" width="127" height="127"
                 rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" markrgb="1 1 1" random="0.01"/>
        <material name="body" texture="body" texuniform="true" rgba="0.8 0.6 .4 1"/>
        <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1=".1 .2 .3" rgb2=".2 .3 .4"/>
        <material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance=".2"/>
    </asset>

    <worldbody>
        <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
        <geom name="floor" size="0 0 .05" type="plane" material="grid"/>

        <body name="torso" pos="0 0 1.4">
            <joint name="root" type="free"/>
            <geom name="torso" type="capsule" size="0.07 0.28" material="body"/>

            <!-- Head -->
            <body name="head" pos="0 0 0.19">
                <geom name="head" type="sphere" size="0.09" material="body"/>
            </body>

            <!-- Left arm -->
            <body name="left_upper_arm" pos="0.13 -0.1 0.1">
                <joint name="left_shoulder_x" type="hinge" axis="1 0 0" range="-85 60"/>
                <joint name="left_shoulder_y" type="hinge" axis="0 1 0" range="-85 85"/>
                <geom name="left_upper_arm" type="capsule" size="0.04 0.11" material="body"/>

                <body name="left_lower_arm" pos="0 0 -0.15">
                    <joint name="left_elbow" type="hinge" axis="0 1 0" range="-90 50"/>
                    <geom name="left_lower_arm" type="capsule" size="0.031 0.1" material="body"/>

                    <body name="left_hand" pos="0 0 -0.1">
                        <geom name="left_hand" type="sphere" size="0.04" material="body"/>
                    </body>
                </body>
            </body>

            <!-- Right arm -->
            <body name="right_upper_arm" pos="-0.13 -0.1 0.1">
                <joint name="right_shoulder_x" type="hinge" axis="1 0 0" range="-60 85"/>
                <joint name="right_shoulder_y" type="hinge" axis="0 1 0" range="-85 85"/>
                <geom name="right_upper_arm" type="capsule" size="0.04 0.11" material="body"/>

                <body name="right_lower_arm" pos="0 0 -0.15">
                    <joint name="right_elbow" type="hinge" axis="0 1 0" range="-90 50"/>
                    <geom name="right_lower_arm" type="capsule" size="0.031 0.1" material="body"/>

                    <body name="right_hand" pos="0 0 -0.1">
                        <geom name="right_hand" type="sphere" size="0.04" material="body"/>
                    </body>
                </body>
            </body>

            <!-- Left leg -->
            <body name="left_upper_leg" pos="0.05 0 -0.35">
                <joint name="left_hip_x" type="hinge" axis="1 0 0" range="-25 5"/>
                <joint name="left_hip_y" type="hinge" axis="0 1 0" range="-25 25"/>
                <joint name="left_hip_z" type="hinge" axis="0 0 1" range="-60 35"/>
                <geom name="left_upper_leg" type="capsule" size="0.04 0.14" material="body"/>

                <body name="left_lower_leg" pos="0 0 -0.18">
                    <joint name="left_knee" type="hinge" axis="0 1 0" range="-160 -2"/>
                    <geom name="left_lower_leg" type="capsule" size="0.033 0.12" material="body"/>

                    <body name="left_foot" pos="0 0 -0.1">
                        <geom name="left_foot" type="capsule" size="0.027 0.06" material="body"/>
                    </body>
                </body>
            </body>

            <!-- Right leg -->
            <body name="right_upper_leg" pos="-0.05 0 -0.35">
                <joint name="right_hip_x" type="hinge" axis="1 0 0" range="-25 5"/>
                <joint name="right_hip_y" type="hinge" axis="0 1 0" range="-25 25"/>
                <joint name="right_hip_z" type="hinge" axis="0 0 1" range="-35 60"/>
                <geom name="right_upper_leg" type="capsule" size="0.04 0.14" material="body"/>

                <body name="right_lower_leg" pos="0 0 -0.18">
                    <joint name="right_knee" type="hinge" axis="0 1 0" range="-160 -2"/>
                    <geom name="right_lower_leg" type="capsule" size="0.033 0.12" material="body"/>

                    <body name="right_foot" pos="0 0 -0.1">
                        <geom name="right_foot" type="capsule" size="0.027 0.06" material="body"/>
                    </body>
                </body>
            </body>
        </body>
    </worldbody>

    <actuator>
        <!-- Arms -->
        <motor name="left_shoulder_x" joint="left_shoulder_x" gear="25"/>
        <motor name="left_shoulder_y" joint="left_shoulder_y" gear="25"/>
        <motor name="left_elbow" joint="left_elbow" gear="25"/>
        <motor name="right_shoulder_x" joint="right_shoulder_x" gear="25"/>
        <motor name="right_shoulder_y" joint="right_shoulder_y" gear="25"/>
        <motor name="right_elbow" joint="right_elbow" gear="25"/>

        <!-- Legs -->
        <motor name="left_hip_x" joint="left_hip_x" gear="120"/>
        <motor name="left_hip_y" joint="left_hip_y" gear="120"/>
        <motor name="left_hip_z" joint="left_hip_z" gear="120"/>
        <motor name="left_knee" joint="left_knee" gear="120"/>
        <motor name="right_hip_x" joint="right_hip_x" gear="120"/>
        <motor name="right_hip_y" joint="right_hip_y" gear="120"/>
        <motor name="right_hip_z" joint="right_hip_z" gear="120"/>
        <motor name="right_knee" joint="right_knee" gear="120"/>
    </actuator>
</mujoco>
"""

class VisualMuJoCoDemo:
    """Interactive MuJoCo simulation with visual UI."""

    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path
        self.running = False
        self.paused = False
        self.model = None
        self.data = None
        self.policy = None
        self.viewer = None

        # Simulation statistics
        self.reset_stats()

        # Create robot model file
        self.robot_file = "enhanced_humanoid.xml"
        with open(self.robot_file, 'w') as f:
            f.write(ROBOT_XML)

        # Load policy
        try:
            self.policy = MacMuJoCoPolicyInterface(checkpoint_path)
            print(f"✅ Policy loaded: {self.policy.model.get_model_info()['total_parameters']:,} params")
        except Exception as e:
            print(f"❌ Error loading policy: {e}")
            return

        # Initialize MuJoCo
        self._init_mujoco()

        # Create UI
        self._create_ui()

    def reset_stats(self):
        """Reset simulation statistics."""
        self.step_count = 0
        self.episode_count = 0
        self.total_reward = 0.0
        self.episode_reward = 0.0
        self.inference_times = []
        self.start_time = time.time()

    def _init_mujoco(self):
        """Initialize MuJoCo simulation."""
        try:
            self.model = mujoco.MjModel.from_xml_path(self.robot_file)
            self.data = mujoco.MjData(self.model)
            print(f"✅ MuJoCo model loaded: {self.model.nu} actuators, {self.model.nq} DOFs")
        except Exception as e:
            print(f"❌ Error loading MuJoCo model: {e}")

    def _create_ui(self):
        """Create the control UI."""
        self.root = tk.Tk()
        self.root.title("🤖 Enhanced Sonic G1 - Visual MuJoCo Demo")
        self.root.geometry("800x600")
        self.root.configure(bg='#2b2b2b')

        # Style configuration
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'), foreground='white', background='#2b2b2b')
        style.configure('Info.TLabel', font=('Arial', 10), foreground='lightgreen', background='#2b2b2b')
        style.configure('Control.TButton', font=('Arial', 12))

        # Title
        title_label = ttk.Label(self.root, text="🚀 Enhanced Sonic G1 Training Demo", style='Title.TLabel')
        title_label.pack(pady=20)

        # Model information
        model_info = self.policy.model.get_model_info()
        info_frame = tk.Frame(self.root, bg='#2b2b2b')
        info_frame.pack(pady=10)

        ttk.Label(info_frame, text=f"Model: {model_info['total_parameters']:,} parameters",
                 style='Info.TLabel').pack()
        ttk.Label(info_frame, text=f"Size: {model_info['parameter_size_mb']:.1f} MB",
                 style='Info.TLabel').pack()
        ttk.Label(info_frame, text=f"Architecture: {model_info.get('d_model', 'N/A')}d, {model_info.get('num_layers', 'N/A')} layers",
                 style='Info.TLabel').pack()

        # Control buttons
        button_frame = tk.Frame(self.root, bg='#2b2b2b')
        button_frame.pack(pady=20)

        self.start_button = ttk.Button(button_frame, text="🚀 Start Simulation",
                                      command=self.start_simulation, style='Control.TButton')
        self.start_button.grid(row=0, column=0, padx=10)

        self.pause_button = ttk.Button(button_frame, text="⏸️ Pause",
                                      command=self.pause_simulation, style='Control.TButton')
        self.pause_button.grid(row=0, column=1, padx=10)

        self.reset_button = ttk.Button(button_frame, text="🔄 Reset",
                                      command=self.reset_simulation, style='Control.TButton')
        self.reset_button.grid(row=0, column=2, padx=10)

        self.stop_button = ttk.Button(button_frame, text="⏹️ Stop",
                                     command=self.stop_simulation, style='Control.TButton')
        self.stop_button.grid(row=0, column=3, padx=10)

        # Statistics display
        stats_frame = tk.LabelFrame(self.root, text="📊 Simulation Statistics",
                                   bg='#2b2b2b', fg='white', font=('Arial', 12, 'bold'))
        stats_frame.pack(pady=20, padx=20, fill='both', expand=True)

        self.stats_text = tk.Text(stats_frame, height=15, width=80,
                                 bg='#1e1e1e', fg='lightgreen', font=('Consolas', 10))
        self.stats_text.pack(padx=10, pady=10, fill='both', expand=True)

        # Real-time updates
        self.update_ui()

    def update_ui(self):
        """Update the UI with current statistics."""
        if hasattr(self, 'stats_text'):
            self.stats_text.delete(1.0, tk.END)

            runtime = time.time() - self.start_time
            avg_inference = np.mean(self.inference_times[-100:]) if self.inference_times else 0.0

            stats = f"""🤖 Enhanced Sonic G1 Simulation Status 🤖

Runtime: {runtime:.1f}s
Status: {"🔴 Stopped" if not self.running else "⏸️ Paused" if self.paused else "🟢 Running"}

📈 Performance Metrics:
├── Steps: {self.step_count:,}
├── Episodes: {self.episode_count}
├── Total Reward: {self.total_reward:.2f}
├── Episode Reward: {self.episode_reward:.2f}
├── Avg Inference: {avg_inference*1000:.2f}ms
└── Simulation FPS: {self.step_count/max(runtime, 1):.1f}

🧠 Model Information:
├── Parameters: {self.policy.model.get_model_info()['total_parameters']:,}
├── Model Size: {self.policy.model.get_model_info()['parameter_size_mb']:.1f} MB
├── Obs Dimension: {self.policy.obs_dim}
├── Action Dimension: {self.policy.action_dim}
└── Sequence Length: {self.policy.max_history}

🎮 Controls:
├── Start: Begin/Resume simulation
├── Pause: Pause simulation
├── Reset: Reset robot to initial position
└── Stop: Stop simulation and close viewer

💡 Tips:
├── The robot uses your trained neural network for control
├── Green = Good performance, Red = Robot fell
├── Higher FPS = Better Mac performance
└── Close this window to exit completely
"""
            self.stats_text.insert(tk.END, stats)

        # Schedule next update
        self.root.after(100, self.update_ui)

    def start_simulation(self):
        """Start the MuJoCo simulation."""
        if self.running:
            self.paused = False
            return

        self.running = True
        self.paused = False

        # Reset simulation
        self.reset_simulation()

        # Start simulation in separate thread
        self.sim_thread = threading.Thread(target=self._simulation_loop, daemon=True)
        self.sim_thread.start()

        # Create viewer
        if self.viewer is None:
            try:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                print("✅ MuJoCo viewer launched")
            except Exception as e:
                print(f"❌ Error launching viewer: {e}")

    def pause_simulation(self):
        """Pause/unpause simulation."""
        self.paused = not self.paused

    def reset_simulation(self):
        """Reset the simulation."""
        if self.model and self.data:
            mujoco.mj_resetData(self.model, self.data)

            # Add small random noise for diversity
            self.data.qpos += np.random.normal(0, 0.01, self.data.qpos.shape)
            mujoco.mj_forward(self.model, self.data)

            if self.policy:
                self.policy.reset()

            # Reset episode stats
            self.episode_reward = 0.0
            self.episode_count += 1

    def stop_simulation(self):
        """Stop the simulation."""
        self.running = False
        self.paused = False

        if self.viewer:
            try:
                self.viewer.close()
                self.viewer = None
            except:
                pass

    def _simulation_loop(self):
        """Main simulation loop."""
        print("🚀 Starting simulation loop...")

        while self.running:
            if self.paused:
                time.sleep(0.1)
                continue

            try:
                # Get observation
                obs = self._get_observation()

                # Get action from policy
                start_time = time.time()
                action = self.policy.get_action(obs)
                inference_time = time.time() - start_time
                self.inference_times.append(inference_time)

                # Apply action
                self.data.ctrl[:len(action)] = action[:self.model.nu]

                # Step simulation
                mujoco.mj_step(self.model, self.data)

                # Update viewer
                if self.viewer:
                    self.viewer.sync()

                # Calculate reward (simple stability reward)
                height = self.data.qpos[2] if len(self.data.qpos) > 2 else 1.0
                reward = 1.0 if height > 0.8 else 0.0
                self.episode_reward += reward
                self.total_reward += reward

                # Update counters
                self.step_count += 1

                # Check for reset conditions
                if height < 0.5 or self.data.time > 10.0:
                    print(f"Episode {self.episode_count} ended: {self.step_count} steps, reward: {self.episode_reward:.2f}")
                    self.reset_simulation()

                # Control simulation speed
                time.sleep(0.01)  # ~100 FPS max

            except Exception as e:
                print(f"Simulation error: {e}")
                break

        print("🛑 Simulation loop ended")

    def _get_observation(self):
        """Get observation from MuJoCo state."""
        obs = []

        # Joint positions (skip root for floating base)
        if self.model.nq > 7:
            obs.extend(self.data.qpos[7:])

        # Joint velocities (skip root)
        if self.model.nv > 6:
            obs.extend(self.data.qvel[6:])

        # Body positions
        obs.extend(self.data.xpos.flatten()[:15])  # Limit to avoid overflow

        # Pad or truncate to expected dimension
        obs_array = np.array(obs, dtype=np.float32)
        if len(obs_array) > self.policy.obs_dim:
            obs_array = obs_array[:self.policy.obs_dim]
        elif len(obs_array) < self.policy.obs_dim:
            padding = np.zeros(self.policy.obs_dim - len(obs_array), dtype=np.float32)
            obs_array = np.concatenate([obs_array, padding])

        return obs_array

    def run(self):
        """Run the visual demo."""
        print("🎮 Starting Visual MuJoCo Demo...")
        print("   Close the window to exit")

        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            print("Demo interrupted by user")
        finally:
            self.stop_simulation()
            # Clean up
            if hasattr(self, 'robot_file'):
                Path(self.robot_file).unlink(missing_ok=True)


def main():
    """Main function to run the visual demo."""
    print("🤖 Enhanced Sonic G1 - Visual MuJoCo Demo")
    print("=" * 50)

    # Find the best available checkpoint
    checkpoint_candidates = [
        "checkpoints/production_large_model/best_checkpoint.pth",
        "checkpoints/enhanced_mac_model/best_checkpoint.pth",
        "checkpoints/mac_training/best_checkpoint.pth",
        "checkpoints/final_model/best_checkpoint.pth"
    ]

    checkpoint_path = None
    for candidate in checkpoint_candidates:
        if Path(candidate).exists():
            checkpoint_path = candidate
            break

    if not checkpoint_path:
        print("❌ No trained model found!")
        print("Train a model first with:")
        print("   python3 mac_optimized_training.py --epochs 50")
        return

    print(f"🎯 Using checkpoint: {checkpoint_path}")

    try:
        demo = VisualMuJoCoDemo(checkpoint_path)
        demo.run()
    except Exception as e:
        print(f"❌ Demo error: {e}")
        print("Make sure MuJoCo is properly installed: pip install mujoco")


if __name__ == "__main__":
    main()