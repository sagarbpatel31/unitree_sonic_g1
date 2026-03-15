# Unitree G1 MuJoCo Environment

A comprehensive MuJoCo-based environment for the Unitree G1 humanoid robot designed for whole-body motion imitation learning and reinforcement learning.

## Features

### 🤖 **Comprehensive Robot Modeling**
- Full Unitree G1 kinematic chain (22+ DOF)
- Accurate physics simulation with MuJoCo
- Realistic joint limits and actuator modeling
- Contact-rich foot dynamics

### 📊 **Rich Observation Space**
- Joint positions and velocities
- Root orientation (quaternion) and velocities
- IMU data (linear acceleration)
- Foot contact states
- Previous actions
- Reference motion features (for imitation)
- Command conditioning signals
- Optional height scanning

### 🎯 **Multi-Component Reward System**
- **Motion Tracking**: Joint position/velocity tracking
- **Root Tracking**: Position, orientation, velocity matching
- **Contact Consistency**: Foot contact and slip penalties
- **Stability**: Upright posture rewards
- **Naturalness**: Action smoothness, energy efficiency
- **Command Following**: Velocity command tracking

### 🔄 **Intelligent Reset System**
- Reference pose initialization
- Configurable noise injection
- Multiple reset strategies (default, reference, random)
- Safety constraint enforcement

### 🌍 **Domain Randomization**
- Physics parameters (friction, mass, damping)
- Actuator properties (strength, bias)
- Sensor noise injection
- Control latency simulation
- External force perturbations
- Ground property variation

### 🎮 **Command Conditioning**
- Walk forward/backward
- Turn left/right
- Strafe left/right
- Stop commands
- Smooth command interpolation
- Curriculum-based command progression

## Quick Start

### Basic Usage

```python
from src.envs.g1 import create_g1_env

# Create environment with minimal config
env = create_g1_env(
    model_path="path/to/unitree_g1.xml",
    config={
        "frame_skip": 10,
        "action_type": "position_delta"
    }
)

# Run environment
obs, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
```

### With Reference Motion

```python
# Create reference motion data
reference_motion = {
    "times": [...],
    "joint_positions": [...],
    "joint_velocities": [...],
    "root_positions": [...],
    "root_orientations": [...],
    # ... more fields
}

# Set reference motion for imitation
env.set_reference_motion(reference_motion)

# Reset will now use reference poses
obs, info = env.reset()
```

### With Command Conditioning

```python
# Enable commands in config
config = {
    "commands": {
        "enabled": True,
        "forward_vel_range": [0.0, 2.0],
        "yaw_rate_range": [-2.0, 2.0]
    }
}

env = create_g1_env(model_path, config)

# Commands are automatically sampled, or set manually:
env.set_command({
    "type": "walk_forward",
    "forward_vel": 1.0,
    "lateral_vel": 0.0,
    "yaw_rate": 0.0
})
```

## Configuration

### Environment Configuration

```yaml
# Basic settings
frame_skip: 10                    # Physics steps per env step
action_type: "position_delta"     # "position_delta" or "position_absolute"
action_scale: 0.1                 # Action scaling factor
render_mode: null                 # Rendering mode

# Component configurations
observations: {...}               # See ObservationConfig
rewards: {...}                   # See RewardConfig
resets: {...}                    # See ResetConfig
randomization: {...}             # See RandomizationConfig
commands: {...}                  # See CommandConfig
safety: {...}                    # Safety parameters
```

### Observation Configuration

```yaml
observations:
  # Core observations
  include_joint_pos: true
  include_joint_vel: true
  include_root_orientation: true
  include_root_linear_vel: true
  include_root_angular_vel: true
  include_previous_action: true

  # Reference motion
  include_reference_motion: true
  reference_horizon: 10           # Future timesteps to include

  # Additional sensors
  include_foot_contacts: true
  include_imu: true
  include_height_scan: false

  # Noise and normalization
  joint_pos_noise: 0.01
  joint_vel_noise: 0.1
  imu_noise: 0.05
  normalize_observations: true
```

### Reward Configuration

```yaml
rewards:
  # Tracking weights
  joint_pos_weight: 1.0
  joint_vel_weight: 0.1
  root_pos_weight: 1.0
  root_orient_weight: 1.0
  root_vel_weight: 0.5

  # Contact and stability
  foot_contact_weight: 0.2
  foot_slip_weight: 0.1
  upright_weight: 0.1

  # Naturalness penalties
  action_smoothness_weight: 0.05
  torque_penalty_weight: 0.001
  energy_penalty_weight: 0.001

  # Bonuses
  alive_bonus: 0.1
  command_tracking_weight: 0.5
```

### Domain Randomization

```yaml
randomization:
  enabled: true

  # Physics randomization
  friction_enabled: true
  friction_range: [0.5, 1.5]
  mass_enabled: true
  mass_range: [0.8, 1.2]
  motor_enabled: true
  motor_range: [0.9, 1.1]

  # Disturbances
  push_enabled: true
  push_probability: 0.002
  push_force_range: [50.0, 200.0]

  # Sensor noise
  sensor_noise_enabled: true
  joint_pos_noise_range: [0.001, 0.01]
  joint_vel_noise_range: [0.01, 0.1]
```

## Architecture

### Core Components

1. **G1Environment**: Main environment class
   - Inherits from `gymnasium.Env`
   - Orchestrates all component managers
   - Handles MuJoCo simulation loop

2. **ObservationManager**: Observation construction
   - Proprioceptive observations
   - Reference motion features
   - Sensor noise injection
   - Observation normalization

3. **RewardManager**: Reward computation
   - Multi-component reward functions
   - Motion tracking rewards
   - Stability and naturalness rewards
   - Command tracking rewards

4. **ResetManager**: Environment resets
   - Reference pose initialization
   - Noise injection
   - Multiple reset strategies
   - Safety constraint enforcement

5. **DomainRandomizer**: Physics randomization
   - Parameter randomization
   - External disturbances
   - Sensor degradation
   - Curriculum-based randomization

6. **CommandManager**: Command conditioning
   - Command sampling and smoothing
   - Velocity command types
   - Command curriculum

### Data Flow

```
MuJoCo Model → G1Environment
      ↓
ObservationManager → Observations
RewardManager → Rewards
ResetManager → Initial States
DomainRandomizer → Physics Variation
CommandManager → Behavior Commands
      ↓
Training Loop / Policy
```

## Action Spaces

### Position Delta (Recommended)
```python
action_type: "position_delta"
action_scale: 0.1  # Maximum position change per step
```
- Actions represent joint position changes
- Bounded by `action_scale`
- More stable for learning

### Position Absolute
```python
action_type: "position_absolute"
```
- Actions represent normalized target positions [-1, 1]
- Mapped to joint limits
- Direct position control

## Observation Space

### Proprioceptive Observations
- **Joint positions**: Current joint angles (normalized)
- **Joint velocities**: Joint angular velocities (normalized)
- **Root orientation**: Quaternion representation
- **Root velocities**: Linear and angular velocities (normalized)
- **Previous action**: Last action taken
- **Foot contacts**: Binary contact states
- **IMU data**: Linear acceleration in body frame

### Reference Motion Features (if enabled)
- **Future joint positions**: Reference trajectory
- **Future root poses**: Reference root motion
- **Temporal horizon**: Configurable lookahead

### Command Features (if enabled)
- **Velocity commands**: [forward_vel, lateral_vel, yaw_rate]

## Reward Components

### Motion Tracking Rewards
```python
joint_reward = exp(-scale * ||q_current - q_reference||)
root_reward = exp(-scale * ||pose_current - pose_reference||)
```

### Contact Rewards
- **Foot contact consistency**: Match reference contact states
- **Foot slip penalty**: Penalize horizontal foot motion during contact

### Stability Rewards
- **Upright reward**: Encourage upright posture
- **Action smoothness**: Penalize large action changes

### Energy Penalties
- **Torque penalty**: Minimize joint torques
- **Energy penalty**: Minimize mechanical power

## Termination Conditions

### Safety Termination
- **Fall detection**: Base height below threshold
- **Excessive tilt**: Base orientation beyond limits
- **Invalid state**: NaN values in state
- **Excessive velocity**: Joint velocities above limits

### Episode Completion
- **Horizon reached**: Maximum episode length
- **Task completion**: Reference motion completed

## Domain Randomization Details

### Physics Randomization
- **Friction coefficients**: Ground and object friction
- **Body masses**: Link mass distribution
- **Joint damping**: Energy dissipation
- **Actuator properties**: Motor strength and bias

### Sensor Degradation
- **Joint encoder noise**: Position and velocity noise
- **IMU noise**: Accelerometer noise
- **Contact noise**: Contact detection uncertainty

### External Disturbances
- **Push forces**: Random external forces
- **Ground variation**: Surface property changes
- **Control latency**: Delayed action application

## Testing

Run the test script to verify environment functionality:

```bash
cd examples
python test_g1_environment.py
```

This will test:
- Basic environment functionality
- Reference motion tracking
- Command conditioning
- Domain randomization
- Safety features

## Best Practices

### For Motion Imitation
1. Use reference pose initialization
2. Enable reference motion observations
3. Focus on tracking rewards
4. Start with minimal randomization

### For Robustness Training
1. Enable comprehensive domain randomization
2. Use push disturbances
3. Add sensor noise
4. Employ command conditioning

### For Curriculum Learning
1. Start with simple commands
2. Gradually increase randomization
3. Progressive noise injection
4. Monitor training stability

## Troubleshooting

### Common Issues

1. **Environment Creation Fails**
   - Check MuJoCo model path
   - Verify model has required joint names
   - Check joint count matches expectations

2. **Observation Dimension Mismatch**
   - Verify observation config
   - Check reference motion dimensions
   - Ensure model compatibility

3. **Training Instability**
   - Reduce domain randomization
   - Lower action scales
   - Check reward weighting
   - Monitor termination frequency

4. **Poor Motion Tracking**
   - Increase tracking reward weights
   - Reduce action smoothness penalty
   - Check reference motion quality
   - Verify joint mappings

### Performance Tips

1. **Speed Optimization**
   - Reduce frame_skip for faster simulation
   - Disable unnecessary observations
   - Use position_delta actions
   - Minimize domain randomization during testing

2. **Training Stability**
   - Start with minimal configuration
   - Gradually add complexity
   - Monitor reward components
   - Use curriculum learning

## Extensions

The environment is designed for extensibility:

1. **New Observation Types**: Extend `ObservationManager`
2. **Custom Rewards**: Add components to `RewardManager`
3. **Additional Commands**: Extend `CommandManager`
4. **New Randomization**: Add to `DomainRandomizer`

See the source code for detailed implementation examples.