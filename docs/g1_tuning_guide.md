# Unitree G1 PPO Training Tuning Guide

This guide provides specific recommendations for tuning PPO hyperparameters for successful motion imitation training on the Unitree G1 humanoid robot.

## Critical Parameters for G1 Success

### 1. Reward Function Tuning

The reward function is the most critical component for successful motion imitation. The G1's 22 DOF and humanoid dynamics require careful balance between different reward components.

#### **Joint Position Tracking (MOST CRITICAL)**
```yaml
joint_pos_weight: 2.0          # Primary objective - increase if tracking is poor
joint_pos_scale: 20.0          # Exponential reward scaling - tune for sensitivity
```

**Tuning Guidelines:**
- **Too Low (< 1.0)**: Robot will deviate significantly from reference motion
- **Too High (> 5.0)**: May cause overly rigid behavior, poor generalization
- **Scale Factor**: Higher values (10-30) create steeper reward gradients
- **Start conservative**: Begin with weight=1.5, scale=15.0 and adjust based on tracking quality

#### **Root Motion Tracking (SECOND MOST CRITICAL)**
```yaml
root_pos_weight: 1.5           # Root position tracking
root_orient_weight: 1.5        # Root orientation tracking
root_pos_scale: 30.0           # Higher scale for root precision
root_orient_scale: 10.0        # Orientation tracking sensitivity
```

**Tuning Guidelines:**
- Root position is crucial for locomotion - weight should be similar to joint_pos_weight
- Orientation tracking prevents drift and maintains upright posture
- Scale factors should be higher than joint scales (root errors are typically smaller)

#### **Joint Velocity Tracking**
```yaml
joint_vel_weight: 0.3          # Secondary objective
joint_vel_scale: 2.0           # Lower scale than positions
```

**Tuning Guidelines:**
- Keep weight much lower than position tracking (0.1-0.5 range)
- Helps with motion smoothness but shouldn't dominate
- If motion is too jerky, increase weight; if too slow, decrease

#### **Contact Consistency**
```yaml
foot_contact_weight: 0.4       # Important for locomotion
foot_slip_weight: 0.2          # Prevents foot sliding
```

**Tuning Guidelines:**
- Critical for walking/running motions
- Disable for non-locomotion tasks (dancing, gestures)
- Increase if foot sliding is observed during training

### 2. Network Architecture

#### **Policy Network**
```yaml
policy:
  hidden_dims: [512, 512, 256]  # Good balance for G1
  activation: "ReLU"
  initial_log_std: -0.5         # CRITICAL: Controls exploration
  min_log_std: -5.0
  max_log_std: 1.0
```

**Tuning Guidelines:**
- **Network Size**: 512-512-256 works well for G1's complexity
- **initial_log_std**: Start at -0.5 for moderate exploration
  - Too high (> 0): Excessive noise, unstable training
  - Too low (< -2): Insufficient exploration, poor coverage
- **Adjust exploration**: Monitor policy entropy in logs

#### **Critic Network**
```yaml
critic:
  hidden_dims: [512, 512, 512, 256]  # Larger than policy
  use_value_norm: true              # ESSENTIAL for stability
```

**Tuning Guidelines:**
- Make critic larger than policy for better value estimation
- **Always enable value normalization** for humanoid control
- Monitor explained variance - should increase over training

### 3. PPO Algorithm Parameters

#### **Core PPO Settings**
```yaml
ppo:
  clip_ratio: 0.2               # Conservative clipping
  epochs: 10                    # Multiple updates per rollout
  minibatch_size: 64            # Balance between stability and speed
  buffer_size: 2048             # Steps per environment
```

**Tuning Guidelines:**
- **clip_ratio**: Keep at 0.2, reduce to 0.1 if training is unstable
- **epochs**: 10 is good for humanoids, reduce if overfitting
- **buffer_size**: 2048 per env gives good sample efficiency
- **minibatch_size**: 64 works well, increase to 128 for larger buffers

#### **Advantage Estimation**
```yaml
gamma: 0.99                     # Standard discount factor
gae_lambda: 0.95               # GAE parameter for bias-variance tradeoff
```

**Tuning Guidelines:**
- Keep gamma at 0.99 for long-horizon tasks
- Reduce gae_lambda to 0.9 if value estimates are noisy
- Increase to 0.98 if you want less bias in advantage estimates

### 4. Learning Rates and Optimization

#### **Learning Rates (CRITICAL)**
```yaml
training:
  policy_lr: 3e-4               # Standard for PPO
  critic_lr: 3e-4               # Same as policy initially
  adam_eps: 1e-5                # Numerical stability
```

**Tuning Guidelines:**
- **Start with 3e-4** for both networks
- **Too high (> 1e-3)**: Training instability, large policy changes
- **Too low (< 1e-4)**: Slow convergence, may not learn
- **Reduce if seeing**: Large policy updates, high KL divergence
- **Consider separate rates**: critic_lr can be higher (1e-3) if value learning is slow

#### **Learning Rate Scheduling**
```yaml
lr_schedule: "linear"           # Linear decay works well
min_lr_ratio: 0.1              # Don't decay to zero
```

### 5. Environment and Reset Configuration

#### **Episode Length**
```yaml
max_episode_steps: 1000         # ~20 seconds at 50Hz control
```

**Tuning Guidelines:**
- Match typical motion clip length
- Longer episodes = more stable gradients but slower iteration
- Shorter episodes = faster training but noisier estimates

#### **Reset Strategy**
```yaml
resets:
  use_reference_pose: true      # ESSENTIAL for imitation
  pose_noise_scale: 0.05        # Small noise for robustness
  reference_time_range: [0.0, 0.8]  # Don't reset at clip end
```

**Tuning Guidelines:**
- **Always use reference pose** for motion imitation
- **pose_noise_scale**: 0.02-0.1 range for robustness
- **reference_time_range**: Avoid resetting at very end of clips

### 6. Domain Randomization

#### **Initial Training (Conservative)**
```yaml
randomization:
  enabled: true
  friction_range: [0.8, 1.2]    # Mild friction variation
  mass_range: [0.95, 1.05]      # Small mass changes
  push_enabled: false           # Disable initially
```

**Tuning Guidelines:**
- **Start conservative**: Small ranges initially
- **Gradually increase**: Add more randomization as training stabilizes
- **Monitor stability**: Too much randomization prevents learning
- **Enable pushes later**: Only after basic motion imitation works

### 7. Curriculum Learning

#### **Curriculum Parameters**
```yaml
curriculum:
  enabled: true
  target_success_rate: 0.8      # When to increase difficulty
  adaptation_rate: 0.02         # How fast to progress
  initial_level: 0.0            # Start with easiest clips
```

**Tuning Guidelines:**
- **target_success_rate**: 0.7-0.9 range works well
- **adaptation_rate**: 0.01-0.05, slower is more stable
- **Monitor curriculum level**: Should gradually increase over training

## Troubleshooting Common Issues

### Problem: Poor Motion Tracking

**Symptoms:**
- Robot deviates significantly from reference motion
- Low reward values throughout training
- Joint positions don't match reference

**Solutions:**
1. **Increase tracking weights**:
   ```yaml
   joint_pos_weight: 3.0  # Increase from 2.0
   root_pos_weight: 2.0   # Increase from 1.5
   ```

2. **Increase reward scaling**:
   ```yaml
   joint_pos_scale: 30.0  # Increase from 20.0
   ```

3. **Reduce competing rewards**:
   ```yaml
   action_smoothness_weight: 0.05  # Reduce from 0.1
   torque_penalty_weight: 0.0001   # Reduce from 0.0005
   ```

### Problem: Training Instability

**Symptoms:**
- Large policy loss spikes
- High KL divergence warnings
- Reward oscillations

**Solutions:**
1. **Reduce learning rate**:
   ```yaml
   policy_lr: 1e-4       # Reduce from 3e-4
   critic_lr: 1e-4       # Reduce from 3e-4
   ```

2. **Reduce clipping**:
   ```yaml
   clip_ratio: 0.1       # Reduce from 0.2
   ```

3. **Reduce exploration**:
   ```yaml
   initial_log_std: -1.0  # Reduce from -0.5
   entropy_coeff: 0.005   # Reduce from 0.01
   ```

### Problem: Slow Convergence

**Symptoms:**
- Reward improvements plateau early
- Policy barely changes over time
- Low explained variance

**Solutions:**
1. **Increase learning rates**:
   ```yaml
   policy_lr: 5e-4       # Increase from 3e-4
   critic_lr: 1e-3       # Increase critic learning
   ```

2. **Increase buffer size**:
   ```yaml
   buffer_size: 4096     # Increase from 2048
   ```

3. **Increase exploration**:
   ```yaml
   initial_log_std: 0.0   # Increase from -0.5
   entropy_coeff: 0.02    # Increase from 0.01
   ```

### Problem: Poor Contact Handling

**Symptoms:**
- Foot sliding during contact
- Unnatural foot trajectories
- Poor locomotion quality

**Solutions:**
1. **Increase contact weights**:
   ```yaml
   foot_contact_weight: 0.6   # Increase from 0.4
   foot_slip_weight: 0.4      # Increase from 0.2
   ```

2. **Ensure contact estimation quality**:
   - Check motion retargeting pipeline
   - Validate foot contact labels

3. **Add contact-specific rewards**:
   - Consider end-effector position tracking
   - Add foot height penalties during swing phase

## Performance Optimization

### Hardware Recommendations

**GPU Memory:**
- Minimum: 8GB for basic training
- Recommended: 16GB+ for large batches
- Optimal: 24GB+ for fast iteration

**Training Speed Tips:**
1. **Increase num_envs**: More parallel environments
   ```yaml
   num_envs: 32          # Scale with GPU memory
   ```

2. **Enable mixed precision** (modern GPUs):
   ```yaml
   use_mixed_precision: true
   ```

3. **Optimize buffer size**:
   ```yaml
   buffer_size: 4096     # Larger buffers = fewer updates
   ```

### Monitoring Training Progress

**Key Metrics to Watch:**

1. **Reward Components**:
   - `train/mean_reward`: Overall progress
   - `joint_pos_reward`: Primary tracking quality
   - `root_pos_reward`: Locomotion quality

2. **Policy Metrics**:
   - `policy_loss`: Should be stable
   - `entropy`: Should decrease gradually
   - `approx_kl`: Should stay below target_kl

3. **Value Function**:
   - `value_loss`: Should decrease
   - `explained_variance`: Should increase (>0.5)

4. **Environment Metrics**:
   - `eval/success_rate`: Key progress indicator
   - `curriculum_level`: Should increase gradually

## Hyperparameter Schedules

### Typical Training Progression

**Phase 1: Initial Learning (0-1M steps)**
- Conservative hyperparameters
- Focus on basic motion tracking
- Minimal domain randomization

**Phase 2: Skill Refinement (1M-3M steps)**
- Gradually increase curriculum difficulty
- Add moderate domain randomization
- Fine-tune reward weights

**Phase 3: Robustness (3M+ steps)**
- Enable all domain randomization
- Add external disturbances
- Focus on generalization

### Example Schedule
```yaml
# Phase 1
joint_pos_weight: 1.5
randomization.push_enabled: false
curriculum.adaptation_rate: 0.01

# Phase 2 (at 1M steps)
joint_pos_weight: 2.0
randomization.push_enabled: true
curriculum.adaptation_rate: 0.02

# Phase 3 (at 3M steps)
joint_pos_weight: 2.5
randomization.push_force_range: [100.0, 300.0]
```

## Summary of Most Important Parameters

**Must Tune for G1:**
1. `joint_pos_weight` and `joint_pos_scale` - Motion tracking quality
2. `root_pos_weight` and `root_orient_weight` - Locomotion stability
3. `initial_log_std` - Exploration level
4. `policy_lr` and `critic_lr` - Learning speed vs stability
5. `curriculum.target_success_rate` - Curriculum progression

**Start with these values:**
```yaml
rewards:
  joint_pos_weight: 2.0
  joint_pos_scale: 20.0
  root_pos_weight: 1.5
  root_orient_weight: 1.5

policy:
  initial_log_std: -0.5

training:
  policy_lr: 3e-4
  critic_lr: 3e-4

curriculum:
  target_success_rate: 0.8
  adaptation_rate: 0.02
```

Then adjust based on training behavior and the troubleshooting guidelines above.