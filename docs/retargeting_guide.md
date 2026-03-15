# Motion Retargeting Pipeline Guide

This guide explains how to use the motion retargeting pipeline to convert human motion capture data into Unitree G1-compatible trajectories for training.

## Overview

The retargeting pipeline consists of several key components:

1. **Skeleton Mapping** - Maps human skeleton joints to G1 robot joints
2. **Motion Retargeting** - Converts human motions to robot trajectories
3. **Contact Estimation** - Detects when feet are in contact with the ground
4. **Normalization** - Standardizes data for training stability
5. **Validation** - Quality assessment and error detection

## Quick Start

### Basic Example

```python
from src.data import (
    MotionRetargeter, ContactEstimator, MotionNormalizer,
    load_motion_clip_from_npz, save_g1_trajectory_to_npz
)

# Load human motion data
motion_clip = load_motion_clip_from_npz("human_motion.npz")

# Retarget to G1 format
retargeter = MotionRetargeter(target_fps=50.0)
g1_trajectory = retargeter.retarget_clip(motion_clip)

# Estimate foot contacts
contact_estimator = ContactEstimator()
contacts = contact_estimator.estimate_contacts(
    g1_trajectory.timestamps,
    joint_positions_dict  # From forward kinematics
)

# Save processed trajectory
save_g1_trajectory_to_npz(g1_trajectory, "g1_trajectory.npz")
```

### Complete Pipeline

```python
from src.data import (
    MotionRetargeter, ContactEstimator, MotionNormalizer,
    RetargetingValidator, ValidationConfig
)

# 1. Initialize components
retargeter = MotionRetargeter(
    target_fps=50.0,
    smoothing_window=5,
    apply_joint_limits=True
)

contact_estimator = ContactEstimator()
normalizer = MotionNormalizer()
validator = RetargetingValidator(
    ValidationConfig(generate_plots=True)
)

# 2. Process motion clip
g1_trajectory = retargeter.retarget_clip(motion_clip)

# 3. Estimate contacts and update trajectory
contacts = contact_estimator.estimate_contacts(
    g1_trajectory.timestamps, joint_positions
)
g1_trajectory.foot_contacts = np.column_stack([
    contacts["left"], contacts["right"]
])

# 4. Validate quality
metrics = validator.validate_trajectory(g1_trajectory)
print(f"Quality score: {metrics.overall_quality_score:.3f}")

# 5. Normalize for training (batch operation)
normalized_trajectories, fitted_normalizer = normalize_trajectory_batch(
    [trajectory_dict], normalizer
)
```

## Configuration

### YAML Configuration

The pipeline can be configured using YAML files. See `configs/data/retargeting.yaml` for a complete example:

```yaml
retargeting:
  target_fps: 50.0
  smoothing_window: 5
  apply_joint_limits: true

contact_estimation:
  velocity_threshold: 0.02
  height_threshold: 0.05
  min_contact_duration: 0.1

normalization:
  normalize_joint_pos: true
  normalize_joint_vel: true
  joint_pos_method: "standard"

validation:
  critical_quality_threshold: 0.3
  generate_plots: true
```

### Programmatic Configuration

```python
from src.data import (
    ContactEstimationConfig, NormalizationConfig, ValidationConfig
)

# Contact estimation config
contact_config = ContactEstimationConfig(
    velocity_threshold=0.02,
    height_threshold=0.05,
    min_contact_duration=0.1,
    combine_method="voting"
)

# Normalization config
norm_config = NormalizationConfig(
    normalize_joint_pos=True,
    normalize_joint_vel=True,
    joint_pos_method="standard",
    clip_outliers=True
)

# Validation config
val_config = ValidationConfig(
    critical_quality_threshold=0.3,
    generate_plots=True,
    plot_output_dir="validation_plots"
)
```

## Input Data Formats

### Expected Input Format

Human motion data should be provided as `MotionClipData` objects or NPZ files with:

```python
{
    "timestamps": np.array,          # (T,) - time stamps
    "joint_positions": dict,         # {joint_name: (T, 3)} - 3D positions
    "joint_rotations": dict,         # {joint_name: (T, 4)} - quaternions
    "root_positions": np.array,      # (T, 3) - root position
    "root_orientations": np.array,   # (T, 4) - root orientation (quaternion)
    "skeleton_type": str             # "amass", "cmu", "mixamo", etc.
}
```

### Supported Skeleton Types

- **AMASS**: Research datasets (CMU, EKUT, TCD, etc.)
- **CMU**: Carnegie Mellon University mocap database
- **Mixamo**: Adobe character animation system

### Loading Data

```python
# From NPZ file
motion_clip = load_motion_clip_from_npz("data.npz")

# From custom format
motion_clip = MotionClipData(
    timestamps=timestamps,
    joint_positions=joint_pos_dict,
    joint_rotations=joint_rot_dict,
    root_positions=root_pos,
    root_orientations=root_orient,
    skeleton_type="amass"
)
```

## Output Format

### G1 Trajectory Data

The retargeting pipeline outputs `G1TrajectoryData` objects containing:

```python
{
    "timestamps": np.array,              # (T,) - time stamps at 50Hz
    "joint_positions": np.array,         # (T, 22) - G1 joint angles
    "joint_velocities": np.array,        # (T, 22) - joint velocities
    "joint_accelerations": np.array,     # (T, 22) - joint accelerations
    "root_positions": np.array,          # (T, 3) - root position
    "root_orientations": np.array,       # (T, 4) - root orientation
    "root_linear_velocities": np.array,  # (T, 3) - root linear velocity
    "root_angular_velocities": np.array, # (T, 3) - root angular velocity
    "foot_contacts": np.array,           # (T, 2) - left/right foot contacts
    "metadata": dict                     # Processing metadata
}
```

### Joint Ordering

G1 joints are ordered as defined in `src/data/skeleton_map.py`:

1. `left_hip_yaw`
2. `left_hip_roll`
3. `left_hip_pitch`
4. `left_knee`
5. `left_ankle_pitch`
6. `left_ankle_roll`
7. `right_hip_yaw`
8. `right_hip_roll`
9. `right_hip_pitch`
10. `right_knee`
11. `right_ankle_pitch`
12. `right_ankle_roll`
13. `torso_yaw`
14. `left_shoulder_pitch`
15. `left_shoulder_roll`
16. `left_shoulder_yaw`
17. `left_elbow`
18. `left_wrist_yaw`
19. `left_wrist_roll`
20. `left_wrist_pitch`
21. `right_shoulder_pitch`
22. `right_shoulder_roll` (truncated for 22 DOF)

## Skeleton Mapping

### How Mapping Works

1. **Joint Correspondence**: Each G1 joint is mapped to a human joint
2. **Axis Transformation**: Rotation axes are transformed between coordinate systems
3. **Sign and Offset**: Sign flips and angular offsets handle different conventions

### Example Mapping

```python
# AMASS to G1 mapping example
mapping = {
    "left_hip_pitch": ["leftHipPitch", {
        "axis": "z",      # Extract Z-axis rotation
        "sign": -1,       # Flip sign for coordinate system
        "offset": 0.0     # No angular offset
    }]
}
```

### Custom Skeleton Types

To add support for new skeleton types:

```python
# In src/data/skeleton_map.py
HUMAN_TO_G1_MAPPING["my_skeleton"] = {
    "left_hip_yaw": ["MyLeftHipYaw", {"axis": "y", "sign": 1, "offset": 0.0}],
    "left_hip_roll": ["MyLeftHipRoll", {"axis": "x", "sign": 1, "offset": 0.0}],
    # ... complete mapping
}
```

## Contact Estimation

### Algorithm Overview

The contact estimator uses multiple methods to detect foot-ground contacts:

1. **Velocity-based**: Contacts when foot velocity < threshold
2. **Height-based**: Contacts when foot height < threshold
3. **Acceleration-based**: Contacts when foot acceleration is low

### Configuration Options

```python
config = ContactEstimationConfig(
    velocity_threshold=0.02,     # m/s
    height_threshold=0.05,       # m
    acceleration_threshold=2.0,  # m/s²
    min_contact_duration=0.1,    # seconds
    combine_method="voting"      # How to combine methods
)
```

### Contact Output

Contact estimation returns a dictionary:

```python
contacts = {
    "left": np.array([0.0, 0.2, 0.8, 1.0, ...]),   # Left foot contact probability
    "right": np.array([1.0, 0.8, 0.2, 0.0, ...])   # Right foot contact probability
}
```

Values range from 0.0 (no contact) to 1.0 (full contact).

## Normalization

### Purpose

Normalization standardizes the data for stable neural network training:

- **Joint angles**: Centered around mean, scaled by standard deviation
- **Velocities**: Scaled to reasonable ranges
- **Root motion**: Normalized for translation invariance

### Statistics Computation

```python
# Compute normalization statistics from multiple trajectories
normalizer = MotionNormalizer()
normalizer.fit(trajectory_list)

# Save statistics for later use
normalizer.save_statistics("norm_stats.pkl")

# Apply normalization
normalized_traj = normalizer.normalize(trajectory)
```

### Normalization Methods

- **Standard**: `(x - mean) / std`
- **MinMax**: `(x - min) / (max - min)`
- **Robust**: Uses median and IQR instead of mean/std

## Validation

### Quality Metrics

The validation system computes multiple quality scores:

1. **Joint Continuity**: Smoothness of joint trajectories
2. **Kinematic Feasibility**: Velocities/accelerations within bounds
3. **Contact Consistency**: Logical foot contact patterns
4. **Root Smoothness**: Smooth root trajectory
5. **Energy Efficiency**: Low accelerations and jerky motions
6. **Naturalness**: Realistic motion patterns

### Quality Thresholds

```python
# Quality score interpretation
score >= 0.7  # High quality (good for training)
score >= 0.3  # Acceptable quality (may need review)
score < 0.3   # Poor quality (should be discarded)
```

### Validation Output

```python
metrics = validator.validate_trajectory(g1_trajectory)
print(f"Overall quality: {metrics.overall_quality_score:.3f}")
print(f"Critical issues: {metrics.has_critical_issues}")
print(f"Joint limit violations: {metrics.joint_limit_violations}")
```

## Batch Processing

### Processing Multiple Files

```python
from pathlib import Path

# Process all NPZ files in a directory
input_dir = Path("human_motion_data")
output_dir = Path("g1_trajectories")

npz_files = list(input_dir.glob("**/*.npz"))

# Initialize processors
retargeter = MotionRetargeter(target_fps=50.0)
validator = RetargetingValidator()

for npz_file in npz_files:
    # Load and process
    motion_clip = load_motion_clip_from_npz(npz_file)
    g1_trajectory = retargeter.retarget_clip(motion_clip)

    # Validate quality
    metrics = validator.validate_trajectory(g1_trajectory)

    # Save if quality is acceptable
    if metrics.overall_quality_score >= 0.5:
        output_file = output_dir / f"{npz_file.stem}_g1.npz"
        save_g1_trajectory_to_npz(g1_trajectory, output_file)
```

### Parallel Processing

```python
from multiprocessing import Pool
from functools import partial

def process_file(npz_file, retargeter, validator, output_dir):
    # Processing logic here
    pass

# Parallel processing
with Pool(processes=4) as pool:
    process_func = partial(
        process_file,
        retargeter=retargeter,
        validator=validator,
        output_dir=output_dir
    )
    pool.map(process_func, npz_files)
```

## Example Scripts

### Basic Retargeting

```bash
# Run the example script
python examples/retargeting_pipeline_example.py --mode single --output output/demo

# Batch processing
python examples/retargeting_pipeline_example.py \
    --mode batch \
    --input data/human_motion \
    --output data/g1_trajectories \
    --max-clips 100
```

### Using Configuration Files

```python
import yaml
from omegaconf import OmegaConf

# Load configuration
config = OmegaConf.load("configs/data/retargeting.yaml")

# Initialize with config
retargeter = MotionRetargeter(
    target_fps=config.retargeting.target_fps,
    smoothing_window=config.retargeting.smoothing_window,
    apply_joint_limits=config.retargeting.apply_joint_limits
)
```

## Troubleshooting

### Common Issues

1. **Joint Limit Violations**
   - Solution: Adjust `orientation_scale` or joint mappings
   - Check: Joint limits in `skeleton_map.py`

2. **Poor Contact Estimation**
   - Solution: Tune contact detection thresholds
   - Check: Foot joint names in configuration

3. **Low Quality Scores**
   - Solution: Improve smoothing parameters
   - Check: Source motion quality

4. **Memory Issues**
   - Solution: Process in smaller batches
   - Check: Available RAM vs trajectory size

### Debugging

```python
# Enable debug logging
logging.getLogger().setLevel(logging.DEBUG)

# Save intermediate results
retargeter = MotionRetargeter(
    # ... config
)
# Save before/after joint angles for inspection

# Visualize joint trajectories
import matplotlib.pyplot as plt
plt.plot(g1_trajectory.timestamps, g1_trajectory.joint_positions)
plt.show()
```

### Performance Tips

1. **Reduce Output Framerate**: Use 30Hz instead of 50Hz for less critical applications
2. **Disable Validation Plots**: Set `generate_plots=False` for batch processing
3. **Use Multiprocessing**: Process files in parallel
4. **Cache Skeleton Mappings**: Reuse mapping computations

## Integration with Training

### Data Loading for Training

```python
# In your training script
from src.data import load_g1_trajectory_from_npz

# Load processed trajectories
trajectories = []
for npz_file in trajectory_files:
    traj = load_g1_trajectory_from_npz(npz_file)
    trajectories.append(traj)

# Convert to training format
training_data = {
    "observations": np.concatenate([t.joint_positions for t in trajectories]),
    "actions": np.concatenate([t.joint_velocities for t in trajectories]),
    # ... additional data
}
```

### Train/Validation Split

```python
# Split trajectories for training
from sklearn.model_selection import train_test_split

train_trajs, val_trajs = train_test_split(
    trajectories,
    test_size=0.2,
    random_state=42
)
```

This completes the motion retargeting pipeline guide. The pipeline provides a robust, configurable system for converting human motion capture data into high-quality robot trajectories suitable for training imitation learning policies.