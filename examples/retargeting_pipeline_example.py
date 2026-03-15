#!/usr/bin/env python3
"""
Complete example demonstrating the motion retargeting pipeline.

This script shows how to:
1. Load human motion capture data
2. Retarget it to Unitree G1 robot format
3. Estimate foot contacts
4. Normalize the data for training
5. Validate the results
6. Save processed trajectories for training
"""

import numpy as np
import logging
from pathlib import Path
import argparse
from typing import List, Dict

# Import motion retargeting components
from src.data import (
    MotionRetargeter, MotionClipData, G1TrajectoryData,
    ContactEstimator, ContactEstimationConfig,
    MotionNormalizer, NormalizationConfig,
    RetargetingValidator, ValidationConfig,
    retarget_motion_clip, estimate_foot_contacts, validate_retargeted_motion,
    load_motion_clip_from_npz, save_g1_trajectory_to_npz
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_motion_clip() -> MotionClipData:
    """
    Create a sample motion clip for demonstration.
    In practice, you would load this from actual mocap data.
    """
    # Generate a simple walking motion (simplified)
    duration = 4.0  # seconds
    fps = 30.0
    n_frames = int(duration * fps)

    timestamps = np.linspace(0, duration, n_frames)

    # Simple sinusoidal walking pattern
    t = timestamps

    # Root motion (walking forward with slight up-down motion)
    root_positions = np.zeros((n_frames, 3))
    root_positions[:, 0] = 0.5 * t  # Forward motion
    root_positions[:, 1] = 0.0      # No lateral motion
    root_positions[:, 2] = 1.0 + 0.02 * np.sin(4 * np.pi * t)  # Slight vertical motion

    # Root orientation (slight yaw oscillation)
    root_orientations = np.zeros((n_frames, 4))
    root_orientations[:, 3] = 1.0  # w component
    # Add slight yaw rotation
    yaw_angle = 0.1 * np.sin(2 * np.pi * t)
    root_orientations[:, 2] = np.sin(yaw_angle / 2)  # z component
    root_orientations[:, 3] = np.cos(yaw_angle / 2)  # w component

    # Joint positions (simplified - just key joints for walking)
    joint_positions = {}
    joint_rotations = {}

    # Define joint motion patterns (simplified walking)
    joints = {
        'leftHip': 0.3 * np.sin(2 * np.pi * t + np.pi),
        'leftKnee': 0.3 * (np.sin(4 * np.pi * t) + 1),
        'leftAnkle': 0.1 * np.sin(4 * np.pi * t),
        'rightHip': 0.3 * np.sin(2 * np.pi * t),
        'rightKnee': 0.3 * (np.sin(4 * np.pi * t + np.pi) + 1),
        'rightAnkle': 0.1 * np.sin(4 * np.pi * t + np.pi),
    }

    for joint_name, angles in joints.items():
        # Convert angles to quaternions (simplified)
        joint_rotations[joint_name] = np.zeros((n_frames, 4))
        joint_rotations[joint_name][:, 3] = 1.0  # w component
        joint_rotations[joint_name][:, 1] = np.sin(angles / 2)  # y component (pitch)
        joint_rotations[joint_name][:, 3] = np.cos(angles / 2)  # w component

        # Approximate joint positions
        joint_positions[joint_name] = np.random.normal(0, 0.1, (n_frames, 3))

    return MotionClipData(
        timestamps=timestamps,
        joint_positions=joint_positions,
        joint_rotations=joint_rotations,
        root_positions=root_positions,
        root_orientations=root_orientations,
        skeleton_type="amass"
    )


def process_single_clip(motion_clip: MotionClipData,
                       output_dir: Path) -> G1TrajectoryData:
    """
    Process a single motion clip through the complete pipeline.
    """
    logger.info("Starting motion retargeting pipeline...")

    # Step 1: Retarget human motion to G1 format
    logger.info("Step 1: Retargeting motion to G1 format...")
    retargeter = MotionRetargeter(
        target_fps=50.0,
        smoothing_window=5,
        position_scale=1.0,
        apply_joint_limits=True,
        filter_velocities=True
    )

    g1_trajectory = retargeter.retarget_clip(motion_clip)
    logger.info(f"Retargeted to G1: {len(g1_trajectory.timestamps)} frames at {g1_trajectory.metadata['target_fps']} fps")

    # Step 2: Estimate foot contacts
    logger.info("Step 2: Estimating foot contacts...")
    contact_config = ContactEstimationConfig(
        velocity_threshold=0.02,
        height_threshold=0.05,
        min_contact_duration=0.1,
        use_velocity=True,
        use_height=True
    )

    contact_estimator = ContactEstimator(contact_config)

    # Create joint positions dict for contact estimation
    joint_positions_dict = {}
    if 'joint_names' in g1_trajectory.metadata:
        joint_names = g1_trajectory.metadata['joint_names']
        for i, joint_name in enumerate(joint_names):
            # For contact estimation, we need 3D positions, but we only have angles
            # In practice, you'd use forward kinematics to get positions
            joint_positions_dict[joint_name] = np.random.normal(0, 0.1, (len(g1_trajectory.timestamps), 3))

    contacts = contact_estimator.estimate_contacts(
        g1_trajectory.timestamps,
        joint_positions_dict
    )

    # Update foot contacts in trajectory
    g1_trajectory.foot_contacts[:, 0] = contacts.get("left", np.zeros(len(g1_trajectory.timestamps)))
    g1_trajectory.foot_contacts[:, 1] = contacts.get("right", np.zeros(len(g1_trajectory.timestamps)))

    logger.info(f"Contact estimation complete: "
               f"left={np.sum(contacts['left'] > 0.5)} frames, "
               f"right={np.sum(contacts['right'] > 0.5)} frames")

    # Step 3: Validate retargeting quality
    logger.info("Step 3: Validating retargeting quality...")
    validation_config = ValidationConfig(
        generate_plots=True,
        plot_output_dir=str(output_dir / "validation_plots")
    )

    validator = RetargetingValidator(validation_config)
    metrics = validator.validate_trajectory(g1_trajectory)

    logger.info(f"Validation complete: Overall quality score = {metrics.overall_quality_score:.3f}")

    if metrics.has_critical_issues:
        logger.warning("Critical quality issues detected!")
    elif metrics.has_warnings:
        logger.warning("Quality warnings detected")
    else:
        logger.info("Trajectory quality is good")

    # Step 4: Save the processed trajectory
    logger.info("Step 4: Saving processed trajectory...")
    output_file = output_dir / "processed_trajectory.npz"
    save_g1_trajectory_to_npz(g1_trajectory, str(output_file))

    # Also save validation metrics
    import json
    metrics_dict = {
        "sequence_duration": metrics.sequence_duration,
        "n_frames": metrics.n_frames,
        "framerate": metrics.framerate,
        "overall_quality_score": metrics.overall_quality_score,
        "has_critical_issues": metrics.has_critical_issues,
        "has_warnings": metrics.has_warnings,
        "joint_continuity_score": metrics.joint_continuity_score,
        "kinematic_feasibility_score": metrics.kinematic_feasibility_score
    }

    with open(output_dir / "validation_metrics.json", 'w') as f:
        json.dump(metrics_dict, f, indent=2)

    logger.info(f"Pipeline complete! Output saved to {output_dir}")
    return g1_trajectory


def process_batch_clips(input_dir: Path, output_dir: Path, max_clips: int = None):
    """
    Process a batch of motion clips.
    """
    logger.info(f"Processing batch of clips from {input_dir}")

    # Find all NPZ files in input directory
    npz_files = list(input_dir.glob("**/*.npz"))
    if max_clips:
        npz_files = npz_files[:max_clips]

    logger.info(f"Found {len(npz_files)} NPZ files to process")

    # Initialize components for batch processing
    retargeter = MotionRetargeter(target_fps=50.0)
    contact_estimator = ContactEstimator()
    validator = RetargetingValidator()

    processed_trajectories = []
    all_validation_metrics = []

    for i, npz_file in enumerate(npz_files):
        logger.info(f"Processing file {i+1}/{len(npz_files)}: {npz_file.name}")

        try:
            # Load motion clip
            motion_clip = load_motion_clip_from_npz(str(npz_file))

            # Retarget to G1
            g1_trajectory = retargeter.retarget_clip(motion_clip)

            # Validate
            metrics = validator.validate_trajectory(g1_trajectory)
            all_validation_metrics.append(metrics)

            # Save if quality is acceptable
            if metrics.overall_quality_score >= 0.5:
                clip_output_dir = output_dir / f"clip_{i:04d}"
                clip_output_dir.mkdir(parents=True, exist_ok=True)

                output_file = clip_output_dir / f"{npz_file.stem}_g1.npz"
                save_g1_trajectory_to_npz(g1_trajectory, str(output_file))

                processed_trajectories.append(g1_trajectory)
                logger.info(f"Saved high-quality trajectory: {output_file}")
            else:
                logger.warning(f"Skipping low-quality trajectory (score: {metrics.overall_quality_score:.3f})")

        except Exception as e:
            logger.error(f"Failed to process {npz_file}: {e}")
            continue

    # Batch normalization (optional)
    if processed_trajectories:
        logger.info("Computing normalization statistics for batch...")

        # Convert to format expected by normalizer
        trajectory_dicts = []
        for traj in processed_trajectories:
            traj_dict = {
                "joint_positions": traj.joint_positions,
                "joint_velocities": traj.joint_velocities,
                "joint_accelerations": traj.joint_accelerations,
                "root_positions": traj.root_positions,
                "root_orientations": traj.root_orientations,
                "root_linear_velocities": traj.root_linear_velocities,
                "root_angular_velocities": traj.root_angular_velocities,
                "joint_names": traj.metadata.get("joint_names", [])
            }
            trajectory_dicts.append(traj_dict)

        # Fit normalizer
        normalizer = MotionNormalizer()
        normalizer.fit(trajectory_dicts)

        # Save normalization statistics
        stats_file = output_dir / "normalization_stats.pkl"
        normalizer.save_statistics(stats_file)

        logger.info(f"Normalization statistics saved to {stats_file}")

        # Print summary
        summary = normalizer.get_statistics_summary()
        logger.info(f"Normalization summary: {summary}")

    # Print batch validation summary
    if all_validation_metrics:
        logger.info("\n" + "="*50)
        logger.info("BATCH PROCESSING SUMMARY")
        logger.info("="*50)

        n_processed = len(processed_trajectories)
        n_total = len(npz_files)
        n_critical = sum(m.has_critical_issues for m in all_validation_metrics)
        n_warnings = sum(m.has_warnings for m in all_validation_metrics)

        logger.info(f"Total files processed: {n_total}")
        logger.info(f"High-quality trajectories saved: {n_processed}")
        logger.info(f"Critical issues: {n_critical}")
        logger.info(f"Warnings: {n_warnings}")

        quality_scores = [m.overall_quality_score for m in all_validation_metrics]
        logger.info(f"Average quality score: {np.mean(quality_scores):.3f}")
        logger.info("="*50)


def main():
    """Main function demonstrating the retargeting pipeline."""
    parser = argparse.ArgumentParser(description="Motion retargeting pipeline example")
    parser.add_argument("--mode", choices=["single", "batch"], default="single",
                       help="Run single clip demo or batch processing")
    parser.add_argument("--input", type=str, help="Input directory for batch processing")
    parser.add_argument("--output", type=str, default="output/retargeting_demo",
                       help="Output directory")
    parser.add_argument("--max-clips", type=int, help="Maximum number of clips to process")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "single":
        # Single clip demonstration
        logger.info("Running single clip demonstration...")

        # Create a sample motion clip
        motion_clip = create_sample_motion_clip()

        # Process through pipeline
        g1_trajectory = process_single_clip(motion_clip, output_dir)

        # Print summary
        logger.info("\nSingle clip processing complete!")
        logger.info(f"Input duration: {motion_clip.timestamps[-1]:.2f}s")
        logger.info(f"Output duration: {g1_trajectory.timestamps[-1]:.2f}s")
        logger.info(f"Output frames: {len(g1_trajectory.timestamps)}")
        logger.info(f"Joint DOF: {g1_trajectory.joint_positions.shape[1]}")

    elif args.mode == "batch":
        # Batch processing
        if not args.input:
            logger.error("Input directory required for batch processing")
            return

        input_dir = Path(args.input)
        if not input_dir.exists():
            logger.error(f"Input directory does not exist: {input_dir}")
            return

        logger.info("Running batch processing...")
        process_batch_clips(input_dir, output_dir, args.max_clips)


if __name__ == "__main__":
    main()