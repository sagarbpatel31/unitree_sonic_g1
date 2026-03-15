#!/usr/bin/env python3
"""
Main training script for PPO-based motion imitation on Unitree G1.

This script orchestrates the complete training pipeline including:
- Environment vectorization
- Motion clip curriculum
- PPO training with proper logging
- Evaluation and model selection
- Checkpointing and resumption
"""

import os
import sys
import argparse
import logging
import random
import numpy as np
import torch
import torch.multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import wandb
from omegaconf import OmegaConf, DictConfig
from tensorboardX import SummaryWriter

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.envs.g1 import create_g1_env
from src.data import load_g1_trajectory_from_npz, MotionNormalizer
from sonic_g1.train.ppo import PPOTrainer
from sonic_g1.models.policy import G1Policy
from sonic_g1.models.critic import G1Critic
from sonic_g1.train.rollout_buffer import RolloutBuffer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    step: int
    episode: int
    mean_reward: float
    mean_episode_length: float
    policy_loss: float
    value_loss: float
    entropy_loss: float
    explained_variance: float
    fps: float
    curriculum_level: Optional[float] = None


class MotionClipCurriculum:
    """Manages curriculum learning for motion clips."""

    def __init__(self, config: DictConfig, motion_clips: List[str]):
        self.config = config
        self.motion_clips = motion_clips
        self.clip_difficulties = self._compute_clip_difficulties()
        self.current_level = 0.0
        self.clips_by_difficulty = self._sort_clips_by_difficulty()

    def _compute_clip_difficulties(self) -> Dict[str, float]:
        """Compute difficulty scores for motion clips."""
        difficulties = {}

        for clip_path in self.motion_clips:
            try:
                # Load clip to analyze difficulty
                trajectory = load_g1_trajectory_from_npz(clip_path)

                # Simple difficulty heuristics
                duration = trajectory.metadata.get('duration', 1.0)
                joint_vel_std = np.mean(np.std(trajectory.joint_velocities, axis=0))
                root_vel_mag = np.mean(np.linalg.norm(trajectory.root_linear_velocities, axis=1))

                # Combine metrics (normalize to 0-1)
                difficulty = (
                    min(duration / 10.0, 1.0) * 0.3 +  # Longer clips are harder
                    min(joint_vel_std / 5.0, 1.0) * 0.4 +  # Faster motions are harder
                    min(root_vel_mag / 2.0, 1.0) * 0.3   # Higher speeds are harder
                )

                difficulties[clip_path] = difficulty

            except Exception as e:
                logger.warning(f"Failed to compute difficulty for {clip_path}: {e}")
                difficulties[clip_path] = 0.5  # Default medium difficulty

        return difficulties

    def _sort_clips_by_difficulty(self) -> List[str]:
        """Sort clips by difficulty."""
        return sorted(self.motion_clips, key=lambda x: self.clip_difficulties.get(x, 0.5))

    def update_curriculum(self, success_rate: float, step: int):
        """Update curriculum level based on performance."""
        if not self.config.curriculum.enabled:
            return

        target_success_rate = self.config.curriculum.target_success_rate
        adaptation_rate = self.config.curriculum.adaptation_rate

        # Adapt curriculum level
        if success_rate > target_success_rate:
            self.current_level = min(1.0, self.current_level + adaptation_rate)
        else:
            self.current_level = max(0.0, self.current_level - adaptation_rate)

        logger.info(f"Curriculum level: {self.current_level:.3f}, Success rate: {success_rate:.3f}")

    def sample_clips(self, batch_size: int) -> List[str]:
        """Sample motion clips based on current curriculum level."""
        if not self.config.curriculum.enabled:
            return random.choices(self.motion_clips, k=batch_size)

        # Determine available clips based on curriculum level
        max_difficulty = self.current_level
        available_clips = [
            clip for clip in self.clips_by_difficulty
            if self.clip_difficulties.get(clip, 0.5) <= max_difficulty
        ]

        if not available_clips:
            available_clips = self.clips_by_difficulty[:1]  # At least one clip

        return random.choices(available_clips, k=batch_size)


class G1TrainingEnvironment:
    """Manages G1 training environment with motion clips."""

    def __init__(self, config: DictConfig, motion_clips: List[str]):
        self.config = config
        self.motion_clips = motion_clips
        self.curriculum = MotionClipCurriculum(config, motion_clips)
        self.normalizer = self._load_normalizer()

    def _load_normalizer(self) -> Optional[MotionNormalizer]:
        """Load motion data normalizer if available."""
        norm_path = self.config.data.get('normalizer_path')
        if norm_path and os.path.exists(norm_path):
            normalizer = MotionNormalizer()
            normalizer.load_statistics(norm_path)
            return normalizer
        return None

    def create_vectorized_env(self, num_envs: int):
        """Create vectorized environment for training."""
        # Note: This is a simplified version - you'd want to implement proper vectorization
        envs = []

        for i in range(num_envs):
            env = create_g1_env(
                model_path=self.config.env.model_path,
                config=self.config.env
            )
            envs.append(env)

        return envs

    def reset_with_motion_clips(self, envs: List, clips: List[str]):
        """Reset environments with specific motion clips."""
        observations = []

        for env, clip_path in zip(envs, clips):
            try:
                # Load and set reference motion
                trajectory = load_g1_trajectory_from_npz(clip_path)
                if self.normalizer:
                    trajectory_dict = {
                        'joint_positions': trajectory.joint_positions,
                        'joint_velocities': trajectory.joint_velocities,
                        'root_positions': trajectory.root_positions,
                        'root_orientations': trajectory.root_orientations,
                        'root_linear_velocities': trajectory.root_linear_velocities,
                        'root_angular_velocities': trajectory.root_angular_velocities
                    }
                    normalized = self.normalizer.normalize(trajectory_dict)
                    # Update trajectory with normalized data
                    trajectory.joint_positions = normalized['joint_positions']
                    trajectory.joint_velocities = normalized['joint_velocities']
                    trajectory.root_positions = normalized['root_positions']
                    trajectory.root_linear_velocities = normalized['root_linear_velocities']
                    trajectory.root_angular_velocities = normalized['root_angular_velocities']

                env.set_reference_motion(trajectory)
                obs, _ = env.reset()
                observations.append(obs)

            except Exception as e:
                logger.warning(f"Failed to load motion clip {clip_path}: {e}")
                obs, _ = env.reset()  # Reset without reference motion
                observations.append(obs)

        return np.array(observations)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Additional settings for deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(config: DictConfig) -> tuple:
    """Setup logging backends (TensorBoard and/or W&B)."""
    writers = {}

    # TensorBoard
    if config.logging.use_tensorboard:
        log_dir = Path(config.logging.log_dir) / "tensorboard"
        log_dir.mkdir(parents=True, exist_ok=True)
        writers['tensorboard'] = SummaryWriter(str(log_dir))

    # Weights & Biases
    if config.logging.use_wandb:
        wandb.init(
            project=config.logging.wandb_project,
            name=config.logging.experiment_name,
            config=OmegaConf.to_container(config, resolve=True),
            tags=config.logging.get('tags', [])
        )
        writers['wandb'] = wandb

    return writers


def log_metrics(writers: Dict, metrics: TrainingMetrics):
    """Log metrics to all configured backends."""
    metrics_dict = {
        'train/mean_reward': metrics.mean_reward,
        'train/mean_episode_length': metrics.mean_episode_length,
        'train/policy_loss': metrics.policy_loss,
        'train/value_loss': metrics.value_loss,
        'train/entropy_loss': metrics.entropy_loss,
        'train/explained_variance': metrics.explained_variance,
        'train/fps': metrics.fps,
    }

    if metrics.curriculum_level is not None:
        metrics_dict['train/curriculum_level'] = metrics.curriculum_level

    # TensorBoard
    if 'tensorboard' in writers:
        for key, value in metrics_dict.items():
            writers['tensorboard'].add_scalar(key, value, metrics.step)

    # Weights & Biases
    if 'wandb' in writers:
        writers['wandb'].log(metrics_dict, step=metrics.step)


def evaluate_policy(policy: G1Policy, env_manager: G1TrainingEnvironment,
                   config: DictConfig, device: torch.device) -> Dict[str, float]:
    """Evaluate policy performance."""
    policy.eval()

    num_eval_episodes = config.eval.num_episodes
    eval_rewards = []
    eval_episode_lengths = []
    eval_success_rates = []

    # Create evaluation environment
    eval_envs = env_manager.create_vectorized_env(1)

    with torch.no_grad():
        for _ in range(num_eval_episodes):
            # Sample random clip for evaluation
            eval_clips = env_manager.curriculum.sample_clips(1)
            obs = env_manager.reset_with_motion_clips(eval_envs, eval_clips)

            episode_reward = 0
            episode_length = 0
            done = False

            while not done and episode_length < config.env.max_episode_steps:
                obs_tensor = torch.FloatTensor(obs).to(device)
                action, _ = policy.act(obs_tensor, deterministic=True)

                obs, reward, terminated, truncated, info = eval_envs[0].step(action.cpu().numpy())
                done = terminated or truncated

                episode_reward += reward
                episode_length += 1

                obs = obs.reshape(1, -1)

            eval_rewards.append(episode_reward)
            eval_episode_lengths.append(episode_length)

            # Success criteria (customize based on your task)
            success = episode_reward > config.eval.success_threshold
            eval_success_rates.append(float(success))

    policy.train()

    return {
        'eval/mean_reward': np.mean(eval_rewards),
        'eval/mean_episode_length': np.mean(eval_episode_lengths),
        'eval/success_rate': np.mean(eval_success_rates),
        'eval/reward_std': np.std(eval_rewards)
    }


def save_checkpoint(step: int, policy: G1Policy, critic: G1Critic,
                   optimizer_policy, optimizer_critic, config: DictConfig):
    """Save training checkpoint."""
    checkpoint_dir = Path(config.training.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'step': step,
        'policy_state_dict': policy.state_dict(),
        'critic_state_dict': critic.state_dict(),
        'optimizer_policy_state_dict': optimizer_policy.state_dict(),
        'optimizer_critic_state_dict': optimizer_critic.state_dict(),
        'config': OmegaConf.to_container(config, resolve=True)
    }

    checkpoint_path = checkpoint_dir / f"checkpoint_step_{step}.pth"
    torch.save(checkpoint, checkpoint_path)

    # Also save as latest
    latest_path = checkpoint_dir / "checkpoint_latest.pth"
    torch.save(checkpoint, latest_path)

    logger.info(f"Saved checkpoint: {checkpoint_path}")


def load_checkpoint(checkpoint_path: str, policy: G1Policy, critic: G1Critic,
                   optimizer_policy, optimizer_critic, device: torch.device) -> int:
    """Load training checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    policy.load_state_dict(checkpoint['policy_state_dict'])
    critic.load_state_dict(checkpoint['critic_state_dict'])
    optimizer_policy.load_state_dict(checkpoint['optimizer_policy_state_dict'])
    optimizer_critic.load_state_dict(checkpoint['optimizer_critic_state_dict'])

    logger.info(f"Loaded checkpoint from step {checkpoint['step']}")
    return checkpoint['step']


def main():
    parser = argparse.ArgumentParser(description="PPO Motion Imitation Training for Unitree G1")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--eval-only", action="store_true", help="Run evaluation only")
    parser.add_argument("--seed", type=int, help="Random seed override")

    args = parser.parse_args()

    # Load configuration
    config = OmegaConf.load(args.config)

    # Override seed if provided
    if args.seed is not None:
        config.training.seed = args.seed

    # Set random seed
    set_seed(config.training.seed)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() and config.training.use_cuda else "cpu")
    logger.info(f"Using device: {device}")

    # Setup logging
    writers = setup_logging(config)

    # Load motion clips
    motion_clips_dir = Path(config.data.motion_clips_dir)
    motion_clips = list(motion_clips_dir.glob("**/*.npz"))
    if not motion_clips:
        raise ValueError(f"No motion clips found in {motion_clips_dir}")

    motion_clips = [str(p) for p in motion_clips]
    logger.info(f"Found {len(motion_clips)} motion clips")

    # Create environment manager
    env_manager = G1TrainingEnvironment(config, motion_clips)

    # Create sample environment to get observation/action dimensions
    sample_env = env_manager.create_vectorized_env(1)[0]
    obs_space = sample_env.observation_space
    action_space = sample_env.action_space

    obs_dim = obs_space.shape[0] if hasattr(obs_space, 'shape') else obs_space.n
    action_dim = action_space.shape[0] if hasattr(action_space, 'shape') else action_space.n

    logger.info(f"Observation dim: {obs_dim}, Action dim: {action_dim}")

    # Initialize networks
    policy = G1Policy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        config=config.policy
    ).to(device)

    critic = G1Critic(
        obs_dim=obs_dim,
        config=config.critic
    ).to(device)

    # Initialize optimizers
    optimizer_policy = torch.optim.Adam(
        policy.parameters(),
        lr=config.training.policy_lr,
        eps=config.training.adam_eps
    )

    optimizer_critic = torch.optim.Adam(
        critic.parameters(),
        lr=config.training.critic_lr,
        eps=config.training.adam_eps
    )

    # Initialize PPO trainer
    ppo_trainer = PPOTrainer(
        policy=policy,
        critic=critic,
        optimizer_policy=optimizer_policy,
        optimizer_critic=optimizer_critic,
        config=config.ppo,
        device=device
    )

    # Load checkpoint if resuming
    start_step = 0
    if args.resume:
        start_step = load_checkpoint(
            args.resume, policy, critic, optimizer_policy, optimizer_critic, device
        )

    # Evaluation only mode
    if args.eval_only:
        logger.info("Running evaluation only...")
        eval_metrics = evaluate_policy(policy, env_manager, config, device)
        for key, value in eval_metrics.items():
            logger.info(f"{key}: {value:.4f}")
        return

    # Training loop
    logger.info("Starting training...")

    step = start_step
    episode = 0
    best_eval_reward = float('-inf')
    episodes_without_improvement = 0

    # Create training environments
    num_envs = config.training.num_envs
    envs = env_manager.create_vectorized_env(num_envs)

    # Create rollout buffer
    buffer = RolloutBuffer(
        buffer_size=config.ppo.buffer_size,
        obs_dim=obs_dim,
        action_dim=action_dim,
        num_envs=num_envs,
        device=device
    )

    # Initialize environments
    current_clips = env_manager.curriculum.sample_clips(num_envs)
    observations = env_manager.reset_with_motion_clips(envs, current_clips)

    import time
    start_time = time.time()

    while step < config.training.max_steps:
        # Collect rollouts
        episode_rewards = []
        episode_lengths = []

        for rollout_step in range(config.ppo.buffer_size):
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(observations).to(device)
                actions, log_probs = policy.act(obs_tensor)
                values = critic(obs_tensor)

            # Step environments
            next_observations = []
            rewards = []
            dones = []
            infos = []

            for i, env in enumerate(envs):
                obs, reward, terminated, truncated, info = env.step(actions[i].cpu().numpy())
                done = terminated or truncated

                next_observations.append(obs)
                rewards.append(reward)
                dones.append(done)
                infos.append(info)

                # Track episode metrics
                if done:
                    episode_rewards.append(info.get('episode_reward', reward))
                    episode_lengths.append(info.get('episode_length', 1))
                    episode += 1

                    # Reset with new motion clip
                    new_clip = env_manager.curriculum.sample_clips(1)[0]
                    obs = env_manager.reset_with_motion_clips([env], [new_clip])[0]
                    next_observations[i] = obs

            # Store transition
            buffer.add(
                observations=observations,
                actions=actions.cpu().numpy(),
                rewards=np.array(rewards),
                dones=np.array(dones),
                values=values.cpu().numpy(),
                log_probs=log_probs.cpu().numpy()
            )

            observations = np.array(next_observations)
            step += num_envs

        # Compute returns
        with torch.no_grad():
            next_values = critic(torch.FloatTensor(observations).to(device))
            buffer.compute_returns(next_values.cpu().numpy(), config.ppo.gamma, config.ppo.gae_lambda)

        # Update policy
        train_metrics = ppo_trainer.update(buffer)
        buffer.clear()

        # Update curriculum
        if episode_rewards:
            success_rate = np.mean([r > config.eval.success_threshold for r in episode_rewards])
            env_manager.curriculum.update_curriculum(success_rate, step)

        # Logging
        if step % config.logging.log_interval == 0:
            elapsed_time = time.time() - start_time
            fps = step / elapsed_time

            metrics = TrainingMetrics(
                step=step,
                episode=episode,
                mean_reward=np.mean(episode_rewards) if episode_rewards else 0,
                mean_episode_length=np.mean(episode_lengths) if episode_lengths else 0,
                policy_loss=train_metrics.get('policy_loss', 0),
                value_loss=train_metrics.get('value_loss', 0),
                entropy_loss=train_metrics.get('entropy_loss', 0),
                explained_variance=train_metrics.get('explained_variance', 0),
                fps=fps,
                curriculum_level=env_manager.curriculum.current_level
            )

            log_metrics(writers, metrics)

            logger.info(
                f"Step: {step:8d} | Episode: {episode:6d} | "
                f"Reward: {metrics.mean_reward:8.3f} | "
                f"FPS: {fps:6.1f} | "
                f"Curriculum: {metrics.curriculum_level:.3f}"
            )

        # Evaluation
        if step % config.eval.eval_interval == 0:
            eval_metrics = evaluate_policy(policy, env_manager, config, device)

            # Log evaluation metrics
            if 'tensorboard' in writers:
                for key, value in eval_metrics.items():
                    writers['tensorboard'].add_scalar(key, value, step)
            if 'wandb' in writers:
                writers['wandb'].log(eval_metrics, step=step)

            logger.info(f"Eval - Reward: {eval_metrics['eval/mean_reward']:.3f}, "
                       f"Success: {eval_metrics['eval/success_rate']:.3f}")

            # Model selection
            current_eval_reward = eval_metrics['eval/mean_reward']
            if current_eval_reward > best_eval_reward:
                best_eval_reward = current_eval_reward
                episodes_without_improvement = 0

                # Save best model
                best_model_path = Path(config.training.checkpoint_dir) / "best_model.pth"
                torch.save({
                    'step': step,
                    'policy_state_dict': policy.state_dict(),
                    'critic_state_dict': critic.state_dict(),
                    'eval_reward': best_eval_reward
                }, best_model_path)

            else:
                episodes_without_improvement += 1

            # Early stopping
            if (config.training.early_stopping.enabled and
                episodes_without_improvement >= config.training.early_stopping.patience):
                logger.info(f"Early stopping triggered after {episodes_without_improvement} evaluations without improvement")
                break

        # Checkpointing
        if step % config.training.checkpoint_interval == 0:
            save_checkpoint(step, policy, critic, optimizer_policy, optimizer_critic, config)

    # Final checkpoint
    save_checkpoint(step, policy, critic, optimizer_policy, optimizer_critic, config)

    # Close logging
    if 'tensorboard' in writers:
        writers['tensorboard'].close()
    if 'wandb' in writers:
        writers['wandb'].finish()

    logger.info("Training completed!")


if __name__ == "__main__":
    main()