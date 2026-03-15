"""
Video recording utilities for G1 policy evaluation.

This module provides video recording capabilities for capturing
policy rollouts during evaluation and analysis.
"""

import numpy as np
import cv2
import mujoco
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
import logging
import time
import tempfile
import subprocess
import os

logger = logging.getLogger(__name__)


class VideoRecorder:
    """
    Video recorder for MuJoCo environments.

    Supports both real-time recording and post-processing of
    evaluation rollouts with customizable camera angles and settings.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize video recorder.

        Args:
            config: Video recording configuration
        """
        self.config = config

        # Video settings
        self.width = config.get('width', 1280)
        self.height = config.get('height', 720)
        self.fps = config.get('fps', 30)
        self.codec = config.get('codec', 'mp4v')

        # Camera settings
        self.camera_name = config.get('camera_name', None)  # None for free camera
        self.camera_distance = config.get('camera_distance', 3.0)
        self.camera_elevation = config.get('camera_elevation', -20)
        self.camera_azimuth = config.get('camera_azimuth', 45)
        self.track_robot = config.get('track_robot', True)

        # Multi-camera recording
        self.multi_camera = config.get('multi_camera', False)
        self.camera_configs = config.get('camera_configs', [])

        # Recording state
        self.is_recording = False
        self.video_writer = None
        self.current_video_path = None
        self.frame_buffer = []
        self.recorded_frames = 0

        # MuJoCo renderer
        self.renderer = None
        self.env = None

        logger.info(f"Initialized VideoRecorder: {self.width}x{self.height} @ {self.fps}fps")

    def start_recording(self, policy_name: str, test_suite: str, episode: int) -> str:
        """
        Start recording a new video.

        Args:
            policy_name: Name of the policy being evaluated
            test_suite: Test suite name
            episode: Episode number

        Returns:
            Path to the video file being recorded
        """
        if self.is_recording:
            logger.warning("Already recording. Stopping current recording.")
            self.stop_recording()

        # Create video filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{policy_name}_{test_suite}_ep{episode:03d}_{timestamp}.mp4"
        video_dir = Path(self.config.get('output_dir', 'videos'))
        video_dir.mkdir(parents=True, exist_ok=True)

        self.current_video_path = str(video_dir / filename)

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self.video_writer = cv2.VideoWriter(
            self.current_video_path, fourcc, self.fps, (self.width, self.height)
        )

        if not self.video_writer.isOpened():
            logger.error("Failed to initialize video writer")
            self.video_writer = None
            return None

        self.is_recording = True
        self.recorded_frames = 0
        self.frame_buffer = []

        logger.info(f"Started recording: {self.current_video_path}")
        return self.current_video_path

    def capture_frame(self, env: Any, custom_text: Optional[str] = None):
        """
        Capture a frame from the environment.

        Args:
            env: MuJoCo environment
            custom_text: Optional text overlay
        """
        if not self.is_recording or not self.video_writer:
            return

        # Store environment reference for camera tracking
        if self.env is None:
            self.env = env
            self._setup_renderer(env)

        # Render frame
        if self.multi_camera:
            frame = self._render_multi_camera_frame(env)
        else:
            frame = self._render_single_camera_frame(env)

        # Add overlays
        if self.config.get('add_overlays', True):
            frame = self._add_overlays(frame, env, custom_text)

        # Write frame
        if frame is not None:
            self.video_writer.write(frame)
            self.recorded_frames += 1

            # Store frame in buffer for post-processing
            if self.config.get('store_frames', False):
                self.frame_buffer.append(frame.copy())

    def _setup_renderer(self, env: Any):
        """Setup MuJoCo renderer."""
        try:
            # Create renderer with appropriate context
            self.renderer = mujoco.Renderer(env.model, self.height, self.width)

            # Configure camera
            if self.camera_name:
                # Use named camera
                camera_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name)
                if camera_id >= 0:
                    self.renderer.camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
                    self.renderer.camera.fixedcamid = camera_id
                else:
                    logger.warning(f"Camera '{self.camera_name}' not found, using free camera")
                    self._setup_free_camera()
            else:
                self._setup_free_camera()

        except Exception as e:
            logger.error(f"Failed to setup renderer: {e}")
            self.renderer = None

    def _setup_free_camera(self):
        """Setup free camera with tracking."""
        if self.renderer:
            self.renderer.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            self.renderer.camera.distance = self.camera_distance
            self.renderer.camera.elevation = self.camera_elevation
            self.renderer.camera.azimuth = self.camera_azimuth

    def _render_single_camera_frame(self, env: Any) -> Optional[np.ndarray]:
        """Render frame from single camera."""
        if not self.renderer:
            return None

        try:
            # Update camera tracking
            if self.track_robot:
                self._update_camera_tracking(env)

            # Render frame
            self.renderer.update_scene(env.data, camera=self.renderer.camera)
            rgb_array = self.renderer.render()

            # Convert RGB to BGR for OpenCV
            frame = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
            return frame

        except Exception as e:
            logger.error(f"Failed to render frame: {e}")
            return None

    def _render_multi_camera_frame(self, env: Any) -> Optional[np.ndarray]:
        """Render multi-camera view."""
        if not self.camera_configs:
            return self._render_single_camera_frame(env)

        try:
            camera_frames = []

            for camera_config in self.camera_configs:
                # Setup camera for this view
                if 'name' in camera_config:
                    camera_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_config['name'])
                    if camera_id >= 0:
                        self.renderer.camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
                        self.renderer.camera.fixedcamid = camera_id
                    else:
                        continue
                else:
                    # Free camera with specified parameters
                    self.renderer.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
                    self.renderer.camera.distance = camera_config.get('distance', 3.0)
                    self.renderer.camera.elevation = camera_config.get('elevation', -20)
                    self.renderer.camera.azimuth = camera_config.get('azimuth', 45)

                # Render this camera view
                self.renderer.update_scene(env.data, camera=self.renderer.camera)
                rgb_array = self.renderer.render()
                frame = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

                # Resize if needed
                view_width = camera_config.get('width', self.width // 2)
                view_height = camera_config.get('height', self.height // 2)
                frame = cv2.resize(frame, (view_width, view_height))

                camera_frames.append(frame)

            # Combine camera views
            if len(camera_frames) == 2:
                # Side by side
                combined_frame = np.hstack(camera_frames)
            elif len(camera_frames) == 4:
                # 2x2 grid
                top_row = np.hstack(camera_frames[:2])
                bottom_row = np.hstack(camera_frames[2:])
                combined_frame = np.vstack([top_row, bottom_row])
            else:
                # Stack vertically
                combined_frame = np.vstack(camera_frames)

            # Resize to target resolution
            combined_frame = cv2.resize(combined_frame, (self.width, self.height))

            return combined_frame

        except Exception as e:
            logger.error(f"Failed to render multi-camera frame: {e}")
            return None

    def _update_camera_tracking(self, env: Any):
        """Update camera to track robot."""
        if not self.track_robot or not self.renderer:
            return

        try:
            # Get robot root position
            root_pos = env.data.qpos[:3]

            # Update camera target
            self.renderer.camera.lookat[:] = root_pos

        except Exception as e:
            logger.debug(f"Failed to update camera tracking: {e}")

    def _add_overlays(self, frame: np.ndarray, env: Any,
                     custom_text: Optional[str] = None) -> np.ndarray:
        """Add information overlays to frame."""
        overlay_frame = frame.copy()

        # Text settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (255, 255, 255)  # White
        thickness = 2

        # Add simulation time
        if hasattr(env, 'data') and hasattr(env.data, 'time'):
            time_text = f"Time: {env.data.time:.2f}s"
            cv2.putText(overlay_frame, time_text, (10, 30), font, font_scale, color, thickness)

        # Add frame number
        frame_text = f"Frame: {self.recorded_frames}"
        cv2.putText(overlay_frame, frame_text, (10, 60), font, font_scale, color, thickness)

        # Add custom text
        if custom_text:
            cv2.putText(overlay_frame, custom_text, (10, 90), font, font_scale, color, thickness)

        # Add performance metrics if available
        if hasattr(env, '_last_reward'):
            reward_text = f"Reward: {env._last_reward:.3f}"
            cv2.putText(overlay_frame, reward_text, (10, overlay_frame.shape[0] - 60),
                       font, font_scale, color, thickness)

        # Add robot state info
        if hasattr(env, 'data'):
            try:
                root_height = env.data.qpos[2]
                height_text = f"Height: {root_height:.3f}m"
                cv2.putText(overlay_frame, height_text, (10, overlay_frame.shape[0] - 30),
                           font, font_scale, color, thickness)
            except:
                pass

        return overlay_frame

    def stop_recording(self):
        """Stop current recording."""
        if not self.is_recording:
            return

        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None

        self.is_recording = False

        logger.info(f"Stopped recording. Saved {self.recorded_frames} frames to: {self.current_video_path}")

        # Post-process video if configured
        if self.config.get('post_process', False):
            self._post_process_video()

        return self.current_video_path

    def _post_process_video(self):
        """Post-process recorded video."""
        if not self.current_video_path or not os.path.exists(self.current_video_path):
            return

        try:
            # Add compression/optimization
            if self.config.get('compress_video', True):
                self._compress_video()

            # Generate thumbnail
            if self.config.get('generate_thumbnail', True):
                self._generate_thumbnail()

        except Exception as e:
            logger.error(f"Failed to post-process video: {e}")

    def _compress_video(self):
        """Compress video using ffmpeg."""
        try:
            input_path = self.current_video_path
            output_path = input_path.replace('.mp4', '_compressed.mp4')

            # Use ffmpeg for compression
            cmd = [
                'ffmpeg', '-y',  # Overwrite output
                '-i', input_path,
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '23',
                '-c:a', 'copy',
                output_path
            ]

            subprocess.run(cmd, capture_output=True, check=True)

            # Replace original with compressed version
            os.replace(output_path, input_path)
            logger.info(f"Compressed video: {input_path}")

        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.warning(f"Failed to compress video (ffmpeg not available?): {e}")
        except Exception as e:
            logger.error(f"Error during video compression: {e}")

    def _generate_thumbnail(self):
        """Generate thumbnail from video."""
        try:
            # Read first frame
            cap = cv2.VideoCapture(self.current_video_path)
            ret, frame = cap.read()
            cap.release()

            if ret:
                # Save thumbnail
                thumbnail_path = self.current_video_path.replace('.mp4', '_thumb.jpg')
                # Resize to thumbnail size
                thumbnail = cv2.resize(frame, (320, 180))
                cv2.imwrite(thumbnail_path, thumbnail)
                logger.debug(f"Generated thumbnail: {thumbnail_path}")

        except Exception as e:
            logger.error(f"Failed to generate thumbnail: {e}")

    def record_episode_highlights(self, episode_data: Dict[str, Any],
                                 policy_name: str, test_suite: str) -> Optional[str]:
        """
        Create highlight video from episode data.

        Args:
            episode_data: Episode data with frame information
            policy_name: Policy name
            test_suite: Test suite name

        Returns:
            Path to highlight video
        """
        if not self.frame_buffer:
            logger.warning("No frames available for highlight creation")
            return None

        try:
            # Create highlight filename
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{policy_name}_{test_suite}_highlights_{timestamp}.mp4"
            video_dir = Path(self.config.get('output_dir', 'videos'))
            highlight_path = str(video_dir / filename)

            # Select highlight frames
            highlight_indices = self._select_highlight_frames(episode_data)

            # Create highlight video
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            highlight_writer = cv2.VideoWriter(
                highlight_path, fourcc, self.fps, (self.width, self.height)
            )

            for idx in highlight_indices:
                if idx < len(self.frame_buffer):
                    frame = self.frame_buffer[idx]
                    # Add highlight text overlay
                    highlight_frame = self._add_highlight_overlay(frame, idx, episode_data)
                    highlight_writer.write(highlight_frame)

            highlight_writer.release()
            logger.info(f"Created highlight video: {highlight_path}")
            return highlight_path

        except Exception as e:
            logger.error(f"Failed to create highlight video: {e}")
            return None

    def _select_highlight_frames(self, episode_data: Dict[str, Any]) -> List[int]:
        """Select interesting frames for highlights."""
        highlights = []

        total_frames = len(self.frame_buffer)
        if total_frames == 0:
            return highlights

        # Always include start and end
        highlights.extend([0, total_frames - 1])

        # Include frames with high rewards
        if 'rewards' in episode_data:
            rewards = episode_data['rewards']
            high_reward_threshold = np.percentile(rewards, 90)

            for i, reward in enumerate(rewards):
                if reward > high_reward_threshold and i not in highlights:
                    highlights.append(i)

        # Include frames with significant state changes
        if 'root_positions' in episode_data:
            positions = episode_data['root_positions']
            for i in range(1, len(positions) - 1):
                # Check for large position changes
                prev_pos = np.array(positions[i-1])
                curr_pos = np.array(positions[i])
                next_pos = np.array(positions[i+1])

                change1 = np.linalg.norm(curr_pos - prev_pos)
                change2 = np.linalg.norm(next_pos - curr_pos)

                if max(change1, change2) > 0.5 and i not in highlights:  # Threshold
                    highlights.append(i)

        # Sort and limit number of highlights
        highlights = sorted(set(highlights))
        max_highlights = self.config.get('max_highlights', 20)

        if len(highlights) > max_highlights:
            # Sample evenly
            step = len(highlights) // max_highlights
            highlights = highlights[::step]

        return highlights

    def _add_highlight_overlay(self, frame: np.ndarray, frame_idx: int,
                              episode_data: Dict[str, Any]) -> np.ndarray:
        """Add highlight-specific overlay to frame."""
        overlay_frame = frame.copy()

        # Highlight text
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (0, 255, 255)  # Yellow
        thickness = 2

        highlight_text = f"Highlight Frame {frame_idx}"
        cv2.putText(overlay_frame, highlight_text, (10, overlay_frame.shape[0] - 10),
                   font, 0.8, color, thickness)

        # Add reason for highlight if available
        if frame_idx < len(episode_data.get('rewards', [])):
            reward = episode_data['rewards'][frame_idx]
            if reward > np.mean(episode_data.get('rewards', [0])):
                reason_text = f"High Reward: {reward:.3f}"
                cv2.putText(overlay_frame, reason_text, (10, overlay_frame.shape[0] - 40),
                           font, 0.6, color, thickness)

        return overlay_frame

    def create_comparison_video(self, video_paths: List[str], output_path: str):
        """
        Create side-by-side comparison video from multiple recordings.

        Args:
            video_paths: List of video file paths
            output_path: Output path for comparison video
        """
        if len(video_paths) < 2:
            logger.warning("Need at least 2 videos for comparison")
            return

        try:
            # Open video captures
            caps = [cv2.VideoCapture(path) for path in video_paths]

            # Get video properties
            fps = int(caps[0].get(cv2.CAP_PROP_FPS))
            frame_count = min([int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in caps])

            # Calculate output dimensions
            if len(video_paths) == 2:
                # Side by side
                out_width = self.width * 2
                out_height = self.height
                layout = 'horizontal'
            elif len(video_paths) == 4:
                # 2x2 grid
                out_width = self.width * 2
                out_height = self.height * 2
                layout = 'grid'
            else:
                # Vertical stack
                out_width = self.width
                out_height = self.height * len(video_paths)
                layout = 'vertical'

            # Create output video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_writer = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))

            # Process frames
            for frame_idx in range(frame_count):
                frames = []

                # Read frame from each video
                for cap in caps:
                    ret, frame = cap.read()
                    if ret:
                        # Resize to standard size
                        frame = cv2.resize(frame, (self.width, self.height))
                        frames.append(frame)
                    else:
                        # Use black frame if video ended
                        black_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                        frames.append(black_frame)

                # Combine frames based on layout
                if layout == 'horizontal':
                    combined_frame = np.hstack(frames)
                elif layout == 'vertical':
                    combined_frame = np.vstack(frames)
                elif layout == 'grid':
                    top_row = np.hstack(frames[:2])
                    bottom_row = np.hstack(frames[2:] if len(frames) > 2 else [frames[0], frames[0]])
                    combined_frame = np.vstack([top_row, bottom_row])

                out_writer.write(combined_frame)

            # Cleanup
            for cap in caps:
                cap.release()
            out_writer.release()

            logger.info(f"Created comparison video: {output_path}")

        except Exception as e:
            logger.error(f"Failed to create comparison video: {e}")

    def __del__(self):
        """Cleanup resources."""
        if self.is_recording:
            self.stop_recording()

        if self.renderer:
            # Cleanup renderer if needed
            pass