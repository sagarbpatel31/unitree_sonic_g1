"""
Real-time control loop for deployed G1 controllers.

This module provides a production-ready control loop that integrates trained
policies with hardware interfaces for real-time robot control.
"""

import time
import signal
import threading
import logging
from typing import Dict, Any, Optional, List
import numpy as np
from pathlib import Path
import json
from dataclasses import dataclass
from enum import Enum

from sonic_g1.deploy import RuntimeInferenceEngine, G1HardwareAdapter, SafetyFilter

logger = logging.getLogger(__name__)


class ControllerState(Enum):
    """Controller state enumeration."""
    IDLE = "idle"
    INITIALIZING = "initializing"
    RUNNING = "running"
    EMERGENCY_STOP = "emergency_stop"
    ERROR = "error"
    SHUTDOWN = "shutdown"


@dataclass
class ControlStats:
    """Control loop performance statistics."""
    loop_count: int = 0
    total_time: float = 0.0
    max_loop_time: float = 0.0
    missed_deadlines: int = 0
    inference_failures: int = 0
    hardware_failures: int = 0
    safety_violations: int = 0
    emergency_stops: int = 0


class RealtimeController:
    """
    Real-time control loop for G1 robot deployment.

    Integrates inference engine, hardware adapter, and safety systems
    for production robot control with deterministic timing.
    """

    def __init__(self, config_path: str):
        """
        Initialize real-time controller.

        Args:
            config_path: Path to controller configuration file
        """
        self.config = self._load_config(config_path)

        # Control timing
        self.control_frequency = self.config['control']['frequency']
        self.control_period = 1.0 / self.control_frequency
        self.deadline_tolerance = self.config['control'].get('deadline_tolerance', 0.1)

        # State management
        self.state = ControllerState.IDLE
        self.running = False
        self.emergency_stop_triggered = False

        # Components (initialized in start())
        self.inference_engine: Optional[RuntimeInferenceEngine] = None
        self.hardware_adapter: Optional[G1HardwareAdapter] = None
        self.safety_filter: Optional[SafetyFilter] = None

        # Threading
        self.control_thread: Optional[threading.Thread] = None
        self.monitor_thread: Optional[threading.Thread] = None
        self.shutdown_event = threading.Event()

        # Statistics and monitoring
        self.stats = ControlStats()
        self.performance_log: List[Dict[str, float]] = []
        self.max_log_entries = self.config.get('logging', {}).get('max_entries', 1000)

        # Safety limits
        self.max_consecutive_failures = self.config['safety'].get('max_consecutive_failures', 5)
        self.consecutive_failures = 0

        # Signal handling
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info("Initialized RealtimeController")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load controller configuration."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)

            # Validate required fields
            required_fields = ['control', 'inference', 'hardware', 'safety']
            for field in required_fields:
                if field not in config:
                    raise ValueError(f"Missing required config field: {field}")

            return config

        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise

    def start(self) -> bool:
        """
        Start the real-time controller.

        Returns:
            True if started successfully, False otherwise
        """
        if self.state != ControllerState.IDLE:
            logger.error(f"Cannot start controller in state: {self.state}")
            return False

        logger.info("Starting real-time controller")
        self.state = ControllerState.INITIALIZING

        try:
            # Initialize inference engine
            logger.info("Initializing inference engine")
            self.inference_engine = RuntimeInferenceEngine(self.config['inference'])

            # Initialize hardware adapter
            logger.info("Initializing hardware adapter")
            self.hardware_adapter = G1HardwareAdapter(self.config['hardware'])

            # Connect to hardware
            if not self.hardware_adapter.connect():
                raise RuntimeError("Failed to connect to hardware")

            # Initialize safety filter
            logger.info("Initializing safety filter")
            self.safety_filter = SafetyFilter(self.config['safety'])

            # Perform health checks
            if not self._health_check():
                raise RuntimeError("Health check failed")

            # Start control thread
            self.running = True
            self.control_thread = threading.Thread(target=self._control_loop, daemon=False)
            self.control_thread.start()

            # Start monitoring thread
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()

            self.state = ControllerState.RUNNING
            logger.info("Real-time controller started successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to start controller: {e}")
            self.state = ControllerState.ERROR
            self._cleanup()
            return False

    def stop(self, emergency: bool = False) -> bool:
        """
        Stop the real-time controller.

        Args:
            emergency: Whether this is an emergency stop

        Returns:
            True if stopped successfully, False otherwise
        """
        if emergency:
            logger.warning("EMERGENCY STOP triggered")
            self.emergency_stop_triggered = True
            self.state = ControllerState.EMERGENCY_STOP
            self.stats.emergency_stops += 1

            # Immediate hardware stop
            if self.hardware_adapter:
                self.hardware_adapter.emergency_stop()

        self.running = False
        self.shutdown_event.set()

        # Wait for threads to finish
        if self.control_thread and self.control_thread.is_alive():
            self.control_thread.join(timeout=5.0)
            if self.control_thread.is_alive():
                logger.warning("Control thread did not shut down gracefully")

        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)

        self._cleanup()
        self.state = ControllerState.SHUTDOWN
        logger.info("Real-time controller stopped")
        return True

    def _control_loop(self):
        """Main real-time control loop."""
        logger.info("Starting control loop")
        next_time = time.time()

        while self.running and not self.shutdown_event.is_set():
            loop_start = time.time()

            try:
                # Check if we're running late
                if loop_start > next_time + self.deadline_tolerance * self.control_period:
                    self.stats.missed_deadlines += 1
                    logger.warning(f"Missed deadline by {(loop_start - next_time) * 1000:.1f}ms")

                # Perform control step
                success = self._control_step()

                if success:
                    self.consecutive_failures = 0
                else:
                    self.consecutive_failures += 1
                    if self.consecutive_failures >= self.max_consecutive_failures:
                        logger.error("Max consecutive failures reached, triggering emergency stop")
                        self.stop(emergency=True)
                        break

                # Update timing statistics
                loop_time = time.time() - loop_start
                self._update_stats(loop_time)

                # Calculate next iteration time
                next_time += self.control_period

                # Sleep until next iteration
                sleep_time = next_time - time.time()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    # Running behind - don't sleep but update next_time
                    next_time = time.time()

            except Exception as e:
                logger.error(f"Control loop error: {e}")
                self.stats.inference_failures += 1
                self.consecutive_failures += 1

                if self.consecutive_failures >= self.max_consecutive_failures:
                    self.stop(emergency=True)
                    break

        logger.info("Control loop finished")

    def _control_step(self) -> bool:
        """
        Execute single control step.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Get observation from hardware
            observation = self.hardware_adapter.get_observation()
            if observation is None:
                logger.warning("Failed to get observation from hardware")
                self.stats.hardware_failures += 1
                return False

            # Convert to numpy array
            obs_array = observation.to_array()

            # Run inference
            try:
                action, inference_info = self.inference_engine.predict(obs_array)

                # Check for inference timeout
                if inference_info.get('timeout', False):
                    logger.warning("Inference timeout detected")
                    action = self.inference_engine.emergency_action()

            except Exception as e:
                logger.error(f"Inference failed: {e}")
                self.stats.inference_failures += 1
                action = self.inference_engine.emergency_action()
                inference_info = {'error': str(e)}

            # Apply safety filter
            filtered_action, safety_info = self.safety_filter.filter_action(
                action, observation, inference_info
            )

            if safety_info.get('violation', False):
                logger.warning("Safety violation detected")
                self.stats.safety_violations += 1

            # Send action to hardware
            success = self.hardware_adapter.send_action(filtered_action)
            if not success:
                logger.warning("Failed to send action to hardware")
                self.stats.hardware_failures += 1
                return False

            # Log performance data
            self._log_performance(observation, action, filtered_action,
                                inference_info, safety_info)

            return True

        except Exception as e:
            logger.error(f"Control step failed: {e}")
            return False

    def _health_check(self) -> bool:
        """Perform comprehensive health check."""
        logger.info("Performing health check")

        health_checks = []

        # Check inference engine
        if self.inference_engine:
            inference_health = self.inference_engine.health_check()
            health_checks.append(all(inference_health.values()))
            logger.info(f"Inference engine health: {inference_health}")

        # Check hardware adapter
        if self.hardware_adapter:
            hardware_health = self.hardware_adapter.health_check()
            health_checks.append(hardware_health)
            logger.info(f"Hardware adapter health: {hardware_health}")

        # Check safety filter
        if self.safety_filter:
            safety_health = self.safety_filter.health_check()
            health_checks.append(safety_health)
            logger.info(f"Safety filter health: {safety_health}")

        overall_health = all(health_checks)
        logger.info(f"Overall health check: {'PASSED' if overall_health else 'FAILED'}")
        return overall_health

    def _monitor_loop(self):
        """Monitoring thread for system health and performance."""
        monitor_interval = self.config.get('monitoring', {}).get('interval', 5.0)

        while self.running and not self.shutdown_event.wait(monitor_interval):
            try:
                # Check system health
                if not self._health_check():
                    logger.warning("Health check failed during monitoring")

                # Log performance statistics
                if self.stats.loop_count > 0:
                    avg_loop_time = self.stats.total_time / self.stats.loop_count
                    logger.info(f"Control loop stats: "
                              f"avg={avg_loop_time*1000:.2f}ms, "
                              f"max={self.stats.max_loop_time*1000:.2f}ms, "
                              f"missed_deadlines={self.stats.missed_deadlines}, "
                              f"failures={self.stats.inference_failures}")

            except Exception as e:
                logger.error(f"Monitor loop error: {e}")

    def _update_stats(self, loop_time: float):
        """Update control loop statistics."""
        self.stats.loop_count += 1
        self.stats.total_time += loop_time
        self.stats.max_loop_time = max(self.stats.max_loop_time, loop_time)

    def _log_performance(self, observation, action, filtered_action,
                        inference_info, safety_info):
        """Log performance data for analysis."""
        if len(self.performance_log) >= self.max_log_entries:
            self.performance_log.pop(0)  # Remove oldest entry

        log_entry = {
            'timestamp': time.time(),
            'inference_time_ms': inference_info.get('inference_time_ms', 0),
            'action_norm': float(np.linalg.norm(action)),
            'filtered_action_norm': float(np.linalg.norm(filtered_action)),
            'safety_violation': safety_info.get('violation', False),
            'timeout': inference_info.get('timeout', False)
        }

        self.performance_log.append(log_entry)

    def _signal_handler(self, signum, frame):
        """Handle system signals for graceful shutdown."""
        logger.info(f"Received signal {signum}, initiating shutdown")
        self.stop(emergency=False)

    def _cleanup(self):
        """Cleanup resources."""
        if self.hardware_adapter:
            self.hardware_adapter.shutdown()

        if self.inference_engine:
            self.inference_engine.shutdown()

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive controller statistics."""
        base_stats = {
            'loop_count': self.stats.loop_count,
            'avg_loop_time_ms': (self.stats.total_time / max(1, self.stats.loop_count)) * 1000,
            'max_loop_time_ms': self.stats.max_loop_time * 1000,
            'missed_deadlines': self.stats.missed_deadlines,
            'deadline_miss_rate': self.stats.missed_deadlines / max(1, self.stats.loop_count),
            'inference_failures': self.stats.inference_failures,
            'hardware_failures': self.stats.hardware_failures,
            'safety_violations': self.stats.safety_violations,
            'emergency_stops': self.stats.emergency_stops,
            'state': self.state.value,
            'control_frequency': self.control_frequency
        }

        # Add inference engine stats if available
        if self.inference_engine:
            inference_stats = self.inference_engine.get_statistics()
            base_stats['inference'] = inference_stats

        return base_stats

    def save_performance_log(self, output_path: str):
        """Save performance log to file."""
        log_data = {
            'controller_stats': self.get_statistics(),
            'performance_log': self.performance_log,
            'config': self.config
        }

        with open(output_path, 'w') as f:
            json.dump(log_data, f, indent=2, default=str)

        logger.info(f"Performance log saved to {output_path}")


def create_realtime_config_template() -> Dict[str, Any]:
    """Create template configuration for real-time controller."""
    return {
        'control': {
            'frequency': 100.0,  # Hz
            'deadline_tolerance': 0.1  # fraction of control period
        },
        'inference': {
            'model': {
                'path': 'path/to/model.onnx',
                'format': 'onnx'
            },
            'action_limits': [-1.0, 1.0],
            'action_scale': 1.0,
            'use_action_filter': True,
            'filter_cutoff_hz': 10.0,
            'enable_watchdog': True,
            'watchdog_timeout_ms': 50.0
        },
        'hardware': {
            'port': '/dev/ttyUSB0',
            'baudrate': 115200,
            'connection_timeout': 10.0,
            'command_timeout': 0.1
        },
        'safety': {
            'joint_position_limits': {
                'min': [-2.0] * 22,
                'max': [2.0] * 22
            },
            'joint_velocity_limits': {
                'max': [10.0] * 22
            },
            'action_rate_limit': 0.1,
            'max_consecutive_failures': 5
        },
        'monitoring': {
            'interval': 5.0,  # seconds
            'enable_logging': True
        },
        'logging': {
            'max_entries': 1000,
            'level': 'INFO'
        }
    }


def main():
    """Main entry point for real-time controller."""
    import argparse

    parser = argparse.ArgumentParser(description='Real-time G1 controller')
    parser.add_argument('--config', required=True, help='Controller configuration file')
    parser.add_argument('--log-level', default='INFO', help='Logging level')
    parser.add_argument('--log-file', help='Log file path')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename=args.log_file
    )

    # Create and start controller
    try:
        controller = RealtimeController(args.config)

        if controller.start():
            logger.info("Controller started successfully")

            # Keep running until interrupted
            try:
                while controller.state == ControllerState.RUNNING:
                    time.sleep(1.0)
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received")

            # Stop controller
            controller.stop()

            # Save performance data
            output_path = f"controller_log_{int(time.time())}.json"
            controller.save_performance_log(output_path)

            # Print final statistics
            stats = controller.get_statistics()
            print("\nFinal Controller Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")

        else:
            logger.error("Failed to start controller")
            return 1

    except Exception as e:
        logger.error(f"Controller error: {e}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())