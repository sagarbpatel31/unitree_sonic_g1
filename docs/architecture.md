# Architecture Documentation

This document describes the architecture and design principles of the Unitree G1 SONIC-inspired training stack.

## Overview

The system is designed as a modular, production-ready framework for training humanoid robots to perform whole-body motion tracking and imitation. The architecture follows clean separation of concerns with well-defined interfaces between components.

## Core Principles

- **Modularity**: Each component has a single responsibility and clear interfaces
- **Configurability**: All behavior controlled through YAML configuration files
- **Extensibility**: Easy to add new environments, models, and training algorithms
- **Reproducibility**: Comprehensive logging, seeding, and checkpointing
- **Safety**: Built-in safety monitoring and filtering for real hardware
- **Production-Ready**: Robust error handling, testing, and deployment support

## System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Sources  │    │   Configuration  │    │   Environment   │
│                 │    │                  │    │                 │
│ • AMASS Dataset │    │ • YAML Configs   │    │ • MuJoCo Sim    │
│ • Custom Mocap  │    │ • Hydra/OmegaConf│    │ • Motion Tasks  │
│ • Synthetic     │    │ • Hierarchical   │    │ • Domain Rand   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
         ┌───────────────────────▼───────────────────────┐
         │              Core Framework                    │
         │                                               │
         │ • Configuration Management                    │
         │ • Logging and Monitoring                      │
         │ • Utilities and Common Functions              │
         └───────────────────┬───────────────────────────┘
                             │
    ┌────────────────────────┼────────────────────────┐
    │                       │                        │
┌───▼────┐         ┌────────▼──────┐        ┌────────▼──────┐
│ Models │         │   Training    │        │  Evaluation   │
│        │         │               │        │               │
│ • TF   │◄────────┤ • BC Trainer  │────────┤ • Metrics     │
│ • Conv │         │ • PPO Trainer │        │ • Scenarios   │
│ • MLP  │         │ • Data Loader │        │ • Reports     │
└────────┘         └───────────────┘        └───────────────┘
    │                       │                        │
    └───────────────────────┼────────────────────────┘
                            │
    ┌───────────────────────▼───────────────────────┐
    │              Deployment                       │
    │                                               │
    │ • ONNX Export        • Hardware Adapters     │
    │ • TorchScript        • Safety Filtering      │
    │ • Model Optimization • Real-time Interface   │
    └───────────────────────────────────────────────┘
```

## Module Breakdown

### Core (`src/core/`)

The foundation layer providing essential services:

- **Configuration**: Hierarchical YAML-based configuration with validation
- **Logging**: Unified logging to console, files, TensorBoard, and W&B
- **Utilities**: Common functions for seeding, device management, math operations

### Environments (`src/envs/`)

MuJoCo-based simulation environments:

- **G1Environment**: Base environment with MuJoCo physics and robot model
- **MotionImitationEnv**: Adds reference motion tracking and rewards
- **RobustTrainingEnv**: Extends with domain randomization and perturbations

### Models (`src/models/`)

Neural network architectures:

- **TransformerPolicy**: Sequence-to-sequence policy with attention mechanisms
- **ValueNetwork**: State value estimation for RL algorithms
- **Model Factory**: Centralized model creation and configuration

### Training (`src/training/`)

Training algorithms and data handling:

- **BehaviorCloningTrainer**: Supervised learning from demonstrations
- **PPOTrainer**: Reinforcement learning for robustness fine-tuning
- **MotionDataLoader**: Handles motion capture data preprocessing

### Evaluation (`src/evaluation/`)

Comprehensive model assessment:

- **ModelEvaluator**: Runs evaluation scenarios and computes metrics
- **Metrics**: Specialized metrics for tracking, robustness, efficiency
- **Reporting**: Automated report generation with plots and statistics

### Hardware (`src/hardware/`)

Real-world deployment interfaces:

- **SafetyFilter**: Real-time safety monitoring and command filtering
- **HardwareAdapter**: Abstraction layer for different robot interfaces
- **StateEstimator**: Sensor fusion and state estimation

### Utilities (`src/utils/`)

Supporting tools and utilities:

- **Export**: ONNX and TorchScript model export with optimization
- **Visualization**: Plotting and analysis tools
- **Data Processing**: Motion data conversion and preprocessing

## Data Flow

### Training Pipeline

1. **Data Loading**: Motion capture data loaded and preprocessed into sequences
2. **Environment Setup**: MuJoCo environment configured with robot model and task
3. **Model Creation**: Neural network instantiated from configuration
4. **Training Loop**:
   - Behavior cloning on reference motions
   - Optional RL fine-tuning for robustness
5. **Evaluation**: Periodic assessment on held-out test scenarios
6. **Checkpointing**: Model and training state saved for resumption

### Evaluation Pipeline

1. **Model Loading**: Trained checkpoint loaded and configured
2. **Scenario Setup**: Multiple evaluation scenarios with different conditions
3. **Episode Execution**: Model tested across scenarios with comprehensive logging
4. **Metrics Computation**: Detailed analysis of tracking, robustness, efficiency
5. **Report Generation**: Automated reports with statistics and visualizations

### Deployment Pipeline

1. **Model Export**: Trained model converted to ONNX/TorchScript formats
2. **Optimization**: Model optimized for inference performance
3. **Safety Integration**: Safety filters and monitoring configured
4. **Hardware Interface**: Model deployed with real-time control interface

## Configuration System

The system uses hierarchical YAML configuration with the following structure:

```yaml
experiment:          # Experiment metadata
  name: string
  description: string
  tags: [string]

env:                # Environment configuration
  name: string
  robot:            # Robot-specific settings
    model_path: string
    joints: {...}
  task:             # Task-specific settings
    rewards: {...}
    termination: {...}

model:              # Neural network architecture
  name: string
  hidden_dim: int
  num_layers: int
  sequence_length: int

training:           # Training algorithm settings
  algorithm: string
  total_steps: int
  batch_size: int
  learning_rate: float

evaluation:         # Evaluation configuration
  scenarios: [...]
  metrics: [...]

logging:            # Logging and monitoring
  log_frequency: int
  save_frequency: int
  wandb: {...}
```

## Key Design Decisions

### Transformer-based Policy

- **Why**: Excellent at processing sequential data and learning temporal dependencies
- **Benefits**: Can handle variable-length sequences and complex motion patterns
- **Trade-offs**: More computationally expensive than MLPs

### MuJoCo Simulation

- **Why**: High-fidelity physics simulation with good robotics support
- **Benefits**: Realistic contact dynamics and accurate robot modeling
- **Trade-offs**: Requires careful tuning of simulation parameters

### Two-Phase Training

- **Phase 1**: Behavior cloning for basic motion imitation
- **Phase 2**: RL fine-tuning for robustness and adaptation
- **Why**: Separates learning motion primitives from robustness

### Modular Architecture

- **Benefits**: Easy testing, development, and extension
- **Implementation**: Clean interfaces, dependency injection, factory patterns
- **Trade-offs**: More initial complexity but better long-term maintainability

## Extensibility Points

### Adding New Environments

1. Inherit from `G1Environment`
2. Override observation space, reward computation, and reset logic
3. Register in environment factory
4. Add corresponding configuration schema

### Adding New Models

1. Inherit from `torch.nn.Module`
2. Implement forward pass and loss computation
3. Add to model factory with configuration support
4. Update training loop if needed

### Adding New Training Algorithms

1. Create trainer class with standard interface
2. Implement training loop with logging and checkpointing
3. Add configuration schema
4. Register in trainer factory

### Adding New Evaluation Metrics

1. Inherit from `BaseMetrics`
2. Implement `compute_metrics` method
3. Register in evaluator
4. Update reporting templates

## Dependencies

### Core Dependencies

- **PyTorch**: Deep learning framework
- **MuJoCo**: Physics simulation
- **Gymnasium**: Environment interface standard
- **OmegaConf/Hydra**: Configuration management

### Optional Dependencies

- **Weights & Biases**: Experiment tracking
- **TensorBoard**: Logging and visualization
- **ONNX**: Model export format
- **Matplotlib/Seaborn**: Plotting and analysis

## Performance Considerations

### Memory Usage

- **Sequence Data**: Use efficient data loading with appropriate batch sizes
- **Model Size**: Monitor parameter count and memory usage during training
- **Checkpointing**: Implement incremental saving to manage disk usage

### Computational Efficiency

- **GPU Utilization**: Use mixed precision training when available
- **Simulation Speed**: Optimize environment step times and vectorization
- **Model Inference**: Profile and optimize model forward pass

### Scalability

- **Multi-GPU**: Support distributed training for large models
- **Data Parallelism**: Efficient data loading across multiple workers
- **Hyperparameter Search**: Support for parallel hyperparameter optimization

## Testing Strategy

### Unit Tests

- Core utilities and mathematical functions
- Configuration loading and validation
- Model forward/backward passes

### Integration Tests

- Environment-model interaction
- Training loop execution
- Evaluation pipeline

### System Tests

- End-to-end training on small datasets
- Model export and deployment
- Safety system validation

## Future Extensions

### Planned Features

- Vision-based perception integration
- Multi-task learning across different behaviors
- Online adaptation and learning
- Advanced domain randomization strategies

### Research Directions

- Integration with language models for task specification
- Hierarchical reinforcement learning
- Meta-learning for rapid adaptation
- Advanced contact modeling and control