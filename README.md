# SONIC G1: Unitree G1 Whole-Body Controller

A comprehensive MuJoCo-based training and deployment framework for the Unitree G1 humanoid robot, supporting motion imitation, behavior cloning, and reinforcement learning approaches.

## 🚀 Features

- **Motion Imitation**: Train policies to imitate human or reference motions
- **Behavior Cloning**: Learn from expert demonstrations
- **PPO + Imitation Learning**: Combine reinforcement learning with imitation
- **Production Deployment**: ONNX/TensorRT export with real-time inference
- **Safety Systems**: Comprehensive safety filters and monitoring
- **Modular Architecture**: Clean separation between training, evaluation, and deployment

## 📁 Repository Structure

```
unitree_sonic_g1/
├── README.md                    # This file
├── pyproject.toml              # Python project configuration
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
├── configs/                    # Configuration files
│   ├── robot/g1.yaml          # Robot-specific settings
│   ├── env/g1_motion_imitation.yaml  # Environment configuration
│   ├── train/                  # Training configurations
│   ├── eval/                   # Evaluation configurations
│   └── data/                   # Data processing configurations
├── scripts/                    # Training and evaluation scripts
│   ├── train_bc.py            # Behavior cloning training
│   ├── train_imitation.py     # PPO + imitation training
│   ├── evaluate_policy.py     # Policy evaluation
│   ├── retarget_to_g1.py      # Motion retargeting
│   ├── export_onnx.py         # Model export
│   ├── benchmark_inference.py # Performance benchmarking
│   └── realtime_loop.py       # Real-time deployment
├── sonic_g1/                   # Main package
│   ├── data/                  # Data processing utilities
│   ├── env/                   # Environment definitions
│   ├── models/                # Neural network architectures
│   ├── train/                 # Training algorithms
│   ├── eval/                  # Evaluation utilities
│   ├── deploy/                # Deployment tools
│   └── utils/                 # Utility functions
├── assets/                     # Robot assets
│   ├── mjcf/                  # MuJoCo XML files
│   └── meshes/               # 3D meshes
└── tests/                     # Unit tests
```

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- MuJoCo 2.3+
- CUDA (optional, for GPU acceleration)

### Quick Install

```bash
# Clone repository
git clone https://github.com/sagarbpatel31/unitree_sonic_g1.git
cd unitree_sonic_g1

# Install in development mode
pip install -e .

# Or install from requirements
pip install -r requirements.txt
```

### Development Setup

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## 🚦 Quick Start

### 1. Data Preparation

Retarget motion data to G1 format:

```bash
python scripts/retarget_to_g1.py --config configs/data/retarget_g1.yaml
```

### 2. Training

**Behavior Cloning:**
```bash
python scripts/train_bc.py --config configs/train/bc.yaml
```

**PPO + Imitation Learning:**
```bash
python scripts/train_imitation.py --config configs/train/ppo_imitation.yaml
```

### 3. Evaluation

```bash
python scripts/evaluate_policy.py \
    --config configs/eval/rollout_eval.yaml \
    evaluation.model.checkpoint_path=checkpoints/best_model.pt
```

### 4. Deployment

**Export to ONNX:**
```bash
python scripts/export_onnx.py \
    --checkpoint checkpoints/best_model.pt \
    --output models/g1_policy.onnx \
    --validate --optimize
```

**Real-time Control:**
```bash
python scripts/realtime_loop.py --config deployment_config.json
```

## 📊 Performance Benchmarking

Benchmark inference performance:

```bash
python scripts/benchmark_inference.py \
    --models models/ \
    --output benchmark_results.json \
    --plot
```

## ⚙️ Configuration

The framework uses Hydra for configuration management. Key configuration files:

- **Robot**: `configs/robot/g1.yaml` - Joint limits, physical properties
- **Environment**: `configs/env/g1_motion_imitation.yaml` - Simulation settings, rewards
- **Training**: `configs/train/` - Algorithm hyperparameters
- **Evaluation**: `configs/eval/` - Evaluation metrics and settings

### Example Configuration Override

```bash
python scripts/train_bc.py \
    train.hyperparameters.learning_rate=1e-4 \
    train.hyperparameters.batch_size=512 \
    train.data.dataset_path=data/custom_demos.pkl
```

## 🔧 Development Status

**✅ Completed:**
- Repository structure and configuration system
- Deployment utilities (ONNX export, inference engine, safety filters)
- Real-time control loop with monitoring
- Performance benchmarking suite

**🚧 In Progress:**
- MuJoCo environment implementation
- Neural network architectures (policy, actor-critic)
- Training algorithms (BC, PPO, imitation learning)
- Data processing and retargeting utilities
- Evaluation and visualization tools

**📋 Planned:**
- Pre-trained model weights
- Comprehensive documentation
- Example notebooks and tutorials
- Hardware integration guides

## 🔐 Safety Features

The deployment system includes comprehensive safety measures:

- **Joint Limits**: Position, velocity, and torque constraints
- **Action Filtering**: Rate limiting and smoothing
- **Watchdog Timers**: Timeout detection and emergency stops
- **Health Monitoring**: System status and performance tracking
- **Emergency Protocols**: Safe shutdown procedures

## 📈 Monitoring and Logging

- Real-time performance metrics
- Training progress tracking with Weights & Biases integration
- Comprehensive logging system
- Automated health checks

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Style

- Follow PEP 8 for Python code
- Use type hints where possible
- Add docstrings for all public functions
- Write tests for new features

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Unitree Robotics for the G1 robot platform
- MuJoCo team for the physics simulator
- SONIC project for inspiration and methodologies

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@software{sonic_g1_2024,
  title={SONIC G1: Unitree G1 Whole-Body Controller},
  author={SONIC G1 Team},
  year={2024},
  url={https://github.com/sagarbpatel31/unitree_sonic_g1}
}
```

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/sagarbpatel31/unitree_sonic_g1/issues)
- **Discussions**: [GitHub Discussions](https://github.com/sagarbpatel31/unitree_sonic_g1/discussions)

---

**Note**: This is an active research project. APIs and interfaces may change as development progresses.