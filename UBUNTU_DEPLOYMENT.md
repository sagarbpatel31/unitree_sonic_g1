# Ubuntu Deployment Guide - Enhanced Sonic G1 Training Pipeline

## 🚀 Quick Start for Ubuntu Systems

This guide helps you deploy the enhanced training pipeline on Ubuntu systems with GPU acceleration and MuJoCo simulation.

### Prerequisites

- **Ubuntu 18.04+** (tested on 20.04, 22.04)
- **NVIDIA GPU** with CUDA support (recommended)
- **Python 3.8+**
- **16GB+ RAM** (32GB+ recommended for large-scale training)
- **10GB+ disk space** for model checkpoints

---

## 📋 Installation Steps

### 1. Clone Repository
```bash
git clone <your-repository-url>
cd unitree_sonic_g1
```

### 2. System Dependencies
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install essential build tools
sudo apt install -y build-essential cmake git wget

# Install Python development tools
sudo apt install -y python3-dev python3-pip python3-venv

# Install graphics libraries (for MuJoCo visualization)
sudo apt install -y libgl1-mesa-dev libgl1-mesa-glx libglew-dev libosmesa6-dev
```

### 3. CUDA Setup (for GPU acceleration)
```bash
# Check if NVIDIA driver is installed
nvidia-smi

# If CUDA not installed, install CUDA toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda-repo-ubuntu2004-12-1-local_12.1.0-530.30.02-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-12-1-local_12.1.0-530.30.02-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-12-1-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
```

### 4. Python Environment Setup
```bash
# Create virtual environment
python3 -m venv sonic_g1_env
source sonic_g1_env/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies
pip install -r requirements.txt

# Install MuJoCo for simulation
pip install mujoco
```

### 5. Verify Installation
```bash
# Test CUDA availability
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python3 -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"

# Test MuJoCo installation
python3 -c "import mujoco; print('MuJoCo installed successfully')"
```

---

## 🎯 Training on Ubuntu

### Quick Training Start
```bash
# Activate environment
source sonic_g1_env/bin/activate

# Run enhanced training with GPU acceleration
python3 final_enhanced_training.py \
    --data data/lightwheel_bevorg_frames.csv \
    --epochs 100 \
    --batch_size 16 \
    --lr 1e-4 \
    --d_model 512 \
    --layers 6 \
    --save_dir checkpoints/ubuntu_training

# Monitor training with nvidia-smi
watch -n 1 nvidia-smi
```

### Advanced Training Options
```bash
# Large-scale training (requires 32GB+ RAM)
python3 final_enhanced_training.py \
    --epochs 200 \
    --batch_size 32 \
    --lr 5e-5 \
    --d_model 1024 \
    --layers 8 \
    --save_dir checkpoints/large_model

# Training with custom data
python3 final_enhanced_training.py \
    --data /path/to/your/data.csv \
    --epochs 50 \
    --batch_size 8 \
    --save_dir checkpoints/custom_training
```

---

## 🎮 MuJoCo Simulation on Ubuntu

### 1. Robot Model Setup
```bash
# Create robot models directory
mkdir -p models/robots

# Place your robot URDF/XML files in models/robots/
# Example: models/robots/sonic_g1.xml
```

### 2. Test MuJoCo Integration
```bash
# Test with dummy simulation
python3 mujoco_demo.py

# Test with actual robot model
python3 mujoco_interface.py --model models/robots/sonic_g1.xml
```

### 3. Real-time Simulation
```python
# Custom simulation script
import mujoco
import mujoco.viewer
from mujoco_demo import MuJoCoPolicyInterface

# Load your trained model
policy = MuJoCoPolicyInterface("checkpoints/ubuntu_training/best_checkpoint.pth")

# Load robot model
model = mujoco.MjModel.from_xml_path("models/robots/sonic_g1.xml")
data = mujoco.MjData(model)

# Create interactive viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    policy.reset()

    for step in range(10000):
        # Get observation from MuJoCo
        obs = policy._get_observation()  # Implement based on your robot

        # Get policy action
        action = policy.get_action(obs)

        # Apply to simulation
        data.ctrl[:] = action
        mujoco.mj_step(model, data)

        # Update viewer
        viewer.sync()
```

---

## ⚡ Performance Optimization

### GPU Memory Optimization
```bash
# Monitor GPU usage during training
watch -n 1 'nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits'

# Reduce batch size if out of memory
python3 final_enhanced_training.py --batch_size 4

# Use gradient checkpointing for memory efficiency
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

### Multi-GPU Training (if available)
```python
# Add to training script for distributed training
import torch.distributed as dist
import torch.multiprocessing as mp

# Launch with:
python3 -m torch.distributed.launch --nproc_per_node=2 final_enhanced_training.py
```

### System Monitoring
```bash
# Monitor system resources
htop

# Monitor GPU usage
nvidia-smi -l 1

# Monitor training logs
tail -f /tmp/sonic_g1_training.log
```

---

## 📊 Model Evaluation on Ubuntu

### Run Comprehensive Evaluation
```bash
# Evaluate trained model
python3 final_model_evaluation.py \
    --checkpoint checkpoints/ubuntu_training/best_checkpoint.pth \
    --data data/lightwheel_bevorg_frames.csv \
    --save_dir evaluation_ubuntu

# View results
ls -la evaluation_ubuntu/
cat evaluation_ubuntu/final_evaluation_results.json
```

### Visualization (requires X11 forwarding for remote)
```bash
# For remote Ubuntu servers
ssh -X username@server

# Install display dependencies
sudo apt install python3-tk

# Generate plots
python3 final_model_evaluation.py --checkpoint checkpoints/ubuntu_training/best_checkpoint.pth
```

---

## 🔧 Troubleshooting

### Common Issues

**CUDA Out of Memory:**
```bash
# Reduce batch size
python3 final_enhanced_training.py --batch_size 4

# Clear GPU cache
python3 -c "import torch; torch.cuda.empty_cache()"
```

**MuJoCo Display Issues:**
```bash
# For headless servers
export MUJOCO_GL=egl

# For X11 issues
export DISPLAY=:0
xhost +local:
```

**Slow Training:**
```bash
# Check GPU utilization
nvidia-smi

# Increase data loading workers
export NUM_WORKERS=4
```

**Memory Issues:**
```bash
# Monitor memory usage
free -h

# Increase swap if needed
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

---

## 🚀 Production Deployment

### Containerized Deployment (Docker)
```dockerfile
FROM nvidia/cuda:12.1-devel-ubuntu20.04

RUN apt update && apt install -y python3 python3-pip git
COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY . /app
WORKDIR /app

CMD ["python3", "final_enhanced_training.py"]
```

### Service Setup (systemd)
```bash
# Create service file
sudo nano /etc/systemd/system/sonic-g1-training.service

[Unit]
Description=Sonic G1 Training Service
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/unitree_sonic_g1
Environment=PATH=/home/ubuntu/sonic_g1_env/bin
ExecStart=/home/ubuntu/sonic_g1_env/bin/python3 final_enhanced_training.py
Restart=always

[Install]
WantedBy=multi-user.target

# Enable and start
sudo systemctl enable sonic-g1-training
sudo systemctl start sonic-g1-training
```

---

## 📈 Expected Performance on Ubuntu

### Training Performance (NVIDIA RTX 4090)
- **OptimizedTransformerPolicy (3.2M params):** ~30 mins for 100 epochs
- **Large Model (10M+ params):** ~2-3 hours for 100 epochs
- **Memory usage:** 8-12GB GPU, 16-32GB RAM

### Simulation Performance
- **MuJoCo real-time:** 60+ FPS with policy inference
- **Policy inference:** <1ms per action
- **Full episode:** 1000 steps in 16-20 seconds

---

## ✅ Success Verification

Your Ubuntu deployment is successful when:

1. **Training runs without errors** and achieves convergence (MSE < 0.4)
2. **GPU utilization** shows >80% during training
3. **MuJoCo demo** runs smoothly with real-time visualization
4. **Model evaluation** completes with expected metrics
5. **Checkpoints** are properly saved and loadable

---

**🎉 Your enhanced Sonic G1 training pipeline is now ready for production use on Ubuntu!**

For additional support, check the evaluation results in `final_evaluation_results/` and training logs in `checkpoints/`.