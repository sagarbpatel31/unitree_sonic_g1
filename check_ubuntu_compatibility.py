#!/usr/bin/env python3
"""
Ubuntu Compatibility Check for Enhanced Sonic G1 Training Pipeline
Verifies system requirements and dependency compatibility.
"""

import sys
import subprocess
import importlib
import platform
from pathlib import Path


def check_python_version():
    """Check Python version compatibility."""
    print("🐍 Python Version Check:")
    version = sys.version_info
    print(f"   Current: {version.major}.{version.minor}.{version.micro}")

    if version.major >= 3 and version.minor >= 8:
        print("   ✅ Python version compatible")
        return True
    else:
        print("   ❌ Python 3.8+ required")
        return False


def check_system_platform():
    """Check operating system compatibility."""
    print("\n💻 System Platform Check:")
    system = platform.system()
    release = platform.release()
    print(f"   OS: {system}")
    print(f"   Release: {release}")

    if system in ["Linux", "Darwin"]:  # Linux (Ubuntu) or macOS
        print("   ✅ Platform compatible")
        return True
    else:
        print("   ⚠️  Windows may require additional setup")
        return True  # Still allow but warn


def check_required_dependencies():
    """Check if required Python packages are available."""
    print("\n📦 Dependency Check:")

    required_packages = [
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("matplotlib", "Matplotlib")
    ]

    optional_packages = [
        ("mujoco", "MuJoCo"),
        ("deeplake", "Deep Lake"),
        ("wandb", "Weights & Biases"),
        ("tensorboard", "TensorBoard")
    ]

    all_good = True

    # Check required packages
    for package, name in required_packages:
        try:
            importlib.import_module(package)
            print(f"   ✅ {name}")
        except ImportError:
            print(f"   ❌ {name} (required)")
            all_good = False

    # Check optional packages
    for package, name in optional_packages:
        try:
            importlib.import_module(package)
            print(f"   ✅ {name}")
        except ImportError:
            print(f"   ⚠️  {name} (optional)")

    return all_good


def check_gpu_availability():
    """Check CUDA/GPU availability."""
    print("\n🔥 GPU Acceleration Check:")

    try:
        import torch
        cuda_available = torch.cuda.is_available()

        if cuda_available:
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
            print(f"   ✅ CUDA available")
            print(f"   🔢 GPU count: {gpu_count}")
            print(f"   🎮 GPU: {gpu_name}")
            return True
        else:
            print("   ⚠️  CUDA not available (CPU training only)")
            return False
    except ImportError:
        print("   ❌ PyTorch not installed")
        return False


def check_file_structure():
    """Check if required files exist."""
    print("\n📁 File Structure Check:")

    required_files = [
        "final_enhanced_training.py",
        "mujoco_demo.py",
        "final_model_evaluation.py",
        "requirements.txt",
        "UBUNTU_DEPLOYMENT.md"
    ]

    optional_files = [
        "data/lightwheel_bevorg_frames.csv",
        "checkpoints/final_model/best_checkpoint.pth"
    ]

    all_good = True

    for file in required_files:
        if Path(file).exists():
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file} (required)")
            all_good = False

    for file in optional_files:
        if Path(file).exists():
            print(f"   ✅ {file}")
        else:
            print(f"   ⚠️  {file} (optional)")

    return all_good


def check_mujoco_compatibility():
    """Check MuJoCo installation and compatibility."""
    print("\n🎮 MuJoCo Compatibility Check:")

    try:
        import mujoco
        version = getattr(mujoco, '__version__', 'Unknown')
        print(f"   ✅ MuJoCo installed (version: {version})")

        # Try to create a simple test
        try:
            # Simple MuJoCo test
            test_xml = """
            <mujoco>
                <worldbody>
                    <body>
                        <geom type="box" size="0.1 0.1 0.1"/>
                    </body>
                </worldbody>
            </mujoco>
            """
            model = mujoco.MjModel.from_xml_string(test_xml)
            data = mujoco.MjData(model)
            print("   ✅ MuJoCo functionality verified")
            return True
        except Exception as e:
            print(f"   ⚠️  MuJoCo test failed: {e}")
            return False

    except ImportError:
        print("   ❌ MuJoCo not installed")
        print("   💡 Install with: pip install mujoco")
        return False


def estimate_system_requirements():
    """Estimate system resource requirements."""
    print("\n💾 System Requirements Estimate:")

    try:
        import psutil

        # Memory check
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        print(f"   RAM: {memory_gb:.1f} GB")

        if memory_gb >= 16:
            print("   ✅ Sufficient RAM for training")
        elif memory_gb >= 8:
            print("   ⚠️  Minimum RAM available (reduce batch size)")
        else:
            print("   ❌ Insufficient RAM (8GB+ recommended)")

        # Disk space
        disk = psutil.disk_usage('/')
        disk_gb_free = disk.free / (1024**3)
        print(f"   Free disk: {disk_gb_free:.1f} GB")

        if disk_gb_free >= 10:
            print("   ✅ Sufficient disk space")
        else:
            print("   ⚠️  Low disk space (10GB+ recommended)")

    except ImportError:
        print("   ⚠️  psutil not installed (install for detailed info)")

    print("\n   💡 Recommended specs for optimal performance:")
    print("     - Ubuntu 20.04+ or compatible Linux")
    print("     - 16GB+ RAM (32GB for large models)")
    print("     - NVIDIA GPU with 8GB+ VRAM")
    print("     - 20GB+ free disk space")


def run_compatibility_check():
    """Run complete compatibility check."""
    print("🔍 Ubuntu Compatibility Check for Enhanced Sonic G1 Training Pipeline")
    print("=" * 70)

    checks = [
        check_python_version(),
        check_system_platform(),
        check_required_dependencies(),
        check_gpu_availability(),
        check_file_structure(),
        check_mujoco_compatibility()
    ]

    estimate_system_requirements()

    print("\n" + "=" * 70)
    print("📋 Compatibility Summary:")

    passed = sum(checks)
    total = len(checks)

    if passed == total:
        print(f"   ✅ All checks passed ({passed}/{total})")
        print("   🚀 System ready for Ubuntu deployment!")
        return True
    elif passed >= 4:  # Core requirements met
        print(f"   ⚠️  {passed}/{total} checks passed")
        print("   🔧 System needs minor setup for optimal use")
        return True
    else:
        print(f"   ❌ Only {passed}/{total} checks passed")
        print("   🛠️  System needs significant setup")
        return False


if __name__ == "__main__":
    success = run_compatibility_check()

    print("\n📖 Next steps:")
    if success:
        print("   1. Review UBUNTU_DEPLOYMENT.md for detailed setup")
        print("   2. Install missing optional dependencies if needed")
        print("   3. Run: python3 final_enhanced_training.py")
        print("   4. Test: python3 mujoco_demo.py")
    else:
        print("   1. Install Python 3.8+")
        print("   2. Install required dependencies: pip install -r requirements.txt")
        print("   3. Install CUDA for GPU acceleration")
        print("   4. Re-run this compatibility check")

    print("\n🎯 For complete Ubuntu setup guide, see: UBUNTU_DEPLOYMENT.md")