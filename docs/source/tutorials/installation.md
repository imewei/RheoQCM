# Installation Guide

This guide covers installing RheoQCM on different platforms and configuring
GPU acceleration for optimal performance.

## Requirements

### Minimum Requirements

- Python 3.12 or later
- 4 GB RAM (8 GB recommended)
- 500 MB disk space

### For GUI Usage

- Display with 1024×768 resolution or higher
- PyQt6-compatible graphics driver

### For Data Acquisition

- Windows operating system
- 32-bit Python (for myVNA compatibility)
- myVNA software installed

## Installation Methods

### From PyPI (Recommended)

The simplest way to install RheoQCM:

```bash
pip install rheoQCM
```

This installs the core package with all required dependencies.

### From Source (Development)

For the latest features or contributing to development:

```bash
git clone https://github.com/imewei/RheoQCM.git
cd RheoQCM
pip install -e .
```

### Using uv (Fast)

If you use [uv](https://github.com/astral-sh/uv) for package management:

```bash
uv pip install rheoQCM
```

## Optional Dependencies

### Fallback Computation

If JAX is unavailable on your system, install the fallback option:

```bash
pip install "rheoQCM[fallback]"
```

This uses mpmath for arbitrary-precision computation (slower but portable).

### Documentation Building

To build the documentation locally:

```bash
pip install "rheoQCM[docs]"
```

### Full Development Environment

For all development tools:

```bash
pip install "rheoQCM[dev]"
```

## GPU Acceleration

GPU acceleration provides 20-100x speedup for large datasets. This requires:

- NVIDIA GPU with CUDA Compute Capability ≥ 5.2
- Linux operating system
- CUDA Toolkit 12.x or 13.x installed

### Check Prerequisites

```bash
# Verify CUDA installation
nvcc --version
# Should show: release 12.x or 13.x

# Check GPU capability
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
# Should show GPU name and compute capability (e.g., "8.9")
```

### Install GPU Support

#### Using Makefile (Recommended)

```bash
cd RheoQCM
make install-jax-gpu  # Auto-detects CUDA version
```

Or explicitly choose your CUDA version:

```bash
make install-jax-gpu-cuda13  # For CUDA 13.x
make install-jax-gpu-cuda12  # For CUDA 12.x
```

#### Manual Installation

```bash
# Uninstall existing JAX
pip uninstall -y jax jaxlib

# For CUDA 13.x (Turing and newer GPUs)
pip install "jax[cuda13-local]"

# OR for CUDA 12.x (Maxwell and newer GPUs)
pip install "jax[cuda12-local]"
```

### Verify GPU Detection

```bash
python -c "import jax; print('Backend:', jax.default_backend())"
# Should print: Backend: gpu
```

If it prints `cpu`, see the troubleshooting section below.

### GPU Compatibility Table

| GPU Generation | Examples | Compute Cap | CUDA 13 | CUDA 12 |
|----------------|----------|-------------|---------|---------|
| Ada Lovelace | RTX 40xx | 8.9 | Yes | Yes |
| Ampere | RTX 30xx, A100 | 8.x | Yes | Yes |
| Turing | RTX 20xx, T4 | 7.5 | Yes | Yes |
| Volta | V100 | 7.0 | No | Yes |
| Pascal | GTX 10xx | 6.x | No | Yes |
| Maxwell | GTX 9xx | 5.2 | No | Yes |

## Verifying Installation

### Basic Check

```python
import rheoQCM
print(f"RheoQCM version: {rheoQCM.__version__}")
```

### Core Functionality

```python
from rheoQCM.core import QCMModel, configure_jax

configure_jax()  # Enable float64 precision

# Create a test model
model = QCMModel(f1=5e6, refh=3)
model.load_delfstars({3: -1000+100j, 5: -1700+180j})
result = model.solve_properties(nh=[3, 5, 3])

print(f"Test successful!")
print(f"drho = {result.drho:.3e} kg/m²")
```

### GUI Launch

```bash
python -m rheoQCM.main
```

The GUI window should appear within a few seconds.

## Troubleshooting

### JAX Installation Issues

**Problem**: `ImportError: No module named 'jax'`

**Solution**: Install JAX explicitly:
```bash
pip install jax jaxlib
```

**Problem**: GPU not detected (shows `cpu` backend)

**Solutions**:
1. Verify CUDA is in PATH:
   ```bash
   export PATH=/usr/local/cuda/bin:$PATH
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
   ```

2. Reinstall JAX with correct CUDA version:
   ```bash
   pip uninstall -y jax jaxlib
   pip install "jax[cuda12-local]"  # Match your CUDA version
   ```

### PyQt6 Issues

**Problem**: `qt.qpa.plugin: Could not load the Qt platform plugin`

**Solution** (Linux):
```bash
sudo apt install libxcb-xinerama0 libxcb-cursor0
```

**Problem**: High-DPI scaling issues

**Solution**: Set environment variable:
```bash
export QT_AUTO_SCREEN_SCALE_FACTOR=1
python -m rheoQCM.main
```

### Memory Issues

**Problem**: Out of memory on large datasets

**Solutions**:
1. Process data in smaller batches
2. Use `batch_analyze_vmap()` with smaller chunk sizes
3. Increase swap space

## Platform-Specific Notes

### Windows

- Use Windows Terminal or PowerShell for best experience
- Data acquisition requires 32-bit Python and myVNA software
- GPU acceleration not supported natively (use WSL2)

### macOS

- Apple Silicon (M1/M2/M3) uses CPU backend (no NVIDIA GPU)
- Metal acceleration not yet supported by JAX
- GUI requires XQuartz for some features

### Linux

- Full GPU acceleration support
- Recommended for production/batch processing
- Use conda environment for isolation if needed

## Next Steps

After installation:

1. {doc}`quickstart` - Run your first analysis
2. {doc}`gui-workflow` - Learn the GUI interface
3. {doc}`scripting-basics` - Write analysis scripts
