:orphan:

# Getting Started

## Installation

RheoQCM targets Python 3.12+. Linux, macOS, and Windows are supported.

```bash
git clone https://github.com/imewei/RheoQCM.git
cd RheoQCM
pip install -e .
```

### Optional GPU Acceleration (Linux + NVIDIA)

If you have CUDA installed, use the Makefile helpers to install the matching
JAX build:

```bash
make install-jax-gpu
# or
make install-jax-gpu-cuda13
make install-jax-gpu-cuda12
```

## Launching the GUI

From the project root:

```bash
python -m rheoQCM.main
```

## Data Import

To import QCM-D data, export from your instrument as `.xlsx`, then use the GUI:

1. `File > Import QCM-D data`
2. Select the base frequency in Settings before import
3. Save or export results when finished
