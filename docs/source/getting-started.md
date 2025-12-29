:orphan:

# Getting Started

## Installation

RheoQCM targets Python 3.12+. For analysis-only workflows, Linux/macOS/Windows
are supported. Hardware acquisition via myVNA is Windows-only.

```bash
git clone https://github.com/imewei/RheoQCM.git
cd RheoQCM
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

### Fallback Mode

If JAX is unavailable, install the fallback dependency:

```bash
pip install "rheoQCM[fallback]"
```

## Launching the GUI

From the project root:

```bash
python -m rheoQCM.main
```

On Windows, you can also use the bundled `rheoQCM.bat` in `src/rheoQCM/` after
updating the Python path.

## Data Import

To import QCM-D data, export from your instrument as `.xlsx`, then use the GUI:

1. `File > Import QCM-D data`
2. Select the base frequency in Settings before import
3. Save or export results when finished
