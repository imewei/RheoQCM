[![DOI](https://zenodo.org/badge/138771761.svg)](https://zenodo.org/badge/latestdoi/138771761)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# RheoQCM - QCM Data Collection and Analysis Software

RheoQCM is a Python package for QCM-D (Quartz Crystal Microbalance with Dissipation) data acquisition and rheological analysis. It features a modern JAX-powered computational core for high-performance modeling and a PyQt6-based GUI for data collection and visualization.

![](Screenshot.png)
<p align="center">
<b>Screenshot of the User Interface</b>
</p>

## Features

- **High-performance modeling** with JAX (GPU-accelerated when available)
- **QCM data collection and analysis** in one integrated package
- **Import and analyze** external QCM-D datasets (.xlsx, .mat, .h5)
- **Multilayer thin-film analysis** using the Small Load Approximation (SLA)
- **Bayesian parameter estimation** with MCMC (NumPyro backend)
- **Uncertainty quantification** via autodiff-based covariance propagation
- **Cross-platform** analysis (Linux/macOS/Windows), data collection on Windows

## Requirements

- Python 3.12+
- JAX 0.8.0+ with jaxlib 0.8.0+
- PyQt6 (for GUI)
- NumPy 2.0+

For data collection with network analyzers:
- Windows (32-bit Python)
- myVNA software and AccessMyVNA

## Installation

### From PyPI (Recommended)

```bash
pip install rheoQCM
```

### From Source

```bash
git clone https://github.com/imewei/RheoQCM.git
cd RheoQCM
pip install -e .
```

### GPU Acceleration (Linux + NVIDIA)

For 20-100x speedup on large datasets:

```bash
# Auto-detect system CUDA version and install matching JAX
make install-jax-gpu

# Or explicitly choose CUDA version:
make install-jax-gpu-cuda13  # CUDA 13.x + SM >= 7.5
make install-jax-gpu-cuda12  # CUDA 12.x + SM >= 5.2
```

**Prerequisites:**
- NVIDIA GPU (Maxwell or newer, SM >= 5.2)
- System CUDA 12.x or 13.x installed (`nvcc` in PATH)

<details>
<summary>GPU Compatibility Table</summary>

| GPU Generation | Example GPUs | SM Version | CUDA 13 | CUDA 12 |
|----------------|--------------|------------|---------|---------|
| Ada Lovelace | RTX 40xx, L40 | 8.9 | Yes | Yes |
| Ampere | RTX 30xx, A100 | 8.x | Yes | Yes |
| Turing | RTX 20xx, T4 | 7.5 | Yes | Yes |
| Volta | V100, Titan V | 7.0 | No | Yes |
| Pascal | GTX 10xx, P100 | 6.x | No | Yes |
| Maxwell | GTX 9xx | 5.x | No | Yes |

</details>

### Fallback Mode

If JAX is unavailable, install the fallback dependency:

```bash
pip install "rheoQCM[fallback]"
```

## Quick Start

### Using the GUI

```bash
python -m rheoQCM.main
```

### Scripting with the Core API

```python
from rheoQCM.core import QCMModel, configure_jax

configure_jax()  # Enable float64 precision

# Create model and load experimental data
model = QCMModel(f1=5e6, refh=3)
model.load_delfstars({3: -1000+100j, 5: -1700+180j, 7: -2500+280j})

# Solve for mechanical properties
result = model.solve_properties(nh=[3, 5, 3])
print(f"drho = {result.drho:.3e} kg/m^2")
print(f"phi = {result.phi:.4f} rad")
print(f"|G*|rho = {result.grho_refh:.3e} Pa·kg/m^3")
```

### Batch Processing with GPU Acceleration

```python
import jax.numpy as jnp
from rheoQCM.core import batch_analyze_vmap

# Process many data points in parallel
delfstars = jnp.array([
    [-1000+100j, -1700+180j, -2500+280j],
    [-1100+110j, -1800+190j, -2600+290j],
    # ... thousands more rows
])
results = batch_analyze_vmap(delfstars, harmonics=[3, 5, 7], f1=5e6, refh=3)
```

## Importing QCM-D Data

1. Export from instrument as `.xlsx` with columns: `t(s), delf1, delg1, delf3, delg3, ...`
2. In GUI: `File > Import QCM-D data`
3. Set base frequency in `Settings > Hardware > Crystal > Base Frequency`
4. Analyze and export results

## Documentation

- [Full Documentation](https://imewei.github.io/RheoQCM/) - API reference, tutorials, and guides
- [Migration Guide](docs/source/migration.md) - Upgrading from QCMFuncs to rheoQCM.core
- [QCM Notes](docs/source/references/qcm-notes.md) - Background theory and references

## Project Structure

```
src/
  rheoQCM/           # Main package
    core/            # JAX-accelerated computational core
      physics.py     # Layer 1: Pure physics functions
      multilayer.py  # Layer 1: Multilayer calculations
      model.py       # Layer 2: State management and solvers
      analysis.py    # Layer 3: Analysis interface
      bayesian.py    # Bayesian MCMC fitting
      uncertainty.py # Uncertainty propagation
    gui/             # PyQt6 GUI components
    io/              # I/O handlers (HDF5, Excel, JSON)
    modules/         # Application modules
    services/        # Service layer
  QCMFuncs/          # Legacy API (deprecated)
tests/               # Test suite
docs/                # Sphinx documentation
```

## Citation

Please cite as:

```bibtex
@software{rheoqcm,
  author = {Wang, Qifeng and Yang, Megan and Shull, Kenneth R.},
  title = {RheoQCM: QCM Data Collection and Analysis Software},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/imewei/RheoQCM},
  doi = {10.5281/zenodo.2486039}
}
```

## Authors

- **Qifeng Wang** - Primary developer
- **Megan Yang** - Developer
- **Kenneth R. Shull** - Project PI

## Related Projects

- [MATLAB QCM Data Acquisition](https://github.com/Shull-Research-Group/QCM_Data_Acquisition_Program) - Original MATLAB version
- [QCM-D Analysis GUI](https://github.com/sadmankazi/QCM-D-Analysis-GUI) - MATLAB analysis by Kazi Sadman

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Josh Yeh - Original MATLAB implementation
- Diethelm Johannsmann - QCM theory
- Lauren Sturdy, Ivan Makarov - Technical contributions
