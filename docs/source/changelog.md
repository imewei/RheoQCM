# Changelog

All notable changes to RheoQCM are documented here.

## [Unreleased]

### Added
- Comprehensive Sphinx documentation with theory, tutorials, and API reference
- ReadTheDocs configuration for automated documentation builds
- Bayesian fitting tutorial with NumPyro MCMC examples
- Batch analysis tutorial with GPU acceleration guidance

### Changed
- Reorganized documentation structure into theory, tutorials, and user-guide sections

## [2.0.0] - 2025-12-28

### Added
- **JAX Backend**: Complete migration from NumPy/SciPy to JAX for GPU acceleration
- **Performance Optimizations**: 22-28x speedup for core analysis functions
  - `thin_film_guess`: 509.7ms → 17.9ms (28.5x speedup)
  - `solve_properties`: 925.6ms → 41.3ms (22.4x speedup)
- **batch_analyze_vmap()**: GPU-accelerated batch processing with `jax.vmap`
- **Optimistix Integration**: Replaced deprecated JAXopt with Optimistix for least-squares
- **Custom Calculation Types**: Extensible `calctype` system for custom physics models
- **Three-Layer Architecture**: Clean separation of physics, model, and analysis layers
- **SolveResult/BatchResult Dataclasses**: Type-safe return values

### Changed
- **Python 3.12+ Required**: Updated minimum Python version
- **JAX Configuration**: Must call `configure_jax()` before analysis
- **API Changes**:
  - `solve_properties()` returns `SolveResult` dataclass
  - `batch_analyze()` returns `BatchResult` dataclass
  - Use `.grho_refh`, `.phi`, `.drho` attributes instead of dict keys

### Fixed
- PyQt6 compatibility issues (scoped enums, exec methods)
- Linux matplotlib marker rendering with RGBA colors
- Bare except clauses replaced with specific exception types
- Mutable default argument patterns (`def f(x=[])`)
- Numerical stability with `arctan2` and `clamp_phi`

### Deprecated
- `QCMFuncs` module: Use `rheoQCM.core` instead (emits FutureWarning)

## [1.x] - Legacy

### Features
- QCM-D data acquisition via N2PK VNA
- Single-layer viscoelastic analysis (SLA)
- Multilayer film calculations
- Excel import/export
- HDF5 data storage
- Real-time plotting during acquisition

### Known Limitations
- CPU-only computation
- Sequential processing for batch data
- Limited uncertainty quantification

---

## Migration Guide

### From v1.x to v2.0

#### 1. Update Imports

```python
# Old
from QCMFuncs import QCM_functions as qcm

# New
from rheoQCM.core import QCMModel, configure_jax
```

#### 2. Configure JAX

```python
# Add at the start of your script
configure_jax()
```

#### 3. Update API Calls

```python
# Old
result = model.solve(...)
grho = result['grho']

# New
result = model.solve_properties(...)
grho = result.grho_refh  # Use attribute access
```

#### 4. Batch Processing

```python
# Old (sequential loop)
results = []
for data in measurements:
    results.append(model.solve(data))

# New (vectorized)
from rheoQCM.core.analysis import batch_analyze
results = batch_analyze(measurements, harmonics=[3,5,7], nhcalc='353', f1=5e6, refh=3)
```

See {doc}`tutorials/scripting-basics` for complete examples.

## Version History

| Version | Date | Python | JAX |
|---------|------|--------|-----|
| 2.0.0 | 2025-12-28 | 3.12+ | 0.8.0+ |
| 1.x | 2024 | 3.8+ | N/A |
