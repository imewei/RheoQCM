# Changelog

All notable changes to RheoQCM are documented here.

For the complete changelog with all historical upstream versions, see
[CHANGELOG.md](https://github.com/imewei/RheoQCM/blob/master/CHANGELOG.md).

## [0.1.1] - 2026-03-17

### Added
- `make verify` and `make verify-fast` pre-push verification targets
- Comprehensive Sphinx documentation with theory, tutorials, and API reference
- Structured logging throughout the codebase
- `ThemeManager` service for system-aware light/dark UI theming
- Professional UI component library and input validation utilities
- `MCMCDiagnostics` structured diagnostic class

### Changed
- Consolidated all optional dependencies into core `dependencies` in pyproject.toml
- Bumped minimum Python to 3.13; minimum NLSQ to >=0.6.8
- Replaced deprecated `jaxopt` with `optimistix` for JAX-native least-squares
- Renamed "About QCMpy" menu item to "About rheoQCM"
- Refined GUI spacing, fonts, and sizing for data-dense layout

### Fixed
- Wired unconnected `Exit` and `Help Manual` menu actions
- Fixed `.fromat()` typo, IEEE 754 NaN comparison bug, `dict.pop` loop-variable bug
- PyQt6 compatibility (scoped enums, exec methods)
- Numerical stability with `arctan2` and `clamp_phi`
- Resolved mypy type errors and CI fixes for GUI test marker and Windows uvloop
- Aligned harmonic dimensions and fixed pandas indexing in I/O layer

### Removed
- VNA/hardware acquisition code (analysis-only tool)
- `QCMFuncs` legacy module
- Dead menubar items: `Maximum Harmonic`, `Open openQCM`, `Delete Selected`

## [0.1.0] - 2025-12-28

### Added
- **JAX Backend**: Complete migration from NumPy/SciPy to JAX for GPU acceleration
- **Performance Optimizations**: 22-28x speedup for core analysis functions
  - `thin_film_guess`: 509.7ms -> 17.9ms (28.5x speedup)
  - `solve_properties`: 925.6ms -> 41.3ms (22.4x speedup)
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
- Bare except clauses replaced with specific exception types
- Mutable default argument patterns (`def f(x=[])`)

### Removed
- `DataSaver` module: Replaced by `rheoQCM.io.data_store.DataStore`

---

## Migration Guide

### From legacy to 0.1.x

#### 1. Update Imports

```python
# Old (removed)
# from QCMFuncs import QCM_functions as qcm

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
| 0.1.1 | 2026-03-17 | 3.13+ | 0.8.0+ |
| 0.1.0 | 2025-12-28 | 3.12+ | 0.8.0+ |
| 0.21.0 | 2024-01-11 | 3.8+ | N/A |
