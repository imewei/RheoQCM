"""
RheoQCM Core Package

This package provides the JAX-accelerated computational core for QCM-D
data analysis and rheological modeling.

Architecture
============

The core package is organized into three layers:

**Layer 1 - Physics Functions (physics.py, multilayer.py)**:
    Pure-JAX stateless functions for physics calculations.
    All functions are designed to be jit-able and vmap-able for
    GPU acceleration and batch processing.

**Layer 2 - Model Logic (model.py)**:
    Unified logic class (QCMModel) with state management, data loading,
    and solver orchestration. Provides a high-level interface to Layer 1.

**Layer 3 - Analysis Interface (analysis.py)**:
    Scripting interface for analysis workflows. Used by UI layers
    (QCM.py, QCM_functions.py) and for standalone processing.


Extending the Core (User Story 4)
=================================

The architecture supports extending physics models without modifying
core code. There are two main extension points:

1. Adding New Physics Functions
-------------------------------
New physics functions can be added to physics.py and will automatically
be available to all layers. Follow these patterns for compatibility:

    - Use @jax.jit decorator for JIT compilation
    - Use jax.numpy (jnp) instead of numpy for operations
    - Avoid Python control flow (if/else on array values)
    - Return jnp arrays with consistent dtypes (float64, complex128)
    - Add to __all__ for public export

Example of a compatible physics function::

    @jax.jit
    def my_custom_modulus(n: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
        '''Calculate custom modulus at harmonic n.

        Parameters
        ----------
        n : array
            Harmonic number
        params : array
            Model parameters

        Returns
        -------
        G_star : complex array
            Complex modulus
        '''
        omega = 2 * jnp.pi * n * 5e6
        # Pure JAX operations only
        return params[0] * jnp.exp(1j * params[1])

2. Adding Custom Calctypes
--------------------------
The QCMModel class supports custom calculation types (calctypes) for
fitting experimental data with different physical models.

To register a custom calctype::

    from rheoQCM.core.model import QCMModel

    def my_residual(params, delfstar_exp, harmonics, f1, refh, Zq):
        '''Custom residual function for fitting.'''
        grho, phi, drho = params[0], params[1], params[2]
        # Calculate predicted delfstar using custom physics
        # Return array of residuals
        return jnp.array([r1, r2, r3])

    model = QCMModel(f1=5e6, refh=3)
    model.register_calctype("MyModel", my_residual)
    result = model.solve_properties(nh=[3, 5, 3], calctype="MyModel")

For global registration (available to all QCMModel instances)::

    from rheoQCM.core.model import register_global_calctype
    register_global_calctype("MyModel", my_residual)

Key Requirements for Custom Residual Functions
----------------------------------------------
- Use pure JAX operations for jit/vmap compatibility
- Accept params as jnp.ndarray (typically [grho_refh, phi, drho])
- Accept delfstar_exp as dict[int, complex]
- Return jnp.ndarray of residuals (same length as data points)
- Avoid Python control flow (use jnp.where instead of if/else)


Usage Examples
==============

Basic single-point analysis::

    from rheoQCM.core import QCMModel, configure_jax

    configure_jax()  # Enable Float64 precision

    model = QCMModel(f1=5e6, refh=3)
    model.load_delfstars({3: -1000+100j, 5: -1700+180j})
    result = model.solve_properties(nh=[3, 5, 3])
    print(f"drho = {result.drho:.3e} kg/m^2")

Batch analysis with GPU acceleration::

    from rheoQCM.core import batch_analyze_vmap

    results = batch_analyze_vmap(
        batch_delfstars,
        harmonics=[3, 5, 3],
        f1=5e6,
        refh=3,
    )

Direct physics calculations::

    from rheoQCM.core import sauerbreyf, grho, grhostar

    # Calculate Sauerbrey frequency shift
    delf = sauerbreyf(3, drho=1e-6)

    # Calculate complex modulus at harmonic 5
    grho_5 = grho(5, grho_refh=1e8, phi=0.5, refh=3)
    gstar_5 = grhostar(grho_5, phi=0.5)


See Also
--------
specs/004-unify-qcm-modules/quickstart.md : Quick start guide with examples
rheoQCM.core.physics : Layer 1 physics module documentation
rheoQCM.core.model : Layer 2 model class documentation
"""

# Analysis module exports - Layer 3
from rheoQCM.core.analysis import (
    QCMAnalyzer,
    analyze_delfstar,
    batch_analyze,
    batch_analyze_vmap,  # T044: Export vmap-enabled batch processing
)

# Bayesian fitting exports
from rheoQCM.core.bayesian import (
    BayesianFitResult,
    BayesianFitter,
    DiagnosticPlot,
    PriorSpec,
    plot_comparison,
)
from rheoQCM.core.jax_config import (
    check_gpu_availability,
    configure_jax,
    get_jax_backend,
    is_gpu_available,
)

# Model module exports - Layer 2
from rheoQCM.core.model import (
    BatchResult,
    CalctypeResidualFn,  # T048: Type alias for custom residual functions
    QCMModel,
    SolveResult,
    bulk_drho,
    get_global_calctypes,  # T048: List globally registered calctypes
    register_global_calctype,  # T048: Global calctype registration
)

# Multilayer module exports - Layer 1
from rheoQCM.core.multilayer import (
    # Validation
    LayerValidationError,
    calc_delfstar_multilayer,
    # Core functions
    calc_ZL,
    calc_Zmot,
    # Utilities
    delete_layer,
    validate_layers,
)

# Physics module exports - Layer 1
from rheoQCM.core.physics import (
    C0byA,
    # Constants
    Zq,
    air_default,
    bulk_props,
    calc_D,
    # SLA equations
    calc_delfstar_sla,
    calc_deltarho,
    calc_dlam,
    calc_lamrho,
    create_interp_func,
    d26,
    deltarho_bulk,
    dlam_refh_range,
    dq,
    drho_range,
    e26,
    electrode_default,
    eps0,
    epsq,
    etarho,
    f1_default,
    # Utility functions
    find_peaks,
    g0_default,
    # Complex modulus calculations
    grho,
    grho_from_dlam,
    grho_refh_range,
    grhostar,
    grhostar_from_refh,
    interp_cubic,
    interp_linear,
    # Kotula model
    kotula_gstar,
    kotula_xi,
    normdelfstar,
    phi_range,
    # Sauerbrey equations
    sauerbreyf,
    sauerbreym,
    savgol_filter,
    water_default,
    zstar_bulk,
)

# Signal processing module exports (JAX-based, scipy-compatible)
from rheoQCM.core.signal import (
    find_peaks as signal_find_peaks,
)
from rheoQCM.core.signal import (
    peak_prominences,
    peak_widths,
)

# Uncertainty calculation exports
from rheoQCM.core.uncertainty import (
    UncertaintyBand,
    UncertaintyCalculator,
    check_degrees_of_freedom,
    regularize_covariance,
)

__all__ = [
    # JAX configuration
    "configure_jax",
    "get_jax_backend",
    "is_gpu_available",
    "check_gpu_availability",
    # Signal processing (scipy-compatible JAX implementations)
    "signal_find_peaks",
    "peak_prominences",
    "peak_widths",
    # Constants
    "Zq",
    "f1_default",
    "e26",
    "d26",
    "g0_default",
    "epsq",
    "eps0",
    "dq",
    "C0byA",
    "electrode_default",
    "water_default",
    "air_default",
    "dlam_refh_range",
    "drho_range",
    "grho_refh_range",
    "phi_range",
    # Sauerbrey equations
    "sauerbreyf",
    "sauerbreym",
    # Complex modulus calculations
    "grho",
    "grhostar",
    "grhostar_from_refh",
    "grho_from_dlam",
    "calc_dlam",
    "calc_lamrho",
    "calc_deltarho",
    "etarho",
    "zstar_bulk",
    # SLA equations
    "calc_delfstar_sla",
    "calc_D",
    "normdelfstar",
    "bulk_props",
    "deltarho_bulk",
    # Kotula model
    "kotula_gstar",
    "kotula_xi",
    # Utility functions
    "find_peaks",
    "interp_linear",
    "interp_cubic",
    "create_interp_func",
    "savgol_filter",
    # Multilayer module - Layer 1
    "LayerValidationError",
    "validate_layers",
    "delete_layer",
    "calc_ZL",
    "calc_delfstar_multilayer",
    "calc_Zmot",
    # Model class - Layer 2
    "QCMModel",
    "SolveResult",
    "BatchResult",
    "CalctypeResidualFn",  # T048: Type for custom residual functions
    "register_global_calctype",  # T048: Global calctype registration
    "get_global_calctypes",  # T048: List globally registered calctypes
    "bulk_drho",
    # Analysis module - Layer 3
    "QCMAnalyzer",
    "analyze_delfstar",
    "batch_analyze",
    "batch_analyze_vmap",  # T044: vmap-enabled batch processing for GPU acceleration
    # Uncertainty calculation
    "UncertaintyBand",
    "UncertaintyCalculator",
    "regularize_covariance",
    "check_degrees_of_freedom",
    # Bayesian fitting
    "BayesianFitResult",
    "BayesianFitter",
    "DiagnosticPlot",
    "PriorSpec",
    "plot_comparison",
]
