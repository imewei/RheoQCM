"""
RheoQCM Core Package

This package provides the JAX-accelerated computational core for QCM-D
data analysis and rheological modeling.

Architecture:
    - Layer 1 (physics.py): Pure-JAX stateless physics module
    - Layer 2 (model.py): Unified logic class with state management
    - Layer 3 (analysis.py): Scripting interface for analysis workflows

All physics functions are designed to be jit-able and vmap-able for
GPU acceleration and batch processing.
"""

from rheoQCM.core.jax_config import configure_jax
from rheoQCM.core.jax_config import get_jax_backend
from rheoQCM.core.jax_config import is_gpu_available

# Physics module exports - Layer 1
from rheoQCM.core.physics import (
    # Constants
    Zq,
    f1_default,
    e26,
    d26,
    g0_default,
    epsq,
    eps0,
    dq,
    C0byA,
    electrode_default,
    water_default,
    air_default,
    dlam_refh_range,
    drho_range,
    grho_refh_range,
    phi_range,
    # Sauerbrey equations
    sauerbreyf,
    sauerbreym,
    # Complex modulus calculations
    grho,
    grhostar,
    grhostar_from_refh,
    grho_from_dlam,
    calc_dlam,
    calc_lamrho,
    calc_deltarho,
    etarho,
    zstar_bulk,
    # SLA equations
    calc_delfstar_sla,
    calc_D,
    normdelfstar,
    bulk_props,
    deltarho_bulk,
    # Kotula model
    kotula_gstar,
    kotula_xi,
    # Utility functions
    find_peaks,
    interp_linear,
    interp_cubic,
    create_interp_func,
    savgol_filter,
)

# Model module exports - Layer 2
from rheoQCM.core.model import (
    QCMModel,
    bulk_drho,
)

# Analysis module exports - Layer 3
from rheoQCM.core.analysis import (
    QCMAnalyzer,
    analyze_delfstar,
    batch_analyze,
)

__all__ = [
    # JAX configuration
    "configure_jax",
    "get_jax_backend",
    "is_gpu_available",
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
    # Model class - Layer 2
    "QCMModel",
    "bulk_drho",
    # Analysis module - Layer 3
    "QCMAnalyzer",
    "analyze_delfstar",
    "batch_analyze",
]
