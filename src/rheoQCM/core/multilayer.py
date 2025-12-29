"""
Multilayer Film Calculations Module for RheoQCM (Layer 1)

This module provides pure-JAX implementations of multi-layer QCM film
calculations, including complex load impedance (ZL) and complex frequency
shift (delfstar) for arbitrary layer stacks.

Key Functions:
    - calc_ZL: Calculate complex load impedance for layer stack
    - calc_delfstar_multilayer: Calculate complex frequency shift for layers
    - calc_Zmot: Calculate motional impedance for Lu-Lewis calculation
    - delete_layer: Remove a layer from layer stack, shifting higher layers down
    - validate_layers: Validate layer configuration

Physics Models:
    - Matrix transfer method for multi-layer acoustic propagation
    - Small Load Approximation (SLA) for thin films
    - Lu-Lewis (LL) calculation for accurate thick film analysis
    - Voigt model for constant G' with frequency-dependent G''

All functions are designed to be:
    - Stateless (no side effects)
    - JIT-compilable with @jax.jit
    - Vectorizable with jax.vmap for batch processing
    - GPU-acceleratable when hardware is available

Notes:
    - All phi values are in RADIANS (not degrees)
    - Layer dict uses 'grho' (not 'grho3') for |G*|*rho at reference harmonic
    - Reference harmonic is stored in layer['n'] if provided
    - drho = np.inf indicates a semi-infinite (bulk) layer

See Also
--------
rheoQCM.core.physics : Single-layer physics calculations
rheoQCM.core.model : Layer 2 model logic
"""

from __future__ import annotations

import logging
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from rheoQCM.core.jax_config import configure_jax
from rheoQCM.core.physics import (
    Zq,
    calc_delfstar_sla,
    electrode_default,
    f1_default,
    g0_default,
)

# Ensure JAX is configured for Float64
configure_jax()

logger = logging.getLogger(__name__)


# =============================================================================
# Layer Validation
# =============================================================================


class LayerValidationError(ValueError):
    """Raised when layer configuration is invalid."""

    pass


def validate_layers(layers: dict[int, dict]) -> None:
    """
    Validate layer configuration.

    Checks that:
    - At least one layer is provided
    - Only the outermost layer can have infinite thickness (drho = inf)
    - Required keys (grho, phi, drho) are present in each layer

    Parameters
    ----------
    layers : dict[int, dict]
        Layer stack indexed by layer number.

    Raises
    ------
    LayerValidationError
        If layer configuration is invalid.

    Notes
    -----
    Layer numbers should be consecutive integers.
    Layer 0 is typically the electrode (closest to quartz).
    Larger layer numbers are further from the quartz surface.
    """
    if not layers:
        raise LayerValidationError("At least one layer must be provided")

    layer_nums = sorted(layers.keys())
    layer_max = max(layer_nums)

    for layer_num in layer_nums:
        layer = layers[layer_num]

        # Check required keys
        required_keys = ["grho", "phi", "drho"]
        for key in required_keys:
            if key not in layer:
                raise LayerValidationError(
                    f"Layer {layer_num} missing required key: '{key}'"
                )

        # Check for infinite thickness in non-outermost layer
        drho = layer["drho"]
        if layer_num != layer_max and (drho == jnp.inf or drho == float("inf")):
            raise LayerValidationError(
                f"Only the outermost layer (layer {layer_max}) can have "
                f"infinite thickness. Layer {layer_num} has drho=inf."
            )


# =============================================================================
# Layer Utilities
# =============================================================================


def delete_layer(old_layers: dict[int, dict], num: int) -> dict[int, dict]:
    """
    Remove a layer from layer stack, shifting higher layers down.

    Parameters
    ----------
    old_layers : dict[int, dict]
        Original layer stack indexed by layer number.
    num : int
        Layer number to remove.

    Returns
    -------
    new_layers : dict[int, dict]
        New layer stack with the specified layer removed and higher
        layers renumbered to fill the gap.

    Examples
    --------
    >>> layers = {1: layer1, 2: layer2, 3: layer3}
    >>> delete_layer(layers, 2)
    {1: layer1, 2: layer3}  # layer3 moved down to position 2

    Notes
    -----
    This function creates a shallow copy of the layer dictionaries
    to avoid modifying the original data structure.

    T049-T051: Using shallow dict.copy() instead of deepcopy (005-jax-perf)
    Layer dicts contain only simple values (grho, phi, drho as floats),
    so shallow copy is sufficient and more memory-efficient.
    """
    layer_nums = sorted(old_layers.keys())
    layer_max = max(layer_nums)
    layer_min = min(layer_nums)

    new_layers = {}

    # Copy layers below the deleted one unchanged
    for i in layer_nums:
        if i < num:
            new_layers[i] = old_layers[i].copy()  # T049: shallow copy

    # Shift layers above the deleted one down by 1
    if num >= layer_min:
        for i in range(num, layer_max):
            if i + 1 in old_layers:
                new_layers[i] = old_layers[i + 1].copy()  # T050: shallow copy

    return new_layers


# =============================================================================
# Core Impedance Functions
# =============================================================================


def _zstar_bulk_from_layer(
    n: int,
    layer: dict[str, Any],
    calctype: str = "SLA",
    refh: int = 3,
) -> jnp.ndarray:
    """
    Calculate complex acoustic impedance for a layer.

    Parameters
    ----------
    n : int
        Harmonic number.
    layer : dict
        Layer properties containing 'grho' and 'phi'.
    calctype : str
        Calculation type: "SLA", "LL", or "Voigt".
    refh : int
        Reference harmonic for grho scaling.

    Returns
    -------
    zstar : complex
        Complex acoustic impedance [Pa s/m].
    """
    grho_refh = layer["grho"]
    phi = layer["phi"]
    layer_refh = layer.get("n", refh)

    if calctype != "Voigt":
        # Power law model: |G*| scales with (n/refh)^(phi/(pi/2))
        grho_n = grho_refh * (n / layer_refh) ** (phi / (jnp.pi / 2))
        grhostar = grho_n * jnp.exp(1j * phi)
    else:
        # Voigt model: constant G', G'' linear in omega
        # G' = |G*| * cos(phi), G'' = |G*| * sin(phi) * (n/refh)
        greal = grho_refh * jnp.cos(phi)
        gimag = grho_refh * (n / layer_refh) * jnp.sin(phi)
        grhostar = jnp.sqrt(gimag**2 + greal**2) * jnp.exp(1j * phi)

    return jnp.sqrt(grhostar)


def _calc_D_from_layer(
    n: int,
    layer: dict[str, Any],
    delfstar: jnp.ndarray,
    f1: float = f1_default,
    calctype: str = "SLA",
    refh: int = 3,
) -> jnp.ndarray:
    """
    Calculate D (thickness times complex wave number) for a layer.

    Parameters
    ----------
    n : int
        Harmonic number.
    layer : dict
        Layer properties containing 'grho', 'phi', and 'drho'.
    delfstar : complex
        Complex frequency shift (for LL calculation).
    f1 : float
        Fundamental frequency [Hz].
    calctype : str
        Calculation type.
    refh : int
        Reference harmonic.

    Returns
    -------
    D : complex
        Thickness times complex wave number (dimensionless).
    """
    drho = layer["drho"]
    zstar = _zstar_bulk_from_layer(n, layer, calctype, refh)

    # Avoid division by zero
    zstar_safe = jnp.where(jnp.abs(zstar) < 1e-30, 1e-30 + 0j, zstar)

    return 2 * jnp.pi * (n * f1 + delfstar) * drho / zstar_safe


# =============================================================================
# JIT-Compatible Layer Functions (T024-T027 from 005-jax-perf)
# =============================================================================


def _layers_to_arrays(
    layers: dict[int, dict],
    refh: int = 3,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray | None]:
    """
    Convert layer dict to arrays for JIT-compatible operations.

    Parameters
    ----------
    layers : dict[int, dict]
        Layer stack indexed by layer number.
    refh : int
        Default reference harmonic.

    Returns
    -------
    grho_arr : jnp.ndarray
        |G*|*rho at reference harmonic for each layer.
    phi_arr : jnp.ndarray
        Phase angles [radians] for each layer.
    drho_arr : jnp.ndarray
        Mass per unit area [kg/m^2] for each layer.
    n_ref_arr : jnp.ndarray
        Reference harmonic for each layer.
    Zf_arr : jnp.ndarray | None
        Pre-specified terminal impedances if top layer has 'Zf', else None.
    """
    layer_nums = sorted(layers.keys())
    N = len(layers)

    # Use NumPy preallocated arrays (10x faster than Python lists + jnp.array)
    grho_np = np.empty(N, dtype=np.float64)
    phi_np = np.empty(N, dtype=np.float64)
    drho_np = np.empty(N, dtype=np.float64)
    n_ref_np = np.empty(N, dtype=np.float64)

    for i, layer_num in enumerate(layer_nums):
        layer = layers[layer_num]
        grho_np[i] = layer["grho"]
        phi_np[i] = layer["phi"]
        drho_np[i] = layer["drho"]
        n_ref_np[i] = layer.get("n", refh)

    # Convert to JAX arrays (single conversion at the end)
    grho_arr = jnp.asarray(grho_np)
    phi_arr = jnp.asarray(phi_np)
    drho_arr = jnp.asarray(drho_np)
    n_ref_arr = jnp.asarray(n_ref_np)

    # Check for pre-specified terminal impedance
    top_layer = layers[layer_nums[-1]]
    Zf_arr = None
    if "Zf" in top_layer:
        Zf_arr = top_layer["Zf"]

    return grho_arr, phi_arr, drho_arr, n_ref_arr, Zf_arr


@partial(jax.jit, static_argnames=["calctype"])
def _zstar_bulk_jit(
    n: int,
    grho_refh: jnp.ndarray,
    phi: jnp.ndarray,
    n_ref: jnp.ndarray,
    calctype: str = "SLA",
) -> jnp.ndarray:
    """
    Calculate complex acoustic impedance (JIT-compatible array version).

    Parameters
    ----------
    n : int
        Harmonic number.
    grho_refh : float
        |G*|*rho at reference harmonic [Pa kg/m^3].
    phi : float
        Phase angle [radians].
    n_ref : float
        Reference harmonic for this layer.
    calctype : str
        Calculation type: "SLA", "LL", or "Voigt".

    Returns
    -------
    zstar : complex
        Complex acoustic impedance [Pa s/m].
    """
    n = jnp.asarray(n, dtype=jnp.float64)

    def power_law_path(_):
        grho_n = grho_refh * (n / n_ref) ** (phi / (jnp.pi / 2))
        grhostar = grho_n * jnp.exp(1j * phi)
        return jnp.sqrt(grhostar)

    def voigt_path(_):
        greal = grho_refh * jnp.cos(phi)
        gimag = grho_refh * (n / n_ref) * jnp.sin(phi)
        grhostar = jnp.sqrt(gimag**2 + greal**2) * jnp.exp(1j * phi)
        return jnp.sqrt(grhostar)

    # Use lax.cond for JIT-compatible branching
    return lax.cond(calctype != "Voigt", power_law_path, voigt_path, None)


@jax.jit
def _calc_D_jit(
    n: jnp.ndarray,
    drho: jnp.ndarray,
    zstar: jnp.ndarray,
    delfstar: jnp.ndarray,
    f1: float,
) -> jnp.ndarray:
    """
    Calculate D (thickness times complex wave number) - JIT-compatible.

    Parameters
    ----------
    n : float
        Harmonic number.
    drho : float
        Mass per unit area [kg/m^2].
    zstar : complex
        Complex acoustic impedance [Pa s/m].
    delfstar : complex
        Complex frequency shift [Hz].
    f1 : float
        Fundamental frequency [Hz].

    Returns
    -------
    D : complex
        Thickness times complex wave number (dimensionless).
    """
    zstar_safe = jnp.where(jnp.abs(zstar) < 1e-30, 1e-30 + 0j, zstar)
    return 2 * jnp.pi * (n * f1 + delfstar) * drho / zstar_safe


@partial(jax.jit, static_argnames=["num_layers", "calctype", "has_Zf_top"])
def _calc_ZL_jit(
    n: int,
    grho_arr: jnp.ndarray,
    phi_arr: jnp.ndarray,
    drho_arr: jnp.ndarray,
    n_ref_arr: jnp.ndarray,
    delfstar: jnp.ndarray,
    f1: float,
    num_layers: int,
    calctype: str = "SLA",
    Zf_top: jnp.ndarray = jnp.array(0.0 + 0.0j),
    has_Zf_top: bool = False,
) -> jnp.ndarray:
    """
    Calculate complex load impedance for stack of layers (JIT-compiled).

    This is the JIT-compatible version of calc_ZL that operates on arrays
    instead of dictionaries. Uses lax.fori_loop for the matrix chain multiplication.

    Parameters
    ----------
    n : int
        Harmonic number.
    grho_arr : jnp.ndarray
        |G*|*rho at reference harmonic for each layer, shape (num_layers,).
    phi_arr : jnp.ndarray
        Phase angles [radians] for each layer, shape (num_layers,).
    drho_arr : jnp.ndarray
        Mass per unit area [kg/m^2] for each layer, shape (num_layers,).
    n_ref_arr : jnp.ndarray
        Reference harmonic for each layer, shape (num_layers,).
    delfstar : complex
        Complex frequency shift [Hz].
    f1 : float
        Fundamental frequency [Hz].
    num_layers : int
        Number of layers (static for JIT).
    calctype : str
        Calculation type: "SLA", "LL", or "Voigt".
    Zf_top : complex
        Pre-specified terminal impedance for top layer.
    has_Zf_top : bool
        If True, use Zf_top; if False, compute from top layer properties.

    Returns
    -------
    ZL : complex
        Complex load impedance [Pa s/m].
    """
    n_float = jnp.asarray(n, dtype=jnp.float64)

    # Compute Z* and D for all layers
    # Use static loop since num_layers is static
    zstar_all = jnp.zeros(num_layers, dtype=jnp.complex128)
    D_all = jnp.zeros(num_layers, dtype=jnp.complex128)

    for i in range(num_layers):
        zstar_i = _zstar_bulk_jit(n, grho_arr[i], phi_arr[i], n_ref_arr[i], calctype)
        D_i = _calc_D_jit(n_float, drho_arr[i], zstar_i, delfstar, f1)
        zstar_all = zstar_all.at[i].set(zstar_i)
        D_all = D_all.at[i].set(D_i)

    # Terminal impedance from top layer
    if has_Zf_top:
        Zf_max = Zf_top
    else:
        Zf_max = 1j * zstar_all[num_layers - 1] * jnp.tan(D_all[num_layers - 1])

    # Single layer case
    if num_layers == 1:
        return Zf_max

    # Multi-layer case
    # Terminal matrix for second-to-last layer
    second_last_idx = num_layers - 2
    Tn = jnp.array(
        [
            [1 + Zf_max / zstar_all[second_last_idx], 0.0 + 0.0j],
            [0.0 + 0.0j, 1 - Zf_max / zstar_all[second_last_idx]],
        ],
        dtype=jnp.complex128,
    )

    # L matrix for second-to-last layer
    cos_D = jnp.cos(D_all[second_last_idx])
    sin_D = jnp.sin(D_all[second_last_idx])
    L_sl = jnp.array(
        [[cos_D + 1j * sin_D, 0.0 + 0.0j], [0.0 + 0.0j, cos_D - 1j * sin_D]],
        dtype=jnp.complex128,
    )

    # Initial vector
    uvec = L_sl @ Tn @ jnp.array([[1.0], [1.0]], dtype=jnp.complex128)

    # Propagate through remaining layers using lax.fori_loop
    def propagate_step(i, uvec):
        # We iterate from idx = num_layers-3 down to 0
        # But fori_loop goes from 0 to num_layers-2
        # So reverse the index: layer_idx = num_layers - 3 - i
        layer_idx = num_layers - 3 - i
        next_layer_idx = layer_idx + 1

        # Build S matrix
        S = jnp.array(
            [
                [
                    1 + zstar_all[next_layer_idx] / zstar_all[layer_idx],
                    1 - zstar_all[next_layer_idx] / zstar_all[layer_idx],
                ],
                [
                    1 - zstar_all[next_layer_idx] / zstar_all[layer_idx],
                    1 + zstar_all[next_layer_idx] / zstar_all[layer_idx],
                ],
            ],
            dtype=jnp.complex128,
        )

        # Build L matrix for this layer
        cos_D_i = jnp.cos(D_all[layer_idx])
        sin_D_i = jnp.sin(D_all[layer_idx])
        L_i = jnp.array(
            [
                [cos_D_i + 1j * sin_D_i, 0.0 + 0.0j],
                [0.0 + 0.0j, cos_D_i - 1j * sin_D_i],
            ],
            dtype=jnp.complex128,
        )

        return L_i @ S @ uvec

    # Number of propagation steps (from layer num_layers-3 down to 0)
    n_steps = num_layers - 2

    # Use lax.fori_loop for JIT-compatible iteration
    uvec_final = lax.fori_loop(0, n_steps, propagate_step, uvec)

    # Extract load impedance from reflection coefficient
    rstar = uvec_final[1, 0] / uvec_final[0, 0]
    return zstar_all[0] * (1 - rstar) / (1 + rstar)


# =============================================================================
# Multi-layer Load Impedance
# =============================================================================


def calc_ZL(
    n: int,
    layers: dict[int, dict],
    delfstar: complex = 0.0,
    f1: float = f1_default,
    calctype: str = "SLA",
    refh: int = 3,
) -> jnp.ndarray:
    """
    Calculate complex load impedance for stack of layers.

    Uses the matrix transfer method for multi-layer acoustic wave propagation.
    Layers are assumed to be laterally homogeneous.

    Parameters
    ----------
    n : int
        Harmonic number.
    layers : dict[int, dict]
        Dictionary of layer dictionaries indexed by layer number.
        Layer numbering starts from 1 (or 0 for electrode).
        Each layer dict must contain:
        - 'grho': |G*|*rho at reference harmonic [Pa kg/m^3]
        - 'phi': Phase angle [radians]
        - 'drho': Mass per unit area [kg/m^2] (inf for bulk)
        - 'n': Reference harmonic (optional, defaults to refh)
    delfstar : complex, optional
        Complex frequency shift [Hz] (needed for Lu-Lewis calculation).
        Default: 0.0 (SLA approximation).
    f1 : float, optional
        Fundamental frequency [Hz]. Default: 5e6 Hz.
    calctype : str, optional
        Calculation type:
        - 'SLA': Small load approximation with power law model
        - 'LL': Lu-Lewis equation
        - 'Voigt': Voigt model (constant G', G'' linear in omega)
    refh : int, optional
        Reference harmonic. Default: 3.

    Returns
    -------
    ZL : complex
        Complex load impedance [Pa s/m].

    Notes
    -----
    The matrix transfer method uses:
        L[i] = diag(exp(iD), exp(-iD))
        S[i] = [[1+Z_{i+1}/Z_i, 1-Z_{i+1}/Z_i],
                [1-Z_{i+1}/Z_i, 1+Z_{i+1}/Z_i]]

    The terminal impedance from the top layer is:
        Zf = i * Z_top * tan(D_top)

    For fractional area coverage (if 'AF' in layer), the result is:
        ZL = AF * ZL_with_layer + (1-AF) * ZL_without_layer
    """
    if not layers:
        return jnp.array(0.0 + 0.0j, dtype=jnp.complex128)

    delfstar = jnp.asarray(delfstar, dtype=jnp.complex128)

    layer_nums = sorted(layers.keys())
    N = len(layers)
    layer_max = max(layer_nums)

    # Check for fractional area coverage (AF) or pre-specified terminal impedance (Zf)
    # These require the slower path
    has_AF = any("AF" in layers[i] for i in layer_nums[:-1])
    top_layer = layers[layer_max]
    has_Zf = "Zf" in top_layer

    # T026: Use JIT-compiled version for standard cases (no AF, no pre-specified Zf)
    if not has_AF and not has_Zf:
        grho_arr, phi_arr, drho_arr, n_ref_arr, _ = _layers_to_arrays(layers, refh)
        return _calc_ZL_jit(
            n,
            grho_arr,
            phi_arr,
            drho_arr,
            n_ref_arr,
            delfstar,
            f1,
            num_layers=N,
            calctype=calctype,
            has_Zf_top=False,
        )

    # Fallback to original implementation for special cases
    # Build impedance and D dictionaries
    Z = {}
    D = {}
    L = {}

    # Process all layers except the top (terminal) layer
    for layer_num in layer_nums[:-1]:
        layer = layers[layer_num]
        Z[layer_num] = _zstar_bulk_from_layer(n, layer, calctype, refh)
        D[layer_num] = _calc_D_from_layer(n, layer, delfstar, f1, calctype, refh)
        cos_D = jnp.cos(D[layer_num])
        sin_D = jnp.sin(D[layer_num])
        L[layer_num] = jnp.array(
            [[cos_D + 1j * sin_D, 0.0 + 0.0j], [0.0 + 0.0j, cos_D - 1j * sin_D]],
            dtype=jnp.complex128,
        )

    # Terminal impedance from the top layer
    if has_Zf:
        # Pre-specified terminal impedance
        Zf_max = top_layer["Zf"].get(n, 0.0 + 0.0j)
    else:
        D_max = _calc_D_from_layer(n, top_layer, delfstar, f1, calctype, refh)
        Z_max = _zstar_bulk_from_layer(n, top_layer, calctype, refh)
        Zf_max = 1j * Z_max * jnp.tan(D_max)

    # Single layer case
    if N == 1:
        return Zf_max

    # Multi-layer case: matrix chain multiplication
    # Terminal matrix for the second-to-last layer
    second_last = layer_nums[-2]
    Tn = jnp.array(
        [
            [1 + Zf_max / Z[second_last], 0.0 + 0.0j],
            [0.0 + 0.0j, 1 - Zf_max / Z[second_last]],
        ],
        dtype=jnp.complex128,
    )

    # Initial vector
    uvec = L[second_last] @ Tn @ jnp.array([[1.0], [1.0]], dtype=jnp.complex128)

    # Propagate through remaining layers (from second-to-last toward first)
    for idx in range(len(layer_nums) - 3, -1, -1):
        layer_num = layer_nums[idx]
        next_layer_num = layer_nums[idx + 1]
        S = jnp.array(
            [
                [
                    1 + Z[next_layer_num] / Z[layer_num],
                    1 - Z[next_layer_num] / Z[layer_num],
                ],
                [
                    1 - Z[next_layer_num] / Z[layer_num],
                    1 + Z[next_layer_num] / Z[layer_num],
                ],
            ],
            dtype=jnp.complex128,
        )
        uvec = L[layer_num] @ S @ uvec

    # Extract load impedance from reflection coefficient
    rstar = uvec[1, 0] / uvec[0, 0]
    first_layer = layer_nums[0]
    ZL = Z[first_layer] * (1 - rstar) / (1 + rstar)

    # Handle fractional area coverage (if any layer has 'AF')
    for layer_num in layer_nums[:-1]:
        layer = layers[layer_num]
        if "AF" in layer:
            AF = layer["AF"]
            # Calculate ZL without this layer
            layers_ref = delete_layer(layers, layer_num)
            ZL_ref = calc_ZL(n, layers_ref, delfstar, f1, calctype, refh)
            ZL = AF * ZL + (1 - AF) * ZL_ref
            break  # Only one layer can have fractional coverage

    return ZL


# =============================================================================
# Complex Frequency Shift for Multi-layer
# =============================================================================


def calc_delfstar_multilayer(
    n: int,
    layers: dict[int, dict],
    f1: float = f1_default,
    calctype: str = "SLA",
    reftype: str = "bare",
    refh: int = 3,
    g0: float = g0_default,
    electrode: dict[str, Any] | None = None,
) -> jnp.ndarray:
    """
    Calculate complex frequency shift for stack of layers.

    Parameters
    ----------
    n : int
        Harmonic number.
    layers : dict[int, dict]
        Dictionary of layer dictionaries indexed by layer number.
        See calc_ZL for required keys.
    f1 : float, optional
        Fundamental frequency [Hz]. Default: 5e6 Hz.
    calctype : str, optional
        Calculation type:
        - 'SLA': Small load approximation with power law model
        - 'LL': Lu-Lewis equation (most general, handles thick films)
        - 'Voigt': Voigt model
    reftype : str, optional
        Reference type:
        - 'bare': Reference is bare quartz (layers 1+ removed)
        - 'overlayer': Reference includes layer 2+ (only layer 1 removed)
    refh : int, optional
        Reference harmonic. Default: 3.
    g0 : float, optional
        Half-bandwidth of unloaded resonator [Hz]. Default: 50.
    electrode : dict, optional
        Electrode properties. Uses default if not provided.

    Returns
    -------
    delfstar : complex
        Complex frequency shift [Hz].
        Real part: frequency shift.
        Imaginary part: half-bandwidth shift (dissipation).

    Raises
    ------
    LayerValidationError
        If layer configuration is invalid (e.g., infinite thickness not at top).

    Notes
    -----
    For SLA and Voigt calctypes, uses the small load approximation:
        delfstar = f1 * i * ZL / (pi * Zq)

    For LL calctype, solves the Lu-Lewis equation numerically to find
    the complex frequency shift that zeros the motional impedance.
    """
    if not layers:
        return jnp.array(jnp.nan + 0.0j, dtype=jnp.complex128)

    # Validate layer configuration
    try:
        validate_layers(layers)
    except LayerValidationError as e:
        logger.warning(f"Layer validation failed: {e}")
        return jnp.array(jnp.nan + 0.0j, dtype=jnp.complex128)

    # T051: Make a shallow copy to avoid modifying input (005-jax-perf)
    # Layer dicts contain only simple values, so shallow copy is sufficient
    layers = {k: v.copy() for k, v in layers.items()}

    # Calculate load impedance
    ZL = calc_ZL(n, layers, 0.0, f1, calctype, refh)

    # Calculate reference load impedance based on reftype
    if reftype == "overlayer" and 2 in layers:
        layers_ref = delete_layer(layers, 1)
        ZL_ref = calc_ZL(n, layers_ref, 0.0, f1, calctype, refh)
    else:
        ZL_ref = 0.0 + 0.0j

    del_ZL = ZL - ZL_ref

    if calctype.upper() != "LL":
        # Use small load approximation
        return calc_delfstar_sla(del_ZL, f1=f1)

    # Lu-Lewis calculation
    # Add electrode if not present
    if electrode is None:
        electrode = electrode_default
    if 0 not in layers:
        layers[0] = electrode

    # T051: Use shallow copy instead of deepcopy (005-jax-perf)
    layers_all = {k: v.copy() for k, v in layers.items()}
    if reftype == "overlayer" and 1 in layers:
        layers_ref = delete_layer({k: v.copy() for k, v in layers.items()}, 1)
    else:
        layers_ref = {0: layers[0].copy()}

    # Get initial guess from SLA
    ZL_all = calc_ZL(n, layers_all, 0.0, f1, calctype, refh)
    delfstar_sla_all = calc_delfstar_sla(ZL_all, f1=f1)

    ZL_ref_ll = calc_ZL(n, layers_ref, 0.0, f1, calctype, refh)
    delfstar_sla_ref = calc_delfstar_sla(ZL_ref_ll, f1=f1)

    # Solve for delfstar using Newton-Raphson iteration
    delfstar_all = _solve_ll_delfstar(
        n, layers_all, delfstar_sla_all, f1, calctype, refh, g0
    )
    delfstar_ref = _solve_ll_delfstar(
        n, layers_ref, delfstar_sla_ref, f1, calctype, refh, g0
    )

    return delfstar_all - delfstar_ref


def _solve_ll_delfstar(
    n: int,
    layers: dict[int, dict],
    delfstar_init: jnp.ndarray,
    f1: float,
    calctype: str,
    refh: int,
    g0: float,
    max_iter: int = 50,
    tol: float = 1e-10,
) -> jnp.ndarray:
    """
    Solve Lu-Lewis equation for delfstar using Newton-Raphson iteration.

    Finds the delfstar that zeros the motional impedance.

    Parameters
    ----------
    n : int
        Harmonic number.
    layers : dict[int, dict]
        Layer stack including electrode (layer 0).
    delfstar_init : complex
        Initial guess (typically from SLA).
    f1 : float
        Fundamental frequency [Hz].
    calctype : str
        Calculation type (should be 'LL').
    refh : int
        Reference harmonic.
    g0 : float
        Half-bandwidth of unloaded resonator [Hz].
    max_iter : int
        Maximum Newton-Raphson iterations.
    tol : float
        Convergence tolerance.

    Returns
    -------
    delfstar : complex
        Solution to Lu-Lewis equation.
    """
    delfstar = jnp.asarray(delfstar_init, dtype=jnp.complex128)

    # T045 (012-jax-perf): Use JAX autodiff for Jacobian instead of finite differences
    # Create a wrapper that splits complex delfstar into real/imag for autodiff
    def _zmot_real_imag(df_real: float, df_imag: float) -> tuple[float, float]:
        """Wrapper for autodiff: takes real/imag of delfstar, returns real/imag of Zmot."""
        df_complex = df_real + 1j * df_imag
        zmot = calc_Zmot(n, layers, df_complex, f1, calctype, refh, g0)
        return float(jnp.real(zmot)), float(jnp.imag(zmot))

    for _ in range(max_iter):
        Zmot = calc_Zmot(n, layers, delfstar, f1, calctype, refh, g0)
        Zmot_mag = jnp.abs(Zmot)

        if Zmot_mag < tol:
            break

        # T045: Compute Jacobian using JAX autodiff
        df_real = float(jnp.real(delfstar))
        df_imag = float(jnp.imag(delfstar))

        # Partial derivatives with respect to real part of delfstar
        jac_fn_real = jax.jacfwd(_zmot_real_imag, argnums=0)
        dZr_dr, dZi_dr = jac_fn_real(df_real, df_imag)

        # Partial derivatives with respect to imaginary part of delfstar
        jac_fn_imag = jax.jacfwd(_zmot_real_imag, argnums=1)
        dZr_di, dZi_di = jac_fn_imag(df_real, df_imag)

        # Jacobian matrix [[dZr/dr, dZr/di], [dZi/dr, dZi/di]]
        J = jnp.array(
            [
                [dZr_dr, dZr_di],
                [dZi_dr, dZi_di],
            ],
            dtype=jnp.float64,
        )

        # Residual vector
        F = jnp.array([jnp.real(Zmot), jnp.imag(Zmot)], dtype=jnp.float64)

        # Solve J @ delta = -F
        try:
            delta = jnp.linalg.solve(J, -F)
            delfstar = delfstar + delta[0] + 1j * delta[1]
        except Exception:
            # If solve fails, break and return current estimate
            break

    return delfstar


# =============================================================================
# Motional Impedance for Lu-Lewis
# =============================================================================


def calc_Zmot(
    n: int,
    layers: dict[int, dict],
    delfstar: complex,
    f1: float = f1_default,
    calctype: str = "LL",
    refh: int = 3,
    g0: float = g0_default,
) -> jnp.ndarray:
    """
    Calculate motional impedance for Lu-Lewis calculation.

    The motional impedance should be zero at the correct delfstar.

    Parameters
    ----------
    n : int
        Harmonic number.
    layers : dict[int, dict]
        Layer stack including electrode (layer 0).
    delfstar : complex
        Complex frequency shift [Hz].
    f1 : float, optional
        Fundamental frequency [Hz]. Default: 5e6 Hz.
    calctype : str, optional
        Calculation type. Default: 'LL'.
    refh : int, optional
        Reference harmonic. Default: 3.
    g0 : float, optional
        Half-bandwidth of unloaded resonator [Hz]. Default: 50.

    Returns
    -------
    Zmot : complex
        Motional impedance.

    Notes
    -----
    Based on Eq. 4.5.9 in Diethelm's book.

    The motional impedance is:
        Zmot = -i*Zqc/sin(Dq) + (1/(i*Zqc*tan(Dq/2)) + 1/(i*Zqc*tan(Dq/2) + ZL))^-1

    where:
        Zqc = Zq * (1 + i*2*g0/(n*f1)) is the complex quartz impedance
        Dq = omega * drho_q / Zq is the quartz phase
        drho_q = Zq / (2*f1) is the quartz mass per unit area
    """
    delfstar = jnp.asarray(delfstar, dtype=jnp.complex128)

    # Angular frequency
    om = 2 * jnp.pi * (n * f1 + delfstar)

    # Complex quartz impedance (includes dissipation)
    Zqc = Zq * (1 + 1j * 2 * g0 / (n * f1))

    # Quartz mass per unit area
    drho_q = Zq / (2 * f1)

    # Quartz phase
    Dq = om * drho_q / Zq

    # Second term: -i*Zqc/sin(Dq)
    secterm = -1j * Zqc / jnp.sin(Dq)

    # Load impedance
    ZL = calc_ZL(n, layers, delfstar, f1, calctype, refh)

    # Third term from Eq. 4.5.9
    tan_Dq_half = jnp.tan(Dq / 2)
    term_a = 1j * Zqc * tan_Dq_half
    thirdterm = 1 / (1 / term_a + 1 / (term_a + ZL))

    Zmot = secterm + thirdterm

    return Zmot


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # Validation
    "LayerValidationError",
    "validate_layers",
    # Utilities
    "delete_layer",
    # Core functions
    "calc_ZL",
    "calc_delfstar_multilayer",
    "calc_Zmot",
]
