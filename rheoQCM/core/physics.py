"""
Physics Core Module for RheoQCM (Layer 1)

This module contains pure-JAX implementations of all QCM physics equations.
All functions are designed to be:
    - Stateless (no side effects)
    - JIT-compilable with @jax.jit
    - Vectorizable with jax.vmap for batch processing
    - GPU-acceleratable when hardware is available

Physics models included:
    - Sauerbrey equations for mass loading
    - Small Load Approximation (SLA) equations
    - Kotula model for viscoelastic films
    - Complex modulus calculations (G*, G', G'')
    - Utility functions replacing scipy (find_peaks, interpolation, filtering)

Constants:
    - Zq: Acoustic impedance of quartz (8.84e6 Pa.s/m)
    - f1_default: Default fundamental frequency (5e6 Hz)
    - Electrode, water, and air default properties

NaN/Inf Handling (T018):
    - check_finite: Check if array contains finite values
    - propagate_nan: Propagate NaN through calculation conditionally
    - safe_divide: Division with protection against inf/nan

Note: All functions use jax.numpy exclusively. No scipy or numpy imports.

See Also
--------
rheoQCM.core.model : Layer 2 model logic
rheoQCM.core.jax_config : JAX configuration
"""

from functools import partial
from typing import Tuple
from typing import Union

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.signal import convolve

import interpax

# =============================================================================
# Physical Constants
# =============================================================================

# Acoustic impedance of AT-cut quartz [kg m^-2 s^-1 = Pa.s/m]
Zq: float = 8.84e6

# Default fundamental resonant frequency [Hz]
f1_default: float = 5e6

# Piezoelectric stress coefficient for AT-cut quartz [C/m^2]
e26: float = 9.65e-2

# Piezoelectric strain coefficient for AT-cut quartz [m/V]
d26: float = 3.1e-9

# Default half bandwidth (HWHM) of unloaded resonator [Hz]
g0_default: float = 50.0

# Quartz permittivity and thickness for piezoelectric stiffening
epsq: float = 4.54
eps0: float = 8.8e-12
dq: float = 330e-6  # Quartz thickness [m]
C0byA: float = epsq * eps0 / dq  # Capacitance per unit area

# Default electrode properties
# grho in Pa kg/m^3 (|G*| * density)
# phi in radians
# drho in kg/m^2 (mass per unit area)
electrode_default: dict = {
    "grho": 3.0e17,
    "phi": 0.0,
    "drho": 2.8e-6,
    "n": 3,  # reference harmonic
}

# Default water properties (bulk semi-infinite)
water_default: dict = {
    "grho": 1e8,  # ~Pa kg/m^3 at room temperature
    "phi": jnp.pi / 2,  # 90 degrees (viscous)
    "drho": jnp.inf,  # semi-infinite
    "n": 3,
}

# Default air properties (negligible)
air_default: dict = {
    "grho": 0.0,
    "phi": jnp.pi / 2,
    "drho": jnp.inf,
    "n": 1,
}

# Variable limits for fitting
dlam_refh_range: Tuple[float, float] = (0.0, 10.0)
drho_range: Tuple[float, float] = (0.0, 3e-2)  # kg/m^2
grho_refh_range: Tuple[float, float] = (1e4, 1e14)  # Pa kg/m^3
phi_range: Tuple[float, float] = (0.0, jnp.pi / 2)  # radians


# =============================================================================
# NaN/Inf Handling Functions (T018)
# =============================================================================


class PhysicsNaNError(ValueError):
    """Raised when NaN is encountered in physics calculations and propagation is disabled."""

    pass


class PhysicsInfError(ValueError):
    """Raised when Inf is encountered in physics calculations and propagation is disabled."""

    pass


@jax.jit
def check_finite(x: jnp.ndarray) -> jnp.ndarray:
    """
    Check if all values in array are finite (not NaN or Inf).

    Parameters
    ----------
    x : array
        Input array to check.

    Returns
    -------
    is_finite : bool
        True if all values are finite.
    """
    return jnp.all(jnp.isfinite(x))


@jax.jit
def propagate_nan(x: jnp.ndarray, condition: jnp.ndarray) -> jnp.ndarray:
    """
    Propagate NaN through calculation based on condition.

    Parameters
    ----------
    x : array
        Input array.
    condition : bool or array
        If True, replace x with NaN.

    Returns
    -------
    result : array
        x if condition is False, NaN otherwise.
    """
    return jnp.where(condition, jnp.nan, x)


@jax.jit
def propagate_nan_complex(x: jnp.ndarray, condition: jnp.ndarray) -> jnp.ndarray:
    """
    Propagate complex NaN through calculation based on condition.

    Parameters
    ----------
    x : complex array
        Input array.
    condition : bool or array
        If True, replace x with complex NaN.

    Returns
    -------
    result : complex array
        x if condition is False, NaN+1j*NaN otherwise.
    """
    nan_complex = jnp.nan + 1j * jnp.nan
    return jnp.where(condition, nan_complex, x)


@jax.jit
def safe_divide(
    numerator: jnp.ndarray,
    denominator: jnp.ndarray,
    fill_value: float = jnp.nan,
) -> jnp.ndarray:
    """
    Safe division that handles zero denominator.

    Parameters
    ----------
    numerator : array
        Numerator values.
    denominator : array
        Denominator values.
    fill_value : float, optional
        Value to use when denominator is zero. Default: NaN.

    Returns
    -------
    result : array
        numerator / denominator, with fill_value where denominator is zero.
    """
    is_zero = jnp.abs(denominator) < 1e-300
    safe_denom = jnp.where(is_zero, 1.0, denominator)
    result = numerator / safe_denom
    return jnp.where(is_zero, fill_value, result)


@jax.jit
def safe_sqrt(x: jnp.ndarray) -> jnp.ndarray:
    """
    Safe square root that returns NaN for negative inputs.

    Parameters
    ----------
    x : array
        Input values.

    Returns
    -------
    result : array
        sqrt(x) for x >= 0, NaN for x < 0.
    """
    is_negative = x < 0
    safe_x = jnp.where(is_negative, 0.0, x)
    result = jnp.sqrt(safe_x)
    return jnp.where(is_negative, jnp.nan, result)


@jax.jit
def safe_log(x: jnp.ndarray) -> jnp.ndarray:
    """
    Safe logarithm that returns NaN for non-positive inputs.

    Parameters
    ----------
    x : array
        Input values.

    Returns
    -------
    result : array
        log(x) for x > 0, NaN for x <= 0.
    """
    is_nonpositive = x <= 0
    safe_x = jnp.where(is_nonpositive, 1.0, x)
    result = jnp.log(safe_x)
    return jnp.where(is_nonpositive, jnp.nan, result)


def validate_inputs(
    *args: jnp.ndarray,
    raise_on_nan: bool = False,
    raise_on_inf: bool = False,
    context: str = "",
) -> bool:
    """
    Validate that inputs are finite (not NaN or Inf).

    Parameters
    ----------
    *args : arrays
        Input arrays to validate.
    raise_on_nan : bool, optional
        If True, raise PhysicsNaNError when NaN is found. Default: False.
    raise_on_inf : bool, optional
        If True, raise PhysicsInfError when Inf is found. Default: False.
    context : str, optional
        Description of the calling function for error messages.

    Returns
    -------
    all_finite : bool
        True if all inputs are finite.

    Raises
    ------
    PhysicsNaNError
        If raise_on_nan=True and NaN is found.
    PhysicsInfError
        If raise_on_inf=True and Inf is found.
    """
    import numpy as np

    for i, arr in enumerate(args):
        arr_np = np.asarray(arr)
        if np.any(np.isnan(arr_np)):
            if raise_on_nan:
                raise PhysicsNaNError(
                    f"NaN detected in input argument {i} of {context or 'physics function'}"
                )
            return False
        if np.any(np.isinf(arr_np)):
            if raise_on_inf:
                raise PhysicsInfError(
                    f"Inf detected in input argument {i} of {context or 'physics function'}"
                )
            return False
    return True


# =============================================================================
# Sauerbrey Equations
# =============================================================================


@jax.jit
def sauerbreyf(
    n: jnp.ndarray | float,
    drho: jnp.ndarray | float,
    f1: float = f1_default,
) -> jnp.ndarray:
    """
    Calculate Sauerbrey frequency shift from mass per unit area.

    The Sauerbrey equation relates mass loading to frequency shift:
        delf = 2 * n * f1^2 * drho / Zq

    Parameters
    ----------
    n : int or array
        Harmonic number(s) (1, 3, 5, 7, 9, ...).
    drho : float or array
        Mass per unit area [kg/m^2].
    f1 : float, optional
        Fundamental resonant frequency [Hz]. Default: 5e6 Hz.

    Returns
    -------
    delf : float or array
        Sauerbrey frequency shift [Hz]. Positive for mass increase.

    Notes
    -----
    This is a pure function suitable for JIT compilation and vmap.
    The convention used here gives positive delf for mass increase,
    but experimentally frequency decreases with mass, so experimental
    delf values are typically negative.

    Examples
    --------
    >>> delf = sauerbreyf(3, 1e-6)  # 3rd harmonic, 1 ug/cm^2
    """
    n = jnp.asarray(n, dtype=jnp.float64)
    drho = jnp.asarray(drho, dtype=jnp.float64)
    return 2 * n * f1**2 * drho / Zq


@jax.jit
def sauerbreym(
    n: jnp.ndarray | float,
    delf: jnp.ndarray | float,
    f1: float = f1_default,
) -> jnp.ndarray:
    """
    Calculate mass per unit area from Sauerbrey frequency shift.

    Inverse of sauerbreyf:
        drho = delf * Zq / (2 * n * f1^2)

    Parameters
    ----------
    n : int or array
        Harmonic number(s) (1, 3, 5, 7, 9, ...).
    delf : float or array
        Frequency shift [Hz]. Typically negative for mass increase.
    f1 : float, optional
        Fundamental resonant frequency [Hz]. Default: 5e6 Hz.

    Returns
    -------
    drho : float or array
        Mass per unit area [kg/m^2].

    Notes
    -----
    For experimental data where frequency decreases with mass loading,
    pass negative delf to get positive drho.
    """
    n = jnp.asarray(n, dtype=jnp.float64)
    delf = jnp.asarray(delf, dtype=jnp.float64)
    return -delf * Zq / (2 * n * f1**2)


# =============================================================================
# Complex Modulus Calculations
# =============================================================================


@partial(jax.jit, static_argnames=["refh"])
def grho(
    n: jnp.ndarray | float,
    grho_refh: jnp.ndarray | float,
    phi: jnp.ndarray | float,
    refh: int = 3,
) -> jnp.ndarray:
    """
    Calculate |G*|*rho at harmonic n using power law frequency scaling.

    The power law model assumes:
        |G*|_n = |G*|_refh * (n/refh)^(phi / (pi/2))

    Parameters
    ----------
    n : int or array
        Harmonic number(s) of interest.
    grho_refh : float or array
        |G*|*rho at reference harmonic [Pa kg/m^3].
    phi : float or array
        Phase angle [radians]. Range: 0 to pi/2.
    refh : int, optional
        Reference harmonic number. Default: 3.

    Returns
    -------
    grho_n : float or array
        |G*|*rho at harmonic n [Pa kg/m^3].

    Notes
    -----
    - phi = 0: purely elastic (grho independent of n)
    - phi = pi/2: purely viscous (grho scales linearly with n)
    """
    n = jnp.asarray(n, dtype=jnp.float64)
    grho_refh = jnp.asarray(grho_refh, dtype=jnp.float64)
    phi = jnp.asarray(phi, dtype=jnp.float64)
    return grho_refh * (n / refh) ** (phi / (jnp.pi / 2))


@jax.jit
def grhostar(
    grho_val: jnp.ndarray | float,
    phi: jnp.ndarray | float,
) -> jnp.ndarray:
    """
    Calculate complex G*rho from magnitude and phase angle.

    G*rho = |G*|rho * exp(i * phi)

    Parameters
    ----------
    grho_val : float or array
        Magnitude |G*|*rho [Pa kg/m^3].
    phi : float or array
        Phase angle [radians]. Range: 0 to pi/2.

    Returns
    -------
    grhostar : complex or array
        Complex modulus times density [Pa kg/m^3].

    Notes
    -----
    The real part is G'*rho (storage modulus * density).
    The imaginary part is G''*rho (loss modulus * density).
    """
    grho_val = jnp.asarray(grho_val, dtype=jnp.float64)
    phi = jnp.asarray(phi, dtype=jnp.float64)
    return grho_val * jnp.exp(1j * phi)


@partial(jax.jit, static_argnames=["refh"])
def grhostar_from_refh(
    n: jnp.ndarray | float,
    grho_refh: jnp.ndarray | float,
    phi: jnp.ndarray | float,
    refh: int = 3,
) -> jnp.ndarray:
    """
    Calculate complex G*rho at harmonic n from reference harmonic properties.

    Combines grho() and grhostar() for convenience.

    Parameters
    ----------
    n : int or array
        Harmonic number(s) of interest.
    grho_refh : float or array
        |G*|*rho at reference harmonic [Pa kg/m^3].
    phi : float or array
        Phase angle [radians].
    refh : int, optional
        Reference harmonic number. Default: 3.

    Returns
    -------
    grhostar_n : complex or array
        Complex modulus times density at harmonic n.
    """
    grho_n = grho(n, grho_refh, phi, refh=refh)
    return grhostar(grho_n, phi)


@jax.jit
def grho_from_dlam(
    n: jnp.ndarray | float,
    drho: jnp.ndarray | float,
    dlam_refh: jnp.ndarray | float,
    phi: jnp.ndarray | float,
    f1: float = f1_default,
) -> jnp.ndarray:
    """
    Calculate |G*|*rho from d/lambda (film thickness / shear wavelength).

    Inverse relationship:
        grho = (drho * n * f1 * cos(phi/2) / dlam_refh)^2

    Parameters
    ----------
    n : int or array
        Harmonic number.
    drho : float or array
        Mass per unit area [kg/m^2].
    dlam_refh : float or array
        d/lambda at reference harmonic (dimensionless).
    phi : float or array
        Phase angle [radians].
    f1 : float, optional
        Fundamental frequency [Hz]. Default: 5e6 Hz.

    Returns
    -------
    grho : float or array
        |G*|*rho [Pa kg/m^3].
    """
    n = jnp.asarray(n, dtype=jnp.float64)
    drho = jnp.asarray(drho, dtype=jnp.float64)
    dlam_refh = jnp.asarray(dlam_refh, dtype=jnp.float64)
    phi = jnp.asarray(phi, dtype=jnp.float64)

    # Avoid division by zero
    dlam_safe = jnp.where(dlam_refh == 0, 1e-10, dlam_refh)

    return (drho * n * f1 * jnp.cos(phi / 2) / dlam_safe) ** 2


@jax.jit
def calc_dlam(
    n: jnp.ndarray | float,
    grho_n: jnp.ndarray | float,
    phi: jnp.ndarray | float,
    drho: jnp.ndarray | float,
    f1: float = f1_default,
) -> jnp.ndarray:
    """
    Calculate d/lambda (film thickness normalized by shear wavelength).

    dlam = drho * n * f1 * cos(phi/2) / sqrt(grho_n)

    Parameters
    ----------
    n : int or array
        Harmonic number.
    grho_n : float or array
        |G*|*rho at harmonic n [Pa kg/m^3].
    phi : float or array
        Phase angle [radians].
    drho : float or array
        Mass per unit area [kg/m^2].
    f1 : float, optional
        Fundamental frequency [Hz]. Default: 5e6 Hz.

    Returns
    -------
    dlam : float or array
        d/lambda (dimensionless).
    """
    n = jnp.asarray(n, dtype=jnp.float64)
    grho_n = jnp.asarray(grho_n, dtype=jnp.float64)
    phi = jnp.asarray(phi, dtype=jnp.float64)
    drho = jnp.asarray(drho, dtype=jnp.float64)

    return drho * n * f1 * jnp.cos(phi / 2) / jnp.sqrt(grho_n)


@jax.jit
def calc_lamrho(
    n: jnp.ndarray | float,
    grho_n: jnp.ndarray | float,
    phi: jnp.ndarray | float,
    f1: float = f1_default,
) -> jnp.ndarray:
    """
    Calculate lambda * rho (shear wavelength times density).

    lamrho = sqrt(grho_n) / (n * f1 * cos(phi/2))

    Parameters
    ----------
    n : int or array
        Harmonic number.
    grho_n : float or array
        |G*|*rho at harmonic n [Pa kg/m^3].
    phi : float or array
        Phase angle [radians].
    f1 : float, optional
        Fundamental frequency [Hz]. Default: 5e6 Hz.

    Returns
    -------
    lamrho : float or array
        Lambda * rho [m kg/m^3 = kg/m^2].
    """
    n = jnp.asarray(n, dtype=jnp.float64)
    grho_n = jnp.asarray(grho_n, dtype=jnp.float64)
    phi = jnp.asarray(phi, dtype=jnp.float64)

    return jnp.sqrt(grho_n) / (n * f1 * jnp.cos(phi / 2))


@jax.jit
def calc_deltarho(
    n: jnp.ndarray | float,
    grho_n: jnp.ndarray | float,
    phi: jnp.ndarray | float,
    f1: float = f1_default,
) -> jnp.ndarray:
    """
    Calculate decay length times density (delta * rho).

    deltarho = lamrho / (2 * pi * tan(phi/2))

    Parameters
    ----------
    n : int or array
        Harmonic number.
    grho_n : float or array
        |G*|*rho at harmonic n [Pa kg/m^3].
    phi : float or array
        Phase angle [radians].
    f1 : float, optional
        Fundamental frequency [Hz]. Default: 5e6 Hz.

    Returns
    -------
    deltarho : float or array
        Decay length * rho [kg/m^2].
    """
    lamrho = calc_lamrho(n, grho_n, phi, f1=f1)
    phi = jnp.asarray(phi, dtype=jnp.float64)
    return lamrho / (2 * jnp.pi * jnp.tan(phi / 2))


@jax.jit
def etarho(
    n: jnp.ndarray | float,
    grho_n: jnp.ndarray | float,
    f1: float = f1_default,
) -> jnp.ndarray:
    """
    Calculate viscosity * density at harmonic n.

    eta*rho = |G*|rho / (2 * pi * n * f1)

    Parameters
    ----------
    n : int or array
        Harmonic number.
    grho_n : float or array
        |G*|*rho at harmonic n [Pa kg/m^3].
    f1 : float, optional
        Fundamental frequency [Hz]. Default: 5e6 Hz.

    Returns
    -------
    etarho : float or array
        |eta*| * rho [Pa s kg/m^3].
    """
    n = jnp.asarray(n, dtype=jnp.float64)
    grho_n = jnp.asarray(grho_n, dtype=jnp.float64)
    return grho_n / (2 * jnp.pi * n * f1)


@jax.jit
def zstar_bulk(grhostar_val: jnp.ndarray) -> jnp.ndarray:
    """
    Calculate acoustic impedance from complex modulus.

    Z* = sqrt(G* * rho)

    Parameters
    ----------
    grhostar_val : complex or array
        Complex modulus times density [Pa kg/m^3].

    Returns
    -------
    zstar : complex or array
        Complex acoustic impedance [Pa s/m].
    """
    return jnp.sqrt(grhostar_val)


# =============================================================================
# Small Load Approximation (SLA) Equations
# =============================================================================


@jax.jit
def calc_delfstar_sla(
    ZL: jnp.ndarray,
    f1: float = f1_default,
) -> jnp.ndarray:
    """
    Calculate complex frequency shift using Small Load Approximation.

    delfstar = f1 * i * ZL / (pi * Zq)

    Parameters
    ----------
    ZL : complex or array
        Complex load impedance [Pa s/m].
    f1 : float, optional
        Fundamental frequency [Hz]. Default: 5e6 Hz.

    Returns
    -------
    delfstar : complex or array
        Complex frequency shift [Hz].
        Real part: frequency shift.
        Imaginary part: half-bandwidth shift (dissipation).
    """
    ZL = jnp.asarray(ZL, dtype=jnp.complex128)
    return f1 * 1j * ZL / (jnp.pi * Zq)


@partial(jax.jit, static_argnames=["refh"])
def calc_D(
    n: jnp.ndarray | float,
    drho: jnp.ndarray | float,
    grhostar_val: jnp.ndarray,
    delfstar: jnp.ndarray | float = 0.0,
    f1: float = f1_default,
    refh: int = 3,
) -> jnp.ndarray:
    """
    Calculate D (film thickness times complex wave number).

    D = 2 * pi * (n * f1 + delfstar) * drho / Z*

    Parameters
    ----------
    n : int or array
        Harmonic number.
    drho : float or array
        Mass per unit area [kg/m^2].
    grhostar_val : complex or array
        Complex modulus times density [Pa kg/m^3].
    delfstar : complex, optional
        Complex frequency shift [Hz]. Default: 0 (SLA).
    f1 : float, optional
        Fundamental frequency [Hz]. Default: 5e6 Hz.
    refh : int, optional
        Reference harmonic (not used here but kept for API consistency).

    Returns
    -------
    D : complex or array
        Film thickness times complex wave number (dimensionless).
    """
    n = jnp.asarray(n, dtype=jnp.float64)
    drho = jnp.asarray(drho, dtype=jnp.float64)
    grhostar_val = jnp.asarray(grhostar_val, dtype=jnp.complex128)
    delfstar = jnp.asarray(delfstar, dtype=jnp.complex128)

    zstar = zstar_bulk(grhostar_val)
    # Avoid division by zero
    zstar_safe = jnp.where(jnp.abs(zstar) == 0, 1e-20, zstar)

    return 2 * jnp.pi * (n * f1 + delfstar) * drho / zstar_safe


@jax.jit
def normdelfstar(
    n: jnp.ndarray | float,
    dlam3: jnp.ndarray | float,
    phi: jnp.ndarray | float,
) -> jnp.ndarray:
    """
    Calculate normalized complex frequency shift.

    Normalized by Sauerbrey shift.

    Parameters
    ----------
    n : int or array
        Harmonic number.
    dlam3 : float or array
        d/lambda at n=3.
    phi : float or array
        Phase angle [radians].

    Returns
    -------
    norm_delfstar : complex or array
        Normalized complex frequency shift.
    """
    n = jnp.asarray(n, dtype=jnp.float64)
    dlam3 = jnp.asarray(dlam3, dtype=jnp.float64)
    phi = jnp.asarray(phi, dtype=jnp.float64)

    # Calculate dlam at harmonic n
    dlam_n = dlam3 * (n / 3) ** (1 - phi / jnp.pi)

    # D = 2*pi*dlam*(1 - i*tan(phi/2))
    D = 2 * jnp.pi * dlam_n * (1 - 1j * jnp.tan(phi / 2))

    # Using sinc definition: sinc(x) = sin(pi*x)/(pi*x)
    # So sin(D)/D = sinc(D/pi)
    return -jnp.sinc(D / jnp.pi) / jnp.cos(D)


@jax.jit
def bulk_props(
    delfstar: jnp.ndarray,
    f1: float = f1_default,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Determine properties of bulk material from complex frequency shift.

    Parameters
    ----------
    delfstar : complex or array
        Complex frequency shift [Hz] at any harmonic.
    f1 : float, optional
        Fundamental frequency [Hz]. Default: 5e6 Hz.

    Returns
    -------
    grho : float or array
        |G*|*rho at the harmonic where delfstar was measured.
    phi : float or array
        Phase angle [radians], capped at pi/2.
    """
    delfstar = jnp.asarray(delfstar, dtype=jnp.complex128)

    grho_val = (jnp.pi * Zq * jnp.abs(delfstar) / f1) ** 2
    phi = jnp.minimum(
        jnp.pi / 2,
        -2 * jnp.arctan(jnp.real(delfstar) / jnp.imag(delfstar)),
    )

    return grho_val, phi


@jax.jit
def deltarho_bulk(
    n: jnp.ndarray | float,
    delfstar: jnp.ndarray,
    f1: float = f1_default,
) -> jnp.ndarray:
    """
    Calculate decay length * density for bulk material.

    Parameters
    ----------
    n : int or array
        Harmonic number.
    delfstar : complex
        Complex frequency shift at harmonic n.
    f1 : float, optional
        Fundamental frequency [Hz]. Default: 5e6 Hz.

    Returns
    -------
    deltarho : float or array
        Decay length times density [kg/m^2].
    """
    n = jnp.asarray(n, dtype=jnp.float64)
    delfstar = jnp.asarray(delfstar, dtype=jnp.complex128)

    return -Zq * jnp.abs(delfstar) ** 2 / (2 * n * f1**2 * jnp.real(delfstar))


# =============================================================================
# Kotula Model (Complex Root Finding)
# =============================================================================


@jax.jit
def _kotula_equation(
    gstar: jnp.ndarray,
    xi: jnp.ndarray,
    Gmstar: jnp.ndarray,
    Gfstar: jnp.ndarray,
    xi_crit: float,
    s: float,
    t: float,
) -> jnp.ndarray:
    """
    Kotula model equation for root finding.

    Returns the residual that should be zero at the solution.

    Parameters
    ----------
    gstar : complex
        Current estimate of complex modulus.
    xi : float
        Filler volume fraction.
    Gmstar : complex
        Matrix complex modulus.
    Gfstar : complex
        Filler complex modulus.
    xi_crit : float
        Critical filler volume fraction.
    s, t : float
        Exponents.

    Returns
    -------
    residual : complex
        Should be zero at solution.
    """
    A = (1 - xi_crit) / xi_crit

    term1 = (1 - xi) * (Gmstar ** (1 / s) - gstar ** (1 / s)) / (
        Gmstar ** (1 / s) + A * gstar ** (1 / s)
    )
    term2 = xi * (Gfstar ** (1 / t) - gstar ** (1 / t)) / (
        Gfstar ** (1 / t) + A * gstar ** (1 / t)
    )

    return term1 + term2


def _newton_step_complex_damped(
    gstar: jnp.ndarray,
    xi: jnp.ndarray,
    Gmstar: jnp.ndarray,
    Gfstar: jnp.ndarray,
    xi_crit: float,
    s: float,
    t: float,
    damping: float = 0.5,
) -> jnp.ndarray:
    """
    Perform one damped Newton-Raphson step for complex root finding.

    Uses finite differences for the Jacobian with damping for stability.
    """
    # Compute function value
    f_val = _kotula_equation(gstar, xi, Gmstar, Gfstar, xi_crit, s, t)

    # Compute derivative with respect to real and imaginary parts
    # Use relative epsilon for numerical stability
    eps = 1e-8 * jnp.maximum(1.0, jnp.abs(gstar))
    f_plus_real = _kotula_equation(gstar + eps, xi, Gmstar, Gfstar, xi_crit, s, t)
    f_plus_imag = _kotula_equation(gstar + eps * 1j, xi, Gmstar, Gfstar, xi_crit, s, t)

    df_dreal = (f_plus_real - f_val) / eps
    df_dimag = (f_plus_imag - f_val) / eps

    # Wirtinger derivative for complex function
    df_dgstar = (df_dreal - 1j * df_dimag) / 2

    # Newton update with damping
    # Avoid division by zero with a small regularization
    df_magnitude = jnp.abs(df_dgstar)
    df_safe = jnp.where(df_magnitude < 1e-20, 1e-20, df_dgstar)
    step = f_val / df_safe

    # Apply damping and limit step size
    step_size = jnp.abs(step)
    max_step = 0.5 * jnp.abs(gstar)  # Limit step to 50% of current value
    scale = jnp.where(step_size > max_step, max_step / step_size, 1.0)

    return gstar - damping * scale * step


def _solve_kotula_single(
    xi: float,
    Gmstar: jnp.ndarray,
    Gfstar: jnp.ndarray,
    xi_crit: float,
    s: float,
    t: float,
    max_iter: int = 100,
    tol: float = 1e-10,
) -> jnp.ndarray:
    """
    Solve Kotula equation for a single xi value.

    Uses damped Newton-Raphson iteration with JAX.

    Returns
    -------
    gstar : complex
        Solution to Kotula equation, or NaN if non-convergent.
        Non-convergence is detectable via ``jnp.isnan(result)``.
    """
    xi = jnp.asarray(xi, dtype=jnp.float64)

    # Initial guess: geometric interpolation between Gmstar and Gfstar
    # This is more stable for complex moduli spanning many orders of magnitude
    log_Gm = jnp.log(jnp.abs(Gmstar) + 1e-20)
    log_Gf = jnp.log(jnp.abs(Gfstar) + 1e-20)
    log_G0 = (1 - xi) * log_Gm + xi * log_Gf
    phase_interp = (1 - xi) * jnp.angle(Gmstar) + xi * jnp.angle(Gfstar)
    gstar = jnp.exp(log_G0) * jnp.exp(1j * phase_interp)

    def body_fn(carry):
        gstar, i = carry
        new_gstar = _newton_step_complex_damped(
            gstar, xi, Gmstar, Gfstar, xi_crit, s, t, damping=0.7
        )
        return (new_gstar, i + 1)

    def cond_fn(carry):
        gstar, i = carry
        residual = jnp.abs(_kotula_equation(gstar, xi, Gmstar, Gfstar, xi_crit, s, t))
        return (residual > tol) & (i < max_iter)

    gstar, final_iter = lax.while_loop(cond_fn, body_fn, (gstar, 0))

    # Check convergence and return NaN if not converged
    final_residual = jnp.abs(_kotula_equation(gstar, xi, Gmstar, Gfstar, xi_crit, s, t))
    converged = final_residual <= tol
    nan_complex = jnp.nan + 1j * jnp.nan
    return jnp.where(converged, gstar, nan_complex)


def kotula_gstar(
    xi: jnp.ndarray | float,
    Gmstar: jnp.ndarray,
    Gfstar: jnp.ndarray,
    xi_crit: float,
    s: float,
    t: float,
    tol: float = 1e-10,
    max_iter: int = 100,
) -> jnp.ndarray:
    """
    Calculate complex modulus using Kotula model.

    The Kotula model describes the effective complex modulus of a
    composite material as a function of filler volume fraction.

    Parameters
    ----------
    xi : float or array
        Filler volume fraction(s). Range: 0 to 1.
    Gmstar : complex
        Matrix complex modulus.
    Gfstar : complex
        Filler complex modulus.
    xi_crit : float
        Critical filler volume fraction (percolation threshold).
    s : float
        Exponent for matrix contribution.
    t : float
        Exponent for filler contribution.
    tol : float, optional
        Convergence tolerance for Newton-Raphson iteration. Default 1e-10.
    max_iter : int, optional
        Maximum number of iterations. Default 100.

    Returns
    -------
    gstar : complex or array
        Effective complex modulus at each xi. Returns NaN (complex NaN)
        for any xi value where the Newton-Raphson iteration fails to
        converge within tolerance. Non-convergence is detectable via
        ``jnp.isnan(result)``.

    Notes
    -----
    This implementation uses JAX-compatible Newton-Raphson iteration
    instead of mpmath.findroot for GPU acceleration and JIT compilation.
    The solver uses damped Newton-Raphson with geometric interpolation
    for the initial guess.
    """
    xi = jnp.asarray(xi, dtype=jnp.float64)
    Gmstar = jnp.asarray(Gmstar, dtype=jnp.complex128)
    Gfstar = jnp.asarray(Gfstar, dtype=jnp.complex128)

    # Handle scalar vs array input
    if xi.ndim == 0:
        return _solve_kotula_single(xi, Gmstar, Gfstar, xi_crit, s, t, max_iter, tol)
    else:
        # Vectorize over xi values
        solve_fn = lambda x: _solve_kotula_single(
            x, Gmstar, Gfstar, xi_crit, s, t, max_iter, tol
        )
        return jax.vmap(solve_fn)(xi)


@jax.jit
def kotula_xi(
    gstar: jnp.ndarray,
    Gmstar: jnp.ndarray,
    Gfstar: jnp.ndarray,
    xi_crit: float,
    s: float,
    t: float,
) -> jnp.ndarray:
    """
    Calculate filler fraction from complex modulus (inverse of kotula_gstar).

    Analytical solution for xi given gstar.

    Parameters
    ----------
    gstar : complex or array
        Effective complex modulus.
    Gmstar : complex
        Matrix complex modulus.
    Gfstar : complex
        Filler complex modulus.
    xi_crit : float
        Critical filler volume fraction.
    s : float
        Exponent for matrix contribution.
    t : float
        Exponent for filler contribution.

    Returns
    -------
    xi : float or array
        Filler volume fraction(s).
    """
    gstar = jnp.asarray(gstar, dtype=jnp.complex128)
    Gmstar = jnp.asarray(Gmstar, dtype=jnp.complex128)
    Gfstar = jnp.asarray(Gfstar, dtype=jnp.complex128)

    A = (1 - xi_crit) / xi_crit

    numerator = (
        -A * Gmstar ** (1 / s) * gstar ** (1 / t)
        + A * gstar ** ((s + t) / (s * t))
        - Gfstar ** (1 / t) * Gmstar ** (1 / s)
        + Gfstar ** (1 / t) * gstar ** (1 / s)
    )

    denominator = (
        A * Gfstar ** (1 / t) * gstar ** (1 / s)
        - A * Gmstar ** (1 / s) * gstar ** (1 / t)
        + Gfstar ** (1 / t) * gstar ** (1 / s)
        - Gmstar ** (1 / s) * gstar ** (1 / t)
    )

    return numerator / denominator


# =============================================================================
# Utility Functions (scipy replacements)
# =============================================================================


def find_peaks(data: jnp.ndarray) -> jnp.ndarray:
    """
    Find local maxima in 1D data.

    Simple implementation using jnp.where for local maxima detection.
    Replaces scipy.signal.find_peaks.

    Parameters
    ----------
    data : array
        1D input data.

    Returns
    -------
    peak_indices : array
        Indices of local maxima.

    Notes
    -----
    A point is considered a peak if it is strictly greater than
    its immediate neighbors.

    Note: This function is NOT JIT-compiled because it returns
    variable-length output. Use jax.numpy operations on the result.
    """
    import numpy as np

    data_np = np.asarray(data)

    # Compare each point to its neighbors
    # peaks[i] is True if data[i] > data[i-1] and data[i] > data[i+1]
    greater_than_left = data_np[1:-1] > data_np[:-2]
    greater_than_right = data_np[1:-1] > data_np[2:]
    is_peak = greater_than_left & greater_than_right

    # Get indices (add 1 to account for slicing)
    peak_mask = np.concatenate([[False], is_peak, [False]])
    indices = np.arange(len(data_np))

    return jnp.array(indices[peak_mask])


@jax.jit
def interp_linear(
    x_new: jnp.ndarray,
    x: jnp.ndarray,
    y: jnp.ndarray,
) -> jnp.ndarray:
    """
    Linear interpolation using JAX.

    Replaces scipy.interpolate.interp1d with kind='linear'.

    Parameters
    ----------
    x_new : array
        New x values at which to interpolate.
    x : array
        Original x values (must be sorted).
    y : array
        Original y values.

    Returns
    -------
    y_new : array
        Interpolated y values.
    """
    x_new = jnp.asarray(x_new, dtype=jnp.float64)
    x = jnp.asarray(x, dtype=jnp.float64)
    y = jnp.asarray(y, dtype=jnp.float64)

    return jnp.interp(x_new, x, y)


def interp_cubic(
    x_new: jnp.ndarray,
    x: jnp.ndarray,
    y: jnp.ndarray,
    extrap: Union[bool, float, tuple] = jnp.nan,
) -> jnp.ndarray:
    """
    Cubic spline interpolation using interpax.

    Replaces scipy.interpolate.interp1d with kind='cubic'.

    Parameters
    ----------
    x_new : array
        New x values at which to interpolate.
    x : array
        Original x values (must be sorted).
    y : array
        Original y values.
    extrap : bool, float, or tuple, optional
        Extrapolation behavior. Default: NaN for out-of-bounds values.
        - False: return NaN for out-of-bounds
        - True: extrapolate
        - float: return this value for out-of-bounds
        - tuple: (left_value, right_value)

    Returns
    -------
    y_new : array
        Interpolated y values.

    Notes
    -----
    Uses interpax.interp1d with method='cubic' for C1 cubic splines.
    This is JIT-compatible when used with JAX arrays.
    """
    x_new = jnp.asarray(x_new, dtype=jnp.float64)
    x = jnp.asarray(x, dtype=jnp.float64)
    y = jnp.asarray(y, dtype=jnp.float64)

    return interpax.interp1d(x_new, x, y, method="cubic", extrap=extrap)


def create_interp_func(
    x: jnp.ndarray,
    y: jnp.ndarray,
    method: str = "linear",
    extrap: Union[bool, float, tuple] = jnp.nan,
):
    """
    Create an interpolation function similar to scipy.interpolate.interp1d.

    This returns a callable that can be used like scipy's interp1d object.

    Parameters
    ----------
    x : array
        Original x values (must be sorted).
    y : array
        Original y values.
    method : str, optional
        Interpolation method: 'linear' or 'cubic'. Default: 'linear'.
    extrap : bool, float, or tuple, optional
        Extrapolation behavior. Default: NaN for out-of-bounds values.

    Returns
    -------
    interp_func : callable
        Function that takes x_new and returns interpolated y values.

    Examples
    --------
    >>> x = jnp.array([0., 1., 2., 3.])
    >>> y = jnp.array([0., 2., 4., 6.])
    >>> f = create_interp_func(x, y, method='linear')
    >>> f(jnp.array([0.5, 1.5, 2.5]))
    Array([1., 3., 5.], dtype=float64)
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    y = jnp.asarray(y, dtype=jnp.float64)

    if method == "linear":

        def interp_func(x_new):
            x_new = jnp.asarray(x_new, dtype=jnp.float64)
            # Handle out-of-bounds with NaN
            result = jnp.interp(x_new, x, y)
            if extrap is False or (
                isinstance(extrap, float) and jnp.isnan(extrap)
            ):
                # Set out-of-bounds to NaN
                mask_low = x_new < x[0]
                mask_high = x_new > x[-1]
                result = jnp.where(mask_low | mask_high, jnp.nan, result)
            elif isinstance(extrap, float):
                mask_low = x_new < x[0]
                mask_high = x_new > x[-1]
                result = jnp.where(mask_low | mask_high, extrap, result)
            return result

    elif method == "cubic":

        def interp_func(x_new):
            x_new = jnp.asarray(x_new, dtype=jnp.float64)
            return interpax.interp1d(x_new, x, y, method="cubic", extrap=extrap)

    else:
        raise ValueError(f"Unknown interpolation method: {method}")

    return interp_func


def _savgol_coeffs(window_length: int, polyorder: int) -> jnp.ndarray:
    """
    Compute Savitzky-Golay filter coefficients.

    Parameters
    ----------
    window_length : int
        Length of filter window (must be odd and > polyorder).
    polyorder : int
        Order of polynomial to fit.

    Returns
    -------
    coeffs : array
        Filter coefficients for convolution.
    """
    # Half window size
    m = window_length // 2

    # Create Vandermonde matrix
    x = jnp.arange(-m, m + 1, dtype=jnp.float64)
    A = jnp.vander(x, N=polyorder + 1, increasing=True)

    # Solve for coefficients using least squares
    # coeffs = (A^T A)^-1 A^T @ delta_0
    # where delta_0 selects the center point (we want smoothed value at center)
    ATA = A.T @ A
    ATA_inv = jnp.linalg.inv(ATA)
    coeffs = ATA_inv @ A.T

    # First row gives smoothing coefficients
    return coeffs[0]


def savgol_filter(
    data: jnp.ndarray,
    window_length: int = 11,
    polyorder: int = 3,
) -> jnp.ndarray:
    """
    Apply Savitzky-Golay filter using convolution.

    Replaces scipy.signal.savgol_filter.

    Parameters
    ----------
    data : array
        Input data (1D).
    window_length : int, optional
        Length of filter window (must be odd). Default: 11.
    polyorder : int, optional
        Order of polynomial. Default: 3.

    Returns
    -------
    filtered : array
        Filtered data.

    Notes
    -----
    Uses pre-calculated coefficients with JAX convolution.
    Edge handling uses 'same' mode (output same size as input).
    """
    data = jnp.asarray(data, dtype=jnp.float64)

    # Get filter coefficients
    coeffs = _savgol_coeffs(window_length, polyorder)

    # Reverse for convolution (convolution flips kernel)
    coeffs_rev = coeffs[::-1]

    # Pad data for edge handling
    pad_size = window_length // 2
    data_padded = jnp.pad(data, pad_size, mode="edge")

    # Convolve
    filtered = convolve(data_padded, coeffs_rev, mode="same")

    # Remove padding
    return filtered[pad_size:-pad_size] if pad_size > 0 else filtered


# =============================================================================
# Exports
# =============================================================================

__all__ = [
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
    # NaN/Inf handling (T018)
    "PhysicsNaNError",
    "PhysicsInfError",
    "check_finite",
    "propagate_nan",
    "propagate_nan_complex",
    "safe_divide",
    "safe_sqrt",
    "safe_log",
    "validate_inputs",
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
    "_kotula_equation",
    # Utility functions
    "find_peaks",
    "interp_linear",
    "interp_cubic",
    "create_interp_func",
    "savgol_filter",
]
