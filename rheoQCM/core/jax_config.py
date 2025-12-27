"""
JAX Configuration Module for RheoQCM

This module configures JAX for scientific computing with Float64 precision
and optimal platform settings. It should be imported before any other JAX
operations to ensure consistent behavior.

Configuration settings:
    - Float64 as default dtype for all calculations
    - Platform configuration (CPU/GPU detection)
    - No subsampling or random SVD optimizations (numerical precision priority)

Usage:
    from rheoQCM.core.jax_config import configure_jax
    configure_jax()  # Call once at application startup
"""

import os
from typing import Literal

# Configure JAX before importing it
# These environment variables must be set BEFORE jax is imported
os.environ.setdefault("JAX_ENABLE_X64", "True")

import jax
import jax.numpy as jnp

# Track configuration state
_configured: bool = False


def configure_jax(
    enable_x64: bool = True,
    platform: Literal["cpu", "gpu", "tpu", "auto"] = "auto",
) -> dict[str, str]:
    """
    Configure JAX for scientific computing with Float64 precision.

    This function should be called once at application startup, before any
    JAX operations are performed. It sets global configuration that affects
    all subsequent JAX computations.

    Parameters
    ----------
    enable_x64 : bool, default=True
        Enable 64-bit floating point precision. Required for scientific
        computing to maintain numerical precision in curve fitting and
        physics calculations.
    platform : {"cpu", "gpu", "tpu", "auto"}, default="auto"
        Target platform for JAX operations.
        - "auto": Automatically detect available hardware (prefer GPU/TPU)
        - "cpu": Force CPU execution
        - "gpu": Force GPU execution (fails if GPU unavailable)
        - "tpu": Force TPU execution (fails if TPU unavailable)

    Returns
    -------
    dict[str, str]
        Configuration summary with keys:
        - "x64_enabled": Whether Float64 is enabled
        - "platform": Active platform
        - "devices": List of available devices

    Raises
    ------
    RuntimeError
        If requested platform is not available.

    Notes
    -----
    - Float64 precision is essential for QCM physics calculations
    - No subsampling or random SVD optimizations are used
    - Numerical precision takes priority over computational speed
    """
    global _configured

    # Enable 64-bit precision
    if enable_x64:
        jax.config.update("jax_enable_x64", True)

    # Configure platform
    if platform != "auto":
        jax.config.update("jax_platform_name", platform)

    # Verify configuration
    devices = jax.devices()
    active_platform = devices[0].platform if devices else "unknown"

    # Validate requested platform
    if platform not in ("auto", active_platform):
        available_platforms = {d.platform for d in devices}
        if platform not in available_platforms:
            raise RuntimeError(
                f"Requested platform '{platform}' not available. "
                f"Available platforms: {available_platforms}"
            )

    _configured = True

    return {
        "x64_enabled": str(jax.config.jax_enable_x64),
        "platform": active_platform,
        "devices": str([str(d) for d in devices]),
    }


def get_jax_backend() -> str:
    """
    Get the current JAX backend platform.

    Returns
    -------
    str
        Platform name: "cpu", "gpu", "tpu", or "unknown"
    """
    devices = jax.devices()
    if not devices:
        return "unknown"
    return devices[0].platform


def is_gpu_available() -> bool:
    """
    Check if GPU acceleration is available.

    Returns
    -------
    bool
        True if at least one GPU device is available for JAX.
    """
    try:
        gpu_devices = jax.devices("gpu")
        return len(gpu_devices) > 0
    except RuntimeError:
        return False


def is_tpu_available() -> bool:
    """
    Check if TPU acceleration is available.

    Returns
    -------
    bool
        True if at least one TPU device is available for JAX.
    """
    try:
        tpu_devices = jax.devices("tpu")
        return len(tpu_devices) > 0
    except RuntimeError:
        return False


def get_default_dtype() -> jnp.dtype:
    """
    Get the default floating point dtype for JAX arrays.

    Returns
    -------
    jnp.dtype
        The default dtype, which is float64 when x64 is enabled.
    """
    if jax.config.jax_enable_x64:
        return jnp.float64
    return jnp.float32


def verify_float64() -> bool:
    """
    Verify that Float64 precision is active.

    Creates a test array and checks its dtype to ensure that Float64
    is properly configured.

    Returns
    -------
    bool
        True if Float64 precision is active, False otherwise.
    """
    test_array = jnp.array([1.0])
    return test_array.dtype == jnp.float64


def print_jax_info() -> None:
    """
    Print JAX configuration information for diagnostics.

    Outputs information about:
    - JAX version
    - Available devices
    - Current platform
    - Float64 status
    - Default dtype
    """
    print(f"JAX version: {jax.__version__}")
    print(f"Available devices: {jax.devices()}")
    print(f"Current platform: {get_jax_backend()}")
    print(f"GPU available: {is_gpu_available()}")
    print(f"TPU available: {is_tpu_available()}")
    print(f"Float64 enabled: {jax.config.jax_enable_x64}")
    print(f"Default dtype: {get_default_dtype()}")
    print(f"Float64 verified: {verify_float64()}")


# Auto-configure on import if not already configured
if not _configured:
    configure_jax()
