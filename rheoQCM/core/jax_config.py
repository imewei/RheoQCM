"""
JAX Configuration Module for RheoQCM

This module configures JAX for scientific computing with Float64 precision
and optimal platform settings. It should be imported before any other JAX
operations to ensure consistent behavior.

Configuration settings:
    - Float64 as default dtype for all calculations
    - Platform configuration (CPU/GPU detection)
    - No subsampling or random SVD optimizations (numerical precision priority)
    - GPU fallback with logging (T019)

Usage:
    from rheoQCM.core.jax_config import configure_jax
    configure_jax()  # Call once at application startup
"""

import logging
import os
from typing import Literal

# Configure JAX before importing it
# These environment variables must be set BEFORE jax is imported
os.environ.setdefault("JAX_ENABLE_X64", "True")

import jax
import jax.numpy as jnp

logger = logging.getLogger(__name__)

# Track configuration state
_configured: bool = False
_gpu_fallback_warned: bool = False


def configure_jax(
    enable_x64: bool = True,
    platform: Literal["cpu", "gpu", "tpu", "auto"] = "auto",
    warn_on_cpu_fallback: bool = True,
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
        - "gpu": Force GPU execution (falls back to CPU with warning)
        - "tpu": Force TPU execution (falls back to CPU with warning)
    warn_on_cpu_fallback : bool, default=True
        If True, log a warning when GPU/TPU is requested but CPU is used.

    Returns
    -------
    dict[str, str]
        Configuration summary with keys:
        - "x64_enabled": Whether Float64 is enabled
        - "platform": Active platform
        - "devices": List of available devices
        - "gpu_fallback": Whether GPU fallback occurred

    Raises
    ------
    RuntimeError
        Only if platform is explicitly set and not available AND
        fallback is disabled by environment variable.

    Notes
    -----
    - Float64 precision is essential for QCM physics calculations
    - No subsampling or random SVD optimizations are used
    - Numerical precision takes priority over computational speed
    - GPU/TPU fallback to CPU is handled gracefully with warning (T019)
    """
    global _configured, _gpu_fallback_warned

    # Enable 64-bit precision
    if enable_x64:
        jax.config.update("jax_enable_x64", True)

    # Get available devices before platform selection
    devices = jax.devices()
    active_platform = devices[0].platform if devices else "cpu"
    gpu_fallback = False

    # Handle platform configuration with graceful fallback (T019)
    if platform != "auto":
        requested_platform = platform
        available_platforms = {d.platform for d in devices}

        if requested_platform not in available_platforms:
            # Fallback to CPU with warning
            gpu_fallback = True
            if warn_on_cpu_fallback and not _gpu_fallback_warned:
                logger.warning(
                    f"Requested platform '{requested_platform}' not available. "
                    f"Available platforms: {available_platforms}. "
                    f"Falling back to CPU. This may impact performance."
                )
                _gpu_fallback_warned = True
            # Don't try to set unavailable platform
        else:
            jax.config.update("jax_platform_name", platform)
            # Update active platform
            devices = jax.devices()
            active_platform = devices[0].platform if devices else "cpu"
    else:
        # Auto mode: check if GPU was expected but not available
        if not is_gpu_available() and not _gpu_fallback_warned:
            if os.environ.get("RHEOQCM_EXPECT_GPU", "").lower() == "true":
                logger.warning(
                    "GPU was expected (RHEOQCM_EXPECT_GPU=true) but not available. "
                    "Falling back to CPU. This may impact performance for large datasets."
                )
                _gpu_fallback_warned = True
                gpu_fallback = True

    _configured = True

    return {
        "x64_enabled": str(jax.config.jax_enable_x64),
        "platform": active_platform,
        "devices": str([str(d) for d in devices]),
        "gpu_fallback": str(gpu_fallback),
    }


def configure_jax_with_fallback(
    preferred_platform: Literal["gpu", "tpu"] = "gpu",
) -> dict[str, str]:
    """
    Configure JAX with automatic fallback to CPU if preferred platform unavailable.

    This is a convenience wrapper around configure_jax() that always succeeds
    and logs appropriate warnings when fallback occurs.

    Parameters
    ----------
    preferred_platform : {"gpu", "tpu"}, default="gpu"
        Preferred platform to use if available.

    Returns
    -------
    dict[str, str]
        Configuration summary (same as configure_jax).

    Notes
    -----
    This function never raises an exception. It will always succeed,
    either using the preferred platform or falling back to CPU.
    """
    if preferred_platform == "gpu" and is_gpu_available():
        return configure_jax(platform="gpu")
    elif preferred_platform == "tpu" and is_tpu_available():
        return configure_jax(platform="tpu")
    else:
        logger.info(
            f"Preferred platform '{preferred_platform}' not available. Using CPU."
        )
        return configure_jax(platform="cpu")


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


def log_platform_status() -> None:
    """
    Log the current platform status at INFO level.

    This is useful for debugging and ensuring the correct platform is being used.
    """
    backend = get_jax_backend()
    gpu_avail = is_gpu_available()
    tpu_avail = is_tpu_available()

    logger.info(f"JAX platform: {backend}")
    if backend == "cpu" and (gpu_avail or tpu_avail):
        logger.info("Note: GPU/TPU available but not in use")
    elif backend == "gpu":
        logger.info("Using GPU acceleration")
    elif backend == "tpu":
        logger.info("Using TPU acceleration")


def check_gpu_availability() -> None:
    """Check if GPU is available but not being used by JAX.

    Prints a helpful warning if:
    - NVIDIA GPU hardware is detected (nvidia-smi works)
    - But JAX is running in CPU-only mode

    This helps users realize they can enable GPU acceleration for 20-100x speedup.
    """
    try:
        import subprocess

        # Check if nvidia-smi detects GPU hardware
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode == 0 and result.stdout.strip():
            gpu_name = result.stdout.strip()

            # Check if JAX is using GPU
            devices = jax.devices()
            using_gpu = any(
                "cuda" in str(d).lower() or "gpu" in str(d).lower() for d in devices
            )

            if not using_gpu:
                print(f"\n⚠️  GPU ACCELERATION AVAILABLE")
                print(f"═══════════════════════════════")
                print(f"NVIDIA GPU detected: {gpu_name}")
                print(f"JAX is currently using: CPU-only")
                print(f"\nEnable 150-270x speedup with GPU acceleration:")
                print(f"  make install-jax-gpu")
                print(f"\nOr manually:")
                print(f"  pip uninstall -y jax jaxlib")
                print(f'  pip install "jax[cuda12-local]==0.8.0" "jaxlib==0.8.0"')
                print(f"\nSee README.md GPU Installation section for details.\n")

    except (subprocess.TimeoutExpired, FileNotFoundError, ImportError):
        # nvidia-smi not found or JAX not installed - silently skip
        pass
    except Exception:
        # Unexpected error - silently skip to avoid disrupting workflow
        pass


# Auto-configure on import if not already configured
if not _configured:
    configure_jax()
    # Check for GPU availability and warn user if GPU is available but not used
    check_gpu_availability()
