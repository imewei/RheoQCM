"""
Tests for JAX Configuration and Headless Operation.

These tests verify:
- Float64 enforcement (T030)
- Device detection CPU/GPU (T031)
- Headless operation without PyQt6 (T032, T035)
"""

import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rheoQCM.core import configure_jax, get_jax_backend


class TestFloat64Enforcement:
    """Tests for float64 precision enforcement (T030)."""

    def test_jax_float64_enabled(self) -> None:
        """Test that JAX Float64 mode is enabled."""
        configure_jax()
        assert jax.config.jax_enable_x64, "JAX Float64 must be enabled"

    def test_float64_array_creation(self) -> None:
        """Test that float64 arrays are created correctly."""
        configure_jax()
        x = jnp.array([1.0, 2.0, 3.0])
        assert x.dtype == jnp.float64

    def test_float64_precision_maintained(self) -> None:
        """Test that float64 precision is maintained in calculations."""
        configure_jax()

        # Use values that require float64 precision
        small = jnp.array(1e-15, dtype=jnp.float64)
        large = jnp.array(1e15, dtype=jnp.float64)

        # This calculation would lose precision in float32
        result = (large + small) - large

        # Float64 should preserve the small value
        assert jnp.isclose(result, small, rtol=1e-10)

    def test_complex128_operations(self) -> None:
        """Test that complex128 operations work correctly."""
        configure_jax()

        z = jnp.array([1.0 + 2.0j, 3.0 - 4.0j], dtype=jnp.complex128)
        assert z.dtype == jnp.complex128

        # Verify precision in complex operations
        z2 = z * z
        expected = jnp.array(
            [1.0**2 - 2.0**2 + 2j * 1.0 * 2.0, 3.0**2 - 4.0**2 + 2j * 3.0 * (-4.0)],
            dtype=jnp.complex128,
        )
        np.testing.assert_allclose(z2, expected, rtol=1e-14)


class TestDeviceDetection:
    """Tests for device detection - CPU/GPU (T031)."""

    def test_get_jax_backend_returns_string(self) -> None:
        """Test that get_jax_backend returns a valid backend string."""
        configure_jax()
        backend = get_jax_backend()
        assert isinstance(backend, str)
        assert backend in ["cpu", "gpu", "tpu"]

    def test_default_backend_available(self) -> None:
        """Test that default backend is available and usable."""
        configure_jax()
        devices = jax.devices()
        assert len(devices) > 0, "At least one device should be available"

    def test_cpu_always_available(self) -> None:
        """Test that CPU backend is always available."""
        configure_jax()
        cpu_devices = jax.devices("cpu")
        assert len(cpu_devices) > 0, "CPU should always be available"

    def test_jit_compilation_works(self) -> None:
        """Test that JIT compilation works on current device."""
        configure_jax()

        @jax.jit
        def simple_fn(x):
            return x * 2.0

        result = simple_fn(jnp.array([1.0, 2.0, 3.0]))
        expected = jnp.array([2.0, 4.0, 6.0])
        np.testing.assert_allclose(result, expected)


class TestHeadlessOperation:
    """Tests for headless operation without PyQt6 (T032)."""

    def test_core_import_no_pyqt(self) -> None:
        """Test that core can be imported without PyQt6."""
        # This test runs in current process, but core modules
        # should not import PyQt6
        import rheoQCM.core  # noqa: F401

        # If we got here without error, PyQt6 was not required
        assert True

    def test_model_import_no_gui(self) -> None:
        """Test that QCMModel can be imported without GUI."""
        from rheoQCM.core.model import QCMModel  # noqa: F401

        assert True

    def test_analysis_import_no_gui(self) -> None:
        """Test that analysis module can be imported without GUI."""
        from rheoQCM.core.analysis import batch_analyze_vmap  # noqa: F401

        assert True

    def test_physics_import_no_gui(self) -> None:
        """Test that physics module can be imported without GUI."""
        from rheoQCM.core.physics import sauerbreyf, sauerbreym  # noqa: F401

        assert True

    def test_headless_subprocess(self) -> None:
        """Test headless import in subprocess without DISPLAY."""
        code = """
import os
os.environ.pop("DISPLAY", None)
os.environ.pop("QT_QPA_PLATFORM", None)

from rheoQCM.core.model import QCMModel
from rheoQCM.core.analysis import batch_analyze_vmap
print("OK")
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env={
                **{
                    k: v
                    for k, v in subprocess.os.environ.items()
                    if k not in ("DISPLAY", "QT_QPA_PLATFORM")
                }
            },
        )
        assert result.returncode == 0, f"Headless import failed: {result.stderr}"
        assert "OK" in result.stdout


class TestBatchAnalyzeHeadless:
    """Tests for batch_analyze_vmap working without GUI (T035)."""

    def test_batch_analyze_simple(self) -> None:
        """Test that batch_analyze_vmap works for simple data."""
        configure_jax()
        from rheoQCM.core.analysis import batch_analyze_vmap

        # Create simple test data
        delfstars = jnp.array(
            [
                [-1000.0 + 100.0j, -1700.0 + 180.0j, -2500.0 + 280.0j],
                [-1100.0 + 110.0j, -1800.0 + 190.0j, -2600.0 + 290.0j],
            ]
        )

        result = batch_analyze_vmap(delfstars, harmonics=[3, 5, 7], nhcalc="357")

        # Verify result structure
        assert hasattr(result, "drho")
        assert hasattr(result, "grho_refh")
        assert hasattr(result, "phi")
        assert len(result.drho) == 2

    def test_batch_analyze_dtype_preservation(self) -> None:
        """Test that batch_analyze_vmap preserves float64."""
        configure_jax()
        from rheoQCM.core.analysis import batch_analyze_vmap

        delfstars = jnp.array(
            [
                [-1000.0 + 100.0j, -1700.0 + 180.0j],
            ],
            dtype=jnp.complex128,
        )

        result = batch_analyze_vmap(delfstars, harmonics=[3, 5], nhcalc="35")

        assert result.drho.dtype == jnp.float64
        assert result.grho_refh.dtype == jnp.float64

    def test_batch_analyze_in_subprocess(self) -> None:
        """Test batch_analyze_vmap in headless subprocess."""
        code = """
import os
os.environ.pop("DISPLAY", None)

import jax.numpy as jnp
from rheoQCM.core import configure_jax
from rheoQCM.core.analysis import batch_analyze_vmap

configure_jax()

delfstars = jnp.array([
    [-1000.0 + 100.0j, -1700.0 + 180.0j, -2500.0 + 280.0j],
])
result = batch_analyze_vmap(delfstars, harmonics=[3, 5, 7], nhcalc="357")
print(f"drho={float(result.drho[0]):.3e}")
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env={k: v for k, v in subprocess.os.environ.items() if k != "DISPLAY"},
        )
        assert result.returncode == 0, f"Batch analyze failed: {result.stderr}"
        assert "drho=" in result.stdout
