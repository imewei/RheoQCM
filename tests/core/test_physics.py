"""
Tests for rheoQCM.core.physics module.

These tests verify the JAX-based physics calculations against known values
and ensure JIT compilation and vmap vectorization work correctly.

Test coverage:
    - sauerbreyf calculation accuracy against known values
    - sauerbreym calculation accuracy
    - grho and grhostar complex modulus calculations
    - grho_from_dlam inverse calculation
    - kotula_gstar root finding with complex numbers
    - jax.jit compilation of all physics functions
    - jax.vmap vectorization across harmonics
    - Float64 precision maintenance
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rheoQCM.core import configure_jax

# Ensure JAX is configured for Float64 before running tests
configure_jax()


pytestmark = pytest.mark.physics


class TestSauerbreyEquations:
    """Test Sauerbrey frequency and mass calculations."""

    def test_sauerbreyf_known_values(self) -> None:
        """Test sauerbreyf calculation against known values.

        The Sauerbrey equation: delf = 2 * n * f1^2 * drho / Zq
        For n=1, f1=5e6 Hz, drho=1e-6 kg/m^2, Zq=8.84e6:
        delf = 2 * 1 * (5e6)^2 * 1e-6 / 8.84e6 = 5.656 Hz (approximately)
        """
        from rheoQCM.core.physics import sauerbreyf, Zq, f1_default

        # Test with fundamental harmonic
        n = 1
        drho = 1e-6  # kg/m^2
        f1 = f1_default

        result = sauerbreyf(n, drho, f1=f1)
        expected = 2 * n * f1**2 * drho / Zq

        assert jnp.isclose(result, expected, rtol=1e-10)
        assert result.dtype == jnp.float64

    def test_sauerbreyf_harmonic_scaling(self) -> None:
        """Test that sauerbreyf scales linearly with harmonic number."""
        from rheoQCM.core.physics import sauerbreyf

        drho = 1e-6  # kg/m^2

        delf_1 = sauerbreyf(1, drho)
        delf_3 = sauerbreyf(3, drho)
        delf_5 = sauerbreyf(5, drho)

        # Should scale linearly with harmonic number
        assert jnp.isclose(delf_3 / delf_1, 3.0, rtol=1e-10)
        assert jnp.isclose(delf_5 / delf_1, 5.0, rtol=1e-10)

    def test_sauerbreym_known_values(self) -> None:
        """Test sauerbreym calculation (inverse of sauerbreyf).

        Note: sauerbreyf returns positive frequency for positive mass (convention),
        while experimentally frequency decreases with mass.
        sauerbreym converts negative frequency shift to positive mass.
        """
        from rheoQCM.core.physics import sauerbreyf, sauerbreym

        # Test round-trip: drho -> delf -> negate -> drho
        # sauerbreyf gives positive delf for positive drho
        # sauerbreym expects negative delf for positive drho
        n = 3
        drho_original = 2.5e-6  # kg/m^2

        delf = sauerbreyf(n, drho_original)
        # Negate because sauerbreym expects experimental convention
        drho_recovered = sauerbreym(n, -delf)

        assert jnp.isclose(drho_recovered, drho_original, rtol=1e-10)

    def test_sauerbreym_negative_frequency(self) -> None:
        """Test sauerbreym with negative frequency shift (mass increase)."""
        from rheoQCM.core.physics import sauerbreym

        n = 3
        delf = -1000.0  # Hz (negative = mass increase)

        drho = sauerbreym(n, delf)

        # Negative frequency shift should give positive mass
        assert drho > 0
        assert drho.dtype == jnp.float64


class TestComplexModulus:
    """Test complex modulus calculations (grho, grhostar)."""

    def test_grho_power_law(self) -> None:
        """Test grho calculation with power law frequency dependence."""
        from rheoQCM.core.physics import grho

        grho_refh = 1e8  # Pa kg/m^3
        phi = jnp.pi / 4  # 45 degrees
        refh = 3
        n = 5

        result = grho(n, grho_refh, phi, refh=refh)

        # grho_n = grho_refh * (n/refh)^(phi / (pi/2))
        expected = grho_refh * (n / refh) ** (phi / (jnp.pi / 2))

        assert jnp.isclose(result, expected, rtol=1e-10)
        assert result.dtype == jnp.float64

    def test_grho_at_reference_harmonic(self) -> None:
        """Test that grho equals grho_refh at reference harmonic."""
        from rheoQCM.core.physics import grho

        grho_refh = 1e8
        phi = jnp.pi / 4
        refh = 3

        result = grho(refh, grho_refh, phi, refh=refh)

        assert jnp.isclose(result, grho_refh, rtol=1e-10)

    def test_grhostar_complex(self) -> None:
        """Test grhostar returns correct complex modulus."""
        from rheoQCM.core.physics import grhostar

        grho_val = 1e8  # Pa kg/m^3
        phi = jnp.pi / 4  # 45 degrees

        result = grhostar(grho_val, phi)

        # grhostar = grho * exp(i * phi)
        expected = grho_val * jnp.exp(1j * phi)

        assert jnp.isclose(result, expected, rtol=1e-10)
        assert result.dtype == jnp.complex128

    def test_grhostar_magnitude_and_phase(self) -> None:
        """Test that grhostar has correct magnitude and phase."""
        from rheoQCM.core.physics import grhostar

        grho_val = 1e8
        phi = jnp.pi / 3  # 60 degrees

        result = grhostar(grho_val, phi)

        assert jnp.isclose(jnp.abs(result), grho_val, rtol=1e-10)
        assert jnp.isclose(jnp.angle(result), phi, rtol=1e-10)


class TestGrhoFromDlam:
    """Test grho_from_dlam inverse calculation."""

    def test_grho_from_dlam_basic(self) -> None:
        """Test grho_from_dlam calculation."""
        from rheoQCM.core.physics import grho_from_dlam

        n = 3
        drho = 1e-6  # kg/m^2
        dlam_refh = 0.1  # d/lambda at reference harmonic
        phi = jnp.pi / 6  # 30 degrees

        result = grho_from_dlam(n, drho, dlam_refh, phi)

        # grho = (drho * n * f1 * cos(phi/2) / dlam_refh)^2
        f1 = 5e6  # default fundamental frequency
        expected = (drho * n * f1 * jnp.cos(phi / 2) / dlam_refh) ** 2

        assert jnp.isclose(result, expected, rtol=1e-10)
        assert result.dtype == jnp.float64

    def test_grho_from_dlam_roundtrip(self) -> None:
        """Test round-trip conversion: grho -> dlam -> grho."""
        from rheoQCM.core.physics import grho_from_dlam, calc_dlam, grho

        grho_refh = 1e10
        phi = jnp.pi / 4
        drho = 1e-6
        refh = 3
        n = 3

        # Calculate dlam from grho
        grho_n = grho(n, grho_refh, phi, refh=refh)
        dlam = calc_dlam(n, grho_n, phi, drho)

        # Calculate grho back from dlam
        grho_recovered = grho_from_dlam(n, drho, dlam, phi)

        assert jnp.isclose(grho_recovered, grho_n, rtol=1e-8)


class TestKotulaModel:
    """Test Kotula model root finding."""

    def test_kotula_gstar_basic(self) -> None:
        """Test kotula_gstar with basic parameters."""
        from rheoQCM.core.physics import kotula_gstar

        xi = 0.3  # filler fraction
        Gmstar = 1e6 + 1e5j  # matrix modulus (complex)
        Gfstar = 1e9 + 1e8j  # filler modulus (complex)
        xi_crit = 0.5
        s = 1.0
        t = 1.0

        result = kotula_gstar(xi, Gmstar, Gfstar, xi_crit, s, t)

        # Result should be complex
        assert jnp.iscomplexobj(result)
        # Verify the solution satisfies the Kotula equation (residual is small)
        from rheoQCM.core.physics import _kotula_equation

        residual = _kotula_equation(result, xi, Gmstar, Gfstar, xi_crit, s, t)
        assert jnp.abs(residual) < 1e-6, f"Residual too large: {jnp.abs(residual)}"

    def test_kotula_gstar_limit_cases(self) -> None:
        """Test kotula_gstar at limit cases (xi=0 and xi~1)."""
        from rheoQCM.core.physics import kotula_gstar

        Gmstar = 1e6 + 1e5j
        Gfstar = 1e9 + 1e8j
        xi_crit = 0.5
        s = 1.0
        t = 1.0

        # At xi=0, result should approach matrix modulus
        result_zero = kotula_gstar(0.001, Gmstar, Gfstar, xi_crit, s, t)
        assert jnp.abs(result_zero - Gmstar) / jnp.abs(Gmstar) < 0.1

        # At xi~1, result should approach filler modulus
        result_one = kotula_gstar(0.999, Gmstar, Gfstar, xi_crit, s, t)
        assert jnp.abs(result_one - Gfstar) / jnp.abs(Gfstar) < 0.1

    def test_kotula_xi_inverse(self) -> None:
        """Test kotula_xi gives back original xi from gstar."""
        from rheoQCM.core.physics import kotula_gstar, kotula_xi

        xi_original = 0.4
        Gmstar = 1e6 + 1e5j
        Gfstar = 1e9 + 1e8j
        xi_crit = 0.5
        s = 1.0
        t = 1.0

        # Get gstar from xi
        gstar = kotula_gstar(xi_original, Gmstar, Gfstar, xi_crit, s, t)

        # Get xi back from gstar
        xi_recovered = kotula_xi(gstar, Gmstar, Gfstar, xi_crit, s, t)

        # Should be close to original (real part)
        assert jnp.isclose(jnp.real(xi_recovered), xi_original, rtol=0.1)


class TestJITCompilation:
    """Test that all physics functions compile with jax.jit."""

    def test_sauerbreyf_jit(self) -> None:
        """Test sauerbreyf JIT compilation."""
        from rheoQCM.core.physics import sauerbreyf

        jitted_fn = jax.jit(sauerbreyf)

        result = jitted_fn(3, 1e-6)
        assert jnp.isfinite(result)

    def test_sauerbreym_jit(self) -> None:
        """Test sauerbreym JIT compilation."""
        from rheoQCM.core.physics import sauerbreym

        jitted_fn = jax.jit(sauerbreym)

        result = jitted_fn(3, -1000.0)
        assert jnp.isfinite(result)

    def test_grho_jit(self) -> None:
        """Test grho JIT compilation."""
        from rheoQCM.core.physics import grho

        jitted_fn = jax.jit(grho, static_argnames=["refh"])

        result = jitted_fn(3, 1e8, jnp.pi / 4, refh=3)
        assert jnp.isfinite(result)

    def test_grhostar_jit(self) -> None:
        """Test grhostar JIT compilation."""
        from rheoQCM.core.physics import grhostar

        jitted_fn = jax.jit(grhostar)

        result = jitted_fn(1e8, jnp.pi / 4)
        assert jnp.isfinite(jnp.abs(result))

    def test_grho_from_dlam_jit(self) -> None:
        """Test grho_from_dlam JIT compilation."""
        from rheoQCM.core.physics import grho_from_dlam

        jitted_fn = jax.jit(grho_from_dlam)

        result = jitted_fn(3, 1e-6, 0.1, jnp.pi / 4)
        assert jnp.isfinite(result)


class TestVmapVectorization:
    """Test that physics functions vectorize with jax.vmap."""

    def test_sauerbreyf_vmap(self) -> None:
        """Test sauerbreyf vectorization across harmonics."""
        from rheoQCM.core.physics import sauerbreyf

        harmonics = jnp.array([1, 3, 5, 7, 9])
        drho = 1e-6

        # Vectorize over harmonics
        vmapped_fn = jax.vmap(sauerbreyf, in_axes=(0, None))
        results = vmapped_fn(harmonics, drho)

        assert results.shape == (5,)
        # Check scaling with harmonic
        assert jnp.allclose(results / results[0], harmonics, rtol=1e-10)

    def test_sauerbreym_vmap(self) -> None:
        """Test sauerbreym vectorization across harmonics."""
        from rheoQCM.core.physics import sauerbreym

        harmonics = jnp.array([1, 3, 5, 7, 9])
        delf = -1000.0

        vmapped_fn = jax.vmap(sauerbreym, in_axes=(0, None))
        results = vmapped_fn(harmonics, delf)

        assert results.shape == (5,)
        assert jnp.all(results > 0)

    def test_grhostar_vmap(self) -> None:
        """Test grhostar vectorization across phi values."""
        from rheoQCM.core.physics import grhostar

        grho_val = 1e8
        phi_values = jnp.array([0.0, jnp.pi / 6, jnp.pi / 4, jnp.pi / 3, jnp.pi / 2])

        vmapped_fn = jax.vmap(grhostar, in_axes=(None, 0))
        results = vmapped_fn(grho_val, phi_values)

        assert results.shape == (5,)
        # All should have same magnitude
        assert jnp.allclose(jnp.abs(results), grho_val, rtol=1e-10)
        # Phases should match input
        assert jnp.allclose(jnp.angle(results), phi_values, rtol=1e-10)


class TestFloat64Precision:
    """Test that Float64 precision is maintained."""

    def test_sauerbreyf_float64(self) -> None:
        """Test sauerbreyf maintains Float64 precision."""
        from rheoQCM.core.physics import sauerbreyf

        result = sauerbreyf(3, 1e-6)

        assert result.dtype == jnp.float64

    def test_sauerbreym_float64(self) -> None:
        """Test sauerbreym maintains Float64 precision."""
        from rheoQCM.core.physics import sauerbreym

        result = sauerbreym(3, -1000.0)

        assert result.dtype == jnp.float64

    def test_grho_float64(self) -> None:
        """Test grho maintains Float64 precision."""
        from rheoQCM.core.physics import grho

        result = grho(3, 1e8, jnp.pi / 4, refh=3)

        assert result.dtype == jnp.float64

    def test_grhostar_complex128(self) -> None:
        """Test grhostar returns complex128 (two Float64)."""
        from rheoQCM.core.physics import grhostar

        result = grhostar(1e8, jnp.pi / 4)

        assert result.dtype == jnp.complex128

    def test_small_differences_preserved(self) -> None:
        """Test that small differences are preserved with Float64."""
        from rheoQCM.core.physics import sauerbreyf

        # Small mass difference that would be lost with Float32
        drho1 = 1e-6
        drho2 = 1e-6 + 1e-15  # Very small difference

        delf1 = sauerbreyf(3, drho1)
        delf2 = sauerbreyf(3, drho2)

        # The difference should be preserved
        assert delf2 > delf1
        assert (delf2 - delf1) > 0


class TestSLAEquations:
    """Test Small Load Approximation (SLA) equations."""

    def test_calc_delfstar_sla_basic(self) -> None:
        """Test basic SLA frequency shift calculation."""
        from rheoQCM.core.physics import calc_delfstar_sla, Zq

        ZL = 1000.0 + 500.0j  # Load impedance (complex)
        f1 = 5e6

        result = calc_delfstar_sla(ZL, f1=f1)

        # delfstar_sla = f1 * 1j * ZL / (pi * Zq)
        expected = f1 * 1j * ZL / (jnp.pi * Zq)

        assert jnp.isclose(result, expected, rtol=1e-10)
        assert result.dtype == jnp.complex128

    def test_calc_delfstar_sla_pure_mass(self) -> None:
        """Test SLA with pure mass loading.

        For pure mass loading (thin rigid film), ZL is purely imaginary:
        ZL = i * omega * drho = i * 2 * pi * f * drho

        delfstar = f1 * i * (i * Z) / (pi * Zq) = -f1 * Z / (pi * Zq)

        So for purely imaginary ZL (positive), we get purely real negative delfstar.
        """
        from rheoQCM.core.physics import calc_delfstar_sla

        # Pure mass loading: ZL is purely imaginary and positive
        ZL = 1000.0j

        result = calc_delfstar_sla(ZL)

        # For pure mass (imaginary ZL), result should be purely real and negative
        # delfstar = f1 * i * (i*1000) / (pi * Zq) = -f1 * 1000 / (pi * Zq) < 0
        assert jnp.real(result) < 0, "Real part should be negative for mass loading"
        assert jnp.abs(jnp.imag(result)) < 1e-10, "Imaginary part should be ~0"


class TestUtilityFunctions:
    """Test scipy utility replacements."""

    def test_find_peaks_basic(self) -> None:
        """Test find_peaks implementation."""
        from rheoQCM.core.physics import find_peaks

        # Simple test data with clear peaks
        data = jnp.array([0.0, 1.0, 0.0, 2.0, 0.0, 1.5, 0.0])

        peak_indices = find_peaks(data)

        # Peaks at indices 1, 3, 5
        expected = jnp.array([1, 3, 5])
        # Convert to numpy for comparison since find_peaks may return different shape
        np.testing.assert_array_equal(np.array(peak_indices), np.array(expected))

    def test_interp_linear(self) -> None:
        """Test linear interpolation using jax.numpy.interp."""
        from rheoQCM.core.physics import interp_linear

        x = jnp.array([0.0, 1.0, 2.0, 3.0])
        y = jnp.array([0.0, 2.0, 4.0, 6.0])
        xnew = jnp.array([0.5, 1.5, 2.5])

        result = interp_linear(xnew, x, y)
        expected = jnp.array([1.0, 3.0, 5.0])

        assert jnp.allclose(result, expected, rtol=1e-10)

    def test_savgol_convolve(self) -> None:
        """Test Savitzky-Golay filter using convolution."""
        from rheoQCM.core.physics import savgol_filter

        # Noisy data
        x = jnp.linspace(0, 10, 100)
        y_true = jnp.sin(x)
        # Add some noise
        key = jax.random.PRNGKey(42)
        noise = jax.random.normal(key, shape=y_true.shape) * 0.1
        y_noisy = y_true + noise

        # Apply filter
        y_filtered = savgol_filter(y_noisy, window_length=11, polyorder=3)

        # Filtered data should be closer to true data than noisy data
        # (in central region, avoiding edge effects)
        center = slice(20, 80)
        error_noisy = jnp.mean(jnp.abs(y_noisy[center] - y_true[center]))
        error_filtered = jnp.mean(jnp.abs(y_filtered[center] - y_true[center]))

        assert error_filtered < error_noisy
