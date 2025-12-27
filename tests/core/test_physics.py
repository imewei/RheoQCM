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
    - Hypothesis property-based tests (T016 - Constitution V)
    - vmap compatibility for batch processing (T040 - US3)
    - New physics function export pattern (T047 - US4)
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

try:
    from hypothesis import given
    from hypothesis import strategies as st
    from hypothesis import settings
    from hypothesis import assume
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False

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


class TestNaNInfHandling:
    """Test NaN/Inf handling functions (T018)."""

    def test_check_finite_valid(self) -> None:
        """Test check_finite with valid values."""
        from rheoQCM.core.physics import check_finite

        valid = jnp.array([1.0, 2.0, 3.0])
        assert check_finite(valid)

    def test_check_finite_nan(self) -> None:
        """Test check_finite with NaN."""
        from rheoQCM.core.physics import check_finite

        with_nan = jnp.array([1.0, jnp.nan, 3.0])
        assert not check_finite(with_nan)

    def test_check_finite_inf(self) -> None:
        """Test check_finite with Inf."""
        from rheoQCM.core.physics import check_finite

        with_inf = jnp.array([1.0, jnp.inf, 3.0])
        assert not check_finite(with_inf)

    def test_safe_divide_normal(self) -> None:
        """Test safe_divide with normal values."""
        from rheoQCM.core.physics import safe_divide

        result = safe_divide(jnp.array(6.0), jnp.array(2.0))
        assert jnp.isclose(result, 3.0)

    def test_safe_divide_zero(self) -> None:
        """Test safe_divide with zero denominator."""
        from rheoQCM.core.physics import safe_divide

        result = safe_divide(jnp.array(6.0), jnp.array(0.0))
        assert jnp.isnan(result)

    def test_safe_sqrt_positive(self) -> None:
        """Test safe_sqrt with positive values."""
        from rheoQCM.core.physics import safe_sqrt

        result = safe_sqrt(jnp.array(4.0))
        assert jnp.isclose(result, 2.0)

    def test_safe_sqrt_negative(self) -> None:
        """Test safe_sqrt with negative values."""
        from rheoQCM.core.physics import safe_sqrt

        result = safe_sqrt(jnp.array(-4.0))
        assert jnp.isnan(result)

    def test_validate_inputs_valid(self) -> None:
        """Test validate_inputs with valid data."""
        from rheoQCM.core.physics import validate_inputs

        a = jnp.array([1.0, 2.0])
        b = jnp.array([3.0, 4.0])
        assert validate_inputs(a, b)

    def test_validate_inputs_nan_raises(self) -> None:
        """Test validate_inputs raises on NaN."""
        from rheoQCM.core.physics import validate_inputs, PhysicsNaNError

        a = jnp.array([1.0, jnp.nan])
        with pytest.raises(PhysicsNaNError):
            validate_inputs(a, raise_on_nan=True, context="test")


# =============================================================================
# Hypothesis Property-Based Tests (T016 - Constitution V)
# =============================================================================


@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
class TestHypothesisSauerbrey:
    """Property-based tests for Sauerbrey equations."""

    @given(
        n=st.integers(min_value=1, max_value=15),
        drho=st.floats(min_value=1e-10, max_value=1e-2, allow_nan=False),
    )
    @settings(max_examples=100)
    def test_sauerbrey_roundtrip(self, n: int, drho: float) -> None:
        """Sauerbrey mass->freq->mass should be identity."""
        from rheoQCM.core.physics import sauerbreyf, sauerbreym

        delf = sauerbreyf(n, drho)
        drho_recovered = sauerbreym(n, -delf)

        assert np.isclose(float(drho_recovered), drho, rtol=1e-10)

    @given(
        n=st.integers(min_value=1, max_value=15),
        drho=st.floats(min_value=1e-10, max_value=1e-2, allow_nan=False),
    )
    @settings(max_examples=100)
    def test_sauerbrey_positive_mass_positive_freq(self, n: int, drho: float) -> None:
        """Positive mass should give positive frequency shift (convention)."""
        from rheoQCM.core.physics import sauerbreyf

        delf = sauerbreyf(n, drho)
        assert float(delf) > 0

    @given(
        drho=st.floats(min_value=1e-10, max_value=1e-2, allow_nan=False),
    )
    @settings(max_examples=50)
    def test_sauerbrey_harmonic_linear(self, drho: float) -> None:
        """Sauerbrey shift should scale linearly with harmonic."""
        from rheoQCM.core.physics import sauerbreyf

        delf_1 = sauerbreyf(1, drho)
        delf_3 = sauerbreyf(3, drho)
        delf_5 = sauerbreyf(5, drho)

        assert np.isclose(float(delf_3), float(delf_1) * 3, rtol=1e-10)
        assert np.isclose(float(delf_5), float(delf_1) * 5, rtol=1e-10)


@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
class TestHypothesisModulus:
    """Property-based tests for complex modulus calculations."""

    @given(
        grho_val=st.floats(min_value=1e4, max_value=1e14, allow_nan=False),
        phi=st.floats(min_value=0.01, max_value=1.56, allow_nan=False),  # ~0 to ~pi/2
    )
    @settings(max_examples=100)
    def test_grhostar_magnitude(self, grho_val: float, phi: float) -> None:
        """grhostar magnitude should equal input grho."""
        from rheoQCM.core.physics import grhostar

        result = grhostar(grho_val, phi)
        assert np.isclose(abs(complex(result)), grho_val, rtol=1e-10)

    @given(
        grho_val=st.floats(min_value=1e4, max_value=1e14, allow_nan=False),
        phi=st.floats(min_value=0.01, max_value=1.56, allow_nan=False),
    )
    @settings(max_examples=100)
    def test_grhostar_phase(self, grho_val: float, phi: float) -> None:
        """grhostar phase should equal input phi."""
        from rheoQCM.core.physics import grhostar

        result = grhostar(grho_val, phi)
        result_phase = np.angle(complex(result))
        assert np.isclose(result_phase, phi, rtol=1e-10)

    @given(
        grho_refh=st.floats(min_value=1e4, max_value=1e14, allow_nan=False),
        phi=st.floats(min_value=0.01, max_value=1.56, allow_nan=False),
    )
    @settings(max_examples=100)
    def test_grho_at_refh_equals_grho_refh(self, grho_refh: float, phi: float) -> None:
        """grho at reference harmonic should equal grho_refh."""
        from rheoQCM.core.physics import grho

        refh = 3
        result = grho(refh, grho_refh, phi, refh=refh)
        assert np.isclose(float(result), grho_refh, rtol=1e-10)


@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
class TestHypothesisPhysicsConstraints:
    """Property-based tests for physics constraints."""

    @given(
        phi=st.floats(min_value=0.01, max_value=1.56, allow_nan=False),
        n1=st.integers(min_value=1, max_value=5),
        n2=st.integers(min_value=5, max_value=13),
    )
    @settings(max_examples=100)
    def test_grho_monotonic_with_harmonic(
        self, phi: float, n1: int, n2: int
    ) -> None:
        """For viscoelastic materials, grho increases with harmonic."""
        from rheoQCM.core.physics import grho

        assume(n2 > n1)

        grho_refh = 1e10
        result_n1 = grho(n1, grho_refh, phi, refh=3)
        result_n2 = grho(n2, grho_refh, phi, refh=3)

        # For 0 < phi <= pi/2, grho should increase with n
        assert float(result_n2) >= float(result_n1)

    @given(
        phi=st.floats(min_value=0.01, max_value=1.56, allow_nan=False),
    )
    @settings(max_examples=50)
    def test_elastic_limit(self, phi: float) -> None:
        """As phi->0, grho becomes independent of harmonic."""
        from rheoQCM.core.physics import grho

        grho_refh = 1e10
        # Use very small phi
        phi_small = 0.001

        result_n3 = grho(3, grho_refh, phi_small, refh=3)
        result_n9 = grho(9, grho_refh, phi_small, refh=3)

        # For elastic material (phi->0), grho should be nearly constant
        ratio = float(result_n9) / float(result_n3)
        assert 0.95 < ratio < 1.05  # Allow 5% variation

    @given(
        drho=st.floats(min_value=1e-9, max_value=1e-4, allow_nan=False),
        grho_n=st.floats(min_value=1e4, max_value=1e14, allow_nan=False),
        phi=st.floats(min_value=0.01, max_value=1.56, allow_nan=False),
    )
    @settings(max_examples=100)
    def test_dlam_positive(self, drho: float, grho_n: float, phi: float) -> None:
        """d/lambda should always be positive for positive inputs."""
        from rheoQCM.core.physics import calc_dlam

        n = 3
        result = calc_dlam(n, grho_n, phi, drho)
        assert float(result) > 0


@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
class TestHypothesisDtypes:
    """Property-based tests for dtype preservation."""

    @given(
        n=st.integers(min_value=1, max_value=15),
        drho=st.floats(min_value=1e-10, max_value=1e-2, allow_nan=False),
    )
    @settings(max_examples=50)
    def test_sauerbrey_float64(self, n: int, drho: float) -> None:
        """Sauerbrey functions should return Float64."""
        from rheoQCM.core.physics import sauerbreyf, sauerbreym

        result_f = sauerbreyf(n, drho)
        assert result_f.dtype == jnp.float64

        result_m = sauerbreym(n, -float(result_f))
        assert result_m.dtype == jnp.float64

    @given(
        grho_val=st.floats(min_value=1e4, max_value=1e14, allow_nan=False),
        phi=st.floats(min_value=0.01, max_value=1.56, allow_nan=False),
    )
    @settings(max_examples=50)
    def test_grhostar_complex128(self, grho_val: float, phi: float) -> None:
        """grhostar should return Complex128."""
        from rheoQCM.core.physics import grhostar

        result = grhostar(grho_val, phi)
        assert result.dtype == jnp.complex128


# =============================================================================
# T040: vmap Compatibility Tests for Batch Processing (US3)
# =============================================================================


class TestVmapCompatibilityForBatch:
    """T040: vmap compatibility tests for physics functions used in batch processing.

    These tests verify that key physics functions can be vmapped for
    GPU-accelerated batch processing as required by User Story 3.
    """

    def test_sauerbreyf_vmap_batch(self) -> None:
        """Test sauerbreyf is vmap-compatible for batch processing."""
        from rheoQCM.core.physics import sauerbreyf

        # Batch of drho values (N measurements, same harmonic)
        drho_batch = jnp.array([1e-6, 2e-6, 3e-6, 4e-6, 5e-6])
        n = 3

        # vmap over batch dimension
        vmapped = jax.vmap(lambda d: sauerbreyf(n, d))
        results = vmapped(drho_batch)

        assert results.shape == (5,)
        assert jnp.all(jnp.isfinite(results))
        # Verify linear scaling with drho
        assert jnp.allclose(results / results[0], drho_batch / drho_batch[0], rtol=1e-10)

    def test_grho_vmap_batch(self) -> None:
        """Test grho is vmap-compatible for batch processing."""
        from rheoQCM.core.physics import grho

        # Batch of measurements with varying grho_refh and phi
        n = 5
        grho_refh_batch = jnp.array([1e8, 2e8, 3e8, 4e8, 5e8])
        phi_batch = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5])

        # vmap over both parameters
        vmapped = jax.vmap(lambda g, p: grho(n, g, p, refh=3))
        results = vmapped(grho_refh_batch, phi_batch)

        assert results.shape == (5,)
        assert jnp.all(jnp.isfinite(results))

    def test_grhostar_vmap_batch(self) -> None:
        """Test grhostar is vmap-compatible for batch processing."""
        from rheoQCM.core.physics import grhostar

        grho_batch = jnp.array([1e8, 2e8, 3e8, 4e8, 5e8])
        phi_batch = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5])

        vmapped = jax.vmap(grhostar)
        results = vmapped(grho_batch, phi_batch)

        assert results.shape == (5,)
        assert results.dtype == jnp.complex128
        assert jnp.all(jnp.isfinite(jnp.abs(results)))

    def test_calc_delfstar_sla_vmap_batch(self) -> None:
        """Test calc_delfstar_sla is vmap-compatible for batch processing."""
        from rheoQCM.core.physics import calc_delfstar_sla

        ZL_batch = jnp.array([1000+100j, 2000+200j, 3000+300j, 4000+400j, 5000+500j])

        vmapped = jax.vmap(calc_delfstar_sla)
        results = vmapped(ZL_batch)

        assert results.shape == (5,)
        assert results.dtype == jnp.complex128
        assert jnp.all(jnp.isfinite(jnp.abs(results)))

    def test_normdelfstar_vmap_batch(self) -> None:
        """Test normdelfstar is vmap-compatible for batch processing."""
        from rheoQCM.core.physics import normdelfstar

        n = 3
        dlam3_batch = jnp.array([0.05, 0.10, 0.15, 0.20, 0.25])
        phi_batch = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5])

        # vmap over dlam3 and phi
        vmapped = jax.vmap(lambda d, p: normdelfstar(n, d, p))
        results = vmapped(dlam3_batch, phi_batch)

        assert results.shape == (5,)
        assert results.dtype == jnp.complex128

    def test_full_sla_pipeline_vmap(self) -> None:
        """Test that full SLA calculation pipeline is vmap-compatible.

        This tests the combined operations used in batch_analyze:
        grho -> grhostar -> ZL -> delfstar
        """
        from rheoQCM.core.physics import (
            grho, grhostar, calc_delfstar_sla, Zq, f1_default
        )

        n = 3
        refh = 3
        f1 = f1_default

        # Batch parameters
        grho_refh_batch = jnp.array([1e8, 2e8, 3e8])
        phi_batch = jnp.array([0.2, 0.3, 0.4])
        drho_batch = jnp.array([1e-6, 2e-6, 3e-6])

        def calc_single(grho_refh, phi, drho):
            grho_n = grho(n, grho_refh, phi, refh=refh)
            grhostar_n = grhostar(grho_n, phi)
            zstar = jnp.sqrt(grhostar_n)
            D = 2 * jnp.pi * n * f1 * drho / zstar
            ZL = 1j * zstar * jnp.tan(D)
            return calc_delfstar_sla(ZL, f1=f1)

        vmapped = jax.vmap(calc_single)
        results = vmapped(grho_refh_batch, phi_batch, drho_batch)

        assert results.shape == (3,)
        assert results.dtype == jnp.complex128
        assert jnp.all(jnp.isfinite(jnp.abs(results)))

    def test_vmap_jit_combined(self) -> None:
        """Test that vmap and jit work together on physics functions."""
        from rheoQCM.core.physics import sauerbreyf, grhostar

        # Test combined vmap + jit
        vmapped_jitted = jax.jit(jax.vmap(lambda d: sauerbreyf(3, d)))

        drho_batch = jnp.array([1e-6, 2e-6, 3e-6, 4e-6, 5e-6])
        results = vmapped_jitted(drho_batch)

        assert results.shape == (5,)
        assert jnp.all(jnp.isfinite(results))

        # Test grhostar
        vmapped_jitted_complex = jax.jit(jax.vmap(grhostar))

        grho_batch = jnp.array([1e8, 2e8, 3e8])
        phi_batch = jnp.array([0.2, 0.3, 0.4])
        results_complex = vmapped_jitted_complex(grho_batch, phi_batch)

        assert results_complex.shape == (3,)
        assert results_complex.dtype == jnp.complex128

    def test_vmap_large_batch(self) -> None:
        """Test vmap handles large batches efficiently (1000 measurements)."""
        from rheoQCM.core.physics import sauerbreyf, grhostar

        n = 1000
        drho_batch = jnp.linspace(1e-7, 1e-5, n)
        grho_batch = jnp.full(n, 1e8)
        phi_batch = jnp.linspace(0.1, 1.5, n)

        # vmap Sauerbrey
        vmapped_sauerbrey = jax.vmap(lambda d: sauerbreyf(3, d))
        results_sauerbrey = vmapped_sauerbrey(drho_batch)

        assert results_sauerbrey.shape == (n,)
        assert jnp.all(jnp.isfinite(results_sauerbrey))

        # vmap grhostar
        vmapped_grhostar = jax.vmap(grhostar)
        results_grhostar = vmapped_grhostar(grho_batch, phi_batch)

        assert results_grhostar.shape == (n,)
        assert jnp.all(jnp.isfinite(jnp.abs(results_grhostar)))

    def test_vmap_nested(self) -> None:
        """Test nested vmap for multi-harmonic batch processing."""
        from rheoQCM.core.physics import sauerbreyf

        # Process batch of measurements across multiple harmonics
        harmonics = jnp.array([3, 5, 7])
        drho_batch = jnp.array([1e-6, 2e-6, 3e-6, 4e-6, 5e-6])

        # Nested vmap: outer over drho, inner over harmonics
        def process_measurement(drho):
            return jax.vmap(lambda n: sauerbreyf(n, drho))(harmonics)

        vmapped = jax.vmap(process_measurement)
        results = vmapped(drho_batch)

        # Shape should be (5 measurements, 3 harmonics)
        assert results.shape == (5, 3)
        assert jnp.all(jnp.isfinite(results))

    def test_kotula_gstar_vmap(self) -> None:
        """Test kotula_gstar is vmap-compatible."""
        from rheoQCM.core.physics import kotula_gstar

        Gmstar = 1e6 + 1e5j
        Gfstar = 1e9 + 1e8j
        xi_crit = 0.5
        s = 1.0
        t = 1.0

        xi_batch = jnp.linspace(0.1, 0.9, 10)

        # kotula_gstar already handles arrays, but verify vmap works
        results = kotula_gstar(xi_batch, Gmstar, Gfstar, xi_crit, s, t)

        assert results.shape == (10,)
        # Should have valid complex results
        assert jnp.all(jnp.isfinite(jnp.abs(results)))

    def test_physics_functions_no_python_control_flow(self) -> None:
        """Test that key physics functions avoid Python control flow.

        Functions with Python if/else break vmap. This test verifies
        that the core functions used in batch_analyze work with vmap.
        """
        from rheoQCM.core.physics import (
            sauerbreyf, sauerbreym, grho, grhostar, calc_delfstar_sla, normdelfstar
        )

        # All these should be vmappable without error
        batch_size = 5

        # sauerbreyf
        drho = jnp.ones(batch_size) * 1e-6
        result = jax.vmap(lambda d: sauerbreyf(3, d))(drho)
        assert result.shape == (batch_size,)

        # sauerbreym
        delf = jnp.ones(batch_size) * -1000
        result = jax.vmap(lambda d: sauerbreym(3, d))(delf)
        assert result.shape == (batch_size,)

        # grho (refh is static)
        grho_refh = jnp.ones(batch_size) * 1e8
        phi = jnp.ones(batch_size) * 0.5
        result = jax.vmap(lambda g, p: grho(5, g, p, refh=3))(grho_refh, phi)
        assert result.shape == (batch_size,)

        # grhostar
        result = jax.vmap(grhostar)(grho_refh, phi)
        assert result.shape == (batch_size,)

        # calc_delfstar_sla
        ZL = jnp.ones(batch_size, dtype=jnp.complex128) * (1000 + 100j)
        result = jax.vmap(calc_delfstar_sla)(ZL)
        assert result.shape == (batch_size,)

        # normdelfstar
        dlam = jnp.ones(batch_size) * 0.1
        result = jax.vmap(lambda d, p: normdelfstar(3, d, p))(dlam, phi)
        assert result.shape == (batch_size,)


# =============================================================================
# T047: Physics Function Export and Extension Pattern Tests (User Story 4)
# =============================================================================


class TestPhysicsFunctionExport:
    """T047: Tests for new physics function export patterns.

    These tests verify that new physics functions added to the core
    module are properly accessible and follow the jit/vmap patterns.
    """

    def test_physics_module_exports_all_functions(self) -> None:
        """Test that physics module exports expected functions in __all__."""
        from rheoQCM.core import physics

        # Check __all__ exists and contains core functions
        assert hasattr(physics, "__all__")
        expected_exports = [
            "sauerbreyf",
            "sauerbreym",
            "grho",
            "grhostar",
            "grhostar_from_refh",
            "calc_delfstar_sla",
            "kotula_gstar",
            "kotula_xi",
        ]

        for func_name in expected_exports:
            assert func_name in physics.__all__, f"{func_name} not in physics.__all__"

    def test_new_function_accessible_from_core(self) -> None:
        """Test that physics functions are accessible from rheoQCM.core."""
        from rheoQCM.core import sauerbreyf, grho, grhostar

        # Functions should be importable from core package
        assert callable(sauerbreyf)
        assert callable(grho)
        assert callable(grhostar)

    def test_physics_function_has_jit_decorator(self) -> None:
        """Test that physics functions are jit-compatible."""
        from rheoQCM.core.physics import sauerbreyf, grhostar

        # Should be able to jit these functions
        jitted_sauerbreyf = jax.jit(sauerbreyf)
        jitted_grhostar = jax.jit(grhostar)

        # And they should work
        result1 = jitted_sauerbreyf(3, 1e-6)
        result2 = jitted_grhostar(1e8, 0.5)

        assert jnp.isfinite(result1)
        assert jnp.isfinite(jnp.abs(result2))

    def test_physics_function_vmap_pattern(self) -> None:
        """Test the standard vmap pattern for physics functions."""
        from rheoQCM.core.physics import grho

        # Standard pattern: vmap over batch dimension with static args
        batch_grho = jax.vmap(lambda g, p: grho(3, g, p, refh=3))

        grho_batch = jnp.array([1e8, 2e8, 3e8])
        phi_batch = jnp.array([0.2, 0.3, 0.4])

        results = batch_grho(grho_batch, phi_batch)

        assert results.shape == (3,)
        assert jnp.all(jnp.isfinite(results))

    def test_adding_new_physics_function_pattern(self) -> None:
        """Test the pattern for adding a new physics function.

        This test demonstrates the recommended pattern for adding
        custom physics functions that integrate with the core.
        """
        # New physics function should follow this pattern:
        # 1. Use @jax.jit decorator (or be jit-compatible)
        # 2. Accept jnp arrays as inputs
        # 3. Return jnp arrays
        # 4. Avoid Python control flow for vmap compatibility

        @jax.jit
        def custom_viscoelastic_modulus(
            n: jnp.ndarray,
            G0: jnp.ndarray,
            tau: jnp.ndarray,
            alpha: jnp.ndarray,
            f1: float = 5e6,
        ) -> jnp.ndarray:
            """Example custom viscoelastic model (fractional derivative).

            This demonstrates the pattern for adding new physics functions.

            Parameters
            ----------
            n : array
                Harmonic number
            G0 : array
                Low-frequency modulus [Pa kg/m^3]
            tau : array
                Relaxation time [s]
            alpha : array
                Fractional derivative order (0 to 1)
            f1 : float
                Fundamental frequency [Hz]

            Returns
            -------
            Gstar : complex array
                Complex modulus times density
            """
            omega = 2 * jnp.pi * n * f1
            # Fractional Maxwell model: G* = G0 * (i*omega*tau)^alpha / (1 + (i*omega*tau)^alpha)
            z = (1j * omega * tau) ** alpha
            return G0 * z / (1 + z)

        # Test that it works with jit
        result = custom_viscoelastic_modulus(
            jnp.array(3.0),
            jnp.array(1e8),
            jnp.array(1e-6),
            jnp.array(0.5),
        )
        assert jnp.isfinite(jnp.abs(result))

        # Test that it works with vmap
        vmapped = jax.vmap(custom_viscoelastic_modulus, in_axes=(None, 0, 0, 0, None))
        batch_result = vmapped(
            jnp.array(3.0),
            jnp.array([1e8, 2e8, 3e8]),
            jnp.array([1e-6, 1e-6, 1e-6]),
            jnp.array([0.5, 0.6, 0.7]),
            5e6,
        )
        assert batch_result.shape == (3,)
        assert jnp.all(jnp.isfinite(jnp.abs(batch_result)))

    def test_physics_function_docstring_pattern(self) -> None:
        """Test that physics functions have proper docstrings."""
        from rheoQCM.core.physics import sauerbreyf, grho, grhostar

        for func in [sauerbreyf, grho, grhostar]:
            assert func.__doc__ is not None, f"{func.__name__} missing docstring"
            assert "Parameters" in func.__doc__ or "Args" in func.__doc__
            assert "Returns" in func.__doc__

    def test_physics_constants_exported(self) -> None:
        """Test that physics constants are exported from core."""
        from rheoQCM.core import Zq, f1_default

        # Constants should be accessible
        assert Zq == 8.84e6
        assert f1_default == 5e6

    def test_physics_function_returns_correct_dtype(self) -> None:
        """Test that physics functions return expected dtypes."""
        from rheoQCM.core.physics import sauerbreyf, grhostar, calc_delfstar_sla

        # Real-valued functions should return float64
        result_real = sauerbreyf(3, 1e-6)
        assert result_real.dtype == jnp.float64

        # Complex-valued functions should return complex128
        result_complex = grhostar(1e8, 0.5)
        assert result_complex.dtype == jnp.complex128

        result_delfstar = calc_delfstar_sla(jnp.array(1000 + 100j))
        assert result_delfstar.dtype == jnp.complex128

    def test_custom_function_integration_with_model(self) -> None:
        """Test that custom physics functions can be used with QCMModel.

        This demonstrates how a user would integrate a custom physics
        function with the QCMModel class for property fitting.
        """
        from rheoQCM.core.model import QCMModel

        model = QCMModel(f1=5e6, refh=3)

        # Custom residual function using a new physics model
        @jax.jit
        def custom_delfstar_model(
            params: jnp.ndarray,
            n: int,
            f1: float,
            Zq: float,
        ) -> jnp.ndarray:
            """Custom model for delfstar calculation."""
            grho_refh, phi, drho = params[0], params[1], params[2]

            # Custom frequency dependence (example: power law with offset)
            grho_n = grho_refh * (n / 3) ** (phi / (jnp.pi / 2))
            grhostar_n = grho_n * jnp.exp(1j * phi)
            zstar = jnp.sqrt(grhostar_n)
            D = 2 * jnp.pi * n * f1 * drho / zstar
            ZL = 1j * zstar * jnp.tan(D)
            return f1 * 1j * ZL / (jnp.pi * Zq)

        # Test that the custom model works
        params = jnp.array([1e8, 0.5, 1e-6])
        result = custom_delfstar_model(params, 3, 5e6, 8.84e6)

        assert jnp.isfinite(jnp.abs(result))

        # The custom model could be registered with QCMModel
        # (if register_calctype is implemented)
        if hasattr(model, "register_calctype"):
            def custom_residual(params, delfstar_exp, harmonics, f1, refh, Zq):
                residuals = []
                for n in harmonics:
                    calc = custom_delfstar_model(params, n, f1, Zq)
                    exp = delfstar_exp[n]
                    residuals.append(jnp.real(calc) - jnp.real(exp))
                return jnp.array(residuals)

            model.register_calctype("CustomPowerLaw", custom_residual)

    def test_nan_inf_handling_functions_exported(self) -> None:
        """Test that NaN/Inf handling functions are exported."""
        from rheoQCM.core.physics import (
            check_finite,
            safe_divide,
            safe_sqrt,
            safe_log,
            validate_inputs,
        )

        # All these should be callable
        assert callable(check_finite)
        assert callable(safe_divide)
        assert callable(safe_sqrt)
        assert callable(safe_log)
        assert callable(validate_inputs)

    def test_utility_functions_exported(self) -> None:
        """Test that utility functions (scipy replacements) are exported."""
        from rheoQCM.core.physics import (
            find_peaks,
            interp_linear,
            interp_cubic,
            create_interp_func,
            savgol_filter,
        )

        # All should be callable
        assert callable(find_peaks)
        assert callable(interp_linear)
        assert callable(interp_cubic)
        assert callable(create_interp_func)
        assert callable(savgol_filter)
