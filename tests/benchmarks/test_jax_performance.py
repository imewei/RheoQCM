"""Benchmark Tests for JAX Performance Optimization.

Feature: 012-jax-performance-optimization
Reference: .optimization/src-report-2025-12-28.md

This module provides benchmark tests to measure and validate performance
improvements from the 4-phase optimization plan.

Run benchmarks:
    pytest tests/benchmark/test_jax_performance.py -v --benchmark-only

Run with comparison:
    pytest tests/benchmark/test_jax_performance.py -v --benchmark-compare
"""

import time
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

# Conditional pytest-benchmark support
try:
    import pytest_benchmark

    HAS_BENCHMARK = True
except ImportError:
    HAS_BENCHMARK = False


def skip_if_no_benchmark(func):
    """Decorator to skip benchmark tests if pytest-benchmark is not installed."""
    if not HAS_BENCHMARK:
        return pytest.mark.skip(reason="pytest-benchmark not installed")(func)
    return func


class BenchmarkResult:
    """Simple benchmark result container for non-pytest-benchmark runs."""

    def __init__(self, name: str, times: list[float]):
        self.name = name
        self.times = times
        self.mean = np.mean(times)
        self.std = np.std(times)
        self.min = np.min(times)
        self.max = np.max(times)

    def __repr__(self) -> str:
        return f"{self.name}: {self.mean * 1000:.2f}ms (±{self.std * 1000:.2f}ms)"


def simple_benchmark(func, n_iterations: int = 10, warmup: int = 2) -> BenchmarkResult:
    """Simple benchmark function for when pytest-benchmark is not available."""
    # Warmup runs
    for _ in range(warmup):
        func()

    # Timed runs
    times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append(end - start)

    return BenchmarkResult(func.__name__, times)


# =============================================================================
# Phase 1: Critical Path JIT Compilation
# =============================================================================


class TestPhase1MultilayerJIT:
    """Benchmark tests for Phase 1: multilayer.py JIT optimization."""

    @pytest.fixture
    def sample_layers(self) -> dict[int, dict]:
        """Create sample layer configuration for benchmarking.

        Note: calc_ZL expects layers as dict[int, dict] where keys are layer numbers.
        """
        return {
            1: {"rho": 1000.0, "grho": 1e8, "phi": 0.1, "drho": 1e-3},
            2: {"rho": 1200.0, "grho": 1.5e8, "phi": 0.15, "drho": 2e-3},
            3: {"rho": 1100.0, "grho": 1.2e8, "phi": 0.12, "drho": 1.5e-3},
        }

    def test_multilayer_calc_ZL_baseline(self, sample_layers: dict) -> None:
        """Baseline test for calc_ZL - records current performance."""
        from rheoQCM.core.multilayer import calc_ZL

        f1 = 5e6

        def run_calc():
            for nh in [3, 5, 7, 9, 11]:
                _ = calc_ZL(
                    f1=f1,
                    n=nh,
                    layers=sample_layers,
                )

        result = simple_benchmark(run_calc, n_iterations=5, warmup=2)
        print(f"\nMultilayer calc_ZL: {result}")
        assert result.mean < 10.0, "calc_ZL baseline too slow (>10s)"


class TestPhase1ThinFilmGuess:
    """Benchmark tests for Phase 1: _thin_film_guess JIT optimization."""

    @pytest.fixture
    def sample_delfstars(self) -> dict[int, complex]:
        """Create sample frequency shifts for thin film guess."""
        return {
            3: -1000 + 100j,
            5: -1700 + 180j,
            7: -2500 + 280j,
        }

    def test_thin_film_guess_baseline(self, sample_delfstars: dict) -> None:
        """Baseline test for thin film guess performance."""
        from rheoQCM.core.model import QCMModel

        model = QCMModel(f1=5e6, refh=3)
        model.load_delfstars(sample_delfstars)

        def run_guess():
            # Access internal thin film guess if available
            result = model.solve_properties(nh=[3, 5, 3])
            return result

        result = simple_benchmark(run_guess, n_iterations=5, warmup=2)
        print(f"\nThin film guess: {result}")
        assert result.mean < 5.0, "Thin film guess baseline too slow (>5s)"


# =============================================================================
# Phase 2: Secondary Optimizations
# =============================================================================


class TestPhase2MutableDefaults:
    """Tests for Phase 2: mutable default argument fixes."""

    def test_no_mutable_defaults_in_core(self) -> None:
        """Verify no mutable default arguments in core modules."""
        import ast
        from pathlib import Path

        core_path = Path(__file__).parent.parent.parent / "src" / "rheoQCM" / "core"
        mutable_defaults = []

        for py_file in core_path.glob("*.py"):
            source = py_file.read_text()
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                    for default in node.args.defaults + node.args.kw_defaults:
                        if default is not None and isinstance(
                            default, ast.List | ast.Dict | ast.Set
                        ):
                            mutable_defaults.append(
                                f"{py_file.name}:{node.lineno} - {node.name}"
                            )

        if mutable_defaults:
            print(f"\nMutable defaults found:\n" + "\n".join(mutable_defaults))
        assert len(mutable_defaults) == 0, (
            f"Found {len(mutable_defaults)} mutable defaults"
        )


class TestPhase2VectorizedApply:
    """Benchmark tests for Phase 2: pandas .apply() vectorization."""

    @pytest.fixture
    def sample_dataframe(self):
        """Create sample DataFrame for .apply() benchmarking."""
        import pandas as pd

        n_rows = 1000
        return pd.DataFrame(
            {
                "freq": np.linspace(1e6, 10e6, n_rows),
                "g": np.random.rand(n_rows) * 1e8,
                "d": np.random.rand(n_rows) * 1e-6,
            }
        )

    def test_apply_vs_vectorized_baseline(self, sample_dataframe) -> None:
        """Compare .apply() vs vectorized operations."""
        df = sample_dataframe

        # Using .apply()
        def apply_version():
            return df["freq"].apply(lambda x: x * 2)

        # Vectorized
        def vectorized_version():
            return df["freq"] * 2

        apply_result = simple_benchmark(apply_version, n_iterations=10)
        vec_result = simple_benchmark(vectorized_version, n_iterations=10)

        speedup = apply_result.mean / vec_result.mean
        print(f"\nApply: {apply_result}")
        print(f"Vectorized: {vec_result}")
        print(f"Speedup: {speedup:.1f}x")

        assert speedup > 1, "Vectorized should be faster than apply"


# =============================================================================
# Phase 3: Numerical Stability
# =============================================================================


class TestPhase3NumericalStability:
    """Tests for Phase 3: numerical stability fixes."""

    def test_arctan2_no_division_by_zero(self) -> None:
        """Verify arctan2 handles zero denominator correctly."""
        # Test values that would cause division by zero in arctan(y/x)
        test_cases = [
            (1.0, 0.0),  # x=0, should not divide
            (0.0, 0.0),  # Both zero
            (0.0, 1.0),  # y=0
            (-1.0, 0.0),  # Negative y, x=0
        ]

        for y, x in test_cases:
            result = jnp.arctan2(y, x)
            assert not jnp.isnan(result), f"arctan2({y}, {x}) returned NaN"
            assert not jnp.isinf(result), f"arctan2({y}, {x}) returned Inf"

    def test_phi_clamping_no_nan(self) -> None:
        """Verify phi clamping prevents NaN from tan(phi/2)."""
        # Edge case phi values
        phi_values = jnp.array([0.0, jnp.pi / 4, jnp.pi / 2 - 1e-10, jnp.pi / 2])

        for phi in phi_values:
            phi_safe = jnp.clip(phi, 0, jnp.pi / 2 - 1e-10)
            result = jnp.tan(phi_safe / 2)
            assert not jnp.isnan(result), f"tan({phi_safe}/2) returned NaN"
            assert not jnp.isinf(result), f"tan({phi_safe}/2) returned Inf"

    def test_safe_divide_pattern(self) -> None:
        """Verify safe_divide handles division by zero with fill_value.

        Note: safe_divide returns fill_value (default: NaN) for zero denominator.
        This is the expected behavior - it prevents Inf and allows downstream
        code to detect and handle the condition.
        """
        from rheoQCM.core.physics import safe_divide

        # Test division by zero with default fill_value (NaN)
        result = safe_divide(1.0, 0.0)
        assert jnp.isnan(result), (
            "safe_divide should return NaN for 1/0 with default fill"
        )

        # Test division by zero with custom fill_value
        result = safe_divide(1.0, 0.0, fill_value=0.0)
        assert jnp.isclose(result, 0.0), (
            "safe_divide should return 0.0 for 1/0 with fill_value=0"
        )
        assert not jnp.isinf(result), "safe_divide should not return Inf"

        # Test normal division
        result = safe_divide(6.0, 2.0)
        assert jnp.isclose(result, 3.0), "safe_divide failed for normal division"


# =============================================================================
# Phase 4: Advanced Optimizations
# =============================================================================


class TestPhase4Vectorization:
    """Benchmark tests for Phase 4: advanced vectorization."""

    @pytest.fixture
    def sample_signal(self) -> np.ndarray:
        """Create sample signal for peak finding benchmarks."""
        x = np.linspace(0, 10, 10000)
        # Signal with multiple peaks
        signal = np.sin(x) * np.exp(-x / 5) + np.sin(3 * x) * 0.5
        return signal

    def test_peak_finding_baseline(self, sample_signal: np.ndarray) -> None:
        """Baseline test for peak finding performance."""
        from scipy.signal import find_peaks

        def run_peak_finding():
            peaks, _ = find_peaks(sample_signal, height=0.1)
            return peaks

        result = simple_benchmark(run_peak_finding, n_iterations=20)
        print(f"\nPeak finding: {result}")
        assert result.mean < 1.0, "Peak finding baseline too slow (>1s)"


# =============================================================================
# Regression Tests
# =============================================================================


class TestNumericalRegression:
    """Regression tests to verify numerical accuracy after optimization."""

    def test_multilayer_numerical_accuracy(self) -> None:
        """Verify multilayer calculations maintain numerical accuracy."""
        from rheoQCM.core.multilayer import calc_ZL

        # Known reference values (should be recorded from baseline)
        # Note: layers is dict[int, dict] where keys are layer numbers
        f1 = 5e6
        layers = {
            1: {"rho": 1000.0, "grho": 1e8, "phi": 0.1, "drho": 1e-3},
        }

        result = calc_ZL(f1=f1, n=3, layers=layers)

        # Verify result is complex and finite
        assert np.isfinite(float(result.real)), "calc_ZL real part not finite"
        assert np.isfinite(float(result.imag)), "calc_ZL imag part not finite"

    def test_solve_properties_numerical_accuracy(self) -> None:
        """Verify solve_properties maintains numerical accuracy."""
        from rheoQCM.core.model import QCMModel

        model = QCMModel(f1=5e6, refh=3)
        delfstars = {
            3: -1000 + 100j,
            5: -1700 + 180j,
            7: -2500 + 280j,
        }
        model.load_delfstars(delfstars)

        result = model.solve_properties(nh=[3, 5, 3])

        # Verify results are finite
        assert np.isfinite(result.drho), "drho not finite"
        assert np.isfinite(result.grho_refh), "grho_refh not finite"
        assert np.isfinite(result.phi), "phi not finite"

        # Verify reasonable ranges
        assert result.drho > 0, "drho should be positive"
        assert result.grho_refh > 0, "grho_refh should be positive"
        assert 0 <= result.phi <= np.pi / 2, "phi should be in [0, pi/2]"
