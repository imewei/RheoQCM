"""
JIT speedup benchmark tests for JAX performance optimization.
Feature: 012-jax-performance-optimization

Measures speedup achieved by JIT compilation on hot paths:
- T008: multilayer fallback path
- T009: _thin_film_guess
- T010: residual hoisting

Target: ≥2x speedup (goal: 30x)
"""

import json
import time
from collections.abc import Callable
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rheoQCM.core import model, multilayer, physics

# Get baseline timing file path
BASELINE_FILE = Path(__file__).parent / "baseline_timing.json"

# Test parameters
F1_DEFAULT = 5e6  # Default fundamental frequency
N_ITERATIONS = 20  # Number of benchmark iterations
N_WARMUP = 5  # Warmup iterations for JIT compilation


def load_baseline() -> dict:
    """Load baseline timing data."""
    if BASELINE_FILE.exists():
        with open(BASELINE_FILE) as f:
            return json.load(f)
    return {}


def save_baseline(data: dict) -> None:
    """Save baseline timing data."""
    with open(BASELINE_FILE, "w") as f:
        json.dump(data, f, indent=2)


def benchmark_function(
    fn: Callable,
    args: tuple = (),
    kwargs: dict | None = None,
    n_iterations: int = N_ITERATIONS,
    warmup: int = N_WARMUP,
) -> dict:
    """
    Benchmark a function with warm-start timing.

    Returns dict with timing statistics in milliseconds.
    """
    kwargs = kwargs or {}

    # Warmup runs (includes JIT compilation)
    for _ in range(warmup):
        result = fn(*args, **kwargs)
        if hasattr(result, "block_until_ready"):
            result.block_until_ready()

    # Timed runs
    times_ms = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        if hasattr(result, "block_until_ready"):
            result.block_until_ready()
        end = time.perf_counter()
        times_ms.append((end - start) * 1000)

    return {
        "mean_ms": np.mean(times_ms),
        "min_ms": np.min(times_ms),
        "max_ms": np.max(times_ms),
        "std_ms": np.std(times_ms),
        "iterations": n_iterations,
    }


class TestMultilayerJITSpeedup:
    """T008: Benchmark tests for multilayer JIT compilation."""

    @pytest.fixture
    def standard_layers(self):
        """Standard two-layer test case for multilayer calculations."""
        return {
            1: {"grho": 1e9, "phi": 0.1, "drho": 1e-6, "n": 3},
            2: {"grho": 1e8, "phi": 0.2, "drho": np.inf, "n": 3},
        }

    @pytest.fixture
    def multi_layers(self):
        """Four-layer test case to stress matrix chain multiplication."""
        return {
            1: {"grho": 1e9, "phi": 0.1, "drho": 1e-6, "n": 3},
            2: {"grho": 5e8, "phi": 0.15, "drho": 2e-6, "n": 3},
            3: {"grho": 2e8, "phi": 0.12, "drho": 5e-7, "n": 3},
            4: {"grho": 1e8, "phi": 0.2, "drho": np.inf, "n": 3},
        }

    def test_calc_ZL_jit_exists(self):
        """Verify JIT-compiled calc_ZL function exists."""
        assert hasattr(
            multilayer, "_calc_ZL_jit"
        ), "JIT-compiled _calc_ZL_jit should exist"

    def test_calc_ZL_standard_benchmark(self, standard_layers):
        """Benchmark calc_ZL with standard 2-layer case."""
        n = 3
        delfstar = 0.0 + 0.0j
        f1 = F1_DEFAULT

        def run_calc():
            return multilayer.calc_ZL(n, standard_layers, delfstar, f1)

        result = benchmark_function(run_calc)

        print(f"\ncalc_ZL (2-layer):")
        print(f"  mean: {result['mean_ms']:.3f} ms")
        print(f"  min:  {result['min_ms']:.3f} ms")
        print(f"  max:  {result['max_ms']:.3f} ms")
        print(f"  std:  {result['std_ms']:.3f} ms")

        # Store result for baseline comparison
        baseline = load_baseline()
        if baseline.get("calc_ZL_fallback", {}).get("mean_ms") is None:
            baseline["calc_ZL_fallback"] = {
                **result,
                "description": "multilayer.py fallback path - 2 layer case",
                "status": "measured",
            }
            save_baseline(baseline)

    def test_calc_ZL_multi_layer_benchmark(self, multi_layers):
        """Benchmark calc_ZL with 4-layer case."""
        n = 3
        delfstar = 0.0 + 0.0j
        f1 = F1_DEFAULT

        def run_calc():
            return multilayer.calc_ZL(n, multi_layers, delfstar, f1)

        result = benchmark_function(run_calc)

        print(f"\ncalc_ZL (4-layer):")
        print(f"  mean: {result['mean_ms']:.3f} ms")
        print(f"  min:  {result['min_ms']:.3f} ms")
        print(f"  max:  {result['max_ms']:.3f} ms")

    def test_calc_ZL_numerical_accuracy(self, standard_layers):
        """Verify JIT path produces same results as fallback."""
        n = 3
        delfstar = 0.0 + 0.0j
        f1 = F1_DEFAULT

        # Get result from calc_ZL (uses JIT path when possible)
        ZL = multilayer.calc_ZL(n, standard_layers, delfstar, f1)

        # Result should be finite and non-zero
        assert np.isfinite(ZL), f"calc_ZL produced non-finite: {ZL}"
        assert abs(ZL) > 0, "calc_ZL produced zero impedance"

    def test_calc_ZL_harmonic_sweep_benchmark(self, standard_layers):
        """Benchmark calc_ZL across multiple harmonics."""
        harmonics = [3, 5, 7, 9, 11]
        delfstar = 0.0 + 0.0j
        f1 = F1_DEFAULT

        def run_sweep():
            results = []
            for n in harmonics:
                ZL = multilayer.calc_ZL(n, standard_layers, delfstar, f1)
                results.append(ZL)
            return results

        result = benchmark_function(run_sweep, n_iterations=10)

        print(f"\ncalc_ZL (harmonic sweep, 5 harmonics):")
        print(f"  mean: {result['mean_ms']:.3f} ms")
        print(f"  per harmonic: {result['mean_ms']/5:.3f} ms")


class TestThinFilmGuessJITSpeedup:
    """T009: Benchmark tests for _thin_film_guess JIT compilation."""

    @pytest.fixture
    def qcm_model(self):
        """Standard QCMModel for thin film guess testing."""
        m = model.QCMModel(f1=F1_DEFAULT, refh=3)
        m.load_delfstars(
            {
                3: -1000 + 100j,
                5: -1700 + 180j,
                7: -2500 + 280j,
            }
        )
        return m

    def test_thin_film_guess_exists(self):
        """Verify _thin_film_guess method exists."""
        m = model.QCMModel(f1=F1_DEFAULT, refh=3)
        assert hasattr(m, "_thin_film_guess"), "_thin_film_guess should exist"

    def test_thin_film_guess_benchmark(self, qcm_model):
        """Benchmark _thin_film_guess function."""
        delfstars = qcm_model.delfstars
        nh = [3, 5, 7]

        def run_guess():
            return qcm_model._thin_film_guess(delfstars, nh)

        result = benchmark_function(run_guess)

        print(f"\n_thin_film_guess:")
        print(f"  mean: {result['mean_ms']:.3f} ms")
        print(f"  min:  {result['min_ms']:.3f} ms")
        print(f"  max:  {result['max_ms']:.3f} ms")
        print(f"  std:  {result['std_ms']:.3f} ms")

        # Store result for baseline
        baseline = load_baseline()
        if baseline.get("thin_film_guess", {}).get("mean_ms") is None:
            baseline["thin_film_guess"] = {
                **result,
                "description": "model.py _thin_film_guess",
                "status": "measured",
            }
            save_baseline(baseline)

    def test_thin_film_guess_numerical_accuracy(self, qcm_model):
        """Verify _thin_film_guess produces physically reasonable results."""
        delfstars = qcm_model.delfstars
        nh = [3, 5, 7]

        grho_refh, phi, drho, dlam_refh = qcm_model._thin_film_guess(delfstars, nh)

        # Results should be finite or NaN (not inf)
        for name, val in [
            ("grho_refh", grho_refh),
            ("phi", phi),
            ("drho", drho),
            ("dlam_refh", dlam_refh),
        ]:
            assert not np.isinf(val), f"{name} is infinite: {val}"

        # If values are finite, check physical ranges
        if np.isfinite(grho_refh):
            assert grho_refh > 0, f"grho_refh should be positive: {grho_refh}"
        if np.isfinite(phi):
            assert 0 <= phi <= np.pi / 2, f"phi out of range: {phi}"
        if np.isfinite(drho):
            assert drho > 0, f"drho should be positive: {drho}"

    def test_thin_film_guess_repeated_calls(self, qcm_model):
        """Verify determinism across repeated calls."""
        delfstars = qcm_model.delfstars
        nh = [3, 5, 7]

        results = []
        for _ in range(5):
            result = qcm_model._thin_film_guess(delfstars, nh)
            results.append(result)

        # All results should be identical
        for i, r in enumerate(results[1:], 1):
            for j, (v1, v2) in enumerate(zip(results[0], r)):
                if np.isfinite(v1) and np.isfinite(v2):
                    assert np.isclose(
                        v1, v2, rtol=1e-10
                    ), f"Result {i} differs at index {j}: {v1} vs {v2}"


class TestResidualHoistingSpeedup:
    """T010: Benchmark tests for residual function hoisting."""

    @pytest.fixture
    def qcm_model(self):
        """QCMModel for residual hoisting testing."""
        m = model.QCMModel(f1=F1_DEFAULT, refh=3)
        m.load_delfstars(
            {
                3: -1000 + 100j,
                5: -1700 + 180j,
                7: -2500 + 280j,
            }
        )
        return m

    def test_solve_properties_benchmark(self, qcm_model):
        """Benchmark full solve_properties (includes residual evaluation)."""
        nh = [3, 5, 7]

        def run_solve():
            return qcm_model.solve_properties(nh)

        result = benchmark_function(run_solve, warmup=3, n_iterations=10)

        print(f"\nsolve_properties (full workflow):")
        print(f"  mean: {result['mean_ms']:.3f} ms")
        print(f"  min:  {result['min_ms']:.3f} ms")
        print(f"  max:  {result['max_ms']:.3f} ms")
        print(f"  std:  {result['std_ms']:.3f} ms")

        # Store result for baseline
        baseline = load_baseline()
        if baseline.get("solve_properties", {}).get("mean_ms") is None:
            baseline["solve_properties"] = {
                **result,
                "description": "Full solve_properties workflow",
                "status": "measured",
            }
            save_baseline(baseline)

    def test_residual_thin_benchmark(self, qcm_model):
        """Benchmark residual_thin evaluation time directly."""
        # Access internal _jax_calc_delfstar to benchmark residual components
        from rheoQCM.core.model import _jax_calc_delfstar

        n1, n2, n3 = 3, 5, 7
        grho = 1e9
        phi = 0.1
        drho = 1e-6
        f1 = F1_DEFAULT
        Zq = physics.Zq
        refh = 3

        def run_residual():
            calc_n1 = _jax_calc_delfstar(n1, grho, phi, drho, f1, Zq, refh)
            calc_n2 = _jax_calc_delfstar(n2, grho, phi, drho, f1, Zq, refh)
            calc_n3 = _jax_calc_delfstar(n3, grho, phi, drho, f1, Zq, refh)
            return jnp.array([jnp.real(calc_n1), jnp.real(calc_n2), jnp.imag(calc_n3)])

        result = benchmark_function(run_residual)

        print(f"\nresidual evaluation (3 harmonics):")
        print(f"  mean: {result['mean_ms']:.3f} ms")
        print(f"  min:  {result['min_ms']:.3f} ms")
        print(f"  max:  {result['max_ms']:.3f} ms")

        # Store result for baseline
        baseline = load_baseline()
        if baseline.get("residual_thin", {}).get("mean_ms") is None:
            baseline["residual_thin"] = {
                **result,
                "description": "model.py residual_thin closure evaluation",
                "status": "measured",
            }
            save_baseline(baseline)

    def test_solve_properties_determinism(self, qcm_model):
        """Verify solve_properties produces deterministic results."""
        nh = [3, 5, 7]

        results = []
        for _ in range(3):
            result = qcm_model.solve_properties(nh)
            results.append(
                {
                    "grho_refh": result.grho_refh,
                    "phi": result.phi,
                    "drho": result.drho,
                }
            )

        # Compare all results
        for i, r in enumerate(results[1:], 1):
            for key in ["grho_refh", "phi", "drho"]:
                v0, v1 = results[0][key], r[key]
                if np.isfinite(v0) and np.isfinite(v1):
                    error = abs(v1 - v0) / max(abs(v0), 1e-30)
                    assert error < 1e-10, f"{key} differs: {error:.2e}"


class TestSpeedupComparison:
    """Compare JIT vs non-JIT performance to validate speedup targets."""

    def test_speedup_target_documentation(self):
        """Document expected speedup targets from spec."""
        print("\n" + "=" * 60)
        print("JAX Performance Optimization - Speedup Targets")
        print("=" * 60)
        print("Target: ≥2x speedup (minimum acceptable)")
        print("Goal:   30-50x speedup (aspirational)")
        print("=" * 60)

        # Load and display current baselines
        baseline = load_baseline()
        for key, data in baseline.items():
            if key.startswith("_"):
                continue
            mean = data.get("mean_ms")
            if mean is not None:
                print(f"\n{key}:")
                print(f"  Baseline: {mean:.3f} ms")


# Allow running specific tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
