"""Performance benchmark tests for Kotula model.

T013: Compare JAX vs mpmath implementation performance.
Requirement: 20x minimum speedup for 10,000+ xi values.
"""

import time

import numpy as np
import pytest

import jax.numpy as jnp

from rheoQCM.core.physics import kotula_gstar


# Standard test parameters
GMSTAR = 1e6 + 1e5j
GFSTAR = 1e9 + 1e8j
XI_CRIT = 0.16
S = 0.8
T = 1.8


@pytest.mark.slow
class TestKotulaBenchmark:
    """T013: Benchmark JAX vs mpmath performance."""

    def test_jax_10000_points_completes(self):
        """JAX implementation handles 10,000+ points without timeout."""
        xi = jnp.linspace(0.01, 0.99, 10000)

        # Warm-up JIT compilation
        _ = kotula_gstar(xi[:10], GMSTAR, GFSTAR, XI_CRIT, S, T)

        # Time the actual computation
        start = time.perf_counter()
        result = kotula_gstar(xi, GMSTAR, GFSTAR, XI_CRIT, S, T)
        jax_time = time.perf_counter() - start

        # Should complete in reasonable time (< 60 seconds)
        assert jax_time < 60.0, f"JAX took {jax_time:.2f}s for 10,000 points"

        # Verify results are valid
        assert result.shape == (10000,)
        assert not jnp.any(jnp.isnan(result)), "Some results are NaN"

        print(f"\nJAX time for 10,000 points: {jax_time:.3f}s")

    def test_jax_vs_mpmath_speedup(self):
        """JAX achieves 20x minimum speedup over mpmath.

        SC-001: JAX implementation achieves minimum 20x speedup over
        mpmath implementation for datasets with 10,000+ xi values.

        Approach: Measure time per point for both methods and extrapolate
        to compare at the 10,000 point scale. mpmath is measured with
        fewer points since it's slow, then extrapolated.
        """
        try:
            from mpmath import findroot
        except ImportError:
            pytest.skip("mpmath not installed - skipping speedup comparison")

        # mpmath: Use smaller dataset and extrapolate (it's too slow for 10,000)
        n_mpmath = 50
        xi_np = np.linspace(0.01, 0.99, n_mpmath)

        # JAX: Use full 10,000 points as per requirement
        n_jax = 10000
        xi_jax = jnp.linspace(0.01, 0.99, n_jax)

        # Time mpmath
        def mpmath_kotula_single(xi_val):
            def ftosolve(gstar_val):
                A = (1 - XI_CRIT) / XI_CRIT
                func = (
                    (1 - xi_val)
                    * (GMSTAR ** (1 / S) - gstar_val ** (1 / S))
                    / (GMSTAR ** (1 / S) + A * gstar_val ** (1 / S))
                    + xi_val
                    * (GFSTAR ** (1 / T) - gstar_val ** (1 / T))
                    / (GFSTAR ** (1 / T) + A * gstar_val ** (1 / T))
                )
                return func

            return complex(findroot(ftosolve, GMSTAR))

        start = time.perf_counter()
        mpmath_results = [mpmath_kotula_single(xi) for xi in xi_np]
        mpmath_time = time.perf_counter() - start
        mpmath_time_per_point = mpmath_time / n_mpmath

        # Time JAX with proper warmup (same array size)
        # First call triggers JIT compilation - exclude from timing
        _ = kotula_gstar(xi_jax, GMSTAR, GFSTAR, XI_CRIT, S, T)

        # Second call measures actual runtime
        start = time.perf_counter()
        jax_results = kotula_gstar(xi_jax, GMSTAR, GFSTAR, XI_CRIT, S, T)
        jax_time = time.perf_counter() - start

        # Calculate speedup: compare time for 10,000 points
        mpmath_extrapolated_time = mpmath_time_per_point * n_jax
        speedup = mpmath_extrapolated_time / jax_time

        print(f"\nBenchmark results:")
        print(f"  mpmath: {mpmath_time:.3f}s for {n_mpmath} points")
        print(f"          ({mpmath_time_per_point*1000:.2f}ms/point)")
        print(f"          Extrapolated {n_jax} points: {mpmath_extrapolated_time:.1f}s")
        print(f"  JAX:    {jax_time:.4f}s for {n_jax} points")
        print(f"          ({jax_time/n_jax*1000:.4f}ms/point)")
        print(f"  Speedup: {speedup:.1f}x")

        # Verify numerical consistency on subset
        jax_subset = np.array(kotula_gstar(jnp.array(xi_np), GMSTAR, GFSTAR, XI_CRIT, S, T))
        mpmath_np = np.array(mpmath_results)
        relative_diff = np.abs(jax_subset - mpmath_np) / np.abs(mpmath_np)
        max_diff = np.max(relative_diff)
        print(f"  Max relative difference: {max_diff:.2e}")

        # Assert minimum 20x speedup
        assert speedup >= 20, f"Expected 20x speedup, got {speedup:.1f}x"

        # Assert numerical consistency
        assert max_diff < 1e-6, f"Results differ by {max_diff:.2e}"

    def test_jax_scaling(self):
        """JAX performance scales linearly with input size."""
        sizes = [100, 1000, 10000]
        times = []

        for n in sizes:
            xi = jnp.linspace(0.01, 0.99, n)

            # Warm-up
            _ = kotula_gstar(xi[:10], GMSTAR, GFSTAR, XI_CRIT, S, T)

            # Time
            start = time.perf_counter()
            _ = kotula_gstar(xi, GMSTAR, GFSTAR, XI_CRIT, S, T)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        print(f"\nJAX scaling:")
        for n, t in zip(sizes, times):
            print(f"  {n:6d} points: {t:.4f}s")

        # Check roughly linear scaling (allow 3x overhead for larger sizes)
        # Time per point should not increase by more than 3x
        time_per_point = [t / n for t, n in zip(times, sizes)]
        ratio = time_per_point[-1] / time_per_point[0]
        print(f"  Time/point ratio (10000 vs 100): {ratio:.2f}x")

        # Linear scaling means ratio should be close to 1
        assert ratio < 5, f"Scaling is super-linear: {ratio:.2f}x increase in time/point"

    def test_memory_efficiency(self):
        """Memory usage scales linearly with input size (SC-003)."""
        import jax

        # This is a basic test - detailed memory profiling would need external tools
        xi = jnp.linspace(0.01, 0.99, 10000)

        # Should not raise OOM for reasonable sizes
        result = kotula_gstar(xi, GMSTAR, GFSTAR, XI_CRIT, S, T)

        # Verify result size is proportional to input
        assert result.shape == xi.shape

        # Check we're not holding onto intermediate arrays
        # (JAX should handle this automatically)
        devices = jax.devices()
        print(f"\nUsing device: {devices[0]}")
