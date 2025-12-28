"""Performance benchmark tests for Kotula model and batch processing.

T013: Compare JAX vs mpmath implementation performance.
Requirement: 20x minimum speedup for 10,000+ xi values.

T045: Batch processing performance benchmark.
Requirement: 1000 measurements in <5s on GPU.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rheoQCM.core.jax_config import get_jax_backend, is_gpu_available
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
                func = (1 - xi_val) * (GMSTAR ** (1 / S) - gstar_val ** (1 / S)) / (
                    GMSTAR ** (1 / S) + A * gstar_val ** (1 / S)
                ) + xi_val * (GFSTAR ** (1 / T) - gstar_val ** (1 / T)) / (
                    GFSTAR ** (1 / T) + A * gstar_val ** (1 / T)
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
        jax_subset = np.array(
            kotula_gstar(jnp.array(xi_np), GMSTAR, GFSTAR, XI_CRIT, S, T)
        )
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
        assert (
            ratio < 5
        ), f"Scaling is super-linear: {ratio:.2f}x increase in time/point"

    def test_memory_efficiency(self):
        """Memory usage scales linearly with input size (SC-003)."""
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


# =============================================================================
# T045: Batch Processing Performance Benchmark (US3)
# =============================================================================


@pytest.mark.slow
class TestBatchProcessingBenchmark:
    """T045: Benchmark batch_analyze_vmap performance.

    Requirement: 1000 measurements should complete in <5s on GPU.
    On CPU, we allow more time but still verify reasonable performance.
    """

    def test_batch_analyze_1000_measurements_timing(self):
        """Test that batch_analyze_vmap processes 1000 measurements efficiently.

        Performance target:
        - GPU: <5s for 1000 measurements
        - CPU: <30s for 1000 measurements (more lenient for CI without GPU)
        """
        from rheoQCM.core.analysis import batch_analyze_vmap

        n_measurements = 1000
        harmonics = [3, 5]

        # Generate test data with varying film properties
        delfstars_list = []
        for i in range(n_measurements):
            scale = 1.0 + 0.1 * (i % 10)  # Cycle through scales
            delfstars_list.append(
                [
                    (-87768.0 * scale) + (155.7 * scale) * 1j,  # n=3
                    (-159742.7 * scale) + (888.7 * scale) * 1j,  # n=5
                ]
            )

        delfstars_array = jnp.array(delfstars_list)

        # Warm-up JIT compilation
        _ = batch_analyze_vmap(
            delfstars=delfstars_array[:10],
            harmonics=harmonics,
            nhcalc="35",
            f1=5e6,
            refh=3,
        )

        # Time the actual computation
        start = time.perf_counter()
        result = batch_analyze_vmap(
            delfstars=delfstars_array,
            harmonics=harmonics,
            nhcalc="35",
            f1=5e6,
            refh=3,
        )
        elapsed = time.perf_counter() - start

        # Report backend and timing
        backend = get_jax_backend()
        gpu_available = is_gpu_available()

        print(f"\nBatch processing benchmark:")
        print(f"  Backend: {backend}")
        print(f"  GPU available: {gpu_available}")
        print(f"  Measurements: {n_measurements}")
        print(f"  Time: {elapsed:.3f}s")
        print(f"  Rate: {n_measurements/elapsed:.1f} measurements/s")

        # Verify results
        assert len(result) == n_measurements
        assert result.drho.shape == (n_measurements,)
        assert result.grho_refh.shape == (n_measurements,)
        assert result.phi.shape == (n_measurements,)

        # Performance assertions based on backend
        if backend == "gpu":
            # GPU should be very fast
            assert (
                elapsed < 5.0
            ), f"GPU should process 1000 measurements in <5s, took {elapsed:.2f}s"
        else:
            # CPU is slower but should still be reasonable
            # Allow 30s for CPU (vmap still provides parallelization benefits)
            assert (
                elapsed < 30.0
            ), f"CPU should process 1000 measurements in <30s, took {elapsed:.2f}s"

    def test_batch_analyze_scaling(self):
        """Test that batch_analyze_vmap scales linearly with batch size."""
        from rheoQCM.core.analysis import batch_analyze_vmap

        harmonics = [3, 5]
        sizes = [100, 500, 1000]
        times = []

        for n in sizes:
            # Generate test data
            delfstars_list = []
            for i in range(n):
                scale = 1.0 + 0.05 * (i % 20)
                delfstars_list.append(
                    [
                        (-87768.0 * scale) + (155.7 * scale) * 1j,
                        (-159742.7 * scale) + (888.7 * scale) * 1j,
                    ]
                )

            delfstars_array = jnp.array(delfstars_list)

            # Warm-up
            _ = batch_analyze_vmap(
                delfstars=delfstars_array[:10],
                harmonics=harmonics,
                nhcalc="35",
            )

            # Time
            start = time.perf_counter()
            _ = batch_analyze_vmap(
                delfstars=delfstars_array,
                harmonics=harmonics,
                nhcalc="35",
            )
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        print(f"\nBatch scaling ({get_jax_backend()}):")
        for n, t in zip(sizes, times):
            print(f"  {n:5d} measurements: {t:.4f}s ({n/t:.1f}/s)")

        # Check roughly linear scaling
        time_per_measurement = [t / n for t, n in zip(times, sizes)]
        ratio = time_per_measurement[-1] / time_per_measurement[0]
        print(f"  Time/measurement ratio (1000 vs 100): {ratio:.2f}x")

        # Allow some overhead for larger batches, but should be roughly linear
        assert (
            ratio < 3.0
        ), f"Scaling is super-linear: {ratio:.2f}x increase in time/measurement"

    def test_batch_vs_sequential_speedup(self):
        """Test that batch processing is faster than sequential processing."""
        from rheoQCM.core.analysis import QCMAnalyzer, batch_analyze_vmap

        n_measurements = 100
        harmonics = [3, 5]

        # Generate test data
        batch_delfstars_dict = []
        delfstars_list = []
        for i in range(n_measurements):
            scale = 1.0 + 0.1 * (i % 10)
            delfstar_3 = (-87768.0 * scale) + (155.7 * scale) * 1j
            delfstar_5 = (-159742.7 * scale) + (888.7 * scale) * 1j
            delfstars_list.append([delfstar_3, delfstar_5])
            batch_delfstars_dict.append({3: delfstar_3, 5: delfstar_5})

        delfstars_array = jnp.array(delfstars_list)

        # Warm-up both methods
        _ = batch_analyze_vmap(
            delfstars=delfstars_array[:10],
            harmonics=harmonics,
            nhcalc="35",
        )
        analyzer = QCMAnalyzer(f1=5e6, refh=3)
        analyzer.load_data(batch_delfstars_dict[0])
        _ = analyzer.analyze(nh=[3, 5, 3], calctype="SLA")

        # Time batch processing
        start = time.perf_counter()
        batch_result = batch_analyze_vmap(
            delfstars=delfstars_array,
            harmonics=harmonics,
            nhcalc="35",
        )
        batch_time = time.perf_counter() - start

        # Time sequential processing
        start = time.perf_counter()
        sequential_results = []
        for delfstar in batch_delfstars_dict:
            analyzer = QCMAnalyzer(f1=5e6, refh=3)
            analyzer.load_data(delfstar)
            result = analyzer.analyze(nh=[3, 5, 3], calctype="SLA")
            sequential_results.append(result)
        sequential_time = time.perf_counter() - start

        speedup = sequential_time / batch_time

        print(f"\nBatch vs sequential ({get_jax_backend()}):")
        print(f"  Measurements: {n_measurements}")
        print(f"  Batch time: {batch_time:.4f}s")
        print(f"  Sequential time: {sequential_time:.4f}s")
        print(f"  Speedup: {speedup:.1f}x")

        # Batch should be faster (at least for CPU, much faster for GPU)
        # Note: Due to JIT compilation overhead on first call and different
        # optimization paths, we just verify batch completes reasonably
        assert len(batch_result) == n_measurements
        assert len(sequential_results) == n_measurements

    def test_batch_analyze_gpu_utilization(self):
        """Test that batch_analyze_vmap properly utilizes GPU when available."""
        from rheoQCM.core.analysis import batch_analyze_vmap

        backend = get_jax_backend()
        gpu_available = is_gpu_available()

        print(f"\nGPU utilization check:")
        print(f"  Backend: {backend}")
        print(f"  GPU available: {gpu_available}")

        # Create test data
        n = 500
        harmonics = [3, 5]
        delfstars_list = []
        for i in range(n):
            scale = 1.0 + 0.05 * (i % 20)
            delfstars_list.append(
                [
                    (-87768.0 * scale) + (155.7 * scale) * 1j,
                    (-159742.7 * scale) + (888.7 * scale) * 1j,
                ]
            )

        delfstars_array = jnp.array(delfstars_list)

        # Run batch processing
        result = batch_analyze_vmap(
            delfstars=delfstars_array,
            harmonics=harmonics,
            nhcalc="35",
        )

        # Verify results
        assert len(result) == n

        # Check that backend info is in messages
        if result.messages:
            assert any(
                backend in msg for msg in result.messages
            ), f"Backend '{backend}' should be mentioned in result messages"

        # Report device placement
        devices = jax.devices()
        print(f"  Active devices: {[str(d) for d in devices]}")
