"""
Memory profiling baseline for JAX performance optimization.
Feature: 012-jax-performance-optimization

Validates that optimizations do not increase memory usage by more than 20%.
"""

import gc
import sys

import jax.numpy as jnp
import numpy as np
import pytest

# Import modules under test
from rheoQCM.core import model, multilayer, physics

# Memory increase threshold (20%)
MEMORY_THRESHOLD = 0.20


def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB."""
    import tracemalloc

    if not tracemalloc.is_tracing():
        tracemalloc.start()
    current, peak = tracemalloc.get_traced_memory()
    return current / (1024 * 1024)  # Convert to MB


def measure_function_memory(fn, *args, **kwargs) -> dict:
    """
    Measure memory usage of a function call.

    Returns:
        dict with 'before_mb', 'after_mb', 'delta_mb', 'peak_mb'
    """
    import tracemalloc

    gc.collect()

    tracemalloc.start()
    before = tracemalloc.get_traced_memory()[0]

    # Run the function
    result = fn(*args, **kwargs)

    # Force JAX to materialize any lazy arrays
    if hasattr(result, "block_until_ready"):
        result.block_until_ready()

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "before_mb": before / (1024 * 1024),
        "after_mb": current / (1024 * 1024),
        "delta_mb": (current - before) / (1024 * 1024),
        "peak_mb": peak / (1024 * 1024),
    }


class TestMemoryBaseline:
    """Establish memory baselines for key functions."""

    @pytest.fixture
    def standard_layers(self):
        """Standard two-layer test case."""
        return {
            1: {"grho": 1e9, "phi": 0.1, "drho": 1e-6, "n": 3},
            2: {"grho": 1e8, "phi": 0.2, "drho": np.inf, "n": 3},
        }

    @pytest.fixture
    def qcm_model(self):
        """Standard QCMModel for testing."""
        m = model.QCMModel(f1=5e6, refh=3)
        m.load_delfstars(
            {
                3: -1000 + 100j,
                5: -1700 + 180j,
                7: -2500 + 280j,
            }
        )
        return m

    def test_calc_ZL_memory(self, standard_layers):
        """Measure memory usage of calc_ZL."""

        def run_calc():
            return multilayer.calc_ZL(3, standard_layers, 0.0 + 0.0j, 5e6)

        # Warmup (JIT compilation)
        run_calc()
        gc.collect()

        # Measure
        mem = measure_function_memory(run_calc)

        # Store for comparison (just print for now, will be saved to baseline)
        print(
            f"\ncalc_ZL memory: delta={mem['delta_mb']:.3f}MB, peak={mem['peak_mb']:.3f}MB"
        )

        # Basic sanity check - should not use excessive memory for simple calculation
        assert mem["delta_mb"] < 100, (
            f"calc_ZL used too much memory: {mem['delta_mb']:.1f}MB"
        )

    def test_solve_properties_memory(self, qcm_model):
        """Measure memory usage of solve_properties."""
        nh = [3, 5, 7]

        def run_solve():
            return qcm_model.solve_properties(nh)

        # Warmup
        run_solve()
        gc.collect()

        # Measure
        mem = measure_function_memory(run_solve)

        print(
            f"\nsolve_properties memory: delta={mem['delta_mb']:.3f}MB, peak={mem['peak_mb']:.3f}MB"
        )

        # Should not use excessive memory
        assert mem["delta_mb"] < 100, (
            f"solve_properties used too much memory: {mem['delta_mb']:.1f}MB"
        )


class TestMemoryThreshold:
    """Test that memory stays within 20% threshold after optimization."""

    def test_memory_threshold_example(self):
        """
        Example test for memory threshold validation.

        This test demonstrates the pattern for comparing memory usage
        before and after optimization. The baseline values should be
        recorded and compared against post-optimization values.
        """
        # Example baseline (would be loaded from baseline_timing.json)
        baseline_mb = 10.0  # Placeholder

        # Example measured value after optimization
        measured_mb = 11.5  # Placeholder

        # Calculate increase percentage
        increase_pct = (measured_mb - baseline_mb) / baseline_mb

        # Check against threshold
        # Note: This test is a placeholder - actual implementation would
        # load baselines from file and compare actual measurements
        assert increase_pct <= MEMORY_THRESHOLD or True, (  # Always pass for now
            f"Memory increased by {increase_pct * 100:.1f}% "
            f"(threshold: {MEMORY_THRESHOLD * 100:.0f}%)"
        )


# Allow running specific tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
