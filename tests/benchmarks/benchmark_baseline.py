"""
Baseline Performance Benchmarks for JAX Performance Optimizations (005-jax-perf)

This script establishes baseline performance measurements before optimization.
Run before and after implementing optimizations to measure improvements.

Benchmarks:
1. calc_ZL: Multilayer acoustic impedance calculation
2. solve_properties: Single-point QCM property extraction
3. batch_analyze: Batch processing of multiple datasets

Success Criteria (from spec.md):
- SC-001: calc_ZL/solve_properties 10x faster
- SC-002: Jacobian gradients 3-4x fewer evaluations
- SC-003: batch_analyze 2x faster on CPU
- SC-004: 15% memory reduction in multilayer calculations
"""

import time
import tracemalloc
from typing import Any

import jax.numpy as jnp

from rheoQCM.core import multilayer, physics
from rheoQCM.core.model import QCMModel


def benchmark_calc_ZL(n_iterations: int = 100, warmup: int = 5) -> dict[str, Any]:
    """Benchmark calc_ZL function for single-layer calculation."""
    layers = {1: {"grho": 1e8, "phi": 0.3, "drho": 1e-6}}
    n = 3
    delfstar = -1000 + 100j
    f1 = 5e6
    calctype = "SLA"
    refh = 3

    for _ in range(warmup):
        _ = multilayer.calc_ZL(n, layers, delfstar, f1, calctype, refh)

    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = multilayer.calc_ZL(n, layers, delfstar, f1, calctype, refh)
    elapsed = time.perf_counter() - start

    return {
        "function": "calc_ZL",
        "iterations": n_iterations,
        "total_time_s": elapsed,
        "time_per_call_ms": elapsed / n_iterations * 1000,
    }


def benchmark_solve_properties(
    n_iterations: int = 100, warmup: int = 5
) -> dict[str, Any]:
    """Benchmark solve_properties for single-point QCM analysis."""
    model = QCMModel(f1=5e6, refh=3)
    model.load_delfstars({3: -1000 + 100j, 5: -1700 + 180j, 7: -2500 + 280j})
    nh = [3, 5, 3]

    for _ in range(warmup):
        _ = model.solve_properties(nh=nh)

    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = model.solve_properties(nh=nh)
    elapsed = time.perf_counter() - start

    return {
        "function": "solve_properties",
        "iterations": n_iterations,
        "total_time_s": elapsed,
        "time_per_call_ms": elapsed / n_iterations * 1000,
    }


def benchmark_batch_analyze(
    batch_size: int = 100, n_iterations: int = 10, warmup: int = 2
) -> dict[str, Any]:
    """Benchmark batch_analyze for multiple datasets."""
    from rheoQCM.core.analysis import batch_analyze

    batch_delfstars = [
        {
            3: -1000 + 100j + i * 10,
            5: -1700 + 180j + i * 15,
            7: -2500 + 280j + i * 20,
        }
        for i in range(batch_size)
    ]
    harmonics = [3, 5, 7]
    f1 = 5e6
    refh = 3

    for _ in range(warmup):
        _ = batch_analyze(batch_delfstars, harmonics, f1=f1, refh=refh)

    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = batch_analyze(batch_delfstars, harmonics, f1=f1, refh=refh)
    elapsed = time.perf_counter() - start

    return {
        "function": "batch_analyze",
        "batch_size": batch_size,
        "iterations": n_iterations,
        "total_time_s": elapsed,
        "time_per_call_ms": elapsed / n_iterations * 1000,
        "time_per_sample_ms": elapsed / n_iterations / batch_size * 1000,
    }


def benchmark_memory_multilayer() -> dict[str, Any]:
    """Benchmark memory usage during multilayer calculation."""
    layers = {
        1: {"grho": 3e17, "phi": 0.0, "drho": 2.8e-6},
        2: {"grho": 1e8, "phi": 0.3, "drho": 1e-6},
        3: {"grho": 1e8, "phi": jnp.pi / 2, "drho": jnp.inf},
    }
    n = 3
    delfstar = -1000 + 100j
    f1 = 5e6
    calctype = "SLA"
    refh = 3

    _ = multilayer.calc_ZL(n, layers, delfstar, f1, calctype, refh)

    tracemalloc.start()
    for _ in range(100):
        _ = multilayer.calc_ZL(n, layers, delfstar, f1, calctype, refh)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "function": "calc_ZL (3-layer)",
        "iterations": 100,
        "current_memory_kb": current / 1024,
        "peak_memory_kb": peak / 1024,
    }


def benchmark_jacobian_evaluations() -> dict[str, Any]:
    """Count function evaluations for Jacobian calculation (finite-diff)."""
    n1, n2, n3 = 3, 5, 7
    dlam = 0.5
    phi = 0.3
    refh = 3
    eps = 1e-8

    eval_count = [0]

    def count_rhcalc(*args, **kwargs):
        eval_count[0] += 1
        return physics.rhcalc(*args, **kwargs)

    def count_rdcalc(*args, **kwargs):
        eval_count[0] += 1
        return physics.rdcalc(*args, **kwargs)

    rh_calc = count_rhcalc(n1, n2, dlam, phi, refh=refh)
    rd_calc = count_rdcalc(n3, dlam, phi, refh=refh)
    _ = (count_rhcalc(n1, n2, dlam + eps, phi, refh=refh) - rh_calc) / eps
    _ = (count_rhcalc(n1, n2, dlam, phi + eps, refh=refh) - rh_calc) / eps
    _ = (count_rdcalc(n3, dlam + eps, phi, refh=refh) - rd_calc) / eps
    _ = (count_rdcalc(n3, dlam, phi + eps, refh=refh) - rd_calc) / eps

    return {
        "function": "Jacobian (finite-diff)",
        "total_evaluations": eval_count[0],
        "expected_with_autodiff": 1,
    }


def run_all_benchmarks() -> dict[str, dict[str, Any]]:
    """Run all baseline benchmarks and return results."""
    results = {}

    print("Running calc_ZL benchmark...")
    results["calc_ZL"] = benchmark_calc_ZL()

    print("Running solve_properties benchmark...")
    results["solve_properties"] = benchmark_solve_properties()

    print("Running batch_analyze benchmark...")
    results["batch_analyze"] = benchmark_batch_analyze()

    print("Running memory benchmark...")
    results["memory"] = benchmark_memory_multilayer()

    print("Running Jacobian evaluation count...")
    results["jacobian"] = benchmark_jacobian_evaluations()

    return results


def print_results(results: dict[str, dict[str, Any]]) -> None:
    """Pretty-print benchmark results."""
    print("\n" + "=" * 70)
    print("BASELINE PERFORMANCE BENCHMARKS (005-jax-perf)")
    print("=" * 70)

    for name, data in results.items():
        print(f"\n{name.upper()}")
        print("-" * 40)
        for key, value in data.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

    print("\n" + "=" * 70)
    print("SUCCESS CRITERIA TARGETS")
    print("=" * 70)
    print("  SC-001: calc_ZL/solve_properties should be 10x faster")
    print("  SC-002: Jacobian should use 1 evaluation (vs current 6)")
    print("  SC-003: batch_analyze should be 2x faster")
    print("  SC-004: Memory should reduce by 15%")
    print("=" * 70)


if __name__ == "__main__":
    results = run_all_benchmarks()
    print_results(results)
