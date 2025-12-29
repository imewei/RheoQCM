"""
Benchmark infrastructure for JAX performance optimization.
Feature: 012-jax-performance-optimization

Provides warm-start timing utilities following JAX best practices:
- Warmup runs to trigger JIT compilation
- block_until_ready() for accurate GPU/accelerator sync
- 10+ iterations for statistical significance
"""

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import pytest


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""

    name: str
    mean_ms: float
    min_ms: float
    max_ms: float
    std_ms: float
    iterations: int
    warmup_iterations: int


def benchmark_function(
    fn: Callable,
    args: tuple = (),
    kwargs: dict | None = None,
    n_iterations: int = 20,
    warmup: int = 5,
    name: str = "benchmark",
) -> BenchmarkResult:
    """
    Benchmark a function using warm-start timing.

    Following JAX best practices:
    - Warmup runs to complete JIT compilation
    - block_until_ready() for accurate timing
    - Statistical aggregation over multiple iterations

    Args:
        fn: Function to benchmark
        args: Positional arguments for fn
        kwargs: Keyword arguments for fn
        n_iterations: Number of timed iterations (default: 20)
        warmup: Number of warmup iterations (default: 5)
        name: Name for the benchmark result

    Returns:
        BenchmarkResult with timing statistics
    """
    if kwargs is None:
        kwargs = {}

    # Warmup runs (include JIT compilation)
    for _ in range(warmup):
        result = fn(*args, **kwargs)
        # Handle JAX arrays - call block_until_ready if available
        if hasattr(result, "block_until_ready"):
            result.block_until_ready()
        elif isinstance(result, tuple):
            for r in result:
                if hasattr(r, "block_until_ready"):
                    r.block_until_ready()

    # Timed runs
    times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        # Sync for accurate timing
        if hasattr(result, "block_until_ready"):
            result.block_until_ready()
        elif isinstance(result, tuple):
            for r in result:
                if hasattr(r, "block_until_ready"):
                    r.block_until_ready()
        times.append(time.perf_counter() - start)

    times_ms = [t * 1000 for t in times]
    mean = sum(times_ms) / len(times_ms)
    variance = sum((t - mean) ** 2 for t in times_ms) / len(times_ms)
    std = variance**0.5

    return BenchmarkResult(
        name=name,
        mean_ms=mean,
        min_ms=min(times_ms),
        max_ms=max(times_ms),
        std_ms=std,
        iterations=n_iterations,
        warmup_iterations=warmup,
    )


def compare_speedup(baseline: BenchmarkResult, optimized: BenchmarkResult) -> float:
    """
    Calculate speedup ratio between baseline and optimized.

    Returns:
        Speedup factor (baseline_time / optimized_time)
    """
    if optimized.mean_ms <= 0:
        return float("inf")
    return baseline.mean_ms / optimized.mean_ms


def load_baseline_timing(name: str) -> dict | None:
    """Load baseline timing from JSON file."""
    baseline_path = Path(__file__).parent / "baseline_timing.json"
    if not baseline_path.exists():
        return None

    with open(baseline_path) as f:
        data = json.load(f)

    return data.get(name)


def save_baseline_timing(name: str, result: BenchmarkResult) -> None:
    """Save baseline timing to JSON file."""
    baseline_path = Path(__file__).parent / "baseline_timing.json"

    if baseline_path.exists():
        with open(baseline_path) as f:
            data = json.load(f)
    else:
        data = {}

    data[name] = {
        "mean_ms": result.mean_ms,
        "min_ms": result.min_ms,
        "max_ms": result.max_ms,
        "std_ms": result.std_ms,
        "iterations": result.iterations,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(baseline_path, "w") as f:
        json.dump(data, f, indent=2)


@pytest.fixture
def benchmark():
    """Pytest fixture providing benchmark_function."""
    return benchmark_function


@pytest.fixture
def assert_speedup():
    """Pytest fixture for asserting minimum speedup."""

    def _assert_speedup(
        baseline: BenchmarkResult,
        optimized: BenchmarkResult,
        min_speedup: float = 2.0,
        target_speedup: float | None = None,
    ):
        actual_speedup = compare_speedup(baseline, optimized)

        # Always pass if we meet minimum threshold
        assert actual_speedup >= min_speedup, (
            f"Speedup {actual_speedup:.2f}x below minimum {min_speedup}x. "
            f"Baseline: {baseline.mean_ms:.3f}ms, Optimized: {optimized.mean_ms:.3f}ms"
        )

        # Log target achievement for documentation
        if target_speedup is not None:
            if actual_speedup >= target_speedup:
                print(f"TARGET MET: {actual_speedup:.2f}x >= {target_speedup}x target")
            else:
                print(
                    f"PARTIAL: {actual_speedup:.2f}x (target was {target_speedup}x, min {min_speedup}x met)"
                )

        return actual_speedup

    return _assert_speedup
