# Quality Checklist: JAX Performance Optimization

**Feature Branch**: `012-jax-performance-optimization`
**Created**: 2025-12-28

## Pre-Implementation Validation

- [ ] Baseline benchmarks recorded for all target functions
- [ ] Baseline memory usage recorded
- [ ] All existing tests pass (`pytest tests/`)
- [ ] Code analysis report reviewed (`.optimization/src-report-2025-12-28.md`)

## Phase 1: Critical Path JIT Compilation

### Task 1.1: JIT multilayer.py fallback path
- [ ] Lines 414-553 identified and understood
- [ ] `@partial(jax.jit, static_argnames=["num_layers", "calctype"])` added
- [ ] Unit tests pass for multilayer calculations
- [ ] Benchmark shows ≥15x speedup
- [ ] Numerical accuracy within 1e-10 vs baseline

### Task 1.2: JIT _thin_film_guess
- [ ] model.py lines 1195-1228 identified
- [ ] `@jax.jit` with static harmonics applied
- [ ] Unit tests pass for thin film guess
- [ ] Benchmark shows ≥12x speedup
- [ ] Numerical accuracy within 1e-10 vs baseline

### Task 1.3: Hoist residual functions
- [ ] model.py lines 1346-1424 identified
- [ ] `residual_thin` moved to module level
- [ ] Pre-compilation verified
- [ ] Unit tests pass
- [ ] Benchmark shows ≥10x speedup
- [ ] Numerical accuracy within 1e-10 vs baseline

**Phase 1 Gate**:
- [ ] All Phase 1 tests pass
- [ ] Cumulative speedup ≥30x verified
- [ ] Memory increase <20%

## Phase 2: Secondary Optimizations

### Task 2.1: Fix mutable defaults
- [ ] MatplotlibWidget.py mutable defaults identified
- [ ] DataSaver.py mutable defaults identified
- [ ] rheoQCM.py mutable defaults identified
- [ ] All `def f(x=[])` replaced with `def f(x=None)` pattern
- [ ] `ruff check --select=B006` reports no errors
- [ ] All tests pass

### Task 2.2: Vectorize top 5 .apply() calls
- [ ] DataSaver.py .apply() instances profiled
- [ ] Top 5 by frequency/impact identified
- [ ] Pandas vectorized replacements implemented
- [ ] Numerical equivalence verified
- [ ] Benchmark shows ≥10x speedup per operation

### Task 2.3: Implement value_and_jacfwd
- [ ] analysis.py lines 762-804 identified
- [ ] `jax.value_and_jacfwd` caching implemented
- [ ] Benchmark shows ≥4x speedup
- [ ] Numerical accuracy verified

### Task 2.4: Fix quadruple .apply() pattern
- [ ] QCM.py lines 963-981 identified
- [ ] Consolidated into single vectorized operation
- [ ] Benchmark shows ≥4x speedup
- [ ] Numerical equivalence verified

**Phase 2 Gate**:
- [ ] All Phase 2 tests pass
- [ ] Additional speedup ≥5x verified
- [ ] Zero mutable default linting errors

## Phase 3: Numerical Stability

### Task 3.1: Fix arctan division
- [ ] model.py line 1167 identified
- [ ] Replaced with `jnp.arctan2(real, imag)`
- [ ] Edge case tests added (zero denominator)
- [ ] No NaN/Inf for zero inputs

### Task 3.2: Add phi clamping
- [ ] physics.py line 640 identified
- [ ] `phi_safe = jnp.clip(phi, 0, jnp.pi/2 - 1e-10)` added
- [ ] Edge case tests added (boundary values)
- [ ] No NaN for phi=0 or phi=π/2

### Task 3.3: Apply safe_divide pattern
- [ ] physics.py lines 531, 809 identified
- [ ] safe_divide pattern applied
- [ ] Edge case tests added
- [ ] No division-by-zero exceptions

**Phase 3 Gate**:
- [ ] All Phase 3 tests pass
- [ ] Zero NaN/Inf outputs on edge cases
- [ ] Numerical stability improved

## Phase 4: Advanced Optimizations

### Task 4.1: Vectorize peak_prominences
- [ ] signal.py lines 266-304 identified
- [ ] NumPy vectorized implementation created
- [ ] Numerical equivalence verified
- [ ] Benchmark shows ≥10x speedup

### Task 4.2: Vectorize peak_widths
- [ ] signal.py lines 392-432 identified
- [ ] NumPy vectorized implementation created
- [ ] Numerical equivalence verified
- [ ] Benchmark shows ≥10x speedup

### Task 4.3: Replace Kotula FD Jacobian
- [ ] physics.py lines 930-973 identified
- [ ] `jax.jacfwd` replacement implemented
- [ ] Benchmark shows ≥2x speedup
- [ ] Numerical accuracy improved

### Task 4.4: Replace Lu-Lewis FD Jacobian
- [ ] multilayer.py lines 901-936 identified
- [ ] `jax.jacfwd` replacement implemented
- [ ] Benchmark shows ≥2x speedup
- [ ] Numerical accuracy improved

### Task 4.5: Cache Savgol coefficients
- [ ] Savgol coefficient computation identified
- [ ] Caching mechanism implemented
- [ ] Benchmark shows ≥1.5x speedup

**Phase 4 Gate**:
- [ ] All Phase 4 tests pass
- [ ] Signal processing speedup ≥10x verified
- [ ] Autodiff Jacobians working correctly

## Final Validation

- [ ] Full test suite passes (`pytest tests/`)
- [ ] Total cumulative speedup ≥50x verified
- [ ] JIT coverage >90%
- [ ] Memory increase <20% from baseline
- [ ] No new linting errors
- [ ] Benchmarks documented
- [ ] CLAUDE.md updated with optimization notes
