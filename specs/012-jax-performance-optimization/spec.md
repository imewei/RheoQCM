# Feature Specification: JAX Performance Optimization

**Feature Branch**: `012-jax-performance-optimization`
**Created**: 2025-12-28
**Status**: Draft
**Input**: Multi-agent optimization analysis report from `.optimization/src-report-2025-12-28.md`

## Overview

This feature implements a 4-phase optimization plan identified by multi-agent analysis of the RheoQCM codebase. The analysis found 53 optimization opportunities across 49 Python files, with an estimated cumulative speedup of 50-70x after full implementation.

**Reference**: `.optimization/src-report-2025-12-28.md`

## User Scenarios & Testing

### User Story 1 - Critical Path JIT Compilation (Priority: P1)

Users running multilayer film analysis experience slow performance due to missing JIT compilation on critical calculation paths. This story addresses the three highest-impact JIT optimizations.

**Why this priority**: These optimizations provide 30-50x cumulative speedup with relatively low implementation risk. They target the most frequently called functions in the hot path.

**Independent Test**: Can be fully tested by running `pytest tests/` and benchmark comparisons showing ≥10x speedup on multilayer calculations.

**Acceptance Scenarios**:

1. **Given** multilayer.py fallback path at lines 414-553, **When** JIT decorator is added with `static_argnames=["num_layers", "calctype"]`, **Then** benchmark shows 15-20x speedup vs baseline.

2. **Given** model.py `_thin_film_guess` at lines 1195-1228, **When** `@jax.jit` is applied with static harmonics, **Then** benchmark shows 12-15x speedup.

3. **Given** model.py thin film residual at lines 1346-1424, **When** `residual_thin` is hoisted to module level and pre-compiled, **Then** benchmark shows 10-12x speedup.

---

### User Story 2 - Secondary JAX Optimizations (Priority: P2)

Users running batch analyses need improved Jacobian computation and array conversion efficiency. This story addresses medium-priority JAX optimizations.

**Why this priority**: These changes provide 5-15x additional speedup and improve code quality (mutable defaults bug prevention).

**Independent Test**: Can be tested by running batch analysis benchmarks and verifying no mutable default argument bugs.

**Acceptance Scenarios**:

1. **Given** analysis.py autodiff Jacobian at lines 762-804, **When** `value_and_jacfwd` caching is implemented, **Then** benchmark shows 4x speedup.

2. **Given** mutable default arguments in MatplotlibWidget.py, DataSaver.py, rheoQCM.py, **When** replaced with `def f(x=None)` pattern, **Then** no mutable default linting errors remain.

3. **Given** DataSaver.py with 37 `.apply()` instances, **When** top 5 are vectorized using pandas vectorized operations, **Then** benchmark shows 10-100x speedup per operation.

4. **Given** QCM.py:963-981 quadruple `.apply()` pattern, **When** consolidated into single vectorized operation, **Then** benchmark shows 4x speedup.

---

### User Story 3 - Numerical Stability Fixes (Priority: P3)

Users encounter NaN values or incorrect results under edge-case inputs due to division-by-zero and unbounded trigonometric operations.

**Why this priority**: These fixes prevent silent numerical errors that could invalidate scientific results.

**Independent Test**: Can be tested by running edge-case inputs through physics calculations and verifying no NaN/Inf outputs.

**Acceptance Scenarios**:

1. **Given** model.py:1167 arctan division, **When** replaced with `jnp.arctan2(real, imag)`, **Then** no division-by-zero for zero denominator inputs.

2. **Given** physics.py:640 `tan(phi/2)` calculation, **When** phi is clamped with `jnp.clip(phi, 0, jnp.pi/2 - 1e-10)`, **Then** no NaN values for boundary phi values.

3. **Given** physics.py:531, 809 division operations, **When** `safe_divide` pattern is applied, **Then** no division-by-zero exceptions.

---

### User Story 4 - Advanced Vectorization (Priority: P4)

Users processing large datasets need vectorized signal processing and autodiff Jacobians for maximum performance.

**Why this priority**: These are larger refactoring efforts with significant speedup potential (10-50x) but higher implementation complexity.

**Independent Test**: Can be tested by comparing vectorized vs loop-based implementations on benchmark datasets.

**Acceptance Scenarios**:

1. **Given** signal.py:266-304 peak_prominences with Python loops, **When** reimplemented with NumPy vectorized operations, **Then** benchmark shows 10-50x speedup.

2. **Given** signal.py:392-432 peak_widths with Python loops, **When** reimplemented with NumPy vectorized operations, **Then** benchmark shows 10-50x speedup.

3. **Given** physics.py:930-973 Kotula FD Jacobian, **When** replaced with `jax.jacfwd`, **Then** benchmark shows 2-3x speedup with better numerical accuracy.

4. **Given** multilayer.py:901-936 Lu-Lewis FD Jacobian, **When** replaced with `jax.jacfwd`, **Then** benchmark shows 2-3x speedup with better numerical accuracy.

---

### Edge Cases

- What happens when multilayer calculation has 0 or 1 layers? (Verify JIT handles edge cases)
- How does system handle phi values at exactly 0 or π/2? (Verify clamping works correctly)
- What happens when arctan2 receives (0, 0)? (Verify proper handling)
- How does vectorized peak finding handle empty or single-element arrays?

## Requirements

### Functional Requirements

- **FR-001**: System MUST maintain numerical accuracy within 1e-10 relative error vs baseline after all optimizations
- **FR-002**: System MUST pass all existing tests (`pytest tests/`) after each optimization phase
- **FR-003**: JIT-compiled functions MUST handle static and dynamic argument combinations correctly
- **FR-004**: Mutable default arguments MUST be eliminated from all active codebase files
- **FR-005**: Vectorized operations MUST produce identical results to loop-based implementations
- **FR-006**: Numerical stability fixes MUST prevent NaN/Inf for all valid input ranges
- **FR-007**: Memory usage MUST NOT increase more than 20% after optimizations

### Performance Requirements

- **PR-001**: Phase 1 optimizations MUST achieve ≥30x cumulative speedup
- **PR-002**: Phase 2 optimizations MUST achieve ≥5x additional speedup
- **PR-003**: JIT coverage MUST increase from 60% to >90%
- **PR-004**: All `.apply()` replacements MUST show ≥10x speedup per operation

### Key Files

| File | Lines | Priority | Changes |
|------|-------|----------|---------|
| model.py | 1,493 | High | JIT on _thin_film_guess, hoist residuals, arctan2 fix |
| multilayer.py | 985 | High | JIT on fallback path, autodiff Jacobian |
| physics.py | 1,449 | Medium | Phi clamping, safe_divide, autodiff Jacobian |
| DataSaver.py | 3,340 | Medium | Vectorize .apply(), mutable defaults |
| analysis.py | 965 | Medium | value_and_jacfwd caching |

## Success Criteria

### Measurable Outcomes

- **SC-001**: Phase 1 benchmark shows ≥30x speedup on multilayer calculations
- **SC-002**: Phase 2 benchmark shows ≥5x additional speedup on batch operations
- **SC-003**: All 600+ tests pass after each optimization phase
- **SC-004**: Zero mutable default argument linting errors
- **SC-005**: Zero NaN/Inf outputs on edge-case numerical inputs
- **SC-006**: JIT coverage metrics show >90% coverage on core functions
- **SC-007**: Memory profiling shows <20% increase from baseline

### Validation Gates

Before merging any optimization:

| Gate | Requirement |
|------|-------------|
| Tests | `pytest tests/` - all pass |
| Numerical | Max rel error < 1e-10 vs baseline |
| Performance | Measured improvement ≥ claimed speedup |
| Memory | Not increased >20% |

## Implementation Phases

### Phase 1: Critical Path Optimization
**Target**: 30-50x speedup

1. JIT multilayer.py fallback path (15-20x)
2. JIT _thin_film_guess (12-15x)
3. Hoist residual functions (10-12x)

### Phase 2: Secondary Optimizations
**Target**: 5-15x additional speedup

4. Fix mutable defaults (bug prevention)
5. Vectorize top 5 `.apply()` calls (10-100x per call)
6. Implement value_and_jacfwd (4x)

### Phase 3: Numerical Stability
**Target**: Bug prevention, no speedup

7. Fix arctan division (stability)
8. Add phi clamping (NaN prevention)

### Phase 4: Advanced Optimizations
**Target**: 10-50x on specific operations

9. Vectorize peak_prominences/peak_widths (10-50x)
10. Replace FD Jacobians with autodiff (2-3x)
11. Cache Savgol coefficients (1.8x)

## Already Optimized (No Action Needed)

| Component | Status | Notes |
|-----------|--------|-------|
| `_calc_ZL_jit` | ✅ Optimized | JIT with lax.fori_loop |
| `_layers_to_arrays` | ✅ Optimized | NumPy preallocation (10x) |
| `batch_analyze_vmap` | ✅ Optimized | vmap (199x documented) |
| `_grho_cache` | ✅ Optimized | Auto-invalidation cache |
| Shallow copies | ✅ Optimized | T049-T051 optimization |
| safe_divide | ✅ Optimized | Proper 1e-300 threshold |
