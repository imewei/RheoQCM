# ADR-001: QCMApp Decomposition

**Status**: Proposed
**Date**: 2026-03-27
**Deciders**: Wei
**Context**: 011-tech-debt-cleanup P3 (deferred)

## Context

`QCMApp` in `src/rheoQCM/rheoQCM.py` is a 7,954-line monolithic `QMainWindow`
subclass with 201 methods covering UI state, file I/O, data processing, plot
management, settings persistence, and solver orchestration.

This creates:

- **Untestability**: Zero unit test coverage for the main application class.
- **Signal/slot opacity**: 237 signal connections with no disconnection or registry.
- **Thread safety risk**: Worker threads interact with shared state on QCMApp.
- **Fragile widget access**: 141 `getattr(self.ui, ...)` calls that silently fail
  if widget names change in the Qt Designer file.

## Decision

Decompose QCMApp into focused service classes following the existing
`ServiceContainer` pattern (`src/rheoQCM/services/base.py`).

### Proposed Services

| Service | Responsibility | Est. Lines |
|---------|---------------|------------|
| `FileService` | Load/save/export data (HDF5, Excel, CSV) | ~800 |
| `PlotService` | Matplotlib widget lifecycle, prop plot grid | ~600 |
| `HarmonicManager` | Dynamic harmonic checkbox/tab creation and state | ~400 |
| `MechanicsService` | Solver orchestration, Bayesian fitting coordination | ~500 |
| `SettingsService` | Settings load/save, theme persistence | ~300 |
| `ExpertModeService` | Dynamic layer widget creation/deletion | ~300 |

### Extraction Strategy

Use the **Strangler Fig** pattern:

1. Extract one service at a time behind the existing API.
2. QCMApp delegates to the service; old methods become thin wrappers.
3. Add unit tests for the extracted service.
4. Remove the thin wrapper once all callers are migrated.

### Extraction Order (by risk/value)

1. **FileService** — highest test value, fewest signal dependencies.
2. **HarmonicManager** — encapsulates fragile `getattr()` patterns.
3. **PlotService** — isolates matplotlib lifecycle from main class.
4. **SettingsService** — low risk, enables settings unit testing.
5. **MechanicsService** — depends on core layer, moderate complexity.
6. **ExpertModeService** — dynamic widget management, highest UI coupling.

## Consequences

**Positive:**
- Each service is independently testable.
- Signal connections become explicit and auditable per service.
- Thread safety boundaries are clearly defined (services own their state).
- New developers can understand one service without reading 8,000 lines.

**Negative:**
- Extraction requires careful preservation of signal wiring.
- Temporary increase in indirection during migration.
- Risk of regressions in a GUI that has minimal test coverage.

**Mitigations:**
- Extract behind existing API (no public interface changes).
- Add integration smoke tests before each extraction.
- One service per PR with manual QA verification.

## Timeline

This is a P3 effort. Suggested pace: one service per development cycle,
starting after current feature work stabilises. No hard deadline.
