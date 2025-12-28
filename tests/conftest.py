"""
Shared pytest fixtures and configuration for RheoQCM test suite.

This module provides:
    - JAX configuration for Float64 precision
    - Common fixtures for physics calculations
    - HDF5 test data loading fixtures
    - Temporary file management
"""

from pathlib import Path
from typing import Any

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rheoQCM.core import configure_jax

# Configure JAX for Float64 precision at module load
configure_jax()


# =============================================================================
# Path Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Return path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def bcb_h5_path(fixtures_dir: Path) -> Path:
    """Return path to BCB thin film test data."""
    path = fixtures_dir / "bcb_4.h5"
    if not path.exists():
        pytest.skip(f"Test fixture not found: {path}")
    return path


@pytest.fixture(scope="session")
def water_h5_path(fixtures_dir: Path) -> Path:
    """Return path to water (bulk) test data."""
    path = fixtures_dir / "water.h5"
    if not path.exists():
        pytest.skip(f"Test fixture not found: {path}")
    return path


@pytest.fixture(scope="session")
def polymer_h5_path(fixtures_dir: Path) -> Path:
    """Return path to polymer test data."""
    path = fixtures_dir / "polymer.h5"
    if not path.exists():
        pytest.skip(f"Test fixture not found: {path}")
    return path


# =============================================================================
# Physics Fixtures
# =============================================================================


@pytest.fixture
def standard_harmonics() -> list[int]:
    """Return standard QCM harmonic numbers."""
    return [1, 3, 5, 7, 9]


@pytest.fixture
def jax_harmonics() -> jnp.ndarray:
    """Return standard QCM harmonics as JAX array."""
    return jnp.array([1, 3, 5, 7, 9])


@pytest.fixture
def bcb_thin_film_delfstars() -> dict[int, complex]:
    """Return known delfstar values for BCB thin film test case.

    This is a well-characterized test case used across multiple tests.
    """
    return {
        1: -28206.4782657343 + 5.6326137881j,
        3: -87768.0313369799 + 155.716064797j,
        5: -159742.686586637 + 888.6642467156j,
    }


@pytest.fixture
def water_bulk_delfstars() -> dict[int, complex]:
    """Return known delfstar values for water (bulk material) test case.

    Water has high dissipation ratio (delg/delf close to 1).
    """
    return {
        1: -694.15609764494 + 762.8726222543j,
        3: -1248.7983004897833 + 1215.1121711257j,
        5: -1641.2310467399657 + 1574.7706516819j,
    }


@pytest.fixture
def default_qcm_params() -> dict[str, Any]:
    """Return default QCM parameters."""
    return {
        "f1": 5e6,  # Fundamental frequency in Hz
        "refh": 3,  # Reference harmonic
        "Zq": 8.84e6,  # AT-cut quartz acoustic impedance
        "calctype": "SLA",  # Calculation type
    }


@pytest.fixture
def thin_film_properties() -> dict[str, Any]:
    """Return typical thin film properties for testing."""
    return {
        "grho_refh": 1e10,  # Pa kg/m^3
        "phi": jnp.pi / 4,  # radians (45 degrees)
        "drho": 1e-6,  # kg/m^2
    }


# =============================================================================
# Model Fixtures
# =============================================================================


@pytest.fixture
def qcm_model():
    """Return initialized QCMModel instance."""
    from rheoQCM.core.model import QCMModel

    return QCMModel(f1=5e6, refh=3)


@pytest.fixture
def qcm_analyzer():
    """Return initialized QCMAnalyzer instance."""
    from rheoQCM.core.analysis import QCMAnalyzer

    return QCMAnalyzer(f1=5e6, refh=3)


@pytest.fixture
def qcm_wrapper():
    """Return initialized QCM wrapper instance."""
    from rheoQCM.modules.QCM import QCM

    qcm = QCM()
    qcm.f1 = 5e6
    qcm.refh = 3
    qcm.calctype = "SLA"
    return qcm


# =============================================================================
# Temporary File Fixtures
# =============================================================================


@pytest.fixture
def temp_h5_file(tmp_path: Path) -> Path:
    """Return path to a temporary HDF5 file."""
    return tmp_path / "test_data.h5"


@pytest.fixture
def temp_excel_file(tmp_path: Path) -> Path:
    """Return path to a temporary Excel file."""
    return tmp_path / "test_data.xlsx"


# =============================================================================
# JAX Configuration Verification
# =============================================================================


@pytest.fixture(autouse=True)
def verify_float64_enabled():
    """Verify Float64 is enabled for all tests.

    This runs automatically before each test to ensure consistent precision.
    """
    # Check that Float64 is enabled
    test_val = jnp.array(1.0)
    assert (
        test_val.dtype == jnp.float64
    ), "JAX Float64 not enabled. Tests require x64 precision."
    yield


# =============================================================================
# Pytest Configuration
# =============================================================================


def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks as integration test")
    config.addinivalue_line("markers", "gui: marks tests requiring PyQt6")
    config.addinivalue_line(
        "markers", "hardware: marks tests requiring hardware (NI DAQ, myVNA)"
    )
    config.addinivalue_line("markers", "physics: marks physics core tests")
    config.addinivalue_line("markers", "model: marks model layer tests")
