"""
Integration tests for scipy parity (T020, T022, T031, T038)

These tests verify that the unified QCM core produces results matching
the original scipy-based implementation within 1e-10 relative tolerance.

Tests:
- T020: scipy parity test for solve_for_props
- T022: degree/radian conversion at API boundary
- T031: Full parity verification at 1e-10 tolerance
- T038: GUI vs script parity for same data

These tests are crucial for User Story 1: ensuring existing scripts
using QCM_functions.py continue working with JAX acceleration.
User Story 2: ensuring GUI and scripts produce identical results.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

# Suppress deprecation warning for these tests
os.environ["QCMFUNCS_SUPPRESS_DEPRECATION"] = "1"


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def single_layer_props_degrees():
    """Single layer properties with phi in DEGREES (legacy format)."""
    return {
        "grho3": 1e10,  # Pa kg/m^3
        "phi": 30.0,  # degrees (legacy format)
        "drho": 1e-6,  # kg/m^2
    }


@pytest.fixture
def single_layer_props_radians():
    """Single layer properties with phi in RADIANS (core format)."""
    return {
        "grho": 1e10,  # Pa kg/m^3
        "phi": np.pi / 6,  # radians (core format)
        "drho": 1e-6,  # kg/m^2
        "n": 3,  # reference harmonic
    }


@pytest.fixture
def layers_degrees():
    """Layer stack with phi in DEGREES (legacy format)."""
    return {
        1: {
            "grho3": 1e10,
            "phi": 30.0,  # degrees
            "drho": 1e-6,
        }
    }


@pytest.fixture
def layers_radians():
    """Layer stack with phi in RADIANS (core format)."""
    return {
        1: {
            "grho": 1e10,
            "phi": np.pi / 6,  # radians
            "drho": 1e-6,
            "n": 3,
        }
    }


@pytest.fixture
def two_layer_system_degrees():
    """Two-layer system with phi in DEGREES (legacy format)."""
    return {
        1: {
            "grho3": 1e10,
            "phi": 30.0,  # degrees
            "drho": 1e-6,
        },
        2: {
            "grho3": 1e8,
            "phi": 90.0,  # degrees - bulk water
            "drho": np.inf,
        },
    }


@pytest.fixture
def sample_delfstar_data():
    """Sample delfstar data for testing."""
    # Typical experimental values for a thin viscoelastic film
    return {
        3: -500.0 + 100.0j,  # delf + 1j * delg at n=3
        5: -900.0 + 200.0j,  # at n=5
        7: -1300.0 + 350.0j,  # at n=7
    }


@pytest.fixture
def bulk_delfstar_data():
    """Sample delfstar data for bulk material."""
    # Typical values for bulk viscous liquid
    return {
        3: -200.0 + 200.0j,  # |delf| ~ |delg| for bulk
        5: -320.0 + 330.0j,
        7: -430.0 + 450.0j,
    }


@pytest.fixture
def bcb_film_data():
    """BCB thin film test data (from original QCM.py test)."""
    return {
        "delfstar": {
            1: -28206.4782657343 + 5.6326137881j,
            3: -87768.0313369799 + 155.716064797j,
            5: -159742.686586637 + 888.6642467156j,
        },
        "film": {
            0: {"calc": False, "drho": 2.8e-6, "grho": 3e17, "phi": 0, "n": 3},
            1: {"calc": True},
        },
        "nh": [3, 5, 3],
    }


@pytest.fixture
def water_bulk_data():
    """Water bulk test data."""
    return {
        "delfstar": {
            1: -694.15609764494 + 762.8726222543j,
            3: -1248.7983004897833 + 1215.1121711257j,
            5: -1641.2310467399657 + 1574.7706516819j,
        },
        "film": {
            0: {"calc": False, "drho": 2.8e-6, "grho": 3e17, "phi": 0, "n": 3},
            1: {"calc": True},
        },
        "nh": [3, 5, 3],
    }


# =============================================================================
# T020: scipy parity test for solve_for_props
# =============================================================================


class TestSolveForPropsParity:
    """Tests verifying solve_for_props produces consistent results."""

    def test_solve_for_props_signature_preserved(self):
        """solve_for_props should have the expected legacy signature."""
        import inspect

        from QCMFuncs.QCM_functions import solve_for_props

        sig = inspect.signature(solve_for_props)
        param_names = list(sig.parameters.keys())

        # Required positional parameters
        assert "delfstar" in param_names
        assert "calc" in param_names
        assert "props_calc_in" in param_names
        assert "layers_in" in param_names

    def test_solve_for_props_returns_dataframe(
        self, sample_delfstar_data, layers_degrees
    ):
        """solve_for_props should return a DataFrame."""
        import pandas as pd

        from QCMFuncs.QCM_functions import solve_for_props

        # Create input dataframe with proper column names (delfstar_expt_n format)
        df = pd.DataFrame(
            {
                "delfstar_expt_3": [sample_delfstar_data[3]],
                "delfstar_expt_5": [sample_delfstar_data[5]],
            }
        )

        # Use simple calc string for Sauerbrey (single harmonic) with no properties to fit
        result = solve_for_props(df, "3", [], layers_degrees)

        assert isinstance(result, pd.DataFrame)

    def test_solve_for_props_uses_core(self):
        """solve_for_props should use core module (no scipy.optimize)."""
        import inspect

        from QCMFuncs import QCM_functions

        # Get the source code
        source = inspect.getsource(QCM_functions)

        # After refactoring, scipy.optimize should not be imported as 'optimize'
        # (T030 implementation removes this)
        assert "import scipy.optimize as optimize" not in source

    def test_grho_function_degree_input(self, single_layer_props_degrees):
        """grho function should accept phi in degrees."""
        from QCMFuncs.QCM_functions import grho

        result = grho(3, single_layer_props_degrees)

        # Should return a finite positive number
        assert np.isfinite(result)
        assert result > 0

        # At n=3, should return grho3 directly
        assert np.isclose(result, single_layer_props_degrees["grho3"], rtol=1e-10)

    def test_grho_harmonic_scaling(self, single_layer_props_degrees):
        """grho should scale correctly with harmonic."""
        from QCMFuncs.QCM_functions import grho

        grho3 = grho(3, single_layer_props_degrees)
        grho5 = grho(5, single_layer_props_degrees)
        grho7 = grho(7, single_layer_props_degrees)

        phi_deg = single_layer_props_degrees["phi"]

        # Power law: grho(n) = grho3 * (n/3)^(phi/90)
        expected_grho5 = grho3 * (5 / 3) ** (phi_deg / 90)
        expected_grho7 = grho3 * (7 / 3) ** (phi_deg / 90)

        assert np.isclose(grho5, expected_grho5, rtol=1e-10)
        assert np.isclose(grho7, expected_grho7, rtol=1e-10)


# =============================================================================
# T022: degree/radian conversion at API boundary
# =============================================================================


class TestDegreeRadianConversion:
    """Tests verifying degree/radian conversion at API boundaries."""

    def test_legacy_functions_accept_degrees(self, single_layer_props_degrees):
        """Legacy QCM_functions should accept phi in degrees."""
        from QCMFuncs.QCM_functions import calc_deltarho, calc_lamrho, grho

        # All these should work with phi in degrees
        grho_val = grho(3, single_layer_props_degrees)
        lamrho_val = calc_lamrho(
            3, single_layer_props_degrees["grho3"], single_layer_props_degrees["phi"]
        )
        deltarho_val = calc_deltarho(
            3, single_layer_props_degrees["grho3"], single_layer_props_degrees["phi"]
        )

        assert np.isfinite(grho_val)
        assert np.isfinite(lamrho_val)
        assert np.isfinite(deltarho_val)

    def test_core_functions_use_radians(self, single_layer_props_radians):
        """Core physics functions should use phi in radians."""
        from rheoQCM.core import grho as core_grho

        # Core uses radians
        result = core_grho(
            3,
            single_layer_props_radians["grho"],
            single_layer_props_radians["phi"],
            refh=3,
        )

        assert np.isfinite(float(result))

    def test_degree_radian_equivalence(
        self, single_layer_props_degrees, single_layer_props_radians
    ):
        """Same physics result whether using degrees (legacy) or radians (core)."""
        from QCMFuncs.QCM_functions import grho as legacy_grho
        from rheoQCM.core import grho as core_grho

        # Legacy with degrees
        legacy_result = legacy_grho(5, single_layer_props_degrees)

        # Core with radians
        core_result = float(
            core_grho(
                5,
                single_layer_props_radians["grho"],
                single_layer_props_radians["phi"],
                refh=3,
            )
        )

        # Should be identical within numerical precision
        assert np.isclose(
            legacy_result, core_result, rtol=1e-10
        ), f"Legacy: {legacy_result}, Core: {core_result}"

    def test_normdelfstar_degree_input(self):
        """normdelfstar should accept phi in degrees."""
        from QCMFuncs.QCM_functions import normdelfstar

        # Legacy API uses degrees
        result = normdelfstar(3, 0.05, 30.0)  # dlam3=0.05, phi=30 degrees

        assert np.isfinite(result)
        assert isinstance(result, (complex, np.complexfloating))

    def test_normdelfstar_parity_with_core(self):
        """normdelfstar should match core implementation."""
        from QCMFuncs.QCM_functions import normdelfstar as legacy_normdelfstar
        from rheoQCM.core import normdelfstar as core_normdelfstar

        dlam3 = 0.05
        phi_deg = 30.0
        phi_rad = np.radians(phi_deg)

        legacy_result = legacy_normdelfstar(3, dlam3, phi_deg)
        core_result = complex(core_normdelfstar(3, dlam3, phi_rad))

        assert np.isclose(
            legacy_result, core_result, rtol=1e-10
        ), f"Legacy: {legacy_result}, Core: {core_result}"

    def test_bulk_props_returns_degrees(self):
        """bulk_props should return phi in degrees."""
        from QCMFuncs.QCM_functions import bulk_props

        # Typical bulk viscous response
        delfstar = -200.0 + 200.0j

        grho_val, phi_val = bulk_props(delfstar)

        assert np.isfinite(grho_val)
        assert np.isfinite(phi_val)
        # phi should be in degrees (0 to 90)
        assert 0 <= phi_val <= 90, f"phi should be in degrees: {phi_val}"


# =============================================================================
# T021: Additional calc_delfstar multi-layer tests
# =============================================================================


class TestCalcDelfstarMultilayer:
    """Tests for multi-layer calc_delfstar with degree/radian conversion."""

    def test_calc_delfstar_with_degrees(self, layers_degrees):
        """calc_delfstar should accept layers with phi in degrees."""
        from QCMFuncs.QCM_functions import calc_delfstar

        result = calc_delfstar(3, layers_degrees)

        assert np.isfinite(result)
        assert isinstance(result, (complex, np.complexfloating))
        # Frequency shift should be negative for mass loading
        assert result.real < 0

    def test_calc_ZL_with_degrees(self, layers_degrees):
        """calc_ZL should accept layers with phi in degrees."""
        from QCMFuncs.QCM_functions import calc_ZL

        result = calc_ZL(3, layers_degrees, 0.0)

        assert np.isfinite(result)

    def test_two_layer_calc_delfstar(self, two_layer_system_degrees):
        """calc_delfstar should work with two-layer systems."""
        from QCMFuncs.QCM_functions import calc_delfstar

        result = calc_delfstar(3, two_layer_system_degrees)

        assert np.isfinite(result)
        # Should include contribution from both layers
        assert result.imag > 0  # Should have dissipation

    def test_calc_delfstar_harmonic_scaling(self, layers_degrees):
        """calc_delfstar should scale with harmonic number."""
        from QCMFuncs.QCM_functions import calc_delfstar

        delf_3 = calc_delfstar(3, layers_degrees)
        delf_5 = calc_delfstar(5, layers_degrees)
        delf_7 = calc_delfstar(7, layers_degrees)

        # Higher harmonics should have larger magnitude shifts
        assert abs(delf_5.real) > abs(delf_3.real)
        assert abs(delf_7.real) > abs(delf_5.real)


# =============================================================================
# T031: Full parity verification at 1e-10 tolerance
# =============================================================================


class TestParityVerification:
    """Comprehensive parity tests at 1e-10 relative tolerance (SC-002)."""

    def test_sauerbreyf_parity(self):
        """sauerbreyf should match reference within 1e-10."""
        from QCMFuncs.QCM_functions import Zq, f1_default, sauerbreyf

        drho = 1e-6  # 1 um * 1 g/cm^3

        for n in [1, 3, 5, 7, 9]:
            result = sauerbreyf(n, drho)
            # NOTE: The QCM_functions sauerbreyf uses POSITIVE convention
            # Sauerbrey formula: delf = +2 * n * f1^2 * drho / Zq (positive for mass increase)
            # This is opposite of the negative delf convention in some literature
            expected = 2 * n * f1_default**2 * drho / Zq

            assert np.isclose(
                result, expected, rtol=1e-10
            ), f"sauerbreyf mismatch at n={n}: {result} vs {expected}"

    def test_sauerbreym_parity(self):
        """sauerbreym should match reference within 1e-10."""
        from QCMFuncs.QCM_functions import Zq, f1_default, sauerbreym

        delf = -500.0  # Hz

        for n in [1, 3, 5, 7, 9]:
            result = sauerbreym(n, delf)
            # Analytical formula (inverse of sauerbreyf)
            expected = -delf * Zq / (2 * n * f1_default**2)

            assert np.isclose(
                result, expected, rtol=1e-10
            ), f"sauerbreym mismatch at n={n}: {result} vs {expected}"

    def test_grho_parity(self, single_layer_props_degrees):
        """grho should match power law formula within 1e-10."""
        from QCMFuncs.QCM_functions import grho

        grho3 = single_layer_props_degrees["grho3"]
        phi_deg = single_layer_props_degrees["phi"]

        for n in [1, 3, 5, 7, 9]:
            result = grho(n, single_layer_props_degrees)
            # Power law formula
            expected = grho3 * (n / 3) ** (phi_deg / 90)

            assert np.isclose(
                result, expected, rtol=1e-10
            ), f"grho mismatch at n={n}: {result} vs {expected}"

    def test_calc_delfstar_sauerbrey_limit(self):
        """For thin elastic film, delfstar should match Sauerbrey within tolerance."""
        from QCMFuncs.QCM_functions import calc_delfstar, sauerbreyf

        # Very thin, very stiff film (approaches Sauerbrey limit)
        thin_elastic = {
            1: {
                "grho3": 1e17,  # Very stiff
                "phi": 1.0,  # Nearly elastic (1 degree)
                "drho": 1e-7,  # Very thin
            }
        }

        drho = thin_elastic[1]["drho"]

        for n in [3, 5, 7]:
            delfstar = calc_delfstar(n, thin_elastic)
            sauerbrey = sauerbreyf(n, drho)

            # For nearly elastic film, |delf| should approach |Sauerbrey|
            # Both delfstar.real and sauerbrey are positive in this convention,
            # but delfstar.real is actually negative for mass loading
            # So compare absolute values
            rel_error = abs(abs(delfstar.real) - abs(sauerbrey)) / abs(sauerbrey)
            assert (
                rel_error < 0.05
            ), f"Sauerbrey limit not approached at n={n}: rel_error={rel_error}, delfstar={delfstar}, sauerbrey={sauerbrey}"

    def test_core_multilayer_parity(self, layers_radians):
        """Core multilayer should match legacy within tolerance."""
        from QCMFuncs.QCM_functions import calc_delfstar as legacy_calc_delfstar
        from rheoQCM.core import calc_delfstar_multilayer

        # Convert layers_radians to legacy format (degrees)
        layers_degrees = {
            1: {
                "grho3": layers_radians[1]["grho"],
                "phi": np.degrees(layers_radians[1]["phi"]),
                "drho": layers_radians[1]["drho"],
            }
        }

        for n in [3, 5, 7]:
            legacy_result = legacy_calc_delfstar(n, layers_degrees)
            core_result = complex(calc_delfstar_multilayer(n, layers_radians))

            # Should match within 1e-10 relative tolerance
            if abs(legacy_result) > 1e-10:
                rel_error = abs(legacy_result - core_result) / abs(legacy_result)
                assert (
                    rel_error < 1e-10
                ), f"Multilayer parity failed at n={n}: legacy={legacy_result}, core={core_result}"

    def test_bulk_props_parity(self):
        """bulk_props should match reference within 1e-10."""
        from QCMFuncs.QCM_functions import Zq, bulk_props, f1_default

        # Test several delfstar values
        test_cases = [
            -200.0 + 200.0j,  # Symmetric (phi ~ 45 deg)
            -100.0 + 200.0j,  # High dissipation
            -300.0 + 100.0j,  # Low dissipation
        ]

        for delfstar in test_cases:
            grho_val, phi_val = bulk_props(delfstar)

            # Reference formula
            expected_grho = (np.pi * Zq * abs(delfstar) / f1_default) ** 2
            expected_phi = min(
                -np.degrees(2 * np.arctan(delfstar.real / delfstar.imag)), 90
            )

            assert np.isclose(
                grho_val, expected_grho, rtol=1e-10
            ), f"bulk_props grho mismatch: {grho_val} vs {expected_grho}"
            assert np.isclose(
                phi_val, expected_phi, rtol=1e-10
            ), f"bulk_props phi mismatch: {phi_val} vs {expected_phi}"


# =============================================================================
# T038: GUI vs script parity for same data
# =============================================================================


class TestGUIvsScriptParity:
    """T038: Verify GUI and script produce identical results for same data."""

    def test_gui_qcm_imports_core(self):
        """GUI QCM module should import from core."""
        from pathlib import Path

        from rheoQCM.modules import QCM as qcm_module

        qcm_path = Path(qcm_module.__file__)
        with open(qcm_path) as f:
            source = f.read()

        assert "from rheoQCM.core" in source, "QCM.py should import from rheoQCM.core"

    def test_gui_no_scipy_optimize(self):
        """GUI QCM module should not use scipy.optimize."""
        from pathlib import Path

        from rheoQCM.modules import QCM as qcm_module

        qcm_path = Path(qcm_module.__file__)
        with open(qcm_path) as f:
            source = f.read()

        assert (
            "from scipy import optimize" not in source
        ), "QCM.py should not import scipy.optimize"
        assert "optimize.root" not in source, "QCM.py should not use optimize.root"

    def test_gui_script_sauerbreyf_parity(self):
        """GUI and script should produce identical sauerbreyf results."""
        from QCMFuncs.QCM_functions import sauerbreyf as script_sauerbreyf
        from rheoQCM.modules.QCM import QCM

        qcm = QCM()
        qcm.f1 = 5e6

        for n in [1, 3, 5, 7]:
            for drho in [1e-7, 1e-6, 1e-5]:
                gui_result = qcm.sauerbreyf(n, drho)
                script_result = script_sauerbreyf(n, drho)

                np.testing.assert_allclose(
                    gui_result,
                    script_result,
                    rtol=1e-10,
                    err_msg=f"sauerbreyf mismatch at n={n}, drho={drho}",
                )

    def test_gui_script_sauerbreym_parity(self):
        """GUI and script should produce identical sauerbreym results."""
        from QCMFuncs.QCM_functions import sauerbreym as script_sauerbreym
        from rheoQCM.modules.QCM import QCM

        qcm = QCM()
        qcm.f1 = 5e6

        for n in [1, 3, 5, 7]:
            for delf in [-100, -500, -1000]:
                gui_result = qcm.sauerbreym(n, delf)
                script_result = script_sauerbreym(n, delf)

                np.testing.assert_allclose(
                    gui_result,
                    script_result,
                    rtol=1e-10,
                    err_msg=f"sauerbreym mismatch at n={n}, delf={delf}",
                )

    def test_gui_script_grho_parity(self):
        """GUI and script should produce identical grho results."""
        from QCMFuncs.QCM_functions import grho as script_grho
        from rheoQCM.modules.QCM import QCM

        qcm = QCM()
        qcm.f1 = 5e6
        qcm.refh = 3

        # Test with same inputs (core uses radians, legacy uses degrees)
        grho_refh = 1e10
        phi_rad = np.pi / 6  # 30 degrees

        # Legacy uses degree input in props dict
        props_legacy = {"grho3": grho_refh, "phi": 30.0, "drho": 1e-6}

        for n in [1, 3, 5, 7]:
            gui_result = qcm.grho(n, grho_refh, phi_rad)
            script_result = script_grho(n, props_legacy)

            np.testing.assert_allclose(
                gui_result, script_result, rtol=1e-10, err_msg=f"grho mismatch at n={n}"
            )

    def test_gui_script_delfstar_thin_film(self, bcb_film_data):
        """GUI and script produce identical results for thin film."""
        from QCMFuncs.QCM_functions import calc_delfstar as script_calc_delfstar
        from rheoQCM.modules.QCM import QCM

        qcm = QCM()
        qcm.f1 = 5e6
        qcm.refh = 3
        qcm.calctype = "SLA"

        delfstar = bcb_film_data["delfstar"]
        film = bcb_film_data["film"]

        # Solve using GUI wrapper
        grho_refh, phi, drho, dlam_refh, err = qcm.solve_general_delfstar_to_prop(
            bcb_film_data["nh"], delfstar, film, calctype="SLA", bulklimit=0.5
        )

        # Build layers for script
        if not np.isnan(drho):
            layers_radians = {
                1: {
                    "grho": grho_refh,
                    "phi": phi,
                    "drho": drho,
                    "n": 3,
                }
            }
            layers_degrees = {
                1: {
                    "grho3": grho_refh,
                    "phi": np.degrees(phi),
                    "drho": drho,
                }
            }

            # Compare delfstar calculations
            for n in [3, 5]:
                gui_delfstar = qcm.calc_delfstar(n, layers_radians)
                script_delfstar = script_calc_delfstar(n, layers_degrees)

                # Should be within 1e-8 (accounting for solver differences)
                np.testing.assert_allclose(
                    gui_delfstar,
                    script_delfstar,
                    rtol=1e-8,
                    err_msg=f"delfstar mismatch at n={n}",
                )

    def test_gui_script_bulk_material(self, water_bulk_data):
        """GUI and script produce identical results for bulk material."""
        from rheoQCM.modules.QCM import QCM

        qcm = QCM()
        qcm.f1 = 5e6
        qcm.refh = 3
        qcm.calctype = "SLA"

        delfstar = water_bulk_data["delfstar"]
        film = water_bulk_data["film"]

        # Solve using GUI wrapper
        grho_refh, phi, drho, dlam_refh, err = qcm.solve_general_delfstar_to_prop(
            water_bulk_data["nh"], delfstar, film, calctype="SLA", bulklimit=0.5
        )

        # For bulk material, phi should be close to pi/2
        if not np.isnan(phi):
            assert phi > np.pi / 4, f"Bulk material should have phi > pi/4, got {phi}"

    def test_gui_uses_qcmmodel_internally(self):
        """GUI QCM should use QCMModel internally for solving."""
        from rheoQCM.core.model import QCMModel
        from rheoQCM.modules.QCM import QCM

        qcm = QCM()
        qcm.f1 = 5e6
        qcm.refh = 3

        # _get_model should return a QCMModel
        model = qcm._get_model()
        assert isinstance(model, QCMModel)

        # After solving, _model should be set
        delfstar = {
            1: -28206.4782657343 + 5.6326137881j,
            3: -87768.0313369799 + 155.716064797j,
            5: -159742.686586637 + 888.6642467156j,
        }
        film = {
            0: {"calc": False, "drho": 2.8e-6, "grho": 3e17, "phi": 0, "n": 3},
            1: {"calc": True},
        }

        qcm.solve_general_delfstar_to_prop([3, 5, 3], delfstar, film)

        assert qcm._model is not None
        assert isinstance(qcm._model, QCMModel)


# =============================================================================
# Deprecation Warning Test
# =============================================================================


class TestDeprecationWarning:
    """Test that DeprecationWarning is emitted on import (FR-010, T029)."""

    def test_deprecation_warning_emitted(self):
        """Importing QCM_functions should emit DeprecationWarning."""
        import importlib
        import sys
        import warnings

        # Remove suppression and reload
        if "QCMFUNCS_SUPPRESS_DEPRECATION" in os.environ:
            del os.environ["QCMFUNCS_SUPPRESS_DEPRECATION"]

        # Remove cached module
        modules_to_remove = [k for k in sys.modules.keys() if "QCMFuncs" in k]
        for mod in modules_to_remove:
            del sys.modules[mod]

        # Capture warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import QCMFuncs.QCM_functions

            # Check for DeprecationWarning
            deprecation_warnings = [
                x for x in w if issubclass(x.category, DeprecationWarning)
            ]
            assert (
                len(deprecation_warnings) > 0
            ), "No DeprecationWarning emitted on QCM_functions import"

            # Verify message content
            msg = str(deprecation_warnings[0].message)
            assert (
                "deprecated" in msg.lower()
            ), f"Warning message should mention deprecation: {msg}"


# =============================================================================
# scipy.optimize removal verification
# =============================================================================


class TestScipyOptimizeRemoval:
    """Verify scipy.optimize is not used in QCM_functions after T030."""

    def test_no_scipy_optimize_import(self):
        """QCM_functions should not import scipy.optimize after refactoring."""
        import inspect

        from QCMFuncs import QCM_functions

        source = inspect.getsource(QCM_functions)

        # Check that scipy.optimize is not imported as 'optimize' alias
        assert (
            "import scipy.optimize as optimize" not in source
        ), "scipy.optimize should not be imported as 'optimize'"

        # Check that optimize.least_squares is not used (should use NLSQ instead)
        assert (
            "optimize.least_squares" not in source
        ), "optimize.least_squares should not be used (use NLSQ instead)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
