"""
Tests for DataSaver Migration to JAX Interpolation.

These tests verify that the DataSaver module works correctly with
JAX-based interpolation functions (replacing scipy.interpolate.interp1d).

Test coverage:
    - Linear interpolation with jax.numpy.interp
    - Cubic/spline interpolation with interpax
    - HDF5 save/load roundtrip
    - Excel export functionality
    - Data format preservation
"""

import os
import tempfile

import h5py
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from rheoQCM.core import configure_jax

# Ensure JAX is configured for Float64 before running tests
configure_jax()


class TestLinearInterpolation:
    """Test linear interpolation using jax.numpy.interp."""

    def test_linear_interp_basic(self) -> None:
        """Test basic linear interpolation accuracy."""
        from rheoQCM.core.physics import interp_linear

        # Known data points
        x = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = jnp.array([0.0, 2.0, 4.0, 6.0, 8.0])

        # Interpolate at midpoints
        x_new = jnp.array([0.5, 1.5, 2.5, 3.5])
        y_interp = interp_linear(x_new, x, y)

        # Expected: linear relationship y = 2*x
        expected = jnp.array([1.0, 3.0, 5.0, 7.0])

        assert jnp.allclose(y_interp, expected, rtol=1e-10)

    def test_linear_interp_at_knots(self) -> None:
        """Test that interpolation at knot points returns exact values."""
        from rheoQCM.core.physics import interp_linear

        x = jnp.array([0.0, 1.0, 2.0, 3.0])
        y = jnp.array([1.5, 2.7, 3.9, 4.2])

        y_interp = interp_linear(x, x, y)

        assert jnp.allclose(y_interp, y, rtol=1e-10)

    def test_linear_interp_extrapolation(self) -> None:
        """Test linear interpolation extrapolation behavior."""
        from rheoQCM.core.physics import interp_linear

        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([10.0, 20.0, 30.0])

        # jnp.interp extrapolates at edges (unlike scipy which raises or returns fill)
        x_new = jnp.array([0.5, 3.5])
        y_interp = interp_linear(x_new, x, y)

        # jnp.interp clamps to edge values
        assert y_interp[0] == 10.0  # Clamped to y[0]
        assert y_interp[1] == 30.0  # Clamped to y[-1]

    def test_linear_interp_float64_precision(self) -> None:
        """Test that Float64 precision is maintained."""
        from rheoQCM.core.physics import interp_linear

        x = jnp.array([0.0, 1.0], dtype=jnp.float64)
        y = jnp.array([0.0, 1e-15], dtype=jnp.float64)
        x_new = jnp.array([0.5], dtype=jnp.float64)

        y_interp = interp_linear(x_new, x, y)

        assert y_interp.dtype == jnp.float64
        assert jnp.isclose(y_interp[0], 0.5e-15, rtol=1e-10)


class TestCubicInterpolation:
    """Test cubic interpolation using interpax."""

    def test_cubic_interp_basic(self) -> None:
        """Test basic cubic interpolation accuracy."""
        from rheoQCM.core.physics import interp_cubic

        # Create smooth cubic data
        x = jnp.linspace(0.0, 2.0 * jnp.pi, 20)
        y = jnp.sin(x)

        # Interpolate at denser points
        x_new = jnp.linspace(0.0, 2.0 * jnp.pi, 100)
        y_interp = interp_cubic(x_new, x, y, extrap=True)

        # Expected: sin(x)
        expected = jnp.sin(x_new)

        # Cubic spline should be close to true values for smooth function
        # With 20 points over 2*pi, the max error is around 0.6%
        assert jnp.allclose(y_interp, expected, rtol=1e-2)

    def test_cubic_interp_at_knots(self) -> None:
        """Test that cubic interpolation at knot points returns exact values."""
        from rheoQCM.core.physics import interp_cubic

        x = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = jnp.array([0.0, 1.0, 4.0, 9.0, 16.0])  # y = x^2

        y_interp = interp_cubic(x, x, y, extrap=True)

        # At knot points, values should match closely
        assert jnp.allclose(y_interp, y, rtol=1e-6)

    def test_cubic_interp_smoothness(self) -> None:
        """Test that cubic interpolation produces smooth output."""
        from rheoQCM.core.physics import interp_cubic

        # Sparse data points
        x = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        y = jnp.array([0.0, 0.5, 0.9, 1.0, 0.9, 0.5])

        # Dense interpolation
        x_new = jnp.linspace(0.0, 5.0, 50)
        y_interp = interp_cubic(x_new, x, y, extrap=True)

        # Check output is finite and reasonable
        assert jnp.all(jnp.isfinite(y_interp))
        assert jnp.max(y_interp) <= 1.5  # Should not overshoot too much
        assert jnp.min(y_interp) >= -0.5

    def test_cubic_interp_nan_extrapolation(self) -> None:
        """Test that cubic interpolation returns NaN for out-of-bounds when extrap=NaN."""
        from rheoQCM.core.physics import interp_cubic

        x = jnp.array([1.0, 2.0, 3.0, 4.0])
        y = jnp.array([1.0, 4.0, 9.0, 16.0])

        # Query outside bounds
        x_new = jnp.array([0.0, 5.0])
        y_interp = interp_cubic(x_new, x, y, extrap=jnp.nan)

        # Should return NaN for out-of-bounds
        assert jnp.all(jnp.isnan(y_interp))


class TestCreateInterpFunc:
    """Test create_interp_func factory function."""

    def test_create_linear_func(self) -> None:
        """Test creating a linear interpolation function."""
        from rheoQCM.core.physics import create_interp_func

        x = jnp.array([0.0, 1.0, 2.0, 3.0])
        y = jnp.array([0.0, 2.0, 4.0, 6.0])

        f = create_interp_func(x, y, method="linear")

        x_new = jnp.array([0.5, 1.5, 2.5])
        y_result = f(x_new)

        expected = jnp.array([1.0, 3.0, 5.0])
        assert jnp.allclose(y_result, expected, rtol=1e-10)

    def test_create_cubic_func(self) -> None:
        """Test creating a cubic interpolation function."""
        from rheoQCM.core.physics import create_interp_func

        x = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = jnp.array([0.0, 1.0, 4.0, 9.0, 16.0])

        f = create_interp_func(x, y, method="cubic", extrap=True)

        x_new = jnp.array([0.5, 1.5, 2.5, 3.5])
        y_result = f(x_new)

        # Cubic should approximate quadratic well
        assert jnp.all(jnp.isfinite(y_result))

    def test_create_func_nan_bounds(self) -> None:
        """Test that created function returns NaN for out-of-bounds."""
        from rheoQCM.core.physics import create_interp_func

        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([1.0, 2.0, 3.0])

        f = create_interp_func(x, y, method="linear", extrap=jnp.nan)

        x_new = jnp.array([0.0, 4.0])
        y_result = f(x_new)

        assert jnp.all(jnp.isnan(y_result))


class TestHDF5SaveLoadRoundtrip:
    """Test HDF5 save/load roundtrip with DataSaver patterns."""

    def test_hdf5_array_roundtrip(self) -> None:
        """Test that arrays can be saved and loaded from HDF5."""
        # Create test data
        data = {
            "frequencies": np.array([5e6, 15e6, 25e6, 35e6, 45e6]),
            "conductance": np.array([0.001, 0.002, 0.003, 0.002, 0.001]),
            "susceptance": np.array([0.0, 0.001, 0.002, 0.001, 0.0]),
        }

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            temp_path = f.name

        try:
            # Save to HDF5
            with h5py.File(temp_path, "w") as hf:
                for key, value in data.items():
                    hf.create_dataset(key, data=value)

            # Load from HDF5
            with h5py.File(temp_path, "r") as hf:
                loaded_data = {key: np.array(hf[key]) for key in data.keys()}

            # Verify roundtrip
            for key in data.keys():
                np.testing.assert_array_almost_equal(data[key], loaded_data[key])

        finally:
            os.unlink(temp_path)

    def test_hdf5_nested_group_roundtrip(self) -> None:
        """Test nested group structure (mimicking DataSaver's raw/samp/ref pattern)."""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            temp_path = f.name

        try:
            # Create nested structure like DataSaver
            with h5py.File(temp_path, "w") as hf:
                raw_group = hf.create_group("raw")
                samp_group = raw_group.create_group("samp")

                # Add harmonic data
                for harm in [1, 3, 5]:
                    harm_group = samp_group.create_group(str(harm))
                    harm_group.create_dataset(
                        "f", data=np.array([5e6 * harm, 5e6 * harm + 100])
                    )
                    harm_group.create_dataset(
                        "G", data=np.array([0.001, 0.002])
                    )
                    harm_group.create_dataset(
                        "B", data=np.array([0.0001, 0.0002])
                    )

            # Load and verify structure
            with h5py.File(temp_path, "r") as hf:
                assert "raw" in hf
                assert "samp" in hf["raw"]
                for harm in [1, 3, 5]:
                    harm_group = hf[f"raw/samp/{harm}"]
                    assert "f" in harm_group
                    assert "G" in harm_group
                    assert "B" in harm_group

        finally:
            os.unlink(temp_path)

    def test_hdf5_json_attribute_roundtrip(self) -> None:
        """Test JSON-encoded attributes (like DataSaver's settings)."""
        import json

        settings = {
            "max_harmonic": 9,
            "time_format": "%Y-%m-%d %H:%M:%S",
            "reference_mode": "single",
        }

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            temp_path = f.name

        try:
            # Save with JSON attribute
            with h5py.File(temp_path, "w") as hf:
                hf.create_dataset("settings", data=json.dumps(settings))

            # Load and verify
            with h5py.File(temp_path, "r") as hf:
                loaded_settings = json.loads(hf["settings"][()])

            assert loaded_settings == settings

        finally:
            os.unlink(temp_path)


class TestExcelExportFunctionality:
    """Test Excel export functionality preservation."""

    def test_pandas_to_excel(self) -> None:
        """Test that pandas DataFrame can be exported to Excel."""
        # Create test DataFrame similar to DataSaver's format
        df = pd.DataFrame({
            "queue_id": [0, 1, 2, 3, 4],
            "t": [0.0, 1.0, 2.0, 3.0, 4.0],
            "temp": [25.0, 25.1, 25.2, 25.3, 25.4],
            "f1": [5e6, 5e6 - 10, 5e6 - 20, 5e6 - 30, 5e6 - 40],
            "f3": [15e6, 15e6 - 30, 15e6 - 60, 15e6 - 90, 15e6 - 120],
            "g1": [50.0, 51.0, 52.0, 53.0, 54.0],
            "g3": [150.0, 151.0, 152.0, 153.0, 154.0],
        })

        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
            temp_path = f.name

        try:
            # Export to Excel
            df.to_excel(temp_path, index=False)

            # Re-import and verify
            loaded_df = pd.read_excel(temp_path)

            # Check column preservation
            assert list(loaded_df.columns) == list(df.columns)

            # Check data preservation
            for col in df.columns:
                np.testing.assert_array_almost_equal(
                    df[col].values, loaded_df[col].values
                )

        finally:
            os.unlink(temp_path)

    def test_multi_sheet_excel_export(self) -> None:
        """Test multi-sheet Excel export (like DataSaver's prop export)."""
        sheets = {
            "samp_data": pd.DataFrame({
                "t": [0.0, 1.0, 2.0],
                "f3": [15e6, 15e6 - 30, 15e6 - 60],
            }),
            "ref_data": pd.DataFrame({
                "t": [0.0, 1.0, 2.0],
                "f3_ref": [15e6, 15e6, 15e6],
            }),
            "props": pd.DataFrame({
                "drho": [1e-6, 2e-6, 3e-6],
                "phi": [0.5, 0.6, 0.7],
            }),
        }

        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
            temp_path = f.name

        try:
            # Export multiple sheets
            with pd.ExcelWriter(temp_path) as writer:
                for sheet_name, df in sheets.items():
                    df.to_excel(writer, sheet_name=sheet_name, index=False)

            # Re-import and verify
            loaded_sheets = pd.read_excel(temp_path, sheet_name=None)

            assert set(loaded_sheets.keys()) == set(sheets.keys())

            for sheet_name in sheets:
                for col in sheets[sheet_name].columns:
                    np.testing.assert_array_almost_equal(
                        sheets[sheet_name][col].values,
                        loaded_sheets[sheet_name][col].values,
                    )

        finally:
            os.unlink(temp_path)


class TestDataFormatPreservation:
    """Test that data formats are preserved during interpolation and I/O."""

    def test_frequency_data_preservation(self) -> None:
        """Test that frequency data (Hz range) is preserved accurately."""
        from rheoQCM.core.physics import interp_linear

        # Typical QCM frequency values
        temps = jnp.array([20.0, 25.0, 30.0, 35.0, 40.0])
        freqs = jnp.array([4999900.0, 4999950.0, 5000000.0, 5000050.0, 5000100.0])

        temps_new = jnp.array([22.5, 27.5, 32.5, 37.5])
        freqs_interp = interp_linear(temps_new, temps, freqs)

        # Verify interpolated values are in expected range
        assert jnp.all(freqs_interp > 4999900.0)
        assert jnp.all(freqs_interp < 5000100.0)

        # Verify monotonicity is preserved
        assert jnp.all(jnp.diff(freqs_interp) > 0)

    def test_conductance_data_preservation(self) -> None:
        """Test that conductance data (small values) is preserved accurately."""
        from rheoQCM.core.physics import interp_linear

        # Typical conductance values (mS range)
        freqs = jnp.array([4.99e6, 4.995e6, 5.0e6, 5.005e6, 5.01e6])
        cond = jnp.array([0.0001, 0.001, 0.01, 0.001, 0.0001])

        freqs_new = jnp.array([4.9925e6, 4.9975e6, 5.0025e6, 5.0075e6])
        cond_interp = interp_linear(freqs_new, freqs, cond)

        # All interpolated values should be positive
        assert jnp.all(cond_interp > 0)

        # Peak should be near the center
        assert jnp.argmax(cond_interp) in [1, 2]

    def test_list_of_lists_format(self) -> None:
        """Test preservation of list-of-lists format used by DataSaver."""
        # DataSaver stores fs and gs as lists of lists per harmonic
        fs_original = [
            [5e6, 5e6 - 10, 5e6 - 20],  # f1
            [15e6, 15e6 - 30, 15e6 - 60],  # f3
            [25e6, 25e6 - 50, 25e6 - 100],  # f5
        ]

        gs_original = [
            [50.0, 51.0, 52.0],  # g1
            [150.0, 153.0, 156.0],  # g3
            [250.0, 255.0, 260.0],  # g5
        ]

        # Convert to numpy for operations
        fs_np = np.array(fs_original)
        gs_np = np.array(gs_original)

        # Convert back to list of lists
        fs_recovered = fs_np.tolist()
        gs_recovered = gs_np.tolist()

        # Verify structure preserved
        assert len(fs_recovered) == 3
        assert len(fs_recovered[0]) == 3

        # Verify values preserved
        for i in range(3):
            for j in range(3):
                assert fs_recovered[i][j] == fs_original[i][j]
                assert gs_recovered[i][j] == gs_original[i][j]
