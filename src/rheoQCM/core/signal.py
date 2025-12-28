"""
Signal Processing Functions for JAX

This module provides JAX-based implementations of signal processing functions,
replacing scipy.signal equivalents while maintaining API compatibility.

Functions:
    - find_peaks: Find peaks in a 1D signal with optional constraints
    - peak_prominences: Calculate peak prominences
    - peak_widths: Calculate peak widths at relative height

Note:
    These functions use a hybrid numpy/JAX approach. Variable-length outputs
    (like peak indices) use numpy for control flow, while numerical computations
    use JAX arrays for compatibility with the JAX ecosystem.

    Peak detection functions are NOT JIT-compatible due to variable-length outputs.
    This is acceptable for the QCM analysis use case where peak detection is not
    in a performance-critical inner loop.
"""

import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike

# =============================================================================
# Type Aliases
# =============================================================================

# Input types
Signal = ArrayLike  # 1D array-like of float64
PeakIndices = jnp.ndarray  # 1D array of int

# Output types - tuples matching scipy signatures
Prominences = tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
# (prominences, left_bases, right_bases)

Widths = tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
# (widths, width_heights, left_ips, right_ips)

# Properties dictionary (scipy compatibility)
PeakProperties = dict[str, jnp.ndarray]


# =============================================================================
# Validation Helpers
# =============================================================================


def _validate_signal(x: ArrayLike, min_length: int = 0) -> np.ndarray:
    """
    Validate and convert input signal to numpy array.

    Parameters
    ----------
    x : ArrayLike
        Input signal.
    min_length : int
        Minimum required length. Default: 0.

    Returns
    -------
    x_np : np.ndarray
        Validated numpy array (float64).

    Raises
    ------
    ValueError
        If signal is not 1D or too short.
    """
    x_np = np.asarray(x, dtype=np.float64)
    if x_np.ndim != 1:
        raise ValueError(f"Signal must be 1D, got {x_np.ndim}D")
    if min_length > 0 and len(x_np) < min_length:
        raise ValueError(
            f"Signal must have at least {min_length} points, got {len(x_np)}"
        )
    return x_np


def _validate_peaks(peaks: ArrayLike, signal_length: int) -> np.ndarray:
    """
    Validate peak indices.

    Parameters
    ----------
    peaks : ArrayLike
        Peak indices.
    signal_length : int
        Length of the signal array.

    Returns
    -------
    peaks_np : np.ndarray
        Validated numpy array of peak indices.

    Raises
    ------
    ValueError
        If peaks are out of bounds or not sorted.
    """
    peaks_np = np.asarray(peaks, dtype=np.intp)
    if peaks_np.ndim != 1:
        raise ValueError(f"Peaks must be 1D, got {peaks_np.ndim}D")
    if len(peaks_np) > 0:
        if peaks_np.min() < 0 or peaks_np.max() >= signal_length:
            raise ValueError(f"Peak indices must be in [0, {signal_length - 1}]")
        if not np.all(np.diff(peaks_np) > 0):
            raise ValueError("Peak indices must be strictly increasing")
    return peaks_np


def _normalize_constraint(
    value: float | tuple[float, float] | None,
) -> tuple[float, float] | None:
    """
    Normalize constraint to (min, max) tuple.

    Parameters
    ----------
    value : float, tuple, or None
        Constraint value.

    Returns
    -------
    tuple or None
        (min, max) tuple or None if value is None.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return (float(value), float("inf"))
    return (float(value[0]), float(value[1]))


# =============================================================================
# Helper Functions
# =============================================================================


def _find_local_maxima(x: np.ndarray) -> np.ndarray:
    """
    Find indices of local maxima in a 1D array.

    A point is a local maximum if it is strictly greater than its neighbors.
    Flat regions (plateaus) are NOT detected as peaks.

    Parameters
    ----------
    x : np.ndarray
        1D input array.

    Returns
    -------
    indices : np.ndarray
        Indices of local maxima.
    """
    if len(x) < 3:
        return np.array([], dtype=np.intp)

    # Compare each point to its neighbors
    greater_than_left = x[1:-1] > x[:-2]
    greater_than_right = x[1:-1] > x[2:]
    is_peak = greater_than_left & greater_than_right

    # Get indices (add 1 to account for slicing)
    return np.where(is_peak)[0] + 1


def _select_by_distance(
    peaks: np.ndarray, heights: np.ndarray, distance: float
) -> np.ndarray:
    """
    Filter peaks by minimum distance, keeping the highest in each cluster.

    Parameters
    ----------
    peaks : np.ndarray
        Peak indices.
    heights : np.ndarray
        Peak heights (signal values at peak indices).
    distance : float
        Minimum distance between peaks (in samples).

    Returns
    -------
    selected : np.ndarray
        Indices of selected peaks (subset of input peaks).
    """
    if len(peaks) <= 1:
        return peaks

    # Sort peaks by height (descending) to prioritize higher peaks
    priority = np.argsort(-heights)
    keep = np.ones(len(peaks), dtype=bool)

    for i in priority:
        if not keep[i]:
            continue
        # Remove peaks within distance of this peak
        for j in range(len(peaks)):
            if i != j and keep[j]:
                if abs(peaks[j] - peaks[i]) < distance:
                    keep[j] = False

    return peaks[keep]


# =============================================================================
# Peak Prominences
# =============================================================================


def peak_prominences(
    x: ArrayLike,
    peaks: ArrayLike,
    wlen: int | None = None,
) -> Prominences:
    """
    Calculate the prominence of peaks.

    The prominence of a peak is the vertical distance between the peak and
    the highest contour line that doesn't contain a higher peak.

    Matches scipy.signal.peak_prominences API.

    Parameters
    ----------
    x : ArrayLike
        1D input signal.
    peaks : ArrayLike
        Indices of peaks in x.
    wlen : int, optional
        Window length for prominence calculation.
        If None, uses entire signal.

    Returns
    -------
    prominences : jnp.ndarray
        Prominence of each peak.
    left_bases : jnp.ndarray
        Index of left base for each peak.
    right_bases : jnp.ndarray
        Index of right base for each peak.

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    x_np = _validate_signal(x)
    peaks_np = _validate_peaks(peaks, len(x_np))

    n_peaks = len(peaks_np)
    if n_peaks == 0:
        return (
            jnp.array([], dtype=jnp.float64),
            jnp.array([], dtype=jnp.int64),
            jnp.array([], dtype=jnp.int64),
        )

    prominences = np.empty(n_peaks, dtype=np.float64)
    left_bases = np.empty(n_peaks, dtype=np.intp)
    right_bases = np.empty(n_peaks, dtype=np.intp)

    for i, peak in enumerate(peaks_np):
        height = x_np[peak]

        # Determine search window
        if wlen is not None:
            left_bound = max(0, peak - wlen // 2)
            right_bound = min(len(x_np), peak + wlen // 2 + 1)
        else:
            left_bound = 0
            right_bound = len(x_np)

        # Search left for base (where signal rises above peak height or boundary)
        # Track the minimum value and its position while searching
        left_min_val = height
        left_min_idx = peak
        for j in range(peak - 1, left_bound - 1, -1):
            if x_np[j] >= height:
                # Found a higher point, stop searching
                break
            if x_np[j] < left_min_val:
                left_min_val = x_np[j]
                left_min_idx = j

        # Search right for base (where signal rises above peak height or boundary)
        right_min_val = height
        right_min_idx = peak
        for j in range(peak + 1, right_bound):
            if x_np[j] >= height:
                # Found a higher point, stop searching
                break
            if x_np[j] < right_min_val:
                right_min_val = x_np[j]
                right_min_idx = j

        # Prominence is the peak height minus the higher of the two contour bases
        prominences[i] = height - max(left_min_val, right_min_val)
        # Left/right bases are the indices of the minima on each side
        left_bases[i] = left_min_idx
        right_bases[i] = right_min_idx

    return (
        jnp.array(prominences),
        jnp.array(left_bases),
        jnp.array(right_bases),
    )


# =============================================================================
# Peak Widths
# =============================================================================


def peak_widths(
    x: ArrayLike,
    peaks: ArrayLike,
    rel_height: float = 0.5,
    prominence_data: Prominences | None = None,
    wlen: int | None = None,
) -> Widths:
    """
    Calculate the width of peaks at a relative height.

    Matches scipy.signal.peak_widths API.

    Parameters
    ----------
    x : ArrayLike
        1D input signal.
    peaks : ArrayLike
        Indices of peaks in x.
    rel_height : float, optional
        Relative height at which to measure width (0 to 1).
        0 = at the peak, 1 = at the base. Default: 0.5.
    prominence_data : tuple, optional
        Pre-computed (prominences, left_bases, right_bases).
        If None, will be computed internally.
    wlen : int, optional
        Window length for prominence calculation if prominence_data is None.

    Returns
    -------
    widths : jnp.ndarray
        Width of each peak in samples.
    width_heights : jnp.ndarray
        Height at which width was measured.
    left_ips : jnp.ndarray
        Left interpolated position of width measurement.
    right_ips : jnp.ndarray
        Right interpolated position of width measurement.

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    x_np = _validate_signal(x)
    peaks_np = _validate_peaks(peaks, len(x_np))

    if not 0 < rel_height <= 1:
        raise ValueError(f"rel_height must be in (0, 1], got {rel_height}")

    n_peaks = len(peaks_np)
    if n_peaks == 0:
        return (
            jnp.array([], dtype=jnp.float64),
            jnp.array([], dtype=jnp.float64),
            jnp.array([], dtype=jnp.float64),
            jnp.array([], dtype=jnp.float64),
        )

    # Get prominence data
    if prominence_data is None:
        prominences, left_bases, right_bases = peak_prominences(x_np, peaks_np, wlen)
        prominences = np.asarray(prominences)
        left_bases = np.asarray(left_bases)
        right_bases = np.asarray(right_bases)
    else:
        prominences = np.asarray(prominence_data[0])
        left_bases = np.asarray(prominence_data[1])
        right_bases = np.asarray(prominence_data[2])

    widths = np.empty(n_peaks, dtype=np.float64)
    width_heights = np.empty(n_peaks, dtype=np.float64)
    left_ips = np.empty(n_peaks, dtype=np.float64)
    right_ips = np.empty(n_peaks, dtype=np.float64)

    for i, peak in enumerate(peaks_np):
        height = x_np[peak]
        prom = prominences[i]
        ref_height = height - rel_height * prom
        width_heights[i] = ref_height

        # Search left for crossing
        left_ip = float(left_bases[i])
        for j in range(peak, int(left_bases[i]) - 1, -1):
            if x_np[j] <= ref_height:
                # Interpolate crossing point
                if j < peak:
                    x1, x2 = j, j + 1
                    y1, y2 = x_np[x1], x_np[x2]
                    if y2 != y1:
                        left_ip = x1 + (ref_height - y1) / (y2 - y1)
                    else:
                        left_ip = x1
                else:
                    left_ip = float(j)
                break

        # Search right for crossing
        right_ip = float(right_bases[i])
        for j in range(peak, int(right_bases[i]) + 1):
            if x_np[j] <= ref_height:
                # Interpolate crossing point
                if j > peak:
                    x1, x2 = j - 1, j
                    y1, y2 = x_np[x1], x_np[x2]
                    if y2 != y1:
                        right_ip = x1 + (ref_height - y1) / (y2 - y1)
                    else:
                        right_ip = x2
                else:
                    right_ip = float(j)
                break

        left_ips[i] = left_ip
        right_ips[i] = right_ip
        widths[i] = right_ip - left_ip

    return (
        jnp.array(widths),
        jnp.array(width_heights),
        jnp.array(left_ips),
        jnp.array(right_ips),
    )


# =============================================================================
# Find Peaks (Main Function)
# =============================================================================


def find_peaks(
    x: ArrayLike,
    height: float | tuple[float, float] | None = None,
    threshold: float | tuple[float, float] | None = None,
    distance: float | None = None,
    prominence: float | tuple[float, float] | None = None,
    width: float | tuple[float, float] | None = None,
    wlen: int | None = None,
    rel_height: float = 0.5,
    plateau_size: int | tuple[int, int] | None = None,
) -> tuple[jnp.ndarray, PeakProperties]:
    """
    Find peaks in a 1D signal with optional filtering.

    Matches scipy.signal.find_peaks API.

    Parameters
    ----------
    x : ArrayLike
        1D input signal.
    height : float or (float, float), optional
        Minimum peak height or (min, max) range.
    threshold : float or (float, float), optional
        Minimum vertical distance to neighbors.
    distance : float, optional
        Minimum horizontal distance between peaks (in samples).
    prominence : float or (float, float), optional
        Minimum peak prominence or (min, max) range.
    width : float or (float, float), optional
        Minimum peak width or (min, max) range.
    wlen : int, optional
        Window length for prominence calculation.
    rel_height : float, optional
        Relative height for width calculation. Default: 0.5.
    plateau_size : int or (int, int), optional
        Plateau size constraint (not commonly used).

    Returns
    -------
    peaks : jnp.ndarray
        Indices of peaks in x.
    properties : dict
        Dictionary containing peak properties:
        - 'peak_heights': heights of peaks (if height specified)
        - 'prominences': prominence values (if prominence or width specified)
        - 'left_bases', 'right_bases': prominence base indices
        - 'widths': peak widths (if width specified)
        - 'width_heights': heights at which widths measured
        - 'left_ips', 'right_ips': interpolated width positions

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    x_np = _validate_signal(x, min_length=0)
    properties: PeakProperties = {}

    # Handle empty or very short signals
    if len(x_np) < 3:
        return jnp.array([], dtype=jnp.int64), properties

    # Find all local maxima
    peaks = _find_local_maxima(x_np)

    if len(peaks) == 0:
        return jnp.array([], dtype=jnp.int64), properties

    # Apply height filter
    height_range = _normalize_constraint(height)
    if height_range is not None:
        heights = x_np[peaks]
        properties["peak_heights"] = jnp.array(heights)
        mask = (heights >= height_range[0]) & (heights <= height_range[1])
        peaks = peaks[mask]
        if "peak_heights" in properties:
            properties["peak_heights"] = properties["peak_heights"][mask]

    # Apply threshold filter (vertical distance to neighbors)
    threshold_range = _normalize_constraint(threshold)
    if threshold_range is not None and len(peaks) > 0:
        left_diff = x_np[peaks] - x_np[np.maximum(peaks - 1, 0)]
        right_diff = x_np[peaks] - x_np[np.minimum(peaks + 1, len(x_np) - 1)]
        min_diff = np.minimum(left_diff, right_diff)
        mask = (min_diff >= threshold_range[0]) & (min_diff <= threshold_range[1])
        peaks = peaks[mask]

    # Apply distance filter
    if distance is not None and len(peaks) > 0:
        heights = x_np[peaks]
        peaks = _select_by_distance(peaks, heights, distance)

    # Compute prominences if needed
    if (prominence is not None or width is not None) and len(peaks) > 0:
        proms, left_bases, right_bases = peak_prominences(x_np, peaks, wlen)
        properties["prominences"] = proms
        properties["left_bases"] = left_bases
        properties["right_bases"] = right_bases

        # Apply prominence filter
        prom_range = _normalize_constraint(prominence)
        if prom_range is not None:
            proms_np = np.asarray(proms)
            mask = (proms_np >= prom_range[0]) & (proms_np <= prom_range[1])
            peaks = peaks[mask]
            properties["prominences"] = properties["prominences"][mask]
            properties["left_bases"] = properties["left_bases"][mask]
            properties["right_bases"] = properties["right_bases"][mask]

    # Compute widths if needed
    if width is not None and len(peaks) > 0:
        prom_data = (
            properties.get("prominences"),
            properties.get("left_bases"),
            properties.get("right_bases"),
        )
        if prom_data[0] is not None:
            widths_arr, width_heights, left_ips, right_ips = peak_widths(
                x_np, peaks, rel_height, prom_data, wlen
            )
        else:
            widths_arr, width_heights, left_ips, right_ips = peak_widths(
                x_np, peaks, rel_height, None, wlen
            )

        properties["widths"] = widths_arr
        properties["width_heights"] = width_heights
        properties["left_ips"] = left_ips
        properties["right_ips"] = right_ips

        # Apply width filter
        width_range = _normalize_constraint(width)
        if width_range is not None:
            widths_np = np.asarray(widths_arr)
            mask = (widths_np >= width_range[0]) & (widths_np <= width_range[1])
            peaks = peaks[mask]
            for key in [
                "prominences",
                "left_bases",
                "right_bases",
                "widths",
                "width_heights",
                "left_ips",
                "right_ips",
            ]:
                if key in properties:
                    properties[key] = properties[key][mask]

    # Note: plateau_size is not commonly used in QCM analysis, so we skip it
    # for simplicity. Can be added if needed.

    return jnp.array(peaks), properties


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Type aliases
    "Signal",
    "PeakIndices",
    "Prominences",
    "Widths",
    "PeakProperties",
    # Main functions
    "find_peaks",
    "peak_prominences",
    "peak_widths",
]
