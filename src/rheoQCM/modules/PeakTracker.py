"""
class for peak tracking and fitting

Migrated from lmfit to NLSQ for peak fitting.
Uses JAX-accelerated nonlinear least squares via nlsq.curve_fit.
"""

import logging

# for debugging
from random import randrange

import jax.numpy as jnp
import numpy as np
from jax import jit
from nlsq import curve_fit

from rheoQCM.core.signal import find_peaks, peak_prominences, peak_widths

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Loading (with fallback for testing)
# =============================================================================

# Default configuration values used by PeakTracker
_CONFIG_DEFAULTS = {
    "peak_min_distance_Hz": 1e3,
    "peak_min_width_Hz": 10,
    "xtol": 1e-10,
    "ftol": 1e-10,
    "cen_range": 0.05,
    "big_move_thresh": 1.5,
    "wid_ratio_range": (8, 20),
    "change_thresh": (0.05, 0.5),
}

# Try to import UISettings, fallback to defaults for testing
try:
    import UISettings

    config_default = UISettings.get_config()
except ImportError:
    # Running in test environment without full GUI stack
    logger.info("UISettings not available, using default config values")
    config_default = _CONFIG_DEFAULTS

# Try to import UIModules for converter function
try:
    from modules import UIModules

    converter_startstop_to_centerspan = UIModules.converter_startstop_to_centerspan
except ImportError:
    # Fallback implementation for testing
    def converter_startstop_to_centerspan(start, stop):
        """Convert start/stop to center/span."""
        center = (start + stop) / 2
        span = stop - start
        return center, span


# peak finder method
peak_finder_method = "py_func"


# =============================================================================
# JAX-Compatible Peak Model Functions
# =============================================================================


@jit
def fun_G(x, amp, cen, wid, phi):
    """
    JAX-compatible function of relation between frequency (f) and conductance (G).
    Lorentzian peak model with phase.
    """
    return (
        amp
        * (
            4 * wid**2 * x**2 * jnp.cos(phi)
            - 2 * wid * x * jnp.sin(phi) * (cen**2 - x**2)
        )
        / (4 * wid**2 * x**2 + (cen**2 - x**2) ** 2)
    )


@jit
def fun_B(x, amp, cen, wid, phi):
    """
    JAX-compatible function of relation between frequency (f) and susceptance (B).
    Lorentzian peak model with phase.
    """
    return (
        amp
        * (
            4 * wid**2 * x**2 * jnp.sin(phi)
            + 2 * wid * x * jnp.cos(phi) * (cen**2 - x**2)
        )
        / (4 * wid**2 * x**2 + (cen**2 - x**2) ** 2)
    )


def fun_G_numpy(x, amp, cen, wid, phi):
    """
    NumPy version of fun_G for evaluation (non-JAX contexts).
    """
    return (
        amp
        * (
            4 * wid**2 * x**2 * np.cos(phi)
            - 2 * wid * x * np.sin(phi) * (cen**2 - x**2)
        )
        / (4 * wid**2 * x**2 + (cen**2 - x**2) ** 2)
    )


def fun_B_numpy(x, amp, cen, wid, phi):
    """
    NumPy version of fun_B for evaluation (non-JAX contexts).
    """
    return (
        amp
        * (
            4 * wid**2 * x**2 * np.sin(phi)
            + 2 * wid * x * np.cos(phi) * (cen**2 - x**2)
        )
        / (4 * wid**2 * x**2 + (cen**2 - x**2) ** 2)
    )


# =============================================================================
# Composite Model Functions for NLSQ Fitting
# =============================================================================


def create_composite_model(n_peaks):
    """
    Create a composite model function for simultaneous G and B fitting.

    The model accepts x_stacked = [freq, freq] (duplicated frequency array)
    and returns [G_values, B_values] matching ydata = [G, B].

    Parameters
    ----------
    n_peaks : int
        Number of peaks in the model.

    Returns
    -------
    callable
        Model function with signature model(x_stacked, *params) where:
        - x_stacked is [freq, freq] (2N points)
        - params contains: [p0_amp, p0_cen, p0_wid, p0_phi, p1_amp, ..., g_c, b_c]
        - Returns [G, B] (2N points)
    """

    def model(x_stacked, *params):
        # x_stacked is the frequency array duplicated: [f, f]
        # First half is for G, second half is for B
        n = len(x_stacked) // 2
        freq = x_stacked[:n]

        # Extract parameters
        n_params_per_peak = 4
        g_c = params[-2]
        b_c = params[-1]

        G_total = jnp.full_like(freq, g_c)
        B_total = jnp.full_like(freq, b_c)

        for i in range(n_peaks):
            idx = i * n_params_per_peak
            amp = params[idx]
            cen = params[idx + 1]
            wid = params[idx + 2]
            phi = params[idx + 3]

            G_total = G_total + fun_G(freq, amp, cen, wid, phi)
            B_total = B_total + fun_B(freq, amp, cen, wid, phi)

        return jnp.concatenate([G_total, B_total])

    return model


# =============================================================================
# Parameter Management System
# =============================================================================


def get_param_index(name, n_peaks):
    """
    Get the index of a parameter in the parameter array.

    Parameters
    ----------
    name : str
        Parameter name (e.g., 'p0_amp', 'p1_cen', 'g_c', 'b_c')
    n_peaks : int
        Number of peaks in the model

    Returns
    -------
    int
        Index of the parameter in the p0/popt array
    """
    if name == "g_c":
        return n_peaks * 4
    elif name == "b_c":
        return n_peaks * 4 + 1
    else:
        # Parse peak parameter: p{i}_{param}
        parts = name.split("_")
        peak_idx = int(parts[0][1:])  # Remove 'p' prefix
        param_name = parts[1]

        param_offset = {"amp": 0, "cen": 1, "wid": 2, "phi": 3}
        return peak_idx * 4 + param_offset[param_name]


def build_p0_and_bounds(params_dict, n_peaks, zerophase=False):
    """
    Build p0 (initial guesses) and bounds arrays from a parameters dictionary.

    Parameters
    ----------
    params_dict : dict
        Dictionary with structure: {i: {'amp': v, 'cen': v, 'wid': v, 'phi': v}, ...}
        where i is the peak index
    n_peaks : int
        Number of peaks
    zerophase : bool
        If True, fix phi to 0 using tight bounds

    Returns
    -------
    p0 : np.ndarray
        Initial parameter values
    bounds : tuple
        (lower_bounds, upper_bounds) arrays
    fixed_mask : np.ndarray
        Boolean mask indicating which parameters are fixed
    """
    n_params = n_peaks * 4 + 2  # 4 params per peak + g_c + b_c

    p0 = np.zeros(n_params)
    lb = np.full(n_params, -np.inf)
    ub = np.full(n_params, np.inf)
    fixed_mask = np.zeros(n_params, dtype=bool)

    # Fill in peak parameters
    for i in range(n_peaks):
        idx = i * 4
        peak_params = params_dict.get(i, {})

        # amp: amplitude (>= 0)
        p0[idx] = peak_params.get("amp", 1.0)
        lb[idx] = peak_params.get("amp_min", 0.0)
        ub[idx] = peak_params.get("amp_max", np.inf)

        # cen: center frequency
        p0[idx + 1] = peak_params.get("cen", 1e6)
        lb[idx + 1] = peak_params.get("cen_min", -np.inf)
        ub[idx + 1] = peak_params.get("cen_max", np.inf)

        # wid: half-width at half-maximum
        p0[idx + 2] = peak_params.get("wid", 1000.0)
        lb[idx + 2] = peak_params.get("wid_min", 1.0)
        ub[idx + 2] = peak_params.get("wid_max", np.inf)

        # phi: phase
        p0[idx + 3] = peak_params.get("phi", 0.0)
        if zerophase:
            # Fix phi to 0 using tight bounds
            eps = 1e-10
            p0[idx + 3] = 0.0
            lb[idx + 3] = -eps
            ub[idx + 3] = eps
            fixed_mask[idx + 3] = True
        else:
            lb[idx + 3] = peak_params.get("phi_min", -np.pi / 2)
            ub[idx + 3] = peak_params.get("phi_max", np.pi / 2)

    # g_c and b_c (baseline offsets)
    g_c_idx = n_peaks * 4
    b_c_idx = n_peaks * 4 + 1

    p0[g_c_idx] = params_dict.get("g_c", 0.0)
    p0[b_c_idx] = params_dict.get("b_c", 0.0)

    # Baselines are unbounded
    lb[g_c_idx] = -np.inf
    ub[g_c_idx] = np.inf
    lb[b_c_idx] = -np.inf
    ub[b_c_idx] = np.inf

    return p0, (lb, ub), fixed_mask


# =============================================================================
# Result Adapter for Backward Compatibility
# =============================================================================


class NLSQResultAdapter:
    """
    Adapter class to provide lmfit-like interface for NLSQ results.

    This allows existing code that expects lmfit result objects to work
    with NLSQ results without modification.
    """

    def __init__(self, popt, pcov, n_peaks, success=True, message=""):
        """
        Initialize the result adapter.

        Parameters
        ----------
        popt : np.ndarray
            Optimized parameter values
        pcov : np.ndarray
            Parameter covariance matrix
        n_peaks : int
            Number of peaks in the model
        success : bool
            Whether the fit was successful
        message : str
            Status message from the optimizer
        """
        self.popt = popt
        self.pcov = pcov
        self.n_peaks = n_peaks
        self.success = success
        self.message = message
        self.lmdif_message = message

        # Calculate standard errors from covariance
        self.stderr = np.sqrt(np.diag(pcov)) if pcov is not None else None

        # Calculate chi-squared (will be set externally)
        self.chisqr = np.nan

        # Create params-like object
        self.params = NLSQParamsAdapter(popt, pcov, n_peaks)

    def __bool__(self):
        """Return True if the result exists (for compatibility checks)."""
        return True


class NLSQParamsAdapter:
    """
    Adapter to provide lmfit Parameters-like interface for NLSQ results.
    """

    def __init__(self, popt, pcov, n_peaks):
        self.popt = popt
        self.pcov = pcov
        self.n_peaks = n_peaks
        self.stderr = (
            np.sqrt(np.diag(pcov)) if pcov is not None else np.full(len(popt), np.nan)
        )

        # Build name-to-index mapping
        self._name_to_idx = {}
        for i in range(n_peaks):
            for j, param_name in enumerate(["amp", "cen", "wid", "phi"]):
                name = f"p{i}_{param_name}"
                self._name_to_idx[name] = i * 4 + j
        self._name_to_idx["g_c"] = n_peaks * 4
        self._name_to_idx["b_c"] = n_peaks * 4 + 1

    def get(self, name):
        """Get a parameter by name (lmfit-compatible interface)."""
        idx = self._name_to_idx.get(name)
        if idx is None:
            return None
        return NLSQParamValue(
            self.popt[idx], self.stderr[idx] if self.stderr is not None else np.nan
        )

    def valuesdict(self):
        """Return a dictionary of parameter values (lmfit-compatible interface)."""
        return {name: self.popt[idx] for name, idx in self._name_to_idx.items()}

    def __repr__(self):
        return f"NLSQParamsAdapter({self.valuesdict()})"


class NLSQParamValue:
    """
    Adapter for individual parameter value with stderr (lmfit-compatible).
    """

    def __init__(self, value, stderr):
        self.value = value
        self.stderr = stderr


# =============================================================================
# Peak Finding Helper Functions (unchanged from original)
# =============================================================================


def findpeaks_py(
    x,
    resonance,
    output=None,
    sortstr=None,
    threshold=None,
    prominence=None,
    distance=None,
    width=None,
):
    """
    A wrap up of scipy.signal.find_peaks.
    advantage of find_peaks function of scipy is
    the peaks can be constrained by more properties
    such as: width, distance etc..
    output: 'indices' or 'values'. if None, return all (indices, heights, prominences, widths)
    sortstr: 'ascend' or 'descend' ordering data by peak height
    """
    if x is None or len(x) == 1:
        logger.warning(f"findpeaks_py input x is not well assigned!\nx = {x}")
        exit(0)

    logger.info(threshold)
    logger.info("f distance %s", distance / (x[1] - x[0]))
    logger.info(prominence)
    logger.info("f width %s", width / (x[1] - x[0]))
    peaks, props = find_peaks(
        resonance,
        threshold=threshold,
        distance=max(1, distance / (x[1] - x[0])),
        prominence=prominence,
        width=max(1, width / (x[1] - x[0])),
    )

    logger.info(peaks)
    logger.info(props)

    indices = np.copy(peaks)
    values = resonance[indices]
    heights = np.array([])
    prominences = np.array([])
    widths = np.array([])
    if sortstr:
        if sortstr.lower() == "ascend":
            order = np.argsort(values)
        elif sortstr.lower() == "descend":
            order = np.argsort(-values)
            values = -np.sort(-values)
        else:
            order = np.argsort(values)
        logger.info(values)
        logger.info(peaks)
        logger.info(order)
        logger.info(props)

        for i in range(order.size):
            indices[i] = indices[order[i]]
            heights = np.append(heights, props["width_heights"][order[i]])
            prominences = np.append(prominences, props["prominences"][order[i]])
            widths = np.append(widths, props["widths"][order[i]] * (x[1] - x[0]))

    if output:
        if output.lower() == "indices":
            return indices
        elif output.lower() == "values":
            return values
    return indices, heights, prominences, widths


def guess_peak_factors(freq, resonance):
    """
    guess the factors of a peak.
    input:
        freq: frequency
        cen_index: index of center of the peak
        resonance: G or B or modulus
    output:
        amp: amplitude
        cen: peak center
        half_wid: half-maxium hal-width (HMHW)
    """
    if peak_finder_method == "simple_func":
        cen_index = np.argmax(resonance)
        cen = freq[cen_index]
        Rmax = resonance[cen_index]
        amp = Rmax - np.amin(resonance)
        half_max = amp / 2 + np.amin(resonance)
        logger.info("Rmax %s", Rmax)
        logger.info("amp %s", amp)
        half_wid = np.absolute(freq[np.argmin(np.abs(half_max - resonance))] - cen)
        logger.info(half_wid)
        return amp, cen, half_wid, half_max
    elif peak_finder_method == "py_func":
        cen_index = np.argmax(resonance)
        prominences = peak_prominences(resonance, np.array([cen_index]))
        widths = peak_widths(
            resonance,
            np.array([cen_index]),
            prominence_data=prominences,
            rel_height=0.5,
        )

        cen = freq[cen_index]
        amp = prominences[0][0]
        half_wid = min(cen_index - widths[2][0], widths[3][0] - cen_index) * (
            freq[1] - freq[0]
        )
        half_max = widths[1][0]

        return amp, cen, half_wid, half_max


# =============================================================================
# Main PeakTracker Class
# =============================================================================


class PeakTracker:
    """
    Tracks QCM resonance peaks across multiple harmonics.

    PeakTracker manages the peak fitting workflow for QCM resonance data,
    including peak detection, Lorentzian fitting, and parameter extraction.

    Parameters
    ----------
    max_harm : int
        Maximum harmonic number to track (odd harmonics only).

    Attributes
    ----------
    harminput : dict
        Input data organized by channel ('samp', 'ref') and harmonic.
    harmoutput : dict
        Fitting results organized by channel and harmonic.
    active_harm : str or None
        Currently selected harmonic for fitting.
    active_chn : str or None
        Currently selected channel ('samp' or 'ref').

    Examples
    --------
    >>> tracker = PeakTracker(max_harm=13)
    >>> tracker.update_input('samp', '3', harmdata=data, freq_span=span, fGB=fGB)
    >>> tracker.process_data('samp', '3', t=0.0)
    >>> result = tracker.harmoutput['samp']['3']
    """

    def __init__(self, max_harm):
        self.max_harm = max_harm
        self.harminput = self.init_harmdict()
        self.harmoutput = self.init_harmdict()
        for harm in range(1, self.max_harm + 2, 2):
            harm = str(harm)
            self.update_input("samp", harm, harmdata={}, freq_span={}, fGB=[[], [], []])
            self.update_input("ref", harm, harmdata={}, freq_span={}, fGB=[[], [], []])

            self.update_output("samp", harm)
            self.update_output("ref", harm)

        self.active_harm = None
        self.active_chn = None
        self.x = None
        self.resonance = None
        self.peak_guess = {}
        self.found_n = None

    def init_harmdict(self):
        """
        create a dict with for saving input or output data
        """
        harm_dict = {}
        for i in range(1, self.max_harm + 2, 2):
            harm_dict[str(i)] = {}
        chn_dict = {
            "samp": harm_dict,
            "ref": harm_dict,
            "refit": harm_dict,
        }
        return chn_dict

    def update_input(self, chn_name, harm, harmdata, freq_span, fGB=None):
        """
        harmdata: it should be from the main ui self.settings['harmdata']
        if empty harmdata, initialize the key to None
        chn_name: 'samp' or 'ref'
        if f, G, B  all(is None): update harmdata, freq_span only.
        harm: int
        """
        logger.info("#### update_input ####")
        if fGB is not None:
            f, G, B = fGB
            self.harminput[chn_name][harm]["isfitted"] = False
            self.harminput[chn_name][harm]["f"] = f
            self.harminput[chn_name][harm]["G"] = G
            self.harminput[chn_name][harm]["B"] = B
            logger.info("f[chn][harm] %s", "None" if f is None else len(f))

        if not harmdata:
            harm_dict = {}
        else:
            harm_dict = harmdata[chn_name][harm]

            logger.info("chn_name:  %s, harm: %s", chn_name, harm)
            logger.info("harmdata[chn][harm] %s", harm_dict)
        logger.info(" #####################")

        if not freq_span:
            self.harminput[chn_name][harm]["current_span"] = [None, None]
        else:
            self.harminput[chn_name][harm]["current_span"] = freq_span[chn_name][harm]
        self.harminput[chn_name][harm]["steps"] = harm_dict.get(
            "lineEdit_scan_harmsteps", None
        )
        self.harminput[chn_name][harm]["method"] = harm_dict.get(
            "comboBox_tracking_method", None
        )
        self.harminput[chn_name][harm]["condition"] = harm_dict.get(
            "comboBox_tracking_condition", None
        )
        self.harminput[chn_name][harm]["fit"] = harm_dict.get("checkBox_harmfit", True)
        self.harminput[chn_name][harm]["factor"] = harm_dict.get(
            "spinBox_harmfitfactor", None
        )
        self.harminput[chn_name][harm]["n"] = harm_dict.get("spinBox_peaks_num", None)

        if harm_dict.get("radioButton_peaks_num_max", None):
            self.harminput[chn_name][harm]["n_policy"] = "max"
        elif harm_dict.get("radioButton_peaks_num_fixed", None):
            self.harminput[chn_name][harm]["n_policy"] = "fixed"
        else:
            self.harminput[chn_name][harm]["n_policy"] = None

        if harm_dict.get("radioButton_peaks_policy_minf", None):
            self.harminput[chn_name][harm]["p_policy"] = "minf"
        elif harm_dict.get("radioButton_peaks_policy_maxamp", None):
            self.harminput[chn_name][harm]["p_policy"] = "maxamp"
        else:
            self.harminput[chn_name][harm]["p_policy"] = None

        self.harminput[chn_name][harm]["zerophase"] = harm_dict.get(
            "checkBox_settings_settings_harmzerophase", False
        )
        self.harminput[chn_name][harm]["threshold"] = harm_dict.get(
            "lineEdit_peaks_threshold", None
        )
        self.harminput[chn_name][harm]["prominence"] = harm_dict.get(
            "lineEdit_peaks_prominence", None
        )

    def update_output(self, chn_name=None, harm=None, **kwargs):
        """
        kwargs: keys to update
        chn_name: 'samp' or 'ref'
        harm: int
        """
        if (chn_name is None) & (harm is None):
            chn_name = self.active_chn
            harm = self.active_harm

        if kwargs:
            for key, val in kwargs.items():
                self.harmoutput[chn_name][harm][key] = val
        else:
            self.harmoutput[chn_name][harm]["span"] = kwargs.get(
                "span", [np.nan, np.nan]
            )
            self.harmoutput[chn_name][harm]["cen_trk"] = kwargs.get("cen_trk", np.nan)
            self.harmoutput[chn_name][harm]["factor_span"] = kwargs.get(
                "span", [np.nan, np.nan]
            )
            self.harmoutput[chn_name][harm]["method"] = kwargs.get("method", "")
            self.harmoutput[chn_name][harm]["found_n"] = kwargs.get("found_n", np.nan)
            self.harmoutput[chn_name][harm]["n_peaks"] = kwargs.get("n_peaks", 1)
            self.harmoutput[chn_name][harm]["params"] = kwargs.get("params", {})
            self.harmoutput[chn_name][harm]["result"] = kwargs.get("result", {})

    def get_input(self, key=None, chn_name=None, harm=None):
        """
        get val of key from self.harmoutput by chn_name and harm
        if key == None:
            return all of the corresponding chn_name and harm
        """
        if (chn_name is None) & (harm is None):
            chn_name = self.active_chn
            harm = self.active_harm

        if key is not None:
            return self.harminput[chn_name][harm].get(key, None)
        else:
            return self.harminput[chn_name][harm]

    def get_output(self, key=None, chn_name=None, harm=None):
        """
        get val of key from self.harmoutput by chn_name and harm
        if key == None:
            return all of the corresponding chn_name and harm
        """
        if (chn_name is None) & (harm is None):
            chn_name = self.active_chn
            harm = self.active_harm

        if key is not None:
            return self.harmoutput[chn_name][harm].get(key, None)
        else:
            return self.harmoutput[chn_name][harm]

    def init_active_val(self, chn_name=None, harm=None, method=None):
        """
        update active values by harm, chn_name
        """
        if harm is None:
            harm = self.active_harm
        if chn_name is None:
            chn_name = self.active_chn
        if method is None:
            method = self.harminput[chn_name][harm]["method"]

        if method == "bmax":
            self.x = self.harminput[chn_name][harm]["f"]
            self.resonance = self.harminput[chn_name][harm]["B"]
        elif method == "derv":
            self.resonance = np.sqrt(
                np.diff(self.harminput[chn_name][harm]["G"]) ** 2
                + np.diff(self.harminput[chn_name][harm]["B"]) ** 2
            )
            self.x = self.harminput[chn_name][harm]["f"][:-1] + np.diff(
                self.harminput[chn_name][harm]["f"]
            )
        elif method == "prev":
            try:
                pre_method = self.harmoutput[chn_name][harm]["method"]
                if pre_method == "prev":
                    pre_method = "gmax"
            except KeyError:
                pre_method = "gmax"

            self.init_active_val(harm=harm, chn_name=chn_name, method=pre_method)
        else:
            self.resonance = self.harminput[chn_name][harm]["G"]
            self.x = self.harminput[chn_name][harm]["f"]

        self.found_n = 0
        self.peak_guess = {}

    ########### peak tracking function ###########
    def peak_tracker(self):
        """
        track the peak and give the span for next scan
        """

        def set_new_cen(freq, cen, current_span, current_xlim):
            cen_range = config_default["cen_range"]
            cen_diff = cen - np.mean(np.array([freq[0], freq[-1]]))
            logger.info("cen_diff %s", cen_diff)
            half_cen_span = cen_range * current_span
            logger.info("half_cen_span %s", half_cen_span)
            if np.absolute(cen_diff) > half_cen_span:
                if cen_diff / half_cen_span < -config_default["big_move_thresh"]:
                    return cen - half_cen_span
                elif cen_diff / half_cen_span > config_default["big_move_thresh"]:
                    return cen + half_cen_span
                else:
                    return cen
            else:
                return np.mean(np.array(current_xlim))

        def set_new_span(current_span, half_wid):
            wid_ratio_range = config_default["wid_ratio_range"]
            change_thresh = config_default["change_thresh"]
            wid_ratio = 0.5 * current_span / half_wid
            logger.info("wid_ratio %s", wid_ratio)
            if wid_ratio < wid_ratio_range[0]:
                new_span = max(
                    min(
                        wid_ratio_range[0] * half_wid * 2,
                        current_span * (1 + change_thresh[1]),
                    ),
                    current_span * (1 + change_thresh[0]),
                )
            elif wid_ratio > wid_ratio_range[1]:
                logger.info(
                    "peak too thin\n %s %s %s %s",
                    half_wid,
                    wid_ratio_range[1] * half_wid * 2,
                    current_span * (1 - change_thresh[1]),
                    current_span * (1 - change_thresh[0]),
                )
                new_span = min(
                    max(
                        wid_ratio_range[1] * half_wid * 2,
                        current_span * (1 - change_thresh[1]),
                    ),
                    current_span * (1 - change_thresh[0]),
                )
            else:
                new_span = current_span

            return max(new_span, config_default["peak_min_width_Hz"])

        def set_new_xlim(new_cen, new_span):
            return [new_cen - 0.5 * new_span, new_cen + 0.5 * new_span]

        chn_name = self.active_chn
        harm = self.active_harm
        track_condition = self.harminput[chn_name][harm]["condition"]
        track_method = self.harminput[chn_name][harm]["method"]
        logger.info("track_condition: %s", track_condition)
        logger.info("track_method: %s", track_method)
        freq = self.harminput[chn_name][harm]["f"]
        if track_method == "bmax":
            resonance = self.harminput[chn_name][harm]["B"]
        else:
            resonance = self.harminput[chn_name][harm]["G"]

        _, cen, half_wid, _ = guess_peak_factors(freq, resonance)

        current_xlim = np.array(self.harminput[chn_name][harm]["current_span"])
        current_center, current_span = converter_startstop_to_centerspan(
            *self.harminput[chn_name][harm]["current_span"]
        )

        logger.info("current_center %s", current_center)
        logger.info("current_span %s", current_span)
        logger.info("current_xlim %s", current_xlim)
        logger.info("f1f2 %s %s", freq[0], freq[-1])
        new_cen = current_center
        new_xlim = self.harminput[chn_name][harm]["current_span"]

        if track_condition == "fixspan":
            new_cen = set_new_cen(freq, cen, current_span, current_xlim)
            new_xlim = set_new_xlim(new_cen, current_span)
        elif track_condition == "fixcenter":
            new_span = (
                set_new_span(current_span, half_wid),
                config_default["peak_min_width_Hz"],
            )
            new_xlim = set_new_xlim(current_center, new_span)
        elif track_condition == "auto":
            new_cen = set_new_cen(freq, cen, current_span, current_xlim)
            new_span = set_new_span(current_span, half_wid)
            new_xlim = set_new_xlim(new_cen, new_span)

            logger.info("current_center %s", current_center)
            logger.info("current_span %s", current_span)
            logger.info("current_xlim %s", current_xlim)
            logger.info("new_cen %s", new_cen)
            logger.info("new_span %s", new_span)
            logger.info("new_xlim %s", new_xlim)
        elif track_condition == "fixcntspn":
            logger.info("fixcntspn")
            pass
        elif track_condition == "usrdef":
            pass

        logger.info("new_cen: %s", new_cen)
        logger.info("new_xlim: %s", new_xlim)

        self.update_output(chn_name, harm, span=new_xlim)
        self.update_output(chn_name, harm, cen_trk=cen)

    ########### initial values guess functions ###
    def params_guess(self, method="gmax"):
        """
        guess initial values based on given method
        """
        phi = 0

        chn_name = self.active_chn
        harm = self.active_harm
        n_policy = self.harminput[chn_name][harm]["n_policy"]
        p_policy = self.harminput[chn_name][harm]["p_policy"]
        logger.info("chn_name %s", chn_name)
        logger.info("harm %s", harm)
        logger.info("p_policy %s", p_policy)

        if p_policy == "maxamp":
            sortstr = "descend"
        elif p_policy == "minf":
            sortstr = "descend"

        indices, heights, prominences, widths = findpeaks_py(
            self.x,
            self.resonance,
            sortstr=sortstr,
            threshold=self.harminput[chn_name][harm]["threshold"],
            prominence=self.harminput[chn_name][harm]["prominence"],
            distance=config_default["peak_min_distance_Hz"],
            width=config_default["peak_min_width_Hz"],
        )

        logger.info("indices %s", indices)
        if indices.size == 0:
            self.found_n = 0
            self.update_output(found_n=0)
            return

        if method == "derv":
            phi = np.arcsin(
                self.harminput[chn_name][harm]["G"][0]
                / np.sqrt(
                    self.harminput[chn_name][harm]["G"][0] ** 2
                    + self.harminput[chn_name][harm]["B"][0] ** 2
                )
            )

        if n_policy == "max":
            self.found_n = min(len(indices), self.harminput[chn_name][harm]["n"])
        elif n_policy == "fixed":
            self.found_n = self.harminput[chn_name][harm]["n"]

        logger.info(type(self.harminput[chn_name][harm]["n"]))
        logger.info(indices)
        logger.info(heights)
        logger.info(prominences)
        logger.info(widths)

        for i in range(self.found_n):
            if i + 1 <= len(indices):
                logger.info(i)
                self.peak_guess[i] = {
                    "amp": prominences[i],
                    "cen": self.x[indices[i]],
                    "wid": widths[i] / 2,
                    "phi": phi,
                }
            else:
                self.peak_guess[i] = {
                    "amp": np.amin(prominences),
                    "cen": self.x[
                        randrange(
                            int(len(self.x) * 0.3), int(len(self.x) * 0.6), self.found_n
                        )
                    ],
                    "wid": np.amin(widths) / 2,
                    "phi": phi,
                }
        self.update_output(found_n=self.found_n)
        logger.info("out found %s", self.harmoutput[chn_name][harm]["found_n"])

    def prev_guess(self, chn_name=None, harm=None):
        """
        get previous calculated values and put them into peak_guess
        """
        if (chn_name is None) & (harm is None):
            chn_name = self.active_chn
            harm = self.active_harm

        result = self.harmoutput[chn_name][harm].get("result", None)
        logger.info(result)
        if not result:
            self.peak_guess = {}
            self.found_n = 0
            self.update_output(found_n=0)
            return

        val = result.params.valuesdict()

        logger.info(self.harmoutput[chn_name][harm]["found_n"])
        logger.info(self.harminput[chn_name][harm]["n"])
        n_policy = self.harminput[chn_name][harm]["n_policy"]

        if n_policy == "max":
            self.found_n = min(
                self.harmoutput[chn_name][harm]["found_n"],
                self.harminput[chn_name][harm]["n"],
            )
        elif n_policy == "fixed":
            self.found_n = self.harminput[chn_name][harm]["n"]

        for i in np.arange(self.found_n):
            if i + 1 <= self.harmoutput[chn_name][harm]["found_n"]:
                pre_str = "p" + str(i) + "_"
                self.peak_guess[i] = {
                    "amp": val[pre_str + "amp"],
                    "cen": val[pre_str + "cen"],
                    "wid": val[pre_str + "wid"],
                    "phi": val[pre_str + "phi"],
                }
            else:
                self.peak_guess[i] = {
                    "amp": (
                        self.peak_guess[self.harmoutput[chn_name][harm]["found_n"] - 1][
                            "amp"
                        ]
                        if self.harmoutput[chn_name][harm]["found_n"] > 0
                        else np.max(self.resonance) - np.min(self.resonance)
                    ),
                    "cen": self.x[
                        randrange(
                            int(len(self.x) * 0.3), int(len(self.x) * 0.6), self.found_n
                        )
                    ],
                    "wid": (
                        self.peak_guess[self.harmoutput[chn_name][harm]["found_n"] - 1][
                            "wid"
                        ]
                        if self.harmoutput[chn_name][harm]["found_n"] > 0
                        else (np.max(self.x) - np.min(self.x)) / 10
                    ),
                    "phi": (
                        self.peak_guess[self.harmoutput[chn_name][harm]["found_n"] - 1][
                            "phi"
                        ]
                        if self.harmoutput[chn_name][harm]["found_n"] > 0
                        else 0
                    ),
                }
        self.update_output(chn_name, harm, found_n=self.found_n)

    def auto_guess(self):
        """
        auto guess the peak parameters by using the given
        method.
        """
        if self.harminput[self.active_chn][self.active_harm]["method"] == "auto":
            method_list = ["gmax", "bmax", "derv", "prev"]
        else:
            method_list = [self.harminput[self.active_chn][self.active_harm]["method"]]

        for method in method_list:
            logger.info(method)
            if method == "prev":
                self.prev_guess()
            else:
                self.params_guess(method=method)

            if self.found_n:
                self.update_output(method=method)
                break

    def set_params(self, chn_name=None, harm=None):
        """
        Set the parameters for fitting.

        Builds the parameter dictionary with initial values and bounds
        for NLSQ curve_fit.
        """
        if (chn_name is None) & (harm is None):
            chn_name = self.active_chn
            harm = self.active_harm

        # Get data for rough guesses
        f = self.get_input(key="f", chn_name=chn_name, harm=harm)
        G = self.get_input(key="G", chn_name=chn_name, harm=harm)
        B = self.get_input(key="B", chn_name=chn_name, harm=harm)
        amp_rough = np.amax(self.resonance) - np.amin(self.resonance)
        cen_rough = np.mean(f)
        wid_rough = (np.amax(f) - np.amin(f)) / 6
        phi_rough = 0

        logger.info("peak_guess %s", self.peak_guess)

        if self.found_n == 0:
            self.found_n = 1
            self.update_output(found_n=1)

        # Build parameters dictionary for NLSQ
        params_dict = {}
        zerophase = self.harminput[chn_name][harm]["zerophase"]

        for i in np.arange(self.found_n):
            if not self.peak_guess:
                amp = amp_rough
                cen = cen_rough
                wid = wid_rough
                phi = phi_rough
            else:
                amp = self.peak_guess[i].get("amp", amp_rough)
                cen = self.peak_guess[i].get("cen", cen_rough)
                wid = self.peak_guess[i].get("wid", wid_rough)
                phi = self.peak_guess[i].get("phi", phi_rough)

            params_dict[i] = {
                "amp": amp,
                "amp_min": 0.0,
                "amp_max": np.inf,
                "cen": cen,
                "cen_min": np.amin(f),
                "cen_max": np.amax(f),
                "wid": wid,
                "wid_min": wid / 10,  # limit width >= 1/10 of guess
                "wid_max": (np.amax(f) - np.amin(f)) * 2,
                "phi": 0.0 if zerophase else phi,
                "phi_min": -np.pi / 2,
                "phi_max": np.pi / 2,
            }

        # Add baseline offsets
        params_dict["g_c"] = np.amin(G)
        params_dict["b_c"] = np.mean(B)

        self.update_output(params=params_dict)
        self.update_output(n_peaks=self.found_n)

    ########### fitting ##########################
    def minimize_GB(self):
        """
        Use NLSQ curve_fit to fit G and B simultaneously.
        """
        chn_name = self.active_chn
        harm = self.active_harm
        factor = self.get_input(key="factor")

        logger.info("chn: %s, harm: %s", chn_name, harm)
        logger.info(self.get_output())
        logger.info("mm factor %s", factor)
        logger.info("self n %s", self.found_n)

        # Guess parameters and set up
        self.auto_guess()
        self.set_params()

        logger.info("self n %s", self.found_n)

        n_peaks = self.found_n
        self.update_output(n_peaks=n_peaks)

        f = self.harminput[chn_name][harm]["f"]
        G = self.harminput[chn_name][harm]["G"]
        B = self.harminput[chn_name][harm]["B"]

        params_dict = self.get_output(key="params")
        zerophase = self.harminput[chn_name][harm]["zerophase"]

        # Set data for fitting by factor
        factor_idx_list = []
        factor_set_list = []
        if factor is not None:
            for i in range(self.found_n):
                cen_i = params_dict[i]["cen"]
                wid_i = params_dict[i]["wid"]
                logger.info("cen_i %s", cen_i)
                logger.info("wid_i %s", wid_i)
                ind_min = np.abs(
                    self.harminput[chn_name][harm]["f"] - (cen_i - wid_i * factor)
                ).argmin()
                ind_max = np.abs(
                    self.harminput[chn_name][harm]["f"] - (cen_i + wid_i * factor)
                ).argmin()
                factor_idx_list.extend([ind_min, ind_max])
                factor_set_list.append(set(np.arange(ind_min, ind_max)))

            idx_list = list(set().union(*factor_set_list))
            [True if i in idx_list else False for i in np.arange(len(f))]

            self.update_output(
                chn_name=chn_name,
                harm=harm,
                factor_span=[min(f[factor_idx_list]), max(f[factor_idx_list])],
            )

            f = f[idx_list]
            G = G[idx_list]
            B = B[idx_list]

            logger.info("data len after factor %s", len(f))

        logger.info("factor\n%s", factor)
        logger.info("mm params %s", self.harmoutput[chn_name][harm]["params"])

        try:
            # Build p0 and bounds for NLSQ
            p0, bounds, fixed_mask = build_p0_and_bounds(
                params_dict, n_peaks, zerophase
            )

            # Create the composite model
            model = create_composite_model(n_peaks)

            # Stack x data: [f, f] so it matches length of [G, B]
            x_stacked = np.concatenate([f, f])
            y_data = np.concatenate([G, B])

            # Call NLSQ curve_fit
            popt, pcov = curve_fit(
                model,
                x_stacked,  # duplicated frequency array
                y_data,
                p0=p0,
                bounds=bounds,
                method="trf",  # Trust Region Reflective (handles bounds)
                ftol=config_default.get("ftol", 1e-18),
                xtol=config_default.get("xtol", 1e-18),
            )

            # Calculate residuals and chi-squared
            y_fit = model(x_stacked, *popt)
            residuals = y_data - np.array(y_fit)
            chisqr = np.sum(residuals**2)

            # Create result adapter for backward compatibility
            result = NLSQResultAdapter(
                popt=popt,
                pcov=pcov,
                n_peaks=n_peaks,
                success=True,
                message="Optimization converged.",
            )
            result.chisqr = chisqr

            logger.info("NLSQ Fit Report:")
            logger.info("  Success: %s", result.success)
            logger.info("  Chi-squared: %.6g", chisqr)
            logger.info("  Parameters: %s", result.params.valuesdict())

        except Exception:
            result = {}
            logger.exception("fitting error occurred.")

        self.update_output(chn_name, harm, result=result)

    def get_fit_values(self, chn_name=None, harm=None):
        """
        get values from calculated result
        """
        if (chn_name is None) & (harm is None):
            chn_name = self.active_chn
            harm = self.active_harm

        p_policy = self.harminput[chn_name][harm]["p_policy"]
        result = self.harmoutput[chn_name][harm]["result"]
        found_n = self.harmoutput[chn_name][harm]["found_n"]

        val = {}
        if result:
            # Check peak order by amp and cen
            amp_array = np.array(
                [result.params.get("p" + str(i) + "_amp").value for i in range(found_n)]
            )
            cen_array = np.array(
                [result.params.get("p" + str(i) + "_cen").value for i in range(found_n)]
            )

            logger.info("found_n %s", found_n)
            logger.info(result.params)
            logger.info("params %s", result.params.valuesdict())
            logger.info(amp_array)
            logger.info(cen_array)

            maxamp_idx = np.argmax(amp_array)
            mincen_idx = np.argmin(cen_array)

            p_trk = maxamp_idx
            if p_policy == "maxamp":
                p_rec = maxamp_idx
            elif p_policy == "minf":
                p_rec = mincen_idx

            # Values for tracking peak
            val["amp_trk"] = {
                "value": result.params.get("p" + str(p_trk) + "_amp").value,
                "stderr": result.params.get("p" + str(p_trk) + "_amp").stderr,
            }
            val["cen_trk"] = {
                "value": result.params.get("p" + str(p_trk) + "_cen").value,
                "stderr": result.params.get("p" + str(p_trk) + "_cen").stderr,
            }
            val["wid_trk"] = {
                "value": result.params.get("p" + str(p_trk) + "_wid").value,
                "stderr": result.params.get("p" + str(p_trk) + "_wid").stderr,
            }
            val["phi_trk"] = {
                "value": result.params.get("p" + str(p_trk) + "_phi").value,
                "stderr": result.params.get("p" + str(p_trk) + "_phi").stderr,
            }

            # Values for recording peak
            val["amp_rec"] = {
                "value": result.params.get("p" + str(p_rec) + "_amp").value,
                "stderr": result.params.get("p" + str(p_rec) + "_amp").stderr,
            }
            val["cen_rec"] = {
                "value": result.params.get("p" + str(p_rec) + "_cen").value,
                "stderr": result.params.get("p" + str(p_rec) + "_cen").stderr,
            }
            val["wid_rec"] = {
                "value": result.params.get("p" + str(p_rec) + "_wid").value,
                "stderr": result.params.get("p" + str(p_rec) + "_wid").stderr,
            }
            val["phi_rec"] = {
                "value": result.params.get("p" + str(p_rec) + "_phi").value,
                "stderr": result.params.get("p" + str(p_rec) + "_phi").stderr,
            }

            val["g_c"] = {
                "value": result.params.get("g_c").value,
                "stderr": result.params.get("g_c").stderr,
            }
            val["b_c"] = {
                "value": result.params.get("b_c").value,
                "stderr": result.params.get("b_c").stderr,
            }

            val["sucess"] = result.success
            val["chisqr"] = result.chisqr

            logger.info("params %s", result.params.valuesdict())
        else:
            # No result - return NaN values
            val["amp_trk"] = {"value": np.nan, "stderr": np.nan}
            val["cen_trk"] = {"value": np.nan, "stderr": np.nan}
            val["wid_trk"] = {"value": np.nan, "stderr": np.nan}
            val["phi_trk"] = {"value": np.nan, "stderr": np.nan}
            val["amp_rec"] = {"value": np.nan, "stderr": np.nan}
            val["cen_rec"] = {"value": np.nan, "stderr": np.nan}
            val["wid_rec"] = {"value": np.nan, "stderr": np.nan}
            val["phi_rec"] = {"value": np.nan, "stderr": np.nan}
            val["g_c"] = {"value": np.nan, "stderr": np.nan}
            val["b_c"] = {"value": np.nan, "stderr": np.nan}
            val["sucess"] = False
            val["chisqr"] = np.nan

        return val

    def eval_mod(self, mod_name, chn_name=None, harm=None, components=False):
        """
        Evaluate gmod or bmod by f.
        Returns ndarray.
        """
        if (chn_name is None) & (harm is None):
            chn_name = self.active_chn
            harm = self.active_harm

        result = self.harmoutput[chn_name][harm]["result"]
        f = self.harminput[chn_name][harm]["f"]

        if result:
            n_peaks = self.harmoutput[chn_name][harm].get("n_peaks", 1)
            popt = result.popt

            if components is False:
                # Total fitting - evaluate the full model
                g_c = popt[-2]
                b_c = popt[-1]

                if mod_name == "gmod":
                    G_total = np.full_like(f, g_c)
                    for i in range(n_peaks):
                        idx = i * 4
                        amp, cen, wid, phi = (
                            popt[idx],
                            popt[idx + 1],
                            popt[idx + 2],
                            popt[idx + 3],
                        )
                        G_total = G_total + fun_G_numpy(f, amp, cen, wid, phi)
                    return G_total
                elif mod_name == "bmod":
                    B_total = np.full_like(f, b_c)
                    for i in range(n_peaks):
                        idx = i * 4
                        amp, cen, wid, phi = (
                            popt[idx],
                            popt[idx + 1],
                            popt[idx + 2],
                            popt[idx + 3],
                        )
                        B_total = B_total + fun_B_numpy(f, amp, cen, wid, phi)
                    return B_total
                else:
                    return np.empty(f.shape) * np.nan
            else:
                # Return divided peaks (components)
                g_c = popt[-2]
                b_c = popt[-1]

                if mod_name == "gmod":
                    g_fit = []
                    for i in range(n_peaks):
                        idx = i * 4
                        amp, cen, wid, phi = (
                            popt[idx],
                            popt[idx + 1],
                            popt[idx + 2],
                            popt[idx + 3],
                        )
                        g_fit.append(fun_G_numpy(f, amp, cen, wid, phi) + g_c)
                    return g_fit
                elif mod_name == "bmod":
                    b_fit = []
                    for i in range(n_peaks):
                        idx = i * 4
                        amp, cen, wid, phi = (
                            popt[idx],
                            popt[idx + 1],
                            popt[idx + 2],
                            popt[idx + 3],
                        )
                        b_fit.append(fun_B_numpy(f, amp, cen, wid, phi) + b_c)
                    return b_fit
                else:
                    dummy = []
                    for _ in range(n_peaks):
                        dummy.append(np.empty(f.shape) * np.nan)
                    return dummy
        else:
            # No result found
            if components is False:
                return np.empty(f.shape) * np.nan
            else:
                return [np.empty(f.shape) * np.nan]

    ########### wrap up functions ################
    def peak_track(self, chn_name=None, harm=None):
        """
        The whole process of peak tracking
        return the predicted span
        """
        if (chn_name is not None) & (harm is not None):
            self.active_chn = chn_name
            self.active_harm = harm

        self.peak_tracker()

        return (
            self.harmoutput[chn_name][harm]["span"],
            self.harmoutput[chn_name][harm]["cen_trk"],
        )

    def peak_fit(self, chn_name=None, harm=None, components=False):
        """
        The whole process of peak fitting
        return:
            the dict of values with std errors
            fitted G value
            fitted B value
            if components == True:
                list fitted G values by peaks
                list fitted B values by peaks
        """
        if (chn_name is not None) & (harm is not None):
            self.active_chn = chn_name
            self.active_harm = harm

        self.init_active_val(chn_name=chn_name, harm=harm)
        logger.info("chn: %s, harm: %s", chn_name, harm)
        logger.info("self.chn: %s,self.harm: %s", self.active_chn, self.active_harm)

        self.minimize_GB()

        if components is False:
            return {
                "v_fit": self.get_fit_values(chn_name=chn_name, harm=harm),
                "fit_g": self.eval_mod("gmod", chn_name=chn_name, harm=harm),
                "fit_b": self.eval_mod("bmod", chn_name=chn_name, harm=harm),
                "factor_span": self.get_output(
                    key="factor_span", chn_name=chn_name, harm=harm
                ),
            }
        elif components is True:
            return {
                "v_fit": self.get_fit_values(chn_name=chn_name, harm=harm),
                "fit_g": self.eval_mod("gmod", chn_name=chn_name, harm=harm),
                "fit_b": self.eval_mod("bmod", chn_name=chn_name, harm=harm),
                "comp_g": self.eval_mod(
                    "gmod", chn_name=chn_name, harm=harm, components=True
                ),
                "comp_b": self.eval_mod(
                    "bmod", chn_name=chn_name, harm=harm, components=True
                ),
            }

    def fit_result_report(self, fit_result=None):
        """
        Generate a human-readable fit result report.
        """
        v_fit = self.get_fit_values()
        logger.info(v_fit)
        keys = {
            "f (Hz)": "cen_rec",
            "\u0393" + " (Hz)": "wid_rec",
            "\u03a6" + " (deg.)": "phi_rec",
            "G_amp (mS)": "amp_rec",
            "G_shift (mS)": "g_c",
            "B_shift (mS)": "b_c",
        }

        buff = []

        buff.append("Sucess:\t{}".format(v_fit["sucess"]))
        buff.append("\u03a7" + "sq:\t{:.7g}".format(v_fit["chisqr"]))

        for txt, key in keys.items():
            txt = f"{txt}:\t"
            if v_fit[key].get("value") is not None:
                txt = "{}{:.7g}".format(
                    txt,
                    (
                        v_fit[key].get("value") * 180 / np.pi
                        if key == "phi_rec"
                        else v_fit[key].get("value")
                    ),
                )
            if v_fit[key].get("stderr") is not None:
                txt = "{} +/- {:.7g}".format(
                    txt,
                    (
                        v_fit[key].get("stderr") * 180 / np.pi
                        if key == "phi_rec"
                        else v_fit[key].get("stderr")
                    ),
                )
            buff.append(txt)
        return "\n".join(buff)

    def plot_peak_fit(
        self,
        chn_name: str | None = None,
        harm: str | None = None,
        *,
        confidence_level: float = 0.95,
        ax=None,
    ):
        """Plot peak fit with uncertainty bands.

        Generates a plot showing the measured data, fitted curve,
        and confidence bands computed using error propagation.

        Parameters
        ----------
        chn_name : str | None
            Channel name ("samp" or "ref"). Default: active channel.
        harm : str | None
            Harmonic number as string. Default: active harmonic.
        confidence_level : float
            Confidence level for bands (default: 0.95)
        ax : matplotlib.axes.Axes | None
            Axes to plot on. Creates new figure if None.

        Returns
        -------
        matplotlib.figure.Figure
            Figure containing the plot.

        Notes
        -----
        Requires that a fit has been completed for the specified
        channel and harmonic.
        """
        import matplotlib.pyplot as plt

        from rheoQCM.core.uncertainty import UncertaintyCalculator

        if chn_name is None:
            chn_name = self.active_chn
        if harm is None:
            harm = self.active_harm

        if chn_name is None or harm is None:
            raise ValueError("No active channel/harmonic. Specify chn_name and harm.")

        harminput = self.harminput[chn_name][harm]
        harmoutput = self.harmoutput[chn_name][harm]
        result = harmoutput.get("result")

        if not result or not hasattr(result, "popt"):
            raise ValueError(f"No fit result available for {chn_name}/{harm}")

        f = np.array(harminput["f"])
        G = np.array(harminput["G"])
        B = np.array(harminput["B"])

        n_peaks = harmoutput.get("found_n", 1)
        model = create_composite_model(n_peaks)

        popt = result.popt
        pcov = result.pcov

        np.concatenate([f, f])
        np.concatenate([G, B])

        f_pred = np.linspace(f.min(), f.max(), 200)
        x_pred = np.concatenate([f_pred, f_pred])

        calc = UncertaintyCalculator()
        band = calc.compute_band(
            model=model,
            x=x_pred,
            popt=popt,
            pcov=pcov,
            confidence_level=confidence_level,
        )

        if ax is None:
            fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
        else:
            fig = ax.figure
            axes = [ax, ax]

        n = len(f_pred)

        # Plot G (conductance)
        axes[0].scatter(f, G, color="black", alpha=0.6, s=10, label="G data")
        axes[0].plot(f_pred, band.y_fit[:n], "b-", linewidth=2, label="G fit")
        axes[0].fill_between(
            f_pred,
            band.y_lower[:n],
            band.y_upper[:n],
            alpha=0.3,
            color="blue",
            label=f"{confidence_level:.0%} CI",
        )
        axes[0].set_ylabel("G (mS)")
        axes[0].legend()
        axes[0].set_title(f"Peak Fit - {chn_name} Harmonic {harm}")

        # Plot B (susceptance)
        axes[1].scatter(f, B, color="black", alpha=0.6, s=10, label="B data")
        axes[1].plot(f_pred, band.y_fit[n:], "r-", linewidth=2, label="B fit")
        axes[1].fill_between(
            f_pred,
            band.y_lower[n:],
            band.y_upper[n:],
            alpha=0.3,
            color="red",
            label=f"{confidence_level:.0%} CI",
        )
        axes[1].set_ylabel("B (mS)")
        axes[1].set_xlabel("Frequency (Hz)")
        axes[1].legend()

        fig.tight_layout()
        return fig


if __name__ == "__main__":
    # Test the JAX model functions
    logger.info("Testing JAX model functions...")

    # Create synthetic test data
    f = np.linspace(4.99e6, 5.01e6, 200)
    amp_true = 0.001
    cen_true = 5.0e6
    wid_true = 1000.0
    phi_true = 0.1

    # Generate G and B with noise
    G_true = fun_G_numpy(f, amp_true, cen_true, wid_true, phi_true) + 1e-5
    B_true = fun_B_numpy(f, amp_true, cen_true, wid_true, phi_true)

    noise = np.random.normal(0, 1e-7, len(f))
    G = G_true + noise
    B = B_true + noise

    logger.info("  Test data generated: %s points", len(f))
    logger.info(
        "  True parameters: amp=%s, cen=%s, wid=%s, phi=%s",
        amp_true,
        cen_true,
        wid_true,
        phi_true,
    )

    # Test NLSQ fitting directly
    logger.info("Testing NLSQ curve_fit...")

    model = create_composite_model(1)

    # Stack x and y data
    x_stacked = np.concatenate([f, f])
    y_data = np.concatenate([G, B])

    p0 = [amp_true * 0.9, cen_true, wid_true * 1.1, phi_true, 1e-5, 0.0]
    bounds = (
        [0, f.min(), 100, -np.pi / 2, -np.inf, -np.inf],
        [np.inf, f.max(), 10000, np.pi / 2, np.inf, np.inf],
    )

    popt, pcov = curve_fit(model, x_stacked, y_data, p0=p0, bounds=bounds)

    logger.info("  Fitted parameters:")
    logger.info("    amp = %.6g (true: %s)", popt[0], amp_true)
    logger.info("    cen = %.6g (true: %s)", popt[1], cen_true)
    logger.info("    wid = %.6g (true: %s)", popt[2], wid_true)
    logger.info("    phi = %.6g (true: %s)", popt[3], phi_true)

    # Calculate relative errors
    rel_err_amp = abs(popt[0] - amp_true) / amp_true * 100
    rel_err_cen = abs(popt[1] - cen_true) / cen_true * 100
    rel_err_wid = abs(popt[2] - wid_true) / wid_true * 100

    logger.info("  Relative errors:")
    logger.info("    amp: %.4f%%", rel_err_amp)
    logger.info("    cen: %.4f%%", rel_err_cen)
    logger.info("    wid: %.4f%%", rel_err_wid)

    if rel_err_cen < 0.1 and rel_err_wid < 0.1:
        logger.info("  SUCCESS: Fitting accuracy within 0.1%% tolerance!")
    else:
        logger.warning("  WARNING: Fitting accuracy exceeds 0.1%% tolerance")

    logger.info("Migration test complete.")
