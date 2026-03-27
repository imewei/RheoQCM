"""Background workers for Bayesian fitting.

T064: BayesianFitWorker for background MCMC execution.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

from PyQt6.QtCore import QThread, pyqtSignal

from rheoQCM.core.constants import MCMC_PRODUCTION

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np
    import numpy.typing as npt

    from rheoQCM.core.bayesian import PriorSpec

    Float64Array = npt.NDArray[np.float64]
    ModelFunc = Callable[..., Float64Array]

_logger = logging.getLogger(__name__)


class BayesianFitWorker(QThread):
    """QThread worker for background Bayesian MCMC fitting.

    Runs BayesianFitter.fit() in a background thread to avoid blocking
    the GUI during long MCMC sampling.

    Thread safety
    -------------
    Cancellation uses a ``threading.Event`` for safe cross-thread
    signalling.  The ``finished`` signal is emitted on every code-path
    (success, error, *and* cancellation) so that modal progress dialogs
    never hang.

    Signals
    -------
    started
        Emitted when MCMC fitting begins
    progress
        Emitted with (percent: int, phase: str) during warmup/sampling
    finished
        Emitted with BayesianFitResult on successful completion
    error
        Emitted with error message string on failure
    cancelled
        Emitted when fitting is cancelled by user
    """

    started = pyqtSignal()
    progress = pyqtSignal(int, str)  # percent, phase
    finished = pyqtSignal(object)  # BayesianFitResult
    error = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(
        self,
        model: ModelFunc,
        x: Float64Array,
        y: Float64Array,
        param_names: list[str],
        *,
        priors: dict[str, PriorSpec] | None = None,
        n_chains: int = MCMC_PRODUCTION.n_chains,
        n_samples: int = MCMC_PRODUCTION.n_samples,
        n_warmup: int = MCMC_PRODUCTION.n_warmup,
        seed: int | None = None,
        parent=None,
    ) -> None:
        """Initialize the worker.

        Parameters
        ----------
        model : ModelFunc
            Model function f(x, *params)
        x : Float64Array
            X-data
        y : Float64Array
            Y-data
        param_names : list[str]
            Parameter names
        priors : dict[str, PriorSpec] | None
            Custom priors (optional)
        n_chains : int
            Number of MCMC chains (default: 4)
        n_samples : int
            Samples per chain (default: 2000)
        n_warmup : int
            Warmup iterations per chain (default: 1000)
        seed : int | None
            Random seed (optional)
        parent : QObject | None
            Parent QObject (optional)
        """
        super().__init__(parent)
        self.model = model
        self.x = x
        self.y = y
        self.param_names = param_names
        self.priors = priors
        self.n_chains = n_chains
        self.n_samples = n_samples
        self.n_warmup = n_warmup
        self.seed = seed
        self._cancel_event = threading.Event()

    def run(self) -> None:
        """Execute Bayesian fitting in background thread."""
        try:
            from rheoQCM.core.bayesian import BayesianFitter

            self.started.emit()
            self.progress.emit(0, "Initializing")

            fitter = BayesianFitter(
                n_chains=self.n_chains,
                n_samples=self.n_samples,
                n_warmup=self.n_warmup,
                seed=self.seed,
            )

            self.progress.emit(5, "Running NLSQ warm-start")

            if self._cancel_event.is_set():
                _logger.info("Bayesian fit cancelled during warm-start")
                self.cancelled.emit()
                return

            self.progress.emit(10, "Starting MCMC sampling")

            result = fitter.fit(
                model=self.model,
                x=self.x,
                y=self.y,
                param_names=self.param_names,
                priors=self.priors,
            )

            if self._cancel_event.is_set():
                _logger.info("Bayesian fit cancelled after sampling")
                self.cancelled.emit()
                return

            self.progress.emit(100, "Complete")
            self.finished.emit(result)

        except ImportError:
            self.error.emit(
                "Bayesian fitting requires NumPyro. Install with: uv add numpyro"
            )
        except (ValueError, TypeError) as e:
            self.error.emit(f"Invalid fitting parameters: {e}")
        except RuntimeError as e:
            self.error.emit(f"MCMC sampling failed: {e}")
        except Exception as e:
            _logger.error("Unexpected error in Bayesian worker: %s", e, exc_info=True)
            self.error.emit(f"Unexpected error: {e}")

    def cancel(self) -> None:
        """Request cancellation of the fitting process.

        Thread-safe: uses ``threading.Event`` for cross-thread signalling.
        """
        self._cancel_event.set()

    def is_cancelled(self) -> bool:
        """Check if cancellation was requested.

        Returns
        -------
        bool
            True if cancel() has been called.
        """
        return self._cancel_event.is_set()
