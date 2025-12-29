"""
QCM UI Wrapper Module (Layer 3).

This module provides a thin UI wrapper around the core physics and model layers.
It delegates all physics calculations to rheoQCM.core.physics and fitting logic
to rheoQCM.core.model.QCMModel.

This module is designed for backward compatibility with the existing UI and
scripting interfaces. For new code, consider using rheoQCM.core.model directly.

Note: Different from other modules, the harmonics used in this module are all INT.

Changes in Phase 4 (T034-T037):
- Removed scipy.optimize import (T033, T035)
- calc_delfstar now delegates to core.multilayer for LL calculation (T035)
- delfstarcalc uses core multilayer functions (T035)
- All fitting methods delegate to QCMModel.solve_properties (T036)
"""

import importlib
import logging
from typing import Any

import numpy as np
import pandas as pd

from rheoQCM.core import multilayer as _core_multilayer
from rheoQCM.core import physics
from rheoQCM.core.model import (
    QCMModel,
    bulk_drho,
    dlam_refh_range,
    drho_range,
    grho_refh_range,
    phi_range,
)

logger = logging.getLogger(__name__)

# Check for kww module
kww_spec = importlib.util.find_spec("kww")
found_kww = kww_spec is not None
if found_kww:
    from kww import kwwc, kwws
else:
    logger.info("kww module is not found!")

# Re-export constants from physics for backward compatibility
Zq = {"AT": physics.Zq, "BT": 0e6}

# Re-export fitting ranges for backward compatibility
dlam_refh_range = dlam_refh_range
drho_range = drho_range
grho_refh_range = grho_refh_range
phi_range = phi_range
bulk_drho = bulk_drho

# Additional constants
e26 = physics.e26
d26 = physics.d26
g0 = physics.g0_default
dq = physics.dq
epsq = physics.epsq
eps0 = physics.eps0
C0byA = physics.C0byA

# Default property dictionaries
prop_default = {
    "electrode": {"calc": False, "grho": 3.0e17, "phi": 0, "drho": 2.8e-6, "n": 3},
    "air": {"calc": False, "grho": 0, "phi": 0, "drho": 0, "n": 1},
    "water": {"calc": False, "grho": 1e8, "phi": np.pi / 2, "drho": bulk_drho, "n": 3},
}


def nh2i(nh: int | str) -> int:
    """
    Convert harmonic (str) to index (int).

    Since only odd harmonics are stored in list.
    """
    if isinstance(nh, str):
        nh = int(nh)
    return int((nh - 1) / 2)


def nhcalc2nh(nhcalc: str) -> list[int]:
    """Convert nhcalc (str) to list of harmonics (int) in nhcalc."""
    return [int(s) for s in nhcalc]


class QCM:
    """
    Thin UI wrapper for QCM analysis.

    This class provides backward-compatible access to QCM physics and fitting
    by delegating to rheoQCM.core.physics and rheoQCM.core.model.QCMModel.

    Parameters
    ----------
    cut : {"AT", "BT"}, optional
        Crystal cut type. Default: "AT".

    Attributes
    ----------
    Zq : float
        Acoustic impedance of quartz [Pa s/m].
    f1 : float or None
        Fundamental resonant frequency [Hz].
    refh : int or None
        Reference harmonic for calculations.
    calctype : str
        Calculation type: "SLA", "LL", or "Voigt".
    """

    def __init__(self, cut: str = "AT") -> None:
        """Initialize QCM with given crystal cut."""
        self.Zq = Zq[cut]
        self.f1: float | None = None
        self.g1: float | None = None
        self.f0s: dict[int, float] | None = None
        self.g0s: dict[int, float] | None = None
        self.g_err_min: float = 1
        self.f_err_min: float = 1
        self.err_frac: float = 3e-2

        self.refh: int | None = None
        self.calctype: str = "SLA"
        self.piezoelectric_stiffening: bool = False

        # Internal model instance (lazy initialization)
        self._model: QCMModel | None = None

    def _get_model(self) -> QCMModel:
        """
        Get or create internal model instance (T034).

        This method ensures the QCMModel is synchronized with the QCM wrapper state.
        Creates a new model if one doesn't exist, otherwise syncs state to existing model.

        Returns
        -------
        QCMModel
            The internal model instance with synced state.
        """
        if self._model is None:
            self._model = QCMModel(
                f1=self.f1,
                refh=self.refh,
                calctype=self.calctype,
            )
        else:
            # Sync state
            self._model.f1 = self.f1
            self._model.refh = self.refh
            self._model.calctype = self.calctype
            self._model.g1 = self.g1
            if self.f0s:
                self._model.f0s = self.f0s
            if self.g0s:
                self._model.g0s = self.g0s
        return self._model

    def get_prop_by_name(self, name: str) -> dict:
        """Get property dictionary by name."""
        return prop_default.get(name, prop_default["air"])

    def fstar_err_calc(self, delfstar: complex) -> complex:
        """Calculate the error in delfstar."""
        f_err = self.f_err_min + self.err_frac * np.imag(delfstar)
        g_err = self.g_err_min + self.err_frac * np.imag(delfstar)
        return complex(f_err, g_err)

    # =========================================================================
    # Physics functions - delegate to core.physics
    # =========================================================================

    def sauerbreyf(self, n: int, drho: float) -> float:
        """Calculate delf_sn from Sauerbrey equation. Delegates to core.physics."""
        return float(physics.sauerbreyf(n, drho, f1=self.f1))

    def sauerbreym(self, n: int, delf: float) -> float:
        """Calculate mass from Sauerbrey equation. Delegates to core.physics."""
        return float(physics.sauerbreym(n, delf, f1=self.f1))

    def grho(self, n: int, grho_refh: float, phi: float) -> float:
        """Calculate grho of n_th harmonic. Delegates to core.physics."""
        return float(physics.grho(n, grho_refh, phi, refh=self.refh))

    def grhostar_from_refh(self, n: int, grho_refh: float, phi: float) -> complex:
        """Calculate complex G*rho at harmonic n. Delegates to core.physics."""
        return complex(physics.grhostar_from_refh(n, grho_refh, phi, refh=self.refh))

    def grhostar(self, grho_val: float, phi: float) -> complex:
        """Return complex value of grhostar from grho (modulus) and phi."""
        return complex(physics.grhostar(grho_val, phi))

    def grho_from_dlam(
        self, n: int, drho: float, dlam_refh: float, phi: float
    ) -> float:
        """Calculate grho from d/lambda."""
        return float(physics.grho_from_dlam(n, drho, dlam_refh, phi, f1=self.f1))

    def etarho(self, n: int, grho_n: float) -> float:
        """Calculate viscosity at nth harmonic. Delegates to core.physics."""
        return float(physics.etarho(n, grho_n, f1=self.f1))

    def calc_lamrho(self, n: int, grho_n: float, phi_n: float) -> float:
        """Calculate rho*lambda. Delegates to core.physics."""
        return float(physics.calc_lamrho(n, grho_n, phi_n, f1=self.f1))

    def calc_delrho(self, n: int, grho: float, phi: float) -> float:
        """Calculate decay length. Delegates to core.physics."""
        return float(physics.calc_deltarho(n, grho, phi, f1=self.f1))

    def zstarbulk(self, grhostar_val: complex) -> complex:
        """Calculate acoustic impedance from complex modulus."""
        return complex(physics.zstar_bulk(np.array(grhostar_val)))

    def calc_delfstar_sla(self, ZL: complex) -> complex:
        """Calculate complex frequency shift using SLA. Delegates to core.physics."""
        return complex(physics.calc_delfstar_sla(np.array(ZL), f1=self.f1))

    # =========================================================================
    # Helper functions for calculations
    # =========================================================================

    def D(
        self, n: int, grho_refh: float, phi: float, drho: float
    ) -> complex | np.ndarray:
        """Calculate D parameter."""
        grho_n = self.grho(n, grho_refh, phi)
        return (
            2
            * np.pi
            * drho
            * n
            * self.f1
            * (np.cos(phi / 2) - 1j * np.sin(phi / 2))
            / grho_n**0.5
        )

    def DfromZ(self, n: int, drho: float, Zstar: complex) -> complex:
        """Calculate D from impedance."""
        return 2 * np.pi * n * self.f1 * drho / Zstar

    def zstarfilm(self, n: int, drho: float, grhostar_val: complex) -> complex:
        """Calculate film acoustic impedance."""
        if grhostar_val == 0:
            return 0
        zstar = self.zstarbulk(grhostar_val)
        return zstar * np.tan(2 * np.pi * n * self.f1 * drho / zstar)

    def rstar(
        self,
        n: int,
        grho_refh: float,
        phi: float,
        drho: float,
        overlayer: dict | None = None,
    ) -> complex:
        """Calculate reflection coefficient."""
        if overlayer is None:
            overlayer = {"drho": 0, "grho_refh": 0, "phi": 0}

        grhostar_1 = self.grhostar_from_refh(n, grho_refh, phi)
        grhostar_2 = self.grhostar_from_refh(
            n, overlayer.get("grho_refh", 0), overlayer.get("phi", 0)
        )
        zstar_1 = self.zstarbulk(grhostar_1)
        zstar_2 = self.zstarfilm(n, overlayer.get("drho", 0), grhostar_2)
        return zstar_2 / zstar_1

    def delfstarcalc(
        self,
        n: int,
        grho_refh: float,
        phi: float,
        drho: float,
        overlayer: dict | None = None,
    ) -> complex | np.ndarray:
        """
        Calculate complex frequency shift for single layer (T035).

        Delegates to core.multilayer for consistent physics.
        """
        # Build layer dict for core multilayer function
        layer = {
            1: {
                "grho": grho_refh,
                "phi": phi,
                "drho": drho,
                "n": self.refh if self.refh else 3,
            }
        }

        # Add overlayer if provided
        if overlayer is not None and overlayer.get("drho", 0) > 0:
            layer[2] = {
                "grho": overlayer.get("grho_refh", 0),
                "phi": overlayer.get("phi", 0),
                "drho": overlayer.get("drho", 0),
                "n": self.refh if self.refh else 3,
            }

        # Use core multilayer function (SLA for delfstarcalc)
        result = _core_multilayer.calc_delfstar_multilayer(
            n, layer, f1=self.f1, calctype="SLA", refh=self.refh if self.refh else 3
        )
        return complex(result)

    def d_lamcalc(self, n: int, grho_refh: float, phi: float, drho: float) -> float:
        """Calculate d/lambda."""
        return (
            drho * n * self.f1 * np.cos(phi / 2) / np.sqrt(self.grho(n, grho_refh, phi))
        )

    def thin_film_gamma(self, n: int, drho: float, jdprime_rho: float) -> float:
        """Calculate thin film gamma."""
        return 8 * np.pi**2 * n**3 * self.f1**4 * drho**3 * jdprime_rho / (3 * self.Zq)

    def grho_refh_func(self, jdprime_rho_refh: float, phi: float) -> float:
        """Calculate grho_refh from jdprime_rho_refh."""
        return np.sin(phi) / jdprime_rho_refh

    def grho_from_material(self, n: int, material: dict) -> float:
        """Calculate grho at harmonic n from material properties."""
        grho_refh = material["grho"]
        phi = material["phi"]
        refh = material["n"]
        return grho_refh * (n / refh) ** (phi / (np.pi / 2))

    def calc_grho_ncalc(
        self, grhostar_val: complex, n: int, ncalc: int
    ) -> tuple[float, float]:
        """Calculate grho at ncalc from grhostar at n."""
        phi = np.angle(grhostar_val)
        grho_n = abs(grhostar_val)
        grho_calc = grho_n * (ncalc / n) ** (phi / (np.pi / 2))
        return grho_calc, phi

    def calc_grho_refh(self, grhostar_val: complex, n: int) -> tuple[float, float]:
        """Calculate grho_refh from grhostar of nth harmonic."""
        return self.calc_grho_ncalc(grhostar_val, n, self.refh)

    def calc_D(self, n: int, material: dict, delfstar: complex) -> complex:
        """Calculate D for a material layer."""
        drho = material["drho"]
        if drho == 0:
            return 0
        return (
            2 * np.pi * (n * self.f1 + delfstar) * drho / self.zstar_bulk(n, material)
        )

    def zstar_bulk(self, n: int, material: dict) -> complex:
        """Calculate bulk acoustic impedance for material."""
        if self.calctype.upper() != "VOIGT":
            grho_n = self.grho_from_material(n, material)
            phi = material["phi"]
            grhostar_val = self.grhostar(grho_n, phi)
        else:
            r = material["n"]
            grho_r = material["grho"]
            greal = grho_r * np.cos(material["phi"])
            gimag = (n / r) * np.sin(material["phi"])
            grhostar_val = (gimag**2 + greal**2) ** 0.5 * np.exp(1j * material["phi"])
        return grhostar_val**0.5

    def dlam(self, n: int, dlam_refh: float, phi: float) -> float:
        """Calculate d/lambda at harmonic n."""
        return dlam_refh * (n / self.refh) ** (1 - phi / np.pi)

    def normdelfstar(self, n: int, dlam_refh: float, phi: float) -> complex:
        """Calculate normalized delfstar."""
        dlam_n = self.dlam(n, dlam_refh, phi)
        D = 2 * np.pi * dlam_n * (1 - 1j * np.tan(phi / 2))
        # Handle D=0 case: limit of tan(D)/D as D→0 is 1, so -tan(D)/D → -1
        with np.errstate(divide="ignore", invalid="ignore"):
            result = -np.tan(D) / D
        # For D=0, the limit is -1 (thin film Sauerbrey limit)
        if np.isscalar(D):
            if D == 0:
                return -1.0 + 0j
        return result

    def calc_drho(self, n1: int, delfstar: dict, dlam_refh: float, phi: float) -> float:
        """Calculate drho from delfstar."""
        return self.sauerbreym(n1, np.real(delfstar[n1])) / np.real(
            self.normdelfstar(n1, dlam_refh, phi)
        )

    def rhcalc(self, nh: list[int], dlam_refh: float, phi: float) -> float:
        """Calculate harmonic ratio (calculated)."""
        return np.real(self.normdelfstar(nh[0], dlam_refh, phi)) / np.real(
            self.normdelfstar(nh[1], dlam_refh, phi)
        )

    def rh_from_delfstar(self, nh: list[int], delfstar: dict) -> float:
        """Calculate harmonic ratio from delfstar."""
        n1 = int(nh[0])
        n2 = int(nh[1])
        if np.real(delfstar[n2]) == 0:
            return np.nan
        return (n2 / n1) * np.real(delfstar[n1]) / np.real(delfstar[n2])

    def rdcalc(self, nh: list[int], dlam_refh: float, phi: float) -> float:
        """Calculate dissipation ratio (calculated)."""
        return -np.imag(self.normdelfstar(nh[2], dlam_refh, phi)) / np.real(
            self.normdelfstar(nh[2], dlam_refh, phi)
        )

    def rdexp(self, nh: list[int], delfstar: dict) -> float:
        """Calculate dissipation ratio (experimental)."""
        return -np.imag(delfstar[nh[2]]) / np.real(delfstar[nh[2]])

    def rd_from_delfstar(self, n: int, delfstar: dict) -> float:
        """Calculate dissipation ratio at harmonic n."""
        if np.real(delfstar[n]) == 0:
            return np.nan
        return -np.imag(delfstar[n]) / np.real(delfstar[n])

    def delrho_bulk(self, n: int, delfstar: dict) -> float:
        """Calculate decay length * density for bulk material."""
        return (
            -self.Zq
            * abs(delfstar[n]) ** 2
            / (2 * n * self.f1**2 * np.real(delfstar[n]))
        )

    def delfstarcalc_bulk(self, n: int, grho_refh: float, phi: float) -> complex:
        """Calculate complex frequency shift for bulk layer."""
        return (self.f1 * np.sqrt(self.grho(n, grho_refh, phi)) / (np.pi * self.Zq)) * (
            -np.sin(phi / 2) + 1j * np.cos(phi / 2)
        )

    def delfstarcalc_bulk_from_film(self, n: int, film: dict) -> complex:
        """Calculate bulk delfstar from film dictionary."""
        material = self.get_calc_material(film)
        grho_refh = self.grho_from_material(self.refh, material)
        phi = material["phi"]
        return self.delfstarcalc_bulk(n, grho_refh, phi)

    def bulk_guess(self, delfstar: dict) -> list[float]:
        """Get bulk solution for dlam_refh and phi."""
        grho_refh = self.grho_bulk(delfstar)
        phi = self.phi_bulk(delfstar)
        dlam_refh = self.bulk_dlam_refh(grho_refh, phi)
        return [dlam_refh, phi]

    def bulk_dlam_refh(self, grho_refh: float, phi: float) -> float:
        """Get bulk d/lambda at reference harmonic."""
        lamrho_refh = self.calc_lamrho(self.refh, grho_refh, phi)
        drho = lamrho_refh / 4
        film = {"drho": drho, "grho": grho_refh, "phi": phi, "n": self.refh}
        return self.calc_dlam(self.refh, film)

    def grho_bulk(self, delfstar: dict, n: int | None = None) -> float:
        """Calculate grho from bulk model."""
        if n is None:
            n = self.refh
        return (np.pi * self.Zq * abs(delfstar[n]) / self.f1) ** 2

    def phi_bulk(self, delfstar: dict, n: int | None = None) -> float:
        """Calculate phi from bulk model."""
        if n is None:
            n = self.refh
        return min(
            np.pi / 2, -2 * np.arctan(np.real(delfstar[n]) / np.imag(delfstar[n]))
        )

    def bulk_props(self, delfstar: dict, n: int | None = None) -> list:
        """Get bulk properties [grho, phi, drho]."""
        return [
            self.grho_bulk(delfstar, n),
            self.phi_bulk(delfstar, n),
            bulk_drho,
        ]

    def isbulk(self, rd_exp: float, bulklimit: float) -> bool:
        """Check if material is bulk based on dissipation ratio."""
        return rd_exp >= bulklimit

    def convert_D_to_gamma(self, D_dsiptn: float, n: int) -> float:
        """Convert D (dissipation) to gamma."""
        return 0.5 * n * self.f1 * D_dsiptn

    def convert_gamma_to_D(self, gamma: float, n: int) -> float:
        """Convert gamma to D (dissipation)."""
        return 2 * gamma / (n * self.f1)

    # =========================================================================
    # Film structure functions - delegate to core.multilayer (T035, T037)
    # =========================================================================

    def calc_ZL(self, n: int, layers: dict, delfstar: complex) -> complex:
        """
        Calculate load impedance for multi-layer structure.

        Delegates to core.multilayer.calc_ZL (T035).
        """
        if not layers:
            return 0

        # Use core multilayer function
        result = _core_multilayer.calc_ZL(
            n,
            layers,
            delfstar,
            f1=self.f1,
            calctype=self.calctype,
            refh=self.refh if self.refh else 3,
        )
        return complex(result)

    def calc_delfstar(self, n: int, layers: dict) -> complex:
        """
        Calculate complex frequency shift for layer structure (T035).

        Delegates to core.multilayer.calc_delfstar_multilayer.
        Removed scipy.optimize dependency (T033).
        """
        if not layers:
            return np.nan

        # Use core multilayer function which handles both SLA and LL
        layers_without_0 = self.remove_layer_0(layers)

        if self.calctype.upper() == "SLA":
            result = _core_multilayer.calc_delfstar_multilayer(
                n,
                layers_without_0,
                f1=self.f1,
                calctype="SLA",
                refh=self.refh if self.refh else 3,
            )
            return complex(result)
        elif self.calctype.upper() == "LL":
            # Add electrode if not present
            layers_with_electrode = layers.copy()
            if 0 not in layers_with_electrode:
                layers_with_electrode[0] = prop_default["electrode"]

            # Use core multilayer function for LL calculation
            result = _core_multilayer.calc_delfstar_multilayer(
                n,
                layers_with_electrode,
                f1=self.f1,
                calctype="LL",
                refh=self.refh if self.refh else 3,
                g0=self.g1 if self.g1 else g0,
            )
            return complex(result)
        else:
            return np.nan

    def calc_delfstar_from_single_material(self, n: int, material: dict) -> complex:
        """Calculate delfstar from single material."""
        layers = self.build_single_layer_film(material)
        return self.calc_delfstar(n, layers)

    def calc_Zmot(self, n: int, layers: dict, delfstar: complex) -> complex:
        """
        Calculate motional impedance.

        Delegates to core.multilayer.calc_Zmot (T035).
        """
        # Use core multilayer function
        result = _core_multilayer.calc_Zmot(
            n,
            layers,
            delfstar,
            f1=self.f1,
            calctype=self.calctype,
            refh=self.refh if self.refh else 3,
            g0=self.g1 if self.g1 else g0,
        )
        return complex(result)

    def calc_dlam(self, n: int, film: dict) -> float:
        """Calculate d/lambda for a film."""
        return np.real(self.calc_D(n, film, 0)) / (2 * np.pi)

    def get_calc_layer_num(self, film: dict) -> int | None:
        """Get layer number with calc=True."""
        for n, layer in film.items():
            if "calc" in layer and layer["calc"]:
                return n
        return None

    def get_calc_material(self, film: dict) -> dict | None:
        """Get material of the layer with calc=True."""
        n = self.get_calc_layer_num(film)
        if n is None:
            return None
        return film[n]

    def get_calc_layer(self, film: dict) -> dict:
        """Get film containing only the calc=True layer."""
        new_film = {}
        n = self.get_calc_layer_num(film)
        if n is not None:
            new_film[n] = film[n]
        return new_film

    def set_calc_layer_val(
        self, film: dict, grho: float, phi: float, drho: float
    ) -> dict:
        """Set values for the calc=True layer."""
        if not film:
            film = self.build_single_layer_film()

        calcnum = self.get_calc_layer_num(film)
        film[calcnum].update(grho=grho, phi=phi, drho=drho, n=self.refh)
        return film

    def build_single_layer_film(self, material: dict | None = None) -> dict:
        """Build a single layer film for calculation."""
        if material is None:
            return self.replace_layer_0_prop_with_known(
                {0: {"calc": False}, 1: {"calc": True}}
            )
        else:
            new_film = self.replace_layer_0_prop_with_known(
                {0: {"calc": False}, 1: {**material}}
            )
            new_film[1]["calc"] = True
            return new_film

    def get_ref_layers(self, film: dict) -> dict:
        """Get layers with calc=False."""
        new_film = {}
        for n, layer in film.items():
            if "calc" not in layer:
                layer["calc"] = False
            if not layer["calc"]:
                new_film[n] = layer
        return new_film

    def separate_calc_ref_layers(self, film: dict) -> tuple[dict, dict]:
        """Separate calc and reference layers."""
        return self.get_calc_layer(film), self.get_ref_layers(film)

    def remove_layer_0(self, film: dict) -> dict:
        """Remove layer 0 from film."""
        new_film = film.copy()
        new_film.pop(0, None)
        return new_film

    def replace_layer_0_prop_with_known(self, film: dict) -> dict:
        """Replace layer 0 with electrode properties."""
        new_film = film.copy()
        if 0 in new_film.keys():
            new_film[0] = prop_default["electrode"]

        for n, layer in new_film.items():
            if "calc" not in layer:
                new_film[n]["calc"] = False

        return new_film

    # =========================================================================
    # Main solving functions - delegate to core.model (T036)
    # =========================================================================

    def guess_from_props(self, film: dict) -> list[float]:
        """Guess dlam_refh and phi from film properties."""
        dlam_refh = self.calc_dlam(self.refh, film)
        phi = self.get_calc_material(film)["phi"]
        return [dlam_refh, phi]

    def thinfilm_guess(
        self, delfstar: dict, nh: list[int]
    ) -> tuple[float, float, float, float]:
        """Guess thin film properties from delfstar."""
        model = self._get_model()
        model.load_delfstars(delfstar)
        return model._thin_film_guess(delfstar, nh)

    def solve_general_delfstar_to_prop(
        self,
        nh: list[int],
        delfstar: dict,
        film: dict,
        calctype: str | None = None,
        prop_guess: dict | None = None,
        bulklimit: float = 0.5,
    ) -> tuple[float, float, float, float, dict]:
        """
        Solve film properties from delfstar using core.model (T036).

        Parameters
        ----------
        nh : list[int]
            Harmonics for calculation [n1, n2, n3].
        delfstar : dict
            Complex frequency shifts {harm: complex}.
        film : dict
            Film structure definition.
        calctype : str, optional
            Calculation type: "SLA" or "LL".
        prop_guess : dict, optional
            Property guess for initialization.
        bulklimit : float, optional
            Dissipation ratio threshold for bulk. Default: 0.5.

        Returns
        -------
        tuple
            (grho_refh, phi, drho, dlam_refh, err)
        """
        if calctype is not None:
            self.calctype = calctype

        # Initialize error dict
        err = {"grho_refh": np.nan, "phi": np.nan, "drho": np.nan}

        # Use core model for solving (T036)
        model = self._get_model()
        model.load_delfstars(delfstar)

        try:
            result = model.solve_properties(
                nh=nh,
                calctype=self.calctype,
                bulklimit=bulklimit,
                calculate_errors=True,
            )

            # Access SolveResult attributes directly (Phase 7 fix)
            grho_refh = result.grho_refh
            phi = result.phi
            drho = result.drho
            dlam_refh = result.dlam_refh
            err = result.errors

        except Exception as e:
            logger.exception("Error solving film properties: %s", e)
            grho_refh, phi, drho, dlam_refh = np.nan, np.nan, np.nan, np.nan

        return grho_refh, phi, drho, dlam_refh, err

    def solve_single_queue_to_prop(
        self,
        nh: list[int],
        qcm_queue: pd.DataFrame,
        calctype: str | None = None,
        film: dict | None = None,
        bulklimit: float = 0.5,
    ) -> tuple[float, float, float, float, dict]:
        """Solve properties from QCM queue data."""
        if calctype is not None:
            self.calctype = calctype

        if film is None:
            film = {}

        # Extract delfstar from queue
        delfstars = qcm_queue.delfstars.iloc[0]
        delfstar = {int(i * 2 + 1): dfstar for i, dfstar in enumerate(delfstars)}

        # Set f1 from queue
        f0s = qcm_queue.f0s.iloc[0]
        f0s = {int(i * 2 + 1): f0 for i, f0 in enumerate(f0s)}
        g0s = qcm_queue.g0s.iloc[0]
        g0s = {int(i * 2 + 1): g0 for i, g0 in enumerate(g0s)}

        self.f0s = f0s
        self.g0s = g0s

        if np.isnan(list(f0s.values())).all():
            self.f1 = np.nan
        else:
            for k in sorted(f0s.keys()):
                if not np.isnan(f0s[k]):
                    self.f1 = f0s[k] / k
                    self.g1 = g0s[k] / k
                    break

        return self.solve_general_delfstar_to_prop(
            nh, delfstar, film, bulklimit=bulklimit
        )

    def solve_single_queue(
        self,
        nh: list[int],
        qcm_queue: pd.DataFrame,
        mech_queue: pd.DataFrame,
        calctype: str | None = None,
        film: dict | None = None,
        bulklimit: float = 0.5,
    ) -> pd.DataFrame:
        """Solve properties and populate mech_queue with results."""
        if calctype is not None:
            self.calctype = calctype

        if film is None:
            film = {}

        film = self.replace_layer_0_prop_with_known(film)

        grho_refh, phi, drho, dlam_refh, err = self.solve_single_queue_to_prop(
            nh, qcm_queue, film=film, bulklimit=bulklimit
        )

        film = self.set_calc_layer_val(film, grho_refh, phi, drho)

        marks = qcm_queue.marks.iloc[0]
        delfstars = qcm_queue.delfstars.iloc[0]
        delfstar = {int(i * 2 + 1): dfstar for i, dfstar in enumerate(delfstars)}
        rd_exp = self.rd_from_delfstar(nh[2], delfstar)

        nhplot = [
            i * 2 + 1
            for i, mark in enumerate(marks)
            if (not np.isnan(mark)) and (mark is not None)
        ]

        delfstar_calc = {}
        delfsn = {
            i * 2 + 1: self.sauerbreyf(i * 2 + 1, drho) for i, _ in enumerate(marks)
        }
        normdelfstar_calcs = {}

        # Copy queue data
        delf_exps = mech_queue.delf_exps.iloc[0].copy()
        delfn_exps = mech_queue.delfn_exps.iloc[0].copy()
        delf_calcs = mech_queue.delf_calcs.iloc[0].copy()
        delfn_calcs = mech_queue.delfn_calcs.iloc[0].copy()
        delg_calcs = mech_queue.delg_calcs.iloc[0].copy()
        delD_exps = mech_queue.delD_exps.iloc[0].copy()
        delD_calcs = mech_queue.delD_calcs.iloc[0].copy()
        sauerbreyms = mech_queue.sauerbreyms.iloc[0].copy()
        rd_exps = mech_queue.rd_exps.iloc[0].copy()
        rd_calcs = mech_queue.rd_calcs.iloc[0].copy()
        dlams = mech_queue.dlams.iloc[0].copy()
        lamrhos = mech_queue.lamrhos.iloc[0].copy()
        delrhos = mech_queue.delrhos.iloc[0].copy()
        grhos = mech_queue.dlams.iloc[0].copy()
        grhos_err = mech_queue.dlams.iloc[0].copy()
        etarhos = mech_queue.etarhos.iloc[0].copy()
        etarhos_err = mech_queue.etarhos_err.iloc[0].copy()
        normdelf_exps = mech_queue.normdelf_exps.iloc[0].copy()
        normdelf_calcs = mech_queue.normdelf_calcs.iloc[0].copy()
        normdelg_exps = mech_queue.normdelg_exps.iloc[0].copy()
        normdelg_calcs = mech_queue.normdelg_calcs.iloc[0].copy()

        delf_exps = qcm_queue.delfs.iloc[0]

        for n in nhplot:
            if self.isbulk(rd_exp, bulklimit):
                delfstar_calc[n] = self.delfstarcalc_bulk_from_film(n, film)
            else:
                delfstar_calc[n] = self.calc_delfstar(n, film)

            delfn_exps[nh2i(n)] = delf_exps[nh2i(n)] / n
            delf_calcs[nh2i(n)] = np.real(delfstar_calc[n])
            delfn_calcs[nh2i(n)] = np.real(delfstar_calc[n]) / n
            delg_calcs[nh2i(n)] = np.imag(delfstar_calc[n])

            delD_exps[nh2i(n)] = self.convert_gamma_to_D(np.imag(delfstar[n]), n)
            delD_calcs[nh2i(n)] = self.convert_gamma_to_D(np.imag(delfstar_calc[n]), n)
            sauerbreyms[nh2i(n)] = self.sauerbreym(n, -np.real(delfstar[n]))

            rd_calcs[nh2i(n)] = self.rd_from_delfstar(n, delfstar_calc)
            rd_exps[nh2i(n)] = self.rd_from_delfstar(n, delfstar)

            dlams[nh2i(n)] = self.dlam(n, dlam_refh, phi)
            grhos[nh2i(n)] = self.grho(n, grho_refh, phi)
            grhos_err[nh2i(n)] = self.grho(n, err["grho_refh"], phi)
            etarhos[nh2i(n)] = self.etarho(n, grhos[nh2i(n)])
            etarhos_err[nh2i(n)] = self.etarho(n, grhos_err[nh2i(n)])
            lamrhos[nh2i(n)] = self.calc_lamrho(n, grhos[nh2i(n)], phi)

            if self.isbulk(rd_exp, bulklimit):
                delrhos[nh2i(n)] = self.delrho_bulk(n, delfstar)
            else:
                delrhos[nh2i(n)] = self.calc_delrho(n, grhos[nh2i(n)], phi)

            normdelfstar_calcs[n] = self.normdelfstar(n, dlam_refh, phi)
            normdelf_exps[nh2i(n)] = np.real(delfstar[n]) / delfsn[n]
            normdelf_calcs[nh2i(n)] = np.real(normdelfstar_calcs[n])
            normdelg_exps[nh2i(n)] = np.imag(delfstar[n]) / delfsn[n]
            normdelg_calcs[nh2i(n)] = np.imag(normdelfstar_calcs[n])

        rh_exp = self.rh_from_delfstar(nh, delfstar)
        rh_calc = self.rh_from_delfstar(nh, delfstar_calc)

        tot_harms = len(delf_calcs)
        mech_queue["drho"] = [[drho] * tot_harms]
        mech_queue["drho_err"] = [[err["drho"]] * tot_harms]
        mech_queue["phi"] = [[min(np.pi / 2, phi)] * tot_harms]
        mech_queue["phi_err"] = [[err["phi"]] * tot_harms]
        mech_queue["rh_exp"] = [[rh_exp] * tot_harms]
        mech_queue["rh_calc"] = [[rh_calc] * tot_harms]

        mech_queue["grhos"] = [grhos]
        mech_queue["grhos_err"] = [grhos_err]
        mech_queue["etarhos"] = [etarhos]
        mech_queue["etarhos_err"] = [etarhos_err]
        mech_queue["dlams"] = [dlams]
        mech_queue["lamrhos"] = [lamrhos]
        mech_queue["delrhos"] = [delrhos]

        mech_queue["delf_exps"] = [delf_exps]
        mech_queue["delfn_exps"] = [delfn_exps]
        mech_queue["delf_calcs"] = [delf_calcs]
        mech_queue["delfn_calcs"] = [delfn_calcs]
        mech_queue["delg_exps"] = qcm_queue["delgs"]
        mech_queue["delg_calcs"] = [delg_calcs]
        mech_queue["delD_exps"] = [delD_exps]
        mech_queue["delD_calcs"] = [delD_calcs]
        mech_queue["sauerbreyms"] = [sauerbreyms]
        mech_queue["normdelf_exps"] = [normdelf_exps]
        mech_queue["normdelf_calcs"] = [normdelf_calcs]
        mech_queue["normdelg_exps"] = [normdelg_exps]
        mech_queue["normdelg_calcs"] = [normdelg_calcs]
        mech_queue["rd_exps"] = [rd_exps]
        mech_queue["rd_calcs"] = [rd_calcs]

        return mech_queue

    def all_nhcaclc_harm_not_na(self, nh: list[int], qcm_queue: pd.DataFrame) -> bool:
        """Check if all harmonics in nhcalc are not NA."""
        delfstars = qcm_queue.delfstars.iloc[0]
        if (
            np.isnan(delfstars[nh2i(nh[0])].real)
            or np.isnan(delfstars[nh2i(nh[1])].real)
            or np.isnan(delfstars[nh2i(nh[2])].imag)
        ):
            return False
        return True

    def analyze(
        self,
        nhcalc: str,
        queue_ids: list,
        qcm_df: pd.DataFrame,
        mech_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Calculate with qcm_df and save to mech_df."""
        nh = nhcalc2nh(nhcalc)
        for queue_id in queue_ids:
            idx = qcm_df[qcm_df.queue_id == queue_id].index.astype(int)[0]
            qcm_queue = qcm_df.loc[[idx], :].copy()
            mech_queue = mech_df.loc[[idx], :].copy()

            if self.all_nhcaclc_harm_not_na(nh, qcm_queue):
                mech_queue = self.solve_single_queue(nh, qcm_queue, mech_queue)
                mech_queue.index = [idx]
                mech_df.update(mech_queue)

        return mech_df

    # =========================================================================
    # Unit conversion
    # =========================================================================

    def convert_mech_unit(self, mech_df: pd.DataFrame) -> pd.DataFrame:
        """Convert units from SI to display units."""
        df = mech_df.copy()
        cols = mech_df.columns
        for col in cols:
            if any(st in col for st in ["drho", "lamrho", "delrho", "sauerbreyms"]):
                df[col] = df[col].apply(
                    lambda x: (
                        list(np.array(x) * 1000) if isinstance(x, list) else x * 1000
                    )
                )
            elif any(st in col for st in ["grho", "etarho"]):
                df[col] = df[col].apply(
                    lambda x: (
                        list(np.array(x) / 1000) if isinstance(x, list) else x / 1000
                    )
                )
            elif "phi" in col:
                df[col] = df[col].apply(
                    lambda x: (
                        list(np.rad2deg(x)) if isinstance(x, list) else np.rad2deg(x)
                    )
                )
            elif "D_" in col:
                df[col] = df[col].apply(
                    lambda x: (
                        list(np.array(x) * 1e6) if isinstance(x, list) else x * 1e6
                    )
                )
        return df

    def convert_mech_unit_data(
        self, data: float | np.ndarray | list, varname: str
    ) -> float | np.ndarray:
        """Convert unit of a single data value."""
        if isinstance(data, list):
            data = np.array(data)

        if varname in ["drho", "lamrho", "delrho", "sauerbreyms"]:
            return data * 1000
        elif varname in ["grho", "etarho"]:
            return data / 1000
        elif "phi" in varname:
            return np.rad2deg(data)
        elif "D" in varname:
            return data * 1e6
        return data

    def single_harm_data(self, var: str, qcm_df: pd.DataFrame) -> pd.Series | None:
        """Get variables calculated from single harmonic."""
        if var == "delf_calcs":
            return qcm_df.delfs
        if var == "delg_calcs":
            return qcm_df.delgs
        if var == "normdelfs":
            return None
        if var == "rds":
            s = qcm_df.delfstars.copy()
            s.apply(
                lambda delfstars: [
                    self.rd_from_delfstar(n, delfstars) for n in range(len(delfstars))
                ]
            )
            return s
        return None

    # =========================================================================
    # Springpot functions (kept for backward compatibility)
    # =========================================================================

    def gstar_maxwell(self, wtau: float | np.ndarray) -> complex | np.ndarray:
        """Maxwell element G*."""
        return 1j * wtau / (1 + 1j * wtau)

    @np.vectorize
    def gstar_kww(self, wtau: float, beta: float) -> complex:
        """KWW (stretched exponential) G*."""
        return wtau * (kwws(wtau, beta) + 1j * kwwc(wtau, beta))

    def gstar_rouse(self, wtau: np.ndarray, n_rouse: int) -> np.ndarray:
        """Rouse model G*."""
        n_rouse = int(n_rouse)
        rouse = np.zeros((len(wtau), n_rouse), dtype=complex)
        for p in 1 + np.arange(n_rouse):
            rouse[:, p - 1] = (wtau / p**2) ** 2 / (1 + wtau / p**2) ** 2 + 1j * (
                wtau / p**2
            ) / (1 + wtau / p**2) ** 2
        return rouse.sum(axis=1) / n_rouse

    def springpot(
        self,
        w: np.ndarray,
        g0: np.ndarray | list | float,
        tau: np.ndarray | list | float,
        beta: np.ndarray | list | float,
        sp_type: np.ndarray | list | int,
        **kwargs: Any,
    ) -> np.ndarray:
        """Springpot model for complex modulus."""
        kww_list = kwargs.get("kww", [])
        maxwell = kwargs.get("maxwell", [])

        tau = np.asarray(tau).reshape(1, -1)[0, :]
        beta = np.asarray(beta).reshape(1, -1)[0, :]
        g0 = np.asarray(g0).reshape(1, -1)[0, :]
        sp_type = np.asarray(sp_type).reshape(1, -1)[0, :]

        nw = len(w)
        n_br = len(sp_type)
        n_sp = sp_type.sum()
        sp_comp = np.empty((nw, n_sp), dtype=np.complex128)
        br_g = np.empty((nw, n_br), dtype=np.complex128)

        for i in np.arange(n_sp):
            if i in maxwell:
                sp_comp[:, i] = 1 / (g0[i] * self.gstar_maxwell(w * tau[i]))
            elif i in kww_list:
                sp_comp[:, i] = 1 / (g0[i] * self.gstar_kww(w * tau[i], beta[i]))
            else:
                sp_comp[:, i] = 1 / (g0[i] * (1j * w * tau[i]) ** beta[i])

        sp_vec = np.append(0, sp_type.cumsum())
        for i in np.arange(n_br):
            sp_i = np.arange(sp_vec[i], sp_vec[i + 1])
            br_g[:, i] = 1 / sp_comp[:, sp_i].sum(1)

        return br_g.sum(1)

    def vogel(
        self, T: float | np.ndarray, Tref: float, B: float, Tinf: float
    ) -> float | np.ndarray:
        """Vogel-Fulcher-Tammann equation for shift factor."""
        return -B / (Tref - Tinf) + B / (T - Tinf)


if __name__ == "__main__":
    qcm = QCM()
    qcm.f1 = 5e6
    qcm.refh = 3

    nh = [3, 5, 3]

    samp = "BCB"

    if samp == "BCB":
        delfstar = {
            1: -28206.4782657343 + 1j * 5.6326137881,
            3: -87768.0313369799 + 1j * 155.716064797,
            5: -159742.686586637 + 1j * 888.6642467156,
        }
        film = {
            0: {"calc": False, "drho": 2.8e-06, "grho": 3e17, "phi": 0, "n": 3},
            1: {"calc": True},
        }
    elif samp == "water":
        delfstar = {
            1: -694.15609764494 + 1j * 762.8726222543,
            3: -1248.7983004897833 + 1j * 1215.1121711257,
            5: -1641.2310467399657 + 1j * 1574.7706516819,
        }
        film = {
            0: {"calc": False, "drho": 2.8e-06, "grho": 3e17, "phi": 0, "n": 3},
            1: {"calc": True},
        }

    grho_refh, phi, drho, dlam_refh, err = qcm.solve_general_delfstar_to_prop(
        nh, delfstar, film, calctype="SLA", bulklimit=0.5
    )

    logger.info("drho %s", drho)
    logger.info("grho_refh %s", grho_refh)
    logger.info("phi %s", phi)
    logger.info("dlam_refh %s", dlam_refh)
    logger.info("err %s", err)
