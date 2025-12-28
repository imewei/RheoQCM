#!/usr/bin/env python3
"""
BCB Thin Film Analysis Example

Demonstrates QCM analysis of a BCB (benzocyclobutene) thin film sample
using the QCMFuncs legacy interface.

For new scripts, consider using the modern rheoQCM.core API:
    from rheoQCM.core.analysis import QCMAnalyzer
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt

# Suppress deprecation warning for this legacy example
os.environ["QCMFUNCS_SUPPRESS_DEPRECATION"] = "1"

# Add QCMFuncs to path (relative to this script)
# Script is in src/example/, QCMFuncs is in src/QCMFuncs/
script_dir = Path(__file__).parent
src_dir = script_dir.parent  # src/
sys.path.insert(0, str(src_dir / "QCMFuncs"))

import QCM_functions as qcm

plt.close("all")

# Read the data
# data is in src/data/
data_path = src_dir / "data" / "samples" / "bcb_4.xlsx"
df = qcm.read_xlsx(str(data_path))

# pick a calculation
# we fit to the frequency shifts for the values before the colon
# we fit to the dissipation shifts for the values after the colon
calc = "3.5_5"

# solve for the properties
layers = {1: {"grho3": 1e12, "phi": 1, "drho": 5e-3}}
soln = qcm.solve_for_props(df, calc, ["grho3", "phi", "drho"], layers)

# now make the property axes and plot the property values on it
figinfo = qcm.make_prop_axes(["grho3.linear", "phi", "drho"], xunit="index")
qcm.plot_props(soln, figinfo, fmt="+-", num="BCB properties", nplot=[3, 5, 7])
