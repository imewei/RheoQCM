#!/usr/bin/env python3
"""
BCB Thin Film Analysis Example

Demonstrates QCM analysis of a BCB (benzocyclobutene) thin film sample
using the modern rheoQCM.core API.
"""

from rheoQCM.core.analysis import analyze_delfstar
from rheoQCM.core.jax_config import configure_jax

configure_jax()

# BCB thin film frequency shift data (harmonics 1, 3, 5)
delfstar = {
    1: -28206.4782657343 + 5.6326137881j,
    3: -87768.0313369799 + 155.716064797j,
    5: -159742.686586637 + 888.6642467156j,
}

# Solve for film properties using harmonics 3, 5
# nh = [n_delf, n_delg, refh]: fit delf at n=3, delg at n=5, ref harmonic = 3
result = analyze_delfstar(delfstar, nh=[3, 5, 3])

print(f"drho     = {result.drho:.3e} kg/m^2")
print(f"grho_refh = {result.grho_refh:.3e} Pa·kg/m^3")
print(f"phi      = {result.phi:.4f} rad")
