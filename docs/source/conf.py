"""Sphinx configuration for RheoQCM documentation."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("QCMFUNCS_SUPPRESS_DEPRECATION", "1")
os.environ.setdefault("RHEOQCM_QUIET", "1")

project = "RheoQCM"
author = "Wei Chen"
copyright = "2025, Wei Chen"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinxcontrib.bibtex",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

html_theme = "alabaster"
html_static_path = ["_static"]

autodoc_default_options = {
    "members": False,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autosummary_generate = True

napoleon_google_docstring = True
napoleon_numpy_docstring = True

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "linkify",
    "substitution",
    "tasklist",
]
myst_heading_anchors = 2

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

autodoc_mock_imports = [
    "PyQt6",
    "numpyro",
    "arviz",
    "pymittagleffler",
    "sympy",
    "nidaqmx",
    "hdf5storage",
    "openpyxl",
    "xlsxwriter",
    "xlrd",
]

autosummary_mock_imports = autodoc_mock_imports

bibtex_bibfiles = [
    "references/QCM.bib",
]
