"""
Entry point for the RheoQCM application.

This module provides the main() function used by the console script entry point.
"""

import os
import sys
import runpy


def main():
    """Launch the RheoQCM GUI application."""
    # Get the path to rheoQCM.py
    package_dir = os.path.dirname(os.path.abspath(__file__))
    rheoqcm_script = os.path.join(package_dir, "rheoQCM.py")

    # Add package directory to path for relative imports
    if package_dir not in sys.path:
        sys.path.insert(0, package_dir)

    # Change to package directory for resource loading
    original_cwd = os.getcwd()
    os.chdir(package_dir)

    try:
        # Run rheoQCM.py as __main__
        runpy.run_path(rheoqcm_script, run_name="__main__")
    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":
    main()
