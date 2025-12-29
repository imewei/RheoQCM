[![DOI](https://zenodo.org/badge/138771761.svg)](https://zenodo.org/badge/latestdoi/138771761)

# QCM Data Collection and Analysis Software

This is the Python project page for the QCM data collection and analysis software used by the Shull research group at Northwestern University. The data collection and analysis are at the testing point. Curently, it is using its own data format (hdf5). The data importing function with QCM-D data is useful to the community doing QCM-D tests. Some of the analysis routines are generally useful, regardless of how the QCM data were generated.

![](Screenshot.png)
<p align = "center">
<b>Screenshot of the User Interface</b>
</p>

## Getting Started

The analysis portions of the software should work on Windows/Mac/Linux platforms. In all cases you'll need some familiarity with running commands from the terminal, however. It's assumed in the following that you know how to do this on your platform. The software to interface with network analyzers and collect the QCM data only runs on Windows-based computers (The analyser, [N2PK Vector Network Analyzer](https://www.makarov.ca/vna.htm), currently interfaced with only works on Windows)

### Capabilities

* Graphical data interface to collect QCM data with network analyzers.
* Fast data recording without openning the dependent external software. Fewer resources are required than in previous MATLAB-based versions of the software.
* Data collection and analysis are combined in one package.
* Other variables (Temperature, for example) can be simultaneously recorded and saved with the QCM data. (with NI devices)

### Prerequisites

* Python 3.12+ is required. For data analysis only, it can run with both 32-bit and 64-bit Python. If you want to use the data collection portion with myVNA, 32-bit Python and Windows are required.

### GPU Acceleration (Linux + System CUDA)

**Performance Impact:** 20-100x speedup for large datasets (>1M points)

**Prerequisites:**
- NVIDIA GPU with SM >= 5.2 (Maxwell or newer)
- System CUDA 12.x or 13.x installed
- `nvcc` in PATH

#### Verify Prerequisites

```bash
# Check CUDA installation
nvcc --version
# Should show: Cuda compilation tools, release 12.x or 13.x

# Check GPU
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
# Should show: GPU name and SM version (e.g., "8.9" for RTX 4090)
```

#### Option 1: Quick Install via Makefile (Recommended)

```bash
git clone https://github.com/imewei/RheoQCM.git
cd RheoQCM

# Auto-detect system CUDA version and install matching JAX
make install-jax-gpu

# Or explicitly choose CUDA version:
make install-jax-gpu-cuda13  # Requires system CUDA 13.x + SM >= 7.5
make install-jax-gpu-cuda12  # Requires system CUDA 12.x + SM >= 5.2
```

This:
- Detects your system CUDA version (via nvcc)
- Validates GPU compatibility
- Installs the matching `jax[cudaXX-local]` package
- Verifies GPU detection

#### Option 2: Manual Installation

**For System CUDA 13.x (Turing and newer GPUs):**

```bash
# Verify you have CUDA 13.x
nvcc --version  # Should show release 13.x

# Verify GPU supports CUDA 13 (SM >= 7.5)
nvidia-smi --query-gpu=compute_cap --format=csv,noheader  # Should be >= 7.5

# Install
pip uninstall -y jax jaxlib
pip install "jax[cuda13-local]"

# Verify
python -c "import jax; print('Backend:', jax.default_backend())"
# Should show: Backend: gpu
```

**For System CUDA 12.x (Maxwell and newer GPUs):**

```bash
# Verify you have CUDA 12.x
nvcc --version  # Should show release 12.x

# Install
pip uninstall -y jax jaxlib
pip install "jax[cuda12-local]"

# Verify
python -c "import jax; print('Backend:', jax.default_backend())"
```

#### GPU Compatibility Guide

| GPU Generation | Example GPUs | SM Version | CUDA 13 | CUDA 12 |
|----------------|--------------|------------|---------|---------|
| Blackwell | B100, B200 | 10.0 | Yes | Yes |
| Hopper | H100, H200 | 9.0 | Yes | Yes |
| Ada Lovelace | RTX 40xx, L40 | 8.9 | Yes | Yes |
| Ampere | RTX 30xx, A100 | 8.x | Yes | Yes |
| Turing | RTX 20xx, T4 | 7.5 | Yes | Yes |
| Volta | V100, Titan V | 7.0 | No | Yes |
| Pascal | GTX 10xx, P100 | 6.x | No | Yes |
| Maxwell | GTX 9xx, Titan X | 5.x | No | Yes |
| Kepler | GTX 7xx, K80 | 3.x | No | No |

**Recommendation based on your GPU:**
- SM >= 7.5 (RTX 20xx or newer): Install CUDA 13 for best performance
- SM 5.2-7.4 (GTX 9xx/10xx, V100): Install CUDA 12

#### GPU Troubleshooting

**Issue:** "nvcc not found"

**Solution:** CUDA toolkit not installed or not in PATH

```bash
# Option 1: Install CUDA toolkit
# Ubuntu/Debian:
sudo apt install nvidia-cuda-toolkit

# Option 2: Add existing CUDA to PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Add to ~/.bashrc for permanent fix
```

**Issue:** "CUDA version mismatch"

**Solution:** JAX package must match your system CUDA version

```bash
# Check your system CUDA version
nvcc --version
# Shows: release 12.6 -> use cuda12-local
# Shows: release 13.x -> use cuda13-local

# Reinstall with correct package
pip uninstall -y jax jaxlib
pip install "jax[cuda12-local]"  # or cuda13-local
```

**Issue:** "GPU SM version doesn't support CUDA 13"

**Solution:** Your GPU is older than Turing architecture

```bash
# Check SM version
nvidia-smi --query-gpu=compute_cap --format=csv,noheader
# If < 7.5, you need CUDA 12

# Install CUDA 12.x toolkit, then:
pip install "jax[cuda12-local]"
```

**Issue:** "libcuda.so not found" or similar library errors

**Solution:** CUDA libraries not in LD_LIBRARY_PATH

```bash
# Add to ~/.bashrc
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Then restart terminal or:
source ~/.bashrc
```

#### Platform Support Summary

| Platform | GPU Support | Notes |
|----------|-------------|-------|
| Linux x86_64/aarch64 | Full | Requires system CUDA 12.x or 13.x |
| Windows WSL2 | Experimental | Use Linux wheels |
| macOS (any) | CPU-only | No NVIDIA support |
| Windows native | CPU-only | No pre-built wheels |

#### Conda/Mamba Users

RheoQCM works seamlessly in conda environments using pip:

```bash
conda create -n rheoqcm python=3.12
conda activate rheoqcm
pip install rheoQCM

# For GPU acceleration (Linux only)
git clone https://github.com/imewei/RheoQCM.git
cd RheoQCM
make install-jax-gpu  # Auto-detects optimal CUDA version
```

**Note:** Conda extras syntax is not supported. Use the Makefile or manual pip installation method above.

### Fallback Behavior

The Kotula model supports multiple backends with automatic fallback:

1. **JAX (default)** - High-performance vectorized computation (100x+ faster)
2. **mpmath (fallback)** - Used when JAX is unavailable

Install mpmath as a fallback option:
```bash
pip install mpmath
# Or install with the fallback optional dependency:
pip install "rheoQCM[fallback]"
```

If neither JAX nor mpmath is available, an informative ImportError is raised with installation instructions.

* If you want to run data collection, hardware and external software for data collection: The AccessMyVNA and myVNA programs were obtained from <http://g8kbb.co.uk/html/downloads.html>.

* The Anaconda python environment is suggested.  You can  download and install the Anaconda distribution of python from [anaconda.com](https://anaconda.com/download).

* Separated scripts works with data stored in a MATLAB-compatible .mat files (collected by our [Matlab data collecting program](https://github.com/Shull-Research-Group/QCM_Data_Acquisition_Program)).  In order to read and write these and get the analysis scripts to work, you need to install the hdf5storage package, which you can add with the following command (assuming you have already added the conda python distribution):

```bash
conda install -c conda-forge hdf5storage
```

## Using Python code

### Installation

To install everything you need from this repository, run the following command from a command window in the directory where you want everthing to be installed:

```bash
git clone https://github.com/zhczq/QCM_py
```

If you just need the updated analysis script, everything you need is in QCMFuncs/QCM_functions.py. In this same directory you will also find some example data obtained with polystyrene films at different temperatures, and a run file called PS_plots.py. You should be able to run PS_plots.py directly and end up with some property plots that illustrate how the process works, and how you need to set up the file structure to include your own data.

All the modules needed for the data collection program are in the `rheoQCM/` folder. Go to that folder and run rheoQCM.py will open the program.

### Running the UI from Terminal

Go to the `rheoQCM/` folder and run `rheoQCM.py` or `rheoQCM` will start the UI and it will check the environment by itself.

## Using UI with QCM-D Data

* Export the QCM-D data as .xlsx file. column names: t(s), delf1, delg1, delf3, delg3, ... The time column name could also be time(s).
* Start the UI and from the menu bar and select `File>Import QCM-D data`.  This will import the QCM-D data and save a .h5 file with the same name in * the same folder. This will save all the calculated property data for future use.
* Select the corresponding fundamental frequency at `Settings>Hardwares>Crystal>Base Frequency` before import the data.
* Now the UI can display your data and do the analysis the same as the data generated with the UI.
* Don't forget to save the data when you finish the calculation.
* Click export to export a .xlsx file with all the data in it.

## Using the Analysis Code for .mat Files

If you just need the updated analysis code for .mat files, everything you really need is in `QCMFuncs/QCM_functions.py`. In order to read and write these and get the analysis scripts to work, you need to install the hdf5storage package, which you can add with the following command (assuming you have already added the conda python distribution):

```bash
conda install -c conda-forge hdf5storage
```

In this same directory you will also find some example data obtained with polystyrene films at different temperatures, and a run file called PS_plots.py. You should be able to run PS_plots.py directly and end up with some property plots that illustrate how the process works, and how you need to set up the file structure to include your own data.

## Using the Analysis Code for .h5 Files

The `QCM_functions.py` code also works with .h5 data files collected by the UI of this project. The file definitions are similar to those of the .mat files. Example files (`example_plot.py` and `example_sampledefs.py`) which demostrate both .mat and .h5 analysis with `QCM_functions.py` can be found in  `QCMFuncs/`.

## Documentation

The QCMnotes.pdf file has some background information on the rheometric mode of the QCM that we utilize, with some useful references included.

Modules `DataSaver` and `QCM` in `Modules/` folder are availabe for dealing with the data and doing ananlysis manually. Those modules include functions run out of the software. You can simply import those modules and do almost the same thing in the software by running your own codes. An example code of extracting data from data file can be found in hte `example/` folder.

The functions for the Matlab version of the data are locoalized in `QCMFuncs/` folder.

Export the current settings as a json file named `settings_default.json` and save in the same folder as `rheoQCM.py` or `rheoQCM.exe`. The UI will use the settings you saved as default after the next time you opend it. If you want the setup go back the original one, just simply delete or rename that file.

There is a `rheoQCM.bat` file in  `rheoQCM/` for running the program with Python by just double clicking it. You need to change the path of python and rheoQCM.py to them on your computer to make it work. Meanwhile, you can make a shortcut of this bat file and put the shortcut in a location of your choosing.
There is a `rheoQCM.bat` file in  `rheoQCM/` for running the program with Python by just double clicking it on Windows. You need to change the path of python and rheoQCM.py to them on your computer to make it work. Meanwhile, you can make a shortcut of this bat file and put the shortcut somewhere you want.

### Known Issues

* Please set MyVNA to `Basic Mode` from the left pannel of MyVNA software by selecting VNA Hardware>Configure CSD / Harmonic Mode and checking Basic Mode in Select Mode. This will make sure the time counting in the Python program fits the hardware scanning time. You will not loose any precision as far as we know.

## To Do List (work in Progress)

* Documentation.
* Property results plotting and exporting.
* Interface with other hardware. (If you have a hardware and interested in interfacing with our software, please feel free to contact us.)

## Authors

* **Qifeng Wang**  - *Primary Developer of the current (Python) version of this project*
* **Megan Yang**  - *Developer of the current (python) version of this project*
* **Kenneth R. Shull** - *Project PI and author of some of the specific functions used in the project*

## Other Versions

If you are a MATLAB user, our previously developed MATLAB version software can be found here: <https://github.com/Shull-Research-Group/QCM_Data_Acquisition_Program>. It was developed Josh Yeh. This Python project is based on this previous MATLAB version developed by Josh.

A MATLAB version of our data analysis software, written by Kazi Sadman can be found here: <https://github.com/sadmankazi/QCM-D-Analysis-GUI>.

## Citation information

Please check how to cite the repository from here: <https://zenodo.org/record/2486039#.XlVoBjJKjcs>

## Acknowledgments

* Josh Yeh
* Diethelm Johannsmann
* Lauren Sturdy
* Ivan Makarov
