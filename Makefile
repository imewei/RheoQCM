# RheoQCM Package Makefile
# ========================
# Development Tools for QCM Analysis Software

.PHONY: help install install-dev install-gui \
        install-jax-gpu install-jax-gpu-cuda13 install-jax-gpu-cuda12 gpu-check env-info \
        test test-fast test-coverage \
        clean clean-all clean-pyc clean-build clean-test clean-venv \
        format lint type-check check quick docs build info version

# Configuration
PYTHON := python
PYTEST := pytest
PACKAGE_NAME := rheoQCM
SRC_DIRS := src/rheoQCM src/QCMFuncs
TEST_DIR := tests
DOCS_DIR := docs
VENV := .venv

# Platform detection
UNAME_S := $(shell uname -s 2>/dev/null || echo "Windows")
ifeq ($(UNAME_S),Linux)
    PLATFORM := linux
else ifeq ($(UNAME_S),Darwin)
    PLATFORM := macos
else
    PLATFORM := windows
endif

# Package manager detection (prioritize uv > conda/mamba > pip)
UV_AVAILABLE := $(shell command -v uv 2>/dev/null)
CONDA_PREFIX := $(shell echo $$CONDA_PREFIX)
MAMBA_AVAILABLE := $(shell command -v mamba 2>/dev/null)

# Determine package manager and commands
ifdef UV_AVAILABLE
    PKG_MANAGER := uv
    PIP := uv pip
    UNINSTALL_CMD := uv pip uninstall -y
    INSTALL_CMD := uv pip install
    RUN_CMD := uv run
else ifdef CONDA_PREFIX
    ifdef MAMBA_AVAILABLE
        PKG_MANAGER := mamba (using pip)
    else
        PKG_MANAGER := conda (using pip)
    endif
    PIP := pip
    UNINSTALL_CMD := pip uninstall -y
    INSTALL_CMD := pip install
    RUN_CMD :=
else
    PKG_MANAGER := pip
    PIP := pip
    UNINSTALL_CMD := pip uninstall -y
    INSTALL_CMD := pip install
    RUN_CMD :=
endif

# GPU installation packages (system CUDA - uses -local suffix)
ifeq ($(PLATFORM),linux)
    JAX_GPU_CUDA13_PKG := "jax[cuda13-local]"
    JAX_GPU_CUDA12_PKG := "jax[cuda12-local]"
else
    JAX_GPU_CUDA13_PKG :=
    JAX_GPU_CUDA12_PKG :=
endif

# Colors for output
BOLD := \033[1m
RESET := \033[0m
BLUE := \033[34m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
CYAN := \033[36m

# Default target
.DEFAULT_GOAL := help

# ===================
# Help target
# ===================
help:
	@echo "$(BOLD)$(BLUE)RheoQCM Development Commands$(RESET)"
	@echo ""
	@echo "$(BOLD)Usage:$(RESET) make $(CYAN)<target>$(RESET)"
	@echo ""
	@echo "$(BOLD)$(GREEN)ENVIRONMENT$(RESET)"
	@echo "  $(CYAN)env-info$(RESET)         Show detailed environment information"
	@echo "  $(CYAN)info$(RESET)             Show project and environment info"
	@echo "  $(CYAN)version$(RESET)          Show package version"
	@echo ""
	@echo "$(BOLD)$(GREEN)INSTALLATION$(RESET)"
	@echo "  $(CYAN)install$(RESET)          Install package in editable mode"
	@echo "  $(CYAN)install-dev$(RESET)      Install with development dependencies"
	@echo "  $(CYAN)install-gui$(RESET)      Install with GUI dependencies (PyQt6)"
	@echo ""
	@echo "$(BOLD)$(GREEN)GPU ACCELERATION (System CUDA)$(RESET)"
	@echo "  $(CYAN)install-jax-gpu$(RESET)        Auto-detect system CUDA and install JAX (Linux only)"
	@echo "  $(CYAN)install-jax-gpu-cuda13$(RESET) Install JAX with system CUDA 13 (requires CUDA 13.x)"
	@echo "  $(CYAN)install-jax-gpu-cuda12$(RESET) Install JAX with system CUDA 12 (requires CUDA 12.x)"
	@echo "  $(CYAN)gpu-check$(RESET)              Check GPU detection and JAX backend status"
	@echo ""
	@echo "$(BOLD)Prerequisites:$(RESET)"
	@echo "  - System CUDA 12.x or 13.x installed (nvcc in PATH)"
	@echo "  - NVIDIA GPU with SM >= 5.2 (Maxwell or newer)"
	@echo ""
	@echo "$(BOLD)GPU Compatibility:$(RESET)"
	@echo "  CUDA 13 (SM 7.5+): RTX 20xx/30xx/40xx, T4, A100, H100, L40"
	@echo "  CUDA 12 (SM 5.2+): GTX 9xx/10xx, P100, V100, + all CUDA 13 GPUs"
	@echo ""
	@echo "$(BOLD)$(GREEN)TESTING$(RESET)"
	@echo "  $(CYAN)test$(RESET)             Run all tests"
	@echo "  $(CYAN)test-fast$(RESET)        Run tests excluding slow tests"
	@echo "  $(CYAN)test-coverage$(RESET)    Run tests with coverage report"
	@echo ""
	@echo "$(BOLD)$(GREEN)CODE QUALITY$(RESET)"
	@echo "  $(CYAN)format$(RESET)           Format code with black and ruff"
	@echo "  $(CYAN)lint$(RESET)             Run linting checks (ruff)"
	@echo "  $(CYAN)type-check$(RESET)       Run type checking (mypy)"
	@echo "  $(CYAN)check$(RESET)            Run all checks (format + lint + type)"
	@echo "  $(CYAN)quick$(RESET)            Fast iteration: format + quick tests"
	@echo ""
	@echo "$(BOLD)$(GREEN)DOCUMENTATION$(RESET)"
	@echo "  $(CYAN)docs$(RESET)             Build documentation"
	@echo ""
	@echo "$(BOLD)$(GREEN)BUILD$(RESET)"
	@echo "  $(CYAN)build$(RESET)            Build distribution packages"
	@echo ""
	@echo "$(BOLD)$(GREEN)CLEANUP$(RESET)"
	@echo "  $(CYAN)clean$(RESET)            Remove build artifacts and caches"
	@echo "  $(CYAN)clean-all$(RESET)        Deep clean of all caches"
	@echo "  $(CYAN)clean-pyc$(RESET)        Remove Python file artifacts"
	@echo "  $(CYAN)clean-build$(RESET)      Remove build artifacts"
	@echo "  $(CYAN)clean-test$(RESET)       Remove test and coverage artifacts"
	@echo "  $(CYAN)clean-venv$(RESET)       Remove virtual environment (use with caution)"
	@echo ""
	@echo "$(BOLD)Environment Detection:$(RESET)"
	@echo "  Platform: $(PLATFORM)"
	@echo "  Package manager: $(PKG_MANAGER)"
	@echo ""

# ===================
# Installation targets
# ===================
install:
	@echo "$(BOLD)$(BLUE)Installing $(PACKAGE_NAME) in editable mode...$(RESET)"
	@$(INSTALL_CMD) -e .
	@echo "$(BOLD)$(GREEN)Done: Package installed!$(RESET)"

install-dev: install
	@echo "$(BOLD)$(BLUE)Installing development dependencies...$(RESET)"
	@$(INSTALL_CMD) -e ".[dev]"
	@echo "$(BOLD)$(GREEN)Done: Dev dependencies installed!$(RESET)"

install-gui:
	@echo "$(BOLD)$(BLUE)Installing GUI dependencies...$(RESET)"
	@$(INSTALL_CMD) -e ".[gui]"
	@echo "$(BOLD)$(GREEN)Done: GUI dependencies installed!$(RESET)"

# Auto-detect system CUDA version and install matching JAX package
install-jax-gpu:
	@echo "$(BOLD)$(BLUE)Installing JAX with GPU support (system CUDA auto-detect)...$(RESET)"
	@echo "============================================================"
	@echo "Platform: $(PLATFORM)"
	@echo "Package manager: $(PKG_MANAGER)"
	@echo ""
ifeq ($(PLATFORM),linux)
	@# Step 1: Detect system CUDA version
	@CUDA_VERSION=$$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+' | head -1); \
	if [ -z "$$CUDA_VERSION" ]; then \
		echo "$(RED)Error: nvcc not found - CUDA toolkit not installed or not in PATH$(RESET)"; \
		echo ""; \
		echo "Please install CUDA toolkit:"; \
		echo "  Ubuntu/Debian: sudo apt install nvidia-cuda-toolkit"; \
		echo "  Or download from: https://developer.nvidia.com/cuda-downloads"; \
		echo ""; \
		echo "After installation, ensure nvcc is in PATH:"; \
		echo "  export PATH=/usr/local/cuda/bin:\$$PATH"; \
		exit 1; \
	fi; \
	CUDA_FULL=$$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+'); \
	echo "Detected system CUDA version: $$CUDA_FULL (major: $$CUDA_VERSION)"; \
	echo ""; \
	\
	# Step 2: Detect GPU SM version \
	SM_VERSION=$$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.'); \
	SM_DISPLAY=$$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1); \
	GPU_NAME=$$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1); \
	if [ -z "$$SM_VERSION" ]; then \
		echo "$(RED)Error: Could not detect GPU (nvidia-smi failed)$(RESET)"; \
		exit 1; \
	fi; \
	echo "Detected GPU: $$GPU_NAME (SM $$SM_DISPLAY)"; \
	echo ""; \
	\
	# Step 3: Validate compatibility and install \
	if [ "$$CUDA_VERSION" = "13" ]; then \
		if [ "$$SM_VERSION" -ge 75 ]; then \
			echo "Compatibility: System CUDA 13 + GPU SM $$SM_DISPLAY = $(GREEN)Compatible$(RESET)"; \
			echo "Installing: $(JAX_GPU_CUDA13_PKG)"; \
			$(MAKE) install-jax-gpu-cuda13; \
		else \
			echo "$(RED)Error: GPU SM $$SM_DISPLAY does not support CUDA 13 (requires SM >= 7.5)$(RESET)"; \
			echo "Your GPU requires CUDA 12. Please install CUDA 12.x toolkit."; \
			exit 1; \
		fi; \
	elif [ "$$CUDA_VERSION" = "12" ]; then \
		if [ "$$SM_VERSION" -ge 52 ]; then \
			echo "Compatibility: System CUDA 12 + GPU SM $$SM_DISPLAY = $(GREEN)Compatible$(RESET)"; \
			echo "Installing: $(JAX_GPU_CUDA12_PKG)"; \
			$(MAKE) install-jax-gpu-cuda12; \
		else \
			echo "$(RED)Error: GPU SM $$SM_DISPLAY too old (requires SM >= 5.2)$(RESET)"; \
			echo "Kepler and older GPUs are not supported by JAX 0.8+"; \
			exit 1; \
		fi; \
	else \
		echo "$(RED)Error: CUDA $$CUDA_VERSION not supported by JAX 0.8+$(RESET)"; \
		echo "JAX requires CUDA 12.x or 13.x"; \
		echo "Please upgrade your CUDA installation."; \
		exit 1; \
	fi
else
	@echo "$(YELLOW)Error: GPU acceleration only available on Linux$(RESET)"
	@echo "  Current platform: $(PLATFORM)"
	@echo "  Keeping CPU-only installation"
	@echo ""
	@echo "$(BOLD)Platform support:$(RESET)"
	@echo "  $(GREEN)✓$(RESET) Linux + NVIDIA GPU + System CUDA: Full GPU acceleration"
	@echo "  $(YELLOW)⚠$(RESET) Windows WSL2: Experimental (use Linux wheels)"
	@echo "  $(RED)✗$(RESET) macOS: CPU-only (no NVIDIA GPU support)"
	@echo "  $(RED)✗$(RESET) Windows native: CPU-only (no pre-built wheels)"
endif

# CUDA 13 installation (requires system CUDA 13.x)
install-jax-gpu-cuda13:
	@echo "$(BOLD)$(BLUE)Installing JAX with system CUDA 13...$(RESET)"
	@echo "======================================"
	@echo "Platform: $(PLATFORM)"
	@echo "Package manager: $(PKG_MANAGER)"
	@echo ""
ifeq ($(PLATFORM),linux)
	@# Validate system CUDA 13.x is installed
	@CUDA_VERSION=$$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+' | head -1); \
	CUDA_FULL=$$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+'); \
	if [ -z "$$CUDA_VERSION" ]; then \
		echo "$(RED)Error: nvcc not found - CUDA toolkit not installed$(RESET)"; \
		exit 1; \
	elif [ "$$CUDA_VERSION" != "13" ]; then \
		echo "$(RED)Error: System CUDA $$CUDA_FULL detected, but CUDA 13.x required$(RESET)"; \
		echo "Either:"; \
		echo "  1. Install CUDA 13.x toolkit"; \
		echo "  2. Use: make install-jax-gpu-cuda12 (if you have CUDA 12.x)"; \
		exit 1; \
	fi; \
	echo "$(GREEN)✓$(RESET) System CUDA: $$CUDA_FULL"
	@# Validate GPU supports CUDA 13 (SM >= 7.5)
	@SM_VERSION=$$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.'); \
	SM_DISPLAY=$$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1); \
	if [ -z "$$SM_VERSION" ]; then \
		echo "$(RED)Error: Could not detect GPU$(RESET)"; \
		exit 1; \
	elif [ "$$SM_VERSION" -lt 75 ]; then \
		echo "$(RED)Error: GPU SM $$SM_DISPLAY does not support CUDA 13$(RESET)"; \
		echo "CUDA 13 requires SM >= 7.5 (Turing or newer)"; \
		echo "Your GPU requires: make install-jax-gpu-cuda12"; \
		exit 1; \
	fi; \
	echo "$(GREEN)✓$(RESET) GPU SM version: $$SM_DISPLAY (compatible with CUDA 13)"
	@echo ""
	@echo "$(BOLD)Step 1/2:$(RESET) Uninstalling CPU-only JAX..."
	@$(UNINSTALL_CMD) jax jaxlib 2>/dev/null || true
	@echo ""
	@echo "$(BOLD)Step 2/2:$(RESET) Installing JAX with system CUDA 13..."
	@echo "Command: $(INSTALL_CMD) $(JAX_GPU_CUDA13_PKG)"
	@$(INSTALL_CMD) $(JAX_GPU_CUDA13_PKG)
	@echo ""
	@$(MAKE) gpu-check
	@echo ""
	@echo "$(BOLD)$(GREEN)✓ JAX GPU support installed successfully$(RESET)"
	@echo "  Package: $(JAX_GPU_CUDA13_PKG)"
	@echo "  Uses: System CUDA 13.x installation"
else
	@echo "$(RED)Error: CUDA 13 GPU acceleration only available on Linux$(RESET)"
endif

# CUDA 12 installation (requires system CUDA 12.x)
install-jax-gpu-cuda12:
	@echo "$(BOLD)$(BLUE)Installing JAX with system CUDA 12...$(RESET)"
	@echo "======================================"
	@echo "Platform: $(PLATFORM)"
	@echo "Package manager: $(PKG_MANAGER)"
	@echo ""
ifeq ($(PLATFORM),linux)
	@# Validate system CUDA 12.x is installed
	@CUDA_VERSION=$$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+' | head -1); \
	CUDA_FULL=$$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+'); \
	if [ -z "$$CUDA_VERSION" ]; then \
		echo "$(RED)Error: nvcc not found - CUDA toolkit not installed$(RESET)"; \
		exit 1; \
	elif [ "$$CUDA_VERSION" != "12" ]; then \
		echo "$(RED)Error: System CUDA $$CUDA_FULL detected, but CUDA 12.x required$(RESET)"; \
		echo "Either:"; \
		echo "  1. Install CUDA 12.x toolkit"; \
		echo "  2. Use: make install-jax-gpu-cuda13 (if you have CUDA 13.x and SM >= 7.5)"; \
		exit 1; \
	fi; \
	echo "$(GREEN)✓$(RESET) System CUDA: $$CUDA_FULL"
	@# Validate GPU supports CUDA 12 (SM >= 5.2)
	@SM_VERSION=$$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.'); \
	SM_DISPLAY=$$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1); \
	if [ -z "$$SM_VERSION" ]; then \
		echo "$(RED)Error: Could not detect GPU$(RESET)"; \
		exit 1; \
	elif [ "$$SM_VERSION" -lt 52 ]; then \
		echo "$(RED)Error: GPU SM $$SM_DISPLAY too old$(RESET)"; \
		echo "CUDA 12 requires SM >= 5.2 (Maxwell or newer)"; \
		echo "Kepler and older GPUs are not supported."; \
		exit 1; \
	fi; \
	echo "$(GREEN)✓$(RESET) GPU SM version: $$SM_DISPLAY (compatible with CUDA 12)"
	@echo ""
	@echo "$(BOLD)Step 1/2:$(RESET) Uninstalling CPU-only JAX..."
	@$(UNINSTALL_CMD) jax jaxlib 2>/dev/null || true
	@echo ""
	@echo "$(BOLD)Step 2/2:$(RESET) Installing JAX with system CUDA 12..."
	@echo "Command: $(INSTALL_CMD) $(JAX_GPU_CUDA12_PKG)"
	@$(INSTALL_CMD) $(JAX_GPU_CUDA12_PKG)
	@echo ""
	@$(MAKE) gpu-check
	@echo ""
	@echo "$(BOLD)$(GREEN)✓ JAX GPU support installed successfully$(RESET)"
	@echo "  Package: $(JAX_GPU_CUDA12_PKG)"
	@echo "  Uses: System CUDA 12.x installation"
else
	@echo "$(RED)Error: CUDA 12 GPU acceleration only available on Linux$(RESET)"
endif

gpu-check:
	@echo "$(BOLD)$(BLUE)Checking JAX GPU Configuration$(RESET)"
	@echo "================================"
	@echo ""
	@echo "$(BOLD)NVIDIA GPU Status:$(RESET)"
	@nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || echo "  nvidia-smi not available or no GPU found"
	@echo ""
	@echo "$(BOLD)JAX Device Detection:$(RESET)"
	@$(PYTHON) -c "import jax; devices = jax.devices(); print(f'  Available devices: {len(devices)}'); [print(f'    - {d}') for d in devices]" 2>&1 || echo "  Failed to import JAX"
	@echo ""
	@echo "$(BOLD)JAX Backend:$(RESET)"
	@$(PYTHON) -c "import jax; print(f'  Default backend: {jax.default_backend()}')" 2>&1 || echo "  Failed to check backend"
	@echo ""

# Environment info target
env-info:
	@echo "$(BOLD)$(BLUE)Environment Information$(RESET)"
	@echo "======================"
	@echo ""
	@echo "$(BOLD)Platform Detection:$(RESET)"
	@echo "  OS: $(UNAME_S)"
	@echo "  Platform: $(PLATFORM)"
	@echo ""
	@echo "$(BOLD)Python Environment:$(RESET)"
	@echo "  Python: $(shell $(PYTHON) --version 2>&1 || echo 'not found')"
	@echo "  Python path: $(shell which $(PYTHON) 2>/dev/null || echo 'not found')"
	@echo ""
	@echo "$(BOLD)Package Manager Detection:$(RESET)"
	@echo "  Active manager: $(PKG_MANAGER)"
ifdef UV_AVAILABLE
	@echo "  $(GREEN)✓$(RESET) uv detected: $(UV_AVAILABLE)"
	@echo "    Install command: $(INSTALL_CMD)"
	@echo "    Uninstall command: $(UNINSTALL_CMD)"
else
	@echo "  $(YELLOW)✗$(RESET) uv not found"
endif
ifdef CONDA_PREFIX
	@echo "  $(GREEN)✓$(RESET) Conda environment detected"
	@echo "    CONDA_PREFIX: $(CONDA_PREFIX)"
ifdef MAMBA_AVAILABLE
	@echo "    Mamba available: $(MAMBA_AVAILABLE)"
else
	@echo "    Mamba: not found"
endif
	@echo "    Note: Using pip within conda for JAX installation"
else
	@echo "  $(YELLOW)✗$(RESET) Not in conda environment"
endif
	@echo "  pip: $(shell which pip 2>/dev/null || echo 'not found')"
	@echo ""
	@echo "$(BOLD)GPU Support:$(RESET)"
ifeq ($(PLATFORM),linux)
	@echo "  Platform: $(GREEN)✓$(RESET) Linux (GPU support available)"
	@echo ""
	@echo "  System CUDA:"
	@CUDA_FULL=$$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+'); \
	CUDA_MAJOR=$$(echo $$CUDA_FULL | cut -d'.' -f1); \
	if [ -n "$$CUDA_FULL" ]; then \
		echo "    Version: $$CUDA_FULL"; \
		echo "    nvcc path: $$(which nvcc)"; \
		if [ "$$CUDA_MAJOR" = "13" ]; then \
			echo "    JAX package: $(JAX_GPU_CUDA13_PKG)"; \
		elif [ "$$CUDA_MAJOR" = "12" ]; then \
			echo "    JAX package: $(JAX_GPU_CUDA12_PKG)"; \
		else \
			echo "    JAX package: Not supported (need CUDA 12 or 13)"; \
		fi; \
	else \
		echo "    $(RED)✗$(RESET) Not found (nvcc not in PATH)"; \
		echo "    Install CUDA toolkit or add to PATH"; \
	fi
	@echo ""
	@echo "  GPU Hardware:"
	@$(PYTHON) -c "\
import subprocess; \
r = subprocess.run(['nvidia-smi', '--query-gpu=name,compute_cap,driver_version', '--format=csv,noheader'], \
    capture_output=True, text=True, timeout=5); \
if r.returncode == 0: \
    for line in r.stdout.strip().split('\n'): \
        parts = line.split(', '); \
        name, sm, driver = parts[0], parts[1], parts[2]; \
        sm_int = int(sm.replace('.', '')); \
        cuda_support = 'CUDA 12 and 13' if sm_int >= 75 else 'CUDA 12 only' if sm_int >= 52 else 'Not supported'; \
        print(f'    Name: {name}'); \
        print(f'    SM version: {sm} ({cuda_support})'); \
        print(f'    Driver: {driver}'); \
else: \
    print('    $(RED)✗$(RESET) Not detected (nvidia-smi failed)')" 2>/dev/null || echo "    $(RED)✗$(RESET) nvidia-smi not found"
	@echo ""
	@echo "  JAX Status:"
	@$(PYTHON) -c "\
import jax; \
print(f'    Version: {jax.__version__}'); \
print(f'    Backend: {jax.default_backend()}'); \
devices = jax.devices(); \
gpu_count = sum(1 for d in devices if 'cuda' in str(d).lower()); \
print(f'    GPU devices: {gpu_count}')" 2>/dev/null || echo "    JAX not installed"
else
	@echo "  Platform: $(RED)✗$(RESET) $(PLATFORM) (GPU not supported)"
endif
	@echo ""
	@echo "$(BOLD)GPU Compatibility Reference:$(RESET)"
	@echo "  CUDA 13 (SM 7.5+): Turing, Ampere, Ada Lovelace, Hopper, Blackwell"
	@echo "    - RTX 20xx, RTX 30xx, RTX 40xx, T4, A100, H100, L40, B100"
	@echo "    - Driver >= 580 required"
	@echo ""
	@echo "  CUDA 12 (SM 5.2+): Maxwell, Pascal, Volta, + all CUDA 13 GPUs"
	@echo "    - GTX 9xx, GTX 10xx, P100, V100, Titan X/V"
	@echo "    - Driver >= 525 required"
	@echo ""
	@echo "$(BOLD)Installation Commands:$(RESET)"
	@echo "  Auto-detect:   make install-jax-gpu"
	@echo "  Force CUDA 13: make install-jax-gpu-cuda13"
	@echo "  Force CUDA 12: make install-jax-gpu-cuda12"
	@echo ""

# ===================
# Testing targets
# ===================
test:
	@echo "$(BOLD)$(BLUE)Running all tests...$(RESET)"
	$(RUN_CMD) $(PYTEST)

test-fast:
	@echo "$(BOLD)$(BLUE)Running fast tests (excluding slow tests)...$(RESET)"
	$(RUN_CMD) $(PYTEST) -m "not slow"

test-coverage:
	@echo "$(BOLD)$(BLUE)Running tests with coverage report...$(RESET)"
	$(RUN_CMD) $(PYTEST) --cov=$(PACKAGE_NAME) --cov-report=term-missing --cov-report=html --cov-report=xml
	@echo "$(BOLD)$(GREEN)Done: Coverage report generated!$(RESET)"
	@echo "View HTML report: open htmlcov/index.html"

# ===================
# Code quality targets
# ===================
format:
	@echo "$(BOLD)$(BLUE)Formatting code with black and ruff...$(RESET)"
	$(RUN_CMD) black $(SRC_DIRS) $(TEST_DIR)
	$(RUN_CMD) ruff check --fix $(SRC_DIRS) $(TEST_DIR)
	@echo "$(BOLD)$(GREEN)Done: Code formatted!$(RESET)"

lint:
	@echo "$(BOLD)$(BLUE)Running linting checks...$(RESET)"
	$(RUN_CMD) ruff check $(SRC_DIRS) $(TEST_DIR)
	@echo "$(BOLD)$(GREEN)Done: No linting errors!$(RESET)"

type-check:
	@echo "$(BOLD)$(BLUE)Running type checks...$(RESET)"
	$(RUN_CMD) mypy $(SRC_DIRS)
	@echo "$(BOLD)$(GREEN)Done: Type checking passed!$(RESET)"

check: lint type-check
	@echo "$(BOLD)$(GREEN)Done: All checks passed!$(RESET)"

quick: format test-fast
	@echo "$(BOLD)$(GREEN)Done: Quick iteration complete!$(RESET)"

# ===================
# Documentation targets
# ===================
docs:
	@echo "$(BOLD)$(BLUE)Building documentation...$(RESET)"
	cd docs && $(MAKE) html
	@echo "$(BOLD)$(GREEN)Done: Documentation built!$(RESET)"
	@echo "Open: docs/_build/html/index.html"

# ===================
# Build targets
# ===================
build: clean
	@echo "$(BOLD)$(BLUE)Building distribution packages...$(RESET)"
	$(PYTHON) -m build
	@echo "$(BOLD)$(GREEN)Done: Build complete!$(RESET)"
	@echo "Distributions in dist/"

# ===================
# Cleanup targets
# ===================
clean-build:
	@echo "$(BOLD)$(BLUE)Removing build artifacts...$(RESET)"
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name "*.egg-info" \
		-not -path "./.venv/*" \
		-not -path "./venv/*" \
		-not -path "./.claude/*" \
		-not -path "./agent-os/*" \
		-exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg" \
		-not -path "./.venv/*" \
		-not -path "./venv/*" \
		-not -path "./.claude/*" \
		-not -path "./agent-os/*" \
		-exec rm -rf {} + 2>/dev/null || true

clean-pyc:
	@echo "$(BOLD)$(BLUE)Removing Python file artifacts...$(RESET)"
	find . -type d -name __pycache__ \
		-not -path "./.venv/*" \
		-not -path "./venv/*" \
		-not -path "./.claude/*" \
		-not -path "./agent-os/*" \
		-exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .nlsq_cache \
		-not -path "./.venv/*" \
		-not -path "./venv/*" \
		-not -path "./.claude/*" \
		-not -path "./agent-os/*" \
		-exec rm -rf {} + 2>/dev/null || true
	find . -type f \( -name "*.pyc" -o -name "*.pyo" \) \
		-not -path "./.venv/*" \
		-not -path "./venv/*" \
		-not -path "./.claude/*" \
		-not -path "./agent-os/*" \
		-delete 2>/dev/null || true

clean-test:
	@echo "$(BOLD)$(BLUE)Removing test and coverage artifacts...$(RESET)"
	find . -type d -name .pytest_cache \
		-not -path "./.venv/*" \
		-not -path "./venv/*" \
		-not -path "./.claude/*" \
		-not -path "./agent-os/*" \
		-exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache \
		-not -path "./.venv/*" \
		-not -path "./venv/*" \
		-not -path "./.claude/*" \
		-not -path "./agent-os/*" \
		-exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache \
		-not -path "./.venv/*" \
		-not -path "./venv/*" \
		-not -path "./.claude/*" \
		-not -path "./agent-os/*" \
		-exec rm -rf {} + 2>/dev/null || true
	find . -type d -name htmlcov \
		-not -path "./.venv/*" \
		-not -path "./venv/*" \
		-not -path "./.claude/*" \
		-not -path "./agent-os/*" \
		-exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .hypothesis \
		-not -path "./.venv/*" \
		-not -path "./venv/*" \
		-not -path "./.claude/*" \
		-not -path "./agent-os/*" \
		-exec rm -rf {} + 2>/dev/null || true
	rm -rf .coverage
	rm -rf coverage.xml

clean: clean-build clean-pyc clean-test
	@echo "$(BOLD)$(GREEN)Done: Cleaned!$(RESET)"
	@echo "$(BOLD)Protected directories preserved:$(RESET) .venv/, venv/, .claude/, agent-os/"

clean-all: clean
	@echo "$(BOLD)$(BLUE)Performing deep clean of additional caches...$(RESET)"
	rm -rf .tox/ 2>/dev/null || true
	rm -rf .nox/ 2>/dev/null || true
	rm -rf .eggs/ 2>/dev/null || true
	rm -rf .cache/ 2>/dev/null || true
	@echo "$(BOLD)$(GREEN)Done: Deep clean complete!$(RESET)"
	@echo "$(BOLD)Protected directories preserved:$(RESET) .venv/, venv/, .claude/, agent-os/"

clean-venv:
	@echo "$(BOLD)$(YELLOW)WARNING: This will remove the virtual environment!$(RESET)"
	@echo "$(BOLD)You will need to recreate it manually.$(RESET)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		echo "$(BOLD)$(BLUE)Removing virtual environment...$(RESET)"; \
		rm -rf $(VENV) venv; \
		echo "$(BOLD)$(GREEN)Done: Virtual environment removed!$(RESET)"; \
	else \
		echo "Cancelled."; \
	fi

# ===================
# Utility targets
# ===================
info:
	@echo "$(BOLD)$(BLUE)Project Information$(RESET)"
	@echo "===================="
	@echo "Project: $(PACKAGE_NAME)"
	@echo "Python: $(shell $(PYTHON) --version 2>&1)"
	@echo "Platform: $(PLATFORM)"
	@echo "Package manager: $(PKG_MANAGER)"
	@echo ""
	@echo "$(BOLD)$(BLUE)Directory Structure$(RESET)"
	@echo "===================="
	@echo "Source: $(SRC_DIRS)"
	@echo "Tests: $(TEST_DIR)/"
	@echo "Docs: $(DOCS_DIR)/"

version:
	@$(PYTHON) -c "from $(PACKAGE_NAME) import __version__; print(__version__)" 2>/dev/null || \
		echo "$(BOLD)$(RED)Error: Package not installed. Run 'make install' first.$(RESET)"
