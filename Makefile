# Default Python interpreter (override per machine if needed, e.g., make env PYTHON=python)
PYTHON ?= python3.11

.PHONY: env install install-dev test clean

# Set up clean virtual environment
env:
	$(PYTHON) -m venv .venv
	@echo "Virtual environment created with $$($(PYTHON) -V)"
	@echo "Activate it with 'source .venv/bin/activate' (Linux/Mac) or '.venv\Scripts\activate' (Windows)."

# Setup for Mac (Apple Silicon / MPS)
install-mac:
	pip install --upgrade pip
	pip install -e ".[dev]"

# Setup for Windows / Linux (NVIDIA GPU with CUDA 12.x)
install-cuda:
	pip install --upgrade pip
	pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
	pip install -e ".[dev]"

# Clean temporary files and build artifacts
clean:
	rm -rf build dist *.egg-info .pytest_cache
	find . -type d -name "__pycache__" -exec rm -r {} +