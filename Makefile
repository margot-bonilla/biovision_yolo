# Default Python interpreter (override per machine if needed, e.g., make env PYTHON=python)
PYTHON ?= python3.11
NPZ_FILE ?= data/chromsome_data.npz
DATA_DIR ?= data/

.PHONY: env install-mac install-cuda clean extract-data

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
	
# --- Data Preparation Targets ---
extract-data:
	@echo "Extracting data from $(NPZ_FILE) to $(DATA_DIR)..."
	$(PYTHON) chromoseg/data/extract_npz.py --npz_path $(NPZ_FILE) --output_dir $(DATA_DIR)

parse-data:
	@echo "Converting masks to YOLO polygon labels..."
	mkdir -p $(DATA_DIR)processed/labels
	$(PYTHON) chromoseg/data/parser.py \
		--images_dir $(DATA_DIR)raw/images \
		--masks_dir $(DATA_DIR)raw/masks \
		--output_dir $(DATA_DIR)processed/labels