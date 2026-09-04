# Default Python interpreter (override per machine if needed, e.g., make env PYTHON=python)
PYTHON ?= python3.11
NPZ_FILE ?= data/chromsome_data.npz
DATA_DIR ?= data/

.PHONY: env install-mac install-cuda clean clean-data extract-data parse-data split-data prepare-data reset-data train-model evaluate-model test

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
	find . -name ".DS_Store" -delete

# Clean generated data
clean-data:
	rm -rf $(DATA_DIR)raw $(DATA_DIR)processed

# --- Data Preparation Targets ---
extract-data:
	@echo "Extracting data from $(NPZ_FILE) to $(DATA_DIR)..."
	$(PYTHON) chromoseg/data/extract_npz.py --npz_path $(NPZ_FILE) --output_dir $(DATA_DIR)

parse-data:
	@echo "Converting masks to YOLO polygon labels..."
	mkdir -p $(DATA_DIR)raw/labels
	$(PYTHON) chromoseg/data/parser.py \
		--images_dir $(DATA_DIR)raw/images \
		--masks_dir $(DATA_DIR)raw/masks \
		--output_dir $(DATA_DIR)raw/labels

split-data:
	@echo "Splitting dataset into training and validation sets..."
	$(PYTHON) chromoseg/data/split.py \
		--images_dir $(DATA_DIR)raw/images \
		--labels_dir $(DATA_DIR)raw/labels \
		--output_dir $(DATA_DIR)processed \
		--train_ratio 0.8

prepare-data: extract-data parse-data split-data

# Wipe and regenerate entire dataset from scratch
reset-data: clean-data prepare-data

train-model:
	@echo "Starting model training..."
	$(PYTHON) chromoseg/engine/trainer.py \
		--data_config dataset.yaml \
		--epochs 50 \
		--img_size 640

evaluate-model:
	@echo "Evaluating model performance..."
	$(PYTHON) chromoseg/engine/evaluator.py \
		--weights runs/segment/models/baseline/weights/best.pt \
		--val data/processed/images/val \
		--labels_dir data/processed/labels/val \
		--img_size 640

test:
	@echo "Running automated test suite..."
	$(PYTHON) -m pytest -v tests/