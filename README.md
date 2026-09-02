# ChromoSeg-YOLO

**Real-Time Instance Segmentation & Automated Cytogenetics Engine**

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c.svg)](https://pytorch.org/)
[![Ultralytics YOLO](https://img.shields.io/badge/Ultralytics-YOLO-00FFFF.svg)](https://docs.ultralytics.com/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Hardware Support](https://img.shields.io/badge/Hardware-CUDA%20%7C%20MPS%20%7C%20CPU-success.svg)](https://pytorch.org/)

---

## Overview

**ChromoSeg-YOLO** is an open-source, high-throughput computer vision pipeline engineered for real-time instance segmentation and classification of human metaphase chromosomes and cellular nuclei.

Built on PyTorch and modern object detection/segmentation paradigms, ChromoSeg-YOLO directly addresses core challenges in automated cytogenetics—specifically severe chromosome overlap, variable fluorescence contrast, and intricate morphological boundaries—to streamline karyotyping workflows and biological feature extraction.

<div align="center">
  <img src="assets/demo_inference.png" alt="Chromosome Segmentation Demo" width="750"/>
  <p><em>Real-time instance segmentation on metaphase chromosome spread using ChromoSeg-YOLO.</em></p>
</div>

---

## Key Features

- **Boundary-Aware Loss**: Integrated Dice-Focal and IoU loss formulations tailored for resolving fine-grained, overlapping biological boundaries.
- **Automated Cytogenetics Data Pipeline**: Fast `.npz` dataset extraction and multi-instance grayscale mask to normalized YOLO polygon segmentation annotation parsing.
- **Bio-Augmentation Pipeline**: Domain-specific transformations including elastic deformations, fluorescence intensity jittering, and morphological scaling.
- **Cross-Platform Hardware Acceleration**: Automatic device detection supporting **NVIDIA CUDA** (Linux/Windows), **Apple Silicon MPS** (macOS Metal), and CPU fallback via `chromoseg.utils.get_device()`.
- **Modular Engine**: Decoupled, extensible modules for data parsing, model definition, training, evaluation, and inference.
- **High-Speed Export & Deployment**: Optimized export pipeline targeting ONNX Runtime and TensorRT for high-throughput laboratory deployment.
- **Experiment Tracking**: Integrated Weights & Biases (W&B) logging for monitoring mAP@50-95, boundary metrics, and resource utilization.

---

## Project Structure

```
biovision_yolo/
├── assets/                 # Visual assets and demo media
│   └── demo_inference.png  # Metaphase chromosome segmentation preview
├── benchmarks/             # Benchmarking scripts & latency/memory profiling
├── chromoseg/              # Core Python package
│   ├── __init__.py
│   ├── utils.py            # Device management (CUDA / MPS / CPU) & helpers
│   ├── data/               # Dataset extraction, parsers, split & augmentations
│   │   ├── extract_npz.py  # Unpacks .npz archives into raw images and masks
│   │   ├── parser.py       # Converts instance masks to YOLO polygon format
│   │   └── split.py        # Generates reproducible train/val dataset splits
│   ├── engine/             # Training, evaluation, and inference engines
│   │   ├── trainer.py      # Modular training loop with checkpointing & W&B
│   │   ├── evaluator.py    # Metric calculation (mAP, Dice, boundary IoU)
│   │   └── predictor.py    # High-throughput batch inference pipeline
│   └── models/             # PyTorch models, backbones, and loss functions
│       ├── backbones.py    # Custom feature extractors & attention layers
│       ├── losses.py       # Dice-Focal & boundary-aware loss layers
│       └── yolo_wrapper.py # Segmentation model wrapper & export utilities
├── data/                   # Dataset root
│   ├── chromsome_data.npz  # Raw dataset archive
│   ├── raw/                # Unpacked raw data & parsed labels
│   │   ├── images/         # Raw metaphase chromosome spread images
│   │   ├── masks/          # Raw instance segmentation masks
│   │   └── labels/         # Raw YOLO polygon label files (.txt)
│   └── processed/          # Processed & split YOLO dataset
│       ├── images/         # Split images (train/ & val/)
│       └── labels/         # Split polygon label files (train/ & val/)
├── deploy/                 # Deployment configs, ONNX / TensorRT export scripts
├── notebooks/              # Interactive Jupyter exploration and karyotyping analysis
├── weights/                # Model checkpoints and pre-trained weights
├── .env.example            # Environment variable configuration template
├── Makefile                # Automated setup, data pipeline, and maintenance tasks
├── pyproject.toml          # Package specifications and dependencies
├── dataset.yaml            # YOLO segmentation dataset configuration
└── LICENSE                 # AGPL-3.0 License
```

---

## Installation & Setup

### Prerequisites
- Python **>= 3.9** (Python 3.11 recommended)
- `git`

### 1. Create Virtual Environment
Use the provided `Makefile` to create an isolated virtual environment:

```bash
# Create virtual environment (uses python3.11 by default)
make env

# Or specify a custom Python interpreter:
make env PYTHON=python3.10
```

Activate the environment:
```bash
# macOS / Linux
source .venv/bin/activate

# Windows (Command Prompt / PowerShell)
.venv\Scripts\activate
```

### 2. Install Dependencies

Select the installation target based on your hardware:

#### macOS (Apple Silicon / MPS)
```bash
make install-mac
```

#### Linux / Windows (NVIDIA GPU with CUDA 12.x)
```bash
make install-cuda
```

#### Manual / Standard Editable Install
```bash
pip install --upgrade pip
pip install -e ".[dev]"
```

### 3. Environment Configuration
Copy the sample environment file and update variables to match your system:

```bash
cp .env.example .env
```

Edit `.env` with your settings:
```ini
# Dataset root directory
DATA_ROOT=./data

# Weights & Biases tracking
WANDB_API_KEY=your_wandb_api_key_here
WANDB_PROJECT=chromoseg-yolo

# Hardware device: 'cuda', 'mps' (Apple Silicon), or 'cpu'
DEVICE=cuda
```

---

## Data Preparation Pipeline

ChromoSeg-YOLO includes a full end-to-end data pipeline to unpack raw cytogenetics `.npz` archives, convert multi-instance masks into normalized YOLO polygon segmentation annotations, and generate reproducible train/validation splits.

### Automated Workflow via Makefile

Run all data preparation steps in one command:
```bash
# Runs extract-data -> parse-data -> split-data
make prepare-data
```

Or run individual pipeline steps:
```bash
# 1. Extract raw images and masks from .npz dataset
make extract-data

# 2. Parse multi-instance masks into YOLO polygon labels
make parse-data

# 3. Create reproducible train/val splits (80% train, 20% val)
make split-data
```

> **Note**: Default variables can be overridden from the command line:
> ```bash
> make prepare-data NPZ_FILE=path/to/data.npz DATA_DIR=data/
> ```

### Manual / CLI Workflow

#### 1. Extracting `.npz` Archives
Unpack the `.npz` archive into raw image and mask folders:
```bash
python chromoseg/data/extract_npz.py \
    --npz_path data/chromsome_data.npz \
    --output_dir data/
```
Output: `data/raw/images/` and `data/raw/masks/`

#### 2. Parsing Masks to YOLO Segmentation Polygons
Convert multi-instance grayscale masks into normalized YOLO polygon labels:
```bash
python chromoseg/data/parser.py \
    --images_dir data/raw/images \
    --masks_dir data/raw/masks \
    --output_dir data/raw/labels
```

#### 3. Splitting into Train & Validation Sets
Generate paired image and label directories in standard YOLO segmentation structure:
```bash
python chromoseg/data/split.py \
    --images_dir data/raw/images \
    --labels_dir data/raw/labels \
    --output_dir data/processed \
    --train_ratio 0.8
```

#### Segmentation Format Details
The parser automatically:
- Identifies individual chromosomes based on unique intensity values in the mask.
- Extracts contours using `cv2.findContours`.
- Filters out non-chromosome debris and imaging artifacts (contour area threshold < 50 px).
- Normalizes polygon vertices to `[0, 1]` relative to image dimensions.
- Produces YOLO segmentation format: `<class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>`.

---

## Quick Start

### Device Verification
ChromoSeg-YOLO automatically selects the fastest available hardware backend:

```python
from chromoseg.utils import get_device

device = get_device()
print(f"Active compute device: {device}")
# Output: 'cuda' on NVIDIA GPU, 'mps' on Apple Silicon, or 'cpu'
```

### Training Pipeline
```python
from chromoseg.engine.trainer import Trainer

# Initialize and launch training with custom configurations
trainer = Trainer(
    data_config="dataset.yaml",
    model="chromoseg-yolov8-seg",
    epochs=100,
    imgsz=1024,
    device=get_device()
)
trainer.train()
```

### Inference & Visualization
```python
from chromoseg.engine.predictor import Predictor

predictor = Predictor(weights="weights/best.pt")
results = predictor.predict(
    source="data/processed/images/val/spread_0001.png",
    conf_threshold=0.25,
    save=True
)
```

---

## Development & Testing

### Code Quality & Linting
The development environment includes `black`, `flake8`, and `pytest`:

```bash
# Format code
black chromoseg/

# Lint code
flake8 chromoseg/

# Run unit tests
pytest tests/
```

### Clean Build & Data Artifacts
Remove build cache, bytecode, and temporary files:
```bash
make clean
```

Reset and re-extract/re-split dataset from scratch:
```bash
make reset-data
```

Wipe all extracted/generated dataset files:
```bash
make clean-data
```

---

## Citation & Contact

If you use ChromoSeg-YOLO in your cytogenetics or computer vision research, please cite:

```bibtex
@software{chromoseg_yolo_2026,
  author = {Margot Bonilla},
  title = {ChromoSeg-YOLO: Real-Time Instance Segmentation & Automated Cytogenetics Engine},
  year = {2026},
  url = {https://github.com/margot-bonilla/biovision_yolo}
}
```

---

## License

Distributed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**. See [`LICENSE`](LICENSE) for complete terms.
