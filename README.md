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
- **Bio-Augmentation Pipeline**: Domain-specific transformations including elastic deformations, fluorescence intensity jittering, and morphological scaling.
- **Cross-Platform Hardware Acceleration**: Automatic device detection supporting **NVIDIA CUDA** (Linux/Windows), **Apple Silicon MPS** (macOS Metal), and CPU fallback via `chromoseg.utils.get_device()`.
- **Modular Engine**: Decoupled, extensible modules for model definition, training, evaluation, and inference.
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
│   ├── data/               # Dataset loaders, bio-formats parsers & augmentations
│   ├── engine/             # Training, evaluation, and inference engines
│   │   ├── trainer.py      # Modular training loop with checkpointing & W&B
│   │   ├── evaluator.py    # Metric calculation (mAP, Dice, boundary IoU)
│   │   └── predictor.py    # High-throughput batch inference pipeline
│   └── models/             # PyTorch models, backbones, and loss functions
│       ├── backbones.py    # Custom feature extractors & attention layers
│       ├── losses.py       # Dice-Focal & boundary-aware loss layers
│       └── yolo_wrapper.py # Segmentation model wrapper & export utilities
├── data/                   # Dataset root (raw & processed cytogenetic data)
├── deploy/                 # Deployment configs, ONNX / TensorRT export scripts
├── notebooks/              # Interactive Jupyter exploration and karyotyping analysis
├── weights/                # Model checkpoints and pre-trained weights
├── .env.example            # Environment variable configuration template
├── Makefile                # Automated setup, installation, and maintenance tasks
├── pyproject.toml          # Package specifications and dependencies
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
    data_config="data/dataset.yaml",
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
    source="data/test/metaphase_spread.jpg",
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

### Clean Build Artifacts
Remove build cache, bytecode, and temporary files:
```bash
make clean
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
