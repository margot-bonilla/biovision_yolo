# 🔬 ChromoSeg-YOLO: Clinical Cytogenetics AI Engine

**Real-Time Instance Segmentation, Overlap Disentanglement & Automated Karyotyping**

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c.svg)](https://pytorch.org/)
[![Ultralytics YOLO](https://img.shields.io/badge/Ultralytics-YOLO-00FFFF.svg)](https://docs.ultralytics.com/)
[![Hugging Face Space](https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Demo-yellow.svg)](https://huggingface.co/spaces/margot-bonilla/chromoseg-yolo)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Hardware Support](https://img.shields.io/badge/Hardware-CUDA%20%7C%20MPS%20%7C%20CPU-success.svg)](https://pytorch.org/)
[![Tests Passing](https://img.shields.io/badge/pytest-9%2F9%20passed-brightgreen.svg)](tests/)

---

## 🧬 Overview

**ChromoSeg-YOLO** is an open-source, high-throughput computer vision engine engineered for clinical cytogenetics. It delivers real-time instance segmentation of human metaphase chromosomes, directly addressing the primary bottleneck in automated karyotyping: **resolving touching and overlapping chromosome clusters**.

In clinical cytogenetics, numerical aberrations (such as trisomies or monosomies) and structural translocations require precise chromosome counting and morphometric analysis. Touching and overlapping chromatids routinely confound standard vision models. ChromoSeg-YOLO introduces a **2-Class Clinical Architecture** coupled with a **GPU-Accelerated Boundary-Aware Bio-Loss** to segment individual chromosome bodies while pinpointing dense crossover junctions at high inference throughput.

```
                  ┌──────────────────────────────────────────────────┐
                  │          Metaphase Microscopic Spread            │
                  └────────────────────────┬─────────────────────────┘
                                           │
                                           ▼
                  ┌──────────────────────────────────────────────────┐
                  │                 ChromoSeg-YOLO                   │
                  │   (Custom v8SegmentationLoss + Morph Boundary)   │
                  └────────────┬────────────────────────┬────────────┘
                               │                        │
                               ▼                        ▼
               ┌────────────────────────┐   ┌────────────────────────┐
               │ Class 0: 'chromosome'  │   │   Class 1: 'overlap'   │
               │ (Full Body Contours)   │   │  (Crossover Junctions) │
               └───────────────┬────────┘   └───────────┬────────────┘
                               │                        │
                               └───────────┬────────────┘
                                           ▼
                  ┌──────────────────────────────────────────────────┐
                  │       Clinical Karyotype Diagnostic Report       │
                  │    Count Error (ΔN) & Automated Cutout Gallery   │
                  └──────────────────────────────────────────────────┘
```

---

## 🚀 Key Innovations & Clinical Features

- **2-Class Clinical Architecture**: Disentangles chromosome clusters into individual body instances (`Class 0`) and dense optical crossover junctions (`Class 1`).
- **GPU-Accelerated Bio-Loss Function**: Replaces standard Cross-Entropy with a differentiable **Dice-Focal Loss** ($\gamma=2.0, \alpha=0.25$) and a **Pure PyTorch Morphological Boundary Loss** computed on GPU/MPS via `F.max_pool2d` (0.5 ms execution, zero CPU transfers).
- **Clinical Evaluation Metrics ($\Delta N$)**: Evaluates Mean Absolute Error in chromosome count ($\text{MAE } \Delta N$), exact spread count accuracy ($\Delta N=0$), within $\pm 1$ clinical tolerance, and Overlapping Cluster IoU.
- **Real-Time High Throughput**: Ultra-fast inference latency (~3.2 ms on GPU, ~14 ms on Apple Silicon MPS), enabling instantaneous laboratory slide screening.
- **Cross-Platform Acceleration**: Native automatic device selection supporting **NVIDIA CUDA** (Linux/Windows), **Apple Silicon MPS** (macOS Metal), and CPU fallback via `chromoseg.utils.get_device()`.
- **Interactive Web Application**: Deployed Gradio application featuring confidence sliders, color-coded diagnostic overlays, and individual chromosome cutout galleries.

---

## 📊 Benchmark Results

Evaluated on the metaphase chromosome validation benchmark (100 multi-chromosome overlapping spreads):

| Architecture / Configuration | MAE Count Error ($\Delta N$) $\downarrow$ | Exact Count Acc ($\Delta N=0$) $\uparrow$ | Within $\pm 1$ Tolerance $\uparrow$ | Overlap Cluster IoU $\uparrow$ | Inference Latency $\downarrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline YOLO-Seg** | $1.84$ chrom | $24.0\%$ | $58.0\%$ | $0.4821$ | **3.1 ms** |
| **ChromoSeg-YOLO (2-Class + Bio-Loss)** | **0.43 chrom** | **63.0%** | **95.0%** | **0.6707** | **3.2 ms** |

> **Clinical Significance**: ChromoSeg-YOLO cuts count error by **$76.6\%$**, raises single-instance clinical tolerance accuracy to **$95.0\%$**, and achieves **$67.1\%$ IoU** on dense overlapping regions.

---

## 🔬 Mathematical Formulation: Custom Bio-Loss

Metaphase spreads feature extreme class imbalance (95% background) and subtle morphological borders. ChromoSeg-YOLO overrides `v8SegmentationLoss.single_mask_loss` with a unified cytogenetics loss:

$$\mathcal{L}_{\text{Cytogenetics}} = \mathcal{L}_{\text{Dice-Focal}}(\hat{M}, M) + \lambda \cdot \mathcal{L}_{\text{Boundary}}(\hat{M}, M)$$

### 1. Dice-Focal Loss
Combines Focal loss with Soft Dice loss to suppress background gradients and optimize global mask overlap:

$$\mathcal{L}_{\text{Focal}} = -\alpha (1 - p_t)^\gamma \log(p_t), \quad \mathcal{L}_{\text{Dice}} = 1 - \frac{2 \sum \hat{M} M + \epsilon}{\sum \hat{M} + \sum M + \epsilon}$$

### 2. GPU Morphological Boundary Loss
Extracts the contour gradient directly on the GPU using 2D Max Pooling:

$$\partial M = \text{MaxPool2D}(M, k) - \left(1 - \text{MaxPool2D}(1 - M, k)\right)$$

$$\mathcal{L}_{\text{Boundary}} = 1 - \frac{2 \sum \partial \hat{M} \cdot \partial M + \epsilon}{\sum \partial \hat{M} + \sum \partial M + \epsilon}$$

---

## 📂 Project Structure

```
biovision_yolo/
├── assets/                 # Visual assets and demo previews
├── chromoseg/              # Core Python package
│   ├── data/               # Data pipeline: extract, 2-class parser, 80/20 split
│   │   ├── extract_npz.py  # Unpacks .npz archive into raw images and masks
│   │   ├── parser.py       # Converts 3-channel masks to 2-class YOLO polygons
│   │   └── split.py        # Generates leak-free train/val splits
│   ├── engine/             # Training & evaluation engines
│   │   ├── trainer.py      # CustomCytogeneticsLoss & CytogeneticsTrainer
│   │   └── evaluator.py    # Clinical count error (ΔN) & overlap IoU evaluator
│   ├── models/             # Custom loss modules
│   │   └── losses.py       # GPU Dice-Focal & Morphological Boundary loss
│   └── utils.py            # Device auto-detection (CUDA / MPS / CPU)
├── hf_space/               # Standalone Hugging Face Space deployment bundle
│   ├── app.py              # ZeroGPU / CPU Gradio application
│   ├── best.pt             # Trained 2-class model weights
│   ├── requirements.txt    # Hugging Face dependencies
│   └── README.md           # Space configuration card
├── tests/                  # Automated pytest test suite
│   ├── test_data_pipeline.py
│   ├── test_evaluator.py
│   └── test_losses.py
├── Makefile                # Automated setup, data pipeline, training & testing
├── pyproject.toml          # Package metadata & dependencies
├── dataset.yaml            # YOLO segmentation dataset config (2 classes)
└── README.md
```

---

## ⚡ Quick Start & Installation

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/margot-bonilla/biovision_yolo.git
cd biovision_yolo

# Create virtual environment
make env
source .venv/bin/activate
```

### 2. Install Dependencies

Select your hardware backend:
```bash
# macOS (Apple Silicon / MPS)
make install-mac

# Linux / Windows (NVIDIA GPU with CUDA)
make install-cuda
```

---

## 🔄 Automated End-to-End Pipeline

The project includes convenient `make` targets for all pipeline operations:

```bash
# Run automated test suite (9 tests)
make test

# Full end-to-end retrain: clean -> extract -> parse -> split -> train
make retrain

# Evaluate trained weights on clinical metrics
make evaluate-model

# Full pipeline: data preparation -> train -> evaluate
make pipeline
```

---

## 🖥️ Interactive Web Demo (Gradio)

Launch the interactive cytogenetics dashboard locally:

```bash
python app.py
```

Features included:
1. **Interactive Confidence Slider**: Adjust detection confidence from 0.10 to 0.90 in real-time.
2. **Color-Coded Visualizations**: Cyan outlines for individual chromosomes (`Class 0`), Red masks for overlap junctions (`Class 1`).
3. **Diagnostic Count Card**: Real-time count error assessment and touching cluster alerts.
4. **Chromosome Gallery**: Automated crop extraction of every detected chromosome for individual inspection.

---

## 🧪 Automated Testing

ChromoSeg-YOLO includes 9 automated unit tests verifying the data pipeline, loss autograd differentiability, and evaluation metrics:

```bash
make test
```

```text
============================== test session starts ===============================
tests/test_data_pipeline.py::test_extract_npz PASSED                       [ 11%]
tests/test_data_pipeline.py::test_parser_area_filter_and_normalization PASSED   [ 22%]
tests/test_data_pipeline.py::test_parser_2class_chromosomes_and_overlap PASSED  [ 33%]
tests/test_data_pipeline.py::test_split_dataset_no_leakage PASSED          [ 44%]
tests/test_evaluator.py::test_evaluator_metrics_structure PASSED           [ 55%]
tests/test_losses.py::test_dice_focal_loss_values_and_gradients PASSED     [ 66%]
tests/test_losses.py::test_extract_boundary_gpu PASSED                     [ 77%]
tests/test_losses.py::test_boundary_loss_differentiability PASSED          [ 88%]
tests/test_losses.py::test_cytogenetics_loss_module PASSED                 [100%]
=============================== 9 passed in 1.90s ================================
```

---

## 🔭 Future Directions & Research Roadmap

1. **Topological Crossover Disentanglement**: Incorporating directional graph pathfinding along chromosome medial axes to reconstruct full individual chromatid geometries under extreme multiple overlaps.
2. **GNN-Based Homolog Pairing**: Developing a Graph Neural Network (GNN) on top of segmented chromosome embeddings to automatically pair homologous autosomes (1–22) and sex chromosomes (X/Y) into standardized **ISCN Karyograms**.
3. **Multi-Spectral Banding Classification**: Expanding the feature extractor with self-supervised vision transformers (DINOv2 / BioViL) to classify Giemsa (G-banding) and Q-banding patterns for automated structural translocation detection.

---

## 📜 Citation & License

```bibtex
@software{chromoseg_yolo_2026,
  author = {Margot Bonilla},
  title = {ChromoSeg-YOLO: Real-Time Instance Segmentation & Automated Cytogenetics Engine},
  year = {2026},
  url = {https://github.com/margot-bonilla/biovision_yolo}
}
```

Distributed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**. See [`LICENSE`](LICENSE) for complete terms.
