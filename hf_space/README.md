---
title: ChromoSeg AI Cytogenetics
emoji: 🔬
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.26.0
app_file: app.py
pinned: false
license: agpl-3.0
---

# 🔬 ChromoSeg: Clinical Cytogenetics AI Engine

[![Ultralytics YOLO](https://img.shields.io/badge/Ultralytics-YOLO-00FFFF.svg)](https://docs.ultralytics.com/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Hardware Support](https://img.shields.io/badge/Hardware-CUDA%20%7C%20MPS%20%7C%20CPU-success.svg)](https://pytorch.org/)

**Real-Time Instance Segmentation & Automated Cytogenetics Engine** for human metaphase chromosome karyotyping, cluster disentanglement, and clinical diagnostics.

---

### 🧬 Key Capabilities
- **2-Class Clinical Architecture**: Segments full individual **chromosome bodies** (`Class 0`) and isolates dense **crossover / overlap junctions** (`Class 1`).
- **Cluster & Touching Disentanglement**: Built on custom Boundary-Aware and Dice-Focal loss formulations to resolve touching chromatid boundaries.
- **Clinical Count Diagnostics ($\Delta N$)**: Real-time evaluation of chromosome numbers to detect numerical abnormalities (e.g., Aneuploidy / Trisomy / Monosomy).
- **Automated Cutout Gallery**: Isolates individual segmented chromosome crops for downstream karyogram pairing.

---

### 📊 Benchmark Performance (Validation Set)
- **Mean Count Error ($\text{MAE } \Delta N$)**: **0.43 chromosomes**
- **Count Accuracy ($\pm 1$ Tolerance)**: **95.0%**
- **Overlapping Cluster IoU**: **0.6707 (67.1%)**
- **Inference Latency**: **~3.2 ms / image** (real-time GPU throughput)

---

### 💻 Local Run
```bash
pip install -r requirements.txt
python app.py
```


