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

Real-Time Instance Segmentation & Automated Cytogenetics Engine for human metaphase chromosome karyotyping.

### 🧬 Key Capabilities:
- **Instant Instance Segmentation**: Color-coded pixel-level chromosome outlines.
- **Touching & Overlapping Cluster Resolution**: Trained with custom Boundary-Focal loss to separate touching chromatids.
- **Clinical Karyotyping Assessment**: Real-time count diagnostic reporting ($\Delta N$) and numerical abnormality detection ($2n=46, 47, 45$).
- **Cutout Gallery**: Automated extraction of individual chromosome crops for karyogram alignment.

### 💻 Local Run:
```bash
pip install -r requirements.txt
python app.py
```

