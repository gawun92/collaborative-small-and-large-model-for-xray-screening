# Collaborative Small-Large Model Pipeline for X-ray Security Screening

## Overview

X-ray security screening requires both **speed** and **accuracy**.
This project implements a **collaborative inference pipeline** that routes each image through a lightweight small model first, and only escalates to a heavier large model when the small model's confidence is low.

```
Input Image
    │
    ▼
[NanoDet-Plus]  ──── confidence ≥ threshold ────▶  Use small model prediction
    │
    └── confidence < threshold ──────────────────▶  [YOLOv11L]  ──▶  Use large model prediction
```

**Best result (t=0.4, mean_score routing):**
- **54% faster** than Large Only (2,186s vs 4,746s on 36,849 test images)
- **+6.97%p AP@50** over Small Only (74.45% vs 67.48%)
- Only **26.2% of images** escalated to the large model

---

## Models

### Small Model — NanoDet-Plus

| Item | Detail |
|---|---|
| Architecture | NanoDet-Plus-m-416 |
| Task | Multi-class object detection |
| Confidence threshold | 0.3 |
| Config | `small_model/ldxray_mainview_nanodet_plus_m_416_test.yml` |
| Weights | `small_model/model_best.pth` |
| Inference entry | `small_model/small_model_inference_single.py` |

NanoDet-Plus is a lightweight anchor-free detector optimized for speed.
It runs efficiently on CPU and serves as the first-stage filter in the pipeline.

---

### Large Model — YOLOv11L

| Item | Detail |
|---|---|
| Architecture | YOLOv11L (fine-tuned) |
| Task | Multi-class object detection |
| Confidence threshold | 0.5 |
| Weights | `YOLOv11L/best.pt` |
| Inference entry | `YOLOv11L/test_model.py` |

YOLOv11L is a high-capacity detector that delivers strong accuracy.
It is only invoked when the small model produces a low-confidence result, keeping overall inference time low.

---

## Dataset — LDXray

- **Source**: [LDXray GitHub](https://github.com/jhb86253817/LDXray)
- **Categories**: 12 prohibited item classes
- **Training set**: 146,997 images
- **Test set**: 36,849 images
- **Annotation format**: COCO JSON
- **Note**: Only the **main view** images were used for both training and evaluation (side/top views excluded)

---

## Project Structure

```
├── pipeline.py                        # Main pipeline (Small / Large / Collaborative)
├── pipeline_with_generated_images.py  # Same pipeline + saves annotated output images
├── pipeline_with_various_routings.py  # Compares all 4 routing methods
├── visualize_results.py               # Generates result charts (fig1~fig4)
├── models.py                          # Model wrappers
├── evaluator.py                       # Per-image AP@50 computation
├── data_loader.py                     # GT annotation loader
├── config.yaml                        # Pipeline configuration
├── routing/
│   ├── max_score.py                   # Routing: max confidence score
│   ├── mean_score.py                  # Routing: mean confidence score
│   ├── entropy.py                     # Routing: prediction entropy
│   └── nms_box_count.py               # Routing: number of detected boxes
├── small_model/
│   ├── model_best.pth                 # NanoDet-Plus weights
│   ├── ldxray_mainview_nanodet_plus_m_416_test.yml
│   └── small_model_inference_single.py
├── YOLOv11L/
│   ├── best.pt                        # YOLOv11L weights
│   └── test_model.py
└── data/
    ├── test.json                      # GT annotations (COCO format) — label file
    ├── images/                        # Test images go here
    └── output/                        # Output charts and visualizations
```

---

## Setup & Usage

### 1. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 2. Install dependencies

```bash
# Install PyTorch first (choose your CUDA version)
# https://pytorch.org/get-started/locally/

# Install NanoDet from source
pip install -e ./small_model/nanodet/

# Install remaining dependencies
pip install -r requirements.txt
```

### 3. Prepare dataset

Place all test images under `data/images/`:

```
data/
├── images/
│   ├── image_00001.jpg
│   ├── image_00002.jpg
│   └── ...
└── test.json        ← COCO-format label file (GT annotations)
```

### 4. Run the pipeline

```bash
# Basic pipeline: Small Only / Large Only / Collaborative (t=0.4~0.7)
python pipeline.py --config config.yaml

# Same pipeline + saves annotated prediction images to data/output/
python pipeline_with_generated_images.py --config config.yaml

# Compare all 4 routing methods
python pipeline_with_various_routings.py --config config.yaml

# Generate result charts
python visualize_results.py
```

**Full example from scratch:**

```bash
python -m venv venv
source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118  # adjust CUDA version
pip install -e ./small_model/nanodet/
pip install -r requirements.txt
python pipeline.py --config config.yaml
```

---

## Configuration

```yaml
# config.yaml
pipeline:
  confidence_threshold: 0.5       # Escalation threshold (images below this go to large model)
  routing_method: "mean_score"    # max_score | mean_score | entropy | nms_box_count

data:
  test_json: "./data/test.json"
  image_dir: "./data/images/"

evaluation:
  iou_threshold: 0.5              # AP@50
```

---

## Results

| Pipeline | Mean AP@50 | Inference Time | Escalation Rate |
|---|---|---|---|
| Small Only (NanoDet-Plus) | 67.48% | 1,487s | 0% |
| Large Only (YOLOv11L) | 74.63% | 4,746s | 100% |
| **Collaborative (t=0.4)** | **74.45%** | **2,186s** | **26.2%** |
| Collaborative (t=0.5) | 74.14% | 4,189s | 68.5% |
| Collaborative (t=0.6) | 74.43% | 5,438s | 94.6% |
| Collaborative (t=0.7) | 74.63% | 5,709s | 100.0% |

Evaluated on 36,849 LDXray test images with `mean_score` routing.

---

## Output Charts

Generated by `visualize_results.py` and saved to `data/output/`:

| File | Description |
|---|---|
| `fig1_pipeline_comparison.png` | AP@50 & inference time for all 6 pipelines |
| `fig2_threshold_sweep.png` | AP@50 / time / escalation rate vs threshold |
| `fig3_tradeoff_scatter.png` | Accuracy–efficiency tradeoff scatter plot |
| `fig4_confidence_histograms.png` | Confidence score distributions (Small vs Large) |
