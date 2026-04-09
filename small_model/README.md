# small_model_inference_single

Run NanoDet-Plus inference on a single image and print detections to the terminal.

`nanodet/` must be placed in the same directory as the script.

## Setup

### 1. Clone NanoDet

```bash
git clone https://github.com/RangiLyu/nanodet.git
```

Place the cloned `nanodet/` folder in the same directory as the script.

### 2. Place config file

Copy your config YAML into `nanodet/config/`:

```
nanodet/
└── config/
    └── ldxray_mainview_nanodet_plus_m_416_test.yml  ← place here
```

### 3. Download model weights

Download `model_best.pth` from the link below and place it in the same directory as the script:

[Google Drive — model_best.pth](https://drive.google.com/file/d/1fZYQEPvwafc4HNQ1XGUP5-Nayd0rzYyB/view?usp=sharing)

```
archive/
├── small_model_inference_single.py
├── model_best.pth  ← place here
└── nanodet/
    └── ...
```

### 4. Install dependencies

```bash
pip install torch opencv-python numpy
pip install -r nanodet/requirements.txt
```

## Usage

```bash
python small_model_inference_single.py \
    --image_path ./path/to/image.jpg
```

### All arguments

| Argument | Default | Description |
|---|---|---|
| `--image_path` | *(required)* | Path to input image |
| `--model_path` | `./model_best.pth` | Path to model weights |
| `--config_path` | `./nanodet/config/ldxray_mainview_nanodet_plus_m_416_test.yml` | Path to NanoDet config YAML |

## Output

Prints detection results to the terminal:

```
==================================================
Inference Result
==================================================
  image:      ./path/to/image.jpg
  detections: 3
    class=0  conf=0.8231  bbox=[120.5, 30.2, 400.1, 280.7]
    class=1  conf=0.6142  bbox=[50.0, 10.0, 200.0, 150.0]
    ...
```

Each detection contains:
- `class`: predicted class ID (int)
- `conf`: confidence score
- `bbox`: bounding box `[x1, y1, x2, y2]`
