import os

import argparse
import torch
import cv2
import numpy as np
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "nanodet"))

from nanodet.data.transform import Pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="NanoDet-Plus single image inference script")
    parser.add_argument("--image_path", type=str, required=True,
                        help="Path to input image")
    parser.add_argument("--model_path", type=str, default='./model_best.pth',
                        help="Path to fine-tuned NanoDet-Plus model weights")
    parser.add_argument("--config_path", type=str, default='./nanodet/config/ldxray_mainview_nanodet_plus_m_416_test.yml',
                        help="Path to NanoDet config YAML")
    return parser.parse_args()


def load_model(config_path, model_path):
    from nanodet.util import cfg, load_config
    from nanodet.model.arch import build_model
    from nanodet.util import load_model_weight

    load_config(cfg, config_path)
    model = build_model(cfg.model)
    ckpt = torch.load(model_path, map_location="cpu")
    load_model_weight(model, ckpt, logger=None)
    model.eval()
    return cfg, model


def run_inference(model, cfg, image_path):
    """
    Run inference on a single image and return a list of detections.
    Return format: list of dict
      [{"bbox": [x1, y1, x2, y2], "confidence": float, "predicted_class": int}, ...]
    """
    pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)
    img = cv2.imread(image_path)
    meta = dict(
        img_info={"height": [img.shape[0]], "width": [img.shape[1]], "id": [0]},
        raw_img=[img],
        img=img,
        warp_matrix=[np.eye(3, dtype=np.float32)],
    )
    meta = pipeline(None, meta, cfg.data.val.input_size)
    if not isinstance(meta["warp_matrix"], list):
        meta["warp_matrix"] = [meta["warp_matrix"]]

    tensor = torch.from_numpy(meta["img"].transpose(2, 0, 1)).unsqueeze(0).float()

    with torch.no_grad():
        meta["img"] = tensor
        results = model.inference(meta)
    
    print("results type:", type(results))
    print("results keys:", list(results.keys()) if isinstance(results, dict) else "not dict")
    print("results[0] type:", type(results[0]))
    print("results[0] keys:", list(results[0].keys()) if isinstance(results[0], dict) else results[0])

    detections = []
    # results is typically {class_id: [[x1, y1, x2, y2, score], ...], ...}
    for class_id, bboxes in results[0].items():
        if len(bboxes) > 0:
            print(f"class {class_id}: count={len(bboxes)}, sample={bboxes[0]}")
        for bbox in bboxes:
            x1, y1, x2, y2, score = bbox
            detections.append({
                "bbox": [round(float(x1), 2), round(float(y1), 2),
                         round(float(x2), 2), round(float(y2), 2)],
                "confidence": round(float(score), 4),
                "predicted_class": int(class_id),
            })
    return detections


def main():
    args = parse_args()

    cfg, model = load_model(args.config_path, args.model_path)
    detections = run_inference(model, cfg, args.image_path)

    print("=" * 50)
    print("Inference Result")
    print("=" * 50)
    print(f"  image:      {args.image_path}")
    print(f"  detections: {len(detections)}")
    for det in detections:
        print(f"    class={det['predicted_class']}  conf={det['confidence']}  bbox={det['bbox']}")


if __name__ == "__main__":
    main()
