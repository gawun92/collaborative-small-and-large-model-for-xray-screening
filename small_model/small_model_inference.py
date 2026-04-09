import json
import os
import argparse
import torch
import cv2
import numpy as np
from tqdm import tqdm
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "nanodet"))

from nanodet.data.transform import Pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="NanoDet-Plus inference script")
    parser.add_argument("--image_dir", type=str, default='./test_images',
                        help="Path to test image directory")
    parser.add_argument("--model_path", type=str, default='./model_best.pth',
                        help="Path to fine-tuned NanoDet-Plus model weights")
    parser.add_argument("--config_path", type=str, default='./nanodet/config/ldxray_mainview_nanodet_plus_m_416_test.yml',
                        help="Path to NanoDet config YAML")
    parser.add_argument("--output_dir", type=str, default="./inference_output",
                        help="Directory to save output files")
    parser.add_argument("--batch_size", type=int, default=8)
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


def run_inference_batch(model, cfg, image_paths, batch_size=16):
    pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)
    all_detections = []

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Inference(Batch)"):
        batch_paths = image_paths[i:i + batch_size]
        tensors = []
        metas_list = []

        for img_path in batch_paths:
            img = cv2.imread(img_path)
            meta = dict(
                img_info={"height": [img.shape[0]], "width": [img.shape[1]], "id": [0]},
                raw_img=[img],
                img=img,
                warp_matrix=[np.eye(3, dtype=np.float32)],
            )
            meta = pipeline(None, meta, cfg.data.val.input_size)
            if not isinstance(meta["warp_matrix"], list):
                meta["warp_matrix"] = [meta["warp_matrix"]]
            tensors.append(torch.from_numpy(meta["img"].transpose(2, 0, 1)).float())
            metas_list.append(meta)

        batch_tensor = torch.stack(tensors)
        batch_meta = {
            "img": batch_tensor,
            "img_info": {
                "height": [m["img_info"]["height"][0] for m in metas_list],
                "width": [m["img_info"]["width"][0] for m in metas_list],
                "id": list(range(len(metas_list))),
            },
            "warp_matrix": [m["warp_matrix"][0] for m in metas_list],
        }

        with torch.no_grad():
            results = model.inference(batch_meta)

        for j in sorted(results.keys()):
            detections = []
            for class_id, bboxes in results[j].items():
                for bbox in bboxes:
                    x1, y1, x2, y2, score = bbox
                    detections.append({
                        "bbox": [round(float(x1), 2), round(float(y1), 2),
                                 round(float(x2), 2), round(float(y2), 2)],
                        "confidence": round(float(score), 4),
                        "predicted_class": int(class_id),
                    })
            all_detections.append(detections)

    return all_detections


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    cfg, model = load_model(args.config_path, args.model_path)

    image_files = sorted([
        f for f in os.listdir(args.image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ])
    image_paths = [os.path.join(args.image_dir, f) for f in image_files]
    all_detections = run_inference_batch(model, cfg, image_paths, batch_size=args.batch_size)

    results = [
        {"image_path": img_path, "detections": detections}
        for img_path, detections in zip(image_paths, all_detections)
    ]

    output_path = os.path.join(args.output_dir, "inference_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    total_dets = sum(len(r["detections"]) for r in results)
    print("=" * 50)
    print("Inference Summary")
    print("=" * 50)
    print(f"  total_images:     {len(results)}")
    print(f"  total_detections: {total_dets}")
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
