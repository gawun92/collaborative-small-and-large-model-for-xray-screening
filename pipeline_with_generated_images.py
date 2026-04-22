"""
pipeline.py
Main entry point. Runs three pipelines sequentially and prints a summary.

Usage:
    python pipeline.py --config config.yaml
"""

import argparse
import os
import sys
import time
from typing import List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import yaml
from tqdm import tqdm

from data_loader import load_gt
from evaluator import EvalResult, compute_image_ap50, print_confidence_histogram
from models import run_small_model, run_large_model
from routing import ROUTING_METHODS

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

PALETTE = [
    (68, 68, 255), (0, 136, 255), (0, 221, 255), (68, 255, 68),
    (255, 187, 0), (255, 68, 136), (187, 68, 255), (204, 255, 0),
    (0, 102, 255), (255, 68, 0), (0, 255, 170), (136, 0, 255),
]


# Config
def load_config(path: str) -> dict:
    with open(path) as f:
        config = yaml.safe_load(f)
    config_dir = os.path.dirname(os.path.abspath(path))
    config["data"]["test_json"] = os.path.join(config_dir, config["data"]["test_json"])
    config["data"]["image_dir"] = os.path.join(config_dir, config["data"]["image_dir"])
    return config


# Visualization
def save_escalated_visualization(image_path: str, small_pred: List, large_pred: List, gt: List, save_path: str) -> None:
    img = cv2.imread(image_path)
    if img is None:
        return
    # Small model predictions: green
    for p in small_pred:
        x1, y1, x2, y2 = [int(v) for v in p["bbox"]]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"S cls{p['predicted_class']} {p['confidence']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 255, 0), -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    # Large model predictions: red
    for p in large_pred:
        x1, y1, x2, y2 = [int(v) for v in p["bbox"]]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = f"L cls{p['predicted_class']} {p['confidence']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 0, 255), -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    # GT bboxes: blue (xywh -> xyxy)
    for g in gt:
        x, y, w, h = [int(v) for v in g["bbox"]]
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        label = f"GT cls{g['category_id']}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(img, (x, y - th - 6), (x + tw + 4, y), (255, 0, 0), -1)
        cv2.putText(img, label, (x + 2, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img)


def save_visualization(image_path: str, pred: List, save_path: str) -> None:
    img = cv2.imread(image_path)
    if img is None:
        return
    for p in pred:
        x1, y1, x2, y2 = [int(v) for v in p["bbox"]]
        cls_id     = p["predicted_class"]
        conf_score = p["confidence"]
        color      = PALETTE[(cls_id - 1) % len(PALETTE)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"cls{cls_id} {conf_score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img)


# Pipeline 1 : Small model only
def run_small_only(images: List[str], gt_dict, config, vis_dir: str) -> EvalResult:
    iou_threshold     = config["evaluation"]["iou_threshold"]
    routing_fn        = ROUTING_METHODS[config["pipeline"]["routing_method"]]
    image_dir         = config["data"]["image_dir"]
    ap50_list         = []
    confidence_scores = []
    start             = time.time()

    for file_name in tqdm(images, desc="Small Only"):
        image_path = os.path.join(image_dir, file_name)
        pred = run_small_model(image_path)
        gt   = gt_dict.get(file_name, [])
        ap50_list.append(compute_image_ap50(pred, gt, iou_threshold))
        confidence_scores.append(routing_fn(pred))
        save_visualization(image_path, pred, os.path.join(vis_dir, "small_only", file_name))

    return EvalResult(
        pipeline_name="Small Only",
        total_time=round(time.time() - start, 4),
        mean_ap50=round(sum(ap50_list) / len(ap50_list), 4) if ap50_list else 0.0,
        small_model_calls=len(images),
        confidence_scores=confidence_scores,
    )


# Pipeline 2 : Large model only
def run_large_only(images: List[str], gt_dict, config, vis_dir: str) -> EvalResult:
    iou_threshold     = config["evaluation"]["iou_threshold"]
    routing_fn        = ROUTING_METHODS[config["pipeline"]["routing_method"]]
    image_dir         = config["data"]["image_dir"]
    ap50_list         = []
    confidence_scores = []
    start             = time.time()

    for file_name in tqdm(images, desc="Large Only"):
        image_path = os.path.join(image_dir, file_name)
        pred = run_large_model(image_path)
        gt   = gt_dict.get(file_name, [])
        ap50_list.append(compute_image_ap50(pred, gt, iou_threshold))
        confidence_scores.append(routing_fn(pred))
        save_visualization(image_path, pred, os.path.join(vis_dir, "large_only", file_name))

    return EvalResult(
        pipeline_name="Large Only",
        total_time=round(time.time() - start, 4),
        mean_ap50=round(sum(ap50_list) / len(ap50_list), 4) if ap50_list else 0.0,
        large_model_calls=len(images),
        confidence_scores=confidence_scores,
    )


# Pipeline 3 : Collaborative (small => large if confidence < threshold)
def run_collaborative(images: List[str], gt_dict, config, vis_dir: str) -> EvalResult:
    threshold         = config["pipeline"]["confidence_threshold"]
    iou_threshold     = config["evaluation"]["iou_threshold"]
    routing_fn        = ROUTING_METHODS[config["pipeline"]["routing_method"]]
    image_dir         = config["data"]["image_dir"]
    ap50_list         = []
    confidence_scores = []
    small_calls, large_calls = 0, 0
    start             = time.time()

    for file_name in tqdm(images, desc="Collaborative"):
        image_path   = os.path.join(image_dir, file_name)
        small_pred   = run_small_model(image_path)
        small_calls += 1
        confidence   = routing_fn(small_pred)
        confidence_scores.append(confidence)

        if confidence >= threshold:
            pred = small_pred
        else:
            pred = run_large_model(image_path)
            large_calls += 1
            gt = gt_dict.get(file_name, [])
            save_escalated_visualization(image_path, small_pred, pred, gt,
                                         os.path.join(vis_dir, "escalated", file_name))

        gt = gt_dict.get(file_name, [])
        ap50_list.append(compute_image_ap50(pred, gt, iou_threshold))
        save_visualization(image_path, pred, os.path.join(vis_dir, "collaborative", file_name))

    return EvalResult(
        pipeline_name="Collaborative",
        total_time=round(time.time() - start, 4),
        mean_ap50=round(sum(ap50_list) / len(ap50_list), 4) if ap50_list else 0.0,
        small_model_calls=small_calls,
        large_model_calls=large_calls,
        confidence_scores=confidence_scores,
    )


# Summary
def print_summary(results: List[EvalResult]) -> None:
    W = 55
    print("\n" + "=" * W)
    print("SUMMARY".center(W))
    print("=" * W)

    for i, r in enumerate(results, 1):
        print(f"\n[P{i}] {r.pipeline_name}")
        print(f"  Total time  : {r.total_time:.4f}s")
        print(f"  Mean AP50   : {r.mean_ap50 * 100:.2f}%")
        print(f"  Small calls : {r.small_model_calls}")
        print(f"  Large calls : {r.large_model_calls}")

        if r.confidence_scores:
            if r.small_model_calls > 0 and r.large_model_calls > 0:
                total = r.small_model_calls
                rate  = r.large_model_calls / total * 100
                print(f"  Escalation  : {r.large_model_calls}/{total} ({rate:.1f}%)")
            print_confidence_histogram(r.confidence_scores)

    print("\n" + "=" * W)


# Main
def main(config_path: str = "config.yaml") -> None:
    config    = load_config(config_path)
    vis_dir   = os.path.join(os.path.dirname(os.path.abspath(config_path)), "data", "output")

    print("Loading GT data...")
    gt_dict   = load_gt(config["data"]["test_json"])
    image_dir = config["data"]["image_dir"]
    images    = sorted([
        f for f in os.listdir(image_dir)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    ])
    print(f"  images found in directory: {len(images)}")
    print(f"  visualizations saved to  : {vis_dir}")

    results = []

    print("\n[1/3] Running Small Only...")
    results.append(run_small_only(images, gt_dict, config, vis_dir))

    print("[2/3] Running Large Only...")
    results.append(run_large_only(images, gt_dict, config, vis_dir))

    print("[3/3] Running Collaborative...")
    results.append(run_collaborative(images, gt_dict, config, vis_dir))

    print_summary(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="X-ray Screening Pipeline")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
