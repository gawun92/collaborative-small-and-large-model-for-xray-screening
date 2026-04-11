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

import yaml

from data_loader import load_gt, get_images
from evaluator import EvalResult, compute_image_ap50, print_confidence_histogram
from dummy_models import run_small_model, run_large_model


# Config
def load_config(path: str) -> dict:
    with open(path) as f:
        config = yaml.safe_load(f)
    config_dir = os.path.dirname(os.path.abspath(path))
    config["data"]["test_json"] = os.path.join(config_dir, config["data"]["test_json"])
    config["data"]["image_dir"] = os.path.join(config_dir, config["data"]["image_dir"])
    return config


# Pipeline 1 : Small model only
def run_small_only(images, gt_dict, config) -> EvalResult:
    iou_threshold = config["evaluation"]["iou_threshold"]
    ap50_list = []
    confidence_scores = []
    start = time.time()

    for img in images:
        file_name = img["file_name"]
        pred = run_small_model(file_name, img["id"])
        gt = gt_dict.get(file_name, [])
        ap50_list.append(compute_image_ap50(pred, gt, iou_threshold))
        confidence_scores.append(pred.model_confidence)

    return EvalResult(
        pipeline_name="Small Only",
        total_time=round(time.time() - start, 4),
        mean_ap50=round(sum(ap50_list) / len(ap50_list), 4) if ap50_list else 0.0,
        small_model_calls=len(images),
        confidence_scores=confidence_scores,
    )


# Pipeline 2 : Large model only
def run_large_only(images, gt_dict, config) -> EvalResult:
    iou_threshold = config["evaluation"]["iou_threshold"]
    ap50_list = []
    confidence_scores = []
    start = time.time()

    for img in images:
        file_name = img["file_name"]
        pred = run_large_model(file_name, img["id"])
        gt = gt_dict.get(file_name, [])
        ap50_list.append(compute_image_ap50(pred, gt, iou_threshold))
        confidence_scores.append(pred.model_confidence)

    return EvalResult(
        pipeline_name="Large Only",
        total_time=round(time.time() - start, 4),
        mean_ap50=round(sum(ap50_list) / len(ap50_list), 4) if ap50_list else 0.0,
        large_model_calls=len(images),
        confidence_scores=confidence_scores,
    )


# Pipeline 3 : Collaborative (small => large if confidence < threshold)
def run_collaborative(images, gt_dict, config) -> EvalResult:
    threshold = config["pipeline"]["confidence_threshold"]
    iou_threshold = config["evaluation"]["iou_threshold"]
    ap50_list = []
    confidence_scores = []
    small_calls, large_calls = 0, 0
    start = time.time()

    for img in images:
        file_name = img["file_name"]
        small_pred = run_small_model(file_name, img["id"])
        small_calls += 1
        confidence_scores.append(small_pred.model_confidence)

        if small_pred.model_confidence >= threshold:
            pred = small_pred
        else:
            pred = run_large_model(file_name, img["id"])
            large_calls += 1

        gt = gt_dict.get(file_name, [])
        ap50_list.append(compute_image_ap50(pred, gt, iou_threshold))

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
            # Show escalation only for collaborative pipeline (both small and large are used)
            if r.small_model_calls > 0 and r.large_model_calls > 0:
                total = r.small_model_calls
                rate = r.large_model_calls / total * 100
                print(f"  Escalation  : {r.large_model_calls}/{total} ({rate:.1f}%)")
            print_confidence_histogram(r.confidence_scores)

    print("\n" + "=" * W)


# Main
def main(config_path: str = "config.yaml") -> None:
    config = load_config(config_path)

    print("Loading GT data...")
    gt_dict = load_gt(config["data"]["test_json"])
    images = get_images()

    results = []

    print("\n[1/3] Running Small Only...")
    results.append(run_small_only(images, gt_dict, config))

    print("[2/3] Running Large Only...")
    results.append(run_large_only(images, gt_dict, config))

    print("[3/3] Running Collaborative...")
    results.append(run_collaborative(images, gt_dict, config))

    print_summary(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="X-ray Screening Pipeline")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
