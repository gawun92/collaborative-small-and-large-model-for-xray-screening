"""
pipeline_with_various_routings.py
Runs the collaborative pipeline with every routing method in ROUTING_METHODS
and prints a unified summary for comparison.

Usage:
    python pipeline_with_various_routings.py --config config.yaml
"""

import argparse
import os
import sys
import time
from typing import List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml
from tqdm import tqdm

from data_loader import load_gt
from evaluator import EvalResult, compute_image_ap50, print_confidence_histogram
from models import run_small_model, run_large_model
from routing import ROUTING_METHODS

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


# Config
def load_config(path: str) -> dict:
    with open(path) as f:
        config = yaml.safe_load(f)
    config_dir = os.path.dirname(os.path.abspath(path))
    config["data"]["test_json"] = os.path.join(config_dir, config["data"]["test_json"])
    config["data"]["image_dir"] = os.path.join(config_dir, config["data"]["image_dir"])
    return config


# Collaborative pipeline with a specific routing method
def run_collaborative(images: List[str], gt_dict, config, routing_name: str) -> EvalResult:
    threshold         = config["pipeline"]["confidence_threshold"]
    iou_threshold     = config["evaluation"]["iou_threshold"]
    routing_fn        = ROUTING_METHODS[routing_name]
    image_dir         = config["data"]["image_dir"]
    ap50_list         = []
    confidence_scores = []
    small_calls, large_calls = 0, 0
    start             = time.time()

    for file_name in tqdm(images, desc=f"Collaborative [{routing_name}]"):
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
        ap50_list.append(compute_image_ap50(pred, gt, iou_threshold))

    return EvalResult(
        pipeline_name=f"Collaborative [{routing_name}]",
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
        print(f"\n[{i}] {r.pipeline_name}")
        print(f"  Total time  : {r.total_time:.4f}s")
        print(f"  Mean AP50   : {r.mean_ap50 * 100:.2f}%")
        print(f"  Small calls : {r.small_model_calls}")
        print(f"  Large calls : {r.large_model_calls}")

        if r.confidence_scores:
            total = r.small_model_calls
            rate  = r.large_model_calls / total * 100
            print(f"  Escalation  : {r.large_model_calls}/{total} ({rate:.1f}%)")
            print_confidence_histogram(r.confidence_scores)

    print("\n" + "=" * W)


# Main
def main(config_path: str = "config.yaml") -> None:
    config = load_config(config_path)

    print("Loading GT data...")
    gt_dict   = load_gt(config["data"]["test_json"])
    image_dir = config["data"]["image_dir"]
    images    = sorted([
        f for f in os.listdir(image_dir)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    ])
    print(f"  images found in directory: {len(images)}")
    print(f"  routing methods          : {list(ROUTING_METHODS.keys())}")
    print(f"  confidence threshold     : {config['pipeline']['confidence_threshold']}")

    results = []

    for i, routing_name in enumerate(ROUTING_METHODS.keys(), 1):
        print(f"\n[{i}/{len(ROUTING_METHODS)}] Running Collaborative [{routing_name}]...")
        results.append(run_collaborative(images, gt_dict, config, routing_name))

    print_summary(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collaborative Pipeline with Various Routing Methods")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
