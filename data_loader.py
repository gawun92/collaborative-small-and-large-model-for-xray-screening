"""
data_loader.py
read test.json and create a dict
"""

import json
from typing import Dict, List

# module-level static cache
_gt_dict: Dict[str, List[dict]] = {}
_images: List[dict] = []


def load_gt(json_path: str) -> Dict[str, List[dict]]:
    global _gt_dict, _images

    with open(json_path) as f:
        data = json.load(f)

    _images = data.get("images", [])
    annotations = data.get("annotations", [])
    id_to_filename = {img["id"]: img["file_name"] for img in _images}

    _gt_dict = {}
    for ann in annotations:
        file_name = id_to_filename[ann["image_id"]]
        if file_name not in _gt_dict:
            _gt_dict[file_name] = []
        _gt_dict[file_name].append({
            "category_id": ann["category_id"],
            "bbox": ann["bbox"],  # [x, y, w, h]
        })

    print(f"  ground truth dict built: {len(_images)} images, {len(annotations)} annotations")

    return _gt_dict


def get_images() -> List[dict]:
    return _images


def get_gt_dict() -> Dict[str, List[dict]]:
    return _gt_dict
