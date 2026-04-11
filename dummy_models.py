"""
Dummy small / large model.
"""

import random
from dataclasses import dataclass
from enum import IntEnum
from typing import List


class Category(IntEnum):
    MOBILE_PHONE = 1
    ORANGE_LIQUID = 2
    POWER_BANK_WITHOUT_BATT = 3
    LAPTOP = 4
    GREEN_LIQUID = 5
    POWER_BANK_WITH_BATT = 6
    TABLET = 7
    BLUE_LIQUID = 8
    CYLINDRICAL_ORANGE_LIQUID = 9
    NON_METAL_LIGHTER = 10
    UMBRELLA = 11
    CYLINDRICAL_GREEN_LIQUID = 12


@dataclass
class Prediction:
    image_id: int
    file_name: str
    boxes: List[List[float]]
    labels: List[int]
    scores: List[float]
    model_confidence: float


def run_small_model(file_name: str, image_id: int) -> Prediction:
    """Dummy small model"""
    n = random.randint(1, 3)
    return Prediction(
        image_id=image_id,
        file_name=file_name,
        boxes=[[round(random.uniform(0, 400), 1), round(random.uniform(0, 400), 1),
                round(random.uniform(30, 150), 1), round(random.uniform(30, 150), 1)]
               for _ in range(n)],
        labels=[random.choice(list(Category)) for _ in range(n)],
        scores=[round(random.uniform(0.3, 0.95), 4) for _ in range(n)],
        model_confidence=round(random.uniform(0.0, 1.0), 4),
    )


def run_large_model(file_name: str, image_id: int) -> Prediction:
    """Dummy large model"""
    n = random.randint(1, 5)
    return Prediction(
        image_id=image_id,
        file_name=file_name,
        boxes=[[round(random.uniform(0, 400), 1), round(random.uniform(0, 400), 1),
                round(random.uniform(30, 150), 1), round(random.uniform(30, 150), 1)]
               for _ in range(n)],
        labels=[random.choice(list(Category)) for _ in range(n)],
        scores=[round(random.uniform(0.5, 0.99), 4) for _ in range(n)],
        model_confidence=round(random.uniform(0.6, 1.0), 4),
    )
