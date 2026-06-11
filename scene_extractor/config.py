"""Configuration for the vision pipeline.

All vocabularies + thresholds live here so the eval harness and unit tests can
override them without touching extractor.py. CLEVR-friendly defaults; the
extractor accepts override lists at construction time.
"""

from __future__ import annotations

from typing import Final

DETECTOR_MODEL_ID: Final[str] = "google/owlvit-base-patch32"
CLIP_MODEL_ID: Final[str] = "openai/clip-vit-base-patch32"

OBJECT_VOCAB: Final[tuple[str, ...]] = (
    "cube",
    "sphere",
    "cylinder",
    "cone",
    "cup",
    "can",
    "box",
    "ball",
    "person",
    "car",
    "dog",
    "cat",
    "chair",
    "table",
    "bottle",
)

ATTRIBUTE_VOCAB: Final[dict[str, tuple[str, ...]]] = {
    "color": (
        "red",
        "blue",
        "green",
        "yellow",
        "purple",
        "orange",
        "brown",
        "gray",
        "white",
        "black",
    ),
    "size": ("small", "medium", "large"),
    "material": ("metal", "rubber", "wood", "plastic", "glass", "ceramic"),
    "shape": ("cube", "sphere", "cylinder", "cone", "pyramid", "flat"),
}

DETECTION_THRESHOLD: Final[float] = 0.1
NMS_IOU_THRESHOLD: Final[float] = 0.5
ATTRIBUTE_THRESHOLD: Final[float] = 0.35

SPATIAL_CENTER_MARGIN: Final[float] = 0.05
SPATIAL_GAP_MARGIN: Final[float] = 0.05
INSIDE_AREA_RATIO: Final[float] = 0.9
ON_TOP_OF_VERTICAL_TOLERANCE: Final[float] = 0.05
NEXT_TO_OVERLAP_FRACTION: Final[float] = 0.5
ON_TOP_OF_OVERLAP_FRACTION: Final[float] = 0.5
IN_FRONT_OF_AREA_RATIO: Final[float] = 1.5
