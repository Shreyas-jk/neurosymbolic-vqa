"""Vision model loading with device selection + process-wide caching.

Module-level singletons via @functools.lru_cache(maxsize=1) on the loaders, so
the heavy weights load once and stay resident for the rest of the process —
critical for the Gradio demo and the eval harness where many inferences share
one process.

Device selection: prefer MPS on Apple Silicon, fall back to CPU on any failure.
The fall-back happens once at first use, surfaced via a warning printed to
stderr — silent CPU fallback would tank inference latency without explanation.
"""

from __future__ import annotations

import functools
import sys
from typing import Optional, Tuple

import torch
from transformers import (
    CLIPModel,
    CLIPProcessor,
    OwlViTForObjectDetection,
    OwlViTProcessor,
)

from scene_extractor.config import CLIP_MODEL_ID, DETECTOR_MODEL_ID


def _select_device(preferred: Optional[str] = None) -> torch.device:
    """Pick the inference device. Preference: explicit > MPS > CPU.

    Set NSVQA_FORCE_CPU=1 to disable MPS (useful when MPS crashes on a layer
    the model uses — happens with rarer ops on older macOS).
    """
    import os

    if preferred:
        return torch.device(preferred)
    if os.environ.get("NSVQA_FORCE_CPU"):
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


_DEVICE: torch.device = _select_device()


def get_device() -> torch.device:
    return _DEVICE


@functools.lru_cache(maxsize=1)
def get_detector(
    model_id: str = DETECTOR_MODEL_ID,
) -> Tuple[OwlViTProcessor, OwlViTForObjectDetection]:
    """Load OWL-ViT processor + model, move model to the inference device.

    Cached for the process lifetime. The model_id is part of the cache key so
    swapping the detector at test time still produces a fresh load.
    """
    processor = OwlViTProcessor.from_pretrained(model_id)
    model = OwlViTForObjectDetection.from_pretrained(model_id)
    model.eval()
    try:
        model = model.to(_DEVICE)
    except (RuntimeError, NotImplementedError) as exc:
        print(
            f"[scene_extractor.models] device={_DEVICE} failed for detector ({exc}); falling back to CPU",
            file=sys.stderr,
        )
        model = model.to("cpu")
    return processor, model


@functools.lru_cache(maxsize=1)
def get_clip(model_id: str = CLIP_MODEL_ID) -> Tuple[CLIPProcessor, CLIPModel]:
    """Load CLIP processor + model on the inference device.

    Same caching strategy as `get_detector`. Both are needed per inference, but
    they share no weights — load them independently so failures in one don't
    block the other from being used in isolation (e.g., unit testing CLIP
    classification alone).
    """
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id)
    model.eval()
    try:
        model = model.to(_DEVICE)
    except (RuntimeError, NotImplementedError) as exc:
        print(
            f"[scene_extractor.models] device={_DEVICE} failed for CLIP ({exc}); falling back to CPU",
            file=sys.stderr,
        )
        model = model.to("cpu")
    return processor, model


def clear_caches() -> None:
    """Drop cached model instances. Used by tests that need fresh loads."""
    get_detector.cache_clear()
    get_clip.cache_clear()
