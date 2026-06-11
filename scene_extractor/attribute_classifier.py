"""CLIP-based attribute classification over a fixed per-family vocabulary.

For each attribute family (color, size, material, shape) we score the box
crop against prompts of the form `"a photo of a {value} object"` and pick the
argmax. If the max softmax probability falls below ATTRIBUTE_THRESHOLD we omit
the attribute entirely — asserting "unknown" to Prolog would pollute downstream
queries.

The classifier is stateless aside from the cached CLIP model; it accepts
overrides for the attribute vocabulary so the eval harness can sweep without
forking this file.
"""

from __future__ import annotations

from typing import Mapping, Optional, Tuple

import torch
from PIL import Image

from scene_extractor.config import ATTRIBUTE_THRESHOLD, ATTRIBUTE_VOCAB
from scene_extractor.models import get_clip, get_device

_PROMPT_TEMPLATES: dict[str, str] = {
    "color": "a photo of a {value} object",
    "size": "a photo of a {value} object",
    "material": "a photo of a {value} object",
    "shape": "a photo of a {value} shape",
}


def _prompt_for(family: str, value: str) -> str:
    template = _PROMPT_TEMPLATES.get(family, "a photo of a {value} object")
    return template.format(value=value)


def classify(
    crop: Image.Image,
    *,
    vocab: Optional[Mapping[str, Tuple[str, ...]]] = None,
    threshold: float = ATTRIBUTE_THRESHOLD,
) -> Tuple[dict[str, str], dict[str, float]]:
    """Score `crop` against each attribute family; return (attrs, confidences).

    Low-confidence attributes (max prob < threshold) are dropped; the caller
    sees an empty entry for that family rather than a guess.
    """
    vocabulary = vocab if vocab is not None else ATTRIBUTE_VOCAB
    if crop.mode != "RGB":
        crop = crop.convert("RGB")

    processor, model = get_clip()
    device = get_device()

    attrs: dict[str, str] = {}
    confidences: dict[str, float] = {}

    for family, values in vocabulary.items():
        if not values:
            continue
        prompts = [_prompt_for(family, v) for v in values]
        inputs = processor(
            text=prompts, images=crop, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = model(**inputs)
        logits = outputs.logits_per_image
        probs = logits.softmax(dim=-1)[0].cpu()
        best_idx = int(torch.argmax(probs).item())
        best_prob = float(probs[best_idx].item())
        if best_prob < threshold:
            continue
        attrs[family] = values[best_idx]
        confidences[family] = best_prob

    return attrs, confidences
