"""CLIP-based attribute classification over a fixed per-family vocabulary.

For each attribute family (color, size, material, shape) we score the box
crop against a SMALL ENSEMBLE of CLIP prompts per candidate value and pick the
argmax. If the max softmax probability falls below ATTRIBUTE_THRESHOLD we omit
the attribute entirely — asserting "unknown" to Prolog would pollute downstream
queries.

Why an ensemble: CLEVR images are Blender-rendered 3D shapes on a flat
backdrop. CLIP's prior over "a photo of a metal object" is dominated by
real-world photos of pots/pans/tools that look nothing like a CLEVR sphere.
Phrasing prompts as "a 3D rendered ..." / "a Blender render of ..." pulls the
text encoder into the right corner of embedding space. Averaging 2–3 prompts
per value (the standard CLIP zero-shot trick — OpenAI's repo ships ~80 templates
for ImageNet) further denoises the per-template variance. We average **logits**
across templates per value, then softmax over values, which is equivalent to
averaging the per-template text embeddings (same math).

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

# Generic ensemble templates per family. {value} interpolates the candidate.
# Kept small (2–3) so a single CLIP forward pass per family stays cheap.
_GENERIC_TEMPLATES: dict[str, Tuple[str, ...]] = {
    "color": (
        "a 3D rendered {value} geometric shape",
        "a Blender render of a {value} colored object",
        "a CGI scene of a solid {value} shape",
    ),
    "size": (
        "a {value} 3D rendered geometric shape",
        "a {value} 3D solid object on a flat background",
    ),
    "shape": (
        "a 3D rendered {value}",
        "a CGI rendering of a {value}",
        "a Blender-rendered {value} shape",
    ),
    "material": (
        # Used only as a fallback when no per-value override is registered.
        "a 3D rendered {value} object",
    ),
}

# Per-(family, value) overrides where the visual cue is highly specific.
# CLEVR uses only two materials (metal, rubber) and they are visually
# distinguished by specular highlight, not by surface texture word.
_PER_VALUE_OVERRIDES: dict[tuple[str, str], Tuple[str, ...]] = {
    ("material", "metal"): (
        "a shiny metallic 3D rendered shape",
        "a specular reflective metal object",
        "a glossy chrome 3D rendered geometric shape",
    ),
    ("material", "rubber"): (
        "a matte rubber 3D rendered shape",
        "a diffuse non-shiny rubber object",
        "a dull rubbery 3D geometric shape",
    ),
    # Local-vocab materials that don't appear in CLEVR but are in
    # ATTRIBUTE_VOCAB. Phrased as 3D renders so the ensemble doesn't collapse
    # on CLEVR crops simply because the generic "a photo of wood" prompt drags
    # in real-world wood-furniture priors.
    ("material", "wood"): (
        "a wooden 3D rendered object",
        "a wood-grain textured 3D shape",
    ),
    ("material", "plastic"): (
        "a smooth plastic 3D rendered object",
        "a plastic geometric shape with a hard surface",
    ),
    ("material", "glass"): (
        "a transparent glass 3D rendered object",
        "a clear glass geometric shape",
    ),
    ("material", "ceramic"): (
        "a glazed ceramic 3D rendered object",
        "a porcelain geometric shape",
    ),
}


def _prompts_for(family: str, value: str) -> Tuple[str, ...]:
    override = _PER_VALUE_OVERRIDES.get((family, value))
    if override is not None:
        return override
    templates = _GENERIC_TEMPLATES.get(family) or ("a 3D rendered {value} object",)
    return tuple(t.format(value=value) for t in templates)


def classify(
    crop: Image.Image,
    *,
    vocab: Optional[Mapping[str, Tuple[str, ...]]] = None,
    threshold: float = ATTRIBUTE_THRESHOLD,
) -> Tuple[dict[str, str], dict[str, float]]:
    """Score `crop` against each attribute family; return (attrs, confidences).

    For each family, we concatenate the per-value prompt ensembles into ONE
    CLIP forward pass, average the resulting logits per value, then softmax
    across values. Low-confidence attributes (max prob < threshold) are
    dropped; the caller sees an empty entry for that family rather than a
    guess.
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

        # Build the flat prompt list plus the per-value group boundaries so
        # we can collapse logits back to one score per value after the forward.
        all_prompts: list[str] = []
        group_sizes: list[int] = []
        for v in values:
            ps = _prompts_for(family, v)
            all_prompts.extend(ps)
            group_sizes.append(len(ps))

        inputs = processor(
            text=all_prompts, images=crop, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = model(**inputs)
        logits = outputs.logits_per_image[0].detach().cpu()  # shape (sum(group_sizes),)

        # Mean-pool logits per value group (= averaging text embeddings).
        per_value = torch.empty(len(values))
        offset = 0
        for i, count in enumerate(group_sizes):
            per_value[i] = logits[offset : offset + count].mean()
            offset += count

        probs = per_value.softmax(dim=0)
        best_idx = int(torch.argmax(probs).item())
        best_prob = float(probs[best_idx].item())
        if best_prob < threshold:
            continue
        attrs[family] = values[best_idx]
        confidences[family] = best_prob

    return attrs, confidences
