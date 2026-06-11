#!/usr/bin/env bash
# Pre-warm the HuggingFace cache with both vision models.
#
# Run once after `pip install -r requirements.txt` so first inference doesn't
# block on a 1GB download. The HF transformers cache (default ~/.cache/huggingface)
# persists across runs, so this is idempotent.
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"

if [ ! -x "$PYTHON" ]; then
  echo "Python not found at $PYTHON. Set PYTHON=<path> or run from project root."
  exit 1
fi

"$PYTHON" - <<'PY'
from transformers import (
    OwlViTProcessor,
    OwlViTForObjectDetection,
    CLIPProcessor,
    CLIPModel,
)

print("Loading OWL-ViT (google/owlvit-base-patch32)...")
OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

print("Loading CLIP (openai/clip-vit-base-patch32)...")
CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

print("Done. Models are now cached at ~/.cache/huggingface/hub/.")
PY
