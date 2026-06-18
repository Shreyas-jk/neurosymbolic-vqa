#!/usr/bin/env python3
"""Precompute SceneGraphs for 5 CLEVR images and copy the PNGs alongside.

Output:
  demo/cached_scenes/CLEVR_val_NNNNNN.json  — pydantic-serialized SceneGraph
  demo/cached_scenes/CLEVR_val_NNNNNN.png   — copy of the source CLEVR image

The Gradio demo loads these on HF Spaces (where running OWL-ViT + CLIP live
is too slow on the free CPU tier). KB generation, NL→Prolog translation,
Prolog execution, and verbalization continue to run live against the cached
SceneGraph.

Run from the repo root with the full vision stack already installed:
  .venv/bin/python scripts/precompute_clevr_scenes.py
"""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scene_extractor import SceneExtractor  # noqa: E402

SOURCE_IMAGES = REPO / "data" / "clevr_test_subset" / "images"
CACHE_DIR = REPO / "demo" / "cached_scenes"
IMAGE_INDICES = range(5)  # CLEVR_val_000000 through CLEVR_val_000004


def main() -> int:
    if not SOURCE_IMAGES.exists():
        print(f"ERROR: source images dir not found: {SOURCE_IMAGES}", file=sys.stderr)
        return 2
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    extractor = SceneExtractor()
    total_t0 = time.perf_counter()
    summary: list[tuple[str, int, int, float]] = []

    for i in IMAGE_INDICES:
        name = f"CLEVR_val_{i:06d}"
        src_png = SOURCE_IMAGES / f"{name}.png"
        if not src_png.exists():
            print(f"SKIP {name}: source PNG missing")
            continue

        t0 = time.perf_counter()
        scene = extractor.extract(str(src_png))
        elapsed_s = time.perf_counter() - t0

        json_path = CACHE_DIR / f"{name}.json"
        png_path = CACHE_DIR / f"{name}.png"
        json_path.write_text(scene.model_dump_json(indent=2))
        shutil.copy2(src_png, png_path)

        n_obj = len(scene.objects)
        n_rel = len(scene.relations)
        summary.append((name, n_obj, n_rel, elapsed_s))
        print(f"{name}: {n_obj} objects, {n_rel} relations, {elapsed_s:.1f}s")

    total = time.perf_counter() - total_t0
    print()
    print(f"Cached {len(summary)} scene(s) to {CACHE_DIR.relative_to(REPO)} "
          f"in {total:.1f}s")

    zero = [s for s in summary if s[1] == 0]
    if zero:
        print(f"\nWARNING: {len(zero)} scene(s) detected 0 objects: "
              f"{[s[0] for s in zero]}")
        print("These will surface as empty SceneGraphs in the demo. Consider "
              "swapping in a different CLEVR image.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
