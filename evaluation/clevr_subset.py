"""CLEVR subset loader for the image-mode eval.

Reads ground-truth scenes from `data/clevr_test_subset/scenes.json` (a 10-scene
slice of CLEVR_v1.0 val) and the matching PNGs from
`data/clevr_test_subset/images/`. Per scene we template a fixed shape of
ImageEvalCase entries from the ground truth:

    1× total count            "How many objects are there?"
    1× count by most-common color    "How many {color} objects are there?"
    1× count by most-common material "How many {material} objects are there?"
    1× positive existence     "Is there a {color} {shape}?"      (combo present)
    1× negative existence     "Is there a {color} {shape}?"      (combo absent)

→ 5 cases per scene × 10 scenes = 50 cases.

The 18GB CLEVR_v1.0 archive is NOT auto-downloaded — see
scripts/download_clevr.sh for the selective-unzip recipe. On any error
(missing files, malformed JSON, empty scenes), `iter_cases()` returns `[]`
and the harness reports the empty list gracefully.

CLEVR vocabularies, used verbatim when templating questions even though the
local CLIP attribute classifier may disagree:
    colors    gray, red, blue, green, brown, purple, cyan, yellow
    shapes    cube, sphere, cylinder
    sizes     small, large
    materials rubber, metal
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple

from evaluation.harness import ImageEvalCase

_REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR: Path = _REPO_ROOT / "data" / "clevr_test_subset"
SCENES_FILE: Path = DATA_DIR / "scenes.json"
IMAGES_DIR: Path = DATA_DIR / "images"

CLEVR_COLORS: Tuple[str, ...] = (
    "gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow",
)
CLEVR_SHAPES: Tuple[str, ...] = ("cube", "sphere", "cylinder")
CLEVR_SIZES: Tuple[str, ...] = ("small", "large")
CLEVR_MATERIALS: Tuple[str, ...] = ("rubber", "metal")


@dataclass(frozen=True)
class LoadStatus:
    """What `iter_cases` saw on disk. The CLI prints this when no cases load."""
    scenes_file_present: bool
    images_dir_present: bool
    scene_count: int
    case_count: int
    error: str | None = None


def _load_scenes() -> list[dict[str, Any]]:
    """Parse scenes.json. Returns the list under `scenes`, or empty on error."""
    if not SCENES_FILE.exists():
        return []
    try:
        with SCENES_FILE.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []
    if isinstance(payload, dict) and "scenes" in payload:
        scenes = payload["scenes"]
    elif isinstance(payload, list):
        scenes = payload
    else:
        return []
    return [s for s in scenes if isinstance(s, dict) and "objects" in s]


def _image_path_for(scene: dict[str, Any]) -> Path | None:
    """Resolve the PNG for `scene` under IMAGES_DIR. None if not on disk."""
    filename = scene.get("image_filename")
    if not filename:
        idx = scene.get("image_index")
        if idx is None:
            return None
        filename = f"CLEVR_val_{int(idx):06d}.png"
    candidate = IMAGES_DIR / filename
    return candidate if candidate.exists() else None


def _most_common(items: list[str]) -> str | None:
    if not items:
        return None
    counter = Counter(items)
    # most_common preserves first-seen on ties; sorting the keys first gives a
    # deterministic tie-break.
    sorted_keys = sorted(counter.keys())
    return max(sorted_keys, key=lambda k: counter[k])


def _missing_combo(scene_objects: list[dict[str, Any]]) -> tuple[str, str]:
    """Return a (color, shape) pair NOT present in the scene.

    Prefers a color absent from the scene paired with the alphabetically-first
    CLEVR shape (deterministic). If every CLEVR color appears, falls back to
    the first absent (color, shape) combination.
    """
    present_colors = {obj.get("color") for obj in scene_objects}
    present_pairs = {(obj.get("color"), obj.get("shape")) for obj in scene_objects}
    sorted_shape = sorted(CLEVR_SHAPES)[0]
    for color in CLEVR_COLORS:
        if color not in present_colors:
            return color, sorted_shape
    for color in CLEVR_COLORS:
        for shape in CLEVR_SHAPES:
            if (color, shape) not in present_pairs:
                return color, shape
    # Pathological: every CLEVR combo is present (impossible with <24 objects).
    return CLEVR_COLORS[0], CLEVR_SHAPES[0]


def _cases_for_scene(scene_index: int, scene: dict[str, Any], image_path: Path) -> list[ImageEvalCase]:
    """Produce 1 total-count + 2 attribute-counts + 1 positive + 1 negative existence."""
    objects = scene["objects"]
    if not objects:
        return []

    cases: list[ImageEvalCase] = []
    img_str = str(image_path)

    cases.append(
        ImageEvalCase(
            case_id=f"CLEVR_{scene_index}_count_total",
            image_path=img_str,
            question="How many objects are there?",
            expected=len(objects),
            qtype="count",
            ground_truth_objects=objects,
            notes="Total object count from CLEVR ground truth.",
        )
    )

    colors_in_scene = [obj["color"] for obj in objects if "color" in obj]
    top_color = _most_common(colors_in_scene)
    if top_color is not None:
        cases.append(
            ImageEvalCase(
                case_id=f"CLEVR_{scene_index}_count_{top_color}",
                image_path=img_str,
                question=f"How many {top_color} objects are there?",
                expected=colors_in_scene.count(top_color),
                qtype="count",
                ground_truth_objects=objects,
                notes=f"Count by most-common color ({top_color}) in the scene.",
            )
        )

    materials_in_scene = [obj["material"] for obj in objects if "material" in obj]
    top_material = _most_common(materials_in_scene)
    if top_material is not None:
        cases.append(
            ImageEvalCase(
                case_id=f"CLEVR_{scene_index}_count_{top_material}",
                image_path=img_str,
                question=f"How many {top_material} objects are there?",
                expected=materials_in_scene.count(top_material),
                qtype="count",
                ground_truth_objects=objects,
                notes=f"Count by most-common material ({top_material}) in the scene.",
            )
        )

    pairs = Counter(
        (obj.get("color"), obj.get("shape"))
        for obj in objects
        if obj.get("color") and obj.get("shape")
    )
    if pairs:
        sorted_pairs = sorted(pairs.keys())
        pos_color, pos_shape = max(sorted_pairs, key=lambda p: pairs[p])
        cases.append(
            ImageEvalCase(
                case_id=f"CLEVR_{scene_index}_exists_{pos_color}_{pos_shape}",
                image_path=img_str,
                question=f"Is there a {pos_color} {pos_shape}?",
                expected=True,
                qtype="boolean",
                ground_truth_objects=objects,
                notes=f"Most-common color+shape combo present: {pos_color} {pos_shape}.",
            )
        )

    neg_color, neg_shape = _missing_combo(objects)
    cases.append(
        ImageEvalCase(
            case_id=f"CLEVR_{scene_index}_not_exists_{neg_color}_{neg_shape}",
            image_path=img_str,
            question=f"Is there a {neg_color} {neg_shape}?",
            expected=False,
            qtype="boolean",
            ground_truth_objects=objects,
            notes=f"Absent combo from CLEVR vocabulary: {neg_color} {neg_shape}.",
        )
    )

    return cases


def iter_cases() -> list[ImageEvalCase]:
    """Build the full ImageEvalCase list from disk.

    Returns `[]` on any I/O or parsing error rather than raising — the harness
    is responsible for handling the empty case (e.g., "no CLEVR data found,
    skipping suite"). This matches the existing iter_cases contract used by
    the CLI.
    """
    scenes = _load_scenes()
    if not scenes:
        return []
    cases: list[ImageEvalCase] = []
    for idx, scene in enumerate(scenes):
        img_path = _image_path_for(scene)
        if img_path is None:
            continue
        cases.extend(_cases_for_scene(idx, scene, img_path))
    return cases


def ensure_clevr_subset() -> tuple[list[ImageEvalCase], dict[str, Any]]:
    """Returns (cases, status_dict). Backwards-compatible with the CLI."""
    scenes = _load_scenes()
    status = LoadStatus(
        scenes_file_present=SCENES_FILE.exists(),
        images_dir_present=IMAGES_DIR.exists(),
        scene_count=len(scenes),
        case_count=0,
    )
    if not scenes:
        return [], {
            "scenes_file_present": status.scenes_file_present,
            "images_dir_present": status.images_dir_present,
            "scene_count": 0,
            "case_count": 0,
            "hint": "Run scripts/download_clevr.sh for the extraction recipe.",
        }
    cases = iter_cases()
    return cases, {
        "scenes_file_present": status.scenes_file_present,
        "images_dir_present": status.images_dir_present,
        "scene_count": len(scenes),
        "case_count": len(cases),
    }
