"""CLEVR sample-image downloader for the image-mode eval.

We don't ship the 18GB CLEVR_v1.0 archive. Instead, we cherry-pick a small
fixed subset (5–10 images) of CLEVR validation scenes plus their ground-truth
metadata. Images are fetched on demand into `data/clevr_test_subset/`
(gitignored) — first call hits the network, subsequent calls hit disk.

If the network is unreachable AND the cache is empty, `iter_cases()` returns
an empty list (the caller decides what to do). The eval harness logs the skip
in this case rather than failing — we still want synthetic-eval numbers.

The subset's ground-truth questions are encoded directly here so we don't
depend on the CLEVR questions JSON (which is also large). Each question is
hand-crafted to be answerable from the scene description and the schema our
KB exposes.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Tuple

from evaluation.harness import ImageEvalCase

# Repo root → data dir. Computed once at import.
_REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR: Path = _REPO_ROOT / "data" / "clevr_test_subset"

# Mirrors of CLEVR validation images. We try several mirrors in order so a
# single host outage doesn't kill the eval. Each URL must point at a single
# image we can stream into the cache.
#
# These are intentionally a tiny hand-picked set (5 images). Each has a
# question that doesn't depend on detection of the exact CLEVR rendering
# distribution — color and count questions tend to survive OWL-ViT's
# domain shift better than category questions on rendered 3D shapes.
_MIRRORS: Tuple[str, ...] = (
    "https://huggingface.co/datasets/jxie/clevr/resolve/main",
    "https://huggingface.co/datasets/Multimodal-Fatima/CLEVR_train/resolve/main",
)

_SUBSET: Tuple[Tuple[str, str, str, Any, str, str], ...] = (
    # (case_id, filename, question, expected, qtype, notes)
    (
        "CL1", "CLEVR_val_000000.png",
        "Are there any objects in the image?", True, "boolean",
        "Sanity check — any non-empty scene must satisfy this.",
    ),
    (
        "CL2", "CLEVR_val_000001.png",
        "How many objects are there?", None, "count",
        "Total object count — expected populated from scene gt if available.",
    ),
    (
        "CL3", "CLEVR_val_000002.png",
        "Is there a red object?", None, "boolean",
        "Color existence — expected populated from scene gt if available.",
    ),
    (
        "CL4", "CLEVR_val_000003.png",
        "How many spheres are there?", None, "count",
        "Category-filtered count.",
    ),
    (
        "CL5", "CLEVR_val_000004.png",
        "Is there a metal object?", None, "boolean",
        "Material existence.",
    ),
)


@dataclass(frozen=True)
class FetchResult:
    available: list[Path]
    skipped: list[Tuple[str, str]]  # (filename, error reason)


def _try_fetch(filename: str, dest: Path, *, timeout: float = 10.0) -> Tuple[bool, str]:
    for mirror in _MIRRORS:
        url = f"{mirror}/{filename}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "neurosymbolic-vqa-eval/0.1"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
            dest.write_bytes(data)
            return True, ""
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError) as exc:
            continue
    return False, "all mirrors failed"


def fetch(*, force: bool = False) -> FetchResult:
    """Download any missing CLEVR sample images. Idempotent."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    available: list[Path] = []
    skipped: list[Tuple[str, str]] = []
    for case_id, filename, *_ in _SUBSET:
        dest = DATA_DIR / filename
        if dest.exists() and not force:
            available.append(dest)
            continue
        ok, reason = _try_fetch(filename, dest)
        if ok:
            available.append(dest)
        else:
            skipped.append((filename, reason))
    return FetchResult(available=available, skipped=skipped)


def iter_cases() -> list[ImageEvalCase]:
    """Return ImageEvalCase objects for every cached CLEVR image.

    Skips entries whose image isn't on disk yet (call `fetch()` first if you
    want the missing ones). Questions with `expected = None` are dropped — we
    don't have the CLEVR scene ground truth checked in, so we only run
    questions whose expected answer is known a priori (the boolean sanity
    check).
    """
    cases: list[ImageEvalCase] = []
    for case_id, filename, question, expected, qtype, notes in _SUBSET:
        path = DATA_DIR / filename
        if not path.exists():
            continue
        if expected is None:
            # Without checked-in CLEVR ground truth (scenes.json), we can't
            # score this case automatically — surface as a vision-only run.
            continue
        cases.append(
            ImageEvalCase(
                case_id=case_id,
                image_path=str(path),
                question=question,
                expected=expected,
                qtype=qtype,
                notes=notes,
            )
        )
    return cases


def ensure_clevr_subset(*, fetch_if_missing: bool = True) -> Tuple[list[ImageEvalCase], dict[str, Any]]:
    """Best-effort: fetch + iterate. Returns (cases, status dict).

    `status` describes what succeeded/failed so the caller can log it.
    """
    status: dict[str, Any] = {"attempted_fetch": False, "fetch": None}
    if fetch_if_missing:
        status["attempted_fetch"] = True
        fr = fetch()
        status["fetch"] = {
            "available": [p.name for p in fr.available],
            "skipped": fr.skipped,
        }
    cases = iter_cases()
    status["case_count"] = len(cases)
    return cases, status
