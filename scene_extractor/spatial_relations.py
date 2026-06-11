"""Pure-geometric spatial relations over normalized bounding boxes.

All inputs and outputs are SceneObject / BoundingBox values; no model calls,
no I/O. Tested without the heavy vision stack — runs in CI.

Conventions:
- Image coordinates are normalized to [0, 1] with y=0 at top.
- Only canonical directions are asserted (left_of, above, inside, on_top_of,
  next_to). Inverses (right_of, below) are derived by the Prolog KB rules.
- next_to is symmetric — emitted only once per pair (sorted by id) so the KB
  doesn't get duplicates.
- in_front_of is a coarse same-category-area heuristic; cross-category
  comparisons return nothing (depth isn't recoverable from 2D bboxes).
- Ambiguous cases (centers inside the margin) assert nothing rather than guess.

These thresholds are tunable via scene_extractor.config.
"""

from __future__ import annotations

from typing import Iterable

from scene_extractor.config import (
    IN_FRONT_OF_AREA_RATIO,
    INSIDE_AREA_RATIO,
    NEXT_TO_OVERLAP_FRACTION,
    ON_TOP_OF_OVERLAP_FRACTION,
    ON_TOP_OF_VERTICAL_TOLERANCE,
    SPATIAL_CENTER_MARGIN,
    SPATIAL_GAP_MARGIN,
)
from scene_extractor.schema import BoundingBox, SceneObject, SceneRelation


def _intersection_area(a: BoundingBox, b: BoundingBox) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return (ix2 - ix1) * (iy2 - iy1)


def _horizontal_overlap(a: BoundingBox, b: BoundingBox) -> float:
    return max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))


def _vertical_overlap(a: BoundingBox, b: BoundingBox) -> float:
    return max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))


def is_left_of(a: BoundingBox, b: BoundingBox) -> bool:
    return a.cx < b.cx - SPATIAL_CENTER_MARGIN


def is_above(a: BoundingBox, b: BoundingBox) -> bool:
    return a.cy < b.cy - SPATIAL_CENTER_MARGIN


def is_inside(a: BoundingBox, b: BoundingBox) -> bool:
    if a.area >= b.area:
        return False
    if a.area <= 0:
        return False
    return _intersection_area(a, b) / a.area > INSIDE_AREA_RATIO


def is_on_top_of(a: BoundingBox, b: BoundingBox) -> bool:
    if not is_above(a, b):
        return False
    if abs(a.y2 - b.y1) > ON_TOP_OF_VERTICAL_TOLERANCE:
        return False
    min_width = min(a.width, b.width)
    if min_width <= 0:
        return False
    return _horizontal_overlap(a, b) / min_width > ON_TOP_OF_OVERLAP_FRACTION


def is_next_to(a: BoundingBox, b: BoundingBox) -> bool:
    horizontal_gap = max(0.0, max(a.x1, b.x1) - min(a.x2, b.x2))
    if horizontal_gap >= SPATIAL_GAP_MARGIN:
        return False
    min_height = min(a.height, b.height)
    if min_height <= 0:
        return False
    return _vertical_overlap(a, b) / min_height > NEXT_TO_OVERLAP_FRACTION


def is_in_front_of(a: SceneObject, b: SceneObject) -> bool:
    """Coarse depth heuristic — only meaningful when both are the same category.

    Returns True iff a is the same category as b and a's box is >=1.5x as large.
    Cross-category comparisons are not depth-recoverable from 2D, so return False.
    """
    if a.category != b.category:
        return False
    if b.bbox.area <= 0:
        return False
    return a.bbox.area / b.bbox.area > IN_FRONT_OF_AREA_RATIO


def compute(objects: Iterable[SceneObject]) -> list[SceneRelation]:
    """Compute all spatial relations for a flat object list.

    Only canonical directions (left_of, above, inside, on_top_of, next_to)
    plus in_front_of (same-category-only) are emitted. Inverses come from
    the Prolog KB rules.
    """
    objs = list(objects)
    relations: list[SceneRelation] = []
    seen_next_to: set[tuple[str, str]] = set()

    for i, a in enumerate(objs):
        for j, b in enumerate(objs):
            if i == j:
                continue
            if is_left_of(a.bbox, b.bbox):
                relations.append(
                    SceneRelation(subject_id=a.id, predicate="left_of", object_id=b.id)
                )
            if is_above(a.bbox, b.bbox):
                relations.append(
                    SceneRelation(subject_id=a.id, predicate="above", object_id=b.id)
                )
            if is_inside(a.bbox, b.bbox):
                relations.append(
                    SceneRelation(subject_id=a.id, predicate="inside", object_id=b.id)
                )
            if is_on_top_of(a.bbox, b.bbox):
                relations.append(
                    SceneRelation(subject_id=a.id, predicate="on_top_of", object_id=b.id)
                )
            if is_in_front_of(a, b):
                relations.append(
                    SceneRelation(
                        subject_id=a.id, predicate="in_front_of", object_id=b.id
                    )
                )
            if is_next_to(a.bbox, b.bbox):
                key = tuple(sorted((a.id, b.id)))
                if key not in seen_next_to:
                    seen_next_to.add(key)
                    relations.append(
                        SceneRelation(
                            subject_id=key[0], predicate="next_to", object_id=key[1]
                        )
                    )

    return relations
