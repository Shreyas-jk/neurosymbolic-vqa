"""Fluent builder for synthetic scene graphs.

Lets tests and the no-image demo path construct a SceneGraph without invoking
the vision pipeline. Bounding boxes are auto-generated on a grid layout for
visual placeholder purposes only — the KB generator does not consult them.
"""

from __future__ import annotations

import math
from typing import Optional

from scene_extractor.schema import (
    BoundingBox,
    SceneGraph,
    SceneObject,
    SceneRelation,
)


class SyntheticScene:
    def __init__(self) -> None:
        self._objects: list[SceneObject] = []
        self._relations: list[SceneRelation] = []
        self._ids: set[str] = set()

    def add_object(
        self,
        id: str,
        category: str,
        bbox: Optional[BoundingBox] = None,
        **attributes: str,
    ) -> "SyntheticScene":
        if id in self._ids:
            raise ValueError(f"Duplicate object id: {id!r}")
        self._ids.add(id)
        if bbox is None:
            bbox = self._auto_bbox(len(self._objects))
        self._objects.append(
            SceneObject(
                id=id,
                category=category,
                bbox=bbox,
                attributes=dict(attributes),
            )
        )
        return self

    def add_relation(self, subject_id: str, predicate: str, object_id: str) -> "SyntheticScene":
        if subject_id not in self._ids:
            raise ValueError(f"Unknown subject_id: {subject_id!r}")
        if object_id not in self._ids:
            raise ValueError(f"Unknown object_id: {object_id!r}")
        self._relations.append(
            SceneRelation(
                subject_id=subject_id,
                predicate=predicate,
                object_id=object_id,
            )
        )
        return self

    def to_scene_graph(self) -> SceneGraph:
        return SceneGraph(
            image_path=None,
            objects=list(self._objects),
            relations=list(self._relations),
            model_versions={"synthetic": "1.0"},
        )

    @staticmethod
    def _auto_bbox(index: int) -> BoundingBox:
        # Place objects on a 4-column grid; each cell is 0.2 wide × 0.2 tall.
        cols = 4
        col = index % cols
        row = index // cols
        x1 = 0.05 + col * 0.22
        y1 = 0.05 + row * 0.22
        # Clamp to keep within [0,1] for very large scenes.
        x1 = min(x1, 0.75)
        y1 = min(y1, 0.75)
        return BoundingBox(x1=x1, y1=y1, x2=x1 + 0.18, y2=y1 + 0.18)
