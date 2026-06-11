"""Vision → SceneGraph pipeline.

Public surface:
    SceneExtractor — top-level orchestrator (image -> SceneGraph)
    ModelDownloadError
    SceneGraph, SceneObject, SceneRelation, BoundingBox — schema types
    spatial_relations.compute(objects) — pure-geometric relation extraction
    attribute_classifier.classify(crop) — CLIP attribute scoring
    models.get_detector / get_clip / get_device — cached model handles
"""

from __future__ import annotations

from scene_extractor.extractor import ModelDownloadError, SceneExtractor
from scene_extractor.schema import (
    BoundingBox,
    SceneGraph,
    SceneObject,
    SceneRelation,
)

__all__ = [
    "BoundingBox",
    "ModelDownloadError",
    "SceneExtractor",
    "SceneGraph",
    "SceneObject",
    "SceneRelation",
]
