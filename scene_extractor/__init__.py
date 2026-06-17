"""Public API for the scene_extractor package.

Schema dataclasses are eagerly importable (no heavy deps). The vision
pipeline classes (SceneExtractor, ModelDownloadError) are lazy-loaded
on first attribute access, so synthetic-mode consumers don't pay the
torch/transformers import cost. This lets the HuggingFace Spaces demo
(synthetic-only) run without installing the vision stack.
"""

from scene_extractor.schema import (
    BoundingBox,
    SceneGraph,
    SceneObject,
    SceneRelation,
)

__all__ = [
    "BoundingBox",
    "SceneGraph",
    "SceneObject",
    "SceneRelation",
    "SceneExtractor",
    "ModelDownloadError",
]


def __getattr__(name: str):
    if name in ("SceneExtractor", "ModelDownloadError"):
        from scene_extractor.extractor import (
            ModelDownloadError,
            SceneExtractor,
        )
        return {"SceneExtractor": SceneExtractor, "ModelDownloadError": ModelDownloadError}[name]
    raise AttributeError(f"module 'scene_extractor' has no attribute {name!r}")
