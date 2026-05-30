"""Pydantic models shared across the pipeline.

Phase 1 uses these types via the synthetic scene builder; Phase 3 will populate
them from the OWL-ViT + CLIP vision stack. Validation lives here so downstream
modules (KB generator, query executor) can trust their inputs.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class BoundingBox(BaseModel):
    model_config = ConfigDict(frozen=True)

    x1: float = Field(ge=0.0, le=1.0)
    y1: float = Field(ge=0.0, le=1.0)
    x2: float = Field(ge=0.0, le=1.0)
    y2: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_ordering(self) -> "BoundingBox":
        if not (self.x1 < self.x2):
            raise ValueError(f"x1 ({self.x1}) must be < x2 ({self.x2})")
        if not (self.y1 < self.y2):
            raise ValueError(f"y1 ({self.y1}) must be < y2 ({self.y2})")
        return self

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height


class SceneObject(BaseModel):
    id: str
    category: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    bbox: BoundingBox
    attributes: dict[str, str] = Field(default_factory=dict)
    attribute_confidences: dict[str, float] = Field(default_factory=dict)

    @field_validator("id")
    @classmethod
    def _id_nonempty(cls, v: str) -> str:
        if not v:
            raise ValueError("SceneObject.id must be a non-empty string")
        return v


class SceneRelation(BaseModel):
    subject_id: str
    predicate: str
    object_id: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _no_self_relation(self) -> "SceneRelation":
        if self.subject_id == self.object_id:
            raise ValueError(
                f"Self-relation not allowed: {self.subject_id} {self.predicate} {self.object_id}"
            )
        return self


class SceneGraph(BaseModel):
    image_path: Optional[str] = None
    objects: list[SceneObject] = Field(default_factory=list)
    relations: list[SceneRelation] = Field(default_factory=list)
    extraction_time_ms: float = 0.0
    model_versions: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _relations_reference_known_objects(self) -> "SceneGraph":
        ids = {obj.id for obj in self.objects}
        for rel in self.relations:
            if rel.subject_id not in ids:
                raise ValueError(
                    f"Relation references unknown subject_id={rel.subject_id!r}"
                )
            if rel.object_id not in ids:
                raise ValueError(
                    f"Relation references unknown object_id={rel.object_id!r}"
                )
        return self

    def object_by_id(self, obj_id: str) -> Optional[SceneObject]:
        for obj in self.objects:
            if obj.id == obj_id:
                return obj
        return None
