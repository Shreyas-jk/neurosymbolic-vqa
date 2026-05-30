"""SceneGraph → Prolog knowledge base.

Pure deterministic Python; no LLM. Produces a string that, when consulted by
SWI-Prolog, exposes the scene as queryable facts under a fixed predicate
schema (object/2, attribute/3, relation/3) plus derived helper predicates.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from scene_extractor.schema import SceneGraph
from kb_generator import templates


@dataclass(frozen=True)
class PredicateSchema:
    """Summary of what's in this scene's KB. Consumed by nl2prolog.

    Built from the SceneGraph (not by re-parsing the Prolog) so dynamic prompt
    construction is cheap and lossless.
    """

    categories: tuple[str, ...]
    attribute_keys: tuple[str, ...]
    attribute_values_by_key: dict[str, tuple[str, ...]] = field(default_factory=dict)
    relation_predicates: tuple[str, ...] = ()


@dataclass(frozen=True)
class KBProgram:
    """Generated Prolog program plus its predicate schema."""

    source: str
    schema: PredicateSchema


def generate(scene: SceneGraph) -> KBProgram:
    """Build a Prolog KB string from a SceneGraph."""
    object_lines: list[str] = []
    attribute_lines: list[str] = []
    relation_lines: list[str] = []

    categories: set[str] = set()
    attr_keys: set[str] = set()
    attr_values: dict[str, set[str]] = {}
    relations: set[str] = set()

    for obj in sorted(scene.objects, key=lambda o: o.id):
        object_lines.append(templates.object_fact(obj.id, obj.category))
        categories.add(obj.category)
        for key, value in sorted(obj.attributes.items()):
            attribute_lines.append(templates.attribute_fact(obj.id, key, value))
            attr_keys.add(key)
            attr_values.setdefault(key, set()).add(value)

    for rel in sorted(
        scene.relations, key=lambda r: (r.subject_id, r.predicate, r.object_id)
    ):
        relation_lines.append(
            templates.relation_fact(rel.subject_id, rel.predicate, rel.object_id)
        )
        relations.add(rel.predicate)

    sections: list[str] = [templates.HEADER]
    sections.append("% --- Object facts ---")
    sections.extend(object_lines if object_lines else ["% (none)"])
    sections.append("")
    sections.append("% --- Attribute facts ---")
    sections.extend(attribute_lines if attribute_lines else ["% (none)"])
    sections.append("")
    sections.append("% --- Relation facts ---")
    sections.extend(relation_lines if relation_lines else ["% (none)"])
    sections.append("")
    sections.append(templates.RULES)

    source = "\n".join(sections)

    schema = PredicateSchema(
        categories=tuple(sorted(categories)),
        attribute_keys=tuple(sorted(attr_keys)),
        attribute_values_by_key={k: tuple(sorted(v)) for k, v in attr_values.items()},
        relation_predicates=tuple(sorted(relations)),
    )
    return KBProgram(source=source, schema=schema)
