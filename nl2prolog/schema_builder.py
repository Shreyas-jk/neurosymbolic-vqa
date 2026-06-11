"""Builds the dynamic predicate menu shown to the LLM.

Listing only categories/attributes/relations that are *actually present* in this
KB keeps the prompt tight and eliminates an entire class of LLM hallucinations
(e.g. asking about an attribute key that doesn't exist).
"""

from __future__ import annotations

from kb_generator.generator import KBProgram


# Predicates that are always emitted by kb_generator.templates, regardless of
# scene content. Listed here so the LLM can rely on them being present.
_ALWAYS_PRESENT_PREDICATES: tuple[str, ...] = (
    "object(?Id, ?Category)",
    "attribute(?Id, ?Key, ?Value)",
    "relation(?Subj, ?Pred, ?Obj)",
    "is_a(?Id, ?Category)",
    "has(?Id, ?Key, ?Value)",
    "left_of(?A, ?B)",
    "right_of(?A, ?B)",
    "above(?A, ?B)",
    "below(?A, ?B)",
    "inside(?A, ?B)",
    "on_top_of(?A, ?B)",
    "next_to(?A, ?B)",
    "same_color(?A, ?B)",
    "same_size(?A, ?B)",
    "same_material(?A, ?B)",
    "same_shape(?A, ?B)",
)


def build_schema(kb: KBProgram) -> str:
    """Return a plain-text predicate menu derived from the KB's contents."""
    s = kb.schema
    lines: list[str] = ["SCHEMA"]

    if s.categories:
        lines.append(f"Object categories present: {', '.join(s.categories)}")
    else:
        lines.append("Object categories present: (none — scene is empty)")

    if s.attribute_keys:
        lines.append(f"Attribute keys present: {', '.join(s.attribute_keys)}")
        for key in s.attribute_keys:
            values = s.attribute_values_by_key.get(key, ())
            if values:
                lines.append(f"  - {key} values: {', '.join(values)}")
    else:
        lines.append("Attribute keys present: (none)")

    if s.relation_predicates:
        lines.append(
            f"Relation predicates present in facts: {', '.join(s.relation_predicates)}"
        )
    else:
        lines.append("Relation predicates present in facts: (none)")

    lines.append("")
    lines.append("Available predicates (use only these):")
    for sig in _ALWAYS_PRESENT_PREDICATES:
        lines.append(f"  {sig}")

    return "\n".join(lines)
