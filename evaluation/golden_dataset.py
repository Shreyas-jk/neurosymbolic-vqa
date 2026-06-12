"""30-triple synthetic golden dataset for the eval harness.

Spread: 6 each of existence / count / attribute / spatial / multi-hop.

Each `EvalCase` carries:
    case_id       — stable identifier (e.g. "E1", "C3", "S5", "M2")
    scene_factory — () -> SceneGraph; builds the scene fresh per run
    question      — natural language question
    expected      — ground-truth answer (bool / int / str / list[str])
    qtype         — one of: boolean | count | attribute | object | list
    notes         — short rationale (why this case, what it stress-tests)

All scenes come from `synthetic.presets`. The harness diffs the pipeline's
QueryResult against `expected` using `evaluation.harness.is_correct` (qtype-aware).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Tuple

from scene_extractor.schema import SceneGraph
from synthetic import presets


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    scene_factory: Callable[[], SceneGraph]
    question: str
    expected: Any
    qtype: str
    notes: str = ""
    tags: Tuple[str, ...] = field(default_factory=tuple)


# ---- Existence (6) -------------------------------------------------------

_EXISTENCE: Tuple[EvalCase, ...] = (
    EvalCase(
        "E1", presets.clevr_like, "Is there a red cube?", True, "boolean",
        "Direct match — red large metal cube exists in clevr_like."
    ),
    EvalCase(
        "E2", presets.clevr_like, "Is there a green sphere?", False, "boolean",
        "Negative — sphere is blue, cylinder is green; no green sphere."
    ),
    EvalCase(
        "E3", presets.kitchen, "Is there a table?", True, "boolean",
        "Category existence in everyday scene."
    ),
    EvalCase(
        "E4", presets.empty_scene, "Is there any cube?", False, "boolean",
        "Empty scene must say no to every existence query."
    ),
    EvalCase(
        "E5", presets.office, "Are there any wooden objects?", True, "boolean",
        "Material existence — desk is wood."
    ),
    EvalCase(
        "E6", presets.clevr_like, "Is there a metal cylinder?", True, "boolean",
        "Two-attribute existence — cylinder is metal and green."
    ),
)

# ---- Count (6) -----------------------------------------------------------

_COUNT: Tuple[EvalCase, ...] = (
    EvalCase(
        "C1", presets.clevr_like, "How many objects are there?", 3, "count",
        "Total count baseline."
    ),
    EvalCase(
        "C2", presets.clevr_like, "How many cubes are there?", 1, "count",
        "Category-filtered count."
    ),
    EvalCase(
        "C3", presets.office, "How many chairs are there?", 2, "count",
        "Counts category with multiple instances."
    ),
    EvalCase(
        "C4", presets.office, "How many black objects are there?", 3, "count",
        "Color-filtered count — 2 chairs + 1 monitor are black."
    ),
    EvalCase(
        "C5", presets.clevr_like, "How many metal objects are there?", 2, "count",
        "Material-filtered count — cube and cylinder are metal."
    ),
    EvalCase(
        "C6", presets.kitchen, "How many green objects are there?", 1, "count",
        "Color-filtered count — bottle is the only green object."
    ),
)

# ---- Attribute (6) -------------------------------------------------------

_ATTRIBUTE: Tuple[EvalCase, ...] = (
    EvalCase(
        "A1", presets.clevr_like, "What color is the cube?", "red", "attribute",
        "Bind color by category."
    ),
    EvalCase(
        "A2", presets.clevr_like, "What size is the sphere?", "small", "attribute",
        "Bind size by category."
    ),
    EvalCase(
        "A3", presets.clevr_like, "What material is the cylinder?", "metal", "attribute",
        "Bind material by category."
    ),
    EvalCase(
        "A4", presets.single_object, "What color is the cube?", "red", "attribute",
        "Single-object scene attribute lookup."
    ),
    EvalCase(
        "A5", presets.kitchen, "What material is the bottle?", "glass", "attribute",
        "Everyday-object material lookup."
    ),
    EvalCase(
        "A6", presets.clevr_like, "What size is the cube?", "large", "attribute",
        "Another size attribute lookup."
    ),
)

# ---- Spatial — answer is the category of the bound object ----------------

_SPATIAL: Tuple[EvalCase, ...] = (
    EvalCase(
        "S1", presets.clevr_like, "What is to the left of the sphere?", "cube",
        "object", "Inverse-direction lookup via left_of fact."
    ),
    EvalCase(
        "S2", presets.clevr_like, "What is to the right of the cube?", "sphere",
        "object", "Forward right_of derived from left_of inverse rule."
    ),
    EvalCase(
        "S3", presets.kitchen, "What is next to the cup?", "apple",
        "object", "Symmetric next_to lookup."
    ),
    EvalCase(
        "S4", presets.office, "What is on top of the desk?", "monitor",
        "object", "on_top_of stack lookup."
    ),
    EvalCase(
        "S5", presets.clevr_like, "What is above the cube?", "cylinder",
        "object", "above lookup."
    ),
    EvalCase(
        "S6", presets.clevr_like, "What is below the cylinder?", "cube",
        "object", "below derived from above inverse rule."
    ),
)

# ---- Multi-hop (6) — chained attribute + spatial / category constraints --

_MULTIHOP: Tuple[EvalCase, ...] = (
    EvalCase(
        "M1", presets.clevr_like, "What color is the object to the left of the sphere?",
        "red", "attribute",
        "Attribute on the object satisfying a spatial constraint."
    ),
    EvalCase(
        "M2", presets.clevr_like, "What size is the metal cube?",
        "large", "attribute",
        "Attribute lookup with category + material filter."
    ),
    EvalCase(
        "M3", presets.clevr_like, "What color is the cylinder above the cube?",
        "green", "attribute",
        "Category + spatial filter."
    ),
    EvalCase(
        "M4", presets.clevr_like, "What material is the small object?",
        "rubber", "attribute",
        "Attribute lookup filtered by another attribute (size)."
    ),
    EvalCase(
        "M5", presets.office, "How many metal chairs are there?", 2, "count",
        "Count filtered by category + material."
    ),
    EvalCase(
        "M6", presets.kitchen, "What color is the cup?",
        "white", "attribute",
        "Hop through scene to a specific category — cup is white ceramic."
    ),
)

GOLDEN_DATASET: Tuple[EvalCase, ...] = (
    _EXISTENCE + _COUNT + _ATTRIBUTE + _SPATIAL + _MULTIHOP
)

assert len(GOLDEN_DATASET) == 30, f"expected 30 golden triples, got {len(GOLDEN_DATASET)}"

QTYPE_BUCKETS: Tuple[Tuple[str, Tuple[EvalCase, ...]], ...] = (
    ("existence", _EXISTENCE),
    ("count", _COUNT),
    ("attribute", _ATTRIBUTE),
    ("spatial", _SPATIAL),
    ("multi_hop", _MULTIHOP),
)
