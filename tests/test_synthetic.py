"""End-to-end Phase 1 tests: synthetic scene → KB → query → verbalized answer.

These use hand-written Prolog queries (no LLM in Phase 1) to lock down the
non-LLM half of the pipeline. Once Phase 2 lands the NL→Prolog translator,
the same scenes feed `eval/harness.py` from English questions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import pytest

from kb_generator.generator import generate
from query_executor.executor import QueryExecutor
from query_executor.result import ParsedQuery
from scene_extractor.schema import SceneGraph
from synthetic import presets
from verbalizer.verbalizer import verbalize


@dataclass
class E2ECase:
    label: str
    scene_factory: Callable[[], SceneGraph]
    question: str
    parsed: ParsedQuery
    expected_answer: Any


CASES: list[E2ECase] = [
    E2ECase(
        label="exists-red-cube",
        scene_factory=presets.clevr_like,
        question="Is there a red cube?",
        parsed=ParsedQuery(
            query="object(X, cube), attribute(X, color, red)",
            type="boolean",
        ),
        expected_answer=True,
    ),
    E2ECase(
        label="exists-purple-cube-no",
        scene_factory=presets.clevr_like,
        question="Is there a purple cube?",
        parsed=ParsedQuery(
            query="object(X, cube), attribute(X, color, purple)",
            type="boolean",
        ),
        expected_answer=False,
    ),
    E2ECase(
        label="count-metal-objects",
        scene_factory=presets.clevr_like,
        question="How many metal objects are there?",
        parsed=ParsedQuery(
            query="findall(X, (object(X, _), attribute(X, material, metal)), L), length(L, N)",
            type="count",
        ),
        expected_answer=2,
    ),
    E2ECase(
        label="count-small-objects",
        scene_factory=presets.clevr_like,
        question="How many small objects are there?",
        parsed=ParsedQuery(
            query="findall(X, (object(X, _), attribute(X, size, small)), L), length(L, N)",
            type="count",
        ),
        expected_answer=1,
    ),
    E2ECase(
        label="attribute-color-of-sphere",
        scene_factory=presets.clevr_like,
        question="What color is the sphere?",
        parsed=ParsedQuery(
            query="object(X, sphere), attribute(X, color, C)",
            type="attribute",
            bind_variable="C",
        ),
        expected_answer="blue",
    ),
    E2ECase(
        label="object-left-of-sphere",
        scene_factory=presets.clevr_like,
        question="What is to the left of the sphere?",
        parsed=ParsedQuery(
            query="object(Y, sphere), left_of(X, Y)",
            type="object",
            bind_variable="X",
        ),
        expected_answer="obj_0",
    ),
    E2ECase(
        label="object-right-of-cube",
        scene_factory=presets.clevr_like,
        question="What is to the right of the cube?",
        parsed=ParsedQuery(
            query="object(Y, cube), right_of(X, Y)",
            type="object",
            bind_variable="X",
        ),
        expected_answer="obj_1",
    ),
    E2ECase(
        label="list-all-objects",
        scene_factory=presets.clevr_like,
        question="What objects are in the scene?",
        parsed=ParsedQuery(
            query="findall(X, object(X, _), L)",
            type="list",
            bind_variable="L",
        ),
        expected_answer=["obj_0", "obj_1", "obj_2"],
    ),
    E2ECase(
        label="kitchen-bottle-on-table",
        scene_factory=presets.kitchen,
        question="Is the bottle on the table?",
        parsed=ParsedQuery(
            query="object(B, bottle), object(T, table), on_top_of(B, T)",
            type="boolean",
        ),
        expected_answer=True,
    ),
    E2ECase(
        label="kitchen-count-glass-objects",
        scene_factory=presets.kitchen,
        question="How many glass objects are there?",
        parsed=ParsedQuery(
            query="findall(X, (object(X, _), attribute(X, material, glass)), L), length(L, N)",
            type="count",
        ),
        expected_answer=1,
    ),
    E2ECase(
        label="office-same-color-chairs",
        scene_factory=presets.office,
        question="Are the two chairs the same color?",
        parsed=ParsedQuery(
            query="object(A, chair), object(B, chair), A \\= B, same_color(A, B)",
            type="boolean",
        ),
        expected_answer=True,
    ),
    E2ECase(
        label="empty-scene-no-red-cube",
        scene_factory=presets.empty_scene,
        question="Is there a red cube?",
        parsed=ParsedQuery(
            query="object(X, cube), attribute(X, color, red)",
            type="boolean",
        ),
        expected_answer=False,
    ),
]


@pytest.fixture(scope="module")
def shared_executor() -> QueryExecutor:
    return QueryExecutor(timeout_s=3.0)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.label)
def test_e2e_synthetic_pipeline(case: E2ECase, shared_executor: QueryExecutor) -> None:
    scene = case.scene_factory()
    kb = generate(scene)
    result = shared_executor.run(kb.source, case.parsed)
    assert result.answer == case.expected_answer, (
        f"{case.label}: expected {case.expected_answer!r}, "
        f"got {result.answer!r} (error={result.error!r})"
    )
    bundle = verbalize(case.question, case.parsed, result, scene)
    assert bundle.answer  # non-empty verbalized string
    assert len(bundle.trace) >= 2  # at least the Q + Prolog translation steps


def test_at_least_ten_e2e_cases() -> None:
    """Phase 1 exit criterion: ≥10 end-to-end synthetic tests."""
    assert len(CASES) >= 10
