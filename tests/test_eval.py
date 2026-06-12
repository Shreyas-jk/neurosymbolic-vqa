"""Tests for the evaluation harness.

Uses a `MockBackend` that returns canned LLM responses, so these tests run
without ollama, without OpenAI, and without any model downloads. The integration
between the mocked backend and the rest of the pipeline (validator, executor,
verbalizer) is exercised against real subprocess swipl + pyswip.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, List, Tuple

import pytest

from evaluation.golden_dataset import GOLDEN_DATASET, QTYPE_BUCKETS, EvalCase
from evaluation.harness import (
    EvalResult,
    classify_failure_stage,
    is_correct,
    run_eval,
)
from kb_generator.validator import ValidationResult
from kb_generator.generator import generate
from query_executor.result import ParsedQuery, QueryResult
from nl2prolog.translator import TranslatorBackend
from synthetic import presets


# ----- MockBackend feeds canned JSON responses ----------------------------


class CannedBackend(TranslatorBackend):
    """Returns the next response per call. Tests inject the exact LLM output."""

    name = "canned"

    def __init__(self, responses: List[str]) -> None:
        self._iter: Iterator[str] = iter(responses)
        self.calls: List[Tuple[str, str]] = []

    def call(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        try:
            return next(self._iter)
        except StopIteration:
            raise AssertionError("CannedBackend ran out of canned responses")


def _json_response(query: str, qtype: str) -> str:
    return json.dumps({"query": query, "type": qtype})


# ----- Golden dataset sanity ----------------------------------------------


def test_golden_dataset_has_thirty_cases() -> None:
    assert len(GOLDEN_DATASET) == 30


def test_golden_dataset_bucket_sizes_match_plan() -> None:
    # Each bucket should be exactly 6 per the plan (existence/count/attribute/spatial/multi_hop × 6).
    for name, bucket in QTYPE_BUCKETS:
        assert len(bucket) == 6, f"bucket {name!r} has {len(bucket)} cases (expected 6)"


def test_golden_dataset_qtypes_are_valid() -> None:
    valid = {"boolean", "count", "attribute", "object", "list"}
    for case in GOLDEN_DATASET:
        assert case.qtype in valid, f"{case.case_id} has invalid qtype {case.qtype!r}"


def test_golden_dataset_case_ids_unique() -> None:
    ids = [c.case_id for c in GOLDEN_DATASET]
    assert len(set(ids)) == len(ids)


def test_golden_dataset_scene_factories_produce_valid_scenes() -> None:
    for case in GOLDEN_DATASET:
        scene = case.scene_factory()
        # Smoke check — just confirm the factory doesn't crash and yields a SceneGraph.
        assert scene is not None
        # Empty scene is allowed (E4 uses it).
        assert hasattr(scene, "objects")


# ----- is_correct comparator ----------------------------------------------


def test_is_correct_boolean_true_true() -> None:
    case = EvalCase("X", presets.empty_scene, "?", True, "boolean")
    scene = presets.empty_scene()
    qr = QueryResult(
        success=True, answer=True, raw_bindings=(), query_string="q",
        execution_time_ms=0.0, error=None, type="boolean",
    )
    assert is_correct(case, scene, qr)


def test_is_correct_boolean_false_when_predicted_true_expected_false() -> None:
    case = EvalCase("X", presets.empty_scene, "?", False, "boolean")
    scene = presets.empty_scene()
    qr = QueryResult(
        success=True, answer=True, raw_bindings=(), query_string="q",
        execution_time_ms=0.0, error=None, type="boolean",
    )
    assert not is_correct(case, scene, qr)


def test_is_correct_count_matches_int() -> None:
    case = EvalCase("X", presets.clevr_like, "?", 3, "count")
    scene = presets.clevr_like()
    qr = QueryResult(
        success=True, answer=3, raw_bindings=(), query_string="q",
        execution_time_ms=0.0, error=None, type="count",
    )
    assert is_correct(case, scene, qr)


def test_is_correct_count_rejects_wrong_int() -> None:
    case = EvalCase("X", presets.clevr_like, "?", 3, "count")
    scene = presets.clevr_like()
    qr = QueryResult(
        success=True, answer=2, raw_bindings=(), query_string="q",
        execution_time_ms=0.0, error=None, type="count",
    )
    assert not is_correct(case, scene, qr)


def test_is_correct_attribute_case_insensitive() -> None:
    case = EvalCase("X", presets.clevr_like, "?", "RED", "attribute")
    scene = presets.clevr_like()
    qr = QueryResult(
        success=True, answer="red", raw_bindings=(), query_string="q",
        execution_time_ms=0.0, error=None, type="attribute",
    )
    assert is_correct(case, scene, qr)


def test_is_correct_object_resolves_obj_id_to_category() -> None:
    case = EvalCase("X", presets.clevr_like, "?", "cube", "object")
    scene = presets.clevr_like()
    qr = QueryResult(
        success=True, answer="obj_0", raw_bindings=(), query_string="q",
        execution_time_ms=0.0, error=None, type="object",
    )
    # obj_0 is the cube in clevr_like.
    assert is_correct(case, scene, qr)


def test_is_correct_object_rejects_wrong_category() -> None:
    case = EvalCase("X", presets.clevr_like, "?", "cube", "object")
    scene = presets.clevr_like()
    qr = QueryResult(
        success=True, answer="obj_1", raw_bindings=(), query_string="q",
        execution_time_ms=0.0, error=None, type="object",
    )
    assert not is_correct(case, scene, qr)


# ----- failure-stage classifier -------------------------------------------


def test_classify_vision_dominates() -> None:
    assert classify_failure_stage(vision_error="boom") == "vision"


def test_classify_kb_validation_when_validation_fails() -> None:
    vr = ValidationResult(ok=False, errors=("ERROR: bad atom",))
    assert classify_failure_stage(kb_validation=vr) == "kb_validation"


def test_classify_translation_when_t_error_set() -> None:
    assert classify_failure_stage(translation_error="retries exhausted") == "translation"


def test_classify_execution_when_executor_errored() -> None:
    assert classify_failure_stage(execution_error="time_limit_exceeded") == "execution"


def test_classify_verbalization_when_verbalizer_crashed() -> None:
    assert classify_failure_stage(verbalization_error="template KeyError") == "verbalization"


def test_classify_correctness_when_pipeline_succeeded_but_wrong() -> None:
    assert classify_failure_stage(correct=False) == "correctness"


def test_classify_none_when_correct() -> None:
    assert classify_failure_stage(correct=True) is None


def test_classify_precedence_vision_over_correctness() -> None:
    assert classify_failure_stage(vision_error="x", correct=False) == "vision"


# ----- end-to-end run_eval with CannedBackend -----------------------------


@pytest.fixture
def tmp_results(tmp_path: Path) -> Path:
    return tmp_path / "results.json"


def test_run_eval_correct_case_is_marked_correct(tmp_results: Path) -> None:
    # One existence case: "Is there a red cube?" against clevr_like.
    case = EvalCase("E1", presets.clevr_like, "Is there a red cube?", True, "boolean")
    backend = CannedBackend([
        _json_response("object(X, cube), attribute(X, color, red)", "boolean"),
    ])
    summary, results = run_eval(backend, [case], tmp_results, quiet=True)
    assert summary["n"] == 1
    assert summary["correct"] == 1
    assert summary["accuracy"] == 1.0
    assert results[0].correct
    assert results[0].failure_stage is None


def test_run_eval_writes_json(tmp_results: Path) -> None:
    case = EvalCase("E1", presets.clevr_like, "Is there a red cube?", True, "boolean")
    backend = CannedBackend([
        _json_response("object(X, cube), attribute(X, color, red)", "boolean"),
    ])
    run_eval(backend, [case], tmp_results, quiet=True)
    payload = json.loads(tmp_results.read_text())
    assert "summary" in payload
    assert "cases" in payload
    assert payload["cases"][0]["case_id"] == "E1"


def test_run_eval_translation_failure_marked_correctly(tmp_results: Path) -> None:
    # All 3 attempts return garbage JSON → translation failure after retries.
    case = EvalCase("E1", presets.clevr_like, "Is there a red cube?", True, "boolean")
    backend = CannedBackend(["not json", "still not json", "no really"])
    summary, results = run_eval(backend, [case], tmp_results, quiet=True)
    assert summary["correct"] == 0
    assert results[0].failure_stage == "translation"


def test_run_eval_wrong_qtype_is_translation_failure(tmp_results: Path) -> None:
    # Valid query but LLM tags it with the wrong qtype.
    case = EvalCase("E1", presets.clevr_like, "Is there a red cube?", True, "boolean")
    backend = CannedBackend([
        _json_response("object(X, cube), attribute(X, color, red)", "count"),
    ])
    summary, results = run_eval(backend, [case], tmp_results, quiet=True)
    assert results[0].failure_stage == "translation"
    assert "qtype mismatch" in (results[0].error or "")


def test_run_eval_wrong_answer_is_correctness_failure(tmp_results: Path) -> None:
    # Pipeline succeeds but the symbolic answer is wrong (asks for green cube,
    # expected True — but no green cube exists, so answer is False).
    case = EvalCase("E2", presets.clevr_like, "Is there a red cube?", True, "boolean")
    backend = CannedBackend([
        _json_response("object(X, cube), attribute(X, color, green)", "boolean"),
    ])
    summary, results = run_eval(backend, [case], tmp_results, quiet=True)
    assert not results[0].correct
    assert results[0].failure_stage == "correctness"


def test_run_eval_records_per_stage_latency(tmp_results: Path) -> None:
    case = EvalCase("E1", presets.clevr_like, "Is there a red cube?", True, "boolean")
    backend = CannedBackend([
        _json_response("object(X, cube), attribute(X, color, red)", "boolean"),
    ])
    _, results = run_eval(backend, [case], tmp_results, quiet=True)
    lat = results[0].latency_ms_by_stage
    assert "kb_generation" in lat
    assert "kb_validation" in lat
    assert "translation" in lat
    assert "execution" in lat
    assert all(v >= 0 for v in lat.values())


def test_run_eval_summary_bucket_by_qtype(tmp_results: Path) -> None:
    # Two cases from different buckets so we see two distinct bucket entries.
    case_a = EvalCase("E1", presets.clevr_like, "Is there a red cube?", True, "boolean")
    case_b = EvalCase("C1", presets.clevr_like, "How many objects are there?", 3, "count")
    backend = CannedBackend([
        _json_response("object(X, cube), attribute(X, color, red)", "boolean"),
        _json_response("findall(X, object(X, _), L), length(L, N)", "count"),
    ])
    summary, _ = run_eval(backend, [case_a, case_b], tmp_results, quiet=True)
    assert "existence" in summary["by_qtype"]
    assert "count" in summary["by_qtype"]
    assert summary["by_qtype"]["existence"]["n"] == 1
    assert summary["by_qtype"]["count"]["n"] == 1


def test_run_eval_failure_stage_counts_populated(tmp_results: Path) -> None:
    case_ok = EvalCase("E1", presets.clevr_like, "Is there a red cube?", True, "boolean")
    case_bad = EvalCase("E2", presets.clevr_like, "Is there a red cube?", True, "boolean")
    backend = CannedBackend([
        _json_response("object(X, cube), attribute(X, color, red)", "boolean"),
        # Trigger correctness failure: returns False when expected True.
        _json_response("object(X, cube), attribute(X, color, green)", "boolean"),
    ])
    summary, _ = run_eval(backend, [case_ok, case_bad], tmp_results, quiet=True)
    assert summary["failure_stage_counts"].get("correctness", 0) == 1
