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


def test_golden_dataset_has_thirty_two_cases() -> None:
    # 30 from the plan's 5 buckets (×6) + 2 list-qtype cases added in cleanup.
    assert len(GOLDEN_DATASET) == 32


def test_golden_dataset_bucket_sizes_match_plan() -> None:
    # Plan buckets (existence/count/attribute/spatial/multi_hop) are 6 each.
    # The list bucket is post-plan and is 2.
    expected = {
        "existence": 6,
        "count": 6,
        "attribute": 6,
        "spatial": 6,
        "multi_hop": 6,
        "list": 2,
    }
    actual = {name: len(bucket) for name, bucket in QTYPE_BUCKETS}
    assert actual == expected, f"bucket sizes {actual!r} != {expected!r}"


def test_golden_dataset_list_cases_have_nonempty_expected() -> None:
    # The list cases compute expected from the preset at import — if the preset
    # ever drops the matching objects, this catches it.
    list_cases = [c for c in GOLDEN_DATASET if c.qtype == "list"]
    assert len(list_cases) == 2
    for c in list_cases:
        assert isinstance(c.expected, list)
        assert len(c.expected) >= 1, f"{c.case_id} has empty expected list"


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


# ----- is_correct list-qtype symmetric normalization ----------------------


def test_is_correct_list_matches_object_ids() -> None:
    # The pipeline returns obj_ids; expected can also be obj_ids and they
    # normalize symmetrically.
    case = EvalCase(
        "X", presets.clevr_like, "?", ["obj_0", "obj_2"], "list"
    )
    scene = presets.clevr_like()
    qr = QueryResult(
        success=True, answer=["obj_0", "obj_2"], raw_bindings=(),
        query_string="q", execution_time_ms=0.0, error=None, type="list",
    )
    assert is_correct(case, scene, qr)


def test_is_correct_list_matches_categories() -> None:
    # Expected as categories works too (existing semantic).
    case = EvalCase("X", presets.clevr_like, "?", ["cube", "cylinder"], "list")
    scene = presets.clevr_like()
    qr = QueryResult(
        success=True, answer=["obj_0", "obj_2"], raw_bindings=(),
        query_string="q", execution_time_ms=0.0, error=None, type="list",
    )
    assert is_correct(case, scene, qr)


def test_is_correct_list_rejects_wrong_membership() -> None:
    case = EvalCase("X", presets.clevr_like, "?", ["obj_0"], "list")
    scene = presets.clevr_like()
    qr = QueryResult(
        success=True, answer=["obj_1"], raw_bindings=(),
        query_string="q", execution_time_ms=0.0, error=None, type="list",
    )
    assert not is_correct(case, scene, qr)


# ----- CLI exit-code semantics --------------------------------------------


def test_compute_exit_code_none_summary_is_zero() -> None:
    from evaluation.cli import compute_exit_code
    assert compute_exit_code(None) == 0


def test_compute_exit_code_high_accuracy_is_zero() -> None:
    from evaluation.cli import compute_exit_code
    assert compute_exit_code({"accuracy": 0.9}) == 0


def test_compute_exit_code_at_threshold_is_zero() -> None:
    from evaluation.cli import compute_exit_code
    # Boundary: exactly 0.5 should pass (>= threshold).
    assert compute_exit_code({"accuracy": 0.5}) == 0


def test_compute_exit_code_below_threshold_is_one() -> None:
    from evaluation.cli import compute_exit_code
    assert compute_exit_code({"accuracy": 0.49}) == 1


def test_compute_exit_code_zero_accuracy_is_one() -> None:
    from evaluation.cli import compute_exit_code
    assert compute_exit_code({"accuracy": 0.0}) == 1


def test_compute_exit_code_custom_threshold() -> None:
    from evaluation.cli import compute_exit_code
    # 0.6 falls below a 0.8 threshold → 1.
    assert compute_exit_code({"accuracy": 0.6}, threshold=0.8) == 1


# ----- CLEVR subset loader -------------------------------------------------


_CLEVR_SCENES_PATH = Path("data/clevr_test_subset/scenes.json")
_CLEVR_SUBSET_AVAILABLE = _CLEVR_SCENES_PATH.exists()


def _scenes_count() -> int:
    if not _CLEVR_SUBSET_AVAILABLE:
        return 0
    with _CLEVR_SCENES_PATH.open() as f:
        payload = json.load(f)
    return len(payload["scenes"])


@pytest.mark.skipif(
    not _CLEVR_SUBSET_AVAILABLE,
    reason="CLEVR subset not extracted (run scripts/download_clevr.sh)",
)
def test_clevr_iter_cases_count_matches_formula() -> None:
    # 5 cases per scene: total_count + 2 attribute-counts + 1 positive existence
    # + 1 negative existence. (Counts may be 4 instead of 5 if a scene has no
    # color or no material — but CLEVR scenes always have both, so 5/scene holds.)
    from evaluation.clevr_subset import iter_cases
    cases = iter_cases()
    expected = _scenes_count() * 5
    assert len(cases) == expected, (
        f"expected {expected} cases ({_scenes_count()} scenes × 5), got {len(cases)}"
    )


@pytest.mark.skipif(
    not _CLEVR_SUBSET_AVAILABLE,
    reason="CLEVR subset not extracted (run scripts/download_clevr.sh)",
)
def test_clevr_iter_cases_image_paths_exist_on_disk() -> None:
    from evaluation.clevr_subset import iter_cases
    cases = iter_cases()
    assert cases, "expected at least one case"
    for c in cases:
        assert Path(c.image_path).exists(), f"image missing: {c.image_path}"
        assert Path(c.image_path).is_file()


@pytest.mark.skipif(
    not _CLEVR_SUBSET_AVAILABLE,
    reason="CLEVR subset not extracted (run scripts/download_clevr.sh)",
)
def test_clevr_existence_cases_match_scene_ground_truth() -> None:
    """Re-verify positive/negative existence expecteds against scenes.json."""
    from evaluation.clevr_subset import iter_cases

    with _CLEVR_SCENES_PATH.open() as f:
        scenes = json.load(f)["scenes"]
    scene_pairs_by_index = {
        i: {(obj.get("color"), obj.get("shape")) for obj in s["objects"]}
        for i, s in enumerate(scenes)
    }

    cases = iter_cases()
    checked_positive = checked_negative = 0
    for c in cases:
        if c.qtype != "boolean":
            continue
        # Case IDs look like CLEVR_{idx}_exists_{color}_{shape} or _not_exists_.
        parts = c.case_id.split("_")
        scene_idx = int(parts[1])
        scene_pairs = scene_pairs_by_index[scene_idx]
        if "not_exists" in c.case_id:
            color = parts[3]
            shape = parts[4]
            assert (color, shape) not in scene_pairs, (
                f"{c.case_id} marked False but ({color},{shape}) IS in scene"
            )
            assert c.expected is False
            checked_negative += 1
        elif "exists" in c.case_id:
            color = parts[3]
            shape = parts[4]
            assert (color, shape) in scene_pairs, (
                f"{c.case_id} marked True but ({color},{shape}) is NOT in scene"
            )
            assert c.expected is True
            checked_positive += 1
    # Sanity: we should have walked through both kinds of cases.
    assert checked_positive > 0
    assert checked_negative > 0


@pytest.mark.skipif(
    not _CLEVR_SUBSET_AVAILABLE,
    reason="CLEVR subset not extracted (run scripts/download_clevr.sh)",
)
def test_clevr_count_total_matches_scene_object_count() -> None:
    from evaluation.clevr_subset import iter_cases

    with _CLEVR_SCENES_PATH.open() as f:
        scenes = json.load(f)["scenes"]
    n_objects_by_index = {i: len(s["objects"]) for i, s in enumerate(scenes)}

    for c in iter_cases():
        if c.case_id.endswith("_count_total"):
            scene_idx = int(c.case_id.split("_")[1])
            assert c.expected == n_objects_by_index[scene_idx]


def test_clevr_iter_cases_returns_empty_when_no_data(tmp_path: Path, monkeypatch) -> None:
    """Without scenes.json + images, iter_cases must return [] (not raise)."""
    import evaluation.clevr_subset as cs
    monkeypatch.setattr(cs, "DATA_DIR", tmp_path / "missing")
    monkeypatch.setattr(cs, "SCENES_FILE", tmp_path / "missing" / "scenes.json")
    monkeypatch.setattr(cs, "IMAGES_DIR", tmp_path / "missing" / "images")
    assert cs.iter_cases() == []
    cases, status = cs.ensure_clevr_subset()
    assert cases == []
    assert status["scenes_file_present"] is False
