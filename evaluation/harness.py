"""Evaluation harness — runs the full pipeline against a dataset of cases.

Two entry points:
    run_eval(backend, dataset, out_path)          — synthetic cases (scene_factory)
    run_eval_on_images(backend, image_cases, ...)  — image cases (scene_extractor)

Per case we record per-stage latencies and the failure stage when wrong:
    vision         — scene extraction crashed (image cases only)
    kb_validation  — KB generator produced invalid Prolog
    translation    — NL → Prolog failed all retries OR returned wrong qtype tag
    execution      — query crashed / timed out at runtime
    verbalization  — verbalizer errored (rare; templates are pure-python)
    correctness    — pipeline succeeded but answer mismatched ground truth

Outputs a JSON file (per-case + summary) and prints a rich-table summary to
stdout. Designed to be deterministic over a fixed seed — re-running on the
same backend + dataset produces identical results modulo LLM variance.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Optional, Sequence

from kb_generator import generator as kb_gen
from kb_generator.validator import ValidationResult, validate
from nl2prolog.schema_builder import build_schema
from nl2prolog.translator import (
    TranslationError,
    TranslationResult,
    TranslatorBackend,
    TranslatorPipeline,
)
from query_executor.executor import QueryExecutor
from query_executor.result import QueryResult
from scene_extractor.schema import SceneGraph
from verbalizer.verbalizer import AnswerBundle, verbalize

from evaluation.golden_dataset import EvalCase, QTYPE_BUCKETS


FailureStage = Literal[
    "vision",
    "kb_validation",
    "translation",
    "execution",
    "verbalization",
    "correctness",
]


@dataclass
class EvalResult:
    case_id: str
    qtype: str
    question: str
    expected: Any
    predicted: Any
    correct: bool
    failure_stage: Optional[FailureStage]
    error: Optional[str]
    latency_ms_by_stage: dict[str, float] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)


# ---- Comparator -----------------------------------------------------------


def _coerce_int(v: Any) -> Optional[int]:
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, str):
        s = v.strip()
        try:
            return int(s)
        except ValueError:
            return None
    return None


def _category_of(scene: SceneGraph, obj_id: Any) -> Optional[str]:
    if not isinstance(obj_id, str):
        return None
    obj = scene.object_by_id(obj_id)
    return obj.category if obj else None


def is_correct(case: EvalCase, scene: SceneGraph, qresult: QueryResult) -> bool:
    """qtype-aware comparison between qresult and the case's expected answer.

    Returns True iff the pipeline's answer matches `case.expected` under the
    semantics of `case.qtype`. Object-type answers are resolved through the
    scene's id → category map before comparison.
    """
    answer = qresult.answer
    expected = case.expected
    qt = case.qtype

    if qt == "boolean":
        return bool(answer) == bool(expected)

    if qt == "count":
        predicted = _coerce_int(answer)
        return predicted is not None and predicted == int(expected)

    if qt == "attribute":
        if answer is None:
            return False
        return str(answer).strip().lower() == str(expected).strip().lower()

    if qt == "object":
        cat = _category_of(scene, answer)
        if cat is None:
            return False
        return cat.strip().lower() == str(expected).strip().lower()

    if qt == "list":
        if not isinstance(answer, list):
            return False
        predicted_cats = sorted(
            c for c in (_category_of(scene, a) for a in answer) if c is not None
        )
        expected_cats = sorted(str(e).strip().lower() for e in expected)
        return [c.strip().lower() for c in predicted_cats] == expected_cats

    return False


# ---- Failure-stage classification -----------------------------------------


def classify_failure_stage(
    *,
    vision_error: Optional[str] = None,
    kb_validation: Optional[ValidationResult] = None,
    translation_error: Optional[str] = None,
    execution_error: Optional[str] = None,
    verbalization_error: Optional[str] = None,
    correct: bool = True,
) -> Optional[FailureStage]:
    if vision_error is not None:
        return "vision"
    if kb_validation is not None and not kb_validation.ok:
        return "kb_validation"
    if translation_error is not None:
        return "translation"
    if execution_error is not None:
        return "execution"
    if verbalization_error is not None:
        return "verbalization"
    if not correct:
        return "correctness"
    return None


# ---- Single-case runner ---------------------------------------------------


def _run_case(
    case: EvalCase,
    pipeline: TranslatorPipeline,
    executor: QueryExecutor,
    *,
    scene_override: Optional[SceneGraph] = None,
    vision_error: Optional[str] = None,
    vision_latency_ms: float = 0.0,
) -> EvalResult:
    latency: dict[str, float] = {}
    extras: dict[str, Any] = {"vision_latency_ms": vision_latency_ms}

    if vision_error is not None:
        return EvalResult(
            case_id=case.case_id,
            qtype=case.qtype,
            question=case.question,
            expected=case.expected,
            predicted=None,
            correct=False,
            failure_stage="vision",
            error=vision_error,
            latency_ms_by_stage={"vision": vision_latency_ms},
            extras=extras,
        )

    scene = scene_override if scene_override is not None else case.scene_factory()

    t0 = time.perf_counter()
    try:
        kb = kb_gen.generate(scene)
    except Exception as exc:
        latency["kb_generation"] = (time.perf_counter() - t0) * 1000
        return EvalResult(
            case_id=case.case_id,
            qtype=case.qtype,
            question=case.question,
            expected=case.expected,
            predicted=None,
            correct=False,
            failure_stage="kb_validation",
            error=f"kb_generation crashed: {exc}",
            latency_ms_by_stage=latency,
            extras=extras,
        )
    latency["kb_generation"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    vresult = validate(kb.source)
    latency["kb_validation"] = (time.perf_counter() - t0) * 1000
    if not vresult.ok:
        return EvalResult(
            case_id=case.case_id,
            qtype=case.qtype,
            question=case.question,
            expected=case.expected,
            predicted=None,
            correct=False,
            failure_stage="kb_validation",
            error="; ".join(vresult.errors)[:500],
            latency_ms_by_stage=latency,
            extras=extras,
        )

    schema_block = build_schema(kb)
    t0 = time.perf_counter()
    try:
        translation: TranslationResult = pipeline.translate(
            case.question, kb.source, schema_block
        )
    except TranslationError as exc:
        latency["translation"] = (time.perf_counter() - t0) * 1000
        return EvalResult(
            case_id=case.case_id,
            qtype=case.qtype,
            question=case.question,
            expected=case.expected,
            predicted=None,
            correct=False,
            failure_stage="translation",
            error=f"translation failed after {len(exc.attempts)} attempts: {exc}"[:500],
            latency_ms_by_stage=latency,
            extras={**extras, "attempts": len(exc.attempts)},
        )
    latency["translation"] = (time.perf_counter() - t0) * 1000
    extras["query"] = translation.parsed.query
    extras["parsed_type"] = translation.parsed.type
    extras["attempts"] = len(translation.attempts) + 1

    # Wrong qtype tag from the LLM counts as a translation failure even
    # if the query executes — we asked for a specific shape.
    if translation.parsed.type != case.qtype:
        return EvalResult(
            case_id=case.case_id,
            qtype=case.qtype,
            question=case.question,
            expected=case.expected,
            predicted=None,
            correct=False,
            failure_stage="translation",
            error=(
                f"qtype mismatch: expected {case.qtype!r}, "
                f"LLM returned {translation.parsed.type!r}"
            ),
            latency_ms_by_stage=latency,
            extras=extras,
        )

    t0 = time.perf_counter()
    try:
        qresult: QueryResult = executor.run(kb.source, translation.parsed)
    except Exception as exc:
        latency["execution"] = (time.perf_counter() - t0) * 1000
        return EvalResult(
            case_id=case.case_id,
            qtype=case.qtype,
            question=case.question,
            expected=case.expected,
            predicted=None,
            correct=False,
            failure_stage="execution",
            error=f"executor crashed: {exc}"[:500],
            latency_ms_by_stage=latency,
            extras=extras,
        )
    latency["execution"] = (time.perf_counter() - t0) * 1000

    if not qresult.success and qresult.error:
        return EvalResult(
            case_id=case.case_id,
            qtype=case.qtype,
            question=case.question,
            expected=case.expected,
            predicted=qresult.answer,
            correct=False,
            failure_stage="execution",
            error=qresult.error[:500],
            latency_ms_by_stage=latency,
            extras=extras,
        )

    t0 = time.perf_counter()
    try:
        bundle: AnswerBundle = verbalize(case.question, translation.parsed, qresult, scene)
    except Exception as exc:
        latency["verbalization"] = (time.perf_counter() - t0) * 1000
        return EvalResult(
            case_id=case.case_id,
            qtype=case.qtype,
            question=case.question,
            expected=case.expected,
            predicted=qresult.answer,
            correct=False,
            failure_stage="verbalization",
            error=f"verbalizer crashed: {exc}"[:500],
            latency_ms_by_stage=latency,
            extras=extras,
        )
    latency["verbalization"] = (time.perf_counter() - t0) * 1000
    extras["verbalized"] = bundle.answer

    correct = is_correct(case, scene, qresult)

    return EvalResult(
        case_id=case.case_id,
        qtype=case.qtype,
        question=case.question,
        expected=case.expected,
        predicted=qresult.answer,
        correct=correct,
        failure_stage=None if correct else "correctness",
        error=None,
        latency_ms_by_stage=latency,
        extras=extras,
    )


# ---- Top-level runner -----------------------------------------------------


def _qtype_bucket_for(case_id: str) -> str:
    for name, bucket in QTYPE_BUCKETS:
        if any(c.case_id == case_id for c in bucket):
            return name
    return "unknown"


def _summarize(results: Sequence[EvalResult]) -> dict[str, Any]:
    n = len(results)
    correct = sum(1 for r in results if r.correct)
    by_bucket: dict[str, dict[str, int]] = {}
    for r in results:
        bucket = _qtype_bucket_for(r.case_id)
        bb = by_bucket.setdefault(bucket, {"n": 0, "correct": 0})
        bb["n"] += 1
        if r.correct:
            bb["correct"] += 1
    failure_counts: dict[str, int] = {}
    for r in results:
        if r.failure_stage:
            failure_counts[r.failure_stage] = failure_counts.get(r.failure_stage, 0) + 1
    latency_means: dict[str, float] = {}
    for stage in ("vision", "kb_generation", "kb_validation", "translation", "execution", "verbalization"):
        vals = [r.latency_ms_by_stage.get(stage, 0.0) for r in results if stage in r.latency_ms_by_stage]
        if vals:
            latency_means[stage] = sum(vals) / len(vals)
    return {
        "n": n,
        "correct": correct,
        "accuracy": correct / n if n else 0.0,
        "by_qtype": {
            k: {
                "n": v["n"],
                "correct": v["correct"],
                "accuracy": v["correct"] / v["n"] if v["n"] else 0.0,
            }
            for k, v in by_bucket.items()
        },
        "failure_stage_counts": failure_counts,
        "mean_latency_ms_by_stage": latency_means,
    }


def _print_table(summary: dict[str, Any]) -> None:
    # Import inside so unit tests don't require rich.
    from rich.console import Console
    from rich.table import Table

    console = Console()
    headline = (
        f"[bold]Overall:[/bold] {summary['correct']}/{summary['n']} "
        f"({summary['accuracy'] * 100:.1f}%)"
    )
    console.print(headline)

    bucket_table = Table(title="Accuracy by question type")
    bucket_table.add_column("Type")
    bucket_table.add_column("Correct")
    bucket_table.add_column("Total")
    bucket_table.add_column("Accuracy")
    for k in ("existence", "count", "attribute", "spatial", "multi_hop"):
        bucket = summary["by_qtype"].get(k, {"correct": 0, "n": 0, "accuracy": 0.0})
        bucket_table.add_row(
            k,
            str(bucket["correct"]),
            str(bucket["n"]),
            f"{bucket['accuracy'] * 100:.1f}%",
        )
    console.print(bucket_table)

    if summary.get("failure_stage_counts"):
        fail_table = Table(title="Failures by stage")
        fail_table.add_column("Stage")
        fail_table.add_column("Count")
        for stage, count in sorted(
            summary["failure_stage_counts"].items(), key=lambda kv: -kv[1]
        ):
            fail_table.add_row(stage, str(count))
        console.print(fail_table)

    if summary.get("mean_latency_ms_by_stage"):
        lat_table = Table(title="Mean latency (ms) by stage")
        lat_table.add_column("Stage")
        lat_table.add_column("Mean ms")
        for stage, mean in summary["mean_latency_ms_by_stage"].items():
            lat_table.add_row(stage, f"{mean:.1f}")
        console.print(lat_table)


def _write_json(out_path: Path, summary: dict[str, Any], results: Sequence[EvalResult]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": summary,
        "cases": [
            {
                **asdict(r),
                # Coerce types we can't dump otherwise.
                "expected": _to_jsonable(r.expected),
                "predicted": _to_jsonable(r.predicted),
            }
            for r in results
        ],
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _to_jsonable(v: Any) -> Any:
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_to_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _to_jsonable(x) for k, x in v.items()}
    return str(v)


def run_eval(
    backend: TranslatorBackend,
    dataset: Iterable[EvalCase],
    out_path: Path | str,
    *,
    max_attempts: int = 3,
    swipl_path: str = "swipl",
    quiet: bool = False,
) -> tuple[dict[str, Any], list[EvalResult]]:
    """Run a full eval against a synthetic dataset and write JSON + print table.

    Returns (summary, per-case results) so callers can post-process programmatically.
    """
    pipeline = TranslatorPipeline(
        backend=backend, max_attempts=max_attempts, swipl_path=swipl_path
    )
    executor = QueryExecutor()
    results: list[EvalResult] = []
    for case in dataset:
        results.append(_run_case(case, pipeline, executor))
    summary = _summarize(results)
    _write_json(Path(out_path), summary, results)
    if not quiet:
        _print_table(summary)
    return summary, results


# ---- Image-mode runner (CLEVR subset) -------------------------------------


@dataclass(frozen=True)
class ImageEvalCase:
    """An eval case backed by a real image (not a scene_factory)."""

    case_id: str
    image_path: str
    question: str
    expected: Any
    qtype: str
    ground_truth_objects: Optional[list[dict[str, Any]]] = None
    notes: str = ""


def run_eval_on_images(
    backend: TranslatorBackend,
    cases: Iterable[ImageEvalCase],
    out_path: Path | str,
    *,
    extractor: Any = None,
    max_attempts: int = 3,
    swipl_path: str = "swipl",
    quiet: bool = False,
) -> tuple[dict[str, Any], list[EvalResult]]:
    """CLEVR-style eval: run vision extractor first, then logic pipeline."""
    if extractor is None:
        from scene_extractor.extractor import SceneExtractor

        extractor = SceneExtractor()

    pipeline = TranslatorPipeline(
        backend=backend, max_attempts=max_attempts, swipl_path=swipl_path
    )
    executor = QueryExecutor()
    results: list[EvalResult] = []

    for img_case in cases:
        synthetic_case = EvalCase(
            case_id=img_case.case_id,
            scene_factory=lambda: None,  # unused — scene_override wins
            question=img_case.question,
            expected=img_case.expected,
            qtype=img_case.qtype,
            notes=img_case.notes,
        )
        t0 = time.perf_counter()
        try:
            scene = extractor.extract(img_case.image_path)
            vision_error = None
        except Exception as exc:
            scene = None
            vision_error = f"vision failed: {exc}"[:500]
        vision_ms = (time.perf_counter() - t0) * 1000

        results.append(
            _run_case(
                synthetic_case,
                pipeline,
                executor,
                scene_override=scene,
                vision_error=vision_error,
                vision_latency_ms=vision_ms,
            )
        )

    summary = _summarize(results)
    _write_json(Path(out_path), summary, results)
    if not quiet:
        _print_table(summary)
    return summary, results
