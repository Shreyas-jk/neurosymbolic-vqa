"""Evaluation harness for the neurosymbolic VQA pipeline.

Public surface:
    EvalCase, EvalResult — record types
    GOLDEN_DATASET — 30-triple synthetic golden set (existence/count/attribute/spatial/multi-hop × 6)
    run_eval — orchestrates one full eval run + writes JSON + prints rich table
    classify_failure_stage — taxonomy: vision / kb_validation / translation / execution / verbalization / correctness
    Named `evaluation` rather than `eval` to avoid shadowing the Python builtin.
"""

from __future__ import annotations

from evaluation.golden_dataset import GOLDEN_DATASET, EvalCase
from evaluation.harness import (
    EvalResult,
    FailureStage,
    classify_failure_stage,
    is_correct,
    run_eval,
)

__all__ = [
    "EvalCase",
    "EvalResult",
    "FailureStage",
    "GOLDEN_DATASET",
    "classify_failure_stage",
    "is_correct",
    "run_eval",
]
