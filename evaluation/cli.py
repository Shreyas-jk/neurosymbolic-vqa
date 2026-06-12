"""Command-line entrypoint for the eval harness.

Usage:
    python -m evaluation.cli --suite synthetic
    python -m evaluation.cli --suite clevr
    python -m evaluation.cli --suite all

Backend selection: honors NL2PROLOG_BACKEND env var (set to `openai` to use
the OpenAI backend, otherwise defaults to local/ollama). The plan defaults to
local backend; we deliberately don't auto-fall-back to OpenAI in the CLI to
avoid surprise paid API usage.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from evaluation.clevr_subset import ensure_clevr_subset
from evaluation.golden_dataset import GOLDEN_DATASET
from evaluation.harness import run_eval, run_eval_on_images
from nl2prolog import NoBackendAvailableError, get_backend

DEFAULT_RESULTS_DIR = Path(__file__).resolve().parents[1] / "evaluation" / "results"


def _make_backend(name: str | None):
    return get_backend(backend_name=name, allow_fallback=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the neurosymbolic VQA eval harness.")
    parser.add_argument(
        "--suite",
        choices=("synthetic", "clevr", "all"),
        default="synthetic",
        help="Which eval suite to run.",
    )
    parser.add_argument(
        "--backend",
        default=None,
        help="Backend override. Defaults to NL2PROLOG_BACKEND or 'local'.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_RESULTS_DIR),
        help="Directory to write JSON results into.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Translation retry budget per case.",
    )
    args = parser.parse_args(argv)

    backend_name = args.backend or os.environ.get("NL2PROLOG_BACKEND", "local")
    try:
        backend = _make_backend(backend_name)
    except NoBackendAvailableError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exit_code = 0
    if args.suite in ("synthetic", "all"):
        print(f"\n=== Synthetic golden dataset ({len(GOLDEN_DATASET)} cases) ===")
        summary, _ = run_eval(
            backend=backend,
            dataset=GOLDEN_DATASET,
            out_path=out_dir / "synthetic.json",
            max_attempts=args.max_attempts,
        )
        if summary["accuracy"] < 0.5:
            print(
                f"WARNING: synthetic accuracy {summary['accuracy']:.0%} is below "
                "the 50% sanity threshold."
            )

    if args.suite in ("clevr", "all"):
        cases, status = ensure_clevr_subset(fetch_if_missing=True)
        print(f"\n=== CLEVR subset ({len(cases)} cases) ===")
        if not cases:
            print(
                f"  No CLEVR cases available. Fetch status: {status}. "
                "Skipping CLEVR suite."
            )
        else:
            run_eval_on_images(
                backend=backend,
                cases=cases,
                out_path=out_dir / "clevr.json",
                max_attempts=args.max_attempts,
            )

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
