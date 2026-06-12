#!/usr/bin/env bash
# Run both eval suites and write results to evaluation/results/.
#
# Usage:
#   scripts/run_eval.sh [synthetic|clevr|all]
#
# Backend selection honors NL2PROLOG_BACKEND from the environment. If unset,
# defaults to "local" (ollama) per the plan.
set -euo pipefail

MODE="${1:-all}"
PYTHON="${PYTHON:-.venv/bin/python}"

if [ ! -x "$PYTHON" ]; then
  echo "Python not found at $PYTHON. Activate the venv or set PYTHON=<path>."
  exit 1
fi

case "$MODE" in
  synthetic)
    "$PYTHON" -m evaluation.cli --suite synthetic
    ;;
  clevr)
    "$PYTHON" -m evaluation.cli --suite clevr
    ;;
  all)
    "$PYTHON" -m evaluation.cli --suite all
    ;;
  *)
    echo "Unknown mode: $MODE (expected: synthetic | clevr | all)"
    exit 1
    ;;
esac
