# Phase Documentation Index

Build narrative for the neurosymbolic VQA project. Each phase doc was written
at the time the phase finished, so language is in the past tense relative to
that point. Read in numerical order to follow the trajectory.

## Phase 2 — NL → Prolog translator
- [PHASE_2_REVIEW.md](PHASE_2_REVIEW.md) — Independent reviewer report (PASS).
  Confirms 15 few-shot pairs, retry loop, dual backend (OpenAI / ollama),
  subprocess-swipl validator.

## Phase 3 — Vision pipeline
- [PHASE_3_REVIEW.md](PHASE_3_REVIEW.md) — Reviewer report (PASS).
  Confirms OWL-ViT detection + CLIP attribute classification + geometric
  spatial relations, all on MPS, with `lru_cache` model singletons.

## Phase 4 — Evaluation harness
- [PHASE_4_REVIEW.md](PHASE_4_REVIEW.md) — Reviewer report (PASS).
  Confirms `evaluation/` module, 30-triple golden dataset, failure-stage
  taxonomy, rich-table summary.
- [PHASE_4_RESULTS.md](PHASE_4_RESULTS.md) — End-to-end numbers from Phase 4
  exit: synthetic **30/30 (100%)** with qwen2.5-coder:7b vs llama3.2:3b
  baseline **16/30 (53%)**. Plan B activated.
- [PHASE_4_CLEANUP.md](PHASE_4_CLEANUP.md) — Four follow-up fixes:
  list-qtype golden cases (now 32 total), CLI exit code, CLEVR loader
  rewrite to load from disk, new tests.

## Phase 4.1–4.3 — CLEVR tuning
- [PHASE_4_TUNING.md](PHASE_4_TUNING.md) — Three sequential tuning sweeps,
  each documented at the time it ran:
  - **4.1** CLIP prompt ensemble (CLEVR-aware phrasing) → 52% → 56%.
  - **4.2** OWL-ViT score-threshold / NMS grid → aborted at 56%
    (no configuration beat baseline and passed the synthetic gate).
  - **4.3** OWL-ViT detection-prompt rephrasing → aborted at 56%
    (both variants dropped to 46% — gray/brown color recall regressed).

## Phase 5 — Repository presentation
- [PHASE_5A_REVIEW.md](PHASE_5A_REVIEW.md) — Reviewer report for the cleanup
  commit (file layout, test status). Written by the post-cleanup reviewer.
- [PHASE_5B_REVIEW.md](PHASE_5B_REVIEW.md) — Reviewer report for the README,
  written by a hiring-manager-simulator reviewer.
- [PHASE_5_SUMMARY.md](PHASE_5_SUMMARY.md) — Final hand-off summary.
- [DEPLOY_SPACES.md](DEPLOY_SPACES.md) — HuggingFace Spaces deployment steps.
