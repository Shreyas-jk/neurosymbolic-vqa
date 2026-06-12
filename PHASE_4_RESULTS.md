# Phase 4 — Evaluation Results

**Date run:** 2026-06-11
**Backend:** ollama (local; HARD constraint, no OpenAI calls)
**Pipeline:** synthetic SceneGraph → KB → NL→Prolog → execute → verbalize

## Headline numbers

| Model | Accuracy | Existence | Count | Attribute | Spatial | Multi-hop |
|---|---|---|---|---|---|---|
| **qwen2.5-coder:7b** (final default) | **30/30 (100.0%)** | 6/6 | 6/6 | 6/6 | 6/6 | 6/6 |
| llama3.2:3b (plan's first pick, baseline) | 16/30 (53.3%) | 5/6 | 2/6 | 3/6 | 3/6 | 3/6 |

The plan's Phase 4 exit target was **25/30 (83%) on the 30-triple golden synthetic dataset**. qwen2.5-coder:7b clears it; llama3.2:3b does not — the plan's documented Plan B (Section 6, Open Questions: "if it's worse than expected: bump to qwen2.5-coder:7b") was activated and is now the default in `nl2prolog/local_backend.py`.

CLEVR-subset live eval did NOT run — every CLEVR mirror I tried returned 404 (HF datasets are gated; the Stanford direct link is the 18GB archive). Details in §"CLEVR" below.

---

## Rich-table summary printed to stdout (qwen2.5-coder:7b)

```
=== Synthetic golden dataset (30 cases) ===
Overall: 30/30 (100.0%)
        Accuracy by question type
┏━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━┓
┃ Type      ┃ Correct ┃ Total ┃ Accuracy ┃
┡━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━┩
│ existence │ 6       │ 6     │ 100.0%   │
│ count     │ 6       │ 6     │ 100.0%   │
│ attribute │ 6       │ 6     │ 100.0%   │
│ spatial   │ 6       │ 6     │ 100.0%   │
│ multi_hop │ 6       │ 6     │ 100.0%   │
└───────────┴─────────┴───────┴──────────┘
Mean latency (ms) by stage
┏━━━━━━━━━━━━━━━┳━━━━━━━━━┓
┃ Stage         ┃ Mean ms ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━┩
│ kb_generation │ 0.1     │
│ kb_validation │ 11.8    │
│ translation   │ 4532.4  │
│ execution     │ 2.0     │
│ verbalization │ 0.0     │
└───━━━━━━━━━━━┻━━━━━━━━━┛

No failures table printed — failure_stage_counts is empty.
```

## Per-question-type accuracy breakdown

### qwen2.5-coder:7b (chosen default)
| Bucket | Correct/Total | Notes |
|---|---|---|
| existence | 6/6 | E1–E6 — both attribute-existence and category-existence; empty-scene negative; material existence |
| count | 6/6 | C1–C6 — total, category-filtered, color-filtered, material-filtered; all use the `findall + length` pattern verbatim |
| attribute | 6/6 | A1–A6 — color, size, material lookups by category; including single-object scene |
| spatial | 6/6 | S1–S6 — `left_of`, `right_of`, `next_to`, `on_top_of`, `above`, `below` (the canonical + derived pair all bound the correct object) |
| multi_hop | 6/6 | M1–M6 — chained spatial+attribute, category+material+size, multi-predicate filters |

### llama3.2:3b (baseline)
| Bucket | Correct/Total | Notes |
|---|---|---|
| existence | 5/6 | E5 failed: treated "wood" as a category (`object(_X, wood)`) instead of `attribute(X, material, wood)` |
| count | 2/6 | C1/C3/C4 hit translation retry-exhaust; C5 omitted `findall+length` wrapping |
| attribute | 3/6 | A1/A4 qtype-mismatch (correct query but tagged `boolean`); A3 returned no binding |
| spatial | 3/6 | S2 confused "right of cube" with "left of cube"; S4 wrong qtype tag; S6 retry-exhaust on `below` |
| multi_hop | 3/6 | M3 qtype-mismatch; M4 wrong filter category; M5 garbled findall clause |

## Top failure modes by stage (llama3.2:3b)

| Failure stage | Count | Pattern |
|---|---|---|
| `translation` | 8/30 | 4× retry-exhaust on count/spatial; 4× qtype-mismatch (query OK, type tag wrong) |
| `correctness` | 6/30 | Wrong filter category, missing `findall` wrap, "wood" as category |
| `vision` | 0 | Synthetic mode; no vision pipeline invoked |
| `kb_validation` | 0 | All scenes produced valid KBs |
| `execution` | 0 | All validated queries ran cleanly |
| `verbalization` | 0 | Pure-Python templates never crashed |

The taxonomy correctly isolated **translation** (LLM-side defect) from **correctness** (LLM produced a structurally valid query that yields the wrong answer). This separation is the eval harness's core contribution.

## qwen2.5-coder:7b generated queries (all correct)

```
E1   boolean    object(X, cube), attribute(X, color, red)
E2   boolean    object(X, sphere), attribute(X, color, green)
E3   boolean    object(X, table)
E4   boolean    object(X, cube)
E5   boolean    object(X, _), attribute(X, material, wood)
E6   boolean    object(X, cylinder), attribute(X, material, metal)
C1   count      findall(X, object(X, _), L), length(L, N)
C2   count      findall(X, object(X, cube), L), length(L, N)
C3   count      findall(X, object(X, chair), L), length(L, N)
C4   count      findall(X, (object(X, _), attribute(X, color, black)), L), length(L, N)
C5   count      findall(X, (object(X, _), attribute(X, material, metal)), L), length(L, N)
C6   count      findall(X, (object(X, _), attribute(X, color, green)), L), length(L, N)
A1   attribute  object(X, cube), attribute(X, color, C)
A2   attribute  object(X, sphere), attribute(X, size, S)
A3   attribute  object(X, cylinder), attribute(X, material, M)
A4   attribute  object(X, cube), attribute(X, color, C)
A5   attribute  object(X, bottle), attribute(X, material, M)
A6   attribute  object(X, cube), attribute(X, size, S)
S1   object     object(Y, sphere), left_of(X, Y), object(X, _)
S2   object     object(Y, cube), right_of(X, Y), object(X, _)
S3   object     object(Y, cup), next_to(X, Y), object(X, _), X \= Y
S4   object     object(Y, desk), on_top_of(X, Y), object(X, _)
S5   object     object(Y, cube), above(X, Y), object(X, _)
S6   object     object(Y, cylinder), below(X, Y), object(X, _)
M1   attribute  object(Y, sphere), left_of(X, Y), attribute(X, color, C)
M2   attribute  object(X, cube), attribute(X, material, metal), attribute(X, size, S)
M3   attribute  object(Y, cube), above(X, Y), attribute(X, color, C)
M4   attribute  object(X, _), attribute(X, size, small), attribute(X, material, M)
M5   count      findall(X, (object(X, chair), attribute(X, material, metal)), L), length(L, N)
M6   attribute  object(X, cup), attribute(X, color, C)
```

Every one is well-formed Prolog. Spatial cases exploit the inverse rules from `kb_generator/templates.py` (`right_of`, `below`, `next_to` derived from canonical `left_of`/`above`/`next_to`). Multi-hop cases compose attribute + spatial filters cleanly.

## Full eval/results/synthetic.json (summary section, qwen2.5-coder:7b)

```json
{
  "accuracy": 1.0,
  "by_qtype": {
    "attribute": {"accuracy": 1.0, "correct": 6, "n": 6},
    "count":     {"accuracy": 1.0, "correct": 6, "n": 6},
    "existence": {"accuracy": 1.0, "correct": 6, "n": 6},
    "multi_hop": {"accuracy": 1.0, "correct": 6, "n": 6},
    "spatial":   {"accuracy": 1.0, "correct": 6, "n": 6}
  },
  "correct": 30,
  "failure_stage_counts": {},
  "mean_latency_ms_by_stage": {
    "execution": 1.96,
    "kb_generation": 0.06,
    "kb_validation": 11.77,
    "translation": 4532.40,
    "verbalization": 0.04
  },
  "n": 30
}
```

Full per-case payload is at `evaluation/results/synthetic.json` (gitignored, regeneratable via `scripts/run_eval.sh synthetic`).

## Full eval/results/synthetic_llama32_3b_baseline.json (summary section)

```json
{
  "accuracy": 0.5333333333333333,
  "by_qtype": {
    "attribute": {"accuracy": 0.5, "correct": 3, "n": 6},
    "count":     {"accuracy": 0.333, "correct": 2, "n": 6},
    "existence": {"accuracy": 0.833, "correct": 5, "n": 6},
    "multi_hop": {"accuracy": 0.5, "correct": 3, "n": 6},
    "spatial":   {"accuracy": 0.5, "correct": 3, "n": 6}
  },
  "correct": 16,
  "failure_stage_counts": {
    "translation": 8,
    "correctness": 6
  },
  "n": 30
}
```

## CLEVR — `evaluation/results/clevr.json` was NOT produced

The CLEVR-subset live eval did not run end-to-end. Reasons:

- The harness code (`evaluation/clevr_subset.py` + `evaluation/harness.py:run_eval_on_images`) is fully wired. CLI invocation: `python -m evaluation.cli --suite clevr`.
- I tried two HuggingFace dataset mirrors (`jxie/clevr`, `Multimodal-Fatima/CLEVR_train`); both returned 404 for the specific filenames the subset module requested. HF dataset repositories were either gated or use different file layouts than the legacy CLEVR_v1.0 archive.
- The Stanford CDN serves the full 18GB CLEVR_v1.0.zip from `downloads.cs.stanford.edu` — too large to fetch in an unattended run.
- `iter_cases()` returns `[]` cleanly; the CLI prints "No CLEVR cases available" and exits 0. This is intentional graceful-degradation behavior.

**Implication for Phase 5 / README:** I have no measured CLEVR recall number. The abort condition you specified ("Phase 3 vision pipeline produces <30% object detection recall on the CLEVR test images") could not be verified against actual CLEVR data. The closest evidence we have is Phase 3's slow test (`tests/test_scene_extractor.py`), which detected 3 objects with correct color attributes on a synthetic high-contrast scene built by `tests/fixtures/synth_image.py`.

**Recommendation:** before any README claim about CLEVR, either (a) manually download a few CLEVR images and drop them into `data/clevr_test_subset/`, then rerun `python -m evaluation.cli --suite clevr`; or (b) replace the CLEVR pitch with synthetic-only eval and the Phase 3 vision-pipeline-works demo.

## Decisions made during this phase

1. **Default model bumped to `qwen2.5-coder:7b`** in `nl2prolog/local_backend.py`. Authorized by plan Section 6 ("Plan B if it's worse than expected: bump to qwen2.5-coder:7b"). Evidence: 53% → 100% on the same 30 triples, same harness. Trade-off: 4.7GB vs 2.0GB on disk, ~4.5s vs ~2.7s per translation. Net win for portfolio scope.

2. **Package renamed `eval/` → `evaluation/`** to avoid shadowing the Python `eval()` builtin. The plan's `python -m eval.harness` becomes `python -m evaluation.cli --suite synthetic`. Reflected in `scripts/run_eval.sh` and the CLI module path.

3. **Strict qtype-mismatch semantic kept as a translation failure**. When the LLM emits the right Prolog query but tags it with the wrong `type` field, the harness counts it as `failure_stage="translation"` (not "correctness"). Rationale: the qtype drives verbalizer template selection, so a wrong tag would produce a malformed answer for the end user. The strict semantic also surfaced 4 of llama3.2:3b's 8 translation failures (vs 0 for qwen), which was diagnostically useful.

4. **No threshold tuning was needed.** All defaults from `scene_extractor/config.py` (detection 0.1, NMS 0.5, attribute 0.35, spatial margins 0.05) survived Phase 3's slow test on the synthetic image and were not touched in Phase 4 — the synthetic eval bypasses the vision pipeline entirely.

5. **No prompt or few-shot changes.** The 15 few-shot pairs from Phase 2 (existence×2, count×3, attribute×3, spatial×4, multi-hop×3) are sufficient for qwen2.5-coder:7b to score 30/30 on this dataset. llama3.2:3b's failures suggest the prompt+few-shot combination is adequate; the model capacity was the bottleneck.

## Open issues for Phase 5 (NOT yet started — awaiting your review)

These are surfaced so the README claim doesn't outrun the evidence:

- **CLEVR live numbers absent.** README must either (a) skip the CLEVR pitch, (b) explain the data is fetched manually, or (c) demo on Phase 3's synthetic image only. I did not write this for you.
- **`list`-qtype golden case not exercised end-to-end.** The 30 triples don't include a `list`-typed question. `is_correct(list, ...)` is defined and unit-tested in shape but not run against the LLM. Recommend adding 1–2 list triples in a future patch.
- **Translation latency (4.5s/case for qwen).** For a 30-case eval that's ~2.3 minutes; for a hypothetical 300-case dataset it'd be ~23 minutes. Acceptable for portfolio; flag if scope grows.
- **CI does not run any eval suite.** The `pytest -m "not slow"` matrix only runs the 28 mocked eval-harness tests; it does NOT spin up ollama or score against the live model. The 30/30 number is reproducible on a developer machine with ollama + qwen2.5-coder:7b pulled.

---

## Test totals at Phase 4 close

- **117 non-slow tests passing** (43 Phase 1 + 33 Phase 2 + 13 spatial + 28 eval).
- **11 slow tests** (9 vision via `test_scene_extractor.py`; 2 ollama-live via `test_ollama_live.py`). Slow tests deselected by CI; verified locally.
- **CI green** through Phase 3 (Phase 4 push CI in flight at time of writing).
- **No Phase 1/2/3 deviations regressed.** The Phase 1 subprocess-swipl validator and `call_with_time_limit/2` executor are intact; the Phase 2 retry loop, 15 few-shot prompt block, and dual-backend factory are intact; the Phase 3 OWL-ViT+CLIP pipeline + spatial-relations module are intact.

---

## Phase 4 reviewer verdict

`PHASE_4_REVIEW.md` — PASS. 28/28 eval tests, 117/117 non-slow tests, synthetic eval ran end-to-end producing the 30/30 numbers above. Concerns surfaced (CLEVR mirrors 404, no `list`-qtype case, strict qtype-mismatch semantic) are all non-blocking.

**HARD STOP. Phase 5 NOT started. Awaiting your review of these results.**
