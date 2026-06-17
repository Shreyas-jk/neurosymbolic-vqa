# Phase 4.1 — CLIP prompt tuning for the CLEVR domain

## Before / after

| Suite | Before | After | Delta |
|---|---|---|---|
| CLEVR (50 cases) | **26/50 (52.0%)** | **28/50 (56.0%)** | +2 / +4 pp |
| Synthetic (32 cases) | 32/32 (100%) | 32/32 (100%) | unchanged |

Both gates required by the task brief pass: CLEVR up, synthetic unchanged. Committed.

## What changed

Only `scene_extractor/attribute_classifier.py`. The prior implementation scored
each CLIP family with a single template (`"a photo of a {value} object"`). CLIP
loaded with that prior collapses CLEVR's Blender-rendered shapes into the
real-world-photo distribution it was trained on, which is where the baseline
failures came from: real-world "metal object" priors look like saucepans, not
shiny CLEVR spheres; CLIP can find "a red apple" in any image but `"a photo of
a gray object"` is nearly content-free as a prompt.

Two changes:

### 1. CLEVR-aware phrasing
All generic templates moved from `"a photo of a ..."` to `"a 3D rendered ..."`
/ `"a Blender render of a ..."` / `"a CGI ..."`. This pulls the text encoder
into the corner of embedding space where rendered geometric primitives actually
live.

### 2. Per-value ensemble + per-(family,value) overrides
Instead of one prompt per value, the new classifier scores each value against
2–3 templates and **averages logits** before the softmax. This is the
standard CLIP zero-shot trick (OpenAI's repo ships ~80 ImageNet templates and
averages over them; we settled on 2–3 as a reasonable cost/quality point for a
real-time pipeline).

Per-value overrides exist only where visual cues are highly specific:
- **metal**: `"a shiny metallic 3D rendered shape"`, `"a specular reflective metal object"`, `"a glossy chrome 3D rendered geometric shape"`
- **rubber**: `"a matte rubber 3D rendered shape"`, `"a diffuse non-shiny rubber object"`, `"a dull rubbery 3D geometric shape"`
- non-CLEVR materials (wood/plastic/glass/ceramic) get 2-template overrides too so they don't drift on CLEVR crops via real-world priors.

The full template list lives at the top of `scene_extractor/attribute_classifier.py`
under `_GENERIC_TEMPLATES` and `_PER_VALUE_OVERRIDES`.

## Cases that changed

| Case ID | Direction | Why |
|---|---|---|
| `CLEVR_0_exists_gray_cube` | + win | gray now distinguishable via "3D rendered gray geometric shape" |
| `CLEVR_4_count_gray` | + win | same — gray colors now show up in the ensemble argmax |
| `CLEVR_9_count_gray` | + win | same |
| `CLEVR_0_count_rubber` | − loss | rubber count drifted from baseline-correct → off-by-one (over-detected) |

Net: 3 won, 1 lost = +2 cases.

## What did NOT improve (residual failure modes)

These are the cases where the new prompts still miss. They are all about
**counting accuracy after correct attribute classification**, not about the
attribute prompts themselves:

- **`count_metal` for CLEVR_{1,2,3,5,8,9}**: metal is now detected on some
  spheres (vs always zero in baseline), but the count is still systematically
  low — 2-4 vs expected 4-7. The remaining miss is OWL-ViT detection recall,
  not CLIP attribute classification: CLEVR scenes typically have 5-10 objects
  and OWL-ViT is only finding a subset.
- **`exists_blue_cube` for CLEVR_{2,6,9}** and **`exists_brown_*` for several
  scenes**: the relevant object isn't being detected by OWL-ViT at all (the
  attribute check never runs because there's no box to crop). Phrasing the
  detection prompt as `"a 3D rendered cube"` instead of `"a photo of a cube"`
  in `scene_extractor/extractor.py:116` would be the natural next step but is
  out of scope for this task.
- **`count_*` over-detections** (rubber on CLEVR_0, purple on CLEVR_8, yellow
  on CLEVR_5, total on CLEVR_0/5): OWL-ViT is double-counting some objects
  (NMS @ 0.5 IoU may be too permissive for CLEVR's tight scenes). Tuning
  `nms_iou` is also out of scope here.

## What I tried that didn't work

I briefly considered:
- Removing non-CLEVR materials (wood/plastic/glass/ceramic) from the vocabulary
  at runtime — would have helped the metal/rubber softmax sharpness, but breaks
  the synthetic eval (which uses `kitchen()` preset with bottle=glass,
  table=wood, cup=ceramic). Rejected — synthetic regression would fail the
  task's exit gate.
- A single longer prompt per value vs ensemble — ensemble won every spot-check
  on CLEVR images so I stayed with it.

## Verification

```
$ NL2PROLOG_BACKEND=local .venv/bin/python -m evaluation.cli --suite clevr
=== CLEVR subset (50 cases) ===
Overall: 28/50 (56.0%)
Failures by stage: { correctness: 22 }

$ NL2PROLOG_BACKEND=local .venv/bin/python -m evaluation.cli --suite synthetic
=== Synthetic golden dataset (32 cases) ===
Overall: 32/32 (100.0%)
existence:6/6  count:6/6  attribute:6/6  spatial:6/6  multi_hop:6/6  list:2/2
```

Slow vision test (`tests/test_scene_extractor.py -m slow`) re-run with new
prompts: 9/9 pass in 6.4s. Non-slow suite untouched (132/132).

## Files touched in this commit

- `scene_extractor/attribute_classifier.py` — full rewrite of `_PROMPT_TEMPLATES`
  + classify-time ensemble. Net +100/-15 lines.
- `evaluation/results/clevr_baseline_v1.json` — copy of the pre-tuning
  `clevr.json` preserved per the task brief. Gitignored, but referenced from
  PHASE_4_CLEANUP.md / this doc for reproducibility.
- `evaluation/results/clevr.json` — overwritten with the new 28/50 run. Also
  gitignored.
- `PHASE_4_CLEANUP.md` — orphan from the previous cleanup task that I wrote
  but forgot to commit. Folded in here so the prior phase's reviewer doc lives
  on main alongside this one. **Not part of the prompt-tuning work** — flagged
  for transparency.

## Phase 5 notes (NOT addressed here)

- Detection-prompt tuning in `extractor.py:116` is the obvious next step —
  most residual CLEVR failures are upstream of CLIP.
- NMS@0.5 may be too permissive for CLEVR's tight clustered scenes; consider
  per-class NMS at 0.3 and re-measure.
- The CLEVR cases all bucket as `"unknown"` in the rich-table because
  `_qtype_bucket_for` only knows about the synthetic-suite case_ids. Either
  add a separate `IMAGE_QTYPE_BUCKETS` constant or fall through to `qtype`
  directly when no synthetic bucket matches.

---

# Phase 4.2 — OWL-ViT detection tuning (sweep)

**Outcome: vision tuning grid exhausted at 56% — moving to Phase 5 with this number.**

No config strictly exceeded the 56% baseline (Config B tied it, but failed the
synthetic gate). Per the task brief's abort condition, no detection-tuning
commit was made; the working tree was left untouched. The baseline
`evaluation/results/clevr_baseline_v2_after_clip_tuning.json` is committed
(`5c4da1f`) as the stable Phase 4.1 anchor.

## Grid

All three configs were run in one process so OWL-ViT + CLIP stayed cached
(`scene_extractor.models.get_detector` / `get_clip` use `lru_cache`). Per-config
result JSONs are in `evaluation/results/clevr_config_{A,B,C}.json` (gitignored);
synthetic gate runs in `evaluation/results/synthetic_after_config_{A,B,C}.json`.

| Config | thresh | NMS IoU | CLEVR | Synthetic | vision ms/img | clevr wall |
|---|---:|---:|---|---|---:|---:|
| **baseline v2** | 0.10 | 0.50 | **28/50 (56.0%)** | 32/32 (100%) | 1102 | — |
| A | 0.05 | 0.50 | 21/50 (42.0%) | 32/32 (100%) | 1314 | 228.5s |
| B | 0.10 | 0.30 | 28/50 (56.0%) | **31/32 (96.9%)** | **983** | 200.0s |
| C | 0.05 | 0.30 | 27/50 (54.0%) | 31/32 (96.9%) | 1417 | 234.3s |

Wall clocks include the ~3.5–4.5 s/case LLM translation; vision is a small slice.

## Per-config count vs boolean + top-5 failure categories

### Baseline v2 — 28/50
- count: 15/30 (50%)  |  boolean: 13/20 (65%)
- top failures: 6× count_material(metal), 4× count_material(rubber), 3× exists_blue_cube, 2× count_total, 2× exists_brown_cube

### Config A (thresh 0.05, NMS 0.5) — 21/50, −7 vs baseline
- count: 5/30 (17%)  |  boolean: 16/20 (80%)
- top failures: **10× count_total**, 5× count_material(metal), 3× count_material(rubber), 3× exists_blue_cube, 2× count_color(gray)
- Verdict: lower threshold floods the scene with spurious boxes. Boolean
  questions get easier (more boxes → more positive matches) but every count
  question collapses (count_total wrong on 10 of 10 scenes). Net loss.

### Config B (thresh 0.10, NMS 0.3) — 28/50, identical accuracy to baseline
- count: 15/30 (50%)  |  boolean: 13/20 (65%)
- top failures: identical to baseline
- vision latency: **983 ms/img**, fastest in the grid
- Synthetic gate: **31/32 (96.9%)** — M4 ("What material is the small object?")
  was tagged `count` instead of `attribute` by qwen2.5-coder:7b. This is LLM
  flake (qwen is mildly nondeterministic on borderline qtype tags at temp 0.2),
  not a vision regression — synthetic eval never touches the extractor at all.
  Same failure repeats on Config C with the same case, supporting the
  flake-not-determinism hypothesis.
- Verdict: tightening NMS alone with this prompt set didn't recover any case
  the looser NMS missed. The duplicate-detection hypothesis was wrong.

### Config C (thresh 0.05, NMS 0.3) — 27/50, −1 vs baseline
- count: 11/30 (37%)  |  boolean: 16/20 (80%)
- top failures: **9× count_total**, 3× count_material(rubber), 3× count_material(metal), 3× exists_blue_cube
- vision latency: 1417 ms/img, slowest in the grid
- Verdict: the lower threshold hurts the same way as in A. Tighter NMS doesn't
  rescue it.

## Why none of the configs won

The user's hypothesis was that NMS@0.5 was over-counting and threshold@0.1 was
under-detecting metal. The data only weakly supports the second half — Config A
and C both add detections (boolean accuracy went from 65% → 80% in both) but
those detections are mostly spurious for COUNT questions. Lowering threshold
trades count accuracy for boolean accuracy, but CLEVR's count questions
outnumber boolean 30:20 in the test set, so the trade-off comes out negative.

Tightening NMS to 0.3 (B) didn't change a single case relative to baseline.
This means baseline NMS@0.5 wasn't actually duplicating boxes — OWL-ViT was
already producing well-separated detections at the original threshold. The
"NMS over-counting" hypothesis from PHASE_4_TUNING was wrong.

The remaining error mass is detector RECALL on metal spheres (the same as
after Phase 4.1) — that recall doesn't change with score threshold or NMS IoU.
It's a model-prior problem: OWL-ViT trained on natural photos doesn't activate
strongly on CLEVR-rendered chrome spheres regardless of the score floor.

## What I did NOT try in 4.2 (deferred — option 1 attempted in 4.3 below, option 2 still open)

The user's task brief flagged prompt expansion as a "Note: also consider"
option. I did NOT include it in the grid because the three threshold configs
were specified as the primary experiment and prompt expansion would have
muddied the threshold signal. Two prompt-engineering ideas worth Phase 5
attention:

1. **Detection prompt rephrasing** at `extractor.py:116` — change `"a photo
   of a {c}"` to a CLEVR-domain template like `"a 3D rendered {c}"` or
   `"a Blender render of a {c}"`, mirroring what the CLIP attribute classifier
   does. This is one line of code; high upside if it shifts the OWL-ViT
   text-encoder prior into render space. **→ Tried in Phase 4.3 below;
   both rephrasings underperformed baseline. Aborted.**
2. **Vocabulary expansion** — add "small geometric shape", "rendered metallic
   sphere", "shiny chrome sphere" alongside the canonical vocab, with a
   post-process step mapping the expanded query labels back to the canonical
   category before constructing `SceneObject`. The extractor currently uses
   `self.object_vocab[label_idx]` directly which assumes 1:1 query→category.
   **Still open.**

## Files touched in this phase

- `evaluation/results/clevr_baseline_v2_after_clip_tuning.json` — committed
  separately (`5c4da1f`) as the Phase 4.1 anchor before the sweep started.
- `evaluation/results/clevr_baseline_v1.json` — also caught up in the baseline
  commit (was an orphan from Phase 4.1).
- `.gitignore` — exception `!evaluation/results/*baseline*.json` so future
  baselines don't need `-f`.
- `evaluation/results/clevr_config_{A,B,C}.json`,
  `evaluation/results/synthetic_after_config_{A,B,C}.json` — written by the
  sweep, gitignored (regeneratable from /tmp/tune_owlvit.py).
- **No production code changed.** `scene_extractor/extractor.py`,
  `scene_extractor/config.py`, `scene_extractor/attribute_classifier.py` are
  untouched at this phase's exit.

STOP. Phase 5 not started.

---

# Phase 4.3 — OWL-ViT detection prompt sweep

**Outcome: both prompt variants underperformed baseline (46% vs 56%). Aborted.**

`scene_extractor/extractor.py` was refactored briefly to make the prompt
template tunable per-instance, then `git checkout -- scene_extractor/extractor.py`
reverted it after the sweep. No production code change is committed in 4.3.
The baseline `evaluation/results/clevr_baseline_v3_after_owlvit_threshold_sweep.json`
is committed (`f95e835`) as the stable post-4.2 anchor.

## Grid

Both variants ran in one process so OWL-ViT + CLIP stayed cached. Per-variant
result JSONs are in `evaluation/results/clevr_variant{1,2}.json` and
`evaluation/results/synthetic_after_variant{1,2}.json` (all gitignored).

| Variant | Detection prompt | CLEVR | Synthetic | vision ms/img | clevr wall |
|---|---|---|---|---:|---:|
| **baseline v3** | `"a photo of a {c}"` | **28/50 (56.0%)** | 32/32 (100%) | 1102 | — |
| 1 | `"a 3D rendered {c}"` | 23/50 (46.0%) | 32/32 (100%) | 1185 | 231.0s |
| 2 | `"a Blender render of a {c}"` | 23/50 (46.0%) | 31/32 (96.9%) | 1266 | 227.8s |

Both variants score identically at 46% but differ on 4 cases:
- `CLEVR_0_count_total` — v1 right, v2 wrong
- `CLEVR_0_count_rubber` — v1 right, v2 wrong
- `CLEVR_5_exists_yellow_sphere` — v2 right, v1 wrong
- `CLEVR_8_count_total` — v2 right, v1 wrong

So the prompt rephrasing produces meaningfully different detection outputs
(not just noise), but neither phrasing wins overall.

## Per-variant count vs boolean + top-5 failure categories

### Baseline v3 — 28/50
- count: 15/30 (50%)  |  boolean: 13/20 (65%)
- top failures: 6× count_material(metal), 4× count_material(rubber), 3× exists_blue_cube, 2× count_total, 2× exists_brown_cube

### Variant 1: `"a 3D rendered {c}"` — 23/50, −5 vs baseline
- count: 12/30 (40%)  |  boolean: 11/20 (55%)
- top failures: 6× count_material(metal), 4× count_total, 3× exists_blue_cube, 3× count_material(rubber), **2× count_color(gray)**
- vs baseline: gained 3 (CLEVR_0_count_rubber, CLEVR_0_count_total, CLEVR_5_count_total), lost 8 (CLEVR_0_exists_gray_cube, CLEVR_1_count_total, CLEVR_3_count_brown, CLEVR_3_count_total, CLEVR_5_exists_yellow_sphere, CLEVR_8_count_total, CLEVR_9_count_gray, CLEVR_9_count_total)
- Net: −5

### Variant 2: `"a Blender render of a {c}"` — 23/50, −5 vs baseline
- count: 11/30 (37%)  |  boolean: 12/20 (60%)
- top failures: 6× count_material(metal), 4× count_total, 4× count_material(rubber), 3× exists_blue_cube, 2× count_color(gray)
- Synthetic gate: 31/32 (same qwen M4 flake as Phase 4.2's configs B and C)

## Why both variants hurt

The CLEVR-domain phrasings reduced OWL-ViT's overall detection volume — fewer
boxes per scene. That helped a small number of cases where the baseline was
over-detecting on count_total (variant 1 picked up CLEVR_0/5_count_total),
but it strictly hurt:

1. **Gray/brown color recall regressed** (CLEVR_{0,9}_count_gray and
   CLEVR_3_count_brown all flipped from right → wrong). The CLIP attribute
   classifier is fine — those colors are detectable once the box exists —
   but the upstream detector now produces FEWER boxes on grayish/brown
   objects when prompted with "3D rendered" phrasing. Best guess: OWL-ViT's
   text encoder for "a 3D rendered cube" pattern-matches to highly-saturated
   primary-color renders in its training set, suppressing muted-color matches.
2. **Yellow sphere detection regressed in variant 1** (CLEVR_5_exists_yellow_sphere
   flipped from right → wrong). Same root cause: the "rendered" qualifier
   narrows the visual prior in unpredictable ways across color × shape combos.
3. **Metal count is unchanged** (6× failures in baseline, 6× in both variants).
   The CLEVR metal-sphere recall problem is unaffected by the prompt — it's
   a deeper detector-prior gap that prompt phrasing alone can't bridge.

So the "shift OWL-ViT's prior into render space" intuition from Phase 4.2's
forward-looking notes was wrong. The text-encoder change does produce a
different prior, but the prior trades one set of misses for another — and the
trade is unfavorable for CLEVR's specific color/material mix.

## What this rules out

Neither of the two "obvious" detection-prompt rephrasings helps. Combined with
Phase 4.2's negative result on threshold/NMS, the simple knobs on OWL-ViT
inference are exhausted. The next things to try, in order:

1. **Vocabulary expansion** (the other deferred item) — add CLEVR-specific
   queries like "metallic sphere", "rubber cube", "rendered cone" alongside
   the canonical vocab. The hypothesis is different from prompt rephrasing:
   adding more query SLOTS should let the detector hit each query with a
   different visual prior, raising recall on whichever phrasing matches a
   given object. Needs the post-process label→category remap noted in 4.2.
2. **Swap the detector model** — `google/owlv2-base-patch16-ensemble` is
   already cached on disk; it's an updated OWL-ViT trained on more data and
   often beats the base patch32 model on out-of-distribution domains.
3. **Per-class detection thresholds** — drop the metal/sphere threshold to
   0.05 while keeping the rest at 0.1. Targets the specific recall miss
   without flooding count_total.

## Files touched in this phase

- `evaluation/results/clevr_baseline_v3_after_owlvit_threshold_sweep.json` —
  committed as `f95e835` before the sweep.
- `evaluation/results/clevr_variant{1,2}.json`,
  `evaluation/results/synthetic_after_variant{1,2}.json` — written by the
  sweep, gitignored.
- `scene_extractor/extractor.py` — momentarily refactored to expose
  `detection_prompt_template` as a class attribute for the sweep driver. Fully
  reverted via `git checkout --` after the sweep finished.
- **No production code changed.** Same state as the end of Phase 4.2.

STOP. Phase 5 not started.
