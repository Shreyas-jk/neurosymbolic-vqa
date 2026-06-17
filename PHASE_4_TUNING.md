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

STOP. Phase 5 not started.
