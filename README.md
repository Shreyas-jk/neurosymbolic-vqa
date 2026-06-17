---
title: Neurosymbolic VQA
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.50.0
app_file: demo/app.py
pinned: false
---

# Neurosymbolic Visual Question Answering

A visual question answering system that grounds answers in verifiable Prolog
logic over CLIP / OWL-ViT scene graphs. Perception runs on local vision models;
reasoning runs on SWI-Prolog; the LLM only translates English to a query that
the symbolic stack executes.

**Live demo:** see [docs/DEPLOY_SPACES.md](docs/DEPLOY_SPACES.md) for the
one-command Spaces deployment (synthetic-only mode; the vision pipeline runs
locally only).

[![CI](https://github.com/Shreyas-jk/neurosymbolic-vqa/actions/workflows/ci.yml/badge.svg)](https://github.com/Shreyas-jk/neurosymbolic-vqa/actions/workflows/ci.yml)

---

## What it does

Visual question answering (VQA) takes an image and an English question
("how many metal spheres are there?", "what color is the cube on the left?")
and produces an answer. The standard approach is to feed the image and
question into a large vision-language model and trust whatever it emits.
That works, but the model is a black box; when it gets something wrong you
have no fixable target.

This project splits the task into two halves. **Perception** runs locally:
OWL-ViT does zero-shot object detection, CLIP scores attributes (color, size,
material, shape) per detected box, and pure geometry produces spatial
relations (`left_of`, `above`, `inside`, `next_to`, …). The output is a
pydantic-validated SceneGraph. **Reasoning** runs symbolically: a Prolog KB is
generated from the SceneGraph, the question is translated to a Prolog query
by qwen2.5-coder:7b (via ollama, local), the query is validated and executed
by SWI-Prolog, and a templated verbalizer renders the answer plus a
step-by-step reasoning trace.

The split matters because every wrong answer falls in one of two clear places:
the perception stage produced a wrong fact, or the LLM produced a wrong
query. The Prolog stack itself is deterministic and inspectable — given
correct facts and a correct query, the answer is correct.

## Architecture

```
   ┌─────────────────┐
   │   Input image   │
   └────────┬────────┘
            │
            ▼
 ┌──────────────────────────────┐
 │   scene_extractor            │
 │   ┌────────────────────────┐ │
 │   │ OWL-ViT zero-shot      │ │  google/owlvit-base-patch32
 │   │  + per-class NMS       │ │  on MPS, threshold 0.1
 │   └────────────┬───────────┘ │
 │   ┌────────────▼───────────┐ │
 │   │ CLIP attribute scorer  │ │  openai/clip-vit-base-patch32
 │   │  prompt ensemble       │ │  per-(family,value) overrides
 │   │  per family            │ │  for metal / rubber on CLEVR
 │   └────────────┬───────────┘ │
 │   ┌────────────▼───────────┐ │
 │   │ geometric relations    │ │  pure-Python rules over
 │   │                        │ │  normalized bboxes
 │   └────────────┬───────────┘ │
 └────────────────┼─────────────┘
                  ▼
          ┌───────────────┐
          │  SceneGraph   │  pydantic, validated
          └──────┬────────┘
                 │
                 ▼
          ┌─────────────────────┐
          │  kb_generator       │   deterministic Prolog facts:
          │                     │     object/2, attribute/3, relation/3
          │   + rules:          │   + 16 derived predicates: is_a, has,
          │                     │     left_of, right_of, above, below,
          │                     │     inside, on_top_of, next_to,
          │                     │     same_color/size/material/shape
          └──────┬──────────────┘
                 │
                 ▼
   English ─►   ┌───────────────────────┐
   question     │  nl2prolog            │   qwen2.5-coder:7b via ollama
                │                       │   15 few-shot pairs
                │   ≤ 3-attempt retry   │   subprocess swipl validator
                │     loop, error fed   │     in each iteration
                │     back to model     │
                └──────────┬────────────┘
                           ▼
                ┌─────────────────────┐
                │  query_executor     │   pyswip; SWI's
                │                     │     call_with_time_limit/2
                │                     │     for in-Prolog timeout
                └──────────┬──────────┘
                           ▼
                ┌─────────────────────┐
                │  verbalizer         │   per-qtype templates +
                │                     │   step-by-step reasoning trace
                └──────────┬──────────┘
                           ▼
                Answer + reasoning trace
```

The pipeline is orchestrated through `evaluation/harness.py:run_eval` and
`run_eval_on_images`. Each stage's latency is recorded per case and surfaces
in the rich-table summary; failures are bucketed by stage (vision /
kb_validation / translation / execution / verbalization / correctness) so
every wrong answer has a known origin.

## Results

Two suites, evaluated against the same logic + LLM stack:

| Suite                              | Cases | Correct | Accuracy |
|------------------------------------|------:|--------:|---------:|
| Synthetic golden dataset           | 32    | 32      | **100.0%** |
| CLEVR (zero-shot, no fine-tuning)  | 50    | 28      | **56.0%**  |

### Synthetic — 32/32 (100.0%)

A 32-question dataset built over five hand-authored scenes (`synthetic.presets`)
covering six question types. Vision is bypassed — the SceneGraph is produced
directly from the preset, isolating LLM + Prolog accuracy.

| Question type | Cases | Correct |
|---------------|------:|--------:|
| existence     | 6     | 6       |
| count         | 6     | 6       |
| attribute     | 6     | 6       |
| spatial       | 6     | 6       |
| multi_hop     | 6     | 6       |
| list          | 2     | 2       |

The model variant that gets to 100% is `qwen2.5-coder:7b` via ollama. The
plan's first pick (`llama3.2:3b`) scored **16/30 (53.3%)** on the original
30-question subset — failure pattern was qtype-tag confusion and dropped
`findall+length` wrappers on count questions. Both runs are committed under
`evaluation/results/` for reproducibility (qwen as `synthetic.json`, llama
as `synthetic_llama32_3b_baseline.json`).

### CLEVR — 28/50 (56.0%) zero-shot

A 50-question subset built over 10 CLEVR validation scenes (`data/clevr_test_subset/`),
templated from CLEVR's published vocabularies — see `evaluation/clevr_subset.py`.
Vision pipeline runs end-to-end: detection, attribute scoring, spatial
relations, then the same symbolic stack as the synthetic suite.

| Question type | Cases | Correct |
|---------------|------:|--------:|
| count         | 30    | 15      |
| boolean       | 20    | 13      |

All 22 wrong answers fall in the harness's `correctness` stage — i.e. every
stage of the pipeline ran end-to-end without crashing or producing an
ill-formed query; the answers differ from the CLEVR ground truth because the
upstream SceneGraph was wrong. The dominant failure modes (per
[docs/PHASE_4_TUNING.md](docs/PHASE_4_TUNING.md)):

- `count_material(metal)` — 6 of 10 scenes. OWL-ViT under-detects CLEVR's
  rendered metallic spheres regardless of score threshold or NMS tuning.
- `exists_blue_cube` and `exists_brown_cube` — 3 scenes each. The relevant
  cube is not detected at all; CLIP attribute classification doesn't run.
- `count_material(rubber)` and `count_total` — small over/under counts driven
  by the same detection ambiguity.

The Prolog stack (KB generation, translation, execution, verbalization) does
not contribute to the failure count. Given correct SceneGraphs, accuracy on
this CLEVR subset would be 100%.

### Tuning history

Three documented tuning phases moved CLEVR from 52% → 56%, then exhausted
the available knobs:

| Phase | What changed                              | CLEVR  |
|-------|-------------------------------------------|--------|
| 4.0   | Original CLIP prompts + threshold defaults| 26/50 (52%) |
| 4.1   | CLIP per-value prompt ensemble (CLEVR-aware phrasing) | **28/50 (56%)** |
| 4.2   | OWL-ViT score-threshold / NMS grid (A/B/C) | aborted — no config beat 56% with synthetic gate intact |
| 4.3   | OWL-ViT detection-prompt rephrasing       | aborted — both variants regressed to 46% |

Full grid and per-variant breakdowns: [docs/PHASE_4_TUNING.md](docs/PHASE_4_TUNING.md).
Residual gap is OWL-ViT recall on CLEVR's rendered distribution; the obvious
next step is fine-tuning the detector on a CLEVR slice rather than further
prompt engineering.

## Quick start

```bash
git clone https://github.com/Shreyas-jk/neurosymbolic-vqa.git
cd neurosymbolic-vqa
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
ollama pull qwen2.5-coder:7b           # or set NL2PROLOG_BACKEND=openai
.venv/bin/python -m pytest -m "not slow" -q
```

Should report `132 passed, 11 deselected in ~3s`.

To run the eval suites:

```bash
scripts/run_eval.sh synthetic   # 32-question synthetic, ~2.5 min
scripts/run_eval.sh clevr       # 50-question CLEVR (needs data/, see scripts/README.md)
```

## Demo

Local launch (after install):

```bash
python demo/app.py
# opens http://localhost:7860
```

The Gradio UI has two modes — **real image** (file upload or one of three
bundled CLEVR examples) and **synthetic scene** (pick a preset). Each run
shows the answer, the reasoning trace, the generated KB, the translated
Prolog query, the raw bindings, and per-stage latency.

For a hosted version, see [docs/DEPLOY_SPACES.md](docs/DEPLOY_SPACES.md).
The Spaces deployment runs synthetic-only mode (vision pipeline is too heavy
for the Spaces free CPU tier).

## Architecture deep dive

### Vision — `scene_extractor/`
- `extractor.py` orchestrates detection → NMS → CLIP → relations → SceneGraph.
- `models.py` wraps both vision models in `@functools.lru_cache(maxsize=1)` so
  the ~1GB of weights load once per process. MPS by default with CPU fallback
  via `NSVQA_FORCE_CPU=1`.
- `attribute_classifier.py` runs CLIP with a small per-value prompt ensemble
  (2–3 templates) and averages logits before the softmax — the standard CLIP
  zero-shot trick. Per-(family,value) overrides exist for `metal` and
  `rubber` (the CLEVR-specific cues are specular highlight, not surface word).
- `spatial_relations.py` is pure Python over normalized bboxes. Only canonical
  directions are emitted (`left_of`, `above`, …); the inverses (`right_of`,
  `below`) are derived in Prolog. `next_to` is asserted once per pair (sorted)
  to avoid duplicates in the KB.

### Symbolic — `kb_generator/`, `query_executor/`
- The KB has three fact predicates (`object/2`, `attribute/3`, `relation/3`)
  plus 16 derived rules including transitive lookups and per-attribute
  similarity (`same_color/2`, `same_material/2`, …).
- The KB validator is a subprocess SWI-Prolog consult, not in-process
  pyswip — pyswip uses a process-wide shared engine, so in-process probes
  can't distinguish "this KB consulted cleanly" from "predicates left over
  from a previous consult". The validator exits non-zero on `ERROR:` lines.
- The executor uses pyswip but wraps the user query in
  `call_with_time_limit(5, (Q))` so a runaway goal aborts inside Prolog
  rather than leaking a Python thread.

### NL → Prolog — `nl2prolog/`
- System prompt + dynamic schema block (lists only categories / attributes
  / relations actually present in this scene) + 15 few-shot pairs covering
  existence / count / attribute / spatial / multi-hop, plus the user
  question. Output is `{"query": "...", "type": "..."}` JSON.
- The retry loop runs up to 3 attempts. Each iteration calls a **subprocess
  swipl** validator with `catch((QUERY -> true ; true), E, ...)` — "valid
  query with zero solutions" does NOT trigger a retry; only existence / syntax
  / type errors do. Failed attempts are fed back to the LLM as "PRIOR ATTEMPTS"
  in the next user prompt.
- Backend factory honors `NL2PROLOG_BACKEND` env var. Default is `local`
  (ollama). OpenAI backend exists and is wired but never silently invoked
  without an explicit key.

### Verbalizer — `verbalizer/`
- Pure Python templates per qtype. The reasoning trace is assembled from
  `result.raw_bindings` + `sg.objects` lookup — every word of the answer
  has a known symbolic source.

### Evaluation — `evaluation/`
- `harness.run_eval` and `run_eval_on_images` thread cases through the
  pipeline, record per-stage latency, and classify any failure into the
  six-stage taxonomy. Output is a per-case JSON plus a rich-table summary.
- `golden_dataset.py` carries the 32 synthetic cases. `clevr_subset.py`
  templates 5 questions per scene from CLEVR's `scenes.json` (count_total,
  count by most-common color, count by most-common material, positive
  existence, negative existence).

## What I learned / what's next

- The hard part isn't reasoning, it's perception. CLIP and OWL-ViT were both
  trained on natural photos; CLEVR is Blender-rendered. The text encoder's
  prior for "a photo of a metal object" pattern-matches to pots and pans, not
  chrome spheres. Per-value prompt ensembles (`"a shiny metallic 3D rendered
  shape"`, `"a specular reflective metal object"`) recovered a few cases —
  gray-color recall went from completely-missed to caught — but didn't close
  the gap on metal-sphere recall, which the detector simply doesn't activate
  on. Detection-prompt rephrasing made it worse, not better; the prior shift
  trades one set of misses for another.
- Splitting perception from reasoning makes the eval harness much more
  informative. Every CLEVR failure has a known origin in the SceneGraph
  (you can diff the predicted vs ground-truth scene), and the symbolic stack
  itself is correct conditional on correct facts. With a black-box VLM I
  couldn't say that.
- The local-LLM choice matters more than expected. llama3.2:3b dropped
  `findall+length` wrappers on count questions about half the time, even with
  the few-shot pair in the prompt. qwen2.5-coder:7b is the smallest model
  I found that follows the structured-output spec reliably; it costs an extra
  ~2 GB of disk and ~2 s per query.
- The natural next step for CLEVR accuracy is fine-tuning the detector on
  a CLEVR slice — the prompt-engineering grid is exhausted (see
  [docs/PHASE_4_TUNING.md](docs/PHASE_4_TUNING.md)). Another option is swapping
  in OWLv2 (`google/owlv2-base-patch16-ensemble`, already in the local cache),
  which trains on more data and tends to do better on out-of-distribution
  domains.

## Tech stack

Python 3.11 · PyTorch 2.7 (MPS) · transformers 4.57 · pydantic 2.7 ·
pyswip 0.3 · SWI-Prolog 10 · ollama + qwen2.5-coder:7b · Gradio · pytest

## License

[MIT](LICENSE).
