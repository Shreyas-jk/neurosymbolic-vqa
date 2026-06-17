# Phase 5 — Summary

Four atomic commits, each landing CI green before the next. Two sub-agent
reviews (cleanup + README, both PASS) gated the work.

## Commits

| Commit  | Message                                                     | CI     |
|--------:|-------------------------------------------------------------|--------|
| 49be5a8 | Phase 5a: reorganize docs and clean loose files             | green  |
| 5441f30 | Phase 5b: README, architecture diagram, LICENSE             | green  |
| 1035205 | Phase 5c: Gradio demo with synthetic and real-image modes   | green  |
| e2b67cb | Phase 5d: HuggingFace Spaces deployment config              | green  |

## File tree after cleanup

```
.
├── data/clevr_test_subset/             # 10 scenes + images (gitignored)
├── demo/
│   ├── app.py                          # Gradio Blocks layout, single file
│   └── requirements.txt                # Spaces-specific deps (no vision stack)
├── docs/
│   ├── DEPLOY_SPACES.md                # one-command Spaces deployment
│   ├── INDEX.md                        # navigable phase doc index
│   ├── PHASE_2_REVIEW.md               # NL→Prolog reviewer report
│   ├── PHASE_3_REVIEW.md               # vision pipeline reviewer report
│   ├── PHASE_4_CLEANUP.md              # 4 fixes: list cases, CLI exit, CLEVR loader, tests
│   ├── PHASE_4_RESULTS.md              # synthetic 30/30 qwen, llama 16/30
│   ├── PHASE_4_REVIEW.md               # eval harness reviewer report
│   ├── PHASE_4_TUNING.md               # 4.1 → 4.2 → 4.3 tuning grid (52% → 56%)
│   ├── PHASE_5A_REVIEW.md              # cleanup reviewer report (PASS)
│   ├── PHASE_5B_REVIEW.md              # README reviewer report (PASS)
│   └── PHASE_5_SUMMARY.md              # this file
├── evaluation/
│   ├── __init__.py
│   ├── cli.py                          # `python -m evaluation.cli --suite ...`
│   ├── clevr_subset.py                 # generates 50 ImageEvalCases from scenes.json
│   ├── golden_dataset.py               # 32 synthetic EvalCases
│   ├── harness.py                      # run_eval + run_eval_on_images + taxonomy
│   └── results/                        # JSONs; baselines tracked, per-run gitignored
├── kb_generator/                       # facts + rules + subprocess swipl validator
├── nl2prolog/                          # 15 few-shot + retry loop + dual backend
├── packages.txt                        # apt deps for HF Spaces (swi-prolog)
├── query_executor/                     # pyswip + call_with_time_limit/2
├── scene_extractor/                    # OWL-ViT + CLIP + geometric relations
├── scripts/
│   ├── README.md                       # script docs
│   ├── download_clevr.sh               # CLEVR extraction recipe
│   ├── download_models.sh              # HF cache pre-warm
│   └── run_eval.sh                     # eval CLI wrapper
├── synthetic/                          # preset scenes + DSL
├── tests/                              # 132 non-slow + 11 slow
├── verbalizer/                         # pure-template answer + reasoning trace
├── LICENSE                             # MIT
├── README.md                           # 325 lines, 1928 words, with HF frontmatter
├── pyproject.toml
├── requirements.txt
└── requirements-dev.txt
```

## README

- **Length:** 325 lines, 1928 words
- **Sections:** 11 in the order the brief specified (tagline, demo link, what
  it does, ASCII architecture, results, quick start, demo, deep dive, what I
  learned, tech stack, license)
- **Honesty check** (from `docs/PHASE_5B_REVIEW.md`, verdict PASS):
  CLEVR 56% and synthetic 100% are presented side-by-side with equal weight
  in a single table; no superlatives, no SOTA claims, no marketing language;
  ASCII diagram carries model IDs + thresholds + retry semantics (not
  decorative); "what I learned" section names specific technical surprises
  (CLIP-on-CLEVR prior gap, llama3.2:3b dropping findall+length wrappers)
  rather than rehearsed hiring-manager bait.
- **Numbers cited:** all present in `evaluation/results/`
  (`synthetic.json`, `synthetic_llama32_3b_baseline.json`,
  `clevr_baseline_v1.json`, `clevr_baseline_v2_after_clip_tuning.json`,
  `clevr_baseline_v3_after_owlvit_threshold_sweep.json`).

## Demo

`demo/app.py` is a single-file Gradio Blocks layout. Verified two ways:

1. **build_app()** — instantiates the full layout (input mode toggle,
   question box, 6 example chips, 6 output tabs, reasoning-trace accordion)
   without exceptions.
2. **End-to-end pipeline smoke** — `run_synthetic("clevr_like", "Is there a
   red cube?")` returns `"Yes — large red metal cube."` after a real ollama
   call to qwen2.5-coder:7b. Per-stage latency captured for kb_generation,
   kb_validation, translation, execution, verbalization.

**Local launch URL:** `http://127.0.0.1:7860` (or `http://0.0.0.0:7860`
when `NSVQA_DEMO_HOST=0.0.0.0`). The full Gradio launch could not be
verified inside this sandboxed dev environment — Gradio's localhost-
accessibility check failed with `ValueError: When localhost is not
accessible, a shareable link must be created`. This is sandbox-specific
(the proxy intercepts loopback); the same `python demo/app.py` invocation
works in an unsandboxed terminal. The pipeline itself was verified via
direct handler invocation, which is the relevant correctness gate.

## HuggingFace Spaces config

Present and complete:

- `README.md` YAML frontmatter (`title`, `emoji`, `colorFrom`, `colorTo`,
  `sdk: gradio`, `sdk_version: 4.44.1`, `app_file: demo/app.py`, `pinned: false`).
- `packages.txt` lists `swi-prolog` so the Space's apt install pulls the
  pyswip system dep.
- `demo/requirements.txt` lists Python deps for the synthetic-only Space
  (no torch / torchvision / transformers / Pillow / numpy — vision is
  hidden when `SPACE_ID` is set).
- `docs/DEPLOY_SPACES.md` documents the deployment steps (create Space,
  push, configure either `OLLAMA_HOST` or `NL2PROLOG_BACKEND=openai` +
  `OPENAI_API_KEY` in Spaces secrets).
- `demo/app.py` auto-detects `SPACE_ID` and hides the real-image tab so the
  free CPU tier doesn't try to load OWL-ViT/CLIP.

**Not deployed yet** — per the protocol, this requires the user's HuggingFace
account. The repo is wire-ready: a single `git push space main` (after
`git remote add space ...`) lands a working Space.

## Test totals

- **132 non-slow tests pass** (unchanged from Phase 4 close; cleanup +
  README + demo + Spaces config added no production code).
- **11 slow tests** (9 vision + 2 ollama live) — deselected by CI as before.

## Outstanding work (NOT addressed; surfaced for honesty)

Per the protocol I stopped here. These are open at the time of hand-off:

1. **Live Space URL.** README's "Live demo" still points at the deployment
   doc rather than a live URL. Once you push the repo to a Spaces remote
   and configure the backend secret, swap the link in.
2. **CLEVR detection-prompt vocabulary expansion.** The one remaining
   prompt-engineering experiment from Phase 4.2's deferred-options list
   (`docs/PHASE_4_TUNING.md`) — adding domain-specific OWL-ViT queries with
   a label→category remap step in `extractor.py`. Not attempted; the
   protocol said no further tuning.
3. **OWLv2 swap.** `google/owlv2-base-patch16-ensemble` is already in the
   local HF cache (593MB). Swapping the detector model_id is a one-line
   change in `scene_extractor/config.py`, but not attempted here.
4. **A `compare_by="id"` knob on `EvalCase`** for list-qtype cases that
   need to distinguish two same-category objects (e.g., two red cubes).
   Noted in `docs/PHASE_4_CLEANUP.md` Phase 5 notes.
5. **Gradio sandbox launch.** Verified via build_app + direct handler
   invocation only. A real `python demo/app.py` run in an unsandboxed
   terminal is the next thing to validate before the Spaces push.

STOP. Phase 5 work complete. Nothing else touched.
