# scripts/

Operational helpers — none of these run automatically. Invoke from the repo
root with the project venv activated.

## download_models.sh
Pre-warms the HuggingFace model cache for both vision models used by
`scene_extractor`:
- `google/owlvit-base-patch32` — object detection (~600MB)
- `openai/clip-vit-base-patch32` — attribute scoring (~350MB)

First call hits the network and writes into `~/.cache/huggingface/hub/`.
Subsequent runs are no-ops. Run once after `pip install -r requirements.txt`
so first inference doesn't block on a 1GB download.

```bash
scripts/download_models.sh
```

## download_clevr.sh
Documentation-only — prints the recipe for extracting a 10-scene CLEVR
subset into `data/clevr_test_subset/`. Does NOT auto-download. The CLEVR_v1.0
archive is 18GB; the user should be in control of large downloads.

The script prints the canonical Stanford URL plus the selective `unzip -p`
recipe (scenes.json slice + 10 PNGs). Output is ~50MB on disk total.

```bash
scripts/download_clevr.sh   # prints the recipe; copy-paste to execute
```

## run_eval.sh
Thin wrapper around `python -m evaluation.cli`. Honors `NL2PROLOG_BACKEND`
from the environment (defaults to `local`/ollama per the plan).

```bash
scripts/run_eval.sh synthetic   # 32-triple golden synthetic dataset
scripts/run_eval.sh clevr       # 50-question CLEVR subset (needs data/)
scripts/run_eval.sh all         # both suites
```

Results are written to `evaluation/results/{synthetic,clevr}.json`.
