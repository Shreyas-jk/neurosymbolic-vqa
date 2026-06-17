# Phase 5f — `requirements.txt` split (minimal vs full)

Commit `7677c94` on `main`.

## What the deploy error was

After Phase 5e bumped Gradio to `5.50.0` everywhere the demo touches
(`demo/app.py`, `demo/requirements.txt`, README YAML `sdk_version`, root
`pyproject.toml` already pinned via the dependency list), the HuggingFace
Space build still failed because the SPACES BUILDER reads the **root**
`requirements.txt`, and the root file still carried Phase 5c's pin:

```
gradio>=4.40,<5.0
```

That constraint is incompatible with `sdk_version: 5.50.0` in the README
frontmatter. The Spaces builder refused to resolve and the build aborted.

In addition, the root `requirements.txt` carried the full vision stack
(`torch`, `torchvision`, `transformers`, `Pillow`, `numpy`), which the
Spaces free CPU tier can't realistically install in a reasonable time even
when the version pins resolve — and the real-image tab is hidden on Spaces
anyway (`demo/app.py:52` keys off `SPACE_ID`).

## Why a split was the right fix (vs. just bumping the gradio pin)

Two independent reasons to keep root `requirements.txt` minimal:

1. **Version drift between the two surfaces becomes impossible.** With the
   minimal file pinned to `gradio==5.50.0` matching the README YAML, the
   only way to break Spaces is to edit both files together — which is what
   PHASE_5_TUNING.md-style protocols catch in review.
2. **Spaces build time and disk usage stay small.** Torch alone is ~800MB.
   The Spaces free tier times out on large pip installs. Even when it
   doesn't, downloading vision weights on cold start would push the demo
   past Spaces' boot timeout. Hiding the real-image tab when `SPACE_ID` is
   set isn't enough on its own — the deps still install at build time
   unless the file omits them.

A single pin bump would have fixed the immediate conflict but left both
problems open. The split fixes the conflict AND the Spaces compatibility
class of bug going forward.

## New file structure

| File                     | Audience            | Has vision stack? | gradio pin          |
|--------------------------|---------------------|-------------------|---------------------|
| `requirements.txt`       | HF Spaces (build)   | No                | `==5.50.0`          |
| `requirements-full.txt`  | local development   | Yes               | `>=4.40,<5.0` (preserved from Phase 4) |
| `demo/requirements.txt`  | Spaces (Gradio app) | No                | `==5.50.0`          |
| `pyproject.toml` deps    | `pip install -e .`  | Yes               | `>=4.40,<5.0`       |

The root `requirements.txt` and `demo/requirements.txt` now both carry
`gradio==5.50.0` exactly. Any future edit to one must mirror the other
(and the README `sdk_version`); drift was the root cause and is now the
maintenance burden.

`requirements-full.txt` is the previous root `requirements.txt` preserved
verbatim under a new name. `git mv` preserved its history.

`pyproject.toml` was not modified per the brief — it has the full dep set
inline (including vision) and is used only by `pip install -e .` (local
dev / CI), where torch is fine. CI installs via `pip install -e .` and
gets the full stack from `pyproject.toml`, not from either requirements
file. CI's behavior is unchanged.

## Install commands

For HuggingFace Spaces — automatic; the platform runs `pip install -r
requirements.txt` at build time. The split is invisible from the Spaces
side: it just picks up the minimal file.

For local development with the full vision pipeline:

```bash
git clone https://github.com/Shreyas-jk/neurosymbolic-vqa.git
cd neurosymbolic-vqa
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-full.txt
ollama pull qwen2.5-coder:7b
.venv/bin/python -m pytest -m "not slow" -q   # 132 passed
```

For a minimal local install (demo only, no real-image mode):

```bash
pip install -r requirements.txt
```

The README's Quick start section documents both forms with a one-line
explanation.

## GitHub CI confirmed no regression

CI run `27717910056` against commit `7677c94`:

```
completed   success   Split requirements.txt: minimal (Spaces) vs full (local with vision)
                                                        CI     main    push    2m40s
```

CI runs `pip install -r requirements-dev.txt` and `pip install -e .` —
neither references `requirements.txt` or `requirements-full.txt`
directly. The full vision stack is pulled from `pyproject.toml`'s inline
dependencies for `-e .`, which is unchanged. The test suite ran the full
non-slow matrix on both Python 3.11 and 3.12 and reported all green.

Local verification before push:

- `pip install -r requirements-full.txt` succeeded (gradio downgraded to
  4.44.1 locally because the file still has the old `<5.0` constraint —
  this is intentional per the protocol's "preserve as the local dev file"
  instruction).
- `pytest -m "not slow" -q` reported `132 passed, 11 deselected in 2.70s`,
  identical to the pre-split count.
- `python -c "import gradio, pydantic, pyswip, openai"` succeeded against
  the minimal set — confirms `requirements.txt` is sufficient for the
  synthetic-mode demo at import time.

## Spaces build status

The HuggingFace remote push (`e497b98..7677c94 main -> main`) succeeded
and triggers a Spaces rebuild server-side. Build status is not polled
from this side; verify manually at the Space's web UI.

## Files touched

- `requirements.txt` — rewritten with the 6-line minimal set (`gradio==5.50.0`
  plus pydantic, pyswip, openai, ollama, rich).
- `requirements-full.txt` — created from the previous root `requirements.txt`
  via `git mv`, content unchanged.
- `README.md` — Quick start section now documents both install commands and
  explains the split in one sentence.
- `pyproject.toml`, `.github/workflows/ci.yml` — staged but unchanged. CI
  uses `pip install -e .` which sources from `pyproject.toml`'s inline
  deps; no reference to either requirements file exists to update.

STOP per protocol. The user will verify the Spaces build manually.
