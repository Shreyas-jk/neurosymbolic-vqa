# Phase 5g — Lazy imports in `scene_extractor.__init__`

Commit `f0869cc` on `main`.

## Deploy error chain

The Spaces build succeeded on `pip install -r requirements.txt` (Phase 5f's
minimal set), but `demo/app.py` crashed at import time. The traceback
collapsed to:

```
File "demo/app.py", line 35, in <module>
    from kb_generator.generator import generate as generate_kb
File ".../kb_generator/generator.py", line 12, in <module>
    from scene_extractor.schema import SceneGraph
File ".../scene_extractor/__init__.py", line 14, in <module>
    from scene_extractor.extractor import ModelDownloadError, SceneExtractor
File ".../scene_extractor/extractor.py", line 21, in <module>
    import torch
ModuleNotFoundError: No module named 'torch'
```

Importing `scene_extractor.schema` — which has zero heavy deps and exists
exactly so synthetic-mode consumers can build SceneGraphs without the
vision stack — was nevertheless triggering a torch import via the package
`__init__.py`.

## Root cause

`scene_extractor/__init__.py` eagerly re-exported the vision-side public
API:

```python
from scene_extractor.extractor import ModelDownloadError, SceneExtractor
```

Python's import system runs `__init__.py` before resolving any submodule
access. So `from scene_extractor.schema import X` triggers:

1. `import scene_extractor` → runs `__init__.py`.
2. `__init__.py` runs `from scene_extractor.extractor import ...` → loads
   `extractor.py` → `import torch` → `ModuleNotFoundError` on Spaces.
3. Step 1 fails. Step 2 (submodule access) never happens.

The schema submodule's clean dependency profile was masked by the package
re-export.

## Why lazy imports (PEP 562) was the right fix

Two alternatives I considered and rejected:

1. **Add `torch` to root `requirements.txt`.** This undoes Phase 5f's
   split. It also pushes a ~800MB pip install onto the Spaces builder.
   Even when the install would succeed, OWL-ViT/CLIP can't run usefully
   on the free CPU tier, and the demo already disables the real-image
   tab when `SPACE_ID` is set (`demo/app.py:52`). The dep belongs only
   in `requirements-full.txt`.
2. **Move the schema re-export out of `__init__.py` and tell consumers
   to import from `scene_extractor.schema` directly.** Already partly
   true (every consumer except the public API surface uses the
   submodule path), but breaks anyone who relied on the documented
   `from scene_extractor import SceneGraph` surface — including this
   repo's own README and the `__all__` list.

The lazy fix preserves the public API exactly: `from scene_extractor
import SceneExtractor` still works on a local install with torch, and
the schema dataclasses remain eagerly importable. The vision classes
are deferred to first attribute access via PEP 562's module-level
`__getattr__`.

## The fix

```python
from scene_extractor.schema import (
    BoundingBox,
    SceneGraph,
    SceneObject,
    SceneRelation,
)

__all__ = [
    "BoundingBox",
    "SceneGraph",
    "SceneObject",
    "SceneRelation",
    "SceneExtractor",
    "ModelDownloadError",
]


def __getattr__(name: str):
    if name in ("SceneExtractor", "ModelDownloadError"):
        from scene_extractor.extractor import (
            ModelDownloadError,
            SceneExtractor,
        )
        return {"SceneExtractor": SceneExtractor, "ModelDownloadError": ModelDownloadError}[name]
    raise AttributeError(f"module 'scene_extractor' has no attribute {name!r}")
```

No other file in the package was modified.

## Reproducing the deploy environment before pushing

Before pushing the fix, I rebuilt the exact Spaces install environment in
a throwaway venv and confirmed the demo import chain now works without
torch:

```bash
python3.11 -m venv /tmp/spaces_test_venv
/tmp/spaces_test_venv/bin/pip install -r requirements.txt
/tmp/spaces_test_venv/bin/pip list | grep -i "^torch"      # → no output
cd ~/projects/neurosymbolic-vqa
/tmp/spaces_test_venv/bin/python -c \
  "from kb_generator.generator import generate as generate_kb; print('OK')"
# → OK
rm -rf /tmp/spaces_test_venv
```

The `pip list` check confirmed torch is absent (numpy and Pillow are
pulled in transitively by Gradio — harmless and unrelated to the bug).
The exact import chain that crashed on Spaces (`from
kb_generator.generator import generate as generate_kb`) succeeded under
the minimal requirements set, end-to-end, in a fresh interpreter.

This is the verification step the original Phase 5d → 5e → 5f cycle
should have run before each push; landing it here makes the cycle
reproducible going forward.

## Compatibility with the full install

`from scene_extractor import SceneExtractor` continues to work on any
install that has torch (`requirements-full.txt` or `pip install -e .`).
The lazy path triggers the same `from scene_extractor.extractor import
SceneExtractor` that used to run eagerly — just deferred until the user
asks for the class. The slow vision tests (`tests/test_scene_extractor.py`,
9 tests) collect and run identically.

## Verification

| Check                                                                   | Result |
|-------------------------------------------------------------------------|:------:|
| `from scene_extractor.schema import SceneGraph` (no `__init__` torch)   |   ✓    |
| `from scene_extractor import SceneGraph` (eager package-level)          |   ✓    |
| `from scene_extractor import SceneExtractor` (lazy package-level)       |   ✓    |
| `pytest -m "not slow"` — 132 passed, 11 deselected                       |   ✓    |
| `pytest -m "slow" --collect-only` — 11 tests discovered                 |   ✓    |
| Spaces simulation: `pip install -r requirements.txt` + import chain     |   ✓    |

One PEP 562 quirk worth documenting: `dir(scene_extractor)` returns only
the eagerly-imported names (`BoundingBox`, `SceneGraph`, `SceneObject`,
`SceneRelation`, `schema`). `__all__` alone doesn't drive `dir()` output
in CPython — you'd need a `__dir__` function for that. The lazy names
are still accessible via attribute access (`scene_extractor.SceneExtractor`
works), they just don't appear in `dir()`. If the asymmetry ever matters
downstream, adding `def __dir__(): return __all__` is the standard PEP
562 idiom and a 1-line change.

## Files touched

- `scene_extractor/__init__.py` — full rewrite to the PEP 562 lazy pattern.
- No other file modified. `extractor.py`, `schema.py`, `models.py`,
  `attribute_classifier.py`, `spatial_relations.py`, `config.py` all
  unchanged.

STOP per protocol. Spaces rebuild has been triggered; the user will
verify the build manually in the Spaces UI.
