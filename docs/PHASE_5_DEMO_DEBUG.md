# Phase 5e — Gradio Demo Debug

## Symptom

The local Gradio demo returned **"Internal Server Error"** (HTTP 500) in the
browser via the `gradio.live` share URL. The terminal showed a recurring
`TypeError: argument of type 'bool' is not iterable` at
`gradio_client/utils.py:863` during `/api` schema introspection.

## Root cause — two independent version incompatibilities

The demo pinned `gradio==4.36.1`, but the active venv had a **modern web
stack** (`fastapi 0.137.1`, `starlette 1.3.1`, `gradio_client 1.0.1`,
`pydantic 2.13`). That pairing is incoherent and breaks in *two* places:

1. **The visible TypeError (API introspection).** The built-in
   `gr.Image(type="filepath")` component — present only on the real-image tab,
   which is why the Spaces/synthetic-only path was unaffected — advertises a
   `FileData` schema whose `meta` property serializes to
   `{"additionalProperties": true}`, i.e. a *boolean* JSON-schema value.
   `gradio_client 1.0.1`'s `_json_schema_to_python_type` recurses into that
   bool and runs `"const" in True` → `TypeError`. Gradio runs this at launch.

2. **The actual browser 500 (page render).** `starlette 1.3.1` changed
   `TemplateResponse` to `(request, name, ...)`, but `gradio 4.36.1`'s
   `routes.py` still calls the legacy `(name, context)` form. Starlette then
   treats the context **dict** as the template name → jinja2
   `TypeError: unhashable type: 'dict'` → 500 on every page load.

The task's hypothesis that a `scene_extractor.schema` pydantic model was at
fault was a **red herring**: the demo already serializes scenes to plain dicts
(`_scene_to_jsonable`), and the synthetic-only path — same pydantic models —
worked fine. The trigger was Gradio's own component schema, not ours.

## Fix

Bug #2 cannot be patched without rewriting Gradio internals, so the
least-invasive *coherent* fix was to move the demo to a modern Gradio that
matches the installed web stack and has the bool-schema guard upstream:

- `demo/app.py`: no code change to the app logic. (An interim monkey-patch of
  `gradio_client.utils._json_schema_to_python_type` fixed bug #1 alone but was
  removed once the upgrade made it dead code.)
- `demo/requirements.txt`: `gradio==4.36.1` → `gradio==5.50.0` (also fixed a
  stray `4.36.1Z` typo that would have failed `pip install`).
- `README.md` frontmatter: `sdk_version: 4.44.1` → `5.50.0` (it disagreed with
  requirements.txt — that mismatch would have re-broken Spaces).

`gradio 5.50.0` pulls `gradio_client 1.14.0` + `starlette 0.52.1`; pip also
nudged `pydantic 2.13.4 → 2.12.3`. `gradio` is imported only by the demo, so
no production module or test is affected.

## Verification

- `pip check` → no broken requirements.
- `app.get_api_info()` → succeeds (no TypeError).
- `curl http://127.0.0.1:7860/` → **HTTP 200** with HTML (was 500).
- Driving `/run_synthetic` via `gradio_client.Client` ran the full stack
  (KB → NL→Prolog via ollama → Prolog exec → verbalize): *"How many metal
  objects are there?"* → **"There are 2 metal objects."** with full trace.
- `pytest` → **143 passed** under the downgraded pydantic; no test touched.

## Portfolio takeaway

The loud error (`bool is not iterable`) was a *symptom* on a side path; the
real 500 was a second, quieter starlette signature break. Pinning only the
top-level package (`gradio`) while leaving its transitive web stack unpinned
let the environment drift into an incompatible combination. Lesson: for a
deployed UI, pin to a *coherent* framework version and keep the deploy manifest
(`requirements.txt`) and the platform manifest (Spaces `sdk_version`) in sync.
