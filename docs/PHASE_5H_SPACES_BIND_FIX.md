# Phase 5h — Spaces bind fix (`0.0.0.0` inside the container)

Commit `848c340` on `main`.

## Symptom

Build passed. Container started. The container log showed Gradio coming up
cleanly:

```
Running on local URL:  http://127.0.0.1:7860, with SSR ⚡
/usr/local/lib/python3.13/site-packages/gradio/blocks.py:2813: UserWarning:
    Setting share=True is not supported on Hugging Face Spaces
To create a public link, set share=True in launch().
Stopping Node.js server...
```

…and then Spaces killed the container 30 minutes later with:

```
Launch timed out, workload was not healthy after 30 min
```

The app was running. Spaces' external health check just couldn't reach it.

## Root cause

`demo/app.py` defaulted `server_name` to `127.0.0.1` from Phase 5c onward
(it was added at the time to dodge a localhost-accessibility quirk on the
dev Mac). Inside a Docker container, `127.0.0.1` is the loopback interface
visible only to processes inside that container. Spaces' reverse proxy and
health checker run *outside* the container — they reach the app through
the published port on `0.0.0.0`, which is exactly the interface Gradio was
NOT bound to.

So:

- Inside the container: `curl http://127.0.0.1:7860` works.
- From the Spaces proxy: connection refused on every interface that's
  actually routable.
- Spaces gives the workload 30 minutes to pass the external health check.
  It never can. Container is killed.

The container log even tells you what's happening — `Running on local URL:
http://127.0.0.1:7860` is the warning sign — but it's easy to miss because
the same line is what Gradio prints on every local launch.

## Why this only surfaces on Spaces

Local development sees `127.0.0.1` work fine because the developer's
browser IS running on the same host as Gradio. Loopback is enough. Any
deploy target that puts the app behind a proxy on a different network
namespace (Docker, Spaces, Cloud Run, k8s) needs `0.0.0.0`. The bug is
invisible until the first such deploy.

This is a classic Spaces gotcha — worth flagging anywhere else this
project pattern (Gradio inside Spaces) appears.

## Fix

```python
default_host = "0.0.0.0" if os.environ.get("SPACE_ID") else "127.0.0.1"
host = os.environ.get("NSVQA_DEMO_HOST", default_host)
app.launch(server_name=host, server_port=7860, show_api=False)
```

Two changes from the previous launch call:

1. **Default host is now conditional on `SPACE_ID`.** Spaces sets that env
   var inside every container; we already use it elsewhere in the same
   file (line 52) to hide the real-image tab. Reusing the same signal for
   the bind address keeps the Spaces-vs-local switch in one place.
   `NSVQA_DEMO_HOST` still wins if set, so the user can override either
   way.
2. **`share=True` removed.** It served a purpose for one early Mac dev
   session before Phase 5e pinned `gradio==5.50.0` — at the time, an older
   Gradio version was failing its localhost-accessibility self-check and
   refused to launch without `share=True`. The pin fixed the underlying
   cause. Meanwhile Spaces explicitly warns the flag is unsupported
   there. Keeping it does no harm but adds a misleading warning to every
   Spaces log; dropping it cleans both surfaces.

## Verification

| Check                                                           | Result |
|-----------------------------------------------------------------|:------:|
| `grep -A3 "app.launch"` — no `share=`, server_name=host         |   ✓    |
| `from demo.app import build_app, get_backend` — backend resolves|   ✓    |
| SPACE_ID set → host=0.0.0.0; unset → host=127.0.0.1             |   ✓    |
| `pytest -m "not slow"` — 132 passed, 11 deselected               |   ✓    |
| Single `app.launch` call (no duplicates introduced)              |   ✓    |

## What was NOT touched

- No production module (`scene_extractor/`, `nl2prolog/`, `kb_generator/`,
  `query_executor/`, `verbalizer/`, `synthetic/`, `evaluation/`) modified.
- No test file modified.
- No `requirements.txt`, `requirements-full.txt`, or `pyproject.toml`
  change — Phase 5f and 5g shipped those correctly.
- README YAML frontmatter unchanged.

Only `demo/app.py` was edited, only the 5-line launch block at the bottom
of `main()`.

## Files touched

- `demo/app.py` — launch block in `main()`, lines 449-455.

STOP per protocol. Spaces rebuild has been triggered by the push; the
user will verify the rebuilt Space manually.
