"""Gradio demo for the neurosymbolic VQA pipeline.

Two input modes (Real image / Synthetic scene) feed the same downstream
stack: KB generation → NL→Prolog translation → Prolog execution →
verbalized answer + reasoning trace. The right-hand tabs surface each
intermediate artifact so the explainability story is in the UI, not just
the docs.

The real-image tab is hidden when SPACE_ID is set (HF Spaces free CPU is
too slow for OWL-ViT + CLIP) — synthetic-only mode is the Spaces target.

Imports only public surfaces from the production modules; the demo does
not modify or extend any of them.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

# Allow `python demo/app.py` from the repo root to find the project modules.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import gradio as gr

# Public surfaces from the production modules — none of these are
# demo-specific and none are modified to make the demo work.
from kb_generator.generator import generate as generate_kb
from kb_generator.validator import validate as validate_kb
from nl2prolog import NoBackendAvailableError, get_backend
from nl2prolog.schema_builder import build_schema
from nl2prolog.translator import (
    TranslationError,
    TranslatorPipeline,
)
from query_executor.executor import QueryExecutor
from scene_extractor.schema import SceneGraph
from synthetic import presets
from verbalizer.verbalizer import verbalize

REPO_ROOT = Path(__file__).resolve().parents[1]
CLEVR_IMAGES_DIR = REPO_ROOT / "data" / "clevr_test_subset" / "images"

# Spaces detection — if SPACE_ID is in env, the real-image tab is hidden.
IS_SPACES = bool(os.environ.get("SPACE_ID"))

PRESET_NAMES: tuple[str, ...] = (
    "clevr_like",
    "kitchen",
    "office",
    "single_object",
    "empty_scene",
)

# Six pre-built questions covering each qtype the pipeline supports.
EXAMPLE_QUESTIONS: tuple[str, ...] = (
    "Is there a red cube?",
    "How many metal objects are there?",
    "What color is the sphere?",
    "What is to the left of the sphere?",
    "What color is the object to the left of the sphere?",
    "List all metal objects.",
)

# Three bundled CLEVR examples (only shown when not on Spaces).
CLEVR_EXAMPLES: list[str] = [
    str(CLEVR_IMAGES_DIR / "CLEVR_val_000000.png"),
    str(CLEVR_IMAGES_DIR / "CLEVR_val_000003.png"),
    str(CLEVR_IMAGES_DIR / "CLEVR_val_000007.png"),
]

# Cached CLEVR scene graphs precomputed by scripts/precompute_clevr_scenes.py.
# Available on both local AND Spaces — the vision pipeline ran offline before
# deploy; only the symbolic reasoning stages run live.
CACHED_CLEVR_DIR = Path(__file__).resolve().parent / "cached_scenes"
CACHED_CLEVR_NAMES: tuple[str, ...] = (
    "CLEVR_val_000000",
    "CLEVR_val_000001",
    "CLEVR_val_000002",
    "CLEVR_val_000003",
    "CLEVR_val_000004",
)
# Original vision extraction wall time on the user's M4, recorded during the
# precompute run — surfaced in the latency table so reviewers can see what
# was traded for the cache hit.
CACHED_VISION_WALL_MS: dict[str, float] = {
    "CLEVR_val_000000": 6700.0,
    "CLEVR_val_000001": 1200.0,
    "CLEVR_val_000002": 800.0,
    "CLEVR_val_000003": 700.0,
    "CLEVR_val_000004": 600.0,
}


# ----- Backend + pipeline construction (lazy, cached) ---------------------


_PIPELINE: Optional[TranslatorPipeline] = None
_EXECUTOR: Optional[QueryExecutor] = None


def _get_pipeline() -> TranslatorPipeline:
    global _PIPELINE
    if _PIPELINE is None:
        backend = get_backend(allow_fallback=False)
        _PIPELINE = TranslatorPipeline(backend=backend, max_attempts=3)
    return _PIPELINE


def _get_executor() -> QueryExecutor:
    global _EXECUTOR
    if _EXECUTOR is None:
        _EXECUTOR = QueryExecutor()
    return _EXECUTOR


# ----- Scene resolution ---------------------------------------------------


def _scene_from_preset(name: str) -> SceneGraph:
    if name not in presets.ALL_PRESETS:
        raise ValueError(f"unknown preset {name!r}")
    return presets.ALL_PRESETS[name]()


def _scene_from_image(image_path: str) -> SceneGraph:
    # Lazy import so import-time on Spaces (synthetic-only) doesn't pay for
    # the vision deps.
    from scene_extractor.extractor import SceneExtractor

    return SceneExtractor().extract(image_path)


def _scene_from_cached(name: str) -> SceneGraph:
    """Load a precomputed SceneGraph JSON. Vision ran offline before deploy."""
    path = CACHED_CLEVR_DIR / f"{name}.json"
    return SceneGraph.model_validate_json(path.read_text())


def _cached_clevr_png(name: str) -> Optional[str]:
    """Return the PNG path for a cached CLEVR name, or None if missing."""
    p = CACHED_CLEVR_DIR / f"{name}.png"
    return str(p) if p.exists() else None


# ----- Per-stage runner — same shape used by the eval harness -------------


def _run_pipeline(scene: SceneGraph, question: str) -> dict[str, Any]:
    latency: dict[str, float] = {}
    extras: dict[str, Any] = {}

    t0 = time.perf_counter()
    kb = generate_kb(scene)
    latency["kb_generation_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    vres = validate_kb(kb.source)
    latency["kb_validation_ms"] = (time.perf_counter() - t0) * 1000
    if not vres.ok:
        return {
            "ok": False,
            "stage": "kb_validation",
            "error": "; ".join(vres.errors),
            "scene": scene,
            "kb_source": kb.source,
            "latency": latency,
        }

    schema_block = build_schema(kb)
    pipeline = _get_pipeline()

    t0 = time.perf_counter()
    try:
        translation = pipeline.translate(question, kb.source, schema_block)
    except TranslationError as exc:
        return {
            "ok": False,
            "stage": "translation",
            "error": f"{exc} (attempts: {len(exc.attempts)})",
            "scene": scene,
            "kb_source": kb.source,
            "schema": schema_block,
            "latency": {
                **latency,
                "translation_ms": (time.perf_counter() - t0) * 1000,
            },
        }
    latency["translation_ms"] = (time.perf_counter() - t0) * 1000
    extras["attempts"] = len(translation.attempts) + 1

    t0 = time.perf_counter()
    qresult = _get_executor().run(kb.source, translation.parsed)
    latency["execution_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    bundle = verbalize(question, translation.parsed, qresult, scene)
    latency["verbalization_ms"] = (time.perf_counter() - t0) * 1000

    return {
        "ok": True,
        "stage": None,
        "scene": scene,
        "kb_source": kb.source,
        "schema": schema_block,
        "parsed": translation.parsed,
        "qresult": qresult,
        "bundle": bundle,
        "latency": latency,
        "extras": extras,
    }


# ----- Gradio event handlers ----------------------------------------------


def _scene_to_jsonable(scene: SceneGraph) -> dict[str, Any]:
    return {
        "image_path": scene.image_path,
        "objects": [
            {
                "id": o.id,
                "category": o.category,
                "confidence": o.confidence,
                "bbox": {
                    "x1": o.bbox.x1,
                    "y1": o.bbox.y1,
                    "x2": o.bbox.x2,
                    "y2": o.bbox.y2,
                },
                "attributes": o.attributes,
                "attribute_confidences": o.attribute_confidences,
            }
            for o in scene.objects
        ],
        "relations": [
            {"subject_id": r.subject_id, "predicate": r.predicate, "object_id": r.object_id}
            for r in scene.relations
        ],
        "extraction_time_ms": scene.extraction_time_ms,
    }


def _latency_table(
    latency: dict[str, float],
    vision_ms: Optional[float],
    cached_original_vision_ms: Optional[float] = None,
) -> list[list[Any]]:
    rows = []
    if cached_original_vision_ms is not None:
        # Cached-mode: vision_ms is the JSON-load time; show that plus the
        # original M4 wall time recorded at precompute, so reviewers can see
        # what the cache traded for.
        rows.append([
            "vision (cached load)",
            f"{vision_ms:.1f}" if vision_ms is not None else "—",
        ])
        rows.append([
            "  ↪ original vision wall (M4)",
            f"{cached_original_vision_ms:.0f}",
        ])
    elif vision_ms is not None:
        rows.append(["vision", f"{vision_ms:.1f}"])
    for k in ("kb_generation_ms", "kb_validation_ms", "translation_ms", "execution_ms", "verbalization_ms"):
        if k in latency:
            rows.append([k.removesuffix("_ms"), f"{latency[k]:.1f}"])
    return rows


def _empty_outputs(message: str, scene_json: dict | None = None) -> tuple:
    return (
        message,              # answer textbox
        "",                   # reasoning trace
        scene_json or {},     # scene graph JSON
        "",                   # KB code
        "",                   # translated query
        "",                   # raw bindings
        [],                   # latency table
    )


def run_synthetic(preset_name: str, question: str) -> tuple:
    if not question.strip():
        return _empty_outputs("Enter a question first.")
    try:
        scene = _scene_from_preset(preset_name)
    except Exception as exc:
        return _empty_outputs(f"Failed to load preset: {exc}")
    result = _run_pipeline(scene, question.strip())
    return _format_outputs(result, vision_ms=None)


def run_real_image(image_path: str | None, question: str) -> tuple:
    if not image_path:
        return _empty_outputs("Upload an image or click an example first.")
    if not question.strip():
        return _empty_outputs("Enter a question first.")
    t0 = time.perf_counter()
    try:
        scene = _scene_from_image(image_path)
    except Exception as exc:
        return _empty_outputs(f"Vision pipeline failed: {exc}")
    vision_ms = (time.perf_counter() - t0) * 1000
    result = _run_pipeline(scene, question.strip())
    return _format_outputs(result, vision_ms=vision_ms)


def run_cached_clevr(cached_name: str, question: str) -> tuple:
    if not cached_name:
        return _empty_outputs("Select a cached CLEVR example first.")
    if not question.strip():
        return _empty_outputs("Enter a question first.")
    t0 = time.perf_counter()
    try:
        scene = _scene_from_cached(cached_name)
    except FileNotFoundError:
        return _empty_outputs(
            f"Cached scene graph missing for {cached_name!r}. "
            "Run scripts/precompute_clevr_scenes.py to regenerate."
        )
    except Exception as exc:
        return _empty_outputs(f"Failed to load cached scene: {exc}")
    load_ms = (time.perf_counter() - t0) * 1000
    result = _run_pipeline(scene, question.strip())
    # Surface BOTH the cached-load time AND the original vision wall time so
    # reviewers can see what was traded. The original wall time is recorded
    # at precompute time, not measured at runtime.
    original_ms = CACHED_VISION_WALL_MS.get(cached_name)
    return _format_outputs(
        result, vision_ms=load_ms, cached_original_vision_ms=original_ms
    )


def _format_outputs(
    result: dict[str, Any],
    *,
    vision_ms: Optional[float],
    cached_original_vision_ms: Optional[float] = None,
) -> tuple:
    scene_json = _scene_to_jsonable(result["scene"])
    latency_rows = _latency_table(
        result["latency"], vision_ms, cached_original_vision_ms
    )

    if not result["ok"]:
        answer = f"[{result['stage']}] {result['error']}"
        return (
            answer,
            "",
            scene_json,
            result.get("kb_source", ""),
            "",
            "",
            latency_rows,
        )

    bundle = result["bundle"]
    parsed = result["parsed"]
    qresult = result["qresult"]
    answer = bundle.answer
    trace = "\n".join(bundle.trace)
    kb_code = result["kb_source"]
    parsed_q = f"% type = {parsed.type}\n{parsed.query}."
    raw_bindings = json.dumps(
        [{k: _coerce_value(v) for k, v in b.items()} for b in qresult.raw_bindings],
        indent=2,
        default=str,
    )
    return (
        answer,
        trace,
        scene_json,
        kb_code,
        parsed_q,
        raw_bindings,
        latency_rows,
    )


def _coerce_value(v: Any) -> Any:
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)


# ----- Layout -------------------------------------------------------------


def build_app() -> gr.Blocks:
    title = "Neurosymbolic VQA"
    subtitle = (
        "Image (or preset) → SceneGraph → Prolog KB → LLM-translated query → "
        "deterministic answer with reasoning trace."
    )

    with gr.Blocks(title=title, theme=gr.themes.Soft()) as app:
        gr.Markdown(f"# {title}")
        gr.Markdown(subtitle)
        gr.Markdown(
            "### Three input modes\n"
            "- **Synthetic scene** — hand-built scene graphs that stress-test "
            "the symbolic stack. 100% on the 32-question golden dataset.\n"
            "- **Cached CLEVR examples** — real CLEVR images with pre-computed "
            "scene graphs (vision pipeline ran locally before deploy; "
            "reasoning runs live). 📦 badge marks pre-computed perception.\n"
            + (
                "- *Real image upload is hidden on HuggingFace Spaces (vision "
                "models too heavy for the free CPU tier). Clone locally to "
                "enable it.*"
                if IS_SPACES
                else "- **Real image** — full live OWL-ViT + CLIP inference on "
                "an uploaded image (requires `requirements-full.txt`)."
            )
        )

        with gr.Row():
            # ----- Left: inputs -----
            with gr.Column(scale=1):
                with gr.Tabs() as input_tabs:
                    with gr.Tab("Cached CLEVR examples"):
                        gr.Markdown(
                            "**📦 Pre-computed scene graph** — the vision "
                            "pipeline (OWL-ViT + CLIP) ran offline on the "
                            "developer's M4 before deploy. Only KB generation, "
                            "NL → Prolog translation, query execution, and "
                            "verbalization run live."
                        )
                        cached_dropdown = gr.Dropdown(
                            label="Cached CLEVR scene",
                            choices=list(CACHED_CLEVR_NAMES),
                            value=CACHED_CLEVR_NAMES[0],
                        )
                        cached_image_preview = gr.Image(
                            label="Source image",
                            value=_cached_clevr_png(CACHED_CLEVR_NAMES[0]),
                            type="filepath",
                            height=240,
                            interactive=False,
                        )
                        cached_dropdown.change(
                            fn=_cached_clevr_png,
                            inputs=[cached_dropdown],
                            outputs=[cached_image_preview],
                        )

                    if not IS_SPACES:
                        with gr.Tab("Real image"):
                            real_image = gr.Image(
                                label="Image",
                                type="filepath",
                                height=300,
                            )
                            real_examples = gr.Examples(
                                examples=[[p] for p in CLEVR_EXAMPLES if Path(p).exists()],
                                inputs=[real_image],
                                label="CLEVR examples",
                            )
                    else:
                        real_image = gr.State(None)

                    with gr.Tab("Synthetic scene"):
                        preset_dropdown = gr.Dropdown(
                            label="Preset",
                            choices=list(PRESET_NAMES),
                            value=PRESET_NAMES[0],
                        )
                        gr.Markdown(
                            "_clevr_like_ — 3 objects (cube, sphere, cylinder). "
                            "_kitchen_ — table + cup + bottle + apple. "
                            "_office_ — 2 chairs + desk + monitor. "
                            "_single_object_ — one red cube. "
                            "_empty_scene_ — no objects."
                        )

                question = gr.Textbox(
                    label="Question",
                    placeholder="e.g. 'How many metal objects are there?'",
                    lines=2,
                )
                gr.Examples(
                    examples=[[q] for q in EXAMPLE_QUESTIONS],
                    inputs=[question],
                    label="Example questions",
                )

                run_cached_btn = gr.Button(
                    "Run (cached CLEVR)", variant="primary"
                )
                if not IS_SPACES:
                    run_real_btn = gr.Button("Run (real image)", variant="primary")
                run_synth_btn = gr.Button("Run (synthetic)", variant="primary")

            # ----- Right: outputs -----
            with gr.Column(scale=1):
                answer_box = gr.Textbox(label="Answer", lines=3, interactive=False)

                with gr.Tabs():
                    with gr.Tab("Reasoning trace"):
                        trace_box = gr.Textbox(
                            label="Step-by-step trace",
                            lines=10,
                            interactive=False,
                        )
                    with gr.Tab("Scene graph"):
                        scene_json = gr.JSON(label="SceneGraph (JSON)")
                    with gr.Tab("Generated KB"):
                        kb_code = gr.Code(label="Prolog KB", language="markdown", lines=18)
                    with gr.Tab("Prolog query"):
                        query_code = gr.Code(label="Translated query", language="markdown", lines=4)
                    with gr.Tab("Raw bindings"):
                        bindings_code = gr.Code(label="pyswip bindings", language="json", lines=10)
                    with gr.Tab("Latency"):
                        latency_table = gr.Dataframe(
                            headers=["stage", "ms"],
                            label="Per-stage latency (ms)",
                            interactive=False,
                        )

        with gr.Accordion("How the reasoning trace is built", open=False):
            gr.Markdown(
                "Every word of the answer is templated from the symbolic "
                "result. The trace lists: (1) the question, (2) the Prolog "
                "query the LLM translated to, (3) each solution binding the "
                "executor returned, resolved against the scene graph. There "
                "is no LLM in the verbalizer — given correct bindings, the "
                "trace is deterministic."
            )

        # ----- Event wiring -----
        run_cached_btn.click(
            fn=run_cached_clevr,
            inputs=[cached_dropdown, question],
            outputs=[
                answer_box, trace_box, scene_json, kb_code,
                query_code, bindings_code, latency_table,
            ],
        )

        run_synth_btn.click(
            fn=run_synthetic,
            inputs=[preset_dropdown, question],
            outputs=[
                answer_box, trace_box, scene_json, kb_code,
                query_code, bindings_code, latency_table,
            ],
        )

        if not IS_SPACES:
            run_real_btn.click(
                fn=run_real_image,
                inputs=[real_image, question],
                outputs=[
                    answer_box, trace_box, scene_json, kb_code,
                    query_code, bindings_code, latency_table,
                ],
            )

    return app


def main() -> None:
    try:
        # Force backend resolution at startup so a missing OPENAI_API_KEY +
        # no-ollama config surfaces immediately, not on the first click.
        _get_pipeline()
    except NoBackendAvailableError as exc:
        print(f"WARNING: no NL→Prolog backend available — {exc}")
        print("Demo will start but Run buttons will error until a backend is configured.")

    app = build_app()
    # Bind 0.0.0.0 inside HuggingFace Spaces so the platform health check
    # and reverse proxy can reach Gradio from outside the container. Stay on
    # 127.0.0.1 locally to avoid exposing the dev server on all interfaces.
    default_host = "0.0.0.0" if os.environ.get("SPACE_ID") else "127.0.0.1"
    host = os.environ.get("NSVQA_DEMO_HOST", default_host)
    app.launch(server_name=host, server_port=7860, show_api=False)


if __name__ == "__main__":
    main()
