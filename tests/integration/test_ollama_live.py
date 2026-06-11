"""Live smoke test against a local ollama server.

Marked `slow` so it is skipped by default — CI runs `pytest -m "not slow"`.
Also auto-skips if the ollama server is unreachable or the model is missing,
so it doesn't break a developer's run when ollama isn't up.
"""

from __future__ import annotations

import pytest

from kb_generator.generator import generate
from nl2prolog import build_schema
from nl2prolog.local_backend import OllamaBackend
from nl2prolog.translator import TranslatorPipeline
from synthetic import presets


pytestmark = pytest.mark.slow


def _ollama_available(backend: OllamaBackend) -> bool:
    try:
        from ollama import Client

        client = Client(host=backend.host)
        models = client.list().get("models", [])
        return any(backend.model in (m.get("name") or m.get("model", "")) for m in models)
    except Exception:
        return False


@pytest.fixture(scope="module")
def backend() -> OllamaBackend:
    b = OllamaBackend()
    if not _ollama_available(b):
        pytest.skip(
            f"ollama not reachable at {b.host} or model {b.model!r} not pulled"
        )
    return b


@pytest.fixture(scope="module")
def kb_and_schema():
    kb = generate(presets.clevr_like())
    return kb.source, build_schema(kb)


def test_ollama_simple_existence_question(backend, kb_and_schema) -> None:
    kb_source, schema = kb_and_schema
    pipeline = TranslatorPipeline(backend=backend, max_attempts=3)
    result = pipeline.translate("Is there a red cube?", kb_source, schema)
    assert result.parsed.type == "boolean"
    # The query should reference object/2 and color=red somehow.
    q = result.parsed.query
    assert "object" in q and "red" in q


def test_ollama_count_question(backend, kb_and_schema) -> None:
    kb_source, schema = kb_and_schema
    pipeline = TranslatorPipeline(backend=backend, max_attempts=3)
    result = pipeline.translate(
        "How many metal objects are there?", kb_source, schema
    )
    assert result.parsed.type == "count"
    assert "findall" in result.parsed.query.lower()
    assert "length" in result.parsed.query.lower()
