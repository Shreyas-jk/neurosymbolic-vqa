"""NL → Prolog translation package.

Public API:
    Attempt, TranslatorBackend, TranslatorPipeline, TranslationError,
    TranslationResult, parse_response, validate_query
    OpenAIBackend, OllamaBackend
    build_schema
    get_backend(): factory honoring NL2PROLOG_BACKEND env var

`get_backend()` is the only entry point that touches env vars and external
services, so tests can construct TranslatorPipeline with a MockBackend
without touching openai or ollama.
"""

from __future__ import annotations

import os
from typing import Optional

from nl2prolog.schema_builder import build_schema
from nl2prolog.translator import (
    Attempt,
    ParsedQuery,
    TranslationError,
    TranslationResult,
    TranslatorBackend,
    TranslatorPipeline,
    parse_response,
    validate_query,
)


class NoBackendAvailableError(RuntimeError):
    """Neither openai (no API key) nor ollama (unreachable) is usable."""


def get_backend(
    backend_name: Optional[str] = None,
    *,
    allow_fallback: bool = True,
) -> TranslatorBackend:
    """Construct a backend by name or env var.

    Resolution order:
      1. Explicit `backend_name` argument (if given)
      2. NL2PROLOG_BACKEND env var
      3. Default: "openai" if OPENAI_API_KEY is set, else "local"

    If the requested backend is "openai" but no API key is present, falls back
    to "local" when `allow_fallback=True`, else raises.
    """
    if backend_name is None:
        backend_name = os.environ.get("NL2PROLOG_BACKEND")
    if backend_name is None:
        backend_name = "openai" if os.environ.get("OPENAI_API_KEY") else "local"
    backend_name = backend_name.lower().strip()

    if backend_name == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            if allow_fallback:
                return _make_local_or_raise()
            raise NoBackendAvailableError(
                "OPENAI_API_KEY not set and allow_fallback=False"
            )
        from nl2prolog.openai_backend import OpenAIBackend

        return OpenAIBackend()

    if backend_name == "local":
        return _make_local_or_raise()

    raise ValueError(
        f"unknown backend {backend_name!r}; expected 'openai' or 'local'"
    )


def _make_local_or_raise() -> TranslatorBackend:
    try:
        from nl2prolog.local_backend import OllamaBackend

        return OllamaBackend()
    except ImportError as exc:
        raise NoBackendAvailableError(
            f"local backend (ollama) is not importable: {exc}"
        ) from exc


__all__ = [
    "Attempt",
    "NoBackendAvailableError",
    "ParsedQuery",
    "TranslationError",
    "TranslationResult",
    "TranslatorBackend",
    "TranslatorPipeline",
    "build_schema",
    "get_backend",
    "parse_response",
    "validate_query",
]
