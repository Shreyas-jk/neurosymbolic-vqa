"""NL → Prolog translation pipeline with structured retry loop.

This is the key upgrade over the AIEA-Lab baseline: instead of a single LLM
call whose output is trusted blindly, every candidate query is dry-run through
an isolated `swipl` subprocess. If it raises an existence or syntax error, the
error string is appended to the prompt and the LLM is asked to repair its own
output, up to N attempts. Process-wide pyswip state is bypassed by using a
subprocess for validation — same isolation pattern as kb_generator.validator.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from query_executor.result import ParsedQuery, QueryType


_VALID_TYPES: frozenset[str] = frozenset(
    {"boolean", "count", "attribute", "object", "list"}
)


# ----- Backend interface ----- #


class TranslatorBackend(ABC):
    """A thin wrapper around an LLM client. One call → one raw text response."""

    name: str = "backend"

    @abstractmethod
    def call(self, system_prompt: str, user_prompt: str) -> str:
        """Return the model's raw text output (ideally a JSON object)."""


# ----- Errors / result types ----- #


class TranslationError(RuntimeError):
    """All retries failed. Carries the full attempt history for debugging."""

    def __init__(self, message: str, attempts: tuple["Attempt", ...]) -> None:
        super().__init__(message)
        self.attempts = attempts


@dataclass(frozen=True)
class Attempt:
    raw_response: str
    error: str


@dataclass(frozen=True)
class TranslationResult:
    parsed: ParsedQuery
    attempts: tuple[Attempt, ...] = field(default_factory=tuple)


# ----- Output parsing ----- #


_MD_FENCE_OPEN_RE = re.compile(r"^```(?:json)?\s*", re.IGNORECASE)
_MD_FENCE_CLOSE_RE = re.compile(r"\s*```\s*$")


def _strip_markdown(raw: str) -> str:
    """Match the AIEA-Lab baseline exactly: strip opening/closing code fences."""
    cleaned = _MD_FENCE_OPEN_RE.sub("", raw.strip())
    cleaned = _MD_FENCE_CLOSE_RE.sub("", cleaned)
    return cleaned.strip()


def parse_response(raw: str) -> ParsedQuery:
    """Parse the LLM JSON response into a ParsedQuery.

    Raises ValueError with a descriptive message on any malformation, so the
    retry loop can feed the error back to the model.
    """
    cleaned = _strip_markdown(raw)
    if not cleaned:
        raise ValueError("response was empty after stripping markdown")
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"response was not valid JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise ValueError(f"expected JSON object, got {type(obj).__name__}")

    query = obj.get("query")
    qtype = obj.get("type")
    bind_variable = obj.get("bind_variable")

    if not isinstance(query, str) or not query.strip():
        raise ValueError("JSON missing non-empty string field 'query'")
    if qtype not in _VALID_TYPES:
        raise ValueError(
            f"'type' must be one of {sorted(_VALID_TYPES)}, got {qtype!r}"
        )
    if bind_variable is not None and not isinstance(bind_variable, str):
        raise ValueError("'bind_variable' must be a string when present")

    return ParsedQuery(
        query=query.strip().rstrip("."),
        type=qtype,  # type: ignore[arg-type]
        bind_variable=bind_variable or None,
    )


# ----- Subprocess-based query validation ----- #


def validate_query(
    kb_source: str,
    query: str,
    *,
    swipl_path: str = "swipl",
    timeout_s: float = 5.0,
) -> Optional[str]:
    """Try the query against a fresh swipl session that has consulted the KB.

    Returns None if the query parses and runs without an exception (success or
    no-solutions both count as valid). Returns an error string otherwise so the
    retry loop can surface it to the LLM.
    """
    # The query may succeed or fail; only existence/syntax/type errors should
    # cause validation to fail. We wrap with catch/3 and a true-fallback so
    # plain failure exits cleanly.
    combined = (
        kb_source
        + "\n:- catch(("
        + query
        + " -> true ; true), E, (print_message(error, E), halt(2))).\n"
        + ":- halt(0).\n"
    )
    tmp = tempfile.NamedTemporaryFile(
        "w", suffix=".pl", delete=False, encoding="utf-8"
    )
    try:
        tmp.write(combined)
        tmp.flush()
        tmp.close()
        try:
            proc = subprocess.run(
                [swipl_path, "-q", "-s", tmp.name],
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except FileNotFoundError:
            return f"swipl not found at {swipl_path!r}"
        except subprocess.TimeoutExpired:
            return f"query validation exceeded {timeout_s}s"

        if proc.returncode == 0 and "ERROR:" not in proc.stderr:
            return None
        # Collect ERROR: lines if any; fall back to whole stderr.
        errors = [
            line.strip()
            for line in proc.stderr.splitlines()
            if line.lstrip().startswith("ERROR:")
        ]
        if errors:
            return " | ".join(errors)
        return (
            proc.stderr.strip()
            or f"swipl exited with code {proc.returncode}"
        )
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


# ----- Retry pipeline ----- #


@dataclass(frozen=True)
class TranslatorPipeline:
    backend: TranslatorBackend
    max_attempts: int = 3
    swipl_path: str = "swipl"
    validation_timeout_s: float = 5.0

    def translate(
        self,
        question: str,
        kb_source: str,
        schema_block: str,
        *,
        system_prompt: Optional[str] = None,
    ) -> TranslationResult:
        from nl2prolog.prompt_templates import (
            SYSTEM_PROMPT,
            build_user_prompt,
        )

        sys_prompt = system_prompt or SYSTEM_PROMPT
        attempts: list[Attempt] = []
        for attempt_idx in range(self.max_attempts):
            user_prompt = build_user_prompt(
                question,
                schema_block,
                prior_attempts=tuple(
                    (a.raw_response, a.error) for a in attempts
                ),
            )
            raw = self.backend.call(sys_prompt, user_prompt)
            try:
                parsed = parse_response(raw)
            except ValueError as exc:
                attempts.append(Attempt(raw_response=raw, error=str(exc)))
                continue
            err = validate_query(
                kb_source,
                parsed.query,
                swipl_path=self.swipl_path,
                timeout_s=self.validation_timeout_s,
            )
            if err is None:
                return TranslationResult(
                    parsed=parsed,
                    attempts=tuple(attempts),
                )
            attempts.append(Attempt(raw_response=raw, error=err))

        raise TranslationError(
            f"NL→Prolog translation failed after {self.max_attempts} attempts",
            attempts=tuple(attempts),
        )
