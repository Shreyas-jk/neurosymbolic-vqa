"""Executes a ParsedQuery against a KB source string.

Each call consults the KB into pyswip's engine (predicates are dynamic +
abolished by the generated header, so re-consult is clean) and runs the user
query under SWI's native `call_with_time_limit/2` so a runaway goal aborts
inside Prolog rather than leaking a Python thread.
"""

from __future__ import annotations

import os
import tempfile
import time
from typing import Any, Optional

import pyswip

from query_executor.result import ParsedQuery, QueryResult


class QueryTimeoutError(TimeoutError):
    """Raised when a Prolog query exceeds the configured timeout."""


# Names the LLM is taught to use for answer bindings. Order matters — we try
# the most specific first when no explicit bind_variable is provided.
_ATTRIBUTE_VAR_HINTS = ("C", "S", "M", "V", "K", "Color", "Size", "Material")
_OBJECT_VAR_HINTS = ("X", "Obj", "Object")
_COUNT_VAR_HINTS = ("N", "Count")
_LIST_VAR_HINTS = ("L", "List", "Objs")


def _is_unbound(value: Any) -> bool:
    return isinstance(value, pyswip.Variable)


def _is_obj_id(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("obj_")


def _pick_value(
    binding: dict[str, Any],
    hints: tuple[str, ...],
    *,
    accept: callable = lambda v: True,
) -> Optional[Any]:
    """Return the first bound value matching `accept`, preferring hint names."""
    for name in hints:
        if name in binding and not _is_unbound(binding[name]) and accept(binding[name]):
            return binding[name]
    for name, value in binding.items():
        if name not in hints and not _is_unbound(value) and accept(value):
            return value
    return None


class QueryExecutor:
    def __init__(self, timeout_s: float = 5.0) -> None:
        self.timeout_s = timeout_s

    def run(self, kb_source: str, parsed: ParsedQuery) -> QueryResult:
        start = time.perf_counter()
        try:
            bindings = self._consult_and_query(
                kb_source, parsed.query, self.timeout_s
            )
        except QueryTimeoutError as exc:
            return QueryResult(
                success=False,
                answer=None if parsed.type != "boolean" else False,
                raw_bindings=(),
                query_string=parsed.query,
                execution_time_ms=(time.perf_counter() - start) * 1000,
                error=str(exc),
                type=parsed.type,
            )
        except Exception as exc:  # pyswip errors bubble as generic Exception
            return QueryResult(
                success=False,
                answer=None if parsed.type != "boolean" else False,
                raw_bindings=(),
                query_string=parsed.query,
                execution_time_ms=(time.perf_counter() - start) * 1000,
                error=f"prolog error: {exc}",
                type=parsed.type,
            )

        success = len(bindings) > 0
        elapsed_ms = (time.perf_counter() - start) * 1000
        if parsed.type == "boolean":
            return QueryResult(
                success=success,
                answer=bool(success),
                raw_bindings=tuple(bindings),
                query_string=parsed.query,
                execution_time_ms=elapsed_ms,
                error=None,
                type=parsed.type,
            )
        return QueryResult(
            success=success,
            answer=self._extract_answer(parsed, bindings) if success else None,
            raw_bindings=tuple(bindings),
            query_string=parsed.query,
            execution_time_ms=elapsed_ms,
            error=None if success else "no solutions",
            type=parsed.type,
        )

    @staticmethod
    def _consult_and_query(
        kb_source: str,
        query: str,
        timeout_s: float,
    ) -> list[dict[str, Any]]:
        tmp = tempfile.NamedTemporaryFile(
            "w", suffix=".pl", delete=False, encoding="utf-8"
        )
        try:
            tmp.write(kb_source)
            tmp.flush()
            tmp.close()
            prolog = pyswip.Prolog()
            list(prolog.query(f"consult('{tmp.name}')"))
            wrapped = f"call_with_time_limit({timeout_s}, ({query}))"
            try:
                return list(prolog.query(wrapped))
            except Exception as exc:
                msg = str(exc)
                if "time_limit_exceeded" in msg or "timeout" in msg.lower():
                    raise QueryTimeoutError(
                        f"query exceeded {timeout_s}s"
                    ) from exc
                raise
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    def _extract_answer(
        self,
        parsed: ParsedQuery,
        bindings: list[dict[str, Any]],
    ) -> Any:
        first = bindings[0]

        if parsed.type == "count":
            if parsed.bind_variable and parsed.bind_variable in first:
                return _coerce_int(first[parsed.bind_variable])
            value = _pick_value(
                first,
                _COUNT_VAR_HINTS,
                accept=lambda v: isinstance(v, int) or (isinstance(v, str) and v.isdigit()),
            )
            return _coerce_int(value) if value is not None else None

        if parsed.type == "attribute":
            if parsed.bind_variable and parsed.bind_variable in first:
                return _atom_to_str(first[parsed.bind_variable])
            value = _pick_value(
                first,
                _ATTRIBUTE_VAR_HINTS,
                accept=lambda v: isinstance(v, str) and not _is_obj_id(v),
            )
            return _atom_to_str(value)

        if parsed.type == "object":
            if parsed.bind_variable and parsed.bind_variable in first:
                return _atom_to_str(first[parsed.bind_variable])
            value = _pick_value(
                first,
                _OBJECT_VAR_HINTS,
                accept=lambda v: isinstance(v, str) and _is_obj_id(v),
            )
            return _atom_to_str(value)

        if parsed.type == "list":
            if parsed.bind_variable and parsed.bind_variable in first:
                return _coerce_list(first[parsed.bind_variable])
            value = _pick_value(
                first,
                _LIST_VAR_HINTS,
                accept=lambda v: isinstance(v, list),
            )
            if value is not None:
                return _coerce_list(value)
            # Fallback: iterate bindings and collect a single var.
            collected: list[str] = []
            for b in bindings:
                v = _pick_value(b, _OBJECT_VAR_HINTS, accept=_is_obj_id)
                if v is not None:
                    collected.append(_atom_to_str(v))
            # Deduplicate, preserve order.
            seen: set[str] = set()
            out: list[str] = []
            for item in collected:
                if item not in seen:
                    seen.add(item)
                    out.append(item)
            return out

        return None


def _atom_to_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


def _coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [_atom_to_str(item) for item in value if item is not None]
