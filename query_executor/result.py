"""Typed result and parsed-query containers.

ParsedQuery lives here (rather than under nl2prolog/, which is built in Phase 2)
so the executor's interface is fully defined in Phase 1 and tests can construct
parsed queries directly with hand-written Prolog.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

QueryType = Literal["boolean", "count", "attribute", "object", "list"]


@dataclass(frozen=True)
class ParsedQuery:
    """A Prolog query plus metadata describing how to interpret its bindings."""

    query: str
    type: QueryType
    bind_variable: Optional[str] = None  # explicit answer-binding name


@dataclass(frozen=True)
class QueryResult:
    success: bool
    answer: Any
    raw_bindings: tuple[dict[str, Any], ...]
    query_string: str
    execution_time_ms: float
    error: Optional[str] = None
    type: Optional[QueryType] = None
