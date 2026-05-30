"""Prolog clause templates and atom sanitization for KB generation."""

from __future__ import annotations

import re

_VALID_ATOM_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def to_atom(value: str) -> str:
    """Render a Python string as a safe Prolog atom.

    Lowercase ASCII identifiers pass through bare; anything else is single-quoted
    with embedded single quotes escaped per SWI-Prolog conventions.
    """
    s = value.strip()
    if not s:
        raise ValueError("Cannot convert empty string to Prolog atom")
    lowered = s.lower().replace(" ", "_").replace("-", "_")
    lowered = re.sub(r"[^a-z0-9_]", "_", lowered)
    if _VALID_ATOM_RE.match(lowered):
        return lowered
    escaped = s.replace("\\", "\\\\").replace("'", "\\'")
    return f"'{escaped}'"


def id_atom(obj_id: str) -> str:
    """Render an object id. Always single-quoted for visual distinction in KB."""
    escaped = obj_id.replace("\\", "\\\\").replace("'", "\\'")
    return f"'{escaped}'"


# Predicates whose prior definitions must be abolished before (re)loading the
# KB. pyswip uses a single SWI engine across calls, so without this every
# subsequent consult would warn "Redefined static procedure". We also mark the
# fact predicates as dynamic so the KB can be re-consulted in the same process.
_ALL_PREDICATES = (
    ("object", 2),
    ("attribute", 3),
    ("relation", 3),
    ("is_a", 2),
    ("has", 3),
    ("left_of", 2),
    ("right_of", 2),
    ("above", 2),
    ("below", 2),
    ("inside", 2),
    ("on_top_of", 2),
    ("next_to", 2),
    ("same_color", 2),
    ("same_size", 2),
    ("same_material", 2),
    ("same_shape", 2),
)


def _build_header() -> str:
    lines = [
        "% Auto-generated knowledge base.",
        "% Do not edit by hand.",
        "",
        ":- set_prolog_flag(verbose_load, silent).",
        "",
    ]
    for name, arity in _ALL_PREDICATES:
        lines.append(
            f":- (current_predicate({name}/{arity}) -> abolish({name}/{arity}) ; true)."
        )
    lines.append("")
    lines.append(":- dynamic(object/2).")
    lines.append(":- dynamic(attribute/3).")
    lines.append(":- dynamic(relation/3).")
    lines.append("")
    return "\n".join(lines)


HEADER = _build_header()


# Asserted unconditionally so query predicates always resolve, even on an
# empty scene. Keeps the NL→Prolog schema stable across scenes.
RULES = """% --- Derived predicates (always present) ---

is_a(X, C) :- object(X, C).
has(X, K, V) :- attribute(X, K, V).

% Spatial inverses
left_of(X, Y)  :- relation(X, left_of, Y).
right_of(X, Y) :- relation(Y, left_of, X).
above(X, Y)    :- relation(X, above, Y).
below(X, Y)    :- relation(Y, above, X).

inside(X, Y)    :- relation(X, inside, Y).
on_top_of(X, Y) :- relation(X, on_top_of, Y).
next_to(X, Y)   :- relation(X, next_to, Y) ; relation(Y, next_to, X).

% Same-attribute predicates
same_color(X, Y)    :- attribute(X, color, C),    attribute(Y, color, C),    X \\= Y.
same_size(X, Y)     :- attribute(X, size, S),     attribute(Y, size, S),     X \\= Y.
same_material(X, Y) :- attribute(X, material, M), attribute(Y, material, M), X \\= Y.
same_shape(X, Y)    :- attribute(X, shape, S),    attribute(Y, shape, S),    X \\= Y.
"""


def object_fact(obj_id: str, category: str) -> str:
    return f"object({id_atom(obj_id)}, {to_atom(category)})."


def attribute_fact(obj_id: str, key: str, value: str) -> str:
    return f"attribute({id_atom(obj_id)}, {to_atom(key)}, {to_atom(value)})."


def relation_fact(subject_id: str, predicate: str, object_id: str) -> str:
    return f"relation({id_atom(subject_id)}, {to_atom(predicate)}, {id_atom(object_id)})."
