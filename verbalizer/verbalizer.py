"""Turns a QueryResult into a human-readable answer plus a reasoning trace.

No LLM here — every word of the answer is derived from the symbolic result.
This is the explainability story: a hiring manager (or reviewer) can point at
any sentence in the answer and trace it back to a specific Prolog binding.
"""

from __future__ import annotations

from dataclasses import dataclass

from scene_extractor.schema import SceneGraph, SceneObject
from query_executor.result import ParsedQuery, QueryResult


@dataclass(frozen=True)
class AnswerBundle:
    answer: str
    trace: tuple[str, ...]
    parsed: ParsedQuery
    result: QueryResult


def verbalize(
    question: str,
    parsed: ParsedQuery,
    result: QueryResult,
    scene: SceneGraph,
) -> AnswerBundle:
    answer = _render_answer(question, parsed, result, scene)
    trace = _build_trace(question, parsed, result, scene)
    return AnswerBundle(answer=answer, trace=trace, parsed=parsed, result=result)


def _render_answer(
    question: str,
    parsed: ParsedQuery,
    result: QueryResult,
    scene: SceneGraph,
) -> str:
    if result.error and parsed.type != "boolean":
        return f"I could not answer that. ({result.error})"

    if parsed.type == "boolean":
        if not result.success:
            return "No."
        # Try to surface what was found so the answer feels grounded.
        first = result.raw_bindings[0] if result.raw_bindings else {}
        obj_id = _first_obj_id(first)
        obj = scene.object_by_id(obj_id) if obj_id else None
        if obj is not None:
            return f"Yes — {_describe(obj)}."
        return "Yes."

    if parsed.type == "count":
        n = result.answer
        if not isinstance(n, int):
            return "I could not determine the count."
        noun = _extract_noun(question)
        if n == 0:
            return f"There are no {noun}."
        if n == 1:
            return f"There is 1 {_singular(noun)}."
        return f"There are {n} {noun}."

    if parsed.type == "attribute":
        value = result.answer
        if not value:
            return "I could not determine that attribute."
        referent = _extract_referent(question)
        return f"The {referent} is {value}."

    if parsed.type == "object":
        obj_id = result.answer
        obj = scene.object_by_id(obj_id) if obj_id else None
        if obj is None:
            return "I could not identify a matching object."
        return f"The {_describe(obj)} (id={obj.id})."

    if parsed.type == "list":
        items = result.answer or []
        if not items:
            return "There are no matching objects."
        descriptors = [_describe(scene.object_by_id(i)) for i in items if scene.object_by_id(i)]
        descriptors = [d for d in descriptors if d]
        if not descriptors:
            descriptors = list(items)
        if len(descriptors) == 1:
            return f"{descriptors[0].capitalize()}."
        return f"{', '.join(descriptors[:-1])}, and {descriptors[-1]}."

    return "I could not interpret the query result."


def _build_trace(
    question: str,
    parsed: ParsedQuery,
    result: QueryResult,
    scene: SceneGraph,
) -> tuple[str, ...]:
    steps: list[str] = []
    steps.append(f"Step 1. Question: {question!r}")
    steps.append(f"Step 2. Translated to Prolog ({parsed.type}): {parsed.query}")

    if result.error and parsed.type != "boolean":
        steps.append(f"Step 3. Execution error: {result.error}")
        return tuple(steps)

    if not result.raw_bindings and parsed.type != "boolean":
        steps.append("Step 3. No solutions found.")
        return tuple(steps)

    if parsed.type == "boolean":
        verdict = "succeeded" if result.success else "failed"
        steps.append(f"Step 3. Goal {verdict}.")
        if result.raw_bindings:
            steps.append(f"Step 4. First binding: {_format_binding(result.raw_bindings[0], scene)}")
        return tuple(steps)

    steps.append(f"Step 3. {len(result.raw_bindings)} solution(s) found.")
    for i, b in enumerate(result.raw_bindings[:3], start=4):
        steps.append(f"Step {i}. Solution: {_format_binding(b, scene)}")
    if len(result.raw_bindings) > 3:
        steps.append(f"        (+ {len(result.raw_bindings) - 3} more solutions)")
    steps.append(f"Step {3 + min(len(result.raw_bindings), 3) + 1}. Answer: {result.answer!r}")
    return tuple(steps)


def _format_binding(binding: dict, scene: SceneGraph) -> str:
    pairs = []
    for k, v in binding.items():
        v_str = str(v)
        if isinstance(v, str) and v.startswith("obj_"):
            obj = scene.object_by_id(v)
            if obj is not None:
                v_str = f"{v} ({_describe(obj)})"
        pairs.append(f"{k}={v_str}")
    return ", ".join(pairs)


def _first_obj_id(binding: dict) -> str | None:
    for v in binding.values():
        if isinstance(v, str) and v.startswith("obj_"):
            return v
    return None


def _describe(obj: SceneObject | None) -> str:
    if obj is None:
        return ""
    parts: list[str] = []
    if "size" in obj.attributes:
        parts.append(obj.attributes["size"])
    if "color" in obj.attributes:
        parts.append(obj.attributes["color"])
    if "material" in obj.attributes:
        parts.append(obj.attributes["material"])
    parts.append(obj.category)
    return " ".join(parts)


def _extract_noun(question: str) -> str:
    """Heuristic: grab the noun-ish phrase the count refers to.

    Looks for 'how many <X>' first; falls back to 'matching objects'.
    """
    q = question.lower().strip().rstrip("?.").strip()
    marker = "how many"
    if marker in q:
        tail = q.split(marker, 1)[1].strip()
        # cut off trailing predicate phrases like "are there", "are in the scene"
        for cut in (" are there", " are in", " do you see", " can you see", " exist"):
            if cut in tail:
                tail = tail.split(cut, 1)[0]
        if tail:
            return tail
    return "matching objects"


def _singular(noun: str) -> str:
    # Naive: strip a trailing 's' for words like 'objects' -> 'object'.
    if noun.endswith("ies") and len(noun) > 3:
        return noun[:-3] + "y"
    if noun.endswith("s") and not noun.endswith("ss"):
        return noun[:-1]
    return noun


def _extract_referent(question: str) -> str:
    q = question.lower().strip().rstrip("?.").strip()
    for prefix in ("what color is the ", "what size is the ", "what material is the ",
                   "what is the color of the ", "what is the size of the ",
                   "what is the material of the "):
        if q.startswith(prefix):
            return q[len(prefix):].strip()
    return "object"
