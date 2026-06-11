"""System prompt + 15-example few-shot block for NL→Prolog translation.

The structure (system, schema, few-shot, question) intentionally mirrors the
AIEA-Lab baseline. JSON output is enforced via the system prompt and reinforced
by either OpenAI's `response_format=json_object` or ollama's `format="json"`,
so we never strip markdown in the happy path — markdown stripping is the retry
fallback when a backend ignores the format flag.
"""

from __future__ import annotations

from dataclasses import dataclass


SYSTEM_PROMPT = """You translate English questions about a scene into Prolog queries for a knowledge base of facts about that scene.

Output ONLY a JSON object with these fields:
  - "query": the Prolog goal as a string (NO trailing period)
  - "type": one of "boolean", "count", "attribute", "object", "list"
  - "bind_variable": (optional) the Prolog variable holding the answer

Conventions:
  - Predicates are lowercase with underscores: object, attribute, relation, left_of, same_color, etc.
  - Variables MUST start with an uppercase letter or underscore: X, Y, Color, _Ignored
  - Use ONLY the predicates listed under SCHEMA below.
  - DO NOT include a trailing period in the query.
  - DO NOT wrap the JSON in markdown code fences. Output raw JSON only.

Question types:
  - boolean: yes/no question. The query should succeed for "yes" and fail for "no".
  - count: "how many" question. Wrap with findall+length:
        findall(X, (...), L), length(L, N)
    and set "bind_variable" to "N".
  - attribute: "what color/size/material is the X". Bind one variable, e.g.
        attribute(X, color, C)
    and set "bind_variable" to "C" (or "S" for size, "M" for material).
  - object: "which object" / "what is to the left of ...". Bind to an object id X.
    Set "bind_variable" to "X".
  - list: "list all X" / "which objects". Use findall to collect into L.
    Set "bind_variable" to "L".
"""


@dataclass(frozen=True)
class FewShot:
    question: str
    output: str  # JSON-encoded response string


# 15 canonical pairs: existence ×2, count ×3, attribute ×3, spatial ×4,
# multi-hop ×3. Cover every question type the executor knows how to dispatch.
FEWSHOT: tuple[FewShot, ...] = (
    # --- existence (2) ---
    FewShot(
        question="Is there a red cube?",
        output='{"query": "object(X, cube), attribute(X, color, red)", "type": "boolean"}',
    ),
    FewShot(
        question="Is there a metal sphere?",
        output='{"query": "object(X, sphere), attribute(X, material, metal)", "type": "boolean"}',
    ),
    # --- count (3) ---
    FewShot(
        question="How many red objects are there?",
        output=(
            '{"query": "findall(X, (object(X, _), attribute(X, color, red)), L),'
            ' length(L, N)", "type": "count", "bind_variable": "N"}'
        ),
    ),
    FewShot(
        question="How many cubes are in the scene?",
        output=(
            '{"query": "findall(X, object(X, cube), L), length(L, N)",'
            ' "type": "count", "bind_variable": "N"}'
        ),
    ),
    FewShot(
        question="Count the small objects.",
        output=(
            '{"query": "findall(X, (object(X, _), attribute(X, size, small)), L),'
            ' length(L, N)", "type": "count", "bind_variable": "N"}'
        ),
    ),
    # --- attribute (3) ---
    FewShot(
        question="What color is the cube?",
        output=(
            '{"query": "object(X, cube), attribute(X, color, C)",'
            ' "type": "attribute", "bind_variable": "C"}'
        ),
    ),
    FewShot(
        question="What size is the sphere?",
        output=(
            '{"query": "object(X, sphere), attribute(X, size, S)",'
            ' "type": "attribute", "bind_variable": "S"}'
        ),
    ),
    FewShot(
        question="What material is the cylinder made of?",
        output=(
            '{"query": "object(X, cylinder), attribute(X, material, M)",'
            ' "type": "attribute", "bind_variable": "M"}'
        ),
    ),
    # --- spatial (4) ---
    FewShot(
        question="What is to the left of the sphere?",
        output=(
            '{"query": "object(Y, sphere), left_of(X, Y), object(X, _)",'
            ' "type": "object", "bind_variable": "X"}'
        ),
    ),
    FewShot(
        question="What is above the cube?",
        output=(
            '{"query": "object(Y, cube), above(X, Y), object(X, _)",'
            ' "type": "object", "bind_variable": "X"}'
        ),
    ),
    FewShot(
        question="Is the bottle on top of the table?",
        output=(
            '{"query": "object(B, bottle), object(T, table), on_top_of(B, T)",'
            ' "type": "boolean"}'
        ),
    ),
    FewShot(
        question="What is next to the cup?",
        output=(
            '{"query": "object(Y, cup), next_to(X, Y), object(X, _),'
            ' X \\\\= Y", "type": "object", "bind_variable": "X"}'
        ),
    ),
    # --- multi-hop (3) ---
    FewShot(
        question="What color is the object to the left of the cube?",
        output=(
            '{"query": "object(Y, cube), left_of(X, Y), attribute(X, color, C)",'
            ' "type": "attribute", "bind_variable": "C"}'
        ),
    ),
    FewShot(
        question="Are there two objects with the same color?",
        output=(
            '{"query": "same_color(X, Y)", "type": "boolean"}'
        ),
    ),
    FewShot(
        question="List all objects that share a color with the cube.",
        output=(
            '{"query": "findall(Z, (object(C, cube), same_color(C, Z)), L)",'
            ' "type": "list", "bind_variable": "L"}'
        ),
    ),
)


def render_fewshot(examples: tuple[FewShot, ...] = FEWSHOT) -> str:
    blocks: list[str] = ["EXAMPLES"]
    for ex in examples:
        blocks.append(f"Q: {ex.question}")
        blocks.append(f"A: {ex.output}")
        blocks.append("")
    return "\n".join(blocks).rstrip()


def build_user_prompt(
    question: str,
    schema_block: str,
    *,
    prior_attempts: tuple[tuple[str, str], ...] = (),
) -> str:
    """Build the user-message content.

    `prior_attempts` is a tuple of (raw_response, error_message) for each
    previous failed try in the retry loop. Surfacing the actual error string
    back to the model gives it a chance to fix its own mistake.
    """
    parts = [schema_block, "", render_fewshot(), ""]
    if prior_attempts:
        parts.append("PRIOR ATTEMPTS (please fix the issue and try again):")
        for i, (raw, err) in enumerate(prior_attempts, start=1):
            parts.append(f"Attempt {i} output: {raw}")
            parts.append(f"Attempt {i} error: {err}")
        parts.append("")
    parts.append(f"Q: {question}")
    parts.append("A:")
    return "\n".join(parts)
