"""Unit tests for nl2prolog: parsing, validation, retry loop, schema builder.

No live LLM is touched here — every backend is a MockBackend that returns
canned responses, so these tests run identically on a developer laptop and CI.
The live-ollama smoke test lives under tests/integration/ and is @pytest.mark.slow.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import pytest

from kb_generator.generator import generate
from nl2prolog import (
    NoBackendAvailableError,
    TranslationError,
    TranslatorBackend,
    TranslatorPipeline,
    build_schema,
    get_backend,
    parse_response,
    validate_query,
)
from nl2prolog.prompt_templates import (
    FEWSHOT,
    SYSTEM_PROMPT,
    build_user_prompt,
    render_fewshot,
)
from synthetic import presets
from synthetic.scene_builder import SyntheticScene


# ----- Mock backend ----- #


@dataclass
class MockBackend(TranslatorBackend):
    """Returns the next queued response each time `call` is invoked."""

    responses: list[str] = field(default_factory=list)
    calls: list[tuple[str, str]] = field(default_factory=list)
    name: str = "mock"

    def call(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        if not self.responses:
            raise AssertionError("MockBackend ran out of canned responses")
        return self.responses.pop(0)


# ----- Fixtures ----- #


@pytest.fixture
def clevr_kb_source():
    return generate(presets.clevr_like()).source


@pytest.fixture
def clevr_schema():
    return build_schema(generate(presets.clevr_like()))


# ----- parse_response ----- #


def test_parse_response_clean_json() -> None:
    raw = '{"query": "object(X, cube)", "type": "object", "bind_variable": "X"}'
    parsed = parse_response(raw)
    assert parsed.query == "object(X, cube)"
    assert parsed.type == "object"
    assert parsed.bind_variable == "X"


def test_parse_response_strips_markdown_fences() -> None:
    raw = "```json\n{\"query\": \"object(X, cube)\", \"type\": \"object\"}\n```"
    parsed = parse_response(raw)
    assert parsed.query == "object(X, cube)"


def test_parse_response_strips_trailing_period() -> None:
    raw = '{"query": "object(X, cube).", "type": "object"}'
    parsed = parse_response(raw)
    assert parsed.query == "object(X, cube)"


def test_parse_response_rejects_non_json() -> None:
    with pytest.raises(ValueError, match="not valid JSON"):
        parse_response("this is not json")


def test_parse_response_rejects_missing_query() -> None:
    with pytest.raises(ValueError, match="'query'"):
        parse_response('{"type": "boolean"}')


def test_parse_response_rejects_bad_type() -> None:
    with pytest.raises(ValueError, match="'type'"):
        parse_response('{"query": "foo", "type": "nonsense"}')


def test_parse_response_rejects_empty_query() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        parse_response('{"query": "   ", "type": "boolean"}')


# ----- validate_query ----- #


def test_validate_query_accepts_valid_succeeds(clevr_kb_source) -> None:
    assert validate_query(clevr_kb_source, "object(X, cube)") is None


def test_validate_query_accepts_valid_fails_no_solutions(clevr_kb_source) -> None:
    """No-solutions ≠ error. Query is well-formed; the answer is just 'no'."""
    assert validate_query(clevr_kb_source, "object(X, hovercraft)") is None


def test_validate_query_rejects_unknown_predicate(clevr_kb_source) -> None:
    err = validate_query(clevr_kb_source, "nonexistent_predicate(X)")
    assert err is not None
    assert "ERROR" in err or "Unknown procedure" in err


def test_validate_query_rejects_syntax_error(clevr_kb_source) -> None:
    err = validate_query(clevr_kb_source, "object(X cube)")
    assert err is not None


def test_validate_query_rejects_arity_mismatch(clevr_kb_source) -> None:
    err = validate_query(clevr_kb_source, "object(X, Y, Z, W)")
    assert err is not None


# ----- TranslatorPipeline ----- #


def test_pipeline_single_attempt_success(clevr_kb_source, clevr_schema) -> None:
    """Backend returns a valid query on the first try → no retries."""
    backend = MockBackend(
        responses=[
            '{"query": "object(X, cube), attribute(X, color, red)", '
            '"type": "boolean"}'
        ]
    )
    pipeline = TranslatorPipeline(backend=backend)
    result = pipeline.translate("Is there a red cube?", clevr_kb_source, clevr_schema)
    assert result.parsed.query == "object(X, cube), attribute(X, color, red)"
    assert result.parsed.type == "boolean"
    assert result.attempts == ()
    assert len(backend.calls) == 1


def test_pipeline_retry_loop_recovers_after_bad_response(
    clevr_kb_source, clevr_schema
) -> None:
    """Phase 2 exit criterion: bad first response → retry → success on 2nd."""
    bad = '{"query": "nonexistent_predicate(X)", "type": "object", "bind_variable": "X"}'
    good = '{"query": "object(X, cube)", "type": "object", "bind_variable": "X"}'
    backend = MockBackend(responses=[bad, good])
    pipeline = TranslatorPipeline(backend=backend)

    result = pipeline.translate("Find any cube.", clevr_kb_source, clevr_schema)

    # Recovered: the final parsed query is the good one.
    assert result.parsed.query == "object(X, cube)"
    # Exactly one prior failed attempt recorded.
    assert len(result.attempts) == 1
    assert "nonexistent_predicate" in result.attempts[0].raw_response
    # The error string from the validator is non-empty and references the bad
    # predicate or an ERROR marker.
    assert result.attempts[0].error
    # Two backend calls happened (one bad, one good).
    assert len(backend.calls) == 2


def test_pipeline_retry_loop_feeds_error_back_to_model(
    clevr_kb_source, clevr_schema
) -> None:
    """The second user prompt should carry the first attempt's error."""
    bad = '{"query": "nonexistent_predicate(X)", "type": "object", "bind_variable": "X"}'
    good = '{"query": "object(X, cube)", "type": "object", "bind_variable": "X"}'
    backend = MockBackend(responses=[bad, good])
    TranslatorPipeline(backend=backend).translate(
        "Find any cube.", clevr_kb_source, clevr_schema
    )

    _, second_user_prompt = backend.calls[1]
    assert "PRIOR ATTEMPTS" in second_user_prompt
    assert bad in second_user_prompt  # raw bad output is shown
    # Error mentions either the bad predicate or an ERROR marker.
    assert (
        "nonexistent_predicate" in second_user_prompt
        or "ERROR" in second_user_prompt
        or "Unknown procedure" in second_user_prompt
    )


def test_pipeline_retry_loop_recovers_after_malformed_json(
    clevr_kb_source, clevr_schema
) -> None:
    """JSON parse errors are also retried, not just validation errors."""
    bad = "not json at all"
    good = '{"query": "object(X, cube)", "type": "object", "bind_variable": "X"}'
    backend = MockBackend(responses=[bad, good])
    result = TranslatorPipeline(backend=backend).translate(
        "Find any cube.", clevr_kb_source, clevr_schema
    )
    assert result.parsed.query == "object(X, cube)"
    assert len(result.attempts) == 1
    assert "JSON" in result.attempts[0].error


def test_pipeline_exhausts_max_retries(clevr_kb_source, clevr_schema) -> None:
    """All 3 attempts bad → TranslationError with full attempt history."""
    bad = '{"query": "nonexistent_predicate(X)", "type": "object"}'
    backend = MockBackend(responses=[bad, bad, bad])
    pipeline = TranslatorPipeline(backend=backend, max_attempts=3)

    with pytest.raises(TranslationError) as exc_info:
        pipeline.translate("Find any cube.", clevr_kb_source, clevr_schema)

    assert len(exc_info.value.attempts) == 3
    assert all(a.error for a in exc_info.value.attempts)
    assert len(backend.calls) == 3


def test_pipeline_max_attempts_one(clevr_kb_source, clevr_schema) -> None:
    """max_attempts=1 means no retries — a bad first response fails immediately."""
    backend = MockBackend(
        responses=['{"query": "nonexistent_predicate(X)", "type": "object"}']
    )
    pipeline = TranslatorPipeline(backend=backend, max_attempts=1)
    with pytest.raises(TranslationError):
        pipeline.translate("?", clevr_kb_source, clevr_schema)
    assert len(backend.calls) == 1


# ----- Schema builder presence-only behavior ----- #


def test_schema_builder_lists_only_present_categories() -> None:
    """Single-object scene should not surface unused categories."""
    kb = generate(presets.single_object())
    schema = build_schema(kb)
    assert "cube" in schema
    assert "sphere" not in schema
    assert "cylinder" not in schema


def test_schema_builder_lists_only_present_attribute_values() -> None:
    """If only red is in the scene, blue should not appear in the menu."""
    sg = (
        SyntheticScene()
        .add_object("obj_0", "cube", color="red")
        .to_scene_graph()
    )
    schema = build_schema(generate(sg))
    assert "red" in schema
    assert "blue" not in schema
    assert "color" in schema
    # No size attribute exists; size values shouldn't show up.
    assert "size values" not in schema


def test_schema_builder_lists_only_present_relations() -> None:
    """Relations section should reflect only the relations asserted in the scene."""
    sg = (
        SyntheticScene()
        .add_object("obj_0", "cube")
        .add_object("obj_1", "sphere")
        .add_relation("obj_0", "left_of", "obj_1")
        .to_scene_graph()
    )
    schema = build_schema(generate(sg))
    assert "Relation predicates present in facts: left_of" in schema
    assert "above" not in schema.split("Available predicates")[0]


def test_schema_builder_empty_scene_lists_no_categories() -> None:
    schema = build_schema(generate(presets.empty_scene()))
    assert "(none" in schema  # the empty-scene markers


def test_schema_builder_always_lists_canonical_predicates() -> None:
    """Even on an empty scene, the predicate menu lists all derived predicates."""
    schema = build_schema(generate(presets.empty_scene()))
    assert "object(?Id, ?Category)" in schema
    assert "same_color(?A, ?B)" in schema


# ----- Prompt templates ----- #


def test_fewshot_has_15_examples() -> None:
    assert len(FEWSHOT) == 15


def test_fewshot_covers_all_question_types() -> None:
    types = {ex.output.split('"type": "', 1)[1].split('"', 1)[0] for ex in FEWSHOT}
    assert types >= {"boolean", "count", "attribute", "object", "list"}


def test_user_prompt_contains_question_and_schema() -> None:
    schema = "SCHEMA\nfake schema content"
    prompt = build_user_prompt("Is there a cube?", schema)
    assert "SCHEMA" in prompt
    assert "Is there a cube?" in prompt
    assert "EXAMPLES" in prompt
    assert "PRIOR ATTEMPTS" not in prompt  # no retries yet


def test_user_prompt_includes_prior_attempts_on_retry() -> None:
    schema = "SCHEMA\n"
    prompt = build_user_prompt(
        "Q?",
        schema,
        prior_attempts=(("bad_output", "error_message_here"),),
    )
    assert "PRIOR ATTEMPTS" in prompt
    assert "bad_output" in prompt
    assert "error_message_here" in prompt


def test_system_prompt_mentions_json() -> None:
    """OpenAI's response_format=json_object requires the word 'JSON' somewhere."""
    assert "JSON" in SYSTEM_PROMPT


# ----- get_backend factory ----- #


def test_get_backend_local_default(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("NL2PROLOG_BACKEND", raising=False)
    backend = get_backend()
    assert backend.name == "ollama"


def test_get_backend_env_var_local(monkeypatch) -> None:
    monkeypatch.setenv("NL2PROLOG_BACKEND", "local")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert get_backend().name == "ollama"


def test_get_backend_openai_without_key_falls_back(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("NL2PROLOG_BACKEND", "openai")
    backend = get_backend()
    assert backend.name == "ollama"


def test_get_backend_openai_without_key_no_fallback_raises(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("NL2PROLOG_BACKEND", "openai")
    with pytest.raises(NoBackendAvailableError):
        get_backend(allow_fallback=False)


def test_get_backend_unknown_name_raises(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("NL2PROLOG_BACKEND", raising=False)
    with pytest.raises(ValueError):
        get_backend("nonsense")
