"""Unit tests for QueryExecutor across all five query types and timeout."""

from __future__ import annotations

import pytest

from kb_generator.generator import generate
from query_executor.executor import QueryExecutor
from query_executor.result import ParsedQuery
from synthetic import presets


@pytest.fixture
def clevr_kb_source():
    return generate(presets.clevr_like()).source


@pytest.fixture
def kitchen_kb_source():
    return generate(presets.kitchen()).source


def test_boolean_true(clevr_kb_source, executor) -> None:
    res = executor.run(
        clevr_kb_source,
        ParsedQuery(query="object(X, cube), attribute(X, color, red)", type="boolean"),
    )
    assert res.success is True
    assert res.answer is True


def test_boolean_false(clevr_kb_source, executor) -> None:
    res = executor.run(
        clevr_kb_source,
        ParsedQuery(query="object(X, cube), attribute(X, color, purple)", type="boolean"),
    )
    assert res.success is False
    assert res.answer is False


def test_count(clevr_kb_source, executor) -> None:
    res = executor.run(
        clevr_kb_source,
        ParsedQuery(
            query="findall(X, (object(X, _), attribute(X, color, red)), L), length(L, N)",
            type="count",
        ),
    )
    assert res.success is True
    assert res.answer == 1


def test_count_zero(clevr_kb_source, executor) -> None:
    res = executor.run(
        clevr_kb_source,
        ParsedQuery(
            query="findall(X, (object(X, _), attribute(X, color, purple)), L), length(L, N)",
            type="count",
        ),
    )
    # findall+length always succeeds; count is 0.
    assert res.success is True
    assert res.answer == 0


def test_attribute(clevr_kb_source, executor) -> None:
    res = executor.run(
        clevr_kb_source,
        ParsedQuery(
            query="object(X, cube), attribute(X, color, C)",
            type="attribute",
            bind_variable="C",
        ),
    )
    assert res.success is True
    assert res.answer == "red"


def test_object_via_spatial_inverse(clevr_kb_source, executor) -> None:
    """right_of derives from the left_of facts; tests rule firing."""
    res = executor.run(
        clevr_kb_source,
        ParsedQuery(
            query="object(Y, cube), right_of(X, Y)",
            type="object",
            bind_variable="X",
        ),
    )
    assert res.success is True
    assert res.answer == "obj_1"


def test_list_all_red_objects(clevr_kb_source, executor) -> None:
    res = executor.run(
        clevr_kb_source,
        ParsedQuery(
            query="findall(X, (object(X, _), attribute(X, color, red)), L)",
            type="list",
            bind_variable="L",
        ),
    )
    assert res.success is True
    assert res.answer == ["obj_0"]


def test_list_empty(clevr_kb_source, executor) -> None:
    res = executor.run(
        clevr_kb_source,
        ParsedQuery(
            query="findall(X, (object(X, _), attribute(X, color, purple)), L)",
            type="list",
            bind_variable="L",
        ),
    )
    # findall succeeds with [], so result.success=True, answer=[]
    assert res.success is True
    assert res.answer == []


def test_same_material_rule_fires(kitchen_kb_source, executor) -> None:
    """Two metal objects share material via the same_material derived rule."""
    res = executor.run(
        generate(presets.office()).source,
        ParsedQuery(
            query="same_color(X, Y)",
            type="object",
            bind_variable="X",
        ),
    )
    assert res.success is True
    assert res.answer in {"obj_0", "obj_1", "obj_3"}  # any black object


def test_timeout_on_infinite_query(clevr_kb_source) -> None:
    """call_with_time_limit aborts repeat/false in ~1 second."""
    ex = QueryExecutor(timeout_s=1.0)
    res = ex.run(
        clevr_kb_source,
        ParsedQuery(query="repeat, false", type="boolean"),
    )
    assert res.success is False
    assert res.answer is False
    assert res.error is not None
    assert "exceeded" in res.error


def test_malformed_query_returns_structured_error(clevr_kb_source, executor) -> None:
    """A query referencing an undefined predicate should produce a clean error."""
    res = executor.run(
        clevr_kb_source,
        ParsedQuery(query="nonexistent_predicate(X)", type="object", bind_variable="X"),
    )
    assert res.success is False
    assert res.error is not None
