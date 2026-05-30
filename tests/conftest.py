"""Shared pytest fixtures for Phase 1 tests."""

from __future__ import annotations

import pytest

from kb_generator.generator import KBProgram, generate
from query_executor.executor import QueryExecutor
from scene_extractor.schema import SceneGraph
from synthetic import presets


@pytest.fixture
def clevr_scene() -> SceneGraph:
    return presets.clevr_like()


@pytest.fixture
def kitchen_scene() -> SceneGraph:
    return presets.kitchen()


@pytest.fixture
def office_scene() -> SceneGraph:
    return presets.office()


@pytest.fixture
def single_object_scene() -> SceneGraph:
    return presets.single_object()


@pytest.fixture
def empty_scene() -> SceneGraph:
    return presets.empty_scene()


@pytest.fixture
def clevr_kb(clevr_scene: SceneGraph) -> KBProgram:
    return generate(clevr_scene)


@pytest.fixture
def executor() -> QueryExecutor:
    return QueryExecutor(timeout_s=3.0)
