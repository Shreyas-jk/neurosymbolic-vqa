"""Unit tests for the pure-geometric spatial relations module.

These run in CI (NOT marked slow) because they don't load any vision models.
The full extractor + model path is exercised by tests/test_scene_extractor.py.
"""

from __future__ import annotations

from scene_extractor import spatial_relations as sr
from scene_extractor.schema import BoundingBox, SceneObject


def _bb(x1: float, y1: float, x2: float, y2: float) -> BoundingBox:
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)


def _obj(id_: str, cat: str, bbox: BoundingBox) -> SceneObject:
    return SceneObject(id=id_, category=cat, bbox=bbox)


# ----- pure geometric predicates -----------------------------------------


def test_left_of_basic() -> None:
    a = _bb(0.0, 0.4, 0.2, 0.6)
    b = _bb(0.6, 0.4, 0.8, 0.6)
    assert sr.is_left_of(a, b)
    assert not sr.is_left_of(b, a)


def test_left_of_ignores_near_equal_centers() -> None:
    # Centers within 0.05 margin → ambiguous → neither direction
    a = _bb(0.30, 0.4, 0.40, 0.6)  # cx = 0.35
    b = _bb(0.32, 0.4, 0.42, 0.6)  # cx = 0.37 — within 0.05 margin
    assert not sr.is_left_of(a, b)
    assert not sr.is_left_of(b, a)


def test_above_basic() -> None:
    top = _bb(0.4, 0.0, 0.6, 0.2)
    bot = _bb(0.4, 0.6, 0.6, 0.8)
    assert sr.is_above(top, bot)
    assert not sr.is_above(bot, top)


def test_inside_requires_smaller_area_and_high_overlap() -> None:
    outer = _bb(0.0, 0.0, 1.0, 1.0)
    inner = _bb(0.4, 0.4, 0.5, 0.5)
    assert sr.is_inside(inner, outer)
    # Reverse direction: outer is not inside inner.
    assert not sr.is_inside(outer, inner)


def test_inside_rejects_equal_size_overlap() -> None:
    a = _bb(0.0, 0.0, 0.5, 0.5)
    b = _bb(0.1, 0.1, 0.6, 0.6)  # overlaps but neither contains the other
    assert not sr.is_inside(a, b)
    assert not sr.is_inside(b, a)


def test_on_top_of_requires_above_plus_bottom_top_contact() -> None:
    top = _bb(0.40, 0.10, 0.60, 0.49)
    bot = _bb(0.40, 0.50, 0.60, 0.90)
    assert sr.is_on_top_of(top, bot)
    # Not on top if separation exceeds tolerance.
    floating = _bb(0.40, 0.05, 0.60, 0.10)
    assert not sr.is_on_top_of(floating, bot)


def test_next_to_horizontal_proximity_required() -> None:
    a = _bb(0.10, 0.40, 0.30, 0.60)
    b = _bb(0.32, 0.40, 0.50, 0.60)  # 0.02 gap, vertically overlapping
    assert sr.is_next_to(a, b)
    assert sr.is_next_to(b, a)  # symmetric
    far = _bb(0.80, 0.40, 0.95, 0.60)
    assert not sr.is_next_to(a, far)


def test_in_front_of_same_category_only_with_size_ratio() -> None:
    big = _obj("o1", "cube", _bb(0.10, 0.10, 0.90, 0.90))
    small = _obj("o2", "cube", _bb(0.40, 0.40, 0.50, 0.50))
    assert sr.is_in_front_of(big, small)
    assert not sr.is_in_front_of(small, big)
    # Different category → no inference.
    diff = _obj("o3", "sphere", _bb(0.40, 0.40, 0.50, 0.50))
    assert not sr.is_in_front_of(big, diff)


# ----- compute(): full pairwise pass -------------------------------------


def test_compute_emits_left_of_only_canonical_direction() -> None:
    a = _obj("a", "cube", _bb(0.0, 0.4, 0.2, 0.6))
    b = _obj("b", "cube", _bb(0.6, 0.4, 0.8, 0.6))
    rels = sr.compute([a, b])
    left_of = [(r.subject_id, r.object_id) for r in rels if r.predicate == "left_of"]
    assert ("a", "b") in left_of
    # right_of is NOT asserted — derived by Prolog rules.
    assert "right_of" not in {r.predicate for r in rels}


def test_compute_emits_next_to_once_per_pair() -> None:
    a = _obj("a", "cube", _bb(0.10, 0.40, 0.30, 0.60))
    b = _obj("b", "sphere", _bb(0.32, 0.40, 0.50, 0.60))
    rels = sr.compute([a, b])
    nexts = [r for r in rels if r.predicate == "next_to"]
    assert len(nexts) == 1
    # Sorted-id canonical form.
    assert (nexts[0].subject_id, nexts[0].object_id) == ("a", "b")


def test_compute_empty_object_list_returns_empty() -> None:
    assert sr.compute([]) == []


def test_compute_skips_self_pairs() -> None:
    a = _obj("a", "cube", _bb(0.0, 0.0, 0.5, 0.5))
    rels = sr.compute([a])
    assert rels == []


def test_compute_above_and_left_of_combine_when_diagonal() -> None:
    top_left = _obj("tl", "cube", _bb(0.0, 0.0, 0.2, 0.2))
    bot_right = _obj("br", "cube", _bb(0.6, 0.6, 0.8, 0.8))
    rels = sr.compute([top_left, bot_right])
    edges = {(r.subject_id, r.predicate, r.object_id) for r in rels}
    assert ("tl", "left_of", "br") in edges
    assert ("tl", "above", "br") in edges
