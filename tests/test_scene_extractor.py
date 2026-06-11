"""Slow end-to-end test of the vision pipeline.

Loads OWL-ViT + CLIP on the system's preferred device (MPS on Apple Silicon)
and runs the full extractor on a synthetic CLEVR-style scene built by the
test fixture. Marked slow so CI excludes it via `pytest -m "not slow"`.

Skips cleanly if the models aren't cached AND no network is available —
keeps `pytest tests/` runnable on a fresh checkout without an out-of-band
download step.
"""

from __future__ import annotations

import socket

import pytest

pytestmark = pytest.mark.slow


def _has_network(host: str = "huggingface.co", timeout: float = 2.0) -> bool:
    try:
        sock = socket.create_connection((host, 443), timeout=timeout)
        sock.close()
        return True
    except OSError:
        return False


def _models_cached() -> bool:
    """True if both vision model snapshots are already on disk."""
    from pathlib import Path

    hf_home = Path.home() / ".cache" / "huggingface" / "hub"
    return (
        (hf_home / "models--google--owlvit-base-patch32").exists()
        and (hf_home / "models--openai--clip-vit-base-patch32").exists()
    )


@pytest.fixture(scope="module")
def extractor():
    if not _models_cached() and not _has_network():
        pytest.skip(
            "vision models not cached and HuggingFace is unreachable; "
            "run scripts/download_models.sh first"
        )
    from scene_extractor.extractor import SceneExtractor

    return SceneExtractor()


@pytest.fixture(scope="module")
def synth_scene_graph(extractor):
    from tests.fixtures.synth_image import make_clevr_like_image

    img = make_clevr_like_image()
    return extractor.extract(img)


def test_extractor_returns_scene_graph(synth_scene_graph) -> None:
    from scene_extractor.schema import SceneGraph

    assert isinstance(synth_scene_graph, SceneGraph)


def test_extractor_records_model_versions(synth_scene_graph) -> None:
    assert "detector" in synth_scene_graph.model_versions
    assert "clip" in synth_scene_graph.model_versions
    assert synth_scene_graph.model_versions["detector"] == "google/owlvit-base-patch32"
    assert synth_scene_graph.model_versions["clip"] == "openai/clip-vit-base-patch32"


def test_extractor_records_latency(synth_scene_graph) -> None:
    # First-call latency is dominated by model load; module-scope ensures we
    # measure ONE warm inference here.
    assert synth_scene_graph.extraction_time_ms > 0


def test_all_bboxes_normalized_in_unit_interval(synth_scene_graph) -> None:
    for obj in synth_scene_graph.objects:
        assert 0.0 <= obj.bbox.x1 < obj.bbox.x2 <= 1.0
        assert 0.0 <= obj.bbox.y1 < obj.bbox.y2 <= 1.0


def test_object_ids_unique(synth_scene_graph) -> None:
    ids = [o.id for o in synth_scene_graph.objects]
    assert len(set(ids)) == len(ids)


def test_relations_reference_known_objects(synth_scene_graph) -> None:
    obj_ids = {o.id for o in synth_scene_graph.objects}
    for rel in synth_scene_graph.relations:
        assert rel.subject_id in obj_ids
        assert rel.object_id in obj_ids


def test_attributes_use_known_vocab(synth_scene_graph) -> None:
    from scene_extractor.config import ATTRIBUTE_VOCAB

    for obj in synth_scene_graph.objects:
        for family, value in obj.attributes.items():
            assert family in ATTRIBUTE_VOCAB
            assert value in ATTRIBUTE_VOCAB[family]


def test_attribute_confidences_match_attributes(synth_scene_graph) -> None:
    for obj in synth_scene_graph.objects:
        assert set(obj.attribute_confidences.keys()) == set(obj.attributes.keys())
        for conf in obj.attribute_confidences.values():
            assert 0.0 <= conf <= 1.0


def test_detection_count_nonzero_on_high_contrast_scene(synth_scene_graph) -> None:
    # Sanity check the pipeline isn't returning an empty list on every input;
    # this is the closest we can get to an "is the system working" assertion
    # without a real CLEVR image. Synthetic colored shapes against a flat
    # background reliably produce ≥1 detection at OWL-ViT threshold 0.1.
    assert len(synth_scene_graph.objects) >= 1
