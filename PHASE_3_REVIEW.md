# Phase 3 Review — Vision Pipeline

## Files Present

Confirmed via `git status` and direct inspection (all paths absolute under `/Users/shreyaskiran/projects/neurosymbolic-vqa/`):

Tracked / modified:
- `pyproject.toml` — adds torch, torchvision, transformers, Pillow, numpy
- `requirements.txt` — same five additions
- `scene_extractor/__init__.py` — rewritten to export `SceneExtractor`, `ModelDownloadError`, schema types

Untracked Phase 3 files:
- `scene_extractor/config.py` — vocabs + thresholds + model IDs
- `scene_extractor/models.py` — `get_detector`, `get_clip`, `get_device`, `clear_caches`
- `scene_extractor/attribute_classifier.py` — CLIP attribute scoring
- `scene_extractor/spatial_relations.py` — pure-geometric rules
- `scene_extractor/extractor.py` — `SceneExtractor` orchestrator + `ModelDownloadError`
- `tests/test_spatial_relations.py` — 13 unit tests for geometric rules
- `tests/test_scene_extractor.py` — 9 slow end-to-end vision tests
- `tests/fixtures/__init__.py` — empty package marker
- `tests/fixtures/synth_image.py` — Pillow-drawn CLEVR-like image
- `scripts/download_models.sh` — pre-warms both HF model caches, executable, idempotent

`scene_extractor/schema.py` was already in place from Phase 1 and is correctly reused.

## Tests Status

- `pytest tests/test_spatial_relations.py -v` → 13 passed in 0.01s.
- `pytest tests/test_scene_extractor.py -v -m slow` → 9 passed in 6.35s (one deprecation `FutureWarning` from transformers about `post_process_object_detection`, non-blocking).
- `pytest -m "not slow" -q` → **89 passed, 11 deselected in 2.34s**. Matches the expected count (43 Phase 1 + 33 Phase 2 + 13 new spatial-relations). No Phase 1/2 regressions.

Test quality notes:
- The slow tests make meaningful assertions (bbox bounds, vocab membership, id uniqueness, relation referent integrity, attribute/confidence key parity, ≥1 detection). None are `assert sg is not None` no-ops.
- `pytestmark = pytest.mark.slow` is applied at module level in `tests/test_scene_extractor.py` (clean, not per-test).
- The slow file gracefully `pytest.skip`s when both the HF cache is empty and HF is unreachable.

## Plan Adherence

Spec items 1–14 from the brief checked one-by-one:

1. **Meaningful test assertions** — yes, see above.
2. **`ModelDownloadError` is raised, not swallowed** — `extractor.py:107-113` wraps the detector load and `extractor.py:161-166` wraps the CLIP classify call. Both re-raise as `ModelDownloadError` with `from exc`. No silent absorption.
3. **Hardcoded model IDs configurable** — `SceneExtractor.__init__` accepts `detector_model_id` and `clip_model_id` (extractor.py:67-68). `get_detector(model_id=...)` and `get_clip(model_id=...)` both take an id arg that participates in the `lru_cache` key.
4. **lru_cache singletons + `clear_caches()`** — both loaders decorated `@functools.lru_cache(maxsize=1)` (models.py:54, 77). `clear_caches()` calls `cache_clear()` on both (models.py:100-103).
5. **MPS selection + CPU fallback + `NSVQA_FORCE_CPU=1`** — `_select_device()` honors `NSVQA_FORCE_CPU` env, picks MPS if available, falls back to CPU. Both `get_detector` and `get_clip` wrap the `model.to(_DEVICE)` call in a `try/except (RuntimeError, NotImplementedError)` with stderr warning and CPU fallback (models.py:66-73, 89-96).
6. **Spatial relations** — every rule matches the plan:
   - `is_left_of`: `cx_A < cx_B - 0.05` ✓
   - `is_above`: `cy_A < cy_B - 0.05` ✓
   - `is_inside`: `area(A) < area(B)` AND `intersection / area(A) > 0.9` ✓
   - `is_on_top_of`: `above` AND `|y2_A - y1_B| < 0.05` AND horizontal overlap > `0.5 * min(width)` ✓
   - `is_next_to`: horizontal gap `< 0.05` AND vertical overlap > `0.5 * min(height)`; symmetry handled in `compute()` via sorted-id `seen_next_to` set, asserted once ✓
   - `is_in_front_of`: same-category only, `area_A / area_B > 1.5` ✓
   - `compute()` emits only canonical `left_of` and `above`; `right_of` / `below` are absent (tests assert this) ✓
7. **Attribute classifier** — fixed vocab per family from `config.ATTRIBUTE_VOCAB`; argmax + 0.35 threshold; sub-threshold attrs are `continue`d (omitted, not asserted as `"unknown"`); returns `(attrs: dict, confidences: dict)` keyed by family. ✓
8. **NMS** — `torchvision.ops.nms` called per-class (loop over `torch.unique(labels)`) at IoU `0.5` from config (`extractor.py:138-145`). ✓
9. **Detection threshold** — default `0.1` via `config.DETECTION_THRESHOLD`; constructor accepts `detection_threshold` override. ✓
10. **Bbox normalization** — `_to_normalized_bbox` divides by `pil.width` / `pil.height`, clamps to `[0,1]`, drops degenerate boxes by returning `None` rather than raising (extractor.py:179-195). Caller skips degenerate ones (`if bbox is None: continue`). ✓
11. **SceneGraph fields** — `image_path`, `objects`, `relations`, `extraction_time_ms`, `model_versions` are all populated (extractor.py:85-94). Pydantic validators in `schema.py` enforce bbox bounds and relation-referent integrity. ✓
12. **Bundled CLEVR image deviation** — the plan called for a ~30KB committed CLEVR image; the implementation uses a Pillow-drawn synthetic image (`tests/fixtures/synth_image.py`). The synthetic image is high-contrast (red disc, blue square, yellow ellipse on gray) and the slow test asserts `len(objects) >= 1` plus all the structural invariants. It exercises the full OWL-ViT → CLIP → spatial pipeline end-to-end, so coverage of the vision pipeline is real. Acceptable deviation; real CLEVR images come in Phase 4. Worth a one-line note in the README or a Phase 3 commit message.
13. **`scripts/download_models.sh`** — pre-warms both models, executable (`-rwxr-xr-x`), idempotent (HF cache no-ops on second run).
14. **CI marker** — `pytestmark = pytest.mark.slow` at module level in `test_scene_extractor.py`. Confirmed.

Deviation audit:

- **torch bumped to `>=2.6,<2.8`** (plan said `>=2.4,<2.6`). The brief notes this is due to CVE-2025-32434 (transformers 4.57 requires torch ≥2.6 to load `.bin` weights). The constraint is reasonable; it is NOT documented in `pyproject.toml`, `requirements.txt`, or a `CHANGELOG`-style file. Recommend a one-line comment in `pyproject.toml` next to the torch pin (non-blocking).
- **CLEVR image deferred to Phase 4** — synthetic image substitute is acceptable as noted in item 12.
- **Phase 1 deviations preserved**:
  - `kb_generator/validator.py` still uses `subprocess.run(["swipl", ...])` (lines 14, 49-78). ✓
  - `query_executor/executor.py` still wraps queries in `call_with_time_limit/2` (line 125). ✓

## Concerns

Nothing blocking. Minor items:

1. **Transformers FutureWarning** — `processor.post_process_object_detection` is deprecated in favor of `post_process_grounded_object_detection`. The call works today but should be migrated before transformers v5. Not a Phase 3 blocker.
2. **torch upper-bound bump undocumented in source** — the CVE rationale lives only in the review brief. A one-line comment in `pyproject.toml` would make this self-explanatory for future maintainers.
3. **`make_clevr_like_image` substitution** — real CLEVR images are deferred to Phase 4's downloader; flag in README when Phase 4 lands so reviewers don't expect a committed CLEVR fixture.

None of these block Phase 3 exit criteria. The vision pipeline runs end-to-end on MPS, yields a populated `SceneGraph` from a synthetic high-contrast image, and all assertions in the slow suite pass.

Verdict: PASS
