# Phase 4 Cleanup

Atomic commit `ba0c96e` (pushed to `main`). Four fixes plus one supporting comparator-symmetry change. CI carries over from Phase 4's last green run.

## What changed per fix

### FIX 1 — list-qtype golden cases

- `evaluation/golden_dataset.py` (+43/-6 lines)
  - Added `_ids_matching(scene_factory, **filters)` helper that walks the SceneGraph at module import.
  - Added `_LIST` tuple with two cases:
    - **L1** "List all red objects." → `expected = _ids_matching(presets.clevr_like, color="red")` → `["obj_0"]`.
    - **L2** "List all metal objects." → `expected = _ids_matching(presets.clevr_like, material="metal")` → `["obj_0", "obj_2"]`.
  - `GOLDEN_DATASET` now concatenates `_LIST` and the assertion is `len == 32`.
  - `QTYPE_BUCKETS` gained `("list", _LIST)`.
- `evaluation/harness.py` (+13/-6 lines)
  - **Comparator symmetry bug fix** required to make FIX 1 work as literally specified. The previous `is_correct` list-branch resolved obj_ids → categories on the *predicted* side but left the *expected* side as raw strings, so `expected=["obj_0"]` could never match `answer=["obj_0"]`. Replaced with a symmetric `_norm(x)` that normalizes either side: if `x` looks like an obj_id it resolves to category; otherwise it passes through. Now both `expected=["obj_0"]` (IDs, as the instruction asks) and `expected=["cube"]` (categories, the prior convention) work.
  - One-line update to `_print_table` so the new "list" bucket shows up in the rich-table summary.

### FIX 2 — CLI exit code

- `evaluation/cli.py` (+30/-11 lines)
  - Added module constant `SYNTHETIC_ACCURACY_FLOOR = 0.50`.
  - Added pure helper `compute_exit_code(synthetic_summary, *, threshold=…)` that's the testable side of the exit-code policy.
  - `main()` now passes `synthetic_summary` through `compute_exit_code` for its return value. CLEVR results never gate exit code — per the instruction, only synthetic is gated.
  - When accuracy falls below the floor, the CLI prints to **stderr** (was stdout) so CI captures it cleanly.

### FIX 3 — `clevr_subset.py` rewrite

- `evaluation/clevr_subset.py` (full rewrite; net +197 lines)
  - Dropped the dead HuggingFace URL fetcher (5 mirror URLs that all 404'd in Phase 4).
  - New module-level constants for the CLEVR vocabularies (`CLEVR_COLORS/SHAPES/SIZES/MATERIALS`).
  - `_load_scenes()` — parses `data/clevr_test_subset/scenes.json`. Returns `[]` on any I/O or JSON error. Accepts both `{info, scenes}` and bare list payloads.
  - `_image_path_for(scene)` — resolves the PNG via `image_filename` field, falling back to `CLEVR_val_{image_index:06d}.png`.
  - `_most_common(items)` — alphabetical tie-break so case_ids stay stable across runs.
  - `_missing_combo(scene_objects)` — first absent color paired with the first CLEVR shape; falls back to any absent `(color, shape)` pair.
  - `_cases_for_scene(...)` — emits exactly 5 cases per scene matching the spec.
  - `iter_cases()` returns `list[ImageEvalCase]` (not a generator); `[]` on any error, no exceptions raised.
  - `ensure_clevr_subset()` keeps the `(cases, status_dict)` shape the CLI already expects, but the status dict is now disk-oriented (no `attempted_fetch` flag).
- `scripts/download_clevr.sh` (new, 38 lines, executable)
  - Does NOT auto-download. Prints the canonical 18GB URL and the selective-unzip recipe (curl + `unzip -p` slicing).

### FIX 4 — tests

- `tests/test_eval.py` (+221/-2 lines)
  - Renamed `test_golden_dataset_has_thirty_cases` → `test_golden_dataset_has_thirty_two_cases`.
  - Replaced `test_golden_dataset_bucket_sizes_match_plan` with one that expects `{existence: 6, count: 6, attribute: 6, spatial: 6, multi_hop: 6, list: 2}` exactly.
  - Added `test_golden_dataset_list_cases_have_nonempty_expected`.
  - Added three comparator tests for the new list-qtype symmetry — IDs side, categories side, and a negative case.
  - Added six tests for `compute_exit_code` (None, ≥threshold, boundary, <threshold, zero accuracy, custom threshold).
  - Added five CLEVR loader tests, all `@pytest.mark.skipif(not Path("data/clevr_test_subset/scenes.json").exists(), ...)`:
    - `test_clevr_iter_cases_count_matches_formula` — count derived from `scenes.json`, not hardcoded.
    - `test_clevr_iter_cases_image_paths_exist_on_disk`
    - `test_clevr_existence_cases_match_scene_ground_truth` — re-verifies positive/negative expected against scenes.json by parsing the case_id and looking up the scene.
    - `test_clevr_count_total_matches_scene_object_count`
    - `test_clevr_iter_cases_returns_empty_when_no_data` (no skipif; uses `tmp_path` + monkeypatch).

## CLEVR cases generated — distribution

50 cases total (5 per scene × 10 scenes). By qtype:

| qtype     | count |
|-----------|------:|
| count     | 30    |
| boolean   | 20    |

By case-id pattern (10 of each):
- `CLEVR_{i}_count_total`
- `CLEVR_{i}_count_{most_common_color}`
- `CLEVR_{i}_count_{most_common_material}`
- `CLEVR_{i}_exists_{color}_{shape}` (positive)
- `CLEVR_{i}_not_exists_{color}_{shape}` (negative)

## Full verification output

### `.venv/bin/python -m pytest -m "not slow" -v`

```
====================== 132 passed, 11 deselected in 2.67s ======================
```

(117 baseline + 15 new tests: 1 list-cases-nonempty, 3 list-comparator, 6 exit-code, 5 CLEVR loader. 11 slow tests deselected as expected.)

### `.venv/bin/python -c "from evaluation.clevr_subset import iter_cases; ..."`

```
CLEVR cases generated: 50
First: CLEVR_0_count_total - How many objects are there? - expected=5
```

### `.venv/bin/python -c "from evaluation.golden_dataset import GOLDEN_DATASET; ..."`

```
Golden cases: 32
List cases: 2
```

## Phase 5 notes (non-blocking but worth surfacing)

1. **Comparator-symmetry change is a real semantic shift** for any future `list`-qtype callers — `expected=["obj_0"]` now resolves to a category, so a list of `obj_` IDs is treated identically to its categories. If a future case wants to distinguish two same-category objects (e.g. two red cubes), the comparator will not currently see them as distinct. Add a `compare_by="id"` knob to `EvalCase` if/when that matters.

2. **CLEVR question style is intentionally narrow** — only `count` and `boolean` are produced from CLEVR scenes. The richer `attribute`/`object`/`spatial` shapes are left for the synthetic suite. If Phase 5 wants vision-mode coverage of attribute/spatial questions on CLEVR, extend `_cases_for_scene` and add per-shape templates.

3. **CLI threshold lives at the CLI layer**, not the harness. If the Gradio demo or a notebook needs the same gate, copy `SYNTHETIC_ACCURACY_FLOOR` and `compute_exit_code` rather than expecting them from `evaluation/harness.py`.

4. **`scripts/download_clevr.sh` is documentation, not automation.** Phase 5 might want to add a `--dataset` URL flag to the CLI that points at an extracted subset directory other than `data/clevr_test_subset/`, so multiple subsets (val/test/CoGenT) can coexist.

5. **No CLEVR vision-mode eval was run** in this task — the instructions explicitly forbade loading OWL-ViT/CLIP. The next phase should run `python -m evaluation.cli --suite clevr` to produce `evaluation/results/clevr.json` and update the README accordingly.

6. **Phase 1/2/3 modules were not touched.** The comparator symmetry fix in `evaluation/harness.py` is Phase-4-internal; everything else (synthetic presets, KB generator, executor, verbalizer, vision pipeline, NL→Prolog translator) is unchanged.

---

STOP. Phase 5 not started.
