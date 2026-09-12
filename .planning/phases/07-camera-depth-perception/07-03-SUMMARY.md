---
phase: 07-camera-depth-perception
plan: 03
subsystem: dataset
tags: [rlds, tfds, tensorflow_datasets, oxe, openvla, depth-perception, libero]

# Dependency graph
requires:
  - phase: 07-camera-depth-perception (plan 02)
    provides: "agentview_depth float32 (H,W,1) [0,1] dataset persisted per-episode in the robomimic HDF5 writer output"
provides:
  - "validate_episode_arrays() extended with a fail-loud agentview_depth dtype/shape/finite/range branch"
  - "load_episodes_from_hdf5() reads agentview_depth per episode (fail-loud, no fallback)"
  - "_episode_to_rlds_steps() includes agentview_depth per RLDS step observation"
  - "hdf5_to_rlds()'s TFDS FeaturesDict declares agentview_depth as a (None,None,1) float32 Tensor (not Image)"
  - "oxe_register.py's depth_obs_keys[\"primary\"] = \"agentview_depth\" in both the in-memory dict and the on-disk patch template (marker bumped v2 -> v3)"
affects: [phase-9-data-collection, phase-9-tune-05-refinetuning]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Depth arrays validated with the same fail-loud dtype/ndim/last-dim/finite pattern as RGB/state/actions, plus a normalized-[0,1]-range check unique to depth"
    - "Legacy-schema detection via explicit pytest.skip (not a silent .get()-style fallback) when a real HDF5 predates a schema-tightening change"

key-files:
  created: []
  modified:
    - LIBERO/libero/libero/datasets/rlds_converter.py
    - LIBERO/libero/libero/datasets/test_rlds_converter.py
    - LIBERO/libero/libero/datasets/oxe_register.py
    - LIBERO/libero/libero/datasets/test_oxe_register.py

key-decisions:
  - "load_episodes_from_hdf5() requires agentview_depth unconditionally (direct dict access, no .get() fallback) -- a legacy pre-Phase-7 HDF5 missing the key surfaces as a KeyError at load time, matching the existing fail-loud style for other obs keys"
  - "test_hdf5_to_rlds_against_real_dataset explicitly detects a legacy HDF5 (missing agentview_depth) via a pre-flight h5py.File check and skips with a documented reason, rather than papering over the schema change with a fallback in production code"
  - "depth_obs_keys marker bumped v2 -> v3 so a previously-patched installed prismatic configs.py file still receives this fix on next apply_soarm_spatial_registration() call"

patterns-established:
  - "Fifth positional parameter (agentview_depth) inserted between eye_in_hand_rgb and state across validate_episode_arrays/load_episodes_from_hdf5/_episode_to_rlds_steps/hdf5_to_rlds, following the image-then-depth-then-proprio grouping already implicit in the arrays dict"

requirements-completed: [DEPTH-03]

coverage:
  - id: D1
    description: "validate_episode_arrays() rejects wrong-dtype, wrong-ndim, wrong-last-dim, non-finite, and out-of-[0,1]-range agentview_depth arrays; accepts well-formed depth"
    requirement: "DEPTH-03"
    verification:
      - kind: unit
        ref: "LIBERO/libero/libero/datasets/test_rlds_converter.py::test_validate_episode_arrays_accepts_well_formed_episode"
        status: pass
      - kind: unit
        ref: "LIBERO/libero/libero/datasets/test_rlds_converter.py::test_validate_episode_arrays_rejects_wrong_dtype_depth"
        status: pass
      - kind: unit
        ref: "LIBERO/libero/libero/datasets/test_rlds_converter.py::test_validate_episode_arrays_rejects_out_of_range_depth"
        status: pass
    human_judgment: false
  - id: D2
    description: "load_episodes_from_hdf5() returns episode dicts containing agentview_depth (float32, last-dim 1)"
    requirement: "DEPTH-03"
    verification:
      - kind: integration
        ref: "LIBERO/libero/libero/datasets/test_rlds_converter.py::test_load_episodes_from_hdf5_preserves_episode_boundaries_and_derives_language"
        status: pass
    human_judgment: false
  - id: D3
    description: "hdf5_to_rlds()'s TFDS FeaturesDict declares agentview_depth as a (None,None,1) float32 Tensor, confirmed via persisted dataset_info.json schema"
    requirement: "DEPTH-03"
    verification:
      - kind: integration
        ref: "LIBERO/libero/libero/datasets/test_rlds_converter.py::test_hdf5_to_rlds_writes_tfds_loadable_dataset"
        status: unknown
    human_judgment: true
    rationale: "tensorflow_datasets is not installed in this project's local libero conda env -- this test is guarded by pytest.importorskip and skips locally (matches Phase 6's established baseline); it must be exercised for real on Colab once tensorflow_datasets is installed there."
  - id: D4
    description: "A pre-Phase-7 legacy HDF5 file lacking agentview_depth causes the real-dataset test to skip with a documented reason, not crash with an unhandled KeyError"
    requirement: "DEPTH-03"
    verification:
      - kind: integration
        ref: "LIBERO/libero/libero/datasets/test_rlds_converter.py::test_hdf5_to_rlds_against_real_dataset"
        status: unknown
    human_judgment: true
    rationale: "This worktree has no real *_demo.hdf5 dataset file on disk (data files are gitignored), so the test skips via the pre-existing 'No real dataset found' path rather than exercising the new depth-presence skip branch. The code path (h5py.File pre-flight check + pytest.skip with the depth-specific message) is implemented and reviewed but not executed against the actual legacy file in this run -- must be verified against the real file (confirmed present in the main checkout per 07-03-PLAN.md's verified_this_session notes) before Phase 7 sign-off."
  - id: D5
    description: "oxe_register.py's depth_obs_keys[\"primary\"] is \"agentview_depth\" in both the in-memory dict and the on-disk patch template; wrist stays None; marker bumped v2->v3"
    requirement: "DEPTH-03"
    verification:
      - kind: unit
        ref: "LIBERO/libero/libero/datasets/test_oxe_register.py::test_register_soarm_spatial_injects_expected_dict_shape"
        status: pass
      - kind: other
        ref: "grep -v '^\\s*#' LIBERO/libero/libero/datasets/oxe_register.py | grep -c '\"agentview_depth\"' == 3 (>= 2 required)"
        status: pass
    human_judgment: false

duration: 18min
completed: 2026-09-12
status: complete
---

# Phase 7 Plan 03: RLDS Depth Conversion + OXE Registration Summary

**`hdf5_to_rlds()` now carries `agentview_depth` through validation, HDF5 loading, and the TFDS `FeaturesDict` as a `(None,None,1)` float32 `Tensor`; `oxe_register.py`'s `depth_obs_keys["primary"]` is `"agentview_depth"` in both the in-memory dict and the on-disk patch template torchrun's subprocess reads.**

## Performance

- **Duration:** 18 min
- **Started:** 2026-09-12T06:33:00Z
- **Completed:** 2026-09-12T06:51:18Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments

- `validate_episode_arrays()` extended with `agentview_depth` as a new third positional parameter, gaining fail-loud dtype/ndim/last-dim/finite/range validation matching the existing RGB/state/actions style.
- `load_episodes_from_hdf5()` reads `obs["agentview_depth"]` per episode directly (no fallback) and includes it in each returned episode dict.
- `_episode_to_rlds_steps()` includes `agentview_depth` in each step's `observation` dict.
- `hdf5_to_rlds()`'s TFDS `FeaturesDict` declares `"agentview_depth": tfds.features.Tensor(shape=(None, None, 1), dtype=np.float32)`, matching OXE/RLDS convention (plain `Tensor`, never `Image`).
- `test_hdf5_to_rlds_against_real_dataset` now pre-flights the real HDF5 for `agentview_depth` presence and explicitly skips (with a documented reason) rather than crashing on a legacy pre-Phase-7 file.
- `oxe_register.py`'s `register_soarm_spatial()` and its on-disk patch template both set `depth_obs_keys["primary"] = "agentview_depth"`; the shared idempotency marker bumped from `v2` to `v3` so a previously-patched installed `configs.py` still receives the fix.
- `test_rlds_converter.py` gained 2 new negative depth tests (wrong-dtype, out-of-range) and a TFDS schema-level depth assertion; `test_oxe_register.py`'s dict-shape assertion updated to expect the new `depth_obs_keys` value.

## Task Commits

Each task was committed atomically:

1. **Task 1: Extend rlds_converter.py's schema, loader, and step-assembly with agentview_depth** - `c8b7750` (feat)
2. **Task 2: Register agentview_depth in oxe_register.py's depth_obs_keys (in-memory + on-disk)** - `8bf984c` (feat)

## Files Created/Modified

- `LIBERO/libero/libero/datasets/rlds_converter.py` - `validate_episode_arrays` gains `agentview_depth` param + validation branch; `load_episodes_from_hdf5` reads it; `_episode_to_rlds_steps` includes it per step; `hdf5_to_rlds`'s `FeaturesDict` declares it as a `Tensor`
- `LIBERO/libero/libero/datasets/test_rlds_converter.py` - `_synthetic_episode`/`_write_synthetic_hdf5` generate depth; all 5 existing validation tests updated for the new param; 2 new negative depth tests; TFDS schema-level depth assertion; legacy-dataset skip guard for the real-dataset test
- `LIBERO/libero/libero/datasets/oxe_register.py` - `register_soarm_spatial`'s dict and the on-disk patch template both set `depth_obs_keys["primary"]`; marker bumped `v2` -> `v3`; docstring corrected to describe the new depth semantics
- `LIBERO/libero/libero/datasets/test_oxe_register.py` - `depth_obs_keys` equality assertion updated to the new expected value

## Decisions Made

- `load_episodes_from_hdf5()` requires `agentview_depth` unconditionally (direct dict access, no `.get()` fallback) — a legacy pre-Phase-7 HDF5 missing the key surfaces as a plain `KeyError` at load time, matching this function's existing direct-indexing style for the other obs keys. The one currently-known legacy file is handled by an explicit, documented `pytest.skip` in the test suite instead.
- The versioned `marker` string in `oxe_register.py` was bumped from `v2` to `v3` (not left unchanged) so an already-patched installed `configs.py` (from a prior, pre-Phase-7 session) still picks up the depth fix on the next `apply_soarm_spatial_registration()` call, per this file's own documented per-version idempotency convention.
- `load_depth=True` was NOT set anywhere in `oxe_register.py` — that flag is Phase 9's TUNE-05 concern (RESEARCH.md Pitfall 4), out of this plan's scope.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- The real pre-Phase-7 HDF5 file (`put_the_cream_cheese_in_the_bowl_demo.hdf5`) that the plan's `<verified_this_session>` notes describe as present on disk is not present in this worktree checkout (dataset files are gitignored and not mirrored into the isolated git worktree used for this parallel execution). `test_hdf5_to_rlds_against_real_dataset` therefore skips via the pre-existing "No real dataset found on disk" path rather than exercising the new depth-presence pre-flight/skip branch added by this plan. The new code path itself (`h5py.File` pre-flight check for `"agentview_depth" in f["data"]["demo_1"]["obs"]`, skip with the depth-specific message) is implemented per spec and structurally correct, but was not executed end-to-end against the real legacy file in this run — flagged in `coverage: D4` above for a follow-up check against the main checkout (or Colab) before Phase 7's final sign-off.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `agentview_depth` now flows all the way from the HDF5 dataset (07-02) through the RLDS/TFDS schema and into `oxe_register.py`'s OXE config — DEPTH-03 is code-complete pending a Colab-side TFDS smoke test (tensorflow_datasets is not installed locally, matching Phase 6's established baseline) and a verification of the legacy-dataset skip against the real HDF5 file noted in "Issues Encountered."
- Phase 9's demo re-collection (DATA-05) and re-fine-tuning (TUNE-05) can flip `load_depth=True` and expect `agentview_depth` to already be present in both the TFDS schema and the OXE registry — zero further plumbing changes needed here.
- Independent of 07-01 (camera recalibration) — no file overlap.

---
*Phase: 07-camera-depth-perception*
*Completed: 2026-09-12*
