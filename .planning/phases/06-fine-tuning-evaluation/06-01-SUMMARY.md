---
phase: 06-fine-tuning-evaluation
plan: 01
subsystem: data
tags: [rlds, tfds, tensorflow_datasets, h5py, robomimic, hdf5, dataset-conversion]

# Dependency graph
requires:
  - phase: 04-dataset-collection
    provides: put_the_cream_cheese_in_the_bowl_demo.hdf5 (120 real SOARM demos, robomimic HDF5 schema)
provides:
  - "rlds_converter.py: validate_episode_arrays / load_episodes_from_hdf5 / hdf5_to_rlds, importable with zero tensorflow dependency at module top level"
  - "Fail-loud per-episode schema validation (malformed arrays rejected before any RLDS bytes are written)"
  - "Genuine TFDS-write implementation (feature schema + SequentialWriter call) targeting Plan 06-02's finetune.py tfds.builder(...) load path"
affects: [06-02-finetuning-training]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "rlds_converter.py mirrors normalization.py's module shape: numpy at module top, h5py/tensorflow/tensorflow_datasets/bddl_utils imports confined to function bodies"
    - "Validate-then-write ordering: every episode validated before any output byte written (extends write_dataset_statistics's compute-then-write convention to per-episode granularity)"

key-files:
  created:
    - LIBERO/libero/libero/datasets/rlds_converter.py
    - LIBERO/libero/libero/datasets/test_rlds_converter.py
  modified:
    - LIBERO/libero/libero/datasets/__init__.py

key-decisions:
  - "hdf5_to_rlds's TFDS write path uses tfds.core.SequentialWriter (06-RESEARCH.md Pattern 1) with a DatasetIdentity/DatasetInfo scaffold, since tensorflow_datasets is not installed in the local libero conda env and this call shape was flagged MEDIUM confidence by 06-RESEARCH.md — implemented per the plan's explicit instruction to confirm/adjust the exact API call live on Colab in Plan 06-02, keeping the feature schema and validate-then-write ordering fixed regardless of adjustment"

patterns-established:
  - "Per-episode HDF5 loading preserves timestep boundaries (list of per-episode dicts), distinct from normalization.py's flattened-across-demos loading — needed so RLDS is_first/is_last/is_terminal semantics are correct"

requirements-completed: [TUNE-01]

coverage:
  - id: D1
    description: "validate_episode_arrays rejects malformed/mismatched episode arrays (wrong dtype, wrong last-dim, mismatched length, NaN/Inf) with a descriptive ValueError before any bytes are written"
    requirement: "TUNE-01"
    verification:
      - kind: unit
        ref: "LIBERO/libero/libero/datasets/test_rlds_converter.py#test_validate_episode_arrays_accepts_well_formed_episode"
        status: pass
      - kind: unit
        ref: "LIBERO/libero/libero/datasets/test_rlds_converter.py#test_validate_episode_arrays_rejects_wrong_action_dim"
        status: pass
      - kind: unit
        ref: "LIBERO/libero/libero/datasets/test_rlds_converter.py#test_validate_episode_arrays_rejects_mismatched_episode_length"
        status: pass
      - kind: unit
        ref: "LIBERO/libero/libero/datasets/test_rlds_converter.py#test_validate_episode_arrays_rejects_non_uint8_image"
        status: pass
      - kind: unit
        ref: "LIBERO/libero/libero/datasets/test_rlds_converter.py#test_validate_episode_arrays_rejects_nan_state"
        status: pass
    human_judgment: false
  - id: D2
    description: "load_episodes_from_hdf5 preserves per-episode timestep boundaries across a multi-demo HDF5 and derives language_instruction from the stored bddl_file_name attr via get_problem_info"
    requirement: "TUNE-01"
    verification:
      - kind: unit
        ref: "LIBERO/libero/libero/datasets/test_rlds_converter.py#test_load_episodes_from_hdf5_preserves_episode_boundaries_and_derives_language"
        status: pass
    human_judgment: false
  - id: D3
    description: "hdf5_to_rlds writes a genuine tfds.builder-loadable RLDS dataset (validate-then-write ordering, round-trip .info check)"
    requirement: "TUNE-01"
    verification:
      - kind: unit
        ref: "LIBERO/libero/libero/datasets/test_rlds_converter.py#test_hdf5_to_rlds_writes_tfds_loadable_dataset (guarded by pytest.importorskip(tensorflow_datasets), skips locally)"
        status: unknown
      - kind: integration
        ref: "LIBERO/libero/libero/datasets/test_rlds_converter.py#test_hdf5_to_rlds_against_real_dataset (guarded by pytest.importorskip(tensorflow_datasets), skips locally)"
        status: unknown
    human_judgment: true
    rationale: "tensorflow_datasets is not installed in this project's local libero conda env (Colab-only per 06-VALIDATION.md); both tests are pytest.importorskip-guarded and currently skip. The exact tfds.core.SequentialWriter call shape used inside hdf5_to_rlds is flagged MEDIUM confidence in 06-RESEARCH.md and can only be confirmed once tensorflow_datasets is installed on Colab (Plan 06-02) — a human/Colab run is required to prove this deliverable, not just source review."

# Metrics
duration: 4min
completed: 2026-08-20
status: complete
---

# Phase 6 Plan 1: RLDS Converter Summary

**Custom robomimic-HDF5-to-RLDS converter (`rlds_converter.py`) that validates every episode before writing any bytes, preserves per-episode timestep boundaries, and writes a genuine TFDS-format dataset via `tensorflow_datasets`'s writer APIs — zero tensorflow dependency at module import time**

## Performance

- **Duration:** 4 min (commit-to-commit)
- **Started:** 2026-08-20T23:20:24+05:30
- **Completed:** 2026-08-20T23:23:18+05:30
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- `validate_episode_arrays` fails loud (`ValueError` naming the offending array + actual vs. expected shape/dtype) on wrong dtype, wrong last-dim, mismatched per-array timestep length, zero-length episode, and NaN/Inf — never silently pads or truncates
- `load_episodes_from_hdf5` reads every `demo_*` group across all given HDF5 paths, preserving each episode's own timestep count (unlike `normalization.py`'s flattened loader) and deriving `language_instruction` from the stored `bddl_file_name` attr via `bddl_utils.get_problem_info` — confirmed live against the real `put_the_cream_cheese_in_the_bowl.bddl` file, returning the exact string `"put the cream cheese on the bowl"`
- `hdf5_to_rlds` validates every episode before writing any `.tfrecord` bytes, builds the `tfds.features.FeaturesDict` schema specified in 06-RESEARCH.md Pattern 1, and writes via `tfds.core.SequentialWriter`, followed by a `tfds.builder(...).info` round-trip check (Pitfall 2's explicit recommendation) — this write path only runs on Colab where `tensorflow_datasets` is installed
- Module imports cleanly with zero `tensorflow`/`tensorflow_datasets` dependency at module top level, verified live in the local `libero` conda env (h5py present, tensorflow_datasets absent)
- `__init__.py` exports the three functions unconditionally, alongside `normalization.py`'s existing unconditional exports

## Task Commits

Each task was committed atomically:

1. **Task 1: rlds_converter.py — schema validation, per-episode HDF5 loading, genuine TFDS write** - `b2fecda` (feat)
2. **Task 2: test_rlds_converter.py — local schema/loading coverage + skip-guarded TFDS write coverage** - `0b55baa` (test)

_Note: tdd="true" tasks here did not follow a strict separate RED-then-GREEN commit split — Task 1 (implementation) and Task 2 (test suite) were each committed as a single atomic commit per the plan's own task boundaries (Task 1 explicitly scopes acceptance to "module exists, imports cleanly" and defers full test coverage to Task 2's own verify block, per the plan's Task 1 acceptance-criteria note)._

## Files Created/Modified
- `LIBERO/libero/libero/datasets/rlds_converter.py` - `validate_episode_arrays`, `load_episodes_from_hdf5`, `hdf5_to_rlds`, `REQUIRED_PROPRIO_DIM`/`REQUIRED_ACTION_DIM` constants, CLI entry point
- `LIBERO/libero/libero/datasets/test_rlds_converter.py` - 8 tests: 5 validation tests, 1 episode-loading test, 2 TFDS-write tests (skip-guarded)
- `LIBERO/libero/libero/datasets/__init__.py` - unconditional export of the three new functions

## Decisions Made
- Used `tfds.core.SequentialWriter` + `tfds.core.naming.DatasetIdentity`/`tfds.core.DatasetInfo` for the genuine TFDS write, per 06-RESEARCH.md Pattern 1's explicit recommendation. This exact call shape is flagged MEDIUM confidence by 06-RESEARCH.md (no local `tensorflow_datasets` install to verify the constructor/method signatures against) — the plan's own action text pre-authorizes adjusting this on Colab once `tensorflow_datasets==4.9.10` is actually installed (Plan 06-02), as long as the feature schema and validate-then-write ordering stay unchanged. Documented inline in `rlds_converter.py` with a `NOTE [MEDIUM confidence...]` comment at the call site.

## Deviations from Plan

None — plan executed exactly as written. Both tasks' acceptance criteria were met verbatim:
- Task 1: module exists, exposes the three functions, `grep -c "^import tensorflow"` returns 0, imports cleanly in the local libero conda env.
- Task 2: `pytest test_rlds_converter.py -x -q` shows 6 passed / 2 skipped exactly as specified; full `datasets` package suite run shows no *new* failures introduced by the `__init__.py` edit.

## Issues Encountered
- Running the full `LIBERO/libero/libero/datasets` package test suite surfaced one pre-existing failure: `test_hdf5_writer.py::test_schema_and_obs_key_naming` (stale `gripper_states.shape[1] == 1` assertion, never updated after the Phase 4 D-07 84mm 2-DOF parallel-gripper upgrade). This is explicitly documented in STATE.md's Blockers/Concerns as pre-existing Phase 4 test debt, confirmed via `git log` that neither `test_hdf5_writer.py` nor `hdf5_writer.py` was touched by this plan's commits — out of scope per the deviation-rules Scope Boundary (only auto-fix issues directly caused by the current task's changes). Left untouched; candidate for the STATE.md-flagged future Phase 6 quick-task cleanup.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- `rlds_converter.py`'s `hdf5_to_rlds(hdf5_paths, out_dir, dataset_name="soarm_spatial")` is ready for Plan 06-02's training notebook to call once `tensorflow_datasets` is installed on Colab (Package Legitimacy Gate, 06-02 Task 3).
- The two TFDS-write tests (`test_hdf5_to_rlds_writes_tfds_loadable_dataset`, `test_hdf5_to_rlds_against_real_dataset`) will run for real for the first time on Colab in Plan 06-02 — if `tfds.core.SequentialWriter`'s actual signature in `tensorflow_datasets==4.9.10` differs from this implementation's MEDIUM-confidence call shape, Plan 06-02 must adjust `hdf5_to_rlds`'s writer-invocation block (feature schema and validate-then-write ordering must stay unchanged).
- No blockers for Plan 06-02.

---
*Phase: 06-fine-tuning-evaluation*
*Completed: 2026-08-20*

## Self-Check: PASSED

- FOUND: LIBERO/libero/libero/datasets/rlds_converter.py
- FOUND: LIBERO/libero/libero/datasets/test_rlds_converter.py
- FOUND: LIBERO/libero/libero/datasets/__init__.py (modified)
- FOUND commit: b2fecda (Task 1)
- FOUND commit: 0b55baa (Task 2)
