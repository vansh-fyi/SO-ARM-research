---
phase: 04-dataset-collection
plan: 04
subsystem: dataset
tags: [normalization, openvla-schema, numpy, hdf5, soarm, data-03]

# Dependency graph
requires:
  - phase: 04-dataset-collection
    plan: 02
    provides: "The real 04-02 dataset (120 demos, put_the_cream_cheese_in_the_bowl_demo.hdf5, DATA-01 complete)"
  - phase: 04-dataset-collection
    plan: 03
    provides: "Confirmed DATA-02 determinism (120/120 demos, 17457/17457 states exact) — same dataset this plan computes stats from"
provides:
  - "normalization.compute_norm_stats(actions, proprio, dataset_key, num_trajectories) -> dict — pure-numpy OpenVLA q01/q99/mean/std/min/max schema"
  - "normalization.load_actions_and_proprio_from_hdf5(hdf5_paths) -> (actions, proprio, total_demo_count) — reads this repo's real HDF5 obs schema directly, no robomimic dependency"
  - "normalization.write_dataset_statistics(hdf5_paths, out_json_path, dataset_key) -> dict — computes + writes the on-disk artifact"
  - "Real DATA-03 artifact: LIBERO/libero/datasets/soarm_spatial/dataset_statistics.json (17457 transitions, 120 trajectories, action/proprio q01<=q99, computed from the real collected SOARM dataset, not copied from Panda)"
affects: [phase-06-finetuning]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Dependency-light module convention: normalization.py imports only numpy/json at module top (h5py confined to the loader function), so it can be imported UNCONDITIONALLY in __init__.py even in a no-sim-stack context — matches vla/__init__.py's graceful-degradation rationale for the modules that DO need it"
    - "Schema parity with an existing consumer: compute_norm_stats's q01/q99/mean/std/min/max schema matches oft_backend.py's already-consumed dataset_statistics.json overlay exactly, so Phase 6 can reuse that same json.load() + dict-key pattern instead of writing a second stats-loading code path"

key-files:
  created:
    - LIBERO/libero/libero/datasets/normalization.py
    - LIBERO/libero/libero/datasets/test_normalization.py
  modified:
    - LIBERO/libero/libero/datasets/__init__.py

key-decisions:
  - "compute_norm_stats(actions, proprio, dataset_key, num_trajectories) is a PURE numpy function (no h5py/sim import at module top) — num_trajectories is caller-supplied, not inferred from the transition array's first dim, since T is the concatenated total across all demos"
  - "[Rule 1 - Correctness] Corrected the plan's stale proprio-dim assumption: 04-04-PLAN.md's Task 1 synthetic test and Task 2 acceptance criteria specify proprio as 6-D (5 SOARM joints + 1 gripper DOF). The REAL dataset's gripper_states is (T, 2) — the roboninecom 84mm parallel gripper (D-07 upgrade) has two independently sliding jaws, a genuine 2-DOF robot0_gripper_qpos, not the old single-DOF value. This is the exact same class of stale pre-gripper-upgrade assumption 04-03-SUMMARY.md already flagged and fixed for gripper_states.shape[1]. Both the synthetic unit test and the real-dataset integration test were written/asserted against proprio dim 7 (5+2), matching hdf5_writer.py's actual obs schema — no silent test-dimension mismatch"
  - "load_actions_and_proprio_from_hdf5 concatenates joint_states + gripper_states along the last axis per demo, across however many *_demo.hdf5 files a glob finds (no hardcoded file count) — the phase's real output is a single file post D-07/D-08 retargeting, but the loader itself doesn't assume that"

requirements-completed: [DATA-03]
requirements-blocked: []

coverage:
  - id: D1
    description: "compute_norm_stats returns the OpenVLA q01/q99/mean/std/min/max schema, non-degenerate on real variance, matching what oft_backend.py already consumes"
    requirement: "DATA-03"
    verification:
      - kind: unit
        ref: "LIBERO/libero/libero/datasets/test_normalization.py#test_compute_norm_stats_schema_and_shapes"
        status: pass
      - kind: unit
        ref: "LIBERO/libero/libero/datasets/test_normalization.py#test_compute_norm_stats_q01_strictly_less_than_q99"
        status: pass
      - kind: unit
        ref: "LIBERO/libero/libero/datasets/test_normalization.py#test_compute_norm_stats_no_nan"
        status: pass
      - kind: unit
        ref: "LIBERO/libero/libero/datasets/test_normalization.py#test_compute_norm_stats_constant_dimension_degenerate_not_nan"
        status: pass
    human_judgment: false
  - id: D2
    description: "dataset_statistics.json exists on disk, computed from the real 04-02 collected SOARM dataset (120 demos, 17457 transitions), with non-degenerate q01<=q99 bounds for both action and proprio — not copied from Panda"
    requirement: "DATA-03"
    verification:
      - kind: integration
        ref: "LIBERO/libero/libero/datasets/test_normalization.py#test_load_and_write_against_04_02_output"
        status: pass
      - kind: other
        ref: "conda run -n libero python3 -c 'write_dataset_statistics(...)' real CLI run against put_the_cream_cheese_in_the_bowl_demo.hdf5"
        status: pass
    human_judgment: false

# Metrics
duration: ~20min
completed: 2026-08-03
status: complete
---

# Phase 4 Plan 04: SOARM Normalization Statistics Summary

**`normalization.py`'s pure-numpy `compute_norm_stats` (OpenVLA q01/q99/mean/std/min/max schema, matching `oft_backend.py`'s existing consumption pattern) was run against the real 04-02 dataset, producing `dataset_statistics.json` — 17457 transitions across 120 trajectories, action dim 7, proprio dim 7 (5 joints + 2-DOF gripper), non-degenerate q01<=q99 throughout — satisfying DATA-03 from the actual collected SOARM data, not copied from Panda.**

## Performance

- **Duration:** ~20 min
- **Tasks:** 2 of 2 complete
- **Files modified:** 3 (2 created, 1 modified)

## Accomplishments

- `compute_norm_stats(actions, proprio, dataset_key, num_trajectories)` — pure numpy, no MuJoCo/robosuite/torch/h5py at module top, returns the exact q01/q99/mean/std/min/max schema `oft_backend.py` already `json.load()`s and dict-keys into `model.norm_stats`.
- `load_actions_and_proprio_from_hdf5(hdf5_paths)` — reads this repo's real `actions`/`obs/joint_states`/`obs/gripper_states` HDF5 schema directly (no robomimic import), concatenating across every demo in every path found.
- `write_dataset_statistics(hdf5_paths, out_json_path, dataset_key)` — computes stats then writes JSON only after a successful full-array computation (T-04-04-01: a load failure raises before any partial file is written).
- Argparse CLI (`--hdf5-glob`, `--out`) for standalone re-runs.
- Real run against `LIBERO/libero/datasets/soarm_spatial/put_the_cream_cheese_in_the_bowl_demo.hdf5` (120 demos): wrote `dataset_statistics.json` — 17457 transitions, 120 trajectories, action q01/q99 both length 7, proprio q01/q99 both length 7, all dims q01<=q99 (three action dims — the FSM's zero rotation-delta convention — are degenerate-constant at exactly 0.0, expected given the collector's fixed top-down orientation, not a bug).
- 5/5 tests pass (4 synthetic unit tests for the 4 specified `compute_norm_stats` behaviors + 1 real-dataset integration test, not skipped since the dataset is present on disk).

## Task Commits

1. **Task 1 (RED): failing test for compute_norm_stats** — `3afe6f8` (test)
2. **Task 1 (GREEN): compute_norm_stats pure numpy stats function** — `b28ae5f` (feat)
3. **Task 2: __init__.py wiring + real DATA-03 run** — `98b8f50` (feat)

_(Task 1 followed the TDD RED→GREEN gate: confirmed `ModuleNotFoundError` with `normalization.py` absent before writing it, then 4/4 pass with it present.)_

## Files Created/Modified

- `LIBERO/libero/libero/datasets/normalization.py` — `compute_norm_stats`, `load_actions_and_proprio_from_hdf5`, `write_dataset_statistics`, CLI entry point (created)
- `LIBERO/libero/libero/datasets/test_normalization.py` — 4 synthetic unit tests + 1 real-dataset-or-skip integration test (created)
- `LIBERO/libero/libero/datasets/__init__.py` — unconditional (no try/except) export of the 3 normalization functions (modified)
- `LIBERO/libero/datasets/soarm_spatial/dataset_statistics.json` — the real DATA-03 artifact (gitignored per `LIBERO/.gitignore`'s `/libero/datasets` rule, not committed)

## Decisions Made

See `key-decisions` in frontmatter — most notably the Rule 1 correction of the plan's stale 6-D proprio assumption (5 joints + 1 gripper DOF) to the real 7-D shape (5 joints + 2-DOF gripper), matching the D-07 84mm parallel-gripper upgrade's actual `gripper_states` shape.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Correctness] Corrected stale 6-D proprio assumption to the real 7-D shape**
- **Found during:** Task 1, before writing the synthetic test (cross-checked against `hdf5_writer.py`'s real obs schema and 04-03-SUMMARY.md's prior identical finding).
- **Issue:** `04-04-PLAN.md`'s Task 1 behavior spec and Task 2 acceptance criteria both specify proprio as `(T, 6)` — "5 SOARM arm joints + 1 gripper DOF" — a pre-gripper-upgrade assumption. The real dataset's `gripper_states` is `(T, 2)`: the roboninecom 84mm parallel gripper (D-07) has two independently sliding jaws, not the old single combined DOF.
- **Fix:** Wrote the synthetic unit test with `proprio_dim=7` (documented inline as 5 joints + 2-DOF gripper) and asserted the real-dataset integration test's `proprio["q01"]` has length 7, not 6. Verified directly against the actual HDF5 file (`gripper_states.shape == (151, 2)` for a sample demo) before writing any test code.
- **Files modified:** `LIBERO/libero/libero/datasets/test_normalization.py` (written correctly from the start with this context — no stale assertion was ever committed).
- **Commit:** `3afe6f8` (test file's first commit already reflected the corrected 7-D shape).

## Threat Model Compliance

- **T-04-04-01 (Tampering, mitigate):** `write_dataset_statistics` calls `compute_norm_stats` (which requires the full concatenated array) BEFORE opening `out_json_path` for writing — a `load_actions_and_proprio_from_hdf5` failure (missing/corrupt HDF5) raises before any file write. Confirmed by code inspection: no partial-write path exists.
- **T-04-04-02 (Information Disclosure, accept):** `dataset_statistics.json` contains only aggregate per-dimension numeric statistics (mean/std/min/max/q01/q99, transition/trajectory counts) — no raw images, states, or trajectories. Local, gitignored output. Confirmed via `git check-ignore -v`.
- **T-04-04-SC (package legitimacy, accept):** No new package installs this plan. Confirmed — only stdlib (`json`, `os`, `argparse`, `glob`) and already-installed `numpy`/`h5py`.

## Known Stubs

None — all three functions are fully implemented and exercised against both synthetic data and the real collected dataset.

## Self-Check: PASSED

- Files: `normalization.py`, `test_normalization.py` FOUND on disk (verified via `Read`/`Write` during implementation, tracked under the lowercase `libero/...` path per this repo's established macOS case-collision convention — same as `collector.py`/`replay.py`/`hdf5_writer.py`); `__init__.py` diff FOUND via `git status`.
- Commits: `3afe6f8`, `b28ae5f`, `98b8f50` all FOUND in `git log --oneline -6`.
- Artifact: `LIBERO/libero/datasets/soarm_spatial/dataset_statistics.json` FOUND on disk (`test -f` succeeds), re-verified via direct `json.load()`: 17457 transitions, 120 trajectories, action/proprio q01 both length 7.
- Tests: `conda run -n libero pytest LIBERO/libero/libero/datasets/test_normalization.py -v` — 5 passed, 0 skipped (real dataset present), 0 failed.

## Next Phase Readiness

- DATA-03 is now COMPLETE. `dataset_statistics.json` is a real, non-degenerate, SOARM-specific normalization-statistics artifact computed from the actual collected dataset, schema-matched to what `oft_backend.py` already consumes — Phase 6 fine-tuning can overlay it via the identical `json.load()` + dict-key pattern, no new loading code needed.
- Plan 04-05 (teleoperation interface, DATA-04) is the last plan in this phase; per STATE.md's resume notes, its Task 3 was written against the old `table_center`/bowl/plate task and needs its BDDL path, output filename, and instructions updated to point at the retargeted `put_the_cream_cheese_in_the_bowl` task before it can run.

---
*Phase: 04-dataset-collection*
*Completed: 2026-08-03*

## Self-Check: PASSED (post-write verification)

- `normalization.py`, `test_normalization.py` FOUND on disk.
- `LIBERO/libero/datasets/soarm_spatial/dataset_statistics.json` FOUND on disk.
- Commits `3afe6f8`, `b28ae5f`, `98b8f50` FOUND in `git log --oneline --all`.
