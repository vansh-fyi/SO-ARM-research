---
phase: 07-camera-depth-perception
plan: 02
subsystem: dataset
tags: [robosuite, mujoco, hdf5, h5py, depth-perception, libero]

# Dependency graph
requires:
  - phase: 07-camera-depth-perception (plan 01)
    provides: recalibrated agentview/eye_in_hand cameras (zero file overlap, independent)
provides:
  - "agentview_depth float32 dataset persisted per-episode in the robomimic HDF5 writer output"
  - "camera_depths=True threaded through OffScreenRenderEnv construction in gather_demonstrations_as_hdf5()"
  - "Fixed pre-existing stale gripper_states shape assertion (D-07 84mm 2-DOF gripper)"
affects: [07-camera-depth-perception (plan 03, RLDS conversion), phase-9-data-collection]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Key-name-tuple dtype dispatch: _DEPTH_KEYS tuple mirrors _RGB_KEYS/_STATE_KEYS, driving a dedicated write-loop branch"

key-files:
  created: []
  modified:
    - LIBERO/libero/libero/datasets/hdf5_writer.py
    - LIBERO/libero/libero/datasets/test_hdf5_writer.py

key-decisions:
  - "Depth persisted as raw normalized [0,1] float32, no unit conversion to meters, matching depth_xyz.py's documented consumer contract"
  - "Depth persisted for agentview only (D-05); camera_depths=True renders depth for both cameras globally (robosuite has no per-camera toggle) but only agentview_depth is added to OBS_KEY_MAPPING/_DEPTH_KEYS, so eye_in_hand's depth render is simply never written"

patterns-established:
  - "Third dtype-dispatch tuple (_DEPTH_KEYS) alongside _RGB_KEYS/_STATE_KEYS for future obs types"

requirements-completed: [DEPTH-01, DEPTH-02]

coverage:
  - id: D1
    description: "gather_demonstrations_as_hdf5() writes demo_N/obs/agentview_depth as float32, shape (H,W,1) per step, values in [0,1]"
    requirement: "DEPTH-01"
    verification:
      - kind: integration
        ref: "LIBERO/libero/libero/datasets/test_hdf5_writer.py::test_schema_and_obs_key_naming"
        status: pass
    human_judgment: false
  - id: D2
    description: "No eye_in_hand/wrist depth dataset is ever written (agentview-only, D-05)"
    requirement: "DEPTH-01"
    verification:
      - kind: integration
        ref: "LIBERO/libero/libero/datasets/test_hdf5_writer.py::test_schema_and_obs_key_naming"
        status: pass
    human_judgment: false
  - id: D3
    description: "agentview_depth schema is source-agnostic across differently-actioned recording sources (DATA-04 parity extended to the new key)"
    requirement: "DEPTH-02"
    verification:
      - kind: integration
        ref: "LIBERO/libero/libero/datasets/test_hdf5_writer.py::test_schema_matches_across_sources"
        status: pass
    human_judgment: false
  - id: D4
    description: "Pre-existing stale gripper_states.shape[1]==1 assertion fixed to ==2 (D-07 84mm 2-DOF gripper), unblocking the test function"
    verification:
      - kind: integration
        ref: "LIBERO/libero/libero/datasets/test_hdf5_writer.py::test_schema_and_obs_key_naming"
        status: pass
    human_judgment: false

duration: 12min
completed: 2026-09-05
status: complete
---

# Phase 7 Plan 02: Depth Persistence in HDF5 Writer Summary

**`gather_demonstrations_as_hdf5()` now persists a per-episode `agentview_depth` float32 dataset (raw normalized `[0,1]`, no unit conversion) alongside the existing RGB/proprio datasets, threading robosuite's `camera_depths=True` through the writer's env construction.**

## Performance

- **Duration:** 12 min
- **Started:** 2026-09-05T16:53:00Z
- **Completed:** 2026-09-05T17:05:09Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- `OBS_KEY_MAPPING` extended with `"agentview_depth": "agentview_depth"` and a new `_DEPTH_KEYS = ("agentview_depth",)` tuple, following the exact `_RGB_KEYS`/`_STATE_KEYS` dtype-dispatch convention.
- `OffScreenRenderEnv(...)` construction inside `gather_demonstrations_as_hdf5()` now passes `camera_depths=True`, rendering depth for the reused regeneration env.
- A new write-loop branch persists `agentview_depth` as a `float32` HDF5 dataset per episode, with no unit conversion — matching `depth_xyz.py`'s documented `[0,1]` normalized-depth consumer contract.
- `test_hdf5_writer.py` extended with dtype/shape/range assertions for `agentview_depth` and an explicit negative assertion that no `eye_in_hand_depth`/`robot0_eye_in_hand_depth` key is ever written (D-05, agentview-only).
- Fixed the pre-existing, previously-blocking `gripper_states.shape[1] == 1` stale assertion (now `== 2`, matching SOARM's real D-07 84mm 2-DOF parallel gripper) — this bug sat earlier in the same test function than the new depth assertion and would have prevented it from ever running under `pytest -x`.
- `test_schema_matches_across_sources`'s `expected_keys` set extended to include `agentview_depth`, proving the writer's five-key schema (not four) is identical regardless of how an episode was actioned.

## Task Commits

Each task was committed atomically:

1. **Task 1: Thread camera_depths=True and persist agentview_depth (DEPTH-01, DEPTH-02)** - `8f9ab3d` (feat)
2. **Task 2: Extend test_hdf5_writer.py with depth assertions and fix the stale gripper_states shape bug** - `125edac` (test)

_Note: Task 1 used the TDD-integration flow (implementation first, verified RED against the not-yet-updated test in Task 2's file, then Task 2 landed the test extension + fix to reach full GREEN) — the plan explicitly sequenced it this way since Task 2's stale-assertion fix is a documented prerequisite for the depth assertion ever executing._

## Files Created/Modified

- `LIBERO/libero/libero/datasets/hdf5_writer.py` - Added `agentview_depth` to `OBS_KEY_MAPPING`, new `_DEPTH_KEYS` tuple, `camera_depths=True` kwarg, depth write-loop branch; corrected a stale "1-DOF gripper" comment on the `_STATE_KEYS` write loop
- `LIBERO/libero/libero/datasets/test_hdf5_writer.py` - Added `agentview_depth` dtype/shape/range assertions and a negative eye_in_hand-depth assertion to `test_schema_and_obs_key_naming`; fixed the stale `gripper_states.shape[1] == 1` → `== 2` assertion; added `agentview_depth` to `test_schema_matches_across_sources`'s `expected_keys`

## Decisions Made

- Depth is written via a dedicated `_DEPTH_KEYS` loop (not folded into `_RGB_KEYS` or `_STATE_KEYS`) since depth arrays are already `(H, W, 1)`-shaped per step (no `atleast_1d` coercion needed, unlike the scalar/1-D proprio values `_STATE_KEYS` handles) but are a distinct dtype (`float32`) from the RGB `uint8` arrays.
- `camera_depths=True` is applied globally at env construction (robosuite has no per-camera depth toggle in 1.4.0); agentview-only persistence (D-05) is enforced purely by which keys are added to `OBS_KEY_MAPPING`/`_DEPTH_KEYS`, never by suppressing rendering.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Corrected a second stale "1-DOF gripper" comment in hdf5_writer.py itself**
- **Found during:** Task 1
- **Issue:** The plan only called out fixing the stale `gripper_states.shape[1] == 1` assertion and its adjacent comment in `test_hdf5_writer.py` (Task 2). The same factually-incorrect comment ("SOARM's 1-DOF gripper returns robot0_gripper_qpos as a 0-d scalar") also exists in `hdf5_writer.py`'s `_STATE_KEYS` write loop, which this plan already modifies (Task 1's `files_modified` scope).
- **Fix:** Updated the comment to correctly describe SOARM's actual D-07 84mm 2-DOF gripper (shape `(2,)`), while preserving the `atleast_1d` rationale (guards any future gripper variant returning a scalar).
- **Files modified:** `LIBERO/libero/libero/datasets/hdf5_writer.py`
- **Verification:** Comment-only change; full test suite still green (`pytest LIBERO/libero/libero/datasets/test_hdf5_writer.py -x` — 3 passed)
- **Committed in:** `8f9ab3d` (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 bug — stale/misleading comment in in-scope file)
**Impact on plan:** Documentation-only correction inside a file already being modified by this plan. No scope creep, no behavior change.

## Issues Encountered

None. Both tasks executed as sequenced; Task 1's implementation was verified against the real local-sim integration test, confirming the expected RED state (obs_keys set correctly included `agentview_depth` but the test's literal set didn't yet expect it) before Task 2 landed the test extension for full GREEN.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `07-03` (RLDS conversion) can now read `demo_N/obs/agentview_depth` directly off disk with the exact float32/`(H,W,1)`/`[0,1]` convention this plan establishes — no writer-side gaps remain for depth.
- Phase 9's demo re-collection (DATA-05) will automatically produce depth-augmented HDF5 files once run against this updated writer, with zero further changes needed here.
- Independent of `07-01` (camera recalibration) — no file overlap, can land in either order.

---
*Phase: 07-camera-depth-perception*
*Completed: 2026-09-05*
