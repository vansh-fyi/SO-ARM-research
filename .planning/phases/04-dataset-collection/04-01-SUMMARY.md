---
phase: 04-dataset-collection
plan: 01
subsystem: dataset
tags: [robosuite, h5py, robomimic, hdf5, mujoco, data-collection-wrapper, soarm]

# Dependency graph
requires:
  - phase: 02-soarm-robot-integration
    provides: Soarm101 robot class, tuned OSC_POSE kinematics, frozen 3-task list, set_init_state() obs-regeneration primitive
  - phase: 03-vla-inference-loop
    provides: libero.libero.vla graceful-degradation package pattern, LIBERO-anchored sys.path convention
provides:
  - "LIBERO/libero/libero/datasets/ importable package (scaffold + raw_recorder + hdf5_writer)"
  - "build_recording_env(): DataCollectionWrapper-wrapped SOARM env builder shared by scripted collector + teleop (later plans)"
  - "gather_demonstrations_as_hdf5(): raw npz states/actions -> robomimic-schema HDF5 with real regenerated image obs under LIBERO renamed keys"
  - "OBS_KEY_MAPPING constant (robosuite -> LIBERO obs key rename)"
  - "gitignore fix unblocking the new source package"
affects: [04-02-scripted-collector, 04-03-teleop, 04-04-replay-verification, 04-05-normalization, phase-06-finetuning]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Two-stage collect-then-extract: record states/actions live, regenerate obs offline via set_init_state()"
    - "LIBERO-anchored absolute env imports (from LIBERO.libero.libero.envs) to hit the registered TASK_MAPPING copy"
    - "OR-accumulated per-episode success gating across DataCollectionWrapper flush chunks"

key-files:
  created:
    - LIBERO/libero/libero/datasets/__init__.py
    - LIBERO/libero/libero/datasets/raw_recorder.py
    - LIBERO/libero/libero/datasets/hdf5_writer.py
    - LIBERO/libero/libero/datasets/test_hdf5_writer.py
  modified:
    - LIBERO/.gitignore

key-decisions:
  - "Import LIBERO envs via the LIBERO-anchored absolute path, not relative ..envs — problems register TASK_MAPPING into the LIBERO.* module copy, so a relative import binds an empty dict and KeyErrors at env construction"
  - "Coerce per-step proprio to np.atleast_1d before stacking so SOARM's 0-d gripper_qpos writes as (N,1), matching robomimic's 2-D obs schema (SOARM gripper is 1-DOF and returns a scalar, unlike Panda's (2,))"
  - "Anchor LIBERO/.gitignore's datasets rule with a leading slash (/libero/datasets) so it stops swallowing the new source package one level deeper"

patterns-established:
  - "Pattern 1: graceful-degradation package __init__ mirroring vla/__init__.py (sim-dependent submodules wrapped in try/except -> None)"
  - "Pattern 2: robomimic/LIBERO HDF5 schema with obs keys renamed at write time via OBS_KEY_MAPPING"
  - "Pattern 3: real (no-mock) local-sim integration test exercising the actual set_init_state() obs-regeneration path"

requirements-completed: [DATA-01]

coverage:
  - id: D1
    description: "raw_recorder.build_recording_env constructs a DataCollectionWrapper-wrapped SOARM env with zero custom controller kwargs (OSC_POSE)"
    requirement: "DATA-01"
    verification:
      - kind: integration
        ref: "LIBERO/libero/libero/datasets/test_hdf5_writer.py#test_schema_and_obs_key_naming"
        status: pass
    human_judgment: false
  - id: D2
    description: "gather_demonstrations_as_hdf5 writes robomimic-schema HDF5 with real regenerated image obs under LIBERO renamed keys, correct SOARM shapes, off-by-one alignment fixed"
    requirement: "DATA-01"
    verification:
      - kind: integration
        ref: "LIBERO/libero/libero/datasets/test_hdf5_writer.py#test_schema_and_obs_key_naming"
        status: pass
    human_judgment: false
  - id: D3
    description: "Only genuinely successful episodes reach the HDF5; unsuccessful attempts are silently dropped (total == 0)"
    requirement: "DATA-01"
    verification:
      - kind: integration
        ref: "LIBERO/libero/libero/datasets/test_hdf5_writer.py#test_unsuccessful_episode_not_written"
        status: pass
    human_judgment: false
  - id: D4
    description: "LIBERO/.gitignore no longer blanket-ignores the new datasets source package while still ignoring upstream's binary demo dir"
    requirement: "DATA-01"
    verification:
      - kind: other
        ref: "git check-ignore -v LIBERO/libero/libero/datasets/__init__.py (exit 1) && git check-ignore -v LIBERO/libero/datasets/x.hdf5 (exit 0)"
        status: pass
    human_judgment: false

# Metrics
duration: 35min
completed: 2026-08-02
status: complete
---

# Phase 4 Plan 01: Dataset Recording Infrastructure Summary

**Shared SOARM recording infrastructure — a DataCollectionWrapper env builder plus an HDF5 writer that turns raw recorded states into a robomimic-schema demo.hdf5 with real regenerated image observations under LIBERO's renamed obs keys, proven by a real headless-sim integration test.**

## Performance

- **Duration:** ~35 min
- **Started:** 2026-08-02
- **Completed:** 2026-08-02
- **Tasks:** 3
- **Files modified:** 5 (4 created, 1 modified)

## Accomplishments
- New importable `LIBERO/libero/libero/datasets/` package with vla-style graceful-degradation exports
- `build_recording_env()` — reusable DataCollectionWrapper-wrapped SOARM env (robots=[Soarm101], generic OSC_POSE, zero custom controller kwargs), the shared foundation for the later scripted collector and teleop plans
- `gather_demonstrations_as_hdf5()` — regenerates real image + proprio observations per recorded state via this repo's own `set_init_state()` primitive, renames to LIBERO's schema keys, gates on OR-accumulated per-episode success, applies the DataCollectionWrapper off-by-one fix
- Fixed a genuine gitignore bug (bare `datasets` matched the new source package at any depth) that would have silently made every file in the new package unstageable
- Real headless-sim integration test proving schema correctness, obs key renaming, SOARM-specific shapes, and honest success gating

## Task Commits

Each task was committed atomically:

1. **Task 1: gitignore scoping fix + package scaffold + raw_recorder.py** - `4384420` (feat)
2. **Task 2: hdf5_writer.py — robomimic-schema HDF5 with regenerated obs** - `b60ada1` (feat)
3. **Task 3: test_hdf5_writer.py — real local-sim integration proof** - `1dc25b7` (test)

_Task 3's commit also carries the two Rule-2/Rule-3 auto-fixes (import anchor + proprio shape) since they were discovered while running the integration test against the real env._

## Files Created/Modified
- `LIBERO/libero/libero/datasets/__init__.py` - graceful-degradation package exports mirroring vla/__init__.py
- `LIBERO/libero/libero/datasets/raw_recorder.py` - `build_recording_env()` shared DataCollectionWrapper setup
- `LIBERO/libero/libero/datasets/hdf5_writer.py` - `gather_demonstrations_as_hdf5()` + `OBS_KEY_MAPPING`
- `LIBERO/libero/libero/datasets/test_hdf5_writer.py` - two real-sim integration tests
- `LIBERO/.gitignore` - anchored `datasets` rule to `/libero/datasets`

## Decisions Made
- **LIBERO-anchored env imports over relative:** LIBERO's problem classes register into `TASK_MAPPING` via `from LIBERO.libero.libero.envs.bddl_base_domain import register_problem`. A relative `from ..envs import TASK_MAPPING` resolves to a separate, unregistered copy of the dict under pytest's `libero.envs` anchor — empty, causing `KeyError: 'libero_tabletop_manipulation'` at env construction. Both new modules therefore import via the LIBERO-anchored absolute path (repo ROOT on sys.path, per the 02-02 STATE.md decision), matching env_wrapper.py / collect_demonstration.py / create_scene.py.
- **atleast_1d proprio coercion:** SOARM's 1-DOF gripper returns `robot0_gripper_qpos` as a 0-d scalar (Panda returns `(2,)`), so naive stacking produced a malformed `(N,)` `gripper_states`. Coercing each per-step value to at least 1-D yields the `(N,1)` robomimic expects.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] LIBERO env imports must use the LIBERO-anchored absolute path**
- **Found during:** Task 3 (running the integration test)
- **Issue:** The plan instructed relative imports (`from ..envs import TASK_MAPPING`). Under pytest this bound an unregistered, empty `TASK_MAPPING` copy (problems register into the `LIBERO.libero.libero.envs` copy), raising `KeyError: 'libero_tabletop_manipulation'` when `build_recording_env` constructed the env.
- **Fix:** Changed `raw_recorder.py` and `hdf5_writer.py` to import envs via `from LIBERO.libero.libero.envs import ...` (the repo-wide convention env_wrapper.py itself uses), with the test inserting repo ROOT on sys.path.
- **Files modified:** LIBERO/libero/libero/datasets/raw_recorder.py, LIBERO/libero/libero/datasets/hdf5_writer.py
- **Verification:** Both integration tests pass; env constructs and obs regenerate end-to-end.
- **Committed in:** `1dc25b7` (Task 3 commit)

**2. [Rule 2 - Missing Critical] Proprio obs must be 2-D (N, D) for robomimic schema**
- **Found during:** Task 3 (schema assertion on `gripper_states`)
- **Issue:** SOARM's 1-DOF gripper returns `robot0_gripper_qpos` as a 0-d scalar; stacking produced a malformed `(N,)` array, and `gripper_states.shape[1]` raised `IndexError`. Phase 6's robomimic reader expects 2-D `(N, D)` proprio.
- **Fix:** Coerce each per-step proprio value with `np.atleast_1d` before stacking, yielding `(N,1)` for gripper and `(N,5)` for joints.
- **Files modified:** LIBERO/libero/libero/datasets/hdf5_writer.py
- **Verification:** `test_schema_and_obs_key_naming` asserts `gripper_states.shape[1] == 1` — passes.
- **Committed in:** `1dc25b7` (Task 3 commit)

---

**Total deviations:** 2 auto-fixed (1 blocking import-resolution, 1 missing-critical schema shape)
**Impact on plan:** Both auto-fixes are correctness-essential and stay within the plan's stated artifacts/behaviors. No scope creep. The plan's relative-import guidance was empirically incorrect given LIBERO's absolute-import registration; the fix aligns with the repo-wide convention and the 02-02 sys.path decision.

## Issues Encountered
- **Git path casing:** the repo tracks the LIBERO tree under a lowercase `libero/` path (pre-existing from the earlier `feat: track LIBERO/ in git` commit) while the macOS filesystem shows `LIBERO`. Commits and staging work correctly; `git ls-files LIBERO/...` returns nothing but `git ls-files | grep libero/libero/libero/datasets` confirms all four files are tracked. Pre-existing repo state, out of scope for this plan.

## Threat Model Compliance
- **T-04-01-01 (Tampering, mitigate):** `assert len(states) == len(actions)` after `del states[-1]`, and `assert os.path.exists(bddl_file_name)` on entry to both public functions — a corrupt/misaligned recording raises immediately rather than silently writing a malformed demo group. Both present and exercised.
- **T-04-01-SC (package legitimacy):** no new package installs this plan — confirmed, only already-vetted robosuite/h5py/mujoco/numpy used.

## Known Stubs
None — all three modules are fully wired and exercised end-to-end against a real headless SOARM env.

## User Setup Required
None - no external service configuration required. Tests run locally via `conda run -n libero pytest`.

## Next Phase Readiness
- `build_recording_env()` is ready for the Wave 2 scripted collector (04-02) and teleop (04-03) to drive with action streams.
- `gather_demonstrations_as_hdf5()` + `OBS_KEY_MAPPING` are ready to assemble those recordings into the Phase-6-compatible HDF5.
- The `data` group carries `bddl_file_name` in its attrs so the replay-verification plan (04-04) can rebuild the regeneration env without a separate argument.
- Concern: images are written exactly as `set_init_state()` returns them (no vertical flip applied); if Phase 6's loader assumes flipped frames this is the single place to revisit.

## Self-Check: PASSED
- Files: __init__.py, raw_recorder.py, hdf5_writer.py, test_hdf5_writer.py all FOUND on disk and tracked in git
- Commits: 4384420, b60ada1, 1dc25b7 all FOUND in git log
- Tests: 2 passed (test_schema_and_obs_key_naming, test_unsuccessful_episode_not_written)

---
*Phase: 04-dataset-collection*
*Completed: 2026-08-02*
