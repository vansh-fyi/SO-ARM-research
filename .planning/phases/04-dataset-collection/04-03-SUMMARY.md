---
phase: 04-dataset-collection
plan: 03
subsystem: dataset
tags: [replay-verification, determinism, hdf5, robosuite, mujoco, soarm, data-02]

# Dependency graph
requires:
  - phase: 04-dataset-collection
    plan: 02
    provides: "The real 04-02 dataset (120 demos, put_the_cream_cheese_in_the_bowl_demo.hdf5, DATA-01 complete)"
  - phase: 04-dataset-collection
    plan: 01
    provides: "ControlEnv.set_state / set_init_state (env_wrapper.py), OBS_KEY_MAPPING (hdf5_writer.py) — the same regeneration primitives this plan proves are round-trip-deterministic"
provides:
  - "replay.verify_states_only(hdf5_path) -> dict — cheap, 100%-of-demos state-setting round-trip check"
  - "replay.verify_full_obs_regeneration(hdf5_path, sample_size=None) -> dict — sampled full obs-regeneration check"
  - "Confirmed DATA-02 result: 120/120 demos, 17457/17457 states round-trip exactly; 1746/1746 sampled obs-regenerations exact"
affects: [04-04-normalization, 04-05-teleop, phase-06-finetuning]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "State-based (not action-replay) determinism verification via ControlEnv.set_state/set_init_state — same primitives the writer used to build the dataset, proving a true round-trip rather than a second replay engine"
    - "Two-tier verification (D-06): cheap 100%-of-demos states-only tier (no rendering) + expensive sampled (~10%, min 5, always >=1/demo) full-obs-regeneration tier"
    - "Deterministic, evenly-spaced sampling (not naive prefix or unseeded random) for the expensive tier, always including each demo's first state"

key-files:
  created:
    - LIBERO/libero/libero/datasets/replay.py
    - LIBERO/libero/libero/datasets/test_replay.py
  modified:
    - LIBERO/libero/libero/datasets/__init__.py

key-decisions:
  - "verify_states_only builds exactly one OffScreenRenderEnv per HDF5 file (one bddl_file_name attr per file, this project's per-file convention) and uses ControlEnv.set_state + sim.forward() — NOT set_init_state, which also renders and would make the cheap tier expensive"
  - "np.allclose(..., atol=0.0) for the states-only round-trip: a pure state-vector round-trip (set then immediately read back) is expected to be exact, so atol=0.0 makes the check meaningfully strict rather than tolerant"
  - "Sample selection always includes the first state of every demo group (structural >=1/demo guarantee), then fills the remaining sample_size budget with a deterministic, evenly-spaced sweep over the rest of the (demo, step) universe — reproducible across runs, not skewed toward early demos/steps"
  - "Corrected a stale plan assumption (per the orchestrator's context note): 04-02's real output is ONE HDF5 file (120 demos in a single put_the_cream_cheese_in_the_bowl_demo.hdf5), not 3 per-task files, because D-07/D-08 retargeted the phase from 3 bowl-position variants down to 1 task. Task 3's verification script and test_replay.py's glob were written to reflect the real single-file output (no hardcoded file count), per the plan's own D-06/DATA-02 intent rather than the stale 3-file draft"

requirements-completed: [DATA-02]
requirements-blocked: []

# Metrics
duration: ~35min
completed: 2026-08-03
status: complete
---

# Phase 4 Plan 03: State-Based Determinism Replay Verification Summary

**Both verification tiers (states-only + sampled full-obs-regeneration) are implemented in `replay.py` and run clean against 04-02's real 120-demo dataset: all 17457 recorded states round-trip exactly via `ControlEnv.set_state`, and all 1746 sampled full-observation regenerations (10% of states, always >=1/demo) exactly match the recorded obs arrays — DATA-02 is satisfied against the real, phase-primary dataset.**

## Accomplishments

- `verify_states_only(hdf5_path)` — the cheap, 100%-of-demos tier. For every
  recorded `demo_N/states` row: `sim.set_state_from_flattened` (via
  `ControlEnv.set_state`) + `sim.forward()`, then re-read the flattened sim
  state and assert exact equality (`atol=0.0`) against the value that was
  just set. No rendering, no env stepping — pure state-vector math. Raises
  `AssertionError` naming the demo/step index on a non-finite recorded value
  or a round-trip mismatch.
- `verify_full_obs_regeneration(hdf5_path, sample_size=None)` — the
  expensive, sampled tier (D-06). Default sample size
  `max(5, ceil(0.10 * total_states))`; the first state of every demo is
  always included (structural >=1/demo guarantee) via a dedicated
  `_select_sample` helper that fills the remaining budget with a
  deterministic, evenly-spaced sweep over the rest of the (demo, step)
  universe. For each sampled state: `set_init_state(recorded_state)` (the
  same primitive `hdf5_writer.py` used to WRITE the dataset), rename via
  `OBS_KEY_MAPPING`, and assert exact equality against the recorded obs array
  for all 4 keys (`agentview_rgb`, `eye_in_hand_rgb`, `gripper_states`,
  `joint_states`).
- 4 tests in `test_replay.py`: a real-dataset-or-skip test per tier (against
  04-02's actual on-disk collection) and a synthetic-corruption test per tier
  proving fail-loudly behavior (not just the happy path).
- Full verification run against the real dataset (Task 3): **120/120 demos,
  17457/17457 states round-trip exactly; 1746/1746 sampled full-obs
  regenerations match exactly.** DATA-02 is satisfied.

## Task Commits

1. **Task 1: verify_states_only + real-dataset/corruption tests** — `c1c9962` (feat)
2. **Task 2: verify_full_obs_regeneration + real-dataset/corruption tests + __init__.py export** — `609e1b4` (feat)
3. **Task 3: full verification run against the real dataset** — no code change (verification-only; the run passed clean on the first attempt, see Deviations below for the one stale-assumption correction applied before running).

## Files Created/Modified

- `LIBERO/libero/libero/datasets/replay.py` — `verify_states_only`,
  `verify_full_obs_regeneration`, and shared helpers (`_open_h5`,
  `_demo_names`, `_build_env`, `_select_sample`) (created)
- `LIBERO/libero/libero/datasets/test_replay.py` — 4 tests, 2 per tier
  (created)
- `LIBERO/libero/libero/datasets/__init__.py` — guarded export of both
  verifiers, matching the package's sim-dependency graceful-degradation
  pattern (modified)

## Full Verification Run Result (Task 3, against 04-02's real dataset)

```
LIBERO/libero/datasets/soarm_spatial/put_the_cream_cheese_in_the_bowl_demo.hdf5
  states: {'total_demos': 120, 'total_states': 17457, 'passed': 17457}
  obs:    {'sampled_states': 1746, 'passed': 1746}
DATA-02 PASS: dataset fully deterministic on states, sampled-clean on obs regen
```

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Correctness] Stale "3 per-task HDF5 files" assumption corrected to the real single-file output**

- **Found during:** Plan start (flagged by the orchestrator's context note before any code was written).
- **Issue:** 04-03-PLAN.md's Task 3 verification/acceptance scripts (and the
  original draft of `test_replay.py`'s Test 3) were written against the
  phase's ORIGINAL 3-bowl-position-variant design (`assert len(paths) == 3`).
  D-07/D-08 (documented in 04-02-SUMMARY.md and 04-CONTEXT.md's amendment)
  retargeted the phase to a single sub-84mm in-reach task
  (`put_the_cream_cheese_in_the_bowl`), so 04-02's real collection produced
  exactly ONE HDF5 file (`put_the_cream_cheese_in_the_bowl_demo.hdf5`, 120
  demos), not 3.
- **Fix:** `test_replay.py`'s real-dataset glob (`_REAL_DATASET_GLOB`) and
  Task 3's inline verification script iterate over however many
  `*_demo.hdf5` files actually exist on disk, with no hardcoded file count
  (`assert len(paths) == 1` used only in the ad hoc Task 3 run script, which
  is disposable — the shipped test file has no such assertion at all, so it
  will not go stale again if a future plan adds more task files).
- **Files modified:** `LIBERO/libero/libero/datasets/test_replay.py` (never
  contained the stale assumption — written correctly from the start, per
  this context).
- **Commit:** `c1c9962` (test_replay.py's first commit already reflected the
  corrected, file-count-agnostic glob).

### Notes (not deviations, but noteworthy)

- The pre-existing `test_hdf5_writer.py::test_schema_and_obs_key_naming`
  failure (already logged in `deferred-items.md`) was investigated further
  during this plan's synthetic-HDF5 test-fixture construction: it is NOT
  about a demo-count regression as the deferred-items note's title implied —
  it is `gripper_states.shape[1] == 1` failing because SOARM's D-07 gripper
  upgrade (roboninecom 84mm parallel gripper, two independent sliding jaws)
  now reports a 2-DOF `robot0_gripper_qpos`, not the old single-DOF value the
  test was written against. This is out of scope for this plan (pre-existing,
  unrelated to `replay.py`) and does not affect `verify_states_only` /
  `verify_full_obs_regeneration`, which never hardcode a gripper DOF count.
  Not fixed here; left in `deferred-items.md` for a future plan to update
  that stale assertion.

## Threat Model Compliance

- **T-04-03-01 (Tampering, mitigate):** Both verifiers raise `AssertionError`
  naming the exact demo/step/key on first divergence rather than aggregating
  a silent pass/fail count. Verified directly by
  `test_verify_states_only_raises_on_corrupt_state` and
  `test_verify_full_obs_regeneration_raises_on_corrupted_obs`.
- **T-04-03-02 (DoS, accept):** The sampled tier's default (~10% of states,
  min 5) kept the full run against the real 17457-state dataset to ~39s
  wall-clock (1746 sampled obs regenerations); the states-only tier (17457
  states, no rendering) completed in seconds. No unbounded loop risk.
- **T-04-03-SC (package legitimacy, accept):** No new package installs this
  plan. Confirmed.

## Known Stubs

None — both verification functions are fully implemented, exercised against
real collected data (not just synthetic fixtures), and both fail-loudly paths
are proven by dedicated corruption tests.

## Self-Check: PASSED

- Files: `replay.py`, `test_replay.py` FOUND on disk (verified via `Read`
  during implementation); `__init__.py` diff FOUND via `git diff`.
- Commits: `c1c9962` and `609e1b4` FOUND in `git log --oneline -5`.
- Tests: `conda run -n libero pytest LIBERO/libero/libero/datasets/test_replay.py -x -q`
  — 4 passed (0 skipped, since 04-02's real dataset is present on disk).
- Full verification run (Task 3): printed `DATA-02 PASS` with
  `passed == total_states` (17457) and `passed == sampled_states` (1746) —
  both tiers green against the real dataset.

## Next Phase Readiness

- DATA-02 is now COMPLETE. Plans 04-04 (normalization statistics) and 04-05
  (teleoperation interface) can proceed against the same real, now
  determinism-verified dataset.
- `verify_states_only` / `verify_full_obs_regeneration` are importable from
  `libero.datasets` (guarded, matching the package's degrade-gracefully
  convention) for any future plan that wants to re-verify a newly collected
  or teleoperated dataset without re-deriving this logic.

---
*Phase: 04-dataset-collection*
*Completed: 2026-08-03*
