---
phase: 04-dataset-collection
verified: 2026-08-03T17:57:33Z
status: passed
score: 4/4 must-haves verified
behavior_unverified: 0
overrides_applied: 0
re_verification:
  previous_status: none
  note: initial verification
---

# Phase 04: Dataset Collection Verification Report

**Phase Goal:** A scripted and teleoperated demonstration collection system that records SOARM task completions as valid robomimic HDF5 datasets with state-based replay and correct SOARM-specific normalization statistics.
**Verified:** 2026-08-03T17:57:33Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths (Roadmap Success Criteria)

| # | Truth | Status | Evidence |
| --- | --- | --- | --- |
| 1 | Scripted collector produces 100+ SOARM demos in robomimic HDF5 with image observations present | ✓ VERIFIED | Real HDF5 inspected: `put_the_cream_cheese_in_the_bowl_demo.hdf5`, `data.attrs['total']==120`, 120 demo groups. `demo_1` has robomimic schema (states/actions/obs). `obs/agentview_rgb` shape (151,128,128,3) uint8, `eye_in_hand_rgb` same; frames non-black (min 0, max 210, mean 136.7). `env_args` = `{"env_name":"libero_tabletop_manipulation","env_type":1,"env_kwargs":{"robots":["Soarm101"],"controller_name":"OSC_POSE"}}` (robomimic env_type=1). |
| 2 | Any recorded demo replays frame-identically via state-setting (not action playback), deterministic on re-run | ✓ VERIFIED | `replay.verify_states_only` uses `ControlEnv.set_state`/`set_state_from_flattened` + `sim.forward()` (no action playback). Behavioral spot-check this session: ran against teleop_v2 demo → 2061/2061 states round-trip exact (atol=0.0). 04-03 full run recorded 17457/17457 scripted states + 1746/1746 sampled obs-regens exact. |
| 3 | SOARM-specific action/observation normalization statistics computed from the collected dataset (not copied from Panda) | ✓ VERIFIED | `dataset_statistics.json` present: 17457 transitions, 120 trajectories, action dim 7, proprio dim 7 (5 SOARM joints + 2-DOF gripper — NOT Panda's 7+2). Values reflect real collected variance (q01<q99 on non-degenerate dims; 3 rotation-delta dims degenerate at 0.0, matching the FSM's fixed top-down convention). OpenVLA q01/q99/mean/std/min/max schema matches `oft_backend.py`'s consumer. |
| 4 | Human operator can record SOARM demos via teleop interface, landing in same HDF5 format | ✓ VERIFIED | `teleop.py` keyboard-only (`input2action` + `robosuite.devices.Keyboard`, no SpaceMouse path — grep empty), reuses `build_recording_env` + `gather_demonstrations_as_hdf5` verbatim. Two real human demos on disk: `cream_cheese_teleop_demo.hdf5` (5814 states) and `_v2.hdf5` (2061 states), each `total==1`, identical obs key set/shapes to scripted path (agentview_rgb (N,128,128,3) uint8, states (N,41), actions (N,7)). Both pass states-only determinism (5814/5814, 2061/2061). |

**Score:** 4/4 truths verified (0 present, behavior-unverified)

### Required Artifacts

| Artifact | Expected | Status | Details |
| -------- | ----------- | ------ | ------- |
| `raw_recorder.py` | build_recording_env → DataCollectionWrapper SOARM env | ✓ VERIFIED | `def build_recording_env` present (90 lines); imported/used by collector.py + teleop.py |
| `hdf5_writer.py` | gather_demonstrations_as_hdf5 + OBS_KEY_MAPPING | ✓ VERIFIED | Function, OBS_KEY_MAPPING dict (4 renamed keys), `del states[-1]` off-by-one fix, `set_init_state` obs regen all present |
| `collector.py` | compute_waypoint_action, run_scripted_episode, collect_task, collect_all, TASKS | ✓ VERIFIED | All 4 functions + TASKS present (453 lines); TASKS retargeted to cream_cheese task, no `next_to_the_plate` |
| `replay.py` | verify_states_only, verify_full_obs_regeneration | ✓ VERIFIED | Both functions present (224 lines); ran clean behaviorally |
| `normalization.py` | compute_norm_stats, load_actions_and_proprio_from_hdf5, write_dataset_statistics | ✓ VERIFIED | All 3 present (155 lines); pure-numpy, exercised by unit + integration tests |
| `teleop.py` | run_teleop_episode, collect_teleop | ✓ VERIFIED | Both present (251 lines); keyboard-only |
| `soarm_spatial/*_demo.hdf5` (scripted) | ≥100 demos robomimic HDF5 | ✓ VERIFIED | 120 demos, 1.7 GB, real image obs (gitignored — present on disk) |
| `soarm_spatial/dataset_statistics.json` | SOARM norm stats | ✓ VERIFIED | Present, 17457 transitions / 120 trajectories |
| `soarm_spatial/teleop/*.hdf5` | human teleop demos | ✓ VERIFIED | 2 real demos, determinism-verified |

### Key Link Verification

| From | To | Via | Status |
| ---- | --- | --- | ------ |
| collector.py | raw_recorder.build_recording_env | headless scripted rollout | ✓ WIRED |
| collector.py | hdf5_writer.gather_demonstrations_as_hdf5 | end-of-task write | ✓ WIRED |
| collector.py | env body_xpos | ground-truth FSM targets | ✓ WIRED |
| replay.py | env_wrapper set_init_state/set_state | round-trip determinism | ✓ WIRED |
| normalization.py | oft_backend.py q01/q99 schema | Phase 6 overlay compatibility | ✓ WIRED (schema parity confirmed) |
| teleop.py | raw_recorder + hdf5_writer + replay | shared infra reuse | ✓ WIRED (verbatim reuse — schema convergence structural) |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
| -------- | ------- | ------ | ------ |
| State-setting replay determinism | `verify_states_only(teleop_v2.hdf5)` | 2061/2061 passed, exact | ✓ PASS |
| Norm-stats schema/quantile behavior | `pytest -k compute_norm_stats` | 4 passed | ✓ PASS |
| Waypoint FSM step logic | `pytest -k waypoint` | 8 passed | ✓ PASS |
| Scripted dataset schema/obs | h5py inspection of real HDF5 | 120 demos, image obs, SOARM shapes | ✓ PASS |
| Teleop dataset schema | h5py inspection of 2 teleop HDF5 | identical obs keys/shapes to scripted | ✓ PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
| ----------- | ---------- | ----------- | ------ | -------- |
| DATA-01 | 04-01, 04-02 | Scripted collector → robomimic HDF5 | ✓ SATISFIED | 120-demo dataset with image obs |
| DATA-02 | 04-03 | Deterministic state-setting replay | ✓ SATISFIED | 17457/17457 states + teleop demos round-trip exact |
| DATA-03 | 04-04 | SOARM-specific norm stats | ✓ SATISFIED | dataset_statistics.json from real data |
| DATA-04 | 04-05 | Human teleop recording | ✓ SATISFIED | 2 real human demos, same HDF5 format |

No orphaned requirements — all Phase-4 IDs (DATA-01..04) claimed by plans. Note: REQUIREMENTS.md traceability table still lists DATA-04 as "Pending" — stale bookkeeping; the deliverable is complete on disk. Recommend updating the table to Complete.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
| ---- | ---- | ------- | -------- | ------ |
| `test_hdf5_writer.py` | 97 | `assert gripper_states.shape[1] == 1` (stale) | ⚠️ Warning | `test_schema_and_obs_key_naming` fails: asserts 1-DOF gripper, but the D-07 roboninecom 84mm gripper upgrade (mid-phase, intentional) makes `gripper_states` 2-DOF. The real dataset is correct (2-DOF); only the test's hardcoded expectation is outdated. Logged in `deferred-items.md`. Does NOT affect any of the 4 success criteria. |

No `TBD`/`FIXME`/`XXX` debt markers in any phase source file.

### Human Verification Required

None outstanding. DATA-04's human-action checkpoint (04-05 Task 3) was executed and confirmed this session — two human-driven teleop demos exist on disk and pass determinism verification. The one behavior requiring a live human (physically driving the arm) has already been performed; its artifacts are verifiable and were re-inspected here.

### Gaps Summary

No goal-blocking gaps. All four roadmap success criteria are observably true against the real (gitignored) on-disk artifacts, cross-checked with direct h5py inspection and a live behavioral determinism run rather than SUMMARY claims. The scripted pipeline produced 120 demos (>100), state-setting replay is exact, SOARM-specific normalization stats reflect the real 5-joint + 2-DOF-gripper embodiment (not Panda), and the human teleop path lands in the identical HDF5 schema.

One WARNING (non-blocking): a stale 1-DOF-gripper assertion in `test_hdf5_writer.py::test_schema_and_obs_key_naming` fails after the intentional mid-phase gripper upgrade. It is a test-maintenance debt already tracked in `deferred-items.md`, not a defect in the phase deliverables — the dataset itself carries the correct 2-DOF gripper shape. Recommend updating that assertion and the REQUIREMENTS.md DATA-04 status in a follow-up.

---

_Verified: 2026-08-03T17:57:33Z_
_Verifier: Claude (gsd-verifier)_
