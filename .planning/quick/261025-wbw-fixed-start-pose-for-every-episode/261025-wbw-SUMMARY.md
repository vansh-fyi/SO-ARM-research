---
phase: quick-261025-wbw
plan: 01
subsystem: control (real-robot episode harness)
status: complete
tags: [safety, start-pose, run_vla_episode, tune_execution_hz, so101]
requires: [vla_bridge.action_contract, keyboard_joint_control.move_to_positions/read_positions]
provides: [control/start_pose.py, always-home episode start and return, <out>/start_pose.json run record]
affects: [control/run_vla_episode.py, control/tune_execution_hz.py, every run_episode caller]
tech-stack:
  added: []
  patterns: [validate-before-motion pre-flight, at-home guard on return move, autouse conftest path redirection]
key-files:
  created:
    - control/start_pose.py
    - control/test_start_pose.py
  modified:
    - control/conftest.py
    - control/run_vla_episode.py
    - control/test_run_vla_episode.py
    - control/test_base_mllm.py
    - control/test_mllm_record.py
    - control/tune_execution_hz.py
    - control/COMMANDS.md
decisions:
  - "One module control/start_pose.py holds the validate/load/capture library and the capture CLI (python start_pose.py [PORT] [ROBOT_ID] [--force])"
  - "The move-home lives in run_episode() (so tune_execution_hz inherits it); main() also loads the pose as a pre-flight check before any backend build"
  - "run_episode(..., home_pose=None) is the test seam, appended last; None loads the committed file; no production opt-out flag"
  - "START_POSE_TOLERANCE=5.0 (deg / %), START_POSE_MOVE_SECONDS=15.0 for both moves, START_POSE_KP=0.2, START_POSE_CONTROL_HZ=30"
  - "No return move if home was never reached (start move missed, raised or was Ctrl-C'd)"
  - "Run record is <out>/start_pose.json; termination.json keeps its {reason, steps_completed} shape with new reason start_pose_not_reached"
metrics:
  duration: ~10 min
  completed: 2026-10-25
  tasks: 3
  files: 9
  tests_final: 1790 passed (baseline 1729)
---

# Quick 261025-wbw: Fixed start pose for every episode Summary

Every real-robot episode now starts at one committed, limit-validated home pose (`control/start_pose.json`), checks arrival within ±5 deg/% before the first model request, records target vs reached in `<out>/start_pose.json`, and returns to that home pose before disconnect on every exit path once it has arrived.

## What was built

- **`control/start_pose.py`** (new): `START_POSE_PATH`, `CAPTURE_COMMAND`, the four tuning constants, `StartPoseError`, `StartPoseNotReachedError` (stores `.missed` and builds a loud per-joint message with a re-capture hint), `validate_start_pose` (collects every problem into one error; rejects bool, non-numeric, NaN/inf, unknown, missing, unlimited and out-of-limit joints), `load_start_pose` (resolves `START_POSE_PATH` at call time; every failure names the path and `python start_pose.py`), `capture_start_pose` (checks exists/`--force` first, reads only, validates, rounds to 2 dp, writes indent-2 JSON with `captured_at`, `units`, `joints`; no robot id), `joints_out_of_tolerance`, and the capture CLI `main()`. The CLI refuses an existing file without `--force` before constructing a robot, falls back to device_map.json via an in-function import (to avoid a circular import), applies calibration with torque off like keyboard_joint_control, and always disconnects.
- **`control/conftest.py`**: autouse `start_pose_file` fixture writes an all-zero home pose under `tmp_path_factory` and patches `start_pose.START_POSE_PATH`, so no test can touch the real file.
- **`control/run_vla_episode.py`**:
  - `run_episode` validates or loads `home_pose` before any motion.
  - Inside the try, it moves home (`label="Moving to start pose"`), reads positions, and writes the `start_pose.json` record whether it passes or fails.
  - If any joint missed, it raises `StartPoseNotReachedError` with zero `get_action` calls and reason `start_pose_not_reached`.
  - The depth recorder is built only after arrival.
  - The return move targets HOME with no label and runs only when `at_home` is set.
  - The Ctrl-C message distinguishes the before-home and after-home cases.
  - `main()` pre-flight loads the pose right after `load_joint_limits_deg()`, before `BACKEND_REGISTRY` builds anything (`parser.error` on failure). It passes `home_pose=` and turns not-reached into `SystemExit("REFUSED TO START EPISODE: ...")`, and its `finally` still shuts the backend down and disconnects.
- **`control/tune_execution_hz.py`**: loads limits and the home pose before robot construction/connect (`SystemExit` on a bad file), centres `SweepActionSource` on the home pose, passes `home_pose=`, and converts not-reached into the same `REFUSED TO START EPISODE` exit. The unused `read_positions` import was dropped.
- **`control/COMMANDS.md`**: a new "Start (home) pose — captured once, used by every episode" section before "Episode recording". It covers the capture command, `--force`, the capture caveats, and the always-home, refusal and recording behaviour.

## Commits

| Task | Commit | Message |
|------|--------|---------|
| 1 RED | ade97e6 | test(261025-wbw): add failing tests for start_pose library and capture CLI |
| 1 GREEN | 2a01662 | feat(261025-wbw): add start_pose library and capture CLI |
| 2 RED | 12ddd02 | test(261025-wbw): add failing tests for always-home episode start and return |
| 2 GREEN | ebef77a | feat(261025-wbw): start every episode at the committed home pose and return to it |
| 3 | 5d54d56 | feat(261025-wbw): align tune_execution_hz with home pose and document capture |

## Existing tests changed (minimal, as planned)

- `test_run_vla_episode.py`:
  - Added `PERMISSIVE_JOINT_LIMITS_DEG` and replaced the empty-limits patch line with it everywhere.
  - `_dispatch_main` now records `home_pose` and takes an optional `run_episode_impl`.
  - The episode-complete test expects 2 moves (start + return), both to the all-zero home.
  - The Ctrl-C ordering test also asserts the final entries are move then disconnect.
- `test_base_mllm.py` and `test_mllm_record.py`: only the `_run` helper changed. It now passes `home_pose=` equal to the MockRobot's pre-set positions.
- No other test files needed changes. No test takes the ~15 s timeout path (the slowest test in the full run is about 2 s, a pre-existing colab_server_utils test).

## Deviations from Plan

- **Site count:** the plan estimated 13 empty-limits patch sites in test_run_vla_episode.py, but there were 12. The one replace_all covered all of them.
- **Extra test:** added `test_interrupt_before_reaching_home_makes_no_return_move`. It covers the planner's "no return move if Ctrl-C'd during the start move" choice directly.
- **Disconnect coverage in the capture CLI:** `robot.bus.write_calibration` sits inside the try/finally, so the robot is also disconnected if the calibration write fails. The plan placed it just before the try. This is a small Rule 2 safety tightening.

Otherwise the plan was executed as written.

## Verification

- `pytest -q` in the worktree's control/ (using the main checkout's venv interpreter): **1790 passed** (baseline 1729; 49 new start_pose tests and 12 new run_vla_episode tests). That includes test_colab_notebooks.py.
- `control/start_pose.json` does not exist and is not tracked. The real file must be captured on the arm with `python start_pose.py` and committed by the user.
- `git diff --stat 6c05ac1..HEAD` touches only the nine files listed in files_modified.

## Known Stubs

None.

## Threat Flags

None. The new surface (a hand-editable pose file whose values drive the harness move) is covered by T-wbw-01..06 in the plan's threat model, and every `mitigate` item is implemented and tested.

## Self-Check: PASSED

- FOUND: control/start_pose.py, control/test_start_pose.py
- FOUND commits: ade97e6, 2a01662, 12ddd02, ebef77a, 5d54d56
