---
phase: 11-vla-hardware-connection
plan: 02
subsystem: robotics-hardware-control
tags: [vla-hardware, io-logger, e-stop, lerobot, jsonl, control-loop]

# Dependency graph
requires:
  - phase: 11-vla-hardware-connection
    provides: "Plan 11-01's action_contract.load_joint_limits_deg() + safety_validator.validate_action(), consumed directly (not reimplemented)"
provides:
  - "JSON Lines I/O logger (control/vla_bridge/io_logger.py) matching RESEARCH.md's Pattern 4 schema exactly, with independent per-camera timestamps and crash-durable flush-per-step writes"
  - "Real-hardware VLA episode harness (control/run_vla_episode.py) with a pluggable ActionSource seam, a scripted dry-run stand-in, and a software keyboard-interrupt e-stop"
affects: ["11-03", "11-04", "11-05 (dry-run/e-stop checkpoint reuses this exact script)"]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "ActionSource protocol (get_action(joint_state, instruction) -> (raw_action, model_version)) as the exact seam Plan 11-04's real Colab-bridged VLA will implement, with ScriptedActionSource as an explicit non-permanent stand-in"
    - "joint_limits_deg loaded once at startup by main() and passed into run_episode() as a parameter, not loaded internally -- keeps the control loop hermetic/testable against a mocked calibration file instead of the real hardware cache"

key-files:
  created:
    - control/vla_bridge/io_logger.py
    - control/test_io_logger.py
    - control/run_vla_episode.py
    - control/test_run_vla_episode.py
  modified: []

key-decisions:
  - "Installed the full lerobot[feetech]==0.6.1 (not just pytest) into this worktree's fresh control/.venv, unlike Plan 11-01's lighter venv, because run_vla_episode.py's <action> spec requires direct-importing keyboard_joint_control.py's read_positions/move_to_positions, and that module imports lerobot's SO101Follower/KeyboardTeleop classes at module load time -- there was no way to satisfy 'reuse ALL of this by direct import, do not reimplement' without a real lerobot install. Verified light: ~970MB venv, ~25s install, no torch/ultralytics chain needed since lerobot[feetech] alone was sufficient."
  - "joint_limits_deg is loaded once in main() (via action_contract.load_joint_limits_deg()) and passed into run_episode() as an explicit parameter, rather than loaded inside run_episode() itself -- found during test-writing that calling load_joint_limits_deg() with no path argument silently depends on this machine's real ~/.cache/huggingface/lerobot/calibration/... file being present, which would make control/test_run_vla_episode.py non-hermetic (passes here, fails on a clean machine/CI). Verified by temporarily removing the real calibration file and re-running the full control/ suite -- all 23 tests still passed."

patterns-established:
  - "run_episode() is a standalone, mockable-robot function separate from main()'s CLI/real-hardware wiring -- the same separation record_episode.py doesn't have, but needed here for hermetic testing of the control loop against Plan 11-01's mock_robot fixture"

requirements-completed: [VLAHW-02, VLAHW-03]

coverage:
  - id: D1
    description: "JSON Lines I/O logger (IOLogger) writes one record per step matching RESEARCH.md's Pattern 4 schema exactly, with independent per-camera timestamps and durable (crash-safe) writes"
    requirement: "VLAHW-03"
    verification:
      - kind: unit
        ref: "control/test_io_logger.py -x -q"
        status: pass
    human_judgment: false
  - id: D2
    description: "Real-hardware episode harness (run_vla_episode.py) gates every send_action() through safety_validator.validate_action(), always writes termination.json with a real reason, and always returns-to-start before disconnect on max-steps, KeyboardInterrupt, and any other exception"
    requirement: "VLAHW-02"
    verification:
      - kind: unit
        ref: "control/test_run_vla_episode.py -x -q"
        status: pass
    human_judgment: false
  - id: D3
    description: "run_vla_episode.py is ready for a real hardware-in-the-loop supervised dry run (Plan 11-05's e-stop checkpoint reuses this exact script) -- not automatable without the physical robot"
    verification: []
    human_judgment: true
    rationale: "Requires the real SO-ARM101, real cameras, and a human observing the e-stop/return-to-start behavior live -- explicitly deferred to Plan 11-05's hardware-in-the-loop checkpoint per this plan's own scope"

duration: 20min
completed: 2026-09-21
status: complete
---

# Phase 11 Plan 02: JSON Lines I/O Logger + Real-Hardware Episode Harness Summary

**JSON Lines I/O logger (`IOLogger`, matching RESEARCH.md's Pattern 4 schema with independent per-camera timestamps and crash-durable flushing) plus a real-hardware `run_vla_episode.py` control loop wiring Plan 11-01's safety validator, a scripted dry-run `ActionSource` stand-in, and a software keyboard-interrupt e-stop that always returns the arm to start before disconnect**

## Performance

- **Duration:** 20 min
- **Started:** 2026-09-21T10:26:00+05:30
- **Completed:** 2026-09-21T10:46:00+05:30
- **Tasks:** 2 completed
- **Files modified:** 4 (all created)

## Accomplishments
- Built `control/vla_bridge/io_logger.py`'s `IOLogger` class: `write_step()` produces one flush-per-call JSON Lines record per inference step with every field from RESEARCH.md's Pattern 4 schema; `capture_camera_frame()` records genuinely independent per-camera timestamps (not a single shared timestamp like `record_episode.py`'s anti-pattern) and never raises on a failed camera read
- Built `control/run_vla_episode.py`: a real-hardware control loop harness with a pluggable `ActionSource` protocol, a `ScriptedActionSource` stand-in (toggles gripper every 30 steps to prove the write path is genuinely live), and a software keyboard-interrupt e-stop that calls `move_to_positions` (return-to-start) strictly before `robot.disconnect()` in every exit path (max-steps, KeyboardInterrupt, or any other exception)
- Every candidate action passes through `safety_validator.validate_action()` before reaching `robot.send_action()` -- no unvalidated write path exists
- `termination.json` always records a real `reason` (`max_steps_reached` | `keyboard_interrupt` | `exception`) and `steps_completed`
- Directly reused (never reimplemented) `keyboard_joint_control.py`'s `read_positions`/`move_to_positions` comms-retry and return-to-start logic, and Plan 11-01's `action_contract`/`safety_validator` modules

## Task Commits

Each task was committed atomically:

1. **Task 1: JSON Lines I/O logger (VLAHW-03)** - `d35b35e` (feat)
2. **Task 2: Real-hardware episode harness with scripted dry-run source + e-stop (VLAHW-02, VLAHW-03)** - `ad0eac2` (feat)

_Both tasks were TDD-tagged in the plan; each task's behavior/tests were written and verified together as a cohesive unit, matching Plan 11-01's precedent for this same reason (each task describes one small, cohesive module, not an incrementally-grown feature)._

## Files Created/Modified
- `control/vla_bridge/io_logger.py` - `IOLogger` class: `write_step()`, `capture_camera_frame()`, context-manager `__enter__`/`__exit__`
- `control/test_io_logger.py` - 6 tests: Pattern 4 schema completeness, 3-step JSONL validity, independent per-camera timestamps, real file-path resolution, failed-read non-raising behavior, mid-episode-crash durability
- `control/run_vla_episode.py` - `ActionSource` protocol, `ScriptedActionSource`, `run_episode()` (mockable control loop), `main()` CLI entrypoint (`--camera`, `--out`, `--instruction`, `--max-steps`, `--control-hz`)
- `control/test_run_vla_episode.py` - 3 tests: scripted gripper-toggle-after-30-steps behavior, max-steps termination reason, keyboard-interrupt return-to-start-before-disconnect ordering

## Decisions Made
- Installed full `lerobot[feetech]==0.6.1` into this worktree's fresh `control/.venv` (see Deviations below) -- required because this task's plan explicitly mandates direct-importing `keyboard_joint_control.py`'s `read_positions`/`move_to_positions`, and that module's own top-level imports pull in `lerobot`'s `SO101Follower`/`KeyboardTeleop` classes; there was no way to satisfy the plan's "reuse ALL of this by direct import" instruction with a lighter venv
- Refactored `joint_limits_deg` loading out of `run_episode()` and into `main()` (loaded once "at startup" per the plan's own phrasing, then passed as a parameter) after discovering the internal-load version silently depended on this machine's real hardware calibration cache -- fixed for test hermeticity, verified by removing the real calibration file and re-running the full suite

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Installed the full `lerobot[feetech]==0.6.1` (not a pytest-only venv) into a fresh `control/.venv`**
- **Found during:** Task 2 (episode harness implementation)
- **Issue:** This worktree had no pre-existing `control/.venv` (gitignored, doesn't carry over from the main checkout). Plan 11-01 established a precedent of installing only `pytest` into a lightweight venv since its code had no runtime dependency on `lerobot`/`opencv`/`ultralytics`. This plan's Task 2, however, explicitly requires direct-importing `keyboard_joint_control.read_positions`/`move_to_positions` (per the plan's `<read_first>` and `<action>` sections: "reuse ALL of this by direct import, do not reimplement any of it") -- and `keyboard_joint_control.py` itself imports `lerobot.robots.so_follower.*`/`lerobot.teleoperators.keyboard.*` at module load time. Any test importing `run_vla_episode` (which imports `keyboard_joint_control`) would fail without a working `lerobot` install, regardless of whether `run_vla_episode.py` itself deferred its own `lerobot` imports.
- **Fix:** Created a fresh `python3.12 -m venv control/.venv` and installed `lerobot[feetech]==0.6.1` (matching the pinned version in `control/requirements.txt`) plus `pytest==8.3.4`, `opencv-python==4.10.0.84`, and `numpy` -- verified this was lightweight in practice (~970MB venv, ~25s install), not the multi-GB torch/ultralytics chain Plan 11-01 was avoiding, since `lerobot[feetech]` alone (without the `[async]`/training extras) was sufficient for this plan's imports.
- **Files modified:** None beyond the two new files this plan already creates (`control/requirements.txt` was not modified; the `opencv-python` version installed into this throwaway venv, `4.10.0.84`, differs from the pinned `5.0.0.93` in `requirements.txt` -- the pin itself was left untouched since updating it was out of this plan's file scope).
- **Verification:** `control/.venv/bin/python -m pytest test_io_logger.py test_run_vla_episode.py test_action_contract.py test_safety_validator.py -q` -- 23 passed.
- **Committed in:** `ad0eac2` (Task 2 commit; `.venv/` itself is gitignored, not committed)

**2. [Rule 1 - Bug] Fixed a test-hermeticity bug: `run_episode()` was loading joint limits from the real hardware calibration cache**
- **Found during:** Task 2 (writing `test_run_vla_episode.py`)
- **Issue:** The plan's `<action>` text describes `joint_limits_deg = action_contract.load_joint_limits_deg()` as part of the main-loop setup. My first implementation called this inside `run_episode()` with no path argument, which defaults to `CALIBRATION_PATH` (`~/.cache/huggingface/lerobot/calibration/robots/so_follower/soarm_follower_02.json`) -- this project's *own real hardware's* live calibration cache. Tests passed, but only because this specific development machine happens to have that file cached from prior hardware bring-up sessions; on a clean machine or CI runner without it, every `test_run_vla_episode.py` test exercising `run_episode()` would raise `FileNotFoundError`. This diverges from Plan 11-01's own precedent (`test_action_contract.py` always injects `mock_calibration_file` explicitly, never relying on the default path).
- **Fix:** Moved the `load_joint_limits_deg()` call to `main()` (matching the plan's "once at startup" framing) and added `joint_limits_deg` as an explicit parameter to `run_episode()`. Updated `test_run_vla_episode.py`'s two `run_episode()`-calling tests to derive `joint_limits_deg` from Plan 11-01's `mock_calibration_file` fixture instead of the real hardware cache.
- **Files modified:** `control/run_vla_episode.py`, `control/test_run_vla_episode.py`
- **Verification:** Temporarily moved the real `soarm_follower_02.json` calibration file out of `~/.cache/huggingface/...` and re-ran the full `control/` test suite (`test_io_logger.py test_run_vla_episode.py test_action_contract.py test_safety_validator.py`) -- all 23 tests still passed with the real file absent, then restored the file. Confirms the fix.
- **Committed in:** `ad0eac2` (Task 2 commit; the fix was made before the commit, so no separate follow-up commit was needed)

---

**Total deviations:** 2 auto-fixed (1 blocking, venv-scoping; 1 bug, test hermeticity)
**Impact on plan:** No scope creep. The venv-scoping deviation was a strict superset of what Plan 11-01 needed (this plan's own explicit reuse requirement forced a heavier but still lightweight install). The hermeticity fix is a pure correctness improvement to the plan's own described data flow (`joint_limits_deg` loaded "once at startup"), caught before it could silently pass on this machine and fail elsewhere.

## Issues Encountered
None beyond the two deviations documented above.

## User Setup Required

None - no external service configuration required. (Note, carried forward from Plan 11-01: a real `control/.venv` with the full `requirements.txt` -- including `lerobot[feetech]` -- is now present in this worktree with `lerobot[feetech]==0.6.1`, `pytest==8.3.4`, `opencv-python==4.10.0.84`, and `numpy` installed; this is a throwaway worktree venv, not a permanent replacement for a real hardware-connected `control/.venv` setup outside this ephemeral environment.)

## Next Phase Readiness
- `control/run_vla_episode.py` is ready for a real hardware-in-the-loop dry run: `cd control && .venv/bin/python run_vla_episode.py <PORT> soarm_follower_02 --out outputs/dry_run_001 --max-steps 20` against the real robot, per this plan's own `<verification>` note -- not run in this session (no physical hardware access from this worktree)
- The `ActionSource` seam (`get_action(joint_state, instruction) -> (raw_action, model_version)`) is the exact interface Plan 11-04 will implement with a real `BridgeActionSource` -- no other part of `run_vla_episode.py` should need to change when that swap happens
- Plan 11-05's e-stop checkpoint reuses this exact script -- `termination.json`'s `keyboard_interrupt` reason and the return-to-start-before-disconnect ordering are both already unit-tested here, ready for the hardware-in-the-loop confirmation
- `MAX_RELATIVE_TARGET_DEG`/`MAX_VELOCITY_DEG_PER_S` (Plan 11-01's `[ASSUMED]` conservative defaults) are now wired end-to-end into the real send path via `SOFollowerRobotConfig(max_relative_target=...)` -- still flagged for tuning during the actual hardware dry run per RESEARCH.md's Assumption A3
- No blockers identified for proceeding to Plan 11-03/11-04 (bridge wiring) or Plan 11-05 (hardware-in-the-loop dry run)

## Self-Check: PASSED

- FOUND: control/vla_bridge/io_logger.py
- FOUND: control/test_io_logger.py
- FOUND: control/run_vla_episode.py
- FOUND: control/test_run_vla_episode.py
- FOUND commit: d35b35e
- FOUND commit: ad0eac2

---
*Phase: 11-vla-hardware-connection*
*Completed: 2026-09-21*
