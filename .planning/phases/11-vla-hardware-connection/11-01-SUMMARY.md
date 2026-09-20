---
phase: 11-vla-hardware-connection
plan: 01
subsystem: robotics-safety
tags: [vla-hardware, safety-validator, action-contract, lerobot, so-101]

# Dependency graph
requires:
  - phase: 10-digital-twin-fidelity
    provides: Live calibration-derived joint limits methodology (scripts/calibration_utils.py's tick->radian formula, reused here for tick->degree)
provides:
  - Documented, tested SO-101 real-hardware action contract (6 joints, DEGREES arm + RANGE_0_100 gripper, absolute targets)
  - Safety validator gating every candidate action before it can reach the real servo bus
affects: [11-02, 11-03, 11-04, later Phase 11 plans wiring the VLA bridge/robot client]

# Tech tracking
tech-stack:
  added: [pytest==8.3.4 (control/requirements.txt)]
  patterns:
    - "control/vla_bridge/ package for VLA-hardware-connection glue code, mirroring scripts/calibration_utils.py's calibration-tick-to-angle formula pattern"
    - "control/conftest.py shared fixtures (mock_robot, mock_calibration_file) for hardware-free unit testing of control/ code"

key-files:
  created:
    - control/vla_bridge/__init__.py
    - control/vla_bridge/action_contract.py
    - control/vla_bridge/safety_validator.py
    - control/conftest.py
    - control/test_action_contract.py
    - control/test_safety_validator.py
  modified:
    - control/requirements.txt
    - control/keyboard_joint_control.py

key-decisions:
  - "Action contract resolved as DEGREES (5 arm joints) + RANGE_0_100 percent (gripper), absolute targets, 6-DOF joint order -- verified directly against installed lerobot 0.6.1 source (config_so_follower.py's use_degrees=True default), not inferred from the stale in-repo comment"
  - "wrist_roll and gripper safety limits are fixed constants, never derived from live calibration ticks, because their calibration entries are known placeholders (full-turn range and percent-mode range respectively)"
  - "Installed only pytest into a fresh control/.venv (not the full requirements.txt) since this worktree had no pre-existing .venv and the new safety-logic code has no runtime dependency on lerobot/opencv/ultralytics"

patterns-established:
  - "Safety validator layers 4 sequential clamps per joint (NaN/inf hold, absolute limit, per-step displacement, dt-aware velocity), then an outer stale-observation override that discards all per-joint results"

requirements-completed: [VLAHW-01, VLAHW-02]

coverage:
  - id: D1
    description: "Documented action contract module (JOINT_ORDER, ACTION_UNITS, ACTION_MODE, calibration-derived joint limits) pinned by automated tests"
    requirement: "VLAHW-01"
    verification:
      - kind: unit
        ref: "control/test_action_contract.py -x -q"
        status: pass
    human_judgment: false
  - id: D2
    description: "Safety validator clamps out-of-range/NaN/inf/oversized-step/oversized-velocity actions and holds position on stale observations"
    requirement: "VLAHW-02"
    verification:
      - kind: unit
        ref: "control/test_safety_validator.py -x -q"
        status: pass
    human_judgment: false

duration: 25min
completed: 2026-09-20
status: complete
---

# Phase 11 Plan 01: Action Contract + Safety Validator Summary

**Documented, test-pinned SO-101 action contract (DEGREES + RANGE_0_100, absolute, 6-DOF) plus a 4-layer safety validator (limit/step/velocity clamp + NaN/stale hold) that every VLA action must pass before reaching the real servo bus**

## Performance

- **Duration:** 25 min
- **Started:** 2026-09-20T21:26:00Z
- **Completed:** 2026-09-20T21:51:49Z
- **Tasks:** 2 completed
- **Files modified:** 8 (6 created, 2 modified)

## Accomplishments
- Resolved VLAHW-01's action-contract ambiguity as a documented, tested Python module (`control/vla_bridge/action_contract.py`) rather than tribal knowledge in a comment -- confirms `use_degrees=True` (DEGREES mode for the 5 arm joints) + `RANGE_0_100` (gripper) + absolute targets + 6-DOF joint order, directly against installed lerobot 0.6.1 source
- Built the safety validator (`control/vla_bridge/safety_validator.py`) VLAHW-02 requires: per-joint NaN/inf rejection (hold current state), absolute joint-limit clamp (calibration-derived for 4 arm joints, fixed constants for wrist_roll/gripper), per-step displacement cap, dt-aware velocity cap, and a stale-observation override that holds every joint
- Fixed the stale `RANGE_M100_100` comment in `control/keyboard_joint_control.py` (no behavior change, DEGREES-mode clarification only)
- Set up `control/`'s first pytest test infrastructure (`conftest.py` with `mock_robot`/`mock_calibration_file` fixtures reusable by every later Phase 11 test file)

## Task Commits

Each task was committed atomically:

1. **Task 1: Test infrastructure + documented action contract (VLAHW-01)** - `486dc00` (feat)
2. **Task 2: Safety validator (VLAHW-02)** - `aea1257` (feat)

_Both tasks were TDD-tagged in the plan; tests and implementation were written and verified together per task rather than as separate RED/GREEN commits, since each task's `<behavior>` section described a small, cohesive unit (a single module) rather than an incrementally-grown feature. All test files pass green as committed._

## Files Created/Modified
- `control/vla_bridge/__init__.py` - empty, makes `vla_bridge` an importable package
- `control/vla_bridge/action_contract.py` - `JOINT_ORDER`, `ACTION_UNITS`, `ACTION_MODE`, `CALIBRATION_PATH`, `JOINTS_FROM_CALIBRATION`, `WRIST_ROLL_LIMIT_DEG`, `GRIPPER_LIMIT_PCT`, `calibration_ticks_to_degrees()`, `load_joint_limits_deg()`
- `control/vla_bridge/safety_validator.py` - `MAX_RELATIVE_TARGET_DEG`, `MAX_VELOCITY_DEG_PER_S`, `STALE_OBSERVATION_S`, `STALE_ACTION_S`, `validate_action()`
- `control/conftest.py` - `mock_robot`/`MockRobot` and `mock_calibration_file` shared pytest fixtures
- `control/test_action_contract.py` - 7 tests pinning the action contract
- `control/test_safety_validator.py` - 7 tests pinning validator behavior
- `control/requirements.txt` - added `pytest==8.3.4`
- `control/keyboard_joint_control.py` - corrected stale `RANGE_M100_100` comment to describe the real DEGREES-mode contract

## Decisions Made
- Action contract resolved as DEGREES (arm) + RANGE_0_100 (gripper), absolute, 6-DOF -- pinned by tests, not left as a comment
- `wrist_roll`/`gripper` safety limits are fixed constants, never calibration-tick-derived, since their calibration entries are known placeholders (full-turn range / percent-mode range respectively) -- mirrors the same exclusion `scripts/calibration_utils.py` already made for Phase 10's URDF work
- Installed only `pytest` into a freshly-created `control/.venv` inside this worktree (see Deviations below) rather than the full `control/requirements.txt`, since none of this plan's new code imports `lerobot`/`opencv`/`ultralytics`

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Created a lightweight `control/.venv` with only `pytest` instead of the full `requirements.txt` install**
- **Found during:** Task 1 (test infrastructure setup)
- **Issue:** The plan's action step says to run `.venv/bin/pip install -r requirements.txt` from `control/` after adding `pytest==8.3.4`. This worktree (a fresh git worktree, isolated from the main repo checkout) had no pre-existing `control/.venv` -- `.venv/` is gitignored and worktrees only check out tracked files, so it does not carry over from the main checkout. Installing the full `requirements.txt` (`lerobot[feetech]==0.6.1`, `opencv-python`, `ultralytics` with its PyTorch dependency chain) would pull in several GB of heavy ML dependencies purely to satisfy a pip-install step, when neither `action_contract.py` nor `safety_validator.py` nor their tests import any of those packages (they use only stdlib `json`/`math`/`pathlib` plus `pytest`).
- **Fix:** Created a fresh `python3.12 -m venv control/.venv` in this worktree and installed only `pytest==8.3.4` into it -- sufficient to run and pass every test file this plan adds, matching the plan's acceptance criterion ("`pytest` is importable from `control/.venv`") without the multi-GB unrelated install.
- **Files modified:** None beyond the already-planned `control/requirements.txt` addition of `pytest==8.3.4` (the line was added as specified; only the *install target* was narrowed).
- **Verification:** `control/.venv/bin/python -m pytest test_action_contract.py test_safety_validator.py -x -q` -- 14 passed.
- **Committed in:** `486dc00` (Task 1 commit; `.venv/` itself is gitignored, not committed)

---

**Total deviations:** 1 auto-fixed (1 blocking, install-target scoping)
**Impact on plan:** No scope creep -- the full heavy `requirements.txt` install remains valid and expected once a real hardware-connected `control/.venv` is set up outside this ephemeral worktree; this plan's own tests only ever needed `pytest`.

## Issues Encountered
None beyond the venv-scoping deviation above.

## User Setup Required

None - no external service configuration required. (Note: a real `control/.venv` with the full `requirements.txt`, including `lerobot[feetech]`, will still be needed before any later Phase 11 plan that actually talks to the physical robot or loads a VLA checkpoint -- that install was intentionally not duplicated in this worktree's throwaway venv.)

## Next Phase Readiness
- `control.vla_bridge.action_contract.load_joint_limits_deg()` and `control.vla_bridge.safety_validator.validate_action()` are ready to be imported and wired into Plan 11-02's dry run / bridge work
- `MAX_RELATIVE_TARGET_DEG`/`MAX_VELOCITY_DEG_PER_S` are explicitly flagged `[ASSUMED]` conservative defaults (per RESEARCH.md Assumption A3) -- Plan 11-02's dry run should exercise and tune these before the first real VLA-driven run
- `control/conftest.py`'s `mock_robot`/`mock_calibration_file` fixtures are ready for reuse by `test_io_logger.py` and any other test file later Phase 11 plans add
- No blockers identified for proceeding to Plan 11-02

## Self-Check: PASSED

- FOUND: control/vla_bridge/__init__.py
- FOUND: control/vla_bridge/action_contract.py
- FOUND: control/vla_bridge/safety_validator.py
- FOUND: control/conftest.py
- FOUND: control/test_action_contract.py
- FOUND: control/test_safety_validator.py
- FOUND commit: 486dc00
- FOUND commit: aea1257

---
*Phase: 11-vla-hardware-connection*
*Completed: 2026-09-20*
