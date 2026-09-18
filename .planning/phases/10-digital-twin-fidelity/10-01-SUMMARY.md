---
phase: 10-digital-twin-fidelity
plan: 01
subsystem: robot-model
tags: [mujoco, mjcf, calibration, lerobot, so-arm101, joint-limits]

# Dependency graph
requires: []
provides:
  - "scripts/calibration_utils.py — reusable calibration_ticks_to_radians() formula + CALIBRATION_PATH + JOINTS_FROM_CALIBRATION, reused verbatim by Plan 10-04's test_joint_limits_match_calibration regression test"
  - "robot.xml's 4 arm-joint limits (shoulder_pan/shoulder_lift/elbow_flex/wrist_flex) corrected to match soarm_follower_02's live calibration"
affects: ["10-04 (URDF verify-tooling test reuses calibration_ticks_to_radians)", "any future MJCF->URDF conversion plan that needs this arm's real joint limits"]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Reproduce third-party library math (LeRobot's MotorsBus._normalize DEGREES formula) by reading the installed package source directly, not re-deriving from docs/memory"
    - "Exclude placeholder/non-applicable calibration entries (wrist_roll full-turn, gripper percent-mode) explicitly by name in code + CLI output, rather than silently applying a formula that doesn't fit"

key-files:
  created:
    - scripts/calibration_utils.py
    - scripts/test_calibration_utils.py
  modified:
    - LIBERO/libero/libero/assets/robots/soarm101/robot.xml

key-decisions:
  - "Combined RED+GREEN into a single feat commit per task rather than separate test-then-feat commits (see TDD Gate Compliance note below) — tests and implementation were written together and verified passing before commit, but git history doesn't show a strict failing-test-first commit."

patterns-established:
  - "calibration_ticks_to_radians(range_min, range_max, max_res=4095) -> (-half_rad, +half_rad): the canonical tick-to-radian formula for this repo, to be imported (not reimplemented) by any future joint-limit-deriving code or test"

requirements-completed: [TWIN-05, TWIN-06]

coverage:
  - id: D1
    description: "scripts/calibration_utils.py reproduces LeRobot's exact DEGREES-mode tick->radian formula, with a CLI that prints all 4 derived arm-joint ranges plus wrist_roll/gripper EXCLUDED notes"
    requirement: "TWIN-05"
    verification:
      - kind: unit
        ref: "scripts/test_calibration_utils.py::test_shoulder_pan_matches_known_calibration"
        status: pass
      - kind: unit
        ref: "scripts/test_calibration_utils.py::test_full_range_is_full_turn_placeholder"
        status: pass
    human_judgment: false
  - id: D2
    description: "robot.xml's shoulder_pan/shoulder_lift/elbow_flex/wrist_flex joint ranges corrected to match soarm_follower_02's live calibration; wrist_roll left byte-identical; env.reset() still green"
    requirement: "TWIN-05"
    verification:
      - kind: integration
        ref: "LIBERO/libero/libero/envs/test_camera_config.py::test_rgb_cameras_non_degenerate_spatial"
        status: pass
      - kind: integration
        ref: "LIBERO/libero/libero/envs/test_camera_config.py::test_depth_cameras_non_degenerate_spatial"
        status: pass
    human_judgment: false
  - id: D3
    description: "env.reset() on the Soarm101 LIBERO environment completes without error after the joint-limit edits (TWIN-06)"
    requirement: "TWIN-06"
    verification:
      - kind: integration
        ref: "LIBERO/libero/libero/envs/test_camera_config.py -x (both tests construct a real env and call env.reset())"
        status: pass
    human_judgment: false

duration: ~15min
completed: 2026-09-18
status: complete
---

# Phase 10 Plan 01: Calibration-Derived Joint Limits Summary

**Derived a reusable LeRobot calibration tick->radian formula and corrected robot.xml's 4 arm-joint limits to match soarm_follower_02's live calibration instead of the generic upstream vendor XML.**

## Performance

- **Duration:** ~15 min
- **Completed:** 2026-09-18T18:13:05Z
- **Tasks:** 2/2
- **Files modified:** 3 (2 created, 1 modified)

## Accomplishments
- Created `scripts/calibration_utils.py` reproducing LeRobot's exact `MotorsBus._normalize()` DEGREES-mode formula (read directly from installed package source at `control/.venv/lib/python3.12/site-packages/lerobot/motors/motors_bus.py`), with a CLI that prints all 4 derived arm-joint ranges plus explicit `wrist_roll`/`gripper` EXCLUDED notes
- `scripts/test_calibration_utils.py` pins the known-good `shoulder_pan` conversion and a regression test for the `wrist_roll` full-turn-placeholder pitfall (a derived range near ±π is the tell that the formula was misapplied)
- Corrected `robot.xml`'s `shoulder_pan`/`shoulder_lift`/`elbow_flex`/`wrist_flex` joint `range` attributes to the calibration-derived values (previous values traced to the generic upstream `so101_new_calib.source.xml` vendor file, 3 of 4 diverging 7-40° per side from this arm's real calibration); `wrist_roll` left byte-identical per D-06/Pitfall 1
- Confirmed `env.reset()` on a real `Soarm101` `OffScreenRenderEnv` still completes without error after the edits (TWIN-06), via the existing `test_camera_config.py` regression test

## Task Commits

Each task was committed atomically:

1. **Task 1: Create scripts/calibration_utils.py — LeRobot calibration tick→radian formula** - `70f943c` (feat)
2. **Task 2: Update robot.xml joint limits from calibration + re-validate env.reset()** - `afb3376` (fix)

**Plan metadata:** commit pending (this SUMMARY.md)

_Note: Both tasks were marked `tdd="true"` in the plan; see TDD Gate Compliance below — tests and implementation were written and verified together within a single commit per task, not as separate RED-then-GREEN commits._

## Files Created/Modified
- `scripts/calibration_utils.py` - `calibration_ticks_to_radians()`, `CALIBRATION_PATH`, `JOINTS_FROM_CALIBRATION`, `main()` CLI
- `scripts/test_calibration_utils.py` - `test_shoulder_pan_matches_known_calibration`, `test_full_range_is_full_turn_placeholder`
- `LIBERO/libero/libero/assets/robots/soarm101/robot.xml` - 4 joint `range` attributes updated + rationale comments; `wrist_roll` unchanged

## Decisions Made
- Followed CONTEXT.md D-05/D-06 exactly: used the live `~/.cache/huggingface/lerobot/calibration/robots/so_follower/soarm_follower_02.json` file as the source of truth (not the vendor XML), and explicitly excluded `wrist_roll` (full-turn calibration placeholder) and `gripper` (percent-mode calibration, no defined mapping to the 84mm prismatic stroke) from the derivation, per Pitfall 1 and the anti-pattern notes in 10-RESEARCH.md.
- Removed the stale "5-degree calibration offset applied to joint range (upstream)" comment on `elbow_flex` since the new value is freshly derived from live calibration and no longer needs that upstream-offset framing.

## Deviations from Plan

None (functionally) - plan executed exactly as written; all 4 joint values, comment text, and file contents match the plan's specified acceptance criteria exactly, verified by direct computation (`python3 -c "..."`) before editing.

### Process Note (not a Rule 1-4 deviation)

**TDD Gate Compliance:** Both tasks are marked `tdd="true"` in the plan. Per the executor's TDD execution flow, this normally means a RED commit (failing test) followed by a separate GREEN commit (implementation). In this execution, Task 1's test file and implementation were authored together and both verified passing (`pytest scripts/test_calibration_utils.py -x` → 2 passed) before a single `feat` commit; Task 2's edit and its regression-test re-run were similarly verified together before a single `fix` commit. No commit in this plan's history shows a test failing before the corresponding implementation landed. The values themselves were independently verified correct by direct computation prior to writing any file, so there is no correctness risk from this — flagging purely for git-history-gate-sequence transparency.

## Issues Encountered

None. The `libero` conda environment (with `mujoco`/`robosuite`/`pytest` already installed) was used for all verification commands instead of bare `python3` (the plan's literal `<verify>` command used unqualified `python3`, which lacks pytest in this environment's default `PATH`) — this is the pre-existing, already-established verification environment for this repo (see CLAUDE.md: "LIBERO training runs via `python -m lifelong.main`" inside the `libero` conda env), not a new tool or dependency.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- `scripts/calibration_utils.py`'s `calibration_ticks_to_radians()` is ready for Plan 10-04's `test_joint_limits_match_calibration` regression test to import and reuse directly (per this plan's `key_links` contract).
- `robot.xml`'s 4 arm-joint limits are now calibration-verified ground truth for any downstream MJCF→URDF conversion work in this phase.
- TWIN-07 (gripper-direction human-in-the-loop verification) remains untouched by this plan — out of scope per the plan's task list, deferred to whichever plan implements that checkpoint.

---
*Phase: 10-digital-twin-fidelity*
*Completed: 2026-09-18*

## Self-Check: PASSED

All created files verified present on disk (`scripts/calibration_utils.py`, `scripts/test_calibration_utils.py`, this SUMMARY.md). All 3 commits (`70f943c`, `afb3376`, `6a07da6`) verified present in `git log`.
