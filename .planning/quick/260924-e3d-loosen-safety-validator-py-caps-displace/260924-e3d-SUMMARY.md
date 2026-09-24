---
phase: quick-260924-e3d
plan: 01
subsystem: hardware-safety
tags: [safety-validator, vla-bridge, so-arm101, pytest]

# Dependency graph
requires:
  - phase: 11-vla-hardware-connection
    provides: safety_validator.py's original conservative caps (VLAHW-02), 11-05's paused live hardware episode
provides:
  - Raised MAX_RELATIVE_TARGET_DEG (arm 5.0->40.0, gripper 15.0->60.0)
  - Raised MAX_VELOCITY_DEG_PER_S (arm 30.0->240.0, gripper 50.0->200.0)
  - Raised STALE_OBSERVATION_S (1.0->10.0) and STALE_ACTION_S (3.0->30.0)
  - Rewritten constant-dependent tests referencing live constants instead of hardcoded numbers
affects: [11-vla-hardware-connection, vla-bridge, safety-validator]

# Tech tracking
tech-stack:
  added: []
  patterns: [tests importing module constants directly instead of hardcoding derived values]

key-files:
  created: []
  modified:
    - control/vla_bridge/safety_validator.py
    - control/test_safety_validator.py

key-decisions:
  - "Loosened per-step/velocity/staleness caps as a values-only change; left NaN/inf rejection, absolute joint-limit clamp, and validate_action() control flow completely untouched, per explicit plan constraint"
  - "Velocity cap scaled by the same multiplier as the per-step cap per joint group (arm 8x, gripper 4x) to preserve the original crossover-dt design while raising the whole envelope"
  - "Staleness thresholds raised to the top of the user's suggested ranges (10s observation, 30s action) since the test environment is physically empty and stale-rejection was the directly observed failure mode blocking 11-05 task 3/4"

requirements-completed: [VLAHW-02]

coverage:
  - id: D1
    description: "Per-step displacement cap raised so a single VLA action step is no longer clamped to ~5 deg (arm) / ~15pp (gripper) but instead allows up to 40 deg (arm) / 60pp (gripper) before further limiting applies"
    requirement: "VLAHW-02"
    verification:
      - kind: unit
        ref: "control/test_safety_validator.py#test_large_step_clamped_to_max_relative_target_default"
        status: pass
    human_judgment: false
  - id: D2
    description: "Velocity cap raised proportionally (arm 8x, gripper 4x) so it does not silently re-clamp what the raised per-step cap allows at the bridge's real dt_s=0.5s control loop"
    requirement: "VLAHW-02"
    verification:
      - kind: unit
        ref: "control/test_safety_validator.py#test_velocity_cap_clamps_further_given_short_dt"
        status: pass
    human_judgment: false
  - id: D3
    description: "STALE_OBSERVATION_S and STALE_ACTION_S raised to 10.0s / 30.0s so a real Colab PolicyServer round-trip is not discarded as stale"
    requirement: "VLAHW-02"
    verification:
      - kind: unit
        ref: "control/test_safety_validator.py#test_stale_observation_holds_all_joints"
        status: pass
    human_judgment: false
  - id: D4
    description: "Absolute joint-limit clamp (step 2) and NaN/inf rejection (step 1) in validate_action() remain completely untouched"
    requirement: "VLAHW-02"
    verification:
      - kind: unit
        ref: "control/test_safety_validator.py#test_out_of_range_value_clamped_to_joint_limit"
        status: pass
      - kind: unit
        ref: "control/test_safety_validator.py#test_nan_value_holds_current_state_and_does_not_propagate"
        status: pass
      - kind: unit
        ref: "control/test_safety_validator.py#test_infinite_value_holds_current_state_and_does_not_propagate"
        status: pass
    human_judgment: false
  - id: D5
    description: "Real hardware VLA episode against the Colab PolicyServer produces visibly large robot motion instead of holding position indefinitely (11-05 task 3/4 unblocked)"
    human_judgment: true
    rationale: "Requires a live hardware run against a real Colab PolicyServer round-trip, which is outside this quick task's automated scope; can only be confirmed by re-running 11-05's paused task 3/4 on real hardware."

# Metrics
duration: 8min
completed: 2026-09-24
status: complete
---

# Quick Task 260924-e3d: Loosen safety_validator.py caps Summary

**Raised safety_validator.py's per-step displacement, velocity, and staleness constants (values-only) so a real Colab PolicyServer round-trip produces visible robot motion instead of being clamped to a few degrees or discarded as stale.**

## Performance

- **Duration:** 8 min
- **Started:** 2026-09-24T04:47:00Z (approx)
- **Completed:** 2026-09-24T04:55:48Z
- **Tasks:** 2/2
- **Files modified:** 2

## Accomplishments
- `MAX_RELATIVE_TARGET_DEG` raised: arm joints 5.0->40.0 deg, gripper 15.0->60.0pp
- `MAX_VELOCITY_DEG_PER_S` raised proportionally: arm joints 30.0->240.0 deg/s, gripper 50.0->200.0pp/s
- `STALE_OBSERVATION_S` raised 1.0->10.0s; `STALE_ACTION_S` raised 3.0->30.0s
- 3 constant-dependent tests rewritten to derive expected values from live imported constants instead of hardcoded numbers, preventing this exact drift from recurring on future retunes
- Full 14-test suite (`test_safety_validator.py` + `test_action_contract.py`) passing

## Task Commits

Each task was committed atomically:

1. **Task 1: Raise safety_validator.py's per-step, velocity, and staleness constants** - `d8c2785` (feat)
2. **Task 2: Update the 3 constant-dependent tests to reference the live constants** - `7214339` (test)

_Note: docs/state metadata commit made separately by the orchestrator._

## Files Created/Modified
- `control/vla_bridge/safety_validator.py` - Raised the four module-level safety constants (values-only; NaN/inf rejection, absolute joint-limit clamp, and control flow untouched)
- `control/test_safety_validator.py` - Rewrote 3 tests to import and reference `MAX_RELATIVE_TARGET_DEG`/`MAX_VELOCITY_DEG_PER_S`/`STALE_OBSERVATION_S` directly instead of hardcoded numbers derived from the old constant values

## Decisions Made
- Followed the plan's pre-derived values exactly: arm 40.0/gripper 60.0 (per-step), arm 240.0/gripper 200.0 (velocity, same multiplier as per-step per group), 10.0s (observation staleness), 30.0s (action staleness)
- No architectural changes; this was a scoped values-only constant edit plus corresponding test updates, as specified

## Deviations from Plan

None - plan executed exactly as written. Only the two files listed in the plan's `files_modified` were touched (`git diff --stat` against the pre-dispatch commit confirms exactly `control/vla_bridge/safety_validator.py` and `control/test_safety_validator.py`, no other file).

## Issues Encountered
- This worktree had no `control/.venv` (gitignored, not present in the freshly created worktree). Ran verification using the main repo's `control/.venv/bin/python` interpreter with `cwd` set to this worktree's `control/` directory and `PYTHONPATH=.`, so imports resolved against this worktree's modified source files while reusing the already-installed interpreter/dependencies. This is a test-execution detail only; no repo files were changed to work around it.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- `safety_validator.py`'s caps are now loosened per the user's explicit request; the absolute joint-limit clamp and NaN/inf rejection remain the final backstop
- 11-05's paused task 3/4 can now be retried on real hardware against the Colab PolicyServer to confirm visible robot motion (D5 above requires that live run to close out — not verifiable from this quick task alone)

---
*Phase: quick-260924-e3d*
*Completed: 2026-09-24*

## Self-Check: PASSED

- FOUND: control/vla_bridge/safety_validator.py
- FOUND: control/test_safety_validator.py
- FOUND: .planning/quick/260924-e3d-loosen-safety-validator-py-caps-displace/260924-e3d-SUMMARY.md
- FOUND commit: d8c2785 (feat: raise safety_validator constants)
- FOUND commit: 7214339 (test: rewrite constant-dependent tests)
