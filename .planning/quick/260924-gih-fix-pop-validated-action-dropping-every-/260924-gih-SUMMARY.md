---
phase: quick-260924-gih
plan: 01
subsystem: control
tags: [lerobot, safety-validator, robot-client, vla-bridge, pytest]

# Dependency graph
requires:
  - phase: 11-vla-hardware-connection
    provides: pop_validated_action()/safety_validator.validate_action() bridge-to-safety-validation seam (11-05)
provides:
  - "pop_validated_action() strips .pos suffixes from raw_action keys before safety validation"
  - "Regression test proving a real bridge-shaped .pos-suffixed action resolves to a non-empty, correctly-valued validated_action"
affects: [11-vla-hardware-connection, control/vla_bridge, live hardware episode retest]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Adapter-boundary key normalization: lerobot wire-format (.pos-suffixed) converted to this project's plain action_contract.JOINT_ORDER convention at the single point where the two meet (pop_validated_action()), not propagated outward"

key-files:
  created: []
  modified:
    - control/vla_bridge/robot_client.py
    - control/test_robot_client.py

key-decisions:
  - "Normalize at the pop_validated_action() adapter boundary only -- safety_validator.py's plain-name convention is left unchanged, per the plan's explicit design decision"
  - "The 3rd tuple element (raw_action, used for episode.jsonl's raw_model_output) returns the same normalized, plain-keyed dict -- not the original .pos-suffixed one -- for internal consistency with validated_action/current_state/prev_action"

patterns-established:
  - "str.removesuffix('.pos') as a no-op-safe normalization: identical behavior for already-plain-keyed callers/fixtures, since removesuffix on an absent suffix returns the string unchanged"

requirements-completed: [VLAHW-02]

coverage:
  - id: D1
    description: "pop_validated_action() normalizes .pos-suffixed raw_action keys to plain action_contract.JOINT_ORDER names before calling safety_validate_fn, and returns the same normalized dict as the 4-tuple's 3rd element"
    requirement: "VLAHW-02"
    verification:
      - kind: unit
        ref: "control/test_robot_client.py#test_pop_validated_action_normalizes_pos_suffixed_keys_from_real_bridge"
        status: pass
      - kind: unit
        ref: "control/test_robot_client.py (11 pre-existing pop_validated_action/BridgeActionSource/connect_bridge tests, unchanged behavior)"
        status: pass
    human_judgment: false
  - id: D2
    description: "Full control/ test suite passes at 93 tests (92 pre-existing + 1 new), confirming no regression to any other bridge/safety-validator/robot-client behavior"
    verification:
      - kind: unit
        ref: "cd control && .venv/bin/python -m pytest -q"
        status: pass
    human_judgment: false

duration: 12min
completed: 2026-09-24
status: complete
---

# Quick Task 260924-gih: Fix pop_validated_action() dropping every real VLA action Summary

**`pop_validated_action()` now strips lerobot's `.pos`-suffixed action keys to plain `action_contract.JOINT_ORDER` names before `safety_validator.validate_action()` runs, fixing a silent empty-dict failure that crashed the last live episode at step 1.**

## Performance

- **Duration:** ~12 min
- **Completed:** 2026-09-24T06:31:30Z
- **Tasks:** 2/2 completed
- **Files modified:** 2

## Accomplishments
- `pop_validated_action()` (`control/vla_bridge/robot_client.py`) normalizes `.pos`-suffixed keys (lerobot's real `SO101Follower.action_features` convention) to plain joint names immediately after `_action_tensor_to_action_dict()`, before the result reaches `safety_validate_fn` -- so every joint's membership check in `safety_validator.validate_action()` now succeeds instead of silently skipping.
- The function's 3rd return value (`raw_action`, used for `episode.jsonl`'s `raw_model_output` field) is the same normalized, plain-keyed dict -- consistent with `validated_action`/`current_state`/`prev_action`, which were already plain-keyed everywhere else.
- New regression test `test_pop_validated_action_normalizes_pos_suffixed_keys_from_real_bridge` reproduces the exact real-world failure mode (a `.pos`-suffixed action dict flowing through `pop_validated_action()`) and asserts a non-empty, correctly-valued, plain-keyed `validated_action`.
- Full `control/` test suite: 93 passed (92 pre-existing + 1 new), 0 failed.

## Task Commits

Each task was committed atomically:

1. **Task 1: Normalize .pos-suffixed keys in pop_validated_action() before safety validation** - `a3e16c4` (fix)
2. **Task 2: Add regression test for the .pos-suffixed key mismatch, verify full suite** - `8e2060a` (test)

**Plan metadata:** committed separately by the orchestrator after this summary.

## Files Created/Modified
- `control/vla_bridge/robot_client.py` - `pop_validated_action()` now strips `.pos` suffixes from `raw_action`'s keys via `str.removesuffix(".pos")` immediately after `_action_tensor_to_action_dict()`, before validation; docstring updated to document the normalization and the crash it fixes.
- `control/test_robot_client.py` - New regression test asserting a `.pos`-suffixed bridge action resolves to a non-empty, plain-keyed, correctly-valued `validated_action` with `flags == []`.

## Decisions Made
- Normalize at the `pop_validated_action()` adapter boundary only -- `safety_validator.py`'s plain-name convention stays unchanged (already decided in the plan's context, followed as specified).
- The normalized dict is used for both the value passed to `safety_validate_fn` and the function's 3rd return value, avoiding a second parallel variable for the pre-normalization form (no remaining consumer needed the original `.pos`-suffixed shape).

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None. The worktree lacked its own `control/.venv`; tests were run using the shared repo's existing `control/.venv/bin/python` (read-only interpreter use, no modification to the venv itself).

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- The root-cause bug blocking 11-05's live-hardware retry (empty `validated_action` from a real bridge-returned action) is fixed and covered by a regression test.
- Full `control/` suite green (93/93). Ready to retry the live SmolVLA episode per Phase 11's outstanding work (`11-05` paused at task 3/4, awaiting retest per STATE.md).
- No other files touched (`git diff --stat` across both commits confirms only `control/vla_bridge/robot_client.py` and `control/test_robot_client.py`).

---
*Phase: quick-260924-gih*
*Completed: 2026-09-24*

## Self-Check: PASSED

- FOUND: control/vla_bridge/robot_client.py
- FOUND: control/test_robot_client.py
- FOUND: .planning/quick/260924-gih-fix-pop-validated-action-dropping-every-/260924-gih-SUMMARY.md
- FOUND: commit a3e16c4
- FOUND: commit 8e2060a
