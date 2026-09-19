---
phase: quick-260919-h8v
plan: 01
subsystem: robot-sim
tags: [mujoco, urdf, gripper, so-arm101, libero, digital-twin]

requires:
  - phase: 10-digital-twin-fidelity
    provides: TWIN-01..06 (URDF/MJCF kinematic chain fix), TWIN-07 hardware-mismatch diagnosis (10-05)
provides:
  - Sim gripper axis/range/ctrlrange polarity flipped to match real SO-ARM101 hardware direction
  - Regenerated So-101.urdf with corrected gripper joint axes
  - Updated TWIN-01..05 regression test expectations
  - Documented (not resolved) TWIN-07 status: fix applied, hardware re-test still required
affects: [phase-10-digital-twin-fidelity, phase-11-vla-hardware-connection, future-phase-8-9-demo-recollection]

tech-stack:
  added: []
  patterns: ["MJCF slide-joint polarity flip via paired axis+range negation (not axis negation alone)"]

key-files:
  created: []
  modified:
    - LIBERO/libero/libero/assets/grippers/soarm_gripper.xml
    - LIBERO/libero/libero/envs/grippers/soarm_gripper.py
    - So-101/So-101.urdf
    - scripts/mjcf_to_urdf.py
    - scripts/test_verify_urdf.py
    - scripts/verify_urdf.py
    - .planning/phases/10-digital-twin-fidelity/10-05-TWIN-07-CHECK.md
    - .planning/phases/10-digital-twin-fidelity/10-VERIFICATION.md
    - .planning/STATE.md

key-decisions:
  - "Flipped axis sign AND negated+reordered range/ctrlrange together (not axis alone) to preserve the exact same physical 84mm stroke while reversing which command sign opens/closes the jaws"
  - "TWIN-07 intentionally left FAIL/gaps_found in both verification docs — this plan's fix is sim-only and cannot itself prove hardware agreement; a fresh human-in-the-loop re-test is still required"
  - "Documented collector.py's OPEN_CMD/CLOSE_CMD polarity regression risk in STATE.md rather than fixing it (out of scope, Phase 8/9 currently paused)"

patterns-established:
  - "When reversing an MJCF slide-joint's command polarity without moving geometry, negate the axis AND swap+negate the range bounds together — axis negation alone would drive the joint past its geometric neutral point into an invalid crossing state"

requirements-completed: [TWIN-07]

coverage:
  - id: D1
    description: "gripper_left/gripper_right MJCF joint axis+range+ctrlrange flipped so sim command polarity matches real hardware direction"
    requirement: "TWIN-07"
    verification:
      - kind: unit
        ref: "MuJoCo empirical load-and-step check: closed_gap=0.018m, open_gap=0.102m, monotonic non-crossing separation"
        status: pass
    human_judgment: false
  - id: D2
    description: "So-101.urdf regenerated with corrected axes; TWIN-01..05 + env-reset regression suite updated and passing"
    requirement: "TWIN-01"
    verification:
      - kind: unit
        ref: "scripts/test_verify_urdf.py + LIBERO/libero/libero/envs/test_camera_config.py (7 tests)"
        status: pass
      - kind: other
        ref: "python3 scripts/verify_urdf.py (standalone CLI, all checkmarks, no warnings)"
        status: pass
    human_judgment: false
  - id: D3
    description: "TWIN-07 hardware verification (real-vs-sim gripper direction agreement)"
    verification: []
    human_judgment: true
    rationale: "This plan's fix is sim-only and self-consistency-checked (MuJoCo + pytest); it cannot itself prove real-hardware agreement. A new human-in-the-loop joint_jog.py comparison against the physical SO-ARM101 is required and explicitly out of this plan's scope — both verification docs state this."

duration: 45min
completed: 2026-09-19
status: complete
---

# Quick Task 260919-h8v: Flip Sim Gripper Axis Polarity to Match Real Hardware Summary

**Flipped `gripper_left`/`gripper_right` MJCF joint axis+range+ctrlrange polarity in `soarm_gripper.xml` (paired axis-negation + range-swap, not axis alone) so the sim's command-to-direction convention matches the real SO-ARM101's positive-command-opens behavior found by TWIN-07's hardware test — regenerated `So-101.urdf`, updated the TWIN-01..05 regression suite, and documented the fix in both verification docs without prematurely marking TWIN-07 resolved.**

## Performance

- **Duration:** ~45 min
- **Started:** 2026-09-19T09:31:00Z (approx, from base commit)
- **Completed:** 2026-09-19T10:16:14Z
- **Tasks:** 3 planned (+ 1 inline bug-fix commit)
- **Files modified:** 9

## Accomplishments
- Reversed the sim gripper's command polarity (real robot: positive command opens; sim previously: `+1` = closed) by negating+reordering `gripper_left`/`gripper_right`'s MJCF `axis`, `range`, and actuator `ctrlrange`, preserving the exact same 84mm physical stroke and mirrored, non-crossing jaw motion (empirically verified via MuJoCo: closed gap 0.018m, open gap 0.102m)
- Regenerated `So-101/So-101.urdf` from the corrected MJCF via `scripts/mjcf_to_urdf.py` (not hand-edited); diff is a minimal, expected 8-line axis/limit change
- Updated `scripts/test_verify_urdf.py` and `scripts/verify_urdf.py`'s hardcoded expected axis literals to match; full TWIN-01..05 + env-reset suite (7 tests) passes with zero regressions
- Documented the code-side fix as an addendum in both `10-05-TWIN-07-CHECK.md` and `10-VERIFICATION.md` without flipping FAIL/gaps_found status — both docs now explicitly require a new human-in-the-loop hardware re-test before TWIN-07/Phase 10 can close
- Logged the newly-discovered `collector.py` `OPEN_CMD`/`CLOSE_CMD` polarity regression risk as a new STATE.md Blocker/Concern (Phase 8/9 demo-recollection FSM, currently paused, will need its constants swapped before reuse)

## Task Commits

1. **Task 1: Flip gripper_left/gripper_right axis polarity in soarm_gripper.xml + companion soarm_gripper.py update** - `5fa8e62` (fix)
2. **Bug-fix (Rule 1, discovered mid-Task-2): remove invalid `--` from XML comments introduced in Task 1** - `fbbf5fb` (fix)
3. **Task 2: Regenerate URDF, update hardcoded axis expectations, confirm no regressions** - `7bf580a` (fix)
4. **Task 3: Document the code-side fix and the still-required hardware re-test; log the collector.py concern** - `aefbb8d` (docs)

## Files Created/Modified
- `LIBERO/libero/libero/assets/grippers/soarm_gripper.xml` - Flipped gripper_left/gripper_right joint axis, range, and actuator ctrlrange (paired negation+reorder); updated descriptive comments
- `LIBERO/libero/libero/envs/grippers/soarm_gripper.py` - `init_qpos` updated to `[-0.042, -0.042]` (new fully-open value); comments corrected; `format_action` arithmetic unchanged
- `So-101/So-101.urdf` - Regenerated via `scripts/mjcf_to_urdf.py`; gripper joint axes/limits now match the corrected MJCF
- `scripts/mjcf_to_urdf.py` - Added `from __future__ import annotations` (Rule 3 blocking fix: script used Python 3.10+ `X | None` type-hint syntax, incompatible with the libero conda env's Python 3.9)
- `scripts/test_verify_urdf.py` - Updated `test_gripper_joint_axes`'s expected axis literals to `[0, 1, 0]` / `[0, -1, 0]`
- `scripts/verify_urdf.py` - Updated the standalone CLI's `axes_ok` check to the same new literals
- `.planning/phases/10-digital-twin-fidelity/10-05-TWIN-07-CHECK.md` - Appended "Gap-Closure Fix Applied" section documenting the fix; original FAIL outcome untouched
- `.planning/phases/10-digital-twin-fidelity/10-VERIFICATION.md` - Added an update note referencing this gap-closure plan; `gaps_found` status and TWIN-07 row untouched
- `.planning/STATE.md` - Added new Blockers/Concerns entry for `collector.py`'s OPEN_CMD/CLOSE_CMD polarity regression risk

## Decisions Made
- **Paired axis+range negation, not axis-sign negation alone:** A bare axis flip with the range left at `0 0.042` would have driven the jaws past each other into an invalid crossing state. Negating the axis AND swapping+negating the range to `-0.042 0` preserves qpos=0 as the same physical touching-closed position while making qpos=-0.042 the same physical 84mm-apart open position — same geometry, reversed command mapping. Confirmed empirically, not just by derivation.
- **TWIN-07 intentionally left unresolved:** Both verification docs (`10-05-TWIN-07-CHECK.md`, `10-VERIFICATION.md`) keep their FAIL/gaps_found status. This plan's fix is sim-only; only a fresh human-in-the-loop hardware re-test can actually close TWIN-07.
- **`collector.py` left untouched, risk documented instead of fixed:** Its `OPEN_CMD`/`CLOSE_CMD` constants now drive the physical opposite of their names post-fix, but the FSM that uses them (Phase 8/9 demo re-collection) is currently paused and out of this plan's scope — logged in STATE.md so it isn't silently rediscovered later.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Added `from __future__ import annotations` to `scripts/mjcf_to_urdf.py`**
- **Found during:** Task 2 (regenerating the URDF)
- **Issue:** The script uses Python 3.10+ `X | None` union type-hint syntax (`def find_body(...) -> ET.Element | None`), which raises `TypeError: unsupported operand type(s) for |: 'type' and 'NoneType'` at import time under the project's `libero` conda env (Python 3.9.25). This is pre-existing script/environment incompatibility, unrelated to the gripper fix, but blocked Task 2's mandatory regeneration step.
- **Fix:** Added `from __future__ import annotations` at the top of the file, deferring all annotation evaluation to strings (PEP 563) — zero behavior change, standard fix for this exact incompatibility.
- **Files modified:** scripts/mjcf_to_urdf.py
- **Verification:** `conda run -n libero python3 scripts/mjcf_to_urdf.py` now runs to completion and regenerates the URDF correctly.
- **Committed in:** `7bf580a` (Task 2 commit)

**2. [Rule 1 - Bug] Fixed invalid `--` inside XML comments introduced by Task 1's own edits**
- **Found during:** Task 2 (regenerating the URDF), when `scripts/mjcf_to_urdf.py`'s stdlib `xml.etree.ElementTree.parse()` raised `ParseError: not well-formed (invalid token)` at the exact line of a comment Task 1 had just written
- **Issue:** XML comments cannot contain the literal sequence `--` anywhere in their body (only at the very end, immediately before `-->`). Three of Task 1's new descriptive comments in `soarm_gripper.xml` used `--` as an em-dash-style separator (e.g., "is now OPEN -- the reverse of..."), which MuJoCo's own (lenient) XML loader accepted silently but the standards-compliant `ElementTree` parser rejected.
- **Fix:** Replaced each `--` with a comma or removed it, preserving the comment's meaning.
- **Files modified:** LIBERO/libero/libero/assets/grippers/soarm_gripper.xml
- **Verification:** Re-ran Task 1's MuJoCo empirical jaw-gap check (unchanged result: closed_gap=0.018m, open_gap=0.102m) to confirm the comment-only edit didn't affect physics; `scripts/mjcf_to_urdf.py` then parsed the file successfully.
- **Committed in:** `fbbf5fb` (standalone fix commit, ahead of Task 2)

---

**Total deviations:** 2 auto-fixed (1 blocking Python-version fix, 1 blocking self-introduced XML-syntax bug)
**Impact on plan:** Both fixes were necessary to complete Task 2's mandatory URDF regeneration step; neither expanded scope beyond what the plan already required. No scope creep.

## Issues Encountered
None beyond the two auto-fixed blocking issues documented above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- The sim-side TWIN-07 fix is complete and regression-tested; Phase 10 still cannot be marked complete.
- **Required next step (out of this plan's scope):** run a fresh human-in-the-loop hardware re-test (repeating the `joint_jog.py ... gripper` real-vs-sim comparison from `10-05-TWIN-07-CHECK.md`) and obtain an explicit PASS before TWIN-07/Phase 10 can close.
- New blocker logged for future Phase 8/9 demo re-collection work: `collector.py`'s `OPEN_CMD`/`CLOSE_CMD` constants need to be swapped before that FSM is used again.

---
*Phase: quick-260919-h8v*
*Completed: 2026-09-19*

## Self-Check: PASSED

All 9 claimed modified files confirmed present on disk (`LIBERO/libero/libero/assets/grippers/soarm_gripper.xml`, `LIBERO/libero/libero/envs/grippers/soarm_gripper.py`, `So-101/So-101.urdf`, `scripts/mjcf_to_urdf.py`, `scripts/test_verify_urdf.py`, `scripts/verify_urdf.py`, `.planning/phases/10-digital-twin-fidelity/10-05-TWIN-07-CHECK.md`, `.planning/phases/10-digital-twin-fidelity/10-VERIFICATION.md`, `.planning/STATE.md`). All 4 task commit hashes (`5fa8e62`, `fbbf5fb`, `7bf580a`, `aefbb8d`) confirmed present in `git log`.
