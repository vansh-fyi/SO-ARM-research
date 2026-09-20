
> **2026-09-20 established-model update:** [10-ESTABLISHED-MODEL.md](10-ESTABLISHED-MODEL.md)
> is authoritative for the accepted Coppelia-aligned assembly, black housing,
> yellow jaws, 0.036 m per-jaw cap, corrected starting pose and LIBERO wiring.
> Earlier settings and verification results below are historical and do not
> establish policy success for the accepted model.

# TWIN-07 Verification: Real vs. Sim Gripper Direction

**Date:** 2026-09-19
**Requirement:** TWIN-07 (Phase 10 — Digital-Twin Fidelity)
**Outcome: FAIL**

## Context

Per CONTEXT.md D-07, TWIN-07 ("driving the simulated gripper with a given joint
command opens/closes it in the same direction as the real gripper under the
identical command") cannot be verified from sim state alone — it requires a
human-in-the-loop side-by-side comparison against the physical SO-ARM101
follower arm. Per D-08, the comparison uses LeRobot's native gripper command
value, translated into the sim's `-1` (open) / `+1` (closed) convention as
documented by `SoarmGripper.format_action`'s EXTERNAL `action` argument (not
the internal `current_action` sign — see Pitfall 2 in 10-05-PLAN.md).

This checkpoint was presented directly to the human by the orchestrator (not a
subagent), because it required physically operating the real robot hardware,
which no agent process can access. The human ran the real-robot test and
reported the result across two rounds of confirmation in-session.

## Real Robot Test

**Command run:**
```
python control/joint_jog.py /dev/cu.usbmodem5B8E1139151 soarm_follower_02 gripper 20
```

This is a positive-delta jog command (`+20`) on the `gripper` joint of
`soarm_follower_02`.

**Observed servo telemetry:** `gripper.pos` moved from **61.72 → 71.64**
(an increase).

**Human's direct visual observation:** this positive-delta command **opened**
the real gripper's jaws (they moved apart).

## Sim Convention (from code)

Source: `LIBERO/libero/libero/envs/grippers/soarm_gripper.py`,
`SoarmGripper.format_action` docstring (EXTERNAL action-sign convention,
not `current_action`'s internal sign):

> `-1 => open, +1 => closed.`

I.e., in sim, a **positive** action value drives the gripper **closed**.

## Comparison

| Side | Direction of positive/increasing command |
|------|---------------------------------------------|
| Real robot (`joint_jog.py ... gripper 20`, pos 61.72 → 71.64) | **Opens** the jaws |
| Sim (`SoarmGripper.format_action`, external convention) | **Closes** the jaws (`+1` = closed) |

The two conventions disagree: the real robot's positive/increasing gripper
command opens the jaws, while the sim's positive (`+1`) action closes them.

## Outcome: FAIL

**FAIL: real robot's positive gripper delta (61.72 → 71.64, `joint_jog.py ...
gripper 20`) opened the jaws, but the sim's `+1` (close) convention closes the
jaws under a positive command — directions disagree.**

This was explicitly confirmed by the human via two rounds of confirmation in
this session: the initial report of the observation, and an explicit
re-confirmation question asking to verify the interpretation.

## Status and Required Next Step

**TWIN-07 remains UNRESOLVED.** Phase 10 **CANNOT** be marked complete while
this outcome stands — TWIN-07 is not satisfied in the phase's requirements
traceability.

**Required next step:** run `/gsd-plan-phase 10 --gaps` to generate a
gap-closure plan that diagnoses and fixes the direction mismatch. The likely
fix is to flip the `gripper_left`/`gripper_right` axis sign in
`LIBERO/libero/libero/assets/grippers/soarm_gripper.xml`, regenerate the URDF
via Plan 10-03's `scripts/mjcf_to_urdf.py` converter, and re-run Plan 10-04's
`test_gripper_joint_axes` verification with updated expected axis values to
match the corrected convention. Only after a re-run of this TWIN-07 check
passes can `/gsd-verify-work` consider TWIN-07 satisfied and Phase 10 be
closed.

This plan (10-05) only detects and documents the mismatch — it does not
contain a remediation task.

## Gap-Closure Fix Applied

**Date:** 2026-09-19
**Applied by:** Gap-closure quick task `260919-h8v`

The `gripper_left`/`gripper_right` joint `axis`, `range`, and `<position>`
actuator `ctrlrange` in `LIBERO/libero/libero/assets/grippers/soarm_gripper.xml`
were flipped: `gripper_left` axis `0 -1 0` → `0 1 0`, `gripper_right` axis
`0 1 0` → `0 -1 0`, both `range`/`ctrlrange` `0 0.042` → `-0.042 0`. This
reverses which command sign (`current_action`/external action) opens vs.
closes the jaws in sim, without moving any geom/body position (same
touching-closed and 84mm-apart-open physical geometry, re-labeled). The
paired `SoarmGripper.init_qpos` in `soarm_gripper.py` was updated to
`[-0.042, -0.042]` (the new fully-open value) and its descriptive comments
corrected; `format_action`'s arithmetic is unchanged.

`So-101/So-101.urdf` was regenerated (not hand-edited) via
`scripts/mjcf_to_urdf.py`, and `scripts/test_verify_urdf.py` /
`scripts/verify_urdf.py`'s hardcoded expected axis values were updated to
match. The full TWIN-01..05 + env-reset regression suite (7 tests) passes
with zero regressions, and an empirical MuJoCo load-and-step check confirms
the jaws still separate monotonically and symmetrically across the same
84mm physical stroke (closed gap ~0.018m, open gap ~0.102m).

**TWIN-07 REMAINS UNVERIFIED AGAINST REAL HARDWARE.** This fix is sim-only
and self-consistency-checked (MuJoCo + pytest); it does NOT constitute a
hardware re-test. **A NEW human-in-the-loop re-test — repeating the
`joint_jog.py ... gripper` real-vs-sim comparison this file originally
documented as FAIL — MUST be run and produce an explicit PASS before
TWIN-07 or Phase 10 can be considered closed.** The "Outcome: FAIL" header
above stays unchanged until that re-test passes.

## Hardware Re-Test: PASS

**Date:** 2026-09-19
**Re-test performed by:** the user, directly with the orchestrator (immediately after gap-closure quick task `260919-h8v` landed)

**Real robot test (repeated, identical command to the original FAIL test):**
```
python control/joint_jog.py /dev/cu.usbmodem5B8E1139151 soarm_follower_02 gripper 20
```
`gripper.pos` moved from 71.61 → 81.55 (a positive/increasing delta, clamped by
`max_relative_target=10.0`). Human's direct observation: **opens** the real
gripper's jaws — identical direction to the original test (expected: the real
robot's servo/calibration behavior is entirely unaffected by this sim-only fix).

**Sim verification (empirical, MuJoCo):** after the gap-closure fix, driving
`SoarmGripper.format_action` with external action `+1` repeatedly converges
`current_action` toward `-1`, which maps (via the corrected `ctrlrange -0.042 0`)
to a jaw center-to-center gap of **0.102m** (open) — versus **0.018m** (closed)
at `current_action=+1`. So external action `+1` (the "positive" direction) now
**opens** the sim gripper, matching the real robot's positive-command-opens
behavior confirmed above.

**Comparison:**

| Side | Direction of positive/increasing command |
|------|---------------------------------------------|
| Real robot (`joint_jog.py ... gripper 20`, pos 71.61 → 81.55) | **Opens** the jaws |
| Sim, post-fix (`SoarmGripper.format_action`, external action `+1`) | **Opens** the jaws (jaw gap 0.018m → 0.102m) |

Directions now **agree**. Confirmed explicitly by the user.

**Outcome: PASS.** TWIN-07 is satisfied. This supersedes the original
"Outcome: FAIL" header above, which is retained for audit history — the
authoritative outcome for this requirement is PASS, as of this re-test.
