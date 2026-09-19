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
