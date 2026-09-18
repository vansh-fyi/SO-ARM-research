# Phase 10: Digital-Twin Fidelity - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-09-18
**Phase:** 10-Digital-Twin Fidelity
**Areas discussed:** URDF rebuild strategy, Joint limit source (calibration ticks), Gripper-direction verification (TWIN-07), URDF's downstream purpose

---

## URDF Rebuild Strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Generate URDF from MuJoCo model | Derive URDF kinematics from robot.xml + soarm_gripper.xml (already correct) | ✓ |
| Patch broken CoppeliaSim-exported URDF directly | Manually fix So-101.urdf in place | |
| Hybrid: patch structure, cross-verify against MuJoCo | Patch structure but copy MuJoCo's joint values | |

**User's choice:** Generate URDF from MuJoCo model.
**Notes:** Initial framing assumed the MuJoCo model was already fully correct. User corrected this: the MuJoCo gripper is also flawed — specifically a visual-only mesh placement bug ("the gear teeth literally go outside of the gripper, other than working in an L shape"), not a structural/joint problem. Motion and grasping physics (box-jaw physics geoms) work correctly. After clarifying this, the user re-confirmed generating the URDF from the MuJoCo model is still the right approach, plus fixing the clamp mesh placement bug in the same phase.

**Follow-up: conversion tool**

| Option | Description | Selected |
|--------|-------------|----------|
| Research best tool during planning | Let researcher/planner pick (MuJoCo has no native URDF export) | ✓ |
| Write custom script now | Hand-write a converter given the simple known structure | |

**User's choice:** Research the best tool during planning.

**Follow-up: scope of the mesh fix**

| Option | Description | Selected |
|--------|-------------|----------|
| Fix inside Phase 10 | Directly relevant to TWIN-03/TWIN-07 gripper fidelity | ✓ |
| Defer as separate follow-up | Cosmetic, doesn't block kinematic TWIN criteria | |

**User's choice:** Fix inside Phase 10.

---

## Joint Limit Source (Calibration Ticks)

| Option | Description | Selected |
|--------|-------------|----------|
| Current follower's live calibration | Whatever control/ currently uses for the real robot | ✓ |
| Re-run calibration fresh | Run lerobot-calibrate/recalibrate_gripper.py fresh | |
| Locate file first | Flag as unknown, researcher must find it | |

**User's choice:** The current follower's live calibration.
**Notes:** No calibration JSON lives in-repo; LeRobot writes it to its local cache keyed to a device name (e.g. `soarm_follower_02`, per `control/COMMANDS.md`).

**Follow-up: re-derive vs. trust existing limits**

| Option | Description | Selected |
|--------|-------------|----------|
| Re-derive from live calibration, verify against robot.xml | Cross-check existing 4 arm-joint limits, correct if mismatched | ✓ |
| Trust robot.xml's existing limits as-is | Only add new wrist_roll/gripper limits | |

**User's choice:** Re-derive and verify against robot.xml.

---

## Gripper-Direction Verification (TWIN-07)

| Option | Description | Selected |
|--------|-------------|----------|
| User does a manual side-by-side check | Send same command to real + sim, visually compare | ✓ |
| Document convention, verify structurally only | Rely on code/comment consistency, no live check | |
| Record reference video first | Capture real gripper on video, compare against that | |

**User's choice:** Manual side-by-side check.
**Notes:** Phase 10 is otherwise sim-side-only; this is a deliberate, acknowledged exception requiring the user's involvement at verification time.

**Follow-up: which command representation to compare**

| Option | Description | Selected |
|--------|-------------|----------|
| LeRobot's native action value | Real command translated into sim's -1/+1 convention | ✓ |
| Sim's normalized -1/+1 value | Translated to real units for the real side | |

**User's choice:** LeRobot's native action value.

---

## URDF's Downstream Purpose

| Option | Description | Selected |
|--------|-------------|----------|
| Documentation/reference artifact only | Must load cleanly and be correct, no integration testing needed | ✓ |
| Feeds a future real-hardware/ROS control path | Needs load-and-use testing | |
| Not sure — flag as open question for research | Let researcher check if anything consumes it | |

**User's choice:** Documentation/reference artifact only.
**Notes:** Nothing in the repo currently loads the URDF programmatically — the MuJoCo sim uses robot.xml directly.

---

## Claude's Discretion

- Exact MuJoCo→URDF conversion tool/script (deferred to research + planning).
- How to represent the corrected clamp mesh placement in the generated URDF's gripper geometry.

## Deferred Ideas

None. The clamp visual-mesh placement bug, discovered mid-discussion, was folded into Phase 10 scope (see CONTEXT.md D-04) rather than deferred, since it falls directly under the existing TWIN-03/TWIN-07 gripper fidelity requirements.
