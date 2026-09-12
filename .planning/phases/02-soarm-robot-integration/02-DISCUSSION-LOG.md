# Phase 2: SOARM Robot Integration - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-07-11
**Phase:** 2-SOARM Robot Integration
**Areas discussed:** BDDL task selection, Robot config & mount, Deliverable format & workspace, Physics tuning scope

---

## BDDL Task Selection

| Option | Description | Selected |
|--------|-------------|----------|
| libero_spatial | Simplest scenes, fewer objects, easiest to validate physics; reusable for Phase 5 spatial-language needs | ✓ |
| libero_goal | Goal-oriented, more diverse but higher risk of contact-force instability with new robot | |
| libero_object | Object-manipulation focused, moderate complexity, less spatial-language relevance | |

**User's choice:** libero_spatial (recommended)
**Notes:** —

| Option | Description | Selected |
|--------|-------------|----------|
| Reuse as-is first | Swap robot name, only modify scene/objects if a task genuinely fails on reach/workspace | ✓ |
| Pre-emptively adjust workspace | Assume SOARM's smaller reach needs closer object placement before testing | |

**User's choice:** Reuse as-is first (recommended)
**Notes:** —

| Option | Description | Selected |
|--------|-------------|----------|
| You decide during planning | Researcher/planner picks 3 tasks informed by workspace constraints found during MJCF work | ✓ |
| I have specific tasks in mind | User names them now | |

**User's choice:** You decide during planning (recommended)
**Notes:** —

---

## Robot Config & Mount

| Option | Description | Selected |
|--------|-------------|----------|
| OnTheGroundPanda-style only | Single ground-mounted variant, matches physical SO101 tabletop arm | ✓ |
| Both mount variants | Build ground + mounted classes mirroring Panda exactly | |

**User's choice:** OnTheGroundPanda-style only (recommended)
**Notes:** —

| Option | Description | Selected |
|--------|-------------|----------|
| Stand-in gripper (e.g. PandaGripper) | Faster, defers accurate gripper modeling | |
| Model SOARM's actual gripper | Geometry already exists in source MJCF, not built from scratch | ✓ |

**User's choice:** Model SOARM's actual gripper
**Notes:** User asked why any "modeling" is needed at all if STL files already exist for SOARM. Clarified: geometry/meshes are reused as-is from `so101_new_calib.xml` (TheRobotStudio/SO-ARM100); the phase's actual work is *adapting* that existing MJCF into robosuite's `ManipulatorModel`/`GripperModel` Python abstractions (grip-site markers, actuator naming conventions, mount points, gripper class split, possible damping/friction retuning) — not building any geometry from scratch. User confirmed this resolved the question.

| Option | Description | Selected |
|--------|-------------|----------|
| Separate Gripper class | Matches robosuite's ManipulatorModel pattern; required for default_gripper property and action-space pipeline | ✓ |
| Keep as single combined body | Simpler if fused in source, but likely breaks robosuite's gripper action-space assumptions | |

**User's choice:** Separate Gripper class (recommended)
**Notes:** —

---

## Deliverable Format & Workspace

| Option | Description | Selected |
|--------|-------------|----------|
| Python files in LIBERO fork | New importable modules mirroring on_the_ground_panda.py, registered in ROBOT_CLASS_MAPPING | ✓ |
| Colab notebook only | Inline definition, simpler to iterate but not reusable as a module for later phases | |

**User's choice:** Python files in LIBERO fork (recommended)
**Notes:** User raised a workflow concern: GPU access is Colab-only. Clarified that MJCF/robot-class Python editing itself needs no GPU (pure MuJoCo physics + Python) and can be done locally using the existing `explorations/create_scene.py` `MUJOCO_GL=glfw` pattern; files are still committed to the LIBERO fork as normal git-tracked modules. Only the verification step (env.reset/render/BDDL runs) needs Colab's GPU runtime. User confirmed: edit locally, verify on Colab.

| Option | Description | Selected |
|--------|-------------|----------|
| Colab verification notebook | New notebook with PASS/FAIL cells per success criterion, consistent with Phase 1's UAT pattern | ✓ |
| Local/CI test script | More automatable but breaks from established Colab-centric verification pattern | |

**User's choice:** Colab verification notebook (recommended)
**Notes:** —

---

## Physics Tuning Scope

| Option | Description | Selected |
|--------|-------------|----------|
| Good enough to pass criteria | Tune only until success criteria met (contact forces <10N, stable rendering, 3+ tasks run); refine later if needed | ✓ |
| Precise tuning now | Invest more time on hardware-accurate physics now, beyond what criteria require | |

**User's choice:** Good enough to pass criteria (recommended)
**Notes:** —

| Option | Description | Selected |
|--------|-------------|----------|
| Use SO101's documented calibration pose if available | Start from hardware default pose already encoded in so101_new_calib.xml | ✓ |
| Tune arbitrarily for sim stability | Ignore hardware calibration, pick whatever keeps contact forces low | |

**User's choice:** Use SO101's documented calibration pose if available (recommended)
**Notes:** —

---

## Claude's Discretion

- Exact joint damping values, base_xpos_offset per scene type, horizontal_radius/top_offset — tuned iteratively to pass success criteria, same pattern as OnTheGroundPanda.
- Which 3+ specific libero_spatial BDDL task files to adapt.
- Whether any BDDL scene/object repositioning is needed (only if reuse-as-is fails during verification).
- Exact structure/cell layout of the Colab verification notebook.

## Deferred Ideas

- Mounted (wall/ceiling) SOARM variant — no current use case, only ground-mounted is in scope.
- Hardware-accurate physics tuning beyond success-criteria requirements — deferred until a later phase surfaces a concrete need.
