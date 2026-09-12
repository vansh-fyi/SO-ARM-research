# Phase 5: Spatial Awareness - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-08-09
**Phase:** 5-Spatial Awareness
**Areas discussed:** Multi-camera → VLA wiring, Depth & XYZ purpose, Spatial BDDL task design, Spatial success predicates

---

## Multi-camera → VLA wiring

| Option | Description | Selected |
|--------|-------------|----------|
| Feed both to VLA (You decide impl.) | Extend both pi0 and OFT backends to accept overhead+wrist together | ✓ |
| Overhead for XYZ/depth only | Keep VLA inference wrist-only; overhead used only for depth/XYZ | |
| Feed both, simple concat fallback | Always pass both images; concat as fallback if not natively supported | |

**User's choice:** Feed both to VLA — Claude's discretion on implementation.

| Option | Description | Selected |
|--------|-------------|----------|
| Concat/tile as fallback | Resize+combine both views if no native multi-view support | |
| Skip that backend's overhead input | Only wrist goes to backends that can't take a second view | |
| You decide per-backend | Research each backend during planning and pick correctly | ✓ |

**User's choice:** You decide per-backend.
**Notes:** eval_loop.py currently only builds `images = {"eye_in_hand": obs[camera_name]}` despite the interface being dict-shaped for multi-view since Phase 3.

---

## Depth & XYZ purpose

| Option | Description | Selected |
|--------|-------------|----------|
| XYZ from sim state (cheap); depth is separate | Ground-truth XYZ from sim.data; depth extracted independently | |
| XYZ derived from depth (more 'real') | Object XYZ computed by back-projecting depth buffer + intrinsics | |
| Both: sim-state XYZ as ground truth, depth as a check | Sim-state primary, depth-derived XYZ validated against it | |

**User's choice:** Asked for a primer on the distinction first (sim-state ground truth vs. depth-derived XYZ), then chose "XYZ computed from depth (simulates real robot)."
**Notes:** User is planning to build the physical SOARM arm soon — this pipeline is meant to transfer to real hardware later, not just satisfy the roadmap's cheaper literal wording.

| Option | Description | Selected |
|--------|-------------|----------|
| Yes — validate against sim state | Cross-check depth-derived XYZ against sim.data ground truth | ✓ |
| No — trust the geometry, no cross-check | Skip runtime validation | |

**User's choice:** Yes — validate against sim state.

---

## Spatial BDDL task design

| Option | Description | Selected |
|--------|-------------|----------|
| Reuse Phase 4's task + objects | Build spatial variants around put_the_cream_cheese_in_the_bowl scene | |
| New scene, same object-size/reach constraints | Fresh scene with small graspable objects for clearer relations | |
| You decide | Claude picks fastest-to-validate option within embodiment constraints | ✓ |

**User's choice:** You decide.

| Option | Description | Selected |
|--------|-------------|----------|
| left/right + near/far + between | Broadest coverage matching libero_spatial conventions | ✓ |
| left/right only, multiple object pairs | Narrower, unambiguous, lower predicate-ambiguity risk | |
| You decide | Claude picks based on LIBERO's predicate library support | |

**User's choice:** left/right + near/far + between.

---

## Spatial success predicates

| Option | Description | Selected |
|--------|-------------|----------|
| Threshold-based (with margin) | Matches LIBERO's own predicate convention | ✓ |
| Strict geometric comparison | Exact coordinate comparison, no margin | |
| You decide | Research LIBERO's existing predicate implementations | |

**User's choice:** Threshold-based (with margin).

| Option | Description | Selected |
|--------|-------------|----------|
| Ground truth judges (Recommended) | MuJoCo's exact internal state scores PASS/FAIL | ✓ |
| Depth estimate judges | Depth-derived XYZ scores PASS/FAIL directly | |

**User's choice:** Ground truth judges (after two rounds of clarification — user asked "explain this to me more" and then a broader question about what the phase actually builds and their real-hardware plans; both answered in plain text before re-asking).
**Notes:** User is building the physical arm soon and was confused about the sim-vs-real distinction; clarified that this phase is entirely simulation, and that judging on ground truth doesn't conflict with D-03's depth pipeline still running and being useful for real-hardware transfer later.

| Option | Description | Selected |
|--------|-------------|----------|
| Avoid at task-design time (Recommended) | Author tasks with enough margin that ambiguous cases can't arise | ✓ |
| Explicit tie-break in predicate logic | Predicate defines a clear rule for borderline cases | |

**User's choice:** Avoid at task-design time.

---

## Claude's Discretion

- Per-backend strategy for consuming two camera views (native multi-view vs. concat/tile fallback).
- Whether to reuse Phase 4's scene/objects vs. author a new one for the spatial tasks.
- Exact tolerance/margin size for spatial predicates.

## Deferred Ideas

- Real-hardware depth camera transfer (real sensor selection, mounting, calibration) — flagged for whichever future phase picks up PHYS-01..03, since the user is building the physical arm soon.
- Third camera angle (front/side view) for occlusion robustness — informative answer given, not adopted since SPAT-01 locks a 2-camera scope (wrist + overhead).
