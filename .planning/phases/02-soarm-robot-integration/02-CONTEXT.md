# Phase 2: SOARM Robot Integration - Context

**Gathered:** 2026-07-11
**Status:** Ready for planning

<domain>
## Phase Boundary

Adapt the existing SOARM SO101 hardware description (MJCF + STL meshes from TheRobotStudio/SO-ARM100, specifically `so101_new_calib.xml`) into robosuite's `ManipulatorModel`/`GripperModel` abstractions, register it in LIBERO's `ROBOT_CLASS_MAPPING` exactly as Panda is today, and validate it against at least 3 `libero_spatial` BDDL tasks. No geometry is built from scratch — this phase is wiring/adaptation work (grip-site markers, actuator naming, mount points, physics tuning) so LIBERO can drive an existing robot model, not robot design.

</domain>

<decisions>
## Implementation Decisions

### BDDL Task Selection
- **D-01:** Adapted tasks come from the `libero_spatial` suite — simplest scenes, sets up reuse for Phase 5 (SPAT-05 needs spatial-language BDDL tasks), lower physics-instability risk than `libero_goal`/`libero_object`.
- **D-02:** Reuse existing BDDL files as-is first (swap robot name only). Only modify scene/object placement if a specific task genuinely fails due to SOARM's reach/workspace during verification — no speculative pre-adjustment.
- **D-03:** Claude (researcher/planner) selects the specific 3+ `libero_spatial` task names during planning, informed by workspace constraints discovered while building the MJCF/robot class.

### Robot Config & Mount
- **D-04:** Ground-mounted only — mirror `OnTheGroundPanda`, not `MountedPanda`. Physical SO101 is a tabletop arm; no mount-hardware use case exists yet.
- **D-05:** Model SOARM's actual gripper (geometry already exists in the source MJCF, not built from scratch).
- **D-06:** Split the gripper into a separate robosuite `GripperModel` class (e.g. `SoarmGripper`), not fused into the arm body — required for robosuite's gripper attachment/action-space pipeline (`default_gripper` property, LIBERO's actionable-gripper assumption).
- **D-07:** SOARM's `init_qpos` (home/rest pose) should start from SO101's documented default/calibration pose (if encoded in `so101_new_calib.xml`) rather than an arbitrary sim-tuned pose — adjust only if that pose proves unstable in sim.

### Deliverable Format & Workspace
- **D-08:** Robot/gripper Python classes live as real git-tracked modules in the LIBERO fork — `LIBERO/libero/libero/envs/robots/soarm.py` and a gripper module (e.g. `LIBERO/libero/libero/envs/grippers/soarm_gripper.py`), mirroring `on_the_ground_panda.py`'s structure, registered in `ROBOT_CLASS_MAPPING` via `LIBERO/libero/libero/envs/robots/__init__.py` (same pattern as existing Panda variants).
- **D-09:** MJCF/robot-class editing and iteration happens **locally** (no GPU required — pure MuJoCo physics + Python; reuse the `MUJOCO_GL=glfw` macOS pattern from `explorations/create_scene.py` for any local rendering checks).
- **D-10:** Verification of ENV-04 through ENV-07 happens on **Colab**, via a new notebook (`LIBERO/notebooks/02-soarm-integration-check.ipynb`) with PASS/FAIL cells per success criterion — consistent with Phase 1's UAT pattern. This is the only part of this phase that needs the Colab GPU runtime.

### Physics Tuning Scope
- **D-11:** Tune joint damping, `init_qpos`, and base offsets only to the point of passing success criteria (contact forces <10N at reset, stable rendering, 3+ BDDL tasks complete without crashing) — do not over-invest in hardware-accurate realism beyond that. Remaining physics issues can be refined in later phases (dataset collection, fine-tuning) if they surface.

### Claude's Discretion
- Exact joint damping values, base_xpos_offset per scene type, and horizontal_radius/top_offset tuning — same pattern as `OnTheGroundPanda`, tuned iteratively to pass success criteria.
- Which 3+ specific `libero_spatial` BDDL task files to adapt (D-03).
- Whether any BDDL scene/object repositioning is needed (only if reuse-as-is fails).
- Exact structure/cell layout of the Colab verification notebook, following Phase 1's established pattern.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements & Roadmap
- `.planning/REQUIREMENTS.md` §Environment Setup — ENV-04, ENV-05, ENV-06, ENV-07 are this phase's requirements.
- `.planning/ROADMAP.md` §Phase 2: SOARM Robot Integration — success criteria (physics stability <10N contact force, correct rendering, ROBOT_CLASS_MAPPING registration, 3+ BDDL tasks running end-to-end).

### Required Environment Reading
- `.planning/phases/01-colab-environment-setup/01-DEBUG-HISTORY.md` — REQUIRED reading before touching the Colab environment: the 7-invariant environment contract, `--no-deps` install contract, restart-vs-clean-slate distinction, dependency debugging lessons that apply to any new notebook work in this phase.

### Existing Robot Integration Pattern (robosuite/LIBERO)
- `LIBERO/libero/libero/envs/robots/on_the_ground_panda.py` — Canonical `ManipulatorModel` subclass pattern to mirror: `init_qpos`, `default_gripper`, `default_controller_config`, `base_xpos_offset` per scene type, `top_offset`, `_horizontal_radius`, `arm_type`, joint damping via `set_joint_attribute`.
- `LIBERO/libero/libero/envs/robots/mounted_panda.py` — Alternate mount variant (not used this phase per D-04, but shows the mount-variant pattern if ever needed later).
- `LIBERO/libero/libero/envs/robots/__init__.py` — `ROBOT_CLASS_MAPPING` registration pattern: `ROBOT_CLASS_MAPPING.update({"ClassName": SingleArm})`.
- `LIBERO/libero/libero/bddl_files/libero_spatial/` — BDDL task suite to adapt (D-01).

### Prior Research Decisions (from STATE.md)
- SOARM MJCF derives from `so101_new_calib.xml` (TheRobotStudio/SO-ARM100), not built from scratch.
- Reference TechLabs Aachen SO100+robosuite integration as prior art for the ManipulatorModel adaptation approach.
- Budget flagged: 1-2 days of iterative MJCF editing — integration is novel, expect iteration.
- Demo replay (relevant to later phases, not this one) must be state-based, not action-replay — LIBERO issue #16.

### Colab Environment Contract
- `explorations/create_scene.py` — Local `MUJOCO_GL=glfw` pattern for macOS rendering checks during local MJCF iteration (D-09).
- Phase 1's notebook (`LIBERO/notebooks/01-colab-env-setup.ipynb`) — Reference for the PASS/FAIL verification cell pattern to replicate in the new Phase 2 verification notebook (D-10).

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `LIBERO/libero/libero/envs/robots/on_the_ground_panda.py`: Direct structural template — class shape (7 properties + `__init__` with `set_joint_attribute` for damping) transfers almost 1:1 to a `SoarmOnTheGround`-style class, just with SOARM-specific `init_qpos`, offsets, and xml path.
- `LIBERO/libero/libero/envs/robots/__init__.py`: Registration pattern — new SOARM class gets added to the same `ROBOT_CLASS_MAPPING.update({...})` call alongside the existing Panda entries.
- `explorations/create_scene.py`: EGL/glfw MUJOCO_GL pattern, reusable for local physics/rendering sanity checks before pushing to Colab.

### Established Patterns
- `MUJOCO_GL` must be set before any MuJoCo import — same constraint as Phase 1, applies to both local iteration and the Colab verification notebook.
- robosuite's `ManipulatorModel` expects `xml_path_completion(...)` for the MJCF path, gripper as a separate attachable model, and specific properties (`default_mount`, `default_gripper`, `default_controller_config`, `init_qpos`, `base_xpos_offset`, `top_offset`, `_horizontal_radius`, `arm_type`) — SOARM's class must implement all of these, same as Panda.

### Integration Points
- New SOARM robot/gripper files integrate into LIBERO's existing `envs/robots/` and (new) `envs/grippers/` directories — no new top-level module structure needed.
- Phase 3 (VLA Inference Loop) and Phase 4 (Dataset Collection) will import the SOARM robot class registered here — the class must be a clean, reusable import, not notebook-embedded code (this is why D-08 locks in real Python files over notebook-only definitions).
- The Colab verification notebook for this phase should be extendable the same way Phase 1→ Phase 2's install cell was designed to be extended (per Phase 1 CONTEXT.md's noted integration point).

</code_context>

<specifics>
## Specific Ideas

- User confirmed the underlying premise directly: STL/mesh geometry for SOARM already exists (open-hardware, TheRobotStudio/SO-ARM100) and should not be re-modeled — this phase is purely an adaptation/wrapping exercise into robosuite's abstractions, not robot design work.
- User's compute constraint (GPU only available via Colab) shapes the workflow split: local editing (no GPU needed) → Colab-only verification (GPU needed for the actual render/BDDL run). This mirrors Phase 1's local-vs-Colab boundary and should be made explicit in planning so the plan doesn't assume Colab access for iterative MJCF debugging.

</specifics>

<deferred>
## Deferred Ideas

- Mounted (wall/ceiling) SOARM variant — no current use case; only ground-mounted is in scope for this phase (D-04). Revisit if a future phase needs it.
- Hardware-accurate physics tuning beyond what success criteria require — deferred to whichever later phase first surfaces a concrete physics-fidelity problem (D-11).

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 2-SOARM Robot Integration*
*Context gathered: 2026-07-11*
