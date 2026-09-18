# Phase 10: Digital-Twin Fidelity - Context

**Gathered:** 2026-09-18
**Status:** Ready for planning

<domain>
## Phase Boundary

The URDF (`So-101/So-101.urdf`) and the MuJoCo XML (`LIBERO/libero/libero/assets/robots/soarm101/robot.xml` + `LIBERO/libero/libero/assets/grippers/soarm_gripper.xml`) must form a correct, complete, 1:1 kinematic match to the real SO-ARM101: a proper parent/child chain ending at the gripper, `wrist_roll` and gripper jaw joints restored with correct axes/direction, correct joint limits derived from real servo calibration, and in-repo mesh paths. Sim-side-only — does not touch the real robot or the `control/` bridge (except for a one-time manual verification step, see below).

</domain>

<decisions>
## Implementation Decisions

### URDF Rebuild Strategy
- **D-01 [informational]:** The MuJoCo side (`robot.xml` + `soarm_gripper.xml`) already has correct kinematic joints — `wrist_roll` and `gripper_left`/`gripper_right` prismatic joints already exist with plausible limits, built independently of the URDF (not derived from it). This was NOT known before this discussion and changes the phase's shape: it is not a from-scratch kinematic rebuild on both files, it's "generate a correct URDF from an already-correct MuJoCo model," plus a separate visual-mesh fix.
- **D-02:** Generate the URDF from the MuJoCo model (`robot.xml` + `soarm_gripper.xml`), rather than patching the broken CoppeliaSim-exported `So-101/So-101.urdf` in place. This guarantees the URDF and MuJoCo XML agree by construction instead of risking two independently-maintained kinematic definitions drifting apart. Visual meshes are still sourced from the CoppeliaSim export (`So-101/*.dae`, `coppelia/meshes/*.stl`).
- **D-03:** Do not lock the conversion tool/method now. The phase-researcher should survey available options (MuJoCo has no native URDF exporter; third-party mjcf-to-urdf converters exist) during Phase 10 research, and the planner should pick based on fidelity/maintainability vs. a small hand-written converter script.
- **D-04:** A known, distinct bug exists in `soarm_gripper.xml`'s clamp **visual** mesh placement (NOT the physics — physics uses separate box-jaw geoms and works correctly for grasping/motion): the clamp STL's gear-tooth detail visibly pokes outside the `main_frame_visual` housing during rendering in Python, instead of sitting flush in an L-shape against the frame. This is a mesh recentering/offset bug in the clamp geom placement (see the asset-note comment in `soarm_gripper.xml` around the `soarm_main_frame` mesh). **Fix this inside Phase 10** — it's directly part of gripper fidelity (TWIN-03/TWIN-07) and touches the same file being worked on.

### Joint Limit Source (TWIN-05)
- **D-05:** No calibration JSON lives in this repo — LeRobot writes/reads it from its local cache keyed to a device name (e.g. `soarm_follower_02`, referenced in `control/COMMANDS.md`). The source of truth for TWIN-05 is the **current follower arm's live LeRobot calibration file** — whatever `control/` presently uses to drive the real robot. Locate it (LeRobot's calibration cache dir) before deriving limits.
- **D-06:** Re-derive joint limits from that live calibration file (tick → radian conversion) and cross-check the result against `robot.xml`'s existing 4 arm-joint limits (`shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`) — these look like they were already derived from calibration in Phase 2 but have never been independently re-verified. If they match, `robot.xml` is confirmed correct; if not, correct it. `wrist_roll` and gripper limits are derived fresh from the same calibration file either way (TWIN-02/TWIN-05).

### Gripper-Direction Verification (TWIN-07)
- **D-07:** Phase 10 is sim-side-only, but TWIN-07 ("driving the simulated gripper with a given joint command opens/closes it in the same direction as the real gripper under the identical command") cannot be verified from sim state alone. Verification method: **the user performs a manual side-by-side check** — send a known gripper command to the real robot via `control/` (e.g. `jog_gripper_raw.py`), drive the sim with the same command, visually confirm both open/close the same way. This requires the user's involvement at verification time during Phase 10 execution — flag this to the planner/executor as a human-in-the-loop checkpoint, not something the executor can auto-verify.
- **D-08:** Compare using **LeRobot's native gripper action value** (the actual command representation `control/` sends to the real servo), translated into the sim's `-1`/`+1` convention (`SoarmGripper.format_action`: `-1`=open, `+1`=closed) for the comparison — not the reverse. Keeps the real-hardware side of the comparison unambiguous.

### URDF's Downstream Purpose
- **D-09:** Nothing in the repo currently loads the URDF programmatically — the MuJoCo sim uses `robot.xml` directly, and no ROS/IK/visualization tooling in this repo consumes `So-101.urdf`. Treat the fixed URDF as a **documentation/reference artifact**: it must load cleanly (via a standard URDF parser) and be kinematically correct per TWIN-01..05, but does not need integration testing against any consuming system beyond that — no ROS stack or IK solver is currently expected to load it in this milestone.

### Claude's Discretion
- Exact conversion tool/script for MuJoCo→URDF generation (research + planning decide, per D-03).
- How to represent the visual-only clamp mesh fix in the generated URDF (the URDF's gripper geometry should reflect the corrected mesh placement once fixed in `soarm_gripper.xml`).

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Kinematic sources (ground truth)
- `LIBERO/libero/libero/assets/robots/soarm101/robot.xml` — MuJoCo arm XML; already has correct `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll` joints with limits, terminates at `right_hand` body (see its header comment on the gripper-boundary split)
- `LIBERO/libero/libero/assets/grippers/soarm_gripper.xml` — MuJoCo gripper XML (robosuite `GripperModel` pattern); has correct `gripper_left`/`gripper_right` prismatic joints and physics (box jaws), but a known visual-mesh placement bug in the clamp geoms (D-04)
- `LIBERO/libero/libero/envs/grippers/soarm_gripper.py` — `SoarmGripper` class; documents the `-1`=open/`+1`=closed action convention (`format_action`) needed for D-08's comparison

### Broken artifacts being fixed
- `So-101/So-101.urdf` — the broken CoppeliaSim-exported URDF: gripper is a separate root-level object near the wrist, never parented under the arm; ends at `wrist_link_respondable` with no `wrist_roll`/gripper joints at all; all mesh `filename` refs are absolute `file:///Users/hp/Downloads/...` paths (30 occurrences)
- `coppelia/export_model_library.py` — its own comments (lines ~59-61) confirm the root-level-gripper bug: "The reference scene keeps the arm (/so101) and its stock gripper (/gripper_link_respondable) as two separate root-level objects — the gripper is positioned to match the wrist but is not parented under the arm."
- `coppelia/soarm_parallel_gripper.urdf` — separate gripper-only URDF fragment (relative mesh paths, correct-looking prismatic joint structure) that may be useful reference/input for the gripper portion of the generated URDF

### Calibration / hardware bridge
- `control/COMMANDS.md` — references the live follower calibration device name and `recalibrate_gripper.py` / `jog_gripper_raw.py` commands needed for D-05/D-08's live calibration lookup and manual verification
- `control/recalibrate_gripper.py`, `control/jog_gripper_raw.py` — scripts for reading/driving the real gripper during the D-07 manual verification step

### Mesh assets (visual-only, trusted geometry)
- `So-101/*.dae`, `So-101/*.stl` — CoppeliaSim-exported visual/collision meshes, geometry trusted as correct per PROJECT.md; only paths need fixing (in-repo relative, not absolute)
- `coppelia/meshes/*.stl` — in-repo mesh copies already used by `soarm_gripper.xml` (relative paths, no fix needed there)

No external ADRs/specs beyond ROADMAP.md and REQUIREMENTS.md TWIN-01..07 — requirements fully captured in decisions above.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `coppelia/soarm_parallel_gripper.urdf`: standalone gripper URDF fragment with correct-looking relative mesh paths and prismatic joint structure — useful reference when generating the gripper portion of the new URDF
- `LIBERO/libero/libero/assets/robots/soarm101/so101_new_calib.source.xml`: the pre-Phase-2 source MJCF this fork's `robot.xml` was derived from — may help trace where the existing joint limits originally came from

### Established Patterns
- `SoarmGripper` follows robosuite's `PandaGripper` integrated-action pattern (single 1-D action collapsed to two jaw actuators) — any URDF/MJCF joint changes must preserve this actuator/action contract, not just the raw kinematic structure
- MuJoCo body/joint tree in `robot.xml` uses a documented gripper-boundary split ("body split at the wrist_roll->gripper boundary") — the generated URDF should preserve this same split point (arm URDF ends at `right_hand`/wrist, gripper URDF/fragment attaches there), mirroring `coppelia/soarm_parallel_gripper.urdf`'s existing separation

### Integration Points
- The URDF and MuJoCo XML are NOT currently cross-loaded by any script — there is no existing "load both, diff the trees" tooling. Phase 10 will likely need to write this verification tooling itself (per TWIN-01's stated verification method: "loading the URDF and walking/printing its parent-child joint chain").

</code_context>

<specifics>
## Specific Ideas

- The clamp visual mesh bug was described by the user as: "The gear teeth literally go outside of the gripper, other than working in an L shape" — visible when running the sim in Python. Motion and grasping physics are unaffected; this is purely a rendered visual-mesh placement issue.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope. (The clamp mesh bug, while newly discovered during this discussion, was folded into Phase 10 scope per D-04 rather than deferred, since it's directly part of the gripper fidelity requirements TWIN-03/TWIN-07 already cover.)

</deferred>

---

*Phase: 10-Digital-Twin Fidelity*
*Context gathered: 2026-09-18*
