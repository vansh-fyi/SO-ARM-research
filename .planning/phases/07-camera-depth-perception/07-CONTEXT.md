# Phase 7: Camera & Depth Perception - Context

**Gathered:** 2026-09-05
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase delivers two things:
1. The sim `agentview`/front camera is recalibrated (position, angle, FOV) to match the real SOARM overhead camera's placement, closing the sim-to-real view gap (CAM-01). This also now includes correcting the `eye_in_hand` (wrist) camera's angle, which was found during discussion to be miscalibrated in the same way (see Decisions).
2. A depth observation stream is plumbed end-to-end: sim obs dict → HDF5 dataset writer → RLDS converter (DEPTH-01/02/03).

Codebase scouting found the sim-side depth *rendering* is already solved (robosuite's `camera_depths=True` flag, proven working in `test_camera_config.py`, plus an existing depth→world back-projection pipeline in `depth_xyz.py` from Phase 5). The real work for the depth requirements is in the **persistence layer** — HDF5 writer and RLDS converter — not new rendering code.

</domain>

<decisions>
## Implementation Decisions

### Agentview (overhead) camera recalibration
- **D-01:** The real overhead camera (Waveshare 32695 AR0144 stereo module) has no finalized mount yet — position/FOV aren't documented in the repo (`diagnostics/PARTS_LIST.md` flags this as "not yet resolved"). Proceed with a **best-estimate recalibration now** using the AR0144 datasheet FOV and a reasonable overhead-workspace placement; do not block Phase 7 on physical hardware completion. Revisit/re-tune once the real mount is built.
- **D-02:** The overhead camera mount should be a **fixed frame/arm bolted to the desk/table edge** (clamp-style, similar to how the robot base is mounted) — static and repeatable, not a tripod or ball-head/gooseneck. This gives stable, remeasurable real-world numbers to recalibrate sim FOV/pos against.
- **D-03:** No pre-made STL exists for the housed AR0144 stereo module (66x30x17.72mm body). The mount design should be: a **custom-printed friction-fit cradle** sized to that body, combined with **generic off-the-shelf 1/4-20 tripod hardware** for the frame/arm itself (e.g., a printable 1/4-20 tripod adapter/bolt — see Specific Ideas below for candidates found during research). This mirrors the pattern already used for the roboninecom gripper (custom part + off-the-shelf mechanical standard).
- **D-04:** Success Criteria #1 ("rendered frame visibly matches real camera's framing via side-by-side/overlay") is **deferred** — do the actual side-by-side photo comparison once the overhead camera's physical mount is built (tracked as a follow-up, likely a UAT step outside strict Phase 7 completion, consistent with how the physical hardware track already runs in parallel per PROJECT.md). Phase 7 sign-off is based on documented estimated params, not a live photo comparison.

### Depth camera scope
- **D-05:** Depth is persisted for **`agentview` only**, not `eye_in_hand`. This matches the real hardware exactly — only the overhead AR0144 is stereo/depth-capable; the wrist IMX335 has no depth capability. `OBS_KEY_MAPPING` in `hdf5_writer.py` should gain an `agentview_depth` entry only; `oxe_register.py`'s `depth_obs_keys["primary"]` becomes `"agentview_depth"`, `"wrist"` stays `None`.

### Eye-in-hand (wrist) camera angle correction — added to scope during discussion
- **D-06:** The current sim `eye_in_hand` camera (`fovy=75`, `quat="0.664463 -0.241845 0.241845 -0.664463"` in `LIBERO/libero/libero/assets/robots/soarm101/robot.xml:108-109`, tuned in Phase 2 plan 02-04) aims **too far forward** along the wrist axis. Reference photos (`progress-documentation/images/20260827_121030-2.jpg`, `20260827_121037-2.jpg`, `20260827_121100-2.jpg`, `20260827_121104-2.jpg`, `20260827_121106-2.jpg`, `20260827_121114-2.jpg`) show the real IMX335 is mounted on a bracket that **tilts the lens down/inward toward the gripper's grasp point** — the point where the fingers converge — not straight out ahead. This needs correcting in Phase 7, in addition to `agentview`. The exact new `quat`/`fovy` values are implementation work for planning/research to derive from these reference photos (e.g. estimating the tilt angle) — not finalized here.
- **D-07:** Unlike the overhead camera, the wrist camera's real mount is already built and photographed — no hardware blocker for this correction. It can be verified against the reference photos directly, unlike the agentview criterion which is deferred pending the overhead mount build.

### Claude's Discretion
- Exact new `fovy`/`quat` numeric values for both `agentview` and `eye_in_hand` — derive from AR0144 datasheet specs + estimated placement (agentview) and from the reference photos (eye_in_hand) during research/planning.
- Depth storage dtype/precision in HDF5 (float32 raw meters vs. quantized) — not discussed, left to planner.
- RLDS depth feature type (plain `Tensor`, not `Image`, per TFDS/OXE convention — already confirmed via codebase scout of `rlds_converter.py`).
- Physical stand design specifics (cradle wall thickness, clamp mechanism) — implementation detail for whoever prints it; not a software planning concern beyond documenting the target mount position it produces.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements & roadmap
- `.planning/ROADMAP.md` §"Phase 7: Camera & Depth Perception" — goal, success criteria, requirement IDs
- `.planning/REQUIREMENTS.md` — CAM-01, DEPTH-01, DEPTH-02, DEPTH-03 definitions

### Camera config (sim)
- `LIBERO/libero/libero/envs/bddl_base_domain.py:275-293` — `_setup_camera()`, the authoritative override point for `agentview`/`canonical_agentview` pos/quat (no `fovy` currently set here — will need one added)
- `LIBERO/libero/libero/assets/robots/soarm101/robot.xml:108-109` — `eye_in_hand` camera element (`fovy="75"`, current quat) — the fovy-on-camera-element syntax precedent to follow for agentview too
- `LIBERO/libero/libero/envs/env_wrapper.py:32-35` — default `camera_names` list and `camera_depths` constructor kwarg (already threaded through to robosuite)
- `LIBERO/libero/libero/envs/problems/libero_coffee_table_manipulation.py:187-202` — example of a problem file overriding `agentview` pos/quat per-scene (pattern to be aware of if other problem files need consistent recalibration)

### Depth rendering (already solved, reference only)
- `LIBERO/libero/libero/envs/test_camera_config.py:86-109` — proof that `camera_depths=True` yields non-degenerate `agentview_depth`/`robot0_eye_in_hand_depth` obs keys
- `LIBERO/libero/libero/perception/depth_xyz.py` — existing depth→world-XYZ back-projection pipeline (Phase 5), not part of this phase's scope but informs depth data semantics

### HDF5 persistence
- `LIBERO/libero/libero/datasets/hdf5_writer.py:49-58` (`OBS_KEY_MAPPING`, `_RGB_KEYS`, `_STATE_KEYS`), `:93-100` (env construction — needs `camera_depths=True`), `:156-170` (dtype-specific write branches — needs a depth branch)
- `LIBERO/libero/libero/datasets/test_hdf5_writer.py` — existing integration test to extend with a depth assertion

### RLDS conversion
- `LIBERO/libero/libero/datasets/rlds_converter.py:35-96` (`validate_episode_arrays`), `:99-174` (`load_episodes_from_hdf5`), `:177-199` (`_episode_to_rlds_steps`), `:254-283` (TFDS `FeaturesDict` — the actual schema fed to OXE)
- `LIBERO/libero/libero/datasets/oxe_register.py:77-91,167-169` — `depth_obs_keys` placeholder, already structurally present and required by `materialize.py`, currently all-`None`
- `LIBERO/libero/libero/datasets/test_rlds_converter.py`, `test_oxe_register.py` — existing tests to extend

### Real hardware specs
- `diagnostics/PARTS_LIST.md:133-141` — camera part specs (IMX335 5MP wrist cam, AR0144 2MP stereo overhead cam, 52mm baseline)
- `diagnostics/PARTS_LIST.md:157-163` — "Not yet resolved" section documenting the overhead mount gap and the generic-tripod-base + custom-cradle plan
- `progress-documentation/images/20260827_121030-2.jpg`, `20260827_121037-2.jpg`, `20260827_121100-2.jpg`, `20260827_121104-2.jpg`, `20260827_121106-2.jpg`, `20260827_121114-2.jpg` — reference photos of the real IMX335 wrist camera mount and tilt angle
- `control/outputs/step5_test/camera_0.png` — sample real camera capture (note: taken with arm in a vertical test pose, not representative of an operating grasp angle — use with caution as a framing reference)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `camera_depths=True` constructor kwarg on `ControlEnv`/`OffScreenRenderEnv` — depth rendering is a config flag away, already verified working
- `depth_xyz.py`'s back-projection utilities — not needed for this phase's persistence work, but establishes the depth data is already calibration-correct at the obs-dict level

### Established Patterns
- Camera `fovy` is set as a literal MJCF attribute on the `<camera>` element (see `eye_in_hand` in `robot.xml`), not via robosuite's `set_camera()` Python call — `agentview`'s fovy correction will need a different mechanism than its pos/quat (which do go through `set_camera()`)
- HDF5 writer distinguishes RGB (uint8) vs state (float64) via key-name tuples (`_RGB_KEYS`, `_STATE_KEYS`) — a `_DEPTH_KEYS` tuple following the same pattern is the natural extension point
- `oxe_register.py` already has a `depth_obs_keys` dict structurally present (all-`None` today) — this confirms the OXE/RLDS side was designed with depth in mind from earlier phases, it just was never populated

### Integration Points
- `hdf5_writer.py`'s env regeneration step (~line 93-100) is where `camera_depths=True` must be added — this is upstream of both `OBS_KEY_MAPPING` and the write-branch logic
- `rlds_converter.py`'s `load_episodes_from_hdf5()` → `_episode_to_rlds_steps()` → `FeaturesDict` chain is the three-point insertion path for a new depth field

</code_context>

<specifics>
## Specific Ideas

- Overhead camera stand: candidate off-the-shelf printable components found during discussion (to inform the physical build, not the sim recalibration itself):
  - [Tripod adapter 1/4-20 to 3/8" thread](https://www.printables.com/model/85177-tripod-adapter-14-20-to-38-thread)
  - [Camera Tripod Bolt Collection UNC 1/4-20](https://www.printables.com/model/489142-camera-tripod-bolt-collection-unc-14-20)
  - No exact pre-made cradle exists for the housed AR0144 module — a custom friction-fit cradle (66x30x17.72mm) must be designed/printed, following the same "custom part + off-the-shelf mechanical standard" pattern used for the roboninecom gripper (see PROJECT.md Key Decisions).
- The `eye_in_hand` angle correction is a **scope addition surfaced during this discussion**, not originally worded into CAM-01 — the user confirmed explicitly they want it fixed in this phase, using the six reference photos as the ground truth for the real mounting tilt.

</specifics>

<deferred>
## Deferred Ideas

- **Physical overhead camera mount fabrication and the real side-by-side photo comparison** (Success Criteria #1's literal verification) — deferred until the parallel physical-hardware track (tracked via `diagnostics/UAT/`) builds and photographs the actual mount. Phase 7 proceeds on a best-estimate basis; the live comparison is a follow-up, not a Phase 7 blocker.

None — discussion stayed within phase scope otherwise (the eye_in_hand angle addition was accepted into scope, not deferred).

</deferred>

---

*Phase: 7-Camera & Depth Perception*
*Context gathered: 2026-09-05*
