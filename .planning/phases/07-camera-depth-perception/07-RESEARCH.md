# Phase 7: Camera & Depth Perception - Research

**Researched:** 2026-09-05
**Domain:** MuJoCo/robosuite camera configuration (MJCF `<camera>` attributes, quaternion math), robomimic HDF5 dataset schema extension, TFDS/RLDS feature schema extension for OXE fine-tuning
**Confidence:** HIGH (mechanics, override-chain, and depth-normalization findings are verified against this repo's own source and cross-checked against upstream robosuite/openvla source) / MEDIUM (AR0144 FOV spec, eye_in_hand target quat — see Assumptions Log)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Agentview (overhead) camera recalibration**
- **D-01:** The real overhead camera (Waveshare 32695 AR0144 stereo module) has no finalized mount yet — position/FOV aren't documented in the repo (`diagnostics/PARTS_LIST.md` flags this as "not yet resolved"). Proceed with a **best-estimate recalibration now** using the AR0144 datasheet FOV and a reasonable overhead-workspace placement; do not block Phase 7 on physical hardware completion. Revisit/re-tune once the real mount is built.
- **D-02:** The overhead camera mount should be a **fixed frame/arm bolted to the desk/table edge** (clamp-style, similar to how the robot base is mounted) — static and repeatable, not a tripod or ball-head/gooseneck.
- **D-03:** No pre-made STL exists for the housed AR0144 stereo module (66x30x17.72mm body). Mount design: custom-printed friction-fit cradle + generic off-the-shelf 1/4-20 tripod hardware.
- **D-04:** Success Criteria #1 ("rendered frame visibly matches real camera's framing via side-by-side/overlay") is **deferred** until the physical overhead mount is built. Phase 7 sign-off is based on documented estimated params, not a live photo comparison.

**Depth camera scope**
- **D-05:** Depth is persisted for **`agentview` only**, not `eye_in_hand`. Matches real hardware exactly — only the overhead AR0144 is stereo/depth-capable. `OBS_KEY_MAPPING` in `hdf5_writer.py` gains an `agentview_depth` entry only; `oxe_register.py`'s `depth_obs_keys["primary"]` becomes `"agentview_depth"`, `"wrist"` stays `None`.

**Eye-in-hand (wrist) camera angle correction — added to scope during discussion**
- **D-06:** The current sim `eye_in_hand` camera (`fovy=75`, `quat="0.664463 -0.241845 0.241845 -0.664463"`) aims incorrectly relative to the real mount. Reference photos show the real IMX335 tilts down/inward toward the gripper's grasp point (where the fingers converge), not straight out ahead. Exact new `quat`/`fovy` are implementation work for planning/research to derive.
- **D-07:** Unlike the overhead camera, the wrist camera's real mount is already built and photographed — no hardware blocker. Verify directly against the reference photos.

### Claude's Discretion
- Exact new `fovy`/`quat` numeric values for both `agentview` and `eye_in_hand` — derive from AR0144 datasheet specs + estimated placement (agentview) and from the reference photos (eye_in_hand) during research/planning.
- Depth storage dtype/precision in HDF5 (float32 raw meters vs. quantized) — not discussed, left to planner. **See Pitfall 2 below — this framing assumes raw depth is already metric, which is false; the actual choice is normalized-[0,1] vs. converted-to-meters, and this research recommends normalized.**
- RLDS depth feature type (plain `Tensor`, not `Image`, per TFDS/OXE convention — already confirmed via codebase scout of `rlds_converter.py`).
- Physical stand design specifics (cradle wall thickness, clamp mechanism) — not a software planning concern beyond documenting the target mount position it produces.

### Deferred Ideas (OUT OF SCOPE)
- Physical overhead camera mount fabrication and the real side-by-side photo comparison (Success Criteria #1's literal verification) — deferred until the physical-hardware track builds and photographs the actual mount.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| CAM-01 | Sim agentview/front camera recalibrated (position, angle, FOV) to match real SOARM camera placement | Pitfall 1 (override-chain correction), Code Example 1 (fovy via `camera_attribs`), AR0144 FOV spec below |
| DEPTH-01 | Depth camera stream added to SOARM MuJoCo scene, exposed as robosuite/LIBERO observation | Already solved upstream (`camera_depths=True`, verified in `test_camera_config.py`) — Phase 7 just needs to thread the flag into `hdf5_writer.py`'s env construction (Code Example 3) |
| DEPTH-02 | Depth frames persisted alongside RGB in HDF5 dataset writer | Pitfall 2 (normalization), Code Example 3 (`_DEPTH_KEYS` write branch) |
| DEPTH-03 | Depth frames carried through RLDS converter into fine-tuning dataset | Code Example 4 (TFDS `Tensor` feature), Pitfall 4 (`load_depth` scoping) |
</phase_requirements>

## Summary

This phase touches three already-working subsystems and extends each by one field: (1) the MJCF camera element that controls `agentview`'s pose/FOV, (2) the HDF5 writer's per-key dtype write branches, and (3) the RLDS converter's TFDS feature schema. The depth *rendering* is already proven working (robosuite's `camera_depths=True` flag, exercised by `test_camera_config.py`) — this phase is a **persistence/plumbing** phase, not a rendering phase, exactly as CONTEXT.md's domain framing states.

The single highest-value finding from this research is a **correction to CONTEXT.md's own canonical-refs list**: the stated "authoritative override point" for `agentview`'s pos/quat/fovy — `bddl_base_domain.py:275-293`'s `_setup_camera()` — is dead code for every BDDL task this project actually uses. All of this project's registered SOARM tasks (`libero_goal/*.bddl` and the custom `libero_spatial_soarm/*.bddl` tasks) declare `(:domain robosuite)` under the `LIBERO_Tabletop_Manipulation` problem class, which maps to `Libero_Tabletop_Manipulation` in `libero_tabletop_manipulation.py` — and that subclass defines its **own** `_setup_camera()` that fully overrides the base class method (no `super()` call). The base class's `_setup_camera()` is only reached by problem classes that don't override it, and every one of the 6 concrete problem classes in this codebase does override it independently, each with its own hardcoded `agentview` pos/quat and no `fovy`. **The fovy addition and any pos/quat recalibration must land in `libero_tabletop_manipulation.py`'s `_setup_camera()`, not `bddl_base_domain.py`'s.**

For the eye-in-hand correction, this research derived a concrete, verified geometric method: because the gripper's root body (`soarm_gripper.xml`) attaches to `right_hand` at identity (`pos="0 0 0" quat="1 0 0 0"`), the grip point and the camera live in the *same local coordinate frame* — so a "look-at" quaternion that aims the camera's boresight at the grip site can be computed in closed form, independent of the arm's current joint pose. This gives the planner a parametrized, reusable function rather than a single guessed number (see Code Example 2). The computed candidate differs by ~44° from the current shipped quat, which is a large enough gap that the planner should treat the computed value as a *starting point* for visual iteration against the reference photos, not a final answer (see Open Questions).

For depth persistence, the decisive finding is that robosuite's `camera_depths=True` returns **normalized [0,1]** depth, not metric meters — confirmed both by the upstream `robosuite.utils.camera_utils.get_real_depth_map()` docstring/formula and by this project's own `test_camera_config.py` assertion (`depth >= 0.0 & depth <= 1.0`). Critically, this project's own Phase 5 `depth_xyz.py` module already depends on receiving **raw normalized** depth as input (its own docstring says so) and calls `get_real_depth_map()` itself internally. Persisting anything other than raw normalized float32 depth in the HDF5 would silently break that existing consumer contract if it's ever pointed at recorded (rather than live) depth.

**Primary recommendation:** Fix `agentview`'s fovy/pos/quat in `libero_tabletop_manipulation.py`'s `_setup_camera()` using `camera_attribs={"fovy": "43"}` (AR0144 vertical FOV) passed to the existing `mujoco_arena.set_camera()` call; correct `eye_in_hand`'s quat using the look-at-the-grip-site method below as a starting point, refined empirically; persist `agentview_depth` in HDF5 as raw normalized float32 `(H, W, 1)` (no meters conversion); register it in the RLDS schema as `tfds.features.Tensor(shape=(None, None, 1), dtype=np.float32)` and set `oxe_register.py`'s `depth_obs_keys["primary"] = "agentview_depth"` — but do NOT set `load_depth=True` anywhere in this phase (that's a Phase 9 fine-tuning-time flag, not a Phase 7 dataset-authoring concern).

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Camera pose/FOV recalibration (CAM-01) | Simulation / MJCF config | — | `<camera>` element attributes are pure MuJoCo scene config, set via robosuite's `MujocoArena.set_camera()` Python API at env-construction time |
| Depth rendering (DEPTH-01) | Simulation / robosuite | — | Already solved: `camera_depths=True` constructor kwarg, robosuite's offscreen renderer emits the buffer; no new rendering code needed |
| Depth persistence (DEPTH-02) | Data / Storage (HDF5 writer) | Simulation (env construction feeds it) | `hdf5_writer.py` owns the write-time schema; it must request depth from the sim (upstream) and write it (its own responsibility) |
| RLDS conversion (DEPTH-03) | Data / Storage (RLDS converter) | Training pipeline (OXE registration consumes it) | `rlds_converter.py` owns the TFDS schema; `oxe_register.py` is the bridge into the training-tier `prismatic` package, which is Phase 9's concern to activate via `load_depth=True` |

## Standard Stack

No new third-party packages are introduced by this phase — it extends three already-integrated libraries already pinned in this project.

### Core (already installed/pinned, reused as-is)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| robosuite | `1.4.0` [VERIFIED: `LIBERO/requirements.txt:16,59`] | MuJoCo scene construction, camera pose/FOV API, offscreen depth rendering | Already the project's simulation backbone; `camera_depths` kwarg and `MujocoArena.set_camera(camera_attribs=...)` are existing, tested entry points |
| h5py | (already a dependency of `hdf5_writer.py`) | HDF5 dataset writing | Already used for RGB/state persistence; adding a depth dataset is the same API |
| tensorflow_datasets | `4.9.10` [CITED: version pinned in `rlds_converter.py`'s own comments, `:294-296`] | RLDS/TFDS dataset writing | Already the project's RLDS backend from Phase 6 (TUNE-01) |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| numpy | (already pinned) | Look-at quaternion computation (Code Example 2) | Pure-numpy implementation avoids adding a `scipy` dependency — `scipy` was confirmed **not installed** in this project's local Python during this research session |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Closed-form look-at quaternion (numpy) | `scipy.spatial.transform.Rotation` | scipy is not currently a project dependency (confirmed absent from local environment); pure-numpy avoids adding one for a single quaternion computation |
| Raw normalized [0,1] depth storage | Pre-convert to metric meters via `get_real_depth_map()` before writing | Would break the existing Phase-5 `depth_xyz.py` consumer contract, which expects raw normalized depth as input and does its own metric conversion internally (Pitfall 2) |

**Installation:** None required — no new packages.

## Package Legitimacy Audit

Not applicable — this phase installs no new external packages. All libraries used (robosuite, h5py, tensorflow_datasets, numpy) are already installed/pinned dependencies exercised by prior phases.

## Architecture Patterns

### System Architecture Diagram

```
                    ┌─────────────────────────────────────────┐
                    │  libero_tabletop_manipulation.py         │
                    │  Libero_Tabletop_Manipulation             │
                    │  ._setup_camera(mujoco_arena)             │  <-- CAM-01 lands HERE,
                    │    mujoco_arena.set_camera(                │      not bddl_base_domain.py
                    │      "agentview", pos, quat,                │      (Pitfall 1)
                    │      camera_attribs={"fovy": "43"})         │
                    └───────────────────┬───────────────────────┘
                                         │ MJCF <camera> element written into scene XML
                                         ▼
                    ┌─────────────────────────────────────────┐
                    │  robosuite offscreen renderer              │
                    │  camera_names=[agentview, robot0_eye_in_hand]│
                    │  camera_depths=True  (global bool, DEPTH-01) │
                    └───────────────────┬───────────────────────┘
                                         │ env.reset()/env.step() obs dict
                                         │  {"agentview_image": uint8 (H,W,3),
                                         │   "agentview_depth": float32 (H,W,1) in [0,1],  <-- normalized! (Pitfall 2)
                                         │   "robot0_eye_in_hand_image": ...,
                                         │   "robot0_eye_in_hand_depth": ... (rendered but NOT persisted, D-05)}
                                         ▼
                    ┌─────────────────────────────────────────┐
                    │  hdf5_writer.py                            │
                    │  OBS_KEY_MAPPING += {"agentview_depth":     │  <-- DEPTH-02 lands HERE
                    │      "agentview_depth"}                     │
                    │  _DEPTH_KEYS = ("agentview_depth",)         │
                    │  write float32, shape (N,H,W,1)             │
                    └───────────────────┬───────────────────────┘
                                         │ demo_N/obs/agentview_depth dataset in HDF5
                                         ▼
                    ┌─────────────────────────────────────────┐
                    │  rlds_converter.py                         │
                    │  load_episodes_from_hdf5(): read depth      │  <-- DEPTH-03 lands HERE
                    │  _episode_to_rlds_steps(): include depth    │
                    │  FeaturesDict: agentview_depth =            │
                    │    tfds.features.Tensor(shape=(H,W,1),      │
                    │      dtype=np.float32)                      │
                    └───────────────────┬───────────────────────┘
                                         │ TFDS record on disk
                                         ▼
                    ┌─────────────────────────────────────────┐
                    │  oxe_register.py (Phase 9 activation point)│
                    │  depth_obs_keys["primary"]="agentview_depth"│
                    │  load_depth=True  <-- NOT set in Phase 7;   │
                    │    this is a finetune.py-time flag (Pitfall 4)│
                    └─────────────────────────────────────────┘
```

### Recommended Project Structure

No new files/directories — this phase edits four existing files in place:
```
LIBERO/libero/libero/envs/problems/libero_tabletop_manipulation.py  # CAM-01: agentview fovy/pos/quat
LIBERO/libero/libero/assets/robots/soarm101/robot.xml               # eye_in_hand quat/fovy correction
LIBERO/libero/libero/datasets/hdf5_writer.py                        # DEPTH-01/02: camera_depths=True, _DEPTH_KEYS
LIBERO/libero/libero/datasets/rlds_converter.py                     # DEPTH-03: depth in FeaturesDict
LIBERO/libero/libero/datasets/oxe_register.py                       # DEPTH-03: depth_obs_keys["primary"]
```

### Pattern 1: Setting `fovy` via `camera_attribs`, not a separate call

**What:** `MujocoArena.set_camera(camera_name, pos, quat, camera_attribs=None)` accepts an optional dict of additional MJCF `<camera>` attributes. If the named camera element doesn't exist yet, it's created with `**camera_attribs` merged in; if it already exists, each attribute in `camera_attribs` is applied via `camera.set(attrib, value)` on the existing XML element [VERIFIED: `robosuite/models/arenas/arena.py`, `v1.4.1` tag, full source fetched this session].

**When to use:** Any time an existing `set_camera()` call needs an extra MJCF camera attribute (fovy, znear, zfar, etc.) beyond pos/quat — no separate post-construction `sim.model.cam_fovy[cam_id] = ...` hack is needed; `camera_attribs` is the intended, idiomatic mechanism and was designed for exactly this.

**Example:**
```python
# Source: robosuite/models/arenas/arena.py set_camera() (v1.4.1, matches 1.4.0's API)
mujoco_arena.set_camera(
    camera_name="agentview",
    pos=[...],   # best-estimate overhead placement (Claude's Discretion)
    quat=[...],
    camera_attribs={"fovy": "43"},  # AR0144 vertical FOV in degrees (string, per MJCF convention)
)
```

Note the existing `eye_in_hand` camera in `robot.xml` sets `fovy` as a literal MJCF attribute because it's declared directly in the robot's XML asset (not constructed via `set_camera()` at env-build time) — that's a different code path (static XML) from `agentview`'s (constructed via Python at `_setup_camera()` time). Both are legitimate; the mechanism differs because the camera lives in a different part of the model tree, not because one is more "correct."

### Pattern 2: Look-at quaternion for a fixed local target (eye_in_hand correction)

**What:** Because `soarm_gripper.xml`'s root body (`right_gripper`) attaches to the arm's `right_hand` body at identity (`pos="0 0 0" quat="1 0 0 0"` — [VERIFIED: `soarm_gripper.xml:58`]), the gripper's `eef`/`grip_site` position (`pos="-0.052 -0.0002 0.0"` relative to `right_gripper`, i.e. also relative to `right_hand`) and the `eye_in_hand` camera's position (`pos="0.14 0 0.02"` relative to `right_hand` — [VERIFIED: `robot.xml:109`]) are both expressed in the **same local frame**, regardless of the arm's current joint configuration. This means a "look-at-the-grasp-point" quaternion can be computed once, in closed form, without needing forward kinematics or a live sim.

**When to use:** Deriving a starting-point quat for `eye_in_hand` (or any fixed-mode camera whose target is a fixed point in its parent body's frame).

**Example:**
```python
# Pure-numpy look-at quaternion (no scipy dependency — scipy confirmed not installed locally)
import numpy as np

def look_at_quat(cam_pos, target_pos, up_hint=(0.0, 0.0, 1.0)):
    """Return a MuJoCo (w,x,y,z) quat that points camera local -Z at target_pos,
    expressed in the same frame as cam_pos/target_pos (e.g. right_hand-local)."""
    fwd = np.asarray(target_pos, dtype=float) - np.asarray(cam_pos, dtype=float)
    fwd_n = fwd / np.linalg.norm(fwd)
    cam_z = -fwd_n  # MuJoCo camera convention: boresight is local -Z
    up = np.asarray(up_hint, dtype=float)
    cam_x = np.cross(up, cam_z)
    if np.linalg.norm(cam_x) < 1e-6:          # up_hint parallel to boresight — pick another hint
        up = np.array([0.0, 1.0, 0.0])
        cam_x = np.cross(up, cam_z)
    cam_x /= np.linalg.norm(cam_x)
    cam_y = np.cross(cam_z, cam_x)
    R = np.stack([cam_x, cam_y, cam_z], axis=1)  # columns = camera axes in parent frame
    # Shepperd's method, robust quat-from-matrix (see full derivation in this
    # research's verification transcript; any standard rotation-matrix-to-quaternion
    # routine works — this one avoids adding scipy).
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25*S, (R[2,1]-R[1,2])/S, (R[0,2]-R[2,0])/S, (R[1,0]-R[0,1])/S
    else:
        ...  # branch on largest diagonal element — see standard quat-from-matrix refs
    return np.array([w, x, y, z])

# Verified in this research session:
cam_pos = [0.14, 0, 0.02]          # eye_in_hand pos, robot.xml:109
grip_site = [-0.052, -0.0002, 0.0]  # eef pos, soarm_gripper.xml:68 (right_gripper attaches at identity)
q = look_at_quat(cam_pos, grip_site)
# => [0.524990, 0.473144, 0.473637, 0.525537]  (candidate starting quat — see Open Questions)
```

**Verified computation from this session** (current shipped quat vs. this look-at target):
- Current `eye_in_hand` boresight in `right_hand`-local frame: `(-0.643, 0, -0.766)` (steeply downward, ~50° off the -X/reach axis)
- Look-at-grip-site boresight: `(-0.995, -0.001, -0.104)` (nearly along -X, only ~6° downward tilt)
- Angle between the two: **~44°**

This is a large enough discrepancy that it should be treated as a hypothesis to visually validate against the reference photos, not a drop-in final answer — see Open Questions.

### Pattern 3: `_DEPTH_KEYS` write branch mirrors the existing `_RGB_KEYS`/`_STATE_KEYS` pattern

**What:** `hdf5_writer.py` already distinguishes dtype-specific write branches by key-name tuple (`_RGB_KEYS` → `uint8`, `_STATE_KEYS` → `float64`). A `_DEPTH_KEYS = ("agentview_depth",)` tuple, written as `float32`, is the natural extension — no new abstraction needed.

**Example:**
```python
# Source: this project's own hdf5_writer.py pattern, extended
OBS_KEY_MAPPING = {
    "agentview_rgb": "agentview_image",
    "eye_in_hand_rgb": "robot0_eye_in_hand_image",
    "agentview_depth": "agentview_depth",   # DEPTH-02: robosuite's own key name, no rename needed
    "gripper_states": "robot0_gripper_qpos",
    "joint_states": "robot0_joint_pos",
}
_RGB_KEYS = ("agentview_rgb", "eye_in_hand_rgb")
_DEPTH_KEYS = ("agentview_depth",)
_STATE_KEYS = ("gripper_states", "joint_states")

# env construction (~line 93-100): add camera_depths=True
regen_env = OffScreenRenderEnv(
    bddl_file_name=bddl_file_name,
    robots=list(robots),
    camera_heights=camera_size,
    camera_widths=camera_size,
    camera_depths=True,          # NEW — renders depth for BOTH agentview and eye_in_hand
                                  # (robosuite's camera_depths is a single global bool, not
                                  #  per-camera — D-05's "agentview only" is enforced by
                                  #  selective WRITING below, not selective rendering)
    has_renderer=False,
    has_offscreen_renderer=True,
)

# write branch (~line 156-170), alongside the existing _RGB_KEYS loop:
for key in _DEPTH_KEYS:
    obs_grp.create_dataset(
        key, data=np.array(obs_acc[key], dtype=np.float32)
    )
    # obs_acc[key] is a list of (H, W, 1) arrays already in [0, 1] — robosuite's
    # own normalized output. Do NOT call get_real_depth_map() here (Pitfall 2).
```

### Pattern 4: TFDS `Tensor` feature for depth, not `Image`

**What:** RLDS/OXE convention represents depth as `tfds.features.Tensor` (a plain numeric array feature), never `tfds.features.Image` (which forces PNG/JPEG encoding semantics meant for photographic RGB) [CITED: TFDS `nyu_franka_play_dataset_converted_externally_to_rlds` catalog page uses `Tensor(shape=(128,128,1), dtype=int32)` for its depth field; openvla-oft's own `transforms.py` casts depth via `tf.cast(trajectory["observation"]["depth"][..., 0], tf.float32)`, confirming a plain-tensor float32 convention]. This matches CONTEXT.md's already-locked decision.

**Example:**
```python
# Source: this project's rlds_converter.py FeaturesDict, extended (matches 06-RESEARCH.md
# Pattern 1's step_features structure)
"observation": tfds.features.FeaturesDict(
    {
        "agentview_rgb": tfds.features.Image(shape=(None, None, 3), dtype=np.uint8),
        "eye_in_hand_rgb": tfds.features.Image(shape=(None, None, 3), dtype=np.uint8),
        "agentview_depth": tfds.features.Tensor(shape=(None, None, 1), dtype=np.float32),  # NEW
        "state": tfds.features.Tensor(shape=(REQUIRED_PROPRIO_DIM,), dtype=np.float32),
    }
),
```

```python
# oxe_register.py: only "primary" changes; "wrist" stays None (D-05)
"depth_obs_keys": {"primary": "agentview_depth", "secondary": None, "wrist": None},
```

### Anti-Patterns to Avoid
- **Editing `bddl_base_domain.py`'s `_setup_camera()` and assuming it fixes `agentview` project-wide:** it doesn't reach any of this project's registered tasks (Pitfall 1). If a "belt and suspenders" fix there is still desired for documentation/future-proofing, it must be applied *in addition to*, not *instead of*, `libero_tabletop_manipulation.py`'s override.
- **Pre-converting depth to real-world meters before writing to HDF5:** breaks the existing `depth_xyz.py` consumer contract, which expects raw normalized [0,1] input and does its own `get_real_depth_map()` conversion (Pitfall 2).
- **Setting `load_depth=True` in `oxe_register.py` during Phase 7:** this flag controls whether the *fine-tuning* data loader actually materializes depth tensors into training batches — it belongs to Phase 9 (TUNE-05, re-fine-tuning), not Phase 7's dataset-authoring scope (Pitfall 4). Setting it early doesn't break anything by itself, but doing so is out of this phase's stated success criteria and risks scope creep into un-researched model-input-shape territory.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Depth rendering | A custom MuJoCo depth-buffer readback | robosuite's `camera_depths=True` | Already implemented, already tested (`test_camera_config.py`), already normalization-documented upstream |
| Depth-to-metric conversion (if ever needed downstream) | A custom near/far unprojection formula | `robosuite.utils.camera_utils.get_real_depth_map()` | Already used correctly by this project's own `depth_xyz.py`; re-deriving the near/far/extent formula independently risks a sign or scale bug |
| Quaternion-from-rotation-matrix conversion | A naive/unstable direct-arcsin formula | Shepperd's method (branch on largest diagonal term) | The naive approach is numerically unstable near singularities (e.g., 180° rotations); Shepperd's method is the standard robust approach used in robotics libraries |

**Key insight:** Every piece of "new" functionality this phase needs (depth rendering, depth-to-meters conversion, TFDS depth conventions) already has a working, tested implementation either upstream (robosuite) or in this project's own Phase 5 code (`depth_xyz.py`). The actual engineering work is *plumbing* — passing the right flag, adding the right key to the right tuple — not building anything new.

## Common Pitfalls

### Pitfall 1: `bddl_base_domain.py`'s `_setup_camera()` is dead code for this project's tasks
**What goes wrong:** A planner or implementer edits `bddl_base_domain.py:275-293` (per CONTEXT.md's stated "authoritative override point") expecting the change to affect `agentview` in the running sim, then finds no change in rendered output.
**Why it happens:** Every problem class actually registered and used by this project (`Libero_Tabletop_Manipulation`, mapped from `LIBERO_Tabletop_Manipulation` in every `.bddl` file under `libero_goal/` and `libero_spatial_soarm/` — [VERIFIED: `grep -h "define (problem"` across all task files used by `collector.py` and `libero_spatial_soarm/`, all return `LIBERO_Tabletop_Manipulation`]) defines its own `_setup_camera()` method that fully overrides the base class's, with no `super()._setup_camera()` call. Six problem-class files in `envs/problems/` each independently override `_setup_camera()` with hardcoded pos/quat (verified: `libero_tabletop_manipulation.py`, `libero_kitchen_tabletop_manipulation.py`, `libero_coffee_table_manipulation.py`, `libero_living_room_tabletop_manipulation.py`, `libero_floor_manipulation.py`, `libero_study_tabletop_manipulation.py`).
**How to avoid:** Apply the `agentview` fovy/pos/quat fix in `libero_tabletop_manipulation.py`'s `_setup_camera()` — this is the ONLY override this project's tasks actually reach. Optionally also update `bddl_base_domain.py` for documentation/future-proofing (in case Phase 8 authors tasks under a different problem domain), but that alone is insufficient.
**Warning signs:** Rendered `agentview` frames look unchanged after editing `bddl_base_domain.py`; a `grep -rn "def _setup_camera" LIBERO/libero/libero/envs/` immediately reveals the 7 definitions (1 base + 6 overrides) and should be the first debugging step if this happens.

### Pitfall 2: robosuite depth is normalized [0,1], not metric meters
**What goes wrong:** CONTEXT.md's "Claude's Discretion" section frames the HDF5 storage choice as "float32 raw meters vs. quantized," implicitly assuming the raw obs-dict depth value is already in meters. It is not.
**Why it happens:** `camera_depths=True` returns MuJoCo's normalized depth buffer directly. Converting to real-world distance requires an explicit extra call: `robosuite.utils.camera_utils.get_real_depth_map(sim, depth_map)`, which applies `near / (1.0 - depth_map * (1.0 - near/far))` using `sim.model.vis.map.zfar/znear` and `sim.model.stat.extent` [VERIFIED: `robosuite/utils/camera_utils.py`, `v1.4.1` tag, docstring + formula fetched this session; cross-confirmed by this project's own `test_camera_config.py` assertion `depth >= 0.0 & depth <= 1.0`].
**How to avoid:** Persist the raw normalized [0,1] float32 depth as-is in the HDF5 (Code Example 3). This matches what the obs dict naturally returns AND matches the existing Phase 5 `depth_xyz.py` module's documented input contract (its own docstring states: `depth_map (np.array): (H, W, 1) depth map normalized in [0, 1] (the raw MuJoCo/robosuite depth obs)` — [VERIFIED: `LIBERO/libero/libero/perception/depth_xyz.py:27-28`]). If a future consumer needs metric depth, it should call `get_real_depth_map()` itself, exactly as `depth_xyz.py` already does internally.
**Warning signs:** A depth-consuming script downstream applies its own unprojection math and produces XYZ estimates that are off by a large, consistent factor — check whether raw or converted depth was written to the HDF5 first.

### Pitfall 3: `camera_depths=True` is a single global flag, not per-camera
**What goes wrong:** Someone looks for a way to render depth for `agentview` only (matching D-05's real-hardware-fidelity intent) and can't find a per-camera toggle in `env_wrapper.py` or robosuite's `ControlEnv.__init__`.
**Why it happens:** `camera_depths` is one boolean applied uniformly across all `camera_names` [VERIFIED: `test_camera_config.py:86-109`'s `test_depth_cameras_non_degenerate_spatial` asserts BOTH `agentview_depth` and `robot0_eye_in_hand_depth` are non-degenerate when `camera_depths=True` is passed with both cameras in `camera_names`].
**How to avoid:** Don't try to suppress `eye_in_hand`'s depth rendering — let robosuite render it (harmless, matches upstream API), and enforce "agentview only" purely at the **persistence** layer: only add `agentview_depth` to `OBS_KEY_MAPPING`/`_DEPTH_KEYS`, never `eye_in_hand_depth`. This is cheaper and matches D-05's actual requirement (persisted depth, not rendered depth).
**Warning signs:** Time spent searching robosuite's API surface for a `camera_depths={"agentview": True, "robot0_eye_in_hand": False}`-style per-camera kwarg — it doesn't exist in 1.4.0.

### Pitfall 4: `load_depth=True` is a fine-tuning-time flag, not a dataset-authoring-time flag
**What goes wrong:** An implementer, trying to be thorough about "making depth work end-to-end," sets `load_depth=True` somewhere in `oxe_register.py`'s registration call during Phase 7, or worries that DEPTH-03's success criterion isn't met without it.
**Why it happens:** `prismatic.vla.datasets.rlds.oxe.materialize.py`'s `make_oxe_dataset_kwargs(..., load_depth: bool = False, ...)` pops `depth_obs_keys` from the returned kwargs entirely when `load_depth` is `False` [VERIFIED: `openvla/openvla` `main` branch, `materialize.py`, fetched this session — `if not load_depth: dataset_kwargs.pop("depth_obs_keys")`]. This flag governs whether the *training* data loader materializes depth tensors into batches — it has nothing to do with whether the TFDS record on disk *contains* a depth field.
**How to avoid:** DEPTH-03's success criterion ("a converted TFDS record includes a depth field readable by the fine-tuning data loader") is satisfied by proving the TFDS record has the field (e.g., `tfds.builder(...).as_dataset()` iterated and `agentview_depth` present per step) — this does NOT require `load_depth=True`. Leave `load_depth` unset/default in Phase 7; Phase 9's TUNE-05 (re-fine-tuning on depth-augmented observations) is the natural place to flip it to `True` and handle any resulting model-input-shape changes.
**Warning signs:** Scope creep into openvla-oft model architecture questions (how does the vision backbone consume a 4th depth channel?) during Phase 7 — that's Phase 9 territory.

### Pitfall 5: `right_gripper`'s "attaches at identity" claim is worth re-verifying at execution time
**What goes wrong:** The look-at quaternion in Code Example 2 assumes `soarm_gripper.xml`'s `right_gripper` root body is welded onto `right_hand` with exactly `pos="0 0 0" quat="1 0 0 0"` (i.e., no additional mount-time offset). This is stated in the gripper XML's own body declaration and its module docstring, but robosuite's gripper-attachment mechanism (`RobotModel.add_gripper()` / mount XML merge) is a separate code path this research did not independently trace end-to-end.
**Why it happens:** robosuite's gripper mounting sometimes applies an additional `hand_pos`/`hand_quat` offset from the *robot's* side (not the gripper's), depending on the manipulator's `gripper_mount_pos`/`eef_name` configuration — this project's own SOARM robot model would need to be checked for such an offset.
**How to avoid:** Before trusting the computed look-at quat numerically, verify empirically: render a frame with the candidate quat and check that the framing plausibly centers the gripper jaws, rather than assuming the closed-form math is final. This is exactly the validation step already recommended (visual iteration against reference photos).
**Warning signs:** Rendered `eye_in_hand` frames with the candidate quat show the jaws off-center or out of frame entirely — that's the signal to re-check the mount offset assumption rather than re-deriving the quat math.

## Code Examples

Verified patterns from this repo and upstream sources — consolidated view (also embedded above under each Pattern):

### Setting agentview fovy (CAM-01)
```python
# LIBERO/libero/libero/envs/problems/libero_tabletop_manipulation.py
def _setup_camera(self, mujoco_arena):
    mujoco_arena.set_camera(
        camera_name="agentview",
        pos=[...],          # best-estimate overhead placement — see Assumptions Log
        quat=[...],
        camera_attribs={"fovy": "43"},  # AR0144 vertical FOV
    )
    # ... existing frontview/galleryview calls unchanged
```

### eye_in_hand quat correction starting point (CAM-01 scope addition)
```python
# LIBERO/libero/libero/assets/robots/soarm101/robot.xml:109
# candidate quat from look_at_quat(cam_pos=[0.14,0,0.02], target=[-0.052,-0.0002,0.0]):
<camera mode="fixed" name="eye_in_hand" pos="0.14 0 0.02"
        quat="0.524990 0.473144 0.473637 0.525537" fovy="75"/>
<!-- fovy unchanged unless the reference photos also suggest a FOV mismatch;
     quat is a STARTING candidate — validate against reference photos before finalizing -->
```

### HDF5 depth write branch (DEPTH-01/02)
See Pattern 3 above (full code block).

### RLDS depth Tensor feature (DEPTH-03)
See Pattern 4 above (full code block).

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|---------------|--------|
| N/A — this phase extends existing, already-current patterns | — | — | No superseded approach; robosuite's `camera_depths`/`get_real_depth_map` and TFDS `Tensor`-for-depth have been stable conventions since well before this project's robosuite 1.4.0 pin |

**Deprecated/outdated:** None identified — all researched mechanisms are the current, actively-used conventions in both robosuite 1.4.x and the openvla-oft/OXE ecosystem.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | AR0144 vertical FOV = 43° (from Waveshare product listing + fabtolab.com cross-check; direct fetch of waveshare.com blocked by 403 during this session, relied on WebSearch snippets from two independent listings that agree with each other and with this project's own already-documented 52mm baseline) | Standard Stack, Code Example 1 | If wrong, `agentview`'s recalibrated FOV is off — but D-04 already defers the live photo-comparison verification, so this is correctable in the deferred follow-up without blocking Phase 7 sign-off |
| A2 | Best-estimate `agentview` overhead pos/quat placement (numeric values not yet chosen — Claude's Discretion per CONTEXT.md) | Code Example 1 | Low — D-01 explicitly frames this as a best-estimate to be revisited once the physical mount exists |
| A3 | The look-at-the-grip-site candidate quat (`0.524990 0.473144 0.473637 0.525537`) is the CORRECT target for the real mount's tilt angle | Pattern 2, Code Example 2 | Medium — this is a geometrically-derived hypothesis (verified math, unverified against the actual photographed tilt angle); the ~44° gap from the current shipped quat means getting the sign/magnitude wrong would visibly mis-frame the wrist view. Mitigated by treating it as a starting point for visual iteration (matches the precedent already set in Phase 2 plan 02-04, per `robot.xml`'s own comment: "tuned in plan 02-04") |
| A4 | `soarm_gripper.xml`'s `right_gripper` root body attaches to `right_hand` at exactly identity with no additional robosuite-side mount offset (Pitfall 5) | Pattern 2 | Medium — if robosuite's gripper-mount mechanism applies an additional offset, the look-at quat's target coordinates would be wrong; mitigated by recommending empirical visual verification before finalizing |
| A5 | robosuite `v1.4.1` GitHub tag's `arena.py`/`camera_utils.py` source is representative of the project's pinned `1.4.0` (a fetched-from-GitHub tag one patch version ahead of the pin, since 1.4.0 isn't separately browsable via raw GitHub in the same way) | Standard Stack, Pattern 1, Pitfall 2 | Low — `set_camera`/`get_real_depth_map` are stable, long-standing APIs; no changelog evidence of a behavioral change between 1.4.0 and 1.4.1 for these specific functions |

## Open Questions

1. **Does the look-at-the-grip-site quat actually match the real mount's tilt, or does the real photo show a shallower/steeper angle?**
   - What we know: the current shipped quat's boresight is ~50° off the reach axis (very steep downward tilt); the closed-form look-at-grip-site target is only ~6° off the reach axis (nearly horizontal); these differ by ~44°.
   - What's unclear: which of these — or something in between — actually matches what the 4 available reference photos show. Photogrammetric angle estimation from the photos wasn't attempted in this research pass (would require identifying corresponding points/known dimensions in the images); the photos were reviewed qualitatively.
   - Recommendation: the planner should schedule a render-and-compare-against-reference-photos verification step (mirroring Phase 2 plan 02-04's original empirical tuning) rather than treating either quat as final. Start from the look-at candidate (it's grounded in known mount geometry, unlike a pure guess) and adjust the `up_hint` parameter or add a small extra rotation if the rendered framing doesn't match.

2. **Should `bddl_base_domain.py`'s `_setup_camera()` also be updated for consistency/documentation, even though it's not reached by current tasks?**
   - What we know: it's dead code today (Pitfall 1).
   - What's unclear: whether Phase 8's new benchmark tasks (BENCH-02) will register under a problem domain that DOES reach the base class method, or whether they'll also use `LIBERO_Tabletop_Manipulation` (in which case they'd inherit `libero_tabletop_manipulation.py`'s fix automatically).
   - Recommendation: fix `libero_tabletop_manipulation.py` (required); optionally also fix `bddl_base_domain.py` as defensive documentation (low cost, protects Phase 8 if it introduces a new problem domain) — leave the decision to the planner/task-breakdown level.

3. **Two of CONTEXT.md's six listed reference-photo filenames don't exist on disk.**
   - What we know: `progress-documentation/images/20260827_121104-2.jpg` does not exist; the actual files present are `20260827_121004-2.jpg`, `20260827_121029-2.jpg`, `20260827_121030-2.jpg`, `20260827_121035-2.jpg`, `20260827_121037-2.jpg`, `20260827_121100-2.jpg`, `20260827_121106-2.jpg`, `20260827_121114-2.jpg` (8 files, not the 6 named in CONTEXT.md).
   - What's unclear: whether CONTEXT.md's filenames were slightly mistyped or refer to since-renamed/removed files.
   - Recommendation: the planner/executor should use the actual 8 files present in `progress-documentation/images/` matching the `20260827_121*` pattern as the reference set, not the exact 6 filenames listed in CONTEXT.md.

## Environment Availability

| Dependency | Required By | Available (this research session, local machine) | Version | Fallback |
|------------|------------|-----|---------|----------|
| robosuite | Depth rendering, camera API (all of CAM-01/DEPTH-01) | ✗ (not installed in the shell used for this research; project uses a dedicated `libero` conda env per STATE.md, not probed directly this session) | 1.4.0 (pinned) | Behavioral verification (actual render calls) must happen in the project's `libero` conda env / Colab, not this research session — matches this project's existing pattern (`rlds_converter.py`'s own comments note tensorflow_datasets is "Colab-only, not available in this project's local libero conda env") |
| tensorflow_datasets | RLDS conversion (DEPTH-03) | ✗ (confirmed absent per existing code comments in `rlds_converter.py`) | 4.9.10 (per code comment) | Colab-only, as already established by Phase 6 |
| h5py | HDF5 writing (DEPTH-02) | Not independently re-verified this session (already a working project dependency per `hdf5_writer.py`'s existing, tested use) | — | — |
| scipy | Would simplify look-at quat math | ✗ (confirmed absent via `ModuleNotFoundError` this session) | — | Pure-numpy implementation provided (Code Example 2) — no scipy dependency needed |

**Missing dependencies with no fallback:** None — all missing tools have an established fallback (Colab / project conda env) already used by prior phases.

**Missing dependencies with fallback:** robosuite and tensorflow_datasets behavioral verification must happen in the project's existing `libero` conda env or Colab, exactly as established by Phases 4-6.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (no `pytest.ini`/`conftest.py` found at repo root; relies on pytest's "prepend" import-mode auto-discovery, matching the documented pattern in `test_camera_config.py`'s own module docstring) |
| Config file | none — see Wave 0 |
| Quick run command | `pytest LIBERO/libero/libero/envs/test_camera_config.py -x` |
| Full suite command | `pytest LIBERO/libero/libero/envs/ LIBERO/libero/libero/datasets/ -x` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| CAM-01 | `agentview` renders with the new fovy/pos/quat applied (non-degenerate frame, correct camera intrinsics) | integration | `pytest LIBERO/libero/libero/envs/test_camera_config.py::test_rgb_cameras_non_degenerate_spatial -x` (extend with an explicit `sim.model.cam_fovy` assertion) | ✅ (extend existing) |
| CAM-01 | `eye_in_hand` renders with corrected quat (qualitative — jaws visible/centered) | manual-only | N/A — visual inspection against reference photos (justification: framing correctness is not a numeric assertion this project can automate without ground-truth photogrammetry) | — |
| DEPTH-01 | `agentview_depth` obs key present, non-degenerate, shape `(H,W,1)`, values in `[0,1]` | integration | `pytest LIBERO/libero/libero/envs/test_camera_config.py::test_depth_cameras_non_degenerate_spatial -x` (already exists, already passes per this session's read) | ✅ (already exists) |
| DEPTH-02 | HDF5 file contains `demo_N/obs/agentview_depth` with correct dtype/shape per timestep | integration | `pytest LIBERO/libero/libero/datasets/test_hdf5_writer.py -x` (extend `test_schema_and_obs_key_naming` with a depth assertion) | ❌ Wave 0 — extend existing test |
| DEPTH-03 | Converted TFDS record includes a readable `agentview_depth` field | integration (Colab-only, `pytest.importorskip("tensorflow_datasets")`) | `pytest LIBERO/libero/libero/datasets/test_rlds_converter.py -x` (extend `test_hdf5_to_rlds_writes_tfds_loadable_dataset`) | ❌ Wave 0 — extend existing test |
| DEPTH-03 | `oxe_register.py`'s `depth_obs_keys["primary"]` correctly set to `"agentview_depth"` | unit | `pytest LIBERO/libero/libero/datasets/test_oxe_register.py -x` (extend `test_register_soarm_spatial_injects_expected_dict_shape`) | ❌ Wave 0 — extend existing test |

### Sampling Rate
- **Per task commit:** `pytest LIBERO/libero/libero/envs/test_camera_config.py -x` (fast, local, no GPU needed)
- **Per wave merge:** Full suite command above, plus a manual visual-comparison step for the two camera-angle corrections (no automated oracle exists for "does this look right")
- **Phase gate:** Full suite green before `/gsd-verify-work`; visual comparison against reference photos documented (screenshots or a short note) since D-04 defers the *real*-camera comparison but the *reference-photo* comparison for `eye_in_hand` is NOT deferred (D-07: real mount already built/photographed)

### Wave 0 Gaps
- [ ] `test_hdf5_writer.py` — extend `test_schema_and_obs_key_naming` (or add a new test) to assert `demo_1/obs/agentview_depth` exists with `dtype==float32`, `shape[-1]==1`, values in `[0,1]`
- [ ] `test_rlds_converter.py` — extend `validate_episode_arrays` (or add a parallel depth-validation function) to check `agentview_depth`'s dtype/shape/finite-values before RLDS write, matching the existing fail-loud pattern for RGB/state/actions
- [ ] `test_oxe_register.py` — extend `test_register_soarm_spatial_injects_expected_dict_shape` to assert `depth_obs_keys == {"primary": "agentview_depth", "secondary": None, "wrist": None}`
- [ ] `test_camera_config.py` — add an explicit `sim.model.cam_fovy[agentview_cam_id] == <recalibrated value>` assertion (currently the test only checks non-degeneracy, not the specific fovy value)

*(Two pre-existing, unrelated test failures noted in STATE.md's Blockers/Concerns — `test_schema_and_obs_key_naming`'s stale `gripper_states.shape[1]==1` assertion, and a `test_replay.py` pixel-mismatch — touch the same files this phase modifies. STATE.md already flags these as "candidate cleanup for Phase 7." Not in scope of this research, but worth the planner deciding explicitly whether to fix them in the same PR since the diff will touch adjacent lines anyway.)*

## Security Domain

`security_enforcement: true`, `security_asvs_level: 1` per `.planning/config.json`.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | No | This phase has no auth surface — local sim/dataset code only |
| V3 Session Management | No | N/A |
| V4 Access Control | No | N/A |
| V5 Input Validation | Yes | Extend the existing fail-loud validation pattern (`rlds_converter.py`'s `validate_episode_arrays`, `hdf5_writer.py`'s dtype-specific write branches) to the new depth field — reject non-finite, wrong-shape, or wrong-dtype depth arrays before writing, exactly as the existing RGB/state/action arrays are validated. This is a direct, mechanical extension of an already-established pattern, not new security design. |
| V6 Cryptography | No | N/A — no secrets, no crypto in this phase's scope |

### Known Threat Patterns for this stack
Not applicable in the traditional sense (no network-facing surface, no user input, no auth) — the only "input validation" concern is data-integrity validation of sim-generated arrays before they're persisted, which is already covered by V5 above and by this project's own established `validate_episode_arrays`/fail-loud conventions (ASVS V5, extended not newly designed).

## Sources

### Primary (HIGH confidence)
- `LIBERO/libero/libero/envs/bddl_base_domain.py`, `libero_tabletop_manipulation.py` (+ 5 sibling problem files), `robot.xml`, `env_wrapper.py`, `test_camera_config.py`, `soarm_gripper.xml`, `hdf5_writer.py`, `rlds_converter.py`, `oxe_register.py`, `depth_xyz.py` — read directly this session
- `diagnostics/PARTS_LIST.md` — read directly this session, camera part specs and "not yet resolved" mount gap
- `progress-documentation/images/20260827_121*.jpg` (4 of 8 present files viewed directly this session) — reference photos for eye_in_hand tilt

### Secondary (MEDIUM confidence — WebFetch/WebSearch cross-checked against official/upstream source)
- [robosuite `arena.py` (`v1.4.1` tag)](https://raw.githubusercontent.com/ARISE-Initiative/robosuite/v1.4.1/robosuite/models/arenas/arena.py) — `set_camera()` full source, fetched and read this session
- [robosuite `camera_utils.py` (`v1.4.1` tag)](https://raw.githubusercontent.com/ARISE-Initiative/robosuite/v1.4.1/robosuite/utils/camera_utils.py) — `get_real_depth_map()` formula, fetched and read this session
- [openvla `materialize.py` (main branch)](https://raw.githubusercontent.com/openvla/openvla/main/prismatic/vla/datasets/rlds/oxe/materialize.py) — `load_depth`/`depth_obs_keys` handling, fetched and read this session
- [Waveshare AR0144 Stereo USB Camera (A) product page](https://www.waveshare.com/ar0144-stereo-usb-camera-a.htm) — FOV spec (74°D/65°H/43°V), via WebSearch snippet (direct WebFetch returned HTTP 403)
- [fabtolab.com AR0144 listing](https://www.fabtolab.com/waveshare-32695-ar0144-2mp-stereo-usb-camera-module-ar0144-sensor-chip-usb2-port-synchronized-same-frame-output-52mm-baseline-featuring-distortion-free-lenses) — independent cross-check of the same FOV/focal-length/baseline spec, fetched and read this session
- [TFDS `nyu_franka_play_dataset_converted_externally_to_rlds` catalog page](https://www.tensorflow.org/datasets/catalog/nyu_franka_play_dataset_converted_externally_to_rlds) — precedent for `Tensor`-typed depth feature, via WebSearch

### Tertiary (LOW confidence)
- None used as authoritative — all WebSearch-sourced claims were cross-checked against at least one independent source or this project's own codebase before being cited above.

## Metadata

**Confidence breakdown:**
- Camera override-chain finding (Pitfall 1): HIGH — verified by direct `grep`/read of every problem file and every `.bddl` task file this project uses
- robosuite `set_camera`/`camera_attribs`/depth-normalization mechanics: HIGH — fetched and read actual upstream source this session, cross-confirmed against this project's own test assertions
- AR0144 FOV spec: MEDIUM — two independent product listings agree, but neither was fetched from an official datasheet PDF (waveshare.com direct fetch blocked)
- eye_in_hand target quat: MEDIUM — geometry/method is HIGH confidence (verified computation from this repo's own shipped values), but the specific numeric target's match to the real mount's actual tilt angle is unverified against the photos with precision (Open Question 1)
- openvla-oft `load_depth`/TFDS Tensor conventions: HIGH — fetched and read actual upstream `openvla/openvla` main-branch source this session

**Research date:** 2026-09-05
**Valid until:** 30 days for the robosuite/TFDS mechanics (stable, pinned-version APIs); the AR0144 FOV spec and eye_in_hand quat should be re-validated the moment the physical overhead mount exists or better reference photos/measurements are available (this is already tracked as a deferred follow-up per D-04).
