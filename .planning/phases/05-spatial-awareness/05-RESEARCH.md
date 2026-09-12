# Phase 5: Spatial Awareness - Research

**Researched:** 2026-08-09
**Domain:** Multi-camera VLA input wiring, MuJoCo depth-buffer-to-3D perception, BDDL spatial-relation success predicates
**Confidence:** MEDIUM-HIGH (core mechanisms verified against installed code / official source repos; VLA-checkpoint-specific behavior partially assumption-flagged pending live Colab confirmation)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Multi-Camera → VLA Wiring (SPAT-01, SPAT-02)**
- **D-01:** Both wrist (`robot0_eye_in_hand`) and overhead (`agentview`) views must actually reach the VLA model as input during inference — a literal reading of SPAT-02, not just "views are captured/logged." `LIBERO/libero/libero/vla/eval_loop.py` currently only builds `images = {"eye_in_hand": obs[camera_name]}`; must be extended to include the overhead view too.
- **D-02 (Claude's discretion):** How each backend (pi0 in `pi0_backend.py`, OFT in `oft_backend.py`) consumes two images — use the model's native multi-view input if the checkpoint supports it; if a specific checkpoint genuinely only accepts one image, fall back to resize+concat/tile so both views still reach the model. Research each backend's actual trained input format before choosing.

**Depth & XYZ Pipeline (SPAT-03, SPAT-04)**
- **D-03 (real-robot-faithful, not privileged-state shortcut):** Object XYZ positions are computed by back-projecting the extracted depth buffer using camera intrinsics/extrinsics — the same computation a real depth camera would require. Chosen deliberately over reading `sim.data` directly, because this code is meant to transfer to the physical SOARM arm later.
- **D-04:** The depth→XYZ pipeline must be validated against MuJoCo's privileged sim-state ground truth as a correctness check (unit/integration tests comparing depth-derived XYZ to `sim.data` ground truth). Sim-state ground truth is used internally for this validation only — not the pipeline's primary output.
- **D-05 (locked, do not confuse with D-03):** Task success/failure (pass/fail) scoring in BDDL predicates uses MuJoCo's privileged sim-state ground truth, NOT the depth-derived estimate. Matches how every other existing LIBERO task is scored; decouples predicate correctness from depth-pipeline accuracy. The depth→XYZ pipeline still runs and produces its own estimates (useful for perception/benchmarking and D-04's validation) — it just isn't the referee for pass/fail.

**Spatial BDDL Task Design (SPAT-05)**
- **D-06 (Claude's discretion):** Reuse Phase 4's validated scene/objects (`put_the_cream_cheese_in_the_bowl` task) vs. author a new scene — pick whichever is fastest to validate correctly, respecting Phase 4's locked embodiment constraints: objects ≲84mm (gripper jaw ceiling), all object-init/place-targets within ~0.45m arm reach, nothing on the forward centerline (collision corridor). Apply the same empirical embodiment-safety checks Phase 4 established (`actuator_force`/`qfrc_bias` + `sim.data.contact`, not assumptions).
- **D-07:** The 3+ spatial tasks should cover **left/right, near/far, and between** relations.

**Spatial Success Predicates**
- **D-08 (Claude's discretion on exact margin):** Spatial relation checks use threshold/tolerance bands, not exact-coordinate comparison — matches LIBERO's own existing predicate convention.
- **D-09:** Ambiguous/borderline spatial relations should be avoided at task-design time via generous margins — no tie-breaking predicate logic needed.

### Claude's Discretion (summary)
- D-02: Per-backend strategy for consuming two camera views (native multi-view vs. concat/tile fallback).
- D-06: Reuse Phase 4's scene/objects vs. author a new one for the spatial tasks.
- D-08: Exact tolerance/margin size for spatial predicates.

### Deferred Ideas (OUT OF SCOPE)
- **Real-hardware depth camera transfer** — when the physical SOARM arm is built (PHYS-01..03, v2 backlog), this phase's depth→XYZ pipeline is the code that will need a real depth camera selection, mounting, and calibration. No work happens on this now.
- **Third camera angle for occlusion robustness** — informative answer given during discussion, but SPAT-01 already locks the 2-camera scope. Not adopted here.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| SPAT-01 | At least 2 camera views (wrist + overhead) configured in LIBERO SOARM environments | Already satisfied by `env_wrapper.py`'s existing `camera_names=["agentview","robot0_eye_in_hand"]` default — no change needed here; verify `camera_depths` flip doesn't disturb it (see Architecture Patterns) |
| SPAT-02 | All camera views passed as input to the VLA during inference | `eval_loop.py`'s `images` dict construction + both backends' native dual-image APIs (see Multi-Camera → VLA Wiring pattern below) |
| SPAT-03 | MuJoCo depth buffer frames extracted alongside RGB frames | `camera_depths=True` flip in `ControlEnv.__init__` kwargs; robosuite emits `{cam}_depth` obs keys automatically (verified locally, see Depth Extraction pattern) |
| SPAT-04 | Object XYZ positions extracted from MuJoCo state, available as structured context | `robosuite.utils.camera_utils` back-projection pipeline (D-03) + `sim.data.body_xpos` ground-truth validation (D-04) — see Depth→XYZ Pipeline pattern |
| SPAT-05 | At least 3 BDDL tasks use spatial language prompts with correctly-evaluating success predicates | New `BinaryAtomic`/relation predicate classes registered into `VALIDATE_PREDICATE_FN_DICT`, BDDL goal-state authoring within Phase 4's embodiment envelope — see Spatial BDDL Predicates pattern |

</phase_requirements>

## Summary

This phase extends three already-existing, well-understood seams rather than building new infrastructure from scratch: (1) `eval_loop.py`'s single-camera `images` dict, (2) `env_wrapper.py`'s already-declared-but-unused `camera_depths` flag, and (3) LIBERO's `bddl.parsing` + `VALIDATE_PREDICATE_FN_DICT` predicate registry. All three extension points were located and read directly in this session; none require new external dependencies.

The most consequential research finding is that **both target VLA backends already have native multi-camera support at the checkpoint/API level** — this was not knowable from the codebase alone and required external verification. OpenVLA-OFT's official inference helper (`experiments/robot/openvla_utils.py::get_vla_action`) processes a `full_image` (third-person) and one or more `wrist_image` entries through the HF processor independently, then concatenates their `pixel_values` tensors along the token dimension before calling `predict_action`, gated by a `num_images_in_input` flag; the project's own checkpoint (`moojink/openvla-7b-oft-finetuned-libero-spatial`) documents a `num_images_in_input=2` quick-start example using exactly this full+wrist pattern. Separately, openpi's LIBERO policy (`Physical-Intelligence/openpi/src/openpi/policies/libero_policy.py`) — the exact server this project's `Pi0Backend` already talks to — natively expects **two distinct** obs keys, `observation/image` (base/third-person) and `observation/wrist_image` (wrist), with an `image_mask` marking padding vs. real views. This project's `Pi0Backend.predict()` is *already* sending both keys, just duplicating the same eye-in-hand frame into both (a Phase 3 placeholder, documented in that file's own docstring) — Phase 5's D-02 work is to stop duplicating, not to invent a new interface.

The second major finding, from reading `LIBERO/libero/libero/envs/problems/libero_tabletop_manipulation.py` directly, is that **LIBERO's existing goal-predicate dispatcher (`_eval_predicate`) only supports unary (2-token) and binary (3-token) predicate tuples** — there is no ternary dispatch path for a natural `(Between obj ref1 ref2)` predicate. The lowest-risk way to satisfy D-07's "between" relation without touching that dispatcher is to express "between" as a conjunction of two binary relations in the BDDL `:goal` block (e.g., `(And (LeftOfX obj ref_right) (RightOfX obj ref_left))`), reusing new but purely-binary predicate classes. This keeps the implementation inside the existing, tested extension point (`VALIDATE_PREDICATE_FN_DICT` + `update_predicate_fn_dict`) instead of requiring a parser/dispatcher change.

Third, MuJoCo/robosuite already ships the exact back-projection math D-03 needs — `robosuite.utils.camera_utils.get_camera_intrinsic_matrix`, `get_camera_extrinsic_matrix`, `get_real_depth_map`, and `transform_from_pixels_to_world` — confirmed present in the project's actually-installed robosuite 1.4.1 (`libero` conda env). This is a clear Don't-Hand-Roll: no custom projective-geometry code is needed, only calling these four functions with the observation dict's `{cam}_image`/`{cam}_depth` arrays and the object pixel location (from a segmentation mask or a known object-of-interest pixel).

**Primary recommendation:** Reuse robosuite's `camera_utils` for the depth→XYZ pipeline (Don't Hand-Roll), route both camera views through each backend's already-native dual-image API (no tile/concat fallback needed for either backend), and implement all three new spatial relations as new binary `BinaryAtomic` predicate subclasses registered in the existing predicate dict — with "between" decomposed into two binary predicates ANDed in BDDL rather than requiring a ternary-predicate code change.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Multi-camera RGB+depth capture | Simulation Environment (`env_wrapper.py` / robosuite / MuJoCo) | — | Camera config and rendering happen entirely inside `ControlEnv`'s robosuite construction; no other tier touches raw pixel/depth buffers |
| Multi-camera → VLA image wiring | Eval/Orchestration (`eval_loop.py`) | VLA Backend (`oft_backend.py`, `pi0_backend.py`) | `eval_loop.py` owns building the `images` dict from obs; each backend owns how it maps that dict into its own model call (per the existing `VLABackend` contract's per-backend-normalization design) |
| Depth buffer → real-world XYZ | Perception/Processing (new module) | Simulation Environment (source depth array) | A new, real-hardware-faithful module consumes `sim`, camera name, and the depth/RGB obs arrays; it must not reach into `sim.data.body_xpos` for its primary output (that would violate D-03) |
| Ground-truth validation of XYZ pipeline | Perception/Processing (test code) | Simulation Environment (`sim.data.body_xpos` as oracle) | D-04's correctness check is explicitly allowed to use privileged state — but only as a test oracle, not as the pipeline's return value |
| BDDL spatial success predicates | Task/Domain (`envs/predicates/`, `envs/problems/*.py`) | Simulation Environment (`sim.data.body_xpos` via `ObjectState.get_geom_state()`) | Predicates already read ground truth through the existing `ObjectState` abstraction — this satisfies D-05 by construction, no new plumbing needed |
| Spatial BDDL task authoring (regions, init, goal) | Task/Domain (`bddl_files/`) | — | Pure data files; must respect Phase 4's embodiment envelope, enforced by the same problem-class placement-initializer machinery already in use |

## Standard Stack

### Core (already pinned, no new packages this phase)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| robosuite | 1.4.0 pinned (`1.4.1` observed installed locally in the `libero` conda env — patch-level drift, not a new dependency) | MuJoCo env wrapper, camera rendering, `camera_utils` projection helpers | Already the project's simulation backbone since Phase 2; `camera_utils` module ships in-tree, no separate install |
| MuJoCo | 2.3.7 (via robosuite) | Physics + offscreen rendering, depth buffer source | Already pinned project-wide |
| bddl | 1.0.1 | BDDL task-file parsing (`bddl.parsing.scan_tokens`) | Already pinned; predicate registry extension uses only `libero`'s own wrapper code, not new `bddl` package APIs |
| numpy | project-pinned (`1.22.4`/`>=1.21.0`) | Matrix math for intrinsics/extrinsics/back-projection | Already used throughout `camera_utils.py` and LIBERO predicate code |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| openvla-oft (moojink fork, `--no-deps`) | pinned per 01-close install contract | Multi-image `get_vla_action`/`predict_action` reference pattern | Only the *pattern* (processor-per-image + `torch.cat` pixel_values) needs replicating inside `oft_backend.py`; the package itself is already installed per Phase 1 |
| openpi_client | pinned per Phase 3 install | `observation/image` + `observation/wrist_image` dual-key obs dict | Already used by `pi0_backend.py`; no interface change needed, only real (non-duplicated) image content |

**No new packages are installed by this phase.** All capabilities (camera projection math, multi-image VLA input, spatial predicates) are achievable through already-pinned, already-installed dependencies plus new first-party Python modules under `LIBERO/libero/libero/`.

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `robosuite.utils.camera_utils` back-projection | Hand-rolled pinhole-camera math | Reinvents exactly what robosuite already provides and tests; higher bug risk in intrinsics/extrinsics sign conventions (MuJoCo's camera axis convention is notoriously easy to get backwards — robosuite's `camera_axis_correction` matrix in `get_camera_extrinsic_matrix` exists precisely because of this) |
| Native dual-image VLA input (both backends) | Manual tile/concat of two images into one frame before sending to either backend | D-02 explicitly names this as the fallback only if a checkpoint "genuinely only accepts one image" — research shows both checkpoints already accept two; tiling would silently degrade both models below their trained input distribution |
| Decomposing "between" into 2 binary predicates | Extending `_eval_predicate` to a 4-token ternary form | Ternary extension touches a shared dispatcher used by every existing/future LIBERO task; the binary-decomposition approach is fully additive (new predicate classes only) and mirrors the existing binary-only predicate convention |

**Installation:** None required — no `pip install` steps for this phase.

**Version verification:** `robosuite==1.4.1` (confirmed via `python -c "import robosuite; print(robosuite.__version__)"` in the project's local `libero` conda env, `/opt/homebrew/Caskroom/miniconda/base/envs/libero`) `[VERIFIED: local package inspection]`. `bddl==1.0.1` and `robosuite==1.4.0` are the values pinned in `LIBERO/requirements.txt` `[VERIFIED: local file read]` — the 1.4.0→1.4.1 drift is a pre-existing local-env patch difference, not something this phase changes.

## Package Legitimacy Audit

**Not applicable this phase — no new external packages are introduced.** All work uses already-installed, already-verified dependencies (robosuite, MuJoCo, bddl, openvla-oft, openpi_client — all installed and legitimacy-checked in Phases 1–3) plus new first-party Python modules. No `package-legitimacy check` run was needed.

## Architecture Patterns

### System Architecture Diagram

```
                         ┌─────────────────────────────┐
                         │  BDDL task file (regions,    │
                         │  init state, goal predicates)│
                         └──────────────┬───────────────┘
                                        │ parsed at env construction
                                        ▼
┌───────────────────────────────────────────────────────────────────┐
│ ControlEnv / OffScreenRenderEnv (env_wrapper.py)                    │
│  camera_names=["agentview","robot0_eye_in_hand"]                    │
│  camera_depths=True  ◄── SPAT-03 flip                               │
│                                                                       │
│  robosuite → MuJoCo render() ──► obs["agentview_image"]   (RGB)     │
│                              ──► obs["agentview_depth"]   (0..1)    │
│                              ──► obs["robot0_eye_in_hand_image"]     │
│                              ──► obs["robot0_eye_in_hand_depth"]     │
└───────────────────┬───────────────────────────┬─────────────────────┘
                    │ obs dict, every env.step()/reset()               │
                    ▼                                                   ▼
┌────────────────────────────────┐          ┌─────────────────────────────────┐
│ eval_loop.py::run_episode        │          │ Depth→XYZ module (NEW)           │
│  images = {                      │          │  get_camera_intrinsic_matrix()   │
│    "eye_in_hand": obs[wrist_img],│          │  get_camera_extrinsic_matrix()   │
│    "agentview": obs[overhead_img]│◄─SPAT-02 │  get_real_depth_map()            │
│  }                                │          │  transform_from_pixels_to_world()│
│  backend.predict(images, lang)   │          │  -> object XYZ (structured data) │
└───────────────┬───────────────────┘          └───────────────┬───────────────────┘
                │                                                │ D-04: validated against
                ▼                                                ▼
┌─────────────────────────────┐              ┌──────────────────────────────────┐
│ OFTBackend / Pi0Backend       │              │ sim.data.body_xpos (ground truth) │
│  both images → native dual-   │              │ via ObjectState.get_geom_state()  │
│  image checkpoint API          │              └──────────────────┬─────────────────┘
│  (D-02: no tile/concat needed)│                                  │ D-05: pass/fail referee
└───────────────────────────────┘                                  ▼
                                              ┌────────────────────────────────────┐
                                              │ Spatial predicates (NEW classes)     │
                                              │  LeftOfX / RightOfX / NearTo / FarFrom│
                                              │  registered in                        │
                                              │  VALIDATE_PREDICATE_FN_DICT           │
                                              │  "between" = And(LeftOfX, RightOfX)   │
                                              │  evaluated by _eval_predicate()        │
                                              │  (binary dispatch only)                │
                                              └────────────────────────────────────┘
```

### Recommended Project Structure
```
LIBERO/libero/libero/
├── envs/
│   ├── predicates/
│   │   ├── base_predicates.py       # add new BinaryAtomic spatial predicate classes here
│   │   └── __init__.py              # register new predicates in VALIDATE_PREDICATE_FN_DICT
│   └── problems/
│       └── libero_tabletop_manipulation.py  # _eval_predicate — verify arity assumptions here, do not need to touch for the binary-decomposition approach
├── perception/                       # NEW module (name is a planning decision, not locked)
│   ├── __init__.py
│   ├── depth_xyz.py                 # wraps robosuite.utils.camera_utils for D-03/D-04
│   └── test_depth_xyz.py            # D-04's sim.data ground-truth comparison tests
├── vla/
│   ├── eval_loop.py                 # extend images dict construction (D-01)
│   ├── oft_backend.py               # extend predict() for 2-image processor+concat pattern (D-02)
│   └── pi0_backend.py               # stop duplicating eye_in_hand into both obs keys (D-02)
└── bddl_files/
    └── libero_spatial/ (or a new libero_spatial_soarm/ dir)
        ├── <task>_left_right.bddl
        ├── <task>_near_far.bddl
        └── <task>_between.bddl
```

### Pattern 1: Multi-Camera → VLA Wiring (D-01, D-02)

**What:** Extend `eval_loop.py`'s `images` dict to include both camera views, then let each backend consume both via its already-native multi-image API.

**When to use:** Any VLA inference call in this phase's eval loop, and any future backend addition (the `VLABackend` Protocol already anticipates this — Phase 3's `interface.py` docstring explicitly says "later phases may add more keys without changing this signature").

**eval_loop.py change:**
```python
# Before (Phase 3, single-view):
images = {"eye_in_hand": obs[camera_name]}

# After (Phase 5, D-01):
images = {
    "eye_in_hand": obs["robot0_eye_in_hand_image"],
    "agentview": obs["agentview_image"],
}
```

**OFTBackend.predict() pattern (D-02), derived from moojink/openvla-oft's own `get_vla_action` reference implementation:**
```python
# Source: github.com/moojink/openvla-oft experiments/robot/openvla_utils.py (get_vla_action),
# adapted to this project's images-dict convention.
prompt = f"In: What action should the robot take to {language}?\nOut:"

primary = Image.fromarray(images["eye_in_hand"])
wrist_views = [Image.fromarray(images["agentview"])]  # additional views beyond primary

primary_inputs = self.processor(prompt, primary).to(self.device, dtype=torch.bfloat16)
extra_inputs = [
    self.processor(prompt, img).to(self.device, dtype=torch.bfloat16)
    for img in wrist_views
]

primary_inputs["pixel_values"] = torch.cat(
    [primary_inputs["pixel_values"]] + [e["pixel_values"] for e in extra_inputs],
    dim=1,
)

with torch.no_grad():
    result = self.model.predict_action(
        **primary_inputs, unnorm_key=self.unnorm_key, do_sample=False
    )
```
`[CITED: github.com/moojink/openvla-oft — experiments/robot/openvla_utils.py, resolved via WebFetch this session]` — verify the exact kwarg name/shape contract live on Colab against the currently-installed openvla-oft commit before relying on it; treat the `torch.cat(..., dim=1)` detail as the load-bearing part, the rest is illustrative.

**Pi0Backend.predict() pattern (D-02):**
```python
# Before (Phase 3 placeholder — both keys reuse the same wrist frame):
img = image_tools.convert_to_uint8(image_tools.resize_with_pad(np.asarray(images["eye_in_hand"]), 224, 224))
obs = {"observation/image": img, "observation/wrist_image": img, ...}

# After (Phase 5, D-02 — real dual views, matching openpi's native libero_policy.py contract):
base_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(np.asarray(images["agentview"]), 224, 224))
wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(np.asarray(images["eye_in_hand"]), 224, 224))
obs = {"observation/image": base_img, "observation/wrist_image": wrist_img, ...}
```
`[CITED: github.com/Physical-Intelligence/openpi — src/openpi/policies/libero_policy.py, resolved via WebFetch this session]`.

### Pattern 2: Depth Extraction (SPAT-03)

**What:** Flip `camera_depths=True` (or a per-camera list) through `ControlEnv`'s existing kwarg, consume the new `{cam}_depth` obs keys robosuite already produces.

**Example:**
```python
# env_wrapper.py already threads camera_depths through to robosuite.make(); no
# structural change needed here, only the call-site default/kwarg used by
# whatever constructs the env for this phase's tasks:
env = OffScreenRenderEnv(
    bddl_file_name=bddl_path,
    robots=["Soarm101"],
    camera_names=["agentview", "robot0_eye_in_hand"],
    camera_depths=True,   # SPAT-03
)
obs = env.reset()
rgb = obs["agentview_image"]          # (H, W, 3) uint8
depth_norm = obs["agentview_depth"]   # (H, W, 1) float, normalized [0,1]
```
`[VERIFIED: local package inspection — robosuite 1.4.1 robot_env.py `_create_camera_sensors`, read directly this session]`.

### Pattern 3: Depth → XYZ Pipeline (D-03, D-04)

**What:** Convert a normalized depth buffer + a target pixel into a world-frame 3D point, using robosuite's ready-made camera utilities — no hand-rolled projective geometry.

**Example:**
```python
# Source: robosuite.utils.camera_utils (installed, robosuite 1.4.1), read directly this session.
from robosuite.utils import camera_utils as cu

def pixel_to_world_xyz(sim, camera_name, camera_height, camera_width, depth_norm, pixel_yx):
    """pixel_yx: (row, col) integer pixel location of the object of interest
    (e.g. from a segmentation mask centroid)."""
    real_depth = cu.get_real_depth_map(sim=sim, depth_map=depth_norm)
    extrinsic = cu.get_camera_extrinsic_matrix(sim=sim, camera_name=camera_name)
    intrinsic = cu.get_camera_intrinsic_matrix(
        sim=sim, camera_name=camera_name,
        camera_height=camera_height, camera_width=camera_width,
    )
    # camera_to_world_transform must be the 4x4 that maps camera-frame homogeneous
    # points to world-frame points — this is `extrinsic` itself (get_camera_extrinsic_matrix
    # already returns "camera pose in the world frame"); do NOT re-invert it.
    pixels = np.array([pixel_yx])  # shape (1, 2), (row, col) order per transform_from_pixels_to_world
    xyz_world = cu.transform_from_pixels_to_world(
        pixels=pixels,
        depth_map=real_depth[None, ...],   # matching leading-shape contract, see docstring
        camera_to_world_transform=extrinsic,
    )
    return xyz_world[0]  # (3,)
```
`[VERIFIED: local package inspection]` for the four function signatures and behavior; the exact leading-shape/broadcasting contract between `pixels` and `depth_map` in `transform_from_pixels_to_world` should be exercised against a real obs array in a unit test before use — the function asserts `depth_map_leading_shape == pixels_leading_shape`, which is easy to get wrong when the depth map is `(H,W,1)`.

**D-04 validation pattern:**
```python
# D-04: compare the depth-derived XYZ estimate against sim.data ground truth for
# the SAME object at the SAME timestep -- this is a correctness *test*, not the
# pipeline's production path.
gt_pos = sim.data.body_xpos[env.obj_body_id["cream_cheese_1"]]  # ground truth, test-only
est_pos = pixel_to_world_xyz(sim, "agentview", H, W, depth_norm, pixel_yx=obj_pixel)
assert np.linalg.norm(gt_pos - est_pos) < TOLERANCE_M  # e.g. 0.02-0.05 m given 128x128 render res
```

### Pattern 4: Spatial BDDL Predicates (D-05, D-07, D-08, D-09)

**What:** Add new `BinaryAtomic` predicate classes to `base_predicates.py`, register them by lowercase name, and author BDDL `:goal` blocks using them — following the exact same pattern as the existing `On`/`In`/`Stack` predicates, which already use `get_geom_state()["pos"]` (i.e. `sim.data.body_xpos`, satisfying D-05 automatically) and threshold-based tolerance (existing precedent: `check_ontop`'s `< 0.03` XY-distance band).

**Critical constraint discovered this session:** `libero_tabletop_manipulation.py::_eval_predicate` only handles 2-token (unary) and 3-token (binary) goal tuples — there is no ternary dispatch. `[VERIFIED: local file read — LIBERO/libero/libero/envs/problems/libero_tabletop_manipulation.py]`. This governs how "between" must be authored (see below).

**Example — new predicate classes (base_predicates.py):**
```python
# Source: pattern mirrors existing On/Stack classes in this same file.
class LeftOfX(BinaryAtomic):
    """arg1 is to the left of arg2 along the table's X axis, beyond a margin."""
    MARGIN = 0.03  # meters; matches check_ontop's existing 0.03 XY-tolerance convention (D-08)

    def __call__(self, arg1, arg2):
        x1 = arg1.get_geom_state()["pos"][0]
        x2 = arg2.get_geom_state()["pos"][0]
        return (x2 - x1) > self.MARGIN


class RightOfX(BinaryAtomic):
    MARGIN = 0.03

    def __call__(self, arg1, arg2):
        x1 = arg1.get_geom_state()["pos"][0]
        x2 = arg2.get_geom_state()["pos"][0]
        return (x1 - x2) > self.MARGIN


class NearTo(BinaryAtomic):
    THRESHOLD = 0.10  # meters; task-design decision (D-08), tune with margin per D-09

    def __call__(self, arg1, arg2):
        p1 = arg1.get_geom_state()["pos"][:2]
        p2 = arg2.get_geom_state()["pos"][:2]
        return np.linalg.norm(p1 - p2) < self.THRESHOLD


class FarFrom(BinaryAtomic):
    THRESHOLD = 0.20  # meters; must be set with a clear gap above NearTo's threshold (D-09)

    def __call__(self, arg1, arg2):
        p1 = arg1.get_geom_state()["pos"][:2]
        p2 = arg2.get_geom_state()["pos"][:2]
        return np.linalg.norm(p1 - p2) > self.THRESHOLD
```

**Registration (predicates/__init__.py):**
```python
VALIDATE_PREDICATE_FN_DICT.update({
    "leftofx": LeftOfX(),
    "rightofx": RightOfX(),
    "nearto": NearTo(),
    "farfrom": FarFrom(),
})
```
(`bddl.parsing.scan_tokens` lowercases the entire file at parse time `[VERIFIED: local file read — bddl/parsing.py]`, so BDDL authors can write `(LeftOfX ...)` in mixed case and it resolves to the lowercase registry key.)

**"Between" via decomposition, not a new ternary predicate (BDDL `:goal` block):**
```
(:goal
  (And
    (LeftOfX cube_1 bowl_right_ref)
    (RightOfX cube_1 bowl_left_ref)
  )
)
```
i.e., "cube_1 is between A and B" is expressed as "cube_1 is right of A AND left of B" — two binary predicates the existing `_eval_predicate` dispatcher already supports without modification.

### Anti-Patterns to Avoid
- **Reading `sim.data` directly inside the depth→XYZ module's production path:** violates D-03's explicit real-hardware-faithful intent — `sim.data.body_xpos` may only appear in that module's *test* file (D-04), never in `depth_xyz.py` itself.
- **Using the depth-derived XYZ estimate inside a BDDL predicate's pass/fail check:** violates D-05 — predicates must keep using `ObjectState.get_geom_state()` (ground truth), even though a spatial-predicate's *design* was informed by "where is this object" reasoning.
- **Tiling/concatenating two camera frames into a single image before sending to either VLA backend:** contradicts D-02's explicit fallback-of-last-resort framing — both backends' native dual-image APIs are confirmed to exist; use them.
- **Extending `_eval_predicate` to accept a 4-token ternary tuple for "between":** works, but touches a shared dispatcher used by every LIBERO task in the codebase (including Phase 4's already-validated `put_the_cream_cheese_in_the_bowl` task) — the binary-decomposition approach achieves the same spatial semantics with zero risk to existing tasks.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Depth buffer → real-world distances | Custom near/far normalization math | `robosuite.utils.camera_utils.get_real_depth_map()` | MuJoCo's depth convention (normalized via `zfar`/`znear` and a nonlinear remap) is easy to get subtly wrong; robosuite's implementation is the same one every robosuite/LIBERO project already relies on |
| Camera intrinsics/extrinsics from MuJoCo camera params | Custom FOV→focal-length or MuJoCo-axis-convention correction | `get_camera_intrinsic_matrix()` / `get_camera_extrinsic_matrix()` | The extrinsic matrix specifically needs a camera-axis correction matrix (`camera_axis_correction`) because MuJoCo's camera body convention differs from standard computer-vision convention — robosuite already encodes this fix; a hand-rolled version is a classic source of "my point cloud is upside down/mirrored" bugs |
| Pixel + depth → world-frame 3D point | Custom homogeneous-coordinate back-projection | `transform_from_pixels_to_world()` | Batched, bilinear-sampled, already-tested implementation; re-deriving the math risks off-by-one pixel/row-col ordering errors (the function's own docstring flags the row/col vs x/y swap as a common gotcha) |
| Multi-image tensor packing for OpenVLA-OFT | A novel image-concatenation scheme | The checkpoint's own `num_images_in_input`-gated pattern (`torch.cat(pixel_values, dim=1)` after per-image processor calls) | The model was trained with this exact packing; any other scheme (e.g. side-by-side image tiling) feeds the model out-of-distribution input |

**Key insight:** Every hard part of this phase (camera math, multi-image model input) is already solved by the pinned dependencies or the checkpoint's own reference code — the actual new work is glue code and BDDL authoring, not novel algorithms.

## Common Pitfalls

### Pitfall 1: Treating `camera_depths=True` as free — some object/geom render settings interact with depth
**What goes wrong:** Depth rendering can behave differently under certain MuJoCo renderer/GL backend configurations (e.g. `MUJOCO_GL=osmesa` vs `glfw` vs `egl`), and this project has hit backend-specific rendering issues before (`MUJOCO_GL` must be set before any MuJoCo import — an existing CLAUDE.md architectural constraint).
**Why it happens:** Depth buffer readback path differs slightly from color-only readback in some MuJoCo renderer backends.
**How to avoid:** Verify non-degenerate depth values (not all-zero, not all-`znear`/`zfar` clamp values) on first render after flipping `camera_depths=True`, on whichever backend the executor is actually running (Colab `egl` per project convention, or local macOS `glfw` per CLAUDE.md).
**Warning signs:** `get_real_depth_map`'s own `assert np.all(depth_map >= 0.0) and np.all(depth_map <= 1.0)` firing, or a suspiciously flat/constant depth image.

### Pitfall 2: `num_images_in_input` behavior is baked into the checkpoint's remote code, not independently controllable
**What goes wrong:** `OFTBackend.__init__` loads the model via `AutoModelForVision2Seq.from_pretrained(checkpoint, trust_remote_code=True)` — the actual multi-image handling logic lives in that checkpoint's own `modeling_prismatic.py`/`processing_prismatic.py` remote code, not in this project's code. If the specific checkpoint's remote code doesn't honor a `num_images_in_input`-style config the way the reference `get_vla_action` helper assumes, the pattern in this document's Pattern 1 code sample may need adjustment.
**Why it happens:** `[ASSUMED]` — this session confirmed the *reference eval script's* pattern and the checkpoint card's *quick-start example*, but did not execute real inference against the live checkpoint (no local GPU, per this project's established Environment Availability constraint from Phase 3's own research).
**How to avoid:** Treat the exact processor/`predict_action` call signature as something to confirm on Colab (where the checkpoint actually loads) before committing to a specific code shape — this is exactly the kind of claim flagged in the Assumptions Log below.
**Warning signs:** `predict_action` raising a shape-mismatch error on `pixel_values`, or actions that look degenerate/constant (suggesting the second image silently didn't influence the forward pass).

### Pitfall 3: Depth-derived XYZ tolerance vs. render resolution
**What goes wrong:** At the project's default `camera_heights=128, camera_widths=128` (per `env_wrapper.py`'s default kwargs), per-pixel depth back-projection has coarser spatial resolution than at higher render resolutions — a naive tight tolerance in D-04's validation test may fail even when the pipeline is correct.
**Why it happens:** Each pixel subtends a larger real-world footprint at low resolution; bilinear depth sampling (`bilinear_interpolate` inside `transform_from_pixels_to_world`) reduces but doesn't eliminate this.
**How to avoid:** Size D-04's tolerance to the render resolution actually used (128×128 by default, or whatever resolution this phase's env construction uses — `eval_loop.py`'s `run_suite` docstring shows a precedent of `camera_heights=256, camera_widths=256` for evaluation), not an arbitrarily tight number.
**Warning signs:** D-04's ground-truth-comparison test flaking intermittently near object edges.

### Pitfall 4: Reusing Phase 4's exact object/region coordinates without re-validating for a *new* task layout
**What goes wrong:** Phase 4's `04-02-SUMMARY.md` documents that the akita_black_bowl's specific placement (not distance alone) caused a real collision with the arm's forward sweep — collision safety in this embodiment is layout-specific, not just "stay within 0.45m."
**Why it happens:** The forward-corridor collision risk is about the *angular/radial arrangement* of objects relative to the base, not a simple reach-distance check.
**How to avoid:** Reuse D-06's empirical validation methodology (direct `actuator_force`/`qfrc_bias` + `sim.data.contact` inspection during a real scripted or FSM-driven trajectory) for any newly-authored region, even if it looks similar to the already-validated `put_the_cream_cheese_in_the_bowl` layout — especially once a 3rd object is added for the "between" task (more objects near each other increases collision-corridor risk).
**Warning signs:** A "stuck" end-effector that looks like a reach/torque limit — Phase 4's own lesson-learned is that this pattern was twice misdiagnosed as a hardware limit before direct contact telemetry revealed it was a collision.

## Code Examples

See Architecture Patterns section above for the primary verified code patterns (Patterns 1-4). All code samples there are either `[VERIFIED: local package inspection]` (robosuite camera_utils, LIBERO predicate registry, bddl parsing) or `[CITED: <repo path>]` (OpenVLA-OFT / openpi reference implementations, fetched from their GitHub source this session) — no code sample in this document is purely `[ASSUMED]`.

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Single eye-in-hand image → VLA (Phase 3 baseline) | Dual third-person + wrist image → VLA, using each checkpoint's native multi-image path | This phase (SPAT-01/02) | Matches the training distribution both checkpoints actually expect (the LIBERO benchmark itself is defined with third-person + wrist input) — Phase 3's single-image baseline was an intentional interim scope, not the final target |
| Privileged `sim.data` object position reads (fast, sim-only) | Depth-buffer back-projection using camera intrinsics/extrinsics (D-03) | This phase (SPAT-04) | This is the project's first perception pathway designed to transfer to the physical SOARM arm — a deliberate scope expansion beyond "whatever works in sim" |
| LIBERO's existing "spatial" tasks (`libero_spatial` suite) — spatial language only disambiguates WHICH object to pick, goal predicate is a plain `On` check | New relation predicates that genuinely evaluate spatial conditions in the goal state itself | This phase (SPAT-05) | `[VERIFIED: local file read]` — confirmed by reading `pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.bddl`'s `:goal` block, which is `(And (On akita_black_bowl_1 plate_1))` despite "between" appearing in the task's language instruction. This project's SPAT-05 genuinely goes beyond LIBERO's own existing "spatial" suite in this respect — there is no existing LIBERO predicate to copy for true relational spatial success checking, it must be authored new |

**Deprecated/outdated:** None identified — the dependencies here (robosuite 1.4.x, bddl 1.0.1) are the project's already-locked stack (robosuite 1.5+ is explicitly Out of Scope per REQUIREMENTS.md due to `SingleArmEnv` removal); this phase does not touch that boundary.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | The specific checkpoint `moojink/openvla-7b-oft-finetuned-libero-spatial` genuinely honors `num_images_in_input=2` and the `torch.cat(pixel_values, dim=1)` packing shown in Pattern 1 — this was inferred from the checkpoint's model-card quick-start example and the reference eval script, not from executing real inference (no local GPU available to this research session) | Multi-Camera → VLA Wiring, Pattern 1, Pitfall 2 | If the checkpoint's actual remote code expects a different packing (e.g. a different dim, or images concatenated before the processor rather than after), `oft_backend.py`'s implementation will need a Colab-side fix during execution — budget for this as a verification step, not a guaranteed-correct implementation |
| A2 | 0.03m (matching `check_ontop`'s existing tolerance) is an appropriate default margin for `LeftOfX`/`RightOfX`; 0.10m/0.20m are appropriate `NearTo`/`FarFrom` thresholds | Spatial BDDL Predicates, Pattern 4 | These are starting points, not empirically validated for the SOARM table layout/object sizes — D-08 explicitly leaves exact margin to planning/execution discretion; if too tight, legitimate task completions could fail; if too loose, D-09's "avoid ambiguity via margin" goal is undermined |
| A3 | Decomposing "between" into two binary predicates (`LeftOfX` + `RightOfX` against two different reference objects) is semantically equivalent to a true "between" relation for this phase's benchmarking purposes | Spatial BDDL Predicates, Pattern 4 | If task design places the two reference objects such that the X-axis decomposition doesn't match an intuitive "between" (e.g. if the natural relation is more about distance-from-midpoint than two one-sided inequalities), the predicate could pass/fail in ways that don't match the natural-language prompt's intent — mitigated by careful task-layout authoring (D-09) |
| A4 | The render resolution used during D-04's XYZ validation should match `eval_loop.py`'s evaluation-time resolution (128×128 default in `env_wrapper.py`, or 256×256 per `run_suite`'s docstring precedent) rather than a fixed arbitrary value | Pitfall 3 | If the wrong resolution is assumed, D-04's tolerance band could be miscalibrated (too tight → flaky test failures; too loose → doesn't actually catch calibration bugs) |

## Open Questions

1. **Does `get_vla_action`'s exact `pixel_values` concatenation dimension (`dim=1`) hold across all `num_images_in_input` values, or is it specific to the 2-image case?**
   - What we know: The reference implementation concatenates along `dim=1` for the 2-image case documented in this session's WebFetch.
   - What's unclear: Whether this generalizes exactly as shown, or whether there's additional masking/positional-encoding logic in `modeling_prismatic.py`'s remote code that this session didn't inspect (no local GPU to run and introspect the actual checkpoint).
   - Recommendation: Treat as a Colab-verification step during execution (mirrors this project's established pattern from Phase 1/3 of "local no-GPU dev iterates the interface, Colab confirms the real forward pass").

2. **Should the "between" tolerance margin differ from the left/right margin, given "between" is a conjunction of two one-sided checks?**
   - What we know: D-08/D-09 want threshold bands and margin-based ambiguity avoidance.
   - What's unclear: Whether a single shared margin constant across all spatial predicates is sufficient, or whether "between" (a stricter compound condition) needs its own, possibly larger, margin to avoid an object accidentally satisfying/failing both halves near the boundary.
   - Recommendation: Start with a shared margin constant (simplicity, matches D-08's "matching LIBERO's own predicate convention" framing) and only differentiate if task-layout validation surfaces boundary flakiness — decide during planning/execution, not here.

3. **Does the SOARM `agentview` camera's actual field-of-view/placement in this project's tuned MJCF produce depth values with good resolution over the ~0.45m reach envelope, or is the camera positioned such that objects occupy very few pixels?**
   - What we know: `agentview` and `robot0_eye_in_hand` are already registered and confirmed visually correct per Phase 2 (ENV-07).
   - What's unclear: Depth-specific image quality (as opposed to RGB visual correctness) was never checked in prior phases — this is genuinely new territory.
   - Recommendation: First implementation step for SPAT-03/04 should be a quick visual/numeric sanity check of the actual depth buffer on this project's tuned SOARM+table scene, before building the full back-projection pipeline on top of it.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| robosuite (local) | Local dev/testing of depth math, predicate logic (non-GPU parts) | Yes | 1.4.1 (`libero` conda env) | — |
| MuJoCo (local) | Same as above, via robosuite | Yes | bundled with robosuite 1.4.1 | — |
| GPU (CUDA) | Real OpenVLA-OFT / real pi0 inference verification | No (local machine has no CUDA GPU, per this project's established Phase 1/3 Environment Availability findings) | — | Colab (A100/T4) — this project's existing, proven pattern; all GPU-backed verification for this phase happens there, not locally |
| Colab + openpi `serve_policy.py` process | Real pi0 dual-image inference smoke test | Not available in this research session | — | Reuse Phase 3's established Notebook B pattern (separate kernel, websocket) |
| bddl (local) | BDDL parsing, predicate registry testing | Yes | 1.0.1 | — |

**Missing dependencies with no fallback:** None — every capability this phase needs either runs locally (depth math, predicate logic, BDDL authoring/parsing — all confirmed available) or has this project's already-proven Colab fallback (real VLA inference).

**Missing dependencies with fallback:** GPU-backed multi-image inference verification (OFT and pi0) — fallback is Colab, exactly as established in Phases 1 and 3.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (no `pytest.ini`/`pyproject.toml` found; invoked directly per this project's established convention) |
| Config file | none — see Wave 0 |
| Quick run command | `conda run -n libero pytest LIBERO/libero/libero/<module>/test_<name>.py -x -q` |
| Full suite command | `conda run -n libero pytest LIBERO/libero/libero -x -q` (mirrors the per-module invocation pattern used throughout Phase 4's SUMMARY.md files; no repo-wide pytest config was found, so this remains the established per-directory convention rather than a single `pytest` invocation from repo root) |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| SPAT-01 | Both cameras configured and render non-degenerate frames | integration (local, MuJoCo, no GPU needed) | `conda run -n libero pytest LIBERO/libero/libero/envs/test_camera_config.py -x -q` | ❌ Wave 0 |
| SPAT-02 | `images` dict passed to `backend.predict()` contains both view keys | unit (mock backend, mirrors `test_eval_loop.py`'s existing `MockEnv` pattern) | `conda run -n libero pytest LIBERO/libero/libero/vla/test_eval_loop.py -x -q -k spatial` | ⚠️ extend existing `test_eval_loop.py`, not a new file |
| SPAT-03 | `{cam}_depth` obs keys present and within `[0,1]` (pre-conversion) / real-distance range (post `get_real_depth_map`) | unit/integration (local, MuJoCo) | `conda run -n libero pytest LIBERO/libero/libero/envs/test_camera_config.py -x -q -k depth` | ❌ Wave 0 |
| SPAT-04 | Depth-derived XYZ within tolerance of `sim.data.body_xpos` ground truth (D-04) | integration (local, MuJoCo) | `conda run -n libero pytest LIBERO/libero/libero/perception/test_depth_xyz.py -x -q` | ❌ Wave 0 |
| SPAT-05 | Each new spatial predicate correctly evaluates true/false cases; each new BDDL task's goal is satisfiable by a known-good state and not satisfied by a known-bad state | unit (predicate logic, local) + integration (real BDDL parse + env construction, local) | `conda run -n libero pytest LIBERO/libero/libero/envs/test_spatial_predicates.py -x -q` | ❌ Wave 0 |

Real multi-image VLA inference (both backends actually consuming both views and producing sane actions) is **manual/Colab-only**, per this project's established GPU-backed-behavior convention (Phase 1/3 precedent) — not part of the local automated suite.

### Sampling Rate
- **Per task commit:** targeted `-k` filtered run of the relevant new test file
- **Per wave merge:** full per-module runs for every file touched that wave
- **Phase gate:** all Wave 0 new test files green locally before `/gsd-verify-work`; Colab-side multi-image inference spot-check documented as a manual verification step (mirrors Phase 3's `VLA-04` proof-cell pattern)

### Wave 0 Gaps
- [ ] `LIBERO/libero/libero/envs/test_camera_config.py` — covers SPAT-01, SPAT-03 (camera registration + depth key presence/range)
- [ ] `LIBERO/libero/libero/perception/test_depth_xyz.py` — covers SPAT-04 (D-04's ground-truth comparison)
- [ ] `LIBERO/libero/libero/envs/test_spatial_predicates.py` — covers SPAT-05 (new predicate unit tests + BDDL goal integration tests)
- [ ] Extend `LIBERO/libero/libero/vla/test_eval_loop.py` — covers SPAT-02 (images dict contains both view keys)
- [ ] Extend `LIBERO/libero/libero/vla/test_pi0_backend.py` — covers D-02 for Pi0Backend (mock-based, verifies both real image arrays reach the obs dict, not duplicated)
- [ ] No framework install needed — pytest already used project-wide

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | No | Research-only local/Colab pipeline, no auth surface introduced |
| V3 Session Management | No | No new session/state surface |
| V4 Access Control | No | No new access-control surface |
| V5 Input Validation | Yes (narrow) | Depth buffer values validated via `get_real_depth_map`'s own `assert np.all(depth_map >= 0.0) and np.all(depth_map <= 1.0)` before conversion (already present in robosuite, no new code needed); new BDDL files are static, developer-authored inputs (not user-supplied at runtime), so standard input-validation concerns (injection, etc.) don't apply the way they would for a web app |
| V6 Cryptography | No | No new cryptographic surface |

### Known Threat Patterns for this stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| openpi websocket server (Pi0Backend) bound beyond localhost | Tampering / Information Disclosure | Already mitigated per Phase 3's established Security Domain finding (`serve_policy.py` must never bind `0.0.0.0`) — this phase does not change that binding, only the obs payload content sent over the existing localhost-only connection |
| Malformed/out-of-range depth values silently propagating into a wrong XYZ estimate that then feeds a BDDL predicate indirectly (e.g. via task-design reasoning) | Tampering (data integrity, not security-adversarial in this research context) | D-04's ground-truth validation test *is* the mitigation — any systematic depth-pipeline bug is caught by comparing against `sim.data` before the XYZ pipeline is trusted for anything beyond D-04-validated demonstration/logging use |

No new network-facing surface, no new secrets/credentials, and no new untrusted-input parsing is introduced by this phase — the security domain here is narrow by the nature of the work (simulation perception + task authoring, not a service with external users).

## Sources

### Primary (HIGH confidence)
- `robosuite.utils.camera_utils` — read directly from the installed package (`/opt/homebrew/Caskroom/miniconda/base/envs/libero/lib/python3.9/site-packages/robosuite/utils/camera_utils.py`, robosuite 1.4.1) `[VERIFIED: local package inspection]`
- `robosuite.environments.robot_env` — `_create_camera_sensors` obs-key naming (`{cam}_image`, `{cam}_depth`), read directly from the installed package `[VERIFIED: local package inspection]`
- `LIBERO/libero/libero/envs/env_wrapper.py`, `bddl_base_domain.py`, `bddl_utils.py`, `envs/predicates/base_predicates.py`, `envs/predicates/__init__.py`, `envs/object_states/base_object_states.py`, `envs/problems/libero_tabletop_manipulation.py`, `vla/eval_loop.py`, `vla/interface.py`, `vla/oft_backend.py`, `vla/pi0_backend.py`, `bddl_files/libero_spatial/*.bddl`, `bddl_files/libero_goal/put_the_cream_cheese_in_the_bowl.bddl` — all read directly this session `[VERIFIED: local file read]`
- `bddl` package `parsing.py` (installed, `bddl==1.0.1`) — read directly this session, confirms case-insensitive predicate name lowering `[VERIFIED: local package inspection]`
- `.planning/phases/04-dataset-collection/04-02-SUMMARY.md` — embodiment-constraint history and empirical-validation methodology `[VERIFIED: local file read]`

### Secondary (MEDIUM confidence)
- github.com/moojink/openvla-oft — `experiments/robot/openvla_utils.py` (`get_vla_action`) — fetched via WebFetch this session `[CITED: github.com/moojink/openvla-oft]`
- huggingface.co/moojink/openvla-7b-oft-finetuned-libero-spatial — model card quick-start example (`num_images_in_input=2`) — fetched via WebFetch this session `[CITED: huggingface.co/moojink/openvla-7b-oft-finetuned-libero-spatial]`
- github.com/Physical-Intelligence/openpi — `src/openpi/policies/libero_policy.py` — fetched via WebFetch this session `[CITED: github.com/Physical-Intelligence/openpi]`

### Tertiary (LOW confidence)
- SpatialVLA (arXiv 2501.15830) — WebSearch summary only, not fetched from the paper directly; context/background only, not adopted this phase (matches ADV-01, v2-deferred) `[ASSUMED — WebSearch summary, not independently verified against the paper text]`
- cVLA (arXiv 2507.02190) — WebSearch summary only; context/background only, not adopted this phase `[ASSUMED — WebSearch summary, not independently verified against the paper text]`

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no new packages; all capabilities confirmed present in already-installed, already-pinned dependencies via direct local inspection
- Architecture (multi-camera VLA wiring, depth pipeline): MEDIUM-HIGH — mechanism confirmed via official source repos (WebFetch) and local package inspection, but the specific checkpoint's exact runtime behavior (A1 in Assumptions Log) is not executable-verified in this session (no local GPU, matches this project's established constraint)
- BDDL predicates: HIGH — the arity limitation and ground-truth-access pattern were confirmed by reading the actual dispatcher code this session, not inferred
- Pitfalls: MEDIUM — pitfalls 1, 3, 4 are grounded in this project's own documented history (Phase 2/4 findings); pitfall 2 is explicitly flagged as an assumption pending Colab verification

**Research date:** 2026-08-09
**Valid until:** 30 days for the local/codebase-grounded findings (robosuite/bddl/LIBERO internals — stable, pinned); ~7-14 days for the OpenVLA-OFT/openpi reference-implementation details (both are actively-developed upstream repos; re-verify the exact `get_vla_action`/`libero_policy.py` code shape immediately before implementation if this research is more than ~2 weeks old)
