# Phase 7: Camera & Depth Perception - Pattern Map

**Mapped:** 2026-09-05
**Files analyzed:** 5 (all existing — no new files this phase)
**Analogs found:** 5 / 5 (self-analog: each file extends its own existing internal pattern)

## File Classification

| File to Modify | Role | Data Flow | Closest Analog | Match Quality |
|---|---|---|---|---|
| `LIBERO/libero/libero/envs/problems/libero_tabletop_manipulation.py` | config (MJCF scene builder) | transform (Python → MJCF camera element) | itself — `_setup_camera()`'s own `frontview`/`galleryview` sibling calls | exact (self) |
| `LIBERO/libero/libero/assets/robots/soarm101/robot.xml` | config (static MJCF asset) | transform (static XML attribute) | itself — the `eye_in_hand` `<camera>` element already present | exact (self) |
| `LIBERO/libero/libero/datasets/hdf5_writer.py` | utility (dataset writer) | file-I/O / batch (CRUD-like: create HDF5 group per episode) | itself — `OBS_KEY_MAPPING`/`_RGB_KEYS`/`_STATE_KEYS` write branches | exact (self) |
| `LIBERO/libero/libero/datasets/rlds_converter.py` | transform (schema converter) | batch / transform (HDF5 → TFRecord) | itself — `validate_episode_arrays`, `_episode_to_rlds_steps`, `step_features` FeaturesDict | exact (self) |
| `LIBERO/libero/libero/datasets/oxe_register.py` | config (registration dict patcher) | transform (dict mutation + on-disk file patch) | itself — `register_soarm_spatial`'s `depth_obs_keys` placeholder, and the on-disk patch marker block | exact (self) |

All five files are the "closest analog to themselves" — this phase is a pure extension of dtype/key-mapping conventions already established in the same file, per the phase's own scoping (no new files, no new abstractions).

## Pattern Assignments

### `LIBERO/libero/libero/envs/problems/libero_tabletop_manipulation.py` (config, transform)

**Analog:** itself, `_setup_camera()` (lines 187-212)

**Current code (lines 187-212):**
```python
def _setup_camera(self, mujoco_arena):
    mujoco_arena.set_camera(
        camera_name="agentview",
        pos=[0.6586131746834771, 0.0, 1.6103500240372423],
        quat=[
            0.6380177736282349,
            0.3048497438430786,
            0.30484986305236816,
            0.6380177736282349,
        ],
    )

    # For visualization purpose
    mujoco_arena.set_camera(
        camera_name="frontview", pos=[1.0, 0.0, 1.48], quat=[0.56, 0.43, 0.43, 0.56]
    )
    mujoco_arena.set_camera(
        camera_name="galleryview",
        pos=[2.844547668904445, 2.1279684793440667, 3.128616846013882],
        quat=[
            0.42261379957199097,
            0.23374411463737488,
            0.41646939516067505,
            0.7702690958976746,
        ],
    )
```

**Pattern to copy:** `mujoco_arena.set_camera(camera_name=..., pos=[...], quat=[...])` is the established call shape for every camera in this method. Extend the `agentview` call with a new `pos`/`quat` (best-estimate overhead placement, Claude's Discretion) plus a `camera_attribs={"fovy": "43"}` kwarg — no other call in this method currently uses `camera_attribs`, so this is a net-new kwarg on an existing call, not a new call shape. Leave `frontview`/`galleryview` calls untouched.

**Note on `bddl_base_domain.py`:** RESEARCH.md Pitfall 1 confirms this file's override is the ONLY one reached by the project's registered tasks (all use `LIBERO_Tabletop_Manipulation`). `bddl_base_domain.py:275-293` has an analogous but unreachable `_setup_camera()` — do not treat it as the fix location; optionally mirror the fix there too for Phase-8 future-proofing (Open Question 2), same call shape.

---

### `LIBERO/libero/libero/assets/robots/soarm101/robot.xml` (config, static MJCF)

**Analog:** itself — the existing `eye_in_hand` `<camera>` element (line 109)

**Current code (line 108-109):**
```xml
<!-- This camera points out from the eef (starting pose from Panda; tuned in plan 02-04). -->
<camera mode="fixed" name="eye_in_hand" pos="0.14 0 0.02" quat="0.664463 -0.241845 0.241845 -0.664463" fovy="75"/>
```

**Pattern to copy:** `fovy` is a literal string attribute directly on the `<camera>` element — this is the ONLY place in the codebase where `fovy` is set via static XML rather than `set_camera(camera_attribs=...)` (that's `agentview`'s Python-construction path, a different mechanism per RESEARCH.md Pattern 1's note). Only `pos`/`quat`/`fovy` attribute values change; element structure, `mode="fixed"`, and the comment-above-camera convention should be preserved (update the comment to record the new tuning rationale, matching the existing "tuned in plan 02-04" style — e.g. "tuned in plan 07-0X, look-at-grip-site method").

**Look-at quaternion starting candidate (from RESEARCH.md Code Example 2, verified computation):**
```
quat="0.524990 0.473144 0.473637 0.525537"
```
(candidate only — RESEARCH.md Open Question 1 flags this as needing visual validation against the 8 reference photos in `progress-documentation/images/20260827_121*.jpg` before finalizing)

---

### `LIBERO/libero/libero/datasets/hdf5_writer.py` (utility, file-I/O/batch)

**Analog:** itself — `OBS_KEY_MAPPING`, `_RGB_KEYS`, `_STATE_KEYS` (lines ~40-58), env construction (~93-100), write branches (~150-170)

**Imports pattern (module header, already present, no change needed):**
```python
from LIBERO.libero.libero.envs import OffScreenRenderEnv
from LIBERO.libero.libero.envs.bddl_utils import get_problem_info
```

**Key-mapping + dtype-tuple pattern (current, lines ~48-58):**
```python
OBS_KEY_MAPPING = {
    "agentview_rgb": "agentview_image",
    "eye_in_hand_rgb": "robot0_eye_in_hand_image",
    "gripper_states": "robot0_gripper_qpos",
    "joint_states": "robot0_joint_pos",
}

# Which renamed keys are uint8 images vs. float proprio arrays.
_RGB_KEYS = ("agentview_rgb", "eye_in_hand_rgb")
_STATE_KEYS = ("gripper_states", "joint_states")
```
**Extend with (per D-05, agentview only):**
```python
OBS_KEY_MAPPING = {
    "agentview_rgb": "agentview_image",
    "eye_in_hand_rgb": "robot0_eye_in_hand_image",
    "agentview_depth": "agentview_depth",   # robosuite's own key name, no rename needed
    "gripper_states": "robot0_gripper_qpos",
    "joint_states": "robot0_joint_pos",
}
_RGB_KEYS = ("agentview_rgb", "eye_in_hand_rgb")
_DEPTH_KEYS = ("agentview_depth",)          # NEW tuple, same pattern as _RGB_KEYS/_STATE_KEYS
_STATE_KEYS = ("gripper_states", "joint_states")
```

**Env construction pattern (current, lines ~93-100):**
```python
regen_env = OffScreenRenderEnv(
    bddl_file_name=bddl_file_name,
    robots=list(robots),
    camera_heights=camera_size,
    camera_widths=camera_size,
    has_renderer=False,
    has_offscreen_renderer=True,
)
```
**Add:** `camera_depths=True,` as a new kwarg (renders depth for BOTH cameras globally per Pitfall 3 — D-05's "agentview only" is enforced downstream at the write-branch level, not here).

**Write-branch pattern (current, lines ~156-170):**
```python
obs_grp = ep_grp.create_group("obs")
for key in _RGB_KEYS:
    obs_grp.create_dataset(
        key, data=np.array(obs_acc[key], dtype=np.uint8)
    )
for key in _STATE_KEYS:
    stacked = np.array(
        [np.atleast_1d(v) for v in obs_acc[key]], dtype=np.float64
    )
    obs_grp.create_dataset(key, data=stacked)
```
**Extend with a parallel `_DEPTH_KEYS` loop** (matching the `_RGB_KEYS` loop shape, since depth is also already array-shaped per-step, dtype float32, no `atleast_1d` needed):
```python
for key in _DEPTH_KEYS:
    obs_grp.create_dataset(
        key, data=np.array(obs_acc[key], dtype=np.float32)
    )
    # obs_acc[key] already (H, W, 1) arrays in [0, 1] — robosuite's own
    # normalized depth output. Do NOT convert to meters (Pitfall 2).
```

**Error handling / validation pattern:** No per-key validation currently exists in this file (validation lives in `rlds_converter.py`'s `validate_episode_arrays`); the `assert os.path.exists(bddl_file_name)` fail-loud style at function entry is the file's only such pattern — no change needed for depth.

---

### `LIBERO/libero/libero/datasets/rlds_converter.py` (transform, batch)

**Analog:** itself — `validate_episode_arrays` (lines ~35-96), `_episode_to_rlds_steps` (~177-199), `step_features` FeaturesDict (~254-283)

**Validation pattern (current, excerpt):**
```python
def validate_episode_arrays(agentview_rgb, eye_in_hand_rgb, state, actions) -> None:
    """Fail-loud schema/shape/dtype validation for one episode's four arrays.
    Raises ValueError ... never silently pads or truncates (D-06, ASVS V5).
    """
    arrays = {
        "agentview_rgb": agentview_rgb,
        "eye_in_hand_rgb": eye_in_hand_rgb,
        "state": state,
        "actions": actions,
    }
    for name, arr in arrays.items():
        if not isinstance(arr, np.ndarray):
            raise ValueError(f"{name} must be a numpy array, got {type(arr)!r}")

    for name in ("agentview_rgb", "eye_in_hand_rgb"):
        arr = arrays[name]
        if arr.dtype != np.uint8:
            raise ValueError(f"{name} must be dtype=uint8, got dtype={arr.dtype}")
        if arr.ndim != 4:
            raise ValueError(f"{name} must be ndim==4 (T, H, W, 3), got ndim={arr.ndim} shape={arr.shape}")
        if arr.shape[-1] != 3:
            raise ValueError(f"{name} last dim must be 3 (RGB), got shape={arr.shape}")
    ...
    if not np.all(np.isfinite(state)):
        raise ValueError("state contains NaN or Inf")
```
**Pattern to copy for `agentview_depth`:** add it to the `arrays` dict, add a dtype/shape branch analogous to the RGB loop but asserting `dtype == np.float32`, `ndim == 4` `(T, H, W, 1)`, last-dim `== 1`, and a `np.isfinite` + range check (`0.0 <= arr <= 1.0`, matching Pitfall 2's normalized-depth contract) instead of the RGB uint8 checks. Signature becomes `validate_episode_arrays(agentview_rgb, eye_in_hand_rgb, agentview_depth, state, actions)` — must update all call sites in this file and its test.

**Per-step assembly pattern (current, `_episode_to_rlds_steps`, lines ~177-199):**
```python
steps.append(
    {
        "observation": {
            "agentview_rgb": episode["agentview_rgb"][t],
            "eye_in_hand_rgb": episode["eye_in_hand_rgb"][t],
            "state": episode["state"][t],
        },
        "action": episode["actions"][t],
        ...
    }
)
```
**Extend with:** `"agentview_depth": episode["agentview_depth"][t],` inside the `"observation"` dict — same indexing pattern as the RGB keys.

**TFDS FeaturesDict pattern (current, lines ~254-283):**
```python
step_features = tfds.features.FeaturesDict(
    {
        "steps": tfds.features.Dataset(
            {
                "observation": tfds.features.FeaturesDict(
                    {
                        "agentview_rgb": tfds.features.Image(
                            shape=(None, None, 3), dtype=np.uint8
                        ),
                        "eye_in_hand_rgb": tfds.features.Image(
                            shape=(None, None, 3), dtype=np.uint8
                        ),
                        "state": tfds.features.Tensor(
                            shape=(REQUIRED_PROPRIO_DIM,), dtype=np.float32
                        ),
                    }
                ),
                "action": tfds.features.Tensor(
                    shape=(REQUIRED_ACTION_DIM,), dtype=np.float32
                ),
                ...
            }
        )
    }
)
```
**Extend with (Tensor, not Image, per Pattern 4):**
```python
"agentview_depth": tfds.features.Tensor(
    shape=(None, None, 1), dtype=np.float32
),
```
placed alongside `agentview_rgb`/`eye_in_hand_rgb` inside the `"observation"` FeaturesDict.

**`load_episodes_from_hdf5`** (loader, lines ~99-174) must also read the new `demo_N/obs/agentview_depth` dataset the same way it currently reads `agentview_rgb`/`eye_in_hand_rgb`/etc. into the episode dict — mirror whatever `f[f"data/{demo_key}/obs/agentview_rgb"][()]`-style access pattern is already used for the RGB keys (same group path convention, just a different key name).

---

### `LIBERO/libero/libero/datasets/oxe_register.py` (config, transform)

**Analog:** itself — `register_soarm_spatial`'s `depth_obs_keys` placeholder (lines ~85-93) and the on-disk patch string template (lines ~150-165)

**Current in-memory dict pattern:**
```python
oxe_dataset_configs[dataset_name] = {
    "image_obs_keys": {
        "primary": "agentview_rgb",
        "secondary": None,
        "wrist": "eye_in_hand_rgb",
    },
    "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
    "state_obs_keys": ["state"],
    "state_encoding": state_encoding,
    "action_encoding": action_encoding,
}
```
**Change (D-05):** `"depth_obs_keys": {"primary": "agentview_depth", "secondary": None, "wrist": None}` — only the `"primary"` value changes; `"wrist"` stays `None` per D-05 (real IMX335 wrist cam has no depth capability).

**On-disk patch template pattern (current, inside `_patch_installed_file` call, mirrors the in-memory dict exactly):**
```python
f'''OXE_DATASET_CONFIGS["{dataset_name}"] = {{
    "image_obs_keys": {{"primary": "agentview_rgb", "secondary": None, "wrist": "eye_in_hand_rgb"}},
    "depth_obs_keys": {{"primary": None, "secondary": None, "wrist": None}},
    "state_obs_keys": ["state"],
    "state_encoding": StateEncoding.POS_EULER,
    ...
'''
```
**Must be updated identically to the in-memory dict** (`"primary": "agentview_depth"`) — these two representations (in-memory mutation + on-disk string template) MUST stay in sync since the on-disk one is what `torchrun`'s subprocess actually reads (per this file's own module docstring explaining why both exist). Also bump the versioned marker comment (`# --- {dataset_name} runtime registration v2 ... ---` → `v3`) per the file's own documented idempotency-versioning convention, so an already-patched installed file picks up this fix.

**Do NOT set `load_depth=True` anywhere in this file** — per RESEARCH.md Pitfall 4, that's a Phase 9 fine-tuning-time flag, out of this phase's scope.

---

## Shared Patterns

### Fail-loud validation (ASVS V5)
**Source:** `LIBERO/libero/libero/datasets/rlds_converter.py`'s `validate_episode_arrays` (raises `ValueError` naming the offending array + actual-vs-expected shape/dtype, never silently pads/truncates)
**Apply to:** The new `agentview_depth` validation branch in the same function — same message format (`f"{name} must be dtype=..., got dtype={arr.dtype}"`), same fail-before-write ordering (`hdf5_to_rlds` validates ALL episodes before writing ANY `.tfrecord` bytes).

### Key-name-tuple dtype dispatch
**Source:** `hdf5_writer.py`'s `_RGB_KEYS`/`_STATE_KEYS` tuples driving separate write-loop branches
**Apply to:** New `_DEPTH_KEYS = ("agentview_depth",)` tuple with its own float32 write loop — no new class/abstraction, just a third tuple + loop following the identical shape.

### `set_camera(camera_attribs=...)` for extra MJCF attributes
**Source:** `robosuite`'s `MujocoArena.set_camera()` API (RESEARCH.md Pattern 1, verified upstream)
**Apply to:** `agentview`'s fovy fix in `libero_tabletop_manipulation.py` — pass `camera_attribs={"fovy": "43"}` alongside the existing `pos`/`quat` kwargs; do not hand-roll a post-construction `sim.model.cam_fovy[...] = ...` patch.

### Colab-only heavy imports confined to function bodies
**Source:** `rlds_converter.py`'s module docstring and `hdf5_to_rlds`'s local `import tensorflow as tf` / `import tensorflow_datasets as tfds`
**Apply to:** No new heavy imports needed this phase (depth uses the same `numpy`/`h5py`/`tensorflow_datasets` already imported this way) — just confirms no new top-level import should be added to `rlds_converter.py` for depth support.

## No Analog Found

None — all 5 files are pre-existing and this phase is a same-file pattern extension in every case; there is no file without a directly-applicable existing convention to mirror.

## Metadata

**Analog search scope:** `LIBERO/libero/libero/envs/problems/`, `LIBERO/libero/libero/assets/robots/soarm101/`, `LIBERO/libero/libero/datasets/`, `LIBERO/libero/libero/envs/` (for `env_wrapper.py`, `test_camera_config.py`, `bddl_base_domain.py` cross-reference)
**Files scanned:** 6 (the 5 target files + `bddl_base_domain.py` for the dead-code override contrast)
**Pattern extraction date:** 2026-09-05
