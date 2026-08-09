# Phase 5: Spatial Awareness - Pattern Map

**Mapped:** 2026-08-09
**Files analyzed:** 11 (5 modified, 6 new)
**Analogs found:** 11 / 11 (all are modify-in-place or same-directory-sibling analogs)

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|--------------------|------|-----------|-----------------|----------------|
| `LIBERO/libero/libero/vla/eval_loop.py` (modify) | orchestration/controller | request-response | itself (existing `run_episode`, lines 20-75) | exact (in-place edit) |
| `LIBERO/libero/libero/vla/pi0_backend.py` (modify) | service (VLA backend adapter) | request-response | itself (`Pi0Backend.predict`, lines 113-138) | exact (in-place edit) |
| `LIBERO/libero/libero/vla/oft_backend.py` (modify) | service (VLA backend adapter) | request-response | itself (`OFTBackend.predict`, lines 121-150) | exact (in-place edit) |
| `LIBERO/libero/libero/vla/interface.py` (modify, docstring/type only) | config/contract (Protocol) | request-response | itself (`VLABackend.predict`, lines 24-43) | exact (in-place edit) |
| `LIBERO/libero/libero/envs/env_wrapper.py` (modify — flip `camera_depths` default/call-site) | config/wrapper | request-response | itself (`ControlEnv.__init__`, lines 12-81) | exact (in-place edit) |
| `LIBERO/libero/libero/perception/depth_xyz.py` (new) | utility/service (perception) | transform | `robosuite.utils.camera_utils` (external, not in-repo) — no in-repo transform-role analog exists; closest structural analog is `envs/object_states/base_object_states.py::ObjectState.get_geom_state` (lines 47-52) for "how this project reads `sim.data.body_xpos`" | role-match (partial — perception module is new territory) |
| `LIBERO/libero/libero/perception/__init__.py` (new) | module init | — | `envs/predicates/__init__.py` (lines 1-3, plain re-export) | role-match |
| `LIBERO/libero/libero/perception/test_depth_xyz.py` (new) | test | integration | no existing perception test; follow project's pytest convention (see STACK.md / RESEARCH.md Validation Architecture) | no analog |
| `LIBERO/libero/libero/envs/predicates/base_predicates.py` (modify — add `LeftOfX`/`RightOfX`/`NearTo`/`FarFrom`) | model/domain logic | CRUD (state-read + boolean eval) | itself — `On` (lines 62-64), `Stack` (lines 84-90) | exact (same file, same pattern) |
| `LIBERO/libero/libero/envs/predicates/__init__.py` (modify — register new predicates) | config/registry | — | itself (`VALIDATE_PREDICATE_FN_DICT`, lines 4-18) | exact (in-place edit) |
| `LIBERO/libero/libero/bddl_files/libero_spatial_soarm/*.bddl` (new, 3+ files) | config/data (task definition) | batch (static data) | `LIBERO/libero/libero/bddl_files/libero_goal/put_the_cream_cheese_in_the_bowl.bddl` (full file, 45 lines) | exact |
| `LIBERO/libero/libero/envs/test_spatial_predicates.py` (new) | test | unit + integration | no existing predicate test file found in repo; follow `envs/predicates/base_predicates.py` structure being tested | no analog |

## Pattern Assignments

### `LIBERO/libero/libero/vla/eval_loop.py` (orchestration, request-response)

**Analog:** itself — `run_episode()` (lines 20-75)

**Current single-view images-dict construction (line 63):**
```python
images = {"eye_in_hand": obs[camera_name]}
```

**Function signature / docstring convention (lines 20-27):**
```python
def run_episode(
    env,
    backend,
    language: str,
    video_path: str,
    max_steps: int = MAX_STEPS_DEFAULT,
    camera_name: str = "robot0_eye_in_hand_image",
) -> dict:
```

**D-01 target change (per RESEARCH.md Pattern 1):**
```python
images = {
    "eye_in_hand": obs["robot0_eye_in_hand_image"],
    "agentview": obs["agentview_image"],
}
```
Keep the existing `camera_name` kwarg (used elsewhere for `vw.append_obs(obs, done, camera_name=camera_name)` at line 69, the video-writer's camera) — do not repurpose it for the new agentview key; add a second explicit obs-key reference or a new `agentview_camera_name` kwarg to preserve the existing default-parameter convention (module docstring style, lines 42-52).

---

### `LIBERO/libero/libero/vla/pi0_backend.py` (service/adapter, request-response)

**Analog:** itself — `Pi0Backend.predict()` (lines 113-167)

**Imports pattern (lines 28-49):**
```python
import socket
import time
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as wcp
```

**Current placeholder (duplicated wrist image into both obs keys) — lines 130-138:**
```python
img = image_tools.convert_to_uint8(
    image_tools.resize_with_pad(np.asarray(images["eye_in_hand"]), 224, 224)
)
obs = {
    "observation/image": img,
    "observation/wrist_image": img,
    "observation/state": self._robot_state(),
    "prompt": language,
}
```

**D-02 target change (per RESEARCH.md Pattern 1 — real dual views, no duplication):**
```python
base_img = image_tools.convert_to_uint8(
    image_tools.resize_with_pad(np.asarray(images["agentview"]), 224, 224)
)
wrist_img = image_tools.convert_to_uint8(
    image_tools.resize_with_pad(np.asarray(images["eye_in_hand"]), 224, 224)
)
obs = {
    "observation/image": base_img,
    "observation/wrist_image": wrist_img,
    "observation/state": self._robot_state(),
    "prompt": language,
}
```

**Error handling / retry pattern to preserve unchanged (lines 140-167):** the `_RETRYABLE_EXCEPTIONS` retry loop around `self.client.infer(obs)` — no change needed here, just the `obs` dict content above it.

---

### `LIBERO/libero/libero/vla/oft_backend.py` (service/adapter, request-response)

**Analog:** itself — `OFTBackend.predict()` (lines 121-150)

**Imports pattern (lines 16-26):**
```python
import importlib, json, subprocess, sys
from pathlib import Path
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
```

**Current single-image predict (lines 121-150):**
```python
def predict(self, images: dict, language: str) -> np.ndarray:
    prompt = f"In: What action should the robot take to {language}?\nOut:"
    pil_image = Image.fromarray(images["eye_in_hand"])
    inputs = self.processor(prompt, pil_image).to(self.device, dtype=torch.bfloat16)
    with torch.no_grad():
        result = self.model.predict_action(
            **inputs, unnorm_key=self.unnorm_key, do_sample=False
        )
    actions = result[0] if isinstance(result, tuple) else result
    actions = np.asarray(actions)
    assert actions.shape == (8, 7), f"expected (8,7) chunk, got {actions.shape}"
    return actions
```

**D-02 target change (per RESEARCH.md Pattern 1 — `torch.cat` dual-image packing, `[CITED: moojink/openvla-oft]`, treat as Colab-verify-before-trust per Pitfall 2):**
```python
primary = Image.fromarray(images["eye_in_hand"])
extra_views = [Image.fromarray(images["agentview"])]

primary_inputs = self.processor(prompt, primary).to(self.device, dtype=torch.bfloat16)
extra_inputs = [
    self.processor(prompt, img).to(self.device, dtype=torch.bfloat16)
    for img in extra_views
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
Keep the existing assert-shape pattern (`assert actions.shape == (8, 7), ...`) unchanged — it's this file's established error-handling convention for backend output validation.

---

### `LIBERO/libero/libero/vla/interface.py` (contract, request-response)

**Analog:** itself — `VLABackend.predict()` Protocol (lines 24-43)

No code-shape change required; only the docstring's "later phases may add more keys without changing this signature" (line 10, line 33) needs updating to state Phase 5 has now populated the `"agentview"` key. This is the **shared contract** every backend/eval_loop file above must keep satisfying — do not change the method signature `predict(self, images: dict[str, Image.Image], language: str) -> np.ndarray`.

---

### `LIBERO/libero/libero/envs/env_wrapper.py` (config/wrapper, request-response)

**Analog:** itself — `ControlEnv.__init__` (lines 12-81)

**Existing camera kwargs already threaded through to robosuite (lines 31-38):**
```python
camera_names=[
    "agentview",
    "robot0_eye_in_hand",
],
camera_heights=128,
camera_widths=128,
camera_depths=False,
camera_segmentations=None,
```

**SPAT-03 change:** flip the call-site default (or the kwarg passed by whichever eval/task-construction code builds the env for this phase) to `camera_depths=True`; the `**kwargs` passthrough (line 41, line 80) already forwards it into `TASK_MAPPING[...]`'s `robosuite.make()`-style construction — no structural change to `ControlEnv` needed beyond the default value itself, per RESEARCH.md Pattern 2.

---

### `LIBERO/libero/libero/perception/depth_xyz.py` (new perception utility, transform)

**No in-repo analog** — this is genuinely new territory (RESEARCH.md's own framing: "Perception/Processing (new module)"). Build directly on the externally-verified `robosuite.utils.camera_utils` API (already installed, robosuite 1.4.1) per RESEARCH.md Pattern 3:
```python
from robosuite.utils import camera_utils as cu

def pixel_to_world_xyz(sim, camera_name, camera_height, camera_width, depth_norm, pixel_yx):
    real_depth = cu.get_real_depth_map(sim=sim, depth_map=depth_norm)
    extrinsic = cu.get_camera_extrinsic_matrix(sim=sim, camera_name=camera_name)
    intrinsic = cu.get_camera_intrinsic_matrix(
        sim=sim, camera_name=camera_name,
        camera_height=camera_height, camera_width=camera_width,
    )
    pixels = np.array([pixel_yx])
    xyz_world = cu.transform_from_pixels_to_world(
        pixels=pixels,
        depth_map=real_depth[None, ...],
        camera_to_world_transform=extrinsic,
    )
    return xyz_world[0]
```

**Anti-pattern reminder (D-03, RESEARCH.md Anti-Patterns):** this file must NEVER read `sim.data.body_xpos`/`sim.data.body_xquat` directly — that ground-truth access pattern (seen in `ObjectState.get_geom_state()`, lines 47-52 of `base_object_states.py`) belongs only in the test file below, not in production perception code.

**Naming/path convention to follow (CLAUDE.md):** `snake_case`, verb-noun function names (`pixel_to_world_xyz` matches `load_*`/`render_*` conventions), module-level path constants in `UPPER_SNAKE_CASE` if any are added.

---

### `LIBERO/libero/libero/perception/__init__.py` (new, module init)

**Analog:** `LIBERO/libero/libero/envs/predicates/__init__.py` (lines 1-3) — simple re-export style:
```python
from .base_predicates import *
```
Mirror with `from .depth_xyz import *` (or explicit names) once `depth_xyz.py` exists.

---

### `LIBERO/libero/libero/perception/test_depth_xyz.py` (new test, integration)

**No direct analog file**, but follow project pytest convention (RESEARCH.md Validation Architecture: no config file, `conda run -n libero pytest <path> -x -q`, per-module invocation). D-04's ground-truth comparison pattern:
```python
gt_pos = sim.data.body_xpos[env.obj_body_id["cream_cheese_1"]]  # test-only ground truth
est_pos = pixel_to_world_xyz(sim, "agentview", H, W, depth_norm, pixel_yx=obj_pixel)
assert np.linalg.norm(gt_pos - est_pos) < TOLERANCE_M  # size per render resolution (Pitfall 3)
```
This is the ONE place in the perception feature where reading `sim.data.body_xpos` directly is correct (test oracle only, per D-04/Anti-Patterns).

---

### `LIBERO/libero/libero/envs/predicates/base_predicates.py` (domain logic, modify)

**Analog:** itself — `On` (lines 62-64) and `Stack` (lines 84-90):
```python
class On(BinaryAtomic):
    def __call__(self, arg1, arg2):
        return arg2.check_ontop(arg1)


class Stack(BinaryAtomic):
    def __call__(self, arg1, arg2):
        return (
            arg1.check_contact(arg2)
            and arg2.check_contain(arg1)
            and arg1.get_geom_state()["pos"][2] > arg2.get_geom_state()["pos"][2]
        )
```

**Base class to subclass (lines 20-25):**
```python
class BinaryAtomic(Expression):
    def __init__(self):
        pass

    def __call__(self, arg1, arg2):
        raise NotImplementedError
```

**New predicate classes to add (RESEARCH.md Pattern 4, D-05/D-07/D-08/D-09), using the same `get_geom_state()["pos"]` ground-truth read as `On`/`Stack`:**
```python
class LeftOfX(BinaryAtomic):
    MARGIN = 0.03  # matches check_ontop's existing 0.03 XY-tolerance (base_object_states.py line 92)

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
    THRESHOLD = 0.10

    def __call__(self, arg1, arg2):
        p1 = arg1.get_geom_state()["pos"][:2]
        p2 = arg2.get_geom_state()["pos"][:2]
        return np.linalg.norm(p1 - p2) < self.THRESHOLD


class FarFrom(BinaryAtomic):
    THRESHOLD = 0.20

    def __call__(self, arg1, arg2):
        p1 = arg1.get_geom_state()["pos"][:2]
        p2 = arg2.get_geom_state()["pos"][:2]
        return np.linalg.norm(p1 - p2) > self.THRESHOLD
```
Note: `np` is not currently imported in `base_predicates.py` (only `from typing import List`) — add `import numpy as np` following the project's import-organization convention (stdlib first, then third-party; see CLAUDE.md).

**Error handling:** this file has none (no try/except anywhere in the 118 lines) — predicates are pure boolean functions; do not introduce exception handling that isn't in the existing style.

---

### `LIBERO/libero/libero/envs/predicates/__init__.py` (registry, modify)

**Analog:** itself — `VALIDATE_PREDICATE_FN_DICT` (lines 4-18):
```python
VALIDATE_PREDICATE_FN_DICT = {
    "true": TruePredicateFn(),
    "false": FalsePredicateFn(),
    "in": In(),
    "on": On(),
    "up": Up(),
    "printjointstate": PrintJointState(),
    "open": Open(),
    "close": Close(),
    "turnon": TurnOn(),
    "turnoff": TurnOff(),
}
```

**Registration to add:**
```python
VALIDATE_PREDICATE_FN_DICT.update({
    "leftofx": LeftOfX(),
    "rightofx": RightOfX(),
    "nearto": NearTo(),
    "farfrom": FarFrom(),
})
```
(lowercase keys — `bddl.parsing.scan_tokens` lowercases the whole file at parse time, per RESEARCH.md line 357, so BDDL authors may write `(LeftOfX ...)` in any case.)

---

### `LIBERO/libero/libero/bddl_files/libero_spatial_soarm/*.bddl` (new, 3+ files)

**Analog:** `LIBERO/libero/libero/bddl_files/libero_goal/put_the_cream_cheese_in_the_bowl.bddl` (full file, 45 lines) — this is the exact structural template (regions/fixtures/objects/obj_of_interest/init/goal blocks):
```lisp
(define (problem LIBERO_Tabletop_Manipulation)
  (:domain robosuite)
  (:language Put the cream cheese on the bowl)
    (:regions
      (akita_black_bowl_region
          (:target main_table)
          (:ranges (
              (-0.21 -0.04 -0.19 -0.02)
            )
          )
      )
      ...
    )
  (:fixtures
    main_table - table
  )
  (:objects
    akita_black_bowl_1 - akita_black_bowl
    cream_cheese_1 - cream_cheese
  )
  (:obj_of_interest
    cream_cheese_1
    akita_black_bowl_1
  )
  (:init
    (On akita_black_bowl_1 main_table_akita_black_bowl_region)
    (On cream_cheese_1 main_table_cream_cheese_region)
  )
  (:goal
    (And (On cream_cheese_1 akita_black_bowl_1))
  )
)
```

**New "between" goal block pattern (RESEARCH.md Pattern 4, decomposed — no ternary predicate):**
```lisp
(:goal
  (And
    (LeftOfX cube_1 bowl_right_ref)
    (RightOfX cube_1 bowl_left_ref)
  )
)
```

**Left/right and near/far goal blocks** follow the same single-binary-predicate `:goal` shape as the existing `(And (On cream_cheese_1 akita_black_bowl_1))` line — just swap in `(LeftOfX ...)`, `(RightOfX ...)`, `(NearTo ...)`, or `(FarFrom ...)`.

**Embodiment constraint to carry forward (D-06/D-09, Pitfall 4):** region `:ranges` must respect Phase 4's 84mm-gripper / ~0.45m-reach / no-forward-centerline constraints — reuse Phase 4's `04-02-SUMMARY.md` empirical validation methodology (`actuator_force`/`qfrc_bias` + `sim.data.contact`) for any newly authored region, even if it looks similar to this analog's already-validated layout.

---

### `LIBERO/libero/libero/envs/test_spatial_predicates.py` (new test)

**No existing predicate test file found in this repo** (searched; none present). Structure as: unit tests instantiating `LeftOfX()`/`RightOfX()`/`NearTo()`/`FarFrom()` directly against mock objects with a stubbed `get_geom_state()`, plus an integration test that constructs a real `OffScreenRenderEnv` from one of the new BDDL files and calls `_check_success()` (the dispatcher at `libero_tabletop_manipulation.py` lines 135-143) against a known-good and a known-bad object placement.

**Dispatcher being tested (read-only reference, do not modify):**
```python
def _check_success(self):
    goal_state = self.parsed_problem["goal_state"]
    result = True
    for state in goal_state:
        result = self._eval_predicate(state) and result
    return result

def _eval_predicate(self, state):
    if len(state) == 3:
        predicate_fn_name = state[0]
        object_1_name = state[1]
        object_2_name = state[2]
        return eval_predicate_fn(
            predicate_fn_name,
            self.object_states_dict[object_1_name],
            self.object_states_dict[object_2_name],
        )
    elif len(state) == 2:
        ...
```
(`LIBERO/libero/libero/envs/problems/libero_tabletop_manipulation.py`, lines 135-163 — confirms the binary-only 3-token dispatch that motivates the "between" decomposition; do not extend this function.)

## Shared Patterns

### Ground-truth state access (D-05)
**Source:** `LIBERO/libero/libero/envs/object_states/base_object_states.py::ObjectState.get_geom_state()` (lines 47-52)
```python
def get_geom_state(self):
    object_pos = self.env.sim.data.body_xpos[self.env.obj_body_id[self.object_name]]
    object_quat = self.env.sim.data.body_xquat[self.env.obj_body_id[self.object_name]]
    return {"pos": object_pos, "quat": object_quat}
```
**Apply to:** all new predicate classes in `base_predicates.py` — this is the ONLY sanctioned way predicates read object position (never depth-derived XYZ, per D-05).

### Tolerance/margin convention (D-08)
**Source:** `check_ontop()`'s `< 0.03` XY-distance band, `base_object_states.py` line 92 (`ObjectState.check_ontop`)
**Apply to:** `LeftOfX`/`RightOfX` `MARGIN`, `NearTo`/`FarFrom` `THRESHOLD` constants — matches existing project convention of small hardcoded float tolerances, not exact-coordinate comparison.

### VLABackend Protocol (shared interface contract)
**Source:** `LIBERO/libero/libero/vla/interface.py` (lines 24-43)
**Apply to:** `eval_loop.py`, `pi0_backend.py`, `oft_backend.py` — every backend's `predict(images: dict, language: str) -> np.ndarray` signature must stay unchanged; only `images` dict content and internal consumption logic change this phase.

### Predicate registry pattern (D-07)
**Source:** `LIBERO/libero/libero/envs/predicates/__init__.py` (lines 4-22)
**Apply to:** any new predicate class — always register via `VALIDATE_PREDICATE_FN_DICT.update({...})` with a lowercase key; never call predicate classes directly from BDDL-parsing code.

### Naming/logging conventions (CLAUDE.md, project-wide)
- `snake_case` filenames and functions; verb-noun function names (`pixel_to_world_xyz`, not `xyz_from_pixel`)
- `print(f"... → {out_path}")` arrow-notation for any new file-output logging (perception module, if it saves anything)
- `MUJOCO_GL` must already be set before any MuJoCo import — applies to `perception/test_depth_xyz.py` if it imports MuJoCo/robosuite directly

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| `LIBERO/libero/libero/perception/depth_xyz.py` | utility/service | transform | First perception/projective-geometry module in the codebase; build directly on `robosuite.utils.camera_utils` (external, verified) per RESEARCH.md Pattern 3, no in-repo transform-role precedent |
| `LIBERO/libero/libero/perception/test_depth_xyz.py` | test | integration | No existing perception test; follow project pytest convention (no config file, per-module `pytest <path> -x -q`) |
| `LIBERO/libero/libero/envs/test_spatial_predicates.py` | test | unit + integration | No existing predicate test file found anywhere in the repo — this phase establishes the first one |

## Metadata

**Analog search scope:** `LIBERO/libero/libero/vla/`, `LIBERO/libero/libero/envs/`, `LIBERO/libero/libero/envs/predicates/`, `LIBERO/libero/libero/envs/object_states/`, `LIBERO/libero/libero/envs/problems/`, `LIBERO/libero/libero/bddl_files/libero_goal/`, `LIBERO/libero/libero/bddl_files/libero_spatial/`
**Files scanned:** 9 read directly this session (eval_loop.py, interface.py, pi0_backend.py, oft_backend.py, env_wrapper.py, base_predicates.py, predicates/__init__.py, libero_tabletop_manipulation.py [_check_success/_eval_predicate section], base_object_states.py [ObjectState section], put_the_cream_cheese_in_the_bowl.bddl)
**Pattern extraction date:** 2026-08-09
