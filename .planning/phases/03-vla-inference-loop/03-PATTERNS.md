# Phase 3: VLA Inference Loop - Pattern Map

**Mapped:** 2026-07-18
**Files analyzed:** 6
**Analogs found:** 6 / 6

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|--------------------|------|-----------|-----------------|----------------|
| `LIBERO/libero/libero/vla/interface.py` | interface/protocol (utility) | request-response | `LIBERO/libero/libero/envs/robots/on_the_ground_panda.py` (abstract-property contract shape) + `LIBERO/libero/libero/envs/env_wrapper.py` (thin-wrapper method contract) | role-match |
| `LIBERO/libero/libero/vla/__init__.py` | module/registration | event-driven (import side-effects) | `LIBERO/libero/libero/envs/robots/__init__.py` | exact |
| `LIBERO/libero/libero/vla/oft_backend.py` | service (model-inference wrapper) | request-response | Phase 1's confirmed OFT loading code (`01-CONTEXT.md` loading pattern) + `LIBERO/libero/lifelong/metric.py::raw_obs_to_tensor_obs`/`evaluate_one_task_success` (obs→tensor→action shape) | role-match |
| `LIBERO/libero/libero/vla/pi0_backend.py` | service (remote-inference client wrapper) | request-response (websocket RPC) | `LIBERO/libero/libero/vla/oft_backend.py` (sibling backend, same interface) — no prior websocket-client code exists in-repo | no analog (new pattern class) |
| `LIBERO/libero/libero/vla/eval_loop.py` | service/orchestrator (eval-loop + success polling + video) | event-driven / batch (per-step polling loop) | `LIBERO/libero/lifelong/metric.py::evaluate_one_task_success` (polling+success loop) + `LIBERO/libero/libero/utils/video_utils.py::VideoWriter` (video save) | exact (loop) / exact (video) |
| `LIBERO/notebooks/03a-oft-inference-eval.ipynb` | notebook (test/verification) | request-response (Colab cell execution) | `LIBERO/notebooks/02-soarm-integration-check.ipynb` (PASS/FAIL block structure, EGL/config/sys.path bootstrap cells) | exact |
| `LIBERO/notebooks/03b-pi0-inference-smoketest.ipynb` | notebook (test/verification) | request-response (Colab cell execution, separate kernel) | `LIBERO/notebooks/02-soarm-integration-check.ipynb` (same PASS/FAIL convention, adapted for a second, lighter kernel/env) | role-match |

## Pattern Assignments

### `LIBERO/libero/libero/vla/__init__.py` (module/registration)

**Analog:** `LIBERO/libero/libero/envs/robots/__init__.py` (21 lines, full file read)

**Full pattern** (lines 1-21):
```python
from .mounted_panda import MountedPanda
from .on_the_ground_panda import OnTheGroundPanda

# Defining the class auto-registers it in robosuite REGISTERED_ROBOTS (metaclass)
from .soarm import MountedSoarm101

from robosuite.robots.single_arm import SingleArm
from robosuite.robots import ROBOT_CLASS_MAPPING

from robosuite.models.grippers import GRIPPER_MAPPING
from ..grippers.soarm_gripper import SoarmGripper

ROBOT_CLASS_MAPPING.update(
    {
        "MountedPanda": SingleArm,
        "OnTheGroundPanda": SingleArm,
        "MountedSoarm101": SingleArm,
    }
)

GRIPPER_MAPPING["SoarmGripper"] = SoarmGripper  # gripper_factory asserts membership
```

**What to copy:** the "import triggers registration side effect, then explicitly wire the mapping dict" convention. `vla/__init__.py` should import `OFTBackend` and `Pi0Backend` (and re-export `VLABackend`/`predict` interface) so `from libero.libero.vla import OFTBackend, Pi0Backend` works cleanly for Phase 4/6 consumers — mirrors this project's "real modules, not notebook-embedded code" convention (CONTEXT.md code_context).

---

### `LIBERO/libero/libero/vla/interface.py` (interface/protocol)

**Analog:** `LIBERO/libero/libero/envs/robots/on_the_ground_panda.py` (abstract-property-contract shape) + `env_wrapper.py`'s thin delegation pattern (lines 87-104)

**Contract-via-properties pattern** (on_the_ground_panda.py lines 22-33 — showing how this codebase defines a class contract other classes must satisfy):
```python
@property
def default_mount(self):
    return None

@property
def default_gripper(self):
    return "PandaGripper"

@property
def default_controller_config(self):
    return "default_panda"
```

**Thin-delegation pattern** (env_wrapper.py lines 87-104 — showing how this codebase writes minimal wrapper methods with a single clear responsibility):
```python
def step(self, action):
    return self.env.step(action)

...

def check_success(self):
    return self.env._check_success()
```

**What to copy:** Follow RESEARCH.md's own `Protocol`-based interface exactly (already drafted concretely in RESEARCH.md Pattern 1) — this project has no existing `typing.Protocol` usage, so the closest *stylistic* precedent is the property-contract style above (small, single-purpose methods, no framework machinery, plain docstring documenting the contract). Do not introduce a new ABC/ABCMeta pattern; keep it as lightweight as the robot classes above.

```python
# Target pattern (from RESEARCH.md, confirmed against project style)
from typing import Protocol
import numpy as np
from PIL import Image

class VLABackend(Protocol):
    def predict(self, images: dict[str, Image.Image], language: str) -> np.ndarray:
        """Returns action chunk (T, 7) or single (7,) action.
        Each backend normalizes internally (D-02) — no shared normalization layer.
        """
        ...
```

---

### `LIBERO/libero/libero/vla/oft_backend.py` (service, request-response)

**Analog:** Phase 1's confirmed OFT loading pattern (per `01-CONTEXT.md` / `01-DEBUG-HISTORY.md` canonical refs) + `metric.py::raw_obs_to_tensor_obs` for the obs-dict → model-input shape convention.

**Obs-to-tensor-input pattern to mirror the *spirit* of** (metric.py lines 19-49 — this project's established way of turning a raw env obs dict into model input, i.e. explicit per-key dict construction, no magic):
```python
def raw_obs_to_tensor_obs(obs, task_emb, cfg):
    """
    Prepare the tensor observations as input for the algorithm.
    """
    env_num = len(obs)
    data = {"obs": {}, "task_emb": task_emb.repeat(env_num, 1)}
    ...
    data = TensorUtils.map_tensor(data, lambda x: safe_device(x, device=cfg.device))
    return data
```

**Confirmed Phase 1 facts to reuse directly (from canonical_refs in 03-CONTEXT.md, not re-derived):**
- `predict_action` returns `(actions, hidden_states)`; `actions` shape `(8, 7)` float64.
- OFT checkpoint norm_stats overlay: `dataset_statistics.json` via `hf_hub_download`, keyed `"libero_spatial_no_noops"`, applied AFTER `from_pretrained`.
- Consume the full 8-step chunk open-loop (D-03/RESEARCH.md recommendation) — do not re-infer every step.

**Target shape** (RESEARCH.md Pattern 1, oft_backend.py example, lines 214-225):
```python
class OFTBackend:
    def __init__(self, checkpoint="moojink/openvla-7b-oft-finetuned-libero-spatial"):
        # from_pretrained(...) then overlay norm_stats from
        # dataset_statistics.json using key "libero_spatial_no_noops" (Phase 1 confirmed)
        ...

    def predict(self, images: dict, language: str) -> np.ndarray:
        actions, _hidden = self.model.predict_action(images["eye_in_hand"], language, ...)
        return actions  # shape (8, 7) float64 — full chunk, consumed open-loop (D-03)
```

---

### `LIBERO/libero/libero/vla/pi0_backend.py` (service, websocket RPC)

**Analog:** No in-repo precedent for a remote/websocket client — this is a genuinely new pattern class for this codebase. Use `oft_backend.py` (sibling file, same phase) purely for structural consistency (same class shape: `__init__` loads/connects, `predict(images, language)` returns ndarray), and openpi's own documented client pattern for the actual RPC mechanics (RESEARCH.md Pattern 1, pi0_backend.py example + Code Examples section).

**Structural consistency to copy from `oft_backend.py`:** same method signature `predict(self, images: dict, language: str) -> np.ndarray`, same "normalize internally, no shared layer" convention (D-02).

**RPC mechanics (from RESEARCH.md, openpi's own documented pattern — treat as authoritative since no in-repo analog exists):**
```python
from openpi_client import websocket_client_policy as wcp
from openpi_client import image_tools

class Pi0Backend:
    def __init__(self, host="localhost", port=8000):
        self.client = wcp.WebsocketClientPolicy(host, port)

    def predict(self, images: dict, language: str) -> np.ndarray:
        obs = {
            "observation/image": image_tools.convert_to_uint8(
                image_tools.resize_with_pad(images["eye_in_hand"], 224, 224)
            ),
            "observation/wrist_image": image_tools.convert_to_uint8(
                image_tools.resize_with_pad(images["eye_in_hand"], 224, 224)
            ),
            "observation/state": self._robot_state(),
            "prompt": language,
        }
        result = self.client.infer(obs)
        return result["actions"]
```

**Security note (from RESEARCH.md Security Domain):** bind server to `localhost`/`127.0.0.1` only, not `0.0.0.0`.

---

### `LIBERO/libero/libero/vla/eval_loop.py` (orchestrator, per-step polling + video)

**Analog A — success-polling loop:** `LIBERO/libero/lifelong/metric.py::evaluate_one_task_success` (lines 130-157, read in full)

```python
while steps < cfg.eval.max_steps:
    steps += 1
    data = raw_obs_to_tensor_obs(obs, task_emb, cfg)
    actions = algo.policy.get_action(data)
    obs, reward, done, info = env.step(actions)
    for k in range(env_num):
        dones[k] = dones[k] or done[k]
    if all(dones):
        break
```
**What to copy:** the per-step `done`-accumulation-then-break pattern (D-12's "poll every step, stop early"). Replace `algo.policy.get_action(data)` with `backend.predict(images, language)`, and simplify away the vectorized-env (`env_num`/`SubprocVectorEnv`) machinery per RESEARCH.md Pitfall 3 — write a simple single-process loop, not the full vectorized `metric.py` apparatus.

**Success-check primitive to call, not reimplement** (`env_wrapper.py` lines 103-104, read in full):
```python
def check_success(self):
    return self.env._check_success()
```
Also note `env.step()`'s `done` return already reflects this internally (env_wrapper.py step delegation, line 87-88) — the eval loop should just read `done` from `env.step()`, per RESEARCH.md Anti-Patterns.

**Analog B — video saving:** `LIBERO/libero/libero/utils/video_utils.py::VideoWriter` (77 lines, read in full)

```python
class VideoWriter:
    def __init__(self, video_path, save_video=False, fps=30, single_video=True):
        ...
    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb): self.save()

    def append_obs(self, obs, done, idx=0, camera_name="agentview_image"):
        # not done -> append obs[camera_name][::-1] (frame is bottom-up, needs flip)
        # done -> overlay a green-tinted "success" blank frame on the last image
        ...

    def save(self):
        if self.save_video:
            os.makedirs(self.video_path, exist_ok=True)
            if self.single_video:
                video_name = os.path.join(self.video_path, f"video.mp4")
                video_writer = imageio.get_writer(video_name, fps=self.fps)
                ...
            print(f"Saved videos to {self.video_path}.")
```
**What to copy:** use `VideoWriter` directly (do not reimplement) with `single_video=True` per D-11; call `.append_obs(obs, done, camera_name="robot0_eye_in_hand_image")` (or `"agentview_image"` for the display camera — confirm exact obs key against Phase 2's `02-VERIFICATION.md` per RESEARCH.md Pitfall 4) inside the same per-step loop as the success check; the `with VideoWriter(...) as vw:` context-manager idiom auto-saves on exit — matches this project's arrow-notation print convention (`Saved videos to {path}.` / project's `Saved → {out_path}` convention from CLAUDE.md Logging section — align `eval_loop.py`'s own print statements with the `→` arrow convention for consistency, e.g. `print(f"Episode {ep_idx}: {'PASS' if done else 'FAIL'} → {video_path}")`).

**PASS/FAIL print pattern** (RESEARCH.md Pattern 2, matches CLAUDE.md Logging conventions):
```python
print(f"Episode {ep_idx}: {'PASS' if done else 'FAIL'} (steps={step+1})")
```

---

### `LIBERO/notebooks/03a-oft-inference-eval.ipynb` (notebook)

**Analog:** `LIBERO/notebooks/02-soarm-integration-check.ipynb` (26 cells, read in full via cell-source scan)

**Structure to replicate exactly:**
1. Markdown header cell stating phase/purpose and referencing prior phase's finalized outputs (cell 0 pattern: "Re-prove every Phase 2 requirement... local `soarm_sanity.py --check all` is 6/6 GREEN; this notebook is the Colab half.")
2. Drive mount + repo unzip cell (cell 1).
3. `REPO_ROOT` user-configuration cell with inline comments explaining Drive path assumption (cell 2).
4. GPU assertion cell early, with an explicit comment on *why* this tier is needed (cell 3 pattern — adapt: OFT notebook needs A100/bf16, unlike Phase 2's T4-is-enough note).
5. `## BLOCK A: Install` / `## BLOCK B: verification` markdown section dividers, with an explicit **"*** STOP — Restart runtime now ***"** cell between them (cell 13) — mandatory per this project's "restart ≠ clean slate" environment contract (01-DEBUG-HISTORY.md, referenced in 03-CONTEXT.md canonical_refs).
6. Each Block A install cell is numbered "Step N of M" with a comment explaining *why* the ordering matters (cells 5-12) — e.g. cell 6: "Step 2 of 6 — PyTorch 2.2.0 (cu121) / Must run BEFORE flash-attn".
7. Post-restart Block B always starts with: EGL bootstrap cell → LIBERO `~/.libero/config.yaml` bootstrap cell → `sys.path` setup cell → matplotlib `Agg` backend + numba shim cell (cells 15-18) — **in that exact order**, per the project's environment-contract invariants.
8. Individual requirement-verification cells are labeled with the requirement ID as a comment header (e.g. `# ENV-04: SOARM MJCF compiles + robot class registered`, `# SC-1: reset physics stability...`) — Phase 3 should label cells `# VLA-01: ...`, `# VLA-02: ...`, etc. to mirror this ID-per-cell convention.
9. Final markdown "Phase N Summary" cell with a checklist table of requirement → check → status (☐) → notes (cell 25) — Phase 3 should end with the aggregated PASS/FAIL summary table (D-14) in this same table format.

**Concrete excerpt — GPU assertion cell** (cell 3):
```python
import torch

assert torch.cuda.is_available(), (
    "No GPU available. Go to Runtime > Change runtime type > Hardware accelerator > GPU."
)

gpu_name = torch.cuda.get_device_name(0)
vram_g = ...
```

**Concrete excerpt — requirement-verification cell labeling convention** (cell 19):
```python
# ENV-04: SOARM MJCF compiles + robot class registered
# Automated: mujoco.MjModel.from_xml_path on the arm and gripper XMLs
# (repo-relative paths from Cell 1), then import libero.libero.envs.robots
# (registration side effects) and assert MountedSoarm101 landed in
# robosuite's REGISTERED_ROBOTS.
```

**Concrete excerpt — Summary table format** (cell 25, markdown):
```markdown
## Phase 2 Summary

| Requirement | Check | Status | Notes |
|-------------|-------|--------|-------|
| ENV-04: SOARM MJCF compiles + class registered | ENV-04 cell | ☐ | Update after running |
```
Adapt directly for Phase 3: rows = VLA-01..VLA-04 plus per-episode PASS/FAIL and the aggregated success-rate table (D-14).

---

### `LIBERO/notebooks/03b-pi0-inference-smoketest.ipynb` (notebook)

**Analog:** Same `02-soarm-integration-check.ipynb` convention as 03a, but this notebook is a **separate kernel/environment** (per D-06/RESEARCH.md — openpi's own `uv`-managed venv, not the OFT torch/transformers stack).

**What differs from 03a (do not copy these parts):**
- Block A install cells install openpi via `uv sync` / `uv pip install -e .` (RESEARCH.md Installation section), not the OFT/transformers-fork install chain.
- No `~/.libero/config.yaml` bootstrap is needed in the client role if this notebook only runs the `WebsocketClientPolicy` client against a server process (openpi's own env may not need the full LIBERO env bootstrap at all — confirm exact split: LIBERO env + client can stay in one kernel while `serve_policy.py` runs in a separate one, per RESEARCH.md's architecture diagram).
- Verification cells are labeled `# VLA-04: ...` (interface swap proof) rather than repeating `# VLA-01..03` (those are OFT's full-suite job per D-09/D-10).

**What to copy identically:** the Drive-mount cell, `REPO_ROOT` cell, GPU assertion cell, BLOCK A/BLOCK B markdown dividers + explicit restart-stop cell, and the final Summary-table markdown convention (same table format, scoped to VLA-04 + the 1-task × 1-2 episode smoke-test PASS/FAIL only).

## Shared Patterns

### Per-step success polling + early stop (D-12)
**Source:** `LIBERO/libero/lifelong/metric.py::evaluate_one_task_success` lines 130-157; primitive at `LIBERO/libero/libero/envs/env_wrapper.py` lines 87-104.
**Apply to:** `eval_loop.py`, both notebooks' eval cells.
```python
while steps < max_steps:
    steps += 1
    obs, reward, done, info = env.step(action)
    if done:
        break
```

### Per-episode video saving (D-11)
**Source:** `LIBERO/libero/libero/utils/video_utils.py::VideoWriter`, `single_video=True` mode.
**Apply to:** `eval_loop.py` (used by both notebooks).
```python
with VideoWriter(video_path=f"outputs/videos/task_{task_id}_ep_{ep_idx}",
                 save_video=True, fps=30, single_video=True) as vw:
    for step in range(max_steps):
        vw.append_obs(obs, done, idx=0, camera_name="robot0_eye_in_hand_image")
        if done:
            break
```

### PASS/FAIL notebook cell + summary table convention
**Source:** `LIBERO/notebooks/02-soarm-integration-check.ipynb` (Block A/B structure, requirement-ID-per-cell comments, final Summary table with ☐ checkboxes).
**Apply to:** both new notebooks — this is this project's only test framework (per RESEARCH.md Validation Architecture: "no pytest/unittest suite ... verification is PASS/FAIL notebook cells").

### Module registration via import side-effect
**Source:** `LIBERO/libero/libero/envs/robots/__init__.py`.
**Apply to:** `LIBERO/libero/libero/vla/__init__.py` — re-export `VLABackend`, `OFTBackend`, `Pi0Backend` so downstream Phase 4/6 code does `from libero.libero.vla import OFTBackend`.

### Arrow-notation logging convention
**Source:** CLAUDE.md Logging conventions (`Saved → {out_path}`) + `video_utils.py`'s own `print(f"Saved videos to {self.video_path}.")`.
**Apply to:** `eval_loop.py` print statements — align new prints (episode PASS/FAIL, summary table) with the existing arrow/descriptive-text style already used across this codebase's scripts.

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| `LIBERO/libero/libero/vla/pi0_backend.py` | service | event-driven / websocket RPC | No existing remote-inference or websocket-client code exists anywhere in this repo (confirmed via INTEGRATIONS.md: "no existing VLA/model-serving integration"). Planner must rely on RESEARCH.md's openpi-sourced code examples (Code Examples section, `WebsocketClientPolicy` usage) as the primary reference, using `oft_backend.py`'s sibling structure only for interface-shape consistency. |

## Metadata

**Analog search scope:** `LIBERO/libero/libero/envs/robots/`, `LIBERO/libero/libero/envs/env_wrapper.py`, `LIBERO/libero/libero/utils/video_utils.py`, `LIBERO/libero/lifelong/metric.py`, `LIBERO/libero/lifelong/utils.py`, `LIBERO/notebooks/01-colab-env-setup.ipynb`, `LIBERO/notebooks/02-soarm-integration-check.ipynb`.
**Files scanned:** 7 (all read in full or via targeted cell-source scan; no file exceeded 2000 lines, no re-reads).
**Pattern extraction date:** 2026-07-18
</content>
