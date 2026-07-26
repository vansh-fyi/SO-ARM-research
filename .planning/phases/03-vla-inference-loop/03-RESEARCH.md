# Phase 3: VLA Inference Loop - Research

**Researched:** 2026-07-18
**Domain:** Vision-Language-Action (VLA) model inference loops in LIBERO/MuJoCo simulation; dual-backend (OpenVLA-OFT + π0/openpi) integration on Colab
**Confidence:** MEDIUM-HIGH (LIBERO/OFT facts are HIGH — confirmed against this repo's own code and Phase 1 Colab evidence; openpi facts are MEDIUM — cross-checked against official GitHub sources but not yet run on Colab in this project)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**VLA Interface & Action Consumption**
- D-01: Shared interface signature is `predict(images: dict[str, Image], language: str) -> action`, where `images` is a dict of named camera views (not a single image) — future-proofs for Phase 5 multi-camera work without a breaking change.
- D-02: Each backend (OpenVLA-OFT, π0) handles its own action/observation normalization internally inside its `predict()` implementation. No shared normalization layer.

**π0 Integration Depth**
- D-05: π0 (openpi) must run **real, working inference on Colab** in this phase — not an interface-only stub. This is locked, not up for debate.

**Task & Prompt Scope**
- D-08: The demo loop runs **all 3** frozen `libero_spatial` tasks from Phase 2, looped automatically in a single notebook run.
- D-09: OpenVLA-OFT runs the **full suite**: 3 tasks × 5-10 episodes each, to produce a real success-rate number for VLA-03.
- D-10: π0 runs a **lighter smoke-test only**: e.g. 1 task × 1-2 episodes. π0's job is to prove the interface swap works, not to match OFT's evaluation depth.
- D-11: Video output is **one video file per episode** (not one combined video per task) — matches LIBERO's existing per-episode `video_utils.py` save pattern.

**Success Detection & Episode Termination**
- D-12: Poll LIBERO's `check_success()` **every step** during an episode; stop the episode early the moment it returns true.
- D-14: Reporting is **printed PASS/FAIL per episode, plus an aggregated success-rate summary table at the end** (across all episodes/tasks/backends).

### Claude's Discretion (research must inform these)
- D-03: OFT action-chunk consumption strategy (open-loop vs. closed-loop re-inference) — **researched below, recommendation given**.
- D-04: Exact module location for the shared VLA interface — **researched below, recommendation given**.
- D-06: Single vs. split notebook/kernel structure for OFT vs. π0 — **researched below, recommendation given**.
- D-07: Specific π0 checkpoint/variant — **researched below, recommendation given**.
- D-13: Exact max-step cap value — **researched below, recommendation given**.
- D-15: Whether to capture a near-miss/partial-credit signal on timeout — **researched below, recommendation given**.

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope. Multi-camera input, 3D localization, and spatial-language prompts remain deferred to Phase 5; fine-tuning remains deferred to Phase 6.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| VLA-01 | OpenVLA-OFT receives (image, language prompt) and outputs 7-D delta EEF action that drives SOARM in sim | Phase 1 confirmed `predict_action` returns `(actions, hidden_states)`, actions shape `(8,7)` float64; norm_stats overlay pattern confirmed. Research below confirms `num_open_loop_steps` chunk-consumption pattern and the exact 7-D action layout (6 OSC_POSE dims + 1 gripper dim, matching SOARM's confirmed 7-DoF action contract from Phase 2). |
| VLA-02 | Episode frames are saved as a video file for inspection after each inference run | `LIBERO/libero/libero/utils/video_utils.py::VideoWriter` (existing, reusable) confirmed to support per-episode single-video mode (`single_video=True`) — directly satisfies D-11. |
| VLA-03 | LIBERO task success rate is measured using LIBERO's evaluation protocol for SOARM tasks | `LIBERO/libero/lifelong/metric.py::evaluate_one_task_success` and `ControlEnv.check_success()` (calls `self.env._check_success()`) confirmed as the canonical polling pattern this phase must replicate per-step (D-12). `max_steps: 600` confirmed as this project's own LIBERO eval config default. |
| VLA-04 | π0 (openpi) inference loop is integrated as a second VLA option alongside OpenVLA-OFT | openpi's `pi0_fast_libero` / `pi0_libero` configs, checkpoint auto-download, and `WebsocketClientPolicy` interface researched below; dependency conflict with OFT's stack confirmed, justifying a split-kernel notebook design. |
</phase_requirements>

## Summary

This phase closes the language→action→sim loop by wiring two independent VLA backends — OpenVLA-OFT (already proven in Phase 1/2) and π0/openpi (new to this project) — behind a single `predict(images, language) -> action` interface, running against the 3 frozen `libero_spatial` SOARM tasks from Phase 2.

The two backends have genuinely incompatible Python dependency stacks: OFT needs `torch==2.2.0` + `transformers==4.40.1` (established in Phase 1); openpi's own `pyproject.toml` pins `torch==2.7.1` + `transformers==4.53.2` + `jax[cuda12]==0.5.3`. This is not a hypothetical risk — it is a confirmed, concrete version mismatch on the same two packages Phase 1's debug history already flagged as the project's most fragile dependency surface (numpy/protobuf/transformers). The two backends **must** run in separate Colab kernels/notebooks, mirroring the LIBERO-training-vs-VLA-inference split already documented in STATE.md.

For OFT's action-chunk consumption (D-03), the OFT reference implementation's own quick-start sets `num_open_loop_steps = NUM_ACTIONS_CHUNK` — i.e., **full open-loop replay of the entire 8-step chunk** before the next inference call. This is the standard/default behavior in the reference LIBERO eval script (`experiments/robot/libero/run_libero_eval.py`) and is the simplest, fastest, most proven path for an MVP phase; closed-loop re-inference after fewer steps is a research-paper-level optimization (see Speculative Verification for VLA) not needed here.

For π0 (D-06/D-07), the recommended integration path is: run openpi as a **separate process/kernel** via its own `serve_policy.py --env=LIBERO` websocket server (openpi's own documented remote-inference pattern, designed exactly for "keeping robot and policy environments separate to avoid dependency conflicts"), with the OFT notebook (or a lightweight client-side script) acting as the `WebsocketClientPolicy` client. This sidesteps the torch/transformers/jax conflict entirely without needing pip trickery, and it is openpi's own blessed pattern — not a workaround this project has to invent. Use the `pi0_fast_libero` config (autoregressive FAST tokenizer, generally faster/lighter than flow-based `pi0_libero`) for the smoke test, loading Physical Intelligence's own LIBERO-finetuned checkpoint from `gs://openpi-assets/checkpoints/pi0_fast_base/params`.

**Primary recommendation:** Two-notebook (two-kernel) design: Notebook A extends Phase 1/2's OFT+LIBERO environment to run the full 3-task × 5-10 episode OFT eval loop with per-step `check_success()` polling, `max_steps=600` (this project's existing LIBERO config default), full open-loop 8-step chunk replay, and per-episode video saving. Notebook B is a minimal openpi environment (fresh venv/kernel via `uv`) that starts `serve_policy.py --env=pi0_fast_libero` and runs a lightweight websocket client against the *same* LIBERO env/task loop for a 1-task × 1-2 episode smoke test, proving the `predict()` interface swap without touching Notebook A's dependency tree.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| LIBERO env stepping / BDDL task execution | Simulation (MuJoCo/robosuite, in-process) | — | Existing `OffScreenRenderEnv` / `ControlEnv` from Phase 1/2; no change needed |
| OpenVLA-OFT inference (`predict_action`) | Simulation-adjacent (same Colab kernel/process as env, in-process torch model) | — | Model and sim share GPU + Python process for simplicity; matches Phase 1's proven pattern |
| π0/openpi inference | Separate process (websocket server), client called from sim process | — | Confirmed real dependency conflict (torch 2.7.1/transformers 4.53.2/jax vs. OFT's torch 2.2.0/transformers 4.40.1) forces process isolation; openpi's own documented remote-inference pattern is designed for exactly this |
| Shared `predict()` interface module | Application/glue code (plain Python module, no framework) | — | Imported by both backends' calling code and by Phase 4/6 later; must have zero heavy deps itself |
| Video recording | Simulation output (existing `VideoWriter`) | — | Reuse, no new component |
| Success/failure detection | Simulation (env-level `_check_success()` / `done` flag) | Reporting (notebook print + summary table) | `check_success()` is already computed inside `env.step()`'s `done` return — the phase adds explicit polling/printing on top, not a new detection mechanism |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `openvla-oft` (moojink fork) | main branch, `--no-deps` (per Phase 1 contract) | OFT VLA backend | Already proven working on Colab A100 in Phase 1; this phase only adds LIBERO eval-loop wiring |
| `openpi` (Physical-Intelligence/openpi) | main branch (git submodules) `[ASSUMED — no pinned release tag found; repo tracks main]` | π0 VLA backend | Official, only maintained implementation of π0/π0-FAST; ships built-in LIBERO configs |
| `openpi-client` | 0.1.2 `[VERIFIED: pypi registry — pip index versions]` | Thin websocket client for calling a remote openpi policy server | Official companion package from the same org; documented as the standard way to call `serve_policy.py` remotely |
| `jax[cuda12]` | ==0.5.3 (pinned by openpi's pyproject.toml) `[CITED: github.com/Physical-Intelligence/openpi/blob/main/pyproject.toml]` | JAX/CUDA backend for π0 inference | Required transitively by openpi; not directly imported by this project's own code |
| `imageio` | already a transitive dep of `LIBERO/libero/libero/utils/video_utils.py` | Video writing | Already used by the existing `VideoWriter` class; no new dependency needed |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `uv` | latest `[CITED: openpi README/DeepWiki install docs]` | Python package/venv manager for openpi's install | Required by openpi's own documented install flow (`uv sync`, `uv run`); do not try to `pip install` openpi's requirements manually — its lockfile is uv-native |
| `websockets` | transitive dep of `openpi-client` | Websocket transport for remote inference | Installed automatically with `openpi-client`; do not pin separately |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Two-notebook/kernel split (recommended) | Single notebook with a subprocess-isolated venv for openpi (e.g. `venv`+`subprocess.run`) | Technically avoids a second Colab tab, but re-creates the same `--no-deps` fragility class Phase 1's debug history warns against; the websocket-server pattern is openpi's own blessed approach and less fragile |
| `pi0_fast_libero` (recommended) | `pi0_libero` (flow-based, non-FAST) | Flow-based pi0 requires iterative denoising at inference (slower); FAST's autoregressive tokenizer is lighter-weight for a smoke test. Either works; FAST is the safer Colab-timeout choice |
| Open-loop full-chunk replay (recommended, D-03) | Closed-loop re-inference every N<8 steps | Re-inference every step is what OFT's paper calls the "slow" baseline it was specifically designed to beat; not needed for an MVP smoke/eval loop, and this project's own Phase 1 confirmed 8-step chunks specifically to amortize inference cost |

**Installation (Notebook A — OFT, extends Phase 1's Block A):**
```bash
# No new installs needed beyond Phase 1's baked environment.
# This phase ADDS eval-loop code, not new packages, to the OFT notebook.
```

**Installation (Notebook B — openpi, separate kernel):**
```bash
git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git
cd openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
pip install openpi-client   # if calling from a separate lightweight client process
```

**Version verification:** `openpi-client` confirmed via `pip index versions openpi-client` → `0.1.2` (latest), `0.1.1` prior. `jax`, `flax`, `orbax-checkpoint` confirmed to exist on PyPI with long version histories (`jax` up to 0.11.0, `flax` up to 0.12.7, `orbax-checkpoint` up to 0.12.1) — openpi's pin (`jax==0.5.3`) is several versions behind current `jax` HEAD, which is normal/expected for a pinned research dependency, not a red flag.

## Package Legitimacy Audit

| Package | Registry | Age | Downloads | Source Repo | Verdict | Disposition |
|---------|----------|-----|-----------|--------------|---------|-------------|
| `openpi-client` | PyPI | published 2026-03-30 (2 versions total) | unknown (registry API returned null) | none listed in registry metadata, but is a subpackage of `github.com/Physical-Intelligence/openpi` (`packages/openpi-client/`) `[CITED]` | SUS (tool heuristic: unknown-downloads, no-repository) | **Flagged — approved with human-verify checkpoint.** This is Physical Intelligence's own official companion package (same org as the primary `openpi` repo); low download count and missing repo-URL metadata are expected for a young, narrow-purpose research-lab package, not evidence of typosquatting. Planner must add `checkpoint:human-verify` before this specific install. |
| `jax` | PyPI | long-established (100+ historical versions back to 0.0) | unknown (registry API returned null) | `github.com/jax-ml/jax` (confirmed in tool output) | SUS (tool heuristic: too-new [latest release], unknown-downloads) | Approved — Google's flagship array/autodiff library, extremely well-known; "too-new" signal refers to the latest point release, not the package's age. No checkpoint needed but noted for completeness. |
| `flax` | PyPI | long-established (60+ historical versions back to 0.1.0) | unknown (registry API returned null) | not returned by registry metadata, but is `github.com/google/flax` `[CITED]` | SUS (tool heuristic: unknown-downloads, no-repository) | Approved — standard JAX-ecosystem neural-net library from Google; registry metadata gap, not a legitimacy signal. |
| `orbax-checkpoint` | PyPI | long-established (80+ historical versions) | unknown (registry API returned null) | not returned by registry metadata, but is `github.com/google/orbax` `[CITED]` | SUS (tool heuristic: too-new, unknown-downloads, no-repository) | Approved — Google's standard JAX checkpointing library; same registry-metadata-gap pattern as above. |

**Packages removed due to [SLOP] verdict:** none.
**Packages flagged as suspicious [SUS]:** `openpi-client` — planner must insert a `checkpoint:human-verify` task immediately before this specific `pip install openpi-client` step (confirm the installed package resolves to `Physical-Intelligence/openpi`'s own client, not a same-named impostor, by checking `pip show openpi-client` output for a matching author/homepage field before proceeding). `jax`, `flax`, `orbax-checkpoint` are also technically flagged SUS by the tool's heuristics (which key heavily on registry metadata completeness) but are treated as approved without a checkpoint given their status as foundational, decades-cited-equivalent infrastructure from Google/JAX-ml — the planner may skip a checkpoint for these three specifically, but should still note the tool's SUS verdict in the plan's audit trail for transparency.

## Architecture Patterns

### System Architecture Diagram

```text
                         ┌─────────────────────────────┐
                         │   Language Prompt (string)   │
                         │  "pick up the black bowl..." │
                         └──────────────┬───────────────┘
                                        │
                    ┌───────────────────┴────────────────────┐
                    │                                          │
                    ▼                                          ▼
      ┌─────────────────────────────┐          ┌──────────────────────────────┐
      │  NOTEBOOK A (Colab kernel 1) │          │  NOTEBOOK B (Colab kernel 2)  │
      │  torch 2.2.0 / transformers  │          │  openpi venv (uv) — jax/torch │
      │  4.40.1 (Phase 1 env)        │          │  2.7.1 / transformers 4.53.2  │
      │                              │          │                               │
      │  ┌────────────────────────┐  │          │  ┌─────────────────────────┐  │
      │  │ LIBERO env.reset()     │  │          │  │ serve_policy.py         │  │
      │  │ (SOARM, libero_spatial)│  │          │  │ --env=pi0_fast_libero   │  │
      │  └───────────┬────────────┘  │          │  │ (websocket server :8000)│  │
      │              │ obs (image,   │          │  └────────────┬────────────┘  │
      │              │  eye_in_hand) │          │               │ actions        │
      │              ▼               │          │               │ (chunk)        │
      │  ┌────────────────────────┐  │          │  ┌────────────▼────────────┐  │
      │  │ predict(images,lang)   │  │          │  │ WebsocketClientPolicy   │  │
      │  │  -> OFT.predict_action │◄─┼──────────┼──┤  .infer(obs) client     │  │
      │  │  (8,7) chunk           │  │  same    │  │  (same LIBERO env loop  │  │
      │  └───────────┬────────────┘  │  shared  │  │   as Notebook A, but    │  │
      │              │ actions[0..7] │  interface│  │   1 task x 1-2 eps)     │  │
      │              ▼               │  contract│  └─────────────────────────┘  │
      │  ┌────────────────────────┐  │          │                               │
      │  │ env.step(action) x8    │  │          └───────────────────────────────┘
      │  │ (open-loop replay,     │  │
      │  │  D-03)                │  │
      │  │ poll check_success()  │  │
      │  │ every step (D-12)      │  │
      │  └───────────┬────────────┘  │
      │              │ done? no -> loop; yes -> stop early
      │              ▼               │
      │  ┌────────────────────────┐  │
      │  │ VideoWriter (per-      │  │
      │  │ episode .mp4, D-11)    │  │
      │  └───────────┬────────────┘  │
      │              ▼               │
      │  ┌────────────────────────┐  │
      │  │ print PASS/FAIL per ep │  │
      │  │ + summary table (D-14) │  │
      │  └────────────────────────┘  │
      └───────────────────────────────┘
```

### Recommended Project Structure
```
LIBERO/libero/libero/vla/              # D-04: shared interface lives in the LIBERO fork,
├── __init__.py                        # not explorations/ — Phase 4/6 import from here too,
├── interface.py                       # and it needs LIBERO env types (images dict keys
├── oft_backend.py                     # match camera_names) already defined in this package
├── pi0_backend.py                     # (mirrors D-08 precedent from Phase 2: real modules,
└── eval_loop.py                       # not notebook-embedded code)
LIBERO/notebooks/
├── 03a-oft-inference-eval.ipynb       # Notebook A: OFT full eval, 3 tasks x 5-10 eps
├── 03b-pi0-inference-smoketest.ipynb  # Notebook B: openpi smoke test, 1 task x 1-2 eps
└── outputs/
    └── videos/                        # per-episode .mp4 files (D-11)
```

### Pattern 1: Shared `predict()` Interface (D-01, D-02, D-04)
**What:** A minimal abstract-ish Python interface (not necessarily a formal ABC — a documented function signature both backends implement) that both `OFTBackend.predict()` and `Pi0Backend.predict()` satisfy.
**When to use:** Any place the eval loop needs an action from a VLA — the eval loop code never branches on which backend is active.
**Example:**
```python
# LIBERO/libero/libero/vla/interface.py
from typing import Protocol
import numpy as np
from PIL import Image

class VLABackend(Protocol):
    def predict(self, images: dict[str, Image.Image], language: str) -> np.ndarray:
        """Returns an action chunk of shape (T, 7) or a single (7,) action.
        Each backend normalizes internally per D-02 — no shared normalization layer.
        """
        ...
```
```python
# LIBERO/libero/libero/vla/oft_backend.py
# Source: Phase 1 confirmed pattern (01-DEBUG-HISTORY.md #7) + moojink/openvla-oft README
class OFTBackend:
    def __init__(self, checkpoint="moojink/openvla-7b-oft-finetuned-libero-spatial"):
        # ... from_pretrained(...) then overlay norm_stats from
        # dataset_statistics.json using key "libero_spatial_no_noops" (Phase 1 confirmed)
        ...

    def predict(self, images: dict, language: str) -> np.ndarray:
        actions, _hidden = self.model.predict_action(images["eye_in_hand"], language, ...)
        return actions  # shape (8, 7) float64 — full chunk, consumed open-loop (D-03)
```
```python
# LIBERO/libero/libero/vla/pi0_backend.py
# Source: Physical-Intelligence/openpi docs/remote_inference.md
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
            "observation/state": self._robot_state(),  # openpi normalizes internally (D-02)
            "prompt": language,
        }
        result = self.client.infer(obs)
        return result["actions"]  # shape (action_horizon, 7)
```

### Pattern 2: Per-Step Success Polling with Early Stop (D-12)
**What:** Poll `env.check_success()` (or read `done` from `env.step()`, which already wraps `_check_success()` internally per `env_wrapper.py:103-104`) every simulation step; break the loop the instant it's true.
**When to use:** Every episode in both notebooks.
**Example:**
```python
# Source: LIBERO/libero/lifelong/metric.py::evaluate_one_task_success (existing pattern)
# and LIBERO/libero/libero/envs/env_wrapper.py::ControlEnv.check_success
max_steps = 600  # this project's LIBERO configs/eval/default.yaml value (D-13)
done = False
for step in range(max_steps):
    action = backend.predict(images, language)  # returns chunk or single action
    for a in action_chunk_to_steps(action):      # D-03: full open-loop replay of chunk
        obs, reward, done, info = env.step(a)
        video_writer.append_obs(obs, done)
        if done:  # done already reflects _check_success() internally
            break
    if done:
        break
print(f"Episode {ep_idx}: {'PASS' if done else 'FAIL'} (steps={step+1})")
```

### Anti-Patterns to Avoid
- **Re-implementing `check_success()` logic from scratch:** LIBERO's `ControlEnv.check_success()` / `_check_success()` already exists and is what `env.step()`'s `done` return reflects. Do not write a new BDDL-goal-checking function — call the existing one.
- **Installing openpi into the same kernel/venv as OFT:** Confirmed real conflict (torch 2.2.0 vs 2.7.1, transformers 4.40.1 vs 4.53.2). Any attempt to reconcile these in one environment will repeat Phase 1's numpy/protobuf whack-a-mole class of failure.
- **Treating `num_open_loop_steps` as a tunable to optimize in this phase:** It's an MVP smoke/eval phase — use the reference default (full chunk), don't spend budget tuning it.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| BDDL task success detection | A custom goal-state checker | `env._check_success()` via `ControlEnv.check_success()` / the `done` flag from `env.step()` | Already implemented, BDDL-aware, exactly what LIBERO's own `evaluate_one_task_success` uses |
| Per-episode video saving | A custom frame-buffer-to-mp4 writer | `LIBERO/libero/libero/utils/video_utils.py::VideoWriter` (`single_video=True` mode) | Already handles the done-state overlay/blank-frame padding and directory creation; exactly matches D-11's per-episode requirement |
| Remote/cross-process model serving | A custom subprocess+pipe or multiprocessing.Queue bridge for openpi | openpi's own `serve_policy.py` + `openpi_client.WebsocketClientPolicy` | This is openpi's documented, maintained solution for exactly this "keep environments separate" problem — reinventing it adds an unmaintained custom protocol for no benefit |
| OFT norm-stats handling | Custom LIBERO-specific unnormalization math | The confirmed Phase 1 pattern: overlay `dataset_statistics.json` (`hf_hub_download`) after `from_pretrained`, key `"libero_spatial_no_noops"` | Already solved and confirmed working in Phase 1; VLA-01 depends on replicating it exactly |

**Key insight:** This phase is almost entirely glue/wiring around existing, already-solved pieces (LIBERO's own success-checking and video-writing, Phase 1's proven OFT loading pattern, openpi's own remote-inference pattern) — the only genuinely new work is the `predict()` interface abstraction and the eval-loop orchestration around it.

## Common Pitfalls

### Pitfall 1: Assuming a single Colab notebook can host both VLA backends
**What goes wrong:** Installing openpi's `torch==2.7.1`/`transformers==4.53.2` into the same kernel as OFT's `torch==2.2.0`/`transformers==4.40.1` silently breaks one or both (transformers caches backend availability at first import per Phase 1 Invariant #2; torch ABI mismatches can corrupt CUDA state).
**Why it happens:** It looks like "just pip install more things" until the version pins collide.
**How to avoid:** Two separate kernels/notebooks (or at minimum two separate `uv`/venv environments) from the start, connected only via the websocket protocol openpi already documents.
**Warning signs:** Any `ImportError`, `AttributeError` on `transformers` classes, or CUDA context errors appearing only after the second backend's install cell runs.

### Pitfall 2: Consuming only `actions[0]` instead of the full 8-step chunk
**What goes wrong:** If the eval loop calls `predict_action` again every single sim step (closed-loop treating the chunk as if it were length 1), inference cost multiplies ~8x for no accuracy benefit in this MVP context, and may blow the Colab session time budget across 3 tasks × 5-10 episodes.
**Why it happens:** It's the more "obviously correct" naive loop structure if you don't know `num_open_loop_steps` exists.
**How to avoid:** Explicitly replay all 8 returned actions before calling `predict_action` again (D-03, confirmed as the OFT reference default).
**Warning signs:** Episode wall-clock time far exceeds expectations; profiling shows `predict_action` called far more often than `max_steps / 8`.

### Pitfall 3: Using LIBERO's multi-process vectorized eval (`SubprocVectorEnv`/`n_eval=20`) unmodified
**What goes wrong:** `LIBERO/libero/lifelong/metric.py`'s reference pattern defaults to `cfg.eval.n_eval=20` and multiprocess vector envs — this phase's D-09 explicitly wants 5-10 episodes per task (not 20), and Colab's single-GPU/limited-CPU environment may not support the same `SubprocVectorEnv` parallelism smoothly.
**Why it happens:** Copy-pasting `evaluate_one_task_success` wholesale pulls in config values and parallelism assumptions tuned for a different (offline cluster) evaluation context.
**How to avoid:** Reuse the *pattern* (per-step `done` polling, `VideoWriter`, `check_success`) but write a simpler single-process loop with an explicit `n_eval=5..10` per D-09, not the full `metric.py` machinery.
**Warning signs:** Multiprocessing/subprocess errors specific to Colab's environment; success-rate numbers that don't match the intended 3×5-10 episode count.

### Pitfall 4: Forgetting camera image key naming consistency in the `images` dict
**What goes wrong:** D-01 requires `images: dict[str, Image]`, but LIBERO's raw obs dict uses keys like `"robot0_eye_in_hand_image"` (from `ControlEnv`'s `camera_names=["agentview","robot0_eye_in_hand"]` default) — if the shared interface's dict keys don't match what Phase 2's tuned camera actually produces, both backends silently receive wrong/black images.
**Why it happens:** LIBERO's raw camera obs keys are verbose and easy to typo or assume incorrectly.
**How to avoid:** Confirm the exact obs key from Phase 2's verification evidence (`robot0_eye_in_hand_image`, flipped `[::-1]` per Phase 2 rendering pattern) before wiring the `images` dict construction; write a one-line assertion/print of `obs.keys()` early in Notebook A.
**Warning signs:** VLA predicts garbage/repeats-same-action; rendered debug frame from the dict is black or upside-down.

### Pitfall 5: openpi's checkpoint auto-download exhausting Colab session time/quota
**What goes wrong:** `pi0_fast_base`/`pi0_base` checkpoints download from `gs://openpi-assets` on first run — if Colab's session or network hiccups mid-download, the smoke test silently hangs or partially downloads a corrupt checkpoint.
**Why it happens:** GCS bucket downloads over Colab's shared network can be slower/less reliable than HuggingFace Hub downloads (which Phase 1's OFT flow already uses and has proven reliable).
**How to avoid:** Set `OPENPI_DATA_HOME` to a Drive-backed path (mirroring Phase 1's Drive-based HF token pattern) so a successful download persists across Colab session restarts; add an explicit timeout/retry or at least a visible progress print in Notebook B.
**Warning signs:** Notebook B hangs with no output during first `serve_policy.py` invocation; subsequent runs redownload the full checkpoint every time (indicates the cache path isn't persisting).

**Resolved (2026-07-26):** This exact pitfall materialized on a real Colab run: the Drive-backed `OPENPI_DATA_HOME` path recommended above in "How to avoid" failed with `gsutil`'s composite-object transfer error (`CommandException: 6 files/objects could not be transferred`) while downloading `pi05_libero`'s 11.6 GiB, 16-sharded checkpoint. Resolution: `OPENPI_DATA_HOME` now points to local Colab disk (`/content/openpi_data`), plus a `crcmod` C-extension reinstall cell per gsutil's own recommendation. See `03-03-PLAN.md`'s T-3-08 amendment for the full threat-model-level rationale and accepted trade-off (loss of restart-persistence).

**Correction (2026-07-26):** The "Resolved" note above turned out to be incomplete — the local-disk switch alone did not fix the download. The actual mechanism is `gsutil`'s bundled Cloud SDK Python being isolated from the Colab kernel's site-packages (`GoogleCloudPlatform/gsutil#1429`), fixed via `CLOUDSDK_PYTHON_SITEPACKAGES=1` plus a `check_hashes=if_fast_else_skip` boto fallback. See `03-03-PLAN.md`'s T-3-08 second amendment for the full threat-model-level rationale.

## Code Examples

### LIBERO's existing per-episode video-saving pattern
```python
# Source: LIBERO/libero/libero/utils/video_utils.py (existing in this repo)
from LIBERO.libero.libero.utils.video_utils import VideoWriter

with VideoWriter(video_path=f"outputs/videos/task_{task_id}_ep_{ep_idx}",
                 save_video=True, fps=30, single_video=True) as vw:
    for step in range(max_steps):
        ...
        vw.append_obs(obs, done, idx=0, camera_name="agentview_image")
        if done:
            break
# __exit__ calls .save() automatically -> outputs/videos/task_X_ep_Y/video.mp4
```

### LIBERO's existing success-polling pattern (this phase's D-12 must replicate)
```python
# Source: LIBERO/libero/lifelong/metric.py::evaluate_one_task_success (existing in this repo)
dones = [False] * env_num
steps = 0
while steps < cfg.eval.max_steps:
    steps += 1
    actions = algo.policy.get_action(data)   # <- this phase replaces with backend.predict(...)
    obs, reward, done, info = env.step(actions)
    for k in range(env_num):
        dones[k] = dones[k] or done[k]
    if all(dones):
        break
```

### openpi remote inference client call
```python
# Source: github.com/Physical-Intelligence/openpi/blob/main/docs/remote_inference.md
from openpi_client import websocket_client_policy as _websocket_client_policy

policy_client = _websocket_client_policy.WebsocketClientPolicy(host="0.0.0.0", port=8000)
action_chunk = policy_client.infer(example_obs)["actions"]
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|---------------|--------|
| Naive closed-loop VLA (re-infer every step) | Action chunking with open-loop replay (`num_open_loop_steps`) | OpenVLA-OFT paper (2025) | ~8x fewer forward passes per episode; this project's Phase 1 already confirmed the (8,7) chunk shape, so this phase should use it |
| Running VLA inference in-process alongside the robot/sim loop | Client-server (websocket) separation for cross-framework VLA serving | openpi's `remote_inference.md` design | Directly solves this phase's OFT/openpi dependency conflict without custom subprocess plumbing |

**Deprecated/outdated:**
- Single-action (non-chunked) VLA output heads are largely superseded by chunked-action heads (OFT, π0-FAST) for inference throughput — not relevant to build custom here since both backends already chunk natively.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | openpi should be installed by tracking `main` branch (no pinned release tag found in research) | Standard Stack | If openpi's `main` branch has since changed its LIBERO config names (`pi0_fast_libero`/`pi0_libero`) or its `pyproject.toml` pins, install could fail differently than researched here — planner should pin a specific commit SHA once Notebook B's install cell is first run successfully, and record it (mirrors Phase 1's `--no-deps` "document the full set next to the install" lesson) |
| A2 | `pi0_fast_libero`'s checkpoint fits comfortably in Colab T4/A100 VRAM for inference (est. ~6.6GB bf16 for the 3.3B-param base model) | Summary / Standard Stack | One GitHub issue reported 37GB usage for a *fine-tuned deployment* server scenario, which likely includes non-inference overhead (JAX compilation caches, training-adjacent buffers); if the smoke-test server genuinely needs >16GB, T4 may not be viable and Notebook B would need an A100, same as OFT |
| A3 | The 3 frozen `libero_spatial` BDDL tasks' `robot0_eye_in_hand_image` obs key is the correct/only key needed for D-01's `images` dict in this phase | Common Pitfalls #4 | If Phase 2's actual obs key differs from this name, the `images` dict construction breaks silently (black/wrong image) rather than erroring — planner should have the first Notebook A cell print `obs.keys()` before building the interface |

**If this table is empty:** N/A — see entries above; all three should be spot-checked once Colab access is available, per this project's established "verify claims against artifacts, not docs" meta-lesson (01-DEBUG-HISTORY.md §3.5).

**A1 materialized (2026-07-26):** This exact anticipated risk occurred on the first real Colab run of Notebook B — the originally-researched config name (`pi0_fast_libero`) is not a valid `serve_policy.py --env` value and has no published checkpoint on openpi's current `main`. Resolution: serve via plain `--env=LIBERO`, which resolves to `pi05_libero`. See `03-CONTEXT.md`'s D-07 amendment for the full rationale. `libero/notebooks/03b-pi0-inference-smoketest.ipynb` was updated accordingly.

## Open Questions

1. **Exact commit/tag of `openpi` to pin for reproducibility**
   - What we know: `Physical-Intelligence/openpi` is under active development; `pyproject.toml` values researched here (`torch==2.7.1`, `transformers==4.53.2`, `jax==0.5.3`) are current as of this research date.
   - What's unclear: Whether these pins will still be current on Colab by the time this phase executes, given openpi's active-development cadence.
   - Recommendation: Planner should have the install task record the actual resolved versions post-install (mirrors Phase 1's Step 5b documentation discipline) rather than assume the versions researched here are exactly what lands.

2. **Whether Colab free-tier T4 is sufficient for the π0 smoke test, or whether it needs the same A100 tier as OFT**
   - What we know: Base model is ~3.3B params (~6.6GB in bf16); D-10 only requires 1 task × 1-2 episodes (low compute budget).
   - What's unclear: openpi's actual peak VRAM during a real inference call (JAX compilation/tracing overhead can spike memory beyond static weight size); one anecdotal GitHub issue reported far higher (37GB) usage in an unspecified deployment context.
   - Recommendation: Attempt Notebook B on the same A100 tier already used for Notebook A (Phase 1 D-05) to avoid a second unknown variable during first implementation; downgrade to T4 later only if this phase needs a lower-cost path in a future iteration.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Colab A100/T4 GPU runtime | Both notebooks (VLA inference is GPU-bound) | N/A (Colab-only; not present in this local dev environment) | — | None — this phase's actual execution and iteration happens on Colab, consistent with Phase 1/2 precedent (local machine has no GPU/torch installed) |
| `torch` (local dev machine) | N/A locally — verification only | ✗ (not installed locally; `ModuleNotFoundError`) | — | Expected — this project's established pattern is Colab-only for GPU-dependent work; no local fallback needed |
| `uv` (Colab, for openpi install) | Notebook B openpi install | Unconfirmed on stock Colab image | — | Install via `pip install uv` or `curl -LsSf https://astral.sh/uv/install.sh \| sh` as Notebook B's first cell if not preinstalled |
| `gs://openpi-assets` GCS bucket access | Checkpoint auto-download for π0/π0-FAST | Unconfirmed — no auth requirement documented, but Colab network reliability to GCS not tested in this project | — | If GCS download is unreliable, `OPENPI_DATA_HOME` can point to a pre-downloaded Drive copy (mirrors Phase 1's Drive-based HF token bootstrap pattern) |

**Missing dependencies with no fallback:**
- None — GPU access is inherent to this phase's scope (VLA inference) and Colab is the established, accepted platform per project constraints.

**Missing dependencies with fallback:**
- `uv` on Colab — install inline if not present (low risk, standard pip-installable tool).
- GCS checkpoint download reliability — Drive-cached fallback available if needed.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | None (project has no pytest/unittest suite) — verification is PASS/FAIL notebook cells, per Phase 1/2 established convention |
| Config file | none — see Wave 0 |
| Quick run command | Run the relevant notebook cell manually on Colab (no CLI test runner in this project) |
| Full suite command | Run both Notebook A and Notebook B end-to-end on Colab |
| A note on the actual approach | This project's PASS/FAIL-cell convention (established Phase 1, continued Phase 2) is the de facto validation architecture. This phase's "tests" are the printed PASS/FAIL-per-episode and the aggregated summary table (D-14), which the human verifier reads on Colab, not automated pytest assertions. |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|--------------------|-------------|
| VLA-01 | OFT produces a valid 7-D action from (image, language) | manual (Colab cell) + printed shape assertion | `assert actions.shape == (8, 7)` inline in Notebook A | ❌ Wave 0 — new notebook cell |
| VLA-02 | Video file saved per episode | manual (Colab cell) + file-existence check | `assert os.path.exists(video_path)` inline in Notebook A/B | ❌ Wave 0 — new notebook cell |
| VLA-03 | Success rate measured via `check_success()` | manual (Colab cell) + printed PASS/FAIL + summary table | Inline print statements in the eval loop (D-14) | ❌ Wave 0 — new notebook cell |
| VLA-04 | π0 backend swappable via same interface, no downstream code change | manual (Colab cell) + smoke-test run | Notebook B end-to-end run against the same `predict()`-calling eval-loop code as Notebook A | ❌ Wave 0 — new notebook + shared interface module |

### Sampling Rate
- **Per task commit:** Manually re-run the affected notebook cell(s) on Colab after each code change (no fast local test loop exists — GPU is Colab-only).
- **Per wave merge:** Re-run the full notebook (Block A install → restart → Block B verification) end-to-end, consistent with Phase 1/2's "restart ≠ clean slate" lesson — verify claims against a genuinely fresh runtime, not a persisted one.
- **Phase gate:** Both notebooks green (all PASS cells) plus the aggregated success-rate summary table produced, before `/gsd-verify-work`.

### Wave 0 Gaps
- [ ] `LIBERO/libero/libero/vla/interface.py` — the shared `predict()` interface module (D-04) does not exist yet.
- [ ] `LIBERO/libero/libero/vla/oft_backend.py` — OFT backend wrapper implementing the interface (net-new; Phase 1 has the loading pattern but not this wrapper).
- [ ] `LIBERO/libero/libero/vla/pi0_backend.py` — π0/openpi backend wrapper implementing the interface (entirely new, no prior art in this project).
- [ ] `LIBERO/libero/libero/vla/eval_loop.py` — shared eval-loop helper (per-step success polling, chunk replay, video writing) reusable by both notebooks.
- [ ] `LIBERO/notebooks/03a-oft-inference-eval.ipynb` — new notebook.
- [ ] `LIBERO/notebooks/03b-pi0-inference-smoketest.ipynb` — new notebook.

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-------------------|
| V2 Authentication | No | This phase has no user-facing auth; Colab session auth is out of scope |
| V3 Session Management | No | N/A — single-user research notebook, no session/token management beyond Phase 1's existing HF-token-via-Drive pattern (unchanged) |
| V4 Access Control | No | N/A — no multi-user access model |
| V5 Input Validation | Marginal | Language prompt is a fixed set of 3 known task strings (from frozen BDDL files), not free user input in this phase — no injection surface. If a free-text prompt box is added later (out of scope here), validate against the benchmark's known task vocabulary before passing to either backend. |
| V6 Cryptography | No | No cryptographic operations introduced by this phase |

### Known Threat Patterns for this stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|----------------------|
| Websocket server (`serve_policy.py`) bound to `0.0.0.0`/all interfaces on a shared Colab VM | Information Disclosure / Tampering (unauthenticated local network exposure) | Bind to `localhost`/`127.0.0.1` only (both client and server run in the same Colab VM in this design) rather than `0.0.0.0`; do not expose the port publicly. This is a research-notebook context, not a deployed service, but the habit matters. |
| Committing cell outputs containing checkpoint download URLs or any inadvertent credentials | Information Disclosure | Same discipline as Phase 1's HF-token lesson (01-DEBUG-HISTORY.md #7): never let a literal secret/token appear in a committed cell output; `gs://openpi-assets` is a public bucket requiring no credentials, so this specific risk is low for the π0 checkpoint download itself, but keep the general habit. |

## Sources

### Primary (HIGH confidence)
- This repository's own code: `LIBERO/libero/lifelong/metric.py`, `LIBERO/libero/libero/envs/env_wrapper.py`, `LIBERO/libero/libero/utils/video_utils.py`, `LIBERO/libero/configs/eval/default.yaml`, `explorations/soarm_sanity.py` (TASKS constant, robot name `Soarm101`), `LIBERO/libero/libero/envs/robots/__init__.py`.
- `.planning/phases/01-colab-environment-setup/01-DEBUG-HISTORY.md` and `01-CONTEXT.md` — confirmed OFT chunk shape `(8,7)`, norm_stats overlay key `libero_spatial_no_noops`, environment contract invariants.
- `.planning/phases/02-soarm-robot-integration/02-CONTEXT.md` and `02-VERIFICATION.md` — confirmed 3 frozen `libero_spatial` task names, `Soarm101` robot registration, 7-DoF action contract.

### Secondary (MEDIUM confidence)
- `github.com/moojink/openvla-oft/blob/main/README.md` (WebFetch) — `num_open_loop_steps`/`NUM_ACTIONS_CHUNK` pattern, `run_libero_eval.py` script location, `moojink/openvla-7b-oft-finetuned-libero-spatial` checkpoint name.
- `github.com/Physical-Intelligence/openpi` README + `pyproject.toml` (WebFetch) — install instructions (`uv sync`), dependency pins (`torch==2.7.1`, `transformers==4.53.2`, `jax[cuda12]==0.5.3`).
- `github.com/Physical-Intelligence/openpi/blob/main/docs/remote_inference.md` (WebFetch) — `serve_policy.py --env=LIBERO`, `WebsocketClientPolicy`, observation dict structure, `image_tools` helpers.
- WebSearch: `pi0_fast_libero`/`pi0_libero` config names and `gs://openpi-assets` checkpoint paths (config.py raw source, HuggingFace Hub community checkpoint mirrors).
- WebSearch: pi0 parameter count (3.3B: PaliGemma 3B backbone + ~300M action expert) via HuggingFace blog and AI Wiki.
- WebSearch: Colab default Python version (3.12) confirming no Python-version conflict with openpi's `>=3.11` requirement.
- `pip index versions` (Bash, this session) — confirmed `openpi-client==0.1.2`, `jax`, `flax`, `orbax-checkpoint` all exist on PyPI with established version histories.

### Tertiary (LOW confidence)
- GitHub issue #599 (37GB VRAM anecdote) — single unverified user report, likely conflates deployment/training overhead with pure inference footprint; treated as a caveat (Open Question #2), not a hard fact.
- No official openpi-on-Colab reference notebook or blog post was found in this research session — the two-notebook design here is this project's own synthesis of openpi's documented remote-inference pattern applied to the Colab constraint, not a directly-cited existing recipe.

## Metadata

**Confidence breakdown:**
- Standard stack: MEDIUM-HIGH — OFT facts are HIGH (already proven in this repo's Phase 1); openpi facts are MEDIUM (official GitHub sources, not yet run in this project's Colab environment)
- Architecture: MEDIUM-HIGH — the two-kernel split is directly justified by a confirmed real version conflict, not a speculative one; the specific notebook boundary is this project's own synthesis
- Pitfalls: HIGH for OFT/LIBERO-specific pitfalls (grounded in this repo's own code and Phase 1's debug history); MEDIUM for openpi-specific pitfalls (grounded in official docs/issues, not this project's direct experience yet)

**Research date:** 2026-07-18
**Valid until:** 2026-08-01 (14 days) — openpi is under active development; re-verify `pyproject.toml` pins and config names (`pi0_fast_libero`) immediately before Notebook B's first real Colab run, per this project's "verify claims against artifacts, not docs" meta-lesson.
