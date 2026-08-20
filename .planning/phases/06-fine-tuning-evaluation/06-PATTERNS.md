# Phase 6: Fine-Tuning & Evaluation - Pattern Map

**Mapped:** 2026-08-20
**Files analyzed:** 9 (new/modified, per RESEARCH.md Recommended Project Structure)
**Analogs found:** 9 / 9

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|--------------------|------|-----------|-----------------|----------------|
| `LIBERO/libero/libero/datasets/rlds_converter.py` | utility (data transform) | batch / file-I/O | `LIBERO/libero/libero/datasets/normalization.py` | exact (standalone script -> data artifact, same package) |
| `LIBERO/libero/libero/datasets/test_rlds_converter.py` | test | batch / file-I/O | `LIBERO/libero/libero/datasets/test_normalization.py` | exact (synthetic-array unit test + skip-if-missing real-dataset integration test, same package) |
| `LIBERO/libero/libero/datasets/oxe_register.py` | config / utility | event-driven (import-time monkeypatch) | `LIBERO/libero/libero/vla/oft_backend.py::_ensure_prismatic` | role-match (guarded import/registration pattern against an external installed package) |
| `LIBERO/libero/libero/vla/eval_loop.py` (MODIFY: add `seed` param) | service (eval driver) | request-response / batch | itself (existing file, additive change) | exact |
| `LIBERO/libero/libero/vla/test_eval_loop.py` (MODIFY: add seed-determinism test) | test | request-response | itself (existing file, additive change) + `test_normalization.py` (assertion style) | exact |
| `LIBERO/libero/libero/vla/adapter_backend.py` (or extend `oft_backend.py`) | service (VLA backend) | request-response | `LIBERO/libero/libero/vla/oft_backend.py::OFTBackend` | exact (subclass extends the exact same class) |
| `LIBERO/notebooks/06a-finetune.ipynb` | config / orchestration (notebook) | batch / event-driven | `LIBERO/notebooks/01-colab-env-setup.ipynb` | role-match (Colab bootstrap + pip install + guarded imports pattern) |
| `LIBERO/notebooks/06b-eval.ipynb` | config / orchestration (notebook) | request-response / batch | Phase 3's eval notebook (per CONTEXT.md two-kernel-group precedent) + `eval_loop.py::run_suite` as the driven logic | role-match |
| HF Hub push / WandB wrapper helper (used inside 06a notebook, per Pattern 3) | utility (event-driven upload) | event-driven / file-I/O | `LIBERO/libero/libero/vla/oft_backend.py` (`hf_hub_download` usage, read-side) | partial (same library, opposite direction — read pattern analog only, no push analog exists yet) |

## Pattern Assignments

### `LIBERO/libero/libero/datasets/rlds_converter.py` (utility, batch/file-I/O)

**Analog:** `LIBERO/libero/libero/datasets/normalization.py` (156 lines, read in full)

**Module docstring / import pattern** (lines 1-24):
```python
"""SOARM-specific action/proprio normalization statistics (DATA-03, Phase 4).
...
"""
from __future__ import annotations

import json
import os

import numpy as np
```
Copy this shape for `rlds_converter.py`: `from __future__ import annotations`, stdlib imports first, keep the module importable without a sim-stack import at top level — confine `h5py`/`tensorflow`/`tensorflow_datasets` imports to inside the function bodies that need them (mirrors `load_actions_and_proprio_from_hdf5`'s local `import h5py`, lines 70, 78).

**HDF5 read pattern** (lines 70-102, `load_actions_and_proprio_from_hdf5`):
```python
def load_actions_and_proprio_from_hdf5(hdf5_paths: list) -> tuple:
    import h5py  # local import: keep this module importable with no sim stack.
    actions_list = []
    proprio_list = []
    total_demo_count = 0
    for hdf5_path in hdf5_paths:
        with h5py.File(hdf5_path, "r") as f:
            grp = f["data"]
            demo_names = [k for k in grp.keys() if k.startswith("demo_")]
            total_demo_count += len(demo_names)
            for demo_name in demo_names:
                demo = grp[demo_name]
                actions_list.append(np.asarray(demo["actions"]))
                obs = demo["obs"]
                proprio_list.append(np.concatenate(
                    [np.asarray(obs["joint_states"]), np.asarray(obs["gripper_states"])], axis=-1))
    actions = np.concatenate(actions_list, axis=0)
    proprio = np.concatenate(proprio_list, axis=0)
    return actions, proprio, total_demo_count
```
`rlds_converter.py` must read the SAME HDF5 schema (`data/demo_N/obs/{agentview_rgb,eye_in_hand_rgb,joint_states,gripper_states}`, `data/demo_N/actions`), but per-episode (not flattened across all demos), since RLDS needs episode boundaries (`is_first`/`is_last`/`is_terminal`) that normalization.py's flattened stats computation does not need. Also reuse `hdf5_writer.py`'s `OBS_KEY_MAPPING` naming convention (lines 49-54) as the canonical schema-key source: `agentview_rgb`, `eye_in_hand_rgb`, `gripper_states`, `joint_states`.

**Fail-loud / write-after-success pattern** (lines 105-126, `write_dataset_statistics`):
```python
def write_dataset_statistics(hdf5_paths, out_json_path, dataset_key="soarm_spatial") -> dict:
    """Per T-04-04-01: the JSON write only happens AFTER stats are successfully
    computed over the full loaded array — a load failure (missing/corrupt
    HDF5) raises before any partial file is written.
    """
    actions, proprio, total_demo_count = load_actions_and_proprio_from_hdf5(hdf5_paths)
    stats = compute_norm_stats(actions, proprio, dataset_key, num_trajectories=total_demo_count)
    out_dir = os.path.dirname(out_json_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_json_path, "w") as f:
        json.dump(stats, f)
    return stats
```
`rlds_converter.py`'s top-level `hdf5_to_rlds(hdf5_paths, out_dir, dataset_name="soarm_spatial") -> int` (per RESEARCH.md Pattern 1) must follow the identical "compute/validate fully in memory, THEN write" ordering — do not write partial `.tfrecord` shards before all schema/shape/dtype validation (V5 ASVS control, RESEARCH.md Security Domain) passes.

**CLI entry point pattern** (lines 129-155, `if __name__ == "__main__":`):
```python
if __name__ == "__main__":
    import argparse
    import glob as glob_module
    parser = argparse.ArgumentParser(description="...")
    parser.add_argument("--hdf5-glob", default="LIBERO/libero/datasets/soarm_spatial/*_demo.hdf5", ...)
    parser.add_argument("--out", default="LIBERO/libero/datasets/soarm_spatial/dataset_statistics.json", ...)
    args = parser.parse_args()
    paths = sorted(glob_module.glob(args.hdf5_glob))
    result = write_dataset_statistics(paths, args.out)
    print(f"wrote stats for ... -> {args.out}")
```
Copy directly for `rlds_converter.py`'s CLI: `--hdf5-glob` default `LIBERO/libero/datasets/soarm_spatial/*_demo.hdf5`, `--out-dir` default `LIBERO/libero/datasets/soarm_spatial/rlds/` (per D-05), and the "print saved path with -> arrow" convention (CLAUDE.md Logging convention).

---

### `LIBERO/libero/libero/datasets/test_rlds_converter.py` (test)

**Analog:** `LIBERO/libero/libero/datasets/test_normalization.py` (143 lines, read in full)

**Import-path bootstrap** (lines 16-28):
```python
import glob
import json
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_LIBERO_LIBERO = os.path.join(_REPO_ROOT, "LIBERO", "libero")
if _LIBERO_LIBERO not in sys.path:
    sys.path.insert(0, _LIBERO_LIBERO)

import numpy as np
import pytest

from libero.datasets.normalization import (compute_norm_stats, load_actions_and_proprio_from_hdf5, write_dataset_statistics)
```
Copy verbatim (same directory depth) for `test_rlds_converter.py`, importing from `libero.datasets.rlds_converter` instead.

**Real-dataset skip-if-missing integration test pattern** (lines 39-45, 121-143):
```python
_REAL_DATASET_GLOB = os.path.join(os.path.dirname(__file__), "..", "..", "datasets", "soarm_spatial", "*_demo.hdf5")

def test_load_and_write_against_04_02_output(tmp_path):
    paths = sorted(glob.glob(_REAL_DATASET_GLOB))
    if not paths:
        pytest.skip("No real soarm_spatial *_demo.hdf5 dataset found on disk")
    out_path = str(tmp_path / "dataset_statistics.json")
    result = write_dataset_statistics(paths, out_path)
    assert os.path.isfile(out_path)
    ...
```
Use this exact `pytest.skip`-if-missing pattern for `test_rlds_converter.py`'s "converted dataset is genuinely `tfds.builder`-loadable" case — RESEARCH.md's Validation Architecture table flags this specific check as Colab-only/manual, but the local skip-if-missing pattern still applies to any schema/shape checks runnable on synthetic HDF5 fixtures without `tensorflow_datasets` installed.

**Synthetic-array fixture pattern** (lines 50-65, `_synthetic_arrays`):
```python
def _synthetic_arrays(rng, t=500, action_dim=7, proprio_dim=7):
    """Realistic-shaped synthetic arrays with distinct per-dim means/scales.
    proprio_dim=7: 5 SOARM arm joints + 2 gripper DOF ...
    """
    means = rng.uniform(-5, 5, size=proprio_dim)
    scales = rng.uniform(0.5, 3.0, size=proprio_dim)
    proprio = rng.normal(loc=means, scale=scales, size=(t, proprio_dim))
    ...
```
`test_rlds_converter.py` should build a small synthetic in-memory HDF5 (via `h5py.File(tmp_path/..., "w")` + `data/demo_0/obs/...`) rather than raw arrays, since the converter's unit is per-episode HDF5 groups, not flat arrays — but keep the "distinct per-dim means/scales, no NaN, degenerate-dimension" assertion style (lines 94-118) for the RLDS feature-schema checks (dtype=uint8 images, dtype=float32 state/action, per Pattern 1's `step_features` schema).

---

### `LIBERO/libero/libero/datasets/oxe_register.py` (config/utility, event-driven)

**Analog:** `LIBERO/libero/libero/vla/oft_backend.py::_prismatic_ok` / `_ensure_prismatic` (lines 32-78)

**Guarded external-package check + fail-loudly pattern**:
```python
def _prismatic_ok() -> bool:
    try:
        importlib.import_module("prismatic.training.train_utils")
        return True
    except Exception:
        return False

def _ensure_prismatic() -> None:
    if _prismatic_ok():
        return
    try:
        ...
    except Exception:
        import traceback
        traceback.print_exc()
        raise RuntimeError("prismatic import failed — see traceback above for the missing dependency")
```
`oxe_register.py` should follow the same shape: a `register_soarm_spatial()` function that (1) imports `OXE_DATASET_CONFIGS`/`OXE_NAMED_MIXTURES`/`StateEncoding`/`ActionEncoding` from the installed `prismatic` package (same import-guard risk as `_ensure_prismatic` — these only exist in the moojink fork), (2) mutates the dicts in-memory per RESEARCH.md Pattern 2, (3) fails loudly (not silently) if `prismatic.vla.datasets.rlds.oxe.configs` isn't importable, printing the traceback before raising — matches this project's "fail loudly, do not swallow" convention (oft_backend.py lines 69-78) and the module docstring convention of stating "does not run any loading/network/GPU logic at import time" (oft_backend.py lines 9-13) — `oxe_register.py`'s mutation should happen inside an explicit `register_soarm_spatial()` call from the training notebook, not at module import time, so it's testable (assert dict keys) without needing prismatic installed for an early smoke-test import guard.

---

### `LIBERO/libero/libero/vla/eval_loop.py` (MODIFY — add seed plumbing, D-08)

**Analog:** itself (existing 163-line file, read in full) — additive change only

**Current `run_episode` signature and reset call** (lines 20-28, 62):
```python
def run_episode(
    env, backend, language: str, video_path: str,
    max_steps: int = MAX_STEPS_DEFAULT,
    camera_name: str = "robot0_eye_in_hand_image",
    agentview_camera_name: str = "agentview_image",
) -> dict:
    ...
    obs = env.reset()
```
Add a `seed: int | None = None` kwarg; before `env.reset()` insert:
```python
    if seed is not None:
        env.seed(seed)
    obs = env.reset()
```
**`ControlEnv.seed` delegation** (`LIBERO/libero/libero/envs/env_wrapper.py` lines 133-134, VERIFIED this session):
```python
def seed(self, seed):
    self.env.seed(seed)
```
Confirms `env.seed(seed)` is already a valid call on every env this project constructs via `OffScreenRenderEnv`/`ControlEnv` — no new env method needed.

**Current `run_suite` signature and per-episode loop** (lines 95-141):
```python
def run_suite(
    env_factory, backend, tasks: list, language_map: dict,
    episodes_per_task: int, video_dir: str, max_steps: int = MAX_STEPS_DEFAULT,
) -> list:
    ...
    for task in tasks:
        env = env_factory(task)
        ...
        for ep_idx in range(episodes_per_task):
            video_path = os.path.join(video_dir, f"{task_slug}_ep{ep_idx}")
            result = run_episode(env, backend, language, video_path, max_steps=max_steps)
```
Add `episode_seeds: list | None = None` kwarg, and inside the `for ep_idx in range(...)` loop:
```python
        for ep_idx in range(episodes_per_task):
            seed = episode_seeds[ep_idx] if episode_seeds is not None else None
            ...
            result = run_episode(env, backend, language, video_path, max_steps=max_steps, seed=seed)
```
Per RESEARCH.md Pitfall 7, `seed` MUST be applied inside the per-episode loop (not once before it) since `env_factory(task)` builds ONE env reused across all `episodes_per_task` episodes.

---

### `LIBERO/libero/libero/vla/test_eval_loop.py` (MODIFY — add seed-determinism test)

**Analog:** itself (existing file) + `MockEnv` fixture pattern (lines 24-40)

**Mock-based no-GPU test pattern** (lines 1-40):
```python
"""Mock-based pytest for eval_loop.py — no GPU/network dependency. ..."""
import numpy as np
from libero.vla.eval_loop import print_episode_result, run_episode, run_suite

class MockEnv:
    """Scripted env: returns done=True on the 3rd env.step() call (mid-chunk)."""
    def __init__(self, done_on_step=3):
        self.done_on_step = done_on_step
        self.step_count = 0
        self._obs = {...}
    def reset(self):
        self.step_count = 0
        return self._obs
```
Add a `seed(self, seed)` method to `MockEnv` that records the last seed it was called with (e.g. `self.last_seed = seed`), then a new test:
```python
def test_run_episode_calls_env_seed_before_reset():
    env = MockEnv()
    ...
    run_episode(env, backend, "do the task", video_path, seed=42)
    assert env.last_seed == 42
```
And, per RESEARCH.md's Pitfall 7 warning sign ("re-running the before benchmark twice with the same seed list produces different summary-table numbers"), add a determinism test asserting `env.seed(...)` is called once per episode inside `run_suite`'s loop with matching values from `episode_seeds`, matching this file's existing per-call assertion style.

---

### `LIBERO/libero/libero/vla/adapter_backend.py` (NEW, or extend `oft_backend.py`)

**Analog:** `LIBERO/libero/libero/vla/oft_backend.py::OFTBackend` (full class, lines 81-193)

**Class structure / `__init__` loading pattern** (lines 84-129):
```python
class OFTBackend:
    def __init__(self, checkpoint: str = CHECKPOINT, device: str = "cuda"):
        _ensure_prismatic()
        self.processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            checkpoint, trust_remote_code=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        ).to(device)
        self.device = device
        self.model.vision_backbone.set_num_images_in_input(2)
        stats_path = hf_hub_download(checkpoint, "dataset_statistics.json")
        with open(stats_path) as f:
            self.model.norm_stats = json.load(f)
        self.unnorm_key = "libero_spatial"
        if self.unnorm_key not in self.model.norm_stats:
            ...
```
`FinetunedOFTBackend` subclasses `OFTBackend` exactly per RESEARCH.md Pattern 4 — call `super().__init__(checkpoint=checkpoint, device=device)` first (reuses the entire prismatic-guard + bf16-load + norm_stats overlay unchanged), THEN apply the PEFT merge:
```python
class FinetunedOFTBackend(OFTBackend):
    def __init__(self, adapter_repo_id: str, checkpoint: str = CHECKPOINT, device: str = "cuda"):
        super().__init__(checkpoint=checkpoint, device=device)
        adapter_dir = snapshot_download(adapter_repo_id)
        self.model = PeftModel.from_pretrained(self.model, adapter_dir).merge_and_unload()
        self.model.vision_backbone.set_num_images_in_input(2)  # re-apply post-merge (verify on Colab, A4)
```
Per Pitfall 6, the norm_stats overlay must use the CHECKPOINT'S OWN `dataset_statistics.json` (pushed alongside the adapter to HF Hub, D-02/D-03), not Phase 4's file — override `self.unnorm_key`/`self.model.norm_stats` resolution using the adapter repo's own stats file, following the same `hf_hub_download(...)` + `json.load` shape as lines 114-117, but pointed at `adapter_repo_id` instead of `checkpoint`.

**`predict()` — unchanged, do not override.** `run_suite`/`run_episode` call `backend.predict(images, language)` polymorphically; `FinetunedOFTBackend` inherits `OFTBackend.predict` (lines 131-192) verbatim — no new code needed here, this is the entire point of Pattern 4 (RESEARCH.md).

---

## Shared Patterns

### Fail-loud data validation (no silent padding/truncation)
**Source:** `LIBERO/libero/libero/datasets/normalization.py::write_dataset_statistics` (docstring, lines 110-115) and `LIBERO/libero/libero/datasets/hdf5_writer.py` (module docstring, lines 12-18, "Unsuccessful episodes are silently dropped ... never written")
**Apply to:** `rlds_converter.py` — validate HDF5 schema (shape/dtype/key presence) BEFORE writing any `.tfrecord` shard; raise on mismatch rather than padding/truncating (ASVS V5, RESEARCH.md Security Domain).

### "Standalone script -> committed data artifact" pattern (D-05)
**Source:** `LIBERO/libero/libero/datasets/normalization.py` (whole-file structure: pure functions + `if __name__ == "__main__":` CLI with `argparse`, `--hdf5-glob`/`--out` flags, "Saved -> path" print)
**Apply to:** `rlds_converter.py` — same CLI shape, run once, output committed to Colab disk under `LIBERO/libero/datasets/soarm_spatial/rlds/`.

### Guarded external-package import + fail loudly, not silently
**Source:** `LIBERO/libero/libero/vla/oft_backend.py::_prismatic_ok`/`_ensure_prismatic` (lines 32-78)
**Apply to:** `oxe_register.py`'s prismatic import guard; `adapter_backend.py`'s `PeftModel`/`snapshot_download` imports should similarly not crash at module import time, only inside the constructor.

### `env.seed(seed)` delegation already exists — no new env API needed
**Source:** `LIBERO/libero/libero/envs/env_wrapper.py::ControlEnv.seed` (lines 133-134)
**Apply to:** `eval_loop.py`'s new seed-plumbing — call it per-episode, inside the loop (Pitfall 7), not once before it.

### `hf_hub_download` + `json.load` for stats/config overlay (read-side); `create_repo(..., private=True)` + `upload_folder` for the new write-side (D-01/D-02)
**Source (read-side):** `LIBERO/libero/libero/vla/oft_backend.py` lines 114-117 — `hf_hub_download(checkpoint, "dataset_statistics.json")` then `json.load`.
**Apply to:** `adapter_backend.py`'s adapter-specific stats loading (read-side, direct copy); the training notebook's periodic-checkpoint push (write-side) has no existing in-repo analog — RESEARCH.md Pattern 3 provides the only reference (`huggingface_hub.create_repo(repo_id, private=True, exist_ok=True)` + `upload_folder(..., run_as_future=True)`), and ASVS Security Domain explicitly requires verifying `private=True` is present in code/tests, not just assumed.

### Test file import-path bootstrap (repo-root + `LIBERO/libero` sys.path insertion)
**Source:** `LIBERO/libero/libero/datasets/test_normalization.py` lines 16-28 (also present in `test_hdf5_writer.py`, `test_eval_loop.py`'s docstring notes an equivalent rootdir-walking mechanism)
**Apply to:** `test_rlds_converter.py` — copy verbatim, same `../../../..` depth (file lives at the same `LIBERO/libero/libero/datasets/` depth).

### Print convention: arrow notation for saved paths
**Source:** CLAUDE.md Logging convention + `normalization.py` line 152-155 (`print(f"wrote stats for ... -> {args.out}")`)
**Apply to:** `rlds_converter.py`'s CLI entry point and any notebook cells reporting HF Hub push destinations.

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| HF Hub periodic checkpoint-push watcher/callback (training notebook helper, D-02) | utility | event-driven (poll `run_root_dir` for new checkpoint dirs) | No existing code in this repo pushes TO HF Hub (all prior HF Hub usage is read-only `hf_hub_download`/`snapshot_download`); planner should follow RESEARCH.md Pattern 3's cited `upload_folder(..., run_as_future=True)` snippet directly, with ASVS `private=True` verification added as a test/assertion. |
| WandB eval-results wrapper (eval notebook, D-13/TUNE-04) | utility | event-driven (`wandb.log` after `run_suite` completes) | No existing code in this repo calls `wandb.log` directly — `LIBERO/libero/lifelong/main.py` is the only WandB touchpoint (a different, unrelated training entrypoint with its own project); no reusable in-repo pattern beyond the `WANDB_API_KEY` env var auth convention (`.planning/codebase/INTEGRATIONS.md`). |
| `oxe_register.py`'s exact `OXE_DATASET_CONFIGS`/`OXE_NAMED_MIXTURES` dict-mutation logic | config | event-driven (monkeypatch) | The dict structure itself is external (`prismatic.vla.datasets.rlds.oxe.configs`, not vendored); only the guard/fail-loudly wrapper pattern has an in-repo analog (`_ensure_prismatic`), not the dict-mutation content — see RESEARCH.md Pattern 2 for the literal snippet to adapt. |

## Metadata

**Analog search scope:** `LIBERO/libero/libero/datasets/`, `LIBERO/libero/libero/vla/`, `LIBERO/libero/libero/envs/`, `LIBERO/notebooks/`
**Files scanned:** `normalization.py`, `test_normalization.py`, `hdf5_writer.py`, `eval_loop.py`, `test_eval_loop.py`, `oft_backend.py`, `env_wrapper.py`, `.planning/codebase/INTEGRATIONS.md`
**Pattern extraction date:** 2026-08-20
</content>
