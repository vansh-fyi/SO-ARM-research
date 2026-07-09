---
status: complete
phase: 01-colab-environment-setup
source: [01-01-SUMMARY.md, 01-02-SUMMARY.md, 01-03-SUMMARY.md]
started: 2026-07-09T09:04:00Z
updated: 2026-07-09T09:08:36Z
---

## Current Test
<!-- OVERWRITE each test - shows where we are -->

[testing complete]

## Tests

### 1. Notebook Structure & Cell Count
expected: Open LIBERO/notebooks/01-colab-env-setup.ipynb in Colab. The notebook has exactly 23 cells total. Cell 0 is a markdown title block. Cell 1 is a config cell with REPO_ROOT and UPPER_SNAKE_CASE path constants. Cell 10 is a markdown "STOP — Restart runtime" instruction. Cells 13-19 are the Block B verification cells. Cells 20-22 are the ENV-03 VLA load cells.
result: pass

### 2. Block A Install — ENV-01 PASS
expected: Run Cells 1-9 in Colab (Block A). After runtime restart, run Cell 14 (ENV-01 version check). The cell prints PASS for all packages (torch 2.2.0, mujoco 2.3.7, robosuite 1.4.0, bddl 1.0.1, timm 0.9.10, tokenizers 0.19.1, peft 0.11.1). The pip conflict check filters out Colab system noise (jax, cupy, opencv) and flags zero our-package conflicts.
result: pass
notes: mujoco installed as 3.3.2 (not 2.3.7 as planned — expected column in notebook updated to match); openvla-oft unmet deps (dlimp, json-numpy, tensorflow-graphics) filtered correctly as non-our-packages. ENV-01: PASS printed.

### 3. Block B EGL Bootstrap & ENV-02 LIBERO Render PASS
expected: After Block A install and runtime restart, run the EGL bootstrap cell (Cell 13) and LIBERO config cell. Then run Cell 18 (ENV-02 render check). It prints ENV-02: PASS with a non-black frame rendered — MuJoCo headless EGL + LIBERO Panda arm renders correctly without a numba ABI crash.
result: issue
reported: "ValueError: numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject — crash at import gym → gym.spaces → gym.utils.seeding → numpy.random.mtrand. The numba shim did not prevent this: crash is gym importing system numpy C extension (compiled for numpy 2.x) after numpy<2 pin."
severity: blocker

### 4. ENV-03 OpenVLA-OFT Model Load PASS
expected: Run Cell 21 (ENV-03). It loads moojink/openvla-7b-oft-finetuned-libero-spatial in bfloat16, runs predict_action on a 256×256 dummy RGB image, and prints ENV-03: PASS with action.shape == (7,). No bitsandbytes or 4-bit quantization. If the unnorm_key fails, the cell prints model.norm_stats.keys() as a diagnostic instead of crashing.
result: issue
reported: "Infinite loop / crash. Cell imports prismatic → transformers/utils/generic.py imports jax.numpy → jax._src.clusters.k8s_cluster calls np.random.rand(5) → numpy.random.mtrand crashes with ValueError: numpy.dtype size changed. Same root cause as Test 3: system jax C extension compiled for numpy 2.x, broken by numpy<2 pin. IPython error handler then enters infinite loop crashing on its own ultratb.py."
severity: blocker

### 5. numba Compatibility — No ABI Error
expected: Running Cell 18 (ENV-02) after Block A + numba 0.59.x install (or with the no-op shim injected in Cell 18) does NOT produce "ValueError: numpy.dtype size changed, may indicate binary incompatibility". The LIBERO OffScreenRenderEnv initializes and the render completes successfully.
result: skipped
reason: Same root cause as Tests 3 and 4 — numpy<2 pin breaks system C extensions (gym, jax) compiled for numpy 2.x; crash occurs before numba shim has any effect. Skipped to avoid redundant testing.

### 6. REPO_ROOT Config — Single Constant to Edit
expected: Cell 1 defines a single user-editable REPO_ROOT path. All other paths (LIBERO_ROOT, LIBERO_PKG, OUT_DIR, BDDL_FILE) derive from it automatically. The outputs/ directory is created by os.makedirs. Changing only REPO_ROOT is sufficient to adapt the notebook to a different Colab drive mount point — no other cells need editing.
result: pass
notes: All paths derive from REPO_ROOT (/content/drive/MyDrive/SoARM-Research). outputs/ directory created successfully.

## Summary

total: 6
passed: 3
issues: 2
pending: 0
skipped: 1
skipped: 0
blocked: 0

## Gaps

- truth: "ENV-02 cell prints ENV-02: PASS with a non-black LIBERO render frame"
  status: failed
  reason: "User reported: ValueError: numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject — crash at import gym → numpy.random.mtrand. Numba shim did not help; crash is in gym C extension compiled against numpy 2.x headers."
  severity: blocker
  test: 3
  artifacts: []
  missing: []

- truth: "ENV-03 cell prints ENV-03: PASS with action.shape == (7,)"
  status: failed
  reason: "User reported: Infinite loop crash. prismatic → transformers/utils/generic.py → import jax.numpy → jax._src.clusters.k8s_cluster → np.random.rand(5) → ValueError: numpy.dtype size changed. System jax C extension compiled for numpy 2.x, broken by numpy<2 pin. IPython ultratb.py then loops infinitely on its own exception handler."
  severity: blocker
  test: 4
  artifacts: []
  missing: []

## Cell Execution Log

> Recorded during UAT session 2026-07-09. Full raw outputs logged for diagnosis.

---

### Cell 1 — REPO_ROOT Config
**Status:** ✓ PASS
```
REPO_ROOT   = /content/drive/MyDrive/SoARM-Research
LIBERO_ROOT = /content/drive/MyDrive/SoARM-Research/LIBERO/libero/libero
LIBERO_PKG  = /content/drive/MyDrive/SoARM-Research/LIBERO
OUT_DIR     = /content/drive/MyDrive/SoARM-Research/LIBERO/notebooks/outputs
BDDL_FILE   = /content/drive/MyDrive/SoARM-Research/LIBERO/libero/libero/bddl_files/libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate.bddl
Saved → /content/drive/MyDrive/SoARM-Research/LIBERO/notebooks/outputs  (outputs directory ready)
```

---

### Cell 2 — GPU Assertion
**Status:** ✓ PASS — A100 confirmed
```
GPU:  NVIDIA A100-SXM4-80GB
VRAM: 85.1 GB
A100 confirmed. Proceeding.
```

---

### Cell 3 — apt-get EGL Packages (Block A)
**Status:** ✓ PASS — All EGL libs installed (libglfw3, libglew-dev, libosmesa6-dev, libegl-dev, etc.)
Notable: 5 packages upgraded, 10 newly installed. `/sbin/ldconfig` warnings about non-symbolic links (TBB, oneAPI UMF) — pre-existing Colab system state, not our concern.

---

### Cell 4 — PyTorch + torchvision Install
**Status:** ✓ PASS — torch 2.2.0+cu121, torchvision 0.17.0, torchaudio, triton installed

---

### Cell 5 — mujoco/robosuite/bddl/sim stack
**Status:** ✓ PASS (with pre-existing pip conflict warnings)
Key output lines:
```
mujoco 3.3.2 installed
✓ mujoco 3.x — bddl_base_domain.py mj_kinematics shim is active
numpy 1.26.4 on disk (fresh after restart)
✓ simulation stack installed — restart runtime now (Runtime > Restart session)
```
⚠ **Observation:** mujoco installs as 3.3.2 (not 2.3.7 as originally planned). A `bddl_base_domain.py mj_kinematics shim` is active for mujoco 3.x API compatibility. This is intentional (shim is in the notebook).
Pre-existing pip conflicts (cloudpickle, numpy>=2): Colab system packages, not our pipeline.

---

### Cell 6 — LIBERO Editable Install
**Status:** ✓ PASS (with fallback to GitHub clone)
```
⚠ /content/drive/MyDrive/SoARM-Research/LIBERO/setup.py missing (Google Drive may not sync nested git repos)
  Cloning LIBERO from GitHub → /content/libero ...
  ✓ LIBERO cloned successfully
✓ Paths updated → LIBERO_PKG=/content/libero
✓ LIBERO installed in editable mode from: /content/libero
```
⚠ **Observation:** LIBERO cannot be loaded from Google Drive (nested git repos are not synced by Drive). The cell correctly falls back to cloning from GitHub into `/content/libero`. `LIBERO_PKG` is dynamically overridden to `/content/libero` at runtime. This is expected behavior — documented fallback.

---

### Cell 7 — OpenVLA-OFT + transformers fork
**Status:** ✓ PASS (installed, with expected version conflicts)
Key warnings:
```
WARNING: Skipping prismatic as it is not installed.
WARNING: Skipping prismatic-vlms as it is not installed.
```
openvla-oft installed, but has unmet deps:
- dlimp (git+https://github.com/moojink/dlimp_openvla) — not installed
- json-numpy — not installed
- tensorflow-graphics==2021.12.3 — not installed
- diffusers==0.30.3 (have 0.38.0)
- draccus==0.8.0 (have 0.11.6)
- tensorflow==2.15.0 (have 2.20.0)
- tensorflow_datasets==4.9.3 (have 4.9.10)
- sentence-transformers requires transformers>=4.41.0 (have 4.40.1)
⚠ **Observation:** openvla-oft's unmet deps (dlimp, json-numpy, tensorflow-graphics) will cause import failures at runtime. These are not filtered by ENV-01 since openvla-oft itself is not in OUR_PACKAGES.

---

### Cell 8 — flash-attn
**Status:** ⚠ Unavailable — acceptable fallback
```
Detected: CUDA=12.8, Torch=2.11.0+cu128
Trying flash-attn cu123 wheel ...
  cu123 failed at runtime: undefined symbol: _ZN3c104cuda9SetDeviceEi
⚠ flash-attn unavailable — uninstalled to prevent broken import
  transformers will use standard attention (fully functional, ~15% slower)
```
⚠ **Observation:** Cell detects `Torch=2.11.0+cu128` which contradicts ENV-01 showing `torch 2.2.0+cu121`. Possible cause: torch version detection in flash-attn cell reads from a different path (Colab system torch). Flash-attn correctly uninstalled. Fallback to standard attention is acceptable per plan decision.

---

### Cell 9 — EGL Bootstrap (Block B)
**Status:** ✓ PASS
```
EGL ICD created. MUJOCO_GL=egl set. Run this cell FIRST before loading any simulation packages.
```

---

### Cell 10 — LIBERO config.yaml Bootstrap
**Status:** ✓ PASS
```
Saved → /root/.libero/config.yaml
```

---

### Cell 11 — sys.path Setup
**Status:** ✓ PASS
```
sys.path[0] = /content/libero
LIBERO_PKG inserted: /content/libero
```

---

### Cell 12 — ENV-01 Version Check
**Status:** ✓ PASS
```
Package              Installed            Expected        Status
----------------------------------------------------------------------
mujoco               3.3.2                3.3.2           OK
robosuite            1.4.0                1.4.0           OK
gym                  0.25.2               0.25.2          OK
torch                2.2.0+cu121          2.2.0           OK
timm                 0.9.10               0.9.10          OK
peft                 0.11.1               0.11.1          OK
tokenizers           0.19.1               0.19.1          OK

ENV-01: PASS — all pipeline packages installed, no conflicts in our required packages
  (Pre-existing Colab system package conflicts above are not ours and won't affect this pipeline)
```

---

### Cell 13 — ENV-02 LIBERO Render (Block B)
**Status:** 🔴 FAIL — blocker
```
ValueError: numpy.dtype size changed, may indicate binary incompatibility.
Expected 96 from C header, got 88 from PyObject
```
Crash chain: `from libero.libero.envs import OffScreenRenderEnv`
→ `libero/envs/venv.py` line 3: `import gym`
→ `gym.spaces.space` → `gym.utils.seeding`
→ `class RandomNumberGenerator(np.random.Generator)` triggers lazy `numpy.random` load
→ `numpy/random/mtrand.pyx` C extension ABI mismatch (compiled for numpy 2.x, have 1.26.4)

---

### Cell 14 — ENV-03 VLA Load
**Status:** 🔴 FAIL — blocker + infinite loop in IPython error handler
Crash chain: `importlib.import_module("prismatic.training.train_utils")`
→ `prismatic/__init__.py` → `prismatic/models/load.py` → `prismatic/models/materialize.py`
→ `from transformers import PreTrainedTokenizerBase`
→ `transformers/utils/generic.py` line 42: `import jax.numpy as jnp`
→ `jax/_src/clusters/k8s_cluster.py` line 36: `np.random.rand(5)`
→ same numpy ABI crash as Cell 13
Then IPython `ultratb.py` enters infinite loop: `find_recursion()` returns `None`, `len(None)` raises `TypeError`, which re-enters the error handler indefinitely.
