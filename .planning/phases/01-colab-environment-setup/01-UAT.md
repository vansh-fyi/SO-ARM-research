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
