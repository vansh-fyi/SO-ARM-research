---
status: testing
phase: 01-colab-environment-setup
source: [01-VERIFICATION.md]
started: 2026-07-09T18:15:00Z
updated: 2026-07-10T09:48:00Z
---

## Current Test

[testing complete]

## Tests

### 1. numpy ABI gate — Block A final gate cell
expected: Run Block A top-to-bottom on a fresh Colab A100. The gate cell (last executable Block A cell, before the STOP restart markdown) prints "numpy ABI gate: PASS — numpy 1.26.4 coherent on disk; safe to restart runtime". On RuntimeError, do NOT restart — report the diagnostics it prints.
result: pass

### 2. ENV-01 — package versions + numpy row + ABI canary
expected: After restart, the ENV-01 cell shows a numpy row with installed 1.26.4 / expected 1.26.4 / OK, the in-kernel numpy.random ABI canary passes, and the cell prints "ENV-01: PASS". A version mismatch in any row now correctly produces FAIL (verdict bug fixed).
result: pass

### 3. ENV-02 — LIBERO EGL render
expected: The ENV-02 cell prints "ENV-02: PASS" with a non-black LIBERO render frame. No "numpy.dtype size changed" ValueError anywhere in the import chain (gym → numpy.random.mtrand). Watch for a NEW failure class at env.reset() (mujoco 3.3.2 API path has never executed post-fix) — if one appears, report the full traceback.
result: pass

### 4. ENV-03 — OpenVLA-OFT load + 7-D action
expected: The ENV-03 cell (with USE_TORCH=1/USE_TF=0/USE_FLAX=0 guards active) loads moojink/openvla-7b-oft-finetuned-libero-spatial in bfloat16 and prints "ENV-03: PASS" with a 7-D per-step action. No jax or tensorflow frames in any traceback; no IPython ultratb infinite loop.
result: issue
reported: "ModuleNotFoundError: No module named 'dlimp' — prismatic/__init__.py eagerly imports prismatic.vla.datasets.rlds.dataset which requires dlimp (a training-time dep not installed by Cell 6). RuntimeError raised before model load."
severity: major

## Summary

total: 4
passed: 3
issues: 1
pending: 0
skipped: 0
blocked: 0

## Gaps

- truth: "ENV-03 cell loads OpenVLA-OFT in bfloat16 and prints ENV-03: PASS with a 7-D action"
  status: failed
  reason: "User reported: ModuleNotFoundError: No module named 'dlimp' — prismatic/__init__.py eagerly imports the RLDS dataset chain (prismatic.vla.datasets.rlds.dataset) which requires dlimp. dlimp was listed as missing in Cell 6 pip install but was never installed. RuntimeError raised before model load."
  severity: major
  test: 4
  root_cause: "prismatic/vla/__init__.py line 1 eagerly imports get_vla_dataset_and_collator from materialize.py, which pulls in the full dataset chain (datasets.py → rlds/dataset.py → import dlimp). dlimp is a required transitive dep (git+https://github.com/moojink/dlimp_openvla) listed in openvla-oft's pyproject.toml but never installed by Cell 8. The ENV-03 cell clones openvla-oft and adds it to sys.path, then calls importlib.import_module('prismatic.training.train_utils') which triggers the full __init__ chain — crashing on the missing dlimp before reaching model load."
  artifacts:
    - path: "LIBERO/notebooks/01-colab-env-setup.ipynb"
      cell_id: "98b86f1a"
      issue: "ENV-03 cell (cell-id 98b86f1a) does not install dlimp before the prismatic import check"
  missing:
    - "Add dlimp install in ENV-03 cell (cell 98b86f1a), immediately after sys.path.insert for openvla-oft and before importlib.import_module call: subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'git+https://github.com/moojink/dlimp_openvla'], check=True)"
  fix_note: "Cannot patch .ipynb directly — apply in Colab. In cell 98b86f1a, after the sys.path.insert block and before the try/importlib.import_module block, add: import subprocess as _sp; _sp.run([sys.executable, '-m', 'pip', 'install', '-q', 'git+https://github.com/moojink/dlimp_openvla'], check=True); print('dlimp installed'). Then re-run the ENV-03 cell."
