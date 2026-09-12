---
status: complete
phase: 01-colab-environment-setup
source: [01-VERIFICATION.md]
started: 2026-07-09T18:15:00Z
updated: 2026-07-10T17:30:00Z
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
result: pass
resolved: 2026-07-10 — Step 5b cell baked into Block A (commits 96cbb0d, b095ee6): dlimp
  via kvablack clone + pip --no-deps (BOTH dlimp repos pin tensorflow==2.15.0, which has
  no cp312 wheel), protobuf runtime restore, and all five missing openvla-oft eager-chain
  deps (tensorflow_graphics --no-deps, draccus, jsonlines, wandb, diffusers). ENV-03 PASS
  observed live on a fresh A100 VM: action shape (8, 7), dtype float64, bf16 on cuda:0,
  unnorm_key libero_spatial_no_noops. Evidence committed in notebook outputs (1a37159).
history: "Initially failed with ModuleNotFoundError: dlimp; then protobuf VersionError
  (gencode 6.31.1 vs runtime 5.29.6); then ModuleNotFoundError: tensorflow_graphics.
  Full failure-chain forensics: .planning/notes/dlimp-install-forensics.md"

## Summary

total: 4
passed: 4
issues: 0
pending: 0
skipped: 0
blocked: 0

## Gaps

- truth: "ENV-03 cell loads OpenVLA-OFT in bfloat16 and prints ENV-03: PASS with a 7-D action"
  status: resolved (2026-07-10)
  resolution: |
    Fix baked into notebook Block A as Step 5b (cell-8b-dlimp), commits 96cbb0d + b095ee6:
    1. dlimp: clone kvablack/dlimp + pip install --no-deps -e (both dlimp repos pin
       tensorflow==2.15.0 — no cp312 wheel exists, so ANY deps-resolving install fails).
    2. protobuf: pip install -U protobuf after Block A installs drag runtime below
       tensorflow_metadata's gencode (observed 5.29.6 vs 6.31.1 → VersionError).
    3. Remaining eager-chain deps (openvla-oft installed --no-deps never pulls them):
       tensorflow_graphics==2021.12.3 (--no-deps), draccus==0.8.0, jsonlines, wandb,
       diffusers==0.30.3 — enumerated from prismatic/ source, not crash-by-crash.
    The failing Block B patch cell (pip install git+.../dlimp_openvla, always exit 1)
    was deleted. Optional HF token bootstrap cell added to Block B (dbcf69a).
    ENV-03 PASS verified live on a fresh A100 VM 2026-07-10; outputs committed (1a37159).
    Residual note: the passing session applied the Step 5b dep commands manually
    (byte-identical to the baked cell); next fresh-VM session start doubles as the
    zero-touch validation.
  original_status: failed
  reason: "User reported: ModuleNotFoundError: No module named 'dlimp' — prismatic/__init__.py eagerly imports the RLDS dataset chain (prismatic.vla.datasets.rlds.dataset) which requires dlimp. dlimp was listed as missing in Cell 6 pip install but was never installed. RuntimeError raised before model load."
  severity: major
  test: 4
  root_cause: "prismatic/vla/__init__.py line 1 eagerly imports get_vla_dataset_and_collator from materialize.py, which pulls in the full dataset chain (datasets.py → rlds/dataset.py → import dlimp). dlimp is a required transitive dep listed in openvla-oft's pyproject.toml as 'git+https://github.com/moojink/dlimp_openvla' but that pip URL fails (non-zero exit). The clone-then-editable-install path is required."
  artifacts:
    - path: "LIBERO/notebooks/01-colab-env-setup.ipynb"
      cell_id: "98b86f1a"
      issue: "ENV-03 cell does not install dlimp before the prismatic import check"
  missing:
    - "In cell 98b86f1a, after sys.path.insert and before importlib.import_module: clone kvablack/dlimp and pip install -e it"
  fix_confirmed: |
    The following snippet (run in Colab before the prismatic import check) resolved the issue:

      import subprocess, sys
      subprocess.run(
          'git clone --depth 1 https://github.com/kvablack/dlimp /content/dlimp_kvablack 2>&1 && '
          f'{sys.executable} -m pip install -q -e /content/dlimp_kvablack',
          shell=True, check=True
      )

    After this, ENV-03 loaded moojink/openvla-7b-oft-finetuned-libero-spatial in bfloat16
    and printed: ENV-03: PASS — action shape: (8, 7), dtype: float64
    unnorm_key resolved to: libero_spatial_no_noops
    Note: pip install git+https://github.com/moojink/dlimp_openvla fails — use kvablack/dlimp instead.
  fix_note: "Add dlimp install to Block A Cell 8 (openvla-oft install) so it is present before Block B ENV-03 runs. Use git clone kvablack/dlimp + pip install -e, NOT the moojink/dlimp_openvla pip URL (that URL returns non-zero exit)."
