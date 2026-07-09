---
status: testing
phase: 01-colab-environment-setup
source: [01-VERIFICATION.md]
started: 2026-07-09T18:15:00Z
updated: 2026-07-09T18:15:00Z
---

## Current Test

number: 1
name: numpy ABI gate — Block A final gate cell
expected: |
  Run Block A top-to-bottom on a fresh Colab A100 runtime. The new final gate cell
  (after flash-attn, before the STOP restart markdown) purges numpy, cleanly
  reinstalls numpy==1.26.4, and runs a fresh-subprocess ABI probe walking the exact
  ENV-02 crash path. It prints: "numpy ABI gate: PASS — numpy 1.26.4 coherent on
  disk; safe to restart runtime". If it raises RuntimeError instead: do NOT restart —
  report the printed diagnostics.
awaiting: user response

## Tests

### 1. numpy ABI gate — Block A final gate cell
expected: Run Block A top-to-bottom on a fresh Colab A100. The gate cell (last executable Block A cell, before the STOP restart markdown) prints "numpy ABI gate: PASS — numpy 1.26.4 coherent on disk; safe to restart runtime". On RuntimeError, do NOT restart — report the diagnostics it prints.
result: [pending]

### 2. ENV-01 — package versions + numpy row + ABI canary
expected: After restart, the ENV-01 cell shows a numpy row with installed 1.26.4 / expected 1.26.4 / OK, the in-kernel numpy.random ABI canary passes, and the cell prints "ENV-01: PASS". A version mismatch in any row now correctly produces FAIL (verdict bug fixed).
result: [pending]

### 3. ENV-02 — LIBERO EGL render
expected: The ENV-02 cell prints "ENV-02: PASS" with a non-black LIBERO render frame. No "numpy.dtype size changed" ValueError anywhere in the import chain (gym → numpy.random.mtrand). Watch for a NEW failure class at env.reset() (mujoco 3.3.2 API path has never executed post-fix) — if one appears, report the full traceback.
result: [pending]

### 4. ENV-03 — OpenVLA-OFT load + 7-D action
expected: The ENV-03 cell (with USE_TORCH=1/USE_TF=0/USE_FLAX=0 guards active) loads moojink/openvla-7b-oft-finetuned-libero-spatial in bfloat16 and prints "ENV-03: PASS" with a 7-D per-step action. No jax or tensorflow frames in any traceback; no IPython ultratb infinite loop.
result: [pending]

## Summary

total: 4
passed: 0
issues: 0
pending: 4
skipped: 0
blocked: 0

## Gaps
