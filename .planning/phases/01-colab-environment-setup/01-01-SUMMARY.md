---
phase: 01-colab-environment-setup
plan: "01"
subsystem: colab-notebook
tags: [jupyter, colab, install, mujoco, openvla, flash-attn, egl]
status: complete

dependencies:
  requires: []
  provides:
    - LIBERO/notebooks/01-colab-env-setup.ipynb
  affects:
    - plan-01-02 (Block B verification cells consume outputs/ directory and BDDL_FILE constant)
    - plan-01-03 (Block B VLA load cell uses REPO_ROOT for checkpoint path)

tech_stack:
  added:
    - torch==2.2.0 (cu121)
    - mujoco==2.3.7
    - robosuite==1.4.0
    - bddl==1.0.1
    - timm==0.9.10
    - tokenizers==0.19.1
    - peft==0.11.1
    - flash-attn==2.5.5
    - git+https://github.com/moojink/transformers-openvla-oft.git
  patterns:
    - UPPER_SNAKE_CASE path constants in config cell (REPO_ROOT, LIBERO_ROOT, LIBERO_PKG, OUT_DIR, BDDL_FILE)
    - Dependency ordering: apt before pip, torch before flash-attn, transformers fork last
    - GPU assertion cell (D-06) checking torch.cuda.get_device_name(0) against "A100"

key_files:
  created:
    - LIBERO/notebooks/01-colab-env-setup.ipynb
    - LIBERO/notebooks/outputs/ (directory placeholder, gitignored at runtime)
  modified: []

decisions:
  - "D-01: Single notebook with restart-aware cell ordering"
  - "D-02: Notebook lives at LIBERO/notebooks/01-colab-env-setup.ipynb"
  - "D-05: Target A100 in bf16 — no bitsandbytes dependency"
  - "D-06: GPU assertion cell using torch.cuda.get_device_name(0) with WARNING on non-A100"
  - "Transformers fork installed last per ordering constraint 3 (prevents pip downgrade to PyPI version)"

metrics:
  duration: "161s (~2m)"
  completed_date: "2026-07-08"
  tasks_completed: 1
  tasks_total: 1
  files_created: 1
  files_modified: 0
---

# Phase 01 Plan 01: Colab Block A Install Notebook Summary

**One-liner:** Jupyter notebook with 11 cells (0-10) implementing the Block A install sequence for the full SoARM + LIBERO + OpenVLA-OFT stack on Colab A100, respecting eight critical dependency ordering constraints.

## What Was Built

Created `LIBERO/notebooks/01-colab-env-setup.ipynb` — a valid nbformat 4.4 Jupyter notebook with exactly 11 cells (0 through 10):

| Cell | Type | Purpose |
|------|------|---------|
| 0 | markdown | Title block — Phase 1 goal + ENV-01/ENV-02/ENV-03 requirement checklist |
| 1 | code | REPO_ROOT config cell — all UPPER_SNAKE_CASE path constants, os.makedirs for outputs/ |
| 2 | code | GPU assertion (D-06) — checks A100 via torch.cuda.get_device_name(0), warns on non-A100 |
| 3 | markdown | Block A header — documents install ordering rationale |
| 4 | code | apt-get EGL system packages (libglfw3, libglew-dev, libosmesa6-dev, libgles2, etc.) |
| 5 | code | PyTorch 2.2.0 + torchvision 0.17.0 via --index-url cu121 |
| 6 | code | MuJoCo 2.3.7, gym 0.25.2, robosuite 1.4.0, bddl 1.0.1 + LIBERO sim stack |
| 7 | code | LIBERO editable install via LIBERO_PKG variable |
| 8 | code | OpenVLA-OFT packages (timm, tokenizers, sentencepiece, peft, accelerate) + transformers fork last |
| 9 | code | flash-attn 2.5.5 --no-build-isolation (with 10-min compile warning comment) |
| 10 | markdown | STOP — Restart runtime instruction |

## Key Technical Decisions

**Dependency ordering** (enforced by cell sequence):
1. apt-get EGL packages BEFORE pip mujoco — C EGL libraries must exist when mujoco Python extension builds
2. PyTorch 2.2.0 BEFORE flash-attn — flash-attn compiles CUDA kernels against installed torch headers
3. transformers fork installed LAST — pip resolver cannot downgrade to PyPI version when fork is the final install

**REPO_ROOT pattern:** Cell 1 defines a single user-editable `REPO_ROOT` constant. All other paths derive from it: `LIBERO_ROOT`, `LIBERO_PKG`, `OUT_DIR`, `BDDL_FILE`. Researcher only needs to change one value.

**GPU assertion (D-06):** Non-fatal warning (not raise) on non-A100 — researcher may want to proceed for install-only testing on T4. A100 is required only for ENV-03 (VLA bf16 load).

**No Block B content:** Cells 0-10 contain zero MuJoCo imports, zero MUJOCO_GL settings, zero verification logic — those belong in Plan 02 per scope boundary.

## Verification Results

```
Total cells: 11  ✓
Notebook structure: OK  ✓
No forbidden imports (import mujoco, import robosuite, import libero, MUJOCO_GL): OK  ✓
Cell 1 (config): OK  ✓
Cell 2 (GPU assert): OK  ✓
Cell 5 (torch): OK  ✓
Cell 6 (mujoco+robosuite): OK  ✓
Cell 8 (openvla fork): OK  ✓
Cell 9 (flash-attn): OK  ✓
Cell 10 (restart markdown): OK  ✓
outputs/ directory: OK  ✓
ALL ACCEPTANCE CRITERIA: PASSED
```

## Deviations from Plan

None — plan executed exactly as written.

## Threat Surface Scan

No new security-relevant surfaces introduced. The notebook installs packages listed in the Package Legitimacy Audit in RESEARCH.md. All approved packages use exact version pins. The transformers fork (T-01-01) is installed as specified; a commit SHA comment placeholder is included in Cell 8 noting where to pin a SHA for reproducibility.

## Known Stubs

None. The notebook is a pure install script with no data wiring. `REPO_ROOT` is a user-editable constant that the researcher must set correctly before running — this is intentional and documented in Cell 0.

## Self-Check: PASSED

- `LIBERO/notebooks/01-colab-env-setup.ipynb` exists: FOUND
- Commit `3fb1d31` exists: FOUND
- All 11 cells present with correct types: VERIFIED
- All acceptance criteria satisfied: VERIFIED
