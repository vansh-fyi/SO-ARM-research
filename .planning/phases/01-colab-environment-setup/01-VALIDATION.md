---
phase: 1
slug: colab-environment-setup
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-07-08
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Python assertions + notebook cell output inspection (no pytest — Colab notebook cells ARE the test harness) |
| **Config file** | none — validation is inline in notebook verification cells |
| **Quick run command** | Run the ENV-01, ENV-02, ENV-03 verification cells in `LIBERO/notebooks/01-colab-env-setup.ipynb` |
| **Full suite command** | Execute all cells top-to-bottom on a fresh Colab A100 runtime |
| **Estimated runtime** | ~15–25 minutes (pip installs dominate) |

---

## Sampling Rate

- **After every task commit:** Run the relevant ENV-XX verification cell
- **After every plan wave:** Re-execute all cells from scratch (full fresh runtime)
- **Before `/gsd-verify-work`:** Full top-to-bottom fresh run must be green
- **Max feedback latency:** 25 minutes (dominated by pip install + model download)

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | Status |
|---------|------|------|-------------|-----------|-------------------|--------|
| 1-01-01 | 01 | 1 | ENV-01 | notebook cell | Run ENV-01 verification cell → prints all versions, no ImportError | ⬜ pending |
| 1-01-02 | 01 | 1 | ENV-01 | notebook cell | NVIDIA EGL ICD cell + MUJOCO_GL cell execute without error | ⬜ pending |
| 1-01-03 | 01 | 1 | ENV-01 | notebook cell | LIBERO config bootstrap cell creates ~/.libero/config.yaml before any import | ⬜ pending |
| 1-01-04 | 01 | 2 | ENV-02 | notebook cell | Run ENV-02 verification cell → renders frame, assert frame.max() > 0 | ⬜ pending |
| 1-01-05 | 01 | 2 | ENV-03 | notebook cell | Run ENV-03 verification cell → action.shape == (7,) printed | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `LIBERO/notebooks/01-colab-env-setup.ipynb` — notebook scaffolded with section headers and empty verification cells before install cells run
- [ ] `~/.libero/config.yaml` bootstrap cell exists before any `import libero`
- [ ] NVIDIA EGL ICD cell (`/usr/share/glvnd/egl_vendor.d/10_nvidia.json`) exists before any MuJoCo import

*All verification in this phase is notebook-cell-based; no pytest framework setup needed.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Rendered frame is visually correct (right-side-up, non-black, visible robot/table) | ENV-02 | Pixel value > 0 proves non-black, but visual sanity requires human eye | Inspect `libero_render_check.png` displayed inline in notebook — confirm robot arm and table are visible |
| OpenVLA-OFT action values are plausible (not all zeros, not NaN) | ENV-03 | Shape check is automated; value sanity is visual | Print action tensor, confirm no NaN, values roughly in [-1, 1] range |

---

## Validation Sign-Off

- [ ] All tasks have notebook cell verification or manual verification instructions
- [ ] Sampling continuity: each ENV requirement has a dedicated verification cell
- [ ] Wave 0 covers notebook scaffolding and required bootstrap cells
- [ ] No watch-mode flags
- [ ] Feedback latency < 25 min (fresh Colab runtime)
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
