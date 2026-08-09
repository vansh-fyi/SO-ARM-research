---
phase: 5
slug: spatial-awareness
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-08-09
---

# Phase 5 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (no `pytest.ini`/`pyproject.toml` found; invoked directly per this project's established convention) |
| **Config file** | none — Wave 0 installs |
| **Quick run command** | `conda run -n libero pytest LIBERO/libero/libero/<module>/test_<name>.py -x -q` |
| **Full suite command** | `conda run -n libero pytest LIBERO/libero/libero -x -q` |
| **Estimated runtime** | ~30-60 seconds (local, MuJoCo, no GPU) |

---

## Sampling Rate

- **After every task commit:** Run targeted `-k` filtered run of the relevant new test file
- **After every plan wave:** Run full per-module test files for everything touched that wave
- **Before `/gsd-verify-work`:** All Wave 0 new test files green locally; Colab-side multi-image inference spot-check documented as a manual verification step
- **Max feedback latency:** ~60 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| TBD-01 | TBD | 0 | SPAT-01 | — | Both cameras configured and render non-degenerate frames | integration | `conda run -n libero pytest LIBERO/libero/libero/envs/test_camera_config.py -x -q` | ❌ W0 | ⬜ pending |
| TBD-02 | TBD | TBD | SPAT-02 | — | `images` dict passed to `backend.predict()` contains both view keys | unit (mock backend) | `conda run -n libero pytest LIBERO/libero/libero/vla/test_eval_loop.py -x -q -k spatial` | ⚠️ extend existing | ⬜ pending |
| TBD-03 | TBD | 0 | SPAT-03 | — | `{cam}_depth` obs keys present and within valid range | unit/integration | `conda run -n libero pytest LIBERO/libero/libero/envs/test_camera_config.py -x -q -k depth` | ❌ W0 | ⬜ pending |
| TBD-04 | TBD | 0 | SPAT-04 | — | Depth-derived XYZ within tolerance of `sim.data.body_xpos` ground truth (D-04) | integration | `conda run -n libero pytest LIBERO/libero/libero/perception/test_depth_xyz.py -x -q` | ❌ W0 | ⬜ pending |
| TBD-05 | TBD | TBD | SPAT-05 | — | Each new spatial predicate correctly evaluates true/false cases; each BDDL task's goal is satisfiable by a known-good state and not by a known-bad state | unit + integration | `conda run -n libero pytest LIBERO/libero/libero/envs/test_spatial_predicates.py -x -q` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

*Task IDs are placeholders — the planner assigns real task/plan/wave IDs; this table's requirement→test mapping is the binding contract.*

Real multi-image VLA inference (both backends actually consuming both views and producing sane actions) is **manual/Colab-only**, per this project's established GPU-backed-behavior convention (Phase 1/3 precedent) — not part of the local automated suite.

---

## Wave 0 Requirements

- [ ] `LIBERO/libero/libero/envs/test_camera_config.py` — stubs for SPAT-01, SPAT-03 (camera registration + depth key presence/range)
- [ ] `LIBERO/libero/libero/perception/test_depth_xyz.py` — stubs for SPAT-04 (D-04's ground-truth comparison)
- [ ] `LIBERO/libero/libero/envs/test_spatial_predicates.py` — stubs for SPAT-05 (new predicate unit tests + BDDL goal integration tests)
- [ ] Extend `LIBERO/libero/libero/vla/test_eval_loop.py` — SPAT-02 (images dict contains both view keys)
- [ ] Extend `LIBERO/libero/libero/vla/test_pi0_backend.py` — D-02 for Pi0Backend (mock-based, verifies both real image arrays reach the obs dict, not duplicated)
- [ ] No framework install needed — pytest already used project-wide

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Both VLA backends (pi0, OFT) actually consume both camera views during real inference and produce coherent actions | SPAT-02 | Requires GPU-backed model inference (Colab), not locally testable per this project's established Phase 1/3 convention | Run the Colab notebook's VLA inference cell with both camera obs populated; confirm no shape/key errors and non-degenerate action output for at least one spatial task |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 60s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
