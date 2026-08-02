---
phase: 4
slug: dataset-collection
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-08-02
---

# Phase 4 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.4.2 (installed, verified) |
| **Config file** | none — rootdir-based collection, same as `LIBERO/libero/libero/vla/test_*.py`'s established pattern |
| **Quick run command** | `pytest LIBERO/libero/libero/datasets/ -x` (from repo root) |
| **Full suite command** | `pytest LIBERO/libero/libero/datasets/ LIBERO/libero/libero/vla/ -v` |
| **Estimated runtime** | ~30s quick / several minutes full (includes real local sim episodes) |

---

## Sampling Rate

- **After every task commit:** Run `pytest LIBERO/libero/libero/datasets/ -x` (fast subset — unit tests + a 1-2 episode integration smoke test, not the full 100+-demo collection run)
- **After every plan wave:** Full suite + a real (small, e.g. 3-5 episode) end-to-end collect → write → replay → normalize run against the actual SOARM env
- **Before `/gsd-verify-work`:** Full 100+-demo collection run, full-dataset states-only replay check, sampled image-regeneration replay check, all green
- **Max feedback latency:** ~30s (quick command)

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| TBD-01 | TBD | 0 | DATA-01 | — | HDF5 obs group uses LIBERO's renamed keys (`agentview_rgb` etc.), not raw robosuite keys | unit | `pytest LIBERO/libero/libero/datasets/test_hdf5_writer.py::test_obs_key_naming -x` | ❌ W0 | ⬜ pending |
| TBD-02 | TBD | 0 | DATA-01 | — | Scripted collector produces valid robomimic-schema HDF5 with image obs present | integration (real local sim, small N) | `pytest LIBERO/libero/libero/datasets/test_hdf5_writer.py -x` | ❌ W0 | ⬜ pending |
| TBD-03 | TBD | 0 | DATA-02 | — | Full-dataset cheap states-only replay determinism check | integration (real local sim, all demos) | `pytest LIBERO/libero/libero/datasets/test_replay.py::test_all_demos_state_replay -x` | ❌ W0 | ⬜ pending |
| TBD-04 | TBD | 0 | DATA-02 | — | Sampled full-image-regeneration replay determinism check | integration (real local sim, sampled subset) | `pytest LIBERO/libero/libero/datasets/test_replay.py::test_sampled_image_replay -x` | ❌ W0 | ⬜ pending |
| TBD-05 | TBD | 0 | DATA-03 | — | Normalization stats match OpenVLA q01/q99/mean/std schema, non-degenerate (not NaN, q01<q99) | unit (synthetic array input) | `pytest LIBERO/libero/libero/datasets/test_normalization.py -x` | ❌ W0 | ⬜ pending |
| TBD-06 | TBD | 0 | DATA-04 | — | Teleop path produces demos in the identical HDF5 schema as the scripted path | manual + integration (live keyboard input not automatable; schema check is) | `pytest LIBERO/libero/libero/datasets/test_hdf5_writer.py::test_schema_matches_across_sources -x` (schema) + manual keyboard session (live input) | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*
*Task IDs finalized once the planner assigns actual plan/task numbers — this table is populated from RESEARCH.md's Phase Requirements → Test Map.*

---

## Wave 0 Requirements

- [ ] `LIBERO/libero/libero/datasets/__init__.py` — package scaffold, mirroring `vla/__init__.py`'s graceful-degradation-import pattern
- [ ] `LIBERO/libero/libero/datasets/test_hdf5_writer.py` — covers DATA-01
- [ ] `LIBERO/libero/libero/datasets/test_replay.py` — covers DATA-02
- [ ] `LIBERO/libero/libero/datasets/test_normalization.py` — covers DATA-03
- [ ] No new pytest framework install needed — already present locally

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Human-operator teleoperation recording session | DATA-04 | Requires live keyboard input from a human operator — not automatable | Run the teleop collection script locally, complete at least one full pick-place episode via keyboard control, confirm the resulting demo lands in the same HDF5 schema as scripted demos (verified by the automated schema-match test) |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s (quick command)
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
