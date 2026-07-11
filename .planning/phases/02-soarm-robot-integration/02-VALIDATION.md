---
phase: 2
slug: soarm-robot-integration
status: approved
nyquist_compliant: true
wave_0_complete: false
created: 2026-07-11
---

# Phase 2 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | None (script-based harness) — `explorations/soarm_sanity.py`, built in Wave 1 (plan 02-01 Task 1) |
| **Config file** | none — Wave 0 IS plan 02-01 Task 1 (harness is the first deliverable) |
| **Quick run command** | `conda run -n libero python explorations/soarm_sanity.py --check compile` (plus the check mode the current task flips GREEN) |
| **Full suite command** | `conda run -n libero python explorations/soarm_sanity.py --check all` |
| **Estimated runtime** | ~60–120 seconds (`--check all`: compile, model, reset, render, soak, tasks) |

---

## Sampling Rate

- **After every task commit:** Run the task's `--check <mode>` plus regression checks named in its acceptance criteria (e.g., `--check model` after physics tuning)
- **After every plan wave:** Run all check modes GREEN so far (RED→GREEN progression: 02-01 → compile-arm; 02-02 → compile+model; 02-03 → +reset+soak; 02-04 → `--check all`)
- **Before `/gsd-verify-work`:** `--check all` exits 0 locally AND Colab notebook `02-soarm-integration-check.ipynb` shows ENV-04..07 PASS (plan 02-05)
- **Max feedback latency:** 120 seconds (local checks; Colab run in 02-05 is a human-gated exception)

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 02-01-01 | 01 | 1 | ENV-04 | — | N/A | script | `conda run -n libero python explorations/soarm_sanity.py --help` | ❌ W0 (this task creates it) | ⬜ pending |
| 02-01-02 | 01 | 1 | ENV-04 | T-02-01 | Pinned-SHA vendoring, sha256 in VENDOR.txt | script | vendor integrity python check (plan 02-01 Task 2 verify block) | ❌ W0 | ⬜ pending |
| 02-01-03 | 01 | 1 | ENV-04 | — | N/A | script | MuJoCo 2.3.7 compile + ElementTree structural asserts (plan 02-01 Task 3 verify block) | ❌ W0 | ⬜ pending |
| 02-02-01 | 02 | 2 | ENV-04 | — | N/A | script | `conda run -n libero python explorations/soarm_sanity.py --check compile` | ✅ (after 02-01) | ⬜ pending |
| 02-02-02 | 02 | 2 | ENV-04, ENV-05 | — | N/A | script | `--check compile` + class import asserts | ✅ | ⬜ pending |
| 02-02-03 | 02 | 2 | ENV-05 | — | N/A | script | `conda run -n libero python explorations/soarm_sanity.py --check model` + ROBOT_CLASS_MAPPING assertion | ✅ | ⬜ pending |
| 02-03-01 | 03 | 3 | ENV-05, ENV-07 | — | N/A | script | `conda run -n libero python explorations/soarm_sanity.py --check reset` (peak force < 10.0 N) | ✅ | ⬜ pending |
| 02-03-02 | 03 | 3 | ENV-07 | — | N/A | script | `conda run -n libero python explorations/soarm_sanity.py --check soak` (500 steps × 3 seeds) | ✅ | ⬜ pending |
| 02-04-01 | 04 | 4 | ENV-07 | — | N/A | script + visual | `--check render` + non-empty PNG asserts; visual judgment of saved frames | ✅ | ⬜ pending |
| 02-04-02 | 04 | 4 | ENV-06, ENV-07 | — | N/A | script | `--check tasks` (3 BDDLs × 50 steps) then `--check all` | ✅ | ⬜ pending |
| 02-05-01 | 05 | 5 | ENV-04..07 | — | N/A | notebook | notebook JSON structural asserts (plan 02-05 Task 1 verify block) | ✅ | ⬜ pending |
| 02-05-02 | 05 | 5 | ENV-04..07 | — | N/A | manual (checkpoint:human-verify) | Colab run of `LIBERO/notebooks/02-soarm-integration-check.ipynb` — human sign-off | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `explorations/soarm_sanity.py` — the validation harness itself (plan 02-01 Task 1, ordered first in Wave 1; all later checks depend on it)

*The harness is a phase deliverable, not pre-existing infrastructure — Wave 0 and Wave 1 coincide by design (RESEARCH.md Validation Architecture).*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Rendered frames visually correct (right-side-up, correct angle, no mesh artifacts) | ENV-07 | Visual quality judgment has no reliable automated oracle | Read `explorations/outputs/soarm_agentview.png` and `soarm_eye_in_hand.png` after `--check render`; gripper jaws visible at frame bottom, workspace centered |
| Colab end-to-end verification (D-10) | ENV-04..07 | Phase closes on GPU-runtime evidence, human confirms PASS cells | Run `LIBERO/notebooks/02-soarm-integration-check.ipynb` on Colab; all ENV-04..07 cells print PASS (plan 02-05 checkpoint:human-verify) |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references (harness built first in plan 02-01)
- [x] No watch-mode flags (all commands are one-shot conda invocations)
- [x] Feedback latency < 120s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved 2026-07-11
