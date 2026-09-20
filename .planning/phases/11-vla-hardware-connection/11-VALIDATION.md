---
phase: 11
slug: vla-hardware-connection
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-09-20
---

# Phase 11 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (already used elsewhere in this repo, e.g. `scripts/test_calibration_utils.py`, `LIBERO/.../test_*.py`); **no test infrastructure currently exists in `control/`** |
| **Config file** | none — Wave 0 installs |
| **Quick run command** | `cd control && source .venv/bin/activate && pytest <new test file> -x` (once added) |
| **Full suite command** | same, no marker/filter needed yet given the small expected surface area |
| **Estimated runtime** | ~5-10 seconds (unit-only, no hardware I/O in the automated tests) |

---

## Sampling Rate

- **After every task commit:** Run the relevant new unit test file (`test_action_contract.py`, `test_safety_validator.py`, or `test_io_logger.py`) with `-x`
- **After every plan wave:** Run all of `control/`'s new test files together (no pre-existing full suite to merge into)
- **Before `/gsd-verify-work`:** All automated unit tests green, plus the mandatory hardware-in-the-loop checkpoints (e-stop verification, full episode recording) signed off by a human
- **Max feedback latency:** ~10 seconds (unit tests are pure-function/mocked, no real hardware or network round-trip)

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 11-01-01 | 01 | 0 | VLAHW-01 | — | Action contract (units/order/joint count) matches documented DEGREES + RANGE_0_100, 6-DOF contract | unit | `pytest control/test_action_contract.py -x` | ❌ W0 | ⬜ pending |
| 11-01-02 | 01 | 0 | VLAHW-02 | T-11-01 | Safety validator clamps out-of-range/NaN/malformed actions correctly | unit | `pytest control/test_safety_validator.py -x` | ❌ W0 | ⬜ pending |
| 11-01-03 | 01 | 0 | VLAHW-02 | T-11-02 | E-stop halts the control loop and does not depend on network state | manual (hardware-in-the-loop) | N/A — human-verify checkpoint | ❌ W0 (harness) | ⬜ pending |
| 11-01-04 | 01 | 0 | VLAHW-03 | — | JSONL log record has all required fields, one record per step, camera paths resolve to real files | unit + integration | `pytest control/test_io_logger.py -x` | ❌ W0 | ⬜ pending |
| 11-02-01 | 02 | 2 | VLAHW-04 | — | Full episode runs end-to-end, produces video + JSONL + termination reason | manual (hardware-in-the-loop) | N/A — human-verify checkpoint, cannot be automated without the real robot | ❌ W0 (harness) | ⬜ pending |
| 11-02-02 | 02 | 2 | VLAHW-05 | — | Findings write-up accurately reflects observed behavior | manual (write-up review) | N/A | — | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

*(Task IDs above are placeholders anchored to the plan/wave structure the researcher assumed; the planner assigns final task IDs — this table is the requirement→test contract, not a literal task list.)*

---

## Wave 0 Requirements

- [ ] `control/test_action_contract.py` — covers VLAHW-01 (mock/no-hardware-required assertions on unit conventions and joint order)
- [ ] `control/test_safety_validator.py` — covers VLAHW-02 (pure-function tests against `JOINT_LIMITS_DEG`, NaN/inf/stale rejection, no real hardware needed)
- [ ] `control/test_io_logger.py` — covers VLAHW-03 (schema validation against a synthetic step record)
- [ ] `pytest` install/config in `control/.venv` — not currently present; `pip install pytest` needed before any of the above can run
- [ ] No existing `conftest.py`/shared fixtures in `control/` — a minimal one (e.g. a fake/mock `Robot` object for unit tests that shouldn't touch real hardware) will likely be needed

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| E-stop (software keyboard interrupt) halts the control loop immediately | VLAHW-02 | Requires a live control loop driving real servos; can't be meaningfully simulated without a real robot/network-drop scenario | Start the VLA-driven control loop against the real SO-ARM101, trigger the e-stop key mid-motion, confirm motor commands stop immediately and the robot holds/returns to a safe position |
| Full observed episode (VLA driving the robot from prompt to termination) | VLAHW-04 | Requires the real robot, real cameras, and the live Colab↔local relay all working together — not unit-testable | Run the red-cube pick task end-to-end on the physical SO-ARM101; confirm synced video + JSONL log + termination reason are all produced and consistent |
| Findings write-up (go/no-go recommendation) | VLAHW-05 | Requires human judgment synthesizing the observed run(s), including whether lerobot#2210 reproduced | Human review of the write-up against what was actually observed during the hardware run |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 10s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
