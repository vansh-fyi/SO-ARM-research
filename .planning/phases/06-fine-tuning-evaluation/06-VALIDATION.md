---
phase: 6
slug: fine-tuning-evaluation
status: draft
nyquist_compliant: true
wave_0_complete: true
created: 2026-08-20
---

# Phase 6 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest [VERIFIED: `LIBERO/libero/libero/conftest.py` exists; `test_*.py` files present throughout `LIBERO/libero/libero/`] |
| **Config file** | `LIBERO/libero/libero/conftest.py` (fixtures only; no dedicated `pytest.ini`) |
| **Quick run command** | `pytest LIBERO/libero/libero/datasets/test_rlds_converter.py -x` |
| **Full suite command** | `pytest LIBERO/libero/libero/ -x` |
| **Estimated runtime** | ~30-60s local suite (excludes Colab-only GPU steps) |

---

## Sampling Rate

- **After every task commit:** Run `pytest LIBERO/libero/libero/datasets/test_rlds_converter.py -x` or `pytest LIBERO/libero/libero/vla/test_eval_loop.py -x -k seed` (whichever module changed)
- **After every plan wave:** Run `pytest LIBERO/libero/libero/ -x` (full local suite — does not cover Colab-only GPU behaviors)
- **Before `/gsd-verify-work`:** Full local suite must be green AND a live Colab run of both notebooks (training completes, eval before/after tables populate, WandB dashboard shows both curve types)
- **Max feedback latency:** ~60 seconds (local suite)

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 06-01-XX | 01 | 0/1 | TUNE-01 | T-06-01 | RLDS converter validates HDF5 schema/shape/dtype, fails loudly on mismatch | unit | `pytest LIBERO/libero/libero/datasets/test_rlds_converter.py -x` | ❌ W0 | ⬜ pending |
| 06-01-XX | 01 | 1 | TUNE-01 | — | Converted dataset is `tfds.builder`-loadable | integration (Colab-only) | manual Colab cell + assertion | ❌ W0 — Colab-only, no local automated equivalent | ⬜ pending |
| 06-02-XX | 02 | 1/2 | TUNE-02 | — | `finetune.py` runs to completion without crashing on Colab A100 | manual-only (GPU required) | Colab notebook run + exit-code/log check | N/A — matches project convention for GPU-only-verifiable steps | ⬜ pending |
| 06-02-XX | 02 | 2 | TUNE-02 | — | Resume-from-checkpoint actually resumes (not restart from step 0) | manual-only (GPU required, disconnect-simulation) | Colab notebook: kill+relaunch mid-run, check WandB step continuity | N/A — Colab-only | ⬜ pending |
| 06-03-XX | 03 | 1 | TUNE-03 | T-06-02 | Seeded before/after `run_suite` produces identical initial object placements across two runs with same seed list | unit/integration | `pytest LIBERO/libero/libero/vla/test_eval_loop.py -x -k seed` | ❌ W0 — new test, locally feasible (only exercises `env.seed()`+`reset()`) | ⬜ pending |
| 06-04-XX | 03/04 | 2/3 | TUNE-04 | T-06-01 | WandB dashboard shows both training-loss curves AND eval success-rate results | manual-only (live WandB run) | Colab notebook + manual dashboard check | N/A — Colab-only | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

*Note: exact Task IDs finalized once PLAN.md files are written by the planner; this table's Plan/Wave columns are placeholders until then.*

---

## Wave 0 Requirements

- [ ] `LIBERO/libero/libero/datasets/test_rlds_converter.py` — unit tests covering TUNE-01 (HDF5→RLDS schema/shape/dtype correctness, fail-loud on mismatch)
- [ ] `LIBERO/libero/libero/vla/test_eval_loop.py` seed-determinism additions — covers TUNE-03's D-08 identical-seed requirement, locally testable without GPU
- [ ] No new fixture files needed — existing `conftest.py` + Phase 4's existing HDF5 test fixture pattern is reusable for RLDS converter unit tests
- [ ] Framework install: none — pytest already used project-wide

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| RLDS dataset is genuinely `tfds.builder`-loadable | TUNE-01 | Requires `tensorflow_datasets` + real TFDS builder registration, Colab-only per this project's kernel-split convention | Run the training notebook's RLDS-load cell; assert `tfds.builder(name, data_dir=...).info` returns expected feature schema without error |
| `finetune.py` completes a LoRA fine-tuning run without crashing on Colab A100 | TUNE-02 | Requires actual GPU + full openvla-oft dependency stack, cannot be simulated locally | Run training notebook end-to-end on Colab A100; check exit code and final log line |
| Resume-from-checkpoint actually resumes training state (not a silent restart from step 0) | TUNE-02 | Requires a real Colab session disconnect/relaunch to exercise the resume path | Kill the Colab runtime mid-training, relaunch notebook with `--resume`, verify WandB step counter continues rather than resetting to 0 |
| WandB dashboard shows both training-loss curves and eval success-rate results in the new dedicated project | TUNE-04 | Requires a live WandB run with real logged data | After a training + eval run, open the WandB project dashboard and visually confirm both metric types are present |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies — every `type="auto"` task across 06-01/06-02/06-03 carries a self-contained `<automated>` command (06-01 Task 1's verify now checks its own module import rather than Task 2's not-yet-created test file; 06-02 Task 1 now authors and runs `test_oxe_register.py` within the same task); `checkpoint:human-verify`/`checkpoint:decision` tasks are exempt by type.
- [x] Sampling continuity: no 3 consecutive tasks without automated verify — 06-02's only non-automated task (Task 3, package-legitimacy checkpoint) sits between two automated tasks (Task 2, Task 4); 06-03's checkpoint (Task 4) is the final task, preceded by three automated tasks.
- [x] Wave 0 covers all MISSING references — `test_rlds_converter.py` created and run in 06-01 Task 2 (self-contained, no longer invoked prematurely by Task 1); `test_eval_loop.py` seed-determinism additions created and run in 06-03 Task 1; no new fixture files needed; no new framework install.
- [x] No watch-mode flags
- [x] Feedback latency < 60s (local suite) — all local `pytest`/`python -c`/notebook-JSON-check commands complete well under 60s; Colab-only GPU steps remain explicitly manual per the Manual-Only Verifications table above, unaffected by this revision.
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** pending re-verification (revision iteration 1, 2026-08-20 — both BLOCKER findings from gsd-plan-checker resolved: 06-01 Task 1's verify decoupled from Task 2's test file; 06-02 Task 1 now authors `test_oxe_register.py` in the same task. Awaiting gsd-plan-checker re-run for formal sign-off.)
