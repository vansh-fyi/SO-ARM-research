---
phase: 3
slug: vla-inference-loop
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-07-18
---

# Phase 3 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | None (project has no pytest/unittest suite) — verification is PASS/FAIL notebook cells, per Phase 1/2 established convention |
| **Config file** | none — see Wave 0 |
| **Quick run command** | Run the relevant notebook cell manually on Colab (no CLI test runner in this project) |
| **Full suite command** | Run both Notebook A (OFT) and Notebook B (π0) end-to-end on Colab |
| **Estimated runtime** | ~15-30 min per notebook (GPU inference across 3 tasks × 5-10 episodes for OFT; smoke test for π0) |

**Note:** This project's PASS/FAIL-cell convention (established Phase 1, continued Phase 2) is the de facto validation architecture. This phase's "tests" are the printed PASS/FAIL-per-episode and the aggregated summary table (CONTEXT.md D-14), read by the human verifier on Colab — not automated pytest assertions.

---

## Sampling Rate

- **After every task commit:** Manually re-run the affected notebook cell(s) on Colab after each code change (no fast local test loop exists — GPU is Colab-only).
- **After every plan wave:** Re-run the full notebook (Block A install → restart → Block B verification) end-to-end, consistent with Phase 1/2's "restart ≠ clean slate" lesson — verify claims against a genuinely fresh runtime, not a persisted one.
- **Before `/gsd-verify-work`:** Both notebooks green (all PASS cells) plus the aggregated success-rate summary table produced.
- **Max feedback latency:** ~30 min (one full Colab notebook run)

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 03-01-xx | 01 | 1 | VLA-01 | — | OFT produces a valid 7-D action from (image, language) | manual (Colab cell) + printed shape assertion | `assert actions.shape == (8, 7)` inline in Notebook A | ❌ Wave 0 — new notebook cell | ⬜ pending |
| 03-01-xx | 01 | 1 | VLA-02 | — | Video file saved per episode | manual (Colab cell) + file-existence check | `assert os.path.exists(video_path)` inline in Notebook A/B | ❌ Wave 0 — new notebook cell | ⬜ pending |
| 03-01-xx | 01 | 1 | VLA-03 | — | Success rate measured via `check_success()` | manual (Colab cell) + printed PASS/FAIL + summary table | Inline print statements in the eval loop (D-14) | ❌ Wave 0 — new notebook cell | ⬜ pending |
| 03-02-xx | 02 | 2 | VLA-04 | T-3-01 | π0 backend swappable via same interface, no downstream code change | manual (Colab cell) + smoke-test run | Notebook B end-to-end run against the same `predict()`-calling eval-loop code as Notebook A | ❌ Wave 0 — new notebook + shared interface module | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*
*Exact Task IDs to be finalized by the planner; requirement coverage must match this table.*

---

## Wave 0 Requirements

- [ ] `LIBERO/libero/libero/vla/interface.py` — the shared `predict()` interface module (D-04) does not exist yet.
- [ ] `LIBERO/libero/libero/vla/oft_backend.py` — OFT backend wrapper implementing the interface (net-new; Phase 1 has the loading pattern but not this wrapper).
- [ ] `LIBERO/libero/libero/vla/pi0_backend.py` — π0/openpi backend wrapper implementing the interface (entirely new, no prior art in this project).
- [ ] `LIBERO/libero/libero/vla/eval_loop.py` — shared eval-loop helper (per-step success polling, chunk replay, video writing) reusable by both notebooks.
- [ ] `LIBERO/notebooks/03a-oft-inference-eval.ipynb` — new notebook.
- [ ] `LIBERO/notebooks/03b-pi0-inference-smoketest.ipynb` — new notebook.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| End-to-end language prompt → rendered video (single cell run) | VLA-01, VLA-02 | Requires live Colab GPU + rendered simulation output; no headless CI in this project | Run Notebook A's demo cell with a task prompt, confirm a video file is produced and visually correct |
| π0 backend swap without downstream code changes | VLA-04 | Requires actually running a second, dependency-conflicting VLA stack on Colab in a separate kernel/notebook | Run Notebook B end-to-end; confirm the same `eval_loop.py`/interface code (unchanged) drives both backends |
| Success/failure detection visually matches simulation outcome | VLA-03 | BDDL goal-condition correctness can only be confirmed by watching the rendered video against the printed PASS/FAIL | Human reviews a sample of saved episode videos against their printed PASS/FAIL result |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < ~30 min (one full Colab notebook run)
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
