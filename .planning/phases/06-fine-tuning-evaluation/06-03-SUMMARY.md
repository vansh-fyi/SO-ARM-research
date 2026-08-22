---
phase: 06-fine-tuning-evaluation
plan: 03
subsystem: vla
tags: [openvla-oft, peft, lora, wandb, huggingface-hub, colab, eval-loop]

# Dependency graph
requires:
  - phase: 06-fine-tuning-evaluation (Plan 06-02)
    provides: "finetune.py LoRA r=32 training run, HF_ADAPTER_REPO_ID naming convention, WANDB_PROJECT constant"
provides:
  - "eval_loop.py per-episode seed plumbing (run_episode/run_suite seed/episode_seeds kwargs)"
  - "adapter_backend.py FinetunedOFTBackend (HF Hub LoRA adapter reload via OFTBackend's proven loading path)"
  - "06b-eval.ipynb before/after seeded benchmark notebook with WandB eval logging"
affects: [phase-6-fine-tuning-evaluation-close, TUNE-03-verification, TUNE-04-verification]

# Tech tracking
tech-stack:
  added: [peft==0.20.0]
  patterns:
    - "Per-episode env.seed(seed) applied fresh inside run_suite's episode loop (not hoisted before it) to keep identical-seed before/after runs reproducible"
    - "Subclass-and-reuse-super().__init__() pattern for reloading a fine-tuned adapter without touching the base backend's proven loading path"
    - "Adapter's own dataset_statistics.json takes precedence over the base checkpoint's or Phase 4's stats, fetched from adapter_repo_id specifically"

key-files:
  created:
    - LIBERO/libero/libero/vla/adapter_backend.py
    - LIBERO/libero/libero/vla/test_adapter_backend.py
    - LIBERO/notebooks/06b-eval.ipynb
  modified:
    - LIBERO/libero/libero/vla/eval_loop.py
    - LIBERO/libero/libero/vla/test_eval_loop.py
    - LIBERO/libero/libero/vla/__init__.py

key-decisions:
  - "eval_loop.py needed `from __future__ import annotations` (not in the original plan) to support PEP 604 `int | None` syntax on this project's local Python 3.9 libero conda env -- Colab runs 3.10+, where this is a no-op."

patterns-established:
  - "Seed plumbing pattern: optional seed/episode_seeds kwargs, purely additive, byte-identical behavior when omitted -- reusable for any future eval-loop determinism requirement."
  - "Fine-tuned-adapter-backend pattern: subclass the zero-shot backend, call super().__init__() first, then layer on adapter-specific merge/stats/re-activation steps -- keeps zero-shot and fine-tuned code paths from diverging."

requirements-completed: []  # TUNE-03/TUNE-04 NOT yet complete -- Task 4's blocking human-verify checkpoint (package legitimacy re-confirmation) has not resolved, and the plan's <human-check> real-Colab run is still pending per its own verification section.

coverage:
  - id: D1
    description: "eval_loop.py's run_episode/run_suite gain optional seed/episode_seeds parameters, applied fresh per-episode inside the loop, byte-identical when omitted"
    requirement: "TUNE-03"
    verification:
      - kind: unit
        ref: "LIBERO/libero/libero/vla/test_eval_loop.py::test_run_episode_calls_env_seed_before_reset"
        status: pass
      - kind: unit
        ref: "LIBERO/libero/libero/vla/test_eval_loop.py::test_run_episode_without_seed_never_calls_env_seed"
        status: pass
      - kind: unit
        ref: "LIBERO/libero/libero/vla/test_eval_loop.py::test_run_suite_applies_seeds_identically_across_two_runs"
        status: pass
    human_judgment: false
  - id: D2
    description: "FinetunedOFTBackend reloads a HF Hub LoRA adapter through OFTBackend's proven loading path, merges via peft, re-applies dual-image mode, overlays the adapter's own dataset_statistics.json"
    requirement: "TUNE-03"
    verification:
      - kind: unit
        ref: "LIBERO/libero/libero/vla/test_adapter_backend.py::test_init_merges_adapter_and_reapplies_dual_image_mode"
        status: pass
      - kind: unit
        ref: "LIBERO/libero/libero/vla/test_adapter_backend.py::test_predict_inherited_from_oft_backend_returns_8x7_chunk"
        status: pass
    human_judgment: false
  - id: D3
    description: "06b-eval.ipynb runs before/after benchmarks under an identical seeded protocol and logs both to the shared soarm-oft-finetune-eval WandB project"
    requirement: "TUNE-03, TUNE-04"
    verification:
      - kind: other
        ref: "python -c notebook JSON-validity + expected-cell-content check (plan's <verify> automated command)"
        status: pass
    human_judgment: true
    rationale: "Requires a real Colab A100 GPU run (this project has no local GPU) to confirm the benchmark actually executes and both tables land in WandB -- source-level checks alone cannot prove the live run works, per this plan's <human-check> section."
  - id: D4
    description: "Task 4: human re-verifies peft==0.20.0 legitimacy for this notebook's separate Colab kernel install"
    requirement: "TUNE-03, TUNE-04"
    verification: []
    human_judgment: true
    rationale: "Blocking checkpoint:human-verify with gate=blocking-human -- never auto-approvable regardless of workflow.auto_advance; requires explicit human confirmation of pip show peft's HuggingFace origin on this notebook's kernel."

# Metrics
duration: 45min
completed: 2026-08-20
status: checkpoint-pending
---

# Phase 6 Plan 03: Before/After Fine-Tuning Benchmark (Tasks 1-3) Summary

**Seed-plumbed eval_loop.py + FinetunedOFTBackend (HF Hub LoRA adapter reload via peft) + 06b-eval.ipynb before/after seeded benchmark notebook — Task 4's blocking package-legitimacy checkpoint is pending human action.**

## Performance

- **Duration:** ~45 min (Tasks 1-3)
- **Started:** 2026-08-20T17:39:00Z
- **Completed (Tasks 1-3):** 2026-08-20T17:52:57Z
- **Tasks:** 3 of 4 completed (Task 4 is a blocking human-verify checkpoint, not yet resolved)
- **Files modified:** 6 (3 created, 3 modified)

## Accomplishments
- `eval_loop.py`'s `run_episode`/`run_suite` gained optional `seed`/`episode_seeds` parameters, applied freshly inside the per-episode loop (not hoisted before it) — the exact correctness requirement from Pitfall 7, proven via a two-independent-runs determinism test.
- `adapter_backend.py`'s `FinetunedOFTBackend(OFTBackend)` reuses `OFTBackend`'s entire loading path via `super().__init__()`, merges a HF-Hub-downloaded PEFT adapter via `merge_and_unload()`, re-applies `set_num_images_in_input(2)` post-merge, and overlays the adapter's OWN `dataset_statistics.json` (not the base checkpoint's or Phase 4's) — `predict()` inherited verbatim.
- `06b-eval.ipynb` created as a separate Colab kernel (D-12): downloads the adapter, re-runs the full zero-shot baseline (D-09), a Pitfall-6 stats-divergence check, the fine-tuned benchmark under the byte-identical seeded protocol (D-08), and a WandB eval-results wrapper logging a before/after table + aggregate scalars to the shared `soarm-oft-finetune-eval` project.

## Task Commits

Each task was committed atomically:

1. **Task 1: eval_loop.py — per-episode seed plumbing** - `e05f9d8` (feat)
2. **Task 2: adapter_backend.py — FinetunedOFTBackend** - `b69767a` (feat)
3. **Task 3: 06b-eval.ipynb — before/after seeded benchmark notebook** - `e1ad20a` (feat)

**Plan metadata:** this SUMMARY.md commit (docs, worktree-local; STATE.md/ROADMAP.md updates deferred to the orchestrator per parallel-worktree execution)

Task 4 (blocking `checkpoint:human-verify`, `gate="blocking-human"`) has NOT been committed — it requires explicit human confirmation and cannot be auto-approved by this executor regardless of `workflow.auto_advance`.

## Files Created/Modified
- `LIBERO/libero/libero/vla/eval_loop.py` - `run_episode` gains `seed: int | None = None`; `run_suite` gains `episode_seeds: list | None = None`; both purely additive
- `LIBERO/libero/libero/vla/test_eval_loop.py` - `MockEnv.seed()` recorder + 3 new tests (8/8 total passing)
- `LIBERO/libero/libero/vla/adapter_backend.py` - new `FinetunedOFTBackend(OFTBackend)` class
- `LIBERO/libero/libero/vla/test_adapter_backend.py` - new fake-injection test suite (2/2 passing)
- `LIBERO/libero/libero/vla/__init__.py` - third guarded try/except import block for `FinetunedOFTBackend`
- `LIBERO/notebooks/06b-eval.ipynb` - new eval notebook (bootstrap, adapter download, seeded before/after `run_suite`, Pitfall-6 divergence check, WandB eval-results wrapper)

## Decisions Made
- Added `from __future__ import annotations` to `eval_loop.py` — not in the original plan's action spec, but required for the plan's literal `seed: int | None = None` / `episode_seeds: list | None = None` PEP 604 syntax to import successfully on this project's local `libero` conda env, which runs Python 3.9 (Colab runs 3.10+, where this deferred-annotation-evaluation is a no-op). Without it, `import libero.vla` raised `TypeError: unsupported operand type(s) for |: 'type' and 'NoneType'` at collection time, blocking every local test in the `vla` package.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Added `from __future__ import annotations` to eval_loop.py**
- **Found during:** Task 1 (running the plan's own verify command)
- **Issue:** Plan's literal `seed: int | None = None` signature spec, taken verbatim, is a runtime `TypeError` on Python 3.9 (this project's local `libero` conda env) because PEP 604 `|`-union syntax is not runtime-evaluable pre-3.10 without deferred annotations.
- **Fix:** Added `from __future__ import annotations` at the top of `eval_loop.py`, deferring all annotation evaluation to strings. No runtime behavior change on Colab (3.10+), where this is already implicit via a different PEP.
- **Files modified:** `LIBERO/libero/libero/vla/eval_loop.py`
- **Verification:** `conda run -n libero pytest LIBERO/libero/libero/vla -x -q` — 17/17 passing after the fix (later 19/19 after Task 2's additions)
- **Committed in:** `e05f9d8` (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking, Rule 3)
**Impact on plan:** Necessary for the plan's own literal signature spec to be importable in this project's local test environment. No scope creep — purely additive, zero behavior change to the plan's design.

## Checkpoint Approvals

**Task 4: Human re-verifies peft package legitimacy for this notebook's separate Colab kernel install** (`checkpoint:human-verify`, `gate="blocking-human"`, non-auto-approvable)

- **What was verified:** `peft==0.20.0` -- the same package/version already fully investigated and verified legitimate in Plan 06-02 Task 3 -- re-confirmed for `06b-eval.ipynb`'s independent Colab kernel install site (D-12: eval runs in a separate kernel from training, so the install happens twice).
- **How-to-verify steps (per plan):** confirm same pinned version as Plan 06-02's verification; after install, `pip show peft` Home-page/Author fields reference `github.com/huggingface/peft`.
- **Resolution:** User responded **"approved"**, confirming the same HuggingFace origin holds for this notebook's independent kernel install site. Package Legitimacy Gate satisfied for both notebooks' separate `peft==0.20.0` install sites (T-06-03-SC closed).
- **Resolved:** 2026-08-22

## Issues Encountered
None beyond the deviation above.

## User Setup Required
None yet completed — Plan 06-03's `user_setup` entries (Hugging Face Hub read access, Weights & Biases `WANDB_API_KEY`) are consumed on Colab when `06b-eval.ipynb` is actually run, which is gated behind Task 4's checkpoint below.

## Next Phase Readiness

**BLOCKED on Task 4 — blocking `checkpoint:human-verify` (`gate="blocking-human"`), never auto-approvable:**

Task 4 requires a human to re-confirm `peft==0.20.0`'s legitimacy (same package/version already verified in Plan 06-02 Task 3, now re-installed in this notebook's separate Colab kernel per D-12). This blocks:
- Running `06b-eval.ipynb`'s Block A install cell (`pip install peft==0.20.0 --no-deps`)
- The plan's `<human-check>` real-Colab before/after benchmark run (TUNE-03/TUNE-04 final verification)
- Marking `TUNE-03`/`TUNE-04` complete in REQUIREMENTS.md

Once Task 4 resolves (human confirms `pip show peft`'s `github.com/huggingface/peft` origin on the eval kernel) and the `<human-check>` Colab run completes (before/after tables + aggregate scalars visible in `https://wandb.ai/<entity>/soarm-oft-finetune-eval`, Pitfall-6 divergence-check result visually confirmed), this plan is fully complete.

All local, no-GPU-dependent work (Tasks 1-3) is done and merged: seed plumbing, `FinetunedOFTBackend`, and the notebook itself are ready for that Colab run — nothing further to implement locally.

---
*Phase: 06-fine-tuning-evaluation*
*Completed (Tasks 1-3): 2026-08-20*
