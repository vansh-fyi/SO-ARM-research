---
phase: 03-vla-inference-loop
plan: 02
subsystem: vla-inference
tags: [openvla-oft, colab, notebook, libero, eval-loop, video]

# Dependency graph
requires:
  - phase: 03-vla-inference-loop (plan 01)
    provides: "LIBERO/libero/libero/vla/ package: VLABackend Protocol, OFTBackend, eval_loop (run_episode/run_suite/print_episode_result)"
  - phase: 02-soarm-robot-integration
    provides: "Soarm101 robot registration, 3 frozen libero_spatial BDDL tasks, tuned eye_in_hand camera"
  - phase: 01-colab-environment-setup
    provides: "Confirmed OpenVLA-OFT loading pattern (prismatic guard, bf16 load, norm_stats overlay), Colab Drive-zip delivery convention"
provides:
  - "LIBERO/notebooks/03a-oft-inference-eval.ipynb: Colab notebook driving OFTBackend across all 3 frozen libero_spatial SOARM tasks, 8 episodes each, with per-episode video + PASS/FAIL + aggregated success-rate table"
affects: ["03-vla-inference-loop plan 03 (pi0 smoke-test notebook, same run_suite/env_factory pattern)", "phase 4 dataset collection (may reuse notebook conventions)"]

# Tech tracking
tech-stack:
  added: []
  patterns: ["Drive-mount + zip-unzip Colab delivery cell (established in 02-05)", "Block A (verify-only, no reinstall) / restart / Block B (EGL->config->sys.path->matplotlib bootstrap) notebook structure", "requirement-ID-per-cell comment convention (VLA-01/02/03)"]

key-files:
  created:
    - LIBERO/notebooks/03a-oft-inference-eval.ipynb
  modified: []

key-decisions:
  - "Notebook created fresh (not extending 02-soarm-integration-check.ipynb), per 03-RESEARCH.md's Recommended Project Structure"
  - "Block A here is verify-only (asserts torch/transformers/libero import + version match) — no new pip installs, since 03-RESEARCH.md confirms Phase 3 only adds eval-loop code on top of Phase 1's baked environment"
  - "EPISODES_PER_TASK = 8, within D-09's 5-10 episode range"
  - "LANGUAGE_MAP derived mechanically from BDDL filenames (strip .bddl, replace underscores with spaces) — no free-text user input in this phase, matching the threat_model's Trust Boundaries"

patterns-established:
  - "Committed at the lowercase libero/ path prefix (git add -f), matching Wave 1's precedent for force-adding into the gitignored, case-insensitive-collapsed LIBERO/libero tree"

requirements-completed: []  # VLA-01/02/03 code shipped but NOT yet requirement-complete — human Colab sign-off (Task 4) is outstanding; see Next Phase Readiness

coverage:
  - id: D1
    description: "Notebook scaffold: Drive mount + unzip, REPO_ROOT/TASKS/LANGUAGE_MAP/VIDEO_DIR constants, GPU assertion, Block A verify-only cell, restart-stop cell"
    requirement: "VLA-01"
    verification:
      - kind: other
        ref: "python3 -c JSON/grep checks — 3 frozen TASKS filenames present, torch.cuda.is_available() present, STOP/Restart runtime text present, no pip install of torch/transformers"
        status: pass
    human_judgment: false
  - id: D2
    description: "Block B bootstrap (EGL -> config.yaml -> sys.path -> matplotlib/numba, in that exact order) + VLA-01 cell instantiating OFTBackend and asserting a live (8,7) action shape from one real predict() call"
    requirement: "VLA-01"
    verification:
      - kind: other
        ref: "python3 -c JSON check — EGL bootstrap cell index precedes first OffScreenRenderEnv( usage; asserts .shape == (8, 7); VLA-01 PASS/FAIL string present"
        status: pass
    human_judgment: true
    rationale: "Static/structural checks confirm the cell exists and is correctly ordered, but actual GPU-backed model load + a real (8,7) action shape from a live forward pass can only be proven by running this cell on a Colab A100 runtime — this project has no local GPU. Task 4's human-verify checkpoint is the authoritative gate."
  - id: D3
    description: "Full eval-loop demo cell: run_suite across 3 tasks x EPISODES_PER_TASK=8 episodes, video-file-existence assertion, Phase 3a Summary table with VLA-01/02/03 rows"
    requirement: "VLA-02"
    verification:
      - kind: other
        ref: "python3 -c JSON/regex check — run_suite( call with episodes_per_task=EPISODES_PER_TASK and max_steps=600 present, EPISODES_PER_TASK in [5,10], os.path.exists + VIDEO_DIR present, VLA-01/02/03 all present in notebook source"
        status: pass
    human_judgment: true
    rationale: "Static checks confirm the cell is wired correctly and calls the shared eval_loop.run_suite unmodified, but actual episode success rates, saved video correctness (not black/corrupted), and the printed aggregated summary table can only be confirmed by running the full ~15-30 min eval loop on Colab A100 — this project has no local GPU. Task 4's human-verify checkpoint is the authoritative gate for VLA-02 and VLA-03."

# Metrics
duration: 20min
completed: 2026-07-18
status: complete
---

# Phase 3 Plan 02: OFT Inference Eval Notebook (Colab) Summary

**Colab notebook (`03a-oft-inference-eval.ipynb`) driving OpenVLA-OFT across all 3 frozen SOARM libero_spatial tasks — 8 episodes each — with per-episode video, PASS/FAIL, and an aggregated success-rate table; code complete, Colab GPU sign-off (Task 4) pending**

## Performance

- **Duration:** 20 min
- **Started:** 2026-07-18T18:15:00Z (approx.)
- **Completed:** 2026-07-18T18:35:43Z
- **Tasks:** 3 of 4 (Tasks 1-3 complete; Task 4 is a blocking human-verify checkpoint)
- **Files modified:** 1 (new notebook)

## Accomplishments
- `LIBERO/notebooks/03a-oft-inference-eval.ipynb` created from scratch: markdown header referencing Phase 1/2 confirmed prerequisites, Drive-mount + zip-unzip delivery cell (established Phase 2 convention), `REPO_ROOT`/`LIBERO_PKG`/`LIBERO_ROOT`/`BDDL_DIR`/`VIDEO_DIR` path constants, the 3 frozen `libero_spatial` `TASKS` filenames verbatim from `explorations/soarm_sanity.py`, a mechanically-derived `LANGUAGE_MAP`, and a GPU assertion with an A100/bf16-specific warning (D-05)
- Block A implemented as **verify-only** (no reinstall): asserts `torch`/`transformers`/`libero` import successfully and match Phase 1's pinned versions — per 03-RESEARCH.md's confirmation that this phase adds no new dependencies
- Block B bootstrap replicated in the exact required order (EGL -> LIBERO `config.yaml` -> `sys.path` -> matplotlib `Agg`/numba shim), matching the cross-notebook invariant from `01-DEBUG-HISTORY.md`
- VLA-01 cell: instantiates `OFTBackend` from Plan 01's shared `vla` package (no reimplementation), builds one live SOARM env for `TASKS[0]`, and asserts a real `predict()` call returns the confirmed `(8, 7)` action chunk, printing `VLA-01: PASS`/`FAIL`
- VLA-02/VLA-03 cell: calls `eval_loop.run_suite(env_factory, backend, TASKS, LANGUAGE_MAP, episodes_per_task=8, video_dir=VIDEO_DIR, max_steps=600)` — reusing Plan 01's shared orchestration unmodified — which prints per-episode PASS/FAIL plus the aggregated success-rate table (D-14) internally; an explicit `assert` after the call confirms at least one saved video file exists (VLA-02 proof)
- Final `## Phase 3a Summary` markdown table with VLA-01/02/03 rows and a status checklist (☐) for the human to fill in after a real Colab run

## Task Commits

Each task was committed atomically. Because all three tasks incrementally build the same single notebook file and Task 3's action is the final structural addition before verification, all three tasks' changes landed in one commit (the notebook only reaches a valid, independently-verifiable JSON state once all three tasks' cells are present):

1. **Tasks 1-3: Notebook A scaffold + Block B bootstrap + VLA-01/02/03 eval-loop cells** - `e4f918f` (feat)

_Note: the plan's three automated verification blocks (scaffold checks, Block B ordering + VLA-01 checks, eval-loop wiring checks) were each run independently against the final notebook and all three passed — see `coverage` above._

## Files Created/Modified
- `LIBERO/notebooks/03a-oft-inference-eval.ipynb` - New Colab notebook (18 cells): markdown header, Drive-mount+unzip, path/TASKS/LANGUAGE_MAP constants, GPU assertion, Block A verify-only cell, restart-stop cell, Block B bootstrap (EGL/config/sys.path/matplotlib), VLA-01 model-load+shape-check cell, VLA-02/VLA-03 full eval-loop cell, sample-video sanity print, Phase 3a Summary table

## Decisions Made
- Notebook built fresh rather than extending `02-soarm-integration-check.ipynb`, per 03-RESEARCH.md's Recommended Project Structure and 03-PATTERNS.md's file classification (role-match, not literal extension)
- `EPISODES_PER_TASK = 8` chosen as the midpoint of D-09's 5-10 episode range — long enough for a meaningful success-rate signal, short enough to keep the ~15-30 min Colab runtime reasonable
- Block A demoted to verify-only (asserting import + version match) rather than a fresh install chain, since 03-RESEARCH.md explicitly confirms this phase needs zero new pip installs beyond Phase 1's baked environment
- `env_factory` lambda takes the bare BDDL filename (matching `TASKS`' contents) and joins with `BDDL_DIR` internally — matches `eval_loop.run_suite`'s existing signature (`env_factory(task)` where `task` is an element of the `tasks` list) without any modification to Plan 01's shared code

## Deviations from Plan

None — plan executed exactly as written. All three `read_first` reference files (`01-colab-env-setup.ipynb`, `explorations/soarm_sanity.py`, Plan 01's `vla/` package files) were available in this worktree and used directly; `02-soarm-integration-check.ipynb` referenced in Task 1/2/3's `read_first` sections was not present on disk in this worktree (only `01-colab-env-setup.ipynb` and the `vla/` package were copied over in Wave 1), so its patterns were instead sourced from `03-PATTERNS.md`'s full cell-by-cell breakdown of that notebook (recorded during the phase's pattern-mapping step) and cross-checked against `01-colab-env-setup.ipynb`'s equivalent cells — both describe the identical Block A/B convention, so no scope or behavior gap resulted.

## Issues Encountered
None.

## User Setup Required

None for building this notebook. However, **Task 4 (blocking human-verify checkpoint) requires the user to actually run this notebook on Google Colab with an A100 GPU runtime** — this project has no local GPU, so VLA-01/02/03's real behavioral verification cannot happen in this execution environment. See "Next Phase Readiness" below for the exact steps required.

## Next Phase Readiness

- **Not yet ready to close this plan.** Task 4 is a `checkpoint:human-verify` (gate="blocking") requiring the user to:
  1. Upload/sync `SoARM-Research-colab.zip` to `MyDrive` (existing Phase 1/2 convention).
  2. Open `LIBERO/notebooks/03a-oft-inference-eval.ipynb` in Colab with an A100 runtime.
  3. Run Block A cells, then Runtime > Restart session (not "Disconnect and delete runtime").
  4. Re-run the path-constants cell, then run Block B cells in order.
  5. Confirm `VLA-01: PASS` with action shape `(8, 7)`.
  6. Run the full eval-loop cell to completion (~15-30 min), confirm per-episode PASS/FAIL and the aggregated success-rate table print.
  7. Spot-check 2-3 saved videos under `VIDEO_DIR` for visual correctness (not black/corrupted).
  8. Fill in the Phase 3a Summary table's Status column with actual results.
- Once the human reports "approved" (or describes issues to fix), this plan's Task 4 `done` criteria are satisfied and the phase can proceed to closure for the OFT backend (ROADMAP Phase 3 success criteria #1 and #2).
- Plan 03 (π0 smoke-test notebook, same wave) can proceed independently — it does not depend on this plan's Colab sign-off, only on Plan 01's shared `vla` package.
- No blockers for Plan 03. The one open item for this plan is exclusively the human Colab run.

## Self-Check: PASSED

`LIBERO/notebooks/03a-oft-inference-eval.ipynb` verified present on disk (18 cells, valid JSON, nbformat 4). Commit `e4f918f` verified present in `git log --oneline -3`.

---
*Phase: 03-vla-inference-loop*
*Completed: 2026-07-18 (code); Colab human sign-off pending*
