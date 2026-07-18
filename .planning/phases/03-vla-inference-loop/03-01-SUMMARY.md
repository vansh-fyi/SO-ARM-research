---
phase: 03-vla-inference-loop
plan: 01
subsystem: vla-inference
tags: [openvla-oft, protocol, typing, libero, pytest, mock-testing, video-utils]

# Dependency graph
requires:
  - phase: 02-soarm-robot-integration
    provides: "Soarm101 robot registration, 3 frozen libero_spatial BDDL tasks, tuned eye_in_hand camera producing the robot0_eye_in_hand_image obs key"
  - phase: 01-colab-environment-setup
    provides: "Confirmed OpenVLA-OFT loading pattern (prismatic guard, bf16 load, norm_stats overlay, unnorm_key resolution, predict_action tuple unpack, (8,7) action chunk shape)"
provides:
  - "LIBERO/libero/libero/vla/ package: VLABackend Protocol interface, OFTBackend implementation, eval_loop helper (run_episode/run_suite/print_episode_result)"
  - "Reusable predict(images, language) -> action contract any VLA backend implements"
  - "Open-loop chunk-replay + per-step success-polling + per-episode video-saving eval loop, reused by Phase 3 Plan 02/03 notebooks and importable by Phase 4/6"
affects: ["03-vla-inference-loop plan 02 (OFT eval notebook)", "03-vla-inference-loop plan 03 (pi0 backend)", "phase 04 dataset collection", "phase 06 fine-tuning"]

# Tech tracking
tech-stack:
  added: [imageio, imageio-ffmpeg, pyyaml (local dev-env only, not project deps)]
  patterns: ["typing.Protocol + @runtime_checkable for lightweight backend contracts (no ABC/ABCMeta)", "graceful ImportError degradation for Colab-only GPU deps in package __init__", "reuse existing VideoWriter/check_success rather than reimplementing"]

key-files:
  created:
    - LIBERO/libero/libero/vla/__init__.py
    - LIBERO/libero/libero/vla/interface.py
    - LIBERO/libero/libero/vla/oft_backend.py
    - LIBERO/libero/libero/vla/eval_loop.py
    - LIBERO/libero/libero/vla/test_eval_loop.py
  modified: []

key-decisions:
  - "D-03 resolved: OFT's 8-step action chunk is replayed fully open-loop before re-inference (RESEARCH.md's confirmed OFT reference default)"
  - "D-04 resolved: shared VLA interface lives at LIBERO/libero/libero/vla/ (not explorations/) so Phase 4/6 can import it directly"
  - "D-12 implemented: check_success (via env.step()'s done flag) polled every single env.step() call, not just after a full chunk — success stops the episode mid-chunk"
  - "D-13 resolved: max_steps defaults to 600, this project's own LIBERO/libero/configs/eval/default.yaml value"
  - "D-15 resolved: max_steps timeout is a hard binary FAIL, no near-miss/partial-credit diagnostic added"
  - "vla/__init__.py's OFTBackend import degrades to None on ImportError so the package (and eval_loop's mock-based test suite) still imports without torch installed — a Colab-only GPU dependency this project has no local fallback for"

patterns-established:
  - "VLABackend as a runtime_checkable typing.Protocol — the lightest-weight formal contract matching this codebase's existing small-methods-no-ABC style (on_the_ground_panda.py)"
  - "eval_loop.py imports VideoWriter via package-relative import (from ..utils.video_utils) rather than the LIBERO.-prefixed absolute import lifelong/metric.py uses, since eval_loop.py lives inside the libero.libero package itself"

requirements-completed: [VLA-01, VLA-02, VLA-03]

coverage:
  - id: D1
    description: "VLABackend Protocol interface with predict(images, language) -> np.ndarray signature, runtime_checkable, satisfied by a plain duck-typed class"
    requirement: "VLA-01"
    verification:
      - kind: unit
        ref: "manual python3 -c isinstance() check against a duck-typed Fake class — see Task 1 acceptance criteria; not captured in a checked-in pytest file"
        status: pass
    human_judgment: false
  - id: D2
    description: "OFTBackend implementing VLABackend: replicates Phase 1's confirmed prismatic-guard + bf16 load + norm_stats overlay + unnorm_key resolution; predict() unpacks predict_action's tuple and returns the full (8,7) action chunk"
    requirement: "VLA-01"
    verification:
      - kind: other
        ref: "ast.parse syntax check + grep-based acceptance criteria (libero_spatial_no_noops fallback, predict_action tuple unpack, torch_dtype=torch.bfloat16, single OFTBackend class) — all pass locally"
        status: pass
    human_judgment: true
    rationale: "Actual model load + real (8,7) action shape from a live forward pass cannot be verified locally — this project has no local GPU and torch is not installed in the local dev environment (Colab-only dependency). GPU-backed behavioral verification happens in Plan 02's Notebook A VLA-01 cell, which a human must review on Colab."
  - id: D3
    description: "eval_loop.run_episode: open-loop 8-step chunk replay, per-step check_success polling with mid-chunk early stop, per-episode VideoWriter recording (single_video=True), hard binary FAIL on max_steps=600 timeout"
    requirement: "VLA-02"
    verification:
      - kind: unit
        ref: "LIBERO/libero/libero/vla/test_eval_loop.py::test_open_loop_chunk_replay_with_mid_chunk_early_stop"
        status: pass
      - kind: unit
        ref: "LIBERO/libero/libero/vla/test_eval_loop.py::test_video_writer_append_obs_called_per_step"
        status: pass
    human_judgment: false
  - id: D4
    description: "print_episode_result (PASS/FAIL arrow-notation line) and run_suite (multi-task/episode orchestration + aggregated success-rate summary table)"
    requirement: "VLA-03"
    verification:
      - kind: unit
        ref: "LIBERO/libero/libero/vla/test_eval_loop.py::test_run_episode_return_dict_and_print"
        status: pass
      - kind: unit
        ref: "LIBERO/libero/libero/vla/test_eval_loop.py::test_run_suite_multi_task_multi_episode_summary"
        status: pass
    human_judgment: false

# Metrics
duration: 45min
completed: 2026-07-18
status: complete
---

# Phase 3 Plan 01: Shared VLA Interface + OFT Backend + Eval Loop Summary

**LIBERO/libero/libero/vla/ package: a runtime_checkable VLABackend Protocol, an OFTBackend replicating Phase 1's confirmed OpenVLA-OFT loading pattern, and a mock-tested eval_loop (open-loop chunk replay, per-step success polling, per-episode video, PASS/FAIL summary table) that Notebook A/B will call in Wave 2**

## Performance

- **Duration:** 45 min
- **Started:** 2026-07-18T18:00:00Z (approx.)
- **Completed:** 2026-07-18T18:25:00Z (approx.)
- **Tasks:** 3
- **Files modified:** 5 (4 new package modules + 1 new test file)

## Accomplishments
- `VLABackend` typing.Protocol (`@runtime_checkable`) defining the shared `predict(images: dict[str, Image], language: str) -> np.ndarray` contract (D-01, D-02) any backend satisfies without inheriting from a base class
- `OFTBackend` implementing that contract: prismatic-import guard with clone fallback, bf16 model load (D-05), `dataset_statistics.json` norm_stats overlay, `unnorm_key` resolution with the confirmed `libero_spatial_no_noops` fallback, and `predict()` returning the full `(8, 7)` OFT action chunk (D-03: eval loop owns replay, not this backend)
- `eval_loop.run_episode()`: resets the env itself, replays every row of the returned action chunk open-loop, polls `done` after every single `env.step()` call (not just after a full chunk) so success stops the episode immediately mid-chunk (D-12), records every step via the existing `VideoWriter` in `single_video=True` mode (D-11), and returns `{"success", "steps", "video_path"}`
- `eval_loop.print_episode_result()` and `eval_loop.run_suite()`: arrow-notation PASS/FAIL printing plus a markdown-style aggregated success-rate summary table across tasks/episodes (D-14)
- `test_eval_loop.py`: a mock env/backend/VideoWriter pytest suite (4/4 passing locally) proving the control-flow contracts above without needing a real VLA model, GPU, or MuJoCo env

## Task Commits

Each task was committed atomically:

1. **Task 1: Shared VLABackend interface + package scaffold** - `267395e` (feat)
2. **Task 2: OFTBackend implementing VLABackend (VLA-01)** - `41d6e82` (feat)
3. **Task 3: eval_loop helper — success polling, chunk replay, video, PASS/FAIL (VLA-02, VLA-03)** - `a6e5e97` (feat), followed by `5745543` (fix — see Deviations)

## Files Created/Modified
- `LIBERO/libero/libero/vla/interface.py` - `VLABackend` runtime_checkable Protocol, single `predict(images, language) -> np.ndarray` method
- `LIBERO/libero/libero/vla/oft_backend.py` - `OFTBackend` class: `CHECKPOINT`/`PRISMATIC_REPO` constants, prismatic-import guard, bf16 load, norm_stats overlay, `predict()` returning the `(8,7)` chunk
- `LIBERO/libero/libero/vla/eval_loop.py` - `run_episode()`, `print_episode_result()`, `run_suite()`, `MAX_STEPS_DEFAULT = 600`
- `LIBERO/libero/libero/vla/test_eval_loop.py` - mock-based pytest suite (`MockEnv`, `MockBackend`, 4 test functions)
- `LIBERO/libero/libero/vla/__init__.py` - re-exports `VLABackend`, `OFTBackend` (ImportError-safe), `run_episode`, `run_suite`, `print_episode_result`

## Decisions Made
- D-03 (Claude's discretion, research-informed): full open-loop replay of all 8 OFT action-chunk steps before re-inference — matches the OFT reference implementation's own default and avoids ~8x redundant inference cost
- D-04 (Claude's discretion): the shared interface lives at `LIBERO/libero/libero/vla/`, inside the vendored LIBERO fork, not `explorations/` — so Phase 4 (dataset collection) and Phase 6 (fine-tuning) can import it directly per CONTEXT.md's Integration Points
- D-12: `done` is read from `env.step()`'s return (which already wraps `_check_success()` internally) rather than a separate `check_success()` poll call — avoids a redundant call while still satisfying "poll every step, stop early"
- D-13: `max_steps` defaults to `600`, this project's own `LIBERO/libero/configs/eval/default.yaml` value, not an arbitrary cutoff
- D-15: a `max_steps` timeout is a hard binary FAIL; no partial-credit/near-miss diagnostic was added, keeping the MVP eval loop simple per the plan's explicit scope boundary
- `vla/__init__.py`'s `OFTBackend` import is wrapped in `try/except ImportError` (degrading to `None` locally) rather than making the whole package fail to import when `torch` isn't installed — this is the only way the local, no-GPU `test_eval_loop.py` suite (explicitly designed by the plan to run without GPU/network) can import the package at all, since Python always executes a package's `__init__.py` before any of its submodules

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Worktree missing the vendored LIBERO/ fork entirely**
- **Found during:** Start of Task 1
- **Issue:** `LIBERO/` is gitignored project-wide (vendored dependency, per CLAUDE.md) and this git worktree only checks out tracked files — so none of the existing LIBERO code (robots/, video_utils.py, metric.py, configs/eval/default.yaml, notebooks/) that Tasks 1-3's `read_first` sections require was present on disk.
- **Fix:** Copied the small, needed subdirectories (`envs`, `utils`, `benchmark`, `bddl_files`, `lifelong`, `configs`, `notebooks`, `setup.py`, `requirements.txt` — 6.6MB total) from the main repo checkout into the worktree, explicitly excluding the large gitignored `assets/` (420MB) and `init_files/` (13MB) directories not needed for this plan's Python-glue-code scope.
- **Files modified:** None (LIBERO/ itself stays gitignored; only the new `vla/` package files are force-added, matching prior-phase precedent of `git add -f` on specific LIBERO glue files despite the wholesale gitignore).
- **Verification:** All `read_first` reference files (env_wrapper.py, video_utils.py, metric.py, robots/__init__.py, configs/eval/default.yaml, Notebook 01) were successfully read and used to derive the exact confirmed patterns.
- **Committed in:** N/A (LIBERO/ vendored tree stays gitignored; not part of any task commit)

**2. [Rule 1 - Bug] vla/__init__.py's OFTBackend import failed locally (torch not installed)**
- **Found during:** Task 2 local verification
- **Issue:** `torch` is a Colab-only GPU dependency per this project's own established pattern (03-RESEARCH.md Environment Availability, CLAUDE.md) — not installed in the local `libero` conda dev environment. A plain `from .oft_backend import OFTBackend` in `__init__.py` made the entire `vla` package fail to import locally, which would also break Task 3's local, no-GPU pytest suite (since importing any submodule always executes the package's `__init__.py` first).
- **Fix:** Wrapped the `OFTBackend` import in `try/except ImportError`, degrading to `OFTBackend = None` when torch is unavailable. On Colab (torch installed) the import succeeds normally and `OFTBackend` resolves to the real class.
- **Files modified:** `LIBERO/libero/libero/vla/__init__.py`
- **Verification:** `sys.path.insert(0, 'LIBERO'); from libero.libero.vla import VLABackend, OFTBackend, run_episode, run_suite, print_episode_result` succeeds locally (OFTBackend resolves to `None`); the plan's own Task 1 acceptance-criteria command still passes unchanged.
- **Committed in:** `a6e5e97` / `5745543` (see deviation 4 below for why this landed in a follow-up commit)

**3. [Rule 3 - Blocking] Missing imageio/imageio-ffmpeg/pyyaml in local dev environment**
- **Found during:** Task 3 local pytest verification
- **Issue:** `LIBERO/libero/libero/utils/video_utils.py` (existing, reused, not modified by this plan) imports `imageio`, which was not installed in the local `libero` conda env; `imageio`'s real video-encoding backend additionally needed `imageio-ffmpeg` for the two tests that exercise the real `VideoWriter` (not the mocked one); and `LIBERO/libero/libero/__init__.py` (existing) imports `pyyaml`, also missing locally.
- **Fix:** `pip install imageio imageio-ffmpeg` into the local `libero` conda environment (pyyaml was already present once re-checked). These are standard, well-known packages already relied upon by existing, unmodified project code — installing them is an environment-completeness fix, not new plan-scope functionality.
- **Files modified:** None (local conda env only; no requirements.txt changes made, since these are Colab-baked/pre-existing project dependencies, not new pins this plan introduces)
- **Verification:** `python3 -m pytest LIBERO/libero/libero/vla/test_eval_loop.py -q` — 4/4 pass after install.
- **Committed in:** N/A (local environment setup, not a code change)

**4. [Rule 1 - Bug] Task 3 commit initially dropped the __init__.py update**
- **Found during:** Post-commit self-check after Task 3
- **Issue:** `git add -f` staged `libero/libero/libero/vla/__init__.py` correctly (confirmed via `git status --short` in the same command), but the subsequent `git commit` in commit `a6e5e97` only actually captured `eval_loop.py`/`test_eval_loop.py` — the `__init__.py` change (adding `run_episode`/`run_suite`/`print_episode_result` exports and the `OFTBackend` ImportError guard) was silently absent from that commit's tree, discovered via a post-commit `git diff` showing an unexpected working-tree delta against `HEAD` on a lowercase-cased path variant (`libero/...` vs `LIBERO/...` — this project's filesystem is case-insensitive, and `LIBERO/` and a sibling `libero/` are literally the same on-disk directory, which appears to have caused `git`'s add/commit sequence to desynchronize for that one file in that specific command batch).
- **Fix:** Re-staged and committed `libero/libero/libero/vla/__init__.py` in a follow-up commit.
- **Files modified:** `LIBERO/libero/libero/vla/__init__.py`
- **Verification:** `git show HEAD:libero/libero/libero/vla/__init__.py` now matches the on-disk file; `git status --short` clean after `git update-index --refresh`; full public-API import (`from libero.libero.vla import VLABackend, OFTBackend, run_episode, run_suite, print_episode_result`) re-verified successfully post-fix.
- **Committed in:** `5745543`

---

**Total deviations:** 4 auto-fixed (1 blocking-environment, 1 bug, 1 blocking-environment, 1 bug/git-hygiene)
**Impact on plan:** All four were necessary to make the plan's own verification criteria achievable in this worktree's environment; none change the plan's scope, design, or the shipped `vla/` package's public API. No scope creep.

## Issues Encountered
- **pytest import-path ambiguity for `test_eval_loop.py` (resolved without a deviation to plan code):** `python3 -m pytest LIBERO/libero/libero/vla/test_eval_loop.py -q` (the plan's mandated verify command) initially failed with `ModuleNotFoundError: No module named 'libero.libero'` when the test imported `from libero.libero.vla.eval_loop import ...`. Root cause: pytest's default rootdir-walking import mode resolves this specific test file's own package identity as `libero.vla.test_eval_loop` (treating `LIBERO/libero/libero/` — the directory containing `__init__.py` nearest to the test file — as the top-level `libero` package for pytest's purposes), which conflicts with the `libero.libero.vla` form used when `LIBERO` itself is placed on `sys.path` (the Colab/notebook convention, confirmed still correct via Task 1's own acceptance-criteria command). Extensive attempts with `conftest.py`, `pytest.ini` (`pythonpath`, `--import-mode=importlib`), and manual `sys.path`/`sys.modules` manipulation all failed to reconcile the two forms simultaneously, because this project's filesystem is case-insensitive (`LIBERO/` and a same-named lowercase `libero/` sibling directory are literally one physical directory), which additionally causes `libero` to resolve as an ambiguous merged namespace package including the current working directory. **Resolution:** changed only `test_eval_loop.py`'s own import statements to use the `libero.vla.*` form pytest naturally resolves for this file's location — both forms point at the identical package on disk; no production code (`eval_loop.py`, `oft_backend.py`, `interface.py`, `__init__.py`) needed any change, and the plan's own Task 1 acceptance criteria (which explicitly uses `sys.path.insert(0,'LIBERO')` then `libero.libero.vla`) continues to pass unmodified.

## User Setup Required

None - no external service configuration required. `oft_backend.py`'s GPU-backed behavior (actual model load, real `(8,7)` action shape from a live forward pass) is deferred to Plan 02's Notebook A on Colab per this project's established no-local-GPU pattern — this is expected, not a gap in this plan.

## Next Phase Readiness

- `LIBERO/libero/libero/vla/` package is fully importable via both this project's established conventions: `sys.path.insert(0, 'LIBERO'); from libero.libero.vla import VLABackend, OFTBackend, run_episode, run_suite, print_episode_result` (Colab/notebook convention) and `from libero.vla.eval_loop import ...` (pytest-local convention for files under `LIBERO/libero/libero/vla/`).
- Plan 02 (Wave 2) can now build Notebook A directly on top of `OFTBackend` and `eval_loop.run_suite()` to run the full 3-task x 5-10 episode OFT eval and produce the real success-rate number for VLA-03 — no remaining glue code needed on the OFT side.
- Plan 03 (Wave 2, pi0 backend) can implement `Pi0Backend` against the same `VLABackend` Protocol and drop it into the same `eval_loop.run_episode`/`run_suite` functions unchanged, proving VLA-04's interface-swap requirement structurally (this plan's `eval_loop.py` never branches on which backend is active).
- No blockers. The one open risk carried forward (unchanged from 03-RESEARCH.md) is confirming `oft_backend.py`'s GPU-backed behavior for real on Colab in Plan 02 — this project has no local GPU, so that verification could not happen in this plan.

---
*Phase: 03-vla-inference-loop*
*Completed: 2026-07-18*
