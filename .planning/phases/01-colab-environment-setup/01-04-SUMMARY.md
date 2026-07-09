---
phase: 01-colab-environment-setup
plan: 04
subsystem: infra
tags: [colab, numpy, abi, pip, gym, transformers, jax, tensorflow, jupyter, notebook]

# Dependency graph
requires:
  - phase: 01-colab-environment-setup (plans 01-01..01-03)
    provides: 21-cell setup notebook (Block A install + Block B verification), UAT evidence identifying the numpy ABI root cause
provides:
  - Block A final numpy ABI gate cell (purge + exact-pin numpy==1.26.4 + fresh-subprocess probe of the gym/numpy.random crash path, run before restart)
  - transformers backend guards (USE_TORCH=1, USE_TF=0, USE_FLAX=0) in the EGL bootstrap cell and defensively at the top of the ENV-03 cell
  - ENV-01 numpy expected-version row, in-kernel numpy.random ABI canary, and verdict logic that fails on version mismatch/canary failure (not only pip conflicts)
affects: [phase-2-soarm-model, any phase re-running the Colab setup notebook]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Fresh-subprocess ABI probe: verify on-disk C-extension coherence from a clean interpreter before instructing a kernel restart (in-kernel imports are cached and lie)"
    - "transformers backend env guards (USE_TORCH/USE_TF/USE_FLAX) set before any transformers import to sever numpy-2.x-compiled jax/tensorflow import paths"
    - "Verification cells gate PASS on all_ok AND conflict-free, not a single signal"

key-files:
  created: []
  modified:
    - libero/notebooks/01-colab-env-setup.ipynb

key-decisions:
  - "numpy enforcement moved to a dedicated final Block A gate cell (after Step 6) instead of strengthening the Step 3 pin — Steps 4-6 pip operations can perturb numpy after Step 3"
  - "Mixed-install repair = uninstall loop + leftover sweep (numpy, numpy-*.dist-info, numpy.libs in purelib only) + clean --no-cache-dir reinstall of exactly numpy==1.26.4"
  - "ABI probe replicates the exact ENV-02 crash path (gym.spaces.Box via gym.utils.seeding -> numpy.random.mtrand) in a fresh subprocess; deliberately imports no mujoco/robosuite/libero pre-restart"
  - "Gap 2 fixed via transformers' official backend env vars, not by uninstalling Colab system jax/flax/tensorflow"
  - "Stale outputs stripped only on cells modified (Step 3, EGL bootstrap, ENV-01, ENV-03) per plan allowance — old outputs contained the misleading restart instruction and the UAT crash tracebacks"

patterns-established:
  - "Notebook JSON edits via python json.load/json.dump (indent=1, ensure_ascii=False, trailing newline), locating cells by unique leading strings, never by index"

requirements-completed: [ENV-01, ENV-02, ENV-03]

coverage:
  - id: D1
    description: "Block A final numpy ABI gate cell purges/exact-pins numpy 1.26.4 and probes the gym/numpy.random crash path in a fresh subprocess before the restart instruction (Gap 1 / ENV-02)"
    requirement: ENV-02
    verification:
      - kind: other
        ref: "python3 JSON assertion (Task 1 verify block in 01-04-PLAN.md) — 22 cells, gate cell position/content/structural region checks"
        status: pass
    human_judgment: true
    rationale: "The runtime truth (gate prints 'numpy ABI gate: PASS' on a fresh Colab A100, then ENV-02 prints PASS post-restart) requires a live Colab session the executor cannot run — see plan human-check"
  - id: D2
    description: "USE_TORCH/USE_TF/USE_FLAX backend guards set in the EGL bootstrap cell and re-set at the top of the ENV-03 cell before any transformers import (Gap 2 / ENV-03)"
    requirement: ENV-03
    verification:
      - kind: other
        ref: "python3 JSON assertion (Task 2 verify block in 01-04-PLAN.md) — guard presence and textual ordering before the prismatic importlib probe and transformers import"
        status: pass
    human_judgment: true
    rationale: "The runtime truth (ENV-03 prints PASS with a 7-D per-step action and no jax/tensorflow import crash) requires a live Colab A100 session with GPU model load"
  - id: D3
    description: "ENV-01 verifies numpy==1.26.4, runs an in-kernel numpy.random ABI canary, and fails on any version mismatch, canary failure, or our-package conflict"
    requirement: ENV-01
    verification:
      - kind: other
        ref: "python3 JSON assertion (Task 3 verify block in 01-04-PLAN.md) — numpy row regex, canary literals, all_ok+our_conflicts combined verdict line"
        status: pass
    human_judgment: true
    rationale: "ENV-01: PASS with the numpy row OK and 'numpy ABI canary: OK' must be observed on a fresh post-restart Colab kernel"

# Metrics
duration: 14min
completed: 2026-07-09
status: complete
---

# Phase 1 Plan 4: numpy ABI Gate + jax/TF Guards Summary

**Block A now ends with a purge/exact-pin/fresh-subprocess numpy ABI gate that proves the post-restart gym import chain survives, and transformers' TF/Flax backends are disabled via env guards so ENV-03 never imports Colab's numpy-2.x-compiled jax — closing both UAT blocker gaps at their shared root cause.**

## Performance

- **Duration:** ~14 min
- **Started:** 2026-07-09T17:37:30Z
- **Completed:** 2026-07-09T17:51:00Z
- **Tasks:** 3
- **Files modified:** 1

## Accomplishments

- New Block A final gate cell (notebook now 22 cells): uninstall-loop purge of numpy, leftover sweep of `numpy`/`numpy-*.dist-info`/`numpy.libs` in purelib (the direct repair for the UAT-observed mixed install), clean `numpy==1.26.4 --no-cache-dir` reinstall, then a fresh-subprocess `_ABI_PROBE` that walks the exact ENV-02 crash path (`import numpy.random` -> `rand(3)` -> `import gym` -> `gym.spaces.Box`) and prints `numpy ABI gate: PASS ... safe to restart runtime` — or raises with do-NOT-restart guidance
- Step 3 no longer instructs a premature restart (`continue with Step 4`), and the Block A header gains ordering rule 4 (gate runs last, must print PASS before restart)
- EGL bootstrap cell (first post-restart cell) sets `USE_TORCH=1`, `USE_TF=0`, `USE_FLAX=0`, preventing `transformers/utils/generic.py` from executing its module-level jax.numpy/tensorflow imports; ENV-03 defensively re-sets the guards at its top, before the prismatic importlib probe
- ENV-01 gains a `numpy: 1.26.4` expected row, an in-kernel numpy.random ABI canary, and a verdict fix — PASS now requires `all_ok and not our_conflicts`, so version mismatches and canary failures fail loudly
- Block B header markdown rewritten without stale cell-number references; documents the TF/Flax guard behavior

## Task Commits

Each task was committed atomically:

1. **Task 1: Block A final numpy ABI gate cell + Step 3 restart-hint fix** - `638a17e` (fix)
2. **Task 2: transformers backend env guards (EGL bootstrap + ENV-03) + Block B header refresh** - `89e03ef` (fix)
3. **Task 3: ENV-01 numpy row, in-kernel ABI canary, verdict gate bug fix** - `ee04e3c` (fix)

## Files Created/Modified

- `libero/notebooks/01-colab-env-setup.ipynb` - 22 cells (was 21): one new gate code cell, three edited code cells (Step 3, EGL bootstrap, ENV-01, ENV-03), two edited markdown cells (Block A + Block B headers). All established UAT facts preserved: mujoco 3.3.2 + mj_kinematics shim, /content/libero clone fallback, flash-attn uninstall fallback, torch 2.2.0 pin, ENV-02 numba stub shim, ENV-01 OUR_PACKAGES filtering and build-tag strip

## Decisions Made

- Stripped recorded UAT outputs on the four modified code cells (plan-permitted): Step 3's stale output contained the removed "restart runtime now" instruction, and ENV-03's outputs held the full crash/ultratb traceback — keeping them would contradict the fixed source
- Gate cell id `cell-9b-numpy-abi-gate` follows the existing `cell-N-name` id convention of Block A cells

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None. All three per-task automated JSON verifications and the whole-notebook structural check (`nb OK 22`) passed on first run; Tasks 1-2 assertions re-verified against the final post-Task-3 state.

## User Setup Required

None locally. Runtime confirmation requires a fresh Colab A100 session (re-run UAT via /gsd-verify-work): Block A top-to-bottom -> gate prints PASS -> restart -> Block B in order -> ENV-01/ENV-02/ENV-03 all print PASS.

## Next Phase Readiness

- Both UAT blocker gaps (Tests 3 and 4) have concrete in-notebook repairs; ready for a fresh-Colab re-verification pass
- If the gate ever prints FAIL on Colab, the diagnostic output (numpy paths + mtrand path + probe traceback) is designed to be reported directly — do not restart on FAIL

---
*Phase: 01-colab-environment-setup*
*Completed: 2026-07-09*

## Self-Check: PASSED
