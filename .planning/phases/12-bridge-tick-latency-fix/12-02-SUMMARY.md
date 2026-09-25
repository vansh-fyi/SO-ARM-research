---
phase: 12-bridge-tick-latency-fix
plan: 02
subsystem: infra
tags: [python, pytest, time.monotonic, io-logger, latency, vla-bridge]

# Dependency graph
requires:
  - phase: 11-vla-hardware-connection
    provides: run_vla_episode.py's run_episode() control loop and io_logger.py's write_step() latency_ms parameter (previously always hardcoded to {0,0})
provides:
  - Real time.monotonic()-derived latency_ms measurements (observation_to_action, action_to_execution) written to every episode.jsonl step record
  - Re-confirmation that DEBT-02 (gripper docstring) and DEBT-03 (mesh asset tracking) remain resolved as of this session
affects: [phase-13-camera-device-resolution, future episode-analysis tooling reading episode.jsonl's latency_ms field]

# Tech tracking
tech-stack:
  added: []
  patterns: [time.monotonic() bracketing around distinct control-loop phases (observation-to-action, action-to-execution), fed straight into an existing logger parameter with no signature change]

key-files:
  created: []
  modified:
    - control/run_vla_episode.py
    - control/test_run_vla_episode.py
    - .planning/REQUIREMENTS.md

key-decisions:
  - "Bracket t0 before get_action(), t1 immediately after get_action() returns, t2 after the send_action try/except block completes (whether it succeeded or hit its except branch) — matches 12-PATTERNS.md's exact call-site mapping"
  - "DEBT-02 and DEBT-03 re-confirmed resolved with no code edits, per CONTEXT.md D-03/D-04 — re-verification found no discrepancy from what REQUIREMENTS.md already states"

patterns-established:
  - "Pattern: instrument an existing control loop with time.monotonic() deltas by bracketing only the two candidate call sites, without touching exception handling or termination-reason logic"

requirements-completed: [LATENCY-03, DEBT-02, DEBT-03]

coverage:
  - id: D1
    description: "run_episode() writes real time.monotonic()-derived observation_to_action and action_to_execution millisecond deltas to io_logger.write_step()'s latency_ms argument, replacing the previous hardcoded {0,0}"
    requirement: "LATENCY-03"
    verification:
      - kind: unit
        ref: "control/test_run_vla_episode.py#test_latency_ms_reflects_real_monotonic_deltas_under_controlled_clock"
        status: pass
      - kind: unit
        ref: "control/test_run_vla_episode.py (full file, 18 tests)"
        status: pass
    human_judgment: false
  - id: D2
    description: "DEBT-02 (gripper format_action docstring) and DEBT-03 (So-101/coppelia mesh asset tracking) re-confirmed already resolved, no code changes needed"
    verification:
      - kind: other
        ref: "grep -c 'opens and -1 closes' LIBERO/libero/libero/envs/grippers/soarm_gripper.py == 1"
        status: pass
      - kind: other
        ref: "git status --porcelain So-101 == empty"
        status: pass
    human_judgment: false

duration: ~10min
completed: 2026-09-25
status: complete
---

# Phase 12 Plan 02: Real latency_ms Instrumentation + DEBT-02/03 Re-confirmation Summary

**`run_episode()`'s per-tick `latency_ms` now carries real `time.monotonic()`-derived millisecond deltas instead of a hardcoded `{0,0}`, proven exact under a controlled clock; DEBT-02/DEBT-03 re-confirmed already resolved with no code changes.**

## Performance

- **Duration:** ~10 min
- **Started:** 2026-09-25 (approx, same session as Task 1 commit)
- **Completed:** 2026-09-25T07:37:36Z
- **Tasks:** 2 completed (Task 2 was verification-only, zero file changes)
- **Files modified:** 3 (`control/run_vla_episode.py`, `control/test_run_vla_episode.py`, `.planning/REQUIREMENTS.md`)

## Accomplishments
- Closed the `latency_ms` logging gap called out in `control/vla_bridge/FINDINGS.md` §2 — `episode.jsonl` step records now carry real measured round-trip timing instead of an always-zero placeholder
- Added a deterministic test proving the exact millisecond values under a monkeypatched, controlled `time.monotonic()` sequence (not just "non-zero")
- Re-confirmed DEBT-02 and DEBT-03 remain resolved (docstring correct, `So-101/` fully tracked, no `coppelia/` mesh directory) with matching git/grep evidence — no invented code changes

## Task Commits

Each task was committed atomically:

1. **Task 1: Instrument run_episode() with real time.monotonic() latency deltas (LATENCY-03)** - `b1e4304` (feat)
2. **Task 2: Re-confirm DEBT-02 and DEBT-03 are already resolved (no code change)** - no commit (verification-only, zero file changes, per plan's explicit "no files are created or modified by this task" acceptance criterion)

**Plan metadata:** committed alongside this SUMMARY.md (see final metadata commit)

## Files Created/Modified
- `control/run_vla_episode.py` - `run_episode()`'s per-tick loop now captures `t0`/`t1`/`t2` via `time.monotonic()` around `action_source.get_action()` and the `robot.send_action()` try/except block, feeding real millisecond deltas into `io_logger.write_step(latency_ms=...)`
- `control/test_run_vla_episode.py` - added `test_latency_ms_reflects_real_monotonic_deltas_under_controlled_clock`, monkeypatching `run_vla_episode.time.monotonic` to a controlled `[0.0, 0.1, 0.25]` sequence and asserting the written `latency_ms` dict equals `{100.0, 150.0}` exactly
- `.planning/REQUIREMENTS.md` - marked `LATENCY-03` complete (checkbox + traceability table row), with a one-line note pointing to this plan; `DEBT-02`/`DEBT-03` were already marked `[X]` Not applicable from plan-phase, unchanged by this plan

## Decisions Made
- Bracket points chosen exactly per `12-PATTERNS.md`'s call-site mapping: `t0` before `get_action()`, `t1` immediately after it returns (before `validate_action()`), `t2` after the `send_action()` try/except block completes regardless of outcome — no changes to control flow, exception handling, or termination-reason logic
- DEBT-02/DEBT-03: re-verified fresh against CONTEXT.md's D-03/D-04 claims (rather than trusting them blindly) — found both claims accurate, no discrepancy, so no edit was made per the plan's explicit "if either re-confirmation surfaces a genuine discrepancy... flag it" escape hatch (not triggered)

## Deviations from Plan

None - plan executed exactly as written. Task 2 made zero file changes as its acceptance criteria required (verification-only); `.planning/REQUIREMENTS.md`'s `LATENCY-03` checkbox update is standard plan-completion bookkeeping (marking a requirement this plan's frontmatter declares as `requirements: [LATENCY-03, DEBT-02, DEBT-03]`), not a deviation — DEBT-02/DEBT-03 were already `[X]` before this plan started, matching the plan's expectation that "REQUIREMENTS.md already reflects both as Not applicable and needs no further edit from this task."

## Issues Encountered

`gsd-tools` CLI (`gsd-tools.cjs`) was not resolvable on this worktree's filesystem or PATH, so `REQUIREMENTS.md`'s `LATENCY-03` checkbox/traceability-row update was applied via a direct `Edit` matching the existing DEBT-02/DEBT-03 formatting convention, rather than via `gsd_run query requirements.mark-complete`. No functional difference in the resulting file content.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- `latency_ms` is now real per-tick timing data; future episode analysis (Phase 13+, and eventual MLLM chunk-size tuning per PROJECT.md's v2.1 Active scope) can read genuine round-trip numbers from `episode.jsonl` instead of reconstructing them from raw timestamps
- DEBT-02/DEBT-03 are fully closed out (re-confirmed, not silently dropped) — REQUIREMENTS.md traceability stays accurate
- LATENCY-01/02/04 (observation-gate fix, `client.latest_action` update, in-flight-request guard) are out of this plan's scope — tracked separately (see `.planning/phases/12-bridge-tick-latency-fix/` sibling plan, if present in this wave)

## Self-Check: PASSED

- FOUND: control/run_vla_episode.py
- FOUND: control/test_run_vla_episode.py
- FOUND: .planning/phases/12-bridge-tick-latency-fix/12-02-SUMMARY.md
- FOUND: commit b1e4304

---
*Phase: 12-bridge-tick-latency-fix*
*Completed: 2026-09-25*
