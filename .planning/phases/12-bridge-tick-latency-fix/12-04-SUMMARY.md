---
phase: 12-bridge-tick-latency-fix
plan: 04
subsystem: infra
tags: [robot-client, watchdog, lerobot, grpc-bridge, deadlock-recovery]

# Dependency graph
requires:
  - phase: 12-01
    provides: pop_validated_action() flags contract (stale-observation / no-action-available flag shapes), BridgeActionSource.get_action() control flow
provides:
  - "force_bridge_recovery(client): drains client.action_queue and sets client.must_go directly"
  - "_is_stale_or_empty(flags): detects held/not-genuinely-fresh pop_validated_action() results"
  - "BridgeActionSource consecutive-stale-count watchdog wired into get_action()"
affects: [12-05, 12-06, live-hardware-episode-runs]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Consecutive-count watchdog forcing vendored-library recovery state directly, bounded and independent of the vendored library's own internal convergence timing"

key-files:
  created: []
  modified:
    - control/vla_bridge/robot_client.py
    - control/test_robot_client.py

key-decisions:
  - "STALE_WATCHDOG_CONSECUTIVE_LIMIT=10 (~5s at 2 Hz control tick), well under the 6+ minute self-recovery window observed live in the 12-UAT.md episode"
  - "Watchdog counter increments only after a successful pop_validated_action() call (not on grpc/ConnectionError/RuntimeError exception paths, which return early before reaching the watchdog check)"
  - "safety_validator.py's STALE_ACTION_S/STALE_OBSERVATION_S left completely unchanged per CONTEXT.md D-01 -- this is a new, independent recovery mechanism layered on top of the existing staleness check"

patterns-established:
  - "Draining a shared queue.Queue via repeated get_nowait() under its lock (never .empty() as a loop condition) to avoid racing a background-thread refill"

requirements-completed: [LATENCY-01, LATENCY-02, LATENCY-04]

coverage:
  - id: D1
    description: "force_bridge_recovery(client) drains client.action_queue to empty and sets client.must_go"
    requirement: "LATENCY-04"
    verification:
      - kind: unit
        ref: "control/test_robot_client.py#test_force_bridge_recovery_drains_queue_and_sets_must_go"
        status: pass
      - kind: unit
        ref: "control/test_robot_client.py#test_force_bridge_recovery_handles_already_empty_queue"
        status: pass
    human_judgment: false
  - id: D2
    description: "BridgeActionSource.get_action() forces recovery after exactly N consecutive stale (growing-stale-action) results, not before"
    requirement: "LATENCY-01"
    verification:
      - kind: unit
        ref: "control/test_robot_client.py#test_bridge_action_source_watchdog_triggers_at_limit_for_growing_stale_action"
        status: pass
    human_judgment: false
  - id: D3
    description: "The same watchdog also triggers on N consecutive empty-queue (no-action-available) results"
    requirement: "LATENCY-01"
    verification:
      - kind: unit
        ref: "control/test_robot_client.py#test_bridge_action_source_watchdog_triggers_at_limit_for_empty_queue"
        status: pass
    human_judgment: false
  - id: D4
    description: "A single stale/empty result below the limit never triggers recovery; a genuine fresh result resets the counter to zero"
    requirement: "LATENCY-02"
    verification:
      - kind: unit
        ref: "control/test_robot_client.py#test_bridge_action_source_watchdog_does_not_trigger_below_limit"
        status: pass
      - kind: unit
        ref: "control/test_robot_client.py#test_bridge_action_source_watchdog_counter_resets_on_fresh_result"
        status: pass
    human_judgment: false
  - id: D5
    description: "Live-hardware confirmation that the watchdog breaks a real multi-minute deadlock and STALE_WATCHDOG_CONSECUTIVE_LIMIT=10 is well-tuned"
    verification: []
    human_judgment: true
    rationale: "Per CONTEXT.md D-06, Claude does not run physical hardware -- this requires a live episode run and inspection of episode.jsonl by the user."

# Metrics
duration: 25min
completed: 2026-09-26
status: complete
---

# Phase 12 Plan 4: Bridge Stale-Watchdog Recovery Summary

**Consecutive-stale watchdog in `BridgeActionSource` that forces `client.action_queue`/`client.must_go` recovery after 10 stuck ticks (~5s at 2 Hz), instead of waiting on the vendored `lerobot` library's own recovery path, which took 6+ minutes to self-converge in the live UAT run.**

## Performance

- **Duration:** 25 min
- **Started:** 2026-09-26T18:01:00Z
- **Completed:** 2026-09-26T18:26:25Z
- **Tasks:** 1
- **Files modified:** 2

## Accomplishments
- `force_bridge_recovery(client)` module-level function that race-safely drains `client.action_queue` (repeated `get_nowait()` under `client.action_queue_lock`, never a check-then-act `.empty()` loop) and sets `client.must_go`
- `_is_stale_or_empty(flags)` helper detecting both held-result shapes `pop_validated_action()` can return (`"no action available, holding position"` and any flag containing `"stale observation"`)
- `BridgeActionSource` gains a `_consecutive_stale_count` counter (and configurable `stale_watchdog_limit` constructor parameter, default `STALE_WATCHDOG_CONSECUTIVE_LIMIT = 10`) wired into `get_action()`: increments on every stale-or-empty result, forces recovery and resets to zero at the limit, and resets to zero unconditionally on any genuine fresh result
- Forced-recovery events are never silent: a `"staleness-watchdog: forced must_go + cleared action_queue"` flag is appended and flows into the returned `model_version` string (and, downstream, `io_logger`'s per-step JSONL record) -- satisfies the threat model's T-12-12 (Repudiation) mitigation
- 9 new tests added (2 direct unit tests of `force_bridge_recovery`, 3 of `_is_stale_or_empty`, 4 `BridgeActionSource`-level: trigger-at-limit for both the growing-stale and empty-queue symptoms, below-limit non-trigger, and counter-reset-on-fresh-result)
- All 18 pre-existing `test_robot_client.py` tests pass unmodified; full `control/` suite (119 tests) passes with 0 failures

## Task Commits

Each task was committed atomically (TDD RED → GREEN):

1. **Task 1 RED: failing tests for watchdog** - `2a6c2fe` (test)
2. **Task 1 GREEN: watchdog implementation** - `086970d` (feat)

**Plan metadata:** (this commit) - `docs(12-04): complete plan`

## Files Created/Modified
- `control/vla_bridge/robot_client.py` - added `STALE_WATCHDOG_CONSECUTIVE_LIMIT` constant, `_is_stale_or_empty()`, `force_bridge_recovery()`; `BridgeActionSource.__init__` gains `stale_watchdog_limit` param + `_consecutive_stale_count`; `get_action()` wires the watchdog check between `pop_validated_action()` and the existing `self._last_action = validated_action` line
- `control/test_robot_client.py` - added `FakeMustGoEvent`, `FakeBridgeClientWithMustGo`, `FakeClientReturnsGrowingStaleAction` fakes and 9 new tests covering the watchdog's unit-level and `BridgeActionSource`-level behavior

## Decisions Made
- `STALE_WATCHDOG_CONSECUTIVE_LIMIT = 10` (~5s at 2 Hz) chosen per the plan's explicit spec, well under the 6+ minute self-recovery window the live UAT episode showed
- Watchdog check placed after the `try`/`except (grpc.RpcError, ConnectionError, RuntimeError)` block around `pop_validated_action()` and before `self._last_action = validated_action`, exactly as specified -- exception paths (bridge unreachable) return early and never touch the counter, matching the plan's intent that this watchdog is orthogonal to the existing bridge-error-holding-position flag
- No changes to `safety_validator.py`'s `STALE_ACTION_S`/`STALE_OBSERVATION_S` (confirmed via grep: still `30.0`/`10.0`), honoring CONTEXT.md D-01

## Deviations from Plan

None - plan executed exactly as written. `force_bridge_recovery()`, `_is_stale_or_empty()`, the `BridgeActionSource` counter, and all specified test cases match the plan's `<action>` and `<behavior>` sections directly.

## Issues Encountered

None. All acceptance criteria verified directly:
- `grep -n "def force_bridge_recovery" control/vla_bridge/robot_client.py` finds the function
- `grep -n "_consecutive_stale_count" control/vla_bridge/robot_client.py` shows init + increment + reset sites
- `cd control && python -m pytest test_robot_client.py -q` -> 27 passed, 0 failures
- `cd control && python -m pytest -q` (full suite) -> 119 passed, 0 failures
- `grep -c "STALE_ACTION_S = \|STALE_OBSERVATION_S = " control/vla_bridge/safety_validator.py` -> 2, values unchanged (30.0/10.0)

## User Setup Required

None - no external service configuration required.

## Live-Hardware Verification Required (D-06)

This is a blocker-severity fix for a failure mode that took 6+ minutes to self-recover on real hardware. Per CONTEXT.md D-06, Claude does not run physical hardware -- the user must:

1. Run a live episode via `python control/run_vla_episode.py ... --server-address ... --checkpoint ...` long enough for the robot to hold still for a few seconds mid-episode.
2. Inspect `episode.jsonl` for any `model_version` values containing `"staleness-watchdog"`.
3. Confirm: (a) if the watchdog fired, the robot resumed receiving real actions within a few seconds afterward, not minutes; (b) if it never fired, `obs_age_s`/staleness values never grew anywhere near the 6+ minute deadlock duration observed in the original UAT run.
4. Report back whether `STALE_WATCHDOG_CONSECUTIVE_LIMIT=10` (~5s at 2 Hz) feels right or needs tuning.

## Next Phase Readiness

- Gap 3 (blocker, `12-UAT.md`) is closed at the unit-test level; live-hardware confirmation remains an open human-check item (D-06) before this can be considered fully verified end-to-end
- `control/vla_bridge/robot_client.py` and `control/test_robot_client.py` are the only files this plan touches -- no overlap with sibling wave-2 plan `12-03` (`run_vla_episode.py`, `io_logger.py`)
- No blockers for downstream phases (13-17); this closes out the last of Phase 12's UAT gaps alongside `12-03`

---
*Phase: 12-bridge-tick-latency-fix*
*Completed: 2026-09-26*

## Self-Check: PASSED

- FOUND: control/vla_bridge/robot_client.py
- FOUND: control/test_robot_client.py
- FOUND: .planning/phases/12-bridge-tick-latency-fix/12-04-SUMMARY.md
- FOUND: 2a6c2fe (test commit)
- FOUND: 086970d (feat commit)
- FOUND: ba3bf4d (docs/summary commit)
