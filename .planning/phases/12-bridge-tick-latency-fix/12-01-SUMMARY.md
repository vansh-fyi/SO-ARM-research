---
phase: 12-bridge-tick-latency-fix
plan: 01
subsystem: infra
tags: [lerobot, grpc, robot-client, threading, bridge, vla]

# Dependency graph
requires:
  - phase: 11-vla-hardware-connection
    provides: BridgeActionSource/pop_validated_action/connect_bridge wiring the safety-validated Colab bridge to the real SO-ARM101, and the FINDINGS.md root-cause diagnosis of the 1/60 real-action yield this plan fixes.
provides:
  - BridgeActionSource.get_action() only sends a fresh camera observation when the local action queue is empty/near-empty, via the vendored lerobot _ready_to_send_observation() gate
  - pop_validated_action() updates client.latest_action after every successful pop, so lerobot's own staleness dedup logic functions as designed
  - BridgeActionSource._inflight_lock structurally prevents overlapping in-flight observation requests (defensive-only)
affects: [12-02-bridge-tick-latency-fix, 13-mllm-raw-json-control-loop, live-hardware-verification]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Vendored-library gate consultation: wrap an existing (not reimplemented) lerobot method call in an `if self.client.<vendored_gate>():` check, mirroring the vendored control_loop() reference pattern"
    - "Non-blocking guard with try/finally release: `lock.acquire(blocking=False)` + early-return-with-flag-string on failure, existing body wrapped in try/finally so an exception can never leave the guard permanently held"
    - "Hand-rolled fake objects with plain call-count attributes for concurrency/gate tests, not unittest.mock -- matches this file's established test convention"

key-files:
  created: []
  modified:
    - control/vla_bridge/robot_client.py
    - control/test_robot_client.py

key-decisions:
  - "Did not touch safety_validator.STALE_ACTION_S (D-01) -- the gate fix is the real remedy; STALE_ACTION_S becomes a rarely-triggered safety fallback, not a value this plan tunes"
  - "Did not add a network request timeout or retry mechanism (D-02) -- explicitly out of scope for this location-and-wire-up task"
  - "In-flight guard implemented as threading.Lock with non-blocking acquire, consistent with the codebase's existing action_queue_lock/latest_action_lock idiom (D-05's implementer's-choice)"
  - "Guard is defensive-only insurance against a future overlap-reintroducing change, not a fix for an active bug -- get_action() is still called synchronously, once per tick, from a single-threaded loop today"

patterns-established:
  - "In-flight guard idiom for future bridge call sites: non-blocking acquire returning a distinct '@bridge-request-inflight-holding-position' flag string, body wrapped in try/finally"

requirements-completed: [LATENCY-01, LATENCY-02, LATENCY-04]

coverage:
  - id: D1
    description: "BridgeActionSource.get_action() only calls control_loop_observation() when the vendored lerobot _ready_to_send_observation() gate returns True, instead of unconditionally every tick"
    requirement: "LATENCY-01"
    verification:
      - kind: unit
        ref: "control/test_robot_client.py#test_bridge_action_source_skips_observation_send_when_gate_is_false"
        status: pass
      - kind: unit
        ref: "control/test_robot_client.py#test_bridge_action_source_sends_observation_when_gate_is_true"
        status: pass
    human_judgment: false
  - id: D2
    description: "pop_validated_action() updates client.latest_action to the popped action's timestep after every successful pop, and leaves it untouched on an empty queue"
    requirement: "LATENCY-02"
    verification:
      - kind: unit
        ref: "control/test_robot_client.py#test_pop_validated_action_updates_latest_action_on_successful_pop"
        status: pass
      - kind: unit
        ref: "control/test_robot_client.py#test_pop_validated_action_leaves_latest_action_unchanged_on_empty_queue"
        status: pass
    human_judgment: false
  - id: D3
    description: "BridgeActionSource._inflight_lock structurally rejects a second, concurrent get_action() call with a distinct holding-position flag, without blocking or touching the client, while normal sequential calls are never rejected"
    requirement: "LATENCY-04"
    verification:
      - kind: unit
        ref: "control/test_robot_client.py#test_bridge_action_source_inflight_guard_rejects_concurrent_get_action_call"
        status: pass
      - kind: unit
        ref: "control/test_robot_client.py#test_bridge_action_source_inflight_guard_never_rejects_sequential_calls"
        status: pass
    human_judgment: false
  - id: D4
    description: "Live-hardware confirmation that a real episode shows a measurably higher real (non-stale) action yield than Phase 11's 1/60 baseline, with the in-flight guard confirmed not to trigger under normal operation"
    verification: []
    human_judgment: true
    rationale: "Requires running the real SO-ARM101 over the live Colab bridge; Claude does not run physical hardware (CONTEXT.md D-06). Owned by the user as a human-check step in Task 3's <verify>, after this plan and 12-02's LATENCY-03 are both merged."

duration: 25min
completed: 2026-09-25
status: complete
---

# Phase 12 Plan 01: Bridge Observation-Gate, Latest-Action Dedup, and In-Flight Guard Summary

**Wired lerobot's already-installed `_ready_to_send_observation()` gate into `BridgeActionSource.get_action()`, made `pop_validated_action()` update `client.latest_action` after every successful pop, and added a non-blocking `_inflight_lock` guard -- closing the three request-cadence/race-safety defects that discarded 59 of 60 real actions in Phase 11's live episode.**

## Performance

- **Duration:** 25 min
- **Started:** 2026-09-25T07:13:00Z (approx)
- **Completed:** 2026-09-25T07:37:51Z
- **Tasks:** 3
- **Files modified:** 2

## Accomplishments
- `BridgeActionSource.get_action()` now consults `self.client._ready_to_send_observation()` before calling `control_loop_observation()`, so a fresh camera observation is only sent to Colab when the local action queue is empty/near-empty, instead of every control tick (LATENCY-01)
- `pop_validated_action()` now sets `client.latest_action = timed_action.get_timestep()` after every successful pop, mirroring the vendored `lerobot` library's own `control_loop_action()` pattern, so `_ready_to_send_observation()`'s queue-size gate and the library's own staleness-dedup logic (`_aggregate_action_queues()`) both function as designed (LATENCY-02)
- `BridgeActionSource._inflight_lock` (a `threading.Lock`) structurally prevents overlapping in-flight observation requests: a failed non-blocking acquire returns immediately with a distinct `@bridge-request-inflight-holding-position` flag, without touching the client; the existing body runs inside `try/finally` so an exception can never leave the guard permanently held (LATENCY-04, defensive-only per CONTEXT.md D-05)
- The existing validation/execution gate separation (T-11-09) is unchanged: `pop_validated_action()` still never calls `send_action()`
- Full `control/test_robot_client.py` suite grew from 14 to 18 tests, all passing; full `control/` suite (99 tests across all files) passes with 0 failures, confirming no cross-file regression

## Task Commits

Each task was committed atomically:

1. **Task 1: Wire the observation-send gate into BridgeActionSource.get_action() (LATENCY-01)** - `5e843e0` (feat)
2. **Task 2: Update client.latest_action after a successful pop (LATENCY-02)** - `18a7d23` (feat)
3. **Task 3: Add an in-flight-request guard around get_action() (LATENCY-04)** - `965172c` (feat)

**Plan metadata:** (this commit, SUMMARY.md)

## Files Created/Modified
- `control/vla_bridge/robot_client.py` - `BridgeActionSource.get_action()` now gates the observation-send on `_ready_to_send_observation()` and wraps its body in a non-blocking `_inflight_lock` guard; `pop_validated_action()` updates `client.latest_action` after a successful pop; `BridgeActionSource.__init__` gains `self._inflight_lock`
- `control/test_robot_client.py` - `FakeClientRaisesOnObservation`/`FakeClientReturnsAction` gain `_ready_to_send_observation()` and a call-count attribute; `FakeBridgeClient`/`FakeClientReturnsAction` gain `latest_action_lock`/`latest_action`; new `FakeClientBlocksOnObservation` for the concurrency test; 6 new test functions covering gate-True/gate-False, latest_action post-pop/empty-queue, and in-flight concurrent-rejection/sequential-regression behavior

## Decisions Made
- Did not change `safety_validator.STALE_ACTION_S` (stays 30s) per D-01 -- the observation gate is the real fix; success is measured by real-action yield on the live episode, not by this constant
- Did not add a network request timeout or retry mechanism per D-02 -- explicitly out of scope for this location-and-wire-up task
- Implemented the in-flight guard as a `threading.Lock` with non-blocking `acquire(blocking=False)`, matching this codebase's existing `action_queue_lock`/`latest_action_lock` idiom, per D-05's implementer's-discretion framing
- Renamed the concurrency/regression test functions to include the substring `inflight` so the plan's specified `pytest -k inflight` verify command selects them deterministically

## Deviations from Plan

None - plan executed exactly as written. All three tasks (LATENCY-01, LATENCY-02, LATENCY-04) were implemented per the `<action>` and `<behavior>` specs in 12-01-PLAN.md, using the exact fix shapes documented in 12-PATTERNS.md.

## Issues Encountered

None. The `.venv` used for test execution is the main repo's `control/.venv` (not present per-worktree, since it is gitignored) -- tests were run against the worktree's checked-out source files using that interpreter, with no code or environment differences from what will be merged.

## User Setup Required

None - no external service configuration required. Live-hardware verification of success criterion #4 (real-action yield on the physical SO-ARM101) is a human-check step the user runs themselves after this plan and 12-02's LATENCY-03 are both merged, per CONTEXT.md D-06 (Claude does not run physical hardware). See Task 3's `<verify><human-check>` in 12-01-PLAN.md for the exact steps.

## Next Phase Readiness
- LATENCY-01/02/04 are complete and covered by passing, deterministic unit tests; `pop_validated_action()`'s validation/execution gate separation (T-11-09) is preserved throughout
- LATENCY-03 (real measured `latency_ms` in `io_logger.py`) and DEBT-02/DEBT-03 (both already resolved as not-applicable per CONTEXT.md D-03/D-04) remain for plan 12-02, if scoped there
- Live-hardware verification of the fix's actual real-action yield improvement is still outstanding and owned by the user

---
*Phase: 12-bridge-tick-latency-fix*
*Completed: 2026-09-25*

## Self-Check: PASSED

All created/modified files found on disk (control/vla_bridge/robot_client.py, control/test_robot_client.py, this SUMMARY.md). All 4 commit hashes (5e843e0, 18a7d23, 965172c, 8c81b8e) confirmed present in `git log`.
