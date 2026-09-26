---
phase: 12-bridge-tick-latency-fix
plan: 03
subsystem: vla-bridge
tags: [io_logger, run_vla_episode, stereo_camera, waypoint-interpolation, cv2, gap-closure]

# Dependency graph
requires:
  - phase: 12-bridge-tick-latency-fix (plan 01, 02)
    provides: LATENCY-01's queue-empty observation gate, LATENCY-03's real-latency io_logger recording -- both untouched by this plan, only extended
provides:
  - "io_logger.capture_stereo_frame() + _record_frame() helper: logs an already-read frame (StereoSplitCamera-shaped) instead of owning a cv2.VideoCapture.read() call"
  - "run_vla_episode._open_cameras(): skip-by-semantic-name camera-opening helper, used to stop opening a second AR0144 capture in the bridge path"
  - "run_vla_episode._interpolate_waypoints(): pure straight-line waypoint interpolation between two already-validated action dicts"
  - "run_episode() stereo_camera/execution_hz keyword parameters, --execution-hz CLI flag"
affects: [13-camera-resolution-unification]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "IOLogger split between cap-owning capture_camera_frame() and already-read-frame capture_stereo_frame(), both delegating to a shared _record_frame() private helper that lazily creates camera_{name}/ subdirectories"
    - "Waypoint interpolation between two already-safety-validated action dicts, sent at a higher --execution-hz decoupled from --control-hz's observation cadence, relying on the convex-combination-stays-in-bounds property (documented in the threat register) rather than re-validating each sub-step"

key-files:
  created: []
  modified:
    - control/vla_bridge/io_logger.py
    - control/run_vla_episode.py
    - control/test_io_logger.py
    - control/test_run_vla_episode.py

key-decisions:
  - "Task 1 and Task 2 landed in a single git commit, not two: run_episode()'s control loop and main()'s wiring interleave both fixes in the same hunks (the send_action call site and the caps/IOLogger construction site are shared), so a clean per-task split would require an artificial, temporarily-broken intermediate state."
  - "main() now always passes stereo_camera=... and execution_hz=... as keyword arguments to run_episode()'s single shared call site (used by both the bridge and non-bridge branches), rather than splitting into two separate call sites -- the non-bridge branch's stereo_camera stays None, functionally identical to omitting the keyword entirely."
  - "Every FakeBridgeClient test double in test_run_vla_episode.py needed a new self._stereo_camera = object() attribute added to __init__, since main() now reads client._stereo_camera unconditionally in the bridge path -- a minimal, non-assertion-changing update required by the real signature change, not a scope expansion."

requirements-completed: [LATENCY-03, LATENCY-04]

coverage:
  - id: D1
    description: "The --server-address bridge path never opens a second cv2.VideoCapture of the AR0144's index; camera_overhead_left/right are recorded from the shared client._stereo_camera every step"
    requirement: "LATENCY-04"
    verification:
      - kind: unit
        ref: "control/test_run_vla_episode.py#test_server_address_never_opens_overhead_camera_index_while_wrist_still_does"
        status: pass
      - kind: unit
        ref: "control/test_run_vla_episode.py#test_run_episode_records_overhead_left_and_right_from_shared_stereo_camera"
        status: pass
    human_judgment: true
    rationale: "Root-cause fix (no second capture, correct instance reused) is fully proven by unit tests, but confirming the recorded frames actually show the real robot workspace (not a wrong physical camera) requires the real AR0144 hardware -- Claude does not run physical hardware per CONTEXT.md D-06."
  - id: D2
    description: "robot.send_action() sends interpolated waypoints between the previous and next validated target at --execution-hz, decoupled from --control-hz; single-substep fallback is behaviorally identical to pre-plan behavior"
    requirement: "LATENCY-04"
    verification:
      - kind: unit
        ref: "control/test_run_vla_episode.py#test_run_episode_sends_multiple_interpolated_waypoints_per_tick_when_execution_hz_exceeds_control_hz"
        status: pass
      - kind: unit
        ref: "control/test_run_vla_episode.py#test_run_episode_default_execution_hz_sends_exactly_one_action_per_tick"
        status: pass
      - kind: unit
        ref: "control/test_run_vla_episode.py#test_interpolate_waypoints_returns_evenly_spaced_intermediate_values"
        status: pass
    human_judgment: true
    rationale: "Interpolation math and send-count/rate-decoupling are fully proven by unit tests, but motion smoothness is inherently a human visual judgment requiring the real physical arm -- Claude does not run physical hardware per CONTEXT.md D-06."

duration: 35min
completed: 2026-09-27
status: complete
---

# Phase 12 Plan 03: Camera Provenance Fix + Waypoint Interpolation Summary

**Bridge path now reuses the shared StereoSplitCamera for diagnostic recording (no second AR0144 open) and sends interpolated waypoints at a decoupled --execution-hz instead of raw per-tick jumps**

## Performance

- **Duration:** 35 min
- **Started:** 2026-09-27
- **Completed:** 2026-09-27
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- `IOLogger.capture_stereo_frame()` + shared `_record_frame()` helper: the `camera_overhead` diagnostic recording in the `--server-address` bridge path now reads from `client._stereo_camera` (the SAME instance feeding the policy) instead of opening a second, colliding `cv2.VideoCapture` of the AR0144 -- eliminating the bug that silently recorded the laptop's FaceTime camera during the live UAT episode.
- `_open_cameras()` helper: the "overhead"-labeled index is never passed to `cv2.VideoCapture` at all in the bridge path (not just discarded after opening); the non-bridge (`ScriptedActionSource`) path is unaffected.
- `_interpolate_waypoints()` + `--execution-hz` flag: `robot.send_action()` now sends interpolated sub-waypoints between the previous and next validated target at a rate decoupled from `--control-hz`'s observation-fetch cadence, tracing a continuous path instead of discrete jumps, while leaving Plan 12-01's `LATENCY-01` observation-gating untouched.
- 36 new/updated tests added across `test_io_logger.py` and `test_run_vla_episode.py`; full `control/` suite (122 tests) passes with 0 failures.

## Task Commits

Both tasks landed in a single commit (see Deviations below for why):

1. **Task 1 + Task 2: Camera provenance fix (Gap 1) + waypoint interpolation (Gap 2)** - `e7ff126` (feat)

**Plan metadata:** committed alongside this SUMMARY (see below)

## Files Created/Modified
- `control/vla_bridge/io_logger.py` - new `capture_stereo_frame()` public method + `_record_frame()` shared private helper; `capture_camera_frame()` refactored to delegate to it with unchanged public signature/behavior
- `control/run_vla_episode.py` - new `_open_cameras()` and `_interpolate_waypoints()` module-level helpers; `run_episode()` gains `stereo_camera`/`execution_hz` keyword parameters; `main()` gains a `--execution-hz` CLI flag, routes caps-opening through `_open_cameras(..., skip_names=...)`, and threads `stereo_camera=client._stereo_camera` through to `run_episode()` in the bridge path
- `control/test_io_logger.py` - 3 new tests: `capture_stereo_frame()` success, `None`-frame failure, lazy subdirectory creation for an undeclared camera name
- `control/test_run_vla_episode.py` - new tests for `_open_cameras()` (skip-by-name, empty-skip regression), a `main()`-level integration test confirming the overhead index never reaches `cv2.VideoCapture`, `run_episode()`-level camera-provenance tests (with/without `stereo_camera`), `_interpolate_waypoints()` unit tests (evenly-spaced + degenerate single-step), and `run_episode()`-level multi-substep/default-substep send-count tests; every pre-existing `FakeBridgeClient` test double updated with a `_stereo_camera` attribute

## Decisions Made
- Both tasks committed together (single commit `e7ff126`) rather than two atomic per-task commits, because `run_episode()`'s control loop and `main()`'s wiring interleave Task 1's camera-provenance changes and Task 2's waypoint-interpolation changes in the same code regions (the single `robot.send_action()` call site, the single `caps`/`IOLogger` construction site). A clean split would require introducing an artificial, temporarily-inconsistent intermediate state not present in the plan's own design.
- `main()`'s single shared `run_episode(...)` call site (used by both the bridge and non-bridge branches) now always passes `stereo_camera=stereo_camera` (a variable set to `client._stereo_camera` in the bridge branch, left `None` otherwise) and `execution_hz=args.execution_hz`, rather than splitting into two separate call sites as the plan's literal phrasing ("the non-bridge branch's call is NOT given this keyword") might suggest -- functionally identical (default is `None`), and avoids duplicating the call site.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Test doubles missing `_stereo_camera` attribute**
- **Found during:** Task 1 (running the pre-existing bridge-selection tests after wiring `stereo_camera=client._stereo_camera` into `main()`)
- **Issue:** `main()` now unconditionally reads `client._stereo_camera` in the `--server-address` branch. All 6 pre-existing `FakeBridgeClient` test doubles in `test_run_vla_episode.py` only set `self.robot = object()` in `__init__`, causing `AttributeError: 'FakeBridgeClient' object has no attribute '_stereo_camera'` in every bridge-path test.
- **Fix:** Added `self._stereo_camera = object()` to every `FakeBridgeClient.__init__` (6 occurrences, identical pattern, applied via `replace_all`). No test assertions changed.
- **Files modified:** control/test_run_vla_episode.py
- **Verification:** All 18 pre-existing `test_run_vla_episode.py` tests pass unmodified (assertions-wise); full suite green.
- **Committed in:** e7ff126 (combined task commit)

**2. [Rule 3 - Blocking] `fake_run_episode` fixture rejected new keyword arguments**
- **Found during:** Task 1 (same test run as above)
- **Issue:** `test_run_vla_episode_selects_bridge_source_when_server_address_given`'s `fake_run_episode` monkeypatch had an explicit positional-only parameter list matching the pre-Task-1 `run_episode()` signature. `main()`'s new `stereo_camera=...`/`execution_hz=...` keyword arguments raised `TypeError: fake_run_episode() got an unexpected keyword argument`.
- **Fix:** Added a trailing `**kwargs` parameter to `fake_run_episode`'s signature. No assertions changed.
- **Files modified:** control/test_run_vla_episode.py
- **Verification:** Test passes; its existing assertions (`action_source_type`, `bridge_action_source_client`, etc.) are unaffected.
- **Committed in:** e7ff126 (combined task commit)

**3. [Rule 1 - Bug] New multi-substep tests initially miscounted `move_to_positions()`'s trailing send**
- **Found during:** Task 2 (writing `test_run_episode_sends_multiple_interpolated_waypoints_per_tick_when_execution_hz_exceeds_control_hz` and `test_run_episode_default_execution_hz_sends_exactly_one_action_per_tick`)
- **Issue:** `run_episode()`'s `finally` block always calls `move_to_positions(robot, start_positions, ...)` (the e-stop return-to-start move), which itself calls `robot.send_action()` at least once even when the arm is already at the target (since `mock_robot`'s positions start and stay at 0.0, `total_error < stop_error` is true after the first P-control iteration, but that iteration still sends one action). This pre-existing behavior was orthogonal to this plan's changes but inflated `mock_robot.sent_actions` counts by 1 in the new tests, which asserted exact counts (20 and 3 respectively, observed as 21 and 4).
- **Fix:** Both new tests now `monkeypatch.setattr(run_vla_episode, "move_to_positions", lambda *a, **k: None)` (the same pattern the pre-existing `test_keyboard_interrupt_triggers_return_to_start_before_disconnect` test already used), isolating the assertion to the main control loop's sends only.
- **Files modified:** control/test_run_vla_episode.py
- **Verification:** Both tests pass with exact counts (20 and 3) after the fix.
- **Committed in:** e7ff126 (combined task commit)

---

**Total deviations:** 3 auto-fixed (all Rule 1/3 -- test-double/fixture corrections required by real signature changes, no assertion or scope changes)
**Impact on plan:** All three fixes were necessary consequences of threading new parameters through shared call sites; none altered the plan's intended behavior or added scope.

## Issues Encountered
None beyond the auto-fixed items documented above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Both Gap 1 (camera provenance) and Gap 2 (motion smoothness) fixes are implemented and fully covered by deterministic, hardware-free unit tests; `cd control && python -m pytest -q` (full 122-test suite) passes with 0 failures.
- Live-hardware confirmation of both fixes (camera framing showing the real robot workspace; visibly smooth interpolated motion; whether `--execution-hz`'s 20.0 default feels right) is a `<human-check>` handoff per CONTEXT.md D-06 -- Claude does not run physical hardware. The user should run a live episode via `python control/run_vla_episode.py ... --server-address ... --checkpoint ...` and report back on both.
- Gap 3 (the stale/similarity deadlock blocker) is out of scope for this plan -- closed by sibling gap-closure plan 12-04 in the same wave (zero `files_modified` overlap: `robot_client.py` vs. this plan's `io_logger.py`/`run_vla_episode.py`).

---
*Phase: 12-bridge-tick-latency-fix*
*Completed: 2026-09-27*

## Self-Check: PASSED

All modified files verified present (control/vla_bridge/io_logger.py, control/run_vla_episode.py,
control/test_io_logger.py, control/test_run_vla_episode.py, this SUMMARY.md). Both commits verified
in git log (e7ff126 task commit, 6e478ff SUMMARY commit).
