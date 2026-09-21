---
phase: 11-vla-hardware-connection
plan: 04
subsystem: vla-hardware
tags: [vla-hardware, smolvla, lerobot, async-inference, grpc, stereo-camera]

# Dependency graph
requires:
  - phase: 11-vla-hardware-connection
    provides: "Plan 11-02's run_vla_episode.py ActionSource seam + IOLogger + safety-validator plumbing; Plan 11-03's selected checkpoint (victorvanhalst/smolvla_so101_cube) and split-stereo camera mapping (policy_server_launch.md)"
provides:
  - "vla_bridge/robot_client.py: connect_bridge()/pop_validated_action()/BridgeActionSource, wrapping lerobot.async_inference.RobotClient without ever calling its unvalidated control_loop_action()/control_loop()"
  - "run_vla_episode.py extended with --server-address/--checkpoint flags selecting BridgeActionSource over Plan 11-02's ScriptedActionSource, plus a bridge_unreachable termination reason"
  - "vla_bridge/stereo_camera.py's StereoSplitCamera, wired into connect_bridge() so camera2/camera3 are real AR0144 left/right split-stereo feeds, not a dummy 3rd slot"
affects: ["11-05 (hardware-in-the-loop dry run/e-stop checkpoint reuses this exact script with the real bridge)"]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Validation-gate/execution-gate separation: pop_validated_action() only reads the bridge's action_queue and calls safety_validator.validate_action() -- it never calls send_action itself; the caller (BridgeActionSource/run_vla_episode.py) owns the actual write, keeping the two gates visibly separate in the code (T-11-09)"
    - "Observation-dict patch over camera-config plugin: robot_client.py wires split-stereo cameras by monkey-patching the connected robot's own get_observation() rather than registering a custom lerobot CameraConfig subclass -- CameraConfig IS a draccus.ChoiceRegistry (a plugin route exists), but no built-in mechanism lets two registered camera configs share one physical device instance, so wrapping get_observation() is the simpler, directly-testable, actually-scriptable mechanism given the installed lerobot==0.6.1 source"
    - "Shared-read-per-tick camera split: StereoSplitCamera caches both halves from one physical cap.read() until both read_left()/read_right() have been consumed once each, so a back-to-back call pair never desyncs the stereo pair with two separate physical reads"

key-files:
  created:
    - control/vla_bridge/robot_client.py
    - control/test_robot_client.py
    - control/vla_bridge/stereo_camera.py
    - control/test_stereo_camera.py
  modified:
    - control/requirements.txt
    - control/run_vla_episode.py
    - control/test_run_vla_episode.py
    - .gitignore

key-decisions:
  - "connect_bridge()'s camera wiring uses an observation-dict patch (monkey-patching client.robot.get_observation), not a custom lerobot camera `type` plugin -- decided after reading the installed lerobot==0.6.1 cameras/ source: CameraConfig is a draccus.ChoiceRegistry so a plugin route is real, but nothing in that registry lets two independently-configured camera instances share one physical cv2 device, which StereoSplitCamera already solves; wrapping the connected robot's own get_observation() is simpler and directly testable without needing a real RobotClient/robot construction."
  - "StereoSplitCamera caches both split halves from a single physical read until both read_left() and read_right() have each been consumed once, then triggers a fresh read on the next call -- matches the real per-step usage pattern (one left+right pair per control-loop tick) without needing an explicit external tick() signal."
  - "Installed the fresh worktree control/.venv with `lerobot[feetech,async]==0.6.1` plus pytest/opencv-python/numpy/pynput (not the full requirements.txt, which also pins `ultralytics` for an unrelated, not-yet-wired YOLO feature) -- matches Plan 11-02/11-03's precedent of installing only what this plan's own imports actually need."

requirements-completed: [VLAHW-01, VLAHW-02, VLAHW-03]

coverage:
  - id: D1
    description: "Bridge wrapper (connect_bridge()/pop_validated_action()/BridgeActionSource) routes every bridge-returned action through safety_validator.validate_action() before it can reach the servo bus, and never calls the installed RobotClient's own control_loop_action()/control_loop() (which call send_action() internally, unvalidated)"
    requirement: "VLAHW-02"
    verification:
      - kind: unit
        ref: "control/test_robot_client.py -x -q"
        status: pass
    human_judgment: false
  - id: D2
    description: "run_vla_episode.py selects BridgeActionSource when --server-address/--checkpoint are given (ScriptedActionSource otherwise, unchanged), and a failed bridge handshake writes termination.json with reason bridge_unreachable and never attempts to drive the robot"
    requirement: "VLAHW-02"
    verification:
      - kind: unit
        ref: "control/test_run_vla_episode.py -x -q"
        status: pass
    human_judgment: false
  - id: D3
    description: "The AR0144's single 2560x720 stereo frame is split into two independent real camera feeds (camera2=left half, camera3=right half) via StereoSplitCamera, opened exactly once per process and wired into the bridge's observation-building step -- not a dummy/duplicated 3rd camera slot"
    requirement: "VLAHW-01"
    verification:
      - kind: unit
        ref: "control/test_stereo_camera.py -x -q"
        status: pass
    human_judgment: false
  - id: D4
    description: "The real, Colab-hosted SmolVLA bridge end-to-end (real ngrok tunnel, real PolicyServer, real robot) has not been exercised against physical hardware in this session -- Plan 11-05's hardware-in-the-loop dry run is the first live test of this code path"
    verification: []
    human_judgment: true
    rationale: "Requires a live Colab PolicyServer + ngrok tunnel + the physical SO-ARM101 and cameras; explicitly out of this plan's own scope (deferred to Plan 11-05 per this plan's success criteria)."

duration: 45min
completed: 2026-09-21
status: complete
---

# Phase 11 Plan 04: Real Colab Bridge + Split-Stereo Cameras Summary

**A validated bridge wrapper (`connect_bridge()`/`pop_validated_action()`/`BridgeActionSource`) that drives the real SO-ARM101 from a real, Colab-hosted SmolVLA checkpoint over `lerobot.async_inference.RobotClient` -- deliberately never calling the library's own unvalidated `control_loop_action()`/`control_loop()` -- plus a `StereoSplitCamera` that turns the AR0144's one physical 2560x720 frame into two genuinely independent `camera2`/`camera3` feeds for the checkpoint's 3-camera-slot config**

## Performance

- **Duration:** 45 min
- **Started:** 2026-09-21 (session start, after Plan 11-03 approval)
- **Completed:** 2026-09-21
- **Tasks:** 3 completed (Task 3 added mid-phase after Plan 11-03's checkpoint discussion, post original plan-checker pass)
- **Files modified:** 8 (4 created, 4 modified)

## Accomplishments

- Built `control/vla_bridge/robot_client.py`: `connect_bridge()` constructs and connects a real `RobotClient` (which itself connects the physical robot as a side effect), returning `None` (never an exception) on a failed handshake; `pop_validated_action()` pops one action from the bridge's `action_queue` and routes it through `safety_validator.validate_action()`, performing NO `send_action` call itself; `BridgeActionSource` implements `run_vla_episode.py`'s `ActionSource` interface, holding position and flagging `bridge-error-holding-position` on any `grpc.RpcError`/`ConnectionError`/`RuntimeError` instead of crashing the control loop.
- Extended `control/run_vla_episode.py` with `--server-address`/`--checkpoint` CLI flags: given, selects the real `BridgeActionSource` path (constructing the robot via `connect_bridge()`, not a second local `SO101Follower`); omitted, behavior is unchanged from Plan 11-02's `ScriptedActionSource` dry-run path. A failed bridge handshake writes `termination.json` with `reason: "bridge_unreachable"` and exits before touching the robot further.
- Built `control/vla_bridge/stereo_camera.py`'s `StereoSplitCamera`: opens the AR0144 (cv2 index 1) exactly once, splits each physical frame into left (`[0:1280]`)/right (`[1280:2560]`) halves, and shares one physical read across a `read_left()`/`read_right()` call pair for the same control-loop tick (never desyncing the stereo pair). Wired into `connect_bridge()` via `_wire_stereo_split_cameras()`, which monkey-patches the connected robot's own `get_observation()` to add real `camera2`/`camera3` keys sourced from this shared camera -- resolved after reading the installed `lerobot==0.6.1` `cameras/` source to confirm a custom camera-type plugin, while technically possible via `draccus.ChoiceRegistry`, doesn't solve the actual problem (sharing one physical device across two independently-configured camera instances) any more directly than this observation-patch approach.
- Every candidate action -- whether scripted (Plan 11-02) or bridge-returned (this plan) -- still passes through the identical `safety_validator.validate_action()` gate before reaching `robot.send_action()`. No new unvalidated write path exists.
- Updated `control/requirements.txt`: `lerobot[feetech]==0.6.1` -> `lerobot[feetech,async]==0.6.1` (pulls in `grpcio`, already audited/approved in `11-RESEARCH.md`'s Package Legitimacy Audit).

## Task Commits

Each task was committed atomically:

1. **Task 1: Bridge wrapper -- connect_bridge() + pop_validated_action() + BridgeActionSource** - `7bb8074` (feat)
2. **Task 2: Wire the real bridge into run_vla_episode.py** - `ee97df6` (feat)
3. **Task 3: Split-stereo camera wrapper (camera2/camera3 from the AR0144's single 2560x720 frame)** - `a38d91b` (feat)

_All 3 tasks were TDD-tagged in the plan; each task's behavior/tests were written and verified together as a cohesive unit (each task delivers one small, cohesive module), matching Plans 11-01/11-02's precedent for this same reason._

## Files Created/Modified

- `control/vla_bridge/robot_client.py` - `connect_bridge()`, `pop_validated_action()`, `BridgeActionSource`, `_wire_stereo_split_cameras()`
- `control/test_robot_client.py` - 9 tests: `pop_validated_action` success/empty-queue, `connect_bridge` handshake-fail/-succeed (incl. stereo-camera wiring assertion), `BridgeActionSource` rpc-error/connection-error/success paths
- `control/run_vla_episode.py` - `--server-address`/`--checkpoint` CLI flags, bridge-vs-scripted `ActionSource` selection, `bridge_unreachable` termination reason
- `control/test_run_vla_episode.py` - 3 new tests: bridge-source-selected, bridge-unreachable-writes-termination-without-driving-robot, checkpoint-without-server-address argparse error
- `control/vla_bridge/stereo_camera.py` - `StereoSplitCamera` class: `read_left()`, `read_right()`, `release()`, `is_opened`
- `control/test_stereo_camera.py` - 8 tests: single-open-per-instance, shared-read-per-tick, fresh-read-next-tick, correct non-swapped crop, open-failure, read-failure, `release()`, and the `robot_client.py` wiring itself
- `control/requirements.txt` - `lerobot[feetech]==0.6.1` -> `lerobot[feetech,async]==0.6.1`
- `.gitignore` - added `control/logs/` (runtime side effect of importing `lerobot.async_inference` at class-definition time, unrelated to this plan's actual deliverables)

## Decisions Made

- **Camera wiring mechanism (Task 3):** chose the observation-dict patch (monkey-patch `client.robot.get_observation`) over a custom `lerobot` camera `type` plugin, after reading the installed `lerobot==0.6.1` `cameras/camera.py`/`cameras/opencv/configuration_opencv.py` source directly. `CameraConfig` is a `draccus.ChoiceRegistry`, so a plugin route genuinely exists, but nothing in `make_cameras_from_configs` lets two independently-registered camera configs share one physical device instance -- a plugin would still need to solve that same problem `StereoSplitCamera` already solves. The observation-patch approach is simpler and directly testable with lightweight fakes (no real `RobotClient`/cv2 device needed).
- **`StereoSplitCamera`'s tick semantics:** rather than requiring an explicit `tick()`/step-boundary API, both halves are cached from one physical read until each of `read_left()`/`read_right()` has been individually consumed once; the next call after both are consumed triggers a fresh read. This matches the real per-step usage pattern (`BridgeActionSource`'s wrapped `get_observation()` calls both exactly once per control-loop tick) without adding an API surface the plan didn't ask for.
- **venv scope:** installed `lerobot[feetech,async]==0.6.1` + `pytest`/`opencv-python`/`numpy`/`pynput` directly (not `pip install -r requirements.txt`, which also pins `ultralytics` for an unrelated, not-yet-wired YOLO feature) -- matches Plan 11-02/11-03's precedent of installing only what this plan's own code imports.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Task 3's `connect_bridge()` change broke Task 1's existing `connect_bridge()` test fakes**
- **Found during:** Task 3 (wiring `_wire_stereo_split_cameras()` into `connect_bridge()`)
- **Issue:** Task 1's `FakeRobotClientHandshakeFails`/`FakeRobotClientHandshakeSucceeds` test doubles set `self.robot = object()`. Once Task 3's `connect_bridge()` started calling `_wire_stereo_split_cameras(client, ...)` internally -- which reads `client.robot.get_observation` -- those fakes' plain `object()` robot attribute raised `AttributeError`, breaking two previously-passing Task 1 tests.
- **Fix:** Added a `FakeRobotWithObservation` double (exposing `get_observation()`) as those fakes' `.robot`, and a `FakeStereoSplitCamera` double monkeypatched onto `robot_client.StereoSplitCamera` for both `connect_bridge()` tests, so no real cv2 device is opened by Task 1's tests. Also added an assertion to the success-path test confirming `camera2`/`camera3` are sourced from the split-stereo feed (covers this plan's own Task 3 acceptance criterion for `connect_bridge()`'s wiring).
- **Files modified:** `control/test_robot_client.py`
- **Verification:** Full `control/` suite (41 tests) passes after the fix.
- **Committed in:** `a38d91b` (Task 3 commit -- caught and fixed before that commit, not a follow-up)

---

**Total deviations:** 1 auto-fixed (1 bug, cross-task test breakage caught same-session)
**Impact on plan:** Necessary correctness fix so Task 1's tests keep passing after Task 3 extends the same function (`connect_bridge()`). No scope creep -- same test file, same tests, just updated fakes plus one new assertion directly relevant to this plan's own acceptance criteria.

## Issues Encountered

- This worktree had no `control/.venv` (gitignored, not part of the git-tracked worktree checkout) -- created a fresh `python3.12 -m venv control/.venv` and installed `lerobot[feetech,async]==0.6.1` (matching this plan's updated `requirements.txt` pin) plus `pytest==8.3.4`, `opencv-python`, `numpy`, `pynput==1.8.2`. Verified `grpcio`/`lerobot.async_inference` import correctly (plan's own verification command). This is local tooling setup, not a plan deviation -- `control/.venv/` remains gitignored.
- Importing `lerobot.async_inference.robot_client` creates a `logs/` directory + timestamped log file as a class-definition-time side effect (`RobotClient.logger = get_logger(prefix)`). Added `control/logs/` to `.gitignore` since this is generated runtime output, not a deliverable.

## User Setup Required

None - no external service configuration required for this session. (Carried forward: a real hardware-in-the-loop run against the live Colab bridge -- per this plan's own `<success_criteria>` -- requires the physical SO-ARM101, cameras, a running Colab `PolicyServer`, and an active ngrok tunnel per Plan 11-03's `policy_server_launch.md`; none of that was available or exercised in this worktree session.)

## Next Phase Readiness

- `run_vla_episode.py --server-address <ngrok host:port> --checkpoint victorvanhalst/smolvla_so101_cube` is ready to drive the real robot from a real Colab-hosted SmolVLA policy, with every returned action still passing through the exact safety validator proven in Plan 11-01/11-02.
- `StereoSplitCamera` and its `connect_bridge()` wiring give the checkpoint's 3-camera-slot config three genuinely distinct real camera views (wrist, AR0144-left, AR0144-right) -- no dummy/black placeholder slot.
- Network-specific failure modes (unreachable bridge, stale bridge response) are handled the same way this session tested them: `connect_bridge()` returns `None` -> `bridge_unreachable` termination reason; a mid-episode `grpc.RpcError`/`ConnectionError` -> hold position, flagged in `model_version`.
- Plan 11-05's hardware-in-the-loop dry run is the next and first live exercise of this bridge code end-to-end (real Colab `PolicyServer`, real ngrok tunnel, real robot, real cameras) -- no blockers identified for proceeding.

## Self-Check: PASSED

- FOUND: control/vla_bridge/robot_client.py
- FOUND: control/test_robot_client.py
- FOUND: control/vla_bridge/stereo_camera.py
- FOUND: control/test_stereo_camera.py
- FOUND commit: 7bb8074
- FOUND commit: ee97df6
- FOUND commit: a38d91b

---
*Phase: 11-vla-hardware-connection*
*Completed: 2026-09-21*
