---
phase: 11-vla-hardware-connection
plan: 05
subsystem: vla-hardware
tags: [vla-hardware, hardware-in-the-loop, device-discovery, findings, go-no-go, checkpoint-paused]

# Dependency graph
requires:
  - phase: 11-vla-hardware-connection
    provides: "Plan 11-04's BridgeActionSource/robot_client.py + StereoSplitCamera, Plan 11-01's safety_validator, Plan 11-02's IOLogger/run_vla_episode.py, Plan 11-03's selected checkpoint + policy_server_launch.md"
provides:
  - "control/detect_devices.py: auto-discovery of follower/leader serial ports (via connect+calibration round trip) and wrist/stereo camera indices (via built-in-name exclusion + AR0144 resolution fingerprinting, with an interactive brightness-check fallback)"
  - "control/device_map.json (schema): the new default source of truth run_vla_episode.py reads PORT/ROBOT_ID/camera indices from when explicit CLI args are omitted"
  - "camera1 (wrist) now genuinely wired into the VLA bridge's observation dict via connect_bridge()'s new wrist_camera_index parameter"
affects: [12-*, any future control/ script that needs port/camera identity]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Camera semantic names (wrist/overhead) are assigned positionally from --camera flag order (or probe order), not from a fixed index->name table -- physical USB enumeration order is not guaranteed stable across sessions on this rig"
    - "Device identity (serial port -> robot id, camera index -> wrist/stereo) is resolved by an actual behavioral round trip (connect+calibration-load, or resolution fingerprint) rather than fixed string/index matching -- both were confirmed to drift session-to-session on this rig"
    - "CLI defaults sourced from a machine-local, gitignored device_map.json (detect_devices.py's output) with explicit CLI args always taking precedence, and a clear error (no silent fallback) when neither is available"

key-files:
  created:
    - control/detect_devices.py
    - control/test_detect_devices.py
  modified:
    - control/run_vla_episode.py
    - control/test_run_vla_episode.py
    - control/vla_bridge/robot_client.py
    - control/test_robot_client.py
    - control/COMMANDS.md
    - .gitignore

key-decisions:
  - "Task 1 (e-stop hardware-in-the-loop) approved after a second attempt with explicit --camera flags; the first attempt silently failed (ran to max_steps, Ctrl+C never registered by the process) because with neither real camera connected, cv2 opened the MacBook's built-in FaceTime camera at index 0 and treated it as the wrist camera"
  - "Discovered during Task 1 verification: physical camera index assignment is inverted from this project's long-standing assumption -- AR0144 stereo is at cv2 index 0 and IMX335 wrist is at cv2 index 1 this session, not the reverse. Confirmed by resolution (2560x720 vs standard) and a physical cover-the-lens test"
  - "Fixed run_vla_episode.py's DEFAULT_CAMERA_NAMES: it was a hardcoded {0: wrist, 1: overhead} table applied regardless of --camera flag order. Replaced with positional _build_camera_names()"
  - "Added --stereo-camera-index CLI flag to run_vla_episode.py, threaded to connect_bridge()"
  - "Task 2: chose an INDEPENDENT cv2.VideoCapture open for the wrist camera's real lerobot camera-config route (robot_config.cameras['camera1']), rather than sharing run_vla_episode.py's own local wrist capture handle -- reading the installed lerobot camera-config source found no built-in mechanism for two independently-registered camera configs to share one physical device the way StereoSplitCamera shares the AR0144's single open. Whether a standard USB webcam driver tolerates two independent opens (unlike the AR0144 stereo pair, which does not) could NOT be verified empirically in this session -- no real hardware was available in this coding environment. This MUST be re-verified against the real camera at Task 3; a second-open failure would surface as an exception inside RobotClient.__init__ before the bridge handshake starts"
  - "Task 2: serial port -> robot identity resolution uses an actual connect+calibration-load round trip per candidate port against each known robot id's calibration file (not fixed port-string matching), matching this session's finding that ports drift"
  - "Task 2: camera identity resolution excludes the built-in webcam by system_profiler SPCameraDataType name (case-insensitive 'facetime' substring match), then splits the remainder by the AR0144's distinctive 2560x720 resolution; a genuinely ambiguous second non-stereo candidate falls back to an interactive cover-the-lens brightness check rather than guessing"
  - "Task 2: run_vla_episode.py's PORT/ROBOT_ID/--camera/--stereo-camera-index are now optional -- when omitted, sourced from control/device_map.json; when device_map.json is ALSO unavailable, the script fails with a clear error rather than silently falling back to the old any-device-that-opens probe behavior"

requirements-completed: []

coverage:
  - id: D1
    description: "E-stop hardware-in-the-loop verification: Ctrl+C mid-episode halts the real SO-ARM101, moves it back to start position, exits cleanly, and records reason: keyboard_interrupt in termination.json"
    requirement: "VLAHW-02"
    verification: []
    human_judgment: true
    rationale: "Requires physically operating the real SO-ARM101 and observing its motion/e-stop behavior firsthand -- no automated test can substitute for hands-on hardware verification of a physical safety-critical behavior. APPROVED this session (see Accomplishments) after a first silent-failure attempt was independently diagnosed and a second attempt genuinely interrupted early."
  - id: D2
    description: "control/detect_devices.py auto-discovers follower/leader serial ports and wrist/stereo camera indices, excludes the built-in webcam by name, and writes device_map.json; run_vla_episode.py reads defaults from it when CLI args are omitted; connect_bridge()'s camera1 wiring gap is fixed"
    requirement: "VLAHW-01"
    verification:
      - kind: unit
        ref: "control/test_detect_devices.py (24 tests: name parsing, built-in exclusion, resolution disambiguation, interactive-fallback, port-identity round trip, top-level detect_devices(), main() CLI)"
        status: pass
      - kind: unit
        ref: "control/test_run_vla_episode.py (device_map.json defaulting, explicit-CLI override, missing-device_map error paths, --help regression)"
        status: pass
      - kind: unit
        ref: "control/test_robot_client.py (connect_bridge camera1 wiring via wrist_camera_index, and the None-default no-op case)"
        status: pass
    human_judgment: false
  - id: D3
    description: "One full VLA-driven episode (Colab SmolVLA PolicyServer over ngrok TCP tunnel -> BridgeActionSource -> safety_validator -> real robot -> IOLogger) recorded end-to-end with real (non-scripted) episode.jsonl, camera frame sequences, and termination.json"
    requirement: "VLAHW-04"
    verification: []
    human_judgment: true
    rationale: "Requires a live Colab PolicyServer, an active ngrok tunnel, the physical SO-ARM101 and cameras, and a human judgment call on pick-up success/failure (no automated vision-based success detection exists yet). Not yet attempted -- plan is paused at this task's checkpoint pending the human's live run."
  - id: D4
    description: "control/vla_bridge/FINDINGS.md go/no-go write-up synthesizing Task 1/Task 3's actual observed outcomes, lerobot#2210 status, and the D-05 no-reasoning-trace framing"
    requirement: "VLAHW-05"
    verification: []
    human_judgment: true
    rationale: "The write-up is a single-observer research artifact requiring human review to confirm every claim traces back to Task 1/Task 3's actual observed outcomes, not invented or assumed. Not yet started -- depends on Task 3."

duration: 0min (Task 1 was human-verification only; Task 2 was ~1 session of automated implementation; paused at Task 3 checkpoint)
completed: 2026-09-22
status: incomplete
---

# Phase 11 Plan 05: Hardware-in-the-Loop E-Stop, Device Auto-Discovery, Live Episode, and Findings Summary

**Device auto-discovery (`detect_devices.py` -> `device_map.json`) now supersedes hardcoded port/camera tables, and the bridge's previously-missing `camera1` (wrist) observation is wired in -- both closing real gaps found live during Task 1's e-stop verification, on top of that verification's own approval.**

## Performance

- **Duration:** 0 min automated work for Task 1 (human-verification only); Task 2 implemented and tested in this session
- **Started:** 2026-09-21 (Task 1 checkpoint first presented)
- **Completed:** N/A -- still paused, now at Task 3
- **Tasks:** 2/4 completed (Task 1 approved, Task 2 implemented+committed; Task 3 and Task 4 remain `checkpoint:human-verify`, `gate="blocking"`)
- **Files modified:** 8 (`control/detect_devices.py` new, `control/test_detect_devices.py` new, `control/run_vla_episode.py`, `control/test_run_vla_episode.py`, `control/vla_bridge/robot_client.py`, `control/test_robot_client.py`, `control/COMMANDS.md`, `.gitignore`)

## Accomplishments

- **Task 1 (E-stop hardware-in-the-loop verification): APPROVED** (prior session -- see key-decisions for the camera-identity bug found and fixed at the time).

- **Task 2 (Device auto-discovery + bridge camera1 wiring fix): IMPLEMENTED, TESTED, COMMITTED.**
  - `control/detect_devices.py`: new script resolving (a) follower/leader serial port identity via an actual connect+calibration-load round trip per candidate `/dev/cu.usbmodem*` port (not fixed port-string matching), and (b) wrist/stereo-overhead camera cv2 indices via `system_profiler SPCameraDataType` name-based built-in-webcam exclusion, then AR0144 2560x720-resolution fingerprinting among the remainder, falling back to an interactive cover-the-lens brightness check only when a second candidate is genuinely ambiguous. Writes `control/device_map.json` (gitignored, machine-local).
  - `control/run_vla_episode.py`: `PORT`/`ROBOT_ID`/`--camera`/`--stereo-camera-index` are now all optional. When omitted, values are sourced from `control/device_map.json` (printed which source was used for each); explicit CLI flags still override. If `device_map.json` is also unavailable, the script now fails with a clear, actionable error instead of silently falling back to the old any-device-that-opens probe behavior.
  - `control/vla_bridge/robot_client.py`: `connect_bridge()` gained a `wrist_camera_index` parameter that wires a real `camera1` entry into `robot_config.cameras` via lerobot's own `OpenCVCameraConfig`, BEFORE `RobotClient(config)` connects the robot (which is when lerobot's own camera-config route actually opens the device). This closes the real gap found live in the prior session: `camera1` was never wired at all, only `camera2`/`camera3` (the AR0144 stereo split via `StereoSplitCamera`) -- meaning the VLA checkpoint's wrist input would have been silently missing during a live episode.
  - Empirical finding NOT possible this session: whether a standard USB webcam driver tolerates a second independent `cv2.VideoCapture` open of the wrist camera's index (one from `run_vla_episode.py`'s own local IOLogger-recording caps, one from lerobot's own camera-config route opened inside `connect_bridge()`) could not be tested -- no real hardware is available in this coding environment. The independent-open path was chosen as the plan's own documented fallback when clean sharing isn't available; **this needs re-verification against the real wrist camera at Task 3** -- if the second open fails, it will surface as an exception raised inside `RobotClient.__init__` before the bridge handshake even starts (a loud failure, not a silent one).
  - 77/77 tests pass across the full `control/` suite (24 new in `test_detect_devices.py`; existing bridge-selection tests in `test_run_vla_episode.py`/`test_robot_client.py` updated for the new `wrist_camera_index` kwarg plus new device_map.json integration/regression tests).

## Task Commits

1. **Task 1: E-stop hardware-in-the-loop verification** -- no code commit (human-verification only); approved a prior session based on live evidence described in key-decisions.
2. **[Deviation, prior session] Fixed camera-index CLI plumbing** -- `b4513dc` (fix)
3. **Task 2: Device auto-discovery (device_map.json) + bridge camera1 wiring fix** -- `fa99468` (feat)

**Plan metadata:** pending (this SUMMARY's own commit)

## Files Created/Modified

- `control/detect_devices.py` (new) -- auto-discovery script: `list_serial_ports()`, `resolve_port_identity()` (connect+calibration round trip), `get_camera_names()`/`_parse_camera_names()` (system_profiler name parsing), `probe_camera_candidates()` (cv2 open+resolution probe), `resolve_cameras()` (built-in exclusion + resolution split + interactive brightness fallback), `detect_devices()` (top-level orchestration), `main()` (CLI writing `device_map.json`).
- `control/test_detect_devices.py` (new) -- 24 tests covering every function above, all hardware/subprocess calls mocked.
- `control/run_vla_episode.py` -- `PORT`/`ROBOT_ID` now `nargs="?"`; `--stereo-camera-index` default changed to `None` (resolved post-parse); added `DEVICE_MAP_PATH`/`_load_device_map()` (with a 24h staleness warning); `main()` now fills port/robot_id/camera/stereo-camera-index from `device_map.json` when CLI args are omitted, erroring clearly if both are unavailable; computes `wrist_camera_index` from `_build_camera_names()`'s positional "wrist" assignment and threads it to `connect_bridge()`.
- `control/test_run_vla_episode.py` -- updated existing `connect_bridge` fakes for the new `wrist_camera_index` kwarg; added `test_wrist_camera_index_threads_through_from_camera_names`, `test_device_map_supplies_port_robot_id_camera_defaults_when_cli_args_omitted`, `test_explicit_cli_args_override_device_map_json`, `test_missing_port_and_device_map_errors_clearly`, `test_missing_camera_and_device_map_errors_clearly`, `test_help_still_documents_port_robot_id_camera_as_explicit_overrides`.
- `control/vla_bridge/robot_client.py` -- `connect_bridge()` gained `wrist_camera_index: int | None = None`; wires `robot_config.cameras["camera1"] = OpenCVCameraConfig(index_or_path=wrist_camera_index)` before constructing `RobotClientConfig`/`RobotClient` when given.
- `control/test_robot_client.py` -- added `test_connect_bridge_wires_camera1_wrist_when_wrist_camera_index_given`, `test_connect_bridge_leaves_cameras_untouched_when_wrist_camera_index_omitted`.
- `control/COMMANDS.md` -- documents `detect_devices.py` as the preferred device-resolution path ahead of the (now explicitly flagged stale) hardcoded port table.
- `.gitignore` -- added `control/device_map.json` (machine-local state, not committed) and `control/.venv` (bare symlink form, in addition to the existing `control/.venv/` directory pattern -- this worktree's own `.venv` is a symlink to the main repo's real venv, which the trailing-slash pattern alone does not match).

## Decisions Made

See `key-decisions` in frontmatter. Summary: Task 1 approved based on a second, genuinely-interrupted attempt after diagnosing the first attempt's silent failure (prior session). Task 2 replaces drift-prone hardcoded port/camera assumptions with behavioral auto-discovery, and fixes the `camera1` wiring gap using lerobot's own camera-config route with an independently-opened capture (empirically unverified against real hardware this session -- flagged for Task 3).

## Deviations from Plan

None beyond what Task 2 itself specifies -- implemented per the plan's `<action>`/`<behavior>`/`<acceptance_criteria>` as written. The independent-vs-shared wrist-camera-open decision was explicitly left open by the plan text itself ("verify actual driver behavior empirically... document whatever is found") -- documented above and in `robot_client.py`'s docstring, not a deviation from the plan's own instructions.

## Issues Encountered

- **Environment constraint, not a code bug:** this coding session has no real SO-ARM101 hardware or cameras attached (a desk/code environment, not the live rig) -- `control/.venv` didn't even exist in this worktree and had to be symlinked from the main repo's checkout to run the test suite. This means the wrist-camera dual-open driver behavior (see key-decisions) and `detect_devices.py`'s actual real-hardware round trip are UNTESTED against physical devices; both are covered by mocked unit tests only, per this task's own testing scope. Task 3's live run is the first real-hardware exercise of `detect_devices.py` and the camera1 fix together.

## User Setup Required

Task 3 requires the human to:
1. Run `python detect_devices.py` first (if any USB device has been unplugged/replugged since Task 1) to produce a fresh `device_map.json`.
2. Measure the red cube per the plan's Task 3 `<how-to-verify>` step 1 (≤78mm graspable width, off the base's forward centerline).
3. Start the Colab notebook per `control/vla_bridge/policy_server_launch.md`.
4. Run `run_vla_episode.py` against the real robot and Colab bridge -- PORT/ROBOT_ID/`--camera` can now be omitted (device_map.json supplies them); pass them explicitly only if `device_map.json` is stale or missing.
5. Report the episode's outcome (success/failure, termination reason, lerobot#2210 status) per the plan's resume-signal.

## Next Phase Readiness

- Not ready -- this plan is now paused at Task 3 of 4 (Task 1 approved, Task 2 complete this session). Task 3 (full VLA-driven episode) has not yet been attempted; its checkpoint is being returned to the human now.
- Task 4 (`FINDINGS.md` write-up) depends on both Task 1 and Task 3's real, human-reported outcomes, and cannot be drafted yet.
- This SUMMARY will be superseded by a complete version once Task 3 and Task 4 are both approved and `control/vla_bridge/FINDINGS.md` exists.

---
*Phase: 11-vla-hardware-connection*
*Completed: N/A (paused at Task 3 checkpoint, 2026-09-22)*

## Self-Check: PASSED

- FOUND: control/detect_devices.py
- FOUND: control/test_detect_devices.py
- FOUND: control/run_vla_episode.py
- FOUND: control/vla_bridge/robot_client.py
- FOUND: .planning/phases/11-vla-hardware-connection/11-05-SUMMARY.md
- FOUND: fa99468 (Task 2 commit) in git log
