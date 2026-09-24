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
    - control/vla_bridge/FINDINGS.md
  modified:
    - control/run_vla_episode.py
    - control/test_run_vla_episode.py
    - control/vla_bridge/robot_client.py
    - control/test_robot_client.py
    - control/COMMANDS.md
    - .gitignore

key-decisions:
  - "Task 1 (e-stop hardware-in-the-loop) approved after a second attempt with explicit --camera flags; the first attempt silently failed (ran to max_steps, Ctrl+C never registered by the process) because with neither real camera connected, cv2 opened the MacBook's built-in FaceTime camera at index 0 and treated it as the wrist camera"
  - "Task 3 (full VLA-driven live episode): two prior live attempts crashed on now-fixed bridge bugs (343fa28: receive_actions() thread never started; e55aa53: staleness checked against STALE_OBSERVATION_S instead of STALE_ACTION_S; a3e16c4: pop_validated_action() built .pos-suffixed keys that silently failed safety_validator's plain-name lookup, producing an empty validated_action and crashing lerobot's ensure_safe_goal_position()). Safety-validator caps were also intentionally loosened ~8-10x (quick task 260924-e3d) to let a real VLA action clear validation. Fourth attempt (control/outputs/11-05-retry-20260924-120530/) ran to max_steps_reached (60/60 steps) with no crash and no lerobot#2210 reproduction; net yield was 1 real executed VLA action out of 60 steps (step 52) -- 59 steps held position because each real control tick took ~11-20s wall-clock (dominated by control_loop_observation() re-sending a full camera observation to Colab on every tick even while draining an already-fetched chunk locally) against a configured control_hz=2.0 (0.5s/tick), so queued actions aged past even the loosened 30s staleness cap before being popped"
  - "Task 4 (FINDINGS.md go/no-go write-up): drafted, grounded in Task 1/Task 3's actual recorded data (episode.jsonl/termination.json read directly, not assumed); recommends GO conditional on fixing the observation-resend-per-tick latency bug before the next live episode attempt -- this is scoped as the concrete next engineering task, not an architecture change. Pending human review per the plan's checkpoint:human-verify gate -- not self-approved"
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
    rationale: "APPROVED this session after two prior crashed attempts were diagnosed and fixed (see key-decisions). Fourth attempt (control/outputs/11-05-retry-20260924-120530/) ran to max_steps_reached (60/60 steps), no crash, no lerobot#2210 reproduction; net yield 1/60 real executed VLA actions -- cube was not fully picked up within max_steps, but the pipeline ran real, non-scripted, end-to-end as the requirement specifies."
  - id: D4
    description: "control/vla_bridge/FINDINGS.md go/no-go write-up synthesizing Task 1/Task 3's actual observed outcomes, lerobot#2210 status, and the D-05 no-reasoning-trace framing"
    requirement: "VLAHW-05"
    verification: []
    human_judgment: true
    rationale: "Drafted this session, grounded directly in episode.jsonl/termination.json (control/outputs/11-05-retry-20260924-120530/) and Task 1's prior approval. Recommends GO conditional on fixing the observation-resend-per-tick latency bug. Pending human review per the plan's checkpoint:human-verify gate -- file exists but is NOT yet approved."

duration: ~1.5hr (Task 3: live hardware session diagnosing/fixing 3 bridge bugs + re-running to a complete episode; Task 4: findings write-up grounded in that episode's real data)
completed: 2026-09-24
status: incomplete
---

# Phase 11 Plan 05: Hardware-in-the-Loop E-Stop, Device Auto-Discovery, Live Episode, and Findings Summary

**Device auto-discovery (`detect_devices.py` -> `device_map.json`) now supersedes hardcoded port/camera tables, the bridge's previously-missing `camera1` (wrist) observation is wired in, a full real-hardware VLA episode ran end-to-end (1/60 steps executed a real action -- see FINDINGS.md), and a go/no-go findings write-up is drafted pending human review.**

## Performance

- **Duration:** 0 min automated work for Task 1 (human-verification only); Task 2 implemented and tested in a prior session; Task 3 was a live hardware session this session diagnosing and fixing 3 bridge bugs before a complete episode ran; Task 4 (`FINDINGS.md`) drafted this session
- **Started:** 2026-09-21 (Task 1 checkpoint first presented)
- **Completed:** N/A -- Task 4 still awaits human "approved" (this plan's final checkpoint)
- **Tasks:** 4/4 executed (Task 1 approved prior session; Task 2 implemented+committed prior session; Task 3 approved this session with real episode data recorded; Task 4 drafted this session, `checkpoint:human-verify` pending review)
- **Files modified/created this session:** 1 (`control/vla_bridge/FINDINGS.md`, new)

## Accomplishments

- **Task 1 (E-stop hardware-in-the-loop verification): APPROVED** (prior session -- see key-decisions for the camera-identity bug found and fixed at the time).

- **Task 2 (Device auto-discovery + bridge camera1 wiring fix): IMPLEMENTED, TESTED, COMMITTED** (prior session -- see key-decisions and Files Created/Modified below).

- **Task 3 (Full VLA-driven episode, recorded end-to-end): APPROVED this session.**
  - Two prior live attempts crashed before completing; both root-caused and fixed this session: (1) `connect_bridge()` never started `receive_actions()`'s background thread, so the action queue was permanently empty (`343fa28`); (2) the bridge's staleness check compared action age against `STALE_OBSERVATION_S` (1.0s) instead of `STALE_ACTION_S` (`e55aa53`); (3) `pop_validated_action()` built `.pos`-suffixed keys that silently failed `safety_validator`'s plain-name lookup, producing an empty `validated_action` and crashing lerobot's `ensure_safe_goal_position()` (`a3e16c4`, quick task `260924-gih`). Safety-validator caps were also intentionally loosened ~8-10x (quick task `260924-e3d`) so a real VLA action could clear validation.
  - Fourth attempt (`control/outputs/11-05-retry-20260924-120530/`) ran to `max_steps_reached` (60/60 steps), no crash, no `lerobot#2210` reproduction. Verified directly against the raw `episode.jsonl`: net yield was 1 real executed VLA action out of 60 steps (step 52) -- the other 59 steps held position because each real control tick took ~11-20s wall-clock (dominated by `control_loop_observation()` re-sending a full camera observation to Colab on every tick even while draining an already-fetched chunk locally) against a configured `control_hz=2.0` (0.5s/tick), so queued actions aged past even the loosened 30s staleness cap before being popped.
  - Step 52's real executed action moved the arm as the checkpoint intended (shoulder_pan 2.7°->-5.6°, shoulder_lift -106.2°->-72.7°, elbow_flex 91.4°->54.1°, gripper 95.8%->49.8% closing), confirmed by the human operator watching live.

- **Task 4 (Findings write-up and go/no-go recommendation): DRAFTED, COMMITTED, pending human review.**
  - `control/vla_bridge/FINDINGS.md` synthesizes lerobot#2210 status (not reproduced), Task 3's actual episode outcome (grounded in the real `episode.jsonl`/`termination.json`, not invented), Task 1's e-stop outcome, D-05's no-reasoning-trace framing, and an explicit GO recommendation conditional on fixing the observation-resend-per-tick latency bug identified in Task 3.
  - This is a `checkpoint:human-verify` task -- NOT self-approved. The file exists and is ready for human review; the plan's own resume-signal ("approved" or corrections) happens in a separate turn.

## Task Commits

1. **Task 1: E-stop hardware-in-the-loop verification** -- no code commit (human-verification only); approved a prior session based on live evidence described in key-decisions.
2. **[Deviation, prior session] Fixed camera-index CLI plumbing** -- `b4513dc` (fix)
3. **Task 2: Device auto-discovery (device_map.json) + bridge camera1 wiring fix** -- `fa99468` (feat)
4. **[Deviation, this session] Fixed receive_actions() thread never starting** -- `343fa28` (fix)
5. **[Deviation, this session] Fixed staleness check using wrong constant** -- `e55aa53` (fix)
6. **[Deviation, prior quick task 260924-gih] Fixed .pos-suffixed key mismatch dropping every real VLA action** -- `a3e16c4`/`8e2060a` (fix/test)
7. **[Deviation, prior quick task 260924-e3d] Loosened safety-validator caps for real-VLA-action testing** -- `d8c2785`/`8e2060a` (fix)
8. **Task 3: Full VLA-driven episode** -- no separate code commit (live hardware run producing gitignored runtime output at `control/outputs/11-05-retry-20260924-120530/`); approved this session based on the recorded episode data.
9. **Task 4: Findings write-up and go/no-go recommendation** -- `434b01a` (docs), pending human review

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
- `control/vla_bridge/FINDINGS.md` (new, this session) -- go/no-go findings write-up grounded in Task 1/Task 3's real recorded data (`control/outputs/11-05-retry-20260924-120530/episode.jsonl`/`termination.json`), covering lerobot#2210 status, Task 3's step-by-step episode outcome, Task 1's e-stop outcome, D-05's no-reasoning-trace restatement, and a scoped GO recommendation.
- `control/vla_bridge/robot_client.py`, `control/vla_bridge/policy_server_launch.md` (this session, prior deviation commits) -- receive_actions() background thread start fix and staleness-constant fix (see Task Commits #4-5).

## Decisions Made

See `key-decisions` in frontmatter. Summary: Task 1 approved based on a second, genuinely-interrupted attempt after diagnosing the first attempt's silent failure (prior session). Task 2 replaces drift-prone hardcoded port/camera assumptions with behavioral auto-discovery, and fixes the `camera1` wiring gap using lerobot's own camera-config route with an independently-opened capture (empirically unverified against real hardware this session -- flagged for Task 3).

## Deviations from Plan

None beyond what Task 2 itself specifies -- implemented per the plan's `<action>`/`<behavior>`/`<acceptance_criteria>` as written. The independent-vs-shared wrist-camera-open decision was explicitly left open by the plan text itself ("verify actual driver behavior empirically... document whatever is found") -- documented above and in `robot_client.py`'s docstring, not a deviation from the plan's own instructions.

## Issues Encountered

- **Environment constraint, not a code bug:** this coding session has no real SO-ARM101 hardware or cameras attached (a desk/code environment, not the live rig) -- `control/.venv` didn't even exist in this worktree and had to be symlinked from the main repo's checkout to run the test suite. This means the wrist-camera dual-open driver behavior (see key-decisions) and `detect_devices.py`'s actual real-hardware round trip are UNTESTED against physical devices; both are covered by mocked unit tests only, per this task's own testing scope. Task 3's live run is the first real-hardware exercise of `detect_devices.py` and the camera1 fix together.

## User Setup Required

Task 4 requires the human to:
1. Read `control/vla_bridge/FINDINGS.md` end to end.
2. Confirm every claim traces back to something actually observed in Task 1 or Task 3 (not invented or assumed), and that the go/no-go recommendation is justified by that evidence.
3. Type "approved", or provide corrections to fold into `FINDINGS.md` before this phase closes.

## Next Phase Readiness

- Not ready -- this plan is now paused at Task 4 of 4 (Tasks 1-3 approved/complete; Task 4 drafted this session, awaiting human review). `control/vla_bridge/FINDINGS.md` exists and is ready to read, but its `checkpoint:human-verify` gate requires an explicit human "approved" (or corrections) before Phase 11 closes.
- Once Task 4 is approved, this SUMMARY should be updated to `status: complete` and the phase can close per the plan's `<verification>`/`<success_criteria>`.
- FINDINGS.md's own go/no-go recommendation (GO, conditional on fixing the observation-resend-per-tick latency bug) is the concrete input for scoping the next milestone phase's first engineering task.

---
*Phase: 11-vla-hardware-connection*
*Completed: N/A (paused at Task 4 checkpoint, pending human review, 2026-09-24)*

## Self-Check: PASSED

- FOUND: control/detect_devices.py
- FOUND: control/test_detect_devices.py
- FOUND: control/run_vla_episode.py
- FOUND: control/vla_bridge/robot_client.py
- FOUND: control/vla_bridge/FINDINGS.md
- FOUND: control/outputs/11-05-retry-20260924-120530/episode.jsonl
- FOUND: control/outputs/11-05-retry-20260924-120530/termination.json
- FOUND: .planning/phases/11-vla-hardware-connection/11-05-SUMMARY.md
- FOUND: fa99468 (Task 2 commit) in git log
- FOUND: 434b01a (Task 4 FINDINGS.md commit) in git log
