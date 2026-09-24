---
phase: 11-vla-hardware-connection
verified: 2026-09-24T07:34:11Z
status: passed
score: 5/5 must-haves verified
behavior_unverified: 0
overrides_applied: 0
---

# Phase 11: VLA Hardware Connection Verification Report

**Phase Goal:** An SO-101-native joint-action VLA (SmolVLA) drives the real SO-ARM101 over the existing `control/` LeRobot bridge, through an explicit safety validator, with complete per-inference-step I/O captured, and at least one full observed run recorded end-to-end to inform the go/no-go decision on later milestone phases.
**Verified:** 2026-09-24T07:34:11Z
**Status:** passed
**Re-verification:** No — initial verification (retroactive, gating phase transition)

## Goal Achievement

### Observable Truths

| # | Truth (ROADMAP Success Criterion) | Status | Evidence |
|---|---|---|---|
| 1 | A harness loads an SO-101-native joint-action VLA and calls it with real wrist+overhead camera frames and live joint-state, against an explicitly documented action contract | ✓ VERIFIED | `control/vla_bridge/action_contract.py` documents JOINT_ORDER/ACTION_UNITS/ACTION_MODE (DEGREES arm + RANGE_0_100 gripper, absolute, 6-DOF), resolved directly against installed `lerobot==0.6.1` source, pinned by `test_action_contract.py`. `control/vla_bridge/robot_client.py`'s `connect_bridge()`/`BridgeActionSource` feed real `read_positions(robot)` joint state and real camera frames (wrist via `wrist_camera_index` wiring, AR0144 split-stereo via `StereoSplitCamera`) into the SmolVLA bridge. Confirmed against the recorded episode: `episode.jsonl` step 0/52 `joint_state` and `camera_frames` fields contain real, non-synthetic values and real PNG paths (`camera_wrist/000052.png`, 1920x1080, verified as real PNG files on disk). |
| 2 | VLA output actions pass a safety validator (limits, per-step/velocity caps, stale/malformed/NaN rejection, e-stop, comms-failure handling) before reaching the robot via the existing `SO101Follower` interface, with no new hand-written raw serial/register code | ✓ VERIFIED | `run_vla_episode.py::run_episode()` line 199-208 routes every candidate action through `safety_validator.validate_action()` before the sole `robot.send_action()` call; `robot_client.py::pop_validated_action()`/`BridgeActionSource` never call `send_action` themselves (validation gate and execution gate are structurally separate — confirmed by reading the code, not just the docstring claim). `robot.send_action()` is the pre-existing `SO101Follower` motor-bus method; `grep` for `serial\.Serial\|write_register\|ser\.write` across `control/vla_bridge/*.py` and `run_vla_episode.py`/`detect_devices.py` found zero matches — no new hand-written serial/register code. NaN/inf rejection (`safety_validator.py` step 1) and absolute joint-limit clamp (step 2) are unmodified. **Per-step/velocity/staleness caps were intentionally loosened this session** (commit `d8c2785`, quick task `260924-e3d`): `MAX_RELATIVE_TARGET_DEG` 5°→40° (gripper 15→60), `MAX_VELOCITY_DEG_PER_S` 30→240°/s (gripper 50→200), `STALE_OBSERVATION_S` 1.0s→10.0s, `STALE_ACTION_S` 3.0s→30.0s — confirmed via `git show d8c2785`, a real, documented, human-approved tuning decision to let a genuine VLA action clear validation in this physically-empty test rig, not a bypass: the validator still runs, still gates every action, and the two untouched checks (NaN/inf, absolute limits) are exactly the ones that protect against a truly malformed/out-of-envelope command. E-stop and comms-failure handling: `run_episode()`'s `KeyboardInterrupt` handler always returns to start position before disconnect (`finally` block, line 239-244); `send_action` write failures (`ConnectionError`/`RuntimeError`) are caught per-tick without crashing the loop (line 209-210); `BridgeActionSource.get_action()` catches `grpc.RpcError`/`ConnectionError`/`RuntimeError` and holds position rather than crashing. Full `control/` test suite (93 tests) passes. |
| 3 | Every inference step's complete I/O (camera frames w/ per-camera timestamps, joint state, instruction, raw model output, validated action, executed action, latency, model version) is written to a durable, timestamped log | ✓ VERIFIED | `control/vla_bridge/io_logger.py`'s `IOLogger.write_step()` writes exactly this schema per JSONL record (confirmed field-by-field against the module source), with `flush()` after every write for crash-durability. `capture_camera_frame()` timestamps each camera independently (not a shared timestamp). Directly confirmed against the real recorded file `control/outputs/11-05-retry-20260924-120530/episode.jsonl`: 60 lines (one per step), every record contains all of `step`, `timestamp_utc`, `instruction`, `camera_frames` (wrist+overhead, each with its own `captured_at_utc`), `joint_state`, `raw_model_output`, `validated_action`, `validator_flags`, `executed_action`, `latency_ms`, `model_version`. |
| 4 | At least one full episode — task prompt to termination — runs end-to-end on the physical robot, producing a saved synchronized video and a matching per-step I/O log, plus recorded termination reason and success/failure outcome | ✓ VERIFIED (literal text met; see reasoning) | `control/outputs/11-05-retry-20260924-120530/termination.json` = `{"reason": "max_steps_reached", "steps_completed": 60}` (read directly from disk, matches FINDINGS.md's claim exactly). `episode.jsonl` has 60 matching per-step records. `camera_wrist/`/`camera_overhead/` each contain 60 real, non-corrupt 1920x1080 PNGs (spot-checked with `file`), one per JSONL record's `camera_frames` path — this project's documented D-04 design decision (`11-CONTEXT.md`) explicitly defines the frame-storage format as "image/video files on disk, referenced by path," consistent with the pre-existing `record_episode.py` convention, so a per-step timestamped PNG sequence satisfies "saved synchronized video" as designed, not as a shortcut invented after the fact. Step 52's real executed action (only 1 of 60 steps) was independently re-verified against raw JSON: `shoulder_pan` 2.73°→-5.61°, `shoulder_lift` -106.15°→-72.71°, `elbow_flex` 91.43°→54.06°, `gripper` 95.76%→49.81% — exact match to FINDINGS.md's cited numbers. Tick-timing spot check (`timestamp_utc` deltas) gives min 10.9s / max 28.3s / avg 13.4s, consistent with FINDINGS.md's "~11-20s/tick" claim. **My own reading of whether this satisfies the criterion's literal text:** yes. The criterion requires the episode to "run end-to-end," produce a saved video + matching I/O log + termination reason + success/failure outcome — it does not require a successful task completion or a minimum action-execution rate. All four literal requirements are met: (a) ran end-to-end (60/60 steps, no crash), (b) saved synchronized frame sequence + matching JSONL, (c) termination reason recorded (`max_steps_reached`), (d) outcome recorded (FINDINGS.md §2 explicitly states "not achieved as a completed pick-and-place... arm moved but the episode ended at max_steps mid-approach, not on a goal-met condition" — a failure outcome, genuinely recorded, not omitted). The 1/60 real-action yield is a real, diagnosed limitation (a tick-latency bug, root-caused in FINDINGS.md §5) that affects *how well* the episode ran, not *whether* it ran end-to-end with a recorded outcome — it is correctly treated as follow-up engineering work, not grounds to say no episode was recorded. |
| 5 | A short findings write-up synthesizes observations (incl. any upstream issues, e.g. lerobot#2210) into an explicit go/no-go recommendation | ✓ VERIFIED | `control/vla_bridge/FINDINGS.md` exists (committed `434b01a`), states lerobot#2210 did not reproduce, cites 3 real bridge bugs found+fixed this session (commits `343fa28`/`e55aa53`/`a3e16c4`, all confirmed to exist in git log), gives a step-by-step account of the recorded episode with numbers independently re-verified above, and closes with an explicit **GO, conditional on** fixing the diagnosed observation-resend-per-tick latency bug. Approved by the human 2026-09-24 (per `11-05-SUMMARY.md` frontmatter `key-decisions`/`Performance.Completed`) with an explicit caveat that the proposed fix needs further design before implementation. This caveat does not invalidate the criterion: a findings document identifying that its own proposed fix needs more design work is doing exactly what a findings document is for (surfacing scoped follow-up work), not failing to be one. |

**Score:** 5/5 truths verified (0 present-but-behavior-unverified)

### Required Artifacts

| Artifact | Expected | Status | Details |
|---|---|---|---|
| `control/vla_bridge/action_contract.py` | Documented action contract | ✓ VERIFIED | Exists, substantive, wired into safety_validator and run_vla_episode.py, pinned by tests |
| `control/vla_bridge/safety_validator.py` | Safety validator gating every action | ✓ VERIFIED | Exists, substantive, wired into every send_action call path; caps intentionally tuned (see Truth 2), core rejection logic untouched |
| `control/vla_bridge/io_logger.py` | JSON Lines per-step I/O logger | ✓ VERIFIED | Exists, substantive, wired into run_vla_episode.py; real output confirmed on disk |
| `control/run_vla_episode.py` | Real-hardware episode harness | ✓ VERIFIED | Exists, substantive, wired to both ScriptedActionSource (dry-run) and BridgeActionSource (real VLA) |
| `control/vla_bridge/robot_client.py` | Bridge wrapper (connect_bridge/pop_validated_action/BridgeActionSource) | ✓ VERIFIED | Exists, substantive; deliberately avoids `RobotClient.control_loop_action()`/`control_loop()` (unvalidated internal send_action) |
| `control/vla_bridge/stereo_camera.py` | Split-stereo camera feed | ✓ VERIFIED | Exists, substantive, wired into connect_bridge() |
| `control/detect_devices.py` | Device auto-discovery | ✓ VERIFIED | Exists, substantive, wired as run_vla_episode.py's default source of PORT/ROBOT_ID/camera indices |
| `control/vla_bridge/FINDINGS.md` | Go/no-go findings write-up | ✓ VERIFIED | Exists, substantive, numbers independently re-verified against raw episode data, human-approved |
| `control/outputs/11-05-retry-20260924-120530/episode.jsonl` + `termination.json` | Recorded full episode | ✓ VERIFIED | Real files on disk, 60/60 steps, contents cross-checked against FINDINGS.md's claims |

### Key Link Verification

| From | To | Via | Status | Details |
|---|---|---|---|---|
| `run_vla_episode.py::run_episode()` | `safety_validator.validate_action()` | direct call before the only `robot.send_action()` | ✓ WIRED | Every candidate action (scripted or bridge) passes through validation first |
| `robot_client.py::BridgeActionSource` | `safety_validator.validate_action()` (via `pop_validated_action`) | direct call | ✓ WIRED | Never calls `send_action` itself; caller does, after validation |
| `robot_client.py::connect_bridge()` | `stereo_camera.py::StereoSplitCamera` | `_wire_stereo_split_cameras()` monkey-patches `get_observation()` | ✓ WIRED | Confirmed via `test_robot_client.py`'s wiring assertion and code read |
| `run_vla_episode.py::main()` | `detect_devices.py`'s `device_map.json` | `_load_device_map()` | ✓ WIRED | CLI args optional, fall back to device_map.json, hard error if neither available |
| `run_vla_episode.py` | `io_logger.IOLogger` | `write_step()`/`capture_camera_frame()` called every tick | ✓ WIRED | Confirmed real output on disk (episode.jsonl + PNGs) |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|---|---|---|---|
| Full `control/` test suite passes | `.venv/bin/python -m pytest -q` (run once) | `93 passed in 3.65s` | ✓ PASS |
| Safety-validator cap change is exactly as documented (loosened, not gutted) | `git show d8c2785 -- control/vla_bridge/safety_validator.py` | NaN/inf + absolute-limit logic untouched; only the 4 numeric constant blocks changed, matching the commit message 1:1 | ✓ PASS |
| Recorded episode termination matches FINDINGS.md's claim | `cat termination.json` | `{"reason": "max_steps_reached", "steps_completed": 60}` | ✓ PASS |
| Episode step count matches claim | `wc -l episode.jsonl` | 60 | ✓ PASS |
| Step 52's real action matches FINDINGS.md's cited joint deltas | Read `episode.jsonl` line 53, diff `joint_state` vs `executed_action` | Exact match (shoulder_pan/shoulder_lift/elbow_flex/gripper deltas all match cited values) | ✓ PASS |
| model_version step-status counts match FINDINGS.md's step-by-step narrative | Python tally of `model_version` substrings across all 60 records | 57 stale, 2 no-action, 1 fresh (`@unknown`) — matches FINDINGS.md's step 0/1-50/51/52/53-59 breakdown exactly | ✓ PASS |
| Tick latency matches FINDINGS.md's "~11-20s/tick" claim | `timestamp_utc` delta computation across all 60 records | min 10.9s, max 28.3s, avg 13.4s | ✓ PASS |
| No new hand-written raw serial/register code | `grep -rn "serial\.Serial\|write_register\|read_register\|ser\.write"` across all vla_bridge/*.py + run_vla_episode.py + detect_devices.py | 0 matches | ✓ PASS |
| All commit hashes cited across the 5 SUMMARY.md files are real | `git cat-file -e` for 17 cited hashes | All 17 present in repo history | ✓ PASS |
| Camera frame files are real, correctly-sized images (not stubs/corrupt) | `file camera_wrist/000000.png camera_overhead/000000.png camera_wrist/000052.png` | All 3: valid PNG, 1920x1080, 8-bit RGB | ✓ PASS |

### Requirements Coverage

| Requirement | Source Plan(s) | Description | Status | Evidence |
|---|---|---|---|---|
| VLAHW-01 | 11-01, 11-03, 11-04, 11-05 | SO-101-native VLA loaded, real observations, documented action contract | ✓ SATISFIED | action_contract.py + robot_client.py + stereo_camera.py + detect_devices.py camera1 fix, all confirmed |
| VLAHW-02 | 11-01, 11-02, 11-04, 11-05 | Safety validator gates every action; existing SO101Follower interface; use_degrees ambiguity resolved | ✓ SATISFIED | safety_validator.py + gating confirmed at every send_action call site; action_contract.py resolves the DEGREES/RANGE_0_100 ambiguity |
| VLAHW-03 | 11-02, 11-04 | Complete per-step I/O captured to durable log | ✓ SATISFIED | io_logger.py + real episode.jsonl confirmed field-complete |
| VLAHW-04 | 11-05 | At least one full episode recorded end-to-end (video + I/O log + termination + outcome) | ✓ SATISFIED | Real episode.jsonl/termination.json/PNG frames on disk, cross-verified above |
| VLAHW-05 | 11-05 | Findings write-up with go/no-go recommendation | ✓ SATISFIED | FINDINGS.md exists, numbers verified, human-approved 2026-09-24 |

Note: `.planning/REQUIREMENTS.md`'s checkboxes/status column still show VLAHW-01..05 as unchecked/"Pending" — this is expected administrative staleness (this VERIFICATION.md is the gate that triggers that update during phase closure), not evidence against satisfaction.

### Anti-Patterns Found

None. Searched all Phase-11-created/modified files under `control/vla_bridge/`, `control/run_vla_episode.py`, and `control/detect_devices.py` for `TBD|FIXME|XXX|TODO|HACK|PLACEHOLDER` — the only match was a substring false-positive (`usbmodemXXXX` in a help string, matching the `XXX` pattern, not a debt marker). No stub returns, no hardcoded empty data flowing to output, no console.log-only implementations found.

### Human Verification Required

None. Per this task's explicit scope, the hardware-in-the-loop human-judgment checkpoints (e-stop physical verification, cube-pick outcome judgment, FINDINGS.md approval) were already recorded as human-verified in `11-05-SUMMARY.md` and `FINDINGS.md`, and are not re-litigated here. All 5 ROADMAP success criteria are independently confirmed against actual code and actual recorded data, not SUMMARY.md narrative alone.

### Gaps Summary

No gaps. All 5 ROADMAP success criteria are verified against the codebase and the real recorded episode data (not SUMMARY.md claims alone):

- Criterion 2's safety-validator cap loosening is confirmed intentional, documented, and scoped (commit `d8c2785`) — the validator still runs and still gates every action; only tunable numeric thresholds changed, not the NaN/inf or absolute-limit rejection logic.
- Criterion 4's low real-action yield (1/60) is a genuine, diagnosed limitation (tick-latency bug, root-caused in FINDINGS.md §5) that does not prevent the criterion's literal text from being met — the episode ran end-to-end, produced a saved synchronized frame sequence + matching I/O log, and recorded both a termination reason and a (failure) outcome. This is correctly scoped as follow-up engineering work in FINDINGS.md, not a phase-goal failure.
- Criterion 5's FINDINGS.md caveat ("the fix needs more design") is itself evidence the findings document did its job (surfacing scoped follow-up work), not evidence of incompleteness.

Phase 11's goal — SmolVLA driving the real SO-ARM101 over the existing LeRobot bridge, gated by a safety validator, with complete I/O logging and at least one full recorded episode plus a findings write-up — is achieved in the codebase, not just claimed in the SUMMARY.md files.

---

_Verified: 2026-09-24T07:34:11Z_
_Verifier: Claude (gsd-verifier)_
