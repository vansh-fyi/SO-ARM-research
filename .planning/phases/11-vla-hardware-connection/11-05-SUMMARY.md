---
phase: 11-vla-hardware-connection
plan: 05
subsystem: vla-hardware
tags: [vla-hardware, hardware-in-the-loop, findings, go-no-go, checkpoint-paused, camera-identity]

# Dependency graph
requires:
  - phase: 11-vla-hardware-connection
    provides: "Plan 11-04's BridgeActionSource/robot_client.py + StereoSplitCamera, Plan 11-01's safety_validator, Plan 11-02's IOLogger/run_vla_episode.py, Plan 11-03's selected checkpoint + policy_server_launch.md"
provides: []
affects: []

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Camera semantic names (wrist/overhead) are assigned positionally from --camera flag order (or probe order), not from a fixed index->name table -- physical USB enumeration order is not guaranteed stable across sessions on this rig"

key-files:
  created: []
  modified:
    - control/run_vla_episode.py
    - control/test_run_vla_episode.py

key-decisions:
  - "Task 1 (e-stop hardware-in-the-loop) approved after a second attempt with explicit --camera flags; the first attempt silently failed (ran to max_steps, Ctrl+C never registered by the process) because with neither real camera connected, cv2 opened the MacBook's built-in FaceTime camera at index 0 and treated it as the wrist camera"
  - "Discovered during Task 1 verification: physical camera index assignment is inverted from this project's long-standing assumption -- AR0144 stereo is at cv2 index 0 and IMX335 wrist is at cv2 index 1 this session, not the reverse. Confirmed by resolution (2560x720 vs standard) and a physical cover-the-lens test"
  - "Fixed run_vla_episode.py's DEFAULT_CAMERA_NAMES: it was a hardcoded {0: wrist, 1: overhead} table applied regardless of --camera flag order, so the script's own comment promising 'override at the CLI if indices have shifted' was not actually true. Replaced with positional _build_camera_names() so --camera order now genuinely controls labeling"
  - "Added --stereo-camera-index CLI flag: vla_bridge.robot_client.connect_bridge()'s stereo_camera_index (which device the VLA bridge's camera2/camera3 split opens) had no CLI override and defaulted to 1 -- given this session's inverted camera mapping, running Task 2 without this fix would have opened the wrist camera as if it were the AR0144 stereo camera, corrupting the split-frame data fed to the model"

requirements-completed: []

coverage:
  - id: D1
    description: "E-stop hardware-in-the-loop verification: Ctrl+C mid-episode halts the real SO-ARM101, moves it back to start position, exits cleanly, and records reason: keyboard_interrupt in termination.json"
    requirement: "VLAHW-02"
    verification: []
    human_judgment: true
    rationale: "Requires physically operating the real SO-ARM101 and observing its motion/e-stop behavior firsthand -- no automated test can substitute for hands-on hardware verification of a physical safety-critical behavior. APPROVED this session (see Accomplishments) after a first silent-failure attempt was independently diagnosed and a second attempt genuinely interrupted early."
  - id: D2
    description: "One full VLA-driven episode (Colab SmolVLA PolicyServer over ngrok TCP tunnel -> BridgeActionSource -> safety_validator -> real robot -> IOLogger) recorded end-to-end with real (non-scripted) episode.jsonl, camera frame sequences, and termination.json"
    requirement: "VLAHW-04"
    verification: []
    human_judgment: true
    rationale: "Requires a live Colab PolicyServer, an active ngrok tunnel, the physical SO-ARM101 and cameras, and a human judgment call on pick-up success/failure (no automated vision-based success detection exists yet). Not yet attempted -- plan is paused at this task's checkpoint pending the human's live run."
  - id: D3
    description: "control/vla_bridge/FINDINGS.md go/no-go write-up synthesizing Task 1/Task 2's actual observed outcomes, lerobot#2210 status, and the D-05 no-reasoning-trace framing"
    requirement: "VLAHW-05"
    verification: []
    human_judgment: true
    rationale: "The write-up is a single-observer research artifact requiring human review to confirm every claim traces back to Task 1/Task 2's actual observed outcomes, not invented or assumed. Not yet started -- depends on Task 2."

duration: 0min (paused at Task 2 checkpoint)
completed: 2026-09-22
status: incomplete
---

# Phase 11 Plan 05: Hardware-in-the-Loop E-Stop, Live Episode, and Findings Summary

**Task 1's e-stop verification approved on real hardware after diagnosing a silent first-attempt failure caused by a camera-identity bug; two related camera-index bugs found and fixed in `run_vla_episode.py` before presenting Task 2's live-episode checkpoint.**

## Performance

- **Duration:** 0 min of automated work this session (task 1 was human-verification only; the code fix below was a short inline deviation)
- **Started:** 2026-09-21 (Task 1 checkpoint first presented)
- **Completed:** N/A -- still paused, now at Task 2
- **Tasks:** 1/3 completed (Task 1 approved; Task 2 and Task 3 remain `checkpoint:human-verify`, `gate="blocking"`)
- **Files modified:** 2 (`control/run_vla_episode.py`, `control/test_run_vla_episode.py` -- a deviation fix, not a plan task)

## Accomplishments

- **Task 1 (E-stop hardware-in-the-loop verification): APPROVED.**
  - First attempt (`outputs/estop_check_001/`) silently failed: `termination.json` recorded `{"reason": "max_steps_reached", "steps_completed": 200}` -- the Ctrl+C was never registered; the script simply ran to completion. Root-caused live: `_probe_camera_indices()` has no way to verify camera identity, and with neither real camera physically connected at that point, it silently opened the MacBook's built-in FaceTime camera at index 0 and treated it as "the wrist camera." This is a real, previously-undocumented gap (not flagged in `11-RESEARCH.md` or any prior PLAN.md).
  - Both real cameras were then physically connected and their identity verified by resolution plus a physical cover-the-lens test:
    - Index 0 = AR0144 stereo (confirmed via its distinctive 2560x720 frame)
    - Index 1 = IMX335 wrist camera (confirmed bright/uncovered when index 2 was covered)
    - Index 2 = MacBook's built-in webcam (confirmed dark when covered)
    - This is the **inverse** of the project's long-standing assumption (`DEFAULT_CAMERA_NAMES = {0: "wrist", 1: "overhead"}` in `run_vla_episode.py`, and `stereo_camera_index=1` default in `robot_client.connect_bridge()`) -- physical USB enumeration order does not match that assumption on this session's hardware.
  - Second attempt (`outputs/estop_check_002/`), run with explicit `--camera 1 --camera 0` to avoid the built-in camera: `termination.json` recorded `{"reason": "keyboard_interrupt", "steps_completed": 33}` -- a genuine interrupt, stopped well short of the 200-step max. Terminal log showed a clean `^C` -> `"Interrupted -- returning to start position..."` -> clean disconnect, no traceback. Human confirmed the arm physically moved back to its start position. Both `camera_wrist/` and `camera_overhead/` output directories were populated with 35 real frames each.
  - **Caveat carried forward, not silently accepted:** because `DEFAULT_CAMERA_NAMES` at the time was a fixed index->name table (see Deviations below), the `camera_wrist/`/`camera_overhead/` directories from this second attempt were almost certainly mislabeled relative to physical reality (whatever was captured at index 0 -- the AR0144 stereo camera -- would have been saved under the `wrist` label, and index 1's real wrist frames under `overhead`), even though `--camera 1 --camera 0` was passed in the "corrected" order. This did not affect the e-stop behavior itself (which is index-agnostic), only the recorded artifact labeling from this specific run. Not re-run, since Task 1's actual pass/fail criteria (four e-stop behaviors) do not depend on camera labeling -- flagged here for the record and fixed going forward (see Deviations).

- **Diagnosed and fixed two real camera-index bugs before presenting Task 2's checkpoint** (see Deviations below): the DEFAULT_CAMERA_NAMES mislabeling bug, and a missing `--stereo-camera-index` CLI override for `vla_bridge.robot_client.connect_bridge()`.

- **Prepared Task 2's checkpoint** (Full VLA-driven episode) with corrected camera-index guidance reflecting this session's verified physical mapping, and a newly-discovered risk flagged for the human's awareness (see Checkpoint below) -- execution now paused there per plan design; Task 2 was not run.

## Task Commits

1. **Task 1: E-stop hardware-in-the-loop verification** -- no code commit (human-verification only); approved this session based on live evidence described above.
2. **[Deviation, not a plan task] Fixed camera-index CLI plumbing** -- `b4513dc` (fix)

**Plan metadata:** pending (this SUMMARY's own commit)

## Files Created/Modified

- `control/run_vla_episode.py` -- replaced the hardcoded `DEFAULT_CAMERA_NAMES` index->name table with `_build_camera_names()` (positional mapping from `--camera` order); added `--stereo-camera-index` CLI flag threaded to `connect_bridge()`.
- `control/test_run_vla_episode.py` -- updated the two existing `connect_bridge` fakes to accept the new `stereo_camera_index` kwarg; added tests for `_build_camera_names`'s positional mapping and for `--stereo-camera-index`'s plumbing.

## Decisions Made

See `key-decisions` in frontmatter. Summary: Task 1 approved based on a second, genuinely-interrupted attempt after diagnosing the first attempt's silent failure; two camera-index bugs (recording-label mismatch, stereo-split device mismatch) fixed inline as Rule 1/Rule 3 deviations before Task 2 could be safely run.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] `DEFAULT_CAMERA_NAMES` ignored `--camera` flag order, contradicting its own documented intent**
- **Found during:** Preparing Task 2's checkpoint (investigating this session's camera-index finding from Task 1 per the resume instructions)
- **Issue:** `run_vla_episode.py` had a hardcoded module-level `DEFAULT_CAMERA_NAMES = {0: "wrist", 1: "overhead"}`, applied to every run regardless of what order `--camera` indices were passed. Its own comment claimed "override at the CLI if indices have shifted," but the code never actually built the name mapping from `--camera`'s order -- it only used `--camera` to decide which indices to open (`caps`), then always applied the fixed table. This meant Task 1's second attempt (`--camera 1 --camera 0`, intended to fix labeling) did NOT actually relabel anything -- index 0's frames (the AR0144 stereo camera) were still recorded under `camera_wrist/`.
- **Fix:** Added `_build_camera_names(camera_indices)`, which builds the index->name mapping positionally from the order `--camera` indices are given (or probe order, if `--camera` omitted): first index -> "wrist", second -> "overhead". `main()` now calls this once and threads the result into both `IOLogger` and `run_episode`, replacing all `DEFAULT_CAMERA_NAMES` usage.
- **Files modified:** `control/run_vla_episode.py`
- **Verification:** Added `test_build_camera_names_assigns_positionally_not_by_raw_index`, `test_build_camera_names_default_order_matches_prior_default_table`, `test_build_camera_names_drops_extra_indices_beyond_known_names` in `control/test_run_vla_episode.py`. Full suite run: `45 passed`.
- **Committed in:** `b4513dc`

**2. [Rule 3 - Blocking] No CLI override existed for the VLA bridge's stereo-camera device index**
- **Found during:** Same investigation as above, cross-checking `vla_bridge/stereo_camera.py` and `vla_bridge/robot_client.py` against this session's verified camera mapping (per the explicit resume instruction to "check the actual code, don't assume")
- **Issue:** `robot_client.connect_bridge()` accepts a `stereo_camera_index` parameter (default `1`) controlling which physical device `StereoSplitCamera` opens for the VLA bridge's `camera2`/`camera3` split observation. `run_vla_episode.py`'s call site never passed this parameter, and no CLI flag existed to set it. Given this session's verified inverted mapping (AR0144 stereo now at index 0, not 1), running Task 2 as originally documented would have opened the IMX335 wrist camera (1920x1080) at the stereo split's expected index, expecting a 2560x720 frame -- `frame[:, 0:1280]`/`frame[:, 1280:2560]` would not raise (numpy silently clips slices to the actual 1920px width) but would silently feed the wrong camera's data, mis-cropped, into the model's `camera2`/`camera3` inputs during Task 2's live run -- a genuine blocking correctness issue for VLAHW-04, not just a labeling cosmetic issue like Deviation 1.
- **Fix:** Added `--stereo-camera-index` CLI argument (default `1`, preserving prior behavior when unset), threaded through to `connect_bridge(..., stereo_camera_index=args.stereo_camera_index)`.
- **Files modified:** `control/run_vla_episode.py`, `control/test_run_vla_episode.py` (updated the two existing `connect_bridge` test fakes to accept the new kwarg, without which they would have raised `TypeError` on the next test run once the call site changed)
- **Verification:** Added `test_stereo_camera_index_flag_threads_through_to_connect_bridge`; confirmed the default-unset path still passes `stereo_camera_index == 1` via an assertion added to the existing `test_run_vla_episode_selects_bridge_source_when_server_address_given`. Full suite run: `45 passed`.
- **Committed in:** `b4513dc`

---

**Total deviations:** 2 auto-fixed (1 bug, 1 blocking) -- both directly required for Task 2 to run correctly against this session's actual (inverted) physical camera mapping.
**Impact on plan:** Both fixes are narrow and non-architectural (no new files, no schema/service changes) -- confined to CLI plumbing and a naming-assignment helper inside `run_vla_episode.py`. No scope creep beyond what Task 2's correctness required.

## Issues Encountered

- **Unresolved, flagged for the human in Task 2's checkpoint (not auto-fixed -- Rule 4 territory):** Reading `vla_bridge/robot_client.py`'s `connect_bridge()`/`_wire_stereo_split_cameras()` closely, `camera1` (the wrist camera slot the selected checkpoint `victorvanhalst/smolvla_so101_cube` expects per `policy_server_launch.md`'s 3-camera mapping table) is **never actually wired into the observation dict sent to the VLA bridge**. `SOFollowerRobotConfig.cameras` defaults to an empty dict and `run_vla_episode.py`'s `robot_config` construction never populates it; `_wire_stereo_split_cameras()` only patches `get_observation()` to add `camera2`/`camera3` (the AR0144 stereo split) -- it does not touch `camera1`. This directly contradicts `11-04-SUMMARY.md`'s claim that the bridge gives the checkpoint "three genuinely distinct real camera views (wrist, AR0144-left, AR0144-right)" -- confirmed by grep: no occurrence of `"camera1"` or `cameras=` anywhere in `robot_client.py`, `run_vla_episode.py`, or `test_robot_client.py`. This is a genuine, unresolved gap that could mean the live episode's `camera1` input is silently missing or zero-valued during Task 2's run. Fixing it properly requires deciding how to source `camera1` without a second concurrent `cv2.VideoCapture` open on the same physical wrist-camera index already used by `run_vla_episode.py`'s own `caps`/IOLogger recording path (per `stereo_camera.py`'s own docstring, "most webcam drivers reject a second concurrent open of the same index") -- an architectural decision (Rule 4), not something safely auto-fixed inline. **Surfaced explicitly in Task 2's checkpoint below rather than silently patched.**

## User Setup Required

None from this session directly, but Task 2 requires the human to:
1. Measure the red cube per the plan's Task 2 `<how-to-verify>` step 1 (≤78mm graspable width, off the base's forward centerline).
2. Start the Colab notebook per `control/vla_bridge/policy_server_launch.md`.
3. Run `run_vla_episode.py` against the real robot and Colab bridge, using this SUMMARY's corrected camera flags (see the checkpoint returned alongside this SUMMARY).
4. Report the episode's outcome (success/failure, termination reason, lerobot#2210 status) per the plan's resume-signal.

## Next Phase Readiness

- Not ready -- this plan is now paused at Task 2 of 3 (Task 1 approved this session). Task 2 (full VLA-driven episode) has not yet been attempted; its checkpoint is being returned to the human now, with corrected camera-index guidance and the unresolved `camera1`-wiring gap flagged explicitly.
- Task 3 (`FINDINGS.md` write-up) depends on both Task 1 (now available) and Task 2's real, human-reported outcome, and cannot be drafted yet.
- This SUMMARY will be superseded by a complete version once Task 2 and Task 3 are both approved and `control/vla_bridge/FINDINGS.md` exists.

---
*Phase: 11-vla-hardware-connection*
*Completed: N/A (paused at Task 2 checkpoint, 2026-09-22)*

## Self-Check: PASSED

- FOUND: control/run_vla_episode.py
- FOUND: control/test_run_vla_episode.py
- FOUND: .planning/phases/11-vla-hardware-connection/11-05-SUMMARY.md
- FOUND: b4513dc (fix commit) in git log
</content>
