---
phase: quick-260927-he3
plan: 01
subsystem: infra
tags: [ffmpeg, avfoundation, camera, stereo, vla-bridge]

# Dependency graph
requires:
  - phase: 11-vla-hardware-connection
    provides: StereoSplitCamera / detect_devices.py camera-identity detection built on the AR0144's captured resolution constants
provides:
  - AR0144 stereo camera capture switched from persistently-frozen 2560x720 to confirmed-working 1280x360 (640x360 per half)
  - Updated docstrings/comments in stereo_camera.py and detect_devices.py describing the current capture resolution accurately
  - Updated policy_server_launch.md Camera mapping section, crop-column table, and Step 5 launch command for the new crop
affects: [12-bridge-tick-latency-fix, 13-camera-resolution-unification, vla_bridge]

# Tech tracking
tech-stack:
  added: []
  patterns: []

key-files:
  created: []
  modified:
    - control/vla_bridge/stereo_camera.py
    - control/test_stereo_camera.py
    - control/detect_devices.py
    - control/vla_bridge/policy_server_launch.md

key-decisions:
  - "Switched AR0144 stereo capture resolution from 2560x720 (confirmed persistently frozen/stuck on this machine via two independent live tests) to 1280x360 (confirmed live via raw ffmpeg testing as a working, correctly-laid-out fallback)."

patterns-established: []

requirements-completed: []

coverage:
  - id: D1
    description: "AR0144 stereo camera captured at 1280x360 (640x360 per half) instead of the frozen 2560x720 mode"
    verification:
      - kind: unit
        ref: "control/test_stereo_camera.py::test_read_left_and_read_right_return_correct_non_swapped_halves"
        status: pass
      - kind: other
        ref: "python3 -c \"from vla_bridge.stereo_camera import STEREO_WIDTH, STEREO_HEIGHT, SPLIT_COL; assert (STEREO_WIDTH, STEREO_HEIGHT, SPLIT_COL) == (1280, 360, 640)\""
        status: pass
    human_judgment: false
  - id: D2
    description: "detect_devices.py descriptive comments no longer cite hardcoded 2560x720 as the current resolution; detection logic (resolve_cameras/_verify_stereo_via_ffmpeg) unchanged since it already reads live constants"
    verification:
      - kind: unit
        ref: "control/test_detect_devices.py (full suite, references imported STEREO_WIDTH/STEREO_HEIGHT symbols not literals)"
        status: pass
    human_judgment: false
  - id: D3
    description: "stereo_calibration.py and depth_camera.py (and their tests) re-confirmed free of hardcoded resolution assumptions tied to the real AR0144 capture"
    verification:
      - kind: other
        ref: "grep -n '2560\\|1280\\|720' control/vla_bridge/stereo_calibration.py control/vla_bridge/depth_camera.py control/test_depth_camera.py control/test_stereo_calibration.py -- all hits confirmed unrelated synthetic fixtures or runtime .shape reads"
        status: pass
    human_judgment: false
  - id: D4
    description: "policy_server_launch.md Camera mapping section, crop-column table, and Step 5 launch command updated to 1280x360/640x360 with [0:640]/[640:1280] crops"
    verification:
      - kind: other
        ref: "grep checks for camera2/camera3 width: 640, height: 360 and [0:640]/[640:1280] crop columns in policy_server_launch.md"
        status: pass
    human_judgment: false
  - id: D5
    description: "Full control/ pytest suite passes after the change"
    verification:
      - kind: unit
        ref: "control/ full pytest suite (153 tests)"
        status: pass
    human_judgment: false

duration: 12min
completed: 2026-09-27
status: complete
---

# Quick Task 260927-he3: Fix AR0144 stereo camera resolution switch Summary

**AR0144 stereo camera capture switched from the persistently-frozen 2560x720 mode to a confirmed-working 1280x360 (640x360 per half), with all dependent docstrings/comments/docs and test assertions updated to match.**

## Performance

- **Duration:** ~12 min
- **Started:** 2026-09-27T07:01:00Z (approx)
- **Completed:** 2026-09-27T07:13:46Z
- **Tasks:** 3
- **Files modified:** 4

## Accomplishments
- `stereo_camera.py`'s `STEREO_WIDTH`/`STEREO_HEIGHT`/`SPLIT_COL` constants changed to 1280/360/640 (was 2560/720/1280), with a rewritten module docstring documenting the full frozen-2560x720 diagnosis and the 1280x360 confirmation
- `test_stereo_camera.py`'s synthetic frame fixture and shape assertions updated to 360x1280/640x360 to match
- `detect_devices.py`'s descriptive prose no longer cites 2560x720 as a hardcoded fact -- rephrased to reference the live `STEREO_WIDTH`/`STEREO_HEIGHT` constants; its actual detection logic (`resolve_cameras()`, `_verify_stereo_via_ffmpeg()`) required zero code changes, confirming the plan's prediction
- `policy_server_launch.md`'s Camera mapping section, crop-column table, and Step 5 `--robot.cameras` launch command updated to the new 640x360/[0:640]/[640:1280] values
- Re-verified live (not just trusted from planning) that `stereo_calibration.py`, `depth_camera.py`, and their test files contain no hardcoded resolution assumption tied to the real AR0144 capture -- all `2560`/`1280`/`720` hits found are either unrelated synthetic test fixtures or runtime-derived `.shape` reads
- Full `control/` pytest suite passes: 153 passed, 0 failures

## Task Commits

Each task was committed atomically:

1. **Task 1: Switch stereo_camera.py's capture resolution to 1280x360** - `caa85da` (fix)
2. **Task 2: Update dependent tests and detection-script comments, verify no hidden hardcoding, run full suite** - `f08bbc0` (test)
3. **Task 3: Update policy_server_launch.md's Camera mapping and launch command** - `4f44205` (docs)

**Plan metadata:** committed separately by the orchestrator after this summary.

## Files Created/Modified
- `control/vla_bridge/stereo_camera.py` - New 1280x360/640x360/[0:640,640:1280] constants and docstrings, full frozen-2560x720 rationale documented once in the module docstring
- `control/test_stereo_camera.py` - Synthetic frame fixture and shape assertions updated to 360x1280/640x360
- `control/detect_devices.py` - Descriptive comments/docstrings updated to reference live `STEREO_WIDTH`/`STEREO_HEIGHT` constants instead of hardcoded 2560x720; zero logic changes
- `control/vla_bridge/policy_server_launch.md` - Camera mapping section, crop-column table, and Step 5 launch command updated to 640x360/[0:640]/[640:1280]

## Decisions Made
- Switched AR0144 stereo capture resolution from 2560x720 (persistently frozen/stuck on this machine, confirmed via raw ffmpeg CLI testing and byte-identical recorded episode frames across 300 ticks) to 1280x360 (confirmed live via raw ffmpeg testing as a working, correctly-laid-out fallback at half resolution per eye). This does not meaningfully affect VLA policy quality since the deployed checkpoint resizes all camera inputs to 256x256 regardless of input resolution, and fits FastFS's depth-input width constraints more natively. Tradeoff: reduced calibration checkerboard-corner-detection precision, to be compensated by holding the checkerboard closer to the camera during calibration sessions.

## Deviations from Plan

None - plan executed exactly as written. All "already verified during planning -- do NOT touch" files (`stereo_calibration.py`, `depth_camera.py`, `test_depth_camera.py`, `test_stereo_calibration.py`, `test_detect_devices.py`) were re-verified live per Task 2's instructions and confirmed free of hardcoded resolution assumptions, exactly as the plan predicted -- no fixes were needed.

## Issues Encountered
None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Live VLA bridge camera feed for the AR0144 stereo pair is unblocked -- it will no longer serve frozen/identical frames every tick under the old 2560x720 mode.
- Phase 13 (camera resolution unification) can proceed against this new confirmed-working 1280x360 baseline for the AR0144 rather than the old frozen 2560x720 assumption.
- Calibration sessions going forward should hold the checkerboard closer to the camera to compensate for the lower per-half resolution, per the documented tradeoff.

---
*Phase: quick-260927-he3*
*Completed: 2026-09-27*
