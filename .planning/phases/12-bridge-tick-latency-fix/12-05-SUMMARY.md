---
phase: 12-bridge-tick-latency-fix
plan: 05
subsystem: infra
tags: [opencv, stereo-vision, calibration, ar0144, depth]

# Dependency graph
requires:
  - phase: 11-vla-hardware-connection
    provides: StereoSplitCamera (control/vla_bridge/stereo_camera.py), the single-open ffmpeg-backed AR0144 stereo capture this plan's run_calibration_session() consumes without ever opening a second capture.
provides:
  - "stereo_calibration.py: build_object_points(), detect_checkerboard_corners(), calibrate_single_camera(), calibrate_stereo_pair(), StereoCalibration dataclass, save_calibration()/load_calibration(), run_calibration_session(), CLI main()"
  - "A saved control/stereo_calibration.json calibration file format (K1/D1/K2/D2/R/T + reprojection_error_px + calibrated_at_utc) Plan 12-06's depth_camera.py will load"
affects: [12-06-depth-camera, live-hardware-verification]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Two-step OpenCV stereo calibration: calibrate_single_camera() per lens first (cv2.calibrateCamera), then calibrate_stereo_pair() with cv2.CALIB_FIX_INTRINSIC solving only R/T -- the standard, better-conditioned recipe vs. joint-from-scratch calibration"
    - "cv2 corner-array shape normalization: reshape(-1, 1, 2) immediately after findChessboardCorners()/cornerSubPix(), since some cv2 builds (confirmed: cv2 5.0.0) return (N, 2) instead of the (N, 1, 2) legacy shape calibrateCamera()/stereoCalibrate() expect"
    - "Synthetic-checkerboard test rendering: a real, axis-aligned black/white grid painted via numpy slicing, with exact computable integer corner pixel locations, mirroring test_stereo_camera.py's _synthetic_stereo_frame() convention for hardware-free but genuinely-detected corner tests"

key-files:
  created:
    - control/vla_bridge/stereo_calibration.py
    - control/test_stereo_calibration.py
  modified: []

key-decisions:
  - "Reused StereoSplitCamera unmodified for all live capture -- run_calibration_session() accepts an already-constructed instance, never imports _open_stereo_capture or references ffmpeg directly (grep-verified 0 matches)"
  - "DEFAULT_CALIBRATION_PATH placed at control/stereo_calibration.json (sibling of control/device_map.json), not nested under vla_bridge/, mirroring detect_devices.py's DEVICE_MAP_PATH convention"
  - "T (inter-camera translation) is stored in meters, not pixels, because build_object_points()'s checkerboard square size is specified in meters -- this is what makes the saved baseline directly usable by Plan 12-06's FastFS integration"

patterns-established:
  - "cv2 output-shape normalization at the function boundary (reshape immediately after any cv2 corner-detection call) rather than downstream at every call site -- keeps the shape contract deterministic across cv2 versions"

requirements-completed: [DEPTH-CAL-01]

coverage:
  - id: D1
    description: "build_object_points(), detect_checkerboard_corners(), calibrate_single_camera(), calibrate_stereo_pair(), save_calibration()/load_calibration(), and run_calibration_session() all implemented with deterministic, hardware-free test coverage"
    requirement: "DEPTH-CAL-01"
    verification:
      - kind: unit
        ref: "control/test_stereo_calibration.py -- all 10 tests"
        status: pass
    human_judgment: false
  - id: D2
    description: "run_calibration_session() never proceeds to calibrate on fewer than 3 valid views (raises RuntimeError), and correctly skips views where corners aren't detected in both halves without counting them toward num_views"
    requirement: "DEPTH-CAL-01"
    verification:
      - kind: unit
        ref: "control/test_stereo_calibration.py#test_run_calibration_session_raises_runtime_error_with_fewer_than_3_valid_views"
        status: pass
      - kind: unit
        ref: "control/test_stereo_calibration.py#test_run_calibration_session_skips_views_with_no_detected_corners"
        status: pass
    human_judgment: false
  - id: D3
    description: "A real control/stereo_calibration.json exists, produced by a human-run live checkerboard calibration session against the physical AR0144, with a human-confirmed plausible reprojection error and plausible K/D/R/T values"
    verification: []
    human_judgment: true
    rationale: "Requires running the real AR0144 and a real physical checkerboard; Claude does not run physical hardware (CONTEXT.md D-06). Owned by the user as Task 2's human-check step in 12-05-PLAN.md, after this plan is merged."

duration: 18min
completed: 2026-09-26
status: complete
---

# Phase 12 Plan 05: AR0144 Stereo Calibration Summary

**Checkerboard-based OpenCV stereo calibration module (`stereo_calibration.py`) producing a saved calibration file with per-camera intrinsics, inter-camera R/T (translation in meters), and reprojection error -- the first of a 3-step depth-calibration extension (D-07), reusing the existing `StereoSplitCamera` for live capture.**

## Performance

- **Duration:** 18 min
- **Started:** 2026-09-26T18:00:00Z (approx)
- **Completed:** 2026-09-26T18:18:25Z
- **Tasks:** 2 (Task 1 code + tests; Task 2 human-check, no code changes required)
- **Files modified:** 2 (both newly created)

## Accomplishments
- `build_object_points()` produces a flat, real-world-scaled (meters) checkerboard object-point grid
- `detect_checkerboard_corners()` finds and subpixel-refines corners via `cv2.findChessboardCorners`/`cornerSubPix`, returning `None` (never raising) on a pattern-not-found image
- `calibrate_single_camera()`/`calibrate_stereo_pair()` wrap `cv2.calibrateCamera`/`cv2.stereoCalibrate` with the two-step, `CALIB_FIX_INTRINSIC` recipe, reducing/flattening their tuple returns into a clean `StereoCalibration` dataclass
- `save_calibration()`/`load_calibration()` round-trip a `StereoCalibration` through JSON, coercing `image_size` back to a tuple on load
- `run_calibration_session()` is the live-capture orchestration loop: prompts the human per view, reads from an injected `StereoSplitCamera`, skips views where either half fails corner detection, and raises `RuntimeError` if fewer than 3 valid views are captured -- never silently calibrating on too little data
- A CLI `main()` wires `StereoSplitCamera` + `run_calibration_session()` + `save_calibration()` together, printing the reprojection error and output path
- All 10 new tests pass; full `control/` suite grew from 100 to 110 tests, all passing -- no cross-file regression

## Task Commits

Each task was committed atomically:

1. **Task 1: Core calibration math -- object points, corner detection, single/stereo calibration, save/load, and the live-session orchestration** - `d8e79bf` (feat)
2. **Task 2: CLI human-run calibration session + reprojection-error checkpoint** - no code commit (no defect surfaced requiring a fix; automated verify re-run in Task 1's commit already covers this task's `<verify><automated>` step; the human-check itself is outstanding, see User Setup Required below)

**Plan metadata:** (this commit, SUMMARY.md)

## Files Created/Modified
- `control/vla_bridge/stereo_calibration.py` - New module: `build_object_points()`, `detect_checkerboard_corners()`, `StereoCalibration` dataclass, `calibrate_single_camera()`, `calibrate_stereo_pair()`, `save_calibration()`/`load_calibration()`, `run_calibration_session()`, CLI `main()`
- `control/test_stereo_calibration.py` - New test file: synthetic-checkerboard rendering helper, `FakeStereoCameraForCalibration` hand-rolled fake, and 10 deterministic tests covering every function plus the never-opens-ffmpeg structural guard

## Decisions Made
- `DEFAULT_CALIBRATION_PATH` placed at `control/stereo_calibration.json`, a sibling of `control/device_map.json`, mirroring `detect_devices.py`'s `DEVICE_MAP_PATH` convention (per the plan's explicit instruction)
- `T` (inter-camera translation) stored in meters, not pixels or arbitrary units, since `build_object_points()`'s square size is specified in meters -- this is exactly what Plan 12-06's Fast-FoundationStereo integration needs for baseline in meters
- Reused `StereoSplitCamera` unmodified; `run_calibration_session()` accepts an already-constructed instance and never opens a second AR0144 capture

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Normalized cv2 corner-array shape to (N, 1, 2)**
- **Found during:** Task 1 (first test run of `detect_checkerboard_corners()`)
- **Issue:** This environment's `cv2` build (5.0.0) returns `findChessboardCorners()`'s corner array as shape `(N, 2)` instead of the legacy `(N, 1, 2)` shape the plan's behavior spec (and `cv2.cornerSubPix()`/`calibrateCamera()`/`stereoCalibrate()`'s expected input) assume. The un-normalized array failed the plan's own `(63, 1, 2)` shape assertion and would have broken `cornerSubPix()`'s and the calibration functions' array-shape expectations downstream.
- **Fix:** Added an explicit `np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2)` normalization immediately after `findChessboardCorners()`, and again after `cornerSubPix()`, so `detect_checkerboard_corners()`'s return contract is stable regardless of the installed `cv2` version's internal shape convention.
- **Files modified:** `control/vla_bridge/stereo_calibration.py`
- **Verification:** `test_detect_checkerboard_corners_finds_corners_near_expected_locations` passes with the exact `(63, 1, 2)` shape the plan's behavior spec requires; full 10-test file and full 110-test suite both pass with 0 failures.
- **Committed in:** `d8e79bf` (part of Task 1 commit -- caught and fixed before the task's own verification step, not a follow-up commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Necessary correctness fix caught during the task's own verify step, before commit. No scope creep -- same functions, same contract, just made robust to a real installed-library version difference.

## Issues Encountered

None beyond the cv2 shape deviation documented above. Tests were run via `control/.venv/bin/python -m pytest` (the repo's existing per-`control/`-directory virtualenv, gitignored, present in this worktree checkout with `pytest`/`opencv-python`/`numpy` already installed) -- no environment setup required.

## User Setup Required

**Outstanding human-check (Task 2 of 12-05-PLAN.md), per CONTEXT.md D-06 -- Claude does not run physical hardware.** The core calibration math is fully covered by deterministic, hardware-free tests (all passing), but a real calibration file has not yet been produced, since that requires:

1. A printed or physical checkerboard (a common size is 9x6 internal corners with ~25mm squares -- adjust `--pattern-cols`/`--pattern-rows`/`--square-size-m` to match whatever board is actually used).
2. Running `cd control && python -m vla_bridge.stereo_calibration --stereo-camera-index "CCB Camera"` (substituting the AR0144's real AVFoundation device name/index, per `control/device_map.json`'s `cameras.stereo_overhead_name` field if available), repositioning the checkerboard between each of the 15 prompted captures.
3. Reporting back the printed reprojection error (under ~1.0 px is a good calibration) and confirming `control/stereo_calibration.json`'s `left_K`/`right_K`/`R`/`T` fields look plausible (no zeros/NaNs, baseline roughly matching the AR0144's real physical lens separation).

Keep the resulting calibration file -- Plan 12-06 loads it directly. See Task 2's `<verify><human-check>` in `12-05-PLAN.md` for the full instructions.

## Next Phase Readiness
- `stereo_calibration.py`'s full API surface (`build_object_points`, `detect_checkerboard_corners`, `calibrate_single_camera`, `calibrate_stereo_pair`, `save_calibration`/`load_calibration`, `run_calibration_session`) is complete, tested, and ready for Plan 12-06's `depth_camera.py` to consume its saved calibration file format
- Plan 12-06 is blocked on the human running the live calibration session above and confirming a real `control/stereo_calibration.json` exists with a plausible reprojection error -- this is the same D-06 hardware-ownership pattern already established in 12-01's live-episode verification

---
*Phase: 12-bridge-tick-latency-fix*
*Completed: 2026-09-26*

## Self-Check: PASSED

All created files found on disk (`control/vla_bridge/stereo_calibration.py`, `control/test_stereo_calibration.py`, this SUMMARY.md). Both commit hashes (`d8e79bf`, `8412905`) confirmed present in `git log`.
