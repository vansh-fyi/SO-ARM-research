---
phase: quick-260927-i3s
plan: 01
subsystem: hardware-io
tags: [ar0144, ffmpeg, avfoundation, stereo-camera, vla-bridge, pytest]

requires:
  - phase: quick-260927-he3
    provides: AR0144 stereo capture fixed at 1280x360 (working stepping-stone resolution after 2560x720 froze)
provides:
  - AR0144 stereo camera now captures at 1600x600 (800x600 per half), the highest confirmed-stable resolution on this hardware
  - Full resolution-history rationale documented once in stereo_camera.py's module docstring (1280x360 stepping stone -> 1600x600 current -> 2560x720 exhausted -> 1280x480/1280x712 ruled out)
  - policy_server_launch.md and detect_devices.py comments kept in sync with the new resolution, no stale "1280x360 is current" text remains
affects: [vla-bridge, phase-13-camera-resolution-unification, live-hardware-episodes]

tech-stack:
  added: []
  patterns:
    - "STEREO_WIDTH/STEREO_HEIGHT/SPLIT_COL module constants remain the single source of truth for AR0144 resolution; ffmpeg command construction and detect_devices.py's live verification both read them dynamically, so a resolution bump only requires editing 3 constants + docs, no downstream code changes."

key-files:
  created: []
  modified:
    - control/vla_bridge/stereo_camera.py
    - control/test_stereo_camera.py
    - control/vla_bridge/policy_server_launch.md
    - control/detect_devices.py

key-decisions:
  - "Chose 1600x600 (800x600/eye) over the prior 1280x360 (640x360/eye) fix -- confirmed stable via 20 consecutive ffmpeg frames with genuine ~4.9 frame-to-frame pixel differences and a visually-confirmed correct side-by-side stereo layout with parallax."
  - "2560x720 remains frozen even after applying macOS's legacy-camera-plugins-without-sw-camera-indication system override plus a fresh reboot -- all known software-level fixes for that mode are now exhausted; 1600x600 is the best available resolution on this hardware since the mode list jumps directly from 1600x600 to the broken 2560x720."
  - "1280x480 and 1280x712 were tested and ruled out as single-lens (non-stereo) crops, not usable candidates."
  - "Extended additional scope beyond the plan's 3-file scope: updated 4 stale '1280x360'-as-current-resolution comment mentions in control/detect_devices.py (comment-only, no logic change -- its resolution checks already read STEREO_WIDTH/STEREO_HEIGHT live from stereo_camera.py) per explicit orchestrator instruction, to avoid the docs going stale a second time."

requirements-completed: []

coverage:
  - id: D1
    description: "AR0144 stereo camera captures at 1600x600 (800x600/eye) via updated STEREO_WIDTH/STEREO_HEIGHT/SPLIT_COL constants"
    verification:
      - kind: unit
        ref: "control/test_stereo_camera.py::test_read_left_and_read_right_return_correct_non_swapped_halves"
        status: pass
      - kind: other
        ref: "python -c \"from vla_bridge.stereo_camera import STEREO_WIDTH, STEREO_HEIGHT, SPLIT_COL; assert (STEREO_WIDTH, STEREO_HEIGHT, SPLIT_COL) == (1600, 600, 800)\""
        status: pass
    human_judgment: false
  - id: D2
    description: "Module docstring in stereo_camera.py documents the full resolution history and rationale (1280x360 -> 1600x600, 2560x720 exhausted, 1280x480/1280x712 ruled out)"
    verification: []
    human_judgment: true
    rationale: "Docstring prose quality/completeness is a judgment call, not something a unit test can assert."
  - id: D3
    description: "stereo_calibration.py and depth_camera.py (and their tests) require zero code changes -- confirmed to derive dimensions from runtime frame shape, not hardcoded literals"
    verification:
      - kind: other
        ref: "grep for 1280/360/640/1600/600/800 literals in stereo_calibration.py, depth_camera.py, test_depth_camera.py, test_stereo_calibration.py -- all hits are unrelated synthetic-fixture values or runtime .shape reads"
        status: pass
    human_judgment: false
  - id: D4
    description: "Full control/ pytest suite passes after the resolution change"
    verification:
      - kind: unit
        ref: "control/ pytest suite (153 tests)"
        status: pass
    human_judgment: false
  - id: D5
    description: "policy_server_launch.md's Camera mapping section, crop-column table, and Step 5 launch command updated to 1600x600/800x600/[0:800]/[800:1600]"
    verification:
      - kind: other
        ref: "grep checks for camera2/camera3 width:800 height:600 and [0:800]/[800:1600] in policy_server_launch.md"
        status: pass
    human_judgment: false

duration: 3min
completed: 2026-09-27
status: complete
---

# Quick Task 260927-i3s: Upgrade AR0144 Stereo Camera Resolution to 1600x600 Summary

**AR0144 stereo camera capture bumped from 1280x360 to 1600x600 (800x600/eye) -- confirmed stable via 20-frame ffmpeg test, with full resolution history documented in `stereo_camera.py` and propagated through tests, launch docs, and detect_devices.py comments.**

## Performance

- **Duration:** ~3 min
- **Started:** 2026-09-27T13:09:18+05:30 (first task commit)
- **Completed:** 2026-09-27T13:11:24+05:30 (last task commit)
- **Tasks:** 3 (plus 1 additional-scope doc update folded into Task 3's commit)
- **Files modified:** 4

## Accomplishments
- `STEREO_WIDTH`/`STEREO_HEIGHT`/`SPLIT_COL` changed from 1280/360/640 to 1600/600/800 in `stereo_camera.py`, with every in-file docstring (module, `_FFmpegAVFoundationCapture`, `StereoSplitCamera`, `read_left()`, `read_right()`) updated to match and the module docstring's full resolution-history narrative rewritten in one place.
- `test_stereo_camera.py`'s synthetic frame fixture and shape assertions updated to 600x1600/800-column-split; re-verified `stereo_calibration.py`, `depth_camera.py`, and their tests contain zero hardcoded AR0144-resolution literals (all derive dimensions from runtime frame shape or use unrelated synthetic fixtures) -- confirmed live via grep, not just trusted from planning.
- `policy_server_launch.md`'s Camera mapping bullet, crop-column table, fallback-option paragraph, and Step 5 launch command all updated to 1600x600/800x600/[0:800]/[800:1600].
- Additional scope: fixed 4 stale "currently 1280x360" comment/docstring mentions in `control/detect_devices.py` (comment-only, its live resolution checks already read `STEREO_WIDTH`/`STEREO_HEIGHT` from `stereo_camera.py` dynamically).
- Full `control/` pytest suite passes: 153 passed.

## Task Commits

Each task was committed atomically:

1. **Task 1: Switch stereo_camera.py's capture resolution to 1600x600** - `0545951` (feat)
2. **Task 2: Update test_stereo_camera.py's dimension assertions, re-verify no hidden hardcoding elsewhere, run full suite** - `1564049` (test)
3. **Task 3: Update policy_server_launch.md's Camera mapping and launch command (+ detect_devices.py doc sync, additional scope)** - `acc99a3` (docs)

_Plan metadata commit (SUMMARY.md/STATE.md) is made separately by the orchestrator, not this executor._

## Files Created/Modified
- `control/vla_bridge/stereo_camera.py` - `STEREO_WIDTH`/`STEREO_HEIGHT`/`SPLIT_COL` = 1600/600/800; module docstring rewritten with full resolution history; all internal docstrings updated.
- `control/test_stereo_camera.py` - synthetic frame fixture and shape assertions updated to 600x1600/800-column split.
- `control/vla_bridge/policy_server_launch.md` - Camera mapping bullet, crop-column table, fallback-option paragraph, and Step 5 launch command updated to 1600x600/800x600.
- `control/detect_devices.py` - 4 stale "currently 1280x360" comment mentions updated to 1600x600 (comment-only, additional scope per orchestrator instruction).

## Decisions Made
- 1600x600 chosen over the prior 1280x360 fix: meaningfully higher per-eye resolution (800x600 vs 640x360) while remaining confirmed stable via 20 consecutive ffmpeg frames with genuine ~4.9 frame-to-frame pixel differences and a visually-confirmed correct side-by-side layout with parallax.
- 2560x720 confirmed still frozen after exhausting the last known macOS system-level fix (legacy-camera-plugins override + reboot) -- no further software fixes remain for that mode.
- 1280x480 and 1280x712 ruled out as single-lens (non-stereo) crops, not viable stereo candidates.
- Extended scope to `control/detect_devices.py`'s stale resolution comments per explicit orchestrator instruction (comment-only, zero logic change) to prevent the docs from going stale again after this second resolution bump.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - orchestrator-directed scope extension] Updated stale "1280x360" comments in `control/detect_devices.py`**
- **Found during:** Task 3 (doc sync pass)
- **Issue:** The prompt's `<additional_scope>` block explicitly noted `detect_devices.py` still referenced the prior "1280x360" resolution in 4 comment/docstring locations (module docstring, `_verify_stereo_via_ffmpeg()` docstring x2, `resolve_cameras()` docstring), left out of scope by the prior quick task but now stale again after this second resolution bump.
- **Fix:** Updated all 4 mentions from "1280x360" to "1600x600" -- comment/docstring text only, no logic changes (the file's actual resolution checks already read `STEREO_WIDTH`/`STEREO_HEIGHT` live from `stereo_camera.py`).
- **Files modified:** `control/detect_devices.py`
- **Verification:** `grep -n "1280x360\|640x360" control/detect_devices.py` returns no hits; full `control/` pytest suite (153 tests) still passes.
- **Committed in:** `acc99a3` (Task 3 commit)

---

**Total deviations:** 1 auto-fixed (orchestrator-directed doc-sync scope extension, explicitly pre-authorized in the execution prompt, not a self-initiated Rule 1-4 fix)
**Impact on plan:** No scope creep beyond what the orchestrator explicitly requested; keeps all AR0144-resolution-describing prose in the repo internally consistent.

## Issues Encountered
- No `python`/`pytest` binary was on PATH inside this worktree checkout (no local `.venv`); resolved by invoking the main repo's existing `control/.venv/bin/python` directly for verification and the test suite. No code or environment changes were made to fix this -- purely a local invocation detail.

## User Setup Required

None - no external service configuration required. This is a pure code/docs change; the AR0144 camera hardware itself already supports the 1600x600 mode (confirmed via live ffmpeg testing prior to this quick task).

## Next Phase Readiness
- `stereo_camera.py` now captures at the best available confirmed-stable resolution (1600x600) for the AR0144 stereo camera on this hardware, ready for the next live VLA episode or Phase 13's camera-resolution-unification work.
- No blockers. `stereo_calibration.py`/`depth_camera.py` remain resolution-agnostic (runtime-shape-derived), so future resolution changes (if the frozen 2560x720 mode is ever unstuck) would similarly only require editing `stereo_camera.py`'s 3 constants + docs.

---
*Phase: quick-260927-i3s*
*Completed: 2026-09-27*
