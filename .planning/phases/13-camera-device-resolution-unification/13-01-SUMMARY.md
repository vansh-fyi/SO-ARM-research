---
phase: 13-camera-device-resolution-unification
plan: 01
subsystem: camera-resolution
tags: [python, opencv, avfoundation, camera]
requires:
  - phase: 12
    provides: Shared stereo capture and name-based ffmpeg addressing
provides:
  - Persisted wrist_name with raw-index fallback
  - Shared resolve_camera_identifier with fresh wrist name correlation
affects: [13-03, 13-04]
tech-stack:
  added: []
  patterns: [name-first resolution, injectable live correlation]
key-files:
  created: []
  modified: [control/detect_devices.py, control/test_detect_devices.py]
key-decisions:
  - Numeric wrist_name fallback values skip live name probing.
requirements-completed: [CAMFIX-01]
coverage:
  - id: D1
    description: Persist wrist camera name alongside its numeric index
    requirement: CAMFIX-01
    verification:
      - kind: unit
        ref: control/test_detect_devices.py
        status: pass
    human_judgment: false
  - id: D2
    description: Resolve both camera roles through one name-first helper
    requirement: CAMFIX-01
    verification:
      - kind: unit
        ref: control/test_detect_devices.py#test_resolve_wrist_identifier_uses_current_index_instead_of_stale_index
        status: pass
    human_judgment: false
duration: 3min
completed: 2026-09-30
status: complete
---

# Phase 13 Plan 01: Shared Camera Resolution Summary

**Camera detection now saves wrist names, and one shared resolver handles stereo names and fresh wrist index correlation.**

## Performance

- Duration: approximately 3 minutes
- Tasks: 2
- Files modified: 2

## Accomplishments

- Added wrist_name symmetrically with stereo_overhead_name, including numeric fallback when no name correlates.
- Added _live_correlate_names using existing camera probes and name enumeration.
- Added resolve_camera_identifier: direct name addressing for stereo; a fresh injectable name-to-index lookup for wrist; raw indices on soft misses.
- Covered legacy maps, absent/None/numeric names, live index drift, default correlation wiring, and probe composition without real hardware calls.

## Task Commits

1. Task 1 RED: `b37a11c`; GREEN: `e5c284d`.
2. Task 2 RED: `711fb86`; GREEN: `9916770`.

## Verification

- Task 1 RED: 6 failures caused by the missing wrist_name field; GREEN: 37 passed.
- Task 2 RED: 11 failures caused by missing resolver functions; GREEN: 48 passed.
- Full control suite: `.venv/bin/python -m pytest -q` — 182 passed in 5.38 seconds.
- `git diff --check` passed; no task commit deletes files.
- Stub and threat-surface review found no new unresolved stubs or unmodeled security surfaces.

## Files Created/Modified

- control/detect_devices.py: saved wrist names and shared camera identifier resolution.
- control/test_detect_devices.py: persisted name expectations and hardware-free resolver coverage.

## Decisions Made

Numeric wrist_name values are persisted fallback indices, so skip live correlation for them. Stereo numeric fallback values remain accepted unchanged.

## Deviations from Plan

**[Rule 2 - Correctness] Recognize numeric wrist_name fallback values.** Task 1 deliberately persists integers when correlation finds no name. Task 2 now avoids probing for those integers, since they cannot match camera name strings. Tests cover this case. Implemented in `9916770`; no change to the intended fallback result.

The resolve_cameras return annotation/docstring was corrected to include string values and wrist_name. The existing top-level detect_devices exact-equality assertion was updated along with the assertions listed in the plan.

## Issues Encountered

None.

## User Setup Required

None for this unit. A live device-map refresh and physical camera verification remain part of plan 13-03.

## Next Phase Readiness

Plans 13-03 and 13-04 can import resolve_camera_identifier. CAMFIX-01's helper contribution is complete; requirement-wide completion still depends on those consumer integrations. Physical correspondence of the OS/cv2 enumeration heuristic remains subject to the planned live verification.

## Self-Check: PASSED

Both modified files exist. All four task commit hashes were verified in git history. All task criteria and the full control regression suite pass.
