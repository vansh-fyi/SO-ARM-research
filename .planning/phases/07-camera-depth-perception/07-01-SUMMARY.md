---
phase: 07-camera-depth-perception
plan: 01
subsystem: simulation
tags: [mujoco, robosuite, mjcf, camera, libero]

# Dependency graph
requires: []
provides:
  - "agentview recalibrated to a real-camera-derived 43 deg vertical FOV (AR0144 datasheet) and a desk-mount-appropriate pose, applied in the reachable _setup_camera() override plus the defensive base-class mirror"
  - "eye_in_hand recalibrated to a human-tuned pos/quat that keeps the grasp target and jaw convergence point in frame together, validated against 8 real IMX335 wrist-mount reference photos"
affects: [07-camera-depth-perception (plan 02, depth persistence — zero file overlap, independent), phase-9-data-collection]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Interactive live-render camera tuner (scratch tool, not committed): matplotlib sliders drive sim.model.cam_pos/cam_quat directly, re-rendering on every change, with a pose selector that replays real recorded demo states (mujoco set_state_from_flattened) — used to catch a pose-dependent framing bug the static single-render checkpoint flow would have missed"

key-files:
  created: []
  modified:
    - LIBERO/libero/libero/envs/problems/libero_tabletop_manipulation.py
    - LIBERO/libero/libero/envs/bddl_base_domain.py
    - LIBERO/libero/libero/assets/robots/soarm101/robot.xml
    - LIBERO/libero/libero/envs/test_camera_config.py

key-decisions:
  - "agentview: applied the planned camera_attribs={\"fovy\": \"43\"} + pos/quat change exactly as specified — automated test assertion covers this, no human judgment needed"
  - "eye_in_hand: the plan's closed-form look-at-the-grip-site quat (pos=\"0.14 0 0.02\") was REJECTED at the Task 3 checkpoint — replayed against a real recorded demo trajectory (reset, early, mid, end states), it rendered as a flat top-down close-up of the gripper's own mechanism with zero forward/table context for the ENTIRE episode, because this arm's wrist barely reorients during a pick (boresight stayed within ~1 deg of straight-down throughout). The old pre-plan quat, by contrast, showed forward-and-down spatial context (table ahead, gripper upper-frame) — the geometrically-precise local-frame derivation was actually a regression for practical wrist-cam usefulness."
  - "Resolved by interactive human tuning instead of a second closed-form guess: built a live slider tool (pos x/y/z, roll/pitch/yaw) rendering both eye_in_hand and agentview in real time, with a pose selector replaying real demo states. User iterated directly against the render and the 8 real reference photos, landing on pos=\"0.0133 0.0017 -0.0718\" quat=\"0.417709 -0.571718 0.576823 -0.407349\" — a real position change (not just a re-tilt at the plan's original spot), which better matches the real mount's forward-and-down framing while keeping the grasp target and jaw convergence point both in frame."

patterns-established:
  - "When a camera checkpoint is 'does the framing look right', validate at more than one arm pose (ideally real recorded states) before presenting to the human — a reset-only render can look correct while a full-episode render exposes exactly the failure that led to rejection here"

requirements-completed: [CAM-01]

coverage:
  - id: D1
    description: "agentview renders non-degenerate with camera_attribs fovy=43 (AR0144 datasheet), applied in the reachable _setup_camera() override and mirrored defensively in the base class"
    requirement: "CAM-01"
    verification:
      - kind: integration
        ref: "LIBERO/libero/libero/envs/test_camera_config.py::test_rgb_cameras_non_degenerate_spatial"
        status: pass
    human_judgment: false
  - id: D2
    description: "eye_in_hand renders with a human-approved pos/quat that keeps the grasp target and jaw convergence point in frame, matching the real IMX335 wrist mount's photographed forward-and-down tilt"
    requirement: "CAM-01"
    verification:
      - kind: integration
        ref: "LIBERO/libero/libero/envs/test_camera_config.py::test_rgb_cameras_non_degenerate_spatial"
        status: pass
    human_judgment: true
    rationale: "No automated oracle exists for 'does this framing match a real photographed mount angle' — explicit human sign-off is the plan's own acceptance criterion (Task 3), and the first candidate quat was in fact rejected and iterated based on that judgment."

duration: "~15min active work across two sessions (2026-09-05 Tasks 1-2; 2026-09-12 Task 3 checkpoint resolution — session was interrupted mid-checkpoint on 09-05, resumed and closed out on 09-12)"
completed: 2026-09-12
status: complete
---

# Phase 7 Plan 01: Camera Recalibration Summary

**`agentview` recalibrated to the AR0144's real 43 deg FOV; `eye_in_hand` recalibrated via interactive human tuning after the plan's closed-form quat was rejected for producing a top-down-only view with no forward/table context across a real demo trajectory.**

## Performance

- **Duration:** ~15 min active work (spread across two sessions — see Issues Encountered)
- **Started:** 2026-09-05T17:02:20Z
- **Completed:** 2026-09-12T06:44:37Z
- **Tasks:** 3
- **Files modified:** 4

## Accomplishments

- `libero_tabletop_manipulation.py`'s `_setup_camera()` (the only override every registered SOARM task actually reaches) now sets `agentview`'s `pos=[0.5, 0.0, 1.45]`, `quat=[0.635968, 0.309103, 0.309103, 0.635968]`, and `camera_attribs={"fovy": "43"}`.
- `bddl_base_domain.py`'s dead-code `_setup_camera()` mirrors the identical `agentview` change defensively (Phase-8 future-proofing).
- `test_camera_config.py` extended with an explicit `cam_fovy == 43.0` assertion.
- `eye_in_hand`'s quat was corrected twice: first to a closed-form look-at-the-grip-site value (Task 2, as planned), then — after that value failed the Task 3 checkpoint — to a human-tuned `pos="0.0133 0.0017 -0.0718" quat="0.417709 -0.571718 0.576823 -0.407349"` found via an interactive live-render tuning session.
- Both camera tests (`test_camera_config.py`) pass with the final values.

## Task Commits

Each task was committed atomically:

1. **Task 1: Recalibrate agentview (CAM-01)** - `b514881` (feat)
2. **Task 2: Correct eye_in_hand quat toward the grip site (initial closed-form candidate)** - `af76618` (feat)
3. **Task 3 resolution: Re-tune eye_in_hand after checkpoint rejection** - `45ec756` (fix)

_Note: Task 3 is a `checkpoint:human-verify` gate — its "commit" is the fix landed after the human sign-off round-trip below, not a task-1-style implementation commit._

## Files Created/Modified

- `LIBERO/libero/libero/envs/problems/libero_tabletop_manipulation.py` - `agentview` pos/quat/fovy recalibration in the reachable override
- `LIBERO/libero/libero/envs/bddl_base_domain.py` - identical `agentview` change mirrored defensively in the dead-code base-class override
- `LIBERO/libero/libero/assets/robots/soarm101/robot.xml` - `eye_in_hand` camera pos/quat corrected (twice — see Deviations)
- `LIBERO/libero/libero/envs/test_camera_config.py` - added explicit `agentview` `cam_fovy` assertion

## Decisions Made

- See `key-decisions` in frontmatter — the central decision was rejecting the plan's closed-form `eye_in_hand` quat after empirical replay against a real recorded demo trajectory showed it produced a top-down-only view for the entire episode (not just at `reset()`), and resolving it via interactive human tuning instead of a second guessed closed-form value.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Invalid XML comment broke MJCF parsing**
- **Found during:** Applying the human-tuned `eye_in_hand` quat to `robot.xml`
- **Issue:** The updated inline comment above the `<camera>` element contained a literal `--` (double hyphen), which is illegal inside an XML comment body. This broke `xml.etree.ElementTree` parsing for the entire robot MJCF, failing both camera tests with `ParseError: not well-formed (invalid token)`.
- **Fix:** Reworded the comment to avoid `--` while preserving the same provenance information.
- **Files modified:** `LIBERO/libero/libero/assets/robots/soarm101/robot.xml`
- **Verification:** `pytest LIBERO/libero/libero/envs/test_camera_config.py -x` — 2 passed
- **Committed in:** `45ec756` (Task 3 resolution commit)

---

**Total deviations:** 1 auto-fixed (1 bug — self-introduced XML syntax error, caught immediately by re-running tests)
**Impact on plan:** No scope creep — comment-only fix inside a file this plan already modifies.

### Plan-scope deviation (not an auto-fix — human-directed)

Task 2's literal instruction (use the closed-form look-at-the-grip-site quat at the plan's original `pos="0.14 0 0.02"`) was superseded at the Task 3 checkpoint. The plan itself anticipated this possibility (RESEARCH.md: "should be treated as a hypothesis to visually validate... not a drop-in final answer"; the task's own `<resume-signal>` explicitly allows "describe the mismatch... so the quat can be iterated"). What actually happened exceeded a quat-only iteration — the human also changed the camera's *position*, not just its orientation, after finding via the interactive tuner that a position closer to the jaws combined with the tilt produced the best framing. This is a direct, checkpoint-authorized resolution of the exact open question RESEARCH.md flagged, not an unplanned scope change.

## Issues Encountered

- **Session interruption between Tasks 2 and 3:** Tasks 1 and 2 were executed and committed on 2026-09-05 in isolated git worktrees as part of a parallel wave-1 dispatch (alongside plan 07-02). The orchestrating session was interrupted before the wave-close worktree merge ever ran, leaving both worktrees' commits unmerged into `master` and Task 3's checkpoint unresolved. A later `/gsd-execute-phase 7` invocation on 2026-09-12 detected this via the safe-resume-gate (production commits present, no SUMMARY.md), verified both worktrees were clean with zero file overlap against `master`'s intervening commits, and merged both back via the standard worktree-cleanup tool before resuming Task 3.
- **First `eye_in_hand` candidate rejected at checkpoint:** see Deviations/Decisions above — resolved via interactive tuning rather than blocking on repeated closed-form guesses.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Both `agentview` and `eye_in_hand` are now grounded in real hardware specs/photos rather than unmodified Panda-benchmark defaults; `07-02`'s depth persistence (already merged, independent, zero file overlap) and `07-03`'s RLDS/OXE registration (Wave 2) are unaffected by this plan's changes.
- The interactive tuner script (`tune_eye_in_hand.py`, not committed — scratch dev tool) is a reusable pattern for any future camera-angle iteration; it replays real recorded demo states rather than only `reset()`, which is what caught this plan's framing bug.

---
*Phase: 07-camera-depth-perception*
*Completed: 2026-09-12*
