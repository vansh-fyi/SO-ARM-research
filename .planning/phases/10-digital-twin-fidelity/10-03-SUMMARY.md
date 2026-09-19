---
phase: 10-digital-twin-fidelity
plan: 03
subsystem: robot-model
tags: [mjcf, urdf, xml.etree.ElementTree, kinematics, so-101]

# Dependency graph
requires:
  - phase: 10-digital-twin-fidelity (10-01)
    provides: robot.xml's calibration-derived joint limits (shoulder_pan/shoulder_lift/elbow_flex/wrist_flex)
  - phase: 10-digital-twin-fidelity (10-02)
    provides: soarm_gripper.xml's corrected clamp visual mesh offsets (D-04 fix)
provides:
  - scripts/mjcf_to_urdf.py — hand-rolled MJCF-to-URDF converter (stdlib only)
  - So-101/So-101.urdf regenerated with correct 9-link/8-joint kinematic chain
affects: [10-04, 10-05]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Hand-rolled ElementTree MJCF->URDF conversion via explicit per-joint mapping table (no generic recursive walker, no third-party converter library)"
    - "Fail-loud precondition assertions (sys.exit) before XML emission, mirroring coppelia/export_model_library.py"

key-files:
  created: [scripts/mjcf_to_urdf.py, So-101/So-101.urdf]
  modified: []

key-decisions:
  - "Read MJCF body/joint attributes live via ElementTree rather than hardcoding numeric values, so future robot.xml/soarm_gripper.xml edits (e.g. a future calibration re-run) propagate to the URDF automatically on re-run"
  - "Hardcoded the per-link CoppeliaSim mesh-placement offset (visual/collision <origin>) and mesh filename lists as module-level constants, since these don't exist in the MJCF at all and are pure CoppeliaSim-export mesh-placement data, not kinematic structure (per the plan's explicit 'reuse mesh filename list, not kinematic structure' instruction)"
  - "Fixed an XML-comment-validity bug (embedded '--' inside <!-- --> comments raises a ParseError under Python's own ElementTree parser) discovered on first self-test run — replaced with single hyphens"

patterns-established:
  - "Any future robot.xml/soarm_gripper.xml edit that changes a joint limit, axis, or gripper mesh offset should be followed by re-running `python3 scripts/mjcf_to_urdf.py` to keep So-101/So-101.urdf in sync by construction (TWIN-06's URDF/MJCF agreement)"

requirements-completed: [TWIN-01, TWIN-02, TWIN-03, TWIN-04, TWIN-06]

coverage:
  - id: D1
    description: "scripts/mjcf_to_urdf.py generates So-101/So-101.urdf with the gripper correctly parented under right_hand (not a root-level sibling of base)"
    requirement: "TWIN-01"
    verification:
      - kind: unit
        ref: "python3 -c \"ElementTree structural check: 9 links, 8 joints, right_hand_to_gripper parent=right_hand child=right_gripper\""
        status: pass
    human_judgment: false
  - id: D2
    description: "Generated URDF's wrist_roll joint carries the exact same asymmetric limit as the corrected MJCF source"
    requirement: "TWIN-02"
    verification:
      - kind: unit
        ref: "grep exact-string match: <limit lower=\"-2.7438472969992493\" upper=\"2.841206309382605\" .../> on joint name=\"wrist_roll\""
        status: pass
    human_judgment: false
  - id: D3
    description: "Generated URDF's gripper_left/gripper_right prismatic joints carry the correct axes matching soarm_gripper.xml"
    requirement: "TWIN-03"
    verification:
      - kind: unit
        ref: "grep: joint name=\"gripper_left\" type=\"prismatic\" axis xyz=\"0 -1 0\"; joint name=\"gripper_right\" type=\"prismatic\" axis xyz=\"0 1 0\""
        status: pass
    human_judgment: false
  - id: D4
    description: "Every mesh filename in the generated URDF is a relative path, not an absolute ~/Downloads/ path"
    requirement: "TWIN-04"
    verification:
      - kind: unit
        ref: "grep -n 'filename=\"/' and 'filename=\"file://' So-101/So-101.urdf -- zero matches"
        status: pass
    human_judgment: false
  - id: D5
    description: "Full existing LIBERO test suite remains green after all Phase-10-so-far MJCF/URDF changes (Plans 10-01, 10-02, 10-03)"
    requirement: "TWIN-06"
    verification:
      - kind: integration
        ref: "conda run -n libero python3 -m pytest LIBERO/libero/libero/envs/ LIBERO/libero/libero/datasets/ -x"
        status: pass
    human_judgment: false

duration: 45min
completed: 2026-09-19
status: complete
---

# Phase 10 Plan 03: Generate So-101.urdf from MJCF Summary

**Hand-rolled `scripts/mjcf_to_urdf.py` (stdlib ElementTree, no MuJoCo runtime) regenerates `So-101/So-101.urdf` from `robot.xml` + `soarm_gripper.xml`, replacing the broken CoppeliaSim export's disconnected-gripper/absolute-path/missing-wrist_roll structure with a correct 9-link/8-joint chain.**

## Performance

- **Duration:** ~45 min
- **Started:** 2026-09-19T03:25:00Z (approx, worktree spawn)
- **Completed:** 2026-09-19T04:10:02Z
- **Tasks:** 2/2 completed
- **Files modified:** 2 (1 new script, 1 regenerated URDF)

## Accomplishments

- `scripts/mjcf_to_urdf.py`: a ~430-line hand-rolled ElementTree-based converter that reads `robot.xml`'s and `soarm_gripper.xml`'s `<worldbody>` trees live (not hardcoded numeric values) and emits a fresh `So-101/So-101.urdf` per an explicit 9-link/8-joint mapping table
- Generated URDF fixes all four confirmed bugs in the broken CoppeliaSim export by construction: gripper is now a proper descendant of `right_hand` via a `right_hand_to_gripper` fixed joint (TWIN-01); `wrist_roll` present with the exact asymmetric mechanical-limit range `-2.7438472969992493`..`2.841206309382605` (TWIN-02); `gripper_left`/`gripper_right` prismatic joints present with correct axes `0 -1 0`/`0 1 0` (TWIN-03); every mesh `filename` is a bare relative path — zero absolute `/` or `file://` references (TWIN-04)
- Fail-loud precondition assertions (`sys.exit`) verify all 6 required MJCF bodies, 5 arm joints, and 2 gripper joints exist before any XML is emitted — mirrors `coppelia/export_model_library.py`'s existing precondition-check pattern (T-10-05 mitigation)
- Full existing LIBERO test suite (`LIBERO/libero/libero/envs/`, `LIBERO/libero/libero/datasets/`) re-run after this plan's change: 50 passed, 5 skipped, 0 failures (TWIN-06's wave-merge regression gate)

## Task Commits

Each task was committed atomically:

1. **Task 1: Write scripts/mjcf_to_urdf.py and generate So-101/So-101.urdf** - `aee8447` (feat)
2. **Task 2: Full-suite regression confirmation (wave-merge gate)** - no commit (pure verification task; no files modified — `LIBERO/libero/libero/envs/`, `LIBERO/libero/libero/datasets/` were read-only inputs to the pytest run)

**Plan metadata:** (this commit, appended after this SUMMARY)

## Files Created/Modified

- `scripts/mjcf_to_urdf.py` - new hand-rolled MJCF->URDF converter (`ROOT`, `ROBOT_XML`, `GRIPPER_XML`, `OUTPUT_URDF` path constants; `mjcf_quat_to_rpy()`; `main()`)
- `So-101/So-101.urdf` - regenerated in place: 9 links (`base`, `shoulder`, `upper_arm`, `lower_arm`, `wrist`, `right_hand`, `right_gripper`, `gripper_left_jaw`, `gripper_right_jaw`), 8 joints (`shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `right_hand_to_gripper`, `gripper_left`, `gripper_right`)

## Decisions Made

- Read MJCF body/joint attributes (pos, quat, range, axis, damping, frictionloss) live from the parsed XML tree rather than hardcoding numeric values as constants in the script — this way, if `robot.xml`/`soarm_gripper.xml` are edited again in a future plan (e.g. a re-run of the calibration cross-check), re-running `mjcf_to_urdf.py` picks up the change automatically, keeping TWIN-06's URDF/MJCF agreement "by construction" rather than requiring a second manual sync step.
- The per-link CoppeliaSim mesh-placement offset (the `<visual>`/`<collision>` `<origin>` shared across every mesh within one arm link) and the mesh filename lists themselves are NOT present anywhere in the MJCF — these are pure CoppeliaSim-export artifacts. Per the plan's explicit instruction to "reuse its per-link CoppeliaSim mesh filename lists, NOT its kinematic structure," these are hardcoded as module-level constants (`MESH_PLACEMENT`, `LINK_MESHES`), sourced from a direct read of the broken `So-101/So-101.urdf` during planning — the script itself does not read that file at runtime.
- Kept `effort`/`velocity`/`dynamics` attributes on the `wrist_roll` `<limit>` consistent with the other 4 arm revolute joints (effort from the shared STS3215 `torq_wrist_roll` actuator's `ctrlrange`, velocity as the documented placeholder, dynamics from the joint's own MJCF `damping`/`frictionloss`) even though the plan's acceptance criteria only exact-string-matched the `lower`/`upper` attributes — this keeps the generated URDF internally consistent rather than leaving `wrist_roll` as a special case with missing attributes.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed invalid XML comment syntax (embedded "--")**

- **Found during:** Task 1, first self-test run of `scripts/mjcf_to_urdf.py`
- **Issue:** Three `ET.Comment(...)` calls in the initial implementation embedded a literal ` -- ` inside the comment text (e.g. `"Generated by ... -- do not hand-edit"`). XML forbids `--` anywhere inside a comment body (only allowed as the closing `-->` delimiter) — `ET.parse()` on the freshly-generated file immediately raised `xml.etree.ElementTree.ParseError: not well-formed (invalid token)`, caught by the plan's own verification step before any commit.
- **Fix:** Replaced all three ` -- ` occurrences with single-hyphen ` - ` in the comment strings (module header comment, velocity-placeholder comment, and the `right_hand`-has-no-mesh link comment).
- **Files modified:** scripts/mjcf_to_urdf.py
- **Verification:** Re-ran the script; `ET.parse('So-101/So-101.urdf')` succeeds; structural check confirms 9 links / 8 joints.
- **Committed in:** aee8447 (Task 1 commit — fixed before the first commit, not a separate follow-up commit)

---

**Total deviations:** 1 auto-fixed (1 bug, caught by the plan's own verification step before commit)
**Impact on plan:** No scope creep — this was a self-contained syntax bug in the script I wrote, caught immediately by the plan's own `<verify>` step, fixed in place before the task's single commit.

## Issues Encountered

- `So-101/` and `coppelia/` are new, currently-*untracked* directories in the main repository checkout (confirmed via the initial `git status` context: `?? So-101/`, `?? coppelia/`) — they exist on disk in the shared checkout but are not part of any git commit, so this isolated worktree (which only materializes committed history) does not have them. This plan's `<read_first>` instructions required reading `So-101/So-101.urdf` (for its per-link mesh filename lists) and `coppelia/soarm_parallel_gripper.urdf` + `coppelia/export_model_library.py` (for structural/convention reference) — all three were read directly via absolute path from the shared checkout (`/Users/hp/Desktop/Work/Repositories/SoARM-Research/So-101/So-101.urdf` etc.), which is a plain filesystem read, not a git operation, and does not touch or modify the shared checkout. No files were copied into the worktree because the converter script only needs `robot.xml`/`soarm_gripper.xml` (both tracked, present in this worktree) at runtime — the mesh filenames and structural conventions from the untracked reference files were captured as hardcoded constants in the script itself. This plan's own deliverables (`scripts/mjcf_to_urdf.py`, `So-101/So-101.urdf`) are committed cleanly to this worktree's branch as usual; the orchestrator's merge will need `So-101/`'s and `coppelia/`'s other (untracked, mesh/reference) files to be committed to the main branch separately at some point for the repo to be self-contained, but that is out of this plan's `files_modified` scope.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `So-101/So-101.urdf` is now correct and available for Plan 10-04's `yourdfpy`-based load verification (TWIN-01..05 test suite) and Plan 10-05's remaining work.
- Full LIBERO test suite green — no regressions introduced by Plans 10-01/10-02/10-03's combined MJCF/URDF changes.
- Flag for the orchestrator/next plan: `So-101/` and `coppelia/` directories (containing the actual `.dae`/`.stl` mesh assets the generated URDF references by relative path) remain untracked in the shared main checkout — worth a `git add` at some point outside this plan's scope so the repo is self-contained for anyone who clones fresh.

---
*Phase: 10-digital-twin-fidelity*
*Completed: 2026-09-19*
