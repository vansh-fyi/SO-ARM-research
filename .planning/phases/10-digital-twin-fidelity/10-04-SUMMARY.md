---
phase: 10-digital-twin-fidelity
plan: 04
subsystem: robot-model
tags: [yourdfpy, urdf, pytest, kinematics, so-101, verification]

# Dependency graph
requires:
  - phase: 10-digital-twin-fidelity (10-03)
    provides: So-101/So-101.urdf regenerated with correct 9-link/8-joint kinematic chain, scripts/mjcf_to_urdf.py
provides:
  - scripts/verify_urdf.py — standalone yourdfpy-based diagnostic CLI (load_generated_urdf, build_parent_map, is_descendant, no_absolute_mesh_paths)
  - scripts/test_verify_urdf.py — pytest suite independently confirming TWIN-01..05 against the generated URDF
  - yourdfpy 0.0.60 installed into the libero conda env
affects: [10-05]

# Tech tracking
tech-stack:
  added: [yourdfpy 0.0.60]
  patterns:
    - "Standalone diagnostic CLI mirrors diagnostics/measure_object_depth.py's print-✓/⚠ idiom (no raised exceptions on the standalone-run path)"
    - "pytest file imports the diagnostic CLI's own tree-walk helpers rather than duplicating logic, keeping one source of truth for the parent-map/descendant-walk algorithm"

key-files:
  created: [scripts/verify_urdf.py, scripts/test_verify_urdf.py]
  modified: []

key-decisions:
  - "Used yourdfpy.URDF.load(..., load_meshes=False, build_collision_scene_graph=False) to load the URDF without requiring the referenced STL/DAE mesh files to be resolvable on disk — this test verifies kinematic/joint structure, not mesh geometry, and the coppelia/ mesh directory is untracked in this worktree (same known gap 10-03 flagged)"
  - "test_joint_limits_match_calibration skips (not fails) if the live LeRobot calibration JSON is absent, since TWIN-05 is an independent recomputation against real hardware calibration data that may not exist in every environment this test suite runs in"

patterns-established:
  - "Any future scripts/mjcf_to_urdf.py re-run should be followed by re-running `pytest scripts/test_verify_urdf.py` to reconfirm TWIN-01..05 against the regenerated URDF"

requirements-completed: [TWIN-01, TWIN-02, TWIN-03, TWIN-04, TWIN-05, TWIN-06]

coverage:
  - id: D1
    description: "yourdfpy successfully loads So-101/So-101.urdf via an independent, standard URDF parser (not the generator's own logic)"
    requirement: "TWIN-06"
    verification:
      - kind: unit
        ref: "scripts/test_verify_urdf.py::test_gripper_is_child_of_wrist (fixture load succeeds as a precondition for all 5 tests)"
        status: pass
    human_judgment: false
  - id: D2
    description: "Gripper (right_gripper) is a proper descendant of right_hand (wrist), not a disconnected root-level body"
    requirement: "TWIN-01"
    verification:
      - kind: unit
        ref: "scripts/test_verify_urdf.py::test_gripper_is_child_of_wrist"
        status: pass
    human_judgment: false
  - id: D3
    description: "wrist_roll is revolute with the exact asymmetric mechanical-limit range from robot.xml"
    requirement: "TWIN-02"
    verification:
      - kind: unit
        ref: "scripts/test_verify_urdf.py::test_wrist_roll_range"
        status: pass
    human_judgment: false
  - id: D4
    description: "gripper_left/gripper_right are prismatic with correct opposing jaw-travel axes"
    requirement: "TWIN-03"
    verification:
      - kind: unit
        ref: "scripts/test_verify_urdf.py::test_gripper_joint_axes"
        status: pass
    human_judgment: false
  - id: D5
    description: "No mesh filename in the generated URDF is an absolute path or file:// URI"
    requirement: "TWIN-04"
    verification:
      - kind: unit
        ref: "scripts/test_verify_urdf.py::test_no_absolute_mesh_paths"
        status: pass
    human_judgment: false
  - id: D6
    description: "The 4 calibration-derived arm joint limits independently recompute-match robot.xml/the URDF via calibration_utils, and wrist_roll's URDF limit does NOT equal the naive full-turn calibration placeholder (Pitfall 1 regression guard)"
    requirement: "TWIN-05"
    verification:
      - kind: unit
        ref: "scripts/test_verify_urdf.py::test_joint_limits_match_calibration"
        status: pass
    human_judgment: false
  - id: D7
    description: "Pre-existing env.reset()/camera-config regression test remains green after the full Phase 10 MJCF+URDF change set"
    requirement: "TWIN-06"
    verification:
      - kind: integration
        ref: "conda run -n libero python3 -m pytest LIBERO/libero/libero/envs/test_camera_config.py -x"
        status: pass
    human_judgment: false

duration: ~30min
completed: 2026-09-19
status: complete
---

# Phase 10 Plan 04: Verify Generated URDF Against TWIN-01..05 Summary

**`yourdfpy`-based independent load-and-assert verification (`scripts/verify_urdf.py` + `scripts/test_verify_urdf.py`) confirms Plan 10-03's generated `So-101/So-101.urdf` satisfies TWIN-01..05, using a standard third-party URDF parser rather than trusting the generator's own output.**

## Performance

- **Duration:** ~30 min
- **Started:** 2026-09-19T05:50:00Z (approx, worktree spawn)
- **Completed:** 2026-09-19T06:19:44Z
- **Tasks:** 2/2 completed
- **Files modified:** 2 (both new)

## Accomplishments

- `yourdfpy` 0.0.60 installed into the `libero` conda env after a pre-approved Package Legitimacy Gate sign-off (PyPI project page, linked GitHub source `clemense/yourdfpy`, and description all independently verified prior to this session)
- `scripts/verify_urdf.py`: a standalone diagnostic CLI (`load_generated_urdf`, `build_parent_map`, `is_descendant`, `no_absolute_mesh_paths`) that loads `So-101/So-101.urdf` via `yourdfpy` and prints `✓`/`⚠` lines for each of the 4 structural checks; run standalone it printed 4 `✓` lines (0 `⚠`)
- `scripts/test_verify_urdf.py`: 5 pytest tests, all passing, independently confirming TWIN-01 (gripper chained under wrist), TWIN-02 (`wrist_roll` range), TWIN-03 (gripper joint axes), TWIN-04 (no absolute mesh paths), and TWIN-05 (calibration-derived joint limits match, with an explicit Pitfall 1 regression guard against the naive full-turn `wrist_roll` value)
- Combined run of `scripts/test_verify_urdf.py` + the existing `LIBERO/libero/libero/envs/test_camera_config.py` regression test: 7 passed, 0 failed (TWIN-06 wave-merge gate)

## Task Commits

Each task was committed atomically:

1. **Task 1: yourdfpy package legitimacy sign-off** - no commit (pure checkpoint task; the install itself is bundled into Task 2's commit since the plan's `files_modified` only lists the two Python files, and `pip install` has no separate git-tracked artifact in this repo's convention)
2. **Task 2: Write verify_urdf.py + test_verify_urdf.py, run TWIN-01..05 tests** - `363a7ca` (feat)

**Plan metadata:** (this commit, appended after this SUMMARY)

## Files Created/Modified

- `scripts/verify_urdf.py` - new standalone diagnostic CLI: `load_generated_urdf()`, `build_parent_map()`, `is_descendant()`, `no_absolute_mesh_paths()`, `main()` (print-✓/⚠ output)
- `scripts/test_verify_urdf.py` - new pytest suite: `test_gripper_is_child_of_wrist`, `test_wrist_roll_range`, `test_gripper_joint_axes`, `test_no_absolute_mesh_paths`, `test_joint_limits_match_calibration`

## Decisions Made

- Loaded the URDF with `load_meshes=False, build_collision_scene_graph=False` — this verification is about kinematic/joint structure (parent/child chaining, joint types/axes/limits, mesh filename strings), not mesh geometry itself; it doesn't need the actual STL/DAE files to be resolvable on disk, which also sidesteps this worktree's known gap (the `coppelia/meshes/` directory referenced by relative path is untracked in the shared main checkout, same issue 10-03's SUMMARY flagged for `So-101/`/`coppelia/`).
- `test_joint_limits_match_calibration` uses `pytest.skip` (not a hard failure) when the live LeRobot calibration JSON is absent from `~/.cache/huggingface/lerobot/calibration/robots/so_follower/`, since TWIN-05 is inherently an environment-dependent cross-check against real hardware calibration data that won't exist on every machine this suite might run on. In this session the calibration file (`soarm_follower_02.json`) was present, so the test ran its full assertion path (not the skip path).
- Followed the plan's exact assertion tolerances: `abs=1e-6` for TWIN-02's `wrist_roll` range, `abs=1e-5` for TWIN-05's calibration-derived limits.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- Same known gap 10-03 flagged: `coppelia/` (containing the `.stl` mesh files the generated URDF references by relative path) is untracked in the shared main checkout, so it isn't present in this isolated worktree. This plan's tests don't need those mesh files to be resolvable on disk (loaded with `load_meshes=False`), so it wasn't a blocker here, but the orchestrator/next plan should still `git add` `So-101/` and `coppelia/` at some point so the repo is self-contained for anyone who clones fresh.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `scripts/verify_urdf.py` and `scripts/test_verify_urdf.py` are now available as the phase's standing automated acceptance gate — any future `scripts/mjcf_to_urdf.py` re-run should be followed by `pytest scripts/test_verify_urdf.py` to reconfirm TWIN-01..05.
- All 6 v2.0 TWIN-* requirements (TWIN-01 through TWIN-06) covered by Phase 10 as of this plan; Plan 10-05 (if it exists) can build on a fully-verified digital twin.
- Flag carried forward from 10-03: `So-101/` and `coppelia/` directories remain untracked in the shared main checkout — needs a `git add` outside this plan's scope.

---
*Phase: 10-digital-twin-fidelity*
*Completed: 2026-09-19*

## Self-Check: PASSED

- FOUND: scripts/verify_urdf.py
- FOUND: scripts/test_verify_urdf.py
- FOUND: .planning/phases/10-digital-twin-fidelity/10-04-SUMMARY.md
- FOUND commit: 363a7ca (Task 2: feat(10-04))
- FOUND commit: 9524b40 (docs(10-04): complete plan)
