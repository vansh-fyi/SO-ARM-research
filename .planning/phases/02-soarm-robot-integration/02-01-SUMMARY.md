---
phase: 02-soarm-robot-integration
plan: 01
subsystem: simulation
tags: [mujoco, robosuite, libero, mjcf, so-arm100, vendoring]

requires:
  - phase: 01-colab-environment-setup
    provides: validated LIBERO dependency stack and local `libero` conda env conventions
provides:
  - explorations/soarm_sanity.py — local CPU validation harness (7 check modes, max_contact_force SC-1 probe)
  - Vendored SO101 hardware description at pinned SHA (source XML + 13 STLs + VENDOR.txt provenance)
  - robosuite-1.4/MuJoCo-2.3.7-compatible arm MJCF (robot.xml) with right_hand body + eye_in_hand camera
affects: [02-02 gripper+classes, 02-03 physics tuning, 02-04 camera tuning, 02-05 colab verification]

tech-stack:
  added: []
  patterns:
    - SHA-pinned asset vendoring with VENDOR.txt (repo URL, 40-hex commit, per-file sha256)
    - soarm_sanity.py check pattern — each check prints PASS/FAIL line, main exits nonzero on any failure

key-files:
  created:
    - explorations/soarm_sanity.py
    - LIBERO/libero/libero/assets/robots/soarm101/robot.xml
    - LIBERO/libero/libero/assets/robots/soarm101/so101_new_calib.source.xml
    - LIBERO/libero/libero/assets/robots/soarm101/VENDOR.txt
    - LIBERO/libero/libero/assets/robots/soarm101/assets/ (13 STL meshes)
  modified:
    - .gitignore (narrowed explorations/ ignore so the harness is git-tracked)

key-decisions:
  - "Files under LIBERO/ commit to the embedded LIBERO git repo (LIBERO/ is gitignored in the outer repo) — all later plans and verifier greps must check BOTH repos"
  - "Upstream pinned at TheRobotStudio/SO-ARM100 fda892cba81032c46c40976a48c9ceadbf40a9ca; fetched via pinned-SHA raw URLs (T-02-01 mitigation)"
  - "qpos=0 is the documented new-calib home pose — no keyframe element needed (D-07)"

patterns-established:
  - "Expected-RED progression: compile check passes arm stage, fails only on missing gripper XML until 02-02"
  - "MJCF adaptation: inline all defaults then delete default blocks (robosuite merge drops them)"

requirements-completed: [ENV-04]

coverage:
  - id: D1
    description: "soarm_sanity.py validation harness with --check compile|model|reset|render|soak|tasks|all, max_contact_force via mj_contactForce"
    requirement: "ENV-04"
    verification:
      - kind: other
        ref: "conda run -n libero python explorations/soarm_sanity.py --help (exit 0); --check compile detects missing artifacts with actionable paths"
        status: pass
    human_judgment: false
  - id: D2
    description: "SO101 geometry vendored with pinned provenance (source XML, 13 STLs, VENDOR.txt with 40-hex SHA + per-file sha256)"
    requirement: "ENV-04"
    verification:
      - kind: other
        ref: "Task 2 verify block — ET mesh enumeration vs assets/ dir + VENDOR.txt SHA regex (VENDOR OK — 13 meshes present, SHA pinned)"
        status: pass
    human_judgment: false
  - id: D3
    description: "Adapted arm MJCF robot.xml — compiles under MuJoCo 2.3.7; 5 motors ctrlrange ±2.94; zero default blocks; right_hand body with eye_in_hand camera; all meshes local assets/ paths"
    requirement: "ENV-04"
    verification:
      - kind: other
        ref: "Task 3 verify block — mujoco.MjModel.from_xml_path + ElementTree structural assertions (ARM XML OK)"
        status: pass
    human_judgment: false

duration: ~15min (prior session) + close-out
completed: 2026-07-11
status: complete
---

# Plan 02-01 Summary

**SOARM SO101 arm MJCF compiling clean under MuJoCo 2.3.7 with vendored SHA-pinned geometry and a 7-mode CPU sanity harness that lands RED exactly where plans 02-02/02-03 will turn it GREEN**

## Performance

- **Duration:** ~15 min execution (prior session, interrupted before close-out); verified and closed out by orchestrator
- **Started:** 2026-07-11T12:19+05:30
- **Completed:** 2026-07-11T12:57+05:30
- **Tasks:** 3
- **Files modified:** 18 (1 harness + .gitignore in outer repo; 16 asset files in LIBERO repo)

## Accomplishments
- `explorations/soarm_sanity.py`: all 7 check modes implemented (compile/model/reset/render/soak/tasks/all), `max_contact_force` via `mj_contactForce` for the SC-1 <10 N criterion, `MUJOCO_GL=glfw` set before any mujoco import, lazy heavy imports so `--help` works anywhere
- SO101 hardware description vendored at pinned commit `fda892cba81032c46c40976a48c9ceadbf40a9ca` (TheRobotStudio/SO-ARM100): pristine `so101_new_calib.source.xml`, 13 STL meshes, VENDOR.txt with per-file sha256 — zero runtime downloads remain
- `robot.xml` adapted per the 9-item checklist: kv attribute stripped (the MuJoCo 2.3.7 schema blocker), defaults fully inlined, 5 motor actuators at ctrlrange ±2.94, gripper body renamed `right_hand` (jaw geoms excluded for 02-02), `eye_in_hand` + `robotview` cameras added

## Task Commits

Cross-repo split — LIBERO/ is an embedded git repo, gitignored in the outer repo:

1. **Task 1: soarm_sanity.py harness** - `31a7a61` (feat, **outer repo**)
2. **Task 2: vendor SO101 XML + 13 STLs** - `2a700d8` (chore, **inner LIBERO repo**)
3. **Task 3: adapted arm robot.xml** - `a9d2c1e` (feat, **inner LIBERO repo**)

## Files Created/Modified
- `explorations/soarm_sanity.py` - CPU validation harness; the phase's success-criteria measuring stick
- `LIBERO/libero/libero/assets/robots/soarm101/robot.xml` - adapted arm MJCF (120 lines, fully inlined)
- `LIBERO/libero/libero/assets/robots/soarm101/so101_new_calib.source.xml` - pristine upstream copy for re-derivation
- `LIBERO/libero/libero/assets/robots/soarm101/VENDOR.txt` - SHA pin + license + per-file sha256
- `LIBERO/libero/libero/assets/robots/soarm101/assets/*.stl` - 13 vendored meshes
- `.gitignore` - narrowed so the harness is tracked

## Decisions Made
- Commits for LIBERO/ paths land in the embedded LIBERO repo — later plans, spot-checks, and the verifier must grep both repos' logs
- Trusted the source XML's mesh enumeration (13 meshes) over the research count, per plan instruction

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- Prior execution session was interrupted after writing robot.xml but before committing it (no SUMMARY.md, Task 3 uncommitted). Orchestrator safe-resume: re-ran all three task verify blocks (all PASS), confirmed the expected-RED harness state (`compile` fails only on the missing gripper XML), committed Task 3, and closed out.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Arm XML compiles standalone; gripper XML + MountedSoarm101/SoarmGripper classes (02-02) are the next RED→GREEN step
- `--check compile` arm stage GREEN; gripper stage RED as designed until 02-02
- Jaw geoms, moving_jaw body, its joint, and the gripper actuator remain only in the pristine source XML, ready for 02-02 extraction

---
*Phase: 02-soarm-robot-integration*
*Completed: 2026-07-11*
