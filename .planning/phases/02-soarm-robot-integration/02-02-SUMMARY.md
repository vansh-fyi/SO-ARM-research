---
phase: 02-soarm-robot-integration
plan: 02
subsystem: simulation
tags: [mujoco, robosuite, libero, mjcf, gripper, registration, so-arm100]

requires:
  - phase: 02-soarm-robot-integration
    plan: 01
    provides: adapted arm robot.xml, vendored SO101 geometry, soarm_sanity.py harness
provides:
  - LIBERO/libero/libero/assets/grippers/soarm_gripper.xml — robosuite-contract gripper MJCF (eef body, grip_site sites, ft_frame sensors, 1 position actuator)
  - MountedSoarm101(ManipulatorModel) with tabletop base placement — LIBERO/libero/libero/envs/robots/soarm.py
  - SoarmGripper(GripperModel) 1-DOF integrated jaw action — LIBERO/libero/libero/envs/grippers/soarm_gripper.py
  - SOARM selectable by name via ROBOT_CLASS_MAPPING/GRIPPER_MAPPING registration in robots/__init__.py
affects: [02-03 physics tuning, 02-04 camera/task selection, 02-05 colab verification]

tech-stack:
  added: []
  patterns:
    - Fork-local ASSETS constant for XML paths (never robosuite's site-packages path completion)
    - Import-time robosuite dict mutation in the LIBERO fork (zero site-packages edits)

key-files:
  created:
    - LIBERO/libero/libero/assets/grippers/soarm_gripper.xml
    - LIBERO/libero/libero/envs/robots/soarm.py
    - LIBERO/libero/libero/envs/grippers/__init__.py
    - LIBERO/libero/libero/envs/grippers/soarm_gripper.py
  modified:
    - LIBERO/libero/libero/envs/robots/__init__.py (registration block extended)
    - explorations/soarm_sanity.py (repo root added to sys.path — Rule 3 fix)

key-decisions:
  - "Gripper actuator keeps source servo values: kp 998.22, ctrlrange -0.17453..1.74533 (source joint range), forcerange ±3.35 — kv stripped for MuJoCo 2.3.7"
  - "Explicit right_gripper inertial (mass 0.034) added so mesh density is not double-counted across the visual+collision geom pair; tune in 02-03"
  - "Harness must insert the repo ROOT on sys.path — the LIBERO working tree imports itself via the absolute 'LIBERO.libero...' prefix (uncommitted pre-existing drift)"
  - "reset and render checks landed GREEN already with Pattern 2/3 starting values — 02-03 starts from a working baseline, not a broken one"

patterns-established:
  - "Dual-name module identity: libero.libero.envs.* and LIBERO.libero.libero.envs.* both execute at import; robosuite dict registration is idempotent so this is harmless"

requirements-completed: [ENV-04, ENV-05]

coverage:
  - id: D1
    description: "soarm_gripper.xml compiles under MuJoCo 2.3.7 with full robosuite contract (eef body, grip_site/grip_site_cylinder/ee* sites, force_ee/torque_ee sensors, 1 position actuator, jaw collision geoms, zero default blocks)"
    requirement: "ENV-04"
    verification:
      - kind: other
        ref: "Task 1 verify block — MjModel.from_xml_path + ElementTree assertions (GRIPPER XML OK); soarm_sanity --check compile exits 0"
        status: pass
    human_judgment: false
  - id: D2
    description: "MountedSoarm101 + SoarmGripper instantiate with full property contract (default_mount None, init_qpos zeros(5), table z=0.90, gripper dof 1)"
    requirement: "ENV-04"
    verification:
      - kind: other
        ref: "Task 2 verify block — CLASSES OK; literal xml_path_completion grep returns 0"
        status: pass
    human_judgment: false
  - id: D3
    description: "MountedSoarm101 in ROBOT_CLASS_MAPPING, SoarmGripper in GRIPPER_MAPPING, Panda entries preserved; SOARM name-selectable exactly as Panda"
    requirement: "ENV-05"
    verification:
      - kind: other
        ref: "soarm_sanity --check model exits 0 (PASS — model); REGISTRATION OK assertion suite"
        status: pass
    human_judgment: false

duration: ~10min
completed: 2026-07-11
status: complete
---

# Plan 02-02 Summary

**SOARM SO101 is now selectable by name in LIBERO: gripper MJCF + MountedSoarm101/SoarmGripper classes + robosuite mapping registration flip `--check compile` and `--check model` GREEN — and `--check reset` (0.024 N contact) plus `--check render` landed GREEN early with the untuned starting values**

## Performance

- **Duration:** ~10 min
- **Started:** 2026-07-11T13:35:18Z
- **Completed:** 2026-07-11T13:44:26Z
- **Tasks:** 3
- **Files modified:** 6 (5 in inner LIBERO repo, 1 harness fix in outer repo)

## Accomplishments

- `soarm_gripper.xml`: jaw parts extracted from the pristine source XML (fixed jaw `wrist_roll_follower_so101_v1` + `moving_jaw_so101_v1` body with its `gripper` hinge), wrapped in the panda_gripper.xml structural contract — `right_gripper` root at identity, `eef` body at the source gripperframe pos `(-0.0079, -0.0002, -0.0981)`, all 7 contract sites, `force_ee`/`torque_ee` sensors, exactly one position actuator, defaults fully inlined
- `MountedSoarm101`: 8-property contract with 5-element damping array, tabletop base placement (`table` lambda z=0.90 per Pitfall 6), `default_mount` None (D-04), `init_qpos` zeros(5) (D-07)
- `SoarmGripper`: 1-DOF integrated jaw action (speed 0.10, init_qpos [0.8]), `_important_geoms` mapped to `fixed_jaw_collision`/`moving_jaw_collision`
- Registration entirely in the fork at import time: `ROBOT_CLASS_MAPPING["MountedSoarm101"] = SingleArm`, `GRIPPER_MAPPING["SoarmGripper"] = SoarmGripper`, Panda entries untouched, zero robosuite site-packages edits (D-08)

## Gripper source values recorded (per plan output spec)

- Joint `gripper` range: `-0.17453297762778586 .. 1.7453291995659765` rad (source joint range)
- Actuator: kp `998.22` (source sts3215 class, `kv="2.731"` stripped — 2.3.7 schema), ctrlrange `-0.17453 1.74533` (source actuator = rounded joint range), forcerange `-3.35 3.35` (source actuator override, NOT the sts3215 class ±2.94)
- Joint inline values: damping `0.6`, frictionloss `0.052`, armature `0.028`
- No deviation from RESEARCH Pattern 2/3 starting values in the classes

## Task Commits

Cross-repo split — LIBERO/ is an embedded git repo, gitignored in the outer repo:

1. **Task 1: gripper MJCF** - `761fef3` (feat, **inner LIBERO repo**)
2. **Task 2: MountedSoarm101 + SoarmGripper classes** - `8ebc8fb` (feat, **inner LIBERO repo**)
3. **Task 3: mapping registration** - `8c7fa6f` (feat, **inner LIBERO repo**)
4. **Deviation fix: harness sys.path** - `d5ca329` (fix, **outer repo**)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Harness could not import the LIBERO fork — repo root missing from sys.path**
- **Found during:** Task 3 (`--check model` failed with `No module named 'LIBERO'` even after registration)
- **Issue:** The LIBERO working tree carries pre-existing uncommitted modifications that rewrite fork-internal imports to the absolute `LIBERO.libero...` prefix (e.g. `env_wrapper.py`, `bddl_base_domain.py`). That prefix only resolves when the repo root is on sys.path as a namespace-package anchor. `python -c` verify snippets get this for free via cwd (`''`); the harness script's `sys.path[0]` is `explorations/`, so its import chain broke.
- **Fix:** Inserted `REPO_ROOT` into sys.path in `soarm_sanity.py`'s bootstrap (outer-repo file, owned by this phase). Did NOT touch the pre-existing inner-repo modifications.
- **Files modified:** `explorations/soarm_sanity.py`
- **Commit:** `d5ca329` (outer repo)

**2. [Rule 2 - Missing critical] Explicit inertial added to `right_gripper`**
- **Found during:** Task 1
- **Issue:** Without an explicit inertial, MuJoCo computes body mass from ALL attached geoms — the fixed jaw's visual+collision mesh pair would double-count density on a moving body once merged under the arm's `right_hand`.
- **Fix:** Added `<inertial pos="0 -0.0002 -0.05" mass="0.034" diaginertia="2e-05 2e-05 1e-05"/>` (approx. fixed-jaw share of the source gripper body's 0.087 kg). `moving_jaw_so101_v1` uses its exact source inertial. Flagged for 02-03 tuning.
- **Files modified:** `LIBERO/libero/libero/assets/grippers/soarm_gripper.xml`
- **Commit:** `761fef3`

Also trivial: docstring/comment references to robosuite's path-completion helper reworded so the literal acceptance-criterion grep returns 0 (intent unchanged).

## TDD Gate Compliance

Tasks 2 and 3 were `tdd="true"`. The RED tests are plan 02-01's committed harness checks (`soarm_sanity.py --check compile|model`, commit `31a7a61`) — this plan's stated objective was to flip them GREEN. RED state was re-confirmed before each implementation step (import failure for Task 2, `--check model` FAIL for Task 3); GREEN confirmed after. No separate `test(...)` commits were created this plan because the test infrastructure predates it by design.

## Learnings for plan 02-03 (merge/namespacing)

- **Dual module identity:** with the repo root on sys.path, the fork loads under BOTH `libero.libero.envs.*` and `LIBERO.libero.libero.envs.*` names. Registration code executes twice into robosuite's shared dicts — harmless (plain dict assignment, idempotent), but debuggers should expect two class objects with the same name.
- **The `LIBERO.`-prefix imports are uncommitted working-tree drift** in the inner repo (HEAD uses clean `libero.` imports). Any new script that imports the fork must put the repo root on sys.path, not just `LIBERO/`.
- **Reset baseline already works:** `--check reset` passes with peak contact force **0.024 N** (SC-1 threshold 10 N) and `--check render` saves non-black agentview + eye_in_hand frames. 02-03's physics tuning starts from a stable baseline; `--check soak` and `--check tasks` remain unverified.

## Issues Encountered

None beyond the documented deviations.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- ENV-04 complete (both MJCFs compile + registered ManipulatorModel subclass); ENV-05 complete (name-selectable via the same mappings as Panda, env reset demonstrated)
- Expected-RED progression ahead of schedule: only `soak` and `tasks` checks remain unproven for 02-03/02-04
- Gripper init_qpos [0.8] / speed 0.10 / right_gripper inertial are flagged tuning knobs for 02-03

---
*Phase: 02-soarm-robot-integration*
*Completed: 2026-07-11*

## Self-Check: PASSED

All 6 created/modified files exist on disk; all 4 task/deviation commits found (3 inner LIBERO repo, 1 outer); no unintended deletions.
