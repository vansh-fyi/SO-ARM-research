---
phase: 04-dataset-collection
plan: 05
subsystem: testing
tags: [teleoperation, keyboard, robosuite, hdf5, mujoco, data-collection]

requires:
  - phase: 04-01
    provides: build_recording_env, gather_demonstrations_as_hdf5 (shared recording/writer infra)
  - phase: 04-03
    provides: verify_states_only (states-only determinism check)
provides:
  - Keyboard-only teleoperation collector (teleop.py) reusing the scripted path's env builder + HDF5 writer verbatim
  - Cross-source schema-convergence unit test (writer output is action-source-agnostic)
  - Two real human-collected teleop demos on disk, both determinism-verified
affects: [05, 06, phase-5-spatial, vla-finetuning]

tech-stack:
  added: []
  patterns:
    - "Teleop reuses the scripted collector's build_recording_env + gather_demonstrations_as_hdf5 (schema convergence by construction, not by parallel maintenance)"
    - "Fresh per-run scratch dir for DataCollectionWrapper so gather never re-sweeps prior runs"

key-files:
  created:
    - LIBERO/libero/libero/datasets/teleop.py
  modified:
    - LIBERO/libero/libero/datasets/test_hdf5_writer.py
    - LIBERO/libero/libero/datasets/__init__.py
    - LIBERO/libero/libero/assets/grippers/soarm_gripper.xml
    - LIBERO/libero/libero/assets/grippers/soarm_parallel/main_frame_visual.stl

key-decisions:
  - "Keyboard-only (D-02) — no SpaceMouse/3D-mouse code path at all"
  - "Robosuite Keyboard starts its own pynput listener; the older GLFW add_keypress_callback wiring was removed (crashed on OpenCVRenderer)"
  - "Added the roboninecom Main frame mesh as a VISUAL-ONLY geom to fix the 'floating claws' render — physics unchanged, prior data stays valid"
  - "collect_teleop defaults to a FRESH mkdtemp staging dir per run so gather_demonstrations_as_hdf5 gathers only that run's episodes"

patterns-established:
  - "Sim visual-fidelity fixes validated by offscreen MuJoCo renders inspected directly (no live viewer needed)"

requirements-completed: [DATA-04]

coverage:
  - id: D1
    description: "teleop.py — keyboard-only collection loop reusing raw_recorder + hdf5_writer verbatim"
    requirement: "DATA-04"
    verification:
      - kind: unit
        ref: "conda run -n libero python3 -c 'from libero.libero.datasets.teleop import run_teleop_episode, collect_teleop' (import) + '! grep SpaceMouse teleop.py'"
        status: pass
    human_judgment: false
  - id: D2
    description: "test_schema_matches_across_sources — writer obs key set/dtype/shape is source-agnostic"
    requirement: "DATA-04"
    verification:
      - kind: unit
        ref: "LIBERO/libero/libero/datasets/test_hdf5_writer.py#test_schema_matches_across_sources"
        status: pass
    human_judgment: false
  - id: D3
    description: "Real human-collected teleop episode(s) on disk, determinism-verified (closes DATA-02 wording for the human path)"
    requirement: "DATA-04"
    verification:
      - kind: manual_procedural
        ref: "collect_teleop -> demos written: 1 (human drove SOARM, cream_cheese into bowl); confirmed by user this session"
        status: pass
      - kind: integration
        ref: "verify_states_only(cream_cheese_teleop_demo.hdf5)=5814/5814 ; verify_states_only(cream_cheese_teleop_demo_v2.hdf5)=2061/2061"
        status: pass
    human_judgment: false

duration: multi-session
completed: 2026-08-03
status: complete
---

# Phase 04 / Plan 05: Keyboard Teleoperation Collector Summary

**Keyboard-only SOARM teleop collector reusing the scripted path's env builder + HDF5 writer, proven schema-convergent and validated by two real human-driven, determinism-verified demos.**

## Performance

- **Duration:** multi-session (Tasks 1–2 in a prior session; Task 3 human checkpoint + two mid-execution fixes this session)
- **Completed:** 2026-08-03
- **Tasks:** 3 (2 auto + 1 blocking human-action checkpoint)
- **Files modified:** 5

## Accomplishments
- `teleop.py`: keyboard-only (`input2action` + `robosuite.devices.Keyboard`) collection loop, structurally identical to the scripted collector's episode loop, reusing `build_recording_env` (`has_renderer=True`) and `gather_demonstrations_as_hdf5` verbatim — schema convergence is structural.
- `test_schema_matches_across_sources`: proves the HDF5 writer's obs key set / dtype / per-step shape is a pure function of the recording mechanism, not the action source (scripted vs teleop).
- **Two real human-collected teleop demos** on disk (`cream_cheese_teleop_demo.hdf5` = 5814 states; `cream_cheese_teleop_demo_v2.hdf5` = 2061 states), each passing `verify_states_only` at 100% — closing DATA-02's "any recorded demonstration" wording for the human-collected path.

## Task Commits

1. **Task 1: teleop.py keyboard-only loop** — `d2a607d` (feat)
2. **Task 2: test_schema_matches_across_sources** — `e165ca7` (test)
3. **Task 3: human teleop session** — human-action checkpoint (data artifacts are gitignored; not a code commit)

**Task 3 support / mid-execution fixes:**
- `5e5d8ef` (docs) — retargeted Task 3's how-to-verify from the abandoned bowl/plate task to the real cream_cheese/bowl task
- `eb2a745` (fix) — removed stale GLFW add_keypress_callback wiring that crashed on OpenCVRenderer; rely on Keyboard's own pynput listener
- `7a834c3` (fix) — added roboninecom Main frame visual so the gripper renders connected (see Deviations)
- `88bdd31` (fix) — collect_teleop uses a fresh staging dir per run (see Deviations)

## Files Created/Modified
- `LIBERO/libero/libero/datasets/teleop.py` — keyboard-only teleop collector (`run_teleop_episode`, `collect_teleop`)
- `LIBERO/libero/libero/datasets/test_hdf5_writer.py` — appended `test_schema_matches_across_sources`
- `LIBERO/libero/libero/datasets/__init__.py` — export teleop symbols (guarded import)
- `LIBERO/libero/libero/assets/grippers/soarm_gripper.xml` — added Main frame visual geom (visual-only)
- `LIBERO/libero/libero/assets/grippers/soarm_parallel/main_frame_visual.stl` — roboninecom Main frame mesh

## Decisions Made
- Keyboard-only per D-02; no SpaceMouse path.
- The gripper Main-frame visual was added at the user's request after the "floating claws" observation during live teleop; verified it's a visual-only gap (physics jaws were always attached — proven by 04-02's 78% scripted grasp success and the teleop demos passing determinism).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 – Missing Visual] Gripper rendered as "floating claws"**
- **Found during:** Task 3 (live human teleop)
- **Issue:** `soarm_gripper.xml` drew only the two moving clamp jaws; the fixed Main frame (roboninecom RB9.01.062.010) that bolts to the wrist was never modeled, so the clamps rendered detached from the arm — making teleop hard to aim.
- **Fix:** Added the real Main frame STL as a visual-only geom on the fixed base body (cyclic-axis rotation `quat 0.5 0.5 0.5 0.5`); placement dialed in via offscreen renders and confirmed on the full arm. No physics/joint/actuator/data change.
- **Files modified:** soarm_gripper.xml, soarm_parallel/main_frame_visual.stl
- **Verification:** Offscreen renders show the gripper connected wrist→jaws; model loads; existing demos still pass determinism.
- **Committed in:** `7a834c3`

**2. [Rule 1 – Correctness] collect_teleop re-gathered prior runs' demos**
- **Found during:** Task 3 (re-collection to a "v2" file)
- **Issue:** `collect_teleop` defaulted `tmp_directory` to a fixed per-task path; `gather_demonstrations_as_hdf5` sweeps every successful episode there, so a second run silently re-gathered the first run's demo (v2 came out with old+new) and the staging dir grew unbounded.
- **Fix:** Default to a fresh `tempfile.mkdtemp()` per run; explicit `tmp_directory` still allowed for deliberate accumulation. Cleaned v2 to hold only the new demo and cleared the stale staging dir.
- **Files modified:** teleop.py
- **Verification:** Cleaned v2 = 1 demo (2061 states), `verify_states_only` 2061/2061.
- **Committed in:** `88bdd31`

---

**Total deviations:** 2 auto-fixed (1 missing-visual, 1 correctness)
**Impact on plan:** Both improve fidelity/correctness of the DATA-04 deliverable. The gripper visual is a shared-robot-model change (used by Phases 4/5/6) — visual-only, no behavioral impact.

## Issues Encountered
- The demo initially appeared to have `demos written: 0` (96-byte file) — this was a mid-write read racing `gather_demonstrations_as_hdf5`; the file completed at 574 MB with a valid demo.

## User Setup Required
None.

## Next Phase Readiness
- DATA-04 complete: keyboard teleop path exists, schema-convergent with the scripted path, with real human demos on disk. DATA-01/02/03 already complete → Phase 4 dataset collection fully delivered.
- Note for Phases 5/6: `soarm_gripper.xml` now renders the full parallel gripper (Main frame + jaws); still a functional (not mechanically-exact) model — gear/rack are abstracted as two prismatic joints.

---
*Phase: 04-dataset-collection*
*Completed: 2026-08-03*
