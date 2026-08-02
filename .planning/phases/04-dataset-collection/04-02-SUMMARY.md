---
phase: 04-dataset-collection
plan: 02
subsystem: dataset
tags: [scripted-collector, waypoint-fsm, robosuite, mujoco, soarm, grasp-blocker]

# Dependency graph
requires:
  - phase: 04-dataset-collection
    plan: 01
    provides: build_recording_env, gather_demonstrations_as_hdf5, OBS_KEY_MAPPING
  - phase: 02-soarm-robot-integration
    provides: Soarm101 robot + SoarmGripper, tuned OSC_POSE, frozen 3-task list
provides:
  - "compute_waypoint_action(): pure, unit-tested 8-phase pick-place FSM step function"
  - "run_scripted_episode()/collect_task()/collect_all(): scripted collection driver + CLI"
  - "collector.TASKS (corrected 3-task list), collector.BDDL_DIR"
  - "Empirical evidence that the SOARM gripper cannot grasp the akita_black_bowl (grasp blocker)"
affects: [04-03-teleop, 04-04-replay-verification, 04-05-normalization, phase-06-finetuning]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Pure FSM step function (no sim access) + thin env-driving loop — testable waypoint control"
    - "Frozen grasp reference: stop tracking the object once grasped so the lift target is stable"
    - "Proportional OSC_POSE position control: action = clip(KP*(target-eef), -1, 1), zero rot delta"

key-files:
  created:
    - LIBERO/libero/libero/datasets/collector.py
    - LIBERO/libero/libero/datasets/test_collector.py
  modified:
    - LIBERO/libero/libero/datasets/__init__.py

key-decisions:
  - "compute_waypoint_action is a PURE function (positions in, action/next_phase/gripper out) so the 8-phase FSM is unit-testable with no sim; the env-driving loop lives separately in run_scripted_episode"
  - "Freeze the bowl reference once grasped — a live bowl+HOVER lift target chases the eef upward forever because the grasped bowl rises with the gripper"
  - "Gripper convention verified empirically: SoarmGripper -1 => open (qpos -0.17), +1 => closed (qpos 1.75)"
  - "Success-target integration test kept as xfail (assertion preserved, not weakened) against the grasp blocker, per plan guidance to surface rather than fake"

requirements-completed: []
requirements-blocked: [DATA-01]

# Metrics
duration: 70min
completed: 2026-08-03
status: blocked
---

# Phase 4 Plan 02: Scripted Waypoint Collector Summary

**A pure, unit-tested 8-phase pick-place waypoint FSM plus a full scripted collection driver/CLI were built on 04-01's recording infrastructure — but the phase's primary dataset artifact could NOT be produced: empirical testing conclusively shows the SOARM gripper cannot grasp or lift the akita_black_bowl, so no scripted trajectory reaches `On(bowl, plate)` and the 100+ successful demos are unattainable with the current robot.**

## Performance

- **Duration:** ~70 min (incl. a full real 50-attempt collection run + ~5 grasp-strategy investigations)
- **Completed:** 2026-08-03
- **Tasks:** 2 of 3 complete; Task 3 (full collection run) BLOCKED
- **Files modified:** 3 (2 created, 1 modified)

## Accomplishments

- `compute_waypoint_action` — a PURE 8-phase FSM step function (approach → descend → grasp → lift → transport → place_descend → release → retreat → done), driven only by ground-truth positions and numpy math, with 8 unit tests covering all 6 specified behaviors + sign conventions.
- `run_scripted_episode` / `collect_task` / `collect_all` + argparse CLI — the full scripted collection driver on top of 04-01's `build_recording_env` + `gather_demonstrations_as_hdf5`, with the robosuite success-hold-count exit, per-attempt success counting, and a single end-of-task HDF5 gather.
- Corrected 3-task `TASKS` constant (table_center / between_plate_ramekin / on_the_ramekin) copied verbatim from `soarm_sanity.py` — the stale `next_to_the_plate` task is absent (asserted in a test).
- A real end-to-end integration test proving the collection PLUMBING is correct (real env → rollout → schema-valid HDF5), independent of task success.
- Conclusive empirical diagnosis of the grasp blocker (see below), turning research assumption A2 / Open Question 1 from "medium risk" into a measured finding.

## Task Commits

1. **Task 1 (RED): failing FSM + integration tests** — `c3f3521` (test)
2. **Task 1 (GREEN): compute_waypoint_action pure FSM** — `edd9d50` (feat)
3. **Task 2: collection driver + honest integration tests** — `7975992` (feat)

_(Task 1 followed the TDD RED→GREEN gate: `test(04-02)` commit precedes the `feat(04-02)` implementation commit.)_

## Files Created/Modified

- `LIBERO/libero/libero/datasets/collector.py` — pure FSM + driver + CLI (created)
- `LIBERO/libero/libero/datasets/test_collector.py` — 9 FSM unit tests, 1 plumbing integration test (pass), 1 xfail'd success-target test (created)
- `LIBERO/libero/libero/datasets/__init__.py` — added guarded collector exports (modified)

## Blocker (Task 3 — DATA-01 dataset artifact unattainable with current robot)

**The SOARM gripper physically cannot grasp/lift the `akita_black_bowl`, so no scripted (or learned) trajectory reaches the `On(akita_black_bowl_1, plate_1)` success condition.** This was established with direct measurement, not inferred:

**Measured geometry (headless SOARM env, all 3 frozen tasks share the same bowl/gripper):**
- Table top surface is at **z = 0.90**. The bowl settles resting with its body origin at **z ≈ 0.898** (rim ≈ 0.928), the plate at z ≈ 0.902 — both simply resting on the table. (The z≈0.97 read at reset is a pre-settle transient.)
- The SOARM arm's **vertical reach bottoms out asymptotically at eef z ≈ 0.965–1.00** near the bowl's xy (200 steps of full-gain downward command only reached z 1.006). The jaw collision geoms sit ~0.02 m below the eef, i.e. ≈ 0.945 — **still ~0.02 m above the settled bowl rim (≈0.928)**. The gripper cannot be lowered around the bowl.
- The **jaw opening is ~0.03 m** (measured fixed↔moving jaw geoms) while the **bowl is ~0.09 m wide** — even at the right height the jaws cannot enclose the bowl; they push it.

**Empirical grasp attempts (all failed to lift the bowl):** center grasp; rim-pinch at ±x offsets (±0.03, ±0.045); position gains KP ∈ {3, 6, 25, 40}; gentle vs. aggressive descent; 20–35-step close holds. In every configuration the bowl remained at its settled height and never rose with the gripper. A **full real 50-attempt collection run on TASKS[0] produced 0 successes over 5m14s.**

**Why `On` can't be reached another way:** `On(bowl, plate)` = `plate.check_ontop(bowl)` requires (plate_z ≤ bowl_z) AND contact AND `‖bowl_xy − plate_xy‖ < 0.03 m`. The bowl starts ~0.22 m from the plate; satisfying this needs the bowl relocated and resting on the plate. Without a working grasp, the only alternative — pushing — displaces the free-floating plate rather than depositing the bowl on it.

**Consistency with prior findings:** This is exactly research assumption **A2** ("SOARM's gripper/reach geometry may make grasp geometry awkward") and **Open Question 1** ("exact grasp feasibility is an implementation-time unknown"), and it aligns with **Phase 3's confirmed baseline** that OFT/π0 reach **0% task success on SOARM** — the robot fundamentally cannot complete these manipulation tasks yet. Phase 2 validated reach *distance*, contact forces, and rendering, but never validated an actual *grasp*; this plan is the first real grasp test, and it fails.

**This requires an architectural / hardware decision (deviation Rule 4 — not auto-fixable):** options include (a) redesign/enlarge the `SoarmGripper` jaw and extend the arm's vertical reach so table-level objects are graspable (revisit Phase 2 MJCF); (b) swap the frozen task set for objects the current gripper *can* grasp (small graspable primitives sized to the ~0.03 m jaw); (c) inject the bowl-on-plate goal state directly (state-set the bowl onto the plate and record the resulting obs) to synthesize demos without a physical grasp — a legitimate scripted-data strategy but a departure from D-01's "waypoint trajectory" framing that needs sign-off; or (d) descope DATA-01's "successful task completion" requirement for the MVP. **None of these should be chosen silently by the executor.**

## Deviations from Plan

### Auto-fixed / corrected during execution

**1. [Rule 1 - Correctness] OSC_POSE action is a normalized delta, not a raw metric offset**
- **Found during:** Task 2 (first real rollout — approach took 123 steps to move ~0.09 m).
- **Issue:** RESEARCH.md Pattern 3's `clip(target - eef, -0.05, 0.05)` treats the action as a raw metric offset; OSC_POSE actions are normalized to the controller's [-1, 1] input range, so a 0.05-magnitude action is ~0.0025 m/step — far too slow.
- **Fix:** Proportional control `action = clip(KP_POS * (target - eef), -1, 1)` (KP_POS = 25). Preserves all FSM sign conventions, so the unit tests are unaffected.
- **Committed in:** `7975992`.

**2. [Rule 1 - Correctness] Freeze the bowl reference after grasp**
- **Found during:** Task 2 grasp/lift trials.
- **Issue:** Using a live `bowl_pos + HOVER` lift target makes the eef chase a target that rises with the (grasped) bowl — an unreachable, self-referential goal.
- **Fix:** `run_scripted_episode` stops re-reading the bowl once past the grasp phase and lifts to a frozen reference.
- **Committed in:** `7975992`.

**3. [Plan guidance followed] Success-target integration test marked xfail, not weakened**
- The plan forbids weakening the `>= 2` assertion. The target test keeps that exact assertion but is `@pytest.mark.xfail(strict=False, reason=<blocker>)`, so the suite is honest-green, the target is documented, and it will flip to `xpass` the day the gripper/reach is fixed. A separate passing test proves the plumbing.

## Threat Model Compliance

- **T-04-02-01 (Tampering, mitigate):** `collect_task`'s `max_attempts` hard cap is present and exercised — the unreachable-target case surfaces as a returned count below target (0) after a bounded number of attempts, never an infinite loop. Verified live (50-attempt run terminated cleanly).
- **T-04-02-02 (DoS, accept):** local CPU-bound sim only. N/A.
- **T-04-02-SC (package legitimacy):** no new package installs. Confirmed.

## Known Stubs

None — all functions are fully implemented and exercised. The missing artifact is the on-disk dataset, blocked by robot capability (above), not by stubbed code.

## Self-Check: PASSED

- Files: `collector.py`, `test_collector.py`, `__init__.py` all FOUND on disk and tracked under the lowercase `libero/...` path (verified `git diff HEAD` is clean after commit — the macOS case-collision staging trap was hit once and corrected by staging via the lowercase tracked path).
- Commits: `c3f3521` (test/RED), the `feat(04-02)` FSM commit, and `7975992` (driver) all FOUND in `git log`.
- Tests: 10 passed (9 FSM unit + 1 plumbing integration), 1 xfailed (success target, documenting the blocker). Zero hard failures.

## Next Phase Readiness / Decision Needed

- `compute_waypoint_action` + the collection driver are correct, tested, and ready to run the instant the grasp blocker is resolved — no code change needed to `collector.py`'s control logic beyond re-tuning clearance constants for whatever gripper/task change is chosen.
- **BLOCKING DECISION for the user/architect:** choose among gripper/reach redesign, task-set swap, state-injection demo synthesis, or DATA-01 descope (see Blocker section). Plans 04-03 (teleop), 04-04 (replay), and 04-05 (normalization) all depend on this dataset existing, so they cannot meaningfully proceed until this is resolved.

---
*Phase: 04-dataset-collection*
*Completed (blocked): 2026-08-03*
