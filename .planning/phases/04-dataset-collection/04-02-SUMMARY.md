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
  - "(2026-08-03 resolution attempt) D-07/D-08 executed: retargeted collector.py to libero_goal/put_the_cream_cheese_in_the_bowl.bddl (cream_cheese_1 pick, akita_black_bowl_1 place) — this DID resolve the original jaw-width blocker (cream_cheese's ~4.3cm grasp face fits the 84mm jaw), but empirical measurement found a SECOND, previously-undocumented embodiment limitation: the arm's vertical reach at this task's radial distance (~0.35m, well inside the 0.479m max reach) bottoms out ~1-3cm ABOVE the object's actual top surface, independent of GRASP_Z_OFFSET tuning, KP_POS, or controller type (OSC_POSE vs OSC_POSITION both tested). A formal 20-attempt validation batch reached 0/20 successes. DATA-01's 100+ demo dataset is STILL not produced. See the new 'Resolution attempt' section below for the full measurement trail and the Rule-4 decision this now requires."

requirements-completed: []
requirements-blocked: [DATA-01]

# Metrics
duration: 70min (original) + ~65min (2026-08-03 retargeting + validation attempt)
completed: 2026-08-03
status: blocked
---

# Phase 4 Plan 02: Scripted Waypoint Collector Summary

**A pure, unit-tested 8-phase pick-place waypoint FSM plus a full scripted collection driver/CLI were built on 04-01's recording infrastructure — but the phase's primary dataset artifact could NOT be produced: empirical testing conclusively shows the SOARM gripper cannot grasp or lift the akita_black_bowl, so no scripted trajectory reaches `On(bowl, plate)` and the 100+ successful demos are unattainable with the current robot.**

## Resolution attempt (2026-08-03) — D-07/D-08 executed, a SECOND blocker found

Following the locked D-07 (84mm faithful gripper, committed `ad0b0d0`) and D-08
(task retargeting) decisions, this session retargeted `collector.py` to the
sub-84mm in-reach task and attempted the real collection run. **The jaw-width
problem is resolved. A second, previously-undocumented embodiment limitation
(arm vertical-reach depth) was found instead, and DATA-01's dataset is still
not producible with the current robot + FSM approach.**

### What was changed (committed `5675163`, BEFORE any run attempt per protocol)

- `BDDL_DIR` -> `libero_goal` (was `libero_spatial`); `TASKS` -> single-element
  `["put_the_cream_cheese_in_the_bowl.bddl"]` (verbatim, no BDDL edits, per the
  locked object/task choice: pick `cream_cheese_1`, place into
  `akita_black_bowl_1`, goal `On(cream_cheese_1, akita_black_bowl_1)`).
- New `TASK_BODY_MAP` dict wires per-task pick/place body names through
  `collect_task` -> `run_scripted_episode` (previously hardcoded to the old
  `akita_black_bowl_1`/`plate_1` names; this would have crashed on the new BDDL
  looking up a nonexistent `plate_1` body).
  `target_per_task`/`max_attempts` raised 40->120 / 200->300 (single task must
  now absorb the full 100+ demo buffer previously split across 3 tasks).
- Module/function docstrings updated with empirically measured facts (below).
- `test_collector.py`: replaced the stale 3-task-list + `next_to_the_plate`
  assertions with retargeted single-task + `TASK_BODY_MAP` coverage; the
  target-behavior integration test's `xfail` reason updated to the new blocker
  (its `>=2` assertion is preserved, not weakened, matching the original
  plan's honesty convention).
- Full suite after the retarget: **11 passed, 1 xfailed** (the honest,
  documented target-behavior gap).

### Measured geometry (new task, faithful 84mm gripper, headless SOARM env)

- `cream_cheese_1` rests FLAT on the table without tipping (the asset is
  already authored lying on its largest face): body origin settles at
  **z ≈ 0.909** (table top z=0.90, confirmed unchanged), i.e. object top
  surface **z ≈ 0.918** (half-thickness ≈ 0.009 m; its ~8.1×4.3 cm faces lie
  horizontal — the graspable Y-width, ~4.3 cm, is comfortably within the
  84 mm jaw's [1.84 cm, 10.06 cm] stroke range).
- `akita_black_bowl_1` (place target, used passively) rests at body origin
  **z ≈ 0.898**, unchanged from the original bowl-task geometry (same object).
- Both regions confirmed reachable per the locked decision: `cream_cheese_1`
  planar distance from the robot base ≈ **0.356 m**; `akita_black_bowl_1`
  planar distance ≈ **0.292 m** — both comfortably under the 0.479 m max reach.
- **The new finding:** driving the eef straight down over `cream_cheese_1`'s
  xy (aggressive proportional control, 400 steps, well past convergence)
  asymptotically bottoms out at **eef z ≈ 0.945–0.952** — never lower,
  regardless of how far below the target is set. Direct geom introspection
  (`gripper0_left_jaw_collision` / `gripper0_right_jaw_collision`) shows the
  jaw pads' lowest reachable point is **≈ 0.923–0.930** at this floor — still
  **≈ 1–1.5 cm ABOVE** the cream_cheese top surface (0.918). None of the arm's
  5 joints were at their `jnt_range` limits at this pose (checked directly),
  ruling out a simple joint-limit explanation; the STS3215 torque actuators
  are hard-capped at ±2.94 N·m (faithful to real hardware), consistent with a
  gravity/torque-saturation floor rather than a position-tolerance artifact.

### What was tried (per the plan's empirical-tuning + 5-iteration cap)

1. Three `GRASP_Z_OFFSET` candidates (0.015 / 0.03 / 0.045 m above object
   origin) via a focused grasp+lift probe — all converged to the same actual
   descended height (~0.945–0.97), confirming the floor is insensitive to the
   target offset (already saturated well before reaching it). 0/3 lifted the
   object.
2. A radial-distance + bearing sweep (0.29 m "near-base" point, and 0.30/0.40/
   0.44/0.47 m along the cream_cheese bearing, plus the bowl's own xy at
   0.29 m) — floor varies mildly (≈0.93–0.99) but never drops meaningfully
   below the object surfaces tested; closer-to-base points are consistently
   *worse* (higher floor), counter to a naive "shorter reach = easier" intuition.
3. Swapped `OSC_POSE` for `OSC_POSITION` (diagnostic only, not committed) to
   test whether orientation-hold was consuming torque budget needed for
   descent — the floor got *worse* (≈0.988), ruling this out as the cause.
4. A formal validation batch via the real `collect_task` path
   (`target_successes=8, max_attempts=20`, ~98 s): **0/20 successes.**

Per the plan's explicit cap ("if you cannot get a reasonably reliable grasp
after ~5 focused iterations, STOP and report... rather than continuing
indefinitely" — this exact investigation has hit the monthly spend limit
twice before), **tuning stopped here.** The full 120-demo collection run was
deliberately NOT attempted: the 0/20 result across a structurally-consistent
floor (not an attempt-count-sensitive probabilistic failure) makes a 300-
attempt run a near-certain 0/300 outcome, burning ~24 min of compute for no
new information.

### Conclusion — Rule 4 (architectural decision required, not auto-fixable)

D-07 (gripper widening) and D-08 (task retargeting for jaw-width + horizontal
reach) are both correctly executed and did fix what they targeted. A THIRD,
previously-unaddressed constraint blocks DATA-01: **this arm's vertical reach
cannot bring the gripper down to table-resting-object height (~0.90–0.92 m
world z) at any of the reachable radii tested (0.29–0.47 m) while holding a
fixed top-down orientation** — a torque/kinematic limit of the small,
faithful (~500 g payload, ±2.94 N·m per-joint) SO-ARM101, not a jaw-width or
horizontal-distance problem. This affects table-resting objects generally,
not just `cream_cheese_1`.

**Options for the user (none auto-selected):**
- (a) Mount the arm on a small riser/pedestal to reduce the required vertical
  excursion — reopens Phase 2's D-04 "no pedestal hardware" decision.
- (b) Target a task where the pick object sits ELEVATED above bare table
  level (e.g. on top of another prop/fixture) rather than resting flush on
  the table, if such a LIBERO task/region exists within reach.
- (c) State-injection demo synthesis — directly set the object's pose to the
  goal state and record the resulting observations, bypassing a physical
  scripted grasp (legitimate per the original blocker's option (c), but a
  documented departure from D-01's "waypoint trajectory" framing).
- (d) Descope DATA-01's "successful task completion via physical grasp"
  requirement for the MVP, or accept teleoperation (04-03/D-02) as the sole
  demo source.

**Current repository state:** `collector.py`/`test_collector.py` are
committed with the correct retargeted task, body-name wiring, and honestly
xfail'd target-behavior test (commit `5675163`). No `soarm_spatial/*.hdf5`
dataset exists yet — DATA-01 remains blocked pending a decision above.

---

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

## Update (2026-08-03) — D-07/D-08 executed; SECOND blocker found, still no DATA-01 dataset

The jaw-width blocker above is RESOLVED (84mm gripper + cream_cheese_1/
akita_black_bowl_1 retargeting, commit `5675163`). A vertical-reach-depth
blocker was found in its place — see the "Resolution attempt (2026-08-03)"
section near the top of this file for the full measurement trail, what was
tried (3 GRASP_Z_OFFSET candidates, a radius/bearing sweep, an OSC_POSE vs
OSC_POSITION controller comparison, and a formal 20-attempt validation batch:
0/20 successes). The full 120-demo collection run was deliberately not
attempted given that structurally-consistent 0/20 result. **DATA-01 remains
blocked; a fresh Rule-4 decision is needed from the user** (pedestal/riser,
an elevated-object task, state-injection synthesis, or descope — see that
section's "Options for the user"). Self-Check for this update: `collector.py`/
`test_collector.py` changes FOUND on disk and in `git log` (commit `5675163`);
full test suite re-run: 11 passed, 1 xfailed.

---
*Phase: 04-dataset-collection*
*Completed (blocked): 2026-08-03*
*Updated (still blocked — second finding): 2026-08-03*
