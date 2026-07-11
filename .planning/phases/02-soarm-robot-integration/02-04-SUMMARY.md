---
phase: 02-soarm-robot-integration
plan: 04
subsystem: simulation
tags: [mujoco, libero, camera-tuning, bddl, libero-spatial]

requires:
  - phase: 02-soarm-robot-integration
    provides: 02-03 finalized physics (base table lambda (-0.38, 0, 0.90), init_qpos zeros, damping 0.6)
provides:
  - Tuned eye_in_hand camera pose frozen in robot.xml (ENV-07 visual half, local evidence)
  - Final 3 libero_spatial BDDL tasks selected from reach math (D-03 resolved; ENV-06 local evidence)
  - Full local gate GREEN — soarm_sanity.py --check all passes 6/6
affects: [02-05 colab verification, phase-3 inference loop, phase-5 wrist camera SPAT-01]

tech-stack:
  added: []
  patterns:
    - "Camera tuning by render iteration: run --check render, Read the PNGs, adjust pos/quat, repeat"
    - "Task selection by region math: parse BDDL :regions, compute distance from tuned base, require < 0.479 m reach"

key-files:
  created:
    - explorations/outputs/soarm_agentview.png
    - explorations/outputs/soarm_eye_in_hand.png
  modified:
    - LIBERO/libero/libero/assets/robots/soarm101/robot.xml (eye_in_hand camera pose)
    - explorations/soarm_sanity.py (TASKS constant finalized)

key-decisions:
  - "eye_in_hand final pose: pos '0.14 0 0.02', quat '0.664463 -0.241845 0.241845 -0.664463' (roll -90° about z + pitch 40° down), fovy 75 — SO101 approach axis is -z, Panda's starting quat looked away from the jaws"
  - "D-03 resolved: table_center (0.305 m), between_plate_ramekin (0.386 m), on_the_ramekin (0.269 m) — next_to_the_plate swapped out at 0.498 m, beyond the 0.479 m reach"
  - "No BDDL files modified — D-02 last resort not needed; swap within the suite sufficed"

patterns-established: []

requirements-completed: [ENV-06, ENV-07]

coverage:
  - id: D1
    description: "eye_in_hand camera tuned — both rendered frames visually correct (right-side-up, arm visible, meshes intact, jaws in wrist frame)"
    requirement: "ENV-07"
    verification:
      - kind: other
        ref: "conda run -n libero python explorations/soarm_sanity.py --check render (PASS); orchestrator Read both PNGs and confirmed visual criteria"
        status: pass
    human_judgment: true
    rationale: "ENV-07's authoritative visual sign-off is the human Colab review in plan 02-05 (D-10); local judgment is agent-eyes-only evidence"
  - id: D2
    description: "3 finalized libero_spatial tasks each run 50 random-action steps with robots=['Soarm101'] without crashing"
    requirement: "ENV-06"
    verification:
      - kind: integration
        ref: "conda run -n libero python explorations/soarm_sanity.py --check tasks (3/3 per-task PASS)"
        status: pass
    human_judgment: false
  - id: D3
    description: "Complete local gate GREEN in one run (compile, model, reset, render, soak, tasks)"
    verification:
      - kind: e2e
        ref: "conda run -n libero python explorations/soarm_sanity.py --check all (6/6 checks PASSED)"
        status: pass
    human_judgment: false

duration: ~20min execution + orchestrator close-out
completed: 2026-07-11
status: complete
---

# Plan 02-04 Summary

**eye_in_hand camera re-aimed along SO101's -z approach axis (jaws bottom-center, workspace centered) and the 3 final libero_spatial tasks locked in by reach math — full 6/6 local gate GREEN, phase ready for Colab**

## Performance

- **Duration:** ~20 min executor run (terminated by session quota after Task 1 commit + Task 2 edit); orchestrator verified and closed out after quota reset
- **Started:** 2026-07-11T19:30+05:30
- **Completed:** 2026-07-11T23:55+05:30 (close-out)
- **Tasks:** 2
- **Files modified:** 2 (+2 generated PNGs)

## Accomplishments
- Camera tuned by render iteration: Panda's starting quat looked along +z (away from the jaws) — SO101's approach axis is -z. Final pose `pos "0.14 0 0.02"` `quat "0.664463 -0.241845 0.241845 -0.664463"` (roll −90° about z + pitch 40° down), fovy 75 unchanged
- D-03 resolved with data: parsed the `:regions` blocks of all 10 libero_spatial BDDLs and computed each bowl region's distance from the tuned base (−0.38, 0) against SO101's 0.479 m horizontal reach
- Full local validation gate GREEN: `--check all` → 6/6 (compile, model, reset, render, soak, tasks)

## Task Commits

1. **Task 1: tune eye_in_hand camera** - `3d6fe90` (feat, **inner LIBERO repo**)
2. **Task 2: finalize 3 BDDL tasks** - `db10090` (feat, **outer repo**)

## Final task selection (D-03)

| Task | Bowl region | Distance from base | Margin vs 0.479 m |
|------|------------|--------------------|--------------------|
| pick_up_the_black_bowl_from_table_center... | (−0.075, 0.00) | 0.305 m | +0.174 |
| pick_up_the_black_bowl_between_the_plate_and_the_ramekin... | (−0.050, 0.20) | 0.386 m | +0.093 |
| pick_up_the_black_bowl_on_the_ramekin... | (−0.200, 0.20) | 0.269 m | +0.210 |

Swapped out: `next_to_the_plate` — bowl region (0.01, 0.31) is 0.498 m from the base, beyond reach. No BDDL files were modified (D-02 last resort not needed).

## Visual confirmation (per-frame, ENV-07 local evidence)

- **soarm_agentview.png**: right-side-up (wall above, table below), SOARM arm fully visible standing on the tabletop, bowls/plate/objects resting ON the table, STL meshes intact, no z-fighting
- **soarm_eye_in_hand.png**: gripper jaw anchored bottom-center, tabletop workspace centered ahead with a bowl visible left, wall at frame top — camera looks along the gripper approach axis as targeted (Open Question 3 closed)

## Decisions Made
- See key-decisions frontmatter; all values now frozen for the 02-05 notebook and Phase 3/5 reuse

## Deviations from Plan

None in content. Process deviation: the executor session was terminated by a provider quota limit after committing Task 1 and writing (not committing) Task 2. The orchestrator resumed after quota reset: re-ran `--check tasks` (3/3 PASS) and `--check all` (6/6 PASS), visually re-judged both PNGs, committed Task 2, and wrote this SUMMARY.

## Issues Encountered
- Session quota kill mid-plan (see above) — no work lost; recovery via spot-check + inline close-out.

## User Setup Required

None.

## Next Phase Readiness
- Entire local slice of the phase is GREEN; only plan 02-05 remains (Colab verification notebook + human sign-off, D-10)
- The notebook consumes verbatim: the 3 TASKS filenames, the tuned camera state in robot.xml, and env construction with zero controller kwargs (02-03)

---
*Phase: 02-soarm-robot-integration*
*Completed: 2026-07-11*
