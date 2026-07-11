---
phase: 02-soarm-robot-integration
plan: 03
subsystem: simulation
tags: [mujoco, robosuite, libero, physics, osc-pose, so-arm100]

requires:
  - phase: 02-soarm-robot-integration
    plan: 02
    provides: MountedSoarm101/SoarmGripper classes, gripper MJCF, robosuite mapping registration, GREEN reset baseline (0.024 N)
provides:
  - Verified-stable SOARM physics baseline — `--check reset` (0.024 N peak, SC-1) and `--check soak` (500 steps x 3 seeds) both GREEN
  - Finalized tuning values for the 02-05 Colab notebook and Phase 3 (all 02-02 starting values confirmed final, zero changes)
  - Confirmation that the generic OSC_POSE controller config suffices — no custom controller_configs kwarg needed anywhere downstream
affects: [02-04 camera/task selection, 02-05 colab verification, phase-3 inference loop]

tech-stack:
  added: []
  patterns:
    - Verify-before-tune — re-run the failing gate before spending the iteration budget; wave-2 baseline already satisfied both gates

key-files:
  created: []
  modified: []

key-decisions:
  - "Generic OSC_POSE config (kp=150) suffices for SOARM soak stability — NO custom controller_configs dict; 02-05 notebook and Phase 3 build the env with no controller kwargs"
  - "All 02-02 starting values are final: damping 0.6 x 5 joints, base_xpos_offset table lambda (-0.38, 0, 0.90), init_qpos zeros(5), gripper speed 0.10 / jaw init 0.8"
  - "RESEARCH A4 primitive-collision fallback NOT needed — convex-hull STL collisions are stable through reset and 1500 random-action steps"
  - "ENV-07 left pending: this plan verifies its physics-stability clause; the visually-correct/camera-angle clause lands in 02-04"

patterns-established: []

requirements-completed: [ENV-05]

coverage:
  - id: D1
    description: "OffScreenRenderEnv(robots=['Soarm101']) resets on the default libero_spatial BDDL with peak contact force < 10 N after 10 settle steps (SC-1)"
    requirement: "ENV-05, ENV-07 (physics-stability clause)"
    verification:
      - kind: other
        ref: "soarm_sanity.py --check reset exits 0; printed peak 0.024 N; re-confirmed after soak run"
        status: pass
    human_judgment: false
  - id: D2
    description: "500 random-action steps x 3 seeds complete with no exception and finite qpos throughout"
    requirement: "ENV-07 (physics-stability clause)"
    verification:
      - kind: other
        ref: "soarm_sanity.py --check soak exits 0 (seeds 0/1/2 each 500 steps OK)"
        status: pass
    human_judgment: false
  - id: D3
    description: "No regression of prior greens; structural XML contracts intact"
    requirement: "ENV-04, ENV-05"
    verification:
      - kind: other
        ref: "--check compile and --check model exit 0; CONTRACTS HOLD assertion (zero default blocks, 5 motors, 1 position actuator); table lambda z == 0.90"
        status: pass
    human_judgment: false

duration: ~10min
completed: 2026-07-11
status: complete
---

# Plan 02-03 Summary

**Physics stabilization complete with ZERO tuning iterations: the 02-02 baseline values already pass both gates — reset at 0.024 N peak contact force (SC-1 threshold 10 N) and a 1500-step random-action soak across 3 seeds — and the generic OSC_POSE config needs no custom controller_configs dict**

## Performance

- **Duration:** ~10 min
- **Started:** 2026-07-11T13:49:09Z
- **Completed:** 2026-07-11T13:58:00Z
- **Tasks:** 2
- **Files modified:** 0 (verification-only — the 1-2 day iteration budget flagged in STATE.md was not needed)

## Accomplishments

- **Task 1 (`--check reset`):** GREEN on first run with the untuned wave-2 values — env creation end-to-end works (merge, namespacing, `robot0_eye_in_hand` camera resolution, mesh paths), peak contact force **0.024 N** after 10 zero-action settle steps. No wiring bugs, no interpenetration, no placement/pose tuning required.
- **Task 2 (`--check soak`):** GREEN on first run — seeds 0, 1, 2 each survive 500 uniform random OSC_POSE actions with finite qpos throughout and no qacc warnings. The generic OSC_POSE config (kp=150, tuned for Panda) is stable on the 0.632 kg SO101 at its ±2.94 Nm torque limits.
- Full no-regression sweep: `--check compile`, `--check model`, structural-contract assertion (`CONTRACTS HOLD`), and a post-soak reset re-run all pass.

## Final tuned values (consumed by 02-04 task selection, 02-05 notebook, Phase 3)

All values are the 02-02 starting values, now verified final:

| Knob | Final value | Where |
|------|-------------|-------|
| `base_xpos_offset` table lambda | `(-0.38, 0, 0.90)` — base ON tabletop, z=0.90 preserved | `LIBERO/libero/libero/envs/robots/soarm.py` |
| `init_qpos` (arm) | `np.zeros(5)` — new-calib home pose, no tuck needed (D-07 unchanged) | `soarm.py` |
| Joint damping | `0.6` per joint x 5 (STS3215 hardware value) | `soarm.py` `set_joint_attribute` |
| `SoarmGripper.speed` | `0.10` | `LIBERO/libero/libero/envs/grippers/soarm_gripper.py` |
| Gripper `init_qpos` (jaw) | `[0.8]` | `soarm_gripper.py` |
| `controller_configs` env kwarg | **NOT NEEDED** — generic OSC_POSE config sufficed | (no kwarg; build env exactly as `soarm_sanity._build_env` does) |
| A4 primitive-collision fallback | **NOT NEEDED** — convex-hull STL collision geoms are stable | (robot.xml / soarm_gripper.xml unchanged) |
| Measured reset peak force | **0.024 N** (SC-1 threshold 10 N; margin ~400x) | `--check reset` output |

**Pitfall 8 note (structural, not a bug):** the 5-DOF arm under the 6-DOF OSC_POSE task sacrifices one rotational direction — expected. Phase 3's 7-D action contract holds (control_dim 6 + gripper dof 1).

## Task Commits

No code commits — neither task required a file change. The wave-2 baseline (inner LIBERO repo commits `761fef3`, `8ebc8fb`, `8c7fa6f`; outer `d5ca329`) already satisfied both gates. This plan's only commit is the outer-repo docs commit carrying this SUMMARY and planning-state updates.

1. **Task 1: stable reset** - no commit (verification only; acceptance criteria all pass unchanged)
2. **Task 2: random-action soak** - no commit (verification only; acceptance criteria all pass unchanged)

## Deviations from Plan

None - plan executed exactly as written. The plan's expected failure classes (wiring/merge errors, contact-force spikes, collision instability, OSC oscillation) simply did not occur; per the D-11 stop condition ("tune only until criteria pass") the correct amount of tuning was zero.

## Requirements note

- **ENV-05**: already complete (02-02); this plan adds the instantiation-and-reset evidence on the default libero_spatial task.
- **ENV-07**: left **pending** deliberately — this plan verifies only its physics-stability clause. The "visually correct (right-side-up, correct camera angle)" clause is plan 02-04's scope; marking it complete now would be premature.

## Verification state at plan end

- GREEN: `compile`, `model`, `reset` (0.024 N), `soak`, plus `render` (early, from 02-02)
- Expected RED remaining: `tasks` (TASKS list unfinalized — 02-04 scope)

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Phase success criterion 1 (reset < 10 N) satisfied locally; Colab re-verification in 02-05
- 02-04 starts from a fully stable physics baseline: camera pose tuning and BDDL task finalization are the only remaining local work
- The 02-05 notebook can build the SOARM env with zero custom kwargs — copy `soarm_sanity._build_env` verbatim

---
*Phase: 02-soarm-robot-integration*
*Completed: 2026-07-11*

## Self-Check: PASSED

SUMMARY claims verified: no files created/modified (git status clean in both repos apart from pre-existing drift), no new commits claimed, all five referenced check commands exited 0 in this session (reset 0.024 N, soak 3x500, compile, model, CONTRACTS HOLD).
