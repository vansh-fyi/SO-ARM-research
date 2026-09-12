---
phase: 02-soarm-robot-integration
verified: 2026-07-18T00:00:00Z
status: passed
score: 4/4 must-haves verified
behavior_unverified: 0
overrides_applied: 0
---

# Phase 2: SOARM Robot Integration — Verification Report

**Phase Goal (ROADMAP):** The SOARM SO101 robot is registered in LIBERO as a validated ManipulatorModel with stable physics, correct rendering, and at least 3 BDDL tasks configured to use it.
**Phase Mode:** mvp — plan-level User Story (validated: `valid=true`): "As a VLA researcher, I want to select the SOARM SO101 robot by name in LIBERO and run libero_spatial tasks with it — with stable physics and correct rendering, so that the Phase 3 inference loop can drive SOARM from language prompts instead of the stock Panda."
**Verified:** 2026-07-18
**Status:** passed
**Re-verification:** No — initial verification

> Verification note: this phase spans TWO git repos (outer + embedded `LIBERO/` repo, gitignored in the outer). All 12 claimed commits were confirmed in the correct repo. All behavioral evidence below was produced by the verifier re-running the local gate in its own process (not taken from SUMMARY claims). The Colab human sign-off (2026-07-18, T4, all cells green, frames approved) is recorded in 02-05-SUMMARY.md and treated as completed human verification per D-10.

## User Flow Coverage (MVP mode)

User story: «As a VLA researcher, I want to select the SOARM SO101 robot by name in LIBERO and run libero_spatial tasks with it — with stable physics and correct rendering, so that the Phase 3 inference loop can drive SOARM from language prompts instead of the stock Panda.»

| Step | Expected | Evidence | Status |
|------|----------|----------|--------|
| Select SOARM by name | `robots=["Soarm101"]` resolves MountedSoarm101 exactly as Panda | `LIBERO/libero/libero/envs/robots/__init__.py:5,17,21` + `envs/__init__.py:4` (`from .robots import *` — registration fires on the standard env import path); verifier ran `--check model` → PASS | ✓ |
| Run a libero_spatial task | `OffScreenRenderEnv(bddl, robots=["Soarm101"])` resets without error | Verifier ran `--check reset` standalone → PASS, peak contact force 0.059 N | ✓ |
| Stable physics | Contact force < 10 N at reset; sustained stepping survives | Reset 0.024–0.059 N (threshold 10 N); `--check soak` → 3 seeds × 500 random actions, finite qpos | ✓ |
| Correct rendering | Right-side-up frames, arm visible, jaws in wrist cam | Verifier regenerated + visually inspected both PNGs; human Colab sign-off 2026-07-18 (02-05-SUMMARY D2) | ✓ |
| Outcome: Phase 3 can drive SOARM from prompts | 7-D action contract, zero custom controller kwargs, 3 frozen tasks | Gripper `dof=1` + OSC 6 = 7-D (soarm_gripper.py:58); 02-03 confirmed no controller_configs kwarg; TASKS frozen in soarm_sanity.py:65-69 and mirrored verbatim in the Colab notebook | ✓ |

## Goal Achievement

### Observable Truths (ROADMAP Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `env.reset()` on a SOARM LIBERO environment completes without errors; contact forces at init < 10 N | ✓ VERIFIED | Verifier ran `--check reset` (standalone, fresh process): PASS, peak 0.059 N; within `--check all` run: 0.024 N. Behavioral, not presence-based. |
| 2 | Rendered frames visually correct: right-side-up, correct camera angle, no mesh artifacts | ✓ VERIFIED | Verifier regenerated frames via `--check render` and inspected both PNGs: agentview right-side-up (wall above, table below, arm standing on tabletop, meshes intact); eye_in_hand shows jaw anchored bottom-center, workspace ahead. Authoritative human sign-off on Colab T4 completed 2026-07-18 (02-05-SUMMARY). |
| 3 | SOARM selectable by name in ROBOT_CLASS_MAPPING exactly as Panda | ✓ VERIFIED | `robots/__init__.py` registers `"MountedSoarm101": SingleArm` alongside both Panda entries + `GRIPPER_MAPPING["SoarmGripper"]`; registration fires via `envs/__init__.py:4` on the standard import path (falsification check: `--check reset` passes WITHOUT `check_model` running first in the process). `--check model` → PASS. |
| 4 | At least 3 BDDL tasks run end-to-end with SOARM without crashing | ✓ VERIFIED | TASKS constant has exactly 3 libero_spatial BDDLs, all 3 files exist on disk; verifier ran `--check tasks`: 3/3 per-task PASS (50 random-action steps each). |

**Score:** 4/4 truths verified (0 present-but-behavior-unverified)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `explorations/soarm_sanity.py` | 7-mode validation harness, `max_contact_force` | ✓ VERIFIED | 305 lines; all 7 `check_*` functions + `max_contact_force` present; `MUJOCO_GL=glfw` set (line 43) before any mujoco/robosuite/libero import; ran green 6/6 |
| `LIBERO/.../robots/soarm101/robot.xml` | Adapted arm MJCF: 5 motors ±2.94, `right_hand`, `eye_in_hand`, no defaults | ✓ VERIFIED | 120 lines; 0 default blocks; 5 motors all ctrlrange "-2.94 2.94"; `right_hand` body with tuned `eye_in_hand` camera (pos "0.14 0 0.02") + `robotview`; all meshes named with local `assets/` paths that resolve on disk; no kv attrs; compiles (compile check PASS) |
| `LIBERO/.../soarm101/so101_new_calib.source.xml` | Pristine upstream XML | ✓ VERIFIED | 165 lines, committed (2a700d8, inner repo) |
| `LIBERO/.../soarm101/VENDOR.txt` | Pinned SHA + per-file sha256 | ✓ VERIFIED | Upstream URL, 40-hex SHA `fda892cb...`, Apache-2.0 note, 14 sha256 lines |
| `LIBERO/.../soarm101/assets/*.stl` | 13 vendored meshes | ✓ VERIFIED | 13 STLs on disk; every mesh referenced by robot.xml and gripper XML resolves |
| `LIBERO/.../grippers/soarm_gripper.xml` | robosuite-contract gripper MJCF | ✓ VERIFIED | `eef` body; all 7 contract sites; `force_ee`/`torque_ee` sensors; exactly 1 position actuator; `fixed_jaw_collision`/`moving_jaw_collision`; 0 defaults; no kv; compiles |
| `LIBERO/.../envs/robots/soarm.py` | `MountedSoarm101(ManipulatorModel)`, 8-property contract | ✓ VERIFIED | Substantive (83 lines); fork-local ASSETS path; damping ×5; `default_mount=None`, `init_qpos=zeros(5)`, table lambda z=0.90; instantiates (model check PASS) |
| `LIBERO/.../envs/grippers/soarm_gripper.py` | `SoarmGripper(GripperModel)` 1-DOF | ✓ VERIFIED | Substantive; `dof=1`, `speed=0.10`, `init_qpos=[0.8]`, `_important_geoms` → jaw collision geoms |
| `LIBERO/.../envs/grippers/__init__.py` | New package | ✓ VERIFIED | Exists; package imports (model check PASS) |
| `LIBERO/.../envs/robots/__init__.py` | Mapping registration | ✓ VERIFIED | Both SOARM entries added, both Panda entries preserved; zero robosuite site-packages edits |
| `LIBERO/notebooks/02-soarm-integration-check.ipynb` | Colab PASS/FAIL notebook | ✓ VERIFIED | 26 cells, valid JSON; markers ENV-04/05/06/07 + SC-1, MUJOCO_GL/egl, max_contact_force, `[::-1]` flip, `robots=["Soarm101"]`; ENV-06 cell's 3 BDDLs exactly match the harness TASKS constant; no token/secret literals; human-run green on Colab T4 |
| `explorations/outputs/soarm_{agentview,eye_in_hand}.png` | Visual evidence | ✓ VERIFIED | Regenerated by verifier this session; both non-black and visually correct |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| soarm_sanity.py | robot.xml + soarm_gripper.xml | ARM_XML/GRIPPER_XML repo-relative paths | ✓ WIRED | Lines 49, 52; compile check loads both |
| robot.xml | soarm101/assets/*.stl | mesh `file="assets/..."` | ✓ WIRED | All mesh files resolve on disk (0 missing) |
| soarm_gripper.xml | soarm101/assets/*.stl | `../robots/soarm101/assets/` | ✓ WIRED | All resolve (0 missing) |
| soarm.py | robot.xml | fork-local ASSETS join | ✓ WIRED | soarm.py:33; instantiation proves resolution |
| soarm_gripper.py | soarm_gripper.xml | fork-local ASSETS join | ✓ WIRED | soarm_gripper.py:28 |
| robots/__init__.py | soarm.py / grippers/soarm_gripper.py | imports + mapping assignments | ✓ WIRED | Lines 5, 11, 17, 21; reachable from `libero.libero.envs` (`envs/__init__.py:4`) — not orphaned |
| soarm_sanity.py check_reset | MountedSoarm101 | `robots=["Soarm101"]` (Mounted-prefix resolution) | ✓ WIRED | Line 87; reset passes standalone |
| Notebook | TASKS / soarm.py | same 3 BDDL names, `robots=["Soarm101"]` | ✓ WIRED | TASKS parity confirmed programmatically |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Full local gate | `conda run -n libero python explorations/soarm_sanity.py --check all` | 6/6 PASS (compile, model, reset 0.024 N, render, soak 3×500, tasks 3/3), exit 0 | ✓ PASS |
| Reset without model-check priming (registration on standard import path) | `... --check reset` (standalone) | PASS, peak 0.059 N, exit 0 | ✓ PASS |
| Frames visually correct | Read regenerated PNGs | agentview right-side-up/intact; eye_in_hand jaws bottom-center | ✓ PASS |

### Probe Execution

No `scripts/*/tests/probe-*.sh` probes exist in this repo; the phase's declared runnable gate is `soarm_sanity.py --check all`, executed above (exit 0).

### Requirements Coverage

| Requirement | Source Plan(s) | Description | Status | Evidence |
|-------------|----------------|-------------|--------|----------|
| ENV-04 | 02-01, 02-02, 02-05 | SO101 MJCF adapted for robosuite 1.4, registered as ManipulatorModel subclass | ✓ SATISFIED | Both MJCFs compile under MuJoCo 2.3.7; `MountedSoarm101(ManipulatorModel)` registered; Colab ENV-04 cell green |
| ENV-05 | 02-02, 02-03, 02-05 | SOARM in ROBOT_CLASS_MAPPING, instantiable in a LIBERO env | ✓ SATISFIED | Mapping entries verified in code; env reset behavioral PASS |
| ENV-06 | 02-04, 02-05 | ≥3 BDDL tasks configured to use SOARM | ✓ SATISFIED | 3 libero_spatial tasks frozen (D-03 by reach math), 3/3 crash-free locally and on Colab |
| ENV-07 | 02-03, 02-04, 02-05 | Frames visually correct, physics stable | ✓ SATISFIED | 0.024–0.059 N reset force; soak green; frames verifier-inspected + human Colab sign-off |

No orphaned requirements: REQUIREMENTS.md maps exactly ENV-04..07 to Phase 2 and all four appear in plan frontmatter.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| .planning/REQUIREMENTS.md | 17-18, 91-92 | Stale bookkeeping: ENV-06/ENV-07 still unchecked / "Pending" although completed and verified | ℹ️ Info | Documentation only — update checkboxes + traceability table to Complete |

Zero TBD/FIXME/XXX/TODO/placeholder markers in any phase-modified code file. Zero empty-implementation or hardcoded-stub patterns.

### Human Verification Required

None outstanding. The phase's blocking human checkpoint (02-05 Task 2) was completed: the user ran `02-soarm-integration-check.ipynb` on Colab T4 on 2026-07-18 and confirmed all cells green and frames visually correct (ENV-07 visual clause + D-10 on-target proof). No `<human-check>` blocks were deferred from auto tasks in any plan.

### Gaps Summary

No gaps. All 4 ROADMAP success criteria are behaviorally verified in the verifier's own process (not from SUMMARY claims), all 12 phase commits exist across the dual-repo split, all artifacts are substantive and wired, and the human Colab sign-off is on record.

Minor observations (non-blocking):
1. REQUIREMENTS.md ENV-06/ENV-07 statuses are stale (still "Pending") — should be flipped to Complete during phase closure.
2. ROADMAP.md Phase 2 has `Mode: mvp` but its Goal field is declarative rather than user-story format; the plans carry a valid user story (validated `valid=true`), which was used for User Flow Coverage. Consider syncing the ROADMAP goal field for future MVP phases.
3. 02-02-SUMMARY notes the inner LIBERO repo working tree carries pre-existing uncommitted `LIBERO.`-prefix import drift; the phase worked around it (sys.path fixes d5ca329 / 4883af2) without owning it. Pre-existing, out of phase scope.

---

_Verified: 2026-07-18_
_Verifier: Claude (gsd-verifier)_
