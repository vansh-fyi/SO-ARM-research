---
phase: 10-digital-twin-fidelity
verified: 2026-09-19T11:16:39Z
status: gaps_found
score: 7/8 must-haves verified
behavior_unverified: 0
overrides_applied: 0
re_verification:
  previous_status: gaps_found
  previous_score: 6/7
  gaps_closed:
    - "Driving the simulated gripper with a given joint command opens/closes it in the same direction as the real gripper under the same command (TWIN-07) — gap-closure quick task 260919-h8v flipped gripper_left/gripper_right MJCF axis+range+ctrlrange polarity; a fresh human-in-the-loop hardware re-test (documented in 10-05-TWIN-07-VERIFICATION.md's 'Hardware Re-Test: PASS' section) confirms real and sim now agree; independently reproduced in a standalone MuJoCo simulation during this re-verification (see Behavioral Spot-Checks)"
  gaps_remaining: []
  regressions:
    - "NEW regression (not present in the original verification's full-suite run): LIBERO/libero/libero/datasets/test_collector.py::test_run_scripted_episode_reaches_success_within_budget now FAILS (0 successes / 50 attempts), caused directly by the TWIN-07 axis-polarity fix. collector.py's OPEN_CMD=-1.0/CLOSE_CMD=1.0 constants (used by the scripted demo-collection FSM's grasp phase) were not updated to match the flipped gripper polarity, so the FSM's CLOSE_CMD action now opens the jaws instead of closing them on the target object. This was disclosed by the gap-closure SUMMARY only as a forward-looking 'risk' in STATE.md, not as a currently-failing, previously-passing automated test. The phase's own validation contract (10-VALIDATION.md: 'Before /gsd-verify-work: Full suite must be green') is therefore currently violated."
gaps:
  - truth: "The full required automated test suite (`pytest LIBERO/libero/libero/envs/ LIBERO/libero/libero/datasets/ -q`) remains green — no new regressions — after the TWIN-07 gap-closure fix, per this phase's own validation contract (10-VALIDATION.md: 'Before /gsd-verify-work: Full suite must be green ... before TWIN-07/Phase 10 can close')"
    status: failed
    reason: "Independently re-ran the full suite during this re-verification: 51 passed, 2 failed, 2 skipped (previously 52 passed, 1 failed, 2 skipped at the original Phase 10 verification). The new failure is LIBERO/libero/libero/datasets/test_collector.py::test_run_scripted_episode_reaches_success_within_budget (0 successes / 50 attempts, asserts >=2). Root cause confirmed by reading collector.py: gap-closure quick task 260919-h8v flipped soarm_gripper.xml's gripper_left/gripper_right axis+range+ctrlrange polarity (correctly, to fix TWIN-07) but did not update the paired OPEN_CMD/CLOSE_CMD constants in LIBERO/libero/libero/datasets/collector.py, whose scripted grasp FSM now sends the wrong-direction command during the 'grasp' phase, so it never actually closes the jaws on the target object. This was already anticipated and disclosed in the gap-closure SUMMARY and a STATE.md 'Blocker/Concern' entry as a *forward-looking risk for future Phase 8/9 resumption* — but it is not merely a future risk: it is a currently-failing, previously-passing test in the live suite today. The pre-existing test_replay.py failure (documented since Phase 5, unrelated to gripper polarity) is unaffected and remains the only carried-over failure."
    artifacts:
      - path: "LIBERO/libero/libero/datasets/collector.py"
        issue: "Lines 81-82: OPEN_CMD = -1.0 / CLOSE_CMD = 1.0 are stale — they drove the correct real-world-matching direction under the PRE-fix gripper polarity, but now drive the physical opposite of their names post-fix (CLOSE_CMD now opens, OPEN_CMD now closes), breaking the scripted grasp FSM"
    missing:
      - "Swap OPEN_CMD/CLOSE_CMD values (or their FSM usage) in LIBERO/libero/libero/datasets/collector.py to match the corrected gripper polarity"
      - "Re-run pytest LIBERO/libero/libero/datasets/test_collector.py::test_run_scripted_episode_reaches_success_within_budget and confirm it passes (>=2 successes within the 50-attempt budget)"
      - "Re-run the full required suite (pytest LIBERO/libero/libero/envs/ LIBERO/libero/libero/datasets/ -q) and confirm it returns to the prior baseline (1 known pre-existing failure: test_replay.py; zero new failures)"
      - "Alternatively, if this regression is to be explicitly accepted rather than fixed (e.g. because Phase 8/9 demo-recollection is paused and out of Phase 10's stated scope), record that decision as an accepted override in this file's frontmatter with a human accepted_by/accepted_at, rather than leaving it silently unresolved"
---

# Phase 10: Digital-Twin Fidelity Verification Report

**Phase Goal:** The URDF and the MuJoCo XML (`LIBERO/libero/libero/assets/robots/soarm101/robot.xml`) form a correct, complete, 1:1 kinematic match to the real SO-ARM101 — a proper parent/child chain ending at the gripper, `wrist_roll` and gripper jaw joints restored with correct axes/direction, correct joint limits, and in-repo mesh paths. This is sim-side-only work; it does not touch the real robot or the `control/` bridge.

**Verified:** 2026-09-19T11:16:39Z
**Status:** gaps_found
**Re-verification:** Yes — after gap closure (quick task `260919-h8v`), second verification pass

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | URDF's gripper is attached at the end of the kinematic chain (child of wrist), not floating off `robot_base` (TWIN-01) | ✓ VERIFIED (regression check) | Re-ran `pytest scripts/test_verify_urdf.py::test_gripper_is_child_of_wrist` live: PASSED. No changes to this structural claim since the original verification. |
| 2 | URDF includes a `wrist_roll` joint with range matching the real robot's calibrated servo limits (TWIN-02) | ✓ VERIFIED (regression check) | Re-ran `pytest scripts/test_verify_urdf.py::test_wrist_roll_range` live: PASSED. Unaffected by the gripper-polarity fix. |
| 3 | URDF includes `gripper_left`/`gripper_right` prismatic joints whose open/close direction matches the real gripper (TWIN-03) | ✓ VERIFIED | Directly read `So-101/So-101.urdf`: `gripper_left` axis `xyz="0 1 0"` limit `-0.042..0`; `gripper_right` axis `xyz="0 -1 0"` limit `-0.042..0` — matches the corrected `soarm_gripper.xml` MJCF source exactly (independently diffed, not just trusting the SUMMARY). `pytest scripts/test_verify_urdf.py::test_gripper_joint_axes` re-run live: PASSED with the new expected literals `[0,1,0]`/`[0,-1,0]`. |
| 4 | All mesh file references in the URDF use in-repo relative paths (TWIN-04) | ✓ VERIFIED (regression check) | Re-ran `pytest scripts/test_verify_urdf.py::test_no_absolute_mesh_paths` live: PASSED. |
| 5 | URDF joint limits for the 4 calibration-derivable arm joints match the real servo calibration ranges (TWIN-05) | ✓ VERIFIED (regression check) | Re-ran `pytest scripts/test_verify_urdf.py::test_joint_limits_match_calibration` live: PASSED. Unaffected by the gripper-polarity fix. |
| 6 | The MuJoCo XML is brought into agreement with the corrected URDF's kinematic structure (TWIN-06) | ✓ VERIFIED (regression check) | Re-ran `pytest LIBERO/libero/libero/envs/test_camera_config.py` live (constructs a real `Soarm101` env, calls `env.reset()`): 2 passed. |
| 7 | Driving the simulated gripper with a given joint command opens/closes it in the same direction as the real gripper under the same command (TWIN-07) | ✓ VERIFIED | **Independently reproduced, not just read.** (a) Read the current `soarm_gripper.xml`: `gripper_left` axis `0 1 0`/range `-0.042 0`, `gripper_right` axis `0 -1 0`/range `-0.042 0`, `ctrlrange -0.042 0` on both actuators — matches the SUMMARY's claimed edit exactly. (b) Read `soarm_gripper.py`: `init_qpos` returns `[-0.042, -0.042]`; `format_action` arithmetic unchanged. (c) Wrote and ran a standalone MuJoCo simulation (not part of the existing test suite) that loads `soarm_gripper.xml` directly, drives `SoarmGripper.format_action` with external action `+1.0` and `-1.0` through robosuite's real `bias/weight` ctrl-mapping (`manipulator.py`), steps physics to settle, and measures the actual jaw-collision-geom separation: external action `+1.0` → jaw gap **0.1020m (open)**; external action `-1.0` → jaw gap **0.0180m (closed)**. This exactly reproduces the numbers claimed in `10-05-TWIN-07-VERIFICATION.md`'s "Hardware Re-Test: PASS" addendum. (d) The addendum documents the human's real-hardware re-test (`joint_jog.py ... gripper 20`, pos 71.61→81.55, observed opening) as agreeing with this direction. Both halves of the real-vs-sim comparison are now independently confirmed by this re-verification, not merely re-read from the SUMMARY. |
| 8 | The full required automated test suite stays green (no new regressions) after the TWIN-07 fix, per this phase's own validation contract (10-VALIDATION.md: "Full suite must be green before `/gsd-verify-work`") | ✗ FAILED | Re-ran `pytest LIBERO/libero/libero/envs/ LIBERO/libero/libero/datasets/ -q` live: **51 passed, 2 failed, 2 skipped** (was 52 passed, 1 failed, 2 skipped at the original verification). New failure: `test_collector.py::test_run_scripted_episode_reaches_success_within_budget` (0 successes / 50 attempts) — a genuine, previously-passing test now broken by the TWIN-07 axis-polarity fix leaking into `collector.py`'s stale `OPEN_CMD`/`CLOSE_CMD` constants. See Gaps Summary. |

**Score:** 7/8 truths verified (0 present-but-behavior-unverified)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `LIBERO/libero/libero/assets/grippers/soarm_gripper.xml` | `gripper_left`/`gripper_right` axis+range+ctrlrange flipped to match real hardware polarity | ✓ VERIFIED | Directly read: axes `0 1 0`/`0 -1 0`, both ranges/ctrlranges `-0.042 0`. Matches SUMMARY claim exactly. |
| `LIBERO/libero/libero/envs/grippers/soarm_gripper.py` | `init_qpos` updated to new fully-open value | ✓ VERIFIED | Directly read: `init_qpos` returns `[-0.042, -0.042]`; comment correctly documents the TWIN-07 fix. `format_action` arithmetic unchanged as claimed. |
| `So-101/So-101.urdf` | Regenerated with corrected gripper axes, not hand-edited | ✓ VERIFIED | Directly read `gripper_left`/`gripper_right` joint blocks: axes and `-0.042..0` limits match the MJCF source exactly, consistent with regeneration via `scripts/mjcf_to_urdf.py` (not a hand patch — matching format/structure of the rest of the file). Git history shows commit `7bf580a` "regenerate URDF and update axis test expectations." |
| `scripts/test_verify_urdf.py` / `scripts/verify_urdf.py` | Updated hardcoded expected axis literals | ✓ VERIFIED | Directly read `test_gripper_joint_axes` and `verify_urdf.py`'s `axes_ok`: both use the new `[0,1,0]`/`[0,-1,0]` literals. Both files re-run live and pass; standalone `verify_urdf.py` CLI re-run live, all 4 checkmarks (including gripper axes) print ✓. |
| `.planning/phases/10-digital-twin-fidelity/10-05-TWIN-07-VERIFICATION.md` | Contains a coherent PASS addendum on top of the original FAIL, not a silent overwrite | ✓ VERIFIED | Read the full file: original "Outcome: FAIL" section preserved verbatim (audit trail), followed by a "Gap-Closure Fix Applied" section and a "Hardware Re-Test: PASS" section with concrete before/after command values (`joint_jog.py ... gripper 20`, pos 71.61→81.55) and an explicit final "Outcome: PASS" that supersedes the FAIL header. Internally consistent, no contradictions. |
| `.planning/REQUIREMENTS.md` | Traceability table reflects verified state | ⚠️ NOT YET UPDATED | At verification time, TWIN-07 is still marked `[ ]` unchecked / "Pending" in the traceability table (line 117, line 201), despite the direction-match claim itself now being verified. **Not updated by this verification** — see Gaps Summary; a full-suite regression exists, so this report does not flip Phase 10 to a clean "complete" state that would justify closing out REQUIREMENTS.md's TWIN-07 row. |
| `LIBERO/libero/libero/datasets/collector.py` | (Not a Phase 10 artifact, but affected by Phase 10's fix) | ✗ REGRESSED | `OPEN_CMD`/`CLOSE_CMD` constants (lines 81-82) are now semantically backwards post-fix; not updated by this or any Phase 10 plan. See Gaps Summary. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| `soarm_gripper.xml`'s corrected axis/range | `So-101/So-101.urdf`'s gripper joints | `scripts/mjcf_to_urdf.py` regeneration | ✓ WIRED | URDF's `gripper_left`/`gripper_right` axes and limits are byte-identical in sign/value to the MJCF source (independently diffed). |
| `soarm_gripper.py`'s `format_action`/`init_qpos` | `soarm_gripper.xml`'s actuator `ctrlrange` | robosuite's `manipulator.py` bias/weight rescale (`applied = bias + weight * gripper_action_actual`) | ✓ WIRED | Independently traced this link into the installed `robosuite` package source (not assumed) and used it in my own standalone simulation — confirmed the full chain from external action sign to physical jaw separation. |
| `LIBERO/libero/libero/datasets/collector.py`'s `OPEN_CMD`/`CLOSE_CMD` | `soarm_gripper.xml`'s corrected polarity | Direct constant reuse (no rescale/translation layer) | ✗ NOT RE-WIRED | This is the source of the new regression: the constants were never updated to reflect the new polarity, so the link between `collector.py`'s grasp-phase intent and the actual physical gripper action is now inverted. Not a Phase 10-owned file, but a direct, verified consequence of a Phase 10 gap-closure commit. |

### Data-Flow Trace (Level 4)

Not applicable — static robot-model asset files (MJCF/URDF), not a UI/API rendering dynamic data. Same as the original verification.

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| TWIN-01..05 independent structural/limit assertions against generated URDF | `pytest scripts/test_verify_urdf.py -v` | 5 passed | ✓ PASS |
| `env.reset()` on real `Soarm101` LIBERO env after the gripper-polarity fix | `pytest LIBERO/libero/libero/envs/test_camera_config.py -v` | 2 passed | ✓ PASS |
| `calibration_utils.py`'s own regression tests | `pytest scripts/test_calibration_utils.py -v` | 2 passed | ✓ PASS |
| Standalone `verify_urdf.py` CLI | `python3 scripts/verify_urdf.py` | All 4 checkmarks (kinematic chain, wrist_roll, gripper axes, mesh paths), no warnings | ✓ PASS |
| **Independent MuJoCo simulation of TWIN-07's direction claim** (not part of the existing test suite — written fresh for this re-verification) | Custom script: load `soarm_gripper.xml`, drive `SoarmGripper.format_action(+1)`/`(-1)` through robosuite's real ctrl-mapping, step physics, measure jaw-collision-geom separation | action=+1 → gap 0.1020m (open); action=-1 → gap 0.0180m (closed) — matches the VERIFICATION addendum's claimed numbers exactly | ✓ PASS |
| Full LIBERO envs+datasets suite regression (run once, per constraint) | `pytest LIBERO/libero/libero/envs/ LIBERO/libero/libero/datasets/ -q` | **51 passed, 2 failed, 2 skipped** | ✗ FAIL (new regression) |

**Full-suite failure investigation:**
1. `test_replay.py::test_verify_full_obs_regeneration_passes_on_04_02_output` — pre-existing since Phase 5 (2026-08-10), unrelated to gripper polarity, already accepted as non-blocking debt in the original verification. Unchanged.
2. `test_collector.py::test_run_scripted_episode_reaches_success_within_budget` — **NEW.** Ran in isolation (`pytest LIBERO/libero/libero/datasets/test_collector.py::test_run_scripted_episode_reaches_success_within_budget -v`): completes cleanly through all 50 scripted attempts with `[collect_task] put_the_cream_cheese_in_the_bowl.bddl: 0 successes / 50 attempts` — a deterministic, total failure to grasp, not flakiness. Root-caused by reading `collector.py`: `CLOSE_CMD = 1.0` is sent during the FSM's "grasp" phase, which under the corrected sim polarity (confirmed above: action `+1` → open) now **opens** the jaws instead of closing them on the object. This is a direct, mechanical consequence of the TWIN-07 fix, not touched by any Phase 10 plan or the gap-closure quick task. **This is counted as a gap** (see Gaps Summary) — it was disclosed only as a forward-looking STATE.md "risk," not as a test that is actually red today, and the phase's own validation contract requires the full suite to be green before closing.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|--------------|--------|----------|
| TWIN-01 | 10-03, 10-04 | Gripper attached at end of kinematic chain | ✓ SATISFIED | Truth #1 |
| TWIN-02 | 10-01, 10-03, 10-04 | `wrist_roll` joint with correct range | ✓ SATISFIED | Truth #2 |
| TWIN-03 | 10-02, 10-03, 10-04 | `gripper_left`/`gripper_right` prismatic joints, correct direction | ✓ SATISFIED | Truth #3 |
| TWIN-04 | 10-03, 10-04 | In-repo relative mesh paths | ✓ SATISFIED | Truth #4 |
| TWIN-05 | 10-01, 10-04 | Calibration-derived joint limits | ✓ SATISFIED | Truth #5 |
| TWIN-06 | 10-01, 10-03, 10-04 | MJCF/URDF agreement | ✓ SATISFIED | Truth #6 |
| TWIN-07 | 10-05, quick-260919-h8v | Real-vs-sim gripper direction match | ✓ SATISFIED (direction claim) — but see full-suite regression | Truth #7. The literal requirement text ("driving the simulated gripper ... opens/closes it in the same direction as the real gripper") is satisfied and independently reproduced. However, achieving it introduced an undisclosed regression elsewhere in the test suite (Truth #8), which this report treats as a phase-closing blocker per the phase's own validation contract — not a failure of TWIN-07's own literal claim. |

**Orphaned requirements check:** All 7 TWIN-01..07 IDs are declared across Phase 10's plans and present in REQUIREMENTS.md's Digital-Twin Fidelity section and Traceability table. No orphaned requirements.

**REQUIREMENTS.md update:** **Not applied by this verification.** TWIN-07's literal claim is satisfied, but this report withholds marking it `[x]`/"Complete" in `.planning/REQUIREMENTS.md` because doing so would represent Phase 10 as cleanly closed while a genuine, newly-discovered regression (Truth #8) remains open and unacknowledged by any human decision. Once the `collector.py` regression is either fixed or explicitly accepted via a human-signed override, TWIN-07 and the REQUIREMENTS.md traceability row should be updated together.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `scripts/mjcf_to_urdf.py` | 65, 371, 406, 441, 476, 511, 546 | `PLACEHOLDER` in `velocity="10.0"` comment/constant | ℹ️ Info | Pre-existing, explicitly documented, intentional (no calibrated velocity data exists). Not a TWIN-01..07 requirement. Not a blocker. Unchanged from original verification. |
| `LIBERO/libero/libero/datasets/collector.py` | 81-82 | `OPEN_CMD`/`CLOSE_CMD` constants now semantically inverted post-fix | 🛑 Blocker (regression) | Causes `test_run_scripted_episode_reaches_success_within_budget` to fail deterministically (0/50). See Gaps Summary. |
| (repo-wide) | — | `So-101/` and `coppelia/` mesh asset directories (34 files, now 35 including a new `.pre-phase10-broken.bak`) remain **untracked** in git | ⚠️ Warning | Unchanged, non-blocking, already flagged in the original verification as a repo-hygiene item outside TWIN-01..07's scope. |

No `TBD`/`FIXME`/`XXX` unresolved debt markers found in any Phase-10 or gap-closure-modified file.

### Human Verification Required

None outstanding for TWIN-07 itself — the required human-in-the-loop hardware re-test has been executed and its PASS outcome is documented and independently corroborated by this report's own MuJoCo reproduction.

One item requires a **human decision** (not human testing): whether to (a) apply the trivial `collector.py` `OPEN_CMD`/`CLOSE_CMD` fix now so the full suite returns to green and Phase 10 can close cleanly, or (b) explicitly accept the regression as out-of-scope (Phase 8/9 demo-recollection is paused) via a signed override in this file's frontmatter. This report does not make that call unilaterally.

### Gaps Summary

**TWIN-07's core claim is achieved and independently corroborated.** The gap-closure quick task `260919-h8v` correctly diagnosed and fixed the original real-vs-sim gripper direction mismatch: `soarm_gripper.xml`'s `gripper_left`/`gripper_right` axis, range, and `ctrlrange` were flipped in a physically coherent way (same 84mm stroke, reversed command mapping), `So-101.urdf` was properly regenerated (not hand-edited), and the TWIN-01..05 + env-reset regression suite (7 tests) passes with zero regressions. This re-verification went further than re-reading the SUMMARY: it independently re-ran all 7 of those tests live, and additionally wrote and ran a standalone MuJoCo simulation (not part of the existing suite) that reproduces the exact jaw-gap numbers (0.018m closed / 0.102m open) claimed in `10-05-TWIN-07-VERIFICATION.md`'s "Hardware Re-Test: PASS" addendum, confirming the sim-side half of the real-vs-sim comparison is genuinely, not just narratively, true. The human's hardware re-test (documented, two-command comparison, `joint_jog.py`) is accepted as the human-in-the-loop half per this phase's established methodology (D-07) — no agent process can access the physical robot.

**New blocking gap (Truth #8): a full-suite regression, not disclosed as a currently-failing test.** Fixing TWIN-07's axis polarity had a side effect the gap-closure SUMMARY and STATE.md both anticipated in the abstract ("collector.py's OPEN_CMD/CLOSE_CMD constants now drive the opposite of their names post-fix") but characterized only as a *forward-looking risk for future Phase 8/9 resumption* — not as an already-red test. Independently re-running the phase's own required full-suite command (`pytest LIBERO/libero/libero/envs/ LIBERO/libero/libero/datasets/ -q`, per `10-VALIDATION.md`'s explicit "Before `/gsd-verify-work`: Full suite must be green" gate) shows this is not hypothetical: `test_collector.py::test_run_scripted_episode_reaches_success_within_budget`, a test that was passing at the time of the original Phase 10 verification (52 passed / 1 pre-existing failure), now deterministically fails (0 successes / 50 attempts) because `collector.py`'s scripted grasp FSM sends `CLOSE_CMD` during its grasp phase, which under the corrected polarity now opens the jaws instead of closing them.

This is a genuine, currently-manifested regression in the live test suite, caused directly by Phase 10 gap-closure work, and it violates the phase's own stated closing gate. It is not one of the TWIN-01..07 roadmap requirements by name, and `collector.py` belongs to the currently-paused Phase 8/9 demo-recollection track — but "the FSM is unused/paused" (STATE.md's framing) does not hold up under direct inspection: the FSM has an active, non-skipped automated test that is failing today, not merely a theoretical future risk.

**Recommended next action:** Either (a) apply the 2-constant swap in `LIBERO/libero/libero/datasets/collector.py` (trivial, ~5 min), re-run `test_collector.py` and the full suite to confirm a clean return to baseline, and then re-run this verification once more to close Phase 10 with a true 8/8 — or (b) have a human explicitly accept this regression as out-of-scope via a signed override entry in this file's frontmatter, at which point TWIN-07 and `.planning/REQUIREMENTS.md`'s traceability row can be marked complete on the same pass.

**Non-blocking follow-up (repo hygiene, carried over, not a phase-goal gap):** `git add` the untracked `So-101/` and `coppelia/` mesh asset directories (35 files including the new `.pre-phase10-broken.bak`) so the URDF's relative mesh references resolve on a fresh clone.

---

_Verified: 2026-09-19T11:16:39Z_
_Verifier: Claude (gsd-verifier)_
