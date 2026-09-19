---
phase: 10-digital-twin-fidelity
verified: 2026-09-19T11:28:32Z
status: passed
score: 8/8 must-haves verified
behavior_unverified: 0
overrides_applied: 0
gap_closure_rounds: 2
re_verification:
  previous_status: gaps_found
  previous_score: 7/8
  gaps_closed:
    - "The full required automated test suite (`pytest LIBERO/libero/libero/envs/ LIBERO/libero/libero/datasets/ -q`) stays green — no new regressions — after the TWIN-07 gap-closure fix. Commit `7cec986` swapped `OPEN_CMD`/`CLOSE_CMD` to `1.0`/`-1.0` in `collector.py` and de-hardcoded `test_collector.py`'s 5 gripper-direction assertions to check `np.sign(action[6]) == np.sign(OPEN_CMD/CLOSE_CMD)` instead of literal signs. Independently re-ran both the isolated `test_collector.py` suite (12/12 pass) and the full `LIBERO/libero/libero/envs/ LIBERO/libero/libero/datasets/` gate (52 passed, 1 pre-existing unrelated failure, 2 skipped) during this re-verification."
  gaps_remaining: []
  regressions: []
gaps: []
---

# Phase 10: Digital-Twin Fidelity Verification Report

**Phase Goal:** The URDF and the MuJoCo XML (`LIBERO/libero/libero/assets/robots/soarm101/robot.xml`) form a correct, complete, 1:1 kinematic match to the real SO-ARM101 — a proper parent/child chain ending at the gripper, `wrist_roll` and gripper jaw joints restored with correct axes/direction, correct joint limits, and in-repo mesh paths. This is sim-side-only work; it does not touch the real robot or the `control/` bridge.

**Verified:** 2026-09-19T11:28:32Z
**Status:** passed
**Re-verification:** Yes — final pass (3rd verification) after **two rounds** of gap closure:
1. Round 1 (`260919-h8v`): fixed TWIN-07's real-vs-sim gripper direction mismatch by flipping `gripper_left`/`gripper_right` MJCF axis/range/ctrlrange polarity.
2. Round 2 (commit `7cec986`, this session): fixed the regression that round 1's fix introduced — `collector.py`'s stale `OPEN_CMD`/`CLOSE_CMD` constants and `test_collector.py`'s hardcoded-literal-sign assertions that had masked the drift.

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | URDF's gripper is attached at the end of the kinematic chain (child of wrist), not floating off `robot_base` (TWIN-01) | ✓ VERIFIED | Re-ran `pytest scripts/test_verify_urdf.py::test_gripper_is_child_of_wrist` live: PASSED. Independently parsed `So-101/So-101.urdf` with `xml.etree.ElementTree`: `gripper_left`/`gripper_right` joints have `parent=right_gripper`, which sits within the wrist-descended chain, not a root-level sibling of `base`. |
| 2 | URDF includes a `wrist_roll` joint with range matching the real robot's calibrated servo limits (TWIN-02) | ✓ VERIFIED | Re-ran `pytest scripts/test_verify_urdf.py::test_wrist_roll_range` live: PASSED. Independently parsed the URDF: `wrist_roll` axis `0 0 1`, limit `(-2.7438..., 2.8412...)` — unaffected by the gripper-polarity fix, unchanged since original verification. |
| 3 | URDF includes `gripper_left`/`gripper_right` prismatic joints whose open/close direction matches the real gripper (TWIN-03) | ✓ VERIFIED | Independently parsed `So-101/So-101.urdf`: `gripper_left` axis `0 1 0` limit `-0.042..0`; `gripper_right` axis `0 -1 0` limit `-0.042..0`. Independently diffed against `LIBERO/libero/libero/assets/grippers/soarm_gripper.xml` MJCF source (lines 57-58, 90-106): axes/ranges/ctrlranges match exactly. `pytest scripts/test_verify_urdf.py::test_gripper_joint_axes` re-run live: PASSED. |
| 4 | All mesh file references in the URDF use in-repo relative paths (TWIN-04) | ✓ VERIFIED | Re-ran `pytest scripts/test_verify_urdf.py::test_no_absolute_mesh_paths` live: PASSED. Independently grepped all `<mesh filename="...">` entries in the URDF: all are bare relative filenames (e.g. `So-101_base_link_visual_vis_1.dae`), no absolute paths. |
| 5 | URDF joint limits for the 4 calibration-derivable arm joints match the real servo calibration ranges (TWIN-05) | ✓ VERIFIED | Re-ran `pytest scripts/test_verify_urdf.py::test_joint_limits_match_calibration` live: PASSED. Unaffected by the gripper-polarity fix. |
| 6 | The MuJoCo XML is brought into agreement with the corrected URDF's kinematic structure (TWIN-06) | ✓ VERIFIED | Re-ran `pytest LIBERO/libero/libero/envs/test_camera_config.py` live (constructs a real `Soarm101` env, calls `env.reset()`): 2 passed. Gripper axes/ranges independently diffed identical between MJCF and regenerated URDF (see Truth #3). |
| 7 | Driving the simulated gripper with a given joint command opens/closes it in the same direction as the real gripper under the same command (TWIN-07) | ✓ VERIFIED | **Independently re-derived from first principles this pass, not re-read from prior reports.** Read `soarm_gripper.xml`: `gripper_left` axis `0 1 0`/range `-0.042 0`, `gripper_right` axis `0 -1 0`/range `-0.042 0`, `ctrlrange -0.042 0` on both actuators. Read `soarm_gripper.py`: `init_qpos` = `[-0.042, -0.042]`; `format_action`'s arithmetic `current_action = clip(current_action + [-1,-1]*speed*sign(action), -1, 1)` unchanged. Wrote and ran a standalone script that instantiates `SoarmGripper` and repeatedly applies `format_action` with the collector's actual constants: external action `CLOSE_CMD=-1.0` converges `current_action -> [1, 1]` (maps via `ctrlrange -0.042 0` to the closed/`0` end); external action `OPEN_CMD=+1.0` converges `current_action -> [-1, -1]` (maps to the open/`-0.042` end). This confirms `collector.py`'s `OPEN_CMD`/`CLOSE_CMD` constants are correctly signed against the actual `format_action` arithmetic and MJCF `ctrlrange`, not just internally self-consistent on paper. Cross-checked against `10-05-TWIN-07-CHECK.md`'s "Hardware Re-Test: PASS" section (read directly, see below) — the human's real-hardware re-test (`joint_jog.py ... gripper 20`, pos 71.61→81.55, observed opening under a positive/increasing command) agrees with sim's `OPEN_CMD=+1.0` → open direction. Both halves of the real-vs-sim comparison, plus the arithmetic that connects `collector.py`'s named constants to the physical jaw position, are now independently confirmed. |
| 8 | The full required automated test suite stays green (no new regressions) after the TWIN-07 fix, per this phase's own validation contract (10-VALIDATION.md: "Full suite must be green before `/gsd-verify-work`") | ✓ VERIFIED | Independently re-ran live (not trusted from SUMMARY): (a) `pytest LIBERO/libero/libero/datasets/test_collector.py -q` → **12 passed**, including `test_run_scripted_episode_reaches_success_within_budget` (previously 0/50, now succeeds within budget). (b) Full gate `pytest LIBERO/libero/libero/envs/ LIBERO/libero/libero/datasets/ -q` → **52 passed, 1 failed, 2 skipped** — matches the original (pre-regression) baseline exactly. The 1 failure is `test_replay.py::test_verify_full_obs_regeneration_passes_on_04_02_output`, confirmed via `.planning/STATE.md` line 94 to be pre-existing Phase 5 debt (offscreen-render pixel non-determinism), unrelated to gripper polarity. |

**Score:** 8/8 truths verified (0 present-but-behavior-unverified)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `LIBERO/libero/libero/assets/grippers/soarm_gripper.xml` | `gripper_left`/`gripper_right` axis+range+ctrlrange flipped to match real hardware polarity | ✓ VERIFIED | Directly read: axes `0 1 0`/`0 -1 0`, both ranges/ctrlranges `-0.042 0`. |
| `LIBERO/libero/libero/envs/grippers/soarm_gripper.py` | `init_qpos` updated to new fully-open value; `format_action` arithmetic connects external action sign to correct physical direction | ✓ VERIFIED | Directly read and numerically exercised `format_action`: confirms `OPEN_CMD=+1.0` drives toward the open ctrlrange end, `CLOSE_CMD=-1.0` toward closed. |
| `So-101/So-101.urdf` | Regenerated with corrected gripper axes, not hand-edited | ✓ VERIFIED | Gripper joint blocks match the MJCF source exactly (independently diffed). |
| `scripts/test_verify_urdf.py` / `scripts/verify_urdf.py` | Updated hardcoded expected axis literals | ✓ VERIFIED | Both re-run live: 5/5 tests pass with the new `[0,1,0]`/`[0,-1,0]` literals. |
| `LIBERO/libero/libero/datasets/collector.py` | `OPEN_CMD`/`CLOSE_CMD` constants match the corrected gripper polarity | ✓ VERIFIED (fixed this round) | Directly read lines 81-82: `OPEN_CMD = 1.0`, `CLOSE_CMD = -1.0`. Numerically confirmed these values drive `format_action`/MJCF `ctrlrange` toward the physically correct open/closed ends (see Truth #7). |
| `LIBERO/libero/libero/datasets/test_collector.py` | Gripper-direction assertions check named constants, not hardcoded literal signs (root-cause fix for the drift that caused Round 2's regression) | ✓ VERIFIED | Directly read: all 6 gripper-sign assertion sites (lines 81, 100, 107, 117, 123, 127) use `np.sign(action[6]) == np.sign(OPEN_CMD)` / `np.sign(CLOSE_CMD)`, importing the named constants (lines 40-41) rather than hardcoded literals. |
| `.planning/phases/10-digital-twin-fidelity/10-05-TWIN-07-CHECK.md` | Contains a coherent PASS addendum on top of the original FAIL, not a silent overwrite | ✓ VERIFIED | Read the full file: original "Outcome: FAIL" section preserved verbatim (audit trail), followed by "Gap-Closure Fix Applied" and "Hardware Re-Test: PASS" sections with concrete before/after servo values (`joint_jog.py ... gripper 20`, pos 71.61→81.55) and an explicit final "Outcome: PASS" superseding the FAIL header. Internally coherent. |
| `.planning/REQUIREMENTS.md` | Traceability table reflects verified state | ✓ UPDATED (this verification) | TWIN-07 flipped from `[ ]`/"Pending" to `[x]`/"Complete" (line 117, line 201) as part of this pass, now consistent with TWIN-01..06. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| `soarm_gripper.xml`'s corrected axis/range | `So-101/So-101.urdf`'s gripper joints | `scripts/mjcf_to_urdf.py` regeneration | ✓ WIRED | URDF's `gripper_left`/`gripper_right` axes and limits are byte-identical in sign/value to the MJCF source (independently diffed). |
| `soarm_gripper.py`'s `format_action`/`init_qpos` | `soarm_gripper.xml`'s actuator `ctrlrange` | robosuite's `manipulator.py` bias/weight rescale | ✓ WIRED | Independently traced and numerically exercised: `format_action`'s `current_action` output correctly maps to the physically correct open/closed `ctrlrange` end for both `OPEN_CMD`/`CLOSE_CMD`. |
| `LIBERO/libero/libero/datasets/collector.py`'s `OPEN_CMD`/`CLOSE_CMD` | `soarm_gripper.xml`'s corrected polarity | Direct constant reuse (no rescale/translation layer) | ✓ RE-WIRED (fixed this round) | `OPEN_CMD = 1.0` / `CLOSE_CMD = -1.0` now correctly match the corrected polarity, confirmed both by direct arithmetic trace and by the previously-failing `test_run_scripted_episode_reaches_success_within_budget` now passing (the scripted FSM successfully grasps and completes the task). |
| `test_collector.py`'s gripper-direction assertions | `collector.py`'s `OPEN_CMD`/`CLOSE_CMD` named constants | `np.sign(action[6]) == np.sign(OPEN_CMD/CLOSE_CMD)` | ✓ WIRED | Structural fix confirmed: assertions now derive their expected sign from the constants themselves rather than a hardcoded literal, so a future polarity change would be caught by these same tests without needing a second re-verification pass to discover it. |

### Data-Flow Trace (Level 4)

Not applicable — static robot-model asset files (MJCF/URDF) and FSM tuning constants, not a UI/API rendering dynamic data.

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| TWIN-01..05 independent structural/limit assertions against generated URDF | `pytest scripts/test_verify_urdf.py -v` | 5 passed | ✓ PASS |
| `env.reset()` on real `Soarm101` LIBERO env after the gripper-polarity fix | `pytest LIBERO/libero/libero/envs/test_camera_config.py -v` | 2 passed | ✓ PASS |
| `OPEN_CMD`/`CLOSE_CMD` numerically drive `format_action`'s `current_action` toward the physically correct `ctrlrange` end | Standalone script: instantiate `SoarmGripper`, repeatedly call `format_action([-1.0])` and `format_action([1.0])`, inspect `current_action` convergence | `CLOSE_CMD=-1.0` → `current_action -> [1,1]` (closed end); `OPEN_CMD=+1.0` → `current_action -> [-1,-1]` (open end) | ✓ PASS |
| Isolated collector test suite (includes the previously-failing scripted-grasp test) | `pytest LIBERO/libero/libero/datasets/test_collector.py -q` | **12 passed** | ✓ PASS |
| Full required suite gate (`10-VALIDATION.md`'s closing contract), run once per constraint | `pytest LIBERO/libero/libero/envs/ LIBERO/libero/libero/datasets/ -q` | **52 passed, 1 failed (pre-existing test_replay.py, Phase 5 debt), 2 skipped (TFDS-gated)** — matches original pre-regression baseline exactly | ✓ PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|--------------|--------|----------|
| TWIN-01 | 10-03, 10-04 | Gripper attached at end of kinematic chain | ✓ SATISFIED | Truth #1 |
| TWIN-02 | 10-01, 10-03, 10-04 | `wrist_roll` joint with correct range | ✓ SATISFIED | Truth #2 |
| TWIN-03 | 10-02, 10-03, 10-04 | `gripper_left`/`gripper_right` prismatic joints, correct direction | ✓ SATISFIED | Truth #3 |
| TWIN-04 | 10-03, 10-04 | In-repo relative mesh paths | ✓ SATISFIED | Truth #4 |
| TWIN-05 | 10-01, 10-04 | Calibration-derived joint limits | ✓ SATISFIED | Truth #5 |
| TWIN-06 | 10-01, 10-03, 10-04 | MJCF/URDF agreement | ✓ SATISFIED | Truth #6 |
| TWIN-07 | 10-05, quick-260919-h8v, commit 7cec986 | Real-vs-sim gripper direction match | ✓ SATISFIED | Truth #7 (direction claim, independently re-derived) + Truth #8 (full-suite regression from Round 1's fix now resolved, no new gaps) |

**Orphaned requirements check:** All 7 TWIN-01..07 IDs are declared across Phase 10's plans and present in REQUIREMENTS.md's Digital-Twin Fidelity section and Traceability table. No orphaned requirements.

**REQUIREMENTS.md update:** Applied by this verification. TWIN-07 flipped to `[x]`/"Complete" in both the checklist (line 117) and Traceability table (line 201), matching TWIN-01..06.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `scripts/mjcf_to_urdf.py` | 65, 371, 406, 441, 476, 511, 546 | `PLACEHOLDER` in `velocity="10.0"` comment/constant | ℹ️ Info | Pre-existing, explicitly documented, intentional (no calibrated velocity data exists). Not a TWIN-01..07 requirement. Not a blocker. Unchanged from original verification. |
| `LIBERO/libero/libero/envs/grippers/soarm_gripper.py` | 46-52 | `format_action`'s docstring prose is internally inconsistent/stale — it labels a `+1` external action as a "CLOSE command" that "drives...toward -1," but `-1` is documented two lines earlier as the *open* end, and the numerically-verified arithmetic (this report's Truth #7) shows `+1` external action is actually `OPEN_CMD` per `collector.py` and correctly drives toward the open end. The comment's *labeling* of which external sign is "OPEN" vs "CLOSE" is muddled/stale post-fix; the underlying arithmetic and `ctrlrange` mapping are correct and independently confirmed. | ⚠️ Warning (docs only, not functional) | No behavioral impact — confirmed by direct numerical exercise and by the passing full test suite (including the scripted grasp test) — but the confusing prose is a latent source of exactly the kind of misunderstanding that caused Round 2's regression, and should be cleaned up in a follow-up. |
| (repo-wide) | — | `So-101/` and `coppelia/` mesh asset directories remain **untracked** in git | ⚠️ Warning | Unchanged, non-blocking, already flagged in prior verifications as a repo-hygiene item outside TWIN-01..07's scope. |

No `TBD`/`FIXME`/`XXX`/`HACK` unresolved debt markers found in `collector.py` or `test_collector.py` (the files modified by this round's gap-closure fix).

### Human Verification Required

None. The one item that previously required a human-in-the-loop physical test (TWIN-07's real-vs-sim gripper direction) was already executed and documented as PASS in `10-05-TWIN-07-CHECK.md`'s "Hardware Re-Test: PASS" section, independently read and confirmed coherent by this verification. No new human-testable surface was introduced by this round's `collector.py`/`test_collector.py` fix (it is a pure constant/assertion correction, fully covered by automated tests re-run live during this verification).

### Gaps Summary

None. This is the third and final verification pass for Phase 10, following two rounds of gap closure:

1. **Round 1** (`260919-h8v`) fixed TWIN-07's original real-vs-sim gripper direction mismatch by flipping `gripper_left`/`gripper_right` axis/range/ctrlrange polarity in `soarm_gripper.xml`, regenerating `So-101.urdf`, and updating `test_verify_urdf.py`'s expected literals. A human-in-the-loop hardware re-test confirmed the fix (documented in `10-05-TWIN-07-CHECK.md`).
2. **Round 2** (commit `7cec986`, this session) fixed a regression Round 1's fix introduced but did not disclose as currently-failing: `collector.py`'s `OPEN_CMD`/`CLOSE_CMD` constants were stale, breaking the scripted-collection FSM's grasp phase. This was caught by the prior re-verification pass (2nd verification, `gaps_found`, 7/8) via an independent full-suite run. The fix swapped the constants to `1.0`/`-1.0` and — root-causing why the drift wasn't caught earlier — de-hardcoded `test_collector.py`'s 5 gripper-direction assertions to check `np.sign(action[6]) == np.sign(OPEN_CMD/CLOSE_CMD)` instead of literal signs, so a future polarity change would be structurally caught by these same tests.

This verification independently re-derived and re-ran every check rather than trusting the SUMMARY or STATE.md narration:
- Confirmed `collector.py`'s `OPEN_CMD = 1.0` / `CLOSE_CMD = -1.0` by direct file read.
- Confirmed `test_collector.py`'s 6 gripper-sign assertion sites use the named constants, not literals.
- Numerically exercised `SoarmGripper.format_action` with both constants and confirmed they drive `current_action` toward the physically correct `ctrlrange` end — this goes beyond re-reading prior claims, independently establishing the arithmetic is correct (and surfacing a stale/confusing docstring comment as a non-blocking follow-up).
- Re-ran `pytest LIBERO/libero/libero/datasets/test_collector.py -q` live: 12/12 pass.
- Re-ran the full required gate `pytest LIBERO/libero/libero/envs/ LIBERO/libero/libero/datasets/ -q` live (once, per constraint): 52 passed, 1 failed (pre-existing `test_replay.py`, Phase 5 debt, confirmed via STATE.md), 2 skipped — an exact match to the pre-regression baseline.
- Re-checked TWIN-01..06 from first principles (not assumed carried-over): re-ran `scripts/test_verify_urdf.py` (5/5) and `test_camera_config.py` (2/2) live, and independently parsed/diffed the URDF's gripper joint blocks against the MJCF source.
- Read `10-05-TWIN-07-CHECK.md`'s "Hardware Re-Test: PASS" section directly and confirmed it is present, coherent, and supersedes the original FAIL header without erasing the audit trail.

All 7 TWIN requirements (TWIN-01..07) are genuinely satisfied in the codebase. `.planning/REQUIREMENTS.md` has been updated to mark TWIN-07 complete, matching TWIN-01..06. Phase 10's goal — a correct, complete, 1:1 kinematic match between the URDF/MJCF and the real SO-ARM101 — is achieved.

**Non-blocking follow-ups (not phase-goal gaps, informational only):**
1. `git add` the untracked `So-101/` and `coppelia/` mesh asset directories so the URDF's relative mesh references resolve on a fresh clone.
2. Clean up `soarm_gripper.py`'s `format_action` docstring — the prose mislabels which external action sign is "OPEN" vs "CLOSE" (stale from before the TWIN-07 fix), even though the underlying arithmetic is correct. This kind of prose/code drift is exactly what caused Round 2's regression in `collector.py`; worth a follow-up doc pass to prevent a third recurrence.

---

_Verified: 2026-09-19T11:28:32Z_
_Verifier: Claude (gsd-verifier)_
