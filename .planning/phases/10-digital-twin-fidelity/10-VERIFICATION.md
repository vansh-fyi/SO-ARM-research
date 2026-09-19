---
phase: 10-digital-twin-fidelity
verified: 2026-09-19T12:15:00Z
status: gaps_found
score: 6/7 must-haves verified
behavior_unverified: 0
overrides_applied: 0
gaps:
  - truth: "Driving the simulated gripper with a given joint command opens/closes it in the same direction as the real gripper under the same command (TWIN-07)"
    status: failed
    reason: "Mandatory human-in-the-loop hardware comparison (Plan 10-05) returned an explicit FAIL: the real SO-ARM101's positive/increasing gripper command (`joint_jog.py ... gripper 20`, gripper.pos 61.72 -> 71.64) opened the real jaws, while the sim's external action convention (SoarmGripper.format_action: +1 = closed) closes the jaws under the equivalent translated command. Directions disagree."
    artifacts:
      - path: "LIBERO/libero/libero/assets/grippers/soarm_gripper.xml"
        issue: "gripper_left/gripper_right joint axis signs (0 -1 0 / 0 1 0) produce a sim close-direction that is the opposite of the real hardware's close-direction under the equivalent command"
    missing:
      - "Diagnose and fix the axis-sign mismatch (likely flip gripper_left/gripper_right axis signs in soarm_gripper.xml)"
      - "Regenerate So-101/So-101.urdf via scripts/mjcf_to_urdf.py after the fix"
      - "Update scripts/test_verify_urdf.py::test_gripper_joint_axes expected axis values to match the corrected convention"
      - "Re-run the TWIN-07 human-in-the-loop real-vs-sim comparison and obtain an explicit PASS before closing the phase"
---

# Phase 10: Digital-Twin Fidelity Verification Report

**Phase Goal:** The URDF and the MuJoCo XML (`LIBERO/libero/libero/assets/robots/soarm101/robot.xml`) form a correct, complete, 1:1 kinematic match to the real SO-ARM101 — a proper parent/child chain ending at the gripper, `wrist_roll` and gripper jaw joints restored with correct axes/direction, correct joint limits, and in-repo mesh paths. This is sim-side-only work; it does not touch the real robot or the `control/` bridge.

**Verified:** 2026-09-19T12:15:00Z
**Status:** gaps_found
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | URDF's gripper is attached at the end of the kinematic chain (child of wrist), not floating off `robot_base` (TWIN-01) | ✓ VERIFIED | Independently parsed `So-101/So-101.urdf`: `right_hand_to_gripper` joint has `<parent link="right_hand"/>` / `<child link="right_gripper"/>`; `right_gripper` is not a root link. `pytest scripts/test_verify_urdf.py::test_gripper_is_child_of_wrist` passes (re-run live, 5/5 passed). |
| 2 | URDF includes a `wrist_roll` joint with range matching the real robot's calibrated servo limits (TWIN-02) | ✓ VERIFIED | `robot.xml` line 109: `range="-2.7438472969992493 2.841206309382605"`. URDF `wrist_roll` `<limit lower="-2.7438472969992493" upper="2.841206309382605".../>` — byte-identical, confirmed by direct grep of both files (not just trusting the generator). `test_wrist_roll_range` passes. |
| 3 | URDF includes `gripper_left`/`gripper_right` prismatic joints whose open/close direction matches the real gripper, fixing the mirrored-gears bug (TWIN-03) | ✓ VERIFIED (structural claim only — see note) | URDF: `gripper_left` `type="prismatic"` `axis xyz="0 -1 0"`; `gripper_right` `type="prismatic"` `axis xyz="0 1 0"` — mirrored/opposite axes matching `soarm_gripper.xml`'s own joint definitions exactly (confirmed via direct file read). `test_gripper_joint_axes` passes. **Note:** the two jaws move in opposite directions from each other (the specific "mirrored-gears" defect this requirement names, per D-01/10-RESEARCH.md, is about the jaws being mirrored relative to each other, not about absolute polarity vs. real hardware). TWIN-07 below found that the *absolute* command-to-direction polarity does NOT match the real robot. See Gaps Summary for why these are tracked as functionally distinct requirements, and why this distinction is a judgment call worth human awareness. |
| 4 | All mesh file references in the URDF use in-repo relative paths, not absolute `~/Downloads/` paths (TWIN-04) | ✓ VERIFIED | `grep -n 'filename="/'` and `grep -n 'filename="file://'` against `So-101/So-101.urdf` both return zero matches. `test_no_absolute_mesh_paths` passes. |
| 5 | URDF joint limits for the 4 calibration-derivable arm joints match the real servo calibration ranges (TWIN-05) | ✓ VERIFIED | Independently recomputed all 4 values via `calibration_utils.calibration_ticks_to_radians()` against the live `soarm_follower_02.json` calibration file (not trusting the plan's own numbers): shoulder_pan (-1.227484, 1.227484), shoulder_lift (-1.875749, 1.875749), elbow_flex (-1.696997, 1.696997), wrist_flex (-1.780619, 1.780619) — all match `robot.xml`'s and the URDF's actual values exactly. `wrist_roll`/gripper correctly excluded per D-06. `test_joint_limits_match_calibration` passes. |
| 6 | The MuJoCo XML is brought into agreement with the corrected URDF's kinematic structure (TWIN-06) | ✓ VERIFIED | Direct side-by-side comparison of `robot.xml`'s 5 arm-joint `range` attributes against the URDF's corresponding `<limit>` values: identical. `LIBERO/libero/libero/envs/test_camera_config.py -x` (constructs a real `Soarm101` env and calls `env.reset()`) re-run live: 2 passed. |
| 7 | Driving the simulated gripper with a given joint command opens/closes it in the same direction as the real gripper under the same command (TWIN-07) | ✗ FAILED | Documented, human-executed hardware comparison in `10-05-TWIN-07-VERIFICATION.md`: real robot's positive gripper command (`joint_jog.py ... gripper 20`, telemetry 61.72→71.64) **opened** the jaws; sim's `+1` (closed) external convention closes them under the equivalent command. Directions disagree — explicit FAIL, confirmed by the human across two rounds of confirmation. |

**Score:** 6/7 truths verified (0 present-but-behavior-unverified)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `scripts/calibration_utils.py` | Calibration tick→radian formula + CLI | ✓ VERIFIED | Exists; `calibration_ticks_to_radians()` independently re-run against live calibration JSON, output matches robot.xml exactly; `pytest scripts/test_calibration_utils.py -x` → 2 passed (re-run live). |
| `scripts/mjcf_to_urdf.py` | MJCF→URDF converter | ✓ VERIFIED | Exists (~430 lines), fail-loud precondition checks present; running it (implicitly, via the already-generated URDF matching its stated 9-link/8-joint output) is consistent with the committed URDF's structure. |
| `So-101/So-101.urdf` | Regenerated, correct 9-link/8-joint URDF | ✓ VERIFIED | Structural check (independent ElementTree parse): exactly 9 `<link>` and 8 `<joint>` elements, matching plan spec exactly. |
| `scripts/verify_urdf.py` + `scripts/test_verify_urdf.py` | Independent yourdfpy-based acceptance gate | ✓ VERIFIED | Both exist; `pytest scripts/test_verify_urdf.py -x` re-run live → 5 passed (TWIN-01..05). |
| `LIBERO/libero/libero/assets/grippers/soarm_gripper.xml` | Corrected clamp visual mesh offsets (D-04) | ✓ VERIFIED (visual fix only) | `left_jaw_visual`/`right_jaw_visual` geom `pos` attributes present with D-04 rationale comments; collision geoms/joints unchanged per plan's own acceptance criteria. Diagnostic PNGs (`diagnostics/outputs/gripper_clamp_{open,closed}.png`) are correctly absent from the working tree because `diagnostics/outputs/` is gitignored (not a gap — human sign-off on these images is recorded in 10-02-SUMMARY.md as "approved"). |
| `.planning/phases/10-digital-twin-fidelity/10-05-TWIN-07-VERIFICATION.md` | TWIN-07 outcome record | ✓ VERIFIED (exists) — but the recorded outcome is FAIL | File exists, contains exact command values, explicit FAIL outcome, and date, per plan's Task 2 acceptance criteria. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| `calibration_utils.calibration_ticks_to_radians()` | `robot.xml`'s 4 arm-joint ranges | Hand-applied calibration-derived values | ✓ WIRED | Independently recomputed values match `robot.xml`'s actual committed values to the documented precision. |
| `soarm_gripper.xml`'s corrected clamp geom offsets | `scripts/mjcf_to_urdf.py`'s gripper visual mesh origins | Live ElementTree read at generation time | ✓ WIRED | `mjcf_to_urdf.py` reads pos/quat live from the MJCF tree per its own SUMMARY; generated URDF's mesh origin values are internally consistent with this design (script reads live, does not hardcode). |
| `robot.xml` + `soarm_gripper.xml` (post-Plan 10-01/10-02) | `So-101/So-101.urdf` | `scripts/mjcf_to_urdf.py` | ✓ WIRED | URDF's joint limits/axes for all 5 revolute + 2 prismatic joints exactly match the corresponding MJCF source values (independently diffed, not just trusting SUMMARY claims). |
| `scripts/test_verify_urdf.py` | `scripts/verify_urdf.py` + `scripts/calibration_utils.py` | Python import | ✓ WIRED | Test file imports and reuses `verify_urdf`'s tree-walk helpers and `calibration_utils`'s formula rather than duplicating logic (confirmed by successful live test run — an import error would surface as a collection failure, and none occurred). |

### Data-Flow Trace (Level 4)

Not applicable in the standard sense — this phase produces static robot-model asset files (MJCF/URDF), not a UI or API that renders dynamic data from a live source. The relevant "data flow" is MJCF → URDF generation, verified above as WIRED with live-read (not hardcoded) values.

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Calibration formula matches live hardware calibration | `python3 -c "...calibration_ticks_to_radians(...)..."` against live `soarm_follower_02.json` | Output matches `robot.xml`'s committed values exactly (see Observable Truths #5) | ✓ PASS |
| `env.reset()` on real `Soarm101` LIBERO env after all Phase 10 MJCF edits | `conda run -n libero python3 -m pytest LIBERO/libero/libero/envs/test_camera_config.py -x` | 2 passed | ✓ PASS |
| TWIN-01..05 independent structural/limit assertions against generated URDF | `conda run -n libero python3 -m pytest scripts/test_verify_urdf.py -x` | 5 passed | ✓ PASS |
| `calibration_utils.py`'s own regression tests | `conda run -n libero python3 -m pytest scripts/test_calibration_utils.py -x` | 2 passed | ✓ PASS |
| Full LIBERO envs+datasets suite regression (run once, per constraint) | `conda run -n libero python3 -m pytest LIBERO/libero/libero/envs/ LIBERO/libero/libero/datasets/ -q` | 52 passed, 1 failed, 2 skipped | ⚠ 1 pre-existing failure (see below) |

**Full-suite failure investigated:** `test_replay.py::test_verify_full_obs_regeneration_passes_on_04_02_output` fails with an `agentview_rgb` pixel mismatch. Cross-referenced against `.planning/STATE.md` line 94 and `06-VERIFICATION.md`: this exact test has been documented, open, pre-existing debt since 2026-08-10 (Phase 5), predating Phase 10 by over a month, attributed to MuJoCo offscreen-render non-determinism and never fixed. `git log` on `test_replay.py`/`replay.py` shows no Phase 10 commit touched either file. **Not a Phase 10 regression** — not counted as a gap.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|--------------|--------|----------|
| TWIN-01 | 10-03, 10-04 | Gripper attached at end of kinematic chain | ✓ SATISFIED | See Observable Truth #1 |
| TWIN-02 | 10-01, 10-03, 10-04 | `wrist_roll` joint with correct range | ✓ SATISFIED | See Observable Truth #2 |
| TWIN-03 | 10-02, 10-03, 10-04 | `gripper_left`/`gripper_right` prismatic joints, correct direction | ✓ SATISFIED (structural reading — see note in Truth #3) | See Observable Truth #3 |
| TWIN-04 | 10-03, 10-04 | In-repo relative mesh paths | ✓ SATISFIED | See Observable Truth #4 |
| TWIN-05 | 10-01, 10-04 | Calibration-derived joint limits | ✓ SATISFIED | See Observable Truth #5 |
| TWIN-06 | 10-01, 10-03, 10-04 | MJCF/URDF agreement | ✓ SATISFIED | See Observable Truth #6 |
| TWIN-07 | 10-05 | Real-vs-sim gripper direction match | ✗ BLOCKED | See Observable Truth #7 — explicit FAIL |

**Orphaned requirements check:** All 7 TWIN-01..07 requirement IDs declared across Phase 10's plan frontmatters (`10-01`: TWIN-05, TWIN-06; `10-02`: TWIN-03, TWIN-07 [scope note only — not claimed complete]; `10-03`: TWIN-01, TWIN-02, TWIN-03, TWIN-04, TWIN-06; `10-04`: TWIN-01, TWIN-02, TWIN-03, TWIN-04, TWIN-05, TWIN-06; `10-05`: TWIN-07) are present in REQUIREMENTS.md's v2.0 Digital-Twin Fidelity section and its Traceability table. No orphaned requirements — every TWIN-01..07 ID is accounted for in a phase plan. REQUIREMENTS.md's Traceability table currently marks TWIN-01..06 "Complete" and TWIN-07 "Pending" — this matches the actual verified state (6/7 pass, TWIN-07 fails) and does NOT need correction.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `scripts/mjcf_to_urdf.py` | 65, 369, 406, 441, 476, 511, 546 | `PLACEHOLDER` in `velocity="10.0"` comment/constant | ℹ️ Info | Explicitly documented, intentional per the plan's own instructions ("no calibrated velocity data exists — comment this as a placeholder"). Not a TWIN-01..07 requirement (none reference joint velocity limits). Not a blocker. |
| (repo-wide) | — | `So-101/` and `coppelia/` mesh asset directories (34 files) remain **untracked** in git | ⚠️ Warning | The generated URDF's mesh references resolve correctly in the current working tree, but a fresh clone of the repo would be missing these files, breaking URDF mesh loading for anyone else. Explicitly flagged as an open item in both 10-03-SUMMARY.md and 10-04-SUMMARY.md ("outside this plan's scope"), not remediated by any Phase 10 plan. Does not block TWIN-01..06 (which only require kinematic/structural correctness, confirmed via `load_meshes=False`), but is a real repo-completeness gap worth closing before calling Phase 10's artifacts durable. |

No `TBD`/`FIXME`/`XXX` unresolved debt markers found in any Phase-10-modified file.

### Human Verification Required

None outstanding. TWIN-07's human-in-the-loop check (the only item requiring human/hardware judgment in this phase) has already been executed and its outcome (FAIL) is documented — this is not an open human-verification item, it is a resolved gap requiring a code fix + re-verification.

### Gaps Summary

**Primary gap — TWIN-07 (blocking):** The phase's final and most safety-critical requirement — that the simulated gripper's direction under a given command matches the real robot's direction under the same command — was tested via the mandatory human-in-the-loop hardware comparison specified in Plan 10-05, and the result is an unambiguous **FAIL**. The real robot's positive gripper delta command opened the jaws; the sim's equivalent (`+1`, closed per `SoarmGripper.format_action`) convention closes them. This is a genuine, human-confirmed (twice) axis/sign disagreement, not a units or observer-error issue.

This directly means the phase goal — "`wrist_roll` and gripper jaw joints restored with correct axes/**direction**" — is **not fully achieved**. All of the automated/structural work (Plans 10-01 through 10-04: calibration-derived joint limits, gripper clamp visual mesh fix, MJCF→URDF generation, and independent yourdfpy-based structural verification) is genuinely complete and independently re-verified in this report — the codebase evidence supports TWIN-01 through TWIN-06 as legitimately satisfied, not just claimed. But TWIN-07 is the requirement that specifically closes the loop against real hardware, and it fails.

**Secondary note (non-blocking, flagged for awareness):** TWIN-03's plain-English wording ("whose open/close direction matches the real gripper") textually overlaps with TWIN-07's substance. This report treats them as satisfying different, narrower technical claims per the roadmap's own decomposition (TWIN-03 = jaws are structurally mirrored/opposed to each other, correctly forming a pincer mechanism, which is independently true and verified; TWIN-07 = the resulting absolute polarity matches real hardware, which is independently false) — both plan-10-02's SUMMARY and this verification's own reasoning treat these as distinct, and REQUIREMENTS.md's existing TWIN-01..06 "Complete" / TWIN-07 "Pending" split already reflects the same call. Flagging this for human awareness since it is a judgment call, not something a grep can settle definitively — but it does not change the recommended action (a gap-closure plan targeting TWIN-07 is required either way, and if that plan also flips the gripper axis sign to fix the polarity, TWIN-03's underlying artifacts will be touched again and should be re-confirmed as part of the same gap-closure pass).

**Recommended next action:** `/gsd-plan-phase 10 --gaps` to generate a gap-closure plan. Per 10-05-SUMMARY.md's own diagnosis (and this report's independent confirmation that the codebase state matches that diagnosis exactly), the likely fix is: flip the `gripper_left`/`gripper_right` axis sign in `LIBERO/libero/libero/assets/grippers/soarm_gripper.xml`, regenerate `So-101/So-101.urdf` via `scripts/mjcf_to_urdf.py`, update `scripts/test_verify_urdf.py::test_gripper_joint_axes`'s hardcoded expected axis values to match, and re-run the TWIN-07 human-in-the-loop hardware comparison to confirm PASS before Phase 10 can be closed.

**Update (2026-09-19, gap-closure plan `260919-h8v`):** The code-side fix described above has been applied — `soarm_gripper.xml`'s gripper joint axis/range/ctrlrange flipped, `soarm_gripper.py`'s `init_qpos` updated, `So-101/So-101.urdf` regenerated, and `scripts/test_verify_urdf.py`/`scripts/verify_urdf.py` updated and passing (see `10-05-TWIN-07-VERIFICATION.md`'s "Gap-Closure Fix Applied" section for full detail). This is a sim-only, self-consistency-checked fix. **Phase completion remains blocked on a fresh human-in-the-loop hardware re-verification of TWIN-07** — this status/score/row is intentionally NOT flipped until that re-test produces an explicit PASS.

**Non-blocking follow-up (repo hygiene, not a phase-goal gap):** `git add` the untracked `So-101/` and `coppelia/` mesh asset directories (34 files) so the URDF's relative mesh references resolve on a fresh clone.

---

_Verified: 2026-09-19T12:15:00Z_
_Verifier: Claude (gsd-verifier)_
