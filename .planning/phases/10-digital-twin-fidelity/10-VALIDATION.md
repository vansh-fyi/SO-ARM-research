---
phase: 10
slug: digital-twin-fidelity
status: established_model_accepted
current_model_full_suite: pending
nyquist_compliant: true
wave_0_complete: false
created: 2026-09-18
---

> **2026-09-20 established-model update:** [10-ESTABLISHED-MODEL.md](10-ESTABLISHED-MODEL.md)
> is authoritative for the accepted Coppelia-aligned assembly, black housing,
> yellow jaws, 0.036 m per-jaw cap, corrected starting pose and LIBERO wiring.
> Earlier settings and verification results below are historical and do not
> establish policy success for the accepted model.


# Phase 10 — Validation Strategy

## Current accepted-model checks (2026-09-20)

The accepted model passes the Coppelia geometry/coupling tests, URDF tests and
LIBERO RGB/depth reset/render integration. Run:

`python -m pytest diagnostics/test_coppelia_alignment.py scripts/test_verify_urdf.py LIBERO/libero/libero/envs/test_camera_config.py -q`

Use `python diagnostics/verify_coppelia_alignment.py` for the 132 numerical
geometry comparisons. User visual acceptance is complete, including the final
36 mm cap. The original planning checklist below is historical. Its full-suite
and collection-success gate has not been re-established for the new geometry.

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (no `pytest.ini`/`conftest.py` at repo or `LIBERO/` root — relies on pytest's "prepend" import-mode auto-discovery, per `test_camera_config.py`'s own module docstring) |
| **Config file** | none — Wave 0 adds `scripts/test_verify_urdf.py` |
| **Quick run command** | `pytest LIBERO/libero/libero/envs/test_camera_config.py -x` (run with `conda activate libero`) |
| **Full suite command** | `pytest LIBERO/libero/libero/envs/ LIBERO/libero/libero/datasets/ -x` |
| **Estimated runtime** | ~30-60s (no GPU needed, matches established Phase 7 pattern) |

---

## Sampling Rate

- **After every task commit:** Run `pytest LIBERO/libero/libero/envs/test_camera_config.py -x`
- **After every plan wave:** Run `pytest LIBERO/libero/libero/envs/ LIBERO/libero/libero/datasets/ -x`
- **Before `/gsd-verify-work`:** Full suite must be green, plus the D-07 human-in-the-loop gripper-direction checkpoint completed and its outcome documented
- **Max feedback latency:** ~60 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 10-01-01 | 01 | 0 | TWIN-01..05 | — | N/A | unit (new) | `pytest scripts/test_verify_urdf.py -x` | ❌ W0 | ⬜ pending |
| 10-0X-0X | TBD | TBD | TWIN-01 | — | Generated URDF loads via `yourdfpy`; gripper is a child of the wrist link, not floating off `robot_base` | unit (new) | `pytest scripts/test_verify_urdf.py::test_gripper_is_child_of_wrist -x` | ❌ W0 | ⬜ pending |
| 10-0X-0X | TBD | TBD | TWIN-02 | — | URDF's `wrist_roll` joint range matches `robot.xml`'s calibrated-servo-limit value | unit (new) | `pytest scripts/test_verify_urdf.py::test_wrist_roll_range -x` | ❌ W0 | ⬜ pending |
| 10-0X-0X | TBD | TBD | TWIN-03 | — | URDF's `gripper_left`/`gripper_right` are prismatic with axes matching `soarm_gripper.xml`'s open/close direction | unit (new) | `pytest scripts/test_verify_urdf.py::test_gripper_joint_axes -x` | ❌ W0 | ⬜ pending |
| 10-0X-0X | TBD | TBD | TWIN-04 | — | No `file://` or absolute-path mesh references anywhere in the generated URDF | unit (new) | `pytest scripts/test_verify_urdf.py::test_no_absolute_mesh_paths -x` | ❌ W0 | ⬜ pending |
| 10-0X-0X | TBD | TBD | TWIN-05 | — | Derived-from-calibration joint limits (5 of 6) match a documented, reproducible formula; `wrist_roll` explicitly excepted | unit (new) | `pytest scripts/test_verify_urdf.py::test_joint_limits_match_calibration -x` | ❌ W0 | ⬜ pending |
| 10-0X-0X | TBD | TBD | TWIN-06 | — | `robot.xml`/`soarm_gripper.xml` edits (if any, from TWIN-05's cross-check) don't break `env.reset()` | integration (existing) | `pytest LIBERO/libero/libero/envs/test_camera_config.py -x` | ✅ already exists | ⬜ pending |
| 10-0X-0X | TBD | TBD | TWIN-07 | — | Sim gripper direction matches real gripper direction under the same command | manual-only (human-in-the-loop, D-07) | N/A — documented checkpoint, not automatable | N/A by design | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*
*Exact task IDs are assigned once the planner writes PLAN.md files; this table's requirement→test mapping is authoritative regardless of final task numbering.*

---

## Wave 0 Requirements

- [ ] `scripts/test_verify_urdf.py` — new test file covering TWIN-01 through TWIN-05 assertions against the generated URDF (via `yourdfpy`)
- [ ] `yourdfpy` install into the `libero` conda env (`pip install yourdfpy`)
- [ ] No `conftest.py` exists for `scripts/` — if the new test file needs repo-root path setup, mirror `test_camera_config.py`'s existing inline `sys.path` pattern rather than adding a new `conftest.py` (keeps consistency with the established no-conftest convention)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Sim gripper opens/closes in the same direction as the real gripper under an identical command | TWIN-07 | Requires hardware-in-the-loop comparison against the physical SO-ARM101 follower arm; Phase 10 is otherwise sim-side-only (per CONTEXT.md D-07/D-08) | Send a known gripper command to the real robot via `control/jog_gripper_raw.py` (or equivalent), drive the sim gripper with the same LeRobot-native command value translated into the sim's `-1`/`+1` convention, visually confirm both open/close the same way. Document the outcome (pass/fail, command value used) in the phase SUMMARY or VERIFICATION.md. |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 60s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
