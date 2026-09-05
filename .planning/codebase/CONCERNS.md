# Codebase Concerns

**Analysis Date:** 2026-09-05

## Tech Debt

**Stale HDF5 gripper-shape test assertion:**
- Issue: `test_schema_and_obs_key_naming` in `LIBERO/libero/libero/datasets/test_hdf5_writer.py` asserts `gripper_states.shape[1] == 1`, but this was never updated after the D-07 gripper upgrade (stock ~2-3cm jaw → roboninecom 84mm 2-DOF parallel gripper). Actual shape is now 2.
- Files: `LIBERO/libero/libero/datasets/test_hdf5_writer.py`
- Impact: Test fails/misleads on every run touching HDF5 writing; masks real regressions in obs-key schema validation.
- Fix approach: Update assertion to `== 2` and add a comment documenting the D-07 gripper DOF change. Logged in `.planning/STATE.md` as candidate cleanup for Phase 7 (touches same HDF5/camera code).

**Replay obs-regeneration pixel mismatch (undiagnosed):**
- Issue: `test_verify_full_obs_regeneration_passes_on_04_02_output` in `LIBERO/libero/libero/datasets/test_replay.py` fails with a pixel mismatch in `(demo_1, 0, agentview_rgb)`. Suspected MuJoCo offscreen-render non-determinism, but not investigated.
- Files: `LIBERO/libero/libero/datasets/test_replay.py`
- Impact: Unknown whether replay-based obs regeneration is trustworthy for demo re-collection (Phase 9 depends on this).
- Fix approach: Root-cause whether the mismatch is renderer non-determinism (seed/GL context) vs. a real state-replay bug; both open since 2026-08-10 per `.planning/STATE.md`.

**Ultralytics installed but unused:**
- Issue: `ultralytics==8.4.138` is pinned in `control/requirements.txt` "for the planned YOLO object-follow feature (adapting XLeRobot's `3_so100_yolo_ee_follow.py`) - not yet wired into any script."
- Files: `control/requirements.txt`
- Impact: Unused heavyweight dependency inflates install size/time in the `control/` venv; risk of silent version drift before the feature is actually built.
- Fix approach: Either build the YOLO follow script soon or move the pin to a comment-only TODO until implementation starts.

**No shared library across exploration/control/diagnostics scripts:**
- Issue: `explorations/`, `control/`, and `diagnostics/` each resolve paths, servo protocol constants, and MuJoCo env setup independently with no shared module.
- Files: `explorations/create_scene.py`, `explorations/soarm_sanity.py`, `diagnostics/servo_scan.py`, `diagnostics/servo_move_test.py`, `diagnostics/servo_set_id.py`, `diagnostics/servo_set_torque_limit.py`, `diagnostics/servo_torque.py`, `diagnostics/servo_drive_to_stall.py`, `diagnostics/servo_set_protection.py`
- Impact: Protocol constants (e.g. `STS_PROTOCOL_END`, `ADDR_PRESENT_POSITION`, baud tables) are duplicated across diagnostics scripts; a bug fix in one script's servo addressing must be manually propagated to the others.
- Fix approach: Extract a small `diagnostics/servo_common.py` (or similar) for shared Feetech STS3215 protocol constants and port-handling helpers.

**Repo-ahead-of-origin drift (recurring, cross-session):**
- Issue: The outer repo has repeatedly drifted commits-ahead of `origin/main` without being pushed, and Colab always clones/pulls from `origin/main` — meaning Colab work can silently run against stale code.
- Files: n/a (repo-level git workflow issue)
- Impact: Confusing failures during Colab runs when local fixes were never pushed; wasted debugging time diagnosing "bugs" that were already fixed locally.
- Fix approach: Check `git status -sb` for an "ahead" count before telling the user to `git pull` on Colab, every session (documented as a standing lesson in `.planning/STATE.md`).

## Known Bugs

**Trivial pass-at-spawn on 3 of 4 Phase 6 eval tasks:**
- Symptoms: `RightOfX`, `NearTo`, and `LeftOfX` eval tasks pass at 100% both before AND after fine-tuning — they are satisfied by the object's spawn position without requiring any real manipulation. Only the one real `On`-predicate task showed genuine 0%→improvement signal.
- Files: LIBERO benchmark task/predicate definitions consumed via `LIBERO/libero/libero/bddl_files/` and Phase 6 eval outputs (`06b_eval_videos/`)
- Trigger: Run the Phase 6 checkpoint benchmark suite; spatial-predicate tasks are satisfied trivially by initial object placement.
- Workaround: None yet — root cause of the v1.1 BENCH-01..04 requirements (Phase 8 is chartered to fix task/object placement so predicates are not trivially satisfied at spawn).

## Security Considerations

**No secrets detected in tracked files:**
- Risk: None identified — no `.env`, credentials, or hardcoded API keys found in `explorations/`, `control/`, `diagnostics/`, or `LIBERO/` during this audit.
- Files: n/a
- Current mitigation: `explorations/data/` and other data directories are gitignored.
- Recommendations: Continue excluding `HF_ADAPTER_REPO_ID` and any Colab secrets from being committed; verify `.gitignore` covers `.venv/` directories under `control/` and `diagnostics/` (both contain full virtualenvs with `site-packages` — confirm these are not accidentally tracked).

## Performance Bottlenecks

**No significant bottlenecks identified in this pass.**
- The codebase is research/exploration-scale (single-episode rendering, single-arm servo control); no high-throughput or latency-critical paths were found that warrant explicit performance concern documentation at this time.

## Fragile Areas

**LIBERO import path setup:**
- Files: `explorations/create_scene.py`, `explorations/soarm_sanity.py`
- Why fragile: Both scripts manually insert `explorations/LIBERO` into `sys.path` before importing `libero.*`, and both hardcode `MUJOCO_GL=glfw` inline for macOS headless rendering (documented as an architectural constraint — `osmesa` is Linux-only and would silently fail on macOS if swapped in). Any new script importing LIBERO must replicate this exact pattern or fail with unclear import errors.
- Safe modification: Any new LIBERO-consuming script must set `MUJOCO_GL` and extend `sys.path` before the first `import libero` statement; do not assume it's set globally.
- Test coverage: No test verifies this setup pattern; failures surface only at runtime with cryptic MuJoCo/GL errors.

**`~/.libero/config.yaml` first-import interactive prompt:**
- Files: LIBERO package import surface (`LIBERO/libero/`)
- Why fragile: The first import of LIBERO reads `~/.libero/config.yaml` from the user's home directory; if absent, the package prompts interactively, causing `EOFError` in non-interactive contexts (Colab, CI, scripted runs).
- Safe modification: Any automation (Colab notebook cells, CI, scripted diagnostics) must ensure `~/.libero/config.yaml` exists before the first LIBERO import.
- Test coverage: None — this is a known operational gotcha, not test-covered.

**Physical hardware scripts have no automated tests:**
- Files: `diagnostics/servo_scan.py`, `diagnostics/servo_move_test.py`, `diagnostics/servo_drive_to_stall.py`, `diagnostics/servo_set_protection.py`, `diagnostics/servo_set_id.py`, `diagnostics/servo_set_torque_limit.py`, `diagnostics/servo_torque.py`, `diagnostics/camera_test.py`, `control/keyboard_joint_control.py`, `control/record_episode.py`, `control/joint_jog.py`
- Why fragile: These scripts directly drive physical Feetech STS3215 servos over serial (torque limits, ID assignment, stall-drive protection tuning) — incorrect protocol constants or baud handling can physically damage hardware (e.g. `servo_drive_to_stall.py`, `servo_set_protection.py`). There is no simulation/mocking layer; verification is manual (`diagnostics/UAT/`).
- Safe modification: Changes to servo addressing constants (`ADDR_PRESENT_POSITION`, protocol end byte, baud tables) must be manually cross-checked against the Feetech STS3215 control table before running against real hardware; test on a single servo ID first via `--ids` scoping.
- Test coverage: `diagnostics/UAT/` contains manual UAT procedures, not automated tests — no CI or simulated hardware-in-the-loop coverage exists.

## Scaling Limits

**Not applicable at current project scale.**
- This is a single-researcher, Colab-GPU-budget project (SO-ARM101 desktop arm, single-arm single-camera setup). No scaling concerns were identified — the relevant constraints are compute-budget and physical-embodiment limits (see Missing Critical Features / embodiment note below), not software scaling.

## Dependencies at Risk

**Loose/unpinned dependency versions in `explorations/requirements.txt`:**
- Risk: `explorations/requirements.txt` uses loose lower-bound pins (`>=1.11.0`, `>=1.21.0`, `>=1.4.0`, etc.) with no lockfile, while `LIBERO/requirements.txt` and `control/requirements.txt` use exact pins.
- Impact: Exploration scripts can silently pick up breaking upstream releases (numpy, torch, transformers) between runs, especially problematic given the project's own documented note that transformers version conflicts exist between LIBERO training and VLA inference (hence "two separate Colab kernel groups needed").
- Migration plan: Pin `explorations/requirements.txt` to exact versions matching what's validated on Colab, mirroring the `control/requirements.txt` pinning pattern already adopted (`chore(control): pin requirements.txt to actual installed versions`).

**No SOARM URDF/MJCF from vendor — derived model:**
- Risk: The SOARM MJCF used in simulation is derived from `so101_new_calib.xml` rather than an authoritative vendor model (documented research decision), since no official SOARM URDF/MJCF exists.
- Impact: Any physical/simulated mismatch (mass, joint limits, gripper geometry) traces back to this derived model, not a vendor spec — compounds with the D-07 gripper upgrade (roboninecom 84mm parallel gripper) which required custom MJCF modeling (see MEMORY.md `soarm-gripper-mjcf-modeling-notes`).
- Migration plan: None planned; this is an accepted project constraint, not a bug — flagged here for future debugging context when sim-to-real discrepancies appear.

## Missing Critical Features

**Physical hardware transfer deferred to v2, but now an active parallel track:**
- Problem: `PHYS-01` to `PHYS-03` (SOARM hardware transfer) were originally deferred to v2, but per the most recent quick task (`260902-kcf`), physical hardware integration is now an active parallel track — yet `.planning/STATE.md`'s "Deferred Items" table still lists it as "v2 deferred."
- Blocks: Risk of stale planning docs causing confusion about whether physical-hardware work (`control/`, `diagnostics/`) is in-scope for the current milestone.
- Files: `.planning/STATE.md` (Deferred Items table), `.planning/PROJECT.md`

**Embodiment constraint must be respected by all new Phase 7/8/9 tasks:**
- Problem: SO-ARM101 is a small ~500g-payload desktop arm. New tasks/objects must keep objects AND targets within ~0.45m reach, objects ≤84mm graspable width (post D-07 gripper upgrade), and avoid the base's forward centerline collision corridor (the forearm sweeps through y≈0 near the base).
- Blocks: Any new BDDL task/object authored in Phase 8 without respecting these constraints will produce ungraspable or arm-colliding tasks — this is the binding constraint carried forward from Phase 4.
- Files: LIBERO BDDL task files under `LIBERO/libero/libero/bddl_files/`, referenced in `.planning/STATE.md` Blockers/Concerns.

## Test Coverage Gaps

**No automated tests for `control/` and `diagnostics/` scripts:**
- What's not tested: `control/keyboard_joint_control.py`, `control/record_episode.py`, `control/joint_jog.py`, and all `diagnostics/servo_*.py` / `diagnostics/camera_test.py` scripts — all physical-hardware-facing.
- Files: `control/*.py`, `diagnostics/*.py`
- Risk: Regressions in servo protocol handling, torque limits, or recording logic could go unnoticed until run against physical hardware, with potential for hardware damage (torque/stall scripts) or silent data-quality loss (episode recording).
- Priority: Medium — mitigated by manual UAT procedures in `diagnostics/UAT/`, but no regression safety net exists for future refactors.

**Two pre-existing open test failures in LIBERO dataset layer:**
- What's not tested/passing: `test_hdf5_writer.py::test_schema_and_obs_key_naming` (stale gripper shape assertion) and `test_replay.py::test_verify_full_obs_regeneration_passes_on_04_02_output` (pixel mismatch, undiagnosed) — see Tech Debt above.
- Files: `LIBERO/libero/libero/datasets/test_hdf5_writer.py`, `LIBERO/libero/libero/datasets/test_replay.py`
- Risk: Both tests touch the exact data pipeline (HDF5 writing, obs replay) that Phase 7 (camera & depth perception) and Phase 9 (demo re-collection) depend on — undiagnosed failures here could mask real regressions introduced by upcoming camera/depth pipeline changes.
- Priority: High — explicitly flagged in `.planning/STATE.md` as candidate cleanup for Phase 7 since it touches the same HDF5/camera-rendering code path.

---

*Concerns audit: 2026-09-05*
