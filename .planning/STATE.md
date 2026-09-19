---
gsd_state_version: 1.0
milestone: v2.0
milestone_name: Real-Hardware MLLM Manipulation Benchmark
current_phase: 10
current_phase_name: Digital-Twin Fidelity
status: executing
stopped_at: Phase 10 context gathered
last_updated: "2026-09-18T18:08:56.292Z"
last_activity: 2026-09-18
last_activity_desc: Phase 10 execution started
progress:
  total_phases: 11
  completed_phases: 7
  total_plans: 33
  completed_plans: 28
  percent: 64
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-09-17)

**Core value:** A researcher types a task prompt and watches SOARM execute it in a LIBERO simulation — the loop from language to embodied action.
**Current focus:** Phase 10 — Digital-Twin Fidelity

## Current Position

Phase: 10 (Digital-Twin Fidelity) — EXECUTING
Plan: 1 of 5
Status: Executing Phase 10
Last activity: 2026-09-19 - Completed quick task 260919-h8v: Fix Phase 10 TWIN-07 gap (gripper axis polarity flip)

## Performance Metrics

**Velocity:**

- Total plans completed: 14
- Average duration: -
- Total execution time: 0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 02 | 5 | - | - |
| 04 | 5 | - | - |
| 05 | 4 | - | - |

**Recent Trend:**

- Last 5 plans: none
- Trend: -

*Updated after each plan completion*
| Phase 02 P02 | 10min | 3 tasks | 6 files |
| Phase 02 P03 | 10min | 2 tasks | 0 files |
| Phase 04 P03 | 35min | 3 tasks | 3 files |
| Phase 04 P04 | 20min | 2 tasks | 3 files |
| Phase 06 P04 | 15min | 1 tasks | 1 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- **v2.0 roadmap (2026-09-18):** Phases 10-11 derived from the 12 v2.0 requirements (TWIN-01..07, VLAHW-01..05). Phase 10 (Digital-Twin Fidelity: URDF + MuJoCo XML kinematic rebuild) and Phase 11 (VLA Hardware Connection: real-hardware HF VLA + reasoning-trace harness) are **independent and parallel-capable, not a sequential chain** — Phase 11's real-hardware experiment runs entirely through `control/`'s existing LeRobot bridge and does not consume Phase 10's sim-twin outputs; Phase 10 only matters for future sim-side verification/re-training. Numbered sequentially (10 then 11) by roadmap convention only.
- **v2.0 scope narrowing (2026-09-17):** An initial general-MLLM-prompting experiment (`experiment-design/`, `docs/multimodal-context-ablation-experiment.md`) failed badly. Milestone narrowed to two concrete workstreams (digital-twin fix + VLA hardware connection) before committing to the full raw-autonomy MLLM-benchmark-suite design (moved to Future Requirements in REQUIREMENTS.md/PROJECT.md).
- **Known upstream risk for Phase 11 (flagged, not yet verified):** [huggingface/lerobot#2210](https://github.com/huggingface/lerobot/issues/2210) reports SmolVLA inference failures on SO-101 — verify early during Phase 11 planning/execution, don't assume it away; identify a fallback open-source HF VLA candidate if SmolVLA proves unworkable.
- **v1.1 roadmap (2026-08-30):** Phases 7-9 derived from the 13 v1.1 requirements (CAM-01, DEPTH-01/02/03, BENCH-01/02/03/04, OBJ-01/02, DATA-05, TUNE-05/06) as a strict sequential chain: Phase 7 (camera calibration + depth pipeline) lands first since task authoring and data collection both depend on the corrected perception stack; Phase 8 (object pool survey/authoring + checkpoint benchmark suite, including the low-risk BENCH-01 exclusion fix) lands before Phase 9 (demo collection + re-fine-tuning + re-evaluation), which needs the finished task suite and dataset. **Phases 8-9 PAUSED 2026-09-15** (v2.0 superseded active focus, not cancelled).
- Research: OpenVLA-OFT chosen as primary VLA (97.1% LIBERO avg, T4-compatible at 4-bit)
- Research: SOARM MJCF to be derived from `so101_new_calib.xml`, not built from scratch
- Research: Two separate Colab kernel groups needed (transformers version conflict between LIBERO training and VLA inference)
- Research: Demo replay must be state-based (not action replay) — LIBERO issue #16
- 01-close: **REQUIRED READING for phases 2-6 before touching the Colab environment:** `.planning/phases/01-colab-environment-setup/01-DEBUG-HISTORY.md` — the environment contract (7 invariants), full ENV-03 failure timeline, and debugging meta-lessons (restart ≠ clean slate, --no-deps contract, source enumeration over whack-a-mole)
- [Phase 02]: 02-03: Generic OSC_POSE config suffices for SOARM soak stability — no custom controller_configs kwarg needed
- [Phase 04]: GRIPPER UPGRADE (LOCKED, D-07): stock ~2-3cm jaw can't grasp any LIBERO object; adopted roboninecom 84mm parallel gripper (faithful, committed `ad0b0d0`) — carries forward to Phase 8/9 object authoring (≤84mm graspable-width constraint) AND to Phase 10's real-gripper-direction success criterion (TWIN-03/07)
- [Phase 04]: TASK RETARGETING (LOCKED, D-08): objects AND targets must stay within the arm's ~0.45m reach, and avoid the base's forward centerline collision corridor — binding constraint for Phase 8's new benchmark tasks/objects
- [Phase 05]: 05-04: OFTBackend needs `set_num_images_in_input(2)` + correct agentview/eye_in_hand channel order for genuine dual-camera consumption — relevant if Phase 7's depth stream is added as a third image-like input
- [Phase 06]: Phase 6 UAT found 3 of 4 eval tasks (RightOfX/NearTo/LeftOfX) pass trivially at spawn (100% before AND after fine-tuning) vs 0% on the one real (`On`-predicate) task — root cause of the v1.1 BENCH-01..04 requirements
- [Phase 10 context]: `coppelia/export_model_library.py`'s own comments confirm the exported URDF's bug: the reference scene keeps the arm and its gripper as two separate root-level objects — the gripper is positioned near the wrist but never parented under the arm. Rebuild must fix this parenting, not just re-export.

### Pending Todos

None yet.

### Blockers/Concerns

- 04 embodiment note for Phase 7/8/9/10: SO-ARM101 is a small ~500g-payload desktop arm; new tasks/objects must keep objects AND targets within ~0.45m reach, objects ≤84mm to be graspable, AND avoid placing objects on the base's forward centerline (y~0 close to the base) — the arm's own forearm sweeps through that corridor and will collide with anything sitting there.
- 05 PRE-EXISTING TEST DEBT (found 2026-08-10): (1) `test_hdf5_writer.py::test_schema_and_obs_key_naming`'s stale `gripper_states.shape[1] == 1` assertion — FIXED in Phase 7 plan 07-02 (now `== 2`, matching D-07's 84mm 2-DOF gripper). (2) `test_replay.py::test_verify_full_obs_regeneration_passes_on_04_02_output` still fails with a pixel mismatch in `(demo_1, 0, agentview_rgb)` — likely MuJoCo offscreen-render non-determinism, not investigated, still open.
- 05 LESSON — private-repo sync friction recurred: the outer repo has repeatedly drifted commits-ahead of `origin/main` without being pushed — Colab always clones/pulls from `origin/main`. Check `git status -sb` for an "ahead" count before telling the user to `git pull` on Colab, every session.
- 07 UNVERIFIED LOCALLY (Phase 7, plan 07-03, 2026-09-12): `rlds_converter.py`'s new legacy-HDF5 skip guard (pre-Phase-7 files lack `agentview_depth`) is implemented in `test_hdf5_to_rlds_against_real_dataset`, but that whole test is gated by `pytest.importorskip("tensorflow_datasets")` — TFDS isn't installed in the local `libero` conda env, so the test (and therefore the new guard) has never actually run, locally or otherwise. Spot-check on Colab (where TFDS is available) before relying on it, ideally before/during Phase 9 (still paused).
- 10 RISK (flagged 2026-09-18, unverified): [huggingface/lerobot#2210](https://github.com/huggingface/lerobot/issues/2210) reports SmolVLA inference failures on SO-101 — verify early in Phase 11, have a fallback open-source HF VLA candidate ready if it proves unworkable.
- 10 gap-closure REGRESSION RISK (flagged 2026-09-19, quick task `260919-h8v`): `LIBERO/libero/libero/datasets/collector.py` lines 81-82 hardcode `OPEN_CMD = -1.0` / `CLOSE_CMD = 1.0` for its Phase-4-era scripted grasp FSM, relying on the gripper's PRE-TWIN-07-fix action polarity. After quick task `260919-h8v` flipped `soarm_gripper.xml`'s joint axis/range polarity to match real hardware, these constants now drive the physical OPPOSITE of their names (`OPEN_CMD` closes, `CLOSE_CMD` opens). `collector.py` was NOT touched by `260919-h8v` (out of scope — Phase 8/9 demo re-collection that uses this FSM is currently paused). **Must swap these two constants before this FSM is used again for any future demo re-collection.**

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 260902-kcf | Update PROJECT.md: physical hardware integration is now an active parallel track (not out of scope) | 2026-09-02 | 1b4201c | [260902-kcf-update-project-md-physical-hardware-inte](./quick/260902-kcf-update-project-md-physical-hardware-inte/) |
| fast-260909 | Update progress on physical hardware build in docs (leader arm + servo swap complete) | 2026-09-09 | 9df5ecb | — |
| 260911-h5h | Create a 3D-printable STL mount bracket for the Waveshare AR0144 Stereo USB Camera module | 2026-09-11 | 47c8691 | [260911-h5h-create-a-3d-printable-stl-mount-bracket-](./quick/260911-h5h-create-a-3d-printable-stl-mount-bracket-/) |
| 260919-h8v | Fix Phase 10 TWIN-07 gap: flip gripper_left/gripper_right axis polarity in soarm_gripper.xml, regenerate URDF, update test expectations (hardware re-test still required) | 2026-09-19 | 5fa8e62 | [260919-h8v-fix-phase-10-twin-07-gap-flip-gripper-le](./quick/260919-h8v-fix-phase-10-twin-07-gap-flip-gripper-le/) |

## Deferred Items

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| Physical Robot | SOARM hardware transfer (PHYS-01 to PHYS-03) | v2 deferred | Init |
| Advanced Spatial | Ego3D encoding, point cloud input (ADV-01 to ADV-03) | v2 deferred | Init |
| Interactive UI | Real-time REPL, web dashboard (INT-01, INT-02) | v2 deferred | Init |
| v1.1 Benchmark Work | Checkpoint Benchmark Suite + Re-Fine-Tuning (Phases 8-9: BENCH-*, OBJ-*, DATA-05, TUNE-05/06) | Paused for v2.0 | 2026-09-15 |
| v2.0 Future Requirements | Deep-reasoning MLLM comparison, raw-autonomy design, provider router, full 4-task suite, failure taxonomy | Deferred pending v2.0 Phases 10-11 results | 2026-09-17 |

## Session Continuity

**Resume file:** .planning/phases/10-digital-twin-fidelity/10-CONTEXT.md

Last session: 2026-09-18T13:00:17.312Z
Stopped at: Phase 10 context gathered

Phases 1-6 (milestone v1.0) are all complete — their earlier resume-sequence notes
are historical, not active blockers.

Phase 7 (milestone v1.1) is complete. Phases 8-9 (v1.1) remain defined in
ROADMAP.md but PAUSED as of 2026-09-15 — not the current priority, not cancelled.

NOTE: 2 pre-existing test failures from Phase 4 (gripper_states.shape assertion,
a replay-obs pixel-mismatch test) remain open — see Blockers/Concerns above.

NOTE (repo sync): the outer repo has repeatedly drifted commits-ahead of
`origin/main` without being pushed — before telling the user to `git pull` on
Colab, always check `git status -sb` for an "ahead" count first.

Next: run `/gsd-plan-phase 10` and/or `/gsd-plan-phase 11` (independent,
parallel-capable — either order, or both, is fine; see ROADMAP.md Overview).
