---
gsd_state_version: 1.0
milestone: v2.0
milestone_name: Real-Hardware MLLM Manipulation Benchmark
status: planning
last_updated: "2026-09-15T09:40:08.919Z"
last_activity: 2026-09-15
progress:
  total_phases: 0
  completed_phases: 0
  total_plans: 0
  completed_plans: 0
  percent: 0
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-08-30)

**Core value:** A researcher types a task prompt and watches SOARM execute it in a LIBERO simulation — the loop from language to embodied action.
**Current focus:** Phase 07 — Camera & Depth Perception

## Current Position

Phase: Not started (defining requirements)
Plan: —
Status: Defining requirements
Last activity: 2026-09-15 — Milestone v2.0 started

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

- **v1.1 roadmap (2026-08-30):** Phases 7-9 derived from the 13 v1.1 requirements (CAM-01, DEPTH-01/02/03, BENCH-01/02/03/04, OBJ-01/02, DATA-05, TUNE-05/06) as a strict sequential chain: Phase 7 (camera calibration + depth pipeline) lands first since task authoring and data collection both depend on the corrected perception stack; Phase 8 (object pool survey/authoring + checkpoint benchmark suite, including the low-risk BENCH-01 exclusion fix) lands before Phase 9 (demo collection + re-fine-tuning + re-evaluation), which needs the finished task suite and dataset.
- Research: OpenVLA-OFT chosen as primary VLA (97.1% LIBERO avg, T4-compatible at 4-bit)
- Research: SOARM MJCF to be derived from `so101_new_calib.xml`, not built from scratch
- Research: Two separate Colab kernel groups needed (transformers version conflict between LIBERO training and VLA inference)
- Research: Demo replay must be state-based (not action replay) — LIBERO issue #16
- 01-close: **REQUIRED READING for phases 2-6 before touching the Colab environment:** `.planning/phases/01-colab-environment-setup/01-DEBUG-HISTORY.md` — the environment contract (7 invariants), full ENV-03 failure timeline, and debugging meta-lessons (restart ≠ clean slate, --no-deps contract, source enumeration over whack-a-mole)
- [Phase 02]: 02-03: Generic OSC_POSE config suffices for SOARM soak stability — no custom controller_configs kwarg needed
- [Phase 04]: GRIPPER UPGRADE (LOCKED, D-07): stock ~2-3cm jaw can't grasp any LIBERO object; adopted roboninecom 84mm parallel gripper (faithful, committed `ad0b0d0`) — carries forward to Phase 8/9 object authoring (≤84mm graspable-width constraint)
- [Phase 04]: TASK RETARGETING (LOCKED, D-08): objects AND targets must stay within the arm's ~0.45m reach, and avoid the base's forward centerline collision corridor — binding constraint for Phase 8's new benchmark tasks/objects
- [Phase 05]: 05-04: OFTBackend needs `set_num_images_in_input(2)` + correct agentview/eye_in_hand channel order for genuine dual-camera consumption — relevant if Phase 7's depth stream is added as a third image-like input
- [Phase 06]: Phase 6 UAT found 3 of 4 eval tasks (RightOfX/NearTo/LeftOfX) pass trivially at spawn (100% before AND after fine-tuning) vs 0% on the one real (`On`-predicate) task — root cause of the v1.1 BENCH-01..04 requirements

### Pending Todos

None yet.

### Blockers/Concerns

- 04 embodiment note for Phase 7/8/9: SO-ARM101 is a small ~500g-payload desktop arm; new tasks/objects must keep objects AND targets within ~0.45m reach, objects ≤84mm to be graspable, AND avoid placing objects on the base's forward centerline (y~0 close to the base) — the arm's own forearm sweeps through that corridor and will collide with anything sitting there.
- 05 PRE-EXISTING TEST DEBT (found 2026-08-10): (1) `test_hdf5_writer.py::test_schema_and_obs_key_naming`'s stale `gripper_states.shape[1] == 1` assertion — FIXED in Phase 7 plan 07-02 (now `== 2`, matching D-07's 84mm 2-DOF gripper). (2) `test_replay.py::test_verify_full_obs_regeneration_passes_on_04_02_output` still fails with a pixel mismatch in `(demo_1, 0, agentview_rgb)` — likely MuJoCo offscreen-render non-determinism, not investigated, still open.
- 05 LESSON — private-repo sync friction recurred: the outer repo has repeatedly drifted commits-ahead of `origin/main` without being pushed — Colab always clones/pulls from `origin/main`. Check `git status -sb` for an "ahead" count before telling the user to `git pull` on Colab, every session.
- 07 UNVERIFIED LOCALLY (Phase 7, plan 07-03, 2026-09-12): `rlds_converter.py`'s new legacy-HDF5 skip guard (pre-Phase-7 files lack `agentview_depth`) is implemented in `test_hdf5_to_rlds_against_real_dataset`, but that whole test is gated by `pytest.importorskip("tensorflow_datasets")` — TFDS isn't installed in the local `libero` conda env, so the test (and therefore the new guard) has never actually run, locally or otherwise. Spot-check on Colab (where TFDS is available) before relying on it, ideally before/during Phase 9.

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 260902-kcf | Update PROJECT.md: physical hardware integration is now an active parallel track (not out of scope) | 2026-09-02 | 1b4201c | [260902-kcf-update-project-md-physical-hardware-inte](./quick/260902-kcf-update-project-md-physical-hardware-inte/) |
| fast-260909 | Update progress on physical hardware build in docs (leader arm + servo swap complete) | 2026-09-09 | 9df5ecb | — |
| 260911-h5h | Create a 3D-printable STL mount bracket for the Waveshare AR0144 Stereo USB Camera module | 2026-09-11 | 47c8691 | [260911-h5h-create-a-3d-printable-stl-mount-bracket-](./quick/260911-h5h-create-a-3d-printable-stl-mount-bracket-/) |

## Deferred Items

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| Physical Robot | SOARM hardware transfer (PHYS-01 to PHYS-03) | v2 deferred | Init |
| Advanced Spatial | Ego3D encoding, point cloud input (ADV-01 to ADV-03) | v2 deferred | Init |
| Interactive UI | Real-time REPL, web dashboard (INT-01, INT-02) | v2 deferred | Init |

## Session Continuity

Last session: 2026-09-05T16:06:49.050Z
Stopped at: Phase 7 context gathered

Phases 1-6 (milestone v1.0) are all complete — their earlier resume-sequence notes
are historical, not active blockers.

NOTE: 2 pre-existing test failures from Phase 4 (gripper_states.shape assertion,
a replay-obs pixel-mismatch test) remain open — see Blockers/Concerns above,
candidate cleanup during Phase 7 since it touches the same HDF5/camera code.

NOTE (repo sync): the outer repo has repeatedly drifted commits-ahead of
`origin/main` without being pushed — before telling the user to `git pull` on
Colab, always check `git status -sb` for an "ahead" count first.

Resume file: .planning/phases/07-camera-depth-perception/07-CONTEXT.md
