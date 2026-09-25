---
gsd_state_version: 1.0
milestone: v2.1
milestone_name: MLLM Raw-Autonomy Benchmark
current_phase: 12
current_phase_name: Bridge Tick-Latency Fix
status: planning
stopped_at: Phase 12 context gathered
last_updated: "2026-09-25T06:46:33.862Z"
last_activity: 2026-09-24
last_activity_desc: ROADMAP.md/REQUIREMENTS.md updated with v2.1 Phases 12-17 (26 requirements, 100% coverage)
progress:
  total_phases: 17
  completed_phases: 9
  total_plans: 38
  completed_plans: 38
  percent: 53
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-09-24)

**Core value:** A researcher types a task prompt and watches SOARM execute it in a LIBERO simulation — the loop from language to embodied action.
**Current focus:** Milestone v2.1 (MLLM Raw-Autonomy Benchmark) — roadmap created, ready to plan Phase 12. Scope: bridge tick-latency fix → camera device-resolution fix → two parallel-capable pipeline-architecture phases (VLA-style backends, MLLM-style backends) → cross-backend Pen Transfer benchmark → safety-cap re-tightening (deliberately last).

## Current Position

Phase: 12 of 17 (Bridge Tick-Latency Fix) — ready to plan
Plan: — of TBD in current phase
Status: Roadmap created, ready to plan
Last activity: 2026-09-24 — ROADMAP.md/REQUIREMENTS.md updated with v2.1 Phases 12-17 (26 requirements, 100% coverage)

Progress: [░░░░░░░░░░] 0%

## Performance Metrics

**Velocity:**

- Total plans completed: 24
- Average duration: -
- Total execution time: 0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 02 | 5 | - | - |
| 04 | 5 | - | - |
| 05 | 4 | - | - |
| 10 | 5 | - | - |
| 11 | 5 | - | - |

**Recent Trend:**

- Last 5 plans: none
- Trend: -

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- **v2.1 roadmap (2026-09-24):** Phases 12-17 derived from the 26 v2.1 requirements (LATENCY ×4, SAFETY ×1, CAMFIX ×3, PIPE ×2, VLAB ×4, MLLM ×7, EVAL ×2, DEBT ×3). Phase 12 (latency fix + 2 debt items) → Phase 13 (camera resolution unification) form a chain. Phases 14 (VLA-style pipeline + 4 backends) and 15 (MLLM-style pipeline + 3 backends) are **independent and parallel-capable** — different `ActionSource` pattern, different backend set — both gated on Phase 13's trustworthy camera pipeline, with Phase 15 additionally gated on Phase 12's real latency data for chunk sizing. Phase 16 (8-backend Pen Transfer comparison) depends on both 14 and 15. Phase 17 (safety-cap re-tightening + wrist_roll debt pin) is deliberately sequenced LAST — resequenced 2026-09-24 per explicit user decision to keep loosened soft-margin caps in place through the whole experiment so they don't mask genuine model behavior; hard limits (absolute joint clamping, NaN/inf rejection, e-stop) stay on unmodified throughout regardless. Full rationale in ROADMAP.md Overview.
- **Requirement-ID collision found during roadmap creation, fixed same session (2026-09-24):** v2.1's VLA Backends requirements (Phase 14) originally reused v1's `VLA-01..04` IDs (VLA Inference Pipeline, Phase 3) — unintentional authoring duplication, not a re-scoping. Renamed to `VLAB-01..04` in REQUIREMENTS.md/ROADMAP.md to remove the collision.
- **Phase 10 established model (2026-09-20):** User accepted the complete
  Coppelia-aligned assembly. Black housing, yellow jaws, `[0, 0.036]` joint and
  actuator ranges, `+1` opens / `-1` closes, mechanical jaw equality, ~88°
  wrist-flex initial pose. This supersedes earlier 84 mm aperture and visual-only
  correction assumptions. See [10-ESTABLISHED-MODEL.md](phases/10-digital-twin-fidelity/10-ESTABLISHED-MODEL.md).

- **v2.0 roadmap (2026-09-18):** Phases 10-11 derived from the 12 v2.0 requirements (TWIN-01..07, VLAHW-01..05). Phase 10 (Digital-Twin Fidelity) and Phase 11 (VLA Hardware Connection) are **independent and parallel-capable, not a sequential chain**.
- **v2.0 scope narrowing (2026-09-17):** An initial general-MLLM-prompting experiment (`experiment-design/`, `docs/multimodal-context-ablation-experiment.md`) failed badly. Milestone narrowed to two concrete workstreams (digital-twin fix + VLA hardware connection) before committing to the full raw-autonomy MLLM-benchmark-suite design.
- Research: OpenVLA-OFT chosen as primary VLA (97.1% LIBERO avg, T4-compatible at 4-bit)
- Research: SOARM MJCF to be derived from `so101_new_calib.xml`, not built from scratch
- 01-close: **REQUIRED READING for phases 2-6 before touching the Colab environment:** `.planning/phases/01-colab-environment-setup/01-DEBUG-HISTORY.md`
- [Phase 04]: GRIPPER UPGRADE (LOCKED, D-07): stock ~2-3cm jaw can't grasp any LIBERO object; adopted roboninecom 84mm parallel gripper — carries forward to Phase 10's real-gripper-direction success criterion (TWIN-03/07)
- [Phase 10 context]: `coppelia/export_model_library.py`'s own comments confirm the exported URDF's bug: the gripper is positioned near the wrist but never parented under the arm.

### Pending Todos

None yet for v2.1 — begin by planning Phase 12 (`/gsd-plan-phase 12`).

### Blockers/Concerns

- **v2.1 Phase 12 is a hard prerequisite for the rest of the milestone (13 → 14/15 → 16 → 17)** — no new live-hardware benchmark work should start until the tick-latency root cause (only 1/60 real actions in Phase 11's episode) is fixed and measured.
- 04 embodiment note (still binding for any new v2.1 task/object work): SO-ARM101 is a small ~500g-payload desktop arm; objects ≤84mm graspable, objects AND targets within ~0.45m reach, avoid the base's forward centerline collision corridor.
- 05 PRE-EXISTING TEST DEBT (unrelated to v2.1, still open): `test_replay.py::test_verify_full_obs_regeneration_passes_on_04_02_output` pixel-mismatch failure, not investigated.
- 10 RISK — RESOLVED (2026-09-24): [huggingface/lerobot#2210](https://github.com/huggingface/lerobot/issues/2210) did NOT reproduce in Phase 11's live episode.
- 11/v2.1 OPEN (root cause diagnosed, fix scoped to Phase 12, not yet implemented): Phase 11's live episode yielded only 1/60 (1.7%) real executed VLA actions because `control_loop_observation()` resends a full camera observation to Colab on every control tick instead of gating on queue-empty. See `control/vla_bridge/FINDINGS.md` §5 and `.planning/research/SUMMARY.md` for the full diagnosis and fix direction (wire up `lerobot`'s existing `_ready_to_send_observation()` gate).
- v2.1 camera bug (root cause diagnosed, fix scoped to Phase 13, not yet implemented): `camera_overhead` recordings in both `vla_episode_001` and the Phase 11 go/no-go episode are confirmed to be the laptop webcam, not the robot workspace — recording path never got the name-based camera-resolution fix already applied to the inference-input path.

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 260902-kcf | Update PROJECT.md: physical hardware integration is now an active parallel track (not out of scope) | 2026-09-02 | 1b4201c | [260902-kcf-update-project-md-physical-hardware-inte](./quick/260902-kcf-update-project-md-physical-hardware-inte/) |
| fast-260909 | Update progress on physical hardware build in docs (leader arm + servo swap complete) | 2026-09-09 | 9df5ecb | — |
| 260911-h5h | Create a 3D-printable STL mount bracket for the Waveshare AR0144 Stereo USB Camera module | 2026-09-11 | 47c8691 | [260911-h5h-create-a-3d-printable-stl-mount-bracket-](./quick/260911-h5h-create-a-3d-printable-stl-mount-bracket-/) |
| 260919-h8v | Fix Phase 10 TWIN-07 gap: flip gripper_left/gripper_right axis polarity in soarm_gripper.xml, regenerate URDF, update test expectations (hardware re-test still required) | 2026-09-19 | 5fa8e62 | [260919-h8v-fix-phase-10-twin-07-gap-flip-gripper-le](./quick/260919-h8v-fix-phase-10-twin-07-gap-flip-gripper-le/) |
| 260924-e3d | Loosen safety_validator.py caps (per-step 5→40deg, velocity 30→240deg/s, staleness 1s/3s→10s/30s) so the real 11-05 VLA episode can produce visible motion despite Colab round-trip latency -- user explicitly deprioritized safety margin in the empty test environment | 2026-09-24 | 9d664b7 | [260924-e3d-loosen-safety-validator-py-caps-displace](./quick/260924-e3d-loosen-safety-validator-py-caps-displace/) |
| 260924-gih | Fix pop_validated_action() silently dropping every real bridge action to `{}` -- lerobot's `.pos`-suffixed action_features keys never matched safety_validator's plain JOINT_ORDER names, crashing the first live 11-05 episode | 2026-09-24 | 41b6a0e | [260924-gih-fix-pop-validated-action-dropping-every-](./quick/260924-gih-fix-pop-validated-action-dropping-every-/) |

## Deferred Items

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| Physical Robot | SOARM hardware transfer (PHYS-01 to PHYS-03) | v2 deferred | Init |
| Advanced Spatial | Ego3D encoding, point cloud input (ADV-01 to ADV-03) | v2 deferred | Init |
| Interactive UI | Real-time REPL, web dashboard (INT-01, INT-02) | v2 deferred | Init |
| v1.1 Benchmark Work | Checkpoint Benchmark Suite + Re-Fine-Tuning (Phases 8-9) | Paused for v2.0 | 2026-09-15 |
| v2.1 Future Requirements | Provider-agnostic router beyond 3 backends, full 4-task suite, Recovery Rate automation | Deferred pending v2.1 Phases 12-17 results | 2026-09-24 |

## Session Continuity

**Resume file:** .planning/phases/12-bridge-tick-latency-fix/12-CONTEXT.md

Last session: 2026-09-25T06:46:33.850Z
Stopped at: Phase 12 context gathered

Phases 1-6 (milestone v1.0), Phase 7 (v1.1), and Phases 10-11 (v2.0) are all
complete. Phases 8-9 (v1.1) remain defined but PAUSED (not cancelled).

Next: user reviews/approves the v2.1 roadmap draft, then `/gsd-plan-phase 12`
to begin Phase 12 (Bridge Tick-Latency Fix) — research flags this as a
standard/mechanical pattern (skip a dedicated research sub-phase), root cause
already fully diagnosed in `control/vla_bridge/FINDINGS.md` §5.

NOTE (repo sync): the outer repo has repeatedly drifted commits-ahead of
`origin/main` without being pushed — before telling the user to `git pull` on
Colab, always check `git status -sb` for an "ahead" count first.
