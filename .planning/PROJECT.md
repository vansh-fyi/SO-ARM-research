# SoARM VLA Research

## What This Is

A research project integrating the SoARM robot arm with Vision-Language-Action (VLA) models (π0/OpenVLA) inside a LIBERO/MuJoCo simulation environment, running on Google Colab for GPU access. The system takes natural language task prompts and executes them as simulated SOARM robot actions, building toward spatial scene awareness in robot manipulation.

## Core Value

A researcher types a task prompt and watches SOARM execute it in a LIBERO simulation — the loop from language to embodied action.

## Requirements

### Validated

- [x] SOARM MuJoCo/robosuite model (MJCF) integrated into LIBERO environment — Validated in Phase 2: SOARM Robot Integration (Colab T4 sign-off 2026-07-18)
- [x] LIBERO task suite configured for SOARM (replacing default Panda arm) — Validated in Phase 2: 3 libero_spatial tasks run crash-free with `robots=["Soarm101"]`

### Active

- [ ] Google Colab notebook that loads π0/OpenVLA and runs inference on GPU
- [ ] End-to-end pipeline: text prompt → VLA → SOARM joint actions → rendered simulation output
- [ ] Dataset collection infrastructure: scripted/teleoperated SOARM demonstrations in LIBERO
- [ ] Spatial awareness: multi-camera views fed to VLA during task execution
- [ ] Spatial awareness: 3D scene understanding (object positions in space)
- [ ] Spatial awareness: spatial language grounding in prompts ("left of the box", "near the wall")
- [ ] Fine-tuning pipeline: collected SOARM demos used to fine-tune VLA on our robot
- [ ] Evaluation benchmark: task suite measuring spatial understanding quality

### Out of Scope

- Real-3DQA point cloud data as training input — inspiration only, not in this pipeline
- Physical SOARM hardware integration — simulation-first; real robot testing is a later milestone after sim pipeline is validated
- Real-time interactive REPL (deferred; start with rendered output)

## Context

- Existing repo has LIBERO cloned as a modifiable fork (`LIBERO/`) with its own git history — will be extended with custom SOARM models, tasks, and datasets; MuJoCo/robosuite environments, Panda arm configs, and lifelong learning training code already present
- Real-3DQA exploration scripts exist (`explorations/real3dqa/`) as a parallel research thread — not directly connected to this pipeline
- Phase 2 complete (2026-07-18): SOARM SO101 registered as `MountedSoarm101`/`SoarmGripper` in the LIBERO fork — vendored SO-ARM100 geometry, stable physics (0.024 N reset contact force), tuned eye_in_hand camera, 3 frozen libero_spatial tasks; verified on Colab T4
- Google Colab is the compute platform for GPU-accelerated VLA inference (π0 or OpenVLA are leading candidates — open-source, trained on robot manipulation data)
- Dataset serves dual purpose: fine-tuning the VLA on SOARM kinematics AND benchmarking spatial understanding
- Spatial awareness goal is three-layered: multi-camera perception, 3D object localization, and spatial language understanding

## Constraints

- **Compute**: Google Colab GPU budget (T4/A100 depending on tier) — inference and training must be Colab-compatible
- **Robot Model**: No existing SOARM URDF/MJCF — must derive or build from SOARM hardware specs
- **Framework**: LIBERO + robosuite stack (MuJoCo 2.3.7, robosuite 1.4.x) — must stay compatible
- **VLA Candidates**: π0 (Physical Intelligence) or OpenVLA — both open weights, robot-data trained
- **Timeline**: MVP full-pipeline sketch target within 2-3 weeks

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| π0 / OpenVLA as VLA backbone | Open-source, trained on robot manipulation data, closer to drop-in for LIBERO | — Pending |
| LIBERO as simulation framework | Already in repo, MuJoCo-based, has task suite infrastructure | — Pending |
| Google Colab for compute | GPU access without local hardware investment | — Pending |
| Simulation-only scope | Derisk by validating pipeline in sim before physical robot | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-07-18 after Phase 2 completion*
