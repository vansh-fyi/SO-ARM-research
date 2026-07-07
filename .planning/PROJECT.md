# SoARM VLA Research

## What This Is

A research project integrating the SoARM robot arm with Vision-Language-Action (VLA) models (π0/OpenVLA) inside a LIBERO/MuJoCo simulation environment, running on Google Colab for GPU access. The system takes natural language task prompts and executes them as simulated SOARM robot actions, building toward spatial scene awareness in robot manipulation.

## Core Value

A researcher types a task prompt and watches SOARM execute it in a LIBERO simulation — the loop from language to embodied action.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] SOARM MuJoCo/robosuite model (URDF/MJCF) integrated into LIBERO environment
- [ ] Google Colab notebook that loads π0/OpenVLA and runs inference on GPU
- [ ] End-to-end pipeline: text prompt → VLA → SOARM joint actions → rendered simulation output
- [ ] LIBERO task suite configured for SOARM (replacing default Panda arm)
- [ ] Dataset collection infrastructure: scripted/teleoperated SOARM demonstrations in LIBERO
- [ ] Spatial awareness: multi-camera views fed to VLA during task execution
- [ ] Spatial awareness: 3D scene understanding (object positions in space)
- [ ] Spatial awareness: spatial language grounding in prompts ("left of the box", "near the wall")
- [ ] Fine-tuning pipeline: collected SOARM demos used to fine-tune VLA on our robot
- [ ] Evaluation benchmark: task suite measuring spatial understanding quality

### Out of Scope

- Real-3DQA point cloud data as training input — inspiration only, not in this pipeline
- Physical SOARM hardware integration — simulation-only for now
- Real-time interactive REPL (deferred; start with rendered output)

## Context

- Existing repo has LIBERO embedded as a vendored submodule (`LIBERO/`) with MuJoCo/robosuite environments, Panda arm configs, and lifelong learning training code
- Real-3DQA exploration scripts exist (`explorations/real3dqa/`) as a parallel research thread — not directly connected to this pipeline
- LIBERO currently configured for Panda arm; SOARM has no existing MuJoCo description — needs to be built
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
*Last updated: 2026-07-07 after initialization*
