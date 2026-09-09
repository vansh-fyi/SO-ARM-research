# SoARM VLA Research

## What This Is

A research project integrating the SoARM robot arm with Vision-Language-Action (VLA) models (π0/OpenVLA) inside a LIBERO/MuJoCo simulation environment, running on Google Colab for GPU access. The system takes natural language task prompts and executes them as simulated SOARM robot actions, building toward spatial scene awareness in robot manipulation.

## Core Value

A researcher types a task prompt and watches SOARM execute it in a LIBERO simulation — the loop from language to embodied action.

## Requirements

### Validated

- [X] SOARM MuJoCo/robosuite model (MJCF) integrated into LIBERO environment — Validated in Phase 2: SOARM Robot Integration (Colab T4 sign-off 2026-07-18)
- [X] LIBERO task suite configured for SOARM (replacing default Panda arm) — Validated in Phase 2: 3 libero_spatial tasks run crash-free with `robots=["Soarm101"]`
- [X] Google Colab notebook that loads π0/OpenVLA and runs inference on GPU — Validated in Phase 3: VLA Inference Loop (Colab A100/L4 sign-off, OFT 2026-07-19, π0 2026-08-02)
- [X] End-to-end pipeline: text prompt → VLA → SOARM joint actions → rendered simulation output — Validated in Phase 3: both OFT and π0 backends drive the shared eval_loop/run_suite unmodified, producing per-episode video and success-rate tables
- [X] Dataset collection infrastructure: scripted/teleoperated SOARM demonstrations in LIBERO — Validated in Phase 4: Dataset Collection (120 demos, `put_the_cream_cheese_in_the_bowl`, 04-02 scripted + 04-05 keyboard teleop, HDF5 schema-verified via replay round-trip)
- [X] Spatial awareness: multi-camera views fed to VLA during task execution — Validated in Phase 5 (SPAT-02): both OFTBackend and Pi0Backend consume genuinely distinct agentview + eye_in_hand views via each checkpoint's native dual-image API; confirmed live on Colab GPU (05-04 gap fix + retest, 2026-08-17)
- [X] Spatial awareness: 3D scene understanding (object positions in space) — Validated in Phase 5 (SPAT-01/03/04): `depth_xyz.py`'s camera-derived back-projection pipeline matches MuJoCo ground truth within ~0.02m using real Colab (Linux egl) instance-segmentation rendering
- [X] Spatial awareness: spatial language grounding in prompts ("left of the box", "near the wall") — Validated in Phase 5 (SPAT-05): 3 new BDDL spatial-relation tasks + binary predicate classes, local test suite green
- [X] Fine-tuning pipeline: collected SOARM demos used to fine-tune VLA on our robot — Validated in Phase 6: OpenVLA-OFT LoRA (r=32) fine-tuned on the 120-demo dataset via RLDS conversion + OXE registration, checkpoint pushed to HF Hub, training curves in WandB
- [X] Evaluation benchmark: task suite measuring spatial understanding quality — Validated in Phase 6, but with a known design flaw: 3 of 4 eval tasks use spatial-relation predicates (RightOfX/NearTo/LeftOfX) satisfied at object spawn, passing at step 1 regardless of policy quality; only `put_the_cream_cheese_in_the_bowl` (an `On` predicate) requires genuine manipulation. This gap is what v1.1 addresses.

## Current Milestone: v1.1 Perception Fidelity & Checkpoint Benchmark

**Goal:** Fix the sim-to-real camera gap, add depth perception, and replace the flawed spatial-task benchmark with a checkpoint-scored task suite — then re-collect demonstrations and re-fine-tune on it.

**Target features:**
- Camera calibration: recalibrate sim agentview/front camera to match real SOARM camera placement/FOV
- Depth camera: new depth observation stream plumbed through robosuite/LIBERO obs → HDF5 writer → RLDS converter
- Retire the 3 spawn-trivial spatial tasks from success metrics (RightOfX/NearTo/LeftOfX predicates satisfied at t=0)
- New benchmark suite(s) mirroring LIBERO's spatial/object/goal category structure, sized ~8-15 tasks
- Each task scored via checkpoint/sub-goal predicates (e.g. reach→grasp→lift→place, ~4 steps) instead of single binary success
- Object pool: survey existing LIBERO objects fitting SOARM constraints (≤84mm, in-reach) + author a few new custom objects via `custom_object_example.ipynb`
- Evaluation: success rate + generalization splits (seen/unseen positions or instructions), same 20-episode/task cadence
- Full loop: author tasks → collect demos → re-run LoRA fine-tuning → re-evaluate on expanded suite

### Active

- [ ] Camera calibration matching real SOARM camera placement/FOV
- [ ] Depth camera observation stream (sim + dataset pipeline)
- [ ] Checkpoint-scored benchmark task suite (~8-15 tasks, LIBERO-style categories)
- [ ] Expanded object pool (existing SOARM-compatible LIBERO objects + a few custom objects)
- [ ] Demo re-collection for the expanded task suite
- [ ] Re-fine-tuned VLA + re-evaluation with success rate + generalization splits

### Out of Scope

- Real-3DQA point cloud data as training input — inspiration only, not in this pipeline
- Real-time interactive REPL (deferred; start with rendered output)

## Context

- Existing repo has LIBERO cloned as a modifiable fork (`LIBERO/`) with its own git history — will be extended with custom SOARM models, tasks, and datasets; MuJoCo/robosuite environments, Panda arm configs, and lifelong learning training code already present
- Real-3DQA exploration scripts exist (`explorations/real3dqa/`) as a parallel research thread — not directly connected to this pipeline
- Phase 2 complete (2026-07-18): SOARM SO101 registered as `MountedSoarm101`/`SoarmGripper` in the LIBERO fork — vendored SO-ARM100 geometry, stable physics (0.024 N reset contact force), tuned eye_in_hand camera, 3 frozen libero_spatial tasks; verified on Colab T4
- Google Colab is the compute platform for GPU-accelerated VLA inference (π0 or OpenVLA are leading candidates — open-source, trained on robot manipulation data)
- Dataset serves dual purpose: fine-tuning the VLA on SOARM kinematics AND benchmarking spatial understanding
- Spatial awareness goal is three-layered: multi-camera perception, 3D object localization, and spatial language understanding
- Physical SOARM hardware bring-up is underway as a parallel track running OUTSIDE the v1.1 milestone/phase structure — tracked via UATs in `diagnostics/UAT/` rather than `.planning/phases/`, and it does not replace or compete with v1.1's sim-only scope: `diagnostics/UAT/components/UAT.md` (electronics bring-up, complete, 7/7 steps), `diagnostics/UAT/assembly/gripper/UAT.md` (gripper build + calibration, complete, 11/11 steps), `diagnostics/UAT/assembly/main/UAT.md` (5-joint arm assembly, complete, 10/10 steps), `diagnostics/UAT/function/UAT.md` (LeRobot-based laptop control + camera recording, complete, 7/7 steps as of 2026-09-09 — now covers a **leader+follower two-arm setup**, not just the single follower: the follower was rebuilt with 12V/30kg-cm servos, and a leader arm was added by repurposing the original 7.4V servos, enabling real leader-follower teleop and a full combined `lerobot-record` episode (joints + both cameras, synced)). Control software lives in `control/` (a separate venv from `diagnostics/`), uses HuggingFace LeRobot for real-robot control, and requires Python 3.12 (not the system default 3.14, which crashes LeRobot's config parser).

## Constraints

- **Compute**: Google Colab GPU budget (T4/A100 depending on tier) — inference and training must be Colab-compatible
- **Robot Model**: No existing SOARM URDF/MJCF — must derive or build from SOARM hardware specs
- **Framework**: LIBERO + robosuite stack (MuJoCo 2.3.7, robosuite 1.4.x) — must stay compatible
- **VLA Candidates**: π0 (Physical Intelligence) or OpenVLA — both open weights, robot-data trained
- **Timeline**: MVP full-pipeline sketch target within 2-3 weeks

## Key Decisions

| Decision                       | Rationale                                                                     | Outcome    |
| ------------------------------ | ----------------------------------------------------------------------------- | ---------- |
| π0 / OpenVLA as VLA backbone  | Open-source, trained on robot manipulation data, closer to drop-in for LIBERO | — Pending |
| LIBERO as simulation framework | Already in repo, MuJoCo-based, has task suite infrastructure                  | — Pending |
| Google Colab for compute       | GPU access without local hardware investment                                  | — Pending |
| Simulation-only scope          | Derisk by validating pipeline in sim before physical robot                    | — Pending |
| Upgrade SOARM gripper to roboninecom 84mm parallel gripper (Phase 4, 2026-08-03) | Stock ~2-3cm jaw physically can't grasp ANY LIBERO object (smallest 4cm); roboninecom is real/printable (STEP+STL), 120-150N, same STS3215 servo. Modeled faithfully at 84mm. Also a real hardware upgrade for the eventual physical arm. | ✓ Committed (ad0b0d0); sim re-validated — 120 real demos collected on `put_the_cream_cheese_in_the_bowl` |
| Retarget Phase 4 off the 3 frozen bowl→plate tasks to a sub-84mm in-reach pick-place task (2026-08-03) | Bowl (11cm) exceeds even the 84mm jaw AND the plate place-target (~0.5m) is beyond the arm's ~0.45m reach; a small object with both pick+place in-reach is completable | ✓ Committed (5675163); collision-corridor fix (6ecd160) unblocked full 120-demo collection |
| Route both camera views through each VLA backend's native multi-image API, not a manual tile/concat fallback (Phase 5 D-02, 2026-08-10) | RESEARCH.md confirmed both OpenVLA-OFT and π0/openpi checkpoints have native dual-image support at the API/checkpoint level — a manual tile would silently degrade both models below their trained input distribution | ✓ Committed; OFTBackend needed a follow-up fix (`set_num_images_in_input(2)` + corrected agentview/eye_in_hand channel order, 05-04) after the initial implementation crashed on real Colab GPU inference — confirmed working end-to-end 2026-08-17 |
| Retire spawn-trivial spatial predicates as benchmark success criteria; require checkpoint/sub-goal predicates for all new v1.1 tasks (2026-08-30) | Phase 6 UAT found 3 of 4 eval tasks (RightOfX/NearTo/LeftOfX) are satisfied by object spawn position alone — pass at step 1 with 100% success regardless of policy quality, drowning out the one task (`On` predicate) that actually measures manipulation. Confirmed live: 100%/100%/100% before AND after fine-tuning on the trivial 3, vs 0%/0% on the real task. | — Pending |
| Target ~8-15 tasks for the v1.1 benchmark suite, mirroring LIBERO's 10-tasks-per-category convention | Research (LIBERO paper, MemoryVLA, VITA, Qwen-VLA) shows small custom-embodiment VLA benchmarks converge on 5-15 tasks for real-world-scale evaluation legs, vs. 50-150+ for large published sim suites | — Pending |
| Start physical hardware bring-up in parallel with v1.1 rather than waiting for sim validation to complete | Physical parts arrived and needed bring-up/testing; this work doesn't block or compete with the sim-focused v1.1 phases since it lives entirely outside the phase/ROADMAP structure (tracked via diagnostics/UAT/ instead) | ✓ Committed |

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

*Last updated: 2026-08-30 — Milestone v1.0 complete (Phases 1-6, ending with Phase 6: Fine-Tuning & Evaluation, 2026-08-29). Started milestone v1.1 (Perception Fidelity & Checkpoint Benchmark): fixes a UAT-surfaced benchmark flaw (3 of 4 eval tasks pass trivially at spawn) and adds camera calibration + depth perception (SEED-001). Note for future phases: SO-ARM101 is a small ~500g-payload arm — tasks must keep objects (<=84mm) and targets within ~0.45m reach.*
