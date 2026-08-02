# Roadmap: SoARM VLA Research

## Overview

This roadmap builds a Vision-Language-Action simulation pipeline in six phases. Phases 1-3 form a hard dependency chain: a working Colab environment with VLA loading gates the SOARM robot integration, which gates the end-to-end inference loop. Once that loop is validated, Phase 4 (dataset collection) and Phase 5 (spatial awareness) run in parallel. Phase 6 (fine-tuning and evaluation) closes the loop by training OpenVLA-OFT on SOARM demonstrations and benchmarking spatial vs. non-spatial task performance.

## Phases

**Phase Numbering:**

- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

**Execution Order:** 1 → 2 → 3 → (4 ∥ 5) → 6

- [x] **Phase 1: Colab Environment Setup** - Install all dependencies conflict-free, verify EGL headless rendering, and confirm OpenVLA-OFT loads on GPU (4 plans) (closed 2026-07-10 — UAT 4/4 PASS after ENV-03 dependency fixes baked into notebook)
- [x] **Phase 2: SOARM Robot Integration** - Build and validate SOARM ManipulatorModel and MJCF, register in LIBERO, configure BDDL tasks (completed 2026-07-18)
- [x] **Phase 3: VLA Inference Loop** - Close the loop from language prompt to rendered SOARM task video with success detection and dual-VLA support (completed 2026-08-02)
- [ ] **Phase 4: Dataset Collection** - Collect 100+ scripted SOARM demonstrations in robomimic HDF5 with state-based replay and normalization stats
- [ ] **Phase 5: Spatial Awareness** - Add multi-camera perception, depth-based 3D localization, and spatial language BDDL task variants
- [ ] **Phase 6: Fine-Tuning & Evaluation** - Fine-tune OpenVLA-OFT on SOARM demonstrations and benchmark spatial vs. non-spatial task success

## Phase Details

### Phase 1: Colab Environment Setup

**Goal**: A working Colab notebook environment where all dependencies install without version conflicts, GPU is accessible for VLA inference, and headless rendering produces valid RGB frames.
**Mode:** mvp
**Depends on**: Nothing (first phase)
**Requirements**: ENV-01, ENV-02, ENV-03
**Success Criteria** (what must be TRUE):

  1. All simulation dependencies (robosuite 1.4.0, MuJoCo 2.3.7, gym 0.25.2, PyTorch 2.1.x, transformers 4.40.1) install in order on a fresh Colab runtime without conflicts
  2. A default LIBERO Panda environment renders non-black RGB frames using EGL headless rendering (MUJOCO_GL=egl set before any MuJoCo import)
  3. OpenVLA-OFT loads onto Colab GPU without OOM errors and returns a 7-D action output given a test image and prompt

**Plans**: 4/4 plans complete
Plans:
**Wave 1**

- [x] 01-01-PLAN.md — Notebook skeleton + Block A install cells (GPU check, apt, pip, restart)

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 01-02-PLAN.md — Post-restart EGL bootstrap, LIBERO config, ENV-01/02 verification

**Wave 3** *(blocked on Wave 2 completion)*

- [x] 01-03-PLAN.md — ENV-03 OpenVLA-OFT load verification + summary table

**Wave 4 — gap closure** *(from 01-UAT.md blockers)*

- [x] 01-04-PLAN.md — numpy ABI gate (Block A final cell) + transformers jax/TF backend guards; closes ENV-02/ENV-03 UAT blockers

### Phase 2: SOARM Robot Integration

**Goal**: The SOARM SO101 robot is registered in LIBERO as a validated ManipulatorModel with stable physics, correct rendering, and at least 3 BDDL tasks configured to use it.
**Mode:** mvp
**Depends on**: Phase 1
**Requirements**: ENV-04, ENV-05, ENV-06, ENV-07
**Success Criteria** (what must be TRUE):

  1. `env.reset()` on a SOARM-based LIBERO environment completes without errors and contact forces at initialization are below 10 N (physics stable)
  2. Rendered frames from SOARM environments are visually correct: right-side-up, correct camera angle, no mesh artifacts
  3. SOARM is selectable by name in LIBERO's ROBOT_CLASS_MAPPING exactly as Panda is today
  4. At least 3 BDDL tasks run end-to-end with the SOARM robot without crashing

**Plans**: 5/5 plans complete

Plans:
**Wave 1**

- [x] 02-01-PLAN.md — Validation harness (soarm_sanity.py) + vendored SO101 assets + arm MJCF adaptation (ENV-04)

**Wave 2** *(blocked on Wave 1)*

- [x] 02-02-PLAN.md — Gripper MJCF + MountedSoarm101/SoarmGripper classes + LIBERO registration (ENV-04, ENV-05)

**Wave 3** *(blocked on Wave 2)*

- [x] 02-03-PLAN.md — Physics stabilization: stable reset <10 N + random-action soak, D-11 tuning (ENV-05, ENV-07)

**Wave 4** *(blocked on Wave 3)*

- [x] 02-04-PLAN.md — Render correctness (eye_in_hand camera tuning) + final 3 BDDL task selection per D-03 (ENV-06, ENV-07)

**Wave 5** *(blocked on Wave 4)*

- [x] 02-05-PLAN.md — Colab verification notebook (ENV-04..07 PASS/FAIL) + blocking human sign-off (D-10)

### Phase 3: VLA Inference Loop

**Goal**: A complete closed loop from language prompt to rendered episode video, with task success/failure detection and a swappable dual-VLA interface (OpenVLA-OFT and pi0).
**Mode:** mvp
**Depends on**: Phase 2
**Requirements**: VLA-01, VLA-02, VLA-03, VLA-04
**Success Criteria** (what must be TRUE):

  1. Running a single Colab notebook cell with a text prompt produces a saved video file of SOARM attempting the task in simulation
  2. Task success or failure is detected and printed after each episode using LIBERO's BDDL evaluation protocol
  3. The pi0 (openpi) VLA backend can be swapped in via the same `predict(image, language) -> action` interface without changing downstream pipeline code

**Plans**: 3/3 plans complete

Plans:
**Wave 1**

- [x] 03-01-PLAN.md — Shared VLABackend interface + OFTBackend + eval_loop module (VLA-01, VLA-02, VLA-03)

**Wave 2** *(blocked on Wave 1)*

- [x] 03-02-PLAN.md — Notebook A: OFT full eval loop, 3 tasks x 5-10 episodes (VLA-01, VLA-02, VLA-03)
- [x] 03-03-PLAN.md — Pi0Backend + Notebook B: π0 smoke test, separate kernel (VLA-04)

### Phase 4: Dataset Collection

**Goal**: A scripted and teleoperated demonstration collection system that records SOARM task completions as valid robomimic HDF5 datasets with state-based replay and correct SOARM-specific normalization statistics.
**Mode:** mvp
**Depends on**: Phase 2 (can start in parallel with Phase 3)
**Requirements**: DATA-01, DATA-02, DATA-03, DATA-04
**Success Criteria** (what must be TRUE):

  1. The scripted collector produces 100+ SOARM demonstrations stored in robomimic HDF5 format with image observations present
  2. Any recorded demonstration replays frame-identically via state-setting (not action playback) — deterministic on re-run
  3. SOARM-specific action and observation normalization statistics are computed from the collected dataset (not copied from Panda)
  4. A human operator can record SOARM demonstrations using the teleoperation interface and they land in the same HDF5 format

**Plans**: 2/5 plans executed

Plans:
**Wave 1**

- [x] 04-01-PLAN.md — Recording infra (raw_recorder.py env builder) + hdf5_writer.py (robomimic-schema HDF5 with regenerated obs), proven by real local-sim integration test (DATA-01 partial)

**Wave 2** *(blocked on Wave 1)*

- [x] 04-02-PLAN.md — Scripted waypoint collector (D-01) + full 100+-demo collection run across the 3 corrected frozen tasks (DATA-01 complete)

**Wave 3** *(blocked on Wave 2)*

- [ ] 04-03-PLAN.md — State-based determinism verification: states-only + sampled obs-regeneration tiers (D-06), run against the real collected dataset (DATA-02)
- [ ] 04-04-PLAN.md — SOARM-specific normalization statistics (OpenVLA q01/q99 schema), run against the real collected dataset (DATA-03)

**Wave 4** *(blocked on Wave 3)*

- [ ] 04-05-PLAN.md — Keyboard-only teleoperation interface (D-02) + schema-convergence proof + blocking human-operated session, closes DATA-02's determinism guarantee for the teleop path too (DATA-04)

### Phase 5: Spatial Awareness

**Goal**: The SOARM environment supports multi-camera RGB input, depth-based 3D object localization, and at least 3 BDDL spatial language task variants for benchmarking.
**Mode:** mvp
**Depends on**: Phase 2 (can start in parallel with Phase 4)
**Requirements**: SPAT-01, SPAT-02, SPAT-03, SPAT-04, SPAT-05
**Success Criteria** (what must be TRUE):

  1. Both wrist and overhead camera views are active in SOARM environments and all views are passed as input to the VLA during inference
  2. MuJoCo depth frames are extracted alongside RGB, and object XYZ positions are available as structured data (extractable from MuJoCo state)
  3. At least 3 BDDL tasks use spatial language prompts (e.g., "pick the cube to the left of the bowl") and success predicates correctly evaluate spatial conditions

**Plans**: TBD

### Phase 6: Fine-Tuning & Evaluation

**Goal**: OpenVLA-OFT is fine-tuned on SOARM demonstrations via LoRA on Colab A100 and evaluated against a spatial benchmark, with before/after results tracked in WandB.
**Mode:** mvp
**Depends on**: Phase 4 and Phase 5
**Requirements**: TUNE-01, TUNE-02, TUNE-03, TUNE-04
**Success Criteria** (what must be TRUE):

  1. The robomimic HDF5 dataset converts to RLDS format and OpenVLA-OFT LoRA fine-tuning (r=32) runs to completion on Colab A100 without crashing
  2. Task success rates on spatial and non-spatial SOARM tasks are measured and compared before and after fine-tuning
  3. Training loss curves and evaluation success rates are visible in a WandB run dashboard

**Plans**: TBD

## Progress

**Execution Order:** 1 → 2 → 3 → (4 ∥ 5) → 6

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Colab Environment Setup | 4/4 | Complete   | 2026-07-09 |
| 2. SOARM Robot Integration | 5/5 | Complete    | 2026-07-18 |
| 3. VLA Inference Loop | 3/3 | Complete   | 2026-08-02 |
| 4. Dataset Collection | 2/5 | In Progress|  |
| 5. Spatial Awareness | 0/TBD | Not started | - |
| 6. Fine-Tuning & Evaluation | 0/TBD | Not started | - |
