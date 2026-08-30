# Requirements: SoARM VLA Research

**Defined:** 2026-07-07
**Core Value:** A researcher types a task prompt and watches SOARM execute it in a LIBERO simulation — the loop from language to embodied action.

## v1 Requirements

Requirements for initial research pipeline. Each maps to roadmap phases.

### Environment Setup

- [ ] **ENV-01**: Colab notebook installs all dependencies in correct order (robosuite 1.4.0, MuJoCo 2.3.7, gym 0.25.2, PyTorch 2.1.x) without conflicts
- [ ] **ENV-02**: EGL headless rendering is configured (MUJOCO_GL=egl set before any MuJoCo import) and produces non-black frames
- [ ] **ENV-03**: OpenVLA-OFT model loads successfully on Colab GPU (A100/T4) via HuggingFace
- [x] **ENV-04**: SOARM SO101 MJCF (from TheRobotStudio/SO-ARM100) is adapted for robosuite 1.4 and registered as a ManipulatorModel subclass
- [x] **ENV-05**: SOARM robot is registered in LIBERO's ROBOT_CLASS_MAPPING and can be instantiated in a LIBERO environment
- [x] **ENV-06**: At least 3 BDDL tasks are configured to use SOARM (replacing Panda arm)
- [x] **ENV-07**: Rendered frames from SOARM environment are visually correct (right-side-up, correct camera angle, physics stable)

### VLA Inference Pipeline

- [ ] **VLA-01**: OpenVLA-OFT receives (image, language prompt) and outputs 7-D delta EEF action that drives SOARM in sim
- [ ] **VLA-02**: Episode frames are saved as a video file for inspection after each inference run
- [ ] **VLA-03**: LIBERO task success rate is measured using LIBERO's evaluation protocol for SOARM tasks
- [ ] **VLA-04**: π0 (openpi) inference loop is integrated as a second VLA option alongside OpenVLA-OFT

### Dataset Collection

- [x] **DATA-01**: Scripted demonstration collector records SOARM task completions to robomimic HDF5 format
- [x] **DATA-02**: Recorded demonstrations replay deterministically via state-setting (not action playback)
- [x] **DATA-03**: SOARM-specific action and observation normalization statistics are computed from the collected dataset
- [x] **DATA-04**: Teleoperation interface allows human-controlled SOARM demonstration recording

### Spatial Awareness

- [x] **SPAT-01**: At least 2 camera views (wrist + overhead) are configured in LIBERO SOARM environments
- [x] **SPAT-02**: All camera views are passed as input to the VLA during inference
- [x] **SPAT-03**: MuJoCo depth buffer frames are extracted alongside RGB frames
- [x] **SPAT-04**: Object XYZ positions are extracted from MuJoCo state and available as structured context
- [x] **SPAT-05**: At least 3 BDDL tasks use spatial language prompts (e.g. "pick the cube to the left of the bowl")

### Fine-Tuning & Evaluation

- [x] **TUNE-01**: Robomimic HDF5 dataset is converted to RLDS format compatible with OpenVLA fine-tuning pipeline
- [x] **TUNE-02**: OpenVLA-OFT is fine-tuned on SOARM demonstrations using LoRA (r=32) on Colab A100
- [x] **TUNE-03**: Spatial vs non-spatial task success rates are benchmarked before and after fine-tuning
- [x] **TUNE-04**: Training metrics and evaluation results are tracked with WandB

## v1.1 Requirements

Requirements for milestone v1.1 (Perception Fidelity & Checkpoint Benchmark). Addresses a UAT-surfaced flaw in Phase 6's benchmark (3 of 4 eval tasks pass trivially at object spawn) and adds camera/depth perception fidelity work (SEED-001).

### Camera Calibration

- [ ] **CAM-01**: Sim agentview/front camera is recalibrated (position, angle, FOV) to match the real SOARM camera placement, closing the sim-to-real view gap

### Depth Perception

- [ ] **DEPTH-01**: A depth camera stream is added to the SOARM MuJoCo scene and exposed as a robosuite/LIBERO observation
- [ ] **DEPTH-02**: Depth frames are persisted alongside RGB in the HDF5 dataset writer
- [ ] **DEPTH-03**: Depth frames are carried through the RLDS converter into the fine-tuning dataset

### Checkpoint Benchmark Suite

- [ ] **BENCH-01**: The 3 spawn-trivial spatial tasks (`RightOfX`/`NearTo`/`LeftOfX` predicates satisfied at t=0) are excluded from success-rate benchmark aggregates
- [ ] **BENCH-02**: A new benchmark task suite of ~8-15 tasks is authored, organized into LIBERO-style categories (mirroring spatial/object/goal or an equivalent split)
- [ ] **BENCH-03**: Each benchmark task defines checkpoint/sub-goal predicates (e.g. reach→grasp→lift→place, ~4 steps) individually markable as passed/failed during an episode
- [ ] **BENCH-04**: Evaluation reports per-checkpoint and final success rate plus generalization splits (seen vs unseen object positions/instructions), at the same 20-episode/task cadence as Phase 6

### Object Pool

- [ ] **OBJ-01**: Existing LIBERO objects are surveyed and filtered for SOARM compatibility (≤84mm graspable width, placeable within ~0.45m reach)
- [ ] **OBJ-02**: At least 1-3 new custom objects are authored (MJCF asset + registration) via the `custom_object_example.ipynb` workflow to diversify the benchmark object pool

### Dataset Collection (v1.1)

- [ ] **DATA-05**: Demonstrations are collected (scripted and/or teleoperated) for every task in the new checkpoint benchmark suite

### Fine-Tuning & Evaluation (v1.1)

- [ ] **TUNE-05**: OpenVLA-OFT is re-fine-tuned (LoRA r=32) on the expanded dataset, including depth-augmented observations
- [ ] **TUNE-06**: The re-fine-tuned model is evaluated on the new checkpoint benchmark suite with before/after comparison tracked in WandB

## v2 Requirements

Deferred to after physical SOARM deployment milestone.

### Physical Robot Transfer

- **PHYS-01**: Fine-tuned VLA policy transfers to physical SOARM hardware
- **PHYS-02**: Real-world task success rate measured against sim baseline
- **PHYS-03**: Sim-to-real gap analysis report

### Advanced Spatial Reasoning

- **ADV-01**: Ego3D position encoding integrated into VLA input representation
- **ADV-02**: Point cloud input pipeline (Real-3DQA-style) for richer spatial context
- **ADV-03**: Multi-object spatial relationship reasoning ("between", "behind", "stacked on")

### Interactive Interface

- **INT-01**: Real-time REPL: send prompts and watch live simulation respond
- **INT-02**: Web-based visualization dashboard for experiment comparison

## Out of Scope

| Feature | Reason |
|---------|--------|
| Real-3DQA point cloud as training input | Inspiration only; separate research thread not connected to this pipeline |
| RL training in simulation | High compute cost, out of scope for VLA fine-tuning approach |
| Real-time interactive REPL (v1) | Deferred; rendered video output sufficient for pipeline validation |
| robosuite 1.5+ | Hard incompatibility with LIBERO (SingleArmEnv removed) |
| Free Colab T4 for OpenVLA-7B inference | T4 = 15GB; OpenVLA-7B needs ~16GB+; requires Colab Pro A100 |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| ENV-01 | Phase 1 | Pending |
| ENV-02 | Phase 1 | Pending |
| ENV-03 | Phase 1 | Pending |
| ENV-04 | Phase 2 | Complete |
| ENV-05 | Phase 2 | Complete |
| ENV-06 | Phase 2 | Complete |
| ENV-07 | Phase 2 | Complete |
| VLA-01 | Phase 3 | Pending |
| VLA-02 | Phase 3 | Pending |
| VLA-03 | Phase 3 | Pending |
| VLA-04 | Phase 3 | Pending |
| DATA-01 | Phase 4 | Complete |
| DATA-02 | Phase 4 | Complete |
| DATA-03 | Phase 4 | Complete |
| DATA-04 | Phase 4 | Complete |
| SPAT-01 | Phase 5 | Complete |
| SPAT-02 | Phase 5 | Complete |
| SPAT-03 | Phase 5 | Complete |
| SPAT-04 | Phase 5 | Complete |
| SPAT-05 | Phase 5 | Complete |
| TUNE-01 | Phase 6 | Complete |
| TUNE-02 | Phase 6 | Complete |
| TUNE-03 | Phase 6 | Complete |
| TUNE-04 | Phase 6 | Complete |
| CAM-01 | Phase 7 | Pending |
| DEPTH-01 | Phase 7 | Pending |
| DEPTH-02 | Phase 7 | Pending |
| DEPTH-03 | Phase 7 | Pending |
| BENCH-01 | Phase 8 | Pending |
| BENCH-02 | Phase 8 | Pending |
| BENCH-03 | Phase 8 | Pending |
| BENCH-04 | Phase 8 | Pending |
| OBJ-01 | Phase 8 | Pending |
| OBJ-02 | Phase 8 | Pending |
| DATA-05 | Phase 9 | Pending |
| TUNE-05 | Phase 9 | Pending |
| TUNE-06 | Phase 9 | Pending |

**Coverage:**

- v1 requirements: 24 total
- v1.1 requirements: 13 total
- Mapped to phases: 24 (v1) + 13 (v1.1) = 37
- Unmapped: 0 (v1) ✓ / 0 (v1.1) ✓

---
*Requirements defined: 2026-07-07*
*Last updated: 2026-08-30 after roadmap creation for milestone v1.1 (Phases 7-9); corrected v1.1 requirement count from a prior miscount of 17 to the actual 13 enumerated IDs (CAM-01, DEPTH-01/02/03, BENCH-01/02/03/04, OBJ-01/02, DATA-05, TUNE-05/06).*
