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

- [ ] **TUNE-01**: Robomimic HDF5 dataset is converted to RLDS format compatible with OpenVLA fine-tuning pipeline
- [ ] **TUNE-02**: OpenVLA-OFT is fine-tuned on SOARM demonstrations using LoRA (r=32) on Colab A100
- [ ] **TUNE-03**: Spatial vs non-spatial task success rates are benchmarked before and after fine-tuning
- [ ] **TUNE-04**: Training metrics and evaluation results are tracked with WandB

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
| TUNE-01 | Phase 6 | Pending |
| TUNE-02 | Phase 6 | Pending |
| TUNE-03 | Phase 6 | Pending |
| TUNE-04 | Phase 6 | Pending |

**Coverage:**

- v1 requirements: 24 total
- Mapped to phases: 24
- Unmapped: 0 ✓

---
*Requirements defined: 2026-07-07*
*Last updated: 2026-07-07 after initial definition*
