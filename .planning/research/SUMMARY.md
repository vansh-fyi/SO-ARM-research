# Project Research Summary

**Project:** SoARM VLA Research — VLA simulation pipeline with spatial awareness
**Domain:** Vision-Language-Action (VLA) robot simulation (SOARM / LIBERO / MuJoCo / OpenVLA)
**Researched:** 2026-07-07
**Confidence:** MEDIUM

## Executive Summary

This project builds a VLA robot simulation pipeline that integrates the SO-ARM100 (SO101) robot into the LIBERO/MuJoCo benchmark framework, runs OpenVLA-OFT or π0 inference, and layers spatial awareness capabilities on top. The core research contribution is threefold: (1) a new robot embodiment for LIBERO (SOARM has never been integrated into this benchmark), (2) a spatial-language evaluation benchmark testing whether VLAs understand relational object positions, and (3) a fine-tuning pipeline adapting a generalist VLA to SOARM kinematics. Standard VLA simulation pipelines follow a strict dependency chain — robot model first, environment second, inference loop third, data collection fourth, fine-tuning fifth — and any deviation creates hard-to-debug failures.

The recommended approach is OpenVLA-OFT as the primary VLA (97.1% LIBERO average success, T4-compatible with 4-bit quantization, standard HuggingFace API), with π0 via LeRobot as a secondary option for smoother motion. The SOARM MJCF must be derived from the validated `so101_new_calib.xml` from TheRobotStudio/SO-ARM100 rather than built from scratch, then adapted for robosuite's `ManipulatorModel` subclass pattern. All simulation runs on Google Colab (T4 for inference, A100 for fine-tuning), which imposes strict dependency pinning and headless rendering requirements.

The biggest risks are: (1) SOARM MJCF integration quality — bad inertia or collision meshes corrupt the physics silently and invalidate all downstream data; (2) version lock fragility — robosuite 1.4.0 / mujoco 2.3.7 / gym 0.25.2 must be pinned exactly and managed alongside a conflicting transformers version requirement between LIBERO training code and VLA inference code; (3) dataset correctness — demo observations must be regenerated from MuJoCo states, action normalization stats must come from SOARM's own dataset, and replay must be state-based not action-based. Get these three right in Phases 1-3 and everything else follows.

## Key Findings

### Recommended Stack

The stack is tightly version-constrained. The simulation layer is: Python 3.10, MuJoCo 2.3.7, robosuite 1.4.0 (not 1.5 — it removed `SingleArmEnv` which LIBERO depends on), gym 0.25.2, LIBERO (vendored in repo), robomimic 0.2.0. PyTorch 2.1.x is the bridge version: old enough for robomimic, new enough for VLA inference. The VLA inference layer requires transformers >= 4.40, which hard-conflicts with LIBERO's lifelong training code that pins transformers 4.21.1 — the recommended mitigation is two separate Colab kernel groups or two conda environments.

**Core technologies:**
- Python 3.10: Last safe Colab target; 3.12+ breaks robomimic `distutils` assumptions
- MuJoCo 2.3.7: LIBERO pins this; do not upgrade (MuJoCo 3.x breaks MJCF loading patterns)
- robosuite 1.4.0: LIBERO's `BDDLBaseDomain` extends `SingleArmEnv` removed in 1.5
- OpenVLA-OFT: 7B, 97.1% LIBERO avg, T4-compatible at 16GB (tight), standard HuggingFace API
- LeRobot pi0: Flow-matching alternative with smoother motion, `lerobot/pi0_libero_base` checkpoint available
- SOARM MJCF (`so101_new_calib.xml`): Community-validated; start here, adapt for robosuite
- `MUJOCO_GL=egl`: Must be set before any MuJoCo import; EGL uses the Colab GPU; GLFW crashes headless

### Expected Features

**Must have (table stakes — MVP gates):**
- SOARM MJCF/URDF in robosuite via `ManipulatorModel` subclass — gates everything
- Single agentview camera → OpenVLA (4-bit quantized for T4) → SOARM actions → env.step() closed loop
- Scripted demonstration collection stored as HDF5 (robomimic format)
- Task success detection reused from LIBERO BDDL predicates
- Rendered video output per episode (imageio/matplotlib, Colab-compatible)

**Should have (spatial research contribution — differentiators):**
- Multi-camera setup (wrist + overhead/agentview) — essential for spatial reasoning
- Depth buffer extraction → 3D object position annotation via MuJoCo depth API
- Spatial language task variants ("to the left of", "near the", "between")
- OpenVLA LoRA fine-tuning on SOARM demonstrations (r=32 on A100 Colab Pro)
- Spatial benchmark: success rate on spatial vs non-spatial prompt variants

**Defer to v2+:**
- Ego3D position encoding (SpatialVLA approach) — needs v1 baseline first
- Physical SOARM hardware transfer — separate milestone after sim validated
- RLDS dataset export for Open X-Embodiment contribution
- RL-generated data augmentation, point clouds as VLA input, multi-robot coordination

### Architecture Approach

The system is a layered pipeline: researcher input (language prompt) → VLA inference (GPU, 7-D delta EEF action) → OSC_POSE controller (Jacobian IK) → MuJoCo physics (mj_step) → observation rendering (RGB + optional depth) → back to VLA. SOARM integration requires exactly two files changed inside LIBERO: a new `soarm.py` ManipulatorModel subclass and a one-line addition to `robots/__init__.py` ROBOT_CLASS_MAPPING. Research code lives in a new `soarm_pipeline/` package at repo root to avoid polluting the LIBERO fork. VLA backends are hidden behind a common `predict(image, language) -> action` interface to make them swappable.

**Major components:**
1. SOARM ManipulatorModel + MJCF — robot kinematics and mesh; derived from `so101_new_calib.xml`
2. LIBERO BDDLBaseDomain + OffScreenRenderEnv — task environment: scene setup, observations, success predicates
3. VLA Inference Layer (OpenVLA-OFT or pi0) — language + image → 7-D delta EEF action
4. OSC_POSE Controller — delta EEF → joint torques via Jacobian IK; needs SOARM-specific controller config JSON
5. HDF5 Dataset Writer — robomimic schema; must regenerate observations from saved states
6. Spatial Awareness Module — depth backprojection, object pose extraction, spatial predicate evaluation

### Critical Pitfalls

1. **MUJOCO_GL must be set before any import** — Set `os.environ["MUJOCO_GL"] = "egl"` as the very first line in every notebook. Setting it after any MuJoCo-related import has no effect. Failure: black frames or GL crash.

2. **SOARM requires a full ManipulatorModel subclass, not just an MJCF file** — Need Python class + adapted MJCF + controller config JSON with matching joint names. Failure: `KeyError: 'SOARM'` or cryptic shape errors in controller.

3. **Validate MJCF inertia and collision meshes before writing any Python wrapper code** — Bad URDF→MJCF conversion produces silent physics corruption. Run `mujoco.MjModel.from_xml_path()` and verify contact forces < 10 N at initialization before proceeding.

4. **Demo replay must be state-based, not action-based** — Action replay drifts (LIBERO issue #16). Always use `sim.set_state(demo["states"][t])`. Action-only demos are invalid training data.

5. **OpenVLA version pinning is exact** — torch 2.2.0, transformers 4.40.1, flash-attn 2.5.5 must all be pinned. Auto-upgraded versions cause `torch.compiler` attribute errors. Test in a clean session.

6. **Action normalization stats must come from SOARM's own dataset** — Copying LIBERO's Panda stats causes near-complete performance collapse. Compute mean/std from collected SOARM demonstrations.

## Implications for Roadmap

### Phase 1: Colab Environment and VLA Loading
**Rationale:** Everything downstream requires working Colab environment with verified MuJoCo headless rendering and confirmed VLA load. Binary blockers (MUJOCO_GL, VRAM, version pinning) all strike here.
**Delivers:** Pinned requirements cell, verified EGL rendering, OpenVLA-OFT loaded and running inference on test image, baseline LIBERO Panda environment confirmed functional.
**Addresses:** Table-stakes "VLA model loading and inference"; unblocks all downstream work.
**Avoids:** MUJOCO_GL misconfiguration, OpenVLA VRAM overrun, version-pinning conflicts.

### Phase 2: SOARM MJCF and Robot Integration
**Rationale:** SOARM MJCF is the root dependency of the entire feature tree. Highest-risk phase with the most silent failure modes. Must be fully validated before writing any pipeline code.
**Delivers:** Validated `soarm.py` ManipulatorModel, adapted MJCF in LIBERO assets, SOARM in ROBOT_CLASS_MAPPING, stable `env.reset()`, verified gripper convention, controller config JSON.
**Addresses:** "SOARM MJCF/URDF model" P1 blocker.
**Avoids:** Building MJCF from scratch, inertia errors, gripper sign mismatch, missing OSC controller config, wrong joint calibration mode.
**Research flag:** Needs verification of robosuite 1.4 ManipulatorModel subclassing specifics (reference TechLabs Aachen SO100+robosuite as prior art).

### Phase 3: End-to-End VLA Inference Loop
**Rationale:** Validate the full closed loop before collecting training data. A broken loop produces worthless demonstrations.
**Delivers:** Colab notebook with zero-shot OpenVLA-OFT rollout on SOARM tasks, rendered video output, task success detection, confirmed vertical image flip applied.
**Addresses:** "Single-camera VLA inference loop" (P1), "Rendered video output" (P1), "Task success detection" (P1).
**Avoids:** Forgetting `[::-1]` vertical image flip; running VLA synchronously on every sim tick (use step-skipping at 3-5 Hz or action chunking).

### Phase 4: Scripted Demonstration Collection
**Rationale:** Fine-tuning requires 50-200 correct demonstrations. Must follow validated inference loop to confirm task mechanics work.
**Delivers:** 100+ SOARM demonstrations in robomimic HDF5, observation regeneration completed (images present), SOARM-specific normalization stats computed.
**Addresses:** "Scripted demonstration collection" (P1); enables fine-tuning (P2).
**Avoids:** Action playback drift (state-based replay), missing HDF5 observations (run regeneration script), wrong normalization stats, rotation composition bug.

### Phase 5: Spatial Awareness Layer
**Rationale:** The primary research contribution. Additive to validated base pipeline. Spatial task definitions and benchmark complete the publishable research story.
**Delivers:** Multi-camera OffScreenRenderEnv (wrist + agentview), depth → 3D object position pipeline, spatial language task variants, spatial benchmark evaluation.
**Addresses:** All P2 differentiator features (multi-camera, 3D localization, spatial language grounding, spatial benchmark).
**Avoids:** Assuming raw depth input gives VLA spatial reasoning without architectural changes — use multi-camera RGB + auxiliary 3D annotations for task scripting.
**Research flag:** Spatial VLA input representation (multi-camera RGB vs RGB+depth vs auxiliary annotations) needs deeper research before committing to implementation.

### Phase 6: SOARM Fine-Tuning Pipeline
**Rationale:** Adapts generalist VLA to SOARM kinematics. Requires completed dataset (Phase 4) and validated baseline (Phase 3). Needs A100 Colab Pro.
**Delivers:** Fine-tuned OpenVLA-OFT LoRA checkpoint, before/after success rate comparison, fine-tuned model on spatial task suite.
**Addresses:** "SOARM-specific fine-tuning" (P2); completes publishable evaluation.
**Avoids:** Using Panda normalization stats, skipping dataset validation before training.

### Phase Ordering Rationale

- Phases 1-3 are strictly sequential (hard dependency chain): Colab env → robot model → inference loop
- Phase 4 (data collection) can begin as soon as Phase 2 (robot env) is stable, in parallel with Phase 3 refinement
- Phase 5 (spatial) can develop depth/camera utilities in parallel with Phase 4, but spatial task variants require Phase 3 working
- Phase 6 (fine-tuning) requires both Phase 4 (dataset) and Phase 3 (baseline), so it comes last
- Anti-features (hardware transfer, RL, point-cloud VLA input) are explicitly deferred and should not appear until Phase 6 completes

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 2:** SOARM robosuite 1.4 ManipulatorModel integration — novel, no LIBERO-specific prior art; reference TechLabs Aachen SO100+robosuite article
- **Phase 5:** Spatial awareness input representation — open research question; study SpatialVLA, VEGA, cVLA papers before committing

Phases with well-documented patterns (can skip deep research):
- **Phase 1:** EGL setup, HuggingFace model loading, version pinning — fully documented in source repos
- **Phase 4:** robomimic HDF5 schema, LIBERO regeneration script — reference implementation in openvla repo
- **Phase 6:** OpenVLA LoRA fine-tuning — documented in openvla-oft repo with LIBERO-specific configs

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | MEDIUM | Version pins cross-verified; transformers conflict well-documented; exact flash-attn compat is version-sensitive |
| Features | MEDIUM | Dependency chain clear; MVP scope well-defined; spatial research approach has SpatialVLA precedent |
| Architecture | MEDIUM | Verified against LIBERO codebase directly; OSC_POSE controller docs are for robosuite 1.5, 1.4 assumed similar |
| Pitfalls | MEDIUM | Most sourced from GitHub issues and official docs; SOARM-specific pitfalls inferred from general MuJoCo integration experience |

**Overall confidence:** MEDIUM

### Gaps to Address

- **SOARM MJCF robosuite adapter completeness:** Exact delta between `so101_new_calib.xml` and robosuite 1.4 ManipulatorModel requirements (site names, actuator group) not fully documented. Budget 1-2 days of iterative MJCF editing in Phase 2.
- **transformers version conflict in Colab:** Two-kernel-group approach is feasible but untested; validate exact cell ordering before committing.
- **Spatial VLA input representation:** Whether multi-camera RGB only, RGB+depth, or auxiliary 3D annotations is the right approach — needs a small ablation.
- **OpenVLA-OFT normalization stats format:** Exact JSON schema for `dataset_statistics.json` needs verification against training script before Phase 6.

## Sources

### Primary (HIGH confidence)
- LIBERO GitHub issue #49 — robosuite 1.5 SingleArmEnv removal confirmed
- LIBERO GitHub issue #16 — action playback drift confirmed
- openpi GitHub issue #416 — rotation composition bug confirmed

### Secondary (MEDIUM confidence)
- moojink/openvla-oft repo and project page — VRAM requirements, LIBERO benchmark results
- TheRobotStudio/SO-ARM100 Simulation/SO101/README — MJCF file location, calibration modes
- HuggingFace LeRobot LIBERO docs and lerobot/pi0_libero_base checkpoint — pi0 inference API
- LIBERO codebase inspection (robots/__init__.py, env_wrapper.py, bddl_base_domain.py) — robot registration pattern
- OpenVLA README — version pins and flash-attn requirements
- robosuite installation docs — EGL/OSMesa/GLFW rendering backends
- openvla regenerate_libero_dataset.py — observation regeneration workflow

### Tertiary (LOW confidence)
- TechLabs Aachen SO100 + SmolVLA + robosuite integration (Medium article) — useful prior art but different robosuite version
- Claru OpenVLA-OFT guide — dataset format details; third-party, needs validation
- cVLA and VEGA papers — spatial VLA input representations; relevant for Phase 5 decisions

---
*Research completed: 2026-07-07*
*Ready for roadmap: yes*
