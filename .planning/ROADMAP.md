# Roadmap: SoARM VLA Research

## Overview

This roadmap builds a Vision-Language-Action simulation pipeline in six phases. Phases 1-3 form a hard dependency chain: a working Colab environment with VLA loading gates the SOARM robot integration, which gates the end-to-end inference loop. Once that loop is validated, Phase 4 (dataset collection) and Phase 5 (spatial awareness) run in parallel. Phase 6 (fine-tuning and evaluation) closes the loop by training OpenVLA-OFT on SOARM demonstrations and benchmarking spatial vs. non-spatial task performance.

Milestone v1.1 (Phases 7-9) fixes a UAT-surfaced benchmark flaw from Phase 6 (3 of 4 eval tasks pass trivially at object spawn) and closes the sim-to-real perception gap. Phase 7 recalibrates the sim camera and plumbs a depth observation stream end-to-end (sim → HDF5 → RLDS). Phase 8 authors a SOARM-compatible object pool and a new checkpoint-scored benchmark task suite, retiring the 3 spawn-trivial spatial tasks from success metrics. Phase 9 collects demonstrations for that suite using the corrected perception pipeline, then re-fine-tunes and re-evaluates OpenVLA-OFT with before/after results in WandB. These three phases form a strict dependency chain (7 → 8 → 9): perception infrastructure must land before task authoring is finalized against it, and both must land before new data is collected and trained on. **Phases 8-9 are PAUSED as of 2026-09-15** (see PROJECT.md) — superseded in active focus by milestone v2.0.

Milestone v2.0 (Phases 10-11) shifts focus from sim/VLA fine-tuning to real-hardware MLLM manipulation, narrowed 2026-09-17 to two concrete workstreams after a general-MLLM-prompting experiment failed. Phase 10 rebuilds the digital twin (URDF + MuJoCo XML) as a correct 1:1 kinematic match to the real SO-ARM101 — this is sim-side-only work, fixing the CoppeliaSim-exported URDF's floating-gripper bug and the mirrored-gripper-gears bug so any future sim-side verification is trustworthy. Phase 11 connects an open-source HuggingFace-hosted VLA to the *real* physical robot over the already-working `control/` LeRobot USB-serial bridge, builds a harness to capture its full reasoning trace, and records at least one full observed run to inform the go/no-go decision on later milestone phases (deep-reasoning MLLM comparison, raw-autonomy design). **Phases 10 and 11 are independent and parallel-capable, not a sequential chain**: Phase 11's VLA experiment runs entirely on real hardware via `control/`'s existing LeRobot bridge and does not consume Phase 10's sim-twin outputs — the digital-twin fix only matters for future sim-side verification/re-training, not for driving the real robot. They are numbered sequentially here (10 then 11) purely by roadmap convention, not by dependency; either can be planned/executed first, or both in parallel, per user preference.

## Phases

**Phase Numbering:**

- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

**Execution Order (v1 / v1.1):** 1 → 2 → 3 → (4 ∥ 5) → 6 → 7 → 8 → 9
**Execution Order (v2.0):** (10 ∥ 11) — independent, parallel-capable (see Overview)

- [x] **Phase 1: Colab Environment Setup** - Install all dependencies conflict-free, verify EGL headless rendering, and confirm OpenVLA-OFT loads on GPU (4 plans) (closed 2026-07-10 — UAT 4/4 PASS after ENV-03 dependency fixes baked into notebook)
- [x] **Phase 2: SOARM Robot Integration** - Build and validate SOARM ManipulatorModel and MJCF, register in LIBERO, configure BDDL tasks (completed 2026-07-18)
- [x] **Phase 3: VLA Inference Loop** - Close the loop from language prompt to rendered SOARM task video with success detection and dual-VLA support (completed 2026-08-02)
- [x] **Phase 4: Dataset Collection** - Collect 100+ scripted SOARM demonstrations in robomimic HDF5 with state-based replay and normalization stats (completed 2026-08-03)
- [x] **Phase 5: Spatial Awareness** - Add multi-camera perception, depth-based 3D localization, and spatial language BDDL task variants (completed 2026-08-10)
- [x] **Phase 6: Fine-Tuning & Evaluation** - Fine-tune OpenVLA-OFT on SOARM demonstrations and benchmark spatial vs. non-spatial task success (completed 2026-08-29)
- [x] **Phase 7: Camera & Depth Perception** - Recalibrate the sim camera to match real SOARM placement/FOV and plumb a depth observation stream through the HDF5 writer and RLDS converter (completed 2026-09-12)
- [ ] **Phase 8: Checkpoint Benchmark Suite** ⏸ PAUSED (2026-09-15, v1.1 paused for v2.0 — see PROJECT.md) - Survey/author a SOARM-compatible object pool and a new ~8-15 task checkpoint-scored benchmark suite, retiring the 3 spawn-trivial spatial tasks from success metrics
- [ ] **Phase 9: Benchmark Data Collection & Re-Fine-Tuning** ⏸ PAUSED (2026-09-15, v1.1 paused for v2.0 — see PROJECT.md) - Collect demonstrations for the new benchmark suite and re-fine-tune/re-evaluate OpenVLA-OFT with before/after results in WandB
- [x] **Phase 10: Digital-Twin Fidelity** - Rebuild the URDF and MuJoCo XML as a correct, complete, 1:1 kinematic match to the real SO-ARM101 (gripper properly chained, wrist_roll + gripper joints restored, correct directions/limits, in-repo mesh paths) (completed 2026-09-19)
- [ ] **Phase 11: VLA Hardware Connection** - Connect an SO-101-native joint-action VLA (SmolVLA) to the real SO-ARM101 over the existing LeRobot bridge with a safety validator and complete per-step I/O logging, and record at least one full observed run

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
**⚠ In-execution amendment (2026-08-03):** the stock SOARM gripper can't grasp any LIBERO object (embodiment blocker). Gripper upgraded to the roboninecom **84mm parallel gripper** (faithful, committed `ad0b0d0`) and the target task retargeted off the bowl→plate tasks to a **sub-84mm object with pick+place both within the arm's ~0.45m reach**. See `04-CONTEXT.md` ⚠ AMENDMENT (D-07/D-08) + `STATE.md` Session Continuity for the resume sequence. Criterion 1 below now means 100+ demos of that retargeted task.
**Success Criteria** (what must be TRUE):

  1. The scripted collector produces 100+ SOARM demonstrations stored in robomimic HDF5 format with image observations present
  2. Any recorded demonstration replays frame-identically via state-setting (not action playback) — deterministic on re-run
  3. SOARM-specific action and observation normalization statistics are computed from the collected dataset (not copied from Panda)
  4. A human operator can record SOARM demonstrations using the teleoperation interface and they land in the same HDF5 format

**Plans**: 5/5 plans complete

Plans:
**Wave 1**

- [x] 04-01-PLAN.md — Recording infra (raw_recorder.py env builder) + hdf5_writer.py (robomimic-schema HDF5 with regenerated obs), proven by real local-sim integration test (DATA-01 partial)

**Wave 2** *(blocked on Wave 1)*

- [x] 04-02-PLAN.md — Scripted waypoint collector (D-01) + full 100+-demo collection run across the 3 corrected frozen tasks (DATA-01 complete)

**Wave 3** *(blocked on Wave 2)*

- [x] 04-03-PLAN.md — State-based determinism verification: states-only + sampled obs-regeneration tiers (D-06), run against the real collected dataset (DATA-02)
- [x] 04-04-PLAN.md — SOARM-specific normalization statistics (OpenVLA q01/q99 schema), run against the real collected dataset (DATA-03)

**Wave 4** *(blocked on Wave 3)*

- [x] 04-05-PLAN.md — Keyboard-only teleoperation interface (D-02) + schema-convergence proof + blocking human-operated session, closes DATA-02's determinism guarantee for the teleop path too (DATA-04)

### Phase 5: Spatial Awareness

**Goal**: The SOARM environment supports multi-camera RGB input, depth-based 3D object localization, and at least 3 BDDL spatial language task variants for benchmarking.
**Mode:** mvp
**Depends on**: Phase 2 (can start in parallel with Phase 4)
**Requirements**: SPAT-01, SPAT-02, SPAT-03, SPAT-04, SPAT-05
**Success Criteria** (what must be TRUE):

  1. Both wrist and overhead camera views are active in SOARM environments and all views are passed as input to the VLA during inference
  2. MuJoCo depth frames are extracted alongside RGB, and object XYZ positions are available as structured data (extractable from MuJoCo state)
  3. At least 3 BDDL tasks use spatial language prompts (e.g., "pick the cube to the left of the bowl") and success predicates correctly evaluate spatial conditions

**Plans**: 4/4 plans complete

Plans:
**Wave 1** (all 3 plans are independent -- zero files_modified overlap, run fully in parallel)

- [x] 05-01-PLAN.md — Multi-camera -> VLA wiring: eval_loop.py images dict, Pi0Backend/OFTBackend real dual-image consumption (D-01, D-02, SPAT-02)
- [x] 05-02-PLAN.md — Depth extraction + depth->XYZ back-projection pipeline: camera-config test, new perception/depth_xyz.py module, D-04 ground-truth validation (D-03, D-04, SPAT-01/03/04)
- [x] 05-03-PLAN.md — Spatial BDDL predicates + task authoring: LeftOfX/RightOfX/NearTo/FarFrom predicates, 3 new spatial BDDL tasks (left/right, near, between) (D-05..D-09, SPAT-05)

**Wave 2 — gap closure** *(from 05-UAT.md blocker)*

- [x] 05-04-PLAN.md — OFTBackend dual-camera fix: activate checkpoint's num_images_in_input=2 mode + correct inverted primary/extra_views order, with local regression tests (D-02, SPAT-02)

### Phase 6: Fine-Tuning & Evaluation

**Goal**: OpenVLA-OFT is fine-tuned on SOARM demonstrations via LoRA on Colab A100 and evaluated against a spatial benchmark, with before/after results tracked in WandB.
**Mode:** mvp
**Depends on**: Phase 4 and Phase 5
**Requirements**: TUNE-01, TUNE-02, TUNE-03, TUNE-04
**Success Criteria** (what must be TRUE):

  1. The robomimic HDF5 dataset converts to RLDS format and OpenVLA-OFT LoRA fine-tuning (r=32) runs to completion on Colab A100 without crashing
  2. Task success rates on spatial and non-spatial SOARM tasks are measured and compared before and after fine-tuning
  3. Training loss curves and evaluation success rates are visible in a WandB run dashboard

**Plans**: 4/4 plans complete

Plans:
**Wave 1** (independent -- zero files_modified overlap, run fully in parallel)

- [x] 06-01-PLAN.md — Custom robomimic HDF5 -> RLDS converter, genuine TFDS write, local schema/loading tests (TUNE-01)
- [x] 06-03-PLAN.md — eval_loop.py seed plumbing + FinetunedOFTBackend (HF Hub adapter reload) + eval notebook: seeded before/after benchmark + WandB eval logging (TUNE-03, TUNE-04)

**Wave 2** *(blocked on 06-01)*

- [x] 06-02-PLAN.md — OXE dataset registration + training notebook: finetune.py LoRA r=32 invocation, resumable HF Hub checkpoint push, WandB training curves (TUNE-02, TUNE-04)

**Wave 3 — gap closure** *(from 06-UAT.md blocker, test 1)*

- [x] 06-04-PLAN.md — Fix wrong `libero.datasets` import path (should be `libero.libero.datasets`) in 06a-finetune.ipynb's RLDS-conversion and OXE-registration cells (TUNE-01, TUNE-02)

### Phase 7: Camera & Depth Perception

**Goal**: The sim agentview/front camera is recalibrated to match real SOARM camera placement/FOV, and a depth observation stream flows end-to-end from the MuJoCo scene through the robosuite/LIBERO observation dict, the HDF5 dataset writer, and the RLDS converter.
**Mode:** mvp
**Depends on**: Phase 6
**Requirements**: CAM-01, DEPTH-01, DEPTH-02, DEPTH-03
**Success Criteria** (what must be TRUE):

  1. The sim agentview/front camera's position, angle, and FOV are recalibrated against documented real SOARM camera mount specs, and a rendered frame from the recalibrated camera visibly matches the real camera's framing (side-by-side/overlay comparison)
  2. A depth camera stream is exposed as a new observation key in the SOARM robosuite/LIBERO environment, retrievable from the observation dict returned by `env.reset()`/`env.step()` alongside RGB
  3. Depth frames are persisted in the HDF5 dataset writer alongside RGB for a recorded demo, verified by loading the file and inspecting the depth array's shape/dtype per timestep
  4. Depth frames survive the RLDS conversion — a converted TFDS record includes a depth field readable by the fine-tuning data loader

**Plans**: 3/3 plans complete

Plans:
**Wave 1** (independent -- zero files_modified overlap, run fully in parallel)

- [x] 07-01-PLAN.md — Camera recalibration: agentview fovy/pos/quat fix (reachable override, not dead-code base class) + eye_in_hand look-at-grip-site quat correction + human sign-off vs. reference photos (CAM-01)
- [x] 07-02-PLAN.md — Depth rendering + HDF5 persistence: camera_depths=True threading, _DEPTH_KEYS/OBS_KEY_MAPPING extension, stale gripper_states test-debt fix (DEPTH-01, DEPTH-02)

**Wave 2** *(blocked on 07-02)*

- [x] 07-03-PLAN.md — RLDS conversion + OXE registration: agentview_depth Tensor FeaturesDict, depth_obs_keys["primary"] registration, legacy-dataset skip handling (DEPTH-03)

### Phase 8: Checkpoint Benchmark Suite

**Goal**: A SOARM-compatible object pool and a new ~8-15 task benchmark suite exist, each task scored via multi-step checkpoint predicates with per-checkpoint/final/generalization reporting, and the 3 spawn-trivial legacy spatial tasks are excluded from aggregate success metrics.
**Mode:** mvp
**Depends on**: Phase 7
**Requirements**: BENCH-01, BENCH-02, BENCH-03, BENCH-04, OBJ-01, OBJ-02
**Status**: ⏸ PAUSED 2026-09-15 (v1.1 paused for v2.0 — see PROJECT.md)
**Success Criteria** (what must be TRUE):

  1. A survey identifies which existing LIBERO objects meet SOARM constraints (≤84mm graspable width, placeable within ~0.45m reach), and at least 1-3 new custom objects are authored via `custom_object_example.ipynb` and registered as usable LIBERO objects
  2. A new benchmark suite of 8-15 BDDL tasks is authored, organized into LIBERO-style categories (spatial/object/goal or an equivalent split), using only surveyed/authored objects that respect the 84mm graspable-width / 0.45m reach constraints and avoid the arm's forward centerline collision corridor
  3. Every new benchmark task defines checkpoint/sub-goal predicates (e.g. reach→grasp→lift→place, ~4 steps) that are individually markable as passed/failed during an episode, not just a single binary success flag
  4. Running the evaluation harness on the new suite reports per-checkpoint and final success rate plus generalization splits (seen vs unseen object positions/instructions), at the same 20-episode/task cadence as Phase 6
  5. The 3 spawn-trivial spatial tasks (`RightOfX`/`NearTo`/`LeftOfX` predicates satisfied at t=0) are excluded from the aggregate success-rate metrics reported by the evaluation harness

**Plans**: TBD

### Phase 9: Benchmark Data Collection & Re-Fine-Tuning

**Goal**: Demonstrations exist for every task in the new checkpoint benchmark suite, collected using the recalibrated camera and depth pipeline, and OpenVLA-OFT is re-fine-tuned and re-evaluated on that suite with before/after results tracked in WandB.
**Mode:** mvp
**Depends on**: Phase 8
**Requirements**: DATA-05, TUNE-05, TUNE-06
**Status**: ⏸ PAUSED 2026-09-15 (v1.1 paused for v2.0 — see PROJECT.md)
**Success Criteria** (what must be TRUE):

  1. Scripted and/or teleoperated demonstrations are collected for every task in the new checkpoint benchmark suite (8-15 tasks), stored in robomimic HDF5 format with both the recalibrated-camera RGB and the new depth observation stream present
  2. OpenVLA-OFT LoRA (r=32) re-fine-tuning runs to completion on Colab A100 using the expanded, depth-augmented dataset, without crashing
  3. The re-fine-tuned checkpoint is evaluated on the full new checkpoint benchmark suite using the Phase 8 checkpoint-scoring harness, producing per-checkpoint and final success rates plus generalization splits
  4. Before-fine-tuning and after-fine-tuning success rates are both tracked and visible side-by-side in a WandB dashboard

**Plans**: TBD

### Phase 10: Digital-Twin Fidelity

**Established model accepted 2026-09-20:** Coppelia-registered assembly, black
housing, yellow jaws, 0.036 m travel per jaw, mechanical coupling and corrected
LIBERO starting pose. [Accepted configuration and source inventory](phases/10-digital-twin-fidelity/10-ESTABLISHED-MODEL.md)
supersede the original assumptions below. Geometry/runtime integration are
verified; task-success and hardware-dynamics revalidation remain separate.

**Goal**: The URDF and the MuJoCo XML (`LIBERO/libero/libero/assets/robots/soarm101/robot.xml`) form a correct, complete, 1:1 kinematic match to the real SO-ARM101 — a proper parent/child chain ending at the gripper, `wrist_roll` and gripper jaw joints restored with correct axes/direction, correct joint limits, and in-repo mesh paths. This is sim-side-only work; it does not touch the real robot or the `control/` bridge.
**Mode:** mvp
**Depends on**: Nothing new (independent of Phase 9's paused work; parallel-capable with Phase 11 — see Overview for reasoning)
**Requirements**: TWIN-01, TWIN-02, TWIN-03, TWIN-04, TWIN-05, TWIN-06, TWIN-07
**Context/Notes**:

- Targets two artifacts: the CoppeliaSim-derived URDF (currently broken per `coppelia/export_model_library.py`'s own comments — the gripper is a separate root-level object positioned near the wrist but never parented under the arm) and the MuJoCo XML used by the paused v1.1 sim/VLA track.
- Mesh geometry (STL/DAE) from the new CoppeliaSim export is trusted as visually correct; only the joint/kinematic structure needs rebuilding.
- Joint limits must be derived by converting the real robot's LeRobot calibration ticks to radians, not guessed.

**Success Criteria** (what must be TRUE):

  1. The URDF's kinematic tree shows the gripper as a descendant of the wrist link, not a sibling of `robot_base` — verified by loading the URDF and walking/printing its parent-child joint chain
  2. The URDF includes a `wrist_roll` joint and `gripper_left`/`gripper_right` prismatic joints, each with limits matching the real robot's calibrated servo ranges (converted from LeRobot calibration ticks to radians)
  3. Every mesh reference in the URDF resolves as an in-repo relative path — loading the URDF from a clean checkout (no `~/Downloads/` or other user-specific absolute paths) succeeds
  4. Driving the simulated gripper (URDF or MuJoCo) with a given joint command opens/closes it in the same direction as the real gripper under the identical command
  5. The MuJoCo XML's joint set, parent/child chain, and joint limits agree with the corrected URDF, and an existing LIBERO SOARM environment's `env.reset()` still completes without errors after the update

**Plans**: 5/5 plans complete
Plans:
**Wave 1** (independent — zero files_modified overlap, run fully in parallel)

- [x] 10-01-PLAN.md — Calibration-derived joint limits: scripts/calibration_utils.py + robot.xml correction (TWIN-05, TWIN-06)
- [x] 10-02-PLAN.md — Gripper clamp visual mesh fix: soarm_gripper.xml D-04 correction + human sign-off (TWIN-03, TWIN-07)

**Wave 2** *(blocked on Wave 1)*

- [x] 10-03-PLAN.md — MJCF→URDF generation: scripts/mjcf_to_urdf.py + regenerated So-101/So-101.urdf (TWIN-01, TWIN-02, TWIN-03, TWIN-04, TWIN-06)

**Wave 3** *(blocked on Wave 2)*

- [x] 10-04-PLAN.md — URDF verification: yourdfpy legitimacy checkpoint + scripts/verify_urdf.py + scripts/test_verify_urdf.py (TWIN-01..06)

**Wave 4** *(blocked on Wave 3 — final phase gate)*

- [x] 10-05-PLAN.md — TWIN-07 human-in-the-loop real-vs-sim gripper-direction checkpoint

### Phase 11: VLA Hardware Connection

**Goal**: An SO-101-native joint-action VLA (SmolVLA — matching the benchmark paper's own tested architecture and avoiding the Cartesian/IK detour the sim OpenVLA-OFT path would require) drives the real SO-ARM101 over the existing `control/` LeRobot bridge, through an explicit safety validator, with complete per-inference-step I/O captured, and at least one full observed run recorded end-to-end to inform the go/no-go decision on later milestone phases (deep-reasoning MLLM comparison, raw-autonomy design).
**Mode:** mvp
**Depends on**: Nothing new (runs entirely on real hardware via the already-working `control/` LeRobot bridge; does not require Phase 10's sim-twin fix — parallel-capable with Phase 10, see Overview for reasoning)
**Requirements**: VLAHW-01, VLAHW-02, VLAHW-03, VLAHW-04, VLAHW-05
**Context/Notes**:

- Known upstream risk to verify early, not assume away: [huggingface/lerobot#2210](https://github.com/huggingface/lerobot/issues/2210) reports SmolVLA inference failures on SO-101. If SmolVLA proves unworkable, the requirement is "an SO-101-native joint-action VLA" generically — a fallback candidate should be identified during planning, not discovered mid-execution.
- Real, pre-existing action-space mismatch to design around: the sim OpenVLA-OFT path outputs 7D Cartesian deltas (`OSC_POSE`); the physical SO-101 takes 6D joint positions. These are not interchangeable — do not reuse the sim VLA pipeline directly. Prefer a joint-action-native VLA (Path A) over Cartesian output + IK (Path B, more moving parts: forward/inverse kinematics, reference frames, singularities) for this first physical experiment.
- Real, pre-existing unit-ambiguity bug to resolve before any policy drives the robot: LeRobot's SO-101 follower config defaults to `use_degrees=True`, but `control/keyboard_joint_control.py`'s own comments describe values as normalized -100..100 — the same value could mean degrees in one path and percent in another. Document and fix the actual contract (units, joint order, gripper scale, absolute vs. relative) before wiring in VLA output.
- Actions must reach the robot through the existing `control/` `SO101Follower` motor-bus interface — no new hand-written serial/register code — and must pass a safety validator first: joint limits, max per-step displacement, max velocity, gripper bounds, stale observation/response rejection, malformed/NaN/infinite action rejection, e-stop, servo comms-failure handling.
- "Complete I/O trace" replaces "reasoning trace" as the framing for VLAHW-03/04: SmolVLA (like other VLAs) maps observations directly to actions and has no natural-language reasoning/thinking output the way an LLM does — the useful analog is logging every inference step's raw camera frames (with per-camera timestamps, not sequential reads that can drift), joint state, instruction, raw model output, validated action, executed action, latency, and model version.
- This is a real-hardware experiment; it deliberately does not wait on Phase 10's sim-only digital-twin fix.

**Success Criteria** (what must be TRUE):

  1. A harness loads an SO-101-native joint-action VLA and calls it with real wrist-camera + overhead-camera frames and joint-state readings pulled live from the SO-ARM101 (not synthetic or simulated inputs), against an explicitly documented action contract (units, joint order, gripper scale, absolute vs. relative)
  2. The VLA's output actions pass a safety validator (joint limits, max per-step displacement, max velocity, gripper bounds, stale/malformed/NaN input and output rejection, e-stop, servo comms-failure handling) before being sent to the robot via the existing `control/` `SO101Follower` motor-bus interface, with no new hand-written raw serial/register code added to reach the servos
  3. Every inference step's complete I/O (raw camera frames with per-camera timestamps, joint state, instruction, raw model output, validated action, executed action, latency, model version) is written to a durable, timestamped log — not just the final executed action
  4. At least one full episode — from a task prompt to episode termination — runs end-to-end on the physical robot, producing a saved synchronized video and a matching per-step I/O log, plus recorded termination reason and success/failure outcome
  5. A short findings write-up synthesizes what was observed (including any upstream issues hit, e.g. lerobot#2210) into an explicit go/no-go recommendation for the next milestone phases

**Plans**: 3/5 plans executed
Plans:
**Wave 1** (independent — zero files_modified overlap, run fully in parallel)

- [x] 11-01-PLAN.md — Action contract + safety validator, unit-ambiguity resolution (VLAHW-01, VLAHW-02)
- [x] 11-03-PLAN.md — Colab PolicyServer/tunnel setup + SmolVLA checkpoint selection (VLAHW-01)

**Wave 2** *(blocked on 11-01)*

- [x] 11-02-PLAN.md — JSON Lines I/O logger + real-hardware episode harness with scripted dry-run source + e-stop (VLAHW-02, VLAHW-03)

**Wave 3** *(blocked on 11-02 and 11-03)*

- [ ] 11-04-PLAN.md — Real Colab-bridged SmolVLA wiring into run_vla_episode.py (VLAHW-01, VLAHW-02, VLAHW-03)

**Wave 4 — final phase gate** *(blocked on 11-04)*

- [ ] 11-05-PLAN.md — E-stop + full episode hardware-in-the-loop checkpoints, findings write-up (VLAHW-02, VLAHW-04, VLAHW-05)

## Progress

**Execution Order (v1 / v1.1):** 1 → 2 → 3 → (4 ∥ 5) → 6 → 7 → 8 → 9
**Execution Order (v2.0):** (10 ∥ 11) — independent, parallel-capable

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Colab Environment Setup | 4/4 | Complete   | 2026-07-09 |
| 2. SOARM Robot Integration | 5/5 | Complete    | 2026-07-18 |
| 3. VLA Inference Loop | 3/3 | Complete   | 2026-08-02 |
| 4. Dataset Collection | 5/5 | Complete    | 2026-08-03 |
| 5. Spatial Awareness | 4/4 | Complete    | 2026-08-10 |
| 6. Fine-Tuning & Evaluation | 4/4 | Complete   | 2026-08-29 |
| 7. Camera & Depth Perception | 3/3 | Complete   | 2026-09-12 |
| 8. Checkpoint Benchmark Suite | 0/TBD | Paused (2026-09-15) | - |
| 9. Benchmark Data Collection & Re-Fine-Tuning | 0/TBD | Paused (2026-09-15) | - |
| 10. Digital-Twin Fidelity | 5/5 | Complete    | 2026-09-19 |
| 11. VLA Hardware Connection | 3/5 | In Progress|  |
