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

Deferred to after physical SOARM deployment milestone (v1-era deferrals; superseded in practice once physical hardware bring-up began — see v2.0 below, which is the actual physical-deployment milestone).

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

## v2.0 Requirements

Requirements for milestone v2.0 (Real-Hardware MLLM Manipulation Benchmark). v1.1 (Phases 7-9, sim/VLA) is PAUSED, not cancelled — see PROJECT.md. Narrowed 2026-09-17 to two concrete workstreams after a general-MLLM-prompting experiment failed; the full raw-autonomy MLLM-benchmark-suite design is captured under Future Requirements below, pending these results.

### Digital-Twin Fidelity

- [x] **TWIN-01**: The URDF's gripper is attached at the end of the kinematic chain (child of the wrist link), not floating off `robot_base`
- [x] **TWIN-02**: The URDF includes a `wrist_roll` joint with range matching the real robot's calibrated servo limits
- [x] **TWIN-03**: The URDF includes `gripper_left`/`gripper_right` prismatic joints whose open/close direction matches the real gripper (fixes the mirrored-gears bug)
- [x] **TWIN-04**: All mesh file references in the URDF use in-repo relative paths, not absolute `~/Downloads/` paths
- [x] **TWIN-05**: URDF joint limits for the 4 arm joints derivable from calibration (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex) match the real servo calibration ranges (converted from LeRobot calibration ticks to radians); wrist_roll and gripper limits are explicitly excepted per D-06 (wrist_roll has no real calibrated range; gripper limits come from the existing physics model, not calibration ticks)
- [x] **TWIN-06**: The MuJoCo XML (`LIBERO/libero/libero/assets/robots/soarm101/robot.xml`) is brought into agreement with the corrected URDF's kinematic structure
- [x] **TWIN-07**: Driving the simulated gripper with a given joint command opens/closes it in the same direction as the real gripper under the same command

### VLA Hardware Connection

- [x] **VLAHW-01**: An SO-101-native joint-action VLA (SmolVLA — matches the benchmark paper's tested architecture and avoids the Cartesian/IK detour the sim OpenVLA-OFT path would require) is loaded and produces valid 6D joint-position actions when called with real wrist-camera + overhead-camera + joint-state observations; the action contract (joint order, units — degrees vs. normalized, gripper scale, absolute vs. relative) is explicitly documented before first use
- [x] **VLAHW-02**: VLA output actions pass a safety validator (joint limits, max per-step displacement, max velocity, gripper bounds, stale observation/response rejection, malformed/NaN/infinite action rejection, e-stop, servo comms-failure handling) before being sent to the real robot via the existing `control/` LeRobot `SO101Follower` interface (not hand-written raw serial code); the pre-existing `use_degrees` vs. normalized-value ambiguity in the LeRobot config path is resolved and documented first
- [x] **VLAHW-03**: Every inference step's complete I/O is captured to a durable log — raw camera frames (with per-camera timestamps, not sequential reads that can drift out of sync), joint state, task instruction, raw model output, validated action, executed action, latency, model version/checkpoint — not just the final executed action (VLAs like SmolVLA map observations directly to actions; there is no natural-language reasoning trace to capture the way there would be for an LLM)
- [x] **VLAHW-04**: At least one full episode (VLA driving the robot from a task prompt to termination) is recorded end-to-end — synchronized video plus the full per-step I/O log from VLAHW-03, plus termination reason and success/failure outcome
- [x] **VLAHW-05**: Findings from the observed run(s) — including whether the known upstream SmolVLA/SO-101 issue ([lerobot#2210](https://github.com/huggingface/lerobot/issues/2210)) reproduces — are written up to inform the go/no-go decision on later milestone phases (deep-reasoning MLLM comparison, raw-autonomy design)

### Future Requirements (deferred pending v2.0's early results)

<!-- Carried forward from the original v2.0 scope draft (2026-09-15). Not dropped — sequenced behind Digital-Twin Fidelity and VLA Hardware Connection above. -->

- Comparable experiment run against a genuine deep-reasoning multimodal model (e.g. Claude, paid tier) using the same reasoning-trace-capture harness as the VLA experiment
- "Raw autonomy" design: model composes its own control functions from floor-level I/O only (no task-level primitives), with servo-onboard torque-limit/overload protection + sandboxed execution of AI-generated code as the safety net (no ESP32 — 12V/5A supply exceeds its safe input rating without extra regulation)
- Provider-agnostic MLLM router (OpenAI/Anthropic/Gemini/HF-hosted), piloted on a free/cheap-tier HF model before paid providers
- Extended synced episode recorder (wrist RGB + raw depth + joint state + reasoning trace + action) as the benchmark dataset format
- Task suite mirroring Yu & Qiu 2026 (arXiv:2606.08881): Pen Transfer, Selective Color Sorting, Multi-Object Packing, Precision Pen Placement
- Failure taxonomy (Grasp Instability, Repetition Loop, State Mismatch, Precision Misalignment) + semantic/execution aggregation + Recovery Rate metric; episode termination modeled on the paper (goal-met / timeout / irreversible-failure / unrecoverable-stagnation), human-judged initially (no automated vision-based detection yet)
- Cross-episode memory ablation (paper-faithful baseline uses independent episodes; memory-as-variable is a later, separate experiment)

## v2.1 Requirements

Requirements for milestone v2.1 (MLLM Raw-Autonomy Benchmark). Builds on v2.0's real-hardware VLA connection (Phase 11). Fixes a diagnosed bridge tick-latency bug and a camera device-resolution bug found during v2.1 scoping, then builds two consistent action-source pipelines — VLA-style (native trained action-output) and MLLM-style (raw-JSON reasoning prompt, no movement primitives) — and runs all backends against Pen Transfer (Yu & Qiu 2026, arXiv:2606.08881) for a cross-backend comparison.

> **Note:** This section's VLA Backends requirements were renamed to `VLAB-01`..`VLAB-04` (2026-09-24) to avoid an ID collision with the v1 section's `VLA-01`..`VLA-04` (VLA Inference Pipeline, Phase 3, above) — both sets are semantically distinct (v1 = OpenVLA-OFT sim inference loop; v2.1 = 4 new real-hardware VLA backends).

### Bridge Latency Fix

- [ ] **LATENCY-01**: The bridge only requests fresh Colab inference when the local action queue is empty/near-empty (wiring up `lerobot`'s existing, currently-unused `_ready_to_send_observation()`/`chunk_size_threshold` gate), instead of resending a full camera observation to Colab on every control tick
- [ ] **LATENCY-02**: `pop_validated_action()` updates `client.latest_action` so the vendored library's own staleness dedup logic functions as designed
- [X] **LATENCY-03**: `io_logger.py`'s `latency_ms` field records real measured latency (via `time.monotonic()` deltas) instead of always `{0,0}` — Complete in Phase 12 Plan 02: `run_episode()`'s t0/t1/t2 `time.monotonic()` brackets feed real `observation_to_action`/`action_to_execution` millisecond deltas into `io_logger.write_step()`, proven exact under a controlled clock (2026-09-25)
- [ ] **LATENCY-04**: An in-flight-request guard prevents duplicate/overlapping observation requests when Colab responds slowly

### Safety Validator

- [ ] **SAFETY-01**: `safety_validator.py`'s caps (`MAX_RELATIVE_TARGET_DEG`, `MAX_VELOCITY_DEG_PER_S`, `STALE_OBSERVATION_S`, `STALE_ACTION_S`) are incrementally re-tightened toward their original conservative defaults, with each step confirmed via a live-hardware re-verification episode (not assumed safe as a blind revert)

### Camera Device Resolution

- [ ] **CAMFIX-01**: A single shared, name-based camera-resolution helper is used by both the model's inference-input path and the `IOLogger` recording path (reusing one `StereoSplitCamera` instance, not two independent camera opens)
- [ ] **CAMFIX-02**: `camera_overhead` recordings are visually confirmed by a human to show the robot workspace (not the laptop webcam) on a live re-verification episode
- [ ] **CAMFIX-03**: The bridge logs a hash/thumbnail of the frame it actually sent to Colab per step, explicitly disclosed as client-side-only provenance (not proof of what Colab received/used)

### Pipeline Architecture

- [ ] **PIPE-01**: A consistent VLA-style `ActionSource` pattern (native trained action-output, one adapter per backend) exists, that the already-working SmolVLA integration fits and new VLA backends plug into without changing the shared control loop/safety/logging path
- [ ] **PIPE-02**: A consistent MLLM-style `ActionSource` pattern exists (shared prompt construction, output JSON schema, and reasoning-trace capture) that all MLLM backends plug into without changing the shared control loop/safety/logging path

### VLA Backends (native action-output, paper-matched)

- [ ] **VLAB-01**: Gemini Robotics is integrated as a VLA-style `ActionSource` backend driving the real SO-ARM101
- [ ] **VLAB-02**: π0/π0.5 is integrated as a VLA-style `ActionSource` backend driving the real SO-ARM101
- [ ] **VLAB-03**: Wall-X is integrated as a VLA-style `ActionSource` backend driving the real SO-ARM101
- [ ] **VLAB-04**: ACT is integrated as a VLA-style `ActionSource` backend driving the real SO-ARM101

### MLLM Backends (raw-JSON reasoning prompt, no movement primitives)

- [ ] **MLLM-01**: A free/cheap HuggingFace-hosted reasoning model is integrated as the first MLLM-style backend
- [ ] **MLLM-02**: Claude (paid tier) is integrated as an MLLM-style backend
- [ ] **MLLM-03**: GPT is integrated as an MLLM-style backend
- [ ] **MLLM-04**: Each MLLM backend's action-chunk JSON schema is generated programmatically from `action_contract.py`'s constants (`JOINT_ORDER`, units) and validated through the unmodified `safety_validator.py`
- [ ] **MLLM-05**: Action-chunk size for MLLM backends is derived from Phase 1's real measured latency/timing data, not guessed
- [ ] **MLLM-06**: Each MLLM backend's staged reasoning trace (restate task → plan → grounded scene claims → action) is captured alongside its action chunk, per step
- [ ] **MLLM-07**: A malformed/parse-failing action within a chunk holds the robot's current position rather than zero-filling or crashing the episode

### Cross-Backend Benchmark

- [ ] **EVAL-01**: One live Pen Transfer episode is run and recorded end-to-end (I/O log + camera footage + termination reason) for each of the 8 backends (SmolVLA baseline + Gemini Robotics + π0/π0.5 + Wall-X + ACT + HF reasoning model + Claude + GPT)
- [ ] **EVAL-02**: All 8 episodes are directly compared, human-judged, using Yu & Qiu 2026's 4-category failure taxonomy (Grasp Instability, Repetition Loop, State Mismatch, Precision Misalignment) where applicable

### Tech Debt (from v2.0 audit)

- [ ] **DEBT-01**: `action_contract.py`'s `WRIST_ROLL_LIMIT_DEG` is pinned to Phase 10's `robot.xml` wrist_roll value with an automated test (or SHA-pinned comment)
- [X] **DEBT-02**: `soarm_gripper.py`'s `format_action` docstring is corrected (currently mislabels OPEN/CLOSE direction post-polarity-fix) — Not applicable: already fixed in commit `1bc5f54` (2026-09-20), predating this phase's scoping. Docstring already reads "External +1 opens and -1 closes", matching the code. Confirmed via `git log` and direct file read during Phase 12 plan-phase (2026-09-25); see `12-CONTEXT.md` D-04.
- [X] **DEBT-03**: The untracked `So-101/` and `coppelia/` mesh asset directories are added to git so the URDF's relative mesh references resolve on a fresh clone — Not applicable: `So-101/` is already fully tracked (39 files); no `coppelia/` mesh directory exists or is referenced by the URDF. "Coppelia" refers to an unrelated one-time local diagnostic reference (see `diagnostics/COPPELIA_MUJOCO_ALIGNMENT.md`). Confirmed during Phase 12 discuss-phase (2026-09-25); see `12-CONTEXT.md` D-03.

## Out of Scope

| Feature | Reason |
|---------|--------|
| Real-3DQA point cloud as training input | Inspiration only; separate research thread not connected to this pipeline |
| RL training in simulation | High compute cost, out of scope for VLA fine-tuning approach |
| Real-time interactive REPL (v1) | Deferred; rendered video output sufficient for pipeline validation |
| robosuite 1.5+ | Hard incompatibility with LIBERO (SingleArmEnv removed) |
| Free Colab T4 for OpenVLA-7B inference | T4 = 15GB; OpenVLA-7B needs ~16GB+; requires Colab Pro A100 |
| New microcontroller/embedded hardware (ESP32) for the v2.0 control path | Existing USB-serial LeRobot bridge already covers arm control; ESP32 remains a possible future physical kill-switch, not a control-path component |
| Fine-tuning any policy on v2.0-collected data | This milestone evaluates VLA/MLLM behavior via inference/prompting, not training |
| Arbitrary code-as-policy execution without a sandboxing/safety design | Superseded (2026-09-24): MLLM backends emit validated JSON through the unmodified `safety_validator.py`, never execute code, so no sandboxing design is needed |
| Provider-agnostic MLLM router beyond the 3 v2.1 backends (OpenAI/Anthropic/HF only) | Deferred; v2.1 proves out the MLLM-style pattern with 3 concrete backends before generalizing to a full router |
| Full 4-task suite beyond Pen Transfer (Selective Color Sorting, Multi-Object Packing, Precision Pen Placement) | Deferred to a later milestone once the 8-backend Pen Transfer comparison validates both pipelines |
| Recovery Rate metric automation / automated vision-based success detection | Deferred; v2.1 uses human-judged outcomes per the paper's own first-version approach |

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
| DATA-05 | Phase 9 (paused) | Pending |
| TUNE-05 | Phase 9 (paused) | Pending |
| TUNE-06 | Phase 9 (paused) | Pending |
| TWIN-01 | Phase 10 | Complete |
| TWIN-02 | Phase 10 | Complete |
| TWIN-03 | Phase 10 | Complete |
| TWIN-04 | Phase 10 | Complete |
| TWIN-05 | Phase 10 | Complete |
| TWIN-06 | Phase 10 | Complete |
| TWIN-07 | Phase 10 | Complete |
| VLAHW-01 | Phase 11 | Complete |
| VLAHW-02 | Phase 11 | Complete |
| VLAHW-03 | Phase 11 | Complete |
| VLAHW-04 | Phase 11 | Complete |
| VLAHW-05 | Phase 11 | Complete |
| LATENCY-01 | Phase 12 | Pending |
| LATENCY-02 | Phase 12 | Pending |
| LATENCY-03 | Phase 12 | Complete |
| LATENCY-04 | Phase 12 | Pending |
| DEBT-02 | Phase 12 | Not applicable (already fixed pre-scoping) |
| DEBT-03 | Phase 12 | Not applicable (already tracked pre-scoping) |
| SAFETY-01 | Phase 17 | Pending |
| DEBT-01 | Phase 17 | Pending |
| CAMFIX-01 | Phase 13 | Pending |
| CAMFIX-02 | Phase 13 | Pending |
| CAMFIX-03 | Phase 13 | Pending |
| PIPE-01 | Phase 14 | Pending |
| VLAB-01 | Phase 14 | Pending |
| VLAB-02 | Phase 14 | Pending |
| VLAB-03 | Phase 14 | Pending |
| VLAB-04 | Phase 14 | Pending |
| PIPE-02 | Phase 15 | Pending |
| MLLM-01 | Phase 15 | Pending |
| MLLM-02 | Phase 15 | Pending |
| MLLM-03 | Phase 15 | Pending |
| MLLM-04 | Phase 15 | Pending |
| MLLM-05 | Phase 15 | Pending |
| MLLM-06 | Phase 15 | Pending |
| MLLM-07 | Phase 15 | Pending |
| EVAL-01 | Phase 16 | Pending |
| EVAL-02 | Phase 16 | Pending |

**Coverage:**

- v1 requirements: 24 total
- v1.1 requirements: 13 total (paused, unmapped to active phases pending resume)
- v2.0 requirements: 12 total (TWIN ×7, VLAHW ×5)
- v2.1 requirements: 26 total (LATENCY ×4, SAFETY ×1, CAMFIX ×3, PIPE ×2, VLA ×4, MLLM ×7, EVAL ×2, DEBT ×3)
- Mapped to phases: 24 (v1) + 13 (v1.1) + 12 (v2.0) + 26 (v2.1) = 75
- Unmapped: 0 (v1) ✓ / 0 (v1.1) ✓ / 0 (v2.0) ✓ / 0 (v2.1) ✓

---
*Requirements defined: 2026-07-07*
*Last updated: 2026-09-24 — mapped v2.1 requirements (26 total) to Phases 12-17 via roadmap creation: LATENCY-01..04 + DEBT-02/03 → Phase 12 (Bridge Tick-Latency Fix); CAMFIX-01..03 → Phase 13 (Camera Device Resolution Unification); PIPE-01 + VLAB-01..04 → Phase 14 (VLA-Style Pipeline & Backends); PIPE-02 + MLLM-01..07 → Phase 15 (MLLM-Style Pipeline & Backends); EVAL-01..02 → Phase 16 (Cross-Backend Pen Transfer Benchmark); SAFETY-01 + DEBT-01 → Phase 17 (Safety-Validator Cap Re-Tightening). Phases 12-13 form a chain; Phases 14 and 15 are independent/parallel-capable (different pipeline patterns); Phase 16 depends on both. Phase 17 is deliberately sequenced LAST (resequenced 2026-09-24 per user decision) so loosened soft-margin safety caps stay in place through the whole experiment rather than being re-tightened before it. The v2.1 VLA Backends requirements were renamed VLA-01..04 → VLAB-01..04 to remove a collision with v1's pre-existing VLA-01..04 (VLA Inference Pipeline, Phase 3).*
