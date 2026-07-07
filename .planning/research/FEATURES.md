# Feature Research

**Domain:** VLA Robot Simulation Research Pipeline (spatial-aware manipulation)
**Researched:** 2026-07-07
**Confidence:** MEDIUM (web sources, cross-checked against multiple arxiv papers and official repos)

## Feature Landscape

### Table Stakes (Researchers Expect These)

Features every VLA simulation pipeline must have. Missing any of these means the pipeline cannot function at all.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Robot URDF/MJCF model in simulation | No robot to control without it; LIBERO defaults to Panda — SOARM has no existing model | HIGH | No existing SOARM MJCF; must derive from hardware specs or CAD; common pitfall: inter-mesh collisions at initialization when converting URDF → MJCF |
| Single wrist/front camera observation | VLA requires visual input at each timestep; camera image is mandatory model input | LOW | LIBERO/robosuite already supports configurable cameras; start with one agentview camera |
| Language-conditioned task prompt input | Defines what "VLA" means; model receives text + image, outputs actions | LOW | OpenVLA accepts raw text strings; no special tokenization beyond the LLM tokenizer |
| Action space definition (joint or EEF delta) | VLA outputs must map to simulator actuators; mismatch causes overshoot/crash | MEDIUM | OpenVLA outputs delta end-effector poses (7-DoF: xyz, rpy, gripper); must define SOARM equivalent |
| VLA model loading and inference | The core computation: text + image → action vector | MEDIUM | OpenVLA 7B runs on Colab A100; 4-bit quantization allows T4 use without accuracy loss |
| Simulation step loop (observation → action → step) | The closed-loop control cycle; missing this = open-loop only | LOW | robosuite env.step() already handles this; must wire VLA output into it |
| Task success/failure detection | Without this, cannot evaluate anything | LOW | LIBERO task definitions include success predicates; reuse for SOARM tasks |
| Rendered video or frame output | Researchers need to see what happened; otherwise debugging is impossible | LOW | robosuite offscreen_renderer or MuJoCo viewer; Colab-compatible via matplotlib/imageio |
| Scripted demonstration collection | Foundation for all fine-tuning; no demos = no fine-tuning | MEDIUM | LIBERO provides scripted policy infrastructure; must adapt for SOARM kinematics |
| HDF5 or RLDS dataset storage | Standard format expected by OpenVLA/Octo training pipelines | LOW | LIBERO natively outputs HDF5; OpenVLA training expects RLDS; conversion script needed |

### Differentiators (What Makes This Research Novel)

Features that go beyond standard VLA eval and constitute the actual research contribution.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| SOARM robot model in LIBERO (new embodiment) | No prior work puts SoARM in LIBERO/MuJoCo; creates a new research artifact others can build on | HIGH | Requires deriving MJCF from SOARM hardware specs; key blocker for all downstream work |
| Multi-camera spatial setup (wrist + overhead + side) | Richer observation enables spatial reasoning the single-camera baseline cannot; SpatialVLA showed spatial encoding significantly improves task success | MEDIUM | robosuite supports multiple named cameras; must define extrinsic calibration between views |
| Depth-based 3D object localization | Converts pixel-space observations to 3D world coordinates; enables "left of the box" style commands | HIGH | MuJoCo provides depth buffer natively; need point cloud reconstruction from depth + camera intrinsics |
| Spatial language grounding in task prompts | Prompts like "pick up the cube to the left of the mug" test whether the VLA understands spatial relations; this is the research gap SpatialVLA and RoboPoint address | HIGH | Requires constructing task variants with spatial language; evaluation needs ground-truth object positions |
| SOARM-specific fine-tuning on collected demos | Adapts a generalist VLA (trained on Panda/UR5/etc.) to SOARM kinematics; without this, zero-shot performance is poor | MEDIUM | OpenVLA LoRA fine-tuning needs ~50-200 demos minimum; runs on Colab A100 with gradient checkpointing |
| Spatial understanding evaluation benchmark | Measures *how well* spatial prompts are understood, not just task completion; differentiates this from a standard LIBERO replication | MEDIUM | Derive from LIBERO-Spatial task suite; add SOARM-specific spatial variants |
| Ego3D position encoding integration (SpatialVLA approach) | Injects depth-derived 3D coordinates into VLM visual features; proven to improve spatial manipulation; research novelty if applied to SOARM | HIGH | Requires monocular depth model (e.g. Depth Anything v2) + camera intrinsics; increases inference cost |

### Anti-Features (Deliberately NOT Building in v1)

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| Real-time interactive REPL / live teleoperation UI | Feels like a natural interface for robot control | Doubles implementation complexity; Colab has latency; delays pipeline validation | Record-and-replay: collect demo, run inference, render video for review |
| Physical SOARM hardware integration | End goal of robotics research is real hardware | Sim pipeline must be validated first; sim-to-real gap is the largest failure mode and needs its own milestone | Simulation-only v1; hardware as explicit later milestone |
| RL training loop (PPO/SAC/etc.) | Generates diverse data without human teleoperation | Requires reward shaping, RL infra, and 10-100x more compute than imitation learning; major scope expansion | Scripted policy demonstrations + fine-tuning; RL as v2+ feature |
| Point cloud as primary VLA input | Richer 3D representation; Real-3DQA inspiration | VLAs are not trained on point clouds natively; requires architectural changes beyond fine-tuning; too heavy for Colab T4 | Use depth-derived 3D object *positions* as auxiliary signals, not raw point clouds as model input |
| Multi-robot coordination / mobile manipulation | Broadens applicability | SOARM is a fixed-base arm; adds complexity without advancing the spatial reasoning research question | Single-arm manipulation tasks covering all spatial dimensions |
| Custom physics simulation from scratch | Full control over dynamics | MuJoCo is already excellent for this; rebuilding is massive wasted effort | Tune MuJoCo contact/friction parameters for SOARM specifically |
| Web UI / dashboard for demos | Nice to share results | Not research-critical; building UI defers actual pipeline work | Colab notebook with inline video rendering is sufficient |
| Automatic speech recognition (voice prompts) | "Natural" language interface | ASR errors compound with VLA errors; debugging becomes impossible | Text prompts in notebook cells only |

## Feature Dependencies

```
SOARM MJCF/URDF model
    └──required by──> LIBERO environment with SOARM
                          └──required by──> Single-camera VLA inference loop
                                                └──required by──> Scripted demonstration collection
                                                                      └──required by──> SOARM fine-tuning pipeline
                                                                                            └──required by──> Spatial benchmark evaluation

Single-camera VLA inference loop
    └──required by──> Multi-camera spatial setup
                          └──required by──> 3D object localization from depth
                                                └──required by──> Spatial language grounding in prompts
                                                                      └──required by──> Ego3D position encoding (SpatialVLA)

SOARM fine-tuning pipeline
    └──enhances──> Spatial benchmark evaluation
```

### Dependency Notes

- **SOARM MJCF requires hardware specs first:** The robot model is the hardest unresolved dependency. Everything downstream depends on having a physically accurate SOARM description; a placeholder model (e.g., a generic 6-DoF arm) can unblock inference testing but will invalidate kinematics results.
- **Single-camera loop gating everything:** The end-to-end inference loop (single camera, no spatial features) must work before adding multi-camera complexity. Debugging a broken pipeline with 3 cameras and depth estimation is much harder.
- **Fine-tuning requires minimum demo count:** OpenVLA LoRA needs ~50-200 task demonstrations to adapt to a new embodiment. Collecting these on a placeholder/incorrect SOARM model wastes the demos.
- **Depth-based 3D localization requires camera calibration:** Extrinsic calibration between cameras (or between camera and robot base) must be defined before 3D coordinates are meaningful. In simulation this is accessible from the scene, but must be explicit.
- **Spatial language grounding requires spatial task definitions:** Can only evaluate spatial prompts once tasks have been constructed with spatial variation (object placement variants). This is a task *design* dependency, not just code.

## MVP Definition

### Launch With (v1 — Colab pipeline validation)

The minimum required to demonstrate the end-to-end loop from language prompt to SOARM simulation and begin collecting fine-tuning data.

- [ ] SOARM MJCF/URDF in robosuite/LIBERO environment — without this, nothing runs
- [ ] Single agentview camera observation feeding OpenVLA (7B, 4-bit quantized for T4) — proves inference loop
- [ ] Language prompt → VLA → SOARM joint/EEF actions → robosuite env.step() — the core loop
- [ ] Rendered video output per episode (Colab-friendly imageio/matplotlib) — validates results visually
- [ ] Scripted policy demonstrations for 3-5 SOARM tasks, stored as HDF5 — seed dataset for fine-tuning
- [ ] Task success detection reused from LIBERO predicates — enables quantitative eval

### Add After Validation (v1.x — spatial awareness layer)

- [ ] Multi-camera setup (wrist + overhead) — add after single-camera baseline succeeds
- [ ] Depth buffer extraction → 3D object position annotation — requires working camera calibration
- [ ] Spatial language task variants ("left of", "near the", "between") — requires task redesign
- [ ] OpenVLA LoRA fine-tuning on SOARM demos — requires 50+ collected demonstrations
- [ ] Spatial benchmark: success rate on spatial vs non-spatial prompt variants

### Future Consideration (v2+)

- [ ] Ego3D position encoding integration (SpatialVLA approach) — significant architectural work; research contribution but needs v1 baseline to compare against
- [ ] Physical SOARM hardware transfer — separate milestone after sim pipeline validated
- [ ] RLDS-format dataset export for Open X-Embodiment contribution — community contribution after pipeline stabilizes
- [ ] RL-generated data augmentation — only if scripted demonstrations prove insufficient for fine-tuning

## Feature Prioritization Matrix

| Feature | Research Value | Implementation Cost | Priority |
|---------|---------------|---------------------|----------|
| SOARM MJCF model | HIGH | HIGH | P1 — gates everything |
| End-to-end inference loop (single cam) | HIGH | MEDIUM | P1 — core pipeline |
| Scripted demonstration collection | HIGH | MEDIUM | P1 — enables fine-tuning |
| Rendered video output | MEDIUM | LOW | P1 — debugging necessity |
| Task success detection | HIGH | LOW | P1 — quantitative eval |
| Multi-camera setup | HIGH | MEDIUM | P2 — spatial research contribution |
| 3D object localization | HIGH | HIGH | P2 — required for spatial grounding |
| Spatial language task variants | HIGH | MEDIUM | P2 — the actual research question |
| OpenVLA LoRA fine-tuning | HIGH | MEDIUM | P2 — improves all metrics |
| Spatial benchmark evaluation | HIGH | MEDIUM | P2 — publishable result |
| Ego3D position encoding | HIGH | HIGH | P3 — novel contribution, deferred |
| RLDS dataset export | LOW | LOW | P3 — community contribution |
| Real hardware transfer | HIGH | HIGH | P3 — separate milestone |

**Priority key:**
- P1: Must have for pipeline to function (Colab MVP)
- P2: Should have — constitutes the actual spatial-awareness research contribution
- P3: Nice to have — future milestones or publications

## Related Pipeline Comparison

| Feature | Standard LIBERO eval (e.g. SemanticVLA) | SpatialVLA | This Project |
|---------|----------------------------------------|-----------|-------------|
| Robot embodiment | Panda | Multiple | SOARM (novel) |
| Camera setup | Single agentview | Single | Multi (wrist + overhead) |
| Spatial grounding | Language only | Ego3D encoding | Depth-based 3D + spatial prompts |
| Action representation | Delta EEF | Adaptive action grids | Delta EEF (OpenVLA-OFT style) |
| Fine-tuning | LoRA on existing robot | Pre-trained + fine-tune | LoRA on new SOARM embodiment |
| Compute platform | GPU server | GPU cluster | Google Colab (constraint) |
| Dataset format | HDF5/RLDS | RLDS | HDF5 → RLDS conversion |

## Sources

- [OpenVLA: An Open-Source Vision-Language-Action Model](https://arxiv.org/abs/2406.09246) — OpenVLA architecture and fine-tuning details
- [OpenVLA-OFT fine-tuning results on LIBERO](https://arxiv.org/abs/2502.19645) — 97.1% LIBERO benchmark, OFT recipe details
- [SpatialVLA: Exploring Spatial Representations for VLA Models](https://arxiv.org/abs/2501.15830) — Ego3D encoding and adaptive action grids
- [pi0: Our First Generalist Policy](https://physicalintelligence.company/blog/pi0) — Physical Intelligence VLA with flow matching
- [LIBERO benchmark task suite](https://www.emergentmind.com/topics/libero-object-benchmark) — Task suite structure (Spatial/Object/Goal/Long)
- [Grounding Sim-to-Real in Dexterous Manipulation](https://arxiv.org/html/2603.22876v1) — Sim-to-real pitfalls with VLA models
- [VLA Models: Concepts, Progress, Applications and Challenges](https://arxiv.org/html/2505.04769v1) — Broad VLA ecosystem survey
- [Teleoperation Data for Robot Learning (2026)](https://claru.ai/training-data/teleoperation) — Dataset collection methods and RLDS format
- [robosuite robot integration docs](https://robosuite.ai/docs/modules/robots.html) — URDF/MJCF custom robot integration
- [RoboPoint: Spatial affordance prediction](https://arxiv.org/pdf/2312.10807) — Spatial language grounding via instruction tuning

---
*Feature research for: VLA robot simulation pipeline with spatial awareness (SOARM/LIBERO/MuJoCo)*
*Researched: 2026-07-07*
