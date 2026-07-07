# Architecture Research

**Domain:** VLA + LIBERO/MuJoCo robot simulation pipeline
**Researched:** 2026-07-07
**Confidence:** MEDIUM (cross-checked codebase inspection + web research; open questions noted)

---

## Standard Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Researcher / Notebook                        │
│              (Google Colab GPU: language prompt + launch)            │
└─────────────────────────┬───────────────────────────────────────────┘
                          │ text prompt
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       VLA Inference Layer                            │
│   ┌──────────────────────────────────────────────────────────────┐  │
│   │  π0 or OpenVLA model (HuggingFace weights, GPU inference)    │  │
│   │  Input: [language tokens] + [RGB image(s)] → action tokens   │  │
│   │  Output: 7-D delta EEF pose (dx,dy,dz,droll,dpitch,dyaw,g)  │  │
│   │  or 50-step action chunk (π0 flow matching)                  │  │
│   └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────┬──────────────────────────────────────────┘
                          │ action vector (7-D float array)
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     Controller / Action Bridge                        │
│   robosuite OSC_POSE controller                                      │
│   (translates delta EEF pose → joint torques via Jacobian IK)       │
│   Alternative for SOARM: JOINT_POSITION controller                   │
│   (if direct 6-DOF joint angle control preferred)                    │
└─────────────────────────┬──────────────────────────────────────────┘
                          │ joint torques / positions
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     Simulation Environment Layer                      │
│   ┌──────────────────┐   ┌──────────────────────────────────────┐   │
│   │ LIBERO BDDLBase  │   │ robosuite SingleArmEnv               │   │
│   │ Domain           │   │ (wraps MuJoCo physics)               │   │
│   │ (task + objects  │   │ env.step(action) → obs, reward, done │   │
│   │  from .bddl file)│   └──────────────────────────────────────┘   │
│   └──────────────────┘                │                             │
│                                       ▼                             │
│   ┌───────────────────────────────────────────────────────────────┐ │
│   │                   MuJoCo Physics Engine                       │ │
│   │  mj_step() → integrates physics, updates body poses          │ │
│   └───────────────────────────────────────────────────────────────┘ │
└─────────────────────────┬──────────────────────────────────────────┘
                          │ observations dict
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Observation / Camera Layer                       │
│  agentview_image (H×W×3 uint8, flipped [::-1] before VLA use)       │
│  robot0_eye_in_hand_image (wrist camera)                             │
│  frontview_image (optional third view)                               │
│  agentview_depth (float, camera_depths=True)                         │
│  robot0_eef_pos (3,), robot0_eef_quat (4,)                          │
│  robot0_joint_pos (N,), robot0_gripper_qpos (2,)                    │
│  object_state (xpos, xmat per object, from sim.data)                │
└─────────────────────────┬──────────────────────────────────────────┘
                          │ back to VLA (closed loop) or to HDF5
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Dataset Collection Layer (offline)                 │
│  Scripted / teleoperated episodes → HDF5 via robomimic format        │
│  /data/demo_N/obs/{images, eef_pos, joint_pos, gripper}             │
│  /data/demo_N/actions (T, 7) delta EEF or joint targets              │
│  Converts to LeRobot Parquet format for HuggingFace sharing         │
└─────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Implementation |
|-----------|---------------|----------------|
| SOARM MJCF Model | Defines robot kinematics, joint limits, meshes for MuJoCo | `so101_new_calib.xml` from TheRobotStudio/SO-ARM100 (adapt for robosuite) |
| SOARM ManipulatorModel | Python class bridging MJCF to robosuite robot system | Subclass `ManipulatorModel`, implement `init_qpos`, `base_xpos_offset`, `arm_type` |
| ROBOT_CLASS_MAPPING | Registry mapping robot name string to SingleArm class | LIBERO `envs/robots/__init__.py` — add `"SOARM": SingleArm` |
| BDDL Task File | Declarative task spec: arena, fixtures, objects, goal | `.bddl` file listing SOARM as robot, tabletop scene, target objects |
| BDDLBaseDomain | LIBERO environment class that parses BDDL and creates MuJoCo scene | Inherits `SingleArmEnv`; handles object placement + goal checking |
| ControlEnv / OffScreenRenderEnv | Gym-like wrapper: `reset()`, `step(action)`, camera obs | `LIBERO/libero/libero/envs/env_wrapper.py` |
| OSC_POSE Controller | Translates 6-D delta EEF pose + gripper to joint torques | robosuite built-in; loaded via `suite.load_controller_config("OSC_POSE")` |
| VLA Model (OpenVLA) | Tokenizes image + language → autoregressive action token decoding → 7-D float vector | `openvla/openvla-7b` HuggingFace checkpoint; outputs rescaled to `[-1, 1]` |
| VLA Model (π0) | Flow matching over 50-step action chunk from image + language | Physical Intelligence `openpi`; replans every 25 steps |
| Action Normalization | Rescales VLA output `[-1, 1]` to physical controller limits | Dataset-specific stats; must match training distribution |
| Observation Camera Stack | Multi-camera RGB + optional depth render | `OffScreenRenderEnv(camera_names=[...], camera_depths=True)` |
| HDF5 Dataset Writer | Persists episodes to robomimic HDF5 format | `h5py`; structure: `/data/demo_N/{obs, actions, rewards, dones}` |
| Spatial Awareness Module | Object pose extraction, depth-to-pointcloud, spatial predicate eval | `sim.data.body_xpos[id]`, camera intrinsics from robosuite `camera_utils.py` |

---

## Recommended Project Structure

```
LIBERO/libero/libero/
├── envs/
│   ├── robots/
│   │   ├── mounted_panda.py        # existing
│   │   ├── soarm.py                # NEW: ManipulatorModel subclass for SOARM
│   │   └── __init__.py             # add SOARM to ROBOT_CLASS_MAPPING
│   ├── assets/
│   │   └── robots/
│   │       └── soarm/
│   │           ├── robot.xml       # NEW: SOARM MJCF (adapted from SO-ARM100)
│   │           └── meshes/         # .obj/.stl mesh files
│   └── ...
│
soarm_pipeline/                     # NEW top-level package (in repo root)
├── models/
│   │   soarm.py                    # MJCF path, init_qpos, constants
├── envs/
│   └── soarm_libero_env.py         # OffScreenRenderEnv factory for SOARM tasks
├── vla/
│   ├── openvla_inference.py        # OpenVLA wrapper: load + predict()
│   ├── pi0_inference.py            # π0 wrapper: load + predict_chunk()
│   └── action_utils.py             # normalize/denormalize, chunk replay
├── data/
│   ├── collect_demos.py            # scripted policy + HDF5 writer
│   ├── dataset_schema.py           # HDF5 structure constants
│   └── convert_lerobot.py          # HDF5 → LeRobot Parquet
├── spatial/
│   ├── camera_utils.py             # depth backprojection, pointcloud
│   └── object_pose.py              # sim.data.body_xpos extraction
└── notebooks/
    ├── 01_soarm_env_check.ipynb    # verify SOARM renders in LIBERO
    ├── 02_vla_inference.ipynb      # end-to-end VLA rollout
    └── 03_data_collection.ipynb    # demo collection pipeline
```

### Structure Rationale

- **envs/robots/soarm.py inside LIBERO tree:** robosuite looks for MJCF via path relative to its own assets directory; keeping the robot class adjacent to MountedPanda follows the existing LIBERO pattern exactly.
- **soarm_pipeline/ at repo root:** Research code that orchestrates the full loop lives outside LIBERO to avoid polluting the benchmark fork. Easier to iterate without rebasing against upstream LIBERO changes.
- **vla/ subdirectory:** OpenVLA and π0 have different APIs (token decoding vs flow matching); isolating them behind a common `predict(image, language) -> action` interface makes the inference loop swappable.
- **spatial/ subdirectory:** Depth and pose extraction are optional augmentations to the base loop; separating them prevents scope creep in the core rollout code.

---

## Architectural Patterns

### Pattern 1: Robot Model Inheritance Chain

**What:** SOARM is wired into robosuite via a three-level class hierarchy: `ManipulatorModel` (MJCF + kinematics) → `SingleArm` (robot controller management) → `ROBOT_CLASS_MAPPING` (string-to-class registry). LIBERO already overrides this registry in its own `robots/__init__.py`.

**When to use:** Required for every new robot added to LIBERO. The pattern is followed by `MountedPanda` and `OnTheGroundPanda`.

**Example:**
```python
# LIBERO/libero/libero/envs/robots/soarm.py
from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.utils.mjcf_utils import xml_path_completion
import numpy as np

class SOARM(ManipulatorModel):
    def __init__(self, idn=0):
        # robot.xml lives at LIBERO/libero/libero/envs/assets/robots/soarm/robot.xml
        super().__init__(xml_path_completion("robots/soarm/robot.xml"), idn=idn)

    @property
    def default_mount(self):
        return "RethinkMount"

    @property
    def default_gripper(self):
        return "PandaGripper"  # or custom SOARM gripper

    @property
    def default_controller_config(self):
        return "default_panda"  # or "default_soarm" with custom YAML

    @property
    def init_qpos(self):
        # 6 joints for SO101; tune to home position
        return np.array([0.0, -0.3, 0.0, -1.5, 0.0, 1.2])

    @property
    def base_xpos_offset(self):
        return {
            "table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
        }

    @property
    def arm_type(self):
        return "single"

# LIBERO/libero/libero/envs/robots/__init__.py (add):
ROBOT_CLASS_MAPPING.update({"SOARM": SingleArm})
```

### Pattern 2: Closed-Loop VLA Inference

**What:** At each control step, the env observation (image array, robot state) is packaged and sent to the VLA, which returns a raw action vector that gets applied to the environment. The loop runs at the controller frequency (20 Hz for LIBERO's OSC_POSE).

**When to use:** The core rollout pattern for all VLA evaluation and dataset augmentation.

**Example:**
```python
env = OffScreenRenderEnv(bddl_file_name=task, robots=["SOARM"],
                         camera_names=["agentview", "robot0_eye_in_hand"],
                         camera_heights=256, camera_widths=256)
obs = env.reset()

for _ in range(horizon):
    # MuJoCo images are vertically flipped — correct before VLA
    image = obs["agentview_image"][::-1]
    # VLA inference (OpenVLA example)
    action = vla.predict(image=image, instruction=language_prompt)
    # action shape: (7,) — [dx, dy, dz, droll, dpitch, dyaw, gripper]
    obs, reward, done, _ = env.step(action)
    if done:
        break
```

### Pattern 3: Action Chunking Replay (π0)

**What:** π0 generates a 50-step chunk at once via flow matching. The caller executes the first H/2 steps before requesting a new chunk, producing smoother control than per-step inference.

**When to use:** π0 inference exclusively. Adds latency per chunk but reduces jitter.

**Example:**
```python
chunk = pi0_model.predict_chunk(image=image, instruction=language_prompt)
# chunk.shape: (50, 7)
for i in range(25):  # execute half the chunk, then replan
    obs, reward, done, _ = env.step(chunk[i])
    if done:
        break
```

### Pattern 4: HDF5 Demonstration Recording

**What:** Each demo episode is recorded as a group in an HDF5 file, following the robomimic schema. LIBERO's lifelong learning system consumes this format directly via `SequenceDataset`.

**When to use:** Any scripted or teleoperated collection run.

**Example:**
```python
import h5py
with h5py.File("soarm_demos.hdf5", "w") as f:
    grp = f.create_group(f"data/demo_{ep_idx}")
    grp.attrs["num_samples"] = len(actions)
    obs_grp = grp.create_group("obs")
    obs_grp.create_dataset("agentview_image", data=images)    # (T, 256, 256, 3)
    obs_grp.create_dataset("robot0_eef_pos", data=eef_pos)   # (T, 3)
    grp.create_dataset("actions", data=actions)               # (T, 7)
    grp.create_dataset("rewards", data=rewards)               # (T,)
    grp.create_dataset("dones", data=dones)                   # (T,)
    f["data"].attrs["env_args"] = json.dumps(env_meta)
    f["data"].attrs["problem_info"] = json.dumps(problem_info)
```

---

## Data Flow

### Primary Inference Flow (Language → Action → Simulation)

```
Researcher types: "Pick up the bowl and place it on the plate"
       │
       ▼ tokenize (VLA tokenizer)
[Language tokens]───────────────────────┐
                                        │
[Camera frame from env.reset()]         │
  obs["agentview_image"] shape (256,256,3)
  ↓ flip vertically [::-1]              │
  ↓ resize / normalize per VLA spec     │
[Image tokens]──────────────────────────┤
                                        ▼
                              VLA forward pass (GPU)
                              OpenVLA: autoregressive decode 7 action tokens
                              π0: flow matching denoising (50-step chunk)
                                        │
                              [raw action: 7 floats in [-1, 1]]
                                        │
                              Action unnormalization
                              (scale by dataset action stats)
                                        │
                              [physical action: delta EEF pose + gripper]
                                        │
                              OSC_POSE controller (Jacobian IK)
                                        │
                              [joint torques]
                                        │
                              MuJoCo mj_step()
                                        │
                              [new body poses, sim state]
                                        │
                              Observation rendering
                              (offscreen render → RGB + depth arrays)
                                        │
                              obs dict → back to VLA (next step)
```

### Dataset Collection Flow (Demos → HDF5)

```
Scripted policy or human teleop
       │
       ▼
env.reset()  →  obs_t0
       │
       ▼ (for each timestep t)
[obs_t: images, eef_pos, joint_pos, gripper]
[action_t: 7-D vector from policy/teleop]
env.step(action_t) → obs_t+1, reward_t, done_t
       │
       ▼
Buffer: episode_obs[], episode_actions[], episode_rewards[]
       │
       ▼ (on episode end)
Write to HDF5 group /data/demo_N/
       │
       ▼
(optional) Convert to LeRobot Parquet via lerobot CLI
```

### Spatial Awareness Flow (Depth + Object Pose)

```
OffScreenRenderEnv(camera_depths=True, camera_segmentations="instance")
       │
       ▼
obs["agentview_depth"]    (H, W, 1) float meters
obs["agentview_segmentation_instance"]  (H, W, 1) int object IDs
       │
       ▼
Backproject depth + segmentation → 3D point cloud per object
       │                (camera intrinsics from robosuite camera_utils.py)
       ▼
Ground truth pose: sim.data.body_xpos[sim.model.body_name2id("bowl_1")]
Camera intrinsics: robosuite camera_utils.get_camera_intrinsic_matrix()
Camera extrinsics: sim.data.cam_xpos[cam_id], sim.data.cam_xmat[cam_id]
       │
       ▼
Spatial predicate evaluation:
"Is bowl to the left of the plate?" → compare world-frame xpos
"Is bowl near the wall?" → compare distance to scene boundary
```

---

## SOARM-Specific Integration Points

### 1. MJCF Source: Use Existing SO101 Model

The SO-ARM100 repository (`TheRobotStudio/SO-ARM100`) already provides validated MuJoCo MJCF files (`Simulation/SO101/so101_new_calib.xml`). These were generated via onshape-to-robot from verified CAD. Use these as the base; do not attempt URDF-from-scratch.

**Adaptation needed for robosuite:**
- robosuite expects the MJCF root body to have specific site names (`grip_site`, actuator group)
- The SOARM model needs a `<mujoco>` configuration block with compiler settings matching robosuite's conventions
- Base collision meshes may need removal (SO-ARM100 README notes they cause simulation issues)

### 2. Controller Choice: OSC_POSE vs JOINT_POSITION

This is the critical architectural decision for SOARM:

| Choice | Pros | Cons |
|--------|------|------|
| `OSC_POSE` (default LIBERO) | Works with OpenVLA/π0 outputs directly (7-D EEF delta); no remapping needed | Requires full Jacobian; SOARM's 6-DOF may create singularities |
| `JOINT_POSITION` | More direct control; matches SOARM's actual servo interface | VLA output needs IK layer; action space is 6-D not 7-D |

**Recommendation:** Start with `OSC_POSE` to stay compatible with pretrained OpenVLA/π0 checkpoints. If control quality is poor (singularities, oscillation), switch to `JOINT_POSITION` with an IK bridge.

### 3. Action Dimension Mismatch

Panda is 7-DOF arm + gripper. SOARM SO101 is 6-DOF arm + linear gripper. OpenVLA outputs 7-D (6D EEF delta + 1D gripper). This is fine with OSC_POSE since the controller solves IK regardless of arm DOF. With JOINT_POSITION, need a 6-D output VLA or append a 7th dummy dim.

### 4. Robot Registration: Only Two Files to Modify

Following the LIBERO pattern (MountedPanda):
1. Create `LIBERO/libero/libero/envs/robots/soarm.py` with `SOARM(ManipulatorModel)`
2. Add to `LIBERO/libero/libero/envs/robots/__init__.py`: `ROBOT_CLASS_MAPPING.update({"SOARM": SingleArm})`

The BDDL task files then reference `robots=["SOARM"]`.

### 5. BDDL Task Registration

New tasks for SOARM go in:
- `LIBERO/libero/libero/bddl_files/libero_soarm/task_name.bddl`

Each file needs `(:domain robosuite)` and a language instruction. The `TASK_MAPPING` registration happens via `@register_problem` decorator on the domain class (following the existing LIBERO pattern).

---

## Anti-Patterns

### Anti-Pattern 1: Building SOARM MJCF from Scratch

**What people do:** Write MJCF from hardware specs (motor torques, link lengths) without validated reference.

**Why it's wrong:** MuJoCo MJCF tuning (friction, damping, joint limits) requires iterative physical validation. The SO-ARM100 repo provides a community-tested model — bypassing it adds weeks of model tuning with no research payoff.

**Do this instead:** Fork `TheRobotStudio/SO-ARM100/Simulation/SO101/so101_new_calib.xml` and adapt for robosuite's conventions. The adaptation is ~50 lines of XML, not a rebuild.

### Anti-Pattern 2: Running VLA Inference Inside the Physics Step

**What people do:** Call the VLA model synchronously on every `mj_step()` tick.

**Why it's wrong:** VLA inference on GPU takes 50-200ms per step. MuJoCo at 20 Hz expects 50ms step budgets. Blocking the sim on VLA inference creates artificial slow-motion execution that doesn't reflect real robot timing and makes dataset collection impractical.

**Do this instead:** Decouple the control frequency. Run VLA at 3-5 Hz (every 4-6 sim steps); hold the last action constant between VLA calls. This is how OpenVLA-OFT achieves 26x throughput improvement. For Colab notebooks, async is impractical — use action chunking (π0) or step-skipping (OpenVLA).

### Anti-Pattern 3: Forgetting the Vertical Image Flip

**What people do:** Pass `obs["agentview_image"]` directly to the VLA.

**Why it's wrong:** MuJoCo offscreen rendering returns images with origin at bottom-left (OpenGL convention). VLMs expect origin at top-left. The image is upside-down. This is confirmed in the existing `explorations/create_scene.py` code.

**Do this instead:** Always apply `image = obs["agentview_image"][::-1]` before any VLA or visualization use. Add this as a utility function so it can't be missed.

### Anti-Pattern 4: Single Camera VLA Input

**What people do:** Feed only the agentview image to the VLA.

**Why it's wrong:** Spatial awareness (left/right/behind, depth estimation) requires multiple viewpoints. Wrist camera (robot0_eye_in_hand) is essential for grasp precision. Training data from Open X-Embodiment used wrist + third-person cameras; VLA performance degrades significantly with fewer views.

**Do this instead:** Always instantiate with both `agentview` and `robot0_eye_in_hand` cameras. When training from SOARM demos, record both.

### Anti-Pattern 5: Using robosuite 1.5 with LIBERO

**What people do:** Upgrade robosuite to 1.5 while keeping LIBERO's existing code.

**Why it's wrong:** `SingleArmEnv` was removed in robosuite 1.5. LIBERO's `BDDLBaseDomain` extends `SingleArmEnv`. This breaks all LIBERO environments (confirmed by GitHub issue #49 in LIBERO repo).

**Do this instead:** Stay pinned to robosuite 1.4.x (LIBERO's `requirements.txt` pins `robosuite==1.4.0`). Do not upgrade.

---

## Integration Points

### External Services

| Service | Integration Pattern | Notes |
|---------|---------------------|-------|
| HuggingFace Hub | `from_pretrained("openvla/openvla-7b")` | Requires internet; cache weights in Colab `/content/` |
| Google Colab GPU | Runtime environment for VLA inference | T4 works for OpenVLA inference; A100 needed for π0 + fine-tuning |
| SO-ARM100 GitHub | Source of validated MJCF files | One-time download; pin to a specific commit |
| openpi (Physical Intelligence) | `pip install openpi`; checkpoint download | Docker eval wrapper provided; strip for Colab use |
| LeRobot | `pip install lerobot`; `lerobot convert` CLI | For Parquet/HF-compatible dataset export |

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| SOARM MJCF ↔ robosuite ManipulatorModel | MJCF loaded via `xml_path_completion()` path string | Path must resolve relative to robosuite assets directory |
| OffScreenRenderEnv ↔ VLA model | NumPy `uint8` array `(H, W, 3)` after `[::-1]` flip | VLA-specific resize/normalize happens inside VLA wrapper |
| VLA action ↔ OSC_POSE controller | NumPy `float32` array `(7,)` in unnormalized physical units | Normalization stats must match training data distribution |
| HDF5 dataset ↔ LIBERO SequenceDataset | h5py file with robomimic schema | `env_args` attrs required; must include `problem_info` JSON |
| Depth camera ↔ spatial module | NumPy `float32` depth array + camera intrinsics dict | Use robosuite `camera_utils.get_camera_intrinsic_matrix()` |

---

## Suggested Build Order

The following ordering respects hard dependencies — each phase produces an artifact required by the next.

1. **SOARM MJCF + Robot Class** (foundation)
   - Artifact: `soarm.py` ManipulatorModel + validated `robot.xml` rendering in MuJoCo
   - Dependency: Nothing (this is the root dependency)
   - Validation: `env.render()` shows SOARM arm in LIBERO scene

2. **SOARM LIBERO Environment** (environment layer)
   - Artifact: `OffScreenRenderEnv` with SOARM producing observations dict
   - Dependency: Phase 1 (robot class must exist)
   - Validation: `obs["agentview_image"]` has correct shape; zero-action steps don't crash

3. **VLA Inference Loop** (closed-loop control)
   - Artifact: Colab notebook running VLA rollout on SOARM LIBERO tasks
   - Dependency: Phase 2 (working env required)
   - Validation: SOARM moves (even random-looking) in response to VLA output

4. **Dataset Collection** (data infrastructure)
   - Artifact: HDF5 file with 50+ SOARM demonstrations
   - Dependency: Phase 2 (env) + scripted policy (can precede VLA)
   - Validation: `dataset_utils.get_dataset_info()` reports expected structure

5. **Spatial Awareness: Multi-camera + Depth** (perception layer)
   - Artifact: Per-step object poses + depth pointclouds
   - Dependency: Phase 2 (env); can be developed in parallel with Phase 3-4
   - Validation: Extracted bowl pose matches visual position in rendered frame

6. **Fine-Tuning Pipeline** (learning loop)
   - Artifact: Fine-tuned VLA checkpoint on SOARM demonstrations
   - Dependency: Phase 4 (need dataset) + Phase 3 (need baseline VLA)
   - Validation: Fine-tuned model outperforms zero-shot on SOARM tasks

7. **Evaluation Benchmark** (measurement)
   - Artifact: Task success rates across SOARM task suite
   - Dependency: All prior phases
   - Validation: Spatial task success rates improve vs. non-spatial baseline

---

## Sources

- Codebase inspection: `LIBERO/libero/libero/envs/robots/__init__.py`, `envs/env_wrapper.py`, `envs/bddl_base_domain.py`, `utils/dataset_utils.py` — MEDIUM confidence (verified directly)
- TheRobotStudio/SO-ARM100 repository (Simulation/SO101/README.md) — MEDIUM confidence (web, cross-checked)
- OpenVLA action space: 7-DOF delta EEF, confirmed via HuggingFace model card and arxiv 2502.19645 — MEDIUM confidence (web)
- π0 action chunking: 50-step chunk at 50Hz via flow matching, replan every 25 steps — MEDIUM confidence (pi.website/download/pi0.pdf and HuggingFace blog)
- robosuite OSC_POSE controller + robot registration pattern — MEDIUM confidence (robosuite.ai docs + codebase)
- HDF5 robomimic schema — MEDIUM confidence (robomimic.github.io docs + codebase `dataset_utils.py`)
- MuJoCo depth + object pose extraction — MEDIUM confidence (web, MuJoCo docs)
- robosuite 1.4→1.5 SingleArmEnv removal: LIBERO GitHub issue #49 — HIGH confidence (direct issue)

---

*Architecture research for: VLA + LIBERO/MuJoCo robot simulation pipeline (SoARM)*
*Researched: 2026-07-07*
