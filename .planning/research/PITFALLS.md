# Pitfalls Research

**Domain:** VLA + LIBERO/MuJoCo robot simulation (SOARM / OpenVLA / π0 / Google Colab)
**Researched:** 2026-07-07
**Confidence:** MEDIUM (cross-checked websearch against official docs and GitHub issues)

---

## Critical Pitfalls

These cause total failure — the pipeline does not run at all, or produces completely wrong output.

---

### Pitfall 1: OpenVLA VRAM Overrun on Colab T4

**What goes wrong:**
OpenVLA-7b requires approximately 21 GB VRAM in bfloat16. The free Colab T4 provides 15 GB. Loading the model either crashes with OOM or (worse, with `device_map="auto"`) silently spreads layers across CPU and GPU, then crashes on the first tensor concatenation. Using `.to("cuda:0")` on a single 11 GB card also fails.

**Why it happens:**
Researchers see "7B parameters" and assume it fits in 15 GB with the same math as language-only models. Robot VLAs carry large vision encoders (ViT-L/14 in OpenVLA) in addition to the LLM backbone, pushing the footprint above language-only 7B estimates.

**How to avoid:**
- Use Colab Pro with A100 (40 GB) for OpenVLA inference.
- Alternatively use openpi (the Physical Intelligence open-source reimplementation of π0) which achieves ~15 Hz on 24 GB GPUs.
- For T4, apply 4-bit QLoRA loading — but verify flash-attn version compatibility first (see Pitfall 8).
- Do NOT use `device_map="auto"` on multi-GPU rigs with heterogeneous VRAM sizes.

**Warning signs:**
- `CUDA out of memory` on model load
- `Expected all tensors to be on the same device` during the first inference call
- Suspiciously long first-token latency (model partially on CPU)

**Phase to address:** Phase 1 — Colab environment setup and VLA loading. Must be validated before any downstream work.

---

### Pitfall 2: π0 (Physical Intelligence) Is Not Self-Hostable — Use openpi

**What goes wrong:**
Researchers try to install or download the Physical Intelligence π0 model directly and find it unavailable. The original π0 is accessible only through PI's API or partnership agreements. Time is wasted trying to scrape weights or set up a non-existent pip package.

**Why it happens:**
The paper "π0: A Vision-Language-Action Flow Model for General Robot Control" is widely cited. The model name "π0" appears in discussions alongside open-source VLAs, causing confusion about availability.

**How to avoid:**
Use `openpi` (github.com/Physical-Intelligence/openpi) — this is the separately released open-source codebase with open weights including `pi0` and `pi0-fast` checkpoints. This is the correct starting point. Alternatively use OpenVLA (github.com/openvla/openvla), which is fully open-source and fine-tuned checkpoints for LIBERO exist on HuggingFace.

**Warning signs:**
- Searching PyPI for `pi0` or `physical-intelligence` returns nothing relevant
- Any instruction saying to "download π0 from Physical Intelligence" without a direct weight URL

**Phase to address:** Phase 1 — VLA selection. Decide at project start: openpi or OpenVLA. The choice affects all downstream integration.

---

### Pitfall 3: MuJoCo Headless Rendering Fails Because MUJOCO_GL Set After Import

**What goes wrong:**
On Colab (no display server), `import mujoco` or `import robosuite` crashes with an OpenGL error, or renders silently fail (black frames). The environment variable `MUJOCO_GL` must be set to `"egl"` **before** any MuJoCo-related import. Setting it in a later cell has no effect.

**Why it happens:**
Colab cells execute sequentially, and researchers often install packages in one cell and set env vars in another after the import has already happened. GLFW (the default) requires a windowed display and crashes headless. OSMesa is software-only and does not use the Colab GPU.

**How to avoid:**
```python
import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
# Only THEN import robosuite or mujoco
import robosuite
```
Put this at the very top of the notebook before any other imports. Verify EGL is available by running `!dpkg -l | grep -i egl` in a Colab cell.

**Warning signs:**
- `GLFW initialization failed` or `cannot connect to X server`
- Black or zero-size rendered frames
- `mujoco.MuJocoException: Could not initialize OpenGL` on first `env.render()` call

**Phase to address:** Phase 1 — Colab environment setup. Must be the first cell in every notebook.

---

### Pitfall 4: SOARM Not Natively Supported in robosuite — Must Build Robot Class

**What goes wrong:**
LIBERO is built on robosuite, which ships with Panda, UR5, Sawyer, etc. SOARM (SO100/SO101) is not in robosuite's robot registry. Attempting to pass `robots="SOARM"` raises `KeyError`. Researchers assume a URDF file is sufficient — it is not. robosuite requires a full Python `RobotModel` subclass, an MJCF file, and a controller config JSON.

**Why it happens:**
SOARM URDF/MJCF files exist in TheRobotStudio/SO-ARM100 repository and MuJoCo Menagerie. Researchers find these files and assume they can be dropped directly into robosuite without additional wiring.

**How to avoid:**
Build the integration in layers:
1. Convert SOARM URDF to MJCF using `mujoco`'s built-in converter or the `onshape-to-robot` pipeline (already used by TheRobotStudio).
2. Create a `ManipulatorModel` subclass in `robosuite/models/robots/manipulators/soarm.py` with correct `eef_name`, `arm_type`, `dof`, and `actuator_config`.
3. Write a controller config JSON specifying joint names matching the MJCF exactly.
4. Register the robot with `@register_robot` decorator.

Use the TechLabs Aachen SmolVLA+robosuite SO100 integration (Medium article) as a reference — they solved exactly this problem.

**Warning signs:**
- `KeyError: 'SOARM'` when creating environment
- Simulation starts but the arm explodes or flies off (wrong inertia or joint limits)
- Controller reports wrong number of joints

**Phase to address:** Phase 2 — SOARM MJCF integration. This is the highest-complexity build task and should be the first thing resolved after the Colab environment is stable.

---

### Pitfall 5: MJCF Inertia Matrix Errors Cause Silent Physics Corruption

**What goes wrong:**
An invalid inertia matrix (diagonal elements must satisfy A+B≥C for all permutations) causes a compile error. Worse, malformed inertial blocks — often from bad URDF→MJCF conversion — produce simulations that appear to run but behave wrong: arms slowly drift, objects jitter, controllers produce unexpected torques.

**Why it happens:**
CAD-derived URDF files often have inertia values estimated from bounding boxes or copy-pasted from a similar robot. The URDF→MJCF converter does not validate physics correctness. Mesh-based inertia inference can silently fall back to surface inertia for flat meshes.

**How to avoid:**
- After conversion, run `mujoco.MjModel.from_xml_path("soarm.xml")` and check for compiler warnings.
- Set `<compiler inertiafromgeom="auto"/>` only if you want MuJoCo to infer inertia automatically; remove explicit `<inertial>` tags if they came from bad estimates.
- Verify collision meshes do not overlap in the default pose (adjacent meshes create large contact forces at initialization).
- Use the SO101 MJCF from MuJoCo Menagerie as a starting point — it has been validated.

**Warning signs:**
- `mujoco.MuJocoException: inertia must satisfy A+B>=C`
- Robot arm shaking or exploding on first `env.step()`
- Contact forces > 1000 N at initialization (check `data.cfrc_ext`)

**Phase to address:** Phase 2 — SOARM MJCF integration. Validate physics before writing any Python wrappers.

---

## Moderate Pitfalls

---

### Pitfall 6: Action Playback Drift — Cannot Replay Demonstrations by Actions Alone

**What goes wrong:**
When trying to replay collected LIBERO/robosuite demonstrations, replayed trajectories diverge from the originals. Known issue: states in replayed trajectories can differ by more than 1 unit from logged states (LIBERO GitHub issue #16). The robosuite documentation explicitly warns: "action playback trajectories tend to drift over time and should not be relied upon to accurately replicate demonstrations."

**Why it happens:**
Discrete timestep integration accumulates floating-point errors. Contact dynamics are especially sensitive to initial conditions. The demonstration was recorded with a SpaceMouse at 20 Hz; replay at the same frequency still introduces small discrepancies that compound.

**How to avoid:**
Use state-setting for replay, not action replay:
```python
sim.set_state(demo["states"][t])
sim.forward()
```
LIBERO stores MuJoCo states precisely for this reason. Never attempt to re-derive a trajectory from the action sequence alone.

**Warning signs:**
- Success rate drops when replaying demonstrations that succeeded during collection
- Object positions diverge visually from the original recording

**Phase to address:** Phase 3 — Dataset collection. Bake in state-based replay from the start.

---

### Pitfall 7: LIBERO HDF5 Stores States, Not Observations — Must Run Regeneration

**What goes wrong:**
Raw LIBERO HDF5 files contain MuJoCo simulator states, not rendered observations (images, proprioception). Scripts that try to read `demo["obs"]` from raw HDF5 find no image data. OpenVLA fine-tuning requires RLDS format (TensorFlow Datasets), not raw HDF5. Researchers skip the regeneration step and wonder why their training data has no images.

**Why it happens:**
HDF5 observation storage is opt-in (large disk footprint). The LIBERO collection pipeline saves states by default. The regeneration step (`regenerate_libero_dataset.py` in openvla) is documented but easy to miss.

**How to avoid:**
- Run `openvla/experiments/robot/libero/regenerate_libero_dataset.py` to render observations from saved states.
- Convert HDF5 → RLDS using the provided scripts before fine-tuning.
- For SOARM, the regeneration step must use the SOARM environment (not Panda) to render correct images.

**Warning signs:**
- HDF5 keys contain `states` and `actions` but not `observations/images`
- Training dataset shows zero non-zero image frames

**Phase to address:** Phase 3 — Dataset collection, and Phase 4 — Fine-tuning pipeline setup.

---

### Pitfall 8: OpenVLA Strict Version Pinning — Colab Auto-Updates Break It

**What goes wrong:**
OpenVLA requires exact versions: PyTorch 2.2.0, torchvision 0.17.0, transformers 4.40.1, tokenizers 0.19.1, timm 0.9.10, flash-attn 2.5.5. Later versions have breaking changes. Colab's `!pip install openvla` without pinning picks up newer versions. Attempting bitsandbytes 8-bit quantization on PyTorch 2.2.0 causes `AttributeError: module 'torch.compiler' has no attribute 'is_compiling'`.

**Why it happens:**
Python package resolution always prefers the latest version unless pinned. OpenVLA's dependency on flash-attn (which has CUDA kernel compilation requirements) makes version management especially fragile.

**How to avoid:**
Pin ALL versions in the Colab install cell:
```bash
pip install torch==2.2.0 torchvision==0.17.0 \
  transformers==4.40.1 tokenizers==0.19.1 \
  timm==0.9.10 flash-attn==2.5.5 \
  --index-url https://download.pytorch.org/whl/cu121
```
Test the exact install cell in a fresh Colab session before depending on it.

**Warning signs:**
- `ImportError` or `AttributeError` on `import transformers` or `from prismatic import ...`
- flash-attn compilation errors during install
- `torch.compiler` attribute errors during model load

**Phase to address:** Phase 1 — Colab environment setup. Create a pinned requirements file and test it.

---

### Pitfall 9: OpenVLA Action Normalization Stats Must Come from YOUR Dataset

**What goes wrong:**
OpenVLA expects actions in [-1, 1] with normalization statistics (mean, std) computed from the fine-tuning dataset. If you use LIBERO's pre-computed statistics but collect SOARM demonstrations with a different workspace or joint configuration, the normalization is wrong. Disabling normalization or dataset shuffling causes near-complete performance collapse — not a gradual degradation.

**Why it happens:**
Researchers copy LIBERO fine-tuning configs directly without updating the statistics. The SOARM workspace bounds will differ from the Panda workspace used in LIBERO.

**How to avoid:**
- Compute normalization stats from your SOARM dataset: `mean = actions.mean(axis=0)`, `std = actions.std(axis=0)`.
- Apply z-score normalization then clip to [-1, 1].
- Never copy `dataset_statistics.json` from a different robot's dataset.
- Always keep dataset shuffling enabled in the training config.

**Warning signs:**
- Training loss plateaus immediately at a high value
- Model produces all-zero or all-extreme actions
- Action distribution in training data has mean far from zero

**Phase to address:** Phase 4 — Fine-tuning pipeline. Add a dataset statistics validation step before training starts.

---

### Pitfall 10: LIBERO→SOARM Action Space Rotation Representation Bug

**What goes wrong:**
LIBERO's action space is 7D: `[dx, dy, dz, dax, day, daz, gripper]` where rotation is axis-angle delta. When adapting to SOARM or converting between delta/absolute representations, subtracting rotation values directly is mathematically wrong (rotations do not compose by subtraction in axis-angle). This produces corrupted rotation commands that silently cause the arm to move to wrong orientations.

**Why it happens:**
Researchers treat the rotation delta as Euler angles and subtract them as scalars. The issue is documented in openpi's LIBERO dataset issue tracker (issue #416: "cannot simply subtract rotation values").

**How to avoid:**
Convert to rotation matrices or quaternions before composing rotations:
```python
from scipy.spatial.transform import Rotation
r_delta = Rotation.from_rotvec(action[3:6])
r_current = Rotation.from_rotvec(current_pose[3:6])
r_new = r_current * r_delta  # compose, not add
```
Always test rotation composition by visualizing the end-effector trajectory.

**Warning signs:**
- End-effector orientation drifts unexpectedly during demos
- Gripper reaches correct position but wrong orientation
- Large rotation residuals in controller during playback

**Phase to address:** Phase 2 (SOARM integration) and Phase 3 (dataset collection).

---

### Pitfall 11: OSC Controller Config Required — Missing Config Causes Cryptic Errors

**What goes wrong:**
LIBERO uses Operational Space Control (OSC) at 500 Hz inner loop with 20 Hz policy rate. Custom robots must provide a controller JSON config matching the robot's joint names and kinematic parameters exactly. Missing or mismatched controller config raises cryptic NumPy shape errors or silent wrong-dimension action application.

**Why it happens:**
The controller config is loaded by name from `robosuite/controllers/config/`. Researchers add the robot class but forget to add the controller config file, or copy a Panda config with Panda joint names.

**How to avoid:**
- Copy `robosuite/controllers/config/osc_pose.json` to `soarm_osc_pose.json`
- Update `joint_names` to match your MJCF exactly
- Set `kp`, `kd`, `damping_ratio` to reasonable values for SOARM's mass/inertia
- Verify `action_limit` bounds match SOARM's workspace

**Warning signs:**
- `ValueError: operands could not be broadcast together with shapes` inside the controller
- End-effector moves in wrong directions despite correct actions
- Controller prints warnings about Jacobian dimension mismatch

**Phase to address:** Phase 2 — SOARM MJCF integration.

---

## Minor Pitfalls

---

### Pitfall 12: VLA Visual Backbone Has No 3D Spatial Awareness by Default

**What goes wrong:**
Adding a depth camera to the environment does not automatically give the VLA spatial reasoning. VLA visual backbones (ViT variants) are pretrained on 2D image data with no 3D geometric supervision. The model cannot reason about object depth or estimate "left of the box" without explicit training on spatial language and 3D cues. Simply concatenating an RGB-D input without architectural changes yields no benefit.

**Why it happens:**
Depth inputs are intuitive from a human perspective, but the model has no learned association between depth values and spatial language. This requires either a 3D-aware vision encoder (e.g., spatial VLA variants) or supervised spatial augmentation data.

**How to avoid:**
For the spatial awareness phases, use multi-camera RGB views (front + overhead + side) rather than relying on depth alone. Depth can be used for task scripting (object position estimation) but not as a raw VLA input without additional training. The VEGA and cVLA papers provide relevant patterns for camera-space grounding.

**Warning signs:**
- Spatial language tasks ("pick object on the left") have the same success rate as control tasks
- Model ignores depth channel (visualize attention maps over depth input)

**Phase to address:** Phase 5 — Spatial awareness. Research the appropriate input representation before building the pipeline.

---

### Pitfall 13: SOARM Joint Calibration Mode Determines Default Pose

**What goes wrong:**
The SO101 MJCF has two calibration modes: (a) virtual zero at mid joint range (recommended), and (b) virtual zero at fully horizontal extended pose. Using the wrong calibration causes the arm to start in an unexpected configuration, which collides with objects in LIBERO task scenes (designed around Panda's default configuration).

**Why it happens:**
The default pose is set in the MJCF `<key>` element and must be adjusted when placing the robot in LIBERO's scene coordinate frame.

**How to avoid:**
- Use calibration mode (a) — mid-range zero — as the SO101 README recommends.
- After placing SOARM in the LIBERO arena, verify the default pose clears all scene objects.
- Run `env.reset()` and visualize before collecting any data.

**Warning signs:**
- Robot arm intersects the table or objects on `env.reset()`
- Contact forces spike immediately at `t=0`

**Phase to address:** Phase 2 — SOARM MJCF integration.

---

### Pitfall 14: Gripper Sign Convention Mismatch

**What goes wrong:**
robosuite normalizes gripper control to [-1, 1] where the convention for open/close varies by gripper type. LIBERO demonstrations use one sign convention. If SOARM's gripper is wired with the opposite sign, the gripper is always doing the opposite of what the VLA commands — picking tasks always fail at the grasp step.

**Why it happens:**
Gripper sign is defined in the MJCF actuator section and in the `GripperModel.format_action()` method. These are robot-specific and must be manually validated.

**How to avoid:**
- After MJCF integration, manually step with gripper action = +1 and verify the gripper opens (or closes) as expected.
- Check the robosuite `GripperModel` subclass `format_action` and `_important_actuators` to verify convention.

**Warning signs:**
- Objects dropped immediately after grasp
- Gripper always fully open or always fully closed during task execution

**Phase to address:** Phase 2 — SOARM MJCF integration.

---

### Pitfall 15: Sim-to-Real Action Frequency and Sensor Delay Mismatch

**What goes wrong:**
LIBERO runs the policy at 20 Hz and the OSC controller at 500 Hz, with deterministic and delay-free MuJoCo sensors. Real SOARM servos (STS3215/Feetech) communicate over serial at lower effective rates, with latency and quantization. A policy trained at 20 Hz sim may be too fast or too slow for the real robot's feedback loop.

**Why it happens:**
MuJoCo sensors are deterministic by default. Real servos have encoder quantization, communication jitter, and backlash. Action frequency that is stable in sim becomes unstable on hardware.

**How to avoid:**
For the simulation phase this is not immediately blocking, but document the mismatch for later. When sim pipeline is validated, add MuJoCo sensor noise and actuator delay before sim-to-real transfer:
```xml
<sensor>
  <jointpos name="joint1_pos" joint="joint1" noise="0.001"/>
</sensor>
```
Match the real servo communication rate in the controller.

**Warning signs:**
- Policy that works in sim produces oscillating or jerky motion on hardware

**Phase to address:** Phase 6 — Sim-to-real transfer (future milestone, not MVP).

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Copy Panda controller config for SOARM | Faster robot integration | Wrong PD gains, unstable control, hard to debug | Never |
| Use LIBERO normalization stats for SOARM demos | Skip stats computation | Performance collapse after fine-tuning | Never |
| Skip observation regeneration step | Smaller disk footprint | Training data has no images | Never |
| Use T4 Colab for OpenVLA inference | Free | OOM crash, wasted Colab session time | Only if using 4-bit QLoRA and tested |
| Set MUJOCO_GL after imports | Feels more organized | Headless rendering silently fails | Never |
| Replay demos via action replay instead of state-setting | Simpler code | Trajectory drift, unreliable evaluation | Never |

---

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| URDF → MJCF | Trust the auto-converter output | Validate inertia, check collision mesh overlap, test in isolation before robosuite integration |
| OpenVLA + LIBERO | Use raw HDF5 for fine-tuning | Run `regenerate_libero_dataset.py`, convert to RLDS format |
| π0/openpi + custom robot | Edit only the model config | Also implement `LiberoInputs`/`LiberoOutputs` transforms for your robot's action/state format |
| Colab + MuJoCo | Set env vars in any cell | Set `MUJOCO_GL=egl` in the first cell, before any import |
| SOARM in robosuite | Pass URDF path directly | Create full `ManipulatorModel` subclass with controller config |
| Action rotation composition | Subtract axis-angle deltas | Use `scipy.spatial.transform.Rotation` composition |

---

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Rendering every step in Colab | Notebook becomes unresponsive | Render every N steps or use offscreen rendering to file | Immediately with any VLA inference loop |
| Loading full OpenVLA-7b for each Colab session | 5-10 min load time per session | Cache weights in Google Drive, load from `/content/drive/` | Every session without Drive mount |
| Episode-level HDF5 writes in Python loop | Very slow dataset creation | Use robosuite's batch HDF5 writer | >100 episodes |
| Regenerating LIBERO observations for all tasks at once | Fills Colab disk (15 GB limit) | Regenerate per-task, delete after conversion to RLDS | All 4 LIBERO suites at once |

---

## "Looks Done But Isn't" Checklist

- [ ] **SOARM MJCF integration:** Check that `env.reset()` produces a stable, collision-free initial pose — not just "no crash on load"
- [ ] **Headless rendering:** Verify rendered frames are non-black (save one PNG and inspect) — not just "no rendering error"
- [ ] **OpenVLA installation:** Confirm inference runs on a real input (random image + text prompt → action) — not just "model loaded"
- [ ] **Dataset collection:** Inspect 3 episodes visually (render saved states) and confirm success labels are correct — not just "HDF5 file created"
- [ ] **Action normalization:** Plot the action distribution and verify near-zero mean, unit variance — not just "normalization code runs"
- [ ] **Gripper convention:** Manually command gripper open/close and verify correct direction — not just "gripper actuator present in MJCF"
- [ ] **RLDS conversion:** Load one episode from RLDS and print images + action shapes — not just "script completed without error"

---

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Wrong Colab GPU tier for OpenVLA | LOW | Upgrade to Colab Pro, remount Drive, restart session |
| Corrupted SOARM MJCF (bad inertia) | MEDIUM | Restart from MuJoCo Menagerie SO101 base, re-apply customizations |
| Wrong normalization stats baked into fine-tuned model | HIGH | Re-collect normalization stats, retrain from base checkpoint |
| Action playback drift in recorded dataset | MEDIUM | Re-record using state-setting replay; existing action-only demos are invalid |
| OpenVLA dependency conflict | LOW | Create fresh Colab session, run pinned install cell first |
| Missing controller config | LOW | Copy and adapt from Panda config, update joint names |

---

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| OpenVLA VRAM overrun | Phase 1 — Colab setup | Load model and run one inference call; confirm GPU memory usage |
| π0 unavailability | Phase 1 — VLA selection | Confirm openpi weights download successfully |
| MUJOCO_GL misconfiguration | Phase 1 — Colab setup | Save a rendered frame to PNG and open it |
| SOARM not in robosuite | Phase 2 — SOARM integration | `env.reset()` with SOARM robot succeeds |
| MJCF inertia errors | Phase 2 — SOARM integration | MuJoCo compile step with no warnings; initial contact forces < 10 N |
| Action playback drift | Phase 3 — Dataset collection | State-based replay exactly reproduces original trajectory |
| HDF5 stores states not observations | Phase 3 — Dataset collection | Regeneration script run; images present in converted dataset |
| OpenVLA version pinning | Phase 1 — Colab setup | Full install + inference test in clean session |
| Action normalization | Phase 4 — Fine-tuning | Plot action distribution before and after normalization |
| Rotation representation bug | Phase 2 + Phase 3 | Unit test rotation composition; visualize EEF trajectory |
| OSC controller config | Phase 2 — SOARM integration | Controller runs without dimension errors for 100 steps |
| VLA lacks spatial awareness | Phase 5 — Spatial features | Multi-camera input ablation; spatial task success rate |
| Joint calibration mode | Phase 2 — SOARM integration | Visual inspection of default pose in LIBERO scene |
| Gripper sign convention | Phase 2 — SOARM integration | Manual gripper open/close test |
| Sim-to-real frequency gap | Phase 6 — Sim-to-real (future) | Document gap; add noise/delay in sim before hardware transfer |

---

## Sources

- robosuite Human Demonstrations documentation: action playback drift warning — https://robosuite.ai/docs/algorithms/demonstrations.html
- LIBERO GitHub issue #16: state drift during demonstration replay — https://github.com/Lifelong-Robot-Learning/LIBERO/issues/16
- OpenVLA GitHub issue #311: multi-GPU OOM with `device_map="auto"` — https://github.com/openvla/openvla/issues/311
- openpi GitHub issue #416: LIBERO Cartesian action space, rotation subtraction bug — https://github.com/Physical-Intelligence/openpi/issues/416
- OpenVLA README: exact version pins and flash-attn requirements — https://github.com/openvla/openvla/blob/main/README.md
- MuJoCo modeling docs: inertia matrix validation (A+B≥C) — https://mujoco.readthedocs.io/en/latest/modeling.html
- MuJoCo headless rendering docs / torchrl guide — https://docs.pytorch.org/rl/main/reference/generated/knowledge_base/MUJOCO_INSTALLATION.html
- SO-ARM100 Simulation README (SO101 calibration modes) — https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/README.md
- TechLabs Aachen SO100 + SmolVLA + robosuite integration report — https://techlabs-aachen.medium.com/organizer-robot-teaching-an-so100-to-restore-order-using-smolvla-and-robosuite-9b5f2d0558ed
- openvla regenerate_libero_dataset.py — https://github.com/openvla/openvla/blob/main/experiments/robot/libero/regenerate_libero_dataset.py
- openpi README: custom robot transforms (LiberoInputs/LiberoOutputs) — https://github.com/Physical-Intelligence/openpi/blob/main/README.md
- VLA action normalization analysis: "Scaling VLA Model Training on a Budget" — https://www.roboticscenter.ai/blog/scaling-vla-training-on-a-budget
- robosuite sim-to-real documentation — https://robosuite.ai/docs/algorithms/sim2real.html
- "On the Role of the Action Space in Robot Manipulation Learning and Sim-to-Real Transfer" (2023) — https://arxiv.org/abs/2312.03673
- cVLA camera-space grounding — https://arxiv.org/html/2507.02190v2

---
*Pitfalls research for: VLA + LIBERO/MuJoCo robot simulation (SoARM Research)*
*Researched: 2026-07-07*
