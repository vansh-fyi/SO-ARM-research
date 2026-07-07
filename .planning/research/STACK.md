# Stack Research

**Domain:** VLA (Vision-Language-Action) robot simulation pipeline with custom robot arm
**Researched:** 2026-07-07
**Confidence:** MEDIUM (cross-verified across official repos, HuggingFace docs, and community sources)

---

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| Python | 3.10 | Primary language | 3.10 is the safest Colab target — 3.11 works for openpi but LIBERO/robomimic have `distutils` assumptions that break on 3.12+ |
| MuJoCo | 2.3.7 | Physics engine | LIBERO pins to 2.3.7; robosuite 1.4.x ships MJCF assets targeting this version; do not upgrade |
| robosuite | 1.4.0 | Robot simulation environments | LIBERO hard-depends on 1.4.x API (`SingleArmEnv`, `OffScreenRenderEnv`); 1.5 broke the API and LIBERO has not migrated |
| LIBERO (vendored) | repo HEAD | Task suites, BDDL, OffScreenRenderEnv wrapper | Already in repo as `LIBERO/`; BDDL task DSL and lifelong learning infrastructure; do not replace with LIBERO from pip |
| PyTorch | 2.1.x + CUDA 11.8 | Tensor ops, model weights | 2.1.x is the last version that cleanly supports both old robomimic code and modern VLA inference; Colab provides this by default |
| OpenVLA-OFT | latest (moojink/openvla-oft) | VLA inference + fine-tuning | 7B model, 16GB VRAM for LIBERO inference (fits T4 exactly), 97.1% avg success on LIBERO benchmark, continuous action space eliminates jitter, action chunking chunk-8 |
| HuggingFace LeRobot | latest (`lerobot[pi0]`) | π0 VLA alternative | More portable than raw openpi (works outside Ubuntu 22.04), `lerobot/pi0_libero_base` checkpoint available, `PI0Policy.from_pretrained()` API |
| robomimic | 0.2.0 | HDF5 dataset handling | Used by LIBERO for `SequenceDataset`, `FileUtils`, `ObsUtils`; version must match LIBERO vendored requirements |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| huggingface_hub | >= 0.20.0 | Model checkpoint download | Every notebook; `snapshot_download` to pull VLA weights to `/root/.cache/huggingface` |
| einops | >= 0.7.0 | Tensor rearrangement | Required by both LeRobot (pi0) and OpenVLA-OFT; install before any VLA import |
| h5py | >= 3.7.0 | HDF5 demo dataset I/O | Reading and writing LIBERO-format demonstration datasets |
| imageio | >= 2.28.0 | Video recording | Saving rollout episodes as MP4 for visual inspection |
| opencv-python | 4.6.0.66 | Image preprocessing | Frame resizing to 224x224; already pinned by LIBERO |
| transformers | 4.40+ (VLA env) / 4.21.1 (LIBERO env) | HuggingFace model loading | **Version conflict — see note below; must be managed per-environment** |
| accelerate | >= 0.26.0 | Multi-GPU / bfloat16 loading | Required by OpenVLA-OFT for `load_in_4bit` / `bfloat16` model loading |
| peft | >= 0.7.0 | LoRA fine-tuning | Required for OpenVLA-OFT fine-tuning with LoRA r=32 |
| bitsandbytes | >= 0.41.0 | 4-bit / 8-bit quantization | Optional on T4 for memory reduction; required if training on Colab free tier |
| wandb | 0.13.x | Experiment tracking | Logging fine-tuning runs; already used by LIBERO training loop |
| imageio-ffmpeg | latest | MP4 encoding backend | Install alongside imageio for `.mp4` output |

### VLA Model Details

#### Primary Recommendation: OpenVLA-OFT

OpenVLA-OFT is the right first choice for this project because it was explicitly benchmarked on LIBERO, its code is written in standard PyTorch/HuggingFace, and its 16GB VRAM inference footprint fits a T4 without quantization tricks.

| Property | Value |
|----------|-------|
| Model size | 7B parameters |
| Base architecture | Prismatic VLM (Llama 2 + DINOv2 + SigLIP) |
| Input image size | 224 x 224 RGB |
| Action dimensions | 7 (3D position delta, 3D orientation delta, gripper open/close) |
| Action representation | Continuous (L1 loss); not discrete bins |
| Action chunking | Chunk size 8 for LIBERO tasks |
| Inference VRAM (LIBERO) | ~15.9 GB (fits T4 16GB, tight) |
| Training VRAM (LoRA, bs=1) | ~25.6 GB (requires A100 Pro Colab) |
| LIBERO benchmark | 97.1% avg (Spatial 97.6%, Object 98.4%, Goal 97.9%, Long 94.5%) |
| Fine-tuning method | LoRA (r=32) via peft |
| GitHub | moojink/openvla-oft |
| HuggingFace | `openvla/openvla-7b` as base |

#### Secondary / Alternative: π0 via LeRobot

Use LeRobot's π0 integration rather than raw openpi when running outside Ubuntu 22.04 (which Colab is) or when you want a simpler inference API.

| Property | Value |
|----------|-------|
| Model family | Flow-matching VLA (not autoregressive) |
| HuggingFace checkpoint | `lerobot/pi0_libero_base` |
| Inference VRAM | >8 GB (fits T4 with headroom) |
| LoRA fine-tuning VRAM | >22.5 GB (requires A100) |
| Full fine-tuning VRAM | >70 GB (H100 / multi-GPU) |
| Inference API | `PI0Policy.from_pretrained(model_id).to(device)` |
| Action space | 7-DOF delta EEF + gripper, chunked |
| Raw openpi hardware note | Officially Ubuntu 22.04 only; Colab is Ubuntu 22.04 as of 2025 so raw openpi is feasible but `uv` tooling conflicts with Colab `pip` workflow |

**Choose OpenVLA-OFT when:** You want battle-tested LIBERO results, standard HuggingFace APIs, and T4 compatibility with no quantization.

**Choose π0/LeRobot when:** You want a flow-based policy (smoother motions, less jitter), or you want to leverage Physical Intelligence's pretraining on diverse manipulation data.

### SOARM Robot Integration Stack

| Component | Source | Purpose | Notes |
|-----------|--------|---------|-------|
| MJCF / URDF model | `TheRobotStudio/SO-ARM100`, `Simulation/SO101/` | Base robot description | `so101_new_calib.xml` recommended (joint zeros at midrange); gripper linear joint: 0=closed, 100=open |
| URDF→MJCF conversion | `mujoco compile urdf_file.urdf out.xml` | Convert if only URDF available | Remove `package://` URIs first; use relative mesh paths |
| robosuite integration | Subclass `ManipulatorModel` | Register SOARM with robosuite | Provide MJCF path, define `default_controller`, `init_qpos`, `joints`, `eef_name` properties |
| Gripper model | Custom `GripperModel` subclass | Control SO-ARM100 gripper | Linear joint; map to binary open/close convention used by LIBERO |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| Google Colab (Pro/Pro+) | GPU access for VLA inference | T4 (16GB) for inference; A100 (40GB) for fine-tuning; session limit 12h Pro |
| `uv` | Fast Python env for raw openpi | Only needed if using raw openpi; Colab can use pip instead via LeRobot |
| `wandb` | Training run logging | Already integrated in LIBERO; use same key for VLA fine-tuning runs |
| MuJoCo viewer (local) | MJCF debugging | Run `python -m mujoco.viewer --mjcf path.xml` to validate robot model before Colab use |

---

## Installation

### Google Colab Installation Order (Critical)

Order matters because MuJoCo rendering mode must be set before any MuJoCo/robosuite/LIBERO import.

```python
# Cell 1: Set rendering env vars FIRST, before any imports
import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

# Cell 2: Install core simulation stack
# robosuite 1.4.0 specifically — 1.5 breaks LIBERO
!pip install mujoco==2.3.7
!pip install robosuite==1.4.0
!pip install gym==0.25.2
!pip install bddl==1.0.1

# Cell 3: Clone and install LIBERO (use vendored fork for SOARM customization)
!pip install -e LIBERO/

# Cell 4a: Install OpenVLA-OFT (recommended VLA)
# In a separate step because it needs transformers >= 4.40
!pip install transformers>=4.40 accelerate peft bitsandbytes einops
!pip install git+https://github.com/moojink/openvla-oft.git

# Cell 4b: OR install LeRobot with pi0 support (alternative VLA)
!pip install lerobot[pi0]

# Cell 5: Supporting utilities
!pip install h5py imageio imageio-ffmpeg wandb huggingface_hub
```

**Critical**: `MUJOCO_GL=egl` must be set before Cell 2. Once any MuJoCo code is imported with a different GL backend, changing the env var has no effect within that session.

### Local Development (macOS)

```bash
# macOS uses GLFW for windowed rendering
export MUJOCO_GL=glfw

pip install mujoco==2.3.7 robosuite==1.4.0
pip install -e LIBERO/
pip install lerobot[pi0]  # or openvla-oft
pip install h5py imageio wandb huggingface_hub einops
```

---

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| robosuite 1.4.0 | robosuite 1.5.x | Never for LIBERO; 1.5 removed `SingleArmEnv` which LIBERO depends on |
| EGL rendering (Colab) | OSMesa (software) | Only when GPU is unavailable; OSMesa is 3-5x slower, CPU-only |
| OpenVLA-OFT | Octo, RT-2, RoboFlamingo | Octo is good but not LIBERO-specific; RT-2 is proprietary; OpenVLA-OFT has published LIBERO numbers |
| LeRobot pi0 integration | Raw openpi (Physical-Intelligence/openpi) | Raw openpi if you need the full training ecosystem (openpi uses JAX under the hood); LeRobot port is PyTorch |
| PyTorch 2.1.x | PyTorch 2.3+ | PyTorch 2.3+ changes `torch.load` weights_only default which breaks older robomimic checkpoints |
| LoRA fine-tuning (A100) | Full fine-tuning | Only use full fine-tuning if you have >70GB VRAM and need maximum expressiveness |
| HDF5 demo format (robomimic) | RLDS (TF Datasets) | RLDS only needed if consuming Open X-Embodiment directly; LIBERO uses HDF5 natively |

---

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| `robosuite>=1.5.0` | Removed `SingleArmEnv`; LIBERO breaks at import | `robosuite==1.4.0` exactly |
| `transformers==4.21.1` for VLA inference | Too old for OpenVLA/LeRobot model loading (flash attention, rope, etc.) | Keep LIBERO env separate; use `transformers>=4.40` in VLA env |
| OSMesa on Colab GPU nodes | Ignores GPU, runs software rasterizer at ~3 FPS | EGL with `MUJOCO_GL=egl` |
| Raw `openpi` with `uv` in Colab | `uv` manages its own venv that conflicts with Colab's package state; GIT LFS requirements add friction | `lerobot[pi0]` which ships the PyTorch port of pi0 via standard pip |
| MuJoCo >= 3.0 | API breaking changes in MJCF loading, asset paths, and `mjModel` access patterns; robosuite 1.4 not tested with it | `mujoco==2.3.7` |
| `gym>=0.26.0` | Changed step() return signature from 4-tuple to 5-tuple (adds `truncated`); robosuite/LIBERO expect the old signature | `gym==0.25.2` |
| GLFW in headless Colab | Requires display server; Colab has no X11/Wayland | EGL backend only |

---

## Version Compatibility Matrix

| Package | Compatible With | Notes |
|---------|-----------------|-------|
| `robosuite==1.4.0` | `mujoco==2.3.7`, `gym==0.25.2` | Do not mix with robosuite 1.5 |
| `LIBERO` (vendored) | `robosuite==1.4.0`, `robomimic==0.2.0` | `LIBERO/requirements.txt` must be satisfied before any `libero.*` import |
| `transformers==4.21.1` | LIBERO lifelong training code only | **Conflicts** with OpenVLA-OFT and LeRobot; run in separate kernel/process |
| `transformers>=4.40` | OpenVLA-OFT, LeRobot pi0 | Install in VLA inference environment only |
| `torch==2.1.x` | All components | Bridge version: old enough for robomimic, new enough for VLA inference |
| `lerobot[pi0]` | `transformers>=4.38`, `torch>=2.0` | LeRobot manages its own pin, generally compatible with Colab defaults |
| `mujoco==2.3.7` | EGL on CUDA 11.8/12.x GPUs | EGL context creation requires GPU driver >=470 (all Colab nodes satisfy this) |

### The transformers Version Conflict — Mitigation Strategy

LIBERO's lifelong training (`LIBERO/libero/lifelong/`) uses `transformers==4.21.1` for task embedding. OpenVLA-OFT requires `transformers>=4.40`. Both cannot coexist in one Python environment.

**Recommended mitigation for Colab**: Run LIBERO task suites (env creation, dataset collection) in the standard LIBERO kernel. For VLA inference, use a second Colab cell-group that reinstalls transformers only when calling the VLA. The `OffScreenRenderEnv` for rendering does not import transformers, so this separation is surgical.

**Recommended mitigation for local development**: Two conda environments — `libero-env` (4.21.1) and `vla-env` (4.40+). Use subprocess calls or a REST server between them.

---

## Stack Patterns by Variant

**Phase 1 — Inference only (validate end-to-end loop):**
- Use OpenVLA-OFT with pre-trained LIBERO checkpoint (no fine-tuning)
- T4 Colab (free) is sufficient
- Swap Panda for SOARM in robosuite environment; resize action space from 7 to SOARM DOF

**Phase 2 — Dataset collection (SOARM demonstrations):**
- Run LIBERO `OffScreenRenderEnv` with SOARM model
- Use scripted policies first (teleoperation adds latency complexity)
- Store demos in HDF5 via robomimic's `DataCollectionWrapper`
- 100-500 demos per task is the typical LIBERO fine-tuning baseline

**Phase 3 — Fine-tuning (SOARM-specific policy):**
- OpenVLA-OFT LoRA (r=32) on A100 Colab Pro
- ~25.6GB VRAM minimum; use gradient checkpointing to fit bs=1 on 40GB A100
- Training images: 224x224, two cameras (agentview + wrist)
- FAST tokenizer optional (15x inference speedup for discrete-token variant)

**Phase 4 — Spatial awareness:**
- Add third camera (frontview) to observation dict
- Feed all three views to VLA (OpenVLA-OFT supports multi-image with code modifications)
- For 3D localization: use depth from MuJoCo's depth buffer + camera intrinsics matrix

---

## Sources

- [Physical-Intelligence/openpi README](https://github.com/Physical-Intelligence/openpi/blob/main/README.md) — hardware requirements, installation, inference API (MEDIUM confidence, cross-verified)
- [moojink/openvla-oft](https://github.com/moojink/openvla-oft) — LIBERO VRAM requirements, LoRA config, benchmark results (MEDIUM confidence, cross-verified)
- [openvla-oft project page](https://openvla-oft.github.io/) — LIBERO success rates, action chunking details (MEDIUM confidence)
- [HuggingFace LeRobot LIBERO docs](https://huggingface.co/docs/lerobot/libero) — LeRobot pi0 integration, action space format (MEDIUM confidence)
- [lerobot/pi0_libero_base checkpoint](https://huggingface.co/lerobot/pi0_libero_base) — pi0 LIBERO checkpoint (MEDIUM confidence)
- [TheRobotStudio/SO-ARM100 Simulation/SO101/README](https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/README.md) — SOARM MJCF file location and calibration variants (MEDIUM confidence)
- [robosuite 1.5 Robot Model docs](https://robosuite.ai/docs/modeling/robot_model.html) — ManipulatorModel subclassing pattern (LOW confidence, 1.5 API, 1.4 pattern is similar)
- [robosuite installation docs](https://robosuite.ai/docs/installation.html) — EGL/OSMesa/GLFW rendering backends (MEDIUM confidence)
- [Claru OpenVLA-OFT guide](https://claru.ai/models/openvla) — dataset format, training data structure (LOW confidence, third-party)
- [OpenVLA arXiv paper v1](https://arxiv.org/html/2406.09246v1) — action format, image size, discretization scheme (MEDIUM confidence)

---

*Stack research for: SoARM VLA Research — VLA simulation pipeline*
*Researched: 2026-07-07*
