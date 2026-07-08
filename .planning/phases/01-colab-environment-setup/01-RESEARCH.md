# Phase 1: Colab Environment Setup - Research

**Researched:** 2026-07-08
**Domain:** Google Colab dependency management, MuJoCo EGL headless rendering, OpenVLA-OFT model loading
**Confidence:** MEDIUM (architecture patterns HIGH from codebase; package versions MEDIUM from PyPI; Colab-specific EGL details LOW from web search)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01:** Single notebook with restart-aware cell ordering — one kernel, clearly marked restart cells between the pip install group and the verification group. No split into multiple files.
- **D-02:** Notebook lives at `LIBERO/notebooks/01-colab-env-setup.ipynb` — co-located with existing LIBERO notebooks.
- **D-03:** Override to `transformers==4.40.1` (OpenVLA-OFT's requirement). Phase 1 only needs LIBERO for EGL rendering, not training, so LIBERO's 4.21.1 pin can be safely overridden for this phase.
- **D-04:** After the runtime restart and pip installs complete, render a default Panda LIBERO environment frame to explicitly verify LIBERO rendering still works under 4.40.1 (this is the ENV-02 check).
- **D-05:** Target Colab Pro A100 (40GB VRAM). Load OpenVLA-OFT in bf16 — no 4-bit quantization needed, no bitsandbytes dependency.
- **D-06:** Include a GPU assertion cell early in the notebook — check `torch.cuda.get_device_name(0)` and print a loud warning (or raise) if the runtime is not A100. Prevents silent OOM failures mid-execution.
- **D-07:** Each ENV requirement (ENV-01, ENV-02, ENV-03) gets its own structured verification cell with explicit `PASS` / `FAIL` output.

### Claude's Discretion

- Exact pip install ordering and pinned versions within the install cells.
- Which specific LIBERO BDDL task and camera config to use for the ENV-02 render check.
- Whether to use `IPython.display.Image` or `matplotlib` for inline display.

### Deferred Ideas (OUT OF SCOPE)

- 4-bit quantization path for T4 (bitsandbytes) — deferred to a future optional cell or Phase 3.
- Google Drive persistence of outputs — deferred.
- π0 (openpi) loading in Colab — deferred to Phase 3.
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| ENV-01 | Colab notebook installs all dependencies in correct order (robosuite 1.4.0, MuJoCo 2.3.7, gym 0.25.2, PyTorch 2.1.x) without conflicts | Dependency install order and conflict analysis documented in Standard Stack and Common Pitfalls sections |
| ENV-02 | EGL headless rendering is configured (MUJOCO_GL=egl set before any MuJoCo import) and produces non-black frames | EGL setup sequence, NVIDIA ICD creation, and MUJOCO_GL ordering constraint documented in Architecture Patterns |
| ENV-03 | OpenVLA-OFT model loads successfully on Colab GPU (A100/T4) via HuggingFace | OpenVLA-OFT model IDs, custom transformers fork requirement, bf16 loading pattern documented in Code Examples |
</phase_requirements>

---

## Summary

Phase 1 is a Colab notebook scaffolding problem. The primary technical challenges are: (1) installing a deeply pinned research stack without pip resolver failures, (2) configuring MuJoCo EGL headless rendering on Colab Linux, and (3) loading a 7B VLA model from a custom transformers fork onto an A100 in bf16.

The most significant discovery from research is that OpenVLA-OFT does **not** use the standard `transformers` package from PyPI. It requires a custom fork: `git+https://github.com/moojink/transformers-openvla-oft.git`. This fork modifies bidirectional attention for parallel decoding. The CONTEXT.md decision D-03 to pin `transformers==4.40.1` is partially correct in spirit but the actual install must be the fork, not the PyPI release. The fork is API-compatible with 4.40.1 and that is the confirmed GPU/version combination from the OFT paper.

The second key discovery is that Colab does not ship the NVIDIA EGL ICD config file (`/usr/share/glvnd/egl_vendor.d/10_nvidia.json`) even when a GPU is attached, because the driver is not installed via apt. This file must be created programmatically before setting `MUJOCO_GL=egl`. Missing this step produces an `EGL display initialization failed` error even with `MUJOCO_GL=egl` set.

LIBERO's `__init__.py` prompts interactively for dataset path on first import if `~/.libero/config.yaml` does not exist. This causes an `EOFError` in non-interactive Colab cells. The config file must be created programmatically before any `import libero` call.

**Primary recommendation:** Build the notebook in three strict phases: (A) System apt + pip install block, (B) Mandatory runtime restart, (C) Config bootstrap + verification cells. All EGL/MUJOCO_GL setup must happen at the top of Phase C, before any MuJoCo import.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Dependency installation | Colab runtime (pip/apt) | — | All packages install into runtime Python environment; no server layer |
| EGL headless rendering | Colab runtime OS | MuJoCo Python binding | OS provides EGL driver; MuJoCo uses it via MUJOCO_GL env var |
| LIBERO env creation | Python process (libero.libero.envs) | MuJoCo physics engine | LIBERO wraps robosuite which wraps MuJoCo |
| VLA model loading | GPU (CUDA / bf16) | HuggingFace hub (download) | Model weights live on GPU; hub provides download |
| Verification output | Notebook cell output | Local file (outputs/) | Inline display + saved PNG for provenance |

---

## Standard Stack

### Core (ENV-01 target versions)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| mujoco | 2.3.7 | Physics engine Python binding | Pinned by LIBERO/requirements.txt; robosuite 1.4.x migrated away from mujoco-py to this native binding [VERIFIED: PyPI registry — `pip index versions mujoco` confirms 2.3.7 exists] |
| robosuite | 1.4.0 | Robot simulation environments | Pinned by LIBERO/requirements.txt; 1.5.x removed SingleArmEnv breaking LIBERO compatibility [VERIFIED: PyPI registry — `pip index versions robosuite` confirms 1.4.0 exists] |
| gym | 0.25.2 | RL environment interface | Pinned by LIBERO/requirements.txt; 0.26.x changed step() return signature [VERIFIED: PyPI registry — `pip index versions gym` confirms 0.25.2 exists] |
| torch | 2.2.0 | Deep learning / GPU inference | Pinned by openvla-oft pyproject.toml [CITED: github.com/moojink/openvla-oft/blob/main/pyproject.toml] |
| torchvision | 0.17.0 | Vision preprocessing | Paired with torch==2.2.0 [CITED: github.com/moojink/openvla-oft/blob/main/pyproject.toml] |
| transformers (fork) | 4.40.1-fork | Custom bidirectional attn for OpenVLA-OFT | Must be installed from git fork, NOT PyPI; fork URL: `git+https://github.com/moojink/transformers-openvla-oft.git` [CITED: github.com/moojink/openvla-oft/blob/main/pyproject.toml] |
| timm | 0.9.10 | Vision encoder (SigLIP) | Pinned by openvla-oft pyproject.toml; later timm versions have regressions [CITED: github.com/moojink/openvla-oft/blob/main/pyproject.toml] |
| tokenizers | 0.19.1 | Fast tokenization | Paired with transformers fork [CITED: github.com/moojink/openvla-oft/blob/main/pyproject.toml] |
| peft | 0.11.1 | LoRA adapter loading | Pinned by openvla-oft pyproject.toml [CITED: github.com/moojink/openvla-oft/blob/main/pyproject.toml] |
| sentencepiece | 0.1.99 | Tokenizer backend | Pinned by openvla-oft pyproject.toml [CITED: github.com/moojink/openvla-oft/blob/main/pyproject.toml] |
| flash-attn | 2.5.5 | Attention acceleration on A100 | Pinned; must be installed AFTER editable install; requires `--no-build-isolation` [CITED: github.com/moojink/openvla-oft/blob/main/pyproject.toml and SETUP.md] |

### LIBERO Supporting Stack

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| hydra-core | 1.2.0 | Config for LIBERO training | LIBERO training; not needed for Phase 1 render check |
| robomimic | 0.2.0 | Dataset pipeline | Data collection phases; not Phase 1 |
| bddl | 1.0.1 | Task definition language | Required for OffScreenRenderEnv init [VERIFIED: PyPI registry] |
| easydict | 1.9 | Config dict access | LIBERO dependency [CITED: LIBERO/requirements.txt] |
| cloudpickle | 2.1.0 | Object serialization | LIBERO dependency [CITED: LIBERO/requirements.txt] |
| wandb | 0.13.1 | Experiment tracking | Training only; not Phase 1 |
| imageio[ffmpeg] | any | Video writing | Required by openvla-oft libero eval [CITED: openvla-oft libero_requirements.txt] |
| einops | 0.4.1 | Tensor ops | LIBERO model dependency [CITED: LIBERO/requirements.txt] |
| numpy | 1.22.4 | Array ops | Pinned by LIBERO/requirements.txt |
| opencv-python | 4.6.0.66 | Image processing | LIBERO dependency |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| mujoco 2.3.7 | mujoco-py | mujoco-py is deprecated, requires MuJoCo 2.1 binary, not compatible with robosuite 1.4 |
| MUJOCO_GL=egl | MUJOCO_GL=osmesa | osmesa is CPU-only, no GPU, produces valid frames but slow; acceptable fallback if EGL fails |
| flash-attn 2.5.5 | No flash-attn | Flash-attn is optional for inference; saves VRAM and speeds attention but adds compile time; include for A100 |
| Custom transformers fork | transformers==4.40.1 from PyPI | PyPI version lacks bidirectional attn patch; model will load but OFT parallel decoding will not work correctly |

### Installation Order (Critical)

```bash
# CELL 1: System packages (must run before pip installs)
!apt-get install -y -q libglfw3 libglew-dev libosmesa6-dev \
    libgles2 libglvnd0 libegl-dev libegl1 libgl1-mesa-glx

# CELL 2: PyTorch first (establishes CUDA context)
!pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu121 -q

# CELL 3: MuJoCo + sim stack (in dependency order)
!pip install mujoco==2.3.7 gym==0.25.2 -q
!pip install robosuite==1.4.0 bddl==1.0.1 easydict==1.9 \
    cloudpickle==2.1.0 einops==0.4.1 numpy==1.22.4 \
    opencv-python==4.6.0.66 imageio[ffmpeg] -q

# CELL 4: LIBERO editable install
# (requires git clone to /content/LIBERO or mount from repo)
!pip install -e /content/drive/MyDrive/SoARM-Research/LIBERO -q

# CELL 5: OpenVLA-OFT and custom transformers fork
!pip install timm==0.9.10 tokenizers==0.19.1 sentencepiece==0.1.99 \
    peft==0.11.1 accelerate huggingface_hub -q
!pip install git+https://github.com/moojink/transformers-openvla-oft.git -q

# CELL 6: flash-attn (must install AFTER everything else; compiles C++)
!pip install packaging ninja -q
!pip install "flash-attn==2.5.5" --no-build-isolation -q

# ---> RESTART RUNTIME HERE <---

# CELL 7 (post-restart): NVIDIA EGL ICD + MUJOCO_GL (first cell after restart)
```

**Version verification commands:**
```bash
pip index versions mujoco       # confirmed 2.3.7 exists
pip index versions robosuite    # confirmed 1.4.0 exists
pip index versions gym          # confirmed 0.25.2 exists
pip index versions transformers # confirmed 4.40.1 exists on PyPI (but use fork)
pip index versions timm         # confirmed 0.9.10 exists
pip index versions tokenizers   # confirmed 0.19.1 exists
```

---

## Package Legitimacy Audit

> The packages below were verified via `pip index versions` on the PyPI registry. All packages have known GitHub source repos and are part of established robotics/ML research stacks. The "SUS" verdicts below are from the legitimacy seam's `unknown-downloads` signal (PyPI does not expose weekly download stats for all packages) — not from any actual suspicion. All packages are from known research organizations (ARISE-Initiative, Stanford, Google DeepMind, OpenAI).

| Package | Registry | Published | Source Repo | Verdict | Disposition |
|---------|----------|-----------|-------------|---------|-------------|
| robosuite | PyPI | 2025-12-24 (latest) | github.com/ARISE-Initiative/robosuite | SUS (unknown-downloads) | Approved — ARISE-Initiative Stanford research lab, well-known in robot learning |
| mujoco | PyPI | 2026-06-22 (latest) | github.com/google-deepmind/mujoco | SUS (unknown-downloads, too-new latest) | Approved — Google DeepMind official Python binding, version 2.3.7 is from 2022 |
| robomimic | PyPI | 2023-07-04 | github.com/ARISE-Initiative/robomimic | SUS (unknown-downloads) | Approved — ARISE-Initiative Stanford, same team as robosuite |
| gym | PyPI | 2022-10-04 | gymlibrary.dev | SUS (unknown-downloads) | Approved — OpenAI/Farama, foundational RL library |
| bddl | PyPI | 2025-06-23 (latest) | github.com/StanfordVL/bddl | SUS (unknown-downloads) | Approved — Stanford Vision and Learning Lab, LIBERO's task DSL |
| transformers (fork) | GitHub only | N/A | github.com/moojink/transformers-openvla-oft | N/A | Approved — fork of HuggingFace transformers; required by openvla-oft |
| flash-attn | PyPI | established | github.com/Dao-AILab/flash-attention | OK | Approved — widely used attention kernel |
| timm | PyPI | established | github.com/huggingface/pytorch-image-models | OK | Approved — PyTorch Image Models, maintained by HuggingFace |

**Packages removed due to SLOP verdict:** none
**Packages flagged as SUS:** robosuite, mujoco, robomimic, gym, bddl — all are legitimate research packages from known institutions; "SUS" verdict is due to PyPI download stats being unavailable for these specialized packages, not actual suspicion.

---

## Architecture Patterns

### System Architecture Diagram

```
[Researcher] → [Colab Notebook]
                     │
         ┌───────────┴─────────────────────┐
         │                                  │
  [CELL BLOCK A: Install]          [CELL BLOCK B: Verify]
  apt-get EGL packages              (post-restart only)
  pip mujoco/gym/robosuite                  │
  pip LIBERO -e                    ┌────────┼────────┐
  pip openvla-oft (fork)    [ENV-01]  [ENV-02]  [ENV-03]
  pip flash-attn            version  render    VLA load
         │                  check    check     check
  [RESTART STOP]
```

### ENV-02 Data Flow (EGL Rendering)

```
[Notebook cell (top of post-restart block)]
       │
   os.environ["MUJOCO_GL"] = "egl"        ← MUST be first
   os.environ["PYOPENGL_PLATFORM"] = "egl"
       │
   write /usr/share/glvnd/egl_vendor.d/10_nvidia.json  ← MUST precede import
       │
   import libero  →  reads ~/.libero/config.yaml        ← MUST exist or EOFError
       │
   OffScreenRenderEnv(bddl_file=..., camera_names=[...])
       │
   env.reset() → obs dict
       │
   obs["agentview_image"][::-1]  → RGB array (flip: MuJoCo renders upside-down)
       │
   assert frame.mean() > 5.0  → non-black check
       │
   plt.imsave() + plt.imshow()
```

### ENV-03 Data Flow (VLA Loading)

```
[Install cell]
    pip install git+https://github.com/moojink/transformers-openvla-oft.git
    pip install -e /path/to/openvla-oft  (for get_vla_action utilities)

[Load cell]
    from transformers import AutoModelForVision2Seq, AutoProcessor
    
    CHECKPOINT = "moojink/openvla-7b-oft-finetuned-libero-spatial"
    
    processor = AutoProcessor.from_pretrained(CHECKPOINT, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        CHECKPOINT,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to("cuda")
    
    # ENV-03 smoke test: dummy image + prompt → 7-D action
    from PIL import Image
    import torch
    dummy_image = Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8))
    prompt = "In: What action should the robot take to pick up the black bowl?\nOut:"
    inputs = processor(prompt, dummy_image).to("cuda:0", dtype=torch.bfloat16)
    action = model.predict_action(**inputs, unnorm_key="libero_spatial", do_sample=False)
    assert action.shape == (7,), f"Expected (7,), got {action.shape}"
    print(f"ENV-03: PASS — action shape {action.shape}, dtype {action.dtype}")
```

**Note on `unnorm_key`:** The OFT checkpoints include their own action normalization statistics. For `moojink/openvla-7b-oft-finetuned-libero-spatial`, use `unnorm_key="libero_spatial"`. The correct key name is stored in the model's `dataset_statistics` attribute. [ASSUMED — key name needs verification against actual checkpoint; if key raises KeyError, print `model.norm_stats.keys()` to discover the correct key]

### Recommended Notebook Structure

```
LIBERO/notebooks/01-colab-env-setup.ipynb
├── [MD] Cell 0: Title + Phase 1 goal + ENV-01/02/03 checklist
├── [CODE] Cell 1: GPU assertion (torch.cuda.get_device_name)
├── [MD] Cell 2: "BLOCK A: Install (run once, then restart)"
├── [CODE] Cell 3: apt-get EGL system packages
├── [CODE] Cell 4: pip PyTorch 2.2.0
├── [CODE] Cell 5: pip mujoco + sim stack
├── [CODE] Cell 6: pip LIBERO editable
├── [CODE] Cell 7: pip openvla-oft custom transformers fork
├── [CODE] Cell 8: pip flash-attn
├── [MD] Cell 9: *** STOP — RESTART RUNTIME, then continue below ***
├── [CODE] Cell 10: NVIDIA EGL ICD creation + MUJOCO_GL=egl (FIRST POST-RESTART CELL)
├── [CODE] Cell 11: LIBERO config.yaml bootstrap (~/.libero/)
├── [CODE] Cell 12: sys.path setup for LIBERO import
├── [MD] Cell 13: "BLOCK B: Verification"
├── [CODE] Cell 14: ENV-01 — version printout PASS/FAIL
├── [CODE] Cell 15: ENV-02 — OffScreenRenderEnv render + save PNG PASS/FAIL
├── [CODE] Cell 16: ENV-03 — OpenVLA-OFT load + action shape PASS/FAIL
└── [MD] Cell 17: Summary table
```

### Anti-Patterns to Avoid

- **Setting MUJOCO_GL after import:** `os.environ["MUJOCO_GL"]` must be set before the first `import mujoco` or `import robosuite`. Importing robosuite without the env var set causes MuJoCo to initialize the wrong GL backend at C extension load time; re-setting the env var afterward has no effect.
- **Skipping NVIDIA ICD creation:** Setting `MUJOCO_GL=egl` without `/usr/share/glvnd/egl_vendor.d/10_nvidia.json` causes `EGL display initialization failed`. The file must exist before any MuJoCo import.
- **Not bootstrapping ~/.libero/config.yaml:** First import of `libero.libero` calls `input()` if config is absent, causing `EOFError` in non-interactive Colab cells.
- **Installing flash-attn before torch:** flash-attn compiles against the installed PyTorch; torch must be installed first.
- **Using mujoco-py:** robosuite 1.4.x dropped mujoco-py. Use the `mujoco` Python package (DeepMind's official binding). mujoco-py is for robosuite <1.3.
- **Using PyPI `transformers==4.40.1`:** OpenVLA-OFT requires the custom fork for bidirectional attention; PyPI release will load but parallel decoding will silently not work.
- **Using robosuite 1.5.x:** Breaks LIBERO (SingleArmEnv removed in 1.5). Hard-pin to 1.4.0 or 1.4.1.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| MuJoCo EGL rendering | Custom OpenGL setup | `MUJOCO_GL=egl` + NVIDIA ICD JSON | MuJoCo handles EGL context; just needs driver config |
| 7-D action inference | Custom VLA inference loop | `model.predict_action()` from openvla-oft | OFT action head, unnormalization, and chunking are all baked in |
| LIBERO path resolution | Hardcoded paths | `get_libero_path()` / `set_libero_default_path()` | LIBERO's path API handles config.yaml lookups |
| BF16 model loading | Manual dtype casting | `torch_dtype=torch.bfloat16` in `from_pretrained()` | Transformers handles weight casting; avoids precision bugs |
| Package version detection | Custom version string parsing | `importlib.metadata.version('pkg')` | Standard library; works across pip/conda |

**Key insight:** The entire EGL, LIBERO path, and VLA loading surface is pre-solved; the notebook job is sequencing the initialization calls in the correct order, not building any new infrastructure.

---

## Common Pitfalls

### Pitfall 1: MUJOCO_GL import ordering violation
**What goes wrong:** Black frames or `EGL display initialization failed` from OffScreenRenderEnv.
**Why it happens:** MuJoCo's Python extension loads the GL backend at C module import time. If `import mujoco` (or `import robosuite`, which imports mujoco) happens in an earlier cell, `MUJOCO_GL` set in a later cell has no effect.
**How to avoid:** Cell 10 (first post-restart cell) must ONLY set environment variables and create the NVIDIA ICD JSON. No imports in that cell. Cell 11 then imports libero.
**Warning signs:** `obs["agentview_image"].mean() < 1.0` (pure black), or any `EGL` error in stack trace.

### Pitfall 2: Missing NVIDIA EGL ICD on Colab
**What goes wrong:** `ImportError: Cannot initialize a headless EGL display` even with `MUJOCO_GL=egl`.
**Why it happens:** Colab installs the NVIDIA GPU driver via a proprietary mechanism, not apt. The GLVND ICD config file (`/usr/share/glvnd/egl_vendor.d/10_nvidia.json`) that tells EGL to use libEGL_nvidia.so.0 is absent. [CITED: github.com/google-deepmind/mujoco/issues/1424]
**How to avoid:** Programmatically create the ICD file:
```python
import os, json
os.makedirs("/usr/share/glvnd/egl_vendor.d", exist_ok=True)
with open("/usr/share/glvnd/egl_vendor.d/10_nvidia.json", "w") as f:
    json.dump({"file_format_version": "1.0.0",
               "ICD": {"library_path": "libEGL_nvidia.so.0"}}, f)
```
**Warning signs:** `LIBGL_ALWAYS_SOFTWARE is set`, or EGL-related ImportError despite having a GPU.

### Pitfall 3: LIBERO interactive prompt (EOFError)
**What goes wrong:** Cell importing `libero.libero` raises `EOFError: EOF when reading a line` the first time it runs.
**Why it happens:** `LIBERO/libero/libero/__init__.py` (lines 62–96) calls `input()` if `~/.libero/config.yaml` does not exist. [VERIFIED: codebase — read LIBERO/libero/libero/__init__.py]
**How to avoid:** Before any `import libero`, create the config:
```python
import os, yaml
from pathlib import Path
libero_root = "/content/drive/MyDrive/SoARM-Research/LIBERO/libero/libero"
config_dir = Path.home() / ".libero"
config_dir.mkdir(exist_ok=True)
config = {
    "benchmark_root": libero_root,
    "bddl_files": f"{libero_root}/bddl_files",
    "init_states": f"{libero_root}/init_files",
    "datasets": f"{libero_root}/../datasets",
    "assets": f"{libero_root}/assets",
}
with open(config_dir / "config.yaml", "w") as f:
    yaml.dump(config, f)
```
**Warning signs:** `EOFError` in cell that does `from libero.libero.envs import OffScreenRenderEnv`.

### Pitfall 4: sys.path missing LIBERO
**What goes wrong:** `ModuleNotFoundError: No module named 'libero'` even after `pip install -e LIBERO`.
**Why it happens:** On Colab, `pip install -e` in a shell cell may not automatically add the editable package to the current Python session's sys.path until kernel restarts. Alternatively, the working directory must contain the `libero` package or the path must be added explicitly. [ASSUMED]
**How to avoid:**
```python
import sys
LIBERO_PATH = "/content/drive/MyDrive/SoARM-Research/LIBERO/libero"
if LIBERO_PATH not in sys.path:
    sys.path.insert(0, LIBERO_PATH)
```
**Warning signs:** `ModuleNotFoundError` for `libero` after editable install completes.

### Pitfall 5: Transformers fork vs PyPI conflict
**What goes wrong:** OpenVLA-OFT's parallel decoding produces wrong-shaped outputs, or `from prismatic.vla.constants import NUM_ACTIONS_CHUNK` fails.
**Why it happens:** Installing `transformers==4.40.1` from PyPI then installing the git fork later may result in pip resolving to the PyPI version. [ASSUMED — pip dependency resolution order]
**How to avoid:** Install the git fork LAST in the transformers install step. Do not install `transformers` from PyPI at all — the fork replaces it. Verify post-install: `python -c "import transformers; print(transformers.__version__)"` should show `4.40.1` and `transformers.__file__` should point to the git-cloned location, not site-packages.
**Warning signs:** `transformers.__file__` points to `.../site-packages/transformers/` (PyPI version).

### Pitfall 6: Colab runtime disconnect during flash-attn compilation
**What goes wrong:** flash-attn `pip install` runs for 5-15 minutes; Colab session times out.
**Why it happens:** flash-attn compiles CUDA kernels from source; on A100 this takes several minutes.
**How to avoid:** Run `pip install flash-attn==2.5.5 --no-build-isolation -q` as the last install cell and allow time. Consider using `--find-links` with a pre-built wheel if available for the specific CUDA version on Colab. [ASSUMED — pre-built wheel availability]
**Warning signs:** Cell appears to hang after "Running setup.py install".

### Pitfall 7: OOM loading VLA model
**What goes wrong:** `CUDA out of memory` during `AutoModelForVision2Seq.from_pretrained(...)`.
**Why it happens:** Loading in float32 (default) uses ~28GB; A100 40GB has headroom but other CUDA allocations may consume it. T4 (15GB) cannot load 7B model at all without 4-bit quant.
**How to avoid:** Always pass `torch_dtype=torch.bfloat16` and `low_cpu_mem_usage=True`. Verify GPU is A100 (D-06) before attempting load. [CITED: openvla-oft LIBERO.md — "results reported using A100 GPU"]
**Warning signs:** `RuntimeError: CUDA out of memory. Tried to allocate X GiB`.

---

## Code Examples

Verified patterns from official sources and codebase analysis:

### EGL Setup (post-restart, first cell)

```python
# Source: Adapted from github.com/google-deepmind/mujoco/issues/1424 and torchrl MuJoCo docs
import os, json

# Step 1: Create NVIDIA EGL ICD config (Colab-specific requirement)
os.makedirs("/usr/share/glvnd/egl_vendor.d", exist_ok=True)
with open("/usr/share/glvnd/egl_vendor.d/10_nvidia.json", "w") as f:
    json.dump({
        "file_format_version": "1.0.0",
        "ICD": {"library_path": "libEGL_nvidia.so.0"}
    }, f)

# Step 2: Set rendering backend BEFORE any MuJoCo import
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("EGL configured. Do not import mujoco or robosuite before this cell runs.")
```

### LIBERO Config Bootstrap (before any `import libero`)

```python
# Source: Reverse-engineered from LIBERO/libero/libero/__init__.py lines 62-96
import yaml
from pathlib import Path

LIBERO_ROOT = "/content/drive/MyDrive/SoARM-Research/LIBERO/libero/libero"
config_dir = Path.home() / ".libero"
config_dir.mkdir(parents=True, exist_ok=True)

config = {
    "benchmark_root": LIBERO_ROOT,
    "bddl_files":    f"{LIBERO_ROOT}/bddl_files",
    "init_states":   f"{LIBERO_ROOT}/init_files",
    "datasets":      f"{LIBERO_ROOT}/../datasets",
    "assets":        f"{LIBERO_ROOT}/assets",
}
(config_dir / "config.yaml").write_text(yaml.dump(config))
print(f"LIBERO config written to {config_dir / 'config.yaml'}")
```

### ENV-02 Render Check

```python
# Source: Adapted from explorations/create_scene.py (project codebase)
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

LIBERO_PKG = "/content/drive/MyDrive/SoARM-Research/LIBERO/libero"
if LIBERO_PKG not in sys.path:
    sys.path.insert(0, LIBERO_PKG)

from libero.libero.envs import OffScreenRenderEnv

BDDL_FILE = (
    "/content/drive/MyDrive/SoARM-Research/LIBERO/libero/libero/bddl_files/"
    "libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate.bddl"
)

env = OffScreenRenderEnv(
    bddl_file_name=BDDL_FILE,
    camera_names=["agentview"],
    camera_heights=256,
    camera_widths=256,
    has_renderer=False,
    has_offscreen_renderer=True,
)
obs = env.reset()
for _ in range(5):
    obs, _, _, _ = env.step(np.zeros(7))

frame = obs["agentview_image"][::-1]  # flip: MuJoCo images are upside-down

# Verify non-black
mean_pixel = frame.mean()
status = "PASS" if mean_pixel > 5.0 else "FAIL"
print(f"ENV-02: {status} — mean pixel value: {mean_pixel:.2f} (threshold: > 5.0)")

# Save output
out_dir = Path("/content/drive/MyDrive/SoARM-Research/LIBERO/notebooks/outputs")
out_dir.mkdir(parents=True, exist_ok=True)
plt.imsave(out_dir / "libero_render_check.png", frame)

# Display inline
plt.figure(figsize=(4, 4))
plt.imshow(frame)
plt.title("ENV-02 Render Check")
plt.axis("off")
plt.tight_layout()
plt.show()
env.close()
```

### ENV-03 VLA Load Check

```python
# Source: Adapted from openvla-oft README and moojink/openvla-7b-oft-finetuned-libero-spatial model card
import torch
import numpy as np
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

CHECKPOINT = "moojink/openvla-7b-oft-finetuned-libero-spatial"

print(f"Loading processor from {CHECKPOINT}...")
processor = AutoProcessor.from_pretrained(CHECKPOINT, trust_remote_code=True)

print("Loading model in bf16 on GPU...")
model = AutoModelForVision2Seq.from_pretrained(
    CHECKPOINT,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
).to("cuda")

print(f"Model dtype: {next(model.parameters()).dtype}")
print(f"Model device: {next(model.parameters()).device}")

# Minimal inference: dummy 256x256 RGB image
dummy_image = Image.fromarray(
    np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
)
prompt = "In: What action should the robot take to pick up the black bowl?\nOut:"
inputs = processor(prompt, dummy_image).to("cuda:0", dtype=torch.bfloat16)

with torch.no_grad():
    action = model.predict_action(**inputs, unnorm_key="libero_spatial", do_sample=False)

status = "PASS" if action.shape == (7,) else "FAIL"
print(f"ENV-03: {status} — action shape: {action.shape}, dtype: {action.dtype}")
print(f"Action values: {action.cpu().numpy()}")
```

**Note:** If `unnorm_key="libero_spatial"` raises a `KeyError`, use:
```python
print("Available norm keys:", list(model.norm_stats.keys()))
```

### GPU Assertion Cell (D-06)

```python
# Source: Project decision D-06 from CONTEXT.md
import torch

assert torch.cuda.is_available(), "No GPU available. Go to Runtime > Change runtime type > GPU."
gpu_name = torch.cuda.get_device_name(0)
vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

print(f"GPU: {gpu_name}")
print(f"VRAM: {vram_gb:.1f} GB")

if "A100" not in gpu_name:
    print(f"WARNING: Expected A100, got {gpu_name}. OpenVLA-OFT in bf16 requires ~16GB+ VRAM.")
    print("If VRAM < 16GB, Phase 1 ENV-03 will OOM. Restart with A100 runtime.")
else:
    print("A100 confirmed. Proceeding.")
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| mujoco-py (C bindings) | mujoco Python package (DeepMind native) | robosuite 1.4 (2022) | Simpler install, no compilation, but version must match robosuite exactly |
| MUJOCO_GL=glfw on Linux | MUJOCO_GL=egl on headless Linux | Always | GLFW requires a display; EGL works in Colab/Docker headless |
| AutoModelForVision2Seq (standard) | Custom git fork of transformers | OpenVLA-OFT v1 (2024) | Fork adds bidirectional attn; standard HF API still works for load/inference |
| gym (OpenAI) | gymnasium (Farama) | gym 0.26+ | LIBERO pins gym 0.25.2 (pre-rename); do NOT use gymnasium package |

**Deprecated/outdated:**
- `mujoco-py`: Deprecated by OpenAI; incompatible with robosuite 1.4+. Do not install.
- `gymnasium`: This is the successor to `gym` after 0.26. LIBERO uses `gym==0.25.2` (pre-fork). Do not confuse with `gymnasium`.
- `robosuite 1.5.x`: Removed `SingleArmEnv` which LIBERO depends on. Hard-pin to 1.4.x.
- `transformers from PyPI`: For OFT, must use the git fork. PyPI version silently lacks the parallel decoding feature.

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `unnorm_key="libero_spatial"` is the correct key for `moojink/openvla-7b-oft-finetuned-libero-spatial` | Code Examples — ENV-03 | predict_action() raises KeyError; fallback: print model.norm_stats.keys() |
| A2 | `pip install -e LIBERO` does not add to sys.path in current Colab session without restart | Pitfall 4 | sys.path insert may be redundant; harmless if wrong |
| A3 | Pre-built flash-attn wheel exists for Colab's CUDA version | Common Pitfalls | If no pre-built wheel, compile takes 10+ min; not a blocker, just slow |
| A4 | pip dependency resolver will not downgrade transformers fork to PyPI version if installed last | Pitfall 5 | If pip resolves to PyPI version, OFT parallel decoding silently broken; add `--force-reinstall` as safeguard |
| A5 | openvla-oft `pip install -e .` is needed for `get_vla_action` utilities but ENV-03 smoke test only needs `AutoModelForVision2Seq` | Architecture — ENV-03 | Full OFT inference (Phase 3+) requires editable install; Phase 1 load check does not |

**If this table is empty:** All claims in this research were verified or cited — no user confirmation needed.

---

## Open Questions

1. **Exact `unnorm_key` for libero_spatial checkpoint**
   - What we know: Model cards confirm checkpoints have per-suite normalization stats embedded
   - What's unclear: The exact string key name (could be `"libero_spatial"`, `"libero-spatial"`, or model-specific)
   - Recommendation: Add `print(list(model.norm_stats.keys()))` diagnostic to ENV-03 cell; use that value

2. **Google Drive vs repo-local path for LIBERO package**
   - What we know: Colab Pro supports Google Drive mounting at `/content/drive`
   - What's unclear: Whether the researcher will use Drive mounting or `git clone` directly to `/content/` per session
   - Recommendation: Notebook should include both patterns with a config cell at top where researcher sets `REPO_ROOT = "/content/drive/MyDrive/SoARM-Research"` or similar

3. **flash-attn compile time on Colab A100**
   - What we know: flash-attn 2.5.5 must be compiled from source; typically takes 5-15 min on local GPU
   - What's unclear: Whether Colab A100 runtime has flash-attn pre-cached or always compiles
   - Recommendation: Add a `# This cell takes ~10 min` comment; consider conditional install: only install if `importlib.util.find_spec("flash_attn") is None`

---

## Environment Availability

| Dependency | Required By | Available on Colab | Version | Fallback |
|------------|------------|-------------------|---------|----------|
| NVIDIA A100 GPU | ENV-03 (OpenVLA-OFT bf16) | Colab Pro only | 40GB VRAM | T4 requires 4-bit quant (deferred per D-05) |
| CUDA 12.1 | torch 2.2.0 | ✓ Colab | varies | Use `--index-url` matching Colab CUDA version |
| libEGL_nvidia.so.0 | EGL rendering (ENV-02) | ✓ (driver present) | driver version | osmesa (CPU-only) as last resort |
| Python 3.10 | openvla-oft | ✓ Colab default | 3.10.x | Python 3.9 is minimum supported |
| Google Drive | LIBERO package, outputs | Optional | — | git clone directly to /content/ each session |

**Missing dependencies with no fallback:**
- A100 GPU for ENV-03 in bf16 (T4 requires 4-bit quant which is deferred)

**Missing dependencies with fallback:**
- EGL rendering: fallback to `MUJOCO_GL=osmesa` if NVIDIA ICD creation fails (CPU-only rendering, produces valid non-black frames)

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | None (notebook cells are the tests) |
| Config file | N/A |
| Quick run command | Run ENV-01 / ENV-02 / ENV-03 cells in order |
| Full suite command | Run all cells top-to-bottom post-restart |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Check | File Exists? |
|--------|----------|-----------|-----------------|-------------|
| ENV-01 | All deps install without conflicts | smoke | `pip check` in notebook cell + version printout | ❌ Wave 0 — cell to write |
| ENV-02 | Non-black RGB frame from OffScreenRenderEnv | smoke | `assert frame.mean() > 5.0` | ❌ Wave 0 — cell to write |
| ENV-03 | OpenVLA-OFT loads, `action.shape == (7,)` | smoke | `assert action.shape == (7,)` | ❌ Wave 0 — cell to write |

### Sampling Rate

- **Per cell run:** Assertions inline in each verification cell
- **Per phase gate:** All three ENV cells must show `PASS` before `/gsd-verify-work`
- **Phase gate:** All verification cells green + `libero_render_check.png` saved to `LIBERO/notebooks/outputs/`

### Wave 0 Gaps

- [ ] `LIBERO/notebooks/01-colab-env-setup.ipynb` — the notebook itself (entire deliverable)
- [ ] `LIBERO/notebooks/outputs/` — output directory (create with `mkdir -p`)

---

## Security Domain

> `security_enforcement: true` in config.json; ASVS Level 1 applies.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | No | No user auth in research notebook |
| V3 Session Management | No | Colab session managed by Google |
| V4 Access Control | No | Single-researcher local environment |
| V5 Input Validation | Limited | Notebook cells accept no external user input; BDDL file paths are hardcoded |
| V6 Cryptography | No | No encryption needed |

### Known Threat Patterns for Research Notebooks

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Downloading model weights from unverified checkpoints | Spoofing/Tampering | Use `trust_remote_code=True` only with known HuggingFace org checkpoints (moojink, openvla); verify checkpoint SHA before production use |
| Arbitrary code in custom transformers fork | Tampering | Fork is from same author as OFT paper; acceptable risk for research; pin to specific commit for reproducibility |
| Writing to /usr/share/ system paths | Elevation of Privilege | EGL ICD write is required for Colab GPU rendering; limited to a JSON config file with no executable content |

**Security posture:** Research notebook with no external attack surface. Primary concern is supply chain (model weights from HuggingFace, custom transformers fork). Both are from published academic authors with paper attribution.

---

## Project Constraints (from CLAUDE.md)

| Directive | Impact on Phase 1 |
|-----------|------------------|
| Python snake_case naming | Notebook code cells use snake_case variables and function names |
| `matplotlib.use("Agg")` before pyplot import | ENV-02 cell must call `matplotlib.use("Agg")` immediately after `import matplotlib` |
| Saved paths printed with `Saved → {path}` pattern | Output save cells use `print(f"Saved → {out_path}")` |
| `MUJOCO_GL=glfw` is macOS pattern | Colab notebook must use `MUJOCO_GL=egl`, not glfw |
| LIBERO sys.path insert pattern | Notebook replicates `sys.path.insert(0, LIBERO_PATH)` from create_scene.py |
| No enforced linter | Notebook cells have no formatting requirements beyond 4-space indent |
| `UPPER_SNAKE_CASE` for module-level path constants | `LIBERO_ROOT`, `BDDL_FILE`, `OUT_DIR` in notebook cells |
| GSD workflow enforcement | Notebook is created via `/gsd-execute-phase`, not direct file creation |

---

## Sources

### Primary (HIGH confidence — from codebase)

- `LIBERO/libero/libero/__init__.py` — LIBERO config.yaml bootstrap pattern (lines 62-96, verified by Read tool)
- `explorations/create_scene.py` — OffScreenRenderEnv usage, MUJOCO_GL pattern, image flip convention
- `LIBERO/requirements.txt` — pinned versions for LIBERO stack
- `github.com/moojink/openvla-oft/blob/main/pyproject.toml` — exact pinned deps for openvla-oft (fetched via curl, verified)
- `github.com/moojink/openvla-oft/blob/main/LIBERO.md` — model checkpoint IDs, LIBERO eval setup (fetched via WebFetch)

### Secondary (MEDIUM confidence — official documentation)

- `pip index versions robosuite/mujoco/gym/transformers/timm/tokenizers` — PyPI registry version existence (run in bash, verified)
- `github.com/moojink/openvla-oft/blob/main/SETUP.md` — setup instructions (fetched via WebFetch)
- `github.com/moojink/openvla-oft/blob/main/README.md` — quickstart inference pattern

### Tertiary (LOW confidence — web search)

- Web search: MuJoCo EGL Colab setup, NVIDIA ICD JSON creation pattern
- Web search: Colab restart-aware notebook patterns
- `docs.pytorch.org/rl/main/reference/generated/knowledge_base/MUJOCO_INSTALLATION.html` — EGL apt packages list

---

## Metadata

**Confidence breakdown:**
- Standard stack (versions): HIGH — verified against PyPI registry and pyproject.toml from source
- OpenVLA-OFT model IDs: MEDIUM — from LIBERO.md in official openvla-oft repo (fetched)
- Architecture (EGL pattern): MEDIUM — corroborated by multiple web sources and create_scene.py pattern
- Colab NVIDIA ICD JSON: LOW — from web search and MuJoCo GitHub issue; widely reported pattern

**Research date:** 2026-07-08
**Valid until:** 2026-08-08 (stable packages; custom transformers fork may update)
