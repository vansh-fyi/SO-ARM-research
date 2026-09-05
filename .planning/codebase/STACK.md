# Technology Stack

**Analysis Date:** 2026-09-05

## Languages

**Primary:**
- Python 3.x - all research, simulation, robot-control, and data-exploration code
  - `LIBERO/` requires Python 3.12 specifically (cp312-only wheels; see `LIBERO/requirements.txt` header comment)
  - `control/` and `diagnostics/` use local `.venv` folders on Python 3.12 (`control/.venv/lib/python3.12/...`)

**Secondary:**
- Jupyter Notebook (`.ipynb`) - Colab-hosted training/eval workflow, `LIBERO/notebooks/*.ipynb` (env setup, SOARM integration check, OFT inference eval, pi0 smoketest, finetune, eval)

## Runtime

**Environment:**
- CPython, orchestrated primarily via Google Colab (T4/A100 GPU tiers)
- Local execution also supported for `explorations/`, `control/`, and `diagnostics/` (macOS/Linux with a physical SOARM arm)
- MuJoCo physics engine (via robosuite/mujoco) — powers all simulation rendering

**Package Manager:**
- pip, per-subproject `requirements.txt` (four independent requirement sets: `LIBERO/requirements.txt`, `explorations/requirements.txt`, `control/requirements.txt`, `diagnostics/requirements.txt`)
- Lockfile: not present (loose/pinned version pins in requirements files, no `pip freeze` lockfile or `uv.lock`)
- `control/` and `diagnostics/` each have a project-local `.venv/` (not committed — see `.gitignore`)

## Frameworks

**Core (simulation / robot learning):**
- robosuite `1.4.0` - MuJoCo-based robot simulation environments (Panda arm, cameras); required by LIBERO env creation (`OffScreenRenderEnv`)
- MuJoCo `3.3.2` (Colab OpenVLA-OFT/π0 stack) / MuJoCo `2.3.7`-class via robosuite `1.4.0` for local LIBERO core — physics engine
- bddl `1.0.1` - LIBERO task-definition DSL (PDDL-like)
- robomimic `0.2.0` - robot-learning dataset/training utilities (`SequenceDataset`, `FileUtils`, `ObsUtils`)
- Hydra `hydra-core==1.2.0` - config management for LIBERO lifelong-learning training runs (`LIBERO/libero/lifelong/main.py`)
- gym `0.25.2` (core) / `gym>=0.21,<=0.26` (Colab Block A) - RL environment interface

**VLA / fine-tuning stack (Colab, referenced in `LIBERO/requirements.txt` "REFERENCE ONLY" section and notebooks):**
- torch `2.2.0` + torchvision `0.17.0` + torchaudio `2.2.0` (cu121 wheels) - OpenVLA-OFT/LoRA stack
- transformers `4.21.1` (core LIBERO) / transformers-openvla-oft fork (Colab, replaces stock transformers for OFT notebooks)
- peft `0.11.1` - LoRA adapters
- diffusers `0.30.3` - force-pinned for peft compatibility
- timm `0.9.10`, tokenizers `0.19.1`, sentencepiece `0.1.99`
- accelerate `1.14.0` - Phase 6 fine-tuning/eval
- openpi/π0 stack - managed by openpi's own `pyproject.toml` (torch `2.7.1`, transformers `4.53.2`), not pip-installed by this repo; referenced in `notebooks/03b-pi0-inference-smoketest.ipynb`
- dlimp, openvla-oft, transformers-openvla-oft - installed via `git+https://...` with `--no-deps`, not expressible in a plain `requirements.txt` (documented as individual pip commands in `LIBERO/requirements.txt`)
- tensorflow `>=2.16` (non-Colab) / Colab's preinstalled tensorflow (stock `2.15.0` has no cp312 wheel) - required by `tensorflow-datasets`/`tensorflow-graphics`/dlimp (OXE dataset registration)
- tensorflow-datasets `4.9.10`, tensorflow-graphics `2021.12.3` - installed `--no-deps`

**Testing:**
- pytest - `.pytest_cache/` present at repo root and under `LIBERO/`; no dedicated test framework config file found beyond cache artifacts

**Robot control (physical hardware):**
- lerobot `0.6.1` (`[feetech]` extra) - `control/requirements.txt`; SOARM servo control, calibration, dataset recording
- feetech-servo-sdk `1.0.0` - `diagnostics/requirements.txt`; low-level Feetech STS/SCS servo protocol
- pyserial `3.5` - serial port communication with servo bus
- opencv-python `5.0.0.93` (`diagnostics/`, `control/`) / `opencv-python==4.6.0.66` (`LIBERO/`) / `opencv-python-headless>=4.7,<4.10` (Colab) - camera capture and image processing
- pynput `1.8.2` - `control/keyboard_joint_control.py` keyboard teleoperation input
- ultralytics `8.4.138` - `control/requirements.txt`; installed for planned YOLO object-follow feature (not yet wired into any script)

**Build/Dev:**
- Hydra `hydra-core==1.2.0` - CLI config composition for training entry points
- setuptools (`LIBERO/setup.py`) - installs `libero` package; exposes CLI entry points (`lifelong.main`, `lifelong.eval`, `libero.config_copy`, `libero.create_template`)

## Key Dependencies

**Critical:**
- `torch` - core tensor/model operations throughout `LIBERO/libero/` and `explorations/`
- `robosuite` - simulation environment; required for LIBERO env creation (`OffScreenRenderEnv`)
- `robomimic` - dataset pipeline (`SequenceDataset`, `FileUtils`, `ObsUtils`)
- `hydra-core` - entry-point config system for `lifelong.main` and `lifelong.eval`
- `transformers` - language task embedding via `AutoModel`/`AutoTokenizer`
- `lerobot[feetech]` - physical SOARM arm control, calibration GUI, teleoperation, dataset recording (`control/`)
- `feetech-servo-sdk` + `pyserial` - direct servo diagnostics/protection tooling (`diagnostics/`)

**Infrastructure:**
- `huggingface_hub>=0.12.0` - dataset download (`snapshot_download`) for Real-3DQA from HF Hub (`explorations/download_real3dqa.py`); also used by `lerobot` internals for adapter/checkpoint downloads
- `pyarrow` / `pandas` - reading LIBERO Parquet episode chunks from `explorations/data/libero/data/chunk-000/`
- `wandb==0.13.1` - training run logging; required by `LIBERO/libero/lifelong/main.py` (`wandb.init(project="libero", config=cfg)` at `LIBERO/libero/lifelong/main.py:135`)
- `datasets>=2.0.0` - HuggingFace datasets API for LIBERO parquet dataset exploration

## Configuration

**Environment:**
- `MUJOCO_GL=glfw` - set inline in `explorations/create_scene.py` and `explorations/soarm_sanity.py` for macOS headless rendering (osmesa is Linux-only per project constraints)
- `TOKENIZERS_PARALLELISM=false` - set in `LIBERO/libero/lifelong/main.py` to suppress HuggingFace warnings
- `LIBERO/.env` - present (contents not read; treated as secret/config file, existence noted only)
- `control/.envrc`, `diagnostics/.envrc` - direnv-style local environment files (present, not read)
- No repo-root `.env` file; per-subproject env files instead (`LIBERO/.env`, `control/.envrc`, `diagnostics/.envrc`)
- `~/.libero/config.yaml` (user home directory) - first import of LIBERO reads this; must exist before running any LIBERO-dependent script or the package prompts interactively, causing `EOFError`

**Build:**
- `LIBERO/setup.py` - package install + CLI entry points
- `LIBERO/requirements.txt` - pinned deps for LIBERO benchmark core, plus an extensive "REFERENCE ONLY" section documenting exact Colab notebook pip pins (kept manually in sync — the file's own comments warn these WILL drift if notebook cells change without updating this file)
- `explorations/requirements.txt` - loose deps for exploration scripts
- `control/requirements.txt`, `diagnostics/requirements.txt` - pinned deps for physical-hardware scripts

## Platform Requirements

**Development:**
- macOS or Linux with MuJoCo-compatible display (GLFW or EGL)
- Python 3.12 specifically for the LIBERO/Colab stack (cp312-only wheels for several pins)
- System packages (apt, Debian/Ubuntu) required for MuJoCo headless rendering on non-Colab machines: `libglfw3 libglew-dev libosmesa6-dev libgles2 libglvnd0 libegl-dev libegl1 libgl1-mesa-glx`
- GPU recommended/required for model training and VLA inference (PyTorch CUDA, cu121 wheels pinned for Colab Block A)

**Production:**
- No production deployment target; this is a research-only codebase
- Primary compute platform: Google Colab (T4/A100 GPU tiers), with notebook-driven cell-by-cell installs for the VLA fine-tuning stack
- Physical SOARM robot arm hardware (via `control/` and `diagnostics/` scripts) as a parallel, non-Colab track — serial/USB connection to servos, USB cameras for observation capture

---

*Stack analysis: 2026-09-05*
