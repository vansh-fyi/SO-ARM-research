# Technology Stack

**Analysis Date:** 2026-10-03

## Languages

**Primary:**
- Python 3.12 - all simulation, robot-control, VLA-bridge, diagnostics, and exploration code
  - `LIBERO/` Colab stack requires Python 3.12 (cp312-only wheels; see header of `LIBERO/requirements.txt`)
  - `control/.venv/` is Python 3.12; `diagnostics/.venv/` is Python 3.14 (local, not committed)

**Secondary:**
- Jupyter Notebook - Colab-hosted workflows: `LIBERO/notebooks/*.ipynb` (env setup, SOARM integration, OFT eval, pi0 smoketest, finetune) and `control/vla_bridge/policy_server.ipynb` (Colab policy/adapter server host)
- URDF/MJCF XML - robot model: `So-101/So-101.urdf` plus `.stl`/`.dae` meshes; SOARM MJCF under `LIBERO/` (see `LIBERO/SOARM_MODEL.md`)

## Runtime

**Environment:**
- CPython on Google Colab (T4/A100) for simulation, fine-tuning, and model serving
- Local macOS/Linux for `control/` (physical SO-101 arm over USB serial + USB cameras), `diagnostics/`, `explorations/`, `scripts/`
- MuJoCo physics (`mujoco==3.3.2` on Colab stack; robosuite `1.4.0`)

**Package Manager:**
- pip with per-subproject requirement files: `LIBERO/requirements.txt`, `explorations/requirements.txt`, `control/requirements.txt`, `diagnostics/requirements.txt`
- Lockfile: missing (exact pins in `control/` and `diagnostics/`; loose pins in `explorations/`)
- Colab-side model environments are installed by notebook cells in `control/vla_bridge/policy_server.ipynb`, often into isolated venvs per backend (e.g. `fastfs_venv`); `scripts/sync_policy_notebook.py` keeps notebook cells in sync with source files

## Frameworks

**Core (simulation / robot learning):**
- robosuite `1.4.0`, bddl `1.0.1`, robomimic `0.2.0`, gym `0.25.2` - LIBERO env stack (`LIBERO/libero/libero/envs/`)
- hydra-core `1.2.0` - LIBERO lifelong training config (`LIBERO/libero/lifelong/main.py`)
- transformers `4.21.1` (LIBERO core) / `transformers-openvla-oft` fork (Colab OFT)

**VLA / fine-tuning (Colab, documented in `LIBERO/requirements.txt` REFERENCE ONLY section):**
- torch `2.2.0` / torchvision `0.17.0` (cu121), peft `0.11.1`, diffusers `0.30.3`, timm `0.9.10`, tokenizers `0.19.1`, draccus `0.8.0`, json-numpy
- openvla-oft, dlimp installed via `git+https` with `--no-deps`; tensorflow-datasets for OXE registration
- openpi (pi0) managed by its own `pyproject.toml`; not pinned here

**Robot control / VLA bridge (`control/requirements.txt`):**
- lerobot `[feetech,async]==0.6.1` - SO-101 servo control, calibration, recording, and `async_inference` gRPC PolicyServer/RobotClient
- opencv-python `5.0.0.93`, pynput `1.8.2`, requests `2.34.2`, flask `3.1.3` (test double for Colab Flask adapters)
- ikpy `4.1.0` - URDF FK/IK for EEF-delta to joint conversion (`control/vla_bridge/backends/eef_to_joint.py`)
- google-genai `2.25.0` - Gemini Robotics-ER backends (`control/vla_bridge/backends/base_gemini.py`)
- anthropic `1.11.0`, openai `3.23.0` - MLLM backends (`control/vla_bridge/backends/mllm_claude.py`, `mllm_gpt.py`, `mllm_hf_*.py`)
- ultralytics `8.4.138` - installed for planned YOLO follow; not wired in

**Diagnostics (`diagnostics/requirements.txt`):**
- feetech-servo-sdk `1.0.0`, pyserial `3.5`, opencv-python `5.0.0.93`, numpy `2.5.2`

**Testing:**
- pytest `8.3.4` - `control/test_*.py`, `control/conftest.py`, `control/fake_providers.py`, `scripts/test_*.py`

**Build/Dev:**
- setuptools via `LIBERO/setup.py` (CLI entry points `lifelong.main`, `lifelong.eval`, `libero.config_copy`, `libero.create_template`)

## Key Dependencies

**Critical:**
- `lerobot==0.6.1` - hardware driver and gRPC policy bridge (`control/vla_bridge/robot_client.py`, `control/vla_bridge/backends/base_grpc.py`)
- `robosuite` / `mujoco` - simulation (`OffScreenRenderEnv`)
- `torch` - model inference/training on Colab
- Vendor SDKs `anthropic`, `openai`, `google-genai` - imported lazily inside backend adapter factories only

**Infrastructure:**
- `flask` - Colab-hosted Tier C HTTP adapters (`control/vla_bridge/*_server.py`, `fastfs_server.py`)
- `pyngrok==8.1.2` (Colab only) - tunnels exposing Colab servers to the local control loop
- `huggingface_hub` - checkpoints/datasets; `wandb` - LIBERO training logging
- `pandas`/`pyarrow`/`datasets` - LIBERO parquet exploration (`explorations/lib/explore.py`)

## Configuration

**Environment:**
- API keys from env vars only (never hard-coded/logged): `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, `HF_TOKEN` (named per backend in `control/vla_bridge/backend_configs/*.json` via `api_key_env`)
- Test-only env vars: `SOARM_TEST_GEMINI_API_KEY`, `SOARM_TEST_MLLM_SMOKE_KEY`, etc.
- `MUJOCO_GL=glfw` (macOS) set inline in `explorations/create_scene.py`; `XLA_PYTHON_CLIENT_PREALLOCATE` for JAX backends; `TOKENIZERS_PARALLELISM=false` in `LIBERO/libero/lifelong/main.py`
- Files present, not read: `LIBERO/.env`, `control/.envrc`, `diagnostics/.envrc`
- `~/.libero/config.yaml` must exist before importing LIBERO
- Per-backend JSON configs: `control/vla_bridge/backend_configs/` (model ids, endpoints, action space); hardware map `control/device_map.json`

**Build:**
- `LIBERO/setup.py`, `LIBERO/requirements.txt`, `control/requirements.txt`, `diagnostics/requirements.txt`, `explorations/requirements.txt`

## Platform Requirements

**Development:**
- Python 3.12; macOS/Linux with GLFW/EGL for MuJoCo
- Physical SO-101 arm (Feetech STS3215 servos) over USB serial; USB/stereo/depth cameras for `control/`
- apt packages for headless MuJoCo on Linux: `libglfw3 libglew-dev libosmesa6-dev libegl1` etc.

**Production:**
- No production target; research codebase. Compute on Google Colab, control loop on a local machine connected to Colab via ngrok tunnels

---

*Stack analysis: 2026-10-03*
