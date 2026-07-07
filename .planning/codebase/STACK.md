# Technology Stack

**Analysis Date:** 2026-07-07

## Languages

**Primary:**
- Python 3.x - All research, simulation, and data exploration code

## Runtime

**Environment:**
- Python (CPython) - research/ML workloads
- MuJoCo physics engine (via robosuite) - robot simulation rendering

**Package Manager:**
- pip
- Lockfile: Not present (loose version pins in requirements files)

## Frameworks

**Core (LIBERO lifelong learning benchmark):**
- PyTorch `>=1.11.0` (explorations) / pinned `(no explicit pin, used via robomimic)` - deep learning backbone
- Hydra `hydra-core==1.2.0` - config management for LIBERO training runs
- robomimic `0.2.0` - robot learning dataset/training utilities
- robosuite `1.4.0` - MuJoCo-based robot simulation environments
- transformers `4.21.1` (HuggingFace) - language model / task embedding

**Data / ML:**
- numpy `>=1.21.0` / `1.22.4` - array ops
- pandas `>=1.4.0` - tabular data (Parquet loading)
- pyarrow `>=8.0.0` - Parquet I/O
- torch (PyTorch) `>=1.11.0` - point-cloud loading, model inference
- datasets `>=2.0.0` - HuggingFace datasets API
- einops `0.4.1` - tensor rearrangement

**Visualization:**
- matplotlib `>=3.5.0` / `3.5.3` - all scene/episode visualizations
- Pillow `>=9.0.0` - image decode/encode (PNG bytes from Parquet)
- opencv-python `4.6.0.66` - image processing in LIBERO

**Simulation & RL:**
- robosuite `1.4.0` - tabletop manipulation environments (Panda arm, cameras)
- gym `0.25.2` - RL environment interface
- bddl `1.0.1` - LIBERO task definition language

**Config & Utilities:**
- easydict `1.9` - dot-access config dicts
- omegaconf - used alongside Hydra
- pyyaml - YAML config loading
- cloudpickle `2.1.0` - object serialization
- future `0.18.2` - Python 2/3 compatibility shim
- thop `0.1.1` - FLOPs counter for models

**Monitoring:**
- wandb `0.13.1` - experiment tracking (LIBERO training/evaluation)

## Key Dependencies

**Critical:**
- `torch` - core tensor/model operations throughout `LIBERO/libero/` and `explorations/`
- `robosuite` - simulation environment; required for LIBERO env creation (`OffScreenRenderEnv`)
- `robomimic` - dataset pipeline (`SequenceDataset`, `FileUtils`, `ObsUtils`)
- `hydra-core` - entry-point config system for `lifelong.main` and `lifelong.eval`
- `transformers` - language task embedding via `AutoModel`/`AutoTokenizer`

**Infrastructure:**
- `huggingface_hub>=0.12.0` - dataset download (`snapshot_download`) for Real-3DQA from HF Hub
- `pyarrow` / `pandas` - reading LIBERO Parquet episode chunks from `data/libero/data/chunk-000/`
- `wandb` - training run logging; required by `LIBERO/libero/lifelong/main.py`

## Configuration

**Environment:**
- `MUJOCO_GL=glfw` - set in `explorations/create_scene.py` for macOS headless rendering
- `TOKENIZERS_PARALLELISM=false` - set in `LIBERO/libero/lifelong/main.py` to suppress HF warnings
- No `.env` file; environment variables set inline in scripts

**Build:**
- `LIBERO/setup.py` - installs `libero` package; exposes CLI entry points (`lifelong.main`, `lifelong.eval`, `libero.config_copy`, `libero.create_template`)
- `LIBERO/requirements.txt` - pinned deps for LIBERO benchmark
- `explorations/requirements.txt` - loose deps for exploration scripts

## Platform Requirements

**Development:**
- macOS or Linux with MuJoCo-compatible display (GLFW or EGL)
- Python >= 3 (3.8+ recommended for robomimic/robosuite compatibility)
- GPU recommended for model training (PyTorch CUDA)

**Production:**
- Research-only codebase; no production deployment target
- LIBERO training runs via `python -m lifelong.main` (Hydra-managed)

---

*Stack analysis: 2026-07-07*
