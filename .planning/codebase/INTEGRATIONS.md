# External Integrations

**Analysis Date:** 2026-07-07

## APIs & External Services

**HuggingFace Hub:**
- Real-3DQA dataset (`Oliver-Ma/Real-3DQA`) - 3D point clouds + QA annotations (~300 MB)
  - SDK/Client: `huggingface_hub.snapshot_download`
  - Auth: None required (public dataset); HF token optional via `HF_TOKEN` env var
  - Download script: `explorations/download_real3dqa.py`

**HuggingFace `datasets` library:**
- LIBERO episode data loaded as Parquet chunks from `data/libero/`
  - SDK/Client: `datasets>=2.0.0`
  - Data format: Parquet files under `data/libero/data/chunk-000/*.parquet`

**HuggingFace `transformers`:**
- Language model / task-embedding backbone used in `LIBERO/libero/lifelong/evaluate.py`
  - SDK/Client: `transformers==4.21.1` (`AutoModel`, `AutoTokenizer`, `pipeline`)
  - Model: loaded at runtime via `get_task_embs` utility in `LIBERO/libero/lifelong/utils.py`

## Data Storage

**Databases:**
- None - all data stored as flat files on disk

**Local Dataset Files:**
- Real-3DQA point clouds: `explorations/Real-3DQA/point_clouds/*.pth` (PyTorch serialized)
- Real-3DQA annotations: `explorations/Real-3DQA/data/test/*.jsonl`
- LIBERO episodes: `explorations/data/libero/data/chunk-000/*.parquet`
- LIBERO task metadata: `explorations/data/libero/meta/info.json`, `tasks.parquet`
- LIBERO HDF5 demonstrations: loaded via `robomimic.utils.dataset.SequenceDataset` (paths configured via Hydra `cfg.folder`)

**File Storage:**
- Local filesystem only; no cloud object storage integrated

**Caching:**
- HuggingFace Hub local cache (default `~/.cache/huggingface/`) used by `snapshot_download`

## Authentication & Identity

**Auth Provider:**
- None - no user authentication; research scripts run locally
- HuggingFace Hub: anonymous access for public datasets; token auth supported but not required

## Monitoring & Observability

**Experiment Tracking:**
- Weights & Biases (wandb `0.13.1`)
  - Used in: `LIBERO/libero/lifelong/main.py`, `LIBERO/libero/lifelong/evaluate.py`
  - Auth: `WANDB_API_KEY` env var (standard wandb login)
  - Logs: training metrics, evaluation results, model checkpoints metadata

**Logs:**
- wandb remote dashboard for training runs
- Local stdout/stderr via Python `pprint` and `print` statements
- Video output written to local disk via `LIBERO/libero/libero/utils/video_utils.py`

## CI/CD & Deployment

**Hosting:**
- No deployment; pure research/experimentation repo

**CI Pipeline:**
- None detected

## Environment Configuration

**Required env vars:**
- `MUJOCO_GL` - rendering backend (`glfw` on macOS, `egl` on headless Linux); set in `explorations/create_scene.py`
- `TOKENIZERS_PARALLELISM` - set to `false` in `LIBERO/libero/lifelong/main.py`
- `WANDB_API_KEY` - required for experiment logging in LIBERO training/evaluation runs

**Secrets location:**
- No secrets files detected; API keys set via shell environment variables

## Webhooks & Callbacks

**Incoming:** None

**Outgoing:** None (wandb uses outbound HTTPS to `api.wandb.ai`)

---

*Integration audit: 2026-07-07*
