# External Integrations

**Analysis Date:** 2026-09-05

## APIs & External Services

**Experiment Tracking:**
- Weights & Biases (wandb) - training/eval run logging for LIBERO lifelong learning
  - SDK/Client: `wandb==0.13.1`
  - Invocation: `wandb.init(project="libero", config=cfg)` in `LIBERO/libero/lifelong/main.py:135`
  - Auth: wandb API key expected via standard wandb login/env var (not found hardcoded in repo)

**Model/Dataset Hub:**
- HuggingFace Hub - dataset and model/adapter distribution
  - SDK/Client: `huggingface_hub` (`snapshot_download`, `hf_hub_download`, `HfApi`)
  - Used directly in `explorations/download_real3dqa.py` (`snapshot_download` to pull the Real-3DQA dataset)
  - Used transitively by `lerobot` (installed in `control/.venv`) for policy checkpoint/adapter download and dataset reward-model weight fetches (`control/.venv/lib/python3.12/site-packages/lerobot/rewards/pretrained.py`, `lerobot/processor/pipeline.py`)
  - Referenced in memory notes as `HF_ADAPTER_REPO_ID` for the Phase 6 fine-tuned adapter (Colab notebooks `LIBERO/notebooks/06a-finetune.ipynb`, `06b-eval.ipynb`) — used to push/pull the fine-tuned OpenVLA-OFT adapter to/from a HF model repo
  - Auth: HF token expected via `huggingface-cli login` / `HF_TOKEN` env var (no token committed in repo)

**Google Colab:**
- Colab MCP server configured for this project — `.mcp.json` registers `colab-mcp` (`git+https://github.com/googlecolab/colab-mcp`, stdio transport via `uvx`) enabling programmatic Colab notebook interaction from the coding agent
- Notebooks in `LIBERO/notebooks/` are the actual execution surface for GPU-dependent training/inference (env setup, SOARM integration check, OFT inference eval, π0 inference smoketest, finetune, eval)

**OpenVLA-OFT / π0 (Physical Intelligence) inference stacks:**
- Not a network API — these are model weights/code pulled via `git+https://github.com/moojink/openvla-oft.git` and `git+https://github.com/moojink/transformers-openvla-oft.git` (installed `--no-deps`, documented as manual pip commands in `LIBERO/requirements.txt` since git-source + `--no-deps` isn't expressible in a plain `requirements.txt` line)
- π0/openpi stack managed by its own `pyproject.toml`, referenced (not vendored) from `LIBERO/notebooks/03b-pi0-inference-smoketest.ipynb`

## Data Storage

**Databases:**
- None. No SQL/NoSQL database detected anywhere in the codebase.

**File Storage:**
- Local filesystem only.
  - `explorations/data/` - downloaded datasets (Real-3DQA `.pth` point clouds, LIBERO HuggingFace parquet chunks), gitignored
  - `explorations/outputs/` - rendered PNG visualizations
  - `control/outputs/` - recorded robot episode data
  - `diagnostics/outputs/` - camera test capture images
  - `06b_eval_videos/before/`, `06b_eval_videos/after/` - eval video artifacts

**Caching:**
- None detected (no Redis/Memcached or similar).

## Authentication & Identity

**Auth Provider:**
- None — this is a local research/CLI tool with no user-facing authentication system.
- Third-party auth only in the sense of API tokens for external services (HuggingFace, wandb), managed via standard CLI login / env vars, not custom-implemented in this repo.

## Monitoring & Observability

**Error Tracking:**
- None (no Sentry/Bugsnag or similar). Errors surface as Python tracebacks.

**Logs:**
- `print()`-based logging throughout exploration scripts (consistent `→` arrow notation for output paths, e.g. `print(f"Saved → {out_path}")`)
- Weights & Biases used as the structured metrics/logging system specifically for LIBERO lifelong-learning training runs (`LIBERO/libero/lifelong/main.py`)

## CI/CD & Deployment

**Hosting:**
- None — research-only codebase, no deployed service.

**CI Pipeline:**
- `.github/` contains only GSD (Get Shit Done) workflow tooling/agents/skills (`.github/gsd-core/`, `.github/agents/`, `.github/skills/`) — this is developer-workflow automation for the Claude Code agent, not a CI/CD pipeline for the research code itself.
- No GitHub Actions workflow files (`.github/workflows/`) detected for build/test/deploy automation.

## Environment Configuration

**Required env vars:**
- `MUJOCO_GL` - rendering backend selector (`glfw` for macOS headless), set inline in scripts, not via `.env`
- `TOKENIZERS_PARALLELISM` - set inline in `LIBERO/libero/lifelong/main.py`
- HuggingFace token (implicit, for `snapshot_download`/`hf_hub_download`/dataset & adapter push-pull) - not found hardcoded; expected via standard HF CLI auth
- wandb API key (implicit, for `wandb.init`) - not found hardcoded; expected via standard wandb CLI auth

**Secrets location:**
- `LIBERO/.env` (present, contents not inspected per security policy)
- `control/.envrc`, `diagnostics/.envrc` (present, direnv-style, contents not inspected)
- No secrets committed directly in tracked `.py`/`.ipynb` source (based on grep of API_KEY/api_key/os.environ patterns — none found hardcoded in project source, only in vendored `.venv` site-packages)

## Webhooks & Callbacks

**Incoming:**
- None.

**Outgoing:**
- None — no webhook dispatch code detected. All external communication is either pull-based dataset/model downloads (HuggingFace Hub) or push-based experiment logging (wandb) via their respective SDKs, not raw HTTP webhook calls.

---

*Integration audit: 2026-09-05*
