# Codebase Structure

**Analysis Date:** 2026-10-18

## Directory Layout

```
SoARM-Research/
├── control/                    # Real-hardware harness, teleop, recording, tests (active core)
│   ├── run_vla_episode.py      # Episode harness entry point (--backend)
│   ├── record_episode.py, joint_jog.py, keyboard_joint_control.py, tune_execution_hz.py
│   ├── render_depth_heatmaps.py, detect_devices.py, calibrate_with_preview.py, ...
│   ├── stereo_calibration.json # AR0144 stereo calibration (input to depth pipeline)
│   ├── fake_providers.py       # Shared MLLM test doubles (not collected by pytest)
│   ├── conftest.py             # MockRobot and shared fixtures
│   ├── test_*.py               # pytest suite (flat, next to sources)
│   ├── COMMANDS.md, requirements.txt, .envrc
│   ├── vla_bridge/             # Bridge package
│   │   ├── backends/           # One module per backend + base_*.py tier helpers + conversions
│   │   ├── backend_configs/    # <backend>.json per backend (+ mllm_defaults.json)
│   │   ├── notebooks/          # GENERATED per-backend Colab notebooks + README.md
│   │   ├── *_server.py         # Colab-resident model servers / patched gRPC launchers
│   │   ├── colab_server_utils.py  # Colab subprocess/port/diagnostic helper
│   │   ├── backend_smoke.py, mllm_smoke.py  # Robot-free live smoke tools
│   │   ├── stereo_camera.py, stereo_calibration.py, depth_camera.py, depth_recorder.py
│   │   ├── safety_validator.py, action_contract.py, robot_client.py, io_logger.py
│   │   ├── mllm_*.py           # MLLM inputs/prompt/schema/urdf/record
│   │   └── FINDINGS.md, checkpoint_candidates.md, check_checkpoint.py
│   └── outputs/, logs/         # Run artifacts
├── scripts/                    # Colab notebook generator, URDF conversion/verification, calibration utils
├── diagnostics/                # Servo bring-up, stereo calibration, Coppelia/MuJoCo alignment, UAT/
├── So-101/                     # SO-101 URDF + meshes
├── LIBERO/                     # Vendored LIBERO benchmark with SOARM robot + Colab notebooks
├── explorations/               # Dataset/scene exploration scripts (data/ gitignored)
├── docs/                       # Prose docs
├── 06b_eval_videos/            # Before/after sim eval videos
├── .planning/                  # GSD state: ROADMAP, STATE, phases/, quick/, codebase/
└── .claude/ .agents/ .codex/ .opencode/ .github/  # GSD tooling per AI CLI
```

## Directory Purposes

**`control/`:**
- Purpose: Everything that drives the physical SO-ARM101
- Key files: `run_vla_episode.py`, `record_episode.py`, `keyboard_joint_control.py`, `joint_jog.py`, `tune_execution_hz.py`, `render_depth_heatmaps.py`, `detect_devices.py`, `stereo_calibration.json`, `COMMANDS.md`

**`control/vla_bridge/`:**
- Purpose: Policy bridge package (imported as top-level `vla_bridge` from `control/`)
- Key files: `safety_validator.py`, `action_contract.py`, `robot_client.py`, `io_logger.py`, `stereo_camera.py`, `stereo_calibration.py`, `depth_camera.py`, `depth_recorder.py`, `mllm_*.py`, `backend_smoke.py`, `colab_server_utils.py`, `check_checkpoint.py`, `FINDINGS.md`, `checkpoint_candidates.md`

**`control/vla_bridge/backends/`:**
- Purpose: Registered backends (`act`, `diffusion_policy`, `gemini_robotics[_2]`, `groot_n15/n17`, `mllm_claude/gpt/hf_deepseek/hf_kimi/hf_qwen`, `octo`, `openvla[_oft]`, `pi0` (+pi05), `rdt1b`, `scripted`, `smolvla`, `wallx`)
- Shared helpers: `base_grpc.py`, `base_http_adapter.py`, `base_gemini.py`, `base_mllm.py`, `base_mllm_build.py`, `base_mllm_chat.py`; conversions `eef_to_joint.py`, `relative_ee_conversion.py`, `er_conversion.py`

**`control/vla_bridge/notebooks/`:**
- Purpose: 13 self-contained Colab notebooks, one per hosted model/service, named `<lane>_<backend>.ipynb`: `grpc_{smolvla,pi0,pi05,act,diffusion,groot_n17}`, `http_{openvla,openvla_oft,octo,wallx,rdt1b,groot_n15}`, `fastfs_depth`; `README.md` index (ports, GPU, licences, smoke/episode commands)
- Generated: Yes, by `scripts/build_colab_notebooks.py`. Do not edit by hand.

**`scripts/`:**
- Purpose: Repo-level tooling. `build_colab_notebooks.py` + `colab_notebook_specs.py` (notebook generation, stdlib only), `mjcf_to_urdf.py`, `verify_urdf.py`, `calibration_utils.py`, plus `test_*.py`

**`diagnostics/`:**
- Purpose: Hardware bring-up and calibration tooling; `UAT/` holds acceptance material (`assembly`, `components`, `function/basic`, `function/depth`), `reference/`

**`LIBERO/`:**
- Purpose: Simulation benchmark; SOARM model in `LIBERO/libero/libero/assets/robots/soarm101`, notes in `LIBERO/SOARM_MODEL.md`, notebooks in `LIBERO/notebooks/`

## Key File Locations

**Entry Points:**
- `control/run_vla_episode.py`: Real-hardware episode
- `control/vla_bridge/backend_smoke.py`: Robot-free backend smoke (`python -m vla_bridge.backend_smoke`)
- `control/record_episode.py`: Demonstration recording
- `control/vla_bridge/notebooks/*.ipynb`: Colab server hosts
- `scripts/build_colab_notebooks.py`: Notebook generator / drift check
- `LIBERO/notebooks/01-colab-env-setup.ipynb` .. `06b-eval.ipynb`: Sim pipeline

**Configuration:**
- `control/vla_bridge/backend_configs/*.json`, `control/stereo_calibration.json`, `control/device_map.json` (local), `control/requirements.txt`, `control/.envrc`, `diagnostics/requirements.txt`, `LIBERO/requirements.txt`, `scripts/colab_notebook_specs.py`, `.planning/config.json`

**Core Logic:**
- `control/vla_bridge/safety_validator.py`, `control/vla_bridge/backends/__init__.py`, `control/vla_bridge/backends/base_mllm.py`, `control/vla_bridge/colab_server_utils.py`

**Testing:**
- `control/test_*.py` (incl. `test_colab_notebooks.py`, `test_colab_server_utils.py`, `test_depth_pipeline.py`, `test_stereo_calibration.py`), `control/conftest.py`, `control/fake_providers.py`, `scripts/test_*.py`

## Naming Conventions

**Files:**
- `snake_case.py`; tests `test_<module>.py`; backend tests `test_backend_<name>.py`
- Backends: `backends/<name>.py`, shared helpers `base_<tier>.py`, private helpers `_<name>.py`
- Colab servers: `<model>_server.py`; configs `backend_configs/<name>.json` matching `^[A-Za-z0-9][A-Za-z0-9_-]*$`
- Notebooks: `<grpc|http>_<backend_underscored>.ipynb`, depth service `fastfs_depth.ipynb`
- Diagnostics: verb-prefixed `servo_*.py`, `tune_*.py`, `capture_*.py`, `verify_*.py`

**Directories:**
- lowercase / snake_case; GSD phases `.planning/phases/NN-slug/`

## Where to Add New Code

**New policy backend:**
- Primary code: `control/vla_bridge/backends/<name>.py` with `@register("<name>")` or `@register_grpc_backend("<name>")`; for MLLM providers use `register_mllm_backend()` and `base_mllm_build.build_mllm_backend`
- Config: `control/vla_bridge/backend_configs/<name>.json`
- Colab server (Tier C): `control/vla_bridge/<name>_server.py`, then add a spec in `scripts/colab_notebook_specs.py` and run `python scripts/build_colab_notebooks.py`
- Tests: `control/test_backend_<name>.py`

**Changing Colab notebook content:**
- Edit `scripts/colab_notebook_specs.py` (per-notebook data) or shared cells in `scripts/build_colab_notebooks.py`; regenerate; `control/test_colab_notebooks.py` must pass

**New bridge utility:**
- Implementation: `control/vla_bridge/<name>.py`; tests `control/test_<name>.py`

**Hardware diagnostic:** `diagnostics/<verb>_<thing>.py`

**URDF/MJCF tooling:** `scripts/`

**Sim/exploration work:** `explorations/` or `LIBERO/notebooks/`

## Special Directories

**`control/vla_bridge/notebooks/`:** Generated: Yes. Committed: Yes.
**`control/outputs/`, `control/logs/`, `diagnostics/outputs/`:** Run artifacts. Generated: Yes.
**`explorations/data/`:** Downloaded datasets. Generated: Yes. Committed: No.
**`control/.venv/`, `diagnostics/.venv/`, `__pycache__/`, `.pytest_cache/`:** Environments/caches. Committed: No.
**`control/vla_bridge/weights/`:** Model weights. Generated: Yes. Committed: No.
**`LIBERO/`:** Vendored upstream with local SOARM additions. Committed: Yes.
**`.claude/`, `.agents/`, `.codex/`, `.opencode/`, `.github/`:** GSD tooling mirrors. Committed: Yes.

---

*Structure analysis: 2026-10-18*
