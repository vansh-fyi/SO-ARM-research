# Codebase Structure

**Analysis Date:** 2026-10-03

## Directory Layout

```
SoARM-Research/
├── control/                 # Real-hardware harness, teleop, recording, tests (active core)
│   ├── run_vla_episode.py   # Episode harness entry point (--backend)
│   ├── vla_bridge/          # Bridge package: safety, contract, cameras, MLLM support, Colab servers
│   │   ├── backends/        # One module per backend + base_*.py tier helpers
│   │   ├── backend_configs/ # <backend>.json per backend
│   │   ├── *_server.py      # Colab-resident model HTTP servers
│   │   └── policy_server.ipynb
│   ├── test_*.py            # pytest suite (flat, next to sources)
│   ├── conftest.py          # MockRobot and shared fixtures
│   └── outputs/, logs/      # Run artifacts
├── diagnostics/             # Servo bring-up, stereo calibration, Coppelia/MuJoCo alignment, UAT/
├── scripts/                 # URDF conversion/verification, notebook sync, calibration utils
├── So-101/                  # SO-101 URDF + meshes (.dae/.stl)
├── LIBERO/                  # Vendored LIBERO benchmark with SOARM robot + Colab notebooks
├── explorations/            # Dataset/scene exploration scripts (data/ gitignored)
├── orientation/             # Joint/link orientation reference images
├── docs/, research_notes/, reports/, experiment-design/  # Prose docs and research
├── progress-documentation/  # Media (images, videos, episodes)
├── 06b_eval_videos/         # Before/after sim eval videos
├── .planning/               # GSD state: ROADMAP, STATE, phases/, codebase/
└── .claude/ .agents/ .codex/ .opencode/ .github/  # GSD tooling per AI CLI
```

## Directory Purposes

**`control/`:**
- Purpose: Everything that drives the physical SO-ARM101
- Key files: `run_vla_episode.py`, `record_episode.py`, `keyboard_joint_control.py`, `joint_jog.py`, `detect_devices.py`, `device_map.json`, `stereo_calibration.json`, `COMMANDS.md`

**`control/vla_bridge/`:**
- Purpose: Policy bridge package
- Key files: `safety_validator.py`, `action_contract.py`, `robot_client.py`, `io_logger.py`, `stereo_camera.py`, `depth_camera.py`, `depth_recorder.py`, `mllm_*.py`, `check_checkpoint.py`, `FINDINGS.md`, `checkpoint_candidates.md`, `policy_server_launch.md`

**`control/vla_bridge/backends/`:**
- Purpose: Registered backends (`act`, `diffusion_policy`, `gemini_robotics[_2]`, `groot_n15/n17`, `mllm_claude/gpt/hf_*`, `octo`, `openvla[_oft]`, `pi0` (+pi05), `rdt1b`, `scripted`, `smolvla`, `wallx`)
- Shared helpers: `base_grpc.py`, `base_http_adapter.py`, `base_gemini.py`, `base_mllm.py`, `base_mllm_build.py`, `base_mllm_chat.py`, conversions `eef_to_joint.py`, `relative_ee_conversion.py`, `er_conversion.py`

**`diagnostics/`:**
- Purpose: Hardware bring-up and calibration tooling; `UAT/` holds acceptance scripts (`assembly`, `components`, `function`)

**`LIBERO/`:**
- Purpose: Simulation benchmark; SOARM model in `LIBERO/libero/libero/assets/robots/soarm101`, notes in `LIBERO/SOARM_MODEL.md`, notebooks in `LIBERO/notebooks/`

## Key File Locations

**Entry Points:**
- `control/run_vla_episode.py`: Real-hardware episode
- `control/record_episode.py`: Demonstration recording
- `LIBERO/notebooks/01-colab-env-setup.ipynb` .. `06b-eval.ipynb`: Sim pipeline
- `control/vla_bridge/policy_server.ipynb`: Colab server host

**Configuration:**
- `control/vla_bridge/backend_configs/*.json`, `control/device_map.json`, `control/stereo_calibration.json`, `control/requirements.txt`, `diagnostics/requirements.txt`, `LIBERO/requirements.txt`, `.planning/config.json`

**Core Logic:**
- `control/vla_bridge/safety_validator.py`, `control/vla_bridge/backends/__init__.py`, `control/vla_bridge/backends/base_mllm.py`

**Testing:**
- `control/test_*.py`, `control/conftest.py`, `scripts/test_*.py`, `diagnostics/test_coppelia_alignment.py`

## Naming Conventions

**Files:**
- `snake_case.py`; tests `test_<module>.py`; backend tests `test_backend_<name>.py`
- Backends: `backends/<name>.py`, shared helpers `base_<tier>.py`, private helpers `_<name>.py`
- Colab servers: `<model>_server.py`; configs `backend_configs/<name>.json` matching `^[A-Za-z0-9][A-Za-z0-9_-]*$`
- Diagnostics: verb-prefixed `servo_*.py`, `tune_*.py`, `capture_*.py`, `verify_*.py`

**Directories:**
- lowercase / snake_case; GSD phases `.planning/phases/NN-slug/`

## Where to Add New Code

**New policy backend:**
- Primary code: `control/vla_bridge/backends/<name>.py` with `@register("<name>")` or `@register_grpc_backend("<name>")`; for MLLM providers use `register_mllm_backend()` and `base_mllm_build.build_mllm_backend`
- Config: `control/vla_bridge/backend_configs/<name>.json`
- Colab server (Tier C): `control/vla_bridge/<name>_server.py` + section in `policy_server_launch.md`
- Tests: `control/test_backend_<name>.py`

**New bridge utility:**
- Implementation: `control/vla_bridge/<name>.py`; tests `control/test_<name>.py`

**Hardware diagnostic:** `diagnostics/<verb>_<thing>.py`

**URDF/MJCF tooling:** `scripts/`

**Sim/exploration work:** `explorations/` or `LIBERO/notebooks/`

## Special Directories

**`control/outputs/`, `control/logs/`, `diagnostics/outputs/`:** Run artifacts. Generated: Yes.
**`explorations/data/`:** Downloaded datasets. Generated: Yes. Committed: No.
**`control/.venv/`, `diagnostics/.venv/`, `__pycache__/`, `.pytest_cache/`:** Environments/caches. Committed: No.
**`control/vla_bridge/weights/`:** Model weights. Generated: Yes.
**`LIBERO/`:** Vendored upstream with local SOARM additions. Committed: Yes.
**`.claude/`, `.agents/`, `.codex/`, `.opencode/`, `.github/`:** GSD tooling mirrors. Committed: Yes.

---

*Structure analysis: 2026-10-03*
