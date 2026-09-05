# Codebase Structure

**Analysis Date:** 2026-09-05

## Directory Layout

```
SoARM-Research/
├── explorations/            # Research scripts: dataset exploration, MuJoCo scene rendering
│   ├── lib/                 # LIBERO HF-dataset exploration (explore.py)
│   ├── real3dqa/             # Real-3DQA point-cloud exploration (explore.py)
│   ├── Real-3DQA/             # Real-3DQA raw data/assets (gitignored-style)
│   ├── dreamzero/manipulation/ # DreamZero exploration scripts
│   ├── data/                 # Local datasets consumed by exploration scripts (gitignored)
│   ├── outputs/               # Rendered PNGs from exploration scripts (gitignored)
│   ├── perspective_images/    # Perspective render outputs
│   ├── create_scene.py        # Instantiate MuJoCo LIBERO env, render multi-camera frames
│   ├── render_scenes.py       # Software-rasterize point clouds to PNG
│   ├── download_real3dqa.py   # Download Real-3DQA dataset
│   └── soarm_sanity.py        # SOARM model/env sanity check
├── LIBERO/                   # Vendored LIBERO benchmark (own .git history, treat as dependency)
│   ├── libero/
│   │   ├── libero/            # Core sim library: envs/, bddl_files/, assets/, benchmark/, perception/, vla/, utils/
│   │   ├── lifelong/          # Continual learning training loop, algos/, models/, main.py, evaluate.py
│   │   ├── configs/            # Hydra configs for training runs
│   │   └── datasets/           # LIBERO dataset definitions
│   ├── notebooks/              # Colab pipeline notebooks (01 through 06b)
│   ├── benchmark_scripts/      # Dataset download / integrity / rendering CLI scripts
│   ├── scripts/                 # Misc LIBERO utility scripts
│   ├── templates/               # BDDL/task templates
│   └── images/                  # Static assets for docs
├── control/                  # Physical SOARM teleoperation and data capture
│   ├── joint_jog.py
│   ├── keyboard_joint_control.py
│   ├── record_episode.py
│   ├── requirements.txt / .venv/
│   └── outputs/               # Captured teleop episodes
├── diagnostics/               # Physical SOARM bring-up, calibration, UAT
│   ├── servo_scan.py, servo_set_id.py, servo_set_torque_limit.py,
│   │   servo_set_protection.py, servo_move_test.py, servo_drive_to_stall.py,
│   │   servo_torque.py, camera_test.py
│   ├── UAT/                   # assembly/, components/, function/ checklists
│   ├── PARTS_LIST.md
│   ├── requirements.txt / .venv/
│   └── outputs/                # Camera test images
├── progress-documentation/    # Physical build docs and 3D print STLs
│   └── 3d_print_stls/arm/, gripper_sim_meshes/, gripper_upstream_full/
├── 06b_eval_videos/           # Generated eval videos: before/ and after/ per episode
├── docs/                      # Project-level markdown docs (PHASES-1-3-EXPLAINER.md)
├── .planning/                 # GSD planning state
│   ├── PROJECT.md, ROADMAP.md, REQUIREMENTS.md, STATE.md, config.json
│   ├── phases/                 # Per-phase plan directories
│   ├── quick/                  # Ad-hoc quick-task logs (dated, slug-named)
│   ├── codebase/                # Generated codebase maps (this document set)
│   ├── notes/, seeds/, todos/, research/
├── .claude/                   # Claude Code CLI: commands/, skills/gsd-*/, agents/, hooks/, gsd-core/
├── .agents/                   # Mirrored GSD skill/agent set (generic agent backend)
├── .codex/                    # Mirrored GSD skill/agent set (Codex backend)
├── .github/                   # Mirrored GSD skill/agent set (GitHub Copilot backend)
├── .opencode/                 # Mirrored GSD skill/agent set (OpenCode backend)
└── .pytest_cache/             # Pytest cache (root-level; LIBERO/ has its own)
```

## Directory Purposes

**`explorations/`:**
- Purpose: Self-contained CLI research scripts for dataset visualization and MuJoCo scene inspection
- Contains: `*.py` scripts, per-topic subdirectories (`lib/`, `real3dqa/`, `dreamzero/`), `data/` and `outputs/` for I/O
- Key files: `explorations/create_scene.py`, `explorations/lib/explore.py`, `explorations/real3dqa/explore.py`

**`LIBERO/`:**
- Purpose: Vendored LIBERO benchmark (MuJoCo/robosuite simulation, BDDL tasks, lifelong learning training)
- Contains: Core sim library (`libero/libero/`), training loop (`libero/lifelong/`), Colab notebooks, benchmark CLI scripts
- Key files: `LIBERO/libero/lifelong/main.py`, `LIBERO/libero/libero/envs/env_wrapper.py`, `LIBERO/notebooks/*.ipynb`

**`control/`:**
- Purpose: Teleoperation and data-capture scripts for the physical SOARM arm
- Contains: Joint control scripts, episode recorder, isolated `.venv`/`requirements.txt`
- Key files: `control/keyboard_joint_control.py`, `control/record_episode.py`

**`diagnostics/`:**
- Purpose: Hardware bring-up, servo calibration, camera checks, and manual UAT tracking
- Contains: Per-servo diagnostic scripts, `UAT/` checklists, `PARTS_LIST.md`, isolated `.venv`
- Key files: `diagnostics/servo_scan.py`, `diagnostics/README.md`, `diagnostics/UAT/`

**`progress-documentation/`:**
- Purpose: Physical build reference material — 3D-printable STL files for arm and gripper
- Contains: STL mesh files organized by part category
- Generated: No (source CAD-derived assets)
- Committed: Yes

**`06b_eval_videos/`:**
- Purpose: Generated evaluation videos from `LIBERO/notebooks/06b-eval.ipynb`, before/after fine-tuning comparison
- Contains: Per-episode video directories under `before/` and `after/`
- Generated: Yes
- Committed: Appears tracked (present in repo tree; verify `.gitignore` before adding more)

**`.planning/`:**
- Purpose: GSD workflow state — roadmap, requirements, phase plans, codebase maps, quick-task logs
- Contains: Markdown/JSON state files, `phases/`, `quick/`, `codebase/`, `notes/`, `seeds/`, `todos/`
- Key files: `.planning/PROJECT.md`, `.planning/ROADMAP.md`, `.planning/STATE.md`

**`.claude/`, `.agents/`, `.codex/`, `.github/`, `.opencode/`:**
- Purpose: GSD command/skill/agent definitions, mirrored per supported AI CLI backend
- Contains: `commands/` (or `command/`), `skills/gsd-*/`, `agents/`, `hooks/`, `gsd-core/` (bin/contexts/references/templates/workflows), `scripts/changeset/`
- Key files: `.claude/commands/*.md`, `.claude/skills/gsd-*/SKILL.md`

## Key File Locations

**Entry Points:**
- `explorations/create_scene.py`: Local MuJoCo scene rendering CLI
- `explorations/lib/explore.py`, `explorations/real3dqa/explore.py`: Dataset exploration CLIs
- `LIBERO/libero/lifelong/main.py`: Hydra-driven training entrypoint
- `LIBERO/notebooks/01-colab-env-setup.ipynb` … `06b-eval.ipynb`: Colab pipeline, run sequentially
- `control/*.py`, `diagnostics/*.py`: Physical hardware CLIs

**Configuration:**
- `LIBERO/libero/configs/`: Hydra training configs
- `~/.libero/config.yaml`: LIBERO runtime config (created on first import, home directory)
- `control/requirements.txt`, `control/.envrc`; `diagnostics/requirements.txt`, `diagnostics/.envrc`
- `explorations/requirements.txt`
- `LIBERO/requirements.txt`
- `.planning/config.json`: GSD workflow configuration

**Core Logic:**
- `LIBERO/libero/libero/envs/env_wrapper.py`: `OffScreenRenderEnv`
- `LIBERO/libero/lifelong/algos/`: Continual learning algorithm implementations
- `LIBERO/libero/lifelong/datasets.py`, `models/`: Training data pipeline and model definitions

**Testing:**
- `LIBERO/.pytest_cache/`, root `.pytest_cache/`: pytest cache directories (test suite location not confirmed under `LIBERO/` — see TESTING.md if generated)
- `diagnostics/UAT/`: manual hardware UAT checklists (assembly/components/function), not automated tests

## Naming Conventions

**Files:**
- `snake_case.py` throughout: `create_scene.py`, `record_episode.py`, `servo_set_torque_limit.py`
- Verb-noun / action-prefixed names: `download_*`, `render_*`, `create_*`, `explore*`, `servo_*`
- Notebooks are numerically prefixed to indicate pipeline order: `01-colab-env-setup.ipynb`, `02-soarm-integration-check.ipynb`, `03a-`/`03b-`, `06a-`/`06b-`

**Directories:**
- Topic-scoped subdirectories under `explorations/` (`lib/`, `real3dqa/`, `dreamzero/`)
- Isolated per-track dependency directories: `control/.venv`, `diagnostics/.venv`, each with sibling `requirements.txt`
- GSD backend directories are dot-prefixed and named after the CLI they target (`.claude/`, `.codex/`, `.opencode/`, `.github/`, `.agents/`)
- `.planning/quick/` entries use `YYMMDD-<hash>-<slug>` naming for ad-hoc task logs

## Where to Add New Code

**New exploration/research script:**
- Primary code: `explorations/<topic>/explore.py` or a new top-level `explorations/*.py` following the `verb_noun.py` pattern
- Data: `explorations/data/<topic>/`
- Outputs: `explorations/outputs/`

**New LIBERO lifelong-learning algorithm:**
- Implementation: `LIBERO/libero/lifelong/algos/<algo>.py`, subclassing `LIBERO/libero/lifelong/algos/base.py`
- Config: `LIBERO/libero/configs/`

**New Colab pipeline step:**
- Notebook: `LIBERO/notebooks/<NN><letter>-<name>.ipynb`, following the existing numeric-prefix ordering convention

**New hardware diagnostic/control script:**
- Diagnostics (bring-up/calibration/UAT): `diagnostics/<verb>_<component>.py`
- Teleop/data capture: `control/<verb>_<noun>.py`
- Both tracks: install into the track's own `.venv` via its `requirements.txt`, do not mix with the LIBERO/exploration environment

**New GSD skill/command:**
- Add to the canonical backend directory (verify authoritative source via `.claude/scripts/changeset` before editing) and mirror across `.agents/`, `.codex/`, `.github/`, `.opencode/` as needed

## Special Directories

**`explorations/data/`, `explorations/outputs/`, `control/outputs/`, `diagnostics/outputs/`:**
- Purpose: Local input datasets and generated output artifacts
- Generated: Outputs yes; data yes (downloaded)
- Committed: No (gitignored research/output data)

**`06b_eval_videos/`:**
- Purpose: Before/after fine-tuning evaluation video output from the eval notebook
- Generated: Yes
- Committed: Present in working tree — verify with `git status`/`.gitignore` before assuming tracked

**`LIBERO/`:**
- Purpose: Vendored external benchmark with its own `.git` history
- Generated: No (cloned/vendored)
- Committed: Yes, as a subtree within the parent repo (not a git submodule based on structure observed)

**`.planning/`:**
- Purpose: GSD workflow state, not application code
- Generated: Partially (phase artifacts generated by GSD commands)
- Committed: Yes

---

*Structure analysis: 2026-09-05*
