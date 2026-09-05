<!-- refreshed: 2026-09-05 -->
# Architecture

**Analysis Date:** 2026-09-05

## System Overview

```text
┌─────────────────────────────────────────────────────────────────────┐
│                     Research Track (Colab / MuJoCo)                  │
├──────────────────────┬────────────────────┬──────────────────────────┤
│  Exploration Scripts  │   LIBERO Benchmark  │   Notebooks (Colab)      │
│  `explorations/`      │   `LIBERO/libero/`  │   `LIBERO/notebooks/`   │
└──────────┬─────────────┴──────────┬──────────┴────────────┬───────────┘
           │                        │                        │
           ▼                        ▼                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│  MuJoCo / robosuite Sim   │  Lifelong Learning Loop │  VLA Inference  │
│  `LIBERO/libero/libero/   │  `LIBERO/libero/         │  (π0 / OFT via  │
│   envs/`                  │   lifelong/`             │  notebooks)     │
└────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                  Physical Hardware Track (parallel)                  │
├──────────────────────────────┬───────────────────────────────────────┤
│  Diagnostics (bring-up/UAT)   │   Control (teleop/data capture)       │
│  `diagnostics/`               │   `control/`                          │
└──────────────────────────────┴───────────────────────────────────────┘
```

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| Real-3DQA Explorer | Load `.pth` point clouds and `.jsonl` QA annotations, render bird's-eye/3D visualizations | `explorations/real3dqa/explore.py` |
| LIBERO Data Explorer | Load HuggingFace parquet episode frames from manipulation dataset, render episode grids | `explorations/lib/explore.py` |
| Scene Creator | Instantiate MuJoCo LIBERO environment from a BDDL file, render multi-camera frames | `explorations/create_scene.py` |
| Point Cloud Renderer | Software-rasterize ScanNet `.pth` point clouds to PNG without GPU | `explorations/render_scenes.py` |
| Dataset Downloader | Download Real-3DQA dataset from remote source | `explorations/download_real3dqa.py` |
| SOARM Sanity Check | Validate SOARM model/env loads correctly before deeper exploration | `explorations/soarm_sanity.py` |
| LIBERO Core Library | Simulation environments (MuJoCo), BDDL-driven task definitions, robot/arena/object classes | `LIBERO/libero/libero/envs/` |
| LIBERO Lifelong Learning | Continual/lifelong learning training loop, algorithms (EWC, AGEM, PackNet, ER), evaluation | `LIBERO/libero/lifelong/` |
| LIBERO Benchmark Scripts | Dataset download, task suite checks, single-task rendering, dataset integrity checks | `LIBERO/benchmark_scripts/` |
| LIBERO Notebooks | Colab-driven pipeline: env setup → integration check → OFT/π0 inference → finetune → eval | `LIBERO/notebooks/*.ipynb` |
| Hardware Diagnostics | Servo bring-up, torque/protection tuning, camera checks, UAT scripts for physical SOARM | `diagnostics/` |
| Hardware Control | Teleop (joint jog/keyboard) and episode recording from the physical arm | `control/` |
| GSD Planning System | Multi-CLI orchestration scaffolding (agents/skills/commands) driving phased execution | `.claude/`, `.agents/`, `.codex/`, `.github/`, `.opencode/` |
| Planning State | Roadmap, phase plans, requirements, and codebase docs consumed by GSD workflow | `.planning/` |

## Pattern Overview

**Overall:** Script-oriented research monorepo with a vendored benchmark dependency (LIBERO) and two parallel hardware/software tracks (simulation research vs. physical robot bring-up), coordinated by a GSD (Get Stuff Done) planning framework rather than a conventional application architecture.

**Key Characteristics:**
- No central application server or client — each script/notebook is an independent, CLI-invoked entry point.
- Simulation (LIBERO/MuJoCo/Colab) and physical hardware (`control/`, `diagnostics/`) are two loosely coupled tracks that will converge once the physical SOARM gripper/embodiment work lands (see `.claude/projects` memory: gripper embodiment blocker).
- `LIBERO/` is a vendored benchmark repo (its own `.git` history) treated as an external dependency, not modified in place except where the project has patched it (see notebooks/quick-task history for LIBERO fixes).
- GSD tooling (`.claude/`, `.agents/`, `.codex/`, `.github/`, `.opencode/`) mirrors the same skill/agent/workflow definitions across multiple AI CLI backends — these directories are near-duplicates of each other, one per supported coding agent.

## Layers

**Exploration Layer:**
- Purpose: One-off and iterative research investigation scripts (dataset exploration, scene rendering, sanity checks)
- Location: `explorations/`
- Contains: Dataset loading, visualization, scene rendering, download utilities
- Depends on: `explorations/data/` for input; numpy/matplotlib/torch/pandas/PIL for processing; `LIBERO/` for MuJoCo env access
- Used by: Researchers running CLI tools directly

**Simulation Environment Layer:**
- Purpose: MuJoCo-based robot simulation environments, task definitions via BDDL
- Location: `LIBERO/libero/libero/envs/`
- Contains: `OffScreenRenderEnv`, arena definitions, robot configs, object models, BDDL parser
- Depends on: MuJoCo 2.3.7, robosuite 1.4.x, BDDL assets in `LIBERO/libero/libero/bddl_files/`
- Used by: `explorations/create_scene.py`, notebooks, lifelong learning training

**Lifelong Learning / Training Layer:**
- Purpose: Training and evaluation of continual learning and VLA-adjacent policies on the LIBERO benchmark
- Location: `LIBERO/libero/lifelong/`
- Contains: `main.py` (Hydra entrypoint), algorithm implementations (`algos/`: EWC, AGEM, ER, PackNet, multitask, single-task), `datasets.py`, `models/`, `metric.py`, `evaluate.py`
- Depends on: Hydra config system (`LIBERO/libero/configs/`), wandb, torch, LIBERO benchmark core
- Used by: Research training runs, invoked from notebooks or CLI

**Notebook Orchestration Layer:**
- Purpose: End-to-end Colab pipeline stitching environment setup, integration checks, VLA inference (OFT/π0), finetuning, and evaluation
- Location: `LIBERO/notebooks/`
- Contains: `01-colab-env-setup.ipynb`, `02-soarm-integration-check.ipynb`, `03a-oft-inference-eval.ipynb`, `03b-pi0-inference-smoketest.ipynb`, `06a-finetune.ipynb`, `06b-eval.ipynb`
- Depends on: LIBERO core + lifelong layers, HuggingFace-hosted adapters/checkpoints
- Used by: Researchers running full pipeline on Colab GPU runtimes

**Hardware Diagnostics Layer:**
- Purpose: Bring-up, calibration, and UAT for the physical SOARM arm and its servos/cameras
- Location: `diagnostics/`
- Contains: Servo scan/torque/protection/ID scripts, camera test, `UAT/` (assembly/components/function checklists), `PARTS_LIST.md`
- Depends on: Physical servo bus hardware, `diagnostics/.venv`, `diagnostics/requirements.txt`
- Used by: Hardware bring-up sessions, run standalone against the physical arm

**Hardware Control Layer:**
- Purpose: Teleoperation and episode data capture from the physical SOARM arm
- Location: `control/`
- Contains: `joint_jog.py`, `keyboard_joint_control.py`, `record_episode.py`, `outputs/` (captured episode data)
- Depends on: Physical hardware, `control/.venv`, `control/requirements.txt`
- Used by: Data collection sessions feeding future physical-embodiment fine-tuning

**Planning/GSD Layer:**
- Purpose: Phase-based project management scaffolding — roadmap, requirements, per-phase plans, codebase docs
- Location: `.planning/`
- Contains: `PROJECT.md`, `ROADMAP.md`, `REQUIREMENTS.md`, `STATE.md`, `phases/`, `quick/` (ad-hoc task logs), `codebase/` (this document set), `notes/`, `seeds/`, `todos/`
- Depends on: Nothing (pure markdown/JSON state)
- Used by: GSD slash commands (`.claude/commands/`, mirrored skill sets in `.agents/`, `.codex/`, `.github/`, `.opencode/`)

## Data Flow

### Simulation Research Flow (Colab)

1. `01-colab-env-setup.ipynb` provisions the Colab runtime and installs LIBERO/robosuite/MuJoCo deps.
2. `02-soarm-integration-check.ipynb` validates SOARM model integration against LIBERO envs (parallels `explorations/soarm_sanity.py`).
3. `03a-oft-inference-eval.ipynb` / `03b-pi0-inference-smoketest.ipynb` run VLA inference (OFT or π0) against `OffScreenRenderEnv` (`LIBERO/libero/libero/envs/env_wrapper.py`) for a BDDL-defined task.
4. `06a-finetune.ipynb` fine-tunes the VLA policy via the lifelong-learning training loop (`LIBERO/libero/lifelong/main.py`), Hydra-configured.
5. `06b-eval.ipynb` evaluates the fine-tuned policy, producing episode videos under `06b_eval_videos/before/` and `06b_eval_videos/after/`.

### Local Exploration Flow

1. `explorations/lib/explore.py` or `explorations/real3dqa/explore.py` loads a local/HF dataset from `explorations/data/`.
2. Data is rendered to `explorations/outputs/` via matplotlib (headless `Agg` backend).
3. `explorations/create_scene.py` separately instantiates a live MuJoCo env from a BDDL file and renders multi-camera frames — this is the direct analog of the notebook `02`/`03` steps, run locally instead of on Colab.

### Physical Hardware Flow

1. `diagnostics/servo_scan.py`, `servo_set_id.py`, `servo_set_torque_limit.py`, `servo_set_protection.py` bring up and calibrate individual servos.
2. `diagnostics/servo_move_test.py`, `servo_drive_to_stall.py`, `camera_test.py` validate motion range and camera feeds; results/checklists tracked in `diagnostics/UAT/`.
3. `control/joint_jog.py` / `keyboard_joint_control.py` provide manual teleoperation once servos are validated.
4. `control/record_episode.py` captures teleop episodes to `control/outputs/` for future use as physical-embodiment training data.

**State Management:**
- No shared runtime state across scripts — each script/notebook is stateless and reads/writes disk on each invocation.
- LIBERO lifelong training persists model checkpoints to disk; wandb tracks metrics remotely.
- `.planning/STATE.md` and `.planning/phases/` track GSD workflow state (not application runtime state).

## Key Abstractions

**OffScreenRenderEnv:**
- Purpose: MuJoCo simulation environment initialized from a BDDL task description, supports headless multi-camera rendering
- Location: `LIBERO/libero/libero/envs/env_wrapper.py`
- Pattern: Wraps robosuite environment; standard gym-like `reset()` / `step(action)` interface

**BDDL Task Definition:**
- Purpose: Declarative task specification (PDDL-like DSL) defining arena, fixtures, objects, initial state, and goal conditions
- Location: `LIBERO/libero/libero/bddl_files/libero_spatial/`, `libero_object/`, `libero_goal/`, `libero_100/`
- Pattern: Each `.bddl` file is one task; parsed at env construction

**Lifelong Algorithm Base:**
- Purpose: Abstract interface for continual learning algorithms
- Location: `LIBERO/libero/lifelong/algos/base.py`
- Pattern: Subclassed by `agem.py`, `er.py`, `ewc.py`, `multitask.py`, `packnet.py`, `single_task.py`

**GSD Skill/Command Pair:**
- Purpose: Each GSD workflow command (e.g. `/gsd-plan-phase`) has a matching skill definition surfaced to the agent
- Location: `.claude/commands/*.md` (invocation) + `.claude/skills/gsd-*/` and mirrored `.agents/`, `.codex/`, `.github/`, `.opencode/` trees (skill content)
- Pattern: Near-identical directory structure duplicated per supported AI CLI backend

## Entry Points

**Real-3DQA Explorer:**
- Location: `explorations/real3dqa/explore.py`
- Triggers: `python real3dqa/explore.py` (run from `explorations/`)
- Responsibilities: Visualize point cloud scenes and print QA annotation stats

**LIBERO Data Explorer:**
- Location: `explorations/lib/explore.py`
- Triggers: `python lib/explore.py` (run from `explorations/`)
- Responsibilities: Visualize HuggingFace LIBERO dataset episodes and task descriptions

**Scene Creator:**
- Location: `explorations/create_scene.py`
- Triggers: `python create_scene.py` (run from `explorations/`, requires `libero` conda env)
- Responsibilities: Render MuJoCo robot simulation frames from a BDDL task

**Lifelong Training Loop:**
- Location: `LIBERO/libero/lifelong/main.py`
- Triggers: `python -m lifelong.main` with Hydra config
- Responsibilities: Full training/evaluation loop for continual robot learning

**Colab Notebook Pipeline:**
- Location: `LIBERO/notebooks/01-colab-env-setup.ipynb` through `06b-eval.ipynb`
- Triggers: Run sequentially in a Colab GPU runtime
- Responsibilities: End-to-end language-to-action pipeline: setup → integration check → inference smoketest → finetune → eval

**Hardware Diagnostics Scripts:**
- Location: `diagnostics/*.py`
- Triggers: `python diagnostics/servo_scan.py` etc., run against physical arm with `diagnostics/.venv`
- Responsibilities: Servo/camera bring-up and validation

**Hardware Control Scripts:**
- Location: `control/*.py`
- Triggers: `python control/keyboard_joint_control.py` etc., run against physical arm with `control/.venv`
- Responsibilities: Teleoperation and episode recording

**GSD Slash Commands:**
- Location: `.claude/commands/*.md` (parsed by Claude Code CLI)
- Triggers: User invokes `/gsd-*` commands
- Responsibilities: Drive the phase-based planning/execution workflow described in `.planning/`

## Architectural Constraints

- **Python path:** `explorations/create_scene.py` manually inserts `explorations/LIBERO` (or the vendored `LIBERO/` path) into `sys.path` before importing `libero.*`. Scripts that import LIBERO must replicate this pattern.
- **MUJOCO_GL:** Must be set to `glfw` on macOS for headless rendering; `osmesa` is Linux-only (used in Colab).
- **LIBERO config:** First import of LIBERO reads `~/.libero/config.yaml` (user home directory). Must exist before running any LIBERO-dependent script or the package prompts interactively, causing `EOFError`.
- **Data location:** Exploration scripts resolve data paths relative to their own `__file__` location. `ROOT = Path(__file__).resolve().parent.parent` is the canonical pattern giving the `explorations/` directory root.
- **Isolated venvs:** `control/` and `diagnostics/` each maintain their own `.venv` and `requirements.txt`, separate from the LIBERO/exploration Python environment — the physical-hardware track and simulation track do not share a dependency environment.
- **Vendored subtree:** `LIBERO/` has its own `.git` history and `.pytest_cache`; treat it as an external dependency — avoid ad hoc edits outside of tracked project patches.
- **GSD backend duplication:** `.claude/`, `.agents/`, `.codex/`, `.github/`, `.opencode/` each contain a near-complete mirror of skills/commands/workflows. Changes to workflow behavior likely need to be replicated across all five (or applied only to the canonical source — verify which is authoritative before editing, e.g. via `.claude/scripts/changeset`).
- **Global state:** None across research/hardware scripts. LIBERO library uses module-level path resolution; GSD tooling uses `.planning/STATE.md` as its state file.
- **Circular imports:** None detected.

## Anti-Patterns

### Hardcoded LIBERO path string

**What happens:** Some exploration scripts hardcode a relative or absolute path to the vendored `LIBERO/` directory when inserting into `sys.path`.
**Why it's wrong:** Breaks when scripts are run from a different working directory or when `LIBERO/` is relocated.
**Do this instead:** Resolve the LIBERO path relative to `Path(__file__).resolve()` as done in `explorations/create_scene.py`, and prefer installing `LIBERO` as an editable package (`pip install -e LIBERO/`) where possible.

### Mixing hardware and simulation dependencies

**What happens:** Physical-hardware scripts (`control/`, `diagnostics/`) risk being run against the LIBERO/simulation Python environment or vice versa.
**Why it's wrong:** The hardware track depends on servo-bus/serial libraries that are irrelevant (and potentially conflicting) with MuJoCo/robosuite/torch pins used in simulation.
**Do this instead:** Always activate `control/.venv` or `diagnostics/.venv` for hardware scripts; keep the LIBERO/exploration Python environment (conda `libero` env per `explorations/create_scene.py` convention) separate.

## Error Handling

**Strategy:** Fail-fast with descriptive Python exceptions; no centralized error handling framework.

**Patterns:**
- `load_scene` raises `FileNotFoundError` with a descriptive message if a `.pth` file is absent (`explorations/real3dqa/`).
- `episode_strip` prints a message and returns early if an episode is not found in parquet files (`explorations/lib/explore.py`).
- No centralized error logging; errors surface as Python tracebacks.

## Cross-Cutting Concerns

**Logging:** Ad hoc `print()` statements with `→` arrow notation for output file paths; `wandb` used for training-run metrics in the lifelong learning layer.

**Validation:** Manual UAT checklists (`diagnostics/UAT/`) rather than automated validation for hardware bring-up; Hydra config validation for training runs.

**Authentication:** HuggingFace Hub tokens for dataset/model downloads (env var driven), no other auth surfaces in this codebase.

---

*Architecture analysis: 2026-09-05*
