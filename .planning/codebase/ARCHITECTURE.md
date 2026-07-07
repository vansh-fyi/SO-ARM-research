<!-- refreshed: 2026-07-07 -->
# Architecture

**Analysis Date:** 2026-07-07

## System Overview

```text
┌──────────────────────────────────────────────────────────────┐
│                     SoARM Research Repo                      │
├──────────────────────────┬───────────────────────────────────┤
│   explorations/          │   LIBERO/  (embedded submodule)   │
│   Research Scripts       │   Benchmark Framework             │
│   `explorations/`        │   `LIBERO/`                       │
└────────────┬─────────────┴───────────────┬───────────────────┘
             │                             │
             ▼                             ▼
┌────────────────────────┐   ┌─────────────────────────────────┐
│  Dataset Explorers     │   │  libero/libero/  (core library) │
│  `explorations/real3dqa│   │  envs/, benchmark/, utils/      │
│   /explore.py`         │   │  bddl_files/, assets/           │
│  `explorations/lib/    │   └────────────────┬────────────────┘
│   explore.py`          │                    │
└────────────┬───────────┘                    ▼
             │                ┌───────────────────────────────┐
             ▼                │  libero/lifelong/ (training)  │
┌────────────────────────┐    │  algos/, models/, datasets.py │
│  explorations/data/    │    │  main.py, evaluate.py         │
│  (gitignored datasets) │    └───────────────────────────────┘
│  data/real3dqa/        │
│  data/libero/          │
└────────────────────────┘
             │
             ▼
┌────────────────────────┐
│  explorations/outputs/ │
│  Rendered images       │
└────────────────────────┘
```

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| Real-3DQA Explorer | Load .pth point clouds and .jsonl QA annotations, render bird's-eye and 3D visualizations | `explorations/real3dqa/explore.py` |
| LIBERO Data Explorer | Load HuggingFace parquet episode frames from manipulation dataset, render episode grids | `explorations/lib/explore.py` |
| Scene Creator | Instantiate MuJoCo LIBERO environment from a BDDL file, render multi-camera frames | `explorations/create_scene.py` |
| Point Cloud Renderer | Software-rasterize ScanNet .pth point clouds to PNG without GPU | `explorations/render_scenes.py` |
| Dataset Downloader | Download Real-3DQA dataset from remote source | `explorations/download_real3dqa.py` |
| LIBERO Core Library | Simulation environments (MuJoCo), BDDL-driven task definitions, robot/arena/object classes | `LIBERO/libero/libero/` |
| LIBERO Lifelong Learning | Continual/lifelong learning training loop, algorithms (EWC, AGEM, PackNet, ER), evaluation | `LIBERO/libero/lifelong/` |
| LIBERO Benchmark Scripts | Dataset download, task suite checks, single-task rendering, dataset integrity checks | `LIBERO/benchmark_scripts/` |

## Pattern Overview

**Overall:** Research exploration monorepo — flat script-per-investigation pattern with a vendored/embedded benchmark library.

**Key Characteristics:**
- Exploration scripts in `explorations/` are self-contained, CLI-driven Python programs.
- All data (datasets, point clouds) lives under `explorations/data/` and is gitignored.
- Outputs (rendered PNG images) go to `explorations/outputs/`.
- The `LIBERO/` subtree is an embedded benchmark repo with its own `.git` history; treated as a vendored dependency.
- No shared library or common module exists across exploration scripts — each script resolves paths via `Path(__file__).resolve().parent`.

## Layers

**Exploration Scripts Layer:**
- Purpose: One-off and iterative research investigation scripts
- Location: `explorations/`
- Contains: Dataset loading, visualization, scene rendering, download utilities
- Depends on: `explorations/data/` for input, numpy/matplotlib/torch/pandas/PIL for processing
- Used by: Researchers running CLI tools directly

**LIBERO Environment Layer:**
- Purpose: MuJoCo-based robot simulation environments, task definitions via BDDL
- Location: `LIBERO/libero/libero/envs/`
- Contains: `OffScreenRenderEnv`, arena definitions, robot configs, object models, BDDL parser
- Depends on: MuJoCo 2.3.7, robosuite 1.4.1, BDDL assets in `LIBERO/libero/libero/bddl_files/`
- Used by: `explorations/create_scene.py`, lifelong learning training

**LIBERO Lifelong Learning Layer:**
- Purpose: Training and evaluation of continual learning policies on the LIBERO benchmark
- Location: `LIBERO/libero/lifelong/`
- Contains: `main.py` (Hydra entrypoint), algorithm implementations (EWC, AGEM, ER, PackNet, multitask, single-task), dataset loading, metric computation
- Depends on: Hydra config system (`LIBERO/libero/configs/`), wandb, torch, LIBERO benchmark core
- Used by: Research training runs

**Data Layer:**
- Purpose: Raw datasets consumed by exploration scripts
- Location: `explorations/data/real3dqa/`, `explorations/data/libero/`
- Contains:
  - Real-3DQA: `point_clouds/*.pth` (ScanNet-format point clouds), `annotations/*.jsonl` (QA pairs)
  - LIBERO: `meta/info.json`, `meta/tasks.parquet`, `data/chunk-000/*.parquet` (episode frames with 256×256 RGB images)
- Generated: Downloaded via `download_real3dqa.py` and HuggingFace
- Committed: No (gitignored)

## Data Flow

### Real-3DQA Visualization Flow

1. User runs `python real3dqa/explore.py [--scene SCENE_ID]` from `explorations/` directory
2. `load_scene(scene_id)` reads `data/real3dqa/point_clouds/<scene_id>.pth` via `torch.load` (`explorations/real3dqa/explore.py:30`)
3. Point cloud (xyz, rgb arrays) is subsampled and rendered with matplotlib scatter
4. `load_annotations()` reads all `data/real3dqa/annotations/*.jsonl` files (`explorations/real3dqa/explore.py:41`)
5. Output PNG saved to `explorations/outputs/`

### LIBERO Parquet Dataset Exploration Flow

1. User runs `python lib/explore.py [--episodes N | --episode ID | --tasks]` from `explorations/`
2. `iter_parquet_files()` enumerates `data/libero/data/chunk-000/*.parquet` (`explorations/lib/explore.py:51`)
3. Each parquet row contains embedded PNG bytes for front and wrist camera images, plus robot state and action vectors
4. `collect_first_frames()` scans files for unique episode_index values (`explorations/lib/explore.py:56`)
5. Images decoded via `PIL.Image.open(io.BytesIO(...))` and displayed in matplotlib grid
6. Output PNG saved to `explorations/outputs/`

### MuJoCo Scene Rendering Flow

1. User runs `python create_scene.py` from `explorations/`
2. `LIBERO/` path appended to `sys.path`; `MUJOCO_GL=glfw` set for macOS headless rendering (`explorations/create_scene.py:13-16`)
3. `OffScreenRenderEnv` instantiated from a `.bddl` task file (`LIBERO/libero/libero/bddl_files/`)
4. `env.reset()` triggers MuJoCo scene initialization; 10 zero-action physics steps settle the scene
5. Observation dict keys `agentview_image`, `frontview_image`, `robot0_eye_in_hand_image` retrieved (images are vertically flipped — use `[::-1]`)
6. 3-panel composite PNG + individual camera PNGs saved to `explorations/outputs/`

### LIBERO Lifelong Training Flow

1. `python main.py` with Hydra config at `LIBERO/libero/configs/` (`LIBERO/libero/lifelong/main.py:37`)
2. Hydra loads YAML config → converted to `EasyDict`
3. `get_benchmark()` returns benchmark suite; `get_dataset()` loads demonstration data
4. `get_algo_class()` instantiates one of: AGEM, ER, EWC, PackNet, Multitask, SingleTask
5. Training loop iterates over tasks; `evaluate_loss` / `evaluate_success` compute metrics
6. Results logged to wandb

**State Management:**
- No global shared state across exploration scripts. Each script is stateless and reads from disk on each invocation.
- LIBERO lifelong training maintains model checkpoint state on disk; wandb for metrics.

## Key Abstractions

**OffScreenRenderEnv:**
- Purpose: MuJoCo simulation environment initialized from a BDDL task description, supports headless multi-camera rendering
- Location: `LIBERO/libero/libero/envs/env_wrapper.py`
- Pattern: Wraps robosuite environment; standard gym-like `reset()` / `step(action)` interface

**BDDL Task Files:**
- Purpose: Declarative task specification (PDDL-like DSL) defining arena, fixtures, objects, initial state, and goal conditions
- Location: `LIBERO/libero/libero/bddl_files/libero_spatial/`, `libero_object/`, `libero_goal/`, `libero_100/`
- Pattern: Each `.bddl` file is one task; parsed at env construction

**Lifelong Algorithm Base:**
- Purpose: Abstract interface for continual learning algorithms
- Location: `LIBERO/libero/lifelong/algos/base.py`
- Pattern: Subclassed by `agem.py`, `er.py`, `ewc.py`, `multitask.py`, `packnet.py`, `single_task.py`

## Entry Points

**Real-3DQA Explorer:**
- Location: `explorations/real3dqa/explore.py`
- Triggers: `python real3dqa/explore.py` (run from `explorations/`)
- Responsibilities: Visualize point cloud scenes and print QA annotation stats

**LIBERO Parquet Explorer:**
- Location: `explorations/lib/explore.py`
- Triggers: `python lib/explore.py` (run from `explorations/`)
- Responsibilities: Visualize HuggingFace LIBERO dataset episodes and task descriptions

**Scene Creator:**
- Location: `explorations/create_scene.py`
- Triggers: `python create_scene.py` (run from `explorations/`, requires `libero` conda env)
- Responsibilities: Render MuJoCo robot simulation frames from a BDDL task

**Lifelong Training:**
- Location: `LIBERO/libero/lifelong/main.py`
- Triggers: `python main.py` with Hydra config
- Responsibilities: Full training/evaluation loop for continual robot learning

## Architectural Constraints

- **Python path:** `explorations/create_scene.py` manually inserts `explorations/LIBERO` into `sys.path` before importing `libero.*`. Scripts that import LIBERO must replicate this pattern.
- **MUJOCO_GL:** Must be set to `glfw` on macOS for headless rendering; `osmesa` is Linux-only.
- **LIBERO config:** First import of LIBERO reads `~/.libero/config.yaml` (user home directory). Must exist before running any LIBERO-dependent script or the package prompts interactively causing `EOFError`.
- **Data location:** All exploration scripts resolve data paths relative to their own `__file__` location. `ROOT = Path(__file__).resolve().parent.parent` is the canonical pattern giving the `explorations/` directory root.
- **Global state:** None across scripts. LIBERO library uses module-level path resolution.
- **Circular imports:** None detected.

## Anti-Patterns

### Hardcoded LIBERO path string

**What happens:** `explorations/create_scene.py` uses `os.path.join(os.path.dirname(__file__), "LIBERO")` as the LIBERO root, assuming LIBERO is a sibling directory of `create_scene.py`.
**Why it's wrong:** The LIBERO package actually lives at `explorations/../LIBERO` (repo root), not inside `explorations/`. This path currently works because `sys.path` insertion uses the string but the actual import resolves via the package name — fragile if files are moved.
**Do this instead:** Use `ROOT.parent / "LIBERO"` where `ROOT = Path(__file__).resolve().parent` to make the path unambiguous.

## Error Handling

**Strategy:** Scripts fail fast with file-not-found exceptions or print messages. Visualization scripts use `try/except FileNotFoundError` around per-scene loading to skip missing point clouds gracefully (`explorations/real3dqa/explore.py:63`).

**Patterns:**
- `load_scene` raises `FileNotFoundError` with a descriptive message if `.pth` file is absent
- `episode_strip` prints a message and returns early if episode not found in parquet files
- No centralized error logging; errors surface as Python tracebacks

## Cross-Cutting Concerns

**Logging:** `print()` statements only. No logging framework.
**Validation:** None — inputs assumed valid (scene IDs, episode IDs from CLI args).
**Authentication:** Not applicable. All data is local or downloaded from public sources.

---

*Architecture analysis: 2026-07-07*
