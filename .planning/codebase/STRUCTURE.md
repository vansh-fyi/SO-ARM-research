# Codebase Structure

**Analysis Date:** 2026-07-07

## Directory Layout

```
SoARM-Research/                     # Research repository root
├── explorations/                   # Research exploration scripts and outputs
│   ├── real3dqa/                   # Real-3DQA dataset exploration module
│   │   └── explore.py              # CLI: visualize Real-3DQA point clouds & annotations
│   ├── lib/                        # LIBERO HuggingFace dataset exploration module
│   │   └── explore.py              # CLI: visualize LIBERO parquet episode data
│   ├── Real-3DQA/                  # Upstream Real-3DQA reference data/assets
│   │   ├── point_clouds/           # .pth point cloud files (ScanNet format)
│   │   ├── data/                   # Annotation files
│   │   └── assets/                 # Reference assets
│   ├── dreamzero/                  # DreamZero dataset exploration
│   │   └── manipulation/           # Manipulation-specific data
│   ├── data/                       # Gitignored local datasets
│   │   ├── real3dqa/               # Downloaded Real-3DQA data
│   │   │   ├── point_clouds/       # *.pth scene point clouds
│   │   │   └── annotations/        # *.jsonl QA pair annotations
│   │   └── libero/                 # Downloaded LIBERO HuggingFace dataset
│   │       ├── meta/               # info.json, tasks.parquet
│   │       └── data/chunk-000/     # *.parquet episode frames
│   ├── outputs/                    # Generated PNG visualizations
│   ├── perspective_images/         # Rendered perspective views
│   ├── create_scene.py             # MuJoCo LIBERO environment renderer
│   ├── render_scenes.py            # Software point cloud rasterizer
│   ├── download_real3dqa.py        # Real-3DQA dataset download script
│   ├── requirements.txt            # Python dependencies for explorations
│   └── LIBERO_SCENE.md             # Setup guide for MuJoCo scene creation
│
├── LIBERO/                         # Embedded LIBERO benchmark repo (own .git)
│   ├── libero/                     # Python package root
│   │   ├── libero/                 # Core simulation library
│   │   │   ├── envs/               # MuJoCo environments
│   │   │   │   ├── env_wrapper.py  # OffScreenRenderEnv entry point
│   │   │   │   ├── problems/       # Per-arena problem classes
│   │   │   │   ├── robots/         # Panda robot definitions
│   │   │   │   ├── arenas/         # Table/kitchen/living room arenas
│   │   │   │   └── objects/        # Manipulable 3D object classes
│   │   │   ├── benchmark/          # Task suite definitions & maps
│   │   │   ├── bddl_files/         # PDDL-like task specification files
│   │   │   │   ├── libero_spatial/ # 10 spatial reasoning tasks
│   │   │   │   ├── libero_object/  # 10 object knowledge tasks
│   │   │   │   ├── libero_goal/    # 10 goal-conditioned tasks
│   │   │   │   └── libero_100/     # 100 complex tasks
│   │   │   ├── assets/             # MuJoCo XML meshes, textures, scene XMLs
│   │   │   ├── utils/              # Dataset, video, log, time, object utilities
│   │   │   └── init_files/         # Task initial state files
│   │   ├── lifelong/               # Lifelong/continual learning training
│   │   │   ├── main.py             # Hydra training entry point
│   │   │   ├── algos/              # Algorithm implementations
│   │   │   │   ├── base.py         # Abstract base algorithm
│   │   │   │   ├── agem.py         # A-GEM continual learning
│   │   │   │   ├── er.py           # Experience Replay
│   │   │   │   ├── ewc.py          # Elastic Weight Consolidation
│   │   │   │   ├── multitask.py    # Multitask baseline
│   │   │   │   ├── packnet.py      # PackNet compression
│   │   │   │   └── single_task.py  # Single-task baseline
│   │   │   ├── models/             # Policy model architectures
│   │   │   ├── datasets.py         # GroupedTaskDataset, SequenceVLDataset
│   │   │   ├── evaluate.py         # Evaluation loop
│   │   │   ├── metric.py           # Success rate / loss metrics
│   │   │   └── utils.py            # Seed control, checkpoint, experiment dir
│   │   └── configs/                # Hydra YAML configuration files
│   ├── benchmark_scripts/          # Standalone CLI benchmark utilities
│   │   ├── download_libero_datasets.py
│   │   ├── check_task_suites.py
│   │   ├── render_single_task.py
│   │   └── shasum_files.py
│   ├── scripts/                    # Dataset creation and demonstration collection
│   │   ├── collect_demonstration.py
│   │   ├── create_dataset.py
│   │   └── libero_100_collect_demonstrations.py
│   ├── templates/                  # Code templates for new task definitions
│   ├── notebooks/                  # Jupyter notebooks
│   ├── setup.py                    # LIBERO package installer
│   └── requirements.txt            # LIBERO Python dependencies
│
├── AGENTS.md                       # GSD agent configuration
├── .gitignore                      # Ignores explorations/, .agents/, .claude/, etc.
├── .planning/
│   └── codebase/                   # GSD codebase analysis documents
├── .claude/                        # Claude Code agent configuration
├── .agents/                        # Multi-agent framework config
├── .codex/                         # Codex agent config
├── .opencode/                      # OpenCode agent config
└── .github/                        # GitHub agent config
```

## Directory Purposes

**`explorations/`:**
- Purpose: All active research exploration code for this project
- Contains: Self-contained CLI Python scripts, one per investigation area
- Key files: `create_scene.py`, `real3dqa/explore.py`, `lib/explore.py`, `render_scenes.py`

**`explorations/data/`:**
- Purpose: Local dataset storage (not committed)
- Contains: Downloaded Real-3DQA point clouds/annotations and LIBERO HuggingFace parquet data
- Generated: Via `download_real3dqa.py` and HuggingFace `datasets` library
- Committed: No (in `.gitignore`)

**`explorations/outputs/`:**
- Purpose: Generated visualization images from exploration scripts
- Contains: PNG files rendered by matplotlib/PIL
- Generated: By all exploration scripts
- Committed: No (in `.gitignore`)

**`LIBERO/`:**
- Purpose: Vendored LIBERO benchmark framework for robot manipulation research
- Contains: Complete simulation library, lifelong learning training code, 130 BDDL task files
- Key files: `libero/libero/envs/env_wrapper.py`, `libero/lifelong/main.py`
- Note: Has its own `.git` history — treat as a submodule

**`LIBERO/libero/libero/bddl_files/`:**
- Purpose: Task library — each `.bddl` file defines one robot manipulation task
- Contains: 130 tasks across four suites (spatial, object, goal, libero_100)
- Key files: Any `.bddl` file can be passed to `OffScreenRenderEnv(bddl_file_name=...)`

## Key File Locations

**Entry Points:**
- `explorations/real3dqa/explore.py`: Real-3DQA visualization CLI
- `explorations/lib/explore.py`: LIBERO parquet dataset exploration CLI
- `explorations/create_scene.py`: MuJoCo scene rendering entry point
- `explorations/render_scenes.py`: Software-rasterized point cloud renderer
- `LIBERO/libero/lifelong/main.py`: Hydra-based lifelong learning training entry point

**Configuration:**
- `LIBERO/libero/configs/`: Hydra YAML configs for lifelong training
- `~/.libero/config.yaml`: User-local LIBERO path config (must exist before first import)
- `explorations/requirements.txt`: Python deps for exploration scripts

**Core Simulation:**
- `LIBERO/libero/libero/envs/env_wrapper.py`: `OffScreenRenderEnv` class
- `LIBERO/libero/libero/bddl_files/`: All 130 task definitions

**Algorithm Implementations:**
- `LIBERO/libero/lifelong/algos/base.py`: Base algorithm class
- `LIBERO/libero/lifelong/algos/ewc.py`, `agem.py`, `er.py`, `packnet.py`: Continual learning algorithms

**Utilities:**
- `LIBERO/libero/libero/utils/`: Dataset, video, logging, object, download utilities
- `LIBERO/libero/lifelong/utils.py`: Training utilities (seed, checkpoint, experiment dir)

**Data:**
- `explorations/data/real3dqa/point_clouds/*.pth`: ScanNet point clouds (xyz, rgb, labels)
- `explorations/data/real3dqa/annotations/*.jsonl`: QA pairs with position/rotation metadata
- `explorations/data/libero/meta/info.json`: Dataset metadata (40 tasks, 1693 episodes)
- `explorations/data/libero/data/chunk-000/*.parquet`: Episode frames with embedded PNG bytes

## Naming Conventions

**Files:**
- Exploration scripts: `snake_case.py` (e.g., `explore.py`, `create_scene.py`, `render_scenes.py`)
- LIBERO algorithm files: `snake_case.py` named after the algorithm (e.g., `ewc.py`, `packnet.py`)
- BDDL task files: `full_english_task_description_with_underscores.bddl`

**Directories:**
- Exploration modules: lowercase single-word or abbreviated names (`real3dqa/`, `lib/`, `dreamzero/`)
- LIBERO internal: lowercase with underscores (`bddl_files/`, `init_files/`)
- Data suites: `libero_spatial/`, `libero_object/`, `libero_goal/`, `libero_100/`

**Variables (in exploration scripts):**
- Path constants: `ROOT`, `DATA`, `OUT` — uppercase module-level constants
- Function names: `snake_case` (e.g., `load_scene`, `bird_eye_grid`, `collect_first_frames`)

## Where to Add New Code

**New Dataset Explorer:**
- Create a new subdirectory under `explorations/` (e.g., `explorations/newdataset/`)
- Add `explore.py` with `ROOT = Path(__file__).resolve().parent.parent` for path resolution
- Output images to `OUT = ROOT / "outputs"`
- Data expected at `DATA = ROOT / "data" / "newdataset"`

**New LIBERO Task:**
- Add a `.bddl` file to the appropriate suite under `LIBERO/libero/libero/bddl_files/`
- Reference it via absolute path in `OffScreenRenderEnv(bddl_file_name=...)`

**New Continual Learning Algorithm:**
- Add `LIBERO/libero/lifelong/algos/<algo_name>.py` subclassing `base.py`
- Register in `LIBERO/libero/lifelong/algos/__init__.py`

**Utilities shared across exploration scripts:**
- No shared utility module exists yet. If needed, create `explorations/utils.py` and import via relative path.

**New rendering/visualization script:**
- Place directly in `explorations/` if it is a one-off top-level investigation
- Place in `explorations/<topic>/` if it belongs to a specific dataset domain

## Special Directories

**`explorations/data/`:**
- Purpose: Downloaded datasets (Real-3DQA .pth files, LIBERO parquet files)
- Generated: Yes — via download scripts and HuggingFace
- Committed: No

**`explorations/outputs/`:**
- Purpose: PNG visualization outputs from all exploration scripts
- Generated: Yes — by matplotlib `savefig` calls
- Committed: No

**`LIBERO/libero/libero/assets/`:**
- Purpose: MuJoCo XML scene files, mesh STLs, texture images for simulation
- Generated: No (part of LIBERO repo)
- Committed: Yes (within LIBERO subtree)

**`.planning/codebase/`:**
- Purpose: GSD codebase analysis documents for AI-assisted development
- Generated: Yes — by GSD map-codebase command
- Committed: Yes

---

*Structure analysis: 2026-07-07*
