# Phase 1: Colab Environment Setup - Pattern Map

**Mapped:** 2026-07-08
**Files analyzed:** 1 (single notebook deliverable with ~18 cells)
**Analogs found:** 4 / 4 analog targets searched

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `LIBERO/notebooks/01-colab-env-setup.ipynb` — Cell block A (install cells) | utility | batch | `LIBERO/requirements.txt` + research install order | role-match |
| `LIBERO/notebooks/01-colab-env-setup.ipynb` — EGL/MUJOCO_GL bootstrap cell | config | request-response | `explorations/create_scene.py` lines 9–19 | exact (GL backend + env var pattern) |
| `LIBERO/notebooks/01-colab-env-setup.ipynb` — LIBERO config.yaml bootstrap cell | config | request-response | `LIBERO/libero/libero/__init__.py` lines 62–96 | exact (config structure, yaml keys) |
| `LIBERO/notebooks/01-colab-env-setup.ipynb` — ENV-02 render verification cell | utility | request-response | `explorations/create_scene.py` lines 35–93 | exact (OffScreenRenderEnv, camera, flip, save) |
| `LIBERO/notebooks/01-colab-env-setup.ipynb` — notebook structure/cell patterns | config | — | `LIBERO/notebooks/quick_walkthrough.ipynb` | role-match |

---

## Pattern Assignments

### Cell Block A — pip/apt Install Cells

**Analog:** `LIBERO/requirements.txt` (full file, 15 lines)

**Pinned versions to satisfy** (all lines):
```
hydra-core==1.2.0
numpy==1.22.4
wandb==0.13.1
easydict==1.9
transformers==4.21.1     ← overridden to fork in Phase 1
opencv-python==4.6.0.66
robomimic==0.2.0
einops==0.4.1
thop==0.1.1-2209072238
robosuite==1.4.0
bddl==1.0.1
future==0.18.2
matplotlib==3.5.3
cloudpickle==2.1.0
gym==0.25.2
```

**Critical install ordering constraint (from RESEARCH.md):**
1. apt-get EGL system packages (libglfw3, libglew-dev, libosmesa6-dev, libgles2, libglvnd0, libegl-dev, libegl1, libgl1-mesa-glx)
2. torch==2.2.0 + torchvision==0.17.0 (via `--index-url https://download.pytorch.org/whl/cu121`)
3. mujoco==2.3.7 + gym==0.25.2
4. robosuite==1.4.0 + bddl==1.0.1 + LIBERO supporting stack
5. pip install -e LIBERO (editable)
6. timm==0.9.10, tokenizers==0.19.1, sentencepiece==0.1.99, peft==0.11.1, accelerate
7. `git+https://github.com/moojink/transformers-openvla-oft.git` (replaces PyPI transformers — install LAST)
8. flash-attn==2.5.5 --no-build-isolation (must be last; compiles C++ against installed torch)

**Note:** Do NOT install `transformers` from PyPI. The git fork replaces it entirely.

---

### Cell: MUJOCO_GL + EGL Bootstrap (first post-restart cell)

**Analog:** `explorations/create_scene.py` lines 9–19

**Exact pattern to replicate** (lines 9–19):
```python
# explorations/create_scene.py lines 9-19
import sys
import os
import numpy as np

# Point Python at the LIBERO package
LIBERO_PATH = os.path.join(os.path.dirname(__file__), "LIBERO")
sys.path.insert(0, LIBERO_PATH)

os.environ["MUJOCO_GL"] = "glfw"               # macOS headless via GLFW
```

**Colab adaptation (change glfw → egl, add NVIDIA ICD creation BEFORE env var):**
```python
# Colab version: NVIDIA ICD must exist before MUJOCO_GL is set
import os, json

os.makedirs("/usr/share/glvnd/egl_vendor.d", exist_ok=True)
with open("/usr/share/glvnd/egl_vendor.d/10_nvidia.json", "w") as f:
    json.dump({
        "file_format_version": "1.0.0",
        "ICD": {"library_path": "libEGL_nvidia.so.0"}
    }, f)

os.environ["MUJOCO_GL"] = "egl"               # Linux/Colab headless via EGL
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
```

**Ordering constraint:** This must be the first executable cell after restart. No `import mujoco`, `import robosuite`, or `import libero` may appear before this cell.

---

### Cell: LIBERO config.yaml Bootstrap

**Analog:** `LIBERO/libero/libero/__init__.py` lines 62–96

**The `input()` call that causes EOFError** (lines 62–70):
```python
# LIBERO/libero/libero/__init__.py lines 62-70
if not os.path.exists(libero_config_path):
    os.makedirs(libero_config_path)

if not os.path.exists(config_file):
    default_path_dict = get_default_path_dict()
    answer = input(
        "Do you want to specify a custom path for the dataset folder? (Y/N): "
    ).lower()
```

**Config dict keys to pre-populate** (from `get_default_path_dict()` lines 11–35):
```python
{
    "benchmark_root": benchmark_root_path,
    "bddl_files":    os.path.join(benchmark_root_path, "./bddl_files"),
    "init_states":   os.path.join(benchmark_root_path, "./init_files"),
    "datasets":      os.path.join(benchmark_root_path, "../datasets"),
    "assets":        os.path.join(benchmark_root_path, "./assets"),
}
```

**Notebook cell must pre-create this file before any `import libero`:**
```python
import os, yaml
from pathlib import Path

LIBERO_ROOT = "/content/drive/MyDrive/SoARM-Research/LIBERO/libero/libero"
config_dir = Path.home() / ".libero"
config_dir.mkdir(parents=True, exist_ok=True)
config = {
    "benchmark_root": LIBERO_ROOT,
    "bddl_files":    f"{LIBERO_ROOT}/bddl_files",
    "init_states":   f"{LIBERO_ROOT}/init_files",
    "datasets":      f"{LIBERO_ROOT}/../datasets",
    "assets":        f"{LIBERO_ROOT}/assets",
}
(config_dir / "config.yaml").write_text(yaml.dump(config))
print(f"Saved → {config_dir / 'config.yaml'}")
```

---

### Cell: ENV-02 Render Verification

**Analog:** `explorations/create_scene.py` — entire file (98 lines)

**sys.path insert pattern** (lines 13–14):
```python
LIBERO_PATH = os.path.join(os.path.dirname(__file__), "LIBERO")
sys.path.insert(0, LIBERO_PATH)
```

**Notebook equivalent** (no `__file__` in Colab):
```python
LIBERO_PKG = "/content/drive/MyDrive/SoARM-Research/LIBERO/libero"
if LIBERO_PKG not in sys.path:
    sys.path.insert(0, LIBERO_PKG)
```

**OffScreenRenderEnv instantiation pattern** (lines 36–43):
```python
env = OffScreenRenderEnv(
    bddl_file_name=BDDL,
    camera_names=["agentview", "frontview", "robot0_eye_in_hand"],
    camera_heights=camera_h,
    camera_widths=camera_w,
    has_renderer=False,
    has_offscreen_renderer=True,
)
```

**Reset + step + image flip pattern** (lines 52–68):
```python
obs = env.reset()
for _ in range(10):
    obs, _, _, _ = env.step(np.zeros(7))

img = obs["agentview_image"][::-1]   # MuJoCo images are upside-down — always flip
```

**Save + print pattern** (lines 81–82, project convention):
```python
out = os.path.join(OUT_DIR, "libero_scene_robot.png")
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved → {out}")
```

**matplotlib.use("Agg") ordering** (lines 18–20, project convention — must precede pyplot import):
```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
```

**BDDL file used in existing analog** (line 26–29) — reuse same task for ENV-02:
```
libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate.bddl
```

**Env close** (line 91):
```python
env.close()
```

---

### Cell: Notebook Structure Pattern

**Analog:** `LIBERO/notebooks/quick_walkthrough.ipynb`

**Existing notebook imports LIBERO path via `get_libero_path`** (cell-1):
```python
from libero.libero import benchmark, get_libero_path, set_libero_default_path
```

**Existing notebook OffScreenRenderEnv usage** (cell-15):
```python
from libero.libero.envs import OffScreenRenderEnv
from IPython.display import display
from PIL import Image

env_args = {
    "bddl_file_name": os.path.join(bddl_files_default_path, task.problem_folder, task.bddl_file),
    "camera_heights": 128,
    "camera_widths": 128
}
env = OffScreenRenderEnv(**env_args)
```

**Image display pattern** (cell-15):
```python
display(Image.fromarray(grid_image.numpy()[::-1]))
```

**Note:** The existing notebooks do NOT set MUJOCO_GL before import (they assume a local display). Phase 1 notebook must add the EGL bootstrap cell before any LIBERO import — this is the key structural difference.

---

## Shared Patterns

### UPPER_SNAKE_CASE path constants (all cells)
**Source:** `explorations/create_scene.py` lines 13, 25, 31
**Apply to:** All cells with file path references
```python
LIBERO_PATH = ...
BDDL = ...
OUT_DIR = ...
```
Colab notebook equivalents: `REPO_ROOT`, `LIBERO_ROOT`, `LIBERO_PKG`, `BDDL_FILE`, `OUT_DIR`

### Saved path print convention (all output-writing cells)
**Source:** `explorations/create_scene.py` lines 82, 89
**Apply to:** Any cell that saves a file
```python
print(f"Saved → {out}")
print(f"  Individual frame → outputs/libero_scene_{name}.png")
```

### matplotlib.use("Agg") ordering (all visualization cells)
**Source:** `explorations/create_scene.py` lines 18–20
**Apply to:** ENV-02 render cell and any cell doing plt operations
```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
```

### PASS/FAIL verification output format
**Source:** CONTEXT.md D-07 decision — no existing codebase analog
**Apply to:** ENV-01, ENV-02, ENV-03 verification cells
```python
status = "PASS" if <condition> else "FAIL"
print(f"ENV-0X: {status} — <detail>")
```

---

## No Analog Found

| File/Cell | Role | Data Flow | Reason |
|-----------|------|-----------|--------|
| GPU assertion cell (D-06) | utility | request-response | No GPU assertion exists in codebase; pattern from RESEARCH.md Code Examples |
| ENV-01 version printout cell | utility | batch | No `importlib.metadata.version()` pattern exists in codebase; standard library |
| ENV-03 VLA load + inference cell | utility | request-response | No VLA/HuggingFace model loading in codebase; pattern from RESEARCH.md Code Examples |
| pip/apt install cells | config | batch | No install scripts in codebase; order from RESEARCH.md Standard Stack |
| Runtime restart markdown cell | — | — | Notebook-specific; no analog needed |

---

## Key Ordering Constraints (Critical for Planner)

These ordering dependencies must be reflected in the plan's action sequence:

1. **apt-get before pip** — EGL system libs must exist before mujoco Python package installs
2. **torch before flash-attn** — flash-attn compiles against installed torch
3. **transformers fork last** — must not be overridden by earlier pip resolver
4. **Runtime restart between Block A and Block C** — mandatory; pip installs do not take effect in current kernel
5. **NVIDIA ICD JSON creation before `os.environ["MUJOCO_GL"]`** — ICD must exist before env var is read by MuJoCo C extension
6. **`os.environ["MUJOCO_GL"] = "egl"` before any `import mujoco`/`import robosuite`/`import libero`** — C extension reads env var at load time only
7. **`~/.libero/config.yaml` creation before `import libero`** — avoids `input()` → EOFError
8. **sys.path insert before `from libero.libero.envs import OffScreenRenderEnv`** — module must be findable

---

## Metadata

**Analog search scope:** `explorations/`, `LIBERO/libero/libero/`, `LIBERO/notebooks/`, `LIBERO/requirements.txt`
**Files scanned:** 4 (create_scene.py, LIBERO/__init__.py, requirements.txt, quick_walkthrough.ipynb)
**Pattern extraction date:** 2026-07-08
