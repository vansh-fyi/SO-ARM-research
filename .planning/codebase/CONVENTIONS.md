# Coding Conventions

**Analysis Date:** 2026-07-07

## Naming Patterns

**Files:**
- `snake_case` for all Python files: `explore.py`, `render_scenes.py`, `download_real3dqa.py`, `create_scene.py`
- Descriptive names reflecting the action: `download_*`, `render_*`, `create_*`, `explore*`

**Functions:**
- `snake_case` throughout: `load_scene()`, `bird_eye_grid()`, `render_point_cloud()`, `collect_first_frames()`
- Verb-noun naming pattern: `load_*`, `render_*`, `collect_*`, `build_*`, `print_*`

**Variables:**
- `snake_case` for all local and module-level variables
- Short abbreviations for commonly used data: `xyz`, `rgb`, `pts_cam`, `idx`, `sid`, `ep`
- Module-level path constants in `UPPER_SNAKE_CASE`: `ROOT`, `DATA`, `OUT`, `OUT_DIR`, `BDDL`, `LIBERO_PATH`

**Constants:**
- `UPPER_SNAKE_CASE` for module-level constants: `ROOT = Path(...)`, `DATA = ROOT / "data"`, `DATASET_REPO = "..."`

## Code Style

**Formatting:**
- No dedicated linter or formatter config detected (no `.flake8`, `.pylintrc`, `pyproject.toml`, `setup.cfg` in project root)
- Consistent 4-space indentation throughout
- Single blank lines between functions; no extra blank lines within short functions
- Inline comments used sparingly for non-obvious math/geometry logic (see `render_scenes.py`)

**Linting:**
- No enforced linting configuration detected

**Line Length:**
- No explicit limit enforced; most lines stay under 100 characters

## Import Organization

**Order (observed pattern):**
1. Standard library imports (`os`, `sys`, `io`, `json`, `argparse`, `pathlib`, `glob`)
2. Third-party numeric/scientific imports (`numpy`, `pandas`, `torch`, `matplotlib`, `PIL`)
3. Local/project imports (`from libero.libero.envs import ...`)

**Style:**
- Standard library imports appear first, then third-party
- `matplotlib.use("Agg")` is called immediately after importing `matplotlib`, before importing `matplotlib.pyplot` — this is a consistent pattern in all visualization scripts
- `pathlib.Path` preferred over `os.path` in newer scripts (`explore.py`); older scripts (`render_scenes.py`, `download_real3dqa.py`) use `os.path` and `os`

**Path Aliases:**
- None detected

## Module-Level Path Setup

Consistent pattern across exploration scripts: define path roots at module level using `Path(__file__).resolve().parent.parent` or `os.path.dirname(os.path.abspath(__file__))`:

```python
# pathlib style (explore.py)
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "libero"
OUT  = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

# os.path style (download_real3dqa.py)
LOCAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Real-3DQA")
```

## Error Handling

**Patterns:**
- Explicit `FileNotFoundError` raised on missing data files:
  ```python
  if not path.exists():
      raise FileNotFoundError(f"Point cloud not found: {path}")
  ```
- Silent fallback via `try/except FileNotFoundError` in grid rendering functions to handle missing scene data gracefully:
  ```python
  try:
      xyz, rgb = load_scene(sid)
      ...
  except FileNotFoundError:
      ax.text(0.5, 0.5, "missing", ...)
  ```
- `for/else` pattern used to detect episode not found:
  ```python
  for fpath in iter_parquet_files():
      ...
      if len(ep_df) > 0:
          break
  else:
      print(f"Episode {episode_id} not found in local data.")
      return
  ```

## Logging

**Framework:** `print()` — no logging library used

**Patterns:**
- Progress messages use `print()` with descriptive text
- Saved file paths always printed with `print(f"Saved → {out_path}")`
- Consistent arrow notation `→` for output paths

## Module Entry Points

All scripts follow the standard Python entry point guard:

```python
if __name__ == "__main__":
    main()
```

`main()` function handles argument parsing (`argparse`) and delegates to specific rendering/loading functions.

## Comments

**Docstrings:**
- Module-level docstrings in triple quotes documenting data layout and CLI usage (see `explorations/lib/explore.py`, `explorations/real3dqa/explore.py`, `explorations/create_scene.py`)
- Function docstrings used selectively for non-obvious functions: `collect_first_frames()`, `episode_strip()`

**Inline Comments:**
- Numbered step comments in complex geometry code (see `render_scenes.py` steps 1–7)
- Brief explanatory comments for non-obvious MuJoCo behaviors:
  ```python
  img = obs[key][::-1]  # MuJoCo images are upside-down
  ```

## Function Design

**Size:** Functions are kept focused; most are under 40 lines
**Parameters:** Keyword arguments with defaults for configurable values (`n_episodes: int = 10`, `out_path: Path = None`)
**Return Values:** Functions either return data structures (loaders) or `None` with side effects (renderers); output path returned from `render_scene()` for caller use

## Module Design

**Exports:** No `__all__` defined; scripts are intended to be run directly, not imported as modules (except `explorations/lib/explore.py` which acts as a shared library)
**Barrel Files:** Not used; this is a scripts-only codebase

---

*Convention analysis: 2026-07-07*
