# Coding Conventions

**Analysis Date:** 2026-09-05

## Naming Patterns

**Files:**
- `snake_case` for every Python file: `explore.py`, `render_scenes.py`, `download_real3dqa.py`, `create_scene.py`, `soarm_sanity.py`, `record_episode.py`, `servo_scan.py`
- Descriptive verb-first names reflecting the action performed: `download_*`, `render_*`, `create_*`, `explore*`, `servo_*`, `record_*`

**Functions:**
- `snake_case` throughout, verb-noun pattern: `load_scene()`, `bird_eye_grid()`, `render_point_cloud()`, `collect_first_frames()`, `episode_grid()`, `episode_strip()`, `open_cameras()`, `record_still()`
- Private/internal helper functions prefixed with underscore: `_report()` in `explorations/soarm_sanity.py`

**Variables:**
- `snake_case` for all locals and module-level variables
- Short, established abbreviations for common data: `xyz`, `rgb`, `pts_cam`, `idx`, `sid`, `ep`, `df`, `fpath`
- Loop/index variables kept terse (`i`, `j`, `row_i`) inside small scopes only

**Types/Constants:**
- Module-level path and config constants in `UPPER_SNAKE_CASE`: `ROOT`, `DATA`, `OUT`, `OUT_DIR`, `BDDL`, `LIBERO_PATH`, `REPO_ROOT`, `ARM_XML`, `GRIPPER_XML`, `TASKS`, `DEFAULT_BAUD`, `COMMON_BAUDS`, `STS_PROTOCOL_END`, `ADDR_PRESENT_POSITION`, `BROADCAST_ID`
- No classes defined in project-owned code (`explorations/`, `diagnostics/`, `control/`) outside vendored `LIBERO/` — all scripts are function + `main()` based

## Code Style

**Formatting:**
- No formatter config present (no `.prettierrc`, `black` config, `pyproject.toml` `[tool.black]`, or `.editorconfig`) — style is manually consistent, not tool-enforced
- 4-space indentation throughout every `.py` file
- Single blank line between top-level function definitions; no blank line padding inside short functions
- Line length not enforced by tooling but stays under ~100 chars in practice; the LIBERO task list comment block in `explorations/soarm_sanity.py:57-64` is a rare example of a longer wrapped comment block

**Linting:**
- No linter config detected (no `.flake8`, `.pylintrc`, `ruff.toml`, `setup.cfg` at repo root)
- `explorations/requirements.txt`, `diagnostics/requirements.txt`, `control/requirements.txt` are separate per-tool dependency lists — no shared root `requirements.txt`

## Import Organization

**Order:**
1. Standard library (`argparse`, `csv`, `io`, `json`, `os`, `sys`, `time`, `pathlib.Path`)
2. Third-party (`numpy`, `pandas`, `matplotlib`, `cv2`, `PIL`)
3. Local/vendored packages (`lerobot.*`, `libero.*`, `scservo_sdk`)

**Special pattern — matplotlib backend:**
- `matplotlib.use("Agg")` is called immediately after `import matplotlib` and before `import matplotlib.pyplot as plt` in every visualization script (`explorations/lib/explore.py:30-32`, `explorations/real3dqa/explore.py`, `explorations/soarm_sanity.py:45-46`). Always replicate this order when adding new rendering scripts — importing `pyplot` first before setting the backend breaks headless rendering.

**Path style:**
- `pathlib.Path` is preferred in newer scripts (`explorations/lib/explore.py`, `control/record_episode.py`): `ROOT = Path(__file__).resolve().parent.parent`
- Older scripts (`render_scenes.py`, `download_real3dqa.py`, `explorations/soarm_sanity.py`) use `os.path` + `os.path.join`/`os.path.abspath` instead — both styles coexist; match whichever style the file you're editing already uses rather than mixing them in one file

**No path aliases** — no `tsconfig`/import-alias mechanism exists (pure Python project).

## Error Handling

- Explicit typed exceptions raised with descriptive messages for missing data: `raise FileNotFoundError(f"Point cloud not found: {path}")` (`explorations/real3dqa/explore.py:33`)
- Silent-fallback `try/except FileNotFoundError` used in grid-rendering functions so a missing scene doesn't kill the whole render loop (`explorations/real3dqa/explore.py:66`)
- `for/else` used to detect "not found" after exhausting an iterable (`explorations/lib/explore.py:120-127`, episode search across parquet files) — falls into `else:` branch to print a message and `return` early
- Broad `except Exception as exc:` used deliberately in the sanity/soak-test harness (`explorations/soarm_sanity.py:134,161,177,215,236,253`) to convert any failure into a PASS/FAIL report line rather than crash the check runner — this is intentional for a validation harness, not a general pattern to copy into normal exploration scripts
- Hardware scripts (`diagnostics/servo_scan.py`, `control/record_episode.py`) check return codes/booleans from SDK calls (`port.openPort()`, `cap.isOpened()`) and `print()` + `sys.exit(1)` or skip-and-continue rather than raising — matches the interactive/CLI nature of hardware diagnostics
- No centralized exception handling, error middleware, or logging framework — errors either propagate as raw Python tracebacks or are converted to printed PASS/FAIL/WARNING lines

## Logging

- No logging framework (`logging` module not used) — all output via `print()`
- Progress and status messages use descriptive `print()` text, not structured logs
- Saved output paths always printed with a consistent arrow notation: `print(f"Saved → {out_path}")` (`explorations/lib/explore.py:115,157`)
- Hardware/diagnostic scripts print `WARNING:`, `FAIL —`, `PASS —` prefixes for scannable status (`control/record_episode.py:35`, `explorations/soarm_sanity.py:72-78`)

## Comments

- Module-level docstrings (triple-quoted) at the top of every script documenting purpose, data layout, and CLI usage — see `explorations/lib/explore.py:1-21`, `explorations/soarm_sanity.py:1-21`, `control/record_episode.py:1-18`, `diagnostics/servo_scan.py`
- Function docstrings used selectively, only for non-obvious behavior: `collect_first_frames()`, `episode_strip()`, `_report()`
- Numbered step comments for complex geometry/math logic (see `explorations/render_scenes.py` steps 1-7)
- Inline comments explain non-obvious hardware/domain facts and carry-forward context, e.g. the LIBERO reach-margin math block in `explorations/soarm_sanity.py:57-64` and the sys.path rationale comments at `explorations/soarm_sanity.py:29-41`
- Comments frequently reference plan/phase IDs from the planning system (e.g. "Phase 2, D-09", "SC-1") — when adding code tied to a GSD phase/decision, follow this pattern and cite the ID inline

## Function Design

- Scripts are structured as a flat list of top-level functions plus a single `main()` that parses `argparse` args and dispatches — no classes, no OOP abstraction layers in project-owned code
- Functions generally do one I/O-adjacent thing (load data, render one figure, run one check) and return early on failure conditions
- `argparse` is the standard CLI pattern: every entry-point script defines a `parser = argparse.ArgumentParser()` in `main()`, adds `--flag` options with `type=`, `default=`, `help=`, and dispatches based on parsed args (`explorations/lib/explore.py:170-183`, `control/record_episode.py:108-138`, `diagnostics/servo_scan.py`)
- Repeatable CLI args use `action="append"`: `parser.add_argument("--camera", type=int, action="append", ...)` (`control/record_episode.py:112`)

## Module Design

- No shared library/common module across exploration, diagnostics, or control scripts — each script is self-contained and resolves its own paths via `Path(__file__).resolve()` or `os.path.dirname(__file__)`
- Each subproject directory (`explorations/`, `diagnostics/`, `control/`) has its own `requirements.txt` and its own `.venv` — dependencies are not shared or centralized
- `if __name__ == "__main__": main()` guard used at the bottom of every entry-point script
- `LIBERO/` is a vendored/embedded benchmark repo with its own git history — treat it as read-only third-party code, do not apply project conventions retroactively to it

---

*Convention analysis: 2026-09-05*
