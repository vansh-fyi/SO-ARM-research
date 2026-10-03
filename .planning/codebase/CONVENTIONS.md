# Coding Conventions

**Analysis Date:** 2026-10-03

Scope: first-party Python in `control/`, `control/vla_bridge/`, `scripts/`, `diagnostics/`, `explorations/`. `LIBERO/` is vendored and follows its own upstream style; do not apply these rules there.

## Naming Patterns

**Files:**
- `snake_case.py` everywhere: `control/run_vla_episode.py`, `control/vla_bridge/safety_validator.py`
- Backends: one module per model under `control/vla_bridge/backends/` named after the model (`pi0.py`, `groot_n15.py`, `mllm_claude.py`)
- Shared backend bases use a `base_` prefix: `base_grpc.py`, `base_http_adapter.py`, `base_mllm.py`, `base_gemini.py`
- Policy servers (Colab side) use `<model>_server.py`: `control/vla_bridge/openvla_server.py`, `rdt1b_server.py`
- Tests: `test_<module>.py` beside the code (`control/test_safety_validator.py`), backend tests as `test_backend_<name>.py`
- Non-collected test helpers: no `test_` prefix (`control/fake_providers.py`)

**Functions:**
- `snake_case`, verb-first: `build_pi0`, `build_grpc_action_source`, `load_backend_config`, `get_backend_names`
- Backend factories are `build_<backend>` and registered via decorator
- Module-private helpers prefixed `_`: `_add`, `_discover_backends`

**Variables:**
- `snake_case`; unit suffixes for physical quantities: `joint_limits_deg`
- Module constants `UPPER_SNAKE_CASE`: `JOINT_ORDER`, `ACTION_UNITS`, `ACTION_MODE`, `CALIBRATION_PATH`, `BACKEND_REGISTRY`, `GRPC_BRIDGE_BACKENDS`

**Types:**
- `PascalCase` classes: `BackendHandles`, `MockRobot`, `ScriptedAdapter`
- Modern type hints (`list[str]`, `dict | None`, `Callable`) on public functions; Python 3.12 (`control/.venv`)
- Immutable sequences as tuples (`JOINT_ORDER = (...)`)

## Code Style

**Formatting:**
- No formatter config (no `pyproject.toml`, `.flake8`, `ruff.toml`); 4-space indent, double quotes, f-strings
- Lines up to ~110 chars are tolerated

**Linting:**
- None enforced. Keep imports sorted isort-style (stdlib / third-party / local) as existing files do

## Import Organization

**Order:**
1. Standard library (`json`, `math`, `time`, `pathlib.Path`, `types.SimpleNamespace`)
2. Third-party (`pytest`, `numpy`, `cv2`)
3. Local package (`from vla_bridge import ...`, `from vla_bridge.action_contract import JOINT_ORDER`)

**Lazy imports:** Heavy or optional deps (vendor SDKs `anthropic`, `openai`, `google-genai`; `numpy`, `cv2` in `control/vla_bridge/io_logger.py`) are imported inside the function that needs them, with a comment explaining why, so backend discovery and pure-logic tests stay importable.

**Path Aliases:**
- None. `control/` is the import root: run code and tests from `control/` so `vla_bridge` resolves.

## Error Handling

**Patterns:**
- Fail loudly with `ValueError` for bad input/config, `RuntimeError` for unexpected environment state; messages include the offending value via `!r`:
  ```python
  raise ValueError(f"camera_map must be a non-empty dict, got {camera_map!r}")
  ```
- Refuse silent overwrites (see `_add` in `control/vla_bridge/backends/__init__.py`)
- Safety-critical action validation is centralized in `control/vla_bridge/safety_validator.py`; never send actions to hardware that bypass it
- Secrets (API keys) come from env vars named in backend configs (`api_key_env`); never hard-code or log them

## Logging

**Framework:** `print()` for CLI progress; structured JSON Lines via `control/vla_bridge/io_logger.py` for episode I/O

**Patterns:**
- Per-episode logs and outputs go under `control/logs/` and `control/outputs/`
- Exploration scripts print saved paths as `print(f"Saved → {path}")`

## Comments

**When to Comment:**
- Explain *why* and cite the verified source (e.g. which installed `lerobot` 0.6.1 file a constant was checked against)
- Reference plan/requirement IDs in docstrings and requirement comments (`VLAHW-01`, `Plan 14-03`, `PIPE-02`)
- Annotate pinned dependencies in `control/requirements.txt` with purpose, source repo, license

**Docstrings:**
- Every module has a substantive triple-quoted docstring describing contract and rationale (`control/vla_bridge/action_contract.py`)
- Public functions get short docstrings with backticked identifiers

## Function Design

**Size:** Small, single-purpose helpers; large orchestration lives in CLI entry scripts (`control/run_vla_episode.py`)

**Parameters:** Pass parsed CLI `args` namespaces into backend builders; optional params default to `None`

**Return Values:** Backends return a `BackendHandles` dataclass (action source + optional `shutdown`)

## Module Design

**Exports:** Registry pattern: decorate builders with `@register("name")` or `@register_grpc_backend("name")`; `_discover_backends()` imports all modules in `backends/`. Per-backend config lives in `control/vla_bridge/backend_configs/`.

**Barrel Files:** `control/vla_bridge/backends/__init__.py` holds the registry; no other re-export barrels.

---

*Convention analysis: 2026-10-03*
