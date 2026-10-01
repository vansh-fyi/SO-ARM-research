"""Auto-discovering registry of Phase 14 VLA-style `ActionSource` backends (PIPE-01).

Every backend is one thin sibling module in this package that registers a
`build(args, camera_names, joint_limits_deg) -> BackendHandles | None`
callable under its `--backend` name, via either:

  - `@register("<name>")` -- any backend (Tier C resident-HTTP-adapter lane,
    Gemini direct-cloud lane, the scripted dry run), or
  - `@register_grpc_backend("<name>")` -- a backend that rides the existing
    Colab `PolicyServer` gRPC bridge (Tier A/B) and therefore requires
    `--server-address`/`--checkpoint` together. `run_vla_episode.py` scopes
    that mutual-requirement check to `GRPC_BRIDGE_BACKENDS` only.

Importing `vla_bridge.backends` anywhere runs `_discover_backends()` once,
which imports every sibling module whose name does not start with `base_`
(shared per-tier helpers: `base_grpc.py`, `base_http_adapter.py`) or `_`
(private helpers), so each module's decorator populates `BACKEND_REGISTRY`.
Adding a backend therefore never requires editing this file.

Backend-module convention: keep heavy or optional third-party imports
(model SDKs, cloud clients) INSIDE `build()`, not at module top level.
Discovery imports every backend module on every `run_vla_episode.py` start,
and an import failure here is deliberately NOT swallowed -- a broken
backend module should fail loudly, not silently vanish from `--backend`'s
choices.

Depth is out of scope for every backend (14-CONTEXT.md D-03): nothing in
this package touches `depth_camera.py`/`fastfs_server.py`/`depth_recorder.py`.
"""

import importlib
import json
import pkgutil
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BACKEND_REGISTRY: dict[str, Callable] = {}
GRPC_BRIDGE_BACKENDS: set[str] = set()

# control/vla_bridge/backend_configs/<name>.json -- one small per-backend
# config (checkpoint / endpoint_url / action_space ...), per 14-CONTEXT.md D-06.
# Module-level so tests can point it at a tmp_path.
BACKEND_CONFIG_DIR = Path(__file__).resolve().parent.parent / "backend_configs"

_CONFIG_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


@dataclass
class BackendHandles:
    """What every registered backend's `build()` returns to `run_vla_episode.py`.

    `action_source` satisfies `run_vla_episode.ActionSource` (structural:
    `get_action(joint_state, instruction) -> (action, model_version)`).
    `robot` is the one true connected robot handle for the control loop.
    `stereo_camera` is the already-open `StereoSplitCamera` (or `None`) that
    `run_episode()` reuses for its diagnostic overhead recording, never a
    second open of the AR0144. `skip_camera_names` are camera names the
    caller's `_open_cameras()` must NOT open itself because this backend
    already holds that device exclusively.
    """

    action_source: Any
    robot: Any
    stereo_camera: Any | None = None
    skip_camera_names: frozenset[str] = frozenset()


def _add(name: str, obj: Callable) -> None:
    existing = BACKEND_REGISTRY.get(name)
    if existing is not None and existing is not obj:
        raise ValueError(
            f"backend name {name!r} is already registered by {existing!r}; "
            f"refusing to silently replace it with {obj!r}"
        )
    BACKEND_REGISTRY[name] = obj


def register(name: str):
    """Decorator factory: store the decorated callable/class under `name` in
    `BACKEND_REGISTRY` and return it unchanged."""

    def decorator(obj):
        _add(name, obj)
        return obj

    return decorator


def register_grpc_backend(name: str):
    """Like `register()`, and also marks `name` as a gRPC-bridge-lane backend
    (adds it to `GRPC_BRIDGE_BACKENDS`)."""

    def decorator(obj):
        _add(name, obj)
        GRPC_BRIDGE_BACKENDS.add(name)
        return obj

    return decorator


def get_backend_names() -> list[str]:
    return sorted(BACKEND_REGISTRY)


def load_backend_config(name: str) -> dict | None:
    """Load `backend_configs/<name>.json` if present and parseable.

    Returns `None` -- never raises -- on any failure (unsafe name, missing
    file, unreadable, invalid JSON, non-object JSON), mirroring
    `run_vla_episode.py::_load_device_map()`'s warn-and-fall-through
    discipline, so callers can fall back to explicit CLI args.
    """
    if not _CONFIG_NAME_RE.match(name):
        print(f"WARNING: refusing to load backend config for unsafe name {name!r}.")
        return None
    path = BACKEND_CONFIG_DIR / f"{name}.json"
    if not path.exists():
        return None
    try:
        config = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        print(f"WARNING: {path} exists but could not be read ({e}); ignoring it.")
        return None
    if not isinstance(config, dict):
        print(f"WARNING: {path} is not a JSON object; ignoring it.")
        return None
    return config


def _discover_backends() -> None:
    """Import every sibling backend module so its decorator registers it."""
    for module_info in pkgutil.iter_modules(__path__):
        module_name = module_info.name
        if module_name.startswith(("base_", "_")):
            continue
        importlib.import_module(f"{__name__}.{module_name}")


__all__ = [
    "BACKEND_REGISTRY",
    "GRPC_BRIDGE_BACKENDS",
    "BACKEND_CONFIG_DIR",
    "BackendHandles",
    "register",
    "register_grpc_backend",
    "get_backend_names",
    "load_backend_config",
]

_discover_backends()
