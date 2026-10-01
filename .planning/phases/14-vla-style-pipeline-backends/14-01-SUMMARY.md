---
phase: 14-vla-style-pipeline-backends
plan: 01
subsystem: control/vla_bridge
status: complete
tags: [backends, registry, grpc-bridge, http-adapter, action-source, pipe-01]
requires: []
provides:
  - "vla_bridge.backends: BACKEND_REGISTRY, GRPC_BRIDGE_BACKENDS, register(), register_grpc_backend(), BackendHandles, load_backend_config(), get_backend_names(), _discover_backends()"
  - "vla_bridge.backends.base_grpc.build_grpc_action_source(policy_type, checkpoint, args, camera_names, joint_limits_deg)"
  - "vla_bridge.backends.base_http_adapter: HttpAdapterActionSource, build_http_adapter_backend(), capture_observation_images()"
  - "robot_client.connect_bridge(..., policy_type='smolvla')"
affects: [14-02, 14-03, 14-04, 14-05, 14-06, 14-07, 14-08, 14-09, 14-10, 14-11, 14-12, 14-13]
tech-stack:
  added: []
  patterns:
    - "decorator-based auto-discovering registry (pkgutil.iter_modules + importlib)"
    - "never-raise hold-position ActionSource contract on the HTTP lane"
    - "final-action JOINT_ORDER shape gate before safety_validator"
key-files:
  created:
    - control/vla_bridge/backends/__init__.py
    - control/vla_bridge/backends/base_grpc.py
    - control/vla_bridge/backends/base_http_adapter.py
    - control/test_backend_registry.py
  modified:
    - control/vla_bridge/robot_client.py
    - control/test_robot_client.py
decisions:
  - "connect_bridge()'s policy_type kwarg sits after `task` (first legal position): a defaulted param before the required robot_config/task is a SyntaxError"
  - "HttpAdapterActionSource accepts any non-null JSON `action` (list or dict), runs convert_fn, then requires the FINAL action to be a dict with every JOINT_ORDER key mapped to a real number, otherwise it holds position"
  - "HTTP-lane frames are downsized to robot_client.OBSERVATION_MAX_SIDE (640) before PNG encoding, reusing _downsize_for_transport"
  - "register() raises ValueError on a second, different object under an existing name"
  - "_discover_backends() does not swallow import errors; backend modules must keep heavy/optional imports inside build()"
metrics:
  duration: "~6 min"
  completed: 2026-10-01
  tasks: 2
  files: 6
---

# Phase 14 Plan 01: Backend Registry and Shared Tier Helpers Summary

Added the `control/vla_bridge/backends/` package: a registry that discovers backend modules automatically, the `BackendHandles` result type, a gRPC-bridge helper (`build_grpc_action_source`) and an HTTP-adapter client (`HttpAdapterActionSource` / `build_http_adapter_backend`) that never raises. `connect_bridge()` now takes an optional `policy_type` that defaults to `"smolvla"`, so existing SmolVLA callers behave exactly as before.

## What Was Built

### Task 1: `connect_bridge(policy_type=...)` (commits c1e208f RED, ab2c744 GREEN)
- New `policy_type: str = "smolvla"` keyword is passed through to `RobotClientConfig.policy_type`. The only production caller (`run_vla_episode.py`) does not pass it, so its behavior is unchanged.
- Checked against the installed `lerobot==0.6.1` source and recorded in the docstring: on the client, `RobotClientConfig.__post_init__` only rejects an **empty** `policy_type`. The allow-list (`async_inference/constants.py::SUPPORTED_POLICIES = ["act", "smolvla", "diffusion", "tdmpc", "vqbet", "pi0", "pi05", "groot"]`) is enforced on the server in `policy_server.py:130`. An unsupported value therefore makes the handshake fail and `connect_bridge` returns `None`; it does not raise on the client.
- `BridgeActionSource` is unchanged. It stays generic: it only stores client, checkpoint and joint limits.
- 2 new tests: an explicit `"pi0"` reaches the config, and omitting the kwarg still gives `"smolvla"`.

### Task 2: the `backends/` package (commits 085a68b RED, 153fddd GREEN)
- **`__init__.py`** provides:
  - `BACKEND_REGISTRY` and `GRPC_BRIDGE_BACKENDS`.
  - `register(name)` and `register_grpc_backend(name)`. Both return the decorated object unchanged and raise `ValueError` on a conflicting duplicate name.
  - The `BackendHandles` dataclass with fields `action_source`, `robot`, `stereo_camera=None` and `skip_camera_names=frozenset()`.
  - `load_backend_config(name)`. It reads `control/vla_bridge/backend_configs/<name>.json` (the directory is the module constant `BACKEND_CONFIG_DIR`, which tests patch). It never raises. It warns and returns `None` for an unsafe name, a missing file, unreadable or invalid JSON, or JSON that isn't an object.
  - `get_backend_names()`.
  - `_discover_backends()`, which runs at import time and skips modules whose names start with `base_` or `_`.
- **`base_grpc.py`**: `build_grpc_action_source()` repeats `run_vla_episode.py`'s current bridge-path construction. It builds `SOFollowerRobotConfig`, takes the first camera named `"wrist"` as the wrist index, and calls `connect_bridge(..., policy_type=policy_type)`. It returns `None` if the bridge is unreachable; otherwise it returns `BackendHandles(BridgeActionSource(...), client.robot, client._stereo_camera, {"overhead"})`.
- **`base_http_adapter.py`**:
  - `HttpAdapterActionSource.get_action()`:
    - **One read per call:** reads the wrist camera once and the stereo pair once.
    - **Encoding:** downsizes each frame to 640 on its longer side, then base64-encodes it as PNG.
    - **Request:** POSTs `{"images": {wrist, overhead_left, overhead_right}, "joint_state", "instruction"}` to `{endpoint_url}/action`.
    - **Conversion:** calls `convert_fn(parsed_action, joint_state)` when one is given.
    - **Fail-safe:** never raises. Every failure path returns `(dict(joint_state), "<checkpoint>@<reason>")`, with these reasons:
      - `http-request-inflight-holding-position`
      - `camera-read-failed-holding-position`
      - `http-error-holding-position`: a transport error, an HTTP 4xx/5xx, a body that isn't JSON or isn't an object, or a missing/null `action`
      - `action-conversion-error-holding-position`
      - `action-format-error-holding-position`
    - **Diagnostics:** on success it sets `last_details` (`raw_model_output`, `bridge_flags`, `observation=None`). A `close()` method releases the dedicated wrist capture.
  - `build_http_adapter_backend()` copies the scripted path's robot construction: `SO101Follower`, `connect(calibrate=False)`, then `write_calibration` with torque disabled. It opens its own `StereoSplitCamera` and a dedicated wrist `cv2.VideoCapture`. If no camera is named `"wrist"`, it raises `SystemExit` before touching the robot.
  - `capture_observation_images()` is factored out ahead of time as the shared capture helper that Plan 14-13's Gemini base is told to reuse.
- `test_backend_registry.py` has 32 tests covering the registry, discovery, config loading, the HTTP source's failure and success paths, the request payload format, the in-flight guard, and both build helpers.

## Verification
- `cd control && <main .venv>/bin/python -m pytest -q`: **242 passed** (baseline 208, plus 2 from Task 1 and 32 from Task 2).
- `import vla_bridge.backends` succeeds, and the registry is empty, as expected until Plan 14-02 adds `scripted`/`smolvla`.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] `policy_type` parameter position**
- **Found during:** Task 1
- **Issue:** The plan said to put `policy_type: str = "smolvla"` right after `checkpoint`. That would put a defaulted parameter before the required `robot_config`/`task`, which is a Python SyntaxError. It would also shift the positional order.
- **Fix:** Placed it right after `task`, the first legal position. Every existing caller is unaffected.
- **Commit:** ab2c744

**2. [Rule 2 - Correctness/Safety] Final-action shape check on the HTTP lane**
- **Found during:** Task 2 (reading `safety_validator.validate_action()` and Plans 14-03/09/10/11/12)
- **Issue:** Later Tier C servers return `"action"` as a **float list**, not a dict. Separately, `validate_action()` calls `math.isfinite(value)`, which raises `TypeError` on a non-numeric value and would crash the live loop. It also skips missing joints, which produces the partial `goal_pos` that lerobot rejects (see the crash already documented in `pop_validated_action`).
- **Fix:** The raw `action` may be a list or a dict. After `convert_fn`, the final action must be a dict with every `JOINT_ORDER` key mapped to a real number (bools rejected); otherwise the source holds position with `@action-format-error-holding-position`. NaN/inf are still passed to the validator, which already rejects them. If `convert_fn` raises, the source holds with `@action-conversion-error-holding-position`.
- **Commit:** 153fddd

**3. [Rule 2 - Robustness] Extra fail-safe paths and an in-flight guard on `HttpAdapterActionSource`**
- A camera read or encode failure now holds position without POSTing. Previously `cv2.imencode(None)` would have raised.
- Added the same non-blocking in-flight lock as `BridgeActionSource`, as 14-PATTERNS.md requires for every ActionSource.
- Added `close()` to release the dedicated wrist `VideoCapture`, which would otherwise leak.
- **Commit:** 153fddd

**4. [Rule 2 - Performance] HTTP-lane frame downsizing**
- Frames are shrunk to `OBSERVATION_MAX_SIDE` (640) before PNG encoding, reusing `robot_client._downsize_for_transport`. The live gRPC lane already showed that full-resolution frames (1920x1080 wrist) stall the control loop for minutes over a tunnel, and every Tier C model's native input is much smaller than 640. `max_side` is a constructor parameter, so a backend can override it.
- **Commit:** 153fddd

**5. [Rule 2 - Security/Correctness] Registry and config guards**
- `register()` and `register_grpc_backend()` raise `ValueError` when a different object claims an existing name, so two modules can't silently both claim, say, `pi0`.
- `load_backend_config()` rejects names outside `[A-Za-z0-9][A-Za-z0-9_-]*` (path traversal) and JSON that isn't an object.
- **Commit:** 153fddd

**6. [Rule 2 - Forward compatibility] `capture_observation_images()` factored out early**
- Plan 14-13 tells its author to factor the HTTP source's capture-and-encode block into a standalone function in `base_http_adapter.py`. It was built that way from the start.
- **Commit:** 153fddd

## Integration Notes for Later Plans
- **Plan 14-02 (cleanup):** the Tier C lane has no `RobotClient`. On cleanup, call `robot.disconnect()`, `handles.stereo_camera.release()`, and `handles.action_source.close()` (wrist capture) when present. The current `finally` block only knows the `client.stop()` / `robot.disconnect()` split.
- **Plans 14-03/08/09/10/11/12 (servers):** the request body sends `joint_state` as a **dict keyed by `JOINT_ORDER`**, not a `proprio_state` list. A server that needs a list must convert on its side. Without a `convert_fn`, a list `action` always holds position, so 14-08's Wall-X HTTP fallback (`convert_fn=None`) needs either a server that returns a JOINT_ORDER dict or a reshaping `convert_fn` like the one 14-12 plans.
- **Plan 14-07:** `build_grpc_action_source()` has no `convert_fn` yet. 14-07 plans to add it as an additive change.
- **Backend modules:** keep model-SDK and cloud-client imports inside `build()`. `_discover_backends()` imports every module at startup and deliberately does not swallow import errors.

## Live-Source Verification Record (execution discipline)
- Verified against the installed `lerobot==0.6.1` source:
  - `RobotClientConfig.__post_init__` validation (`async_inference/configs.py:156-170`)
  - `SUPPORTED_POLICIES` (`async_inference/constants.py:26`) and where it is enforced (`policy_server.py:130`)
  - the `SOFollowerRobotConfig` / `SO101Follower` import paths, the same ones `run_vla_episode.py` already uses
- Verified against in-repo source:
  - `requests==2.34.2`: `requests.JSONDecodeError` subclasses `RequestException` (checked in the venv)
  - `safety_validator.validate_action()`'s `math.isfinite` and missing-joint behavior
  - `run_episode()`'s `last_details` consumption (`raw_model_output`, `observation`, `bridge_flags`)
  - `StereoSplitCamera`'s one-read-per-pair cache
- No backend-specific upstream API (model cards, adapter servers) is involved in this plan. Nothing was left unverified.
- No new dependencies were installed.

## Known Stubs
None. The registry is empty on purpose until Plan 14-02 registers `scripted` and `smolvla`.

## Threat Flags
None. T-14-01 holds: HTTP-lane actions still go through `run_episode()`'s unmodified `safety_validator.validate_action()` (`run_vla_episode.py:303-308`), and the new shape check adds a pre-filter, not an exemption. T-14-02 (auto-import) is accepted as planned.

## Human Checks Deferred
The plan has no `<human-check>` blocks. One hardware item carries forward from `connect_bridge()`'s existing docstring and now also applies to `build_http_adapter_backend()`:
- **test:** On real hardware, start a Tier C (HTTP-lane) episode and confirm the wrist USB camera tolerates two independent `cv2.VideoCapture` opens (the backend's own open plus `run_vla_episode.py`'s IOLogger open).
- **expected:** Both opens succeed, and wrist frames arrive in both the POSTed payload and `episode.jsonl`.
- **why_human:** No camera hardware is available in the coding environment.

## Self-Check: PASSED
- FOUND: control/vla_bridge/backends/__init__.py, base_grpc.py, base_http_adapter.py, control/test_backend_registry.py
- FOUND commits: c1e208f, ab2c744, 085a68b, 153fddd
