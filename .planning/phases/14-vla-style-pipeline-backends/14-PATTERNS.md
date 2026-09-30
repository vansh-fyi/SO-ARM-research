# Phase 14: VLA-Style Pipeline & Backends - Pattern Map

**Mapped:** 2026-10-01
**Files analyzed:** ~22 (registry + 12 backend subclasses + 3 base/mixin classes + backend configs + 3 new test files + CLI extension)
**Analogs found:** 22 / 22 (all have a strong in-repo analog; no "no analog" files)

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---|---|---|---|---|
| `control/vla_bridge/backends/__init__.py` (registry) | config/registry | request-response | `control/run_vla_episode.py::_load_device_map()` (small dict-producing loader pattern) | role-match |
| `control/vla_bridge/backends/base_grpc.py` (`GrpcBridgeMixin`) | service (base class) | streaming (gRPC bridge) | `control/vla_bridge/robot_client.py` (`connect_bridge`, `BridgeActionSource`, `pop_validated_action`) | exact |
| `control/vla_bridge/backends/pi0.py`, `act.py`, `diffusion_policy.py` (Tier A) | service (thin subclass) | streaming (gRPC bridge) | `control/vla_bridge/robot_client.py::connect_bridge()` (policy_type swap only) | exact |
| `control/vla_bridge/backends/groot_n17.py`, `wallx.py` (Tier B) | service (thin subclass + action conversion) | streaming (gRPC bridge) or fallback HTTP | `control/vla_bridge/robot_client.py::connect_bridge()` + `action_contract.py` (for EE-delta conversion) | role-match |
| `control/vla_bridge/backends/base_http_adapter.py` (`HttpAdapterMixin`, client-side) | service (base class) | request-response (HTTP) | `control/vla_bridge/depth_camera.py` (`DepthCameraClient`/`compute_depth()` — POST body build, base64 encode, fail-safe None-on-error) | exact |
| `control/vla_bridge/fastfs_server.py`-style adapter servers for Tier C (`openvla_server.py`, `octo_server.py`, `rdt1b_server.py`, `groot_n15_server.py`, and Wall-X/GR00T-N17 fallback servers) — Colab-side, not in `control/` | service (resident HTTP server) | file-I/O + request-response (load-once model server) | `control/vla_bridge/fastfs_server.py` (`FastFSModel`, `create_app`, `main()`) | exact |
| `control/vla_bridge/backends/openvla.py`, `openvla_oft.py`, `octo.py`, `rdt1b.py`, `groot_n15.py` (Tier C client subclasses) | service (thin subclass) | request-response (HTTP) | `control/vla_bridge/depth_camera.py::DepthCameraClient` (HTTP client shape) + `control/vla_bridge/robot_client.py::BridgeActionSource.get_action()` (ActionSource contract/flags/last_details shape) | role-match |
| `control/vla_bridge/backends/eef_to_joint.py` (IK conversion helper) | utility/transform | transform | none direct — new problem shape; `control/vla_bridge/action_contract.py` (defines the target `JOINT_ORDER`/units the conversion must produce) | role-match (target contract only) |
| `control/vla_bridge/backends/base_gemini.py` (`GeminiActionSourceBase`) | service (base class) | request-response (HTTPS cloud API) | `control/vla_bridge/depth_camera.py::DepthCameraClient` (closest HTTP-client shape in repo; no cloud-auth analog exists) | partial-match |
| `control/vla_bridge/backends/gemini_robotics.py`, `gemini_robotics_2.py` | service (thin subclass) | request-response (HTTPS) + transform (ER spatial output → joint action) | `control/vla_bridge/backends/base_gemini.py` + `eef_to_joint.py`-style conversion | role-match |
| `control/vla_bridge/backend_configs/*.json` | config | — | `control/device_map.json` (read via `_load_device_map()`) | role-match |
| `control/run_vla_episode.py` (add `--backend` flag, CLI branch) | controller/CLI | request-response | same file's existing `--checkpoint`/`--server-address` argparse block + `action_source = BridgeActionSource(...)` branch (~line 647-665) | exact (in-place extension) |
| `control/test_backend_adapters.py` | test | request-response (mocked HTTP) | `control/test_depth_pipeline.py` (FastFS-adapter mocked-HTTP test pattern) | exact |
| `control/test_gemini_action_source.py` | test | request-response (mocked HTTP) | `control/test_depth_camera.py` (mocked `requests`/HTTP client test pattern) | exact |
| `control/test_robot_client.py` (extend) | test | streaming (mocked gRPC client) | itself — existing `MockRobot`/`RobotClient` doubles, extend with new `policy_type` cases | exact |
| Shared `BACKEND_REGISTRY` conformance test (new, in `control/test_backend_adapters.py` or its own file) | test | structural conformance | `control/run_vla_episode.py::ActionSource` Protocol (~line 143) | exact |

## Pattern Assignments

### `control/vla_bridge/backends/base_grpc.py` (`GrpcBridgeMixin`) — service, streaming

**Analog:** `control/vla_bridge/robot_client.py` (full file, 438 lines — already read in full this session)

**Imports pattern** (lines 37-51):
```python
import queue
import threading
import time
from typing import Any

import cv2
import grpc
import numpy as np

from lerobot.async_inference.configs import RobotClientConfig
from lerobot.async_inference.robot_client import RobotClient
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

from vla_bridge import safety_validator
from vla_bridge.stereo_camera import StereoSplitCamera
```

**Core connect/bridge pattern — the ONLY thing Tier A subclasses change** (lines 168-177):
```python
config = RobotClientConfig(
    policy_type="smolvla",                 # <- swap per backend: "pi0", "pi05", "act", "diffusion"
    pretrained_name_or_path=checkpoint,     # <- swap per backend (HF repo id)
    robot=robot_config,
    actions_per_chunk=actions_per_chunk,
    task=task,
    server_address=server_address,
    policy_device=policy_device,
    chunk_size_threshold=chunk_size_threshold,
)
client = RobotClient(config)
```
Everything else in `connect_bridge()` (stereo camera wiring, `start_barrier` 1-party patch, `receive_actions()` daemon thread) is backend-agnostic and must NOT be duplicated per backend — factor it into `GrpcBridgeMixin.connect()` once, called by every Tier A/B subclass.

**get_action / ActionSource contract pattern** (lines 344-430, `BridgeActionSource.get_action()`):
```python
class BridgeActionSource:
    def __init__(self, client, checkpoint, joint_limits_deg, dt_s=0.5,
                 stale_watchdog_limit=STALE_WATCHDOG_CONSECUTIVE_LIMIT):
        self.client = client
        self.checkpoint = checkpoint
        self.joint_limits_deg = joint_limits_deg
        self._dt_s = dt_s
        self._last_action = None
        self._inflight_lock = threading.Lock()
        self._stale_watchdog_limit = stale_watchdog_limit
        self._consecutive_stale_count = 0
        self.last_details = {}

    def get_action(self, joint_state, instruction):
        self.last_details = dict(raw_model_output={}, bridge_flags=[], observation=None)
        if not self._inflight_lock.acquire(blocking=False):
            return joint_state, f"{self.checkpoint}@bridge-request-inflight-holding-position"
        try:
            # ... observation fetch, pop_validated_action(), stale watchdog ...
            if flags:
                return validated_action, f"{self.checkpoint}@{'+'.join(flags)}"
            return validated_action, f"{self.checkpoint}@unknown"
        finally:
            self._inflight_lock.release()
```
This `{checkpoint}@{flags}` model_version string convention, the non-blocking in-flight lock, `last_details` diagnostic dict, and the "never raise, always return (joint_state, reason-string) on failure" contract must be preserved identically by every new backend's `ActionSource` implementation — this is what keeps `run_vla_episode.py`'s control loop unchanged (PIPE-01).

**Error handling pattern** (lines 391-405):
```python
try:
    observation = self.client.control_loop_observation(task=instruction)
    ...
except (grpc.RpcError, ConnectionError, RuntimeError):
    return joint_state, f"{self.checkpoint}@bridge-error-holding-position"
```
Never let a transport exception propagate out of `get_action()` — always degrade to "hold position" with a descriptive model_version string.

**Anti-pattern warning to carry into every subclass's docstring** (lines 5-11, module docstring): never call `client.control_loop_action()`/`client.control_loop()` — both bypass `safety_validator.validate_action()` internally.

---

### `control/vla_bridge/backends/base_http_adapter.py` (`HttpAdapterMixin`, client-side) — service, request-response

**Analog:** `control/vla_bridge/depth_camera.py` (client) + `control/vla_bridge/fastfs_server.py` (server, for the Colab-side counterpart)

**Imports pattern** (depth_camera.py lines 21-31):
```python
import argparse
import base64
import io
import hashlib
import json
from dataclasses import asdict
import time
from pathlib import Path

import cv2
import numpy as np
import requests

from vla_bridge import stereo_calibration
```

**Fail-safe client call pattern** (depth_camera.py docstring, lines 14-18 — the exact convention to replicate for every Tier C backend's HTTP client):
```python
# compute_depth() never raises on a network/endpoint failure -- it always
# returns None, matching io_logger.capture_camera_frame()'s fail-safe
# convention, since a flaky Colab connection must not crash the live
# episode's real control loop.
```
Every Tier C `ActionSource.get_action()` must apply this same discipline: catch `requests.RequestException`/timeout, return `(joint_state, f"{checkpoint}@http-error-holding-position")` rather than raising — mirroring `BridgeActionSource`'s `grpc.RpcError` handling above, just for `requests` instead of `grpc`.

**Resident server pattern (Colab-side, one per Tier C backend)** — `control/vla_bridge/fastfs_server.py` full file:
```python
class FastFSModel:
    def __init__(self, checkpoint, valid_iters=8, max_disp=192):
        # Load once, resident -- never reload per request.
        ...

def create_app(infer):
    from flask import Flask, jsonify, request
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024
    inference_lock = threading.Lock()

    @app.get('/health')
    def health():
        return jsonify(status='ready', protocol_version=1, model=infer.metadata)

    @app.post('/depth')      # -> rename '/action' per RESEARCH.md's Pattern 2
    def depth():
        if not inference_lock.acquire(blocking=False):
            return jsonify(error='... busy'), 429
        try:
            body = request.get_json()
            # decode/validate inputs, raise ValueError on bad input
            result = infer(...)
            return jsonify(...)
        except (ValueError, KeyError, TypeError) as exc:
            return jsonify(error=str(exc)), 400
        finally:
            inference_lock.release()
    return app

def main():
    # argparse --repo/--checkpoint/--port, sys.path.insert for vendored repo,
    # warmup call BEFORE create_app(...).run(...) reports ready
    ...
```
Key conventions every Tier C `<backend>_server.py` must copy exactly: (1) model loaded once in `__init__`, never per-request; (2) `inference_lock` + `429` busy response, not a request queue; (3) `/health` GET returning `status='ready'` + model metadata for the same startup-polling pattern `policy_server_launch.md` already uses; (4) explicit warmup call before `app.run()` so `/health` never reports ready before the model can actually serve; (5) `ValueError/KeyError/TypeError` → 400, anything else propagates (crash-is-visible, not swallowed).

---

### `control/vla_bridge/backends/eef_to_joint.py` (IK conversion helper) — utility, transform

**Analog (target contract only, not the conversion logic itself):** `control/vla_bridge/action_contract.py`

**Contract excerpt every conversion function must produce** (lines 25-49):
```python
JOINT_ORDER = (
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
)
ACTION_UNITS = {
    "shoulder_pan": "degrees", "shoulder_lift": "degrees", "elbow_flex": "degrees",
    "wrist_flex": "degrees", "wrist_roll": "degrees", "gripper": "percent_0_100",
}
ACTION_MODE = "absolute"   # targets, never deltas, at the point they reach safety_validator
```
Any EEF-delta output (OpenVLA, OpenVLA-OFT, Octo, RDT-1B if EEF-configured, GR00T N1.7) MUST be converted to this exact dict shape/units/absolute-target convention BEFORE being passed to `safety_validator.validate_action()` — there is no analog for the IK math itself in this repo (per RESEARCH.md, use an existing SO-101 IK reference, not hand-rolled).

---

### `control/vla_bridge/backends/base_gemini.py` (`GeminiActionSourceBase`) — service, request-response (HTTPS)

**Analog (closest available — no cloud-auth precedent exists in-repo):** `control/vla_bridge/depth_camera.py::DepthCameraClient` for HTTP client shape + `control/vla_bridge/robot_client.py::BridgeActionSource` for the `ActionSource` contract/flags/`last_details` shape (see excerpt above).

**What to reuse:** the fail-safe "never raise from get_action(), return holding-position with a descriptive reason string" convention, and the `last_details` diagnostic-dict pattern (for `io_logger` fields) — NOT a gRPC pattern, since Gemini uses `google-genai` over HTTPS directly (D-02, no Colab hop).

**Security note to carry into this file specifically** (from RESEARCH.md's Known Threat Patterns): log only the response body/parsed action, never the outgoing request's auth header — mirrors `io_logger.py`'s existing disclosure discipline of never persisting secrets to `episode.jsonl`.

---

### `control/run_vla_episode.py` (extend with `--backend` flag) — controller, request-response

**Analog:** itself — existing argparse + branch pattern (already read in full this session, lines 428-665)

**Existing argparse pattern to extend, not replace** (lines 466-478):
```python
parser.add_argument(
    "--server-address", type=str, default=None,
    help="Colab PolicyServer bridge address (host:port). When given, drives the robot "
    "with a real Colab-hosted VLA via BridgeActionSource instead of ScriptedActionSource.",
)
parser.add_argument(
    "--checkpoint", type=str, default=None,
    help="HF Hub SmolVLA checkpoint id. Required when --server-address is given.",
)
```
Per D-06: add `--backend` (choices=[...13 names...], default `"smolvla"`) as a NEW flag selecting which per-backend config/class to use; keep `--checkpoint` as the existing Colab-lane-specific override it already is (do not overload it as the backend-type selector).

**Existing validation-pairing pattern to mirror for new flags** (lines 520-523):
```python
if args.server_address and not args.checkpoint:
    parser.error("--checkpoint is required when --server-address is given")
if args.checkpoint and not args.server_address:
    parser.error("--server-address is required when --checkpoint is given")
```

**Existing branch point to extend** (lines 613-665, inside the `try:` after cameras open): today it's a hard `if args.server_address: ... BridgeActionSource(...) else: ... ScriptedActionSource()`. Per D-04/D-06, this becomes a dispatch through `BACKEND_REGISTRY[args.backend]` that returns the right `ActionSource` instance (gRPC-mixin-based, HTTP-adapter-mixin-based, or Gemini-based) while keeping the exact same `action_source = ...` single-assignment shape the control loop below is unaware of.

---

## Shared Patterns

### ActionSource structural contract (applies to ALL 13 backends)
**Source:** `control/run_vla_episode.py` lines 143-156 (`ActionSource` Protocol)
```python
class ActionSource(Protocol):
    def get_action(
        self, joint_state: dict[str, float], instruction: str
    ) -> tuple[dict[str, float], str]:
        """Return (raw_action, model_version) for the given observation."""
        ...
```
Every new backend class — regardless of tier — must satisfy this exact signature. This is what the "Shared `BACKEND_REGISTRY` conformance test" (Wave 0 gap) asserts.

### Safety validation gate (unmodified, applies to ALL 13 backends)
**Source:** `control/vla_bridge/safety_validator.py` lines 1-16 (module docstring) + the per-joint loop referenced in RESEARCH.md line 344-351
```python
for joint in action_contract.JOINT_ORDER:
    if joint not in raw_action:
        continue
    value = raw_action[joint]
    # NaN/inf rejection, absolute clamp, per-step clamp, velocity clamp --
    # identical regardless of which backend produced `raw_action`.
```
No backend subclass may call `safety_validate_fn` with modified caps or skip it — this is VLAB success criteria #3 across all 12 new backends.

### Fail-safe "never raise from get_action()" convention
**Source:** `control/vla_bridge/robot_client.py::BridgeActionSource.get_action()` (lines 372-430) + `control/vla_bridge/depth_camera.py`'s `compute_depth()` docstring (lines 14-18)
Apply to: every gRPC-mixin, HTTP-adapter-mixin, and Gemini subclass — any transport/inference failure degrades to `(joint_state, f"{checkpoint}@<reason>-holding-position")`, never an unhandled exception reaching `run_vla_episode.py`'s control loop.

### Resident/warm model-server process pattern (Colab-side, Tier C only)
**Source:** `control/vla_bridge/fastfs_server.py` (full file, see excerpt above)
Apply to: every Tier C backend's Colab-side adapter server (OpenVLA, OpenVLA-OFT, Octo, RDT-1B, GR00T N1.5, and GR00T N1.7/Wall-X if Tier B dispatch verification fails). Load-once in `__init__`, explicit warmup before `/health` reports ready, `inference_lock` + 429-busy rather than a request queue, `/health` + one POST action endpoint.

### Small per-integration config file pattern
**Source:** `control/run_vla_episode.py::_load_device_map()` (lines 197-209) + `control/device_map.json`
```python
def _load_device_map() -> dict | None:
    if not DEVICE_MAP_PATH.exists():
        return None
    try:
        device_map = json.loads(DEVICE_MAP_PATH.read_text())
    except (json.JSONDecodeError, OSError) as e:
        print(f"WARNING: {DEVICE_MAP_PATH} exists but could not be read ({e}); ignoring it.")
        return None
```
Apply to: `control/vla_bridge/backend_configs/<name>.json` loading (D-06's per-backend config) — same "never raise, warn and fall through" discipline.

### Test-double pattern for mocked transport
**Source:** `control/test_robot_client.py` (849 lines, existing `MockRobot`/`RobotClient` doubles used to test `BridgeActionSource` without real gRPC) and `control/test_depth_camera.py`/`control/test_depth_pipeline.py` (mocked `requests`-based HTTP doubles used to test `DepthCameraClient`/FastFS adapter without a real network call)
Apply to: `control/test_robot_client.py` extension for Tier A/B `policy_type` cases; new `control/test_backend_adapters.py` for Tier C HTTP adapters; new `control/test_gemini_action_source.py` for Gemini's mocked HTTP responses.

## No Analog Found

None — every file classified above has at least a role-match or better analog in the existing codebase. The one partial-match (Gemini's HTTPS/cloud-auth transport) has no direct in-repo precedent for cloud API auth specifically, but its `ActionSource` contract, fail-safe error handling, and diagnostic-logging shape are fully covered by `BridgeActionSource`/`DepthCameraClient`.

## Metadata

**Analog search scope:** `control/vla_bridge/` (all modules), `control/run_vla_episode.py`, `control/test_*.py`
**Files scanned:** `robot_client.py` (438 lines, read in full), `run_vla_episode.py` (relevant ~300 lines of ~700+), `fastfs_server.py` (122 lines, read in full), `depth_camera.py` (~70 lines read), `action_contract.py` (~60 lines read), `safety_validator.py` (~40 lines read), plus `Bash`/`grep` sweeps over `control/test_*.py` and `control/vla_bridge/*.md`
**Pattern extraction date:** 2026-10-01
