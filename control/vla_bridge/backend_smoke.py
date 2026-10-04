"""Live smoke check for one Phase 14 VLA backend, with no robot attached (quick task 261004-dz7).

What this is
============
Before a Phase 14 hardware UAT run (14-UAT.md tests 2 and 6-18), this tool
proves that one backend works end to end against its live server: the Colab
PolicyServer (gRPC), the Colab `<backend>_server.py` adapter (HTTP) or the
Gemini API. It runs the backend's REAL registered factory,
`vla_bridge.backends.BACKEND_REGISTRY[NAME]` (the same callable
`run_vla_episode.py --backend NAME` calls), and the REAL `ActionSource`
`get_action()`. Only the inputs are synthetic: a 640x480 wrist frame, two
800x600 overhead halves (the rig's sizes, as in `mllm_smoke`) and an in-limits
joint state at the midpoint of every joint's range.

Lanes
-----
  - gRPC (act, diffusion, groot-n17, pi0, pi05, smolvla): TCP reachability of
    `--server-address`, then the real `connect_bridge()` handshake and
    `BridgeActionSource`, polled until `--steps` non-held actions arrive or
    `--wait-s` runs out.
  - HTTP (groot-n15, octo, openvla, openvla-oft, rdt1b, wallx): `GET /health`
    (every field is printed), then `--steps` `HttpAdapterActionSource` calls.
  - Gemini (gemini-robotics, gemini-robotics-2): `--steps` Gemini
    Robotics-ER calls through `GeminiActionSourceBase` and the ER conversion.
  - mllm-* backends belong to `python -m vla_bridge.mllm_smoke`; `scripted`
    has no server to smoke.

Hardware isolation
------------------
Everything runs inside `synthetic_hardware()`, which replaces every hardware
construction seam the three lanes use (`make_robot_from_config` inside
lerobot's `RobotClient`, the `SO101Follower` and `StereoSplitCamera` names in
the backend bases and `robot_client`, `cv2.VideoCapture`) with synthetic
doubles for the whole run. Tripwires make the real `SO101Follower.connect`
and the real `StereoSplitCamera.__init__` raise if they are reached anyway,
and the synthetic robot's `send_action()` raises. No serial port, robot or
camera is opened, and no action is ever sent.

Each step prints PASS/WARN/FAIL lines: server reachable, health ready (HTTP),
action returned and schema valid, not held, validator, saturation (FAIL when
all five arm joints sit at or beyond a limit: the 14-UAT tests 13/18
units-error symptom), latency and model_version; then `RESULT: PASS` or
`RESULT: FAIL`. The exit code is 0 only when no FAIL-class check failed.

SPEND WARNING (Gemini lane)
---------------------------
Every Gemini step is one paid Gemini Robotics-ER call billed to the key's
Google AI project. Without `--confirm-spend` the tool prints what it would
call and returns 2 before the API key is read or any client is built. The
key must be a RESTRICTED key (the API rejects unrestricted keys with 403); it
is read only by `base_gemini`, from the environment variable the backend
config names, and is never printed.

Usage::

    cd control
    .venv/bin/python -m vla_bridge.backend_smoke --backend smolvla \\
        --server-address 0.tcp.ngrok.io:12345 --checkpoint victorvanhalst/smolvla_so101_cube
    .venv/bin/python -m vla_bridge.backend_smoke --backend wallx \\
        --server-address https://<tunnel>.ngrok-free.app --instruction 'Put the object in the bin'
    export GEMINI_API_KEY=...            # restricted key, in this shell only
    .venv/bin/python -m vla_bridge.backend_smoke --backend gemini-robotics \\
        --instruction 'red cube' --confirm-spend

Add `2>&1 | grep -v objc` to hide the harmless cv2/av duplicate-class warnings.
"""

import argparse
import contextlib
import json
import math
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import requests

from vla_bridge import action_contract, robot_client, safety_validator
from vla_bridge.action_contract import JOINT_ORDER
from vla_bridge.backends import (
    BACKEND_REGISTRY,
    base_gemini,
    base_http_adapter,
    base_mllm,
    get_backend_names,
    load_backend_config,
)
from vla_bridge.mllm_smoke import NEUTRAL_JOINT_LIMITS_DEG, STEREO_SHAPE, WRIST_SHAPE, synthetic_frame

# --- CLI defaults ------------------------------------------------------------------------------

DEFAULT_INSTRUCTION = "Pick up the red cube"  # run_vla_episode's default
MAX_STEPS = 5
DEFAULT_WAIT_S = 60.0
DEFAULT_POLL_INTERVAL_S = 0.5
TCP_TIMEOUT_S = 5.0
HEALTH_TIMEOUT_S = 10.0
HEALTH_PROTOCOL_VERSION = 1
MAX_PRINTED_ER_TEXT_CHARS = 200

GEMINI_SPEND_WARNING = (
    "WARNING: --confirm-spend makes {steps} paid Gemini Robotics-ER API call(s), one per --steps, "
    "billed to the key's Google AI project. The key must be a RESTRICTED key (unrestricted keys get 403)."
)

# --- Lanes -------------------------------------------------------------------------------------

# Kept explicit (not inferred from module names) so the drift test catches a new backend.
GRPC_LANE = frozenset({"act", "diffusion", "groot-n17", "pi0", "pi05", "smolvla"})
HTTP_LANE = frozenset({"groot-n15", "octo", "openvla", "openvla-oft", "rdt1b", "wallx"})
GEMINI_LANE = frozenset({"gemini-robotics", "gemini-robotics-2"})
SMOKEABLE_BACKENDS = GRPC_LANE | HTTP_LANE | GEMINI_LANE

# backend name -> backend_configs/<stem>.json
CONFIG_NAMES = {
    "act": "act",
    "diffusion": "diffusion_policy",
    "groot-n17": "groot_n17",
    "pi0": "pi0",
    "pi05": "pi05",
    "smolvla": "smolvla",
    "groot-n15": "groot_n15",
    "octo": "octo",
    "openvla": "openvla",
    "openvla-oft": "openvla_oft",
    "rdt1b": "rdt1b",
    "wallx": "wallx",
    "gemini-robotics": "gemini_robotics",
    "gemini-robotics-2": "gemini_robotics_2",
}

# --- Check thresholds --------------------------------------------------------------------------

# An arm joint this close to (or beyond) a limit counts as pinned.
SATURATION_MARGIN_DEG = 2.0
# gRPC has no client-side request timeout; this is the latency budget there.
CLIENT_TIMEOUT_S = 5.0
LATENCY_WARN_FRACTION = 0.8
NOOP_TOLERANCE = 1e-6
# First-tick validation as run_episode does it at the default --control-hz 2.0.
FIRST_TICK_DT_S = 0.5
ARM_JOINTS = tuple(joint for joint in JOINT_ORDER if joint != "gripper")

_HELD_MARKERS = ("holding-position", "holding position", "held-position", "stale observation")
_VALIDATOR_FAIL_MARKERS = ("rejected non-finite", "stale observation")

# Modules whose hardware construction names synthetic_hardware() replaces. The
# seam-coverage test fails if a construction site appears anywhere else.
PATCHED_MODULES = (
    "vla_bridge.robot_client",
    "vla_bridge.backends.base_http_adapter",
    "vla_bridge.backends.base_gemini",
)


# --- Lane classification -----------------------------------------------------------------------


def lane_of(name: str) -> str:
    """`"grpc"`, `"http"` or `"gemini"`; `SystemExit` for anything this tool cannot smoke."""
    if name in GRPC_LANE:
        return "grpc"
    if name in HTTP_LANE:
        return "http"
    if name in GEMINI_LANE:
        return "gemini"
    if name in base_mllm.MLLM_BACKENDS:
        raise SystemExit(f"--backend {name} is an MLLM backend: use python -m vla_bridge.mllm_smoke instead.")
    if name == "scripted":
        raise SystemExit("--backend scripted has no server to smoke (it is the local dry-run backend).")
    if name in get_backend_names():
        raise SystemExit(f"--backend {name} is registered but not classified for backend_smoke; "
                         "add it to a lane (GRPC_LANE / HTTP_LANE / GEMINI_LANE).")
    raise SystemExit(f"--backend {name!r} is not a smoke-able backend "
                     f"(known: {', '.join(sorted(SMOKEABLE_BACKENDS))}).")


# --- Synthetic hardware ------------------------------------------------------------------------


@dataclass
class HardwareLog:
    """What the synthetic doubles saw. Both attempt lists must stay empty."""

    robots_built: list = field(default_factory=list)
    send_action_attempts: list = field(default_factory=list)
    tripwire_hits: list = field(default_factory=list)


def _wrist_frame() -> np.ndarray:
    return synthetic_frame(WRIST_SHAPE, (90, 160, 210), (40, 40, 220), (250, 200, 390, 300), "SYNTHETIC WRIST")


class SyntheticWristCap:
    """What `cv2.VideoCapture(idx)` returns during the smoke: one fixed wrist frame."""

    def __init__(self, index=None, *args, **kwargs):
        self.index = index
        self.frame = _wrist_frame()

    def read(self):
        return True, self.frame.copy()

    def grab(self):
        return True

    def isOpened(self):
        return True

    def release(self):
        pass


class SyntheticStereoCamera:
    """`StereoSplitCamera` stand-in: left/right views of one box with a small disparity."""

    def __init__(self, index=None, **kwargs):
        self.index = index
        self.left = synthetic_frame(STEREO_SHAPE, (150, 150, 150), (40, 200, 40), (330, 200, 470, 400),
                                    "SYNTHETIC LEFT")
        self.right = synthetic_frame(STEREO_SHAPE, (140, 140, 160), (40, 200, 40), (300, 200, 440, 400),
                                     "SYNTHETIC RIGHT")
        self.last_pair_metadata = {"pair_id": 0, "source": "synthetic"}

    def read_left(self):
        return self.left.copy()

    def read_right(self):
        return self.right.copy()

    def release(self):
        pass


class _SyntheticBus:
    """No-op stand-in for `SO101Follower.bus` (mirrors conftest's `_MockBus`)."""

    @contextlib.contextmanager
    def torque_disabled(self):
        yield

    def write_calibration(self, calibration) -> None:
        pass


class SyntheticRobot:
    """Robot double for every lane: built from `RobotClient`'s robot config (gRPC,
    via `make_robot_from_config`) or `SOFollowerRobotConfig` (HTTP/Gemini, via the
    `SO101Follower` name). It serves a fixed joint state and synthetic frames for
    the configured cameras, and refuses to send anything."""

    def __init__(self, config=None, joint_state=None, hardware_log=None):
        self.config = config
        self.joint_state = dict(joint_state) if joint_state is not None else dict.fromkeys(JOINT_ORDER, 0.0)
        self.hardware_log = hardware_log if hardware_log is not None else HardwareLog()
        self.camera_names = list(getattr(config, "cameras", None) or {})
        self.calibration = {}
        self.bus = _SyntheticBus()
        self.is_connected = False
        self._frame = _wrist_frame()
        self.hardware_log.robots_built.append(self)

    @property
    def observation_features(self) -> dict:
        features = {f"{joint}.pos": float for joint in JOINT_ORDER}
        for name in self.camera_names:
            features[name] = (*WRIST_SHAPE, 3)
        return features

    @property
    def action_features(self) -> dict:
        return {f"{joint}.pos": float for joint in JOINT_ORDER}

    def connect(self, calibrate: bool = True) -> None:
        self.is_connected = True

    def disconnect(self) -> None:
        self.is_connected = False

    def get_observation(self) -> dict:
        obs = {f"{joint}.pos": float(self.joint_state[joint]) for joint in JOINT_ORDER}
        for name in self.camera_names:
            obs[name] = self._frame.copy()
        return obs

    def send_action(self, action):
        self.hardware_log.send_action_attempts.append(action)
        raise RuntimeError("backend_smoke never sends actions")


@contextlib.contextmanager
def synthetic_hardware(joint_state: dict):
    """Replace every hardware construction seam with synthetic doubles; yield a `HardwareLog`.

    Everything is restored on exit, including exit by exception.
    """
    import lerobot.async_inference.robot_client as lerobot_robot_client
    from lerobot.robots.so_follower.so_follower import SO101Follower as RealSO101Follower

    from vla_bridge.stereo_camera import StereoSplitCamera as RealStereoSplitCamera

    log = HardwareLog()

    def make_robot(config=None, *args, **kwargs):
        return SyntheticRobot(config, joint_state=joint_state, hardware_log=log)

    def follower_tripwire(self, *args, **kwargs):
        log.tripwire_hits.append("SO101Follower.connect")
        raise RuntimeError("backend_smoke tripwire: the real SO101Follower.connect was reached")

    def stereo_tripwire(self, *args, **kwargs):
        log.tripwire_hits.append("StereoSplitCamera.__init__")
        raise RuntimeError("backend_smoke tripwire: the real StereoSplitCamera was constructed")

    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.object(lerobot_robot_client, "make_robot_from_config", make_robot))
        for module in (robot_client, base_http_adapter, base_gemini):
            stack.enter_context(mock.patch.object(module, "StereoSplitCamera", SyntheticStereoCamera))
        for module in (base_http_adapter, base_gemini):
            stack.enter_context(mock.patch.object(module, "SO101Follower", make_robot))
        stack.enter_context(mock.patch.object(cv2, "VideoCapture", SyntheticWristCap))
        stack.enter_context(mock.patch.object(RealSO101Follower, "connect", follower_tripwire))
        stack.enter_context(mock.patch.object(RealStereoSplitCamera, "__init__", stereo_tripwire))
        yield log


# --- Per-step checks ---------------------------------------------------------------------------


@dataclass
class Check:
    level: str  # "PASS" | "WARN" | "FAIL"
    name: str
    detail: str


def format_check(check: Check) -> str:
    return f"{check.level} {check.name}: {check.detail}"


def _is_finite_joint_action(action) -> bool:
    return base_http_adapter._is_joint_action(action) and all(
        math.isfinite(float(action[joint])) for joint in JOINT_ORDER
    )


def _bridge_flags(details) -> list[str]:
    flags = (details or {}).get("bridge_flags") or []
    return [str(flag) for flag in flags]


def hold_reason(action, model_version, details, joint_state) -> str | None:
    """Why this step is a hold, or `None` when the action is a genuine move."""
    version = str(model_version or "")
    if any(marker in version for marker in ("holding-position", "holding position")):
        return f"model_version {version!r}"
    flags = _bridge_flags(details)
    held_flags = [f for f in flags if any(marker in f for marker in _HELD_MARKERS)]
    if held_flags or robot_client._is_stale_or_empty(flags):
        return f"bridge flags {held_flags or flags}"
    if all(abs(float(action[joint]) - float(joint_state[joint])) <= NOOP_TOLERANCE for joint in JOINT_ORDER):
        return "no-op: the action equals the current joint state"
    return None


def classify_validator_flags(flags: list[str], safe_action: dict) -> Check:
    missing = [joint for joint in JOINT_ORDER if joint not in safe_action]
    failing = [f for f in flags if any(marker in f for marker in _VALIDATOR_FAIL_MARKERS)]
    if missing or failing:
        detail = f"missing joints {missing}" if missing else "; ".join(failing)
        return Check("FAIL", "validator", detail)
    if flags:
        return Check("WARN", "validator", "accepted with clamps: " + "; ".join(flags))
    return Check("PASS", "validator", "safety_validator.validate_action accepts it unchanged")


def _saturation_candidate(action, details) -> tuple[dict, str]:
    raw = (details or {}).get("raw_model_output")
    if _is_finite_joint_action(raw):
        return raw, "pre-validation model output"
    return action, "returned action"


def check_saturation(action, details, joint_limits_deg) -> Check:
    candidate, source = _saturation_candidate(action, details)
    pinned = []
    for joint in ARM_JOINTS:
        lo, hi = joint_limits_deg[joint]
        value = float(candidate[joint])
        if value <= lo + SATURATION_MARGIN_DEG or value >= hi - SATURATION_MARGIN_DEG:
            pinned.append(f"{joint}={value:.2f} (limit {lo:.2f}/{hi:.2f})")
    if len(pinned) == len(ARM_JOINTS):
        return Check("FAIL", "saturated",
                     f"every arm joint pinned at a limit: units error? (14-UAT tests 13/18); {source}: "
                     + ", ".join(pinned))
    if pinned:
        return Check("WARN", "near joint limits", f"{source}: " + ", ".join(pinned))
    return Check("PASS", "not saturated",
                 f"{source}: no arm joint within {SATURATION_MARGIN_DEG} deg of a limit")


def check_latency(lane: str, latency_s: float, client_timeout_s: float | None) -> Check:
    if lane == "grpc":
        if latency_s > safety_validator.STALE_ACTION_S:
            return Check("FAIL", "latency", f"{latency_s:.2f} s > STALE_ACTION_S "
                         f"{safety_validator.STALE_ACTION_S} s: the validator would hold this action as stale")
        if latency_s > CLIENT_TIMEOUT_S:
            return Check("WARN", "latency", f"{latency_s:.2f} s over the {CLIENT_TIMEOUT_S} s budget")
        return Check("PASS", "latency", f"{latency_s:.2f} s (budget {CLIENT_TIMEOUT_S} s)")
    timeout = float(client_timeout_s if client_timeout_s is not None else CLIENT_TIMEOUT_S)
    if latency_s >= timeout:
        return Check("FAIL", "latency", f"{latency_s:.2f} s at or over the {timeout} s client timeout")
    if latency_s >= LATENCY_WARN_FRACTION * timeout:
        return Check("WARN", "latency", f"{latency_s:.2f} s is over {int(LATENCY_WARN_FRACTION * 100)}% "
                     f"of the {timeout} s client timeout")
    return Check("PASS", "latency", f"{latency_s:.2f} s (client timeout {timeout} s)")


def check_model_version(lane: str, model_version, checkpoint) -> Check:
    version = str(model_version or "")
    if lane == "grpc":
        if checkpoint and version.startswith(f"{checkpoint}@"):
            return Check("PASS", "model_version", version)
        return Check("WARN", "model_version", f"{version!r} does not start with '{checkpoint}@'")
    if lane == "gemini":
        if version.endswith(f"@{base_gemini.CONVERSION_SUFFIX}"):
            return Check("PASS", "model_version", version)
        return Check("WARN", "model_version", f"{version!r} is not an @{base_gemini.CONVERSION_SUFFIX} action")
    if version.endswith("@unknown"):
        return Check("WARN", "model_version", f"server sent no model_version ({version})")
    if any(marker in version for marker in ("holding-position", "holding position")):
        return Check("WARN", "model_version", f"hold tag, not a server model_version ({version})")
    return Check("PASS", "model_version", version)


def evaluate_step(lane, action, model_version, details, joint_state, joint_limits_deg,
                  latency_s, client_timeout_s, checkpoint) -> list[Check]:
    """Every check for one `get_action()` result, in report order. Pure: no I/O."""
    checks = []
    if _is_finite_joint_action(action):
        checks.append(Check("PASS", "action returned", f"schema valid: {len(JOINT_ORDER)} finite JOINT_ORDER values"))
        reason = hold_reason(action, model_version, details, joint_state)
        if reason:
            checks.append(Check("FAIL", "not held", f"held: {reason}"))
        else:
            checks.append(Check("PASS", "not held", "the action moves at least one joint"))
        safe_action, flags = safety_validator.validate_action(
            dict(action), joint_state, joint_limits_deg, obs_age_s=0.0, prev_action=None, dt_s=FIRST_TICK_DT_S
        )
        checks.append(classify_validator_flags(flags, safe_action))
        checks.append(check_saturation(action, details, joint_limits_deg))
    else:
        checks.append(Check("FAIL", "action returned",
                            f"not a dict of finite numbers for every JOINT_ORDER key: {action!r}"[:300]))
    checks.append(check_latency(lane, latency_s, client_timeout_s))
    checks.append(check_model_version(lane, model_version, checkpoint))
    return checks


# --- Pre-flight --------------------------------------------------------------------------------


class _Report:
    """Prints checks and remembers whether any FAIL was printed."""

    def __init__(self):
        self.failed = False

    def emit(self, check: Check) -> None:
        print(format_check(check))
        if check.level == "FAIL":
            self.failed = True


def _json(value) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _load_limits(path: Path) -> dict:
    try:
        return action_contract.load_joint_limits_deg(Path(path))
    except (FileNotFoundError, OSError, KeyError, TypeError, ValueError) as exc:
        print(f"WARNING: no usable follower calibration at {path} ({type(exc).__name__}); using neutral "
              "joint limits (-90 to 90 degrees for the arm joints, 0 to 100 for the gripper). The "
              "saturation check is then only indicative; no robot is driven.")
        return dict(NEUTRAL_JOINT_LIMITS_DEG)


def _parse_host_port(address: str) -> tuple[str, int]:
    host, sep, port = str(address).rpartition(":")
    if "://" in str(address) or not sep or not host or not port.isdigit():
        raise ValueError(f"{address!r} is not host:port")
    return host, int(port)


def check_tcp(host: str, port: int, timeout_s: float = TCP_TIMEOUT_S) -> Check:
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            pass
    except OSError as exc:
        return Check("FAIL", "server reachable", f"TCP {host}:{port}: {type(exc).__name__}: {exc}")
    return Check("PASS", "server reachable", f"TCP {host}:{port} accepted a connection")


def check_health(endpoint: str, report: _Report) -> bool:
    """`GET <endpoint>/health`: emit reachability + readiness, print every field. True when ready."""
    url = f"{endpoint.rstrip('/')}/health"
    try:
        response = requests.get(url, timeout=HEALTH_TIMEOUT_S)
        response.raise_for_status()
    except requests.RequestException as exc:
        report.emit(Check("FAIL", "server reachable", f"GET {url}: {type(exc).__name__}: {exc}"))
        return False
    report.emit(Check("PASS", "server reachable", f"GET {url} -> {response.status_code}"))
    try:
        body = response.json()
    except ValueError:
        body = None
    if not isinstance(body, dict):
        report.emit(Check("FAIL", "health ready", f"/health body is not a JSON object: {body!r}"[:300]))
        return False
    for key in sorted(body):
        value = body[key]
        if key == "model" and isinstance(value, dict):
            for model_key in sorted(value):
                print(f"health model.{model_key}: {_json(value[model_key])}")
        else:
            print(f"health {key}: {_json(value)}")
    status, protocol = body.get("status"), body.get("protocol_version")
    if status == "ready" and protocol == HEALTH_PROTOCOL_VERSION:
        report.emit(Check("PASS", "health ready", f"status ready, protocol_version {protocol}"))
        return True
    report.emit(Check("FAIL", "health ready", f"status {status!r}, protocol_version {protocol!r} "
                      f"(expected 'ready' and {HEALTH_PROTOCOL_VERSION}); not POSTing /action"))
    return False


# --- Lane runners ------------------------------------------------------------------------------


@dataclass
class StepResult:
    index: int
    action: object
    model_version: str
    details: dict
    latency_s: float
    waited_s: float | None = None  # gRPC: time since the first poll (or the previous action)


def _details_of(source) -> dict:
    details = getattr(source, "last_details", None)
    return dict(details) if isinstance(details, dict) else {}


def run_direct_steps(source, joint_state: dict, instruction: str, steps: int) -> list[StepResult]:
    """HTTP / Gemini lanes: `steps` timed `get_action()` calls (each one real inference)."""
    results = []
    for index in range(1, steps + 1):
        start = time.perf_counter()
        action, model_version = source.get_action(dict(joint_state), instruction)
        latency = time.perf_counter() - start
        results.append(StepResult(index, action, model_version, _details_of(source), latency))
    return results


def run_grpc_steps(source, joint_state: dict, instruction: str, steps: int, wait_s: float,
                   poll_interval_s: float) -> tuple[list[StepResult], list[str]]:
    """gRPC lane: poll until `steps` non-held actions arrive or `wait_s` elapses.

    Latency is the action's observation-to-action age (`last_details["action_age_s"]`,
    the same number the validator compares with STALE_ACTION_S), so a cold
    server's warm-up wait does not count against it; `waited_s` (time from
    the first poll, or the previous collected action) is reported alongside.
    Returns the collected steps and the last held step's flags.
    """
    results, last_flags = [], []
    start = previous = time.perf_counter()
    deadline = start + wait_s
    while len(results) < steps:
        action, model_version = source.get_action(dict(joint_state), instruction)
        details = _details_of(source)
        now = time.perf_counter()
        if not _is_finite_joint_action(action) or hold_reason(action, model_version, details, joint_state) is None:
            waited = now - previous
            age = details.get("action_age_s")
            usable_age = isinstance(age, (int, float)) and not isinstance(age, bool) and math.isfinite(age) and age >= 0
            latency = float(age) if usable_age else waited
            results.append(StepResult(len(results) + 1, action, model_version, details, latency, waited))
            previous = now
            continue
        last_flags = _bridge_flags(details) or [str(model_version)]
        if now >= deadline:
            break
        time.sleep(poll_interval_s)
    return results, last_flags


def _rounded(value):
    if isinstance(value, dict):
        return {key: (round(float(v), 2) if isinstance(v, (int, float)) and not isinstance(v, bool) else v)
                for key, v in value.items()}
    return value


def _print_gemini_details(details: dict) -> None:
    print(f"  er_model: {details.get('er_model')}")
    print(f"  action_path: {details.get('action_path')}")
    raw = details.get("raw_model_output")
    er_text = raw.get("er_text") if isinstance(raw, dict) else None
    if er_text is not None:
        print(f"  er_text: {str(er_text)[:MAX_PRINTED_ER_TEXT_CHARS]}")
    if details.get("error_type") or details.get("error"):
        # Already key-redacted by base_gemini._hold().
        print(f"  error: {details.get('error_type')}: {details.get('error')}")


def report_steps(lane, results, joint_state, joint_limits_deg, client_timeout_s, checkpoint,
                 report: _Report) -> None:
    held = [
        _is_finite_joint_action(r.action) and hold_reason(r.action, r.model_version, r.details, joint_state)
        for r in results
    ]
    any_moving = any(not h for h in held)
    for result in results:
        print(f"step {result.index}: action {_rounded(result.action)}")
        print(f"  joint_state {_rounded(joint_state)}")
        print(f"  bridge flags: {_bridge_flags(result.details)}")
        print(f"  model_version: {result.model_version}")
        if result.waited_s is not None:
            print(f"  waited {result.waited_s:.2f} s for this action; observation-to-action age "
                  f"{result.latency_s:.2f} s")
        if lane == "gemini":
            _print_gemini_details(result.details)
        for check in evaluate_step(lane, result.action, result.model_version, result.details, joint_state,
                                   joint_limits_deg, result.latency_s, client_timeout_s, checkpoint):
            if check.name == "not held" and check.level == "FAIL" and any_moving:
                # The UAT criterion is at least one validated non-held action.
                check = Check("WARN", check.name, check.detail + " (another step moved)")
            report.emit(check)


def _shutdown(handles) -> None:
    """Mirror of run_vla_episode._shutdown_backend: each step runs even if one raises."""
    if handles is None:
        return
    try:
        if handles.shutdown is not None:
            handles.shutdown()
        elif handles.robot is not None:
            handles.robot.disconnect()
    finally:
        try:
            close = getattr(handles.action_source, "close", None)
            if callable(close):
                close()
        finally:
            stereo = handles.stereo_camera
            if stereo is not None and hasattr(stereo, "release"):
                stereo.release()


# --- CLI ---------------------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Smoke one Phase 14 VLA backend against its live server with synthetic inputs "
        "(no robot, serial port or camera is opened; no action is sent)."
    )
    parser.add_argument("--backend", required=True, help="A gRPC, HTTP or Gemini backend, e.g. smolvla")
    parser.add_argument("--server-address", default=None,
                        help="gRPC: host:port of the PolicyServer tunnel. HTTP: the adapter's http(s):// "
                        "URL (default: the backend config's endpoint_url). Ignored for Gemini.")
    parser.add_argument("--checkpoint", default=None,
                        help="HF checkpoint id (required for gRPC; HTTP defaults to the backend config's)")
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION,
                        help=f"Task prompt (default {DEFAULT_INSTRUCTION!r}, run_vla_episode's default)")
    parser.add_argument("--steps", type=int, default=1,
                        help=f"1..{MAX_STEPS}. HTTP/Gemini: get_action calls (each a real inference; on "
                        "Gemini each is a PAID call). gRPC: non-held actions to collect.")
    parser.add_argument("--wait-s", type=float, default=DEFAULT_WAIT_S,
                        help=f"gRPC: seconds to wait for the first action chunk (default {DEFAULT_WAIT_S:g})")
    parser.add_argument("--poll-interval-s", type=float, default=DEFAULT_POLL_INTERVAL_S,
                        help=f"gRPC: poll period in seconds (default {DEFAULT_POLL_INTERVAL_S:g})")
    parser.add_argument("--camera-map", default=None,
                        help="gRPC only: name=source,... exactly as run_vla_episode --camera-map")
    parser.add_argument("--joint-calibration", type=Path, default=action_contract.CALIBRATION_PATH,
                        help="Follower calibration JSON for the joint limits (default: "
                        "action_contract.CALIBRATION_PATH; neutral limits if missing)")
    parser.add_argument("--confirm-spend", action="store_true",
                        help="Gemini only: actually make the paid calls (otherwise print the plan, exit 2)")
    return parser


def main(argv=None) -> int:
    """Smoke one backend; see the module docstring. 0 = PASS, 1 = FAIL, 2 = Gemini spend refused."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    name = args.backend
    lane = lane_of(name)  # unknown / mllm-* / scripted exit here, before anything else

    if not 1 <= args.steps <= MAX_STEPS:
        parser.error(f"--steps must be between 1 and {MAX_STEPS}")
    if args.wait_s <= 0 or args.poll_interval_s <= 0:
        parser.error("--wait-s and --poll-interval-s must be positive")
    camera_map = None
    if args.camera_map is not None:
        if lane != "grpc":
            parser.error("--camera-map applies to gRPC-bridge backends only")
        try:
            camera_map = robot_client.parse_camera_map(args.camera_map)
        except ValueError as exc:
            parser.error(str(exc))

    config = load_backend_config(CONFIG_NAMES[name]) or {}
    server_address, checkpoint, endpoint = None, None, None
    if lane == "grpc":
        if not args.server_address or not args.checkpoint:
            parser.error(f"--backend {name} (gRPC bridge) needs both --server-address and --checkpoint")
        try:
            host, port = _parse_host_port(args.server_address)
        except ValueError as exc:
            parser.error(f"--server-address for a gRPC backend must be host:port without a scheme ({exc})")
        server_address, checkpoint = args.server_address, args.checkpoint
    elif lane == "http":
        endpoint = args.server_address or config.get("endpoint_url")
        if not endpoint or not str(endpoint).startswith(("http://", "https://")):
            parser.error(f"--backend {name} needs an http(s):// --server-address (the {name} server's "
                         f"tunnel URL); got {endpoint!r}")
        server_address, checkpoint = endpoint, args.checkpoint or config.get("checkpoint")

    joint_limits_deg = _load_limits(args.joint_calibration)
    joint_state = {joint: (float(lo) + float(hi)) / 2 for joint, (lo, hi) in joint_limits_deg.items()}

    print(f"Backend smoke: backend {name}, lane {lane}")
    if lane == "grpc":
        source = "--camera-map" if camera_map else ("backend config" if config.get("camera_map") else "default")
        resolved = camera_map or config.get("camera_map") or robot_client.DEFAULT_CAMERA_MAP
        print(f"  server: {server_address}  checkpoint: {checkpoint}")
        print(f"  camera_map ({source}): {_json(resolved)}")
    elif lane == "http":
        print(f"  endpoint: {endpoint}  checkpoint: {checkpoint}")
    else:
        model, api_key_env, source_kwargs = base_gemini.load_er_backend_config(CONFIG_NAMES[name], name)
        print(f"  model: {model}  api_key_env: {api_key_env} (name only; the value is never printed)  "
              f"settings: {_json(source_kwargs)}")
        if args.server_address or args.checkpoint:
            print("  note: --server-address/--checkpoint are ignored on the Gemini lane")
    print(f"  steps: {args.steps}  instruction: {args.instruction!r}")
    print("  inputs: synthetic wrist/overhead_left/overhead_right images and a synthetic joint state; "
          "no robot, serial port or camera is opened and no action is sent")
    if lane != "gemini" and args.confirm_spend:
        print("  note: --confirm-spend only applies to the Gemini lane; ignored")

    report = _Report()
    # Pre-flight, before building anything.
    if lane == "grpc":
        check = check_tcp(host, port)
        report.emit(check)
        if check.level == "FAIL":
            print("RESULT: FAIL")
            return 1
    elif lane == "http":
        if not check_health(endpoint, report):
            print("RESULT: FAIL")
            return 1
    else:
        print(GEMINI_SPEND_WARNING.format(steps=args.steps))
        if not args.confirm_spend:
            print("Not calling: re-run with --confirm-spend to make the paid call(s).")
            return 2

    factory_args = argparse.Namespace(
        port="synthetic-no-serial",
        robot_id="backend_smoke",
        stereo_camera_index="synthetic",
        server_address=server_address,
        checkpoint=checkpoint if lane != "gemini" else None,
        instruction=args.instruction,
        camera_map=camera_map,
    )
    camera_names = {0: "wrist"}  # a synthetic index: the wrist double never opens anything

    with synthetic_hardware(joint_state) as hw:
        handles = None
        try:
            built = False
            try:
                handles = BACKEND_REGISTRY[name](factory_args, camera_names, joint_limits_deg)
                built = True
            except SystemExit as exc:
                report.emit(Check("FAIL", "build", str(exc)))
            except Exception as exc:  # noqa: BLE001 -- report, then tear down
                detail = type(exc).__name__ if lane == "gemini" else f"{type(exc).__name__}: {exc}"
                report.emit(Check("FAIL", "build", detail))
            if built and handles is None:
                report.emit(Check("FAIL", "bridge handshake", "PolicyServer refused or unreachable "
                                  "(check the Colab log for 'Policy type: ...')"))
            elif handles is not None:
                source = handles.action_source
                if lane == "grpc":
                    results, last_flags = run_grpc_steps(source, joint_state, args.instruction, args.steps,
                                                         args.wait_s, args.poll_interval_s)
                    client_timeout_s = None
                else:
                    results = run_direct_steps(source, joint_state, args.instruction, args.steps)
                    client_timeout_s = getattr(source, "timeout_s", None)
                report_steps(lane, results, joint_state, joint_limits_deg, client_timeout_s,
                             checkpoint if lane == "grpc" else None, report)
                if lane == "grpc" and not results:
                    report.emit(Check("FAIL", "not held", f"no action chunk within {args.wait_s:g} s "
                                      f"(bridge flags: {last_flags})"))
                elif lane == "grpc" and len(results) < args.steps:
                    report.emit(Check("WARN", "steps", f"only {len(results)} of {args.steps} actions "
                                      f"arrived within {args.wait_s:g} s"))
        finally:
            _shutdown(handles)
        if hw.send_action_attempts or hw.tripwire_hits:
            report.emit(Check("FAIL", "hardware isolation",
                              f"send_action attempts {len(hw.send_action_attempts)}, "
                              f"tripwires {hw.tripwire_hits}"))
        else:
            report.emit(Check("PASS", "hardware isolation",
                              f"no send_action and no real device construction "
                              f"({len(hw.robots_built)} synthetic robot(s))"))

    print(f"RESULT: {'FAIL' if report.failed else 'PASS'}")
    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARM_JOINTS",
    "CLIENT_TIMEOUT_S",
    "CONFIG_NAMES",
    "Check",
    "GEMINI_LANE",
    "GRPC_LANE",
    "HTTP_LANE",
    "HardwareLog",
    "PATCHED_MODULES",
    "SATURATION_MARGIN_DEG",
    "SMOKEABLE_BACKENDS",
    "SyntheticRobot",
    "SyntheticStereoCamera",
    "SyntheticWristCap",
    "check_latency",
    "check_model_version",
    "check_saturation",
    "classify_validator_flags",
    "evaluate_step",
    "format_check",
    "hold_reason",
    "lane_of",
    "synthetic_hardware",
]
