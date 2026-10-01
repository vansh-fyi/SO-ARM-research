"""Shared client side of the Tier C resident-HTTP-adapter lane (14-CONTEXT.md D-01/D-04).

Tier C backends (OpenVLA, OpenVLA-OFT, Octo, RDT-1B, GR00T N1.5, and any
Tier B fallback) have no LeRobot `PolicyServer` integration, so each runs a
small Colab-hosted Flask adapter (`<backend>_server.py`, modelled on
`fastfs_server.py`). This module is the one control-loop-side client every
such backend reuses.

Wire contract (what every `<backend>_server.py` must implement):

  POST {endpoint_url}/action, JSON body:
    {"images": {"wrist": <b64 PNG>, "overhead_left": <b64 PNG>,
                "overhead_right": <b64 PNG>},
     "joint_state": {<JOINT_ORDER name>: <degrees / gripper percent>},
     "instruction": "<task prompt>"}
  Images are OpenCV BGR arrays encoded with `cv2.imencode(".png", ...)`, so
  they decode back to true colors (`cv2.imdecode` -> BGR, `PIL.Image.open`
  -> RGB). Each frame's longer edge is first shrunk to
  `robot_client.OBSERVATION_MAX_SIDE` (640) -- every Tier C model's native
  input is far smaller, and the live gRPC lane already showed full-res
  frames stall the loop for minutes over a tunnel.

  200 response JSON: {"action": <list or dict, model-native format>,
                      "model_version": "<string>" (optional)}

`convert_fn(parsed_action, joint_state)` turns the model-native `"action"`
(e.g. a 7-float delta-EEF list) into this project's absolute
`action_contract.JOINT_ORDER`-keyed dict. With no `convert_fn`, the server
must already return that dict. The FINAL action is checked to be a dict
with every `JOINT_ORDER` key mapped to a real number before it is returned:
`safety_validator.validate_action()` would raise `TypeError` on a
non-numeric value (`math.isfinite`), and silently drops missing joints,
producing the partial `goal_pos` lerobot rejects. NaN/inf ARE passed
through -- the validator's own non-finite rejection handles those.

`HttpAdapterActionSource.get_action()` never raises -- every camera,
transport, parse, or conversion failure degrades to
`(dict(joint_state), "<checkpoint>@<reason>-holding-position")`, matching
`BridgeActionSource`'s fail-safe contract and `depth_camera.py`'s
"a flaky Colab connection must not crash the live control loop" rule.
Its output still flows through `run_episode()`'s unmodified
`safety_validator.validate_action()` before any `send_action()` (T-14-01).

Depth is untouched here (D-03).
"""

import base64
import numbers
import threading

import cv2
import requests
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SO101Follower

from vla_bridge import safety_validator
from vla_bridge.action_contract import JOINT_ORDER
from vla_bridge.backends import BackendHandles
from vla_bridge.robot_client import OBSERVATION_MAX_SIDE, _downsize_for_transport
from vla_bridge.stereo_camera import StereoSplitCamera


def _encode_png(frame) -> str | None:
    """Same encoding as `DepthCameraClient._encode_png`; `None` on failure."""
    ok, buf = cv2.imencode(".png", frame)
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode("ascii")


def capture_observation_images(wrist_cap, stereo_camera, max_side: int = OBSERVATION_MAX_SIDE) -> dict | None:
    """Read one wrist frame and ONE stereo pair (one physical AR0144 read
    shared by `read_left()`/`read_right()`), downsize, and base64-PNG-encode.

    Returns `{"wrist", "overhead_left", "overhead_right"}` -> b64 strings,
    or `None` if any read/encode failed. Never raises on a failed read.
    Shared with Plan 14-13's Gemini base so both lanes capture identically.
    """
    ok, wrist = wrist_cap.read()
    left = stereo_camera.read_left()
    right = stereo_camera.read_right()
    if not ok or wrist is None or left is None or right is None:
        return None
    images = {}
    for key, frame in (("wrist", wrist), ("overhead_left", left), ("overhead_right", right)):
        encoded = _encode_png(_downsize_for_transport(frame, max_side))
        if encoded is None:
            return None
        images[key] = encoded
    return images


def _is_joint_action(action) -> bool:
    if not isinstance(action, dict):
        return False
    for joint in JOINT_ORDER:
        value = action.get(joint)
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            return False
    return True


class HttpAdapterActionSource:
    """`run_vla_episode.ActionSource` backed by a Colab-hosted HTTP adapter."""

    def __init__(
        self,
        endpoint_url: str,
        checkpoint: str,
        wrist_cap,
        stereo_camera,
        convert_fn=None,
        timeout_s: float = 5.0,
        max_side: int = OBSERVATION_MAX_SIDE,
    ):
        self.endpoint_url = endpoint_url
        self.checkpoint = checkpoint
        self.wrist_cap = wrist_cap
        self.stereo_camera = stereo_camera
        self.convert_fn = convert_fn
        self.timeout_s = timeout_s
        self.max_side = max_side
        self._action_url = f"{endpoint_url.rstrip('/')}/action"
        self._inflight_lock = threading.Lock()
        self.last_details = {}

    def _hold(self, joint_state, reason: str):
        self.last_details["bridge_flags"] = [reason]
        return dict(joint_state), f"{self.checkpoint}@{reason}"

    def get_action(self, joint_state: dict[str, float], instruction: str) -> tuple[dict[str, float], str]:
        self.last_details = dict(raw_model_output={}, bridge_flags=[], observation=None)
        # Same defensive non-blocking in-flight guard as BridgeActionSource (LATENCY-04).
        if not self._inflight_lock.acquire(blocking=False):
            return self._hold(joint_state, "http-request-inflight-holding-position")
        try:
            images = capture_observation_images(self.wrist_cap, self.stereo_camera, self.max_side)
            if images is None:
                return self._hold(joint_state, "camera-read-failed-holding-position")

            payload = {"images": images, "joint_state": dict(joint_state), "instruction": instruction}
            try:
                response = requests.post(self._action_url, json=payload, timeout=self.timeout_s)
                response.raise_for_status()
                body = response.json()
            except (requests.RequestException, ValueError):
                return self._hold(joint_state, "http-error-holding-position")
            if not isinstance(body, dict) or body.get("action") is None:
                return self._hold(joint_state, "http-error-holding-position")

            parsed_action = body["action"]
            self.last_details = dict(raw_model_output=parsed_action, bridge_flags=[], observation=None)
            if self.convert_fn is not None:
                try:
                    converted_action = self.convert_fn(parsed_action, joint_state)
                except Exception:  # never let a conversion bug crash the live loop
                    return self._hold(joint_state, "action-conversion-error-holding-position")
            else:
                converted_action = parsed_action
            if not _is_joint_action(converted_action):
                return self._hold(joint_state, "action-format-error-holding-position")

            model_version = body.get("model_version")
            if not isinstance(model_version, str) or not model_version:
                model_version = f"{self.checkpoint}@unknown"
            return converted_action, model_version
        finally:
            self._inflight_lock.release()

    def close(self) -> None:
        """Release this source's dedicated wrist capture (the stereo camera is
        released by the caller via `BackendHandles.stereo_camera`)."""
        if self.wrist_cap is not None and hasattr(self.wrist_cap, "release"):
            self.wrist_cap.release()


def build_http_adapter_backend(
    endpoint_url: str,
    checkpoint: str,
    args,
    camera_names: dict[int, str],
    convert_fn=None,
) -> BackendHandles:
    """Connect the robot directly (no `RobotClient` on this lane) and build
    an `HttpAdapterActionSource` with its own camera opens.

    Robot construction mirrors `run_vla_episode.py`'s pre-Phase-14 scripted
    path exactly. The `StereoSplitCamera` is this lane's own open of the
    AR0144 (there is no `connect_bridge()` to share one with) and is
    returned as `BackendHandles.stereo_camera` so `run_episode()` reuses it
    rather than opening the device a second time. The wrist gets a
    dedicated `cv2.VideoCapture` -- an independent second open of the index
    `run_vla_episode.py` also opens for IOLogger, the same already-accepted
    precedent documented in `connect_bridge()` (must be re-verified on real
    hardware). `args` needs `.port`, `.robot_id`, `.stereo_camera_index`.

    Raises `SystemExit` before touching the robot if no camera is named
    "wrist" -- every Tier C request needs a wrist frame.
    """
    wrist_idx = next((idx for idx, name in camera_names.items() if name == "wrist"), None)
    if wrist_idx is None:
        raise SystemExit(
            f"HTTP-adapter backend for {checkpoint!r} needs a wrist camera "
            "(first --camera index / device_map.json wrist entry)."
        )

    robot = SO101Follower(
        SOFollowerRobotConfig(
            port=args.port,
            id=args.robot_id,
            max_relative_target=safety_validator.MAX_RELATIVE_TARGET_DEG,
        )
    )
    robot.connect(calibrate=False)
    with robot.bus.torque_disabled():
        robot.bus.write_calibration(robot.calibration)

    stereo_camera = StereoSplitCamera(index=args.stereo_camera_index)
    wrist_cap = cv2.VideoCapture(wrist_idx)
    if hasattr(wrist_cap, "isOpened") and not wrist_cap.isOpened():
        print(f"WARNING: wrist camera index {wrist_idx} did not open; every step will hold position.")

    return BackendHandles(
        action_source=HttpAdapterActionSource(
            endpoint_url, checkpoint, wrist_cap, stereo_camera, convert_fn=convert_fn
        ),
        robot=robot,
        stereo_camera=stereo_camera,
        skip_camera_names=frozenset({"overhead"}),
    )


__all__ = ["HttpAdapterActionSource", "build_http_adapter_backend", "capture_observation_images"]
