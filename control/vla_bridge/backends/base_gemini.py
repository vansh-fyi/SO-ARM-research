"""Shared base for the Gemini-family backends (Plan 14-13; VLAB-01, VLAB-02).

What this is, and what it is NOT
================================
This is NOT a native Gemini Robotics VLA policy. The Gemini Robotics and
Gemini Robotics 2 VLA models, which output robot actions directly, are open
only to early-access partners. Per 14-CONTEXT.md D-02's 2026-10-01
supersession, both Gemini backends are instead built on the PUBLIC
**Gemini Robotics-ER 2** model. "ER" means embodied reasoning: it returns
points, boxes and trajectories as JSON text, NOT joint or motor commands.

A local conversion layer turns that spatial output into a small, bounded
joint-space action: `convert_fn`, normally
`er_conversion.convert_er_output_to_action`. So these backends differ
architecturally from the other backends' trained-action-output path, and this
module never presents them as the same:

  - The `model_version` of every converted action is
    `"<checkpoint_label>@gemini-er-conversion"`. Every hold is
    `"<checkpoint_label>@<reason>-holding-position"`. No Gemini step ever
    carries a bare policy label or `@unknown`.
  - Every `last_details` (and therefore every `episode.jsonl`
    `action_details`) carries `action_path` = `ACTION_PATH`, which states the
    ER + conversion path in words.

Transport (D-02): direct HTTPS from the control-loop machine to Google's
Gemini API, via the official `google-genai` SDK (`google-genai==2.25.0`,
pinned in control/requirements.txt). There is no Colab hop and no ngrok tunnel.

API shape (verified 2026-10-01; not guessed)
--------------------------------------------
Sources: ai.google.dev/gemini-api/docs/robotics-overview and
/robotics-spatial (both "Last updated 2026-09-04"), plus the installed
google-genai 2.25.0 source.

  - Model ids: `gemini-robotics-er-2-preview`, and a Live-API-only
    `gemini-robotics-er-2-streaming-preview`. ER 1.6 is shut down. The docs
    use the Interactions API:
        client.interactions.create(model=..., input=[<image>, <text>],
                                   generation_config={"thinking_level": ...})
    and read `response.output_text`.
  - Inline images are `{"type": "image", "data": <base64>, "mime_type":
    "image/png"}` (google-genai `ImageContentParam`).
  - Point output is the JSON text `[{"point": [y, x], "label": ...}]`, with
    coordinates "in [y, x] format normalized to 0-1000".
  - Auth: the SDK sends the key in the `x-goog-api-key` request header, and
    nowhere else (checked against the real SDK in test_base_gemini.py).
    The docs also say the API rejects UNRESTRICTED keys with 403.
  - Latency: the docs recommend thinking level "medium" as the balance
    between latency and quality, so that is the default here.
  - Retries: the SDK's `HttpOptions.timeout` is in ms. With
    `HttpRetryOptions(attempts=1)`, the SDK's Interactions path still made
    ONE retry on 408/409/429/5xx when measured against 2.25.0 (none on other
    4xx). So the worst-case blocking time per step is about 2 x `timeout_s`
    plus backoff. The robot holds position meanwhile, because `get_action()`
    is synchronous.

Secret discipline (T-14-11; mirrors io_logger.py's rule that secrets never
reach episode.jsonl)
------------------------------------------------------------------------
  - The key is read only from the environment variable named by
    `api_key_env`. A missing or blank variable raises `SystemExit` at
    construction, before any robot or camera is touched. The module never
    makes an auth attempt with an empty string.
  - `last_details` holds only the parsed ER response, its truncated text,
    the model id and the conversion marker. It never holds the client,
    request headers or the key. An exception message is copied in only
    after the key has been redacted from it.

`get_action()` never raises. Any camera, SDK, transport, parse or conversion
failure degrades to `(dict(joint_state), "<label>@<reason>-holding-position")`.
This is the same fail-safe contract as `BridgeActionSource` and
`HttpAdapterActionSource`. The converted action still passes through
`run_episode()`'s unmodified `safety_validator.validate_action()`.
"""

import base64
import json
import os
import re
import threading

import cv2
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SO101Follower

from vla_bridge import safety_validator
from vla_bridge.backends import BackendHandles
from vla_bridge.backends.base_http_adapter import _is_joint_action, capture_observation_images
from vla_bridge.robot_client import OBSERVATION_MAX_SIDE
from vla_bridge.stereo_camera import StereoSplitCamera

ACTION_PATH = (
    "gemini-robotics-er embodied-reasoning points + local bounded visual-servo "
    "conversion (er_conversion); not native VLA action output (14-CONTEXT.md D-02 supersession)"
)
CONVERSION_SUFFIX = "gemini-er-conversion"

# Docs' recommended latency/quality balance (robotics-overview "Limitations").
DEFAULT_THINKING_LEVEL = "medium"

# Built on the documented pointing prompt (robotics-overview / robotics-spatial),
# narrowed to ONE target point. `{instruction}` is substituted with
# str.replace, never str.format, because the JSON example contains braces.
DEFAULT_ER_PROMPT_TEMPLATE = (
    "You are guiding a robot arm whose camera took this image. Task: \"{instruction}\". "
    "Point to the single object or location the robot gripper should move toward next "
    "to make progress on this task. The answer should follow the json format: "
    "[{\"point\": <point>, \"label\": <label1>}]. "
    "The points are in [y, x] format normalized to 0-1000."
)

# Truncation for the model's raw text in last_details (diagnostics only).
MAX_LOGGED_TEXT_CHARS = 2000

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def _make_genai_client(api_key: str, timeout_s: float, httpx_client=None):
    """Build the official google-genai client. Imported lazily: backend
    discovery imports every backend module on each run_vla_episode.py start."""
    from google import genai
    from google.genai import types

    http_options = types.HttpOptions(
        timeout=int(round(timeout_s * 1000)),  # HttpOptions.timeout is in milliseconds
        retry_options=types.HttpRetryOptions(attempts=1),
        httpx_client=httpx_client,
    )
    return genai.Client(api_key=api_key, http_options=http_options)


def parse_er_json(text) -> list | dict:
    """Parse Gemini Robotics-ER's JSON text answer (bare, or in a ```json fence).

    Raises `ValueError` if no JSON list or object can be parsed.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("empty or non-text ER response")
    candidate = text.strip()
    fenced = _CODE_FENCE_RE.search(candidate)
    if fenced:
        candidate = fenced.group(1).strip()
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        start, end = candidate.find("["), candidate.rfind("]")
        if start == -1 or end <= start:
            raise ValueError("ER response is not JSON") from None
        try:
            parsed = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError:
            raise ValueError("ER response is not JSON") from None
    if not isinstance(parsed, (list, dict)):
        raise ValueError("ER response JSON is not a list or object")
    return parsed


def _png_shape(b64_png: str) -> list[int]:
    """`[height, width]` read from a base64 PNG's IHDR chunk."""
    raw = base64.b64decode(b64_png[:64])
    if raw[:8] != _PNG_SIGNATURE or raw[12:16] != b"IHDR":
        raise ValueError("not a PNG")
    width = int.from_bytes(raw[16:20], "big")
    height = int.from_bytes(raw[20:24], "big")
    return [height, width]


class GeminiActionSourceBase:
    """`run_vla_episode.ActionSource` backed by the public Gemini Robotics-ER 2 API
    plus a local ER-to-joint conversion step (`convert_fn`).

    `convert_fn(er_output, joint_state)` receives
    `{"er_entries": <parsed ER JSON>, "image_shape": [height, width]}`, where
    `image_shape` is the shape of the frame actually sent. It must return
    an absolute `JOINT_ORDER` dict.
    """

    def __init__(
        self,
        model: str,
        api_key_env: str,
        checkpoint_label: str,
        wrist_cap,
        stereo_camera,
        convert_fn,
        timeout_s: float = 10.0,
        thinking_level: str | None = DEFAULT_THINKING_LEVEL,
        prompt_template: str = DEFAULT_ER_PROMPT_TEMPLATE,
        er_camera: str = "wrist",
        max_side: int = OBSERVATION_MAX_SIDE,
    ):
        if er_camera not in ("wrist", "overhead_left", "overhead_right"):
            raise ValueError(f"er_camera must be wrist/overhead_left/overhead_right, got {er_camera!r}")
        api_key = (os.environ.get(api_key_env) or "").strip()
        if not api_key:
            raise SystemExit(f"Environment variable {api_key_env} is not set")
        self.model = model
        self.api_key_env = api_key_env
        self.checkpoint_label = checkpoint_label
        self.wrist_cap = wrist_cap
        self.stereo_camera = stereo_camera
        self.convert_fn = convert_fn
        self.timeout_s = timeout_s
        self.thinking_level = thinking_level
        self.prompt_template = prompt_template
        self.er_camera = er_camera
        self.max_side = max_side
        self._api_key = api_key  # only used to redact error text; never logged
        self._client = _make_genai_client(api_key, timeout_s)
        self._inflight_lock = threading.Lock()
        self.last_details = self._details({})

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(model={self.model!r}, api_key_env={self.api_key_env!r}, "
                f"checkpoint_label={self.checkpoint_label!r})")

    def _details(self, raw_model_output: dict, bridge_flags: list | None = None) -> dict:
        return dict(
            raw_model_output=raw_model_output,
            bridge_flags=list(bridge_flags or []),
            observation=None,
            action_path=ACTION_PATH,
            er_model=self.model,
        )

    def _redact(self, text: str) -> str:
        return text.replace(self._api_key, "<redacted>") if self._api_key else text

    def _hold(self, joint_state, reason: str, error: BaseException | None = None):
        raw = self.last_details.get("raw_model_output", {})
        self.last_details = self._details(raw, [reason])
        if error is not None:
            self.last_details["error_type"] = type(error).__name__
            self.last_details["error"] = self._redact(str(error))[:MAX_LOGGED_TEXT_CHARS]
        return dict(joint_state), f"{self.checkpoint_label}@{reason}"

    def _request(self, image_b64: str, instruction: str):
        prompt = self.prompt_template.replace("{instruction}", str(instruction))
        kwargs = dict(
            model=self.model,
            input=[
                {"type": "image", "data": image_b64, "mime_type": "image/png"},
                {"type": "text", "text": prompt},
            ],
            timeout=self.timeout_s,
        )
        if self.thinking_level:
            kwargs["generation_config"] = {"thinking_level": self.thinking_level}
        return self._client.interactions.create(**kwargs)

    def get_action(self, joint_state: dict[str, float], instruction: str) -> tuple[dict[str, float], str]:
        self.last_details = self._details({})
        if not self._inflight_lock.acquire(blocking=False):
            return self._hold(joint_state, "gemini-request-inflight-holding-position")
        try:
            images = capture_observation_images(self.wrist_cap, self.stereo_camera, self.max_side)
            if images is None:
                return self._hold(joint_state, "camera-read-failed-holding-position")
            image_b64 = images[self.er_camera]

            try:
                image_shape = _png_shape(image_b64)
                response = self._request(image_b64, instruction)
                text = getattr(response, "output_text", None)
                entries = parse_er_json(text)
            except Exception as exc:  # SDK/network/parse: never crash the live loop
                return self._hold(joint_state, "gemini-error-holding-position", exc)

            er_output = {"er_entries": entries, "image_shape": image_shape}
            self.last_details = self._details(
                dict(er_output, er_text=self._redact(text)[:MAX_LOGGED_TEXT_CHARS])
            )
            try:
                converted = self.convert_fn(er_output, dict(joint_state))
            except Exception as exc:
                return self._hold(joint_state, "gemini-error-holding-position", exc)
            if not _is_joint_action(converted):
                return self._hold(joint_state, "action-format-error-holding-position")
            return converted, f"{self.checkpoint_label}@{CONVERSION_SUFFIX}"
        finally:
            self._inflight_lock.release()

    def close(self) -> None:
        """Release this source's dedicated wrist capture. The stereo camera is
        released by the caller through `BackendHandles.stereo_camera`."""
        if self.wrist_cap is not None and hasattr(self.wrist_cap, "release"):
            self.wrist_cap.release()


def build_gemini_backend(
    model: str,
    api_key_env: str,
    checkpoint_label: str,
    args,
    camera_names: dict[int, str],
    convert_fn,
    **source_kwargs,
) -> BackendHandles:
    """Connect the robot directly and build a `GeminiActionSourceBase`.

    This lane has no `RobotClient` and no Colab bridge. Robot and camera
    construction mirrors `base_http_adapter.build_http_adapter_backend()`:
    `SO101Follower`, then `connect(calibrate=False)`, then `write_calibration`
    with torque disabled. It also opens its own `StereoSplitCamera`, returned
    so `run_episode()` reuses it, and a dedicated wrist `cv2.VideoCapture`.

    It fails with `SystemExit` BEFORE touching the robot when there is no
    camera named "wrist" or the API-key variable is unset. To guarantee that,
    the action source, and with it the key check, is built before
    `robot.connect()`. `args` needs `.port`, `.robot_id` and
    `.stereo_camera_index`. `source_kwargs` pass through to
    `GeminiActionSourceBase` (thinking_level, timeout_s, ...).
    """
    wrist_idx = next((idx for idx, name in camera_names.items() if name == "wrist"), None)
    if wrist_idx is None:
        raise SystemExit(
            f"Gemini backend {checkpoint_label!r} needs a wrist camera "
            "(first --camera index / device_map.json wrist entry)."
        )

    action_source = GeminiActionSourceBase(
        model, api_key_env, checkpoint_label, None, None, convert_fn, **source_kwargs
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
    action_source.wrist_cap = wrist_cap
    action_source.stereo_camera = stereo_camera

    return BackendHandles(
        action_source=action_source,
        robot=robot,
        stereo_camera=stereo_camera,
        skip_camera_names=frozenset({"overhead"}),
    )


__all__ = [
    "ACTION_PATH",
    "CONVERSION_SUFFIX",
    "DEFAULT_ER_PROMPT_TEMPLATE",
    "GeminiActionSourceBase",
    "build_gemini_backend",
    "parse_er_json",
]
