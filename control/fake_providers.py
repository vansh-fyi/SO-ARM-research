"""Shared test doubles for the Phase 15 MLLM pipeline (PIPE-02).

Not a test module (no `test_` prefix, so pytest never collects it). Every
MLLM test file imports its fakes from here so provider adapters, collectors
and cameras behave identically across `test_base_mllm.py` and the later
per-backend tests.

Nothing here touches real hardware, a real camera or the network:

  - `ScriptedAdapter` stands in for a provider adapter. It replays a fixed
    script of responses and raises `AssertionError` when the script runs out,
    so an unexpected extra (paid) provider call fails a test loudly.
  - `FakeCollector` stands in for the per-chunk input collector (camera
    frames, depth note, capture metadata).
  - `FakeCaptureCamera`, `FakeWristCap` and `FakeStereoCamera` stand in for
    `cv2.VideoCapture` and `StereoSplitCamera`.
  - `FakeDepthClient` stands in for `DepthCameraClient` (the FastFS depth
    service), with scripted failures for the HTTP 429 retry tests.

`vla_bridge.backends.base_mllm` is imported lazily inside the functions that
need its types, so this module imports cleanly on its own.
"""

import json

import cv2
import numpy as np

from vla_bridge.action_contract import JOINT_ORDER

DEFAULT_PLAN = "Move toward the pen, then close the gripper around it."
DEFAULT_TASK_RESTATEMENT = "Pick up the pen from the table."
DEFAULT_SCENE_CLAIMS = (
    {"claim": "A pen lies on the table in front of the arm.", "source": "wrist", "confidence": "high"},
    {"claim": "The gripper is open and above the table.", "source": "left", "confidence": "medium"},
)


def make_step(base: float, step: float, k: int) -> dict:
    """One action step: every `JOINT_ORDER` key mapped to `base + step*(k+1)`."""
    return {joint: float(base + step * (k + 1)) for joint in JOINT_ORDER}


def make_chunk_doc(
    n: int,
    plan: str = DEFAULT_PLAN,
    task_restatement: str = DEFAULT_TASK_RESTATEMENT,
    scene_claims=None,
    task_complete: bool = False,
    base: float = 5.0,
    step: float = 0.5,
) -> dict:
    """A python dict shaped like the action-chunk schema, with `n` steps."""
    claims = [dict(c) for c in (DEFAULT_SCENE_CLAIMS if scene_claims is None else scene_claims)]
    return {
        "task_restatement": task_restatement,
        "plan": plan,
        "scene_claims": claims,
        "actions": [make_step(base, step, k) for k in range(n)],
        "task_complete": task_complete,
    }


def make_chunk_text(n: int, **kwargs) -> str:
    """`make_chunk_doc(...)` serialised as the JSON text a provider returns."""
    return json.dumps(make_chunk_doc(n, **kwargs))


def png_bytes(value: int = 0, shape: tuple = (8, 8, 3)) -> bytes:
    """PNG bytes of a small constant uint8 array."""
    ok, buffer = cv2.imencode(".png", np.full(shape, value, dtype=np.uint8))
    assert ok
    return buffer.tobytes()


def make_provider_result(
    text: str,
    stop_reason: str = "complete",
    raw_payload=None,
    usage: dict | None = None,
    **kwargs,
):
    """A `ProviderResult` with default usage and a `complete` stop."""
    from vla_bridge.backends.base_mllm import ProviderResult

    if usage is None:
        usage = {
            "input_tokens": 100,
            "output_tokens": 50,
            "reasoning_tokens": 10,
            "cached_input_tokens": None,
            "raw": {},
        }
    stop = {"reason": stop_reason, "refusal_category": None, "incomplete_reason": None}
    return ProviderResult(
        text=text,
        raw_payload={"fake": True} if raw_payload is None else raw_payload,
        usage=usage,
        stop=stop,
        **kwargs,
    )


class ScriptedAdapter:
    """Provider adapter that replays a fixed script.

    Each scripted item is consumed by one `call()`:
      - a `str` becomes a complete `ProviderResult` via `make_provider_result`;
      - a `ProviderResult` is returned as is;
      - an `Exception` instance is raised (a transport failure).
    """

    provider = "fake"

    def __init__(self, responses, model: str = "fake-model"):
        self.responses = list(responses)
        self.model = model
        self.sdk = {"name": "fake", "version": "0"}
        self.settings_summary = {
            "requested": {"temperature": 1.0},
            "applied": {},
            "omitted": {"temperature": "fake provider"},
        }
        self.requests = []
        self.calls = 0
        self.closed = False

    def call(self, request):
        self.calls += 1
        self.requests.append(request)
        if not self.responses:
            raise AssertionError(
                f"ScriptedAdapter script exhausted: unexpected provider call #{self.calls}"
            )
        item = self.responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, str):
            return make_provider_result(item)
        return item

    def close(self):
        self.closed = True


class FakeCollector:
    """Per-chunk input collector returning fixed PNG frames and metadata."""

    def __init__(self, include_depth: bool = False, fail: bool = False):
        self.include_depth = include_depth
        self.fail = fail
        self.calls = 0
        self.joint_states = []
        self.closed = False

    def collect(self, joint_state):
        from vla_bridge.backends.base_mllm import MllmInputs

        self.calls += 1
        self.joint_states.append(dict(joint_state))
        if self.fail:
            return None
        images = {"wrist": png_bytes(10), "left": png_bytes(20), "right": png_bytes(30)}
        if self.include_depth:
            images["depth"] = png_bytes(40)
        return MllmInputs(
            images=images,
            depth_note="Depth unavailable this call (fake collector).",
            meta={
                "timings_ms": {"depth_fetch": None, "image_encode": 1.0},
                "depth": {"status": "not_configured"},
                "left_image_source_requested": "raw",
                "left_image_source_sent": "raw",
            },
            sidecar_files={},
        )

    def close(self):
        self.closed = True


class FakeCaptureCamera:
    """`cv2.VideoCapture` stand-in for `run_episode()`'s diagnostic recording."""

    def read(self):
        return True, np.zeros((4, 4, 3), dtype=np.uint8)


class FakeWristCap:
    """Wrist `cv2.VideoCapture` stand-in with an optional failing read."""

    def __init__(self, ok: bool = True, shape: tuple = (48, 64, 3), value: int = 7, calls=None):
        self.ok = ok
        self.shape = shape
        self.value = value
        self.calls = calls if calls is not None else []
        self.released = False

    def grab(self):
        self.calls.append("wrist.grab")
        return self.ok

    def read(self):
        self.calls.append("wrist.read")
        if not self.ok:
            return False, None
        return True, np.full(self.shape, self.value, dtype=np.uint8)

    def isOpened(self):
        return True

    def release(self):
        self.released = True


class FakeStereoCamera:
    """`StereoSplitCamera` stand-in returning distinct left/right arrays and
    recording the call order (shared `calls` list) for read-order tests."""

    def __init__(self, ok: bool = True, shape: tuple = (60, 80, 3), calls=None):
        self.ok = ok
        self.shape = shape
        self.calls = calls if calls is not None else []
        self.released = False
        self.last_pair_metadata = {"pair_id": 1}

    def read_left(self):
        self.calls.append("stereo.read_left")
        return np.full(self.shape, 11, dtype=np.uint8) if self.ok else None

    def read_right(self):
        self.calls.append("stereo.read_right")
        return np.full(self.shape, 22, dtype=np.uint8) if self.ok else None

    def release(self):
        self.released = True


def make_depth_array(shape: tuple = (576, 800)) -> np.ndarray:
    """A float32 metres depth map: a left-to-right ramp 0.3 -> 1.4 m with one
    invalid (NaN) 24x25 block in the top-left corner."""
    rows, cols = shape
    depth = np.tile(np.linspace(0.3, 1.4, cols, dtype=np.float32), (rows, 1))
    depth[0:min(24, rows), 0:min(25, cols)] = np.nan
    return depth


class FakeDepthClient:
    """`DepthCameraClient` stand-in honouring the `compute_depth_result` dict
    contract (never raises unless scripted to).

    Without `results`, every call succeeds with `depth` (default
    `make_depth_array()`). With `results`, each call consumes one scripted
    item: `None` -> success with the default depth; an `np.ndarray` ->
    success with that depth; a `str` -> a failure record with that error
    text; a `dict` -> returned as is; an `Exception` -> raised. An exhausted
    script raises `AssertionError` so an unexpected extra depth request fails
    loudly. `received` keeps `(left, right, request_id)` per call.
    """

    ALIGNED_VALUE = 99
    CALIBRATION_SHA256 = "fake-calibration-sha256"
    MODEL = "fake-fastfs"

    def __init__(self, depth=None, results=None, inference_ms: float = 12.5,
                 aligned_shape: tuple = (576, 800, 3), calls=None):
        self.depth = make_depth_array() if depth is None else depth
        self.results = None if results is None else list(results)
        self.inference_ms = inference_ms
        self.aligned_shape = aligned_shape
        self.call_log = calls if calls is not None else []
        self.calls = 0
        self.received = []
        self.ready_checks = 0
        self.last_depth = None

    def _success(self, depth: np.ndarray) -> dict:
        depth = np.asarray(depth, dtype=np.float32).copy()
        valid = np.isfinite(depth) & (depth > 0)
        self.last_depth = depth
        return {
            "depth": depth,
            "aligned_left": np.full(self.aligned_shape, self.ALIGNED_VALUE, dtype=np.uint8),
            "aligned_right": np.full(self.aligned_shape, self.ALIGNED_VALUE + 1, dtype=np.uint8),
            "valid_fraction": float(valid.mean()),
            "calibration_sha256": self.CALIBRATION_SHA256,
            "units": "metres",
            "pixel_frame": "rectified_left_resized",
            "inference_ms": self.inference_ms,
            "model": self.MODEL,
        }

    def _failure(self, error: str) -> dict:
        return {
            "depth": None,
            "calibration_sha256": self.CALIBRATION_SHA256,
            "units": "metres",
            "pixel_frame": "rectified_left_resized",
            "error": error,
        }

    def compute_depth_result(self, left_frame, right_frame, request_id=None):
        self.calls += 1
        self.call_log.append("depth.compute")
        self.received.append((left_frame, right_frame, request_id))
        if self.results is None:
            return self._success(self.depth)
        if not self.results:
            raise AssertionError(f"FakeDepthClient script exhausted: unexpected depth call #{self.calls}")
        item = self.results.pop(0)
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, str):
            return self._failure(item)
        if isinstance(item, dict):
            return item
        return self._success(self.depth if item is None else item)

    def check_ready(self) -> dict:
        self.ready_checks += 1
        return {"status": "ready", "protocol_version": 1}


__all__ = [
    "DEFAULT_PLAN",
    "DEFAULT_SCENE_CLAIMS",
    "DEFAULT_TASK_RESTATEMENT",
    "FakeCaptureCamera",
    "FakeCollector",
    "FakeDepthClient",
    "FakeStereoCamera",
    "FakeWristCap",
    "ScriptedAdapter",
    "make_chunk_doc",
    "make_chunk_text",
    "make_depth_array",
    "make_provider_result",
    "make_step",
    "png_bytes",
]
