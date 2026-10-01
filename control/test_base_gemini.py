"""Tests for the shared Gemini Robotics-ER foundation (Plan 14-13, VLAB-01/02).

`vla_bridge.backends.base_gemini` (GeminiActionSourceBase,
build_gemini_backend) and `vla_bridge.backends.er_conversion`
(convert_er_output_to_action).

Never touches real hardware, a real camera, or the real Gemini API: the SDK
client is replaced by a fake (or, in one test, the real `google-genai` SDK is
pointed at an `httpx.MockTransport`, so no request ever leaves the process).
"""

import ast
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import MockRobot
from vla_bridge import safety_validator
from vla_bridge.action_contract import JOINT_ORDER
from vla_bridge.backends import BackendHandles, base_gemini, eef_to_joint, er_conversion
from vla_bridge.backends.base_gemini import GeminiActionSourceBase

API_KEY_ENV = "SOARM_TEST_GEMINI_API_KEY"
SECRET = "AIzaSy-TEST-SECRET-do-not-log-0123456789"
MODEL = "gemini-robotics-er-2-preview"
LABEL = "gemini-robotics-er-2-preview"
JOINT_STATE = {joint: float(i) for i, joint in enumerate(JOINT_ORDER)}
POINT_TEXT = '[{"point": [376, 508], "label": "red cube"}]'


class FakeWristCap:
    def __init__(self, ok=True, shape=(48, 64, 3)):
        self.ok = ok
        self.shape = shape
        self.read_count = 0
        self.released = False

    def read(self):
        self.read_count += 1
        if not self.ok:
            return False, None
        return True, np.full(self.shape, 7, dtype=np.uint8)

    def release(self):
        self.released = True


class FakeStereo:
    def __init__(self, ok=True):
        self.ok = ok
        self.left_reads = 0
        self.right_reads = 0

    def read_left(self):
        self.left_reads += 1
        return np.full((30, 40, 3), 11, dtype=np.uint8) if self.ok else None

    def read_right(self):
        self.right_reads += 1
        return np.full((30, 40, 3), 22, dtype=np.uint8) if self.ok else None


class FakeInteractions:
    def __init__(self, output_text=POINT_TEXT, exc=None):
        self.output_text = output_text
        self.exc = exc
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.exc is not None:
            raise self.exc
        return SimpleNamespace(output_text=self.output_text)


@pytest.fixture
def api_key(monkeypatch):
    monkeypatch.setenv(API_KEY_ENV, SECRET)
    return SECRET


@pytest.fixture
def fake_client(monkeypatch):
    """Replace the lazily-built google-genai client with a recording fake."""
    state = {"interactions": FakeInteractions(), "factory_calls": []}

    def fake_make_client(api_key, timeout_s):
        state["factory_calls"].append(dict(api_key=api_key, timeout_s=timeout_s))
        return SimpleNamespace(interactions=state["interactions"])

    monkeypatch.setattr(base_gemini, "_make_genai_client", fake_make_client)
    return state


def _identity_convert(er_output, joint_state):
    return dict(joint_state)


def _source(convert_fn=_identity_convert, **kwargs):
    defaults = dict(
        model=MODEL,
        api_key_env=API_KEY_ENV,
        checkpoint_label=LABEL,
        wrist_cap=FakeWristCap(),
        stereo_camera=FakeStereo(),
        convert_fn=convert_fn,
    )
    defaults.update(kwargs)
    return GeminiActionSourceBase(**defaults)


def _dump(obj) -> str:
    return json.dumps(obj, default=repr)


# --- construction / API key handling (T-14-11) ---------------------------------


def test_missing_api_key_env_var_raises_clear_system_exit(monkeypatch, fake_client):
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    with pytest.raises(SystemExit, match=f"Environment variable {API_KEY_ENV} is not set"):
        _source()
    assert fake_client["factory_calls"] == []  # no silent empty-key auth attempt


def test_empty_api_key_env_var_raises_system_exit(monkeypatch, fake_client):
    monkeypatch.setenv(API_KEY_ENV, "   ")
    with pytest.raises(SystemExit, match=API_KEY_ENV):
        _source()
    assert fake_client["factory_calls"] == []


def test_api_key_is_read_from_the_named_env_var(api_key, fake_client):
    source = _source(timeout_s=7.5)
    assert fake_client["factory_calls"] == [dict(api_key=SECRET, timeout_s=7.5)]
    assert source.api_key_env == API_KEY_ENV
    assert SECRET not in repr(source)


def test_no_api_key_is_hardcoded_in_the_module():
    text = Path(base_gemini.__file__).read_text()
    assert "AIza" not in text
    assert "api_key=\"" not in text and "api_key='" not in text


def test_google_genai_is_imported_lazily_not_at_module_top_level():
    # Backend discovery imports every backend module (and therefore this base)
    # on every run_vla_episode.py start -- the cloud SDK must stay lazy.
    tree = ast.parse(Path(base_gemini.__file__).read_text())
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert not any(alias.name.startswith("google") for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("google")


# --- get_action(): success path --------------------------------------------------


def test_success_returns_converted_action_with_er_conversion_marker(api_key, fake_client):
    seen = {}

    def convert(er_output, joint_state):
        seen["er_output"] = er_output
        seen["joint_state"] = joint_state
        return {joint: value + 1.0 for joint, value in joint_state.items()}

    action, model_version = _source(convert_fn=convert).get_action(dict(JOINT_STATE), "pick up the red cube")

    assert action == {joint: value + 1.0 for joint, value in JOINT_STATE.items()}
    assert model_version == f"{LABEL}@gemini-er-conversion"
    assert seen["er_output"]["er_entries"] == [{"point": [376, 508], "label": "red cube"}]
    assert seen["er_output"]["image_shape"] == [48, 64]  # the frame actually sent
    assert seen["joint_state"] == JOINT_STATE


def test_request_follows_the_documented_interactions_shape(api_key, fake_client):
    _source(thinking_level="low").get_action(dict(JOINT_STATE), "pick up the red cube")

    (call,) = fake_client["interactions"].calls
    assert call["model"] == MODEL
    assert call["generation_config"] == {"thinking_level": "low"}
    assert call["timeout"] == pytest.approx(10.0)
    image, text = call["input"]
    assert image["type"] == "image"
    assert image["mime_type"] == "image/png"
    assert isinstance(image["data"], str) and image["data"]
    assert text["type"] == "text"
    assert "pick up the red cube" in text["text"]
    assert "[y, x]" in text["text"] and "0-1000" in text["text"]


def test_only_the_wrist_frame_is_sent_but_capture_matches_the_http_lane(api_key, fake_client):
    wrist, stereo = FakeWristCap(), FakeStereo()
    _source(wrist_cap=wrist, stereo_camera=stereo).get_action(dict(JOINT_STATE), "go")
    assert (wrist.read_count, stereo.left_reads, stereo.right_reads) == (1, 1, 1)
    (call,) = fake_client["interactions"].calls
    assert [part["type"] for part in call["input"]] == ["image", "text"]


def test_success_last_details_hold_parsed_er_output_and_path_marker(api_key, fake_client):
    source = _source()
    source.get_action(dict(JOINT_STATE), "go")
    details = source.last_details
    assert details["raw_model_output"]["er_entries"] == [{"point": [376, 508], "label": "red cube"}]
    assert details["raw_model_output"]["image_shape"] == [48, 64]
    assert details["bridge_flags"] == []
    assert details["observation"] is None
    assert "gemini-robotics-er" in details["action_path"]
    assert "not native" in details["action_path"]


def test_json_wrapped_in_markdown_code_fences_is_parsed(api_key, fake_client):
    fake_client["interactions"].output_text = "```json\n" + POINT_TEXT + "\n```"
    seen = {}

    def convert(er_output, joint_state):
        seen.update(er_output)
        return dict(joint_state)

    _, model_version = _source(convert_fn=convert).get_action(dict(JOINT_STATE), "go")
    assert model_version == f"{LABEL}@gemini-er-conversion"
    assert seen["er_entries"] == [{"point": [376, 508], "label": "red cube"}]


# --- get_action(): fail-safe, never raises ---------------------------------------


@pytest.mark.parametrize(
    "exc",
    [RuntimeError("sdk exploded"), ConnectionError("network down"), TimeoutError("slow"),
     ValueError("bad"), OSError("socket")],
    ids=lambda e: type(e).__name__,
)
def test_sdk_exception_holds_position_and_never_raises(api_key, fake_client, exc):
    fake_client["interactions"].exc = exc
    action, model_version = _source().get_action(dict(JOINT_STATE), "go")
    assert action == JOINT_STATE
    assert model_version == f"{LABEL}@gemini-error-holding-position"


@pytest.mark.parametrize("text", ["", "not json at all", "{\"point\": ", None, 42])
def test_malformed_response_text_holds_position(api_key, fake_client, text):
    fake_client["interactions"].output_text = text
    action, model_version = _source().get_action(dict(JOINT_STATE), "go")
    assert action == JOINT_STATE
    assert model_version == f"{LABEL}@gemini-error-holding-position"


def test_convert_fn_exception_holds_position(api_key, fake_client):
    def boom(er_output, joint_state):
        raise ZeroDivisionError("conversion bug")

    action, model_version = _source(convert_fn=boom).get_action(dict(JOINT_STATE), "go")
    assert action == JOINT_STATE
    assert model_version == f"{LABEL}@gemini-error-holding-position"


def test_non_joint_final_action_holds_with_format_error(api_key, fake_client):
    action, model_version = _source(convert_fn=lambda er, js: [1, 2, 3]).get_action(dict(JOINT_STATE), "go")
    assert action == JOINT_STATE
    assert model_version == f"{LABEL}@action-format-error-holding-position"


def test_camera_failure_holds_without_calling_gemini(api_key, fake_client):
    action, model_version = _source(wrist_cap=FakeWristCap(ok=False)).get_action(dict(JOINT_STATE), "go")
    assert action == JOINT_STATE
    assert model_version == f"{LABEL}@camera-read-failed-holding-position"
    assert fake_client["interactions"].calls == []


def test_inflight_request_holds_without_calling_gemini(api_key, fake_client):
    source = _source()
    assert source._inflight_lock.acquire(blocking=False)
    try:
        action, model_version = source.get_action(dict(JOINT_STATE), "go")
    finally:
        source._inflight_lock.release()
    assert action == JOINT_STATE
    assert model_version == f"{LABEL}@gemini-request-inflight-holding-position"
    assert fake_client["interactions"].calls == []


def test_hold_returns_a_copy_not_the_callers_dict(api_key, fake_client):
    fake_client["interactions"].exc = RuntimeError("x")
    state = dict(JOINT_STATE)
    action, _ = _source().get_action(state, "go")
    assert action == state and action is not state


# --- secret-disclosure discipline (T-14-11) --------------------------------------


def test_last_details_never_contain_the_api_key_on_success(api_key, fake_client):
    source = _source()
    source.get_action(dict(JOINT_STATE), "go")
    dumped = _dump(source.last_details)
    assert SECRET not in dumped
    assert "x-goog-api-key" not in dumped.lower()
    assert "authorization" not in dumped.lower()


def test_last_details_redact_the_api_key_even_if_an_error_message_echoes_it(api_key, fake_client):
    fake_client["interactions"].exc = RuntimeError(f"403 for key {SECRET} (x-goog-api-key: {SECRET})")
    source = _source()
    action, model_version = source.get_action(dict(JOINT_STATE), "go")
    assert model_version.endswith("@gemini-error-holding-position")
    dumped = _dump(source.last_details)
    assert SECRET not in dumped
    assert source.last_details["bridge_flags"] == ["gemini-error-holding-position"]
    assert source.last_details["error_type"] == "RuntimeError"


def test_real_sdk_request_shape_and_key_only_in_auth_header(api_key, monkeypatch):
    """Drive the REAL pinned google-genai SDK against an in-process
    httpx.MockTransport: checks the documented Interactions request shape and
    that the key travels only in the x-goog-api-key header, never in
    last_details. No network I/O happens."""
    pytest.importorskip("google.genai")
    httpx = pytest.importorskip("httpx")
    requests_seen = []

    def handler(request):
        requests_seen.append(request)
        return httpx.Response(200, json={
            "id": "interaction-1",
            "status": "completed",
            "steps": [{"type": "model_output", "content": [{"type": "text", "text": POINT_TEXT}]}],
        })

    real_make_client = base_gemini._make_genai_client

    def make_client_with_mock_transport(api_key_value, timeout_s):
        return real_make_client(
            api_key_value, timeout_s, httpx_client=httpx.Client(transport=httpx.MockTransport(handler))
        )

    monkeypatch.setattr(base_gemini, "_make_genai_client", make_client_with_mock_transport)
    seen = {}

    def convert(er_output, joint_state):
        seen.update(er_output)
        return dict(joint_state)

    source = _source(convert_fn=convert)
    _, model_version = source.get_action(dict(JOINT_STATE), "pick up the red cube")

    assert model_version == f"{LABEL}@gemini-er-conversion"
    assert seen["er_entries"] == [{"point": [376, 508], "label": "red cube"}]
    (request,) = requests_seen
    assert request.method == "POST"
    assert request.url.path == "/v1beta/interactions"
    assert request.headers["x-goog-api-key"] == SECRET
    body = json.loads(request.content)
    assert body["model"] == MODEL
    (turn,) = body["input"]
    assert [part["type"] for part in turn["content"]] == ["image", "text"]
    assert turn["content"][0]["mime_type"] == "image/png"
    assert SECRET not in request.content.decode()
    assert SECRET not in _dump(source.last_details)


# --- close() / build_gemini_backend() --------------------------------------------


def test_close_releases_the_dedicated_wrist_capture(api_key, fake_client):
    wrist = FakeWristCap()
    _source(wrist_cap=wrist).close()
    assert wrist.released is True


class RecordingMockRobot(MockRobot):
    instances = []

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.connect_kwargs = None
        self.calibration_written = False
        self.bus.write_calibration = self._write_calibration
        RecordingMockRobot.instances.append(self)

    def _write_calibration(self, calibration):
        self.calibration_written = True

    def connect(self, calibrate: bool = False) -> None:
        self.connect_kwargs = dict(calibrate=calibrate)
        self.connected = True


def _args(**overrides):
    defaults = dict(
        port="/dev/tty.usbmodem1",
        robot_id="soarm_follower_02",
        instruction="pick the cube",
        stereo_camera_index=1,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _patch_gemini_lane_hardware(monkeypatch):
    opened = {}
    RecordingMockRobot.instances = []
    monkeypatch.setattr(base_gemini, "SO101Follower", RecordingMockRobot)

    def fake_stereo(index=1):
        opened["stereo_index"] = index
        return FakeStereo()

    def fake_video_capture(index):
        opened["wrist_index"] = index
        return FakeWristCap()

    monkeypatch.setattr(base_gemini, "StereoSplitCamera", fake_stereo)
    monkeypatch.setattr(base_gemini.cv2, "VideoCapture", fake_video_capture)
    return opened


def test_build_gemini_backend_constructs_robot_cameras_and_action_source(monkeypatch, api_key, fake_client):
    opened = _patch_gemini_lane_hardware(monkeypatch)

    handles = base_gemini.build_gemini_backend(
        MODEL, API_KEY_ENV, LABEL, _args(stereo_camera_index=3), {5: "wrist", 3: "overhead"},
        convert_fn=_identity_convert,
    )

    assert isinstance(handles, BackendHandles)
    robot = RecordingMockRobot.instances[-1]
    assert handles.robot is robot
    assert robot.connect_kwargs == dict(calibrate=False)
    assert robot.calibration_written is True
    assert robot.config.port == "/dev/tty.usbmodem1"
    assert robot.config.id == "soarm_follower_02"
    assert robot.config.max_relative_target == safety_validator.MAX_RELATIVE_TARGET_DEG
    assert opened == {"stereo_index": 3, "wrist_index": 5}
    source = handles.action_source
    assert isinstance(source, GeminiActionSourceBase)
    assert (source.model, source.api_key_env, source.checkpoint_label) == (MODEL, API_KEY_ENV, LABEL)
    assert source.convert_fn is _identity_convert
    assert source.stereo_camera is handles.stereo_camera
    assert handles.skip_camera_names == frozenset({"overhead"})
    assert handles.shutdown is None  # no RobotClient on this lane: default robot.disconnect()


def test_build_gemini_backend_missing_key_exits_before_touching_robot(monkeypatch, fake_client):
    _patch_gemini_lane_hardware(monkeypatch)
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    with pytest.raises(SystemExit, match=API_KEY_ENV):
        base_gemini.build_gemini_backend(
            MODEL, API_KEY_ENV, LABEL, _args(), {5: "wrist", 3: "overhead"}, convert_fn=_identity_convert
        )
    assert RecordingMockRobot.instances == []


def test_build_gemini_backend_requires_a_wrist_camera_before_touching_robot(monkeypatch, api_key, fake_client):
    _patch_gemini_lane_hardware(monkeypatch)
    with pytest.raises(SystemExit, match="wrist"):
        base_gemini.build_gemini_backend(
            MODEL, API_KEY_ENV, LABEL, _args(), {3: "overhead"}, convert_fn=_identity_convert
        )
    assert RecordingMockRobot.instances == []


# =============================================================================
# Task 2: er_conversion.convert_er_output_to_action() -- bounded visual servo
# =============================================================================

# A generic, well-conditioned pose (inside the URDF limits, not singular).
SERVO_POSE = {"shoulder_pan": 10.0, "shoulder_lift": -20.0, "elbow_flex": 30.0,
              "wrist_flex": 40.0, "wrist_roll": 0.0, "gripper": 30.0}
SHAPE = (480, 640)


def _point(y, x):
    return {"er_entries": [{"point": [y, x], "label": "target"}], "image_shape": list(SHAPE)}


@pytest.fixture
def ik_spy(monkeypatch):
    calls = []
    real = eef_to_joint.solve_ik

    def spy(current, eef_delta, *args, **kwargs):
        calls.append(dict(current=dict(current), eef_delta=dict(eef_delta), kwargs=kwargs))
        return real(current, eef_delta, *args, **kwargs)

    monkeypatch.setattr(er_conversion.eef_to_joint, "solve_ik", spy)
    return calls


def _eef_xyz(joints):
    return np.array(eef_to_joint.forward_kinematics(joints)[:3])


def _heading(joints):
    x, y = eef_to_joint.forward_kinematics(joints)[:2]
    r = math.hypot(x, y)
    return np.array([x / r, y / r])


@pytest.mark.parametrize("yx", [(0, 0), (0, 1000), (1000, 0), (1000, 1000), (0, 500), (500, 1000), (380, 560)])
def test_well_formed_point_gives_finite_joint_order_bounded_action(yx, ik_spy):
    action = er_conversion.convert_er_output_to_action(_point(*yx), dict(SERVO_POSE))
    assert list(action) == list(JOINT_ORDER)
    assert all(math.isfinite(v) for v in action.values())
    (call,) = ik_spy
    delta = call["eef_delta"]
    assert math.hypot(delta["dx"], delta["dy"]) <= er_conversion.MAX_NUDGE_M + 1e-12
    assert (delta["dz"], delta["droll"], delta["dpitch"], delta["dyaw"]) == (0.0, 0.0, 0.0, 0.0)
    assert delta["gripper"] == SERVO_POSE["gripper"]
    moved = np.linalg.norm(_eef_xyz(action) - _eef_xyz(SERVO_POSE))
    assert moved <= 2 * er_conversion.MAX_NUDGE_M
    assert max(abs(action[j] - SERVO_POSE[j]) for j in JOINT_ORDER) < 10.0
    assert action["gripper"] == SERVO_POSE["gripper"]  # never decides to grasp


def test_far_off_points_saturate_at_the_same_small_maximum(ik_spy):
    for yx in [(0, 0), (1000, 1000)]:
        er_conversion.convert_er_output_to_action(_point(*yx), dict(SERVO_POSE))
    for call in ik_spy:
        delta = call["eef_delta"]
        assert math.hypot(delta["dx"], delta["dy"]) == pytest.approx(er_conversion.MAX_NUDGE_M)


def test_nudge_is_proportional_below_saturation(ik_spy):
    # 20 px right of centre in a 640-wide frame: x = 500 + 20/640*1000.
    er_conversion.convert_er_output_to_action(_point(500, 500 + 20 / 640 * 1000), dict(SERVO_POSE))
    delta = ik_spy[0]["eef_delta"]
    expected = 20 * er_conversion.PIXELS_TO_EEF_GAIN
    assert expected < er_conversion.MAX_NUDGE_M
    assert math.hypot(delta["dx"], delta["dy"]) == pytest.approx(expected, rel=1e-6)


def test_target_inside_deadband_holds_without_ik(ik_spy):
    action = er_conversion.convert_er_output_to_action(_point(500, 501), dict(SERVO_POSE))
    assert action == SERVO_POSE
    assert ik_spy == []


def test_image_up_moves_radially_outward_and_image_right_moves_tangentially(ik_spy):
    heading = _heading(SERVO_POSE)
    right = np.array([heading[1], -heading[0]])
    er_conversion.convert_er_output_to_action(_point(0, 500), dict(SERVO_POSE))  # straight up
    er_conversion.convert_er_output_to_action(_point(500, 1000), dict(SERVO_POSE))  # straight right
    up_delta = np.array([ik_spy[0]["eef_delta"]["dx"], ik_spy[0]["eef_delta"]["dy"]])
    right_delta = np.array([ik_spy[1]["eef_delta"]["dx"], ik_spy[1]["eef_delta"]["dy"]])
    unit = er_conversion.MAX_NUDGE_M
    assert up_delta @ heading == pytest.approx(er_conversion.IMAGE_UP_SIGN * unit)
    assert right_delta @ right == pytest.approx(er_conversion.IMAGE_RIGHT_SIGN * unit)


@pytest.mark.parametrize(
    "er_response",
    [
        {"er_entries": [{"box_2d": [100, 200, 300, 400], "label": "b"}], "image_shape": [480, 640]},
        {"er_entries": [{"y": 100, "x": 200, "y2": 300, "x2": 400, "label": "b"}], "image_shape": [480, 640]},
        {"er_entries": [{"point": [200, 300]}], "image_shape": [480, 640]},
        [{"point": [200, 300], "label": "bare list"}],
        {"point": [200, 300], "label": "single entry"},
        {"er_entries": [{"label": "no point"}, {"point": [200, 300]}], "image_shape": [480, 640]},
    ],
    ids=["box_2d", "box_xy", "point", "bare-list", "single-entry", "first-valid-entry"],
)
def test_points_and_boxes_resolve_to_the_same_target(er_response):
    assert er_conversion.extract_target_yx(er_response) == pytest.approx((200.0, 300.0))


def test_image_shape_comes_from_the_response_then_argument_then_default(ik_spy):
    entry = [{"point": [500, 1000]}]  # right edge: 500 normalized units right of centre
    # Tiny frame from the response: 500/1000*20 = 10 px -> below saturation.
    er_conversion.convert_er_output_to_action({"er_entries": entry, "image_shape": [10, 20]}, dict(SERVO_POSE))
    # An explicit argument overrides the response.
    er_conversion.convert_er_output_to_action(
        {"er_entries": entry, "image_shape": [10, 20]}, dict(SERVO_POSE), image_shape=(10, 40)
    )
    # Neither: DEFAULT_IMAGE_SHAPE (640 wide -> 320 px -> saturates).
    er_conversion.convert_er_output_to_action(entry, dict(SERVO_POSE))
    norms = [math.hypot(c["eef_delta"]["dx"], c["eef_delta"]["dy"]) for c in ik_spy]
    gain = er_conversion.PIXELS_TO_EEF_GAIN
    assert norms == pytest.approx([10 * gain, 20 * gain, er_conversion.MAX_NUDGE_M])
    assert er_conversion.DEFAULT_IMAGE_SHAPE == (480, 640)


@pytest.mark.parametrize(
    "er_response",
    [None, "text", 42, [], {}, {"er_entries": []}, [{"label": "x"}], [{"point": [1]}],
     [{"point": ["a", "b"]}], [{"point": [float("nan"), 5]}], [{"point": [float("inf"), 5]}],
     [{"point": [2000, 5]}], [{"point": [-1, 5]}], [{"point": [True, 5]}], [{"box_2d": [1, 2, 3]}],
     {"er_entries": [{"point": [10, 10]}], "image_shape": [0, 640]},
     {"er_entries": [{"point": [10, 10]}], "image_shape": "big"}],
)
def test_malformed_or_missing_point_holds_position_and_never_raises(er_response, ik_spy):
    action = er_conversion.convert_er_output_to_action(er_response, dict(SERVO_POSE))
    assert action == SERVO_POSE
    assert ik_spy == []


def test_bad_joint_state_holds_and_never_raises():
    state = dict(SERVO_POSE)
    del state["elbow_flex"]
    assert er_conversion.convert_er_output_to_action(_point(0, 0), state) == state
    nan_state = dict(SERVO_POSE, wrist_flex=float("nan"))
    held = er_conversion.convert_er_output_to_action(_point(0, 0), nan_state)
    assert set(held) == set(nan_state) and math.isnan(held["wrist_flex"])


def test_ik_failure_holds_and_never_raises(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("ik blew up")

    monkeypatch.setattr(er_conversion.eef_to_joint, "solve_ik", boom)
    assert er_conversion.convert_er_output_to_action(_point(0, 0), dict(SERVO_POSE)) == SERVO_POSE


def test_hold_returns_a_copy(ik_spy):
    state = dict(SERVO_POSE)
    held = er_conversion.convert_er_output_to_action(None, state)
    assert held == state and held is not state


def test_make_er_convert_fn_forwards_calibrated_joint_limits(ik_spy):
    limits = {joint: (-60.0, 60.0) for joint in JOINT_ORDER[:5]}
    convert = er_conversion.make_er_convert_fn(limits)
    action = convert(_point(0, 0), dict(SERVO_POSE))
    assert ik_spy[0]["kwargs"]["joint_limits_deg"] == limits
    assert all(-60.0 <= action[j] <= 60.0 for j in JOINT_ORDER[:5])


def test_er_conversion_module_documents_it_is_not_a_native_vla_policy():
    doc = er_conversion.__doc__
    assert "NOT" in doc and "native" in doc and "D-02" in doc


def test_gemini_source_with_real_conversion_end_to_end(api_key, fake_client):
    fake_client["interactions"].output_text = '[{"point": [100, 900], "label": "red cube"}]'
    source = _source(convert_fn=er_conversion.make_er_convert_fn(None))
    action, model_version = source.get_action(dict(SERVO_POSE), "pick up the red cube")
    assert model_version == f"{LABEL}@gemini-er-conversion"
    assert list(action) == list(JOINT_ORDER)
    assert all(math.isfinite(v) for v in action.values())
    assert action != SERVO_POSE  # an actual (small) nudge, not a hold
    assert np.linalg.norm(_eef_xyz(action) - _eef_xyz(SERVO_POSE)) <= 2 * er_conversion.MAX_NUDGE_M
    assert source.last_details["bridge_flags"] == []


def test_gemini_source_no_target_is_flagged_as_held(api_key, fake_client):
    fake_client["interactions"].output_text = '[{"label": "nothing to point at"}]'
    source = _source(convert_fn=er_conversion.make_er_convert_fn(None))
    action, model_version = source.get_action(dict(SERVO_POSE), "go")
    assert action == SERVO_POSE
    assert model_version == f"{LABEL}@gemini-er-conversion"
    assert source.last_details["bridge_flags"] == ["gemini-er-conversion-held-position"]
