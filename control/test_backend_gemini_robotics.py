"""Tests for `--backend gemini-robotics` (Plan 14-14, VLAB-01).

The backend is the public Gemini Robotics-ER 2 model plus Plan 14-13's local
`er_conversion` layer, NOT a native Gemini Robotics VLA (14-CONTEXT.md D-02
supersession). No test here calls the real Gemini API: the builder is
stubbed, or the REAL pinned google-genai SDK is driven against an in-process
`httpx.MockTransport`.
"""

import ast
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import MockRobot
from vla_bridge import backends, safety_validator
from vla_bridge.action_contract import JOINT_ORDER
from vla_bridge.backends import BACKEND_REGISTRY, GRPC_BRIDGE_BACKENDS, base_gemini, er_conversion
from vla_bridge.backends import gemini_robotics

BACKEND_NAME = "gemini-robotics"
CONFIG_NAME = "gemini_robotics"
PUBLIC_ER_MODEL = "gemini-robotics-er-2-preview"
API_KEY_ENV = "SOARM_TEST_GEMINI_ROBOTICS_KEY"
SECRET = "AIzaSy-TEST-SECRET-gemini-robotics-0123456789"
CAMERA_NAMES = {5: "wrist", 3: "overhead"}
JOINT_LIMITS_DEG = {joint: (-180.0, 180.0) for joint in JOINT_ORDER}
JOINT_LIMITS_DEG["gripper"] = (0.0, 100.0)
SERVO_POSE = {"shoulder_pan": 10.0, "shoulder_lift": -20.0, "elbow_flex": 30.0,
              "wrist_flex": 40.0, "wrist_roll": 0.0, "gripper": 30.0}
MODULE_PATH = Path(gemini_robotics.__file__)
SHIPPED_CONFIG_PATH = backends.BACKEND_CONFIG_DIR / f"{CONFIG_NAME}.json"


def _args(**overrides):
    defaults = dict(
        port="/dev/tty.usbmodem1",
        robot_id="soarm_follower_02",
        instruction="pick up the red cube",
        stereo_camera_index=1,
        server_address=None,
        checkpoint=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


@pytest.fixture
def config_dir(monkeypatch, tmp_path):
    """Point load_backend_config() at an empty tmp dir; returns a writer."""
    monkeypatch.setattr(backends, "BACKEND_CONFIG_DIR", tmp_path)

    def write(config):
        path = tmp_path / f"{CONFIG_NAME}.json"
        path.write_text(config if isinstance(config, str) else json.dumps(config))
        return path

    return write


@pytest.fixture
def builder_spy(monkeypatch):
    """Replace base_gemini.build_gemini_backend with a recorder (no robot, no SDK)."""
    calls = []
    sentinel = object()

    def fake_build(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    monkeypatch.setattr(base_gemini, "build_gemini_backend", fake_build)
    return SimpleNamespace(calls=calls, sentinel=sentinel)


# =============================================================================
# Task 1: registration, config resolution, builder wiring
# =============================================================================


def test_gemini_robotics_is_registered_with_plain_register_not_the_grpc_lane():
    assert BACKEND_NAME in BACKEND_REGISTRY
    assert BACKEND_REGISTRY[BACKEND_NAME] is gemini_robotics.build
    assert BACKEND_NAME not in GRPC_BRIDGE_BACKENDS


def test_build_calls_the_shared_gemini_builder_with_the_config_model_and_key_env(config_dir, builder_spy):
    config_dir({"model": PUBLIC_ER_MODEL, "api_key_env": API_KEY_ENV})
    args = _args()

    result = BACKEND_REGISTRY[BACKEND_NAME](args, CAMERA_NAMES, JOINT_LIMITS_DEG)

    assert result is builder_spy.sentinel
    ((positional, kwargs),) = builder_spy.calls
    assert positional == (PUBLIC_ER_MODEL, API_KEY_ENV, "gemini-robotics", args, CAMERA_NAMES)
    assert callable(kwargs.pop("convert_fn"))
    assert kwargs == {}  # no optional tuning keys in this config -> base defaults


def test_build_convert_fn_is_er_conversion_bound_to_the_calibrated_joint_limits(
    config_dir, builder_spy, monkeypatch
):
    config_dir({"model": PUBLIC_ER_MODEL, "api_key_env": API_KEY_ENV})
    seen = []

    def spy_convert(er_response, joint_state, image_shape=None, joint_limits_deg=None):
        seen.append(dict(er_response=er_response, joint_limits_deg=joint_limits_deg))
        return dict(joint_state)

    monkeypatch.setattr(er_conversion, "convert_er_output_to_action", spy_convert)
    BACKEND_REGISTRY[BACKEND_NAME](_args(), CAMERA_NAMES, JOINT_LIMITS_DEG)
    convert_fn = builder_spy.calls[0][1]["convert_fn"]

    er_output = {"er_entries": [{"point": [100, 900]}], "image_shape": [480, 640]}
    assert convert_fn(er_output, dict(SERVO_POSE)) == SERVO_POSE
    assert seen == [dict(er_response=er_output, joint_limits_deg=JOINT_LIMITS_DEG)]


def test_optional_thinking_level_and_timeout_pass_through_to_the_action_source(config_dir, builder_spy):
    config_dir({"model": PUBLIC_ER_MODEL, "api_key_env": API_KEY_ENV,
                "thinking_level": "low", "timeout_s": 6.5, "notes": "ignored"})
    BACKEND_REGISTRY[BACKEND_NAME](_args(), CAMERA_NAMES, JOINT_LIMITS_DEG)
    kwargs = builder_spy.calls[0][1]
    assert kwargs["thinking_level"] == "low"
    assert kwargs["timeout_s"] == 6.5


def test_missing_config_file_raises_a_clear_system_exit(config_dir, builder_spy):
    with pytest.raises(SystemExit, match=r"gemini_robotics\.json"):
        BACKEND_REGISTRY[BACKEND_NAME](_args(), CAMERA_NAMES, JOINT_LIMITS_DEG)
    assert builder_spy.calls == []


def test_unparseable_config_file_raises_a_clear_system_exit(config_dir, builder_spy):
    config_dir("{not json")
    with pytest.raises(SystemExit, match=r"gemini_robotics\.json"):
        BACKEND_REGISTRY[BACKEND_NAME](_args(), CAMERA_NAMES, JOINT_LIMITS_DEG)
    assert builder_spy.calls == []


@pytest.mark.parametrize("missing", ["model", "api_key_env"])
def test_config_lacking_a_required_field_raises_system_exit_naming_it(config_dir, builder_spy, missing):
    config = {"model": PUBLIC_ER_MODEL, "api_key_env": API_KEY_ENV}
    del config[missing]
    config_dir(config)
    with pytest.raises(SystemExit, match=missing):
        BACKEND_REGISTRY[BACKEND_NAME](_args(), CAMERA_NAMES, JOINT_LIMITS_DEG)
    assert builder_spy.calls == []


@pytest.mark.parametrize("field,value", [
    ("model", ""), ("model", "   "), ("model", 7), ("model", None),
    ("api_key_env", ""), ("api_key_env", 3), ("api_key_env", "not a var-name"),
])
def test_blank_or_wrongly_typed_required_fields_raise_system_exit(config_dir, builder_spy, field, value):
    config = {"model": PUBLIC_ER_MODEL, "api_key_env": API_KEY_ENV}
    config[field] = value
    config_dir(config)
    with pytest.raises(SystemExit, match=field):
        BACKEND_REGISTRY[BACKEND_NAME](_args(), CAMERA_NAMES, JOINT_LIMITS_DEG)
    assert builder_spy.calls == []


def test_a_pasted_key_in_api_key_env_is_rejected_without_echoing_it(config_dir, builder_spy):
    config_dir({"model": PUBLIC_ER_MODEL, "api_key_env": "AIzaSy-pasted-key-value-0123"})
    with pytest.raises(SystemExit) as excinfo:
        BACKEND_REGISTRY[BACKEND_NAME](_args(), CAMERA_NAMES, JOINT_LIMITS_DEG)
    assert "AIzaSy-pasted-key-value-0123" not in str(excinfo.value)
    assert builder_spy.calls == []


@pytest.mark.parametrize("model", [
    "gemini-robotics-2",  # a native VLA name: would need a different decoder, not er_conversion
    "gemini-3.8-flash",  # not an embodied-reasoning model
])
def test_non_er_models_are_refused_because_this_path_only_converts_er_output(config_dir, builder_spy, model):
    config_dir({"model": model, "api_key_env": API_KEY_ENV})
    with pytest.raises(SystemExit, match="gemini-robotics-er-"):
        BACKEND_REGISTRY[BACKEND_NAME](_args(), CAMERA_NAMES, JOINT_LIMITS_DEG)
    assert builder_spy.calls == []


def test_the_live_api_only_streaming_er_model_is_refused(config_dir, builder_spy):
    config_dir({"model": "gemini-robotics-er-2-streaming-preview", "api_key_env": API_KEY_ENV})
    with pytest.raises(SystemExit, match="Live API"):
        BACKEND_REGISTRY[BACKEND_NAME](_args(), CAMERA_NAMES, JOINT_LIMITS_DEG)
    assert builder_spy.calls == []


@pytest.mark.parametrize("field,value", [
    ("thinking_level", "extreme"), ("thinking_level", 3),
    ("timeout_s", 0), ("timeout_s", -1), ("timeout_s", float("inf")), ("timeout_s", True),
    ("timeout_s", "10"),
])
def test_invalid_optional_tuning_values_raise_system_exit(config_dir, builder_spy, field, value):
    config = {"model": PUBLIC_ER_MODEL, "api_key_env": API_KEY_ENV, field: value}
    config_dir(json.dumps(config))  # Python's json round-trips inf as Infinity
    with pytest.raises(SystemExit, match=field):
        BACKEND_REGISTRY[BACKEND_NAME](_args(), CAMERA_NAMES, JOINT_LIMITS_DEG)
    assert builder_spy.calls == []


def test_null_optional_tuning_values_fall_back_to_base_defaults(config_dir, builder_spy):
    config_dir({"model": PUBLIC_ER_MODEL, "api_key_env": API_KEY_ENV,
                "thinking_level": None, "timeout_s": None})
    BACKEND_REGISTRY[BACKEND_NAME](_args(), CAMERA_NAMES, JOINT_LIMITS_DEG)
    assert set(builder_spy.calls[0][1]) == {"convert_fn"}


def test_shipped_config_targets_the_verified_public_er_2_model_and_env_var():
    config = json.loads(SHIPPED_CONFIG_PATH.read_text())
    assert config["model"] == PUBLIC_ER_MODEL
    assert config["api_key_env"] == "GEMINI_API_KEY"
    assert config.get("thinking_level") in (None, "minimal", "low", "medium", "high")
    # Honesty fields: this is VLAB-01 over ER + conversion, no native VLA model id.
    assert config["requirement"] == "VLAB-01"
    assert config["native_vla_model"] is None
    assert "er" in config["action_path"].lower()


def test_shipped_config_and_module_never_contain_an_api_key():
    for path in (SHIPPED_CONFIG_PATH, MODULE_PATH):
        assert "AIza" not in path.read_text(), path


def test_module_has_no_top_level_google_import():
    tree = ast.parse(MODULE_PATH.read_text())
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert not any(alias.name.startswith("google") for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("google")


def test_module_docstring_states_it_is_er_plus_conversion_not_native_vla():
    doc = gemini_robotics.__doc__
    assert "NOT" in doc and "native" in doc and "D-02" in doc
    assert PUBLIC_ER_MODEL in doc
    assert "VLAB-02" in doc  # documents how it relates to gemini-robotics-2
    assert "waitlist" in doc.lower() or "trusted tester" in doc.lower()


# =============================================================================
# Task 2: full mocked round trip -- registry -> shipped config -> real pinned
# google-genai SDK (in-process httpx.MockTransport) -> er_conversion -> action
# =============================================================================

ROUND_TRIP_KEY_ENV = "GEMINI_API_KEY"  # the shipped config's api_key_env
POINT_TEXT = '[{"point": [100, 900], "label": "red cube"}]'


class FakeWristCap:
    def __init__(self):
        self.released = False

    def isOpened(self):
        return True

    def read(self):
        return True, np.full((480, 640, 3), 90, dtype=np.uint8)

    def release(self):
        self.released = True


class FakeStereo:
    def read_left(self):
        return np.full((480, 640, 3), 40, dtype=np.uint8)

    def read_right(self):
        return np.full((480, 640, 3), 50, dtype=np.uint8)


def _wire_round_trip(monkeypatch, status=200, text=POINT_TEXT):
    """Patch only the hardware and the SDK's HTTP transport; everything else is real."""
    httpx = pytest.importorskip("httpx")
    pytest.importorskip("google.genai")
    monkeypatch.setenv(ROUND_TRIP_KEY_ENV, SECRET)
    monkeypatch.setattr(base_gemini, "SO101Follower", lambda config: MockRobot())
    monkeypatch.setattr(base_gemini, "StereoSplitCamera", lambda index=1: FakeStereo())
    monkeypatch.setattr(base_gemini.cv2, "VideoCapture", lambda index: FakeWristCap())
    requests_seen = []

    def handler(request):
        requests_seen.append(request)
        if status != 200:
            return httpx.Response(status, json={"error": {"code": status, "message": "denied",
                                                          "status": "PERMISSION_DENIED"}})
        return httpx.Response(200, json={
            "id": "interaction-1",
            "status": "completed",
            "steps": [{"type": "model_output", "content": [{"type": "text", "text": text}]}],
        })

    real_make_client = base_gemini._make_genai_client

    def make_client(api_key_value, timeout_s):
        return real_make_client(
            api_key_value, timeout_s, httpx_client=httpx.Client(transport=httpx.MockTransport(handler))
        )

    monkeypatch.setattr(base_gemini, "_make_genai_client", make_client)
    return requests_seen


def test_full_mocked_round_trip_returns_a_validated_finite_er_conversion_action(monkeypatch):
    requests_seen = _wire_round_trip(monkeypatch)

    handles = BACKEND_REGISTRY[BACKEND_NAME](_args(), CAMERA_NAMES, JOINT_LIMITS_DEG)
    source = handles.action_source
    action, model_version = source.get_action(dict(SERVO_POSE), "pick up the red cube")

    # Tagged as ER + conversion, never as a native policy.
    assert model_version == "gemini-robotics@gemini-er-conversion"
    assert model_version.endswith("@gemini-er-conversion")
    assert source.last_details["action_path"] == base_gemini.ACTION_PATH
    assert source.last_details["er_model"] == PUBLIC_ER_MODEL
    assert source.last_details["bridge_flags"] == []  # a real nudge, not a held no-op

    # Validated shape: full JOINT_ORDER, all finite, a small bounded step.
    assert list(action) == list(JOINT_ORDER)
    assert all(math.isfinite(value) for value in action.values())
    assert action != SERVO_POSE
    assert action["gripper"] == SERVO_POSE["gripper"]  # the conversion never grasps
    safe_action, flags = safety_validator.validate_action(action, SERVO_POSE, JOINT_LIMITS_DEG)
    assert list(safe_action) == list(JOINT_ORDER)
    assert all(math.isfinite(value) for value in safe_action.values())
    assert safe_action == action and flags == []  # the bounded nudge needs no clamping

    # The request went to the verified public ER 2 model with the configured settings.
    (request,) = requests_seen
    assert request.url.path == "/v1beta/interactions"
    body = json.loads(request.content)
    assert body["model"] == PUBLIC_ER_MODEL
    assert body["generation_config"]["thinking_level"] == "medium"
    assert "pick up the red cube" in request.content.decode()
    # The key travels only in the auth header, never in the body or the logged details.
    assert request.headers["x-goog-api-key"] == SECRET
    assert SECRET not in request.content.decode()
    assert SECRET not in json.dumps(source.last_details, default=repr)
    assert SECRET not in repr(source)


def test_round_trip_api_rejection_holds_position_and_never_raises(monkeypatch):
    # 403 is what the docs say an UNRESTRICTED key gets; the SDK does not retry it.
    requests_seen = _wire_round_trip(monkeypatch, status=403)

    handles = BACKEND_REGISTRY[BACKEND_NAME](_args(), CAMERA_NAMES, JOINT_LIMITS_DEG)
    action, model_version = handles.action_source.get_action(dict(SERVO_POSE), "pick up the red cube")

    assert action == SERVO_POSE
    assert model_version == "gemini-robotics@gemini-error-holding-position"
    assert len(requests_seen) == 1
    assert SECRET not in json.dumps(handles.action_source.last_details, default=repr)


def test_round_trip_teardown_handles_match_the_gemini_lane(monkeypatch):
    _wire_round_trip(monkeypatch)
    handles = BACKEND_REGISTRY[BACKEND_NAME](_args(), CAMERA_NAMES, JOINT_LIMITS_DEG)
    assert handles.shutdown is None  # default robot.disconnect(); no RobotClient on this lane
    assert handles.skip_camera_names == frozenset({"overhead"})
    wrist = handles.action_source.wrist_cap
    handles.action_source.close()
    assert wrist.released is True
