"""Tests for `--backend gemini-robotics-2` (Plan 14-15, VLAB-02).

The backend is the public Gemini Robotics-ER 2 model plus Plan 14-13's local
`er_conversion` layer, NOT the native Gemini Robotics 2 VLA (14-CONTEXT.md D-02
supersession). Per 14-14-SUMMARY.md "VLAB-01 vs VLAB-02 distinction" it calls
the SAME model with the SAME settings as `gemini-robotics` (VLAB-01); the
parity tests below pin that so the two cannot drift apart silently. No test here calls the real Gemini API: the builder is
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
from vla_bridge.backends import gemini_robotics, gemini_robotics_2

BACKEND_NAME = "gemini-robotics-2"
CONFIG_NAME = "gemini_robotics_2"
PUBLIC_ER_MODEL = "gemini-robotics-er-2-preview"
API_KEY_ENV = "SOARM_TEST_GEMINI_ROBOTICS_2_KEY"
SECRET = "AIzaSy-TEST-SECRET-gemini-robotics-2-0123456789"
CAMERA_NAMES = {5: "wrist", 3: "overhead"}
JOINT_LIMITS_DEG = {joint: (-180.0, 180.0) for joint in JOINT_ORDER}
JOINT_LIMITS_DEG["gripper"] = (0.0, 100.0)
SERVO_POSE = {"shoulder_pan": 10.0, "shoulder_lift": -20.0, "elbow_flex": 30.0,
              "wrist_flex": 40.0, "wrist_roll": 0.0, "gripper": 30.0}
MODULE_PATH = Path(gemini_robotics_2.__file__)
SHIPPED_CONFIG_PATH = backends.BACKEND_CONFIG_DIR / f"{CONFIG_NAME}.json"
VLAB01_CONFIG_PATH = backends.BACKEND_CONFIG_DIR / f"{gemini_robotics.CONFIG_NAME}.json"


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


def test_gemini_robotics_2_is_registered_with_plain_register_not_the_grpc_lane():
    assert BACKEND_NAME in BACKEND_REGISTRY
    assert BACKEND_REGISTRY[BACKEND_NAME] is gemini_robotics_2.build
    assert BACKEND_REGISTRY[BACKEND_NAME] is not BACKEND_REGISTRY[gemini_robotics.BACKEND_NAME]
    assert BACKEND_NAME not in GRPC_BRIDGE_BACKENDS


def test_build_calls_the_shared_gemini_builder_with_the_config_model_and_key_env(config_dir, builder_spy):
    config_dir({"model": PUBLIC_ER_MODEL, "api_key_env": API_KEY_ENV})
    args = _args()

    result = BACKEND_REGISTRY[BACKEND_NAME](args, CAMERA_NAMES, JOINT_LIMITS_DEG)

    assert result is builder_spy.sentinel
    ((positional, kwargs),) = builder_spy.calls
    assert positional == (PUBLIC_ER_MODEL, API_KEY_ENV, "gemini-robotics-2", args, CAMERA_NAMES)
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
    with pytest.raises(SystemExit, match=r"gemini_robotics_2\.json"):
        BACKEND_REGISTRY[BACKEND_NAME](_args(), CAMERA_NAMES, JOINT_LIMITS_DEG)
    assert builder_spy.calls == []


def test_unparseable_config_file_raises_a_clear_system_exit(config_dir, builder_spy):
    config_dir("{not json")
    with pytest.raises(SystemExit, match=r"gemini_robotics_2\.json"):
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
    "gemini-robotics-2",  # the native Gemini Robotics 2 VLA name: needs a different decoder, not er_conversion
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
    # Honesty fields: this is VLAB-02 over ER + conversion, no native VLA model id.
    assert config["requirement"] == "VLAB-02"
    assert config["product_name"] == "Gemini Robotics 2"
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
    doc = gemini_robotics_2.__doc__
    assert "NOT" in doc and "native" in doc and "D-02" in doc
    assert PUBLIC_ER_MODEL in doc
    assert "VLAB-01" in doc and "14-14" in doc  # states the equivalence and points back
    assert "waitlist" in doc.lower() or "trusted tester" in doc.lower()


# =============================================================================
# VLAB-01 / VLAB-02 parity: same model, same settings, different identity
# (14-14-SUMMARY.md "VLAB-01 vs VLAB-02 distinction" -- no artificial variation)
# =============================================================================

PARITY_KEYS = ("model", "api_key_env", "thinking_level", "timeout_s", "action_path", "native_vla_model")


def test_shipped_config_matches_vlab01_on_every_behavioural_setting():
    vlab02 = json.loads(SHIPPED_CONFIG_PATH.read_text())
    vlab01 = json.loads(VLAB01_CONFIG_PATH.read_text())
    for key in PARITY_KEYS:
        assert vlab02.get(key) == vlab01.get(key), key
    assert (vlab01["requirement"], vlab02["requirement"]) == ("VLAB-01", "VLAB-02")


def test_both_backends_resolve_to_identical_builder_calls_except_the_label(builder_spy):
    args = _args()
    gemini_robotics.build(args, CAMERA_NAMES, JOINT_LIMITS_DEG)
    gemini_robotics_2.build(args, CAMERA_NAMES, JOINT_LIMITS_DEG)
    (pos1, kw1), (pos2, kw2) = builder_spy.calls
    assert pos1[2] == "gemini-robotics" and pos2[2] == "gemini-robotics-2"
    assert pos1[:2] + pos1[3:] == pos2[:2] + pos2[3:]
    kw1.pop("convert_fn"), kw2.pop("convert_fn")
    assert kw1 == kw2 and kw2["thinking_level"] == "medium"


def test_module_identity_constants_are_distinct_from_vlab01():
    assert (gemini_robotics_2.BACKEND_NAME, gemini_robotics_2.CONFIG_NAME, gemini_robotics_2.CHECKPOINT_LABEL) == (
        "gemini-robotics-2", "gemini_robotics_2", "gemini-robotics-2")
    assert gemini_robotics_2.CONFIG_NAME != gemini_robotics.CONFIG_NAME
