"""Tests for `control/vla_bridge/backends/` (Plan 14-01, PIPE-01).

Covers the shared Phase 14 scaffolding every later backend module builds on:
the auto-discovering `BACKEND_REGISTRY`, `load_backend_config()`, and the two
shared per-tier construction helpers (`base_grpc.build_grpc_action_source()`
for the Tier A/B gRPC-bridge lane, `base_http_adapter.HttpAdapterActionSource`
/ `build_http_adapter_backend()` for the Tier C resident-HTTP-adapter lane).

Never touches real hardware, a real camera, a real gRPC bridge, or the
network -- every external seam is a monkeypatched double.
"""

import base64
import inspect
import json
import pkgutil
import threading
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import requests

from conftest import MockRobot
from vla_bridge import backends, robot_client, safety_validator
from vla_bridge.action_contract import JOINT_ORDER
from vla_bridge.backends import BackendHandles, base_grpc, base_http_adapter
from vla_bridge.backends.base_http_adapter import HttpAdapterActionSource

JOINT_STATE = {joint: float(i) for i, joint in enumerate(JOINT_ORDER)}


# --- register() / register_grpc_backend() ------------------------------------


@pytest.fixture
def fresh_registry(monkeypatch):
    """Swap in empty registry containers so these tests never leak fake
    names into the real, process-wide BACKEND_REGISTRY."""
    registry: dict = {}
    grpc_names: set = set()
    monkeypatch.setattr(backends, "BACKEND_REGISTRY", registry)
    monkeypatch.setattr(backends, "GRPC_BRIDGE_BACKENDS", grpc_names)
    return registry, grpc_names


def test_register_adds_to_registry_and_returns_input_unchanged(fresh_registry):
    registry, grpc_names = fresh_registry

    def build(args, camera_names, joint_limits_deg):
        return "handles"

    returned = backends.register("foo")(build)

    assert returned is build
    assert registry["foo"] is build
    assert "foo" not in grpc_names


def test_register_grpc_backend_adds_to_registry_and_grpc_set(fresh_registry):
    registry, grpc_names = fresh_registry

    class SomeBackend:
        pass

    returned = backends.register_grpc_backend("bar")(SomeBackend)

    assert returned is SomeBackend
    assert registry["bar"] is SomeBackend
    assert grpc_names == {"bar"}


def test_register_rejects_a_second_different_backend_under_the_same_name(fresh_registry):
    """Two backend modules silently claiming the same --backend name would
    make whichever imports last win -- fail loudly instead."""
    backends.register("dup")(lambda *a: None)
    with pytest.raises(ValueError, match="dup"):
        backends.register("dup")(lambda *a: None)


def test_get_backend_names_is_sorted(fresh_registry):
    backends.register("zeta")(object)
    backends.register("alpha")(object)
    assert backends.get_backend_names() == ["alpha", "zeta"]


# --- _discover_backends() -------------------------------------------------------


def test_discover_backends_imports_only_non_base_non_private_siblings(monkeypatch):
    fake_modules = [
        pkgutil.ModuleInfo(None, "smolvla", False),
        pkgutil.ModuleInfo(None, "base_grpc", False),
        pkgutil.ModuleInfo(None, "base_http_adapter", False),
        pkgutil.ModuleInfo(None, "_helpers", False),
        pkgutil.ModuleInfo(None, "act", False),
    ]
    seen_paths = []

    def fake_iter_modules(path):
        seen_paths.append(path)
        return iter(fake_modules)

    imported = []
    monkeypatch.setattr(backends.pkgutil, "iter_modules", fake_iter_modules)
    monkeypatch.setattr(backends.importlib, "import_module", imported.append)

    backends._discover_backends()

    assert seen_paths == [backends.__path__]
    assert imported == ["vla_bridge.backends.smolvla", "vla_bridge.backends.act"]


# --- load_backend_config() ------------------------------------------------------


def test_load_backend_config_returns_none_for_missing_file(monkeypatch, tmp_path):
    monkeypatch.setattr(backends, "BACKEND_CONFIG_DIR", tmp_path)
    assert backends.load_backend_config("does-not-exist") is None


def test_load_backend_config_default_dir_missing_name_returns_none():
    """Same never-raise contract against the real, unpatched config dir."""
    assert backends.load_backend_config("does-not-exist") is None


def test_load_backend_config_reads_valid_json(monkeypatch, tmp_path):
    monkeypatch.setattr(backends, "BACKEND_CONFIG_DIR", tmp_path)
    (tmp_path / "pi0.json").write_text(json.dumps({"checkpoint": "lerobot/pi0_base"}))
    assert backends.load_backend_config("pi0") == {"checkpoint": "lerobot/pi0_base"}


def test_load_backend_config_returns_none_for_invalid_json(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(backends, "BACKEND_CONFIG_DIR", tmp_path)
    (tmp_path / "broken.json").write_text("{not json")
    assert backends.load_backend_config("broken") is None
    assert "WARNING" in capsys.readouterr().out


def test_load_backend_config_returns_none_for_non_object_json(monkeypatch, tmp_path):
    monkeypatch.setattr(backends, "BACKEND_CONFIG_DIR", tmp_path)
    (tmp_path / "listy.json").write_text("[1, 2, 3]")
    assert backends.load_backend_config("listy") is None


def test_load_backend_config_rejects_path_traversal_names(monkeypatch, tmp_path):
    configs = tmp_path / "configs"
    configs.mkdir()
    (tmp_path / "secret.json").write_text(json.dumps({"leak": True}))
    monkeypatch.setattr(backends, "BACKEND_CONFIG_DIR", configs)
    assert backends.load_backend_config("../secret") is None


# --- HttpAdapterActionSource ----------------------------------------------------


class FakeWristCap:
    def __init__(self, ok=True):
        self.ok = ok
        self.read_count = 0
        self.released = False

    def read(self):
        self.read_count += 1
        if not self.ok:
            return False, None
        return True, np.full((48, 64, 3), 7, dtype=np.uint8)

    def release(self):
        self.released = True


class FakeStereo:
    def __init__(self, ok=True):
        self.ok = ok
        self.left_reads = 0
        self.right_reads = 0
        self.last_pair_metadata = {}

    def read_left(self):
        self.left_reads += 1
        return np.full((30, 40, 3), 11, dtype=np.uint8) if self.ok else None

    def read_right(self):
        self.right_reads += 1
        return np.full((30, 40, 3), 22, dtype=np.uint8) if self.ok else None


class FakeResponse:
    def __init__(self, payload=None, status_code=200, raise_on_json=False):
        self._payload = payload
        self.status_code = status_code
        self._raise_on_json = raise_on_json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")

    def json(self):
        if self._raise_on_json:
            raise requests.JSONDecodeError("Expecting value", "<html>", 0)
        return self._payload


def _source(**kwargs):
    defaults = dict(
        endpoint_url="https://adapter.example.ngrok.app/",
        checkpoint="openvla/openvla-7b",
        wrist_cap=FakeWristCap(),
        stereo_camera=FakeStereo(),
    )
    defaults.update(kwargs)
    return HttpAdapterActionSource(**defaults)


def test_http_adapter_holds_position_on_connection_error(monkeypatch):
    def raise_connection_error(*args, **kwargs):
        raise requests.ConnectionError("tunnel down")

    monkeypatch.setattr(base_http_adapter.requests, "post", raise_connection_error)
    source = _source()

    action, model_version = source.get_action(JOINT_STATE, "pick the cube")

    assert action == JOINT_STATE
    assert action is not JOINT_STATE  # a copy, never the caller's own dict
    assert model_version == "openvla/openvla-7b@http-error-holding-position"


@pytest.mark.parametrize(
    "response",
    [
        FakeResponse(status_code=500),
        FakeResponse(raise_on_json=True),
        FakeResponse({"model_version": "x"}),  # missing "action"
        FakeResponse({"action": None}),
        FakeResponse(["not", "an", "object"]),
    ],
    ids=["http-500", "non-json", "missing-action", "null-action", "non-object-body"],
)
def test_http_adapter_holds_position_on_bad_response(monkeypatch, response):
    monkeypatch.setattr(base_http_adapter.requests, "post", lambda *a, **k: response)

    action, model_version = _source().get_action(JOINT_STATE, "pick the cube")

    assert action == JOINT_STATE
    assert model_version == "openvla/openvla-7b@http-error-holding-position"


def test_http_adapter_holds_position_on_timeout(monkeypatch):
    def raise_timeout(*args, **kwargs):
        raise requests.Timeout("slow colab")

    monkeypatch.setattr(base_http_adapter.requests, "post", raise_timeout)
    action, model_version = _source().get_action(JOINT_STATE, "pick")
    assert action == JOINT_STATE
    assert model_version.endswith("@http-error-holding-position")


def test_http_adapter_calls_convert_fn_with_parsed_action_and_joint_state(monkeypatch):
    raw = [0.01, 0.0, -0.02, 0.0, 0.0, 0.1, 1.0]
    monkeypatch.setattr(
        base_http_adapter.requests,
        "post",
        lambda *a, **k: FakeResponse({"action": raw, "model_version": "openvla-7b@abc123"}),
    )
    converted = {joint: 1.5 for joint in JOINT_ORDER}
    calls = []

    def convert_fn(parsed_action, joint_state):
        calls.append((parsed_action, joint_state))
        return converted

    source = _source(convert_fn=convert_fn)
    action, model_version = source.get_action(JOINT_STATE, "pick the cube")

    assert calls == [(raw, JOINT_STATE)]
    assert action == converted
    assert model_version == "openvla-7b@abc123"
    assert source.last_details["raw_model_output"] == raw
    assert source.last_details["bridge_flags"] == []
    assert source.last_details["observation"] is None


def test_http_adapter_identity_when_no_convert_fn_and_joint_dict_action(monkeypatch):
    joint_action = {joint: 2.0 for joint in JOINT_ORDER}
    monkeypatch.setattr(
        base_http_adapter.requests, "post", lambda *a, **k: FakeResponse({"action": joint_action})
    )
    action, model_version = _source().get_action(JOINT_STATE, "pick")
    assert action == joint_action
    assert model_version == "openvla/openvla-7b@unknown"


def test_http_adapter_holds_position_when_final_action_is_not_a_joint_dict(monkeypatch):
    """A raw list with no convert_fn (or a convert_fn returning a partial or
    non-numeric dict) must never reach safety_validator.validate_action()
    -- whose math.isfinite() would raise TypeError on a string, and whose
    missing-joint skip would produce a partial goal_pos lerobot rejects."""
    monkeypatch.setattr(
        base_http_adapter.requests, "post", lambda *a, **k: FakeResponse({"action": [1, 2, 3]})
    )
    action, model_version = _source().get_action(JOINT_STATE, "pick")
    assert action == JOINT_STATE
    assert model_version == "openvla/openvla-7b@action-format-error-holding-position"

    partial = {"shoulder_pan": 1.0}
    action, model_version = _source(convert_fn=lambda a, s: partial).get_action(JOINT_STATE, "pick")
    assert action == JOINT_STATE
    assert model_version.endswith("@action-format-error-holding-position")

    stringy = {joint: "1.0" for joint in JOINT_ORDER}
    action, model_version = _source(convert_fn=lambda a, s: stringy).get_action(JOINT_STATE, "pick")
    assert action == JOINT_STATE
    assert model_version.endswith("@action-format-error-holding-position")


def test_http_adapter_holds_position_when_convert_fn_raises(monkeypatch):
    monkeypatch.setattr(
        base_http_adapter.requests, "post", lambda *a, **k: FakeResponse({"action": [0.0] * 7})
    )

    def boom(parsed_action, joint_state):
        raise ValueError("IK diverged")

    action, model_version = _source(convert_fn=boom).get_action(JOINT_STATE, "pick")
    assert action == JOINT_STATE
    assert model_version == "openvla/openvla-7b@action-conversion-error-holding-position"


def test_http_adapter_posts_expected_payload_and_reads_each_camera_once(monkeypatch):
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured.update(url=url, json=json, timeout=timeout)
        return FakeResponse({"action": {joint: 0.0 for joint in JOINT_ORDER}})

    monkeypatch.setattr(base_http_adapter.requests, "post", fake_post)
    wrist, stereo = FakeWristCap(), FakeStereo()
    source = _source(wrist_cap=wrist, stereo_camera=stereo, timeout_s=3.5)

    source.get_action(JOINT_STATE, "pick the cube")

    assert captured["url"] == "https://adapter.example.ngrok.app/action"
    assert captured["timeout"] == 3.5
    body = captured["json"]
    assert set(body) == {"images", "joint_state", "instruction"}
    assert body["joint_state"] == JOINT_STATE
    assert body["instruction"] == "pick the cube"
    assert set(body["images"]) == {"wrist", "overhead_left", "overhead_right"}
    decoded_left = cv2.imdecode(
        np.frombuffer(base64.b64decode(body["images"]["overhead_left"]), np.uint8), cv2.IMREAD_COLOR
    )
    assert decoded_left.shape == (30, 40, 3)
    assert decoded_left[0, 0, 0] == 11
    assert (wrist.read_count, stereo.left_reads, stereo.right_reads) == (1, 1, 1)


def test_http_adapter_downsizes_large_frames_before_encoding(monkeypatch):
    captured = {}

    class BigWrist(FakeWristCap):
        def read(self):
            return True, np.zeros((1080, 1920, 3), dtype=np.uint8)

    def fake_post(url, json=None, timeout=None):
        captured["json"] = json
        return FakeResponse({"action": {joint: 0.0 for joint in JOINT_ORDER}})

    monkeypatch.setattr(base_http_adapter.requests, "post", fake_post)
    _source(wrist_cap=BigWrist()).get_action(JOINT_STATE, "pick")

    wrist = cv2.imdecode(
        np.frombuffer(base64.b64decode(captured["json"]["images"]["wrist"]), np.uint8), cv2.IMREAD_COLOR
    )
    assert max(wrist.shape[:2]) == robot_client.OBSERVATION_MAX_SIDE


@pytest.mark.parametrize("which", ["wrist", "stereo"])
def test_http_adapter_holds_position_without_posting_on_camera_failure(monkeypatch, which):
    posted = []
    monkeypatch.setattr(base_http_adapter.requests, "post", lambda *a, **k: posted.append(1))
    source = _source(
        wrist_cap=FakeWristCap(ok=(which != "wrist")),
        stereo_camera=FakeStereo(ok=(which != "stereo")),
    )

    action, model_version = source.get_action(JOINT_STATE, "pick")

    assert posted == []
    assert action == JOINT_STATE
    assert model_version == "openvla/openvla-7b@camera-read-failed-holding-position"


def test_http_adapter_inflight_guard_rejects_overlapping_call(monkeypatch):
    entered, release = threading.Event(), threading.Event()

    def slow_post(*args, **kwargs):
        entered.set()
        release.wait(timeout=2.0)
        return FakeResponse({"action": {joint: 0.0 for joint in JOINT_ORDER}})

    monkeypatch.setattr(base_http_adapter.requests, "post", slow_post)
    source = _source()
    worker = threading.Thread(target=source.get_action, args=(JOINT_STATE, "pick"))
    worker.start()
    assert entered.wait(timeout=2.0)

    action, model_version = source.get_action(JOINT_STATE, "pick")
    release.set()
    worker.join(timeout=2.0)

    assert action == JOINT_STATE
    assert model_version == "openvla/openvla-7b@http-request-inflight-holding-position"


def test_http_adapter_close_releases_wrist_capture():
    wrist = FakeWristCap()
    _source(wrist_cap=wrist).close()
    assert wrist.released is True


# --- base_grpc.build_grpc_action_source() --------------------------------------


def _args(**overrides):
    defaults = dict(
        port="/dev/tty.usbmodem1",
        robot_id="soarm_follower_02",
        server_address="0.tcp.ngrok.io:12345",
        instruction="pick the cube",
        stereo_camera_index=1,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_build_grpc_action_source_returns_none_when_bridge_unreachable(monkeypatch):
    monkeypatch.setattr(base_grpc.robot_client, "connect_bridge", lambda *a, **k: None)
    result = base_grpc.build_grpc_action_source(
        "pi0", "lerobot/pi0_base", _args(), {0: "wrist", 1: "overhead"}, {}
    )
    assert result is None


def test_build_grpc_action_source_forwards_policy_type_and_wraps_bridge_source(monkeypatch):
    calls = {}
    fake_stereo = object()
    fake_robot = object()
    fake_client = SimpleNamespace(robot=fake_robot, _stereo_camera=fake_stereo, stop=lambda: None)

    def fake_connect_bridge(server_address, checkpoint, **kwargs):
        calls.update(server_address=server_address, checkpoint=checkpoint, **kwargs)
        return fake_client

    monkeypatch.setattr(base_grpc.robot_client, "connect_bridge", fake_connect_bridge)
    joint_limits = {"shoulder_pan": (-90.0, 90.0)}

    handles = base_grpc.build_grpc_action_source(
        "pi0", "lerobot/pi0_base", _args(), {4: "wrist", 1: "overhead"}, joint_limits
    )

    assert isinstance(handles, BackendHandles)
    assert calls["server_address"] == "0.tcp.ngrok.io:12345"
    assert calls["checkpoint"] == "lerobot/pi0_base"
    assert calls["policy_type"] == "pi0"
    assert calls["task"] == "pick the cube"
    assert calls["policy_device"] == "cuda"
    assert calls["stereo_camera_index"] == 1
    assert calls["wrist_camera_index"] == 4
    robot_config = calls["robot_config"]
    assert robot_config.port == "/dev/tty.usbmodem1"
    assert robot_config.id == "soarm_follower_02"
    assert robot_config.max_relative_target == safety_validator.MAX_RELATIVE_TARGET_DEG
    assert isinstance(handles.action_source, robot_client.BridgeActionSource)
    assert handles.action_source.client is fake_client
    assert handles.action_source.checkpoint == "lerobot/pi0_base"
    assert handles.action_source.joint_limits_deg is joint_limits
    assert handles.robot is fake_robot
    assert handles.stereo_camera is fake_stereo
    assert handles.skip_camera_names == frozenset({"overhead"})
    # Plan 14-02: RobotClient.stop() (disconnect + channel close) replaces a
    # bare robot.disconnect() at episode end.
    assert handles.shutdown is fake_client.stop


# --- base_http_adapter.build_http_adapter_backend() -----------------------------


class RecordingMockRobot(MockRobot):
    instances: list = []

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.connect_kwargs = None
        self.calibration_written = False
        robot = self

        class _Bus:
            from contextlib import contextmanager

            @contextmanager
            def torque_disabled(self_inner):
                yield

            def write_calibration(self_inner, calibration):
                robot.calibration_written = True

        self.bus = _Bus()
        RecordingMockRobot.instances.append(self)

    def connect(self, calibrate: bool = False) -> None:
        self.connect_kwargs = dict(calibrate=calibrate)
        self.connected = True


def _patch_http_lane_hardware(monkeypatch):
    opened = {}
    RecordingMockRobot.instances = []
    monkeypatch.setattr(base_http_adapter, "SO101Follower", RecordingMockRobot)

    def fake_stereo(index=1):
        opened["stereo_index"] = index
        return FakeStereo()

    def fake_video_capture(index):
        opened["wrist_index"] = index
        return FakeWristCap()

    monkeypatch.setattr(base_http_adapter, "StereoSplitCamera", fake_stereo)
    monkeypatch.setattr(base_http_adapter.cv2, "VideoCapture", fake_video_capture)
    return opened


def test_build_http_adapter_backend_constructs_robot_cameras_and_action_source(monkeypatch):
    opened = _patch_http_lane_hardware(monkeypatch)

    def convert(parsed_action, joint_state):
        return parsed_action

    handles = base_http_adapter.build_http_adapter_backend(
        "https://adapter.example", "openvla/openvla-7b", _args(stereo_camera_index=3),
        {5: "wrist", 3: "overhead"}, convert_fn=convert,
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
    assert isinstance(handles.action_source, HttpAdapterActionSource)
    assert handles.action_source.endpoint_url == "https://adapter.example"
    assert handles.action_source.checkpoint == "openvla/openvla-7b"
    assert handles.action_source.convert_fn is convert
    assert handles.action_source.stereo_camera is handles.stereo_camera
    assert handles.skip_camera_names == frozenset({"overhead"})


def test_build_http_adapter_backend_requires_a_wrist_camera_before_touching_robot(monkeypatch):
    _patch_http_lane_hardware(monkeypatch)
    with pytest.raises(SystemExit, match="wrist"):
        base_http_adapter.build_http_adapter_backend(
            "https://adapter.example", "openvla/openvla-7b", _args(), {3: "overhead"}
        )
    assert RecordingMockRobot.instances == []


# --- Plan 14-02: "scripted" and "smolvla" registry entries ---------------------


def _scripted_args(**overrides):
    defaults = dict(port="/dev/tty.usbmodem1", robot_id="soarm_follower_02")
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_scripted_backend_builds_connected_robot_and_scripted_source(monkeypatch):
    from vla_bridge.backends import scripted

    RecordingMockRobot.instances = []
    monkeypatch.setattr(scripted, "SO101Follower", RecordingMockRobot)

    handles = backends.BACKEND_REGISTRY["scripted"](
        _scripted_args(), {0: "wrist", 1: "overhead"}, {}
    )

    assert isinstance(handles, BackendHandles)
    robot = RecordingMockRobot.instances[-1]
    assert handles.robot is robot
    assert robot.connect_kwargs == dict(calibrate=False)
    assert robot.calibration_written is True
    assert robot.config.port == "/dev/tty.usbmodem1"
    assert robot.config.id == "soarm_follower_02"
    assert robot.config.max_relative_target == safety_validator.MAX_RELATIVE_TARGET_DEG
    assert isinstance(handles.action_source, scripted.ScriptedActionSource)
    assert handles.stereo_camera is None
    assert handles.skip_camera_names == frozenset()


def test_scripted_is_a_plain_backend_and_smolvla_is_a_grpc_backend():
    assert "scripted" in backends.BACKEND_REGISTRY
    assert "scripted" not in backends.GRPC_BRIDGE_BACKENDS
    assert "smolvla" in backends.BACKEND_REGISTRY
    assert "smolvla" in backends.GRPC_BRIDGE_BACKENDS


def test_run_vla_episode_scripted_action_source_is_the_relocated_class():
    """Backward compatibility: run_vla_episode.ScriptedActionSource must keep
    resolving -- to the single relocated copy, not a duplicate."""
    import run_vla_episode
    from vla_bridge.backends import scripted

    assert run_vla_episode.ScriptedActionSource is scripted.ScriptedActionSource


def _record_grpc_build(monkeypatch):
    calls = {}
    sentinel = object()

    def fake_build(policy_type, checkpoint, args, camera_names, joint_limits_deg):
        calls.update(policy_type=policy_type, checkpoint=checkpoint, args=args,
                     camera_names=camera_names, joint_limits_deg=joint_limits_deg)
        return sentinel

    monkeypatch.setattr(base_grpc, "build_grpc_action_source", fake_build)
    return calls, sentinel


def test_smolvla_backend_uses_cli_checkpoint_when_given(monkeypatch, tmp_path):
    monkeypatch.setattr(backends, "BACKEND_CONFIG_DIR", tmp_path)
    (tmp_path / "smolvla.json").write_text(json.dumps({"checkpoint": "config/ckpt"}))
    calls, sentinel = _record_grpc_build(monkeypatch)
    args = _args(checkpoint="cli/ckpt")
    camera_names = {0: "wrist"}
    joint_limits = {"gripper": (0.0, 100.0)}

    result = backends.BACKEND_REGISTRY["smolvla"](args, camera_names, joint_limits)

    assert result is sentinel
    assert calls == dict(policy_type="smolvla", checkpoint="cli/ckpt", args=args,
                         camera_names=camera_names, joint_limits_deg=joint_limits)


def test_smolvla_backend_falls_back_to_config_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setattr(backends, "BACKEND_CONFIG_DIR", tmp_path)
    (tmp_path / "smolvla.json").write_text(json.dumps({"checkpoint": "config/ckpt"}))
    calls, _ = _record_grpc_build(monkeypatch)

    backends.BACKEND_REGISTRY["smolvla"](_args(checkpoint=None), {0: "wrist"}, {})

    assert calls["policy_type"] == "smolvla"
    assert calls["checkpoint"] == "config/ckpt"


def test_smolvla_backend_without_any_checkpoint_exits_before_connecting(monkeypatch, tmp_path):
    monkeypatch.setattr(backends, "BACKEND_CONFIG_DIR", tmp_path)  # no smolvla.json
    calls, _ = _record_grpc_build(monkeypatch)

    with pytest.raises(SystemExit, match="checkpoint"):
        backends.BACKEND_REGISTRY["smolvla"](_args(checkpoint=None), {0: "wrist"}, {})
    assert calls == {}


def test_shipped_smolvla_config_names_the_production_checkpoint():
    # Verified live on the HF Hub (2026-10-01): public, lerobot "type": "smolvla",
    # camera1/camera2/camera3 inputs matching connect_bridge()'s wiring.
    assert backends.load_backend_config("smolvla") == {
        "checkpoint": "victorvanhalst/smolvla_so101_cube"
    }


# --- Conformance (PIPE-01): durable as the registry grows ----------------------


def test_registry_contains_the_baseline_backends():
    assert {"scripted", "smolvla"}.issubset(backends.get_backend_names())
    assert "smolvla" in backends.GRPC_BRIDGE_BACKENDS
    assert "scripted" not in backends.GRPC_BRIDGE_BACKENDS


def _action_source_bases():
    from vla_bridge.backends.scripted import ScriptedActionSource

    # Every backend from Wave 3 onward is built on one of these shared bases,
    # so checking the bases covers the whole registry -- no per-backend
    # conformance test is needed. Plan 14-13 (Gemini foundation) MUST add its
    # GeminiActionSourceBase here when it lands.
    return [
        robot_client.BridgeActionSource,
        HttpAdapterActionSource,
        ScriptedActionSource,
    ]


@pytest.mark.parametrize("cls", _action_source_bases(), ids=lambda cls: cls.__name__)
def test_action_source_bases_match_the_action_source_protocol_signature(cls):
    import run_vla_episode

    expected = ["self", "joint_state", "instruction"]
    protocol_params = list(inspect.signature(run_vla_episode.ActionSource.get_action).parameters)
    assert protocol_params == expected
    assert list(inspect.signature(cls.get_action).parameters) == expected
