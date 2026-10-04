"""Tests for vla_bridge.backend_smoke (quick task 261004-dz7).

No Colab server, gRPC PolicyServer, Gemini API, robot, serial port or camera is
contacted: the HTTP adapter's `requests` calls, the gRPC `RobotClient` and the
google-genai client are all replaced by in-process fakes, and every hardware
construction seam runs inside `backend_smoke.synthetic_hardware()`.
"""

import contextlib
import queue
import re
import socket
import threading
import time
import types
from pathlib import Path

import cv2
import numpy as np
import pytest
import requests

import lerobot.async_inference.robot_client as lerobot_robot_client
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SO101Follower as RealSO101Follower

from vla_bridge import action_contract, backend_smoke, backends, robot_client, safety_validator, stereo_camera
from vla_bridge.action_contract import JOINT_ORDER
from vla_bridge.backends import GRPC_BRIDGE_BACKENDS, base_gemini, base_http_adapter, base_mllm, get_backend_names

ARM_JOINTS = [joint for joint in JOINT_ORDER if joint != "gripper"]
LIMITS = {joint: (-90.0, 90.0) for joint in JOINT_ORDER}
LIMITS["gripper"] = (0.0, 100.0)
STATE = {joint: (lo + hi) / 2 for joint, (lo, hi) in LIMITS.items()}
CKPT = "org/model"


def _offset(state: dict, delta: float) -> dict:
    return {joint: value + delta for joint, value in state.items()}


def _levels(checks) -> dict:
    return {check.name: check.level for check in checks}


def _evaluate(action, model_version="srv@abc", details=None, lane="http", latency_s=0.5,
              client_timeout_s=5.0, limits=None, state=None):
    return backend_smoke.evaluate_step(
        lane, action, model_version, details if details is not None else {"bridge_flags": []},
        state if state is not None else STATE, limits if limits is not None else LIMITS,
        latency_s, client_timeout_s, CKPT,
    )


# --- Lane classification ------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(GRPC_BRIDGE_BACKENDS))
def test_grpc_backends_are_in_the_grpc_lane(name):
    assert backend_smoke.lane_of(name) == "grpc"


@pytest.mark.parametrize("name", ["openvla-oft", "openvla", "octo", "rdt1b", "groot-n15", "wallx"])
def test_http_backends_are_in_the_http_lane(name):
    assert backend_smoke.lane_of(name) == "http"


@pytest.mark.parametrize("name", ["gemini-robotics", "gemini-robotics-2"])
def test_gemini_backends_are_in_the_gemini_lane(name):
    assert backend_smoke.lane_of(name) == "gemini"


def test_mllm_backend_is_refused_with_a_pointer_to_mllm_smoke():
    with pytest.raises(SystemExit) as exc:
        backend_smoke.lane_of("mllm-claude")
    assert "vla_bridge.mllm_smoke" in str(exc.value)


def test_scripted_backend_is_refused():
    with pytest.raises(SystemExit) as exc:
        backend_smoke.lane_of("scripted")
    assert "scripted" in str(exc.value)


def test_unknown_backend_lists_every_smokeable_name():
    with pytest.raises(SystemExit) as exc:
        backend_smoke.lane_of("no-such-backend")
    message = str(exc.value)
    for name in backend_smoke.GRPC_LANE | backend_smoke.HTTP_LANE | backend_smoke.GEMINI_LANE:
        assert name in message


def test_lanes_match_the_live_registry():
    lanes = (backend_smoke.GRPC_LANE, backend_smoke.HTTP_LANE, backend_smoke.GEMINI_LANE)
    assert backend_smoke.GRPC_LANE == frozenset(GRPC_BRIDGE_BACKENDS)
    for i, a in enumerate(lanes):
        for b in lanes[i + 1:]:
            assert not a & b
    covered = set().union(*lanes) | set(base_mllm.MLLM_BACKENDS) | {"scripted"}
    assert covered == set(get_backend_names())


# --- Synthetic hardware seams --------------------------------------------------------------------


def _seam_values():
    return {
        "base_http_adapter.SO101Follower": base_http_adapter.SO101Follower,
        "base_gemini.SO101Follower": base_gemini.SO101Follower,
        "base_http_adapter.StereoSplitCamera": base_http_adapter.StereoSplitCamera,
        "base_gemini.StereoSplitCamera": base_gemini.StereoSplitCamera,
        "robot_client.StereoSplitCamera": robot_client.StereoSplitCamera,
        "cv2.VideoCapture": cv2.VideoCapture,
        "make_robot_from_config": lerobot_robot_client.make_robot_from_config,
        "SO101Follower.connect": RealSO101Follower.__dict__.get("connect"),
        "StereoSplitCamera.__init__": stereo_camera.StereoSplitCamera.__dict__.get("__init__"),
    }


def test_synthetic_hardware_replaces_every_seam_and_restores_them():
    before = _seam_values()
    with backend_smoke.synthetic_hardware(STATE) as hw:
        config = SOFollowerRobotConfig(port="none", id="smoke")
        assert isinstance(base_http_adapter.SO101Follower(config), backend_smoke.SyntheticRobot)
        assert isinstance(base_gemini.SO101Follower(config), backend_smoke.SyntheticRobot)
        for cls in (base_http_adapter.StereoSplitCamera, base_gemini.StereoSplitCamera,
                    robot_client.StereoSplitCamera):
            assert isinstance(cls(index="x"), backend_smoke.SyntheticStereoCamera)
        assert isinstance(cv2.VideoCapture(0), backend_smoke.SyntheticWristCap)
        assert isinstance(lerobot_robot_client.make_robot_from_config(config), backend_smoke.SyntheticRobot)
        assert len(hw.robots_built) == 3
    assert _seam_values() == before


def test_synthetic_hardware_restores_seams_after_an_exception():
    before = _seam_values()
    with pytest.raises(ValueError):
        with backend_smoke.synthetic_hardware(STATE):
            raise ValueError("boom")
    assert _seam_values() == before


def test_every_hardware_construction_site_is_a_patched_module():
    root = Path(backend_smoke.__file__).resolve().parent
    candidates = [root / "robot_client.py", *sorted((root / "backends").glob("*.py"))]
    markers = ("SO101Follower(", "StereoSplitCamera(", "cv2.VideoCapture(", "RobotClient(")
    missing = []
    for path in candidates:
        stem = path.stem
        if stem.startswith(("mllm_", "base_mllm")) or stem == "scripted":
            continue
        if any(marker in path.read_text() for marker in markers):
            module = "vla_bridge.robot_client" if path.parent == root else f"vla_bridge.backends.{stem}"
            if module not in backend_smoke.PATCHED_MODULES:
                missing.append(module)
    assert not missing, f"unpatched hardware construction sites: {missing}"


def test_synthetic_robot_surface_and_raising_send_action():
    config = SOFollowerRobotConfig(port="none", id="smoke")
    config.cameras["camera1"] = object()
    log = backend_smoke.HardwareLog()
    robot = backend_smoke.SyntheticRobot(config, joint_state=STATE, hardware_log=log)
    features = robot.observation_features
    for joint in JOINT_ORDER:
        assert features[f"{joint}.pos"] is float
    assert features["camera1"] == (480, 640, 3)
    assert list(robot.action_features) == [f"{joint}.pos" for joint in JOINT_ORDER]
    robot.connect(calibrate=False)
    with robot.bus.torque_disabled():
        robot.bus.write_calibration(robot.calibration)
    obs = robot.get_observation()
    assert {f"{joint}.pos": STATE[joint] for joint in JOINT_ORDER}.items() <= obs.items()
    assert obs["camera1"].shape == (480, 640, 3) and obs["camera1"].dtype == np.uint8
    with pytest.raises(RuntimeError, match="never sends actions"):
        robot.send_action({"shoulder_pan.pos": 0.0})
    assert log.send_action_attempts == [{"shoulder_pan.pos": 0.0}]
    robot.disconnect()


def test_synthetic_cameras_serve_rig_sized_frames():
    ok, wrist = backend_smoke.SyntheticWristCap(0).read()
    assert ok and wrist.shape == (480, 640, 3)
    stereo = backend_smoke.SyntheticStereoCamera(index="x")
    assert stereo.read_left().shape == (600, 800, 3)
    assert stereo.read_right().shape == (600, 800, 3)
    assert not np.array_equal(stereo.read_left(), stereo.read_right())
    assert stereo.last_pair_metadata["source"] == "synthetic"


def test_tripwires_fire_on_the_real_hardware_classes():
    with backend_smoke.synthetic_hardware(STATE) as hw:
        with pytest.raises(RuntimeError, match="backend_smoke"):
            object.__new__(RealSO101Follower).connect()
        with pytest.raises(RuntimeError, match="backend_smoke"):
            stereo_camera.StereoSplitCamera(index=1)
    assert len(hw.tripwire_hits) == 2


# --- Per-step checks -----------------------------------------------------------------------------


def test_moving_action_with_server_version_has_no_fail():
    checks = _evaluate(_offset(STATE, 3.0))
    assert not [c for c in checks if c.level == "FAIL"]
    names = [c.name for c in checks]
    assert names[:2] == ["action returned", "not held"]
    assert "not saturated" in names and "latency" in names and "model_version" in names


@pytest.mark.parametrize("model_version, flags", [
    (f"{CKPT}@http-error-holding-position", ["http-error-holding-position"]),
    (f"{CKPT}@no action available, holding position", ["no action available, holding position"]),
    ("srv@abc", ["gemini-er-conversion-held-position"]),
    ("srv@abc", ["stale observation (40s > 30s), holding all joints"]),
])
def test_held_actions_fail_not_held(model_version, flags):
    checks = _evaluate(dict(STATE), model_version=model_version, details={"bridge_flags": flags})
    assert _levels(checks)["not held"] == "FAIL"


def test_noop_action_equal_to_the_state_fails_not_held():
    checks = _evaluate(_offset(STATE, 1e-9))
    assert _levels(checks)["not held"] == "FAIL"


def test_all_arm_joints_at_limits_fail_saturated():
    action = dict(STATE)
    for joint in ARM_JOINTS:
        action[joint] = LIMITS[joint][1] - 1.0
    checks = _evaluate(action)
    saturated = [c for c in checks if c.name == "saturated"]
    assert saturated and saturated[0].level == "FAIL"
    assert "units error" in saturated[0].detail


def test_some_arm_joints_pinned_warns_and_names_them():
    action = _offset(STATE, 3.0)
    action["shoulder_pan"] = LIMITS["shoulder_pan"][0]
    action["elbow_flex"] = LIMITS["elbow_flex"][1] + 20.0
    checks = _evaluate(action)
    warn = [c for c in checks if c.level == "WARN" and "shoulder_pan" in c.detail]
    assert warn and "elbow_flex" in warn[0].detail
    assert not [c for c in checks if c.name == "saturated"]


def test_gripper_at_its_limit_never_counts_as_saturated():
    action = _offset(STATE, 3.0)
    action["gripper"] = 100.0
    assert _levels(_evaluate(action))["not saturated"] == "PASS"


def test_grpc_saturation_uses_the_pre_validation_raw_output():
    raw = {joint: LIMITS[joint][1] + 50.0 for joint in JOINT_ORDER}
    returned = _offset(STATE, 3.0)  # post-validation: clamped back into range
    checks = _evaluate(returned, model_version=f"{CKPT}@unknown", lane="grpc",
                       details={"bridge_flags": [], "raw_model_output": raw})
    assert _levels(checks)["saturated"] == "FAIL"


def test_validator_classification():
    assert _levels(_evaluate(_offset(STATE, 3.0)))["validator"] == "PASS"
    clamped = _offset(STATE, 3.0)
    clamped["shoulder_pan"] = STATE["shoulder_pan"] + 60.0  # beyond the 40 deg per-step cap
    warn = [c for c in _evaluate(clamped) if c.name == "validator"][0]
    assert warn.level == "WARN" and "max per-step displacement" in warn.detail
    non_finite = _offset(STATE, 3.0)
    non_finite["wrist_flex"] = float("nan")
    assert _levels(_evaluate(non_finite))["action returned"] == "FAIL"
    rejected = backend_smoke.classify_validator_flags(["wrist_flex: rejected non-finite value (nan)"],
                                                      dict(STATE))
    assert rejected.level == "FAIL"
    stale = backend_smoke.classify_validator_flags(["stale observation (40s > 10s), holding all joints"],
                                                   dict(STATE))
    assert stale.level == "FAIL"
    partial = dict(STATE)
    del partial["gripper"]
    assert backend_smoke.classify_validator_flags([], partial).level == "FAIL"


def test_missing_joint_fails_schema_and_skips_action_checks():
    action = _offset(STATE, 3.0)
    del action["gripper"]
    checks = _evaluate(action)
    assert _levels(checks)["action returned"] == "FAIL"
    assert "not held" not in _levels(checks)


@pytest.mark.parametrize("lane, latency, timeout, level", [
    ("http", 1.0, 5.0, "PASS"),
    ("http", 4.5, 5.0, "WARN"),
    ("http", 5.0, 5.0, "FAIL"),
    ("gemini", 7.9, 10.0, "PASS"),
    ("grpc", 3.0, None, "PASS"),
    ("grpc", 12.0, None, "WARN"),
    ("grpc", safety_validator.STALE_ACTION_S + 1, None, "FAIL"),
])
def test_latency_classification(lane, latency, timeout, level):
    version = f"{CKPT}@unknown" if lane == "grpc" else "srv@abc"
    checks = _evaluate(_offset(STATE, 3.0), model_version=version, lane=lane, latency_s=latency,
                       client_timeout_s=timeout)
    assert _levels(checks)["latency"] == level


@pytest.mark.parametrize("lane, version, level", [
    ("grpc", f"{CKPT}@unknown", "PASS"),
    ("grpc", "other@unknown", "WARN"),
    ("http", "Zekai-Chen/wallx@ef8ce55f", "PASS"),
    ("http", f"{CKPT}@unknown", "WARN"),
    ("gemini", "gemini-robotics@gemini-er-conversion", "PASS"),
    ("gemini", "gemini-robotics@whatever", "WARN"),
])
def test_model_version_is_never_fail(lane, version, level):
    check = [c for c in _evaluate(_offset(STATE, 3.0), model_version=version, lane=lane)
             if c.name == "model_version"][0]
    assert check.level == level


def test_format_check():
    check = backend_smoke.Check("PASS", "not held", "moved")
    assert backend_smoke.format_check(check) == "PASS not held: moved"
    assert re.match(r"^(PASS|WARN|FAIL) ", backend_smoke.format_check(check))


# --- CLI end to end (offline) --------------------------------------------------------------------

WALLX_VERSION = "Zekai-Chen/wallx_pick_place_3cam_200ep@ef8ce55f9772"
GEMINI_SENTINEL = "AIzaSENTINEL-not-a-real-key-0123456789"


class FakeResponse:
    def __init__(self, body, status_code=200):
        self._body = body
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")

    def json(self):
        return self._body


@pytest.fixture
def hw_logs(monkeypatch):
    """Record every HardwareLog the CLI's synthetic_hardware() context yields."""
    logs = []
    real = backend_smoke.synthetic_hardware

    @contextlib.contextmanager
    def recording(joint_state):
        with real(joint_state) as log:
            logs.append(log)
            yield log

    monkeypatch.setattr(backend_smoke, "synthetic_hardware", recording)
    return logs


@pytest.fixture
def synthetic_state(mock_calibration_file):
    limits = action_contract.load_joint_limits_deg(mock_calibration_file)
    return limits, {joint: (float(lo) + float(hi)) / 2 for joint, (lo, hi) in limits.items()}


def _health_ok():
    return FakeResponse({"status": "ready", "protocol_version": 1,
                         "model": {"checkpoint": "Zekai-Chen/wallx_pick_place_3cam_200ep",
                                   "revision": "ef8ce55f9772", "action_space": "absolute_joint"}})


def _install_http(monkeypatch, get=None, post=None):
    calls = {"get": [], "post": []}

    def fake_get(url, timeout=None, **kwargs):
        calls["get"].append(url)
        return get(url) if callable(get) else get

    def fake_post(url, json=None, timeout=None, **kwargs):
        calls["post"].append({"url": url, "json": json, "timeout": timeout})
        return post(url) if callable(post) else post

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(requests, "post", fake_post)
    return calls


def _wallx_argv(calibration, *extra):
    return ["--backend", "wallx", "--server-address", "http://smoke.test",
            "--joint-calibration", str(calibration), *extra]


def test_http_pass_path(monkeypatch, capsys, hw_logs, mock_calibration_file, synthetic_state):
    _limits, state = synthetic_state
    action = _offset(state, 3.0)
    calls = _install_http(monkeypatch, get=_health_ok(),
                          post=FakeResponse({"action": action, "model_version": WALLX_VERSION}))
    code = backend_smoke.main(_wallx_argv(mock_calibration_file))
    out = capsys.readouterr().out
    assert code == 0, out
    for line in ("PASS server reachable", "PASS health ready", "PASS not held", "PASS not saturated",
                 "RESULT: PASS"):
        assert line in out
    assert "health model.revision" in out and "health model.action_space" in out
    assert calls["get"] == ["http://smoke.test/health"]
    payload = calls["post"][0]["json"]
    assert calls["post"][0]["url"] == "http://smoke.test/action"
    assert set(payload["images"]) == {"wrist", "overhead_left", "overhead_right"}
    assert payload["joint_state"] == pytest.approx(state)
    assert len(hw_logs) == 1
    assert hw_logs[0].send_action_attempts == [] and hw_logs[0].tripwire_hits == []
    assert hw_logs[0].robots_built


def test_http_health_not_ready_stops_before_action(monkeypatch, capsys, mock_calibration_file):
    calls = _install_http(monkeypatch, get=FakeResponse({"status": "loading"}), post=None)
    code = backend_smoke.main(_wallx_argv(mock_calibration_file))
    out = capsys.readouterr().out
    assert code == 1
    assert "FAIL health ready" in out and "RESULT: FAIL" in out
    assert calls["post"] == []


def test_http_unreachable_server(monkeypatch, capsys, mock_calibration_file):
    def refuse(url):
        raise requests.ConnectionError("connection refused")

    calls = _install_http(monkeypatch, get=refuse, post=None)
    code = backend_smoke.main(_wallx_argv(mock_calibration_file))
    out = capsys.readouterr().out
    assert code == 1 and "FAIL server reachable" in out
    assert calls["post"] == []


@pytest.mark.parametrize("post", ["timeout", "http500"])
def test_http_held_action_fails(monkeypatch, capsys, mock_calibration_file, post):
    def timeout(url):
        raise requests.Timeout("read timed out")

    response = timeout if post == "timeout" else FakeResponse({"error": "boom"}, status_code=500)
    _install_http(monkeypatch, get=_health_ok(), post=response)
    code = backend_smoke.main(_wallx_argv(mock_calibration_file))
    out = capsys.readouterr().out
    assert code == 1
    assert re.search(r"FAIL not held: .*http-error-holding-position", out)
    assert "RESULT: FAIL" in out


def test_http_saturated_action_fails(monkeypatch, capsys, mock_calibration_file, synthetic_state):
    limits, state = synthetic_state
    action = dict(state)
    for joint in ARM_JOINTS:
        action[joint] = limits[joint][1] + 50.0
    _install_http(monkeypatch, get=_health_ok(),
                  post=FakeResponse({"action": action, "model_version": WALLX_VERSION}))
    code = backend_smoke.main(_wallx_argv(mock_calibration_file))
    out = capsys.readouterr().out
    assert code == 1 and "FAIL saturated" in out


def test_missing_calibration_warns_and_uses_neutral_limits(monkeypatch, capsys, tmp_path):
    state = {joint: (lo + hi) / 2 for joint, (lo, hi) in backend_smoke.NEUTRAL_JOINT_LIMITS_DEG.items()}
    _install_http(monkeypatch, get=_health_ok(),
                  post=FakeResponse({"action": _offset(state, 3.0), "model_version": WALLX_VERSION}))
    code = backend_smoke.main(_wallx_argv(tmp_path / "missing.json"))
    out = capsys.readouterr().out
    assert code == 0, out
    assert "WARNING: no usable follower calibration" in out


# gRPC lane fakes: modelled on test_robot_client.py's FakeTimedAction / FakeBridgeClient.


class FakeTimedAction:
    def __init__(self, action, timestamp: float, timestep: int = 0):
        self._action = action
        self._timestamp = timestamp
        self._timestep = timestep

    def get_action(self):
        return self._action

    def get_timestamp(self) -> float:
        return self._timestamp

    def get_timestep(self) -> int:
        return self._timestep


class FakeRobotClient:
    """`RobotClient` surface `connect_bridge()` and `BridgeActionSource` touch. Its
    robot comes from lerobot's `make_robot_from_config`, so the smoke's seam runs."""

    instances: list = []
    actions_to_send: list = []
    action_age_s: float = 0.0

    def __init__(self, config):
        self.config = config
        self.robot = lerobot_robot_client.make_robot_from_config(config.robot)
        self.robot.connect()
        self.policy_config = types.SimpleNamespace(lerobot_features={})
        self.action_queue = queue.Queue()
        self.action_queue_lock = threading.Lock()
        self.latest_action_lock = threading.Lock()
        self.latest_action = -1
        self.action_chunk_size = -1
        self.must_go = threading.Event()
        self.must_go.set()
        self.observations = []
        self.stopped = False
        FakeRobotClient.instances.append(self)

    def start(self):
        return True

    def receive_actions(self):
        for i, action in enumerate(FakeRobotClient.actions_to_send):
            self.action_queue.put(FakeTimedAction(action, time.time() - FakeRobotClient.action_age_s, i))

    def _ready_to_send_observation(self):
        return True

    def control_loop_observation(self, task):
        obs = self.robot.get_observation()
        self.observations.append(obs)
        return obs

    def _action_tensor_to_action_dict(self, action_tensor):
        return {key: action_tensor[i] for i, key in enumerate(self.robot.action_features)}

    def stop(self):
        self.stopped = True
        self.robot.disconnect()


@pytest.fixture
def listening_port():
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen()
    yield server.getsockname()[1]
    server.close()


@pytest.fixture
def closed_port():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


@pytest.fixture
def fake_robot_client(monkeypatch):
    FakeRobotClient.instances = []
    FakeRobotClient.actions_to_send = []
    FakeRobotClient.action_age_s = 0.0
    monkeypatch.setattr(robot_client, "RobotClient", FakeRobotClient)
    return FakeRobotClient


def _grpc_argv(port, calibration, *extra):
    return ["--backend", "smolvla", "--server-address", f"127.0.0.1:{port}", "--checkpoint", "x/y",
            "--joint-calibration", str(calibration), *extra]


def test_grpc_unreachable_server_never_builds(monkeypatch, capsys, closed_port, mock_calibration_file):
    built = []
    monkeypatch.setitem(backends.BACKEND_REGISTRY, "smolvla", lambda *a, **k: built.append(a))
    code = backend_smoke.main(_grpc_argv(closed_port, mock_calibration_file))
    out = capsys.readouterr().out
    assert code == 1 and "FAIL server reachable" in out
    assert built == []


def test_real_robot_client_runs_on_the_synthetic_robot(closed_port):
    """The REAL lerobot RobotClient (not a fake) builds on SyntheticRobot via the
    make_robot_from_config seam; a refused loopback handshake returns None."""
    limits = dict(backend_smoke.NEUTRAL_JOINT_LIMITS_DEG)
    state = {joint: (lo + hi) / 2 for joint, (lo, hi) in limits.items()}
    args = types.SimpleNamespace(port="synthetic-no-serial", robot_id="backend_smoke",
                                 stereo_camera_index="synthetic", server_address=f"127.0.0.1:{closed_port}",
                                 checkpoint="x/y", instruction="smoke", camera_map=None)
    with backend_smoke.synthetic_hardware(state) as hw:
        handles = backends.BACKEND_REGISTRY["smolvla"](args, {0: "wrist"}, limits)
    assert handles is None
    assert len(hw.robots_built) == 1
    assert "camera1" in hw.robots_built[0].observation_features
    assert not hw.robots_built[0].is_connected  # connect_bridge's cleanup stop() disconnected it
    assert hw.send_action_attempts == [] and hw.tripwire_hits == []


def test_grpc_pass_path(capsys, hw_logs, listening_port, fake_robot_client, mock_calibration_file,
                        synthetic_state):
    _limits, state = synthetic_state
    fake_robot_client.actions_to_send = [[state[joint] + 2.0 for joint in JOINT_ORDER]]
    code = backend_smoke.main(_grpc_argv(listening_port, mock_calibration_file, "--wait-s", "5",
                                         "--poll-interval-s", "0.05"))
    out = capsys.readouterr().out
    assert code == 0, out
    assert "PASS server reachable" in out and "PASS not held" in out and "RESULT: PASS" in out
    client = fake_robot_client.instances[0]
    assert isinstance(client.robot, backend_smoke.SyntheticRobot)
    assert "camera1" in client.robot.observation_features
    assert client.observations and client.observations[0]["camera2"].shape[1] == 640
    assert client.stopped
    assert hw_logs[0].send_action_attempts == [] and hw_logs[0].tripwire_hits == []


def test_grpc_latency_is_the_observation_to_action_age(capsys, listening_port, fake_robot_client,
                                                        mock_calibration_file, synthetic_state):
    _limits, state = synthetic_state
    fake_robot_client.actions_to_send = [[state[joint] + 2.0 for joint in JOINT_ORDER]]
    fake_robot_client.action_age_s = 12.0  # slow round trip, but under STALE_ACTION_S
    code = backend_smoke.main(_grpc_argv(listening_port, mock_calibration_file, "--poll-interval-s", "0.05"))
    out = capsys.readouterr().out
    assert code == 0, out
    assert re.search(r"WARN latency: 12\.\d\d s over the 5\.0 s budget", out)
    assert "observation-to-action age 12." in out


def test_grpc_no_action_within_wait_fails_and_tears_down(capsys, listening_port, fake_robot_client,
                                                         mock_calibration_file):
    code = backend_smoke.main(_grpc_argv(listening_port, mock_calibration_file, "--wait-s", "1",
                                         "--poll-interval-s", "0.1"))
    out = capsys.readouterr().out
    assert code == 1
    assert re.search(r"FAIL not held: no action chunk within 1(\.0)? s", out)
    assert fake_robot_client.instances[0].stopped


# Gemini lane.


def test_gemini_without_confirm_spend_refuses(monkeypatch, capsys, mock_calibration_file):
    monkeypatch.setenv("GEMINI_API_KEY", GEMINI_SENTINEL)
    clients, built = [], []
    monkeypatch.setattr(base_gemini, "_make_genai_client", lambda *a, **k: clients.append(a))
    monkeypatch.setitem(backends.BACKEND_REGISTRY, "gemini-robotics", lambda *a, **k: built.append(a))
    code = backend_smoke.main(["--backend", "gemini-robotics", "--joint-calibration",
                               str(mock_calibration_file)])
    captured = capsys.readouterr()
    assert code == 2
    assert "gemini-robotics-er-2-preview" in captured.out
    assert "--confirm-spend" in captured.out and "paid" in captured.out
    assert clients == [] and built == []
    assert GEMINI_SENTINEL not in captured.out + captured.err


def test_gemini_confirmed_path(monkeypatch, capsys, hw_logs, mock_calibration_file):
    _model, api_key_env, _kwargs = base_gemini.load_er_backend_config("gemini_robotics", "gemini-robotics")
    monkeypatch.setenv(api_key_env, GEMINI_SENTINEL)
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return types.SimpleNamespace(output_text='[{"point": [250, 750], "label": "cube"}]')

    fake_client = types.SimpleNamespace(interactions=types.SimpleNamespace(create=create))
    monkeypatch.setattr(base_gemini, "_make_genai_client", lambda api_key, timeout_s, **k: fake_client)
    code = backend_smoke.main(["--backend", "gemini-robotics", "--steps", "2", "--instruction", "cube",
                               "--joint-calibration", str(mock_calibration_file), "--confirm-spend"])
    captured = capsys.readouterr()
    assert len(calls) == 2
    assert re.search(r"model_version: gemini-robotics@", captured.out)
    assert GEMINI_SENTINEL not in captured.out + captured.err
    # At the synthetic midpoint state the ER conversion produces a real (bounded) move.
    assert code == 0, captured.out
    assert "PASS not held" in captured.out and "RESULT: PASS" in captured.out
    assert hw_logs[0].send_action_attempts == [] and hw_logs[0].tripwire_hits == []


# Argument handling.


def test_unknown_backend_exits_listing_smokeable_names():
    with pytest.raises(SystemExit) as exc:
        backend_smoke.main(["--backend", "no-such-backend"])
    assert "smolvla" in str(exc.value) and "wallx" in str(exc.value)


def test_mllm_backend_points_to_mllm_smoke():
    with pytest.raises(SystemExit) as exc:
        backend_smoke.main(["--backend", "mllm-claude"])
    assert "vla_bridge.mllm_smoke" in str(exc.value)


@pytest.mark.parametrize("argv", [
    ["--backend", "smolvla", "--server-address", "127.0.0.1:5173"],
    ["--backend", "wallx", "--server-address", "127.0.0.1:5000"],
    ["--backend", "wallx", "--server-address", "https://x.test", "--camera-map", "camera1=wrist"],
    ["--backend", "wallx", "--server-address", "https://x.test", "--steps", "6"],
])
def test_argument_errors_exit_2(argv):
    with pytest.raises(SystemExit) as exc:
        backend_smoke.main(argv)
    assert exc.value.code == 2
