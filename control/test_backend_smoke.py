"""Tests for vla_bridge.backend_smoke (quick task 261004-dz7).

No Colab server, gRPC PolicyServer, Gemini API, robot, serial port or camera is
contacted: the HTTP adapter's `requests` calls, the gRPC `RobotClient` and the
google-genai client are all replaced by in-process fakes, and every hardware
construction seam runs inside `backend_smoke.synthetic_hardware()`.
"""

import re
from pathlib import Path

import cv2
import numpy as np
import pytest

import lerobot.async_inference.robot_client as lerobot_robot_client
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SO101Follower as RealSO101Follower

from vla_bridge import backend_smoke, robot_client, safety_validator, stereo_camera
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
