"""Tests for `control/start_pose.py` (quick 261025-wbw: fixed home pose).

Uses `conftest.py`'s `MockRobot` and `mock_calibration_file` -- no real
hardware, serial port or calibration cache is ever touched, and the real
`control/start_pose.json` is never read or written (conftest's autouse
`start_pose_file` fixture repoints `START_POSE_PATH` at a temp file).
"""

import json
import sys

import pytest

import start_pose
from conftest import MockRobot
from vla_bridge import action_contract
from vla_bridge.action_contract import JOINT_ORDER

VALID_JOINTS = {
    "shoulder_pan": 10.0,
    "shoulder_lift": -45.5,
    "elbow_flex": 60.25,
    "wrist_flex": -20.0,
    "wrist_roll": 150.0,
    "gripper": 30.0,
}


@pytest.fixture
def limits(mock_calibration_file):
    return action_contract.load_joint_limits_deg(mock_calibration_file)


def _write_pose(path, joints=None, captured_at="2026-10-25T12:00:00Z"):
    doc = {"joints": dict(VALID_JOINTS if joints is None else joints)}
    if captured_at is not None:
        doc["captured_at"] = captured_at
    path.write_text(json.dumps(doc))
    return path


# --- validate_start_pose ---


def test_validate_returns_floats_in_joint_order_for_a_valid_pose(limits):
    shuffled = {joint: VALID_JOINTS[joint] for joint in reversed(JOINT_ORDER)}
    shuffled["gripper"] = 30  # int is accepted and converted
    result = start_pose.validate_start_pose(shuffled, limits)
    assert list(result) == list(JOINT_ORDER)
    assert result["gripper"] == 30.0
    assert all(type(value) is float for value in result.values())


def test_validate_names_a_missing_joint(limits):
    joints = dict(VALID_JOINTS)
    del joints["elbow_flex"]
    with pytest.raises(start_pose.StartPoseError, match="elbow_flex"):
        start_pose.validate_start_pose(joints, limits)


def test_validate_names_an_unknown_joint(limits):
    joints = dict(VALID_JOINTS, elbow_twist=1.0)
    with pytest.raises(start_pose.StartPoseError, match="elbow_twist"):
        start_pose.validate_start_pose(joints, limits)


@pytest.mark.parametrize("bad_value", ["10", None, True, False, [1.0]])
def test_validate_rejects_non_numeric_and_bool_values(limits, bad_value):
    joints = dict(VALID_JOINTS, wrist_flex=bad_value)
    with pytest.raises(start_pose.StartPoseError, match="wrist_flex"):
        start_pose.validate_start_pose(joints, limits)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_validate_rejects_non_finite_values(limits, bad_value):
    joints = dict(VALID_JOINTS, shoulder_lift=bad_value)
    with pytest.raises(start_pose.StartPoseError, match="shoulder_lift.*finite"):
        start_pose.validate_start_pose(joints, limits)


def test_validate_rejects_an_out_of_limit_value_naming_joint_value_and_limits(limits):
    joints = dict(VALID_JOINTS, shoulder_pan=80.0)
    with pytest.raises(start_pose.StartPoseError) as excinfo:
        start_pose.validate_start_pose(joints, limits)
    message = str(excinfo.value)
    assert "shoulder_pan" in message
    assert "80.0" in message
    assert "-70.33" in message and "70.33" in message


def test_validate_rejects_a_joint_with_no_limits_entry(limits):
    del limits["wrist_roll"]
    with pytest.raises(start_pose.StartPoseError, match="wrist_roll.*no safety limits"):
        start_pose.validate_start_pose(dict(VALID_JOINTS), limits)


def test_validate_lists_every_problem_in_one_error(limits):
    joints = dict(VALID_JOINTS, shoulder_pan=500.0, gripper=float("nan"), extra=1.0)
    del joints["elbow_flex"]
    with pytest.raises(start_pose.StartPoseError) as excinfo:
        start_pose.validate_start_pose(joints, limits)
    message = str(excinfo.value)
    for name in ("shoulder_pan", "gripper", "extra", "elbow_flex"):
        assert name in message


def test_validate_rejects_a_non_dict_joints_value(limits):
    with pytest.raises(start_pose.StartPoseError):
        start_pose.validate_start_pose([1.0] * 6, limits)


# --- load_start_pose ---


def test_load_returns_validated_joints_for_a_valid_file(tmp_path, limits):
    path = _write_pose(tmp_path / "pose.json")
    assert start_pose.load_start_pose(limits, path) == VALID_JOINTS


def test_load_missing_file_names_path_and_capture_command(tmp_path, limits):
    path = tmp_path / "absent.json"
    with pytest.raises(start_pose.StartPoseError) as excinfo:
        start_pose.load_start_pose(limits, path)
    assert str(path) in str(excinfo.value)
    assert "python start_pose.py" in str(excinfo.value)


@pytest.mark.parametrize(
    "raw_text",
    [
        "{not json",
        "[]",
        '{"captured_at": "2026-10-25T12:00:00Z"}',
        '{"captured_at": "2026-10-25T12:00:00Z", "joints": [1, 2, 3]}',
        json.dumps({"joints": VALID_JOINTS}),
        json.dumps({"captured_at": "", "joints": VALID_JOINTS}),
        json.dumps({"captured_at": 123, "joints": VALID_JOINTS}),
        json.dumps({"captured_at": "2026-10-25T12:00:00Z", "joints": VALID_JOINTS}).replace(
            "10.0", "NaN"
        ),
        json.dumps({"captured_at": "2026-10-25T12:00:00Z", "joints": VALID_JOINTS}).replace(
            "10.0", "Infinity"
        ),
    ],
    ids=[
        "broken_json",
        "not_an_object",
        "missing_joints",
        "joints_not_a_dict",
        "missing_captured_at",
        "empty_captured_at",
        "non_string_captured_at",
        "nan_literal",
        "infinity_literal",
    ],
)
def test_load_rejects_invalid_files_with_path_and_capture_command(tmp_path, limits, raw_text):
    path = tmp_path / "pose.json"
    path.write_text(raw_text)
    with pytest.raises(start_pose.StartPoseError) as excinfo:
        start_pose.load_start_pose(limits, path)
    assert str(path) in str(excinfo.value)
    assert "python start_pose.py" in str(excinfo.value)


def test_load_rejects_nan_literal_as_non_finite(tmp_path, limits):
    path = tmp_path / "pose.json"
    path.write_text('{"captured_at": "x", "joints": {"shoulder_pan": NaN}}')
    with pytest.raises(start_pose.StartPoseError, match="shoulder_pan.*finite"):
        start_pose.load_start_pose(limits, path)


def test_load_rejects_an_unreadable_path(tmp_path, limits):
    with pytest.raises(start_pose.StartPoseError, match="python start_pose.py"):
        start_pose.load_start_pose(limits, tmp_path)  # a directory, not a file


def test_load_without_path_reads_start_pose_path_at_call_time(tmp_path, limits, monkeypatch):
    path = _write_pose(tmp_path / "patched.json", dict(VALID_JOINTS, gripper=55.0))
    monkeypatch.setattr(start_pose, "START_POSE_PATH", path)
    assert start_pose.load_start_pose(limits)["gripper"] == 55.0


def test_conftest_fixture_points_start_pose_path_at_an_all_zero_temp_pose(start_pose_file, limits):
    assert start_pose.START_POSE_PATH == start_pose_file
    assert start_pose_file.name == "start_pose.json"
    assert start_pose.load_start_pose(limits) == {joint: 0.0 for joint in JOINT_ORDER}


# --- capture_start_pose ---


def test_capture_writes_rounded_joints_units_and_utc_timestamp(tmp_path, limits):
    robot = MockRobot()
    robot._positions = dict(VALID_JOINTS, shoulder_pan=10.123456)
    path = tmp_path / "pose.json"
    doc = start_pose.capture_start_pose(robot, limits, path)

    written = json.loads(path.read_text())
    assert written == doc
    assert path.read_text().endswith("\n")
    assert path.read_text().startswith('{\n  "')  # indent=2
    assert list(written["joints"]) == list(JOINT_ORDER)
    assert written["joints"]["shoulder_pan"] == 10.12
    assert written["units"] == action_contract.ACTION_UNITS
    from datetime import datetime

    datetime.strptime(written["captured_at"], "%Y-%m-%dT%H:%M:%SZ")
    assert written["captured_at"].endswith("Z")
    assert "robot_id" not in written and "id" not in written


def test_captured_file_round_trips_through_load(tmp_path, limits):
    robot = MockRobot()
    robot._positions = dict(VALID_JOINTS)
    path = tmp_path / "pose.json"
    start_pose.capture_start_pose(robot, limits, path)
    assert start_pose.load_start_pose(limits, path) == VALID_JOINTS


def test_capture_never_sends_an_action(tmp_path, limits):
    robot = MockRobot()
    start_pose.capture_start_pose(robot, limits, tmp_path / "pose.json")
    assert robot.sent_actions == []


def test_capture_refuses_to_overwrite_without_force(tmp_path, limits):
    path = tmp_path / "pose.json"
    path.write_text("original bytes")
    with pytest.raises(FileExistsError, match="--force"):
        start_pose.capture_start_pose(MockRobot(), limits, path)
    assert path.read_text() == "original bytes"


def test_capture_overwrites_with_force(tmp_path, limits):
    path = tmp_path / "pose.json"
    path.write_text("original bytes")
    robot = MockRobot()
    robot._positions = dict(VALID_JOINTS)
    start_pose.capture_start_pose(robot, limits, path, force=True)
    assert json.loads(path.read_text())["joints"] == VALID_JOINTS


def test_capture_refuses_an_out_of_limit_reading_and_writes_nothing(tmp_path, limits):
    robot = MockRobot()
    robot._positions = dict(VALID_JOINTS, gripper=-1.0)
    path = tmp_path / "pose.json"
    with pytest.raises(start_pose.StartPoseError, match="gripper"):
        start_pose.capture_start_pose(robot, limits, path)
    assert not path.exists()


def test_capture_without_path_writes_start_pose_path(tmp_path, limits, monkeypatch):
    path = tmp_path / "default.json"
    monkeypatch.setattr(start_pose, "START_POSE_PATH", path)
    start_pose.capture_start_pose(MockRobot(), limits)
    assert path.exists()


# --- joints_out_of_tolerance ---


def test_out_of_tolerance_is_empty_when_every_joint_is_within_tolerance_inclusive():
    target = {"shoulder_pan": 10.0, "gripper": 30.0}
    reached = {"shoulder_pan": 15.0, "gripper": 25.0}
    assert start_pose.joints_out_of_tolerance(target, reached, 5.0) == {}


def test_out_of_tolerance_reports_signed_error_for_joints_beyond_it():
    target = {"shoulder_pan": 10.0, "elbow_flex": 0.0, "gripper": 30.0}
    reached = {"shoulder_pan": 16.5, "elbow_flex": 1.0, "gripper": 20.0}
    assert start_pose.joints_out_of_tolerance(target, reached, 5.0) == {
        "shoulder_pan": 6.5,
        "gripper": -10.0,
    }


def test_out_of_tolerance_reports_none_for_a_joint_absent_from_reached():
    assert start_pose.joints_out_of_tolerance({"wrist_roll": 0.0}, {}, 5.0) == {"wrist_roll": None}


def test_out_of_tolerance_defaults_to_the_module_tolerance():
    assert start_pose.START_POSE_TOLERANCE == 5.0
    assert start_pose.joints_out_of_tolerance({"gripper": 0.0}, {"gripper": 5.0}) == {}
    assert start_pose.joints_out_of_tolerance({"gripper": 0.0}, {"gripper": 5.1}) != {}


# --- StartPoseNotReachedError ---


def test_not_reached_error_names_missed_joints_tolerance_and_hint():
    error = start_pose.StartPoseNotReachedError(
        target={"shoulder_pan": 10.0, "wrist_roll": 0.0},
        reached={"shoulder_pan": 0.0},
        missed={"shoulder_pan": -10.0, "wrist_roll": None},
        tolerance=5.0,
    )
    message = str(error)
    assert error.missed == {"shoulder_pan": -10.0, "wrist_roll": None}
    assert "shoulder_pan" in message and "wrist_roll" in message
    assert "not reported" in message
    assert "5.0" in message
    assert "not starting the episode" in message
    assert "python start_pose.py --force" in message


# --- capture CLI main() ---


class _CountingRobot(MockRobot):
    constructed = 0
    last = None

    def __init__(self, config=None):
        super().__init__()
        self.config = config
        type(self).constructed += 1
        type(self).last = self


@pytest.fixture
def cli(monkeypatch, limits):
    _CountingRobot.constructed = 0
    _CountingRobot.last = None
    monkeypatch.setattr(start_pose, "SO101Follower", _CountingRobot)
    monkeypatch.setattr(start_pose, "SOFollowerRobotConfig", lambda **kwargs: kwargs)
    monkeypatch.setattr(action_contract, "load_joint_limits_deg", lambda: dict(limits))

    def run(*argv):
        monkeypatch.setattr(sys, "argv", ["start_pose.py", *argv])
        start_pose.main()

    return run


def test_cli_refuses_existing_file_without_force_before_constructing_a_robot(cli, start_pose_file, capsys):
    original = start_pose_file.read_text()
    with pytest.raises(SystemExit):
        cli("/dev/fake", "fake_robot")
    assert "--force" in capsys.readouterr().err
    assert _CountingRobot.constructed == 0
    assert start_pose_file.read_text() == original


def test_cli_with_force_captures_and_disconnects(cli, start_pose_file, capsys):
    cli("/dev/fake", "fake_robot", "--force")
    robot = _CountingRobot.last
    assert robot.config == {"port": "/dev/fake", "id": "fake_robot"}
    assert robot.connected is False
    assert json.loads(start_pose_file.read_text())["joints"] == {joint: 0.0 for joint in JOINT_ORDER}
    assert f"Saved → {start_pose_file}" in capsys.readouterr().out


def test_cli_writes_a_new_file_without_force_when_none_exists(cli, tmp_path, monkeypatch):
    path = tmp_path / "fresh.json"
    monkeypatch.setattr(start_pose, "START_POSE_PATH", path)
    cli("/dev/fake", "fake_robot")
    assert path.exists()
    assert _CountingRobot.last.connected is False


def test_cli_disconnects_even_when_capture_raises(cli, start_pose_file, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("bus fault")

    monkeypatch.setattr(start_pose, "capture_start_pose", boom)
    with pytest.raises(RuntimeError, match="bus fault"):
        cli("/dev/fake", "fake_robot", "--force")
    assert _CountingRobot.last.connected is False


def test_cli_turns_an_out_of_limit_reading_into_system_exit(cli, start_pose_file, monkeypatch):
    original_init = _CountingRobot.__init__

    def init(self, config=None):
        original_init(self, config)
        self._positions["gripper"] = -1.0

    monkeypatch.setattr(_CountingRobot, "__init__", init)
    with pytest.raises(SystemExit, match="gripper"):
        cli("/dev/fake", "fake_robot", "--force")
    assert _CountingRobot.last.connected is False


def test_cli_falls_back_to_device_map_follower(cli, start_pose_file, monkeypatch):
    import run_vla_episode

    monkeypatch.setattr(
        run_vla_episode, "_load_device_map",
        lambda: {"follower": {"port": "/dev/mapped", "id": "mapped_robot"}},
    )
    cli("--force")
    assert _CountingRobot.last.config == {"port": "/dev/mapped", "id": "mapped_robot"}


def test_cli_errors_without_args_or_device_map(cli, start_pose_file, monkeypatch, capsys):
    import run_vla_episode

    monkeypatch.setattr(run_vla_episode, "_load_device_map", lambda: None)
    with pytest.raises(SystemExit):
        cli("--force")
    assert "detect_devices.py" in capsys.readouterr().err
    assert _CountingRobot.constructed == 0
