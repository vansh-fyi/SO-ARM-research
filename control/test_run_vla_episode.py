"""Tests for `control/run_vla_episode.py` (VLAHW-02, VLAHW-03).

Uses Plan 11-01's `mock_robot` fixture (from `conftest.py`) and a fake camera
object -- never a real robot or real `cv2.VideoCapture`.
"""

import json

import numpy as np
import pytest

import run_vla_episode
from run_vla_episode import ScriptedActionSource, _build_camera_names, run_episode
from vla_bridge import action_contract
from vla_bridge.io_logger import IOLogger


class _NeverOpensCapture:
    """Stand-in for `cv2.VideoCapture` that never actually opens a device --
    keeps the bridge-selection tests below hermetic (no real camera probing)."""

    def __init__(self, *args, **kwargs):
        pass

    def isOpened(self):
        return False

    def release(self):
        pass


class FakeCamera:
    def read(self):
        return True, np.zeros((4, 4, 3), dtype=np.uint8)


CAMERA_NAMES = {0: "wrist", 1: "overhead"}


def _fake_caps():
    return {0: FakeCamera(), 1: FakeCamera()}


def test_scripted_action_source_toggles_gripper_after_30_steps():
    source = ScriptedActionSource()
    joint_state = {
        "shoulder_pan": 0.0,
        "shoulder_lift": 0.0,
        "elbow_flex": 0.0,
        "wrist_flex": 0.0,
        "wrist_roll": 0.0,
        "gripper": 50.0,
    }

    first_action, model_version = source.get_action(joint_state, "instruction")
    assert first_action["gripper"] == 0.0
    assert model_version == "scripted-dry-run-v1"
    for joint in joint_state:
        if joint != "gripper":
            assert first_action[joint] == joint_state[joint]

    # Calls 2..30 (internal step counter 1..29) stay in the same 30-step
    # block as the first call (internal step 0) -> gripper stays 0.0.
    for _ in range(29):
        action, _ = source.get_action(joint_state, "instruction")
        assert action["gripper"] == 0.0

    # Call 31 (internal step counter reaches 30) crosses into the next
    # 30-step block -> gripper toggles to 100.0.
    action, _ = source.get_action(joint_state, "instruction")
    assert action["gripper"] == 100.0


def test_loop_writes_termination_json_with_max_steps_reason(tmp_path, mock_robot, mock_calibration_file):
    action_source = ScriptedActionSource()
    caps = _fake_caps()
    joint_limits_deg = action_contract.load_joint_limits_deg(mock_calibration_file)
    with IOLogger(tmp_path, CAMERA_NAMES) as io_logger:
        run_episode(
            mock_robot,
            caps,
            CAMERA_NAMES,
            io_logger,
            action_source,
            instruction="Pick up the red cube",
            max_steps=3,
            control_hz=100.0,  # fast, so the test doesn't sleep meaningfully
            joint_limits_deg=joint_limits_deg,
        )

    termination = json.loads((tmp_path / "termination.json").read_text())
    assert termination["reason"] == "max_steps_reached"
    assert termination["steps_completed"] == 3


def test_keyboard_interrupt_triggers_return_to_start_before_disconnect(
    tmp_path, mock_robot, mock_calibration_file, monkeypatch
):
    call_order = []

    def fake_move_to_positions(robot, targets, kp, control_freq, max_seconds, **kwargs):
        call_order.append("move_to_positions")

    monkeypatch.setattr(run_vla_episode, "move_to_positions", fake_move_to_positions)

    original_disconnect = mock_robot.disconnect

    def tracked_disconnect():
        call_order.append("disconnect")
        original_disconnect()

    mock_robot.disconnect = tracked_disconnect

    class InterruptingActionSource:
        def __init__(self):
            self.calls = 0

        def get_action(self, joint_state, instruction):
            self.calls += 1
            if self.calls == 2:
                raise KeyboardInterrupt
            return dict(joint_state), "scripted-dry-run-v1"

    caps = _fake_caps()
    joint_limits_deg = action_contract.load_joint_limits_deg(mock_calibration_file)
    with IOLogger(tmp_path, CAMERA_NAMES) as io_logger:
        run_episode(
            mock_robot,
            caps,
            CAMERA_NAMES,
            io_logger,
            InterruptingActionSource(),
            instruction="Pick up the red cube",
            max_steps=10,
            control_hz=100.0,
            joint_limits_deg=joint_limits_deg,
        )
    mock_robot.disconnect()  # simulate main()'s outer finally calling disconnect after run_episode

    assert call_order.index("move_to_positions") < call_order.index("disconnect")

    termination = json.loads((tmp_path / "termination.json").read_text())
    assert termination["reason"] == "keyboard_interrupt"


# --- _build_camera_names (Plan 11-05, Task 1/2 prep -- positional relabeling) ---


def test_build_camera_names_assigns_positionally_not_by_raw_index():
    # Plan 11-05 Task 1 found the AR0144 stereo camera at index 0 and the
    # IMX335 wrist camera at index 1 on this session's hardware -- the
    # reverse of the previous fixed {0: "wrist", 1: "overhead"} table. Passing
    # --camera in physical-reality order must relabel correctly.
    assert _build_camera_names([1, 0]) == {1: "wrist", 0: "overhead"}


def test_build_camera_names_default_order_matches_prior_default_table():
    assert _build_camera_names([0, 1]) == {0: "wrist", 1: "overhead"}


def test_build_camera_names_drops_extra_indices_beyond_known_names():
    assert _build_camera_names([0, 1, 2]) == {0: "wrist", 1: "overhead"}


# --- --server-address / --checkpoint bridge selection (Plan 11-04, Task 2) ---


def _point_device_map_at_nonexistent_path(monkeypatch, tmp_path):
    """Keeps bridge-selection tests hermetic regardless of whether a real
    control/device_map.json happens to exist on the machine running these
    tests -- device_map.json resolution is exercised by its own dedicated
    tests below."""
    monkeypatch.setattr(run_vla_episode, "DEVICE_MAP_PATH", tmp_path / "nonexistent_device_map.json")


def test_run_vla_episode_selects_bridge_source_when_server_address_given(monkeypatch, tmp_path):
    calls = {}

    class FakeBridgeClient:
        def __init__(self):
            self.robot = object()

        def stop(self):
            calls["stopped"] = True

    fake_client = FakeBridgeClient()

    def fake_connect_bridge(
        server_address,
        checkpoint,
        robot_config,
        task,
        policy_device="cuda",
        stereo_camera_index=1,
        wrist_camera_index=None,
    ):
        calls["connect_bridge_args"] = (server_address, checkpoint, task, policy_device)
        calls["stereo_camera_index"] = stereo_camera_index
        calls["wrist_camera_index"] = wrist_camera_index
        return fake_client

    class FakeBridgeActionSource:
        def __init__(self, client, checkpoint, joint_limits_deg):
            calls["bridge_action_source_client"] = client
            calls["bridge_action_source_checkpoint"] = checkpoint

    def fake_run_episode(
        robot, caps, camera_names, io_logger, action_source, instruction, max_steps, control_hz, joint_limits_deg
    ):
        calls["action_source_type"] = type(action_source).__name__

    _point_device_map_at_nonexistent_path(monkeypatch, tmp_path)
    monkeypatch.setattr(run_vla_episode, "connect_bridge", fake_connect_bridge)
    monkeypatch.setattr(run_vla_episode, "BridgeActionSource", FakeBridgeActionSource)
    monkeypatch.setattr(run_vla_episode, "run_episode", fake_run_episode)
    monkeypatch.setattr(action_contract, "load_joint_limits_deg", lambda: {})
    monkeypatch.setattr(run_vla_episode.cv2, "VideoCapture", lambda idx: _NeverOpensCapture())
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_vla_episode.py",
            "/dev/fake_port",
            "fake_robot_id",
            "--out",
            str(tmp_path),
            "--camera",
            "0",
            "--server-address",
            "0.tcp.ngrok.io:12345",
            "--checkpoint",
            "victorvanhalst/smolvla_so101_cube",
        ],
    )

    run_vla_episode.main()

    assert calls["action_source_type"] == "FakeBridgeActionSource"
    assert calls["bridge_action_source_client"] is fake_client
    assert calls["bridge_action_source_checkpoint"] == "victorvanhalst/smolvla_so101_cube"
    assert calls["connect_bridge_args"][0] == "0.tcp.ngrok.io:12345"
    assert calls["connect_bridge_args"][1] == "victorvanhalst/smolvla_so101_cube"
    assert calls.get("stopped") is True
    # Default --stereo-camera-index is 1 when neither the CLI nor device_map.json
    # supplies one.
    assert calls["stereo_camera_index"] == 1
    # Single --camera 0 given -> _build_camera_names([0]) == {0: "wrist"}.
    assert calls["wrist_camera_index"] == 0


def test_stereo_camera_index_flag_threads_through_to_connect_bridge(monkeypatch, tmp_path):
    calls = {}

    class FakeBridgeClient:
        def __init__(self):
            self.robot = object()

        def stop(self):
            pass

    fake_client = FakeBridgeClient()

    def fake_connect_bridge(
        server_address,
        checkpoint,
        robot_config,
        task,
        policy_device="cuda",
        stereo_camera_index=1,
        wrist_camera_index=None,
    ):
        calls["stereo_camera_index"] = stereo_camera_index
        return fake_client

    _point_device_map_at_nonexistent_path(monkeypatch, tmp_path)
    monkeypatch.setattr(run_vla_episode, "connect_bridge", fake_connect_bridge)
    monkeypatch.setattr(run_vla_episode, "BridgeActionSource", lambda *a, **k: None)
    monkeypatch.setattr(run_vla_episode, "run_episode", lambda *a, **k: None)
    monkeypatch.setattr(action_contract, "load_joint_limits_deg", lambda: {})
    monkeypatch.setattr(run_vla_episode.cv2, "VideoCapture", lambda idx: _NeverOpensCapture())
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_vla_episode.py",
            "/dev/fake_port",
            "fake_robot_id",
            "--out",
            str(tmp_path),
            "--camera",
            "0",
            "--server-address",
            "0.tcp.ngrok.io:12345",
            "--checkpoint",
            "victorvanhalst/smolvla_so101_cube",
            "--stereo-camera-index",
            "0",
        ],
    )

    run_vla_episode.main()

    # --stereo-camera-index is type=str now (accepts a device name, not just a
    # numeric index) -- an explicit "0" arrives as the string "0", not int 0.
    assert calls["stereo_camera_index"] == "0"


def test_wrist_camera_index_threads_through_from_camera_names(monkeypatch, tmp_path):
    """The camera1 (wrist) wiring gap fix: connect_bridge() must receive
    whichever physical index _build_camera_names() positionally assigned to
    "wrist" -- here, --camera 5 (first) --camera 7 (second) -> wrist=5."""
    calls = {}

    class FakeBridgeClient:
        def __init__(self):
            self.robot = object()

        def stop(self):
            pass

    fake_client = FakeBridgeClient()

    def fake_connect_bridge(
        server_address,
        checkpoint,
        robot_config,
        task,
        policy_device="cuda",
        stereo_camera_index=1,
        wrist_camera_index=None,
    ):
        calls["wrist_camera_index"] = wrist_camera_index
        return fake_client

    _point_device_map_at_nonexistent_path(monkeypatch, tmp_path)
    monkeypatch.setattr(run_vla_episode, "connect_bridge", fake_connect_bridge)
    monkeypatch.setattr(run_vla_episode, "BridgeActionSource", lambda *a, **k: None)
    monkeypatch.setattr(run_vla_episode, "run_episode", lambda *a, **k: None)
    monkeypatch.setattr(action_contract, "load_joint_limits_deg", lambda: {})
    monkeypatch.setattr(run_vla_episode.cv2, "VideoCapture", lambda idx: _NeverOpensCapture())
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_vla_episode.py",
            "/dev/fake_port",
            "fake_robot_id",
            "--out",
            str(tmp_path),
            "--camera",
            "5",
            "--camera",
            "7",
            "--server-address",
            "0.tcp.ngrok.io:12345",
            "--checkpoint",
            "victorvanhalst/smolvla_so101_cube",
        ],
    )

    run_vla_episode.main()

    assert calls["wrist_camera_index"] == 5


def test_bridge_unreachable_writes_termination_reason_without_driving_robot(monkeypatch, tmp_path):
    calls = {"run_episode_called": False}

    def fake_connect_bridge_returns_none(
        server_address,
        checkpoint,
        robot_config,
        task,
        policy_device="cuda",
        stereo_camera_index=1,
        wrist_camera_index=None,
    ):
        return None

    def fake_run_episode(*args, **kwargs):
        calls["run_episode_called"] = True

    _point_device_map_at_nonexistent_path(monkeypatch, tmp_path)
    monkeypatch.setattr(run_vla_episode, "connect_bridge", fake_connect_bridge_returns_none)
    monkeypatch.setattr(run_vla_episode, "run_episode", fake_run_episode)
    monkeypatch.setattr(action_contract, "load_joint_limits_deg", lambda: {})
    monkeypatch.setattr(run_vla_episode.cv2, "VideoCapture", lambda idx: _NeverOpensCapture())
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_vla_episode.py",
            "/dev/fake_port",
            "fake_robot_id",
            "--out",
            str(tmp_path),
            "--camera",
            "0",
            "--server-address",
            "0.tcp.ngrok.io:12345",
            "--checkpoint",
            "victorvanhalst/smolvla_so101_cube",
        ],
    )

    run_vla_episode.main()

    assert calls["run_episode_called"] is False
    termination = json.loads((tmp_path / "termination.json").read_text())
    assert termination["reason"] == "bridge_unreachable"
    assert termination["steps_completed"] == 0


def test_checkpoint_without_server_address_errors_clearly(monkeypatch, tmp_path):
    _point_device_map_at_nonexistent_path(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_vla_episode.py",
            "/dev/fake_port",
            "fake_robot_id",
            "--out",
            str(tmp_path),
            "--checkpoint",
            "victorvanhalst/smolvla_so101_cube",
        ],
    )

    with pytest.raises(SystemExit):
        run_vla_episode.main()


# --- device_map.json integration (Plan 11-05 Task 2) --------------------------------


def test_device_map_supplies_port_robot_id_camera_defaults_when_cli_args_omitted(monkeypatch, tmp_path):
    device_map_path = tmp_path / "device_map.json"
    device_map_path.write_text(
        json.dumps(
            {
                "detected_at": "2026-09-22T10:00:00Z",
                "follower": {"port": "/dev/cu.fake999", "id": "soarm_follower_02"},
                "leader": None,
                "cameras": {"wrist": 3, "stereo_overhead": 4},
            }
        )
    )
    monkeypatch.setattr(run_vla_episode, "DEVICE_MAP_PATH", device_map_path)

    calls = {}

    class FakeBridgeClient:
        def __init__(self):
            self.robot = object()

        def stop(self):
            calls["stopped"] = True

    fake_client = FakeBridgeClient()

    def fake_connect_bridge(
        server_address,
        checkpoint,
        robot_config,
        task,
        policy_device="cuda",
        stereo_camera_index=1,
        wrist_camera_index=None,
    ):
        calls["port"] = robot_config.port
        calls["robot_id"] = robot_config.id
        calls["stereo_camera_index"] = stereo_camera_index
        calls["wrist_camera_index"] = wrist_camera_index
        return fake_client

    monkeypatch.setattr(run_vla_episode, "connect_bridge", fake_connect_bridge)
    monkeypatch.setattr(run_vla_episode, "BridgeActionSource", lambda *a, **k: None)
    monkeypatch.setattr(run_vla_episode, "run_episode", lambda *a, **k: None)
    monkeypatch.setattr(action_contract, "load_joint_limits_deg", lambda: {})
    monkeypatch.setattr(run_vla_episode.cv2, "VideoCapture", lambda idx: _NeverOpensCapture())
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_vla_episode.py",
            "--out",
            str(tmp_path / "out"),
            "--server-address",
            "0.tcp.ngrok.io:12345",
            "--checkpoint",
            "victorvanhalst/smolvla_so101_cube",
        ],
    )

    run_vla_episode.main()

    assert calls["port"] == "/dev/cu.fake999"
    assert calls["robot_id"] == "soarm_follower_02"
    assert calls["stereo_camera_index"] == 4


def test_device_map_prefers_stereo_overhead_name_over_numeric_index_when_present(monkeypatch, tmp_path):
    """The 11-05 Task 3 fix: cameras.stereo_overhead_name (a device NAME
    string) must win over cameras.stereo_overhead (a numeric cv2 index) when
    both are present -- the numeric index has been confirmed to drift between
    process launches on macOS, the name string has not."""
    device_map_path = tmp_path / "device_map.json"
    device_map_path.write_text(
        json.dumps(
            {
                "detected_at": "2026-09-22T10:00:00Z",
                "follower": {"port": "/dev/cu.fake999", "id": "soarm_follower_02"},
                "leader": None,
                "cameras": {"wrist": 3, "stereo_overhead": 4, "stereo_overhead_name": "CCB Camera"},
            }
        )
    )
    monkeypatch.setattr(run_vla_episode, "DEVICE_MAP_PATH", device_map_path)

    calls = {}

    class FakeBridgeClient:
        def __init__(self):
            self.robot = object()

        def stop(self):
            pass

    fake_client = FakeBridgeClient()

    def fake_connect_bridge(
        server_address,
        checkpoint,
        robot_config,
        task,
        policy_device="cuda",
        stereo_camera_index=1,
        wrist_camera_index=None,
    ):
        calls["stereo_camera_index"] = stereo_camera_index
        calls["wrist_camera_index"] = wrist_camera_index
        return fake_client

    monkeypatch.setattr(run_vla_episode, "connect_bridge", fake_connect_bridge)
    monkeypatch.setattr(run_vla_episode, "BridgeActionSource", lambda *a, **k: None)
    monkeypatch.setattr(run_vla_episode, "run_episode", lambda *a, **k: None)
    monkeypatch.setattr(action_contract, "load_joint_limits_deg", lambda: {})
    monkeypatch.setattr(run_vla_episode.cv2, "VideoCapture", lambda idx: _NeverOpensCapture())
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_vla_episode.py",
            "--out",
            str(tmp_path / "out"),
            "--server-address",
            "0.tcp.ngrok.io:12345",
            "--checkpoint",
            "victorvanhalst/smolvla_so101_cube",
        ],
    )

    run_vla_episode.main()

    assert calls["stereo_camera_index"] == "CCB Camera"
    assert calls["wrist_camera_index"] == 3


def test_explicit_cli_args_override_device_map_json(monkeypatch, tmp_path):
    device_map_path = tmp_path / "device_map.json"
    device_map_path.write_text(
        json.dumps(
            {
                "detected_at": "2026-09-22T10:00:00Z",
                "follower": {"port": "/dev/cu.should_not_be_used", "id": "wrong_id"},
                "leader": None,
                "cameras": {"wrist": 99, "stereo_overhead": 98},
            }
        )
    )
    monkeypatch.setattr(run_vla_episode, "DEVICE_MAP_PATH", device_map_path)

    calls = {}

    class FakeBridgeClient:
        def __init__(self):
            self.robot = object()

        def stop(self):
            pass

    fake_client = FakeBridgeClient()

    def fake_connect_bridge(
        server_address,
        checkpoint,
        robot_config,
        task,
        policy_device="cuda",
        stereo_camera_index=1,
        wrist_camera_index=None,
    ):
        calls["port"] = robot_config.port
        calls["robot_id"] = robot_config.id
        calls["stereo_camera_index"] = stereo_camera_index
        return fake_client

    monkeypatch.setattr(run_vla_episode, "connect_bridge", fake_connect_bridge)
    monkeypatch.setattr(run_vla_episode, "BridgeActionSource", lambda *a, **k: None)
    monkeypatch.setattr(run_vla_episode, "run_episode", lambda *a, **k: None)
    monkeypatch.setattr(action_contract, "load_joint_limits_deg", lambda: {})
    monkeypatch.setattr(run_vla_episode.cv2, "VideoCapture", lambda idx: _NeverOpensCapture())
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_vla_episode.py",
            "/dev/cu.explicit_port",
            "explicit_robot_id",
            "--camera",
            "0",
            "--out",
            str(tmp_path / "out"),
            "--server-address",
            "0.tcp.ngrok.io:12345",
            "--checkpoint",
            "victorvanhalst/smolvla_so101_cube",
            "--stereo-camera-index",
            "1",
        ],
    )

    run_vla_episode.main()

    assert calls["port"] == "/dev/cu.explicit_port"
    assert calls["robot_id"] == "explicit_robot_id"
    # --stereo-camera-index is type=str now (accepts a device name like "CCB
    # Camera", not just a numeric index -- see the flag's help text), so an
    # explicit "1" arrives as the string "1", not int 1.
    assert calls["stereo_camera_index"] == "1"


def test_missing_port_and_device_map_errors_clearly(monkeypatch, tmp_path):
    _point_device_map_at_nonexistent_path(monkeypatch, tmp_path)
    monkeypatch.setattr("sys.argv", ["run_vla_episode.py", "--out", str(tmp_path / "out")])

    with pytest.raises(SystemExit):
        run_vla_episode.main()


def test_missing_camera_and_device_map_errors_clearly(monkeypatch, tmp_path):
    _point_device_map_at_nonexistent_path(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        ["run_vla_episode.py", "/dev/fake_port", "fake_robot_id", "--out", str(tmp_path / "out")],
    )

    with pytest.raises(SystemExit):
        run_vla_episode.main()


def test_help_still_documents_port_robot_id_camera_as_explicit_overrides(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["run_vla_episode.py", "--help"])

    with pytest.raises(SystemExit) as exc_info:
        run_vla_episode.main()

    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "port" in captured.out
    assert "robot_id" in captured.out
    assert "--camera" in captured.out
