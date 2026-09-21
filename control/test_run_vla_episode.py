"""Tests for `control/run_vla_episode.py` (VLAHW-02, VLAHW-03).

Uses Plan 11-01's `mock_robot` fixture (from `conftest.py`) and a fake camera
object -- never a real robot or real `cv2.VideoCapture`.
"""

import json

import numpy as np
import pytest

import run_vla_episode
from run_vla_episode import ScriptedActionSource, run_episode
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


# --- --server-address / --checkpoint bridge selection (Plan 11-04, Task 2) ---


def test_run_vla_episode_selects_bridge_source_when_server_address_given(monkeypatch, tmp_path):
    calls = {}

    class FakeBridgeClient:
        def __init__(self):
            self.robot = object()

        def stop(self):
            calls["stopped"] = True

    fake_client = FakeBridgeClient()

    def fake_connect_bridge(server_address, checkpoint, robot_config, task, policy_device="cuda"):
        calls["connect_bridge_args"] = (server_address, checkpoint, task, policy_device)
        return fake_client

    class FakeBridgeActionSource:
        def __init__(self, client, checkpoint, joint_limits_deg):
            calls["bridge_action_source_client"] = client
            calls["bridge_action_source_checkpoint"] = checkpoint

    def fake_run_episode(
        robot, caps, camera_names, io_logger, action_source, instruction, max_steps, control_hz, joint_limits_deg
    ):
        calls["action_source_type"] = type(action_source).__name__

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


def test_bridge_unreachable_writes_termination_reason_without_driving_robot(monkeypatch, tmp_path):
    calls = {"run_episode_called": False}

    def fake_connect_bridge_returns_none(server_address, checkpoint, robot_config, task, policy_device="cuda"):
        return None

    def fake_run_episode(*args, **kwargs):
        calls["run_episode_called"] = True

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
