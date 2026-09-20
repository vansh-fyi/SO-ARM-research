"""Shared pytest fixtures for every `control/test_*.py` file this phase adds.

Both fixtures exist so tests can exercise `control/vla_bridge/` code
deterministically, without touching real hardware or the real
`~/.cache/huggingface/...` calibration file.
"""

import json
from contextlib import contextmanager

import pytest

from vla_bridge.action_contract import JOINT_ORDER


class MockRobot:
    """Test double exposing the same public surface `SO101Follower` provides
    that this phase's code touches -- never sends anything to real hardware.
    """

    def __init__(self):
        self.connected = False
        self.calibration = {}
        self.sent_actions = []
        self._positions = {joint: 0.0 for joint in JOINT_ORDER}
        self.bus = _MockBus()

    def connect(self, calibrate: bool = False) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def get_observation(self) -> dict:
        return {f"{joint}.pos": self._positions[joint] for joint in JOINT_ORDER}

    def send_action(self, action: dict) -> None:
        self.sent_actions.append(action)


class _MockBus:
    """No-op stand-in for `SO101Follower.bus`."""

    @contextmanager
    def torque_disabled(self):
        yield

    def write_calibration(self, calibration) -> None:
        pass


@pytest.fixture
def mock_robot():
    yield MockRobot()


@pytest.fixture
def mock_calibration_file(tmp_path):
    """Write the live `soarm_follower_02.json` values (read from the real
    hardware calibration cache this session) to a tmp_path JSON file, so
    tests can exercise `load_joint_limits_deg()` deterministically."""
    calibration = {
        "shoulder_pan": {
            "id": 1,
            "drive_mode": 0,
            "homing_offset": -1653,
            "range_min": 1269,
            "range_max": 2869,
        },
        "shoulder_lift": {
            "id": 2,
            "drive_mode": 0,
            "homing_offset": -1992,
            "range_min": 904,
            "range_max": 3349,
        },
        "elbow_flex": {
            "id": 3,
            "drive_mode": 0,
            "homing_offset": 1518,
            "range_min": 832,
            "range_max": 3044,
        },
        "wrist_flex": {
            "id": 4,
            "drive_mode": 0,
            "homing_offset": 1349,
            "range_min": 899,
            "range_max": 3220,
        },
        "wrist_roll": {
            "id": 5,
            "drive_mode": 0,
            "homing_offset": -2033,
            "range_min": 0,
            "range_max": 4095,
        },
        "gripper": {
            "id": 6,
            "drive_mode": 1,
            "homing_offset": -197,
            "range_min": 48,
            "range_max": 3637,
        },
    }
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(calibration))
    return path
