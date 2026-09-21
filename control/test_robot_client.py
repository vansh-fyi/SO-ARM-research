"""Tests for `vla_bridge.robot_client` (VLAHW-02).

Uses small, test-local fake objects -- NEVER a real `RobotClient` -- since
constructing one tries to open a real serial port via
`make_robot_from_config()`/`robot.connect()` inside its own `__init__`.
"""

import queue
import threading
import time

import grpc

from vla_bridge import action_contract, robot_client, safety_validator
from vla_bridge.action_contract import JOINT_ORDER


class FakeTimedAction:
    """Stand-in for `lerobot.async_inference.helpers.TimedAction` -- exposes
    just the public surface `pop_validated_action()` actually calls."""

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


class FakeBridgeClient:
    """Test-local fake of `RobotClient`'s public surface `pop_validated_action`
    touches -- a real `queue.Queue` + `threading.Lock`, and a stub
    `_action_tensor_to_action_dict` that returns its input unchanged (tests
    populate the queue with plain dicts already, not real torch tensors)."""

    def __init__(self):
        self.action_queue = queue.Queue()
        self.action_queue_lock = threading.Lock()

    def _action_tensor_to_action_dict(self, action_tensor):
        return action_tensor


# --- pop_validated_action ---------------------------------------------------


def test_pop_validated_action_returns_validated_action_and_never_calls_send_action(
    mock_calibration_file,
):
    client = FakeBridgeClient()
    raw_action = dict.fromkeys(JOINT_ORDER, 0.0)
    client.action_queue.put(FakeTimedAction(raw_action, timestamp=time.time()))

    joint_limits_deg = action_contract.load_joint_limits_deg(mock_calibration_file)
    current_state = dict.fromkeys(JOINT_ORDER, 0.0)

    validated_action, flags, raw_out, obs_age_s = robot_client.pop_validated_action(
        client,
        safety_validator.validate_action,
        joint_limits_deg,
        current_state,
        prev_action=None,
        dt_s=1.0,
    )

    assert isinstance(validated_action, dict)
    assert isinstance(flags, list)
    assert raw_out == raw_action
    assert obs_age_s >= 0.0
    # pop_validated_action must never itself call send_action -- FakeBridgeClient
    # has no such method at all, so any attempt would raise AttributeError.
    assert not hasattr(client, "send_action")


def test_pop_validated_action_empty_queue_holds_position(mock_calibration_file):
    client = FakeBridgeClient()  # empty action_queue
    joint_limits_deg = action_contract.load_joint_limits_deg(mock_calibration_file)
    current_state = dict.fromkeys(JOINT_ORDER, 1.0)

    validated_action, flags, raw_out, obs_age_s = robot_client.pop_validated_action(
        client,
        safety_validator.validate_action,
        joint_limits_deg,
        current_state,
        prev_action=None,
        dt_s=1.0,
    )

    assert validated_action == current_state
    assert flags == ["no action available, holding position"]
    assert raw_out == {}
    assert obs_age_s == 0.0


# --- connect_bridge ----------------------------------------------------------


class FakeRobotClientHandshakeFails:
    """Stand-in for `RobotClient` simulating an unreachable Colab bridge:
    `start()` returns False (matches the installed library's own contract --
    it catches `grpc.RpcError` internally and returns False, never raises)."""

    def __init__(self, config):
        self.config = config
        self.stopped = False
        self.robot = object()

    def start(self) -> bool:
        return False

    def stop(self) -> None:
        self.stopped = True


def test_connect_bridge_returns_none_and_cleans_up_when_start_fails(monkeypatch):
    created = {}

    def fake_ctor(config):
        instance = FakeRobotClientHandshakeFails(config)
        created["instance"] = instance
        return instance

    monkeypatch.setattr(robot_client, "RobotClient", fake_ctor)

    result = robot_client.connect_bridge(
        server_address="0.tcp.ngrok.io:12345",
        checkpoint="victorvanhalst/smolvla_so101_cube",
        robot_config=object(),
        task="Pick the red cube and place it in the bowl",
    )

    assert result is None
    assert created["instance"].stopped is True


class FakeRobotClientHandshakeSucceeds(FakeRobotClientHandshakeFails):
    def start(self) -> bool:
        return True


def test_connect_bridge_returns_connected_client_on_success(monkeypatch):
    monkeypatch.setattr(robot_client, "RobotClient", FakeRobotClientHandshakeSucceeds)

    result = robot_client.connect_bridge(
        server_address="0.tcp.ngrok.io:12345",
        checkpoint="victorvanhalst/smolvla_so101_cube",
        robot_config=object(),
        task="Pick the red cube and place it in the bowl",
    )

    assert isinstance(result, FakeRobotClientHandshakeSucceeds)
    assert result.stopped is False


# --- BridgeActionSource --------------------------------------------------------


class FakeClientRaisesOnObservation:
    """Simulates a dead/unreachable bridge: `control_loop_observation()`
    raises instead of returning normally."""

    def __init__(self, exc):
        self.action_queue = queue.Queue()
        self.action_queue_lock = threading.Lock()
        self._exc = exc

    def control_loop_observation(self, task: str):
        raise self._exc

    def _action_tensor_to_action_dict(self, action_tensor):
        return action_tensor


def test_bridge_action_source_holds_position_on_rpc_error():
    client = FakeClientRaisesOnObservation(grpc.RpcError("bridge unreachable"))
    source = robot_client.BridgeActionSource(
        client, checkpoint="victorvanhalst/smolvla_so101_cube", joint_limits_deg={}
    )
    joint_state = dict.fromkeys(JOINT_ORDER, 0.0)

    action, model_version = source.get_action(joint_state, "Pick the red cube and place it in the bowl")

    assert action == joint_state
    assert model_version == "victorvanhalst/smolvla_so101_cube@bridge-error-holding-position"


def test_bridge_action_source_holds_position_on_connection_error():
    client = FakeClientRaisesOnObservation(ConnectionError("tunnel dropped"))
    source = robot_client.BridgeActionSource(
        client, checkpoint="victorvanhalst/smolvla_so101_cube", joint_limits_deg={}
    )
    joint_state = dict.fromkeys(JOINT_ORDER, 0.0)

    action, model_version = source.get_action(joint_state, "Pick the red cube and place it in the bowl")

    assert action == joint_state
    assert model_version == "victorvanhalst/smolvla_so101_cube@bridge-error-holding-position"


class FakeClientReturnsAction:
    """Simulates a healthy bridge: `control_loop_observation()` succeeds, one
    action is already queued for `pop_validated_action()` to consume."""

    def __init__(self, action):
        self.action_queue = queue.Queue()
        self.action_queue_lock = threading.Lock()
        self.action_queue.put(FakeTimedAction(action, timestamp=time.time()))

    def control_loop_observation(self, task: str):
        return {"task": task}

    def _action_tensor_to_action_dict(self, action_tensor):
        return action_tensor


def test_bridge_action_source_returns_validated_action_on_success(mock_calibration_file):
    joint_limits_deg = action_contract.load_joint_limits_deg(mock_calibration_file)
    raw_action = dict.fromkeys(JOINT_ORDER, 0.0)
    client = FakeClientReturnsAction(raw_action)
    source = robot_client.BridgeActionSource(
        client, checkpoint="victorvanhalst/smolvla_so101_cube", joint_limits_deg=joint_limits_deg
    )
    joint_state = dict.fromkeys(JOINT_ORDER, 0.0)

    action, model_version = source.get_action(joint_state, "Pick the red cube and place it in the bowl")

    assert isinstance(action, dict)
    assert model_version == "victorvanhalst/smolvla_so101_cube@unknown"
