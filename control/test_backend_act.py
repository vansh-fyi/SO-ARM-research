"""Tests for `vla_bridge.backends.act`: `--backend act` (VLAB-05).

ACT is a Tier A gRPC-bridge backend. It differs from SmolVLA/pi0 only in the
`policy_type` string and the checkpoint, and goes through
`base_grpc.build_grpc_action_source()`. Nothing here opens a serial port, a
camera, or a gRPC channel.
"""

import json
import math
from types import SimpleNamespace

import pytest
from test_backend_pi0 import _FakePiBridgeClient
from test_run_vla_episode import _ClosableActionSource, _dispatch_main, _FakeBackendRobot

from vla_bridge import action_contract, backends, robot_client
from vla_bridge.action_contract import JOINT_ORDER
from vla_bridge.backends import BackendHandles, base_grpc


def _args(**overrides):
    defaults = dict(
        port="/dev/tty.usbmodem1",
        robot_id="soarm_follower_02",
        server_address="0.tcp.ngrok.io:12345",
        instruction="pick the cube",
        stereo_camera_index=1,
        checkpoint=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _record_grpc_build(monkeypatch):
    calls = {}
    sentinel = object()

    def fake_build(policy_type, checkpoint, args, camera_names, joint_limits_deg):
        calls.update(policy_type=policy_type, checkpoint=checkpoint, args=args,
                     camera_names=camera_names, joint_limits_deg=joint_limits_deg)
        return sentinel

    monkeypatch.setattr(base_grpc, "build_grpc_action_source", fake_build)
    return calls, sentinel


# --- Registration ------------------------------------------------------------------


def test_act_is_registered_on_the_grpc_bridge_lane():
    assert "act" in backends.BACKEND_REGISTRY
    assert "act" in backends.GRPC_BRIDGE_BACKENDS
    assert "act" in backends.get_backend_names()


def test_act_registry_entry_is_the_module_build_function():
    from vla_bridge.backends import act as act_module

    assert backends.BACKEND_REGISTRY["act"] is act_module.build


def test_act_policy_type_is_accepted_by_the_installed_policy_server():
    """The handshake fails server-side for any policy_type outside
    lerobot 0.6.1's SUPPORTED_POLICIES."""
    from lerobot.async_inference.constants import SUPPORTED_POLICIES

    assert "act" in SUPPORTED_POLICIES


# --- Checkpoint resolution -----------------------------------------------------------


def test_act_uses_cli_checkpoint_when_given(monkeypatch, tmp_path):
    monkeypatch.setattr(backends, "BACKEND_CONFIG_DIR", tmp_path)
    (tmp_path / "act.json").write_text(json.dumps({"checkpoint": "config/ckpt"}))
    calls, sentinel = _record_grpc_build(monkeypatch)
    args = _args(checkpoint="cli/ckpt")
    camera_names = {0: "wrist", 1: "overhead"}
    joint_limits = {"gripper": (0.0, 100.0)}

    result = backends.BACKEND_REGISTRY["act"](args, camera_names, joint_limits)

    assert result is sentinel
    assert calls == dict(policy_type="act", checkpoint="cli/ckpt", args=args,
                         camera_names=camera_names, joint_limits_deg=joint_limits)


def test_act_falls_back_to_its_own_config_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setattr(backends, "BACKEND_CONFIG_DIR", tmp_path)
    # Another backend's config is present too: ACT must read its OWN file.
    (tmp_path / "act.json").write_text(json.dumps({"checkpoint": "config/act-ckpt"}))
    (tmp_path / "pi0.json").write_text(json.dumps({"checkpoint": "config/pi0-ckpt"}))
    calls, _ = _record_grpc_build(monkeypatch)

    backends.BACKEND_REGISTRY["act"](_args(checkpoint=None), {0: "wrist"}, {})

    assert calls["policy_type"] == "act"
    assert calls["checkpoint"] == "config/act-ckpt"


def test_act_without_any_checkpoint_exits_before_connecting(monkeypatch, tmp_path):
    monkeypatch.setattr(backends, "BACKEND_CONFIG_DIR", tmp_path)  # no act.json
    calls, _ = _record_grpc_build(monkeypatch)

    with pytest.raises(SystemExit, match=r"--backend act .*checkpoint"):
        backends.BACKEND_REGISTRY["act"](_args(checkpoint=None), {0: "wrist"}, {})
    assert calls == {}


def test_act_with_config_lacking_checkpoint_field_exits(monkeypatch, tmp_path):
    monkeypatch.setattr(backends, "BACKEND_CONFIG_DIR", tmp_path)
    (tmp_path / "act.json").write_text(json.dumps({"notes": "no checkpoint here"}))
    calls, _ = _record_grpc_build(monkeypatch)

    with pytest.raises(SystemExit, match="checkpoint"):
        backends.BACKEND_REGISTRY["act"](_args(checkpoint=None), {0: "wrist"}, {})
    assert calls == {}


def test_act_returns_none_when_bridge_is_unreachable(monkeypatch):
    """`None` is the documented bridge_unreachable signal run_vla_episode.main() handles."""
    monkeypatch.setattr(base_grpc.robot_client, "connect_bridge", lambda *a, **k: None)

    assert backends.BACKEND_REGISTRY["act"](_args(checkpoint="some/ckpt"), {0: "wrist"}, {}) is None


# --- Shipped config (HF-Hub-verified 2026-10-01) ------------------------------------

# Checked live: public, not gated, lerobot "type": "act", input_features have
# observation.images.camera1 as the ONLY image key (ACT has no missing-camera
# masking and the bridge delivers only camera1, see backends/act.py), 6-D
# state and action, and config + processor pipeline parse under lerobot 0.6.1.
SHIPPED_ACT_CHECKPOINT = "hrl7/so101-pregrasp-act-l4"
SHIPPED_ACT_REVISION = "1fb2021f5378c71f11ec95c76ecbdc066f61b5c2"


def test_shipped_config_names_the_verified_camera1_only_checkpoint():
    config = backends.load_backend_config("act")

    assert config is not None
    assert config["checkpoint"] == SHIPPED_ACT_CHECKPOINT
    assert config["verified_revision"] == SHIPPED_ACT_REVISION


def test_shipped_act_checkpoint_is_the_first_act_candidate_row():
    from vla_bridge import check_checkpoint as cc

    assert cc.CANDIDATE_GROUPS["act"][0] == SHIPPED_ACT_CHECKPOINT


# --- Mocked round trip (Task 2): real base_grpc + real BridgeActionSource ----------
#
# Only `connect_bridge()` is faked. Everything after the handshake is the
# production code path: base_grpc wraps the client in the unmodified
# BridgeActionSource, whose get_action() pops one queued chunk action through
# pop_validated_action() -> safety_validator.validate_action(). The client
# double is test_robot_client.FakeBridgeClient (+ FakeTimedAction), extended in
# test_backend_pi0 with the handshake-time surface base_grpc touches.


def test_act_round_trip_returns_validated_action_without_send_action(
    monkeypatch, mock_robot, mock_calibration_file
):
    joint_limits = action_contract.load_joint_limits_deg(mock_calibration_file)
    queued = {f"{joint}.pos": 12.5 for joint in JOINT_ORDER}
    connect_calls = {}

    def fake_connect_bridge(server_address, checkpoint, **kwargs):
        connect_calls.update(server_address=server_address, checkpoint=checkpoint, **kwargs)
        connect_calls["client"] = _FakePiBridgeClient(mock_robot, queued)
        return connect_calls["client"]

    monkeypatch.setattr(robot_client, "connect_bridge", fake_connect_bridge)

    handles = backends.BACKEND_REGISTRY["act"](
        _args(checkpoint=None), {0: "wrist", 1: "overhead"}, joint_limits
    )

    client = connect_calls["client"]
    assert isinstance(handles, BackendHandles)
    assert connect_calls["policy_type"] == "act"
    assert connect_calls["checkpoint"] == SHIPPED_ACT_CHECKPOINT
    assert connect_calls["server_address"] == "0.tcp.ngrok.io:12345"
    assert connect_calls["wrist_camera_index"] == 0
    assert handles.robot is mock_robot
    assert handles.shutdown == client.stop
    assert handles.skip_camera_names == frozenset({"overhead"})
    assert isinstance(handles.action_source, robot_client.BridgeActionSource)

    joint_state = dict.fromkeys(JOINT_ORDER, 10.0)
    action, model_version = handles.action_source.get_action(joint_state, "pick the cube")

    assert set(action) == set(JOINT_ORDER)
    assert all(isinstance(v, float) and math.isfinite(v) for v in action.values())
    assert action == dict.fromkeys(JOINT_ORDER, 12.5)  # executed, not held at 10.0
    assert model_version == f"{SHIPPED_ACT_CHECKPOINT}@unknown"  # no hold/stale flags
    assert client.observations_sent == ["pick the cube"]
    assert mock_robot.sent_actions == []  # the source never sends; run_episode does


def test_act_round_trip_holds_position_when_queue_is_empty(
    monkeypatch, mock_robot, mock_calibration_file
):
    """What the robot does if the server never returns a chunk (e.g. a
    checkpoint whose image keys the bridge can't satisfy): it holds."""
    joint_limits = action_contract.load_joint_limits_deg(mock_calibration_file)
    client = _FakePiBridgeClient(mock_robot, {f"{j}.pos": 12.5 for j in JOINT_ORDER})
    client.action_queue.get_nowait()  # nothing queued
    monkeypatch.setattr(robot_client, "connect_bridge", lambda *a, **k: client)

    handles = backends.BACKEND_REGISTRY["act"](_args(checkpoint="x/y"), {0: "wrist"}, joint_limits)
    joint_state = dict.fromkeys(JOINT_ORDER, 10.0)
    action, model_version = handles.action_source.get_action(joint_state, "pick the cube")

    assert action == joint_state
    assert "holding position" in model_version
    assert mock_robot.sent_actions == []


# --- CLI dispatch through run_vla_episode.main() ------------------------------------


def test_cli_backend_flag_dispatches_to_real_act_build(monkeypatch, tmp_path):
    calls = {}
    handles = BackendHandles(action_source=_ClosableActionSource(), robot=_FakeBackendRobot())

    def fake_build(policy_type, checkpoint, args, camera_names, joint_limits_deg):
        calls.update(policy_type=policy_type, checkpoint=checkpoint)
        return handles

    monkeypatch.setattr(base_grpc, "build_grpc_action_source", fake_build)

    result = _dispatch_main(monkeypatch, tmp_path, [
        "--backend", "act", "--server-address", "0.tcp.ngrok.io:12345", "--checkpoint", "cli/ckpt",
    ], {})

    assert calls == dict(policy_type="act", checkpoint="cli/ckpt")
    assert result["run_episode_called"] is True
    assert result["action_source"] is handles.action_source


@pytest.mark.parametrize(
    "partial_args",
    [["--checkpoint", "cli/ckpt"], ["--server-address", "0.tcp.ngrok.io:12345"], []],
    ids=["checkpoint-only", "server-address-only", "neither"],
)
def test_cli_act_requires_bridge_args(monkeypatch, tmp_path, partial_args):
    calls, _ = _record_grpc_build(monkeypatch)

    with pytest.raises(SystemExit):
        _dispatch_main(monkeypatch, tmp_path, ["--backend", "act", *partial_args], {})
    assert calls == {}
