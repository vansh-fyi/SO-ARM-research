"""Tests for `vla_bridge.backends.diffusion_policy`: `--backend diffusion` (VLAB-12).

Diffusion Policy is a Tier A gRPC-bridge backend. Client-side, it differs from
SmolVLA/pi0/ACT only in the `policy_type` string and the checkpoint, and goes
through `base_grpc.build_grpc_action_source()`. Nothing here opens a serial
port, a camera, or a gRPC channel.
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

    def fake_build(policy_type, checkpoint, args, camera_names, joint_limits_deg, camera_map=None):
        calls.update(policy_type=policy_type, checkpoint=checkpoint, args=args,
                     camera_names=camera_names, joint_limits_deg=joint_limits_deg)
        if camera_map is not None:
            calls["camera_map"] = camera_map
        return sentinel

    monkeypatch.setattr(base_grpc, "build_grpc_action_source", fake_build)
    return calls, sentinel


# --- Registration ------------------------------------------------------------------


def test_diffusion_is_registered_on_the_grpc_bridge_lane():
    assert "diffusion" in backends.BACKEND_REGISTRY
    assert "diffusion" in backends.GRPC_BRIDGE_BACKENDS
    assert "diffusion" in backends.get_backend_names()


def test_diffusion_registry_entry_is_the_module_build_function():
    from vla_bridge.backends import diffusion_policy as diffusion_module

    assert backends.BACKEND_REGISTRY["diffusion"] is diffusion_module.build


def test_diffusion_policy_type_is_accepted_by_the_installed_policy_server():
    """The registry key is the exact `policy_type` string the PolicyServer dispatches on."""
    from lerobot.async_inference.constants import SUPPORTED_POLICIES
    from vla_bridge.backends import diffusion_policy as diffusion_module

    assert diffusion_module.POLICY_TYPE == "diffusion"
    assert "diffusion" in SUPPORTED_POLICIES


# --- Checkpoint resolution -----------------------------------------------------------


def test_diffusion_uses_cli_checkpoint_when_given(monkeypatch, tmp_path):
    monkeypatch.setattr(backends, "BACKEND_CONFIG_DIR", tmp_path)
    (tmp_path / "diffusion_policy.json").write_text(json.dumps({"checkpoint": "config/ckpt"}))
    calls, sentinel = _record_grpc_build(monkeypatch)
    args = _args(checkpoint="cli/ckpt")
    camera_names = {0: "wrist", 1: "overhead"}
    joint_limits = {"gripper": (0.0, 100.0)}

    result = backends.BACKEND_REGISTRY["diffusion"](args, camera_names, joint_limits)

    assert result is sentinel
    assert calls == dict(policy_type="diffusion", checkpoint="cli/ckpt", args=args,
                         camera_names=camera_names, joint_limits_deg=joint_limits)


def test_diffusion_falls_back_to_diffusion_policy_json_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setattr(backends, "BACKEND_CONFIG_DIR", tmp_path)
    # The config FILE is diffusion_policy.json even though the registry key is "diffusion";
    # a stray diffusion.json or another backend's config must not be read.
    (tmp_path / "diffusion_policy.json").write_text(json.dumps({"checkpoint": "config/dp-ckpt"}))
    (tmp_path / "diffusion.json").write_text(json.dumps({"checkpoint": "wrong/file"}))
    (tmp_path / "act.json").write_text(json.dumps({"checkpoint": "config/act-ckpt"}))
    calls, _ = _record_grpc_build(monkeypatch)

    backends.BACKEND_REGISTRY["diffusion"](_args(checkpoint=None), {0: "wrist"}, {})

    assert calls["policy_type"] == "diffusion"
    assert calls["checkpoint"] == "config/dp-ckpt"


def test_diffusion_without_any_checkpoint_exits_before_connecting(monkeypatch, tmp_path):
    monkeypatch.setattr(backends, "BACKEND_CONFIG_DIR", tmp_path)  # no diffusion_policy.json
    calls, _ = _record_grpc_build(monkeypatch)

    with pytest.raises(SystemExit, match=r"--backend diffusion .*checkpoint"):
        backends.BACKEND_REGISTRY["diffusion"](_args(checkpoint=None), {0: "wrist"}, {})
    assert calls == {}


def test_diffusion_with_config_lacking_checkpoint_field_exits(monkeypatch, tmp_path):
    monkeypatch.setattr(backends, "BACKEND_CONFIG_DIR", tmp_path)
    (tmp_path / "diffusion_policy.json").write_text(json.dumps({"notes": "no checkpoint here"}))
    calls, _ = _record_grpc_build(monkeypatch)

    with pytest.raises(SystemExit, match="checkpoint"):
        backends.BACKEND_REGISTRY["diffusion"](_args(checkpoint=None), {0: "wrist"}, {})
    assert calls == {}


def test_diffusion_returns_none_when_bridge_is_unreachable(monkeypatch):
    """`None` is the documented bridge_unreachable signal run_vla_episode.main() handles."""
    monkeypatch.setattr(base_grpc.robot_client, "connect_bridge", lambda *a, **k: None)

    result = backends.BACKEND_REGISTRY["diffusion"](_args(checkpoint="some/ckpt"), {0: "wrist"}, {})

    assert result is None


# --- Shipped config (HF-Hub-verified 2026-10-01) ------------------------------------
#
# No HF Hub Diffusion Policy checkpoint declares observation.images.camera1 as its
# ONLY image key (415 lerobot "type": "diffusion" configs scanned). Diffusion does not
# mask missing cameras and today's bridge delivers only camera1, so NO default
# checkpoint is shipped: --checkpoint is required. The config records the candidate
# that becomes loadable once the bridge follow-up in deferred-items.md lands.
CANDIDATE_DIFFUSION_CHECKPOINT = "hrl7/so101-touch-diffusion-l4"
CANDIDATE_DIFFUSION_REVISION = "56c702bac49b28894fe4e6afcaa0789d6551cb63"


def test_shipped_config_ships_the_candidate_with_a_matching_camera_map():
    """The bridge now sends the cameras a backend's camera_map names, so the SO-101
    candidate (camera1 + camera2) is the default checkpoint."""
    config = backends.load_backend_config("diffusion_policy")

    assert config["checkpoint"] == CANDIDATE_DIFFUSION_CHECKPOINT
    assert config["verified_revision"] == CANDIDATE_DIFFUSION_REVISION
    assert config["camera_map"] == {"camera1": "wrist", "camera2": "overhead_left"}


def test_shipped_config_build_passes_its_checkpoint_and_camera_map(monkeypatch):
    calls, sentinel = _record_grpc_build(monkeypatch)

    result = backends.BACKEND_REGISTRY["diffusion"](_args(checkpoint=None), {0: "wrist"}, {})

    assert result is sentinel
    assert calls["checkpoint"] == CANDIDATE_DIFFUSION_CHECKPOINT
    assert calls["camera_map"] == {"camera1": "wrist", "camera2": "overhead_left"}


def test_check_checkpoint_treats_diffusion_as_non_masking():
    from vla_bridge import check_checkpoint as cc

    assert "diffusion" in cc.NO_CAMERA_MASKING_BACKENDS
    assert "diffusion" in cc.SECTION_TITLES
    assert cc.CANDIDATE_GROUPS["diffusion"][0] == CANDIDATE_DIFFUSION_CHECKPOINT


# --- Mocked round trip (Task 2): real base_grpc + real BridgeActionSource ----------
#
# Only `connect_bridge()` is faked. Everything after the handshake is the
# production code path: base_grpc wraps the client in the unmodified
# BridgeActionSource, whose get_action() pops one queued chunk action through
# pop_validated_action() -> safety_validator.validate_action(). The client
# double is test_robot_client.FakeBridgeClient (+ FakeTimedAction), extended in
# test_backend_pi0 with the handshake-time surface base_grpc touches.

ROUND_TRIP_CHECKPOINT = "someone/so101-diffusion-camera1-only"


def test_diffusion_round_trip_returns_validated_action_without_send_action(
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

    handles = backends.BACKEND_REGISTRY["diffusion"](
        _args(checkpoint=ROUND_TRIP_CHECKPOINT), {0: "wrist", 1: "overhead"}, joint_limits
    )

    client = connect_calls["client"]
    assert isinstance(handles, BackendHandles)
    assert connect_calls["policy_type"] == "diffusion"
    assert connect_calls["checkpoint"] == ROUND_TRIP_CHECKPOINT
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
    assert model_version == f"{ROUND_TRIP_CHECKPOINT}@unknown"  # no hold/stale flags
    assert client.observations_sent == ["pick the cube"]
    assert mock_robot.sent_actions == []  # the source never sends; run_episode does


def test_diffusion_round_trip_clamps_an_out_of_range_chunk_action(
    monkeypatch, mock_robot, mock_calibration_file
):
    """A server-returned action outside the joint limits still goes through
    safety_validator.validate_action() (T-14-02): it is clamped, never sent raw."""
    joint_limits = action_contract.load_joint_limits_deg(mock_calibration_file)
    queued = {f"{joint}.pos": 12.5 for joint in JOINT_ORDER}
    queued["elbow_flex.pos"] = 10_000.0
    monkeypatch.setattr(
        robot_client, "connect_bridge", lambda *a, **k: _FakePiBridgeClient(mock_robot, queued)
    )

    handles = backends.BACKEND_REGISTRY["diffusion"](
        _args(checkpoint=ROUND_TRIP_CHECKPOINT), {0: "wrist"}, joint_limits
    )
    action, _ = handles.action_source.get_action(dict.fromkeys(JOINT_ORDER, 10.0), "pick the cube")

    low, high = joint_limits["elbow_flex"]
    assert low <= action["elbow_flex"] <= high
    assert action["elbow_flex"] != 10_000.0
    assert mock_robot.sent_actions == []


def test_diffusion_round_trip_holds_position_when_queue_is_empty(
    monkeypatch, mock_robot, mock_calibration_file
):
    """What the robot does if the server never returns a chunk (the stock 0.6.1 server's
    n_obs_steps assertion, or an undelivered camera key): it holds."""
    joint_limits = action_contract.load_joint_limits_deg(mock_calibration_file)
    client = _FakePiBridgeClient(mock_robot, {f"{j}.pos": 12.5 for j in JOINT_ORDER})
    client.action_queue.get_nowait()  # nothing queued
    monkeypatch.setattr(robot_client, "connect_bridge", lambda *a, **k: client)

    handles = backends.BACKEND_REGISTRY["diffusion"](_args(checkpoint="x/y"), {0: "wrist"}, joint_limits)
    joint_state = dict.fromkeys(JOINT_ORDER, 10.0)
    action, model_version = handles.action_source.get_action(joint_state, "pick the cube")

    assert action == joint_state
    assert "holding position" in model_version
    assert mock_robot.sent_actions == []


# --- CLI dispatch through run_vla_episode.main() ------------------------------------


def test_cli_backend_flag_dispatches_to_real_diffusion_build(monkeypatch, tmp_path):
    calls = {}
    handles = BackendHandles(action_source=_ClosableActionSource(), robot=_FakeBackendRobot())

    def fake_build(policy_type, checkpoint, args, camera_names, joint_limits_deg, camera_map=None):
        calls.update(policy_type=policy_type, checkpoint=checkpoint)
        return handles

    monkeypatch.setattr(base_grpc, "build_grpc_action_source", fake_build)

    result = _dispatch_main(monkeypatch, tmp_path, [
        "--backend", "diffusion", "--server-address", "0.tcp.ngrok.io:12345",
        "--checkpoint", "cli/ckpt",
    ], {})

    assert calls == dict(policy_type="diffusion", checkpoint="cli/ckpt")
    assert result["run_episode_called"] is True
    assert result["action_source"] is handles.action_source


@pytest.mark.parametrize(
    "partial_args",
    [["--checkpoint", "cli/ckpt"], ["--server-address", "0.tcp.ngrok.io:12345"], []],
    ids=["checkpoint-only", "server-address-only", "neither"],
)
def test_cli_diffusion_requires_bridge_args(monkeypatch, tmp_path, partial_args):
    calls, _ = _record_grpc_build(monkeypatch)

    with pytest.raises(SystemExit):
        _dispatch_main(monkeypatch, tmp_path, ["--backend", "diffusion", *partial_args], {})
    assert calls == {}
