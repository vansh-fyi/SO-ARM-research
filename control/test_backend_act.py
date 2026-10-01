"""Tests for `vla_bridge.backends.act`: `--backend act` (VLAB-05).

ACT is a Tier A gRPC-bridge backend. It differs from SmolVLA/pi0 only in the
`policy_type` string and the checkpoint, and goes through
`base_grpc.build_grpc_action_source()`. Nothing here opens a serial port, a
camera, or a gRPC channel.
"""

import json
from types import SimpleNamespace

import pytest

from vla_bridge import backends
from vla_bridge.backends import base_grpc


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
