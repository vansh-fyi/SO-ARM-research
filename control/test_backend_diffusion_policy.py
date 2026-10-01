"""Tests for `vla_bridge.backends.diffusion_policy`: `--backend diffusion` (VLAB-12).

Diffusion Policy is a Tier A gRPC-bridge backend. Client-side, it differs from
SmolVLA/pi0/ACT only in the `policy_type` string and the checkpoint, and goes
through `base_grpc.build_grpc_action_source()`. Nothing here opens a serial
port, a camera, or a gRPC channel.
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


def test_shipped_config_has_no_default_checkpoint_and_records_the_candidate():
    config = backends.load_backend_config("diffusion_policy")

    assert config is not None
    assert "checkpoint" not in config
    assert config["candidate_checkpoint"] == CANDIDATE_DIFFUSION_CHECKPOINT
    assert config["candidate_verified_revision"] == CANDIDATE_DIFFUSION_REVISION
    assert "camera1" in config["notes"]


def test_shipped_config_makes_build_exit_with_the_reason(monkeypatch):
    calls, _ = _record_grpc_build(monkeypatch)

    with pytest.raises(SystemExit, match=r"(?s)--backend diffusion .*checkpoint.*camera1"):
        backends.BACKEND_REGISTRY["diffusion"](_args(checkpoint=None), {0: "wrist"}, {})
    assert calls == {}


def test_check_checkpoint_treats_diffusion_as_non_masking():
    from vla_bridge import check_checkpoint as cc

    assert "diffusion" in cc.NO_CAMERA_MASKING_BACKENDS
    assert "diffusion" in cc.SECTION_TITLES
    assert cc.CANDIDATE_GROUPS["diffusion"][0] == CANDIDATE_DIFFUSION_CHECKPOINT
