"""Tests for `vla_bridge.backends.pi0`: `--backend pi0` / `--backend pi05` (VLAB-03).

Both are Tier A gRPC-bridge backends. They differ from SmolVLA only in the
`policy_type` string and the checkpoint, and both go through
`base_grpc.build_grpc_action_source()`. Nothing here opens a serial port, a
camera, or a gRPC channel.
"""

import json
from types import SimpleNamespace

import pytest

from vla_bridge import backends
from vla_bridge.backends import base_grpc

PI_BACKENDS = ("pi0", "pi05")


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


@pytest.mark.parametrize("name", PI_BACKENDS)
def test_pi_backend_is_registered_on_the_grpc_bridge_lane(name):
    assert name in backends.BACKEND_REGISTRY
    assert name in backends.GRPC_BRIDGE_BACKENDS
    assert name in backends.get_backend_names()


def test_pi0_and_pi05_are_distinct_build_functions_exported_by_the_module():
    from vla_bridge.backends import pi0 as pi0_module

    assert backends.BACKEND_REGISTRY["pi0"] is pi0_module.build_pi0
    assert backends.BACKEND_REGISTRY["pi05"] is pi0_module.build_pi05
    assert pi0_module.build_pi0 is not pi0_module.build_pi05


def test_policy_type_strings_are_accepted_by_the_installed_policy_server():
    """The handshake fails server-side for any policy_type outside
    lerobot 0.6.1's SUPPORTED_POLICIES, so the registry names have to match it."""
    from lerobot.async_inference.constants import SUPPORTED_POLICIES

    assert set(PI_BACKENDS).issubset(SUPPORTED_POLICIES)


# --- Checkpoint resolution -----------------------------------------------------------


@pytest.mark.parametrize("name", PI_BACKENDS)
def test_pi_backend_uses_cli_checkpoint_when_given(monkeypatch, tmp_path, name):
    monkeypatch.setattr(backends, "BACKEND_CONFIG_DIR", tmp_path)
    (tmp_path / f"{name}.json").write_text(json.dumps({"checkpoint": "config/ckpt"}))
    calls, sentinel = _record_grpc_build(monkeypatch)
    args = _args(checkpoint="cli/ckpt")
    camera_names = {0: "wrist", 1: "overhead"}
    joint_limits = {"gripper": (0.0, 100.0)}

    result = backends.BACKEND_REGISTRY[name](args, camera_names, joint_limits)

    assert result is sentinel
    assert calls == dict(policy_type=name, checkpoint="cli/ckpt", args=args,
                         camera_names=camera_names, joint_limits_deg=joint_limits)


@pytest.mark.parametrize("name", PI_BACKENDS)
def test_pi_backend_falls_back_to_its_own_config_checkpoint(monkeypatch, tmp_path, name):
    monkeypatch.setattr(backends, "BACKEND_CONFIG_DIR", tmp_path)
    # Both configs present: each backend must read its OWN file, not the other's.
    (tmp_path / "pi0.json").write_text(json.dumps({"checkpoint": "config/pi0-ckpt"}))
    (tmp_path / "pi05.json").write_text(json.dumps({"checkpoint": "config/pi05-ckpt"}))
    calls, _ = _record_grpc_build(monkeypatch)

    backends.BACKEND_REGISTRY[name](_args(checkpoint=None), {0: "wrist"}, {})

    assert calls["policy_type"] == name
    assert calls["checkpoint"] == f"config/{name}-ckpt"


@pytest.mark.parametrize("name", PI_BACKENDS)
def test_pi_backend_without_any_checkpoint_exits_before_connecting(monkeypatch, tmp_path, name):
    monkeypatch.setattr(backends, "BACKEND_CONFIG_DIR", tmp_path)  # no <name>.json
    calls, _ = _record_grpc_build(monkeypatch)

    with pytest.raises(SystemExit, match=rf"--backend {name} .*checkpoint"):
        backends.BACKEND_REGISTRY[name](_args(checkpoint=None), {0: "wrist"}, {})
    assert calls == {}


@pytest.mark.parametrize("name", PI_BACKENDS)
def test_pi_backend_with_config_lacking_checkpoint_field_exits(monkeypatch, tmp_path, name):
    monkeypatch.setattr(backends, "BACKEND_CONFIG_DIR", tmp_path)
    (tmp_path / f"{name}.json").write_text(json.dumps({"notes": "no checkpoint here"}))
    calls, _ = _record_grpc_build(monkeypatch)

    with pytest.raises(SystemExit, match="checkpoint"):
        backends.BACKEND_REGISTRY[name](_args(checkpoint=None), {0: "wrist"}, {})
    assert calls == {}


@pytest.mark.parametrize("name", PI_BACKENDS)
def test_pi_backend_returns_none_when_bridge_is_unreachable(monkeypatch, name):
    """`None` is the documented bridge_unreachable signal run_vla_episode.main() handles."""
    monkeypatch.setattr(base_grpc.robot_client, "connect_bridge", lambda *a, **k: None)

    assert backends.BACKEND_REGISTRY[name](_args(checkpoint="some/ckpt"), {0: "wrist"}, {}) is None


# --- Shipped configs (HF-Hub-verified 2026-10-01) ------------------------------------

# Each checkpoint was checked live: public, not gated, lerobot "type" matching
# the backend, input_features contain observation.images.camera1 (the only
# camera key the bridge actually delivers, see backends/pi0.py), 6-D state and
# 6-D action named exactly JOINT_ORDER (.pos), and config + processor pipeline
# parse under the installed lerobot 0.6.1.
SHIPPED = {
    "pi0": ("amindt/pi0_so101_pickandplace_redblue", "6d052c61e84eec978070c361d7f86e7d71e08b4f"),
    "pi05": ("yathAg/pi05_so101_pick_place", "8913fc8168dfada97c676598897ac3deaaaf6f0d"),
}


@pytest.mark.parametrize("name", PI_BACKENDS)
def test_shipped_config_names_the_verified_so101_joint_space_checkpoint(name):
    config = backends.load_backend_config(name)
    checkpoint, revision = SHIPPED[name]

    assert config is not None
    assert config["checkpoint"] == checkpoint
    assert config["verified_revision"] == revision


def test_pi0_and_pi05_ship_different_checkpoints():
    assert backends.load_backend_config("pi0")["checkpoint"] != backends.load_backend_config("pi05")["checkpoint"]
