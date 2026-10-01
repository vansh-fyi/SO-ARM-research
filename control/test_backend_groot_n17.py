"""Tests for `--backend groot-n17` (VLAB-10): GR00T N1.7 over the gRPC bridge.

Plan 14-07 had to settle two open questions against the installed
`lerobot==0.6.1` source before choosing a transport tier:

1. Dispatch. `"groot"` is in `async_inference/constants.py::SUPPORTED_POLICIES`,
   and `get_policy_class("groot")` resolves to `GrootPolicy`, which wraps GR00T
   N1.7 (N1.5 support was removed). So the gRPC-bridge tier applies, not the
   Tier C HTTP-adapter fallback.
2. Action space. Every SO-101 N1.7 checkpoint on the HF Hub declares joint-space
   action groups (`single_arm` RELATIVE NON_EEF + `gripper` ABSOLUTE, or plain
   absolute joints). The server-side postprocessor (`GrootN17ActionDecodeStep`)
   un-normalizes them and turns relative joints back into absolute targets with
   the state cached at preprocess time. The client therefore receives ABSOLUTE
   joint targets. Applying a relative-EE/IK conversion on top would re-add a
   delta to an absolute target (the pinned-to-limits failure in 14-RESEARCH.md
   Pitfall 1), so the declared action space is `absolute_joint`.

The stock 0.6.1 PolicyServer post-processes one action at a time, which raises
`NotImplementedError` for native relative-action N1.7 checkpoints. The
`groot_n17_server.py` launcher decodes the whole chunk instead. Its tests run the
real installed `GrootN17PackInputsStep` / `GrootN17ActionDecodeStep` /
`PolicyServer._predict_action_chunk` path; only the 3B model itself is faked.

Nothing here opens a serial port, a camera, a gRPC channel or the network.
"""

import json
import math
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from lerobot.async_inference import constants as async_constants
from lerobot.async_inference.helpers import TimedObservation
from lerobot.async_inference.policy_server import PolicyServer
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.groot.processor_groot import (
    GrootActionUnpackUnnormalizeStep,
    GrootN17ActionDecodeStep,
    GrootN17PackInputsStep,
)
from lerobot.processor import PolicyProcessorPipeline
from lerobot.processor.converters import (
    batch_to_transition,
    policy_action_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)
from lerobot.utils.constants import OBS_STATE

from vla_bridge import backends, robot_client
from vla_bridge.action_contract import JOINT_ORDER
from vla_bridge.backends import base_grpc, relative_ee_conversion

BACKEND = "groot-n17"
CONFIG_NAME = "groot_n17"
CAMERA1 = "observation.images.camera1"


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


def _write_config(tmp_path, monkeypatch, **fields):
    monkeypatch.setattr(backends, "BACKEND_CONFIG_DIR", tmp_path)
    config = {"action_space": "absolute_joint", "action_units": "degrees"}
    config.update(fields)
    config = {k: v for k, v in config.items() if v is not None}
    (tmp_path / f"{CONFIG_NAME}.json").write_text(json.dumps(config))


def _record_grpc_build(monkeypatch):
    calls = {}
    sentinel = object()

    def fake_build(policy_type, checkpoint, args, camera_names, joint_limits_deg, convert_fn=None):
        calls.update(policy_type=policy_type, checkpoint=checkpoint, args=args,
                     camera_names=camera_names, joint_limits_deg=joint_limits_deg,
                     convert_fn=convert_fn)
        return sentinel

    monkeypatch.setattr(base_grpc, "build_grpc_action_source", fake_build)
    return calls, sentinel


# --- Dispatch verification against the INSTALLED lerobot (the tier decision) -------


def test_installed_policy_server_dispatches_groot():
    """The live-source fact the gRPC-tier decision rests on."""
    from vla_bridge.backends import groot_n17

    assert groot_n17.POLICY_TYPE == "groot"
    assert "groot" in async_constants.SUPPORTED_POLICIES
    assert groot_n17.policy_server_dispatches(groot_n17.POLICY_TYPE) is True


def test_installed_groot_policy_is_n17_not_n15():
    from lerobot.policies.factory import get_policy_class
    from lerobot.policies.groot import configuration_groot
    from lerobot.policies.groot.modeling_groot import GrootPolicy

    assert get_policy_class("groot") is GrootPolicy
    assert configuration_groot.GROOT_N1_7_BASE_MODEL == "nvidia/GR00T-N1.7-3B"
    assert configuration_groot.GrootConfig.max_action_dim == 132
    with pytest.raises(ValueError, match="N1.5 support was removed"):
        configuration_groot.normalize_groot_model_version("n1.5")


# --- Registration ------------------------------------------------------------------


def test_groot_n17_is_registered_on_the_grpc_bridge_lane():
    from vla_bridge.backends import groot_n17

    assert BACKEND in backends.BACKEND_REGISTRY
    assert BACKEND in backends.GRPC_BRIDGE_BACKENDS
    assert backends.BACKEND_REGISTRY[BACKEND] is groot_n17.build


# --- build(): dispatch-supported branch -----------------------------------------------


def test_build_uses_grpc_bridge_with_groot_policy_type_and_convert_fn(monkeypatch, tmp_path):
    _write_config(tmp_path, monkeypatch)
    calls, sentinel = _record_grpc_build(monkeypatch)
    args = _args(checkpoint="cli/groot-ckpt")
    camera_names = {0: "wrist", 1: "overhead"}
    joint_limits = {"gripper": (0.0, 100.0)}

    result = backends.BACKEND_REGISTRY[BACKEND](args, camera_names, joint_limits)

    assert result is sentinel
    assert calls["policy_type"] == "groot"
    assert calls["checkpoint"] == "cli/groot-ckpt"
    assert calls["args"] is args
    assert calls["camera_names"] == camera_names
    assert calls["joint_limits_deg"] == joint_limits
    convert_fn = calls["convert_fn"]
    assert callable(convert_fn)
    # The wired convert_fn is relative_ee_conversion's absolute-joint gate.
    action = dict.fromkeys(JOINT_ORDER, 30.0)
    assert convert_fn(action, dict.fromkeys(JOINT_ORDER, 25.0)) == action


def test_build_falls_back_to_config_checkpoint(monkeypatch, tmp_path):
    _write_config(tmp_path, monkeypatch, checkpoint="config/groot-ckpt")
    calls, _ = _record_grpc_build(monkeypatch)

    backends.BACKEND_REGISTRY[BACKEND](_args(checkpoint=None), {0: "wrist"}, {})

    assert calls["checkpoint"] == "config/groot-ckpt"


def test_build_without_checkpoint_exits_before_connecting(monkeypatch, tmp_path):
    _write_config(tmp_path, monkeypatch)
    calls, _ = _record_grpc_build(monkeypatch)

    with pytest.raises(SystemExit, match=r"--backend groot-n17 .*checkpoint"):
        backends.BACKEND_REGISTRY[BACKEND](_args(checkpoint=None), {0: "wrist"}, {})
    assert calls == {}


# --- build(): dispatch-unsupported branch -----------------------------------------------


def test_build_refuses_when_policy_server_does_not_dispatch_groot(monkeypatch, tmp_path):
    """If a lerobot upgrade drops "groot" from the async dispatch table, build() fails
    fast instead of handshaking into a server-side ValueError (silent hold)."""
    _write_config(tmp_path, monkeypatch)
    calls, _ = _record_grpc_build(monkeypatch)
    monkeypatch.setattr(async_constants, "SUPPORTED_POLICIES", ["act", "smolvla", "pi0"])

    with pytest.raises(SystemExit, match="SUPPORTED_POLICIES"):
        backends.BACKEND_REGISTRY[BACKEND](_args(checkpoint="cli/ckpt"), {0: "wrist"}, {})
    assert calls == {}


# --- build(): explicit action-space declaration ------------------------------------------


@pytest.mark.parametrize(
    "fields, pattern",
    [
        (dict(action_space=None), "action_space"),
        (dict(action_space="relative_ee"), "relative_ee"),
        (dict(action_units=None), "action_units"),
        (dict(action_units="radians"), "radians"),
    ],
    ids=["missing-space", "relative-ee", "missing-units", "radians"],
)
def test_build_requires_a_supported_explicit_action_space(monkeypatch, tmp_path, fields, pattern):
    _write_config(tmp_path, monkeypatch, **fields)
    calls, _ = _record_grpc_build(monkeypatch)

    with pytest.raises(SystemExit, match=pattern):
        backends.BACKEND_REGISTRY[BACKEND](_args(checkpoint="cli/ckpt"), {0: "wrist"}, {})
    assert calls == {}


# --- relative_ee_conversion.convert_groot_n17_action --------------------------------------


def test_convert_returns_joint_order_floats_for_an_absolute_joint_dict():
    raw = {"shoulder_pan": 1, "shoulder_lift": -2.5, "elbow_flex": 3.0,
           "wrist_flex": 60.0, "wrist_roll": -4.0, "gripper": 20.0}

    out = relative_ee_conversion.convert_groot_n17_action(raw, dict.fromkeys(JOINT_ORDER, 0.0))

    assert list(out) == list(JOINT_ORDER)
    assert all(type(v) is float and math.isfinite(v) for v in out.values())
    assert out["shoulder_pan"] == 1.0
    assert out["shoulder_lift"] == -2.5


def test_convert_does_not_reapply_a_delta_to_an_absolute_target():
    """The server already decoded relative joints against the state. Adding the
    state again is the pinned-to-limits failure (14-RESEARCH.md Pitfall 1)."""
    joint_state = dict.fromkeys(JOINT_ORDER, 25.0)
    raw = dict.fromkeys(JOINT_ORDER, 30.0)

    out = relative_ee_conversion.convert_groot_n17_action(raw, joint_state)

    assert out == dict.fromkeys(JOINT_ORDER, 30.0)


def test_convert_slices_a_padded_vector_to_the_first_six_checkpoint_dims():
    """Mirrors lerobot's own slice (`actions[..., :original_action_dim]`): the SO-101
    action groups (single_arm[5] + gripper[1]) are the first six of the 132-D vector."""
    padded = [float(i) for i in range(1, 7)] + [999.0] * 126
    assert len(padded) == 132

    out = relative_ee_conversion.convert_groot_n17_action(padded, dict.fromkeys(JOINT_ORDER, 0.0))

    assert out == {joint: float(i) for i, joint in enumerate(JOINT_ORDER, start=1)}


@pytest.mark.parametrize(
    "raw",
    [
        {**dict.fromkeys(JOINT_ORDER, 1.0), "elbow_flex": float("nan")},
        {**dict.fromkeys(JOINT_ORDER, 1.0), "gripper": float("inf")},
        {k: 1.0 for k in JOINT_ORDER if k != "wrist_roll"},
        {**dict.fromkeys(JOINT_ORDER, 1.0), "gripper": True},
        {**dict.fromkeys(JOINT_ORDER, 1.0), "gripper": "open"},
        [1.0, 2.0, 3.0],
        None,
    ],
    ids=["nan", "inf", "missing-joint", "bool", "string", "short-vector", "none"],
)
def test_convert_rejects_malformed_actions(raw):
    with pytest.raises(ValueError):
        relative_ee_conversion.convert_groot_n17_action(raw, dict.fromkeys(JOINT_ORDER, 0.0))


def test_convert_rejects_relative_ee_action_space():
    with pytest.raises(ValueError, match="relative_ee"):
        relative_ee_conversion.convert_groot_n17_action(
            dict.fromkeys(JOINT_ORDER, 1.0), dict.fromkeys(JOINT_ORDER, 0.0),
            action_space="relative_ee",
        )


# --- base_grpc.build_grpc_action_source(convert_fn=...) ---------------------------------


class _FakeClient:
    def __init__(self, robot):
        self.robot = robot
        self._stereo_camera = object()

    def stop(self):
        pass


def test_build_grpc_action_source_without_convert_fn_is_unchanged(monkeypatch, mock_robot):
    """Backward compatibility for smolvla/pi0/pi05/act/diffusion: no wrapper."""
    monkeypatch.setattr(robot_client, "connect_bridge", lambda *a, **k: _FakeClient(mock_robot))

    handles = base_grpc.build_grpc_action_source("act", "x/y", _args(), {0: "wrist"}, {})

    assert type(handles.action_source) is robot_client.BridgeActionSource


def test_build_grpc_action_source_wraps_with_convert_fn(monkeypatch, mock_robot):
    monkeypatch.setattr(robot_client, "connect_bridge", lambda *a, **k: _FakeClient(mock_robot))
    seen = []

    def convert_fn(action, joint_state):
        seen.append((dict(action), dict(joint_state)))
        return {k: v + 1.0 for k, v in action.items()}

    handles = base_grpc.build_grpc_action_source(
        "groot", "x/y", _args(), {0: "wrist"}, {}, convert_fn=convert_fn
    )
    inner = handles.action_source.inner
    assert type(inner) is robot_client.BridgeActionSource
    joint_state = dict.fromkeys(JOINT_ORDER, 10.0)
    monkeypatch.setattr(inner, "get_action", lambda js, instr: (dict.fromkeys(JOINT_ORDER, 12.0), "x/y@unknown"))

    action, version = handles.action_source.get_action(joint_state, "pick")

    assert action == dict.fromkeys(JOINT_ORDER, 13.0)
    assert version == "x/y@unknown"
    assert seen == [(dict.fromkeys(JOINT_ORDER, 12.0), joint_state)]


def test_converted_source_holds_position_when_convert_fn_raises(monkeypatch, mock_robot):
    monkeypatch.setattr(robot_client, "connect_bridge", lambda *a, **k: _FakeClient(mock_robot))

    def boom(action, joint_state):
        raise ValueError("bad action")

    handles = base_grpc.build_grpc_action_source(
        "groot", "x/y", _args(), {0: "wrist"}, {}, convert_fn=boom
    )
    inner = handles.action_source.inner
    inner.last_details = {}

    def fake_get_action(js, instr):
        inner.last_details = dict(raw_model_output={"a": 1}, bridge_flags=[], observation=None)
        return dict.fromkeys(JOINT_ORDER, 12.0), "x/y@unknown"

    monkeypatch.setattr(inner, "get_action", fake_get_action)
    joint_state = dict.fromkeys(JOINT_ORDER, 10.0)

    action, version = handles.action_source.get_action(joint_state, "pick")

    assert action == joint_state
    assert action is not joint_state
    assert version == "x/y@action-conversion-error-holding-position"
    assert any("bad action" in f for f in handles.action_source.last_details["bridge_flags"])


# --- groot_n17_server.py: Colab PolicyServer launcher -----------------------------------
#
# Real installed lerobot steps, configured exactly like the shipped candidate
# checkpoint's serialized pipelines (tsangb34/groot-n17-so101-cube-drawer-40k):
# state/action groups single_arm[5] + gripper[1]; single_arm is RELATIVE NON_EEF with
# per-chunk-step relative stats, gripper ABSOLUTE; use_percentiles=True.

HORIZON = 4
STATE = [1.0, 2.0, 3.0, 60.0, 4.0, 20.0]


def _stats_entry(low, high, rows=None):
    if rows is None:
        return {"min": list(low), "max": list(high), "q01": list(low), "q99": list(high)}
    return {
        "min": [[lo - r for lo in low] for r in range(rows)],
        "max": [[hi + r for hi in high] for r in range(rows)],
    }


def _groot_pipelines(relative=True):
    raw_stats = {
        "state": {"single_arm": _stats_entry([-100.0] * 5, [100.0] * 5), "gripper": _stats_entry([0.0], [100.0])},
        "action": {"single_arm": _stats_entry([-100.0] * 5, [100.0] * 5), "gripper": _stats_entry([0.0], [50.0])},
        "relative_action": {"single_arm": _stats_entry([-10.0] * 5, [10.0] * 5, rows=HORIZON)},
    }
    modality_config = {
        "state": {"modality_keys": ["single_arm", "gripper"]},
        "action": {
            "modality_keys": ["single_arm", "gripper"],
            "action_configs": [
                {"rep": "RELATIVE" if relative else "ABSOLUTE", "type": "NON_EEF", "format": "DEFAULT", "state_key": None},
                {"rep": "ABSOLUTE", "type": "NON_EEF", "format": "DEFAULT", "state_key": None},
            ],
            "delta_indices": list(range(HORIZON)),
        },
        "video": {"modality_keys": ["front", "wrist"], "delta_indices": [0]},
    }
    pack = GrootN17PackInputsStep(
        action_horizon=HORIZON, valid_action_horizon=HORIZON, embodiment_mapping={"new_embodiment": 10},
        use_percentiles=True, raw_stats=raw_stats, modality_config=modality_config,
        video_modality_keys=["front", "wrist"],
    )
    decode = GrootN17ActionDecodeStep(
        env_action_dim=6, raw_stats=raw_stats, modality_config=modality_config,
        use_percentiles=True, use_relative_action=relative, pack_step=pack,
    )
    pre = PolicyProcessorPipeline(steps=[pack], to_transition=batch_to_transition, to_output=transition_to_batch)
    post = PolicyProcessorPipeline(steps=[decode], to_transition=policy_action_to_transition,
                                   to_output=transition_to_policy_action)
    return pre, post


def _expected_relative_chunk(normalized):
    """Hand-computed decode: per-step relative stats, + state for single_arm; absolute gripper."""
    out = []
    for t, row in enumerate(normalized):
        lo, hi = -10.0 - t, 10.0 + t
        arm = [(row[j] + 1.0) * 0.5 * (hi - lo) + lo + STATE[j] for j in range(5)]
        q01, q99 = 0.0, 50.0  # use_percentiles -> q01/q99 for absolute groups
        out.append(arm + [(row[5] + 1.0) * 0.5 * (q99 - q01) + q01])
    return out


class _FakeGrootPolicy:
    """Stands in for the 3B GrootPolicy: returns a fixed normalized (1, T, 6) chunk."""

    def __init__(self, normalized):
        self.normalized = torch.tensor([normalized], dtype=torch.float32)
        self.config = SimpleNamespace(
            image_features={CAMERA1: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 48, 64))}
        )

    def predict_action_chunk(self, observation):
        assert observation["state"].shape == (1, 1, 132)  # packed by the real pack step
        return self.normalized.clone()


def _make_server(policy_type="groot", relative=True, normalized=None):
    if normalized is None:
        normalized = [[0.1 * t, -0.2, 0.0, 0.5, -0.5, 0.2 * t] for t in range(HORIZON)]
    from lerobot.async_inference.configs import PolicyServerConfig

    server = PolicyServer(PolicyServerConfig())
    server.policy_type = policy_type
    server.policy = _FakeGrootPolicy(normalized)
    server.preprocessor, server.postprocessor = _groot_pipelines(relative=relative)
    server.actions_per_chunk = 50
    server.lerobot_features = {
        OBS_STATE: {"dtype": "float32", "shape": (6,), "names": [f"{j}.pos" for j in JOINT_ORDER]},
        CAMERA1: {"dtype": "image", "shape": (48, 64, 3), "names": ["height", "width", "channels"]},
    }
    return server, normalized


def _timed_observation():
    """What today's bridge sends: 6 joint positions + camera1 only + the task."""
    raw = {f"{joint}.pos": value for joint, value in zip(JOINT_ORDER, STATE)}
    raw["camera1"] = np.zeros((48, 64, 3), dtype=np.uint8)
    raw["task"] = "pick the cube"
    return TimedObservation(timestamp=time.time(), timestep=7, observation=raw)


@pytest.fixture
def restore_predict_action_chunk(monkeypatch):
    monkeypatch.setattr(PolicyServer, "_predict_action_chunk", PolicyServer._predict_action_chunk)


def test_stock_policy_server_cannot_decode_a_relative_groot_chunk(restore_predict_action_chunk):
    """Canary for the launcher: the stock 0.6.1 server post-processes one (B, D) action
    at a time, which GrootN17ActionDecodeStep refuses for native relative actions. If a
    lerobot upgrade makes this pass, re-review groot_n17_server.py."""
    server, _ = _make_server()

    with pytest.raises(NotImplementedError, match="one step at a time"):
        server._predict_action_chunk(_timed_observation())


def test_patched_server_decodes_the_whole_relative_chunk(restore_predict_action_chunk):
    from vla_bridge import groot_n17_server

    groot_n17_server.apply_patch()
    server, normalized = _make_server()

    chunk = server._predict_action_chunk(_timed_observation())

    assert len(chunk) == HORIZON
    assert [a.get_timestep() for a in chunk] == [7, 8, 9, 10]
    got = [a.get_action().tolist() for a in chunk]
    assert np.allclose(got, _expected_relative_chunk(normalized), atol=1e-4)
    assert all(a.get_action().shape == (6,) for a in chunk)


def test_patched_server_keeps_absolute_groot_checkpoints_working(restore_predict_action_chunk):
    from vla_bridge import groot_n17_server

    groot_n17_server.apply_patch()
    server, normalized = _make_server(relative=False)
    unnorm = GrootActionUnpackUnnormalizeStep(
        env_action_dim=6, stats={"action": {"min": torch.full((6,), -100.0), "max": torch.full((6,), 100.0)}},
        normalize_min_max=True,
    )
    server.postprocessor = PolicyProcessorPipeline(
        steps=[unnorm], to_transition=policy_action_to_transition, to_output=transition_to_policy_action
    )

    chunk = server._predict_action_chunk(_timed_observation())

    got = np.array([a.get_action().tolist() for a in chunk])
    assert np.allclose(got, np.array(normalized) * 100.0, atol=1e-4)


def test_patched_server_leaves_other_policy_types_on_the_stock_path(monkeypatch, restore_predict_action_chunk):
    from vla_bridge import groot_n17_server

    calls = []
    monkeypatch.setattr(PolicyServer, "_predict_action_chunk",
                        lambda self, obs: calls.append(self.policy_type) or "stock")
    groot_n17_server.apply_patch()
    server, _ = _make_server(policy_type="act")

    assert server._predict_action_chunk(_timed_observation()) == "stock"
    assert calls == ["act"]


def test_apply_patch_is_idempotent(restore_predict_action_chunk):
    from vla_bridge import groot_n17_server

    groot_n17_server.apply_patch()
    first = PolicyServer._predict_action_chunk
    groot_n17_server.apply_patch()

    assert PolicyServer._predict_action_chunk is first
    server, _ = _make_server(policy_type="groot")
    assert len(server._predict_action_chunk(_timed_observation())) == HORIZON  # no recursion


def test_apply_patch_refuses_an_unreviewed_lerobot_version(monkeypatch, restore_predict_action_chunk):
    from vla_bridge import groot_n17_server

    original = PolicyServer._predict_action_chunk
    monkeypatch.setattr(groot_n17_server, "_installed_lerobot_version", lambda: "0.7.0")

    with pytest.raises(SystemExit, match="0.7.0"):
        groot_n17_server.apply_patch()
    assert PolicyServer._predict_action_chunk is original


def test_main_patches_before_serving(monkeypatch, restore_predict_action_chunk):
    from lerobot.async_inference import policy_server

    from vla_bridge import groot_n17_server

    seen = {}

    def fake_serve():
        seen["patched"] = groot_n17_server.is_patched()

    monkeypatch.setattr(policy_server, "serve", fake_serve)

    groot_n17_server.main()

    assert seen == {"patched": True}


# --- Shipped config ------------------------------------------------------------------------


def test_shipped_config_declares_the_action_space_explicitly():
    config = backends.load_backend_config(CONFIG_NAME)

    assert config is not None
    assert config["action_space"] == "absolute_joint"
    assert config["action_units"] == "degrees"
    assert config["action_space"] in relative_ee_conversion.SUPPORTED_ACTION_SPACES


def test_shipped_config_has_no_default_checkpoint_and_records_candidates():
    """No public SO-101 N1.7 checkpoint declares camera1, the only camera today's
    bridge delivers, so --checkpoint is required (same as diffusion)."""
    config = backends.load_backend_config(CONFIG_NAME)

    assert "checkpoint" not in config
    assert config["candidate_checkpoint"] == "tsangb34/groot-n17-so101-cube-drawer-40k"
    assert config["candidate_verified_revision"] == "12e001ccd3f6146c50819b20384714c4903b1faa"
    assert "camera1" in config["notes"]
    assert "nvidia/GR00T-N1.7-3B" in config["notes"]


def test_shipped_config_makes_build_exit_with_the_reason(monkeypatch):
    calls, _ = _record_grpc_build(monkeypatch)

    with pytest.raises(SystemExit, match=r"(?s)--backend groot-n17 .*checkpoint.*camera1"):
        backends.BACKEND_REGISTRY[BACKEND](_args(checkpoint=None), {0: "wrist"}, {})
    assert calls == {}


# --- Mocked round trip (Task 2): real base_grpc + real BridgeActionSource + gate ---------
#
# Only `connect_bridge()` is faked (test_backend_pi0._FakePiBridgeClient, built on
# test_robot_client.FakeBridgeClient/FakeTimedAction). Everything after the handshake is
# production code: base_grpc wraps the client in BridgeActionSource, pop_validated_action()
# -> safety_validator.validate_action(), then ConvertedBridgeActionSource applies the
# shipped config's absolute_joint gate.

ROUND_TRIP_CHECKPOINT = "someone/so101-groot-n17-camera1"


def _groot_round_trip(monkeypatch, mock_robot, mock_calibration_file, queued):
    from test_backend_pi0 import _FakePiBridgeClient

    from vla_bridge import action_contract

    joint_limits = action_contract.load_joint_limits_deg(mock_calibration_file)
    connect_calls = {}

    def fake_connect_bridge(server_address, checkpoint, **kwargs):
        connect_calls.update(server_address=server_address, checkpoint=checkpoint, **kwargs)
        connect_calls["client"] = _FakePiBridgeClient(mock_robot, queued)
        return connect_calls["client"]

    monkeypatch.setattr(robot_client, "connect_bridge", fake_connect_bridge)
    handles = backends.BACKEND_REGISTRY[BACKEND](
        _args(checkpoint=ROUND_TRIP_CHECKPOINT), {0: "wrist", 1: "overhead"}, joint_limits
    )
    return handles, connect_calls, joint_limits


def test_groot_round_trip_returns_validated_action_without_send_action(
    monkeypatch, mock_robot, mock_calibration_file
):
    queued = {f"{joint}.pos": 12.5 for joint in JOINT_ORDER}
    handles, connect_calls, _ = _groot_round_trip(monkeypatch, mock_robot, mock_calibration_file, queued)

    client = connect_calls["client"]
    assert isinstance(handles, backends.BackendHandles)
    assert connect_calls["policy_type"] == "groot"
    assert connect_calls["checkpoint"] == ROUND_TRIP_CHECKPOINT
    assert connect_calls["wrist_camera_index"] == 0
    assert handles.robot is mock_robot
    assert handles.shutdown == client.stop
    assert handles.skip_camera_names == frozenset({"overhead"})
    assert isinstance(handles.action_source, base_grpc.ConvertedBridgeActionSource)
    assert type(handles.action_source.inner) is robot_client.BridgeActionSource

    joint_state = dict.fromkeys(JOINT_ORDER, 10.0)
    action, model_version = handles.action_source.get_action(joint_state, "pick the cube")

    assert list(action) == list(JOINT_ORDER)
    assert all(type(v) is float and math.isfinite(v) for v in action.values())
    assert action == dict.fromkeys(JOINT_ORDER, 12.5)  # absolute target: not 22.5, not held at 10
    assert model_version == f"{ROUND_TRIP_CHECKPOINT}@unknown"
    assert handles.action_source.last_details["raw_model_output"] == dict.fromkeys(JOINT_ORDER, 12.5)
    assert client.observations_sent == ["pick the cube"]
    assert mock_robot.sent_actions == []  # the source never sends; run_episode does


def test_groot_round_trip_clamps_an_out_of_range_chunk_action(
    monkeypatch, mock_robot, mock_calibration_file
):
    """A server-returned target outside the joint limits is clamped by
    safety_validator.validate_action(), never passed through raw (T-14-08)."""
    queued = {f"{joint}.pos": 12.5 for joint in JOINT_ORDER}
    queued["elbow_flex.pos"] = 10_000.0
    handles, _, joint_limits = _groot_round_trip(monkeypatch, mock_robot, mock_calibration_file, queued)

    action, _ = handles.action_source.get_action(dict.fromkeys(JOINT_ORDER, 10.0), "pick the cube")

    low, high = joint_limits["elbow_flex"]
    assert low <= action["elbow_flex"] <= high
    assert action["elbow_flex"] != 10_000.0
    assert mock_robot.sent_actions == []


def test_groot_round_trip_holds_position_when_queue_is_empty(
    monkeypatch, mock_robot, mock_calibration_file
):
    """What the robot does if the server never returns a chunk (stock server's per-step
    decode NotImplementedError, or an undelivered camera key): it holds."""
    queued = {f"{joint}.pos": 12.5 for joint in JOINT_ORDER}
    handles, connect_calls, _ = _groot_round_trip(monkeypatch, mock_robot, mock_calibration_file, queued)
    connect_calls["client"].action_queue.get_nowait()  # nothing queued
    joint_state = dict.fromkeys(JOINT_ORDER, 10.0)

    action, model_version = handles.action_source.get_action(joint_state, "pick the cube")

    assert action == joint_state
    assert "holding position" in model_version
    assert mock_robot.sent_actions == []


# --- CLI dispatch through run_vla_episode.main() ----------------------------------------


def test_cli_backend_flag_dispatches_to_real_groot_build(monkeypatch, tmp_path):
    from test_run_vla_episode import _ClosableActionSource, _dispatch_main, _FakeBackendRobot

    calls = {}
    handles = backends.BackendHandles(action_source=_ClosableActionSource(), robot=_FakeBackendRobot())

    def fake_build(policy_type, checkpoint, args, camera_names, joint_limits_deg, convert_fn=None):
        calls.update(policy_type=policy_type, checkpoint=checkpoint, has_convert_fn=callable(convert_fn))
        return handles

    monkeypatch.setattr(base_grpc, "build_grpc_action_source", fake_build)

    result = _dispatch_main(monkeypatch, tmp_path, [
        "--backend", BACKEND, "--server-address", "0.tcp.ngrok.io:12345",
        "--checkpoint", "cli/ckpt",
    ], {})

    assert calls == dict(policy_type="groot", checkpoint="cli/ckpt", has_convert_fn=True)
    assert result["run_episode_called"] is True
    assert result["action_source"] is handles.action_source


@pytest.mark.parametrize(
    "partial_args",
    [["--checkpoint", "cli/ckpt"], ["--server-address", "0.tcp.ngrok.io:12345"], []],
    ids=["checkpoint-only", "server-address-only", "neither"],
)
def test_cli_groot_requires_bridge_args(monkeypatch, tmp_path, partial_args):
    from test_run_vla_episode import _dispatch_main

    calls, _ = _record_grpc_build(monkeypatch)

    with pytest.raises(SystemExit):
        _dispatch_main(monkeypatch, tmp_path, ["--backend", BACKEND, *partial_args], {})
    assert calls == {}

