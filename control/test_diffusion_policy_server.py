"""Tests for `vla_bridge/diffusion_policy_server.py`, the Colab-side launcher that
fixes lerobot 0.6.1's PolicyServer for Diffusion Policy (Plan 14-06).

These run the real installed lerobot code path: `raw_observation_to_observation()`
-> the default diffusion preprocessor -> `PolicyServer._get_action_chunk()` ->
`DiffusionPolicy.predict_action_chunk()` -> `DiffusionModel.generate_actions()`,
on a tiny untrained `DiffusionPolicy`. Only the `diffusers` noise scheduler is
stubbed, because `diffusers` is a Colab-only dependency that is not installed here.
"""

import types

import numpy as np
import pytest
import torch
from lerobot.async_inference.helpers import raw_observation_to_observation
from lerobot.async_inference.policy_server import PolicyServer
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.diffusion import modeling_diffusion
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils import import_utils
from lerobot.utils.constants import OBS_STATE

from vla_bridge import diffusion_policy_server
from vla_bridge.action_contract import JOINT_ORDER

CAMERA1 = "observation.images.camera1"
CAMERA2 = "observation.images.camera2"
N_ACTION_STEPS = 8


class _StubScheduler:
    """Just enough of diffusers' DDPMScheduler for `conditional_sample()`."""

    def __init__(self, **kwargs):
        self.config = types.SimpleNamespace(num_train_timesteps=kwargs["num_train_timesteps"])
        self.timesteps = []

    def set_timesteps(self, n):
        self.timesteps = list(range(n - 1, -1, -1))

    def step(self, model_output, t, sample, generator=None):
        return types.SimpleNamespace(prev_sample=sample - 0.01 * model_output)


@pytest.fixture
def restore_predict_action_chunk(monkeypatch):
    """Undo any class-level patch a test applies."""
    monkeypatch.setattr(
        modeling_diffusion.DiffusionPolicy,
        "predict_action_chunk",
        modeling_diffusion.DiffusionPolicy.predict_action_chunk,
    )


@pytest.fixture
def make_server(monkeypatch, restore_predict_action_chunk):
    monkeypatch.setitem(import_utils._require_package_cache, "diffusers", True)
    monkeypatch.setattr(modeling_diffusion, "DDPMScheduler", _StubScheduler)

    def _make(image_keys=(CAMERA1,)):
        torch.manual_seed(0)
        config = DiffusionConfig(
            input_features={
                OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(6,)),
                **{k: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)) for k in image_keys},
            },
            output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(6,))},
            pretrained_backbone_weights=None,
            crop_shape=None,
            down_dims=(16, 32),
            horizon=16,
            n_action_steps=N_ACTION_STEPS,
            n_obs_steps=2,
            num_inference_steps=2,
            device="cpu",
        )
        policy = modeling_diffusion.DiffusionPolicy(config).eval()
        minmax = {"min": torch.full((6,), -150.0), "max": torch.full((6,), 150.0)}
        stats = {OBS_STATE: minmax, "action": minmax}
        for key in image_keys:
            stats[key] = {"mean": torch.zeros(3, 1, 1), "std": torch.ones(3, 1, 1)}
        preprocessor, _ = make_pre_post_processors(config, dataset_stats=stats)
        server = types.SimpleNamespace(policy=policy, actions_per_chunk=50)
        return server, preprocessor

    return _make


def _bridge_observation(preprocessor, policy):
    """Exactly what today's bridge sends: 6 joint positions + camera1 only."""
    lerobot_features = {
        OBS_STATE: {"dtype": "float32", "shape": (6,), "names": [f"{j}.pos" for j in JOINT_ORDER]},
        CAMERA1: {"dtype": "image", "shape": (480, 640, 3), "names": ["height", "width", "channels"]},
    }
    raw = {f"{joint}.pos": 10.0 for joint in JOINT_ORDER}
    raw["camera1"] = np.zeros((480, 640, 3), dtype=np.uint8)
    raw["task"] = "pick the cube"
    observation = raw_observation_to_observation(raw, lerobot_features, policy.config.image_features)
    return preprocessor(observation)


def test_stock_policy_server_cannot_run_diffusion(make_server):
    """Upstream huggingface/lerobot#3861, reproduced on the installed 0.6.1. If a lerobot
    upgrade makes this pass, the launcher's patch is no longer needed."""
    server, preprocessor = make_server()
    observation = _bridge_observation(preprocessor, server.policy)
    assert observation[OBS_STATE].shape == (1, 6)  # no n_obs_steps dimension

    with pytest.raises(AssertionError):
        PolicyServer._get_action_chunk(server, observation)


def test_patched_policy_server_returns_an_action_chunk(make_server):
    server, preprocessor = make_server()
    diffusion_policy_server.apply_patch()

    for _ in range(2):  # stateless: works on every call, not just the first
        chunk = PolicyServer._get_action_chunk(server, _bridge_observation(preprocessor, server.policy))
        assert chunk.shape == (1, N_ACTION_STEPS, 6)
        assert torch.isfinite(chunk).all()


def test_patch_leaves_batches_with_a_time_dimension_unchanged(make_server):
    """Offline dataloader batches already carry n_obs_steps; upstream PR #3886 keeps them as-is."""
    server, _ = make_server()
    diffusion_policy_server.apply_patch()
    calls = {}
    real_generate = server.policy.diffusion.generate_actions

    def spy(batch, noise=None):
        calls["state"] = batch[OBS_STATE].shape
        return real_generate(batch, noise=noise)

    server.policy.diffusion.generate_actions = spy
    batch = {OBS_STATE: torch.zeros(3, 2, 6), CAMERA1: torch.zeros(3, 2, 3, 64, 64)}

    actions = server.policy.predict_action_chunk(batch)

    assert calls["state"] == (3, 2, 6)
    assert actions.shape == (3, N_ACTION_STEPS, 6)


def test_patched_diffusion_still_does_not_mask_a_missing_camera(make_server):
    """Why "diffusion" is in check_checkpoint.NO_CAMERA_MASKING_BACKENDS: a checkpoint that
    also declares camera2 KeyErrors on today's camera1-only bridge, patched or not."""
    server, preprocessor = make_server(image_keys=(CAMERA1, CAMERA2))
    diffusion_policy_server.apply_patch()
    observation = _bridge_observation(preprocessor, server.policy)

    with pytest.raises(KeyError, match="camera2"):
        PolicyServer._get_action_chunk(server, observation)


def test_apply_patch_is_idempotent(restore_predict_action_chunk):
    diffusion_policy_server.apply_patch()
    diffusion_policy_server.apply_patch()

    patched = modeling_diffusion.DiffusionPolicy.predict_action_chunk
    assert patched is diffusion_policy_server.predict_action_chunk


def test_apply_patch_refuses_an_unreviewed_lerobot_version(monkeypatch, restore_predict_action_chunk):
    original = modeling_diffusion.DiffusionPolicy.predict_action_chunk
    monkeypatch.setattr(diffusion_policy_server, "_installed_lerobot_version", lambda: "0.7.0")

    with pytest.raises(SystemExit, match="0.7.0"):
        diffusion_policy_server.apply_patch()
    assert modeling_diffusion.DiffusionPolicy.predict_action_chunk is original


def test_main_patches_before_serving(monkeypatch, restore_predict_action_chunk):
    from lerobot.async_inference import policy_server

    seen = {}

    def fake_serve():
        seen["patched"] = (
            modeling_diffusion.DiffusionPolicy.predict_action_chunk
            is diffusion_policy_server.predict_action_chunk
        )

    monkeypatch.setattr(policy_server, "serve", fake_serve)

    diffusion_policy_server.main()

    assert seen == {"patched": True}
