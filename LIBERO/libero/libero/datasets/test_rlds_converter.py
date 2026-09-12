"""Tests for rlds_converter.py — TUNE-01 RLDS conversion (Phase 6).

Local unit tests (schema validation + per-episode HDF5 loading) exercise pure
numpy/h5py code paths, no tensorflow needed -- matches this project's
confirmed local ``libero`` conda env capability (h5py 3.14.0 available,
tensorflow_datasets NOT available). The genuine-TFDS-write tests are guarded
by ``pytest.importorskip("tensorflow_datasets")`` and are expected to skip
locally, running for real only on Colab once Plan 06-02's training notebook
installs ``tensorflow_datasets``.

Import-path note (mirrors test_normalization.py): insert the repo root so
``LIBERO.libero.libero.envs`` absolute imports resolve, and ``LIBERO/libero``
so ``libero.datasets.*`` resolves regardless of invocation dir.
"""

import glob
import os
import sys

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_LIBERO_LIBERO = os.path.join(_REPO_ROOT, "LIBERO", "libero")
if _LIBERO_LIBERO not in sys.path:
    sys.path.insert(0, _LIBERO_LIBERO)

import numpy as np
import pytest

from libero.datasets.rlds_converter import (
    validate_episode_arrays,
    load_episodes_from_hdf5,
    hdf5_to_rlds,
)

# 04-02's real collected dataset (single-file layout, post D-07/D-08
# retargeting). The glob does not assert a fixed file count.
_REAL_DATASET_GLOB = os.path.join(
    os.path.dirname(__file__), "..", "..", "datasets", "soarm_spatial", "*_demo.hdf5"
)
_REAL_BDDL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "bddl_files",
    "libero_goal",
    "put_the_cream_cheese_in_the_bowl.bddl",
)
_REAL_LANGUAGE_INSTRUCTION = "put the cream cheese on the bowl"


def _synthetic_episode(rng, t=3, h=4, w=4):
    agentview_rgb = rng.integers(0, 255, size=(t, h, w, 3), dtype=np.uint8)
    eye_in_hand_rgb = rng.integers(0, 255, size=(t, h, w, 3), dtype=np.uint8)
    # Uniform [0,1) values are valid normalized depth by construction --
    # unlike state/actions (rng.normal), depth must stay in-range.
    agentview_depth = rng.random(size=(t, h, w, 1)).astype(np.float32)
    state = rng.normal(size=(t, 7)).astype(np.float32)
    actions = rng.normal(size=(t, 7)).astype(np.float32)
    return agentview_rgb, eye_in_hand_rgb, agentview_depth, state, actions


def test_validate_episode_arrays_accepts_well_formed_episode():
    rng = np.random.default_rng(0)
    agentview_rgb, eye_in_hand_rgb, agentview_depth, state, actions = _synthetic_episode(
        rng, t=3
    )

    # No exception raised.
    validate_episode_arrays(agentview_rgb, eye_in_hand_rgb, agentview_depth, state, actions)


def test_validate_episode_arrays_rejects_wrong_action_dim():
    rng = np.random.default_rng(1)
    agentview_rgb, eye_in_hand_rgb, agentview_depth, state, actions = _synthetic_episode(
        rng, t=3
    )
    actions = actions[:, :6]  # last-dim 6, not 7

    with pytest.raises(ValueError):
        validate_episode_arrays(agentview_rgb, eye_in_hand_rgb, agentview_depth, state, actions)


def test_validate_episode_arrays_rejects_mismatched_episode_length():
    rng = np.random.default_rng(2)
    agentview_rgb, eye_in_hand_rgb, agentview_depth, state, actions = _synthetic_episode(
        rng, t=3
    )
    actions = actions[:2]  # 2 timesteps vs. state's 3

    with pytest.raises(ValueError):
        validate_episode_arrays(agentview_rgb, eye_in_hand_rgb, agentview_depth, state, actions)


def test_validate_episode_arrays_rejects_non_uint8_image():
    rng = np.random.default_rng(3)
    agentview_rgb, eye_in_hand_rgb, agentview_depth, state, actions = _synthetic_episode(
        rng, t=3
    )
    agentview_rgb = agentview_rgb.astype(np.float32)

    with pytest.raises(ValueError):
        validate_episode_arrays(agentview_rgb, eye_in_hand_rgb, agentview_depth, state, actions)


def test_validate_episode_arrays_rejects_nan_state():
    rng = np.random.default_rng(4)
    agentview_rgb, eye_in_hand_rgb, agentview_depth, state, actions = _synthetic_episode(
        rng, t=3
    )
    state = state.copy()
    state[0, 0] = np.nan

    with pytest.raises(ValueError):
        validate_episode_arrays(agentview_rgb, eye_in_hand_rgb, agentview_depth, state, actions)


def test_validate_episode_arrays_rejects_wrong_dtype_depth():
    rng = np.random.default_rng(7)
    agentview_rgb, eye_in_hand_rgb, agentview_depth, state, actions = _synthetic_episode(
        rng, t=3
    )
    agentview_depth = agentview_depth.astype(np.float64)

    with pytest.raises(ValueError):
        validate_episode_arrays(agentview_rgb, eye_in_hand_rgb, agentview_depth, state, actions)


def test_validate_episode_arrays_rejects_out_of_range_depth():
    rng = np.random.default_rng(8)
    agentview_rgb, eye_in_hand_rgb, agentview_depth, state, actions = _synthetic_episode(
        rng, t=3
    )
    agentview_depth = agentview_depth.copy()
    agentview_depth[0, 0, 0, 0] = 1.5

    with pytest.raises(ValueError):
        validate_episode_arrays(agentview_rgb, eye_in_hand_rgb, agentview_depth, state, actions)


def _write_synthetic_hdf5(path, rng, demo_lengths=(3, 2)):
    import h5py

    with h5py.File(path, "w") as f:
        grp = f.create_group("data")
        grp.attrs["bddl_file_name"] = _REAL_BDDL_PATH
        for i, t in enumerate(demo_lengths, start=1):
            ep = grp.create_group(f"demo_{i}")
            obs = ep.create_group("obs")
            obs.create_dataset(
                "agentview_rgb",
                data=rng.integers(0, 255, size=(t, 4, 4, 3), dtype=np.uint8),
            )
            obs.create_dataset(
                "eye_in_hand_rgb",
                data=rng.integers(0, 255, size=(t, 4, 4, 3), dtype=np.uint8),
            )
            obs.create_dataset(
                "agentview_depth",
                data=rng.random(size=(t, 4, 4, 1)).astype(np.float32),
            )
            obs.create_dataset(
                "joint_states", data=rng.normal(size=(t, 5)).astype(np.float64)
            )
            obs.create_dataset(
                "gripper_states", data=rng.normal(size=(t, 2)).astype(np.float64)
            )
            ep.create_dataset("actions", data=rng.normal(size=(t, 7)).astype(np.float64))


def test_load_episodes_from_hdf5_preserves_episode_boundaries_and_derives_language(tmp_path):
    rng = np.random.default_rng(5)
    hdf5_path = tmp_path / "synth.hdf5"
    _write_synthetic_hdf5(hdf5_path, rng, demo_lengths=(3, 2))

    episodes = load_episodes_from_hdf5([str(hdf5_path)])

    assert len(episodes) == 2
    assert episodes[0]["actions"].shape[0] == 3
    assert episodes[1]["actions"].shape[0] == 2
    for episode in episodes:
        assert episode["language_instruction"] == _REAL_LANGUAGE_INSTRUCTION
        assert episode["state"].shape[-1] == 7
        assert episode["actions"].shape[-1] == 7
        assert episode["agentview_depth"].dtype == np.float32
        assert episode["agentview_depth"].shape[-1] == 1


def test_hdf5_to_rlds_writes_tfds_loadable_dataset(tmp_path):
    tfds = pytest.importorskip("tensorflow_datasets")

    rng = np.random.default_rng(6)
    hdf5_path = tmp_path / "synth.hdf5"
    _write_synthetic_hdf5(hdf5_path, rng, demo_lengths=(3, 2))

    out_dir = str(tmp_path / "rlds_out")
    n = hdf5_to_rlds([str(hdf5_path)], out_dir, dataset_name="test_soarm_spatial")

    assert n == 2
    # Round-trip proof (Pitfall 2): does not raise.
    info = tfds.builder("test_soarm_spatial", data_dir=out_dir).info
    # Schema-level depth check (07-03, DEPTH-03): confirms the persisted
    # dataset_info.json declares agentview_depth as a float32 Tensor, matching
    # this file's own step_features construction.
    assert "agentview_depth" in info.features["steps"]["observation"]
    assert info.features["steps"]["observation"]["agentview_depth"].dtype == np.float32


def test_hdf5_to_rlds_against_real_dataset(tmp_path):
    pytest.importorskip("tensorflow_datasets")
    import h5py  # local import, matches this file's importorskip-adjacent pattern.

    paths = sorted(glob.glob(_REAL_DATASET_GLOB))
    if not paths:
        pytest.skip("No real soarm_spatial *_demo.hdf5 dataset found on disk")

    with h5py.File(paths[0], "r") as f:
        if "agentview_depth" not in f["data"]["demo_1"]["obs"]:
            pytest.skip(
                "real dataset predates Phase 7 depth persistence (07-02) -- "
                "will be superseded by Phase 9's DATA-05 depth-augmented re-collection"
            )

    out_dir = str(tmp_path / "rlds_out_real")
    n = hdf5_to_rlds(paths, out_dir, dataset_name="test_soarm_spatial_real")

    assert n > 0
