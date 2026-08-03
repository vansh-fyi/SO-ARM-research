"""Tests for normalization.py — DATA-03 SOARM normalization statistics (Phase 4).

Task 1 tests are pure unit tests (synthetic numpy arrays, no sim/HDF5
dependency — matches 04-RESEARCH.md's "unit, synthetic array input"
classification for DATA-03), exercising ``compute_norm_stats``'s 4 specified
behaviors. Task 2 adds a real-dataset-or-skip integration test for
``write_dataset_statistics`` against 04-02's actual collected HDF5 output.

Import-path note (mirrors test_replay.py / test_collector.py): insert the
repo root so ``LIBERO.libero.libero.envs`` absolute imports resolve if ever
needed, and ``LIBERO/libero`` so ``libero.datasets.*`` resolves regardless of
invocation dir. normalization.py itself has no sim dependency, but this
import-path setup is kept consistent with the package's sibling test files.
"""

import glob
import json
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

from libero.datasets.normalization import (
    compute_norm_stats,
    load_actions_and_proprio_from_hdf5,
    write_dataset_statistics,
)

# 04-02's real collected dataset. Single-file layout post D-07/D-08
# retargeting (see 04-02-SUMMARY.md / 04-03-SUMMARY.md) — the glob does not
# assert a fixed file count, it reflects whatever *_demo.hdf5 files actually
# exist on disk.
_REAL_DATASET_GLOB = os.path.join(
    os.path.dirname(__file__), "..", "..", "datasets", "soarm_spatial", "*_demo.hdf5"
)

_STAT_KEYS = {"mean", "std", "max", "min", "q01", "q99"}


def _synthetic_arrays(rng, t=500, action_dim=7, proprio_dim=7):
    """Realistic-shaped synthetic arrays with distinct per-dim means/scales.

    proprio_dim=7: 5 SOARM arm joints + 2 gripper DOF (the real roboninecom
    84mm parallel gripper has two independently sliding jaws — a 2-DOF
    robot0_gripper_qpos, NOT Panda's single combined DOF), matching
    joint_states + gripper_states concatenation this dataset actually uses.
    """
    means = rng.uniform(-5, 5, size=proprio_dim)
    scales = rng.uniform(0.5, 3.0, size=proprio_dim)
    proprio = rng.normal(loc=means, scale=scales, size=(t, proprio_dim))

    a_means = rng.uniform(-1, 1, size=action_dim)
    a_scales = rng.uniform(0.1, 1.0, size=action_dim)
    actions = rng.normal(loc=a_means, scale=a_scales, size=(t, action_dim))
    return actions, proprio


def test_compute_norm_stats_schema_and_shapes():
    rng = np.random.default_rng(0)
    actions, proprio = _synthetic_arrays(rng)

    result = compute_norm_stats(actions, proprio, num_trajectories=17)

    assert set(result.keys()) == {"soarm_spatial"}
    entry = result["soarm_spatial"]
    assert set(entry["action"].keys()) == _STAT_KEYS
    assert set(entry["proprio"].keys()) == _STAT_KEYS
    assert entry["num_transitions"] == 500
    assert entry["num_trajectories"] == 17


def test_compute_norm_stats_q01_strictly_less_than_q99():
    rng = np.random.default_rng(1)
    actions, proprio = _synthetic_arrays(rng)

    result = compute_norm_stats(actions, proprio)["soarm_spatial"]

    for sub in ("action", "proprio"):
        q01 = np.asarray(result[sub]["q01"])
        q99 = np.asarray(result[sub]["q99"])
        assert np.all(q01 < q99), f"{sub}: q01 not strictly < q99"


def test_compute_norm_stats_no_nan():
    rng = np.random.default_rng(2)
    actions, proprio = _synthetic_arrays(rng)

    result = compute_norm_stats(actions, proprio)["soarm_spatial"]

    for sub in ("action", "proprio"):
        for key in _STAT_KEYS:
            values = np.asarray(result[sub][key])
            assert not np.any(np.isnan(values)), f"{sub}.{key} contains NaN"


def test_compute_norm_stats_constant_dimension_degenerate_not_nan():
    rng = np.random.default_rng(3)
    actions, proprio = _synthetic_arrays(rng)
    # Force one proprio dimension to be constant (std=0).
    proprio = proprio.copy()
    proprio[:, 0] = 3.14

    result = compute_norm_stats(actions, proprio)["soarm_spatial"]

    assert not np.any(np.isnan(np.asarray(result["proprio"]["q01"])))
    assert result["proprio"]["q01"][0] == pytest.approx(3.14)
    assert result["proprio"]["q99"][0] == pytest.approx(3.14)
    assert result["proprio"]["mean"][0] == pytest.approx(3.14)


def test_load_and_write_against_04_02_output(tmp_path):
    paths = sorted(glob.glob(_REAL_DATASET_GLOB))
    if not paths:
        pytest.skip("No real soarm_spatial *_demo.hdf5 dataset found on disk")

    out_path = str(tmp_path / "dataset_statistics.json")
    result = write_dataset_statistics(paths, out_path)

    assert os.path.isfile(out_path)
    with open(out_path) as f:
        loaded = json.load(f)
    assert loaded == result

    assert "soarm_spatial" in result
    entry = result["soarm_spatial"]
    assert len(entry["action"]["q01"]) == 7
    assert len(entry["proprio"]["q01"]) == 7  # 5 joints + 2-DOF gripper

    for sub in ("action", "proprio"):
        q01 = np.asarray(entry[sub]["q01"])
        q99 = np.asarray(entry[sub]["q99"])
        assert np.all(q01 <= q99), f"{sub}: q01 > q99 in real dataset stats"
