"""Tests for replay.py — DATA-02 state-based determinism verification (Phase 4).

Two tiers, mirroring replay.py's own split:
  * ``test_verify_states_only_*`` — cheap states-only tier (Task 1).
  * ``test_verify_full_obs_regeneration_*`` — sampled full-obs-regeneration
    tier (Task 2).

Each tier has a real-dataset-or-skip test (against 04-02's actual on-disk
collection output) and a synthetic-corruption test (proves the fail-loudly
behavior, not just the happy path).

NOTE on dataset shape: 04-02's collection was retargeted (D-07/D-08) from 3
frozen bowl->plate tasks down to a SINGLE task
(``put_the_cream_cheese_in_the_bowl``), so the real output is ONE HDF5 file
containing 120 demo groups, not 3 per-task files. The glob below intentionally
does not assert a fixed file count — it reflects whatever ``*_demo.hdf5``
files actually exist on disk (see 04-02-SUMMARY.md's "RESOLVED" section).

Import-path note (mirrors test_hdf5_writer.py / test_collector.py): insert the
repo root so ``LIBERO.libero.libero.envs`` absolute imports resolve, and
``LIBERO/libero`` so ``libero.datasets.*`` resolves regardless of invocation dir.
"""

import glob
import os
import sys

os.environ.setdefault("MUJOCO_GL", "glfw")

# repo root = four levels up from this file's dir
# (datasets -> libero -> libero -> LIBERO -> repo root)
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_LIBERO_LIBERO = os.path.join(_REPO_ROOT, "LIBERO", "libero")
if _LIBERO_LIBERO not in sys.path:
    sys.path.insert(0, _LIBERO_LIBERO)

import h5py
import numpy as np
import pytest

from libero.datasets.raw_recorder import build_recording_env
from libero.datasets.hdf5_writer import gather_demonstrations_as_hdf5
from libero.datasets.replay import verify_states_only, verify_full_obs_regeneration

# Same task/BDDL used by test_hdf5_writer.py's synthetic-HDF5 plumbing tests —
# any schema-valid HDF5 works for these corruption-detection tests,
# independent of collector.py's real (cream_cheese) task set.
_BDDL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "bddl_files",
    "libero_spatial",
    "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate.bddl",
)

# 04-02's real collected dataset. Single-file layout post-retargeting (see
# module docstring) — glob for whatever *_demo.hdf5 files exist, do NOT
# hardcode an expected count.
_REAL_DATASET_GLOB = os.path.join(
    _REPO_ROOT, "LIBERO", "libero", "datasets", "soarm_spatial", "*_demo.hdf5"
)


def _make_synthetic_hdf5(tmp_path, n_steps=4):
    """Build a tiny, schema-valid synthetic HDF5 (1 demo) via the real
    recording + writer path — same pattern as test_hdf5_writer.py's plumbing
    test. Zero actions won't complete the task, so success is flagged
    manually to exercise the write path (this is a structural, not
    task-success, test fixture)."""
    tmp_dir = str(tmp_path / "raw")
    hdf5_path = str(tmp_path / "synthetic_demo.hdf5")

    env = build_recording_env(_BDDL_PATH, tmp_dir, has_renderer=False)
    env.reset()
    for _ in range(n_steps):
        env.step(np.zeros(7))
    env.successful = True
    env.close()

    n = gather_demonstrations_as_hdf5(tmp_dir, hdf5_path, _BDDL_PATH)
    assert n == 1
    return hdf5_path


# ----------------------------------------------------------------------------
# Tier 1: verify_states_only
# ----------------------------------------------------------------------------


def test_verify_states_only_passes_on_04_02_output():
    paths = sorted(glob.glob(_REAL_DATASET_GLOB))
    if not paths:
        pytest.skip("04-02's real collected dataset not present on disk")
    for p in paths:
        result = verify_states_only(p)
        assert result["passed"] == result["total_states"]
        assert result["total_states"] > 0
        assert result["total_demos"] > 0


def test_verify_states_only_raises_on_corrupt_state(tmp_path):
    hdf5_path = _make_synthetic_hdf5(tmp_path)

    # Directly corrupt one recorded state value via h5py.
    with h5py.File(hdf5_path, "r+") as f:
        f["data"]["demo_1"]["states"][0, 0] = np.nan

    with pytest.raises(AssertionError):
        verify_states_only(hdf5_path)


# ----------------------------------------------------------------------------
# Tier 2: verify_full_obs_regeneration
# ----------------------------------------------------------------------------


def test_verify_full_obs_regeneration_passes_on_04_02_output():
    paths = sorted(glob.glob(_REAL_DATASET_GLOB))
    if not paths:
        pytest.skip("04-02's real collected dataset not present on disk")
    for p in paths:
        result = verify_full_obs_regeneration(p)
        assert result["passed"] == result["sampled_states"]
        # >=5 overall, and >=1/demo guaranteed structurally by
        # verify_full_obs_regeneration's always-include-first-state rule.
        assert result["sampled_states"] >= 5


def test_verify_full_obs_regeneration_raises_on_corrupted_obs(tmp_path):
    hdf5_path = _make_synthetic_hdf5(tmp_path)

    # Directly mutate one pixel in the recorded agentview_rgb obs via h5py.
    # (255 - v) != v for any integer v, so this always produces a divergence.
    with h5py.File(hdf5_path, "r+") as f:
        rgb = f["data"]["demo_1"]["obs"]["agentview_rgb"]
        rgb[0, 0, 0, 0] = 255 - int(rgb[0, 0, 0, 0])

    with pytest.raises(AssertionError):
        verify_full_obs_regeneration(hdf5_path)
