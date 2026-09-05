"""Real (no-mock) local-sim integration proof for hdf5_writer.py.

Classified in 04-RESEARCH.md's Validation Architecture as "integration, real
local sim, small N": these tests exercise the real headless SOARM env and the
real ``set_init_state()`` obs-regeneration path end-to-end, because the whole
point is to prove the writer's plumbing — HDF5 schema correctness, LIBERO obs
key renaming, the off-by-one alignment fix, and genuine per-episode success
gating — not to mock it. A mock would prove none of those.

Two tests:
  * ``test_schema_and_obs_key_naming`` — a successful (manually flagged) episode
    produces a robomimic-schema demo group with LIBERO's renamed obs keys and
    correct SOARM-specific shapes.
  * ``test_unsuccessful_episode_not_written`` — an episode that never reports
    success is excluded from the HDF5 (total == 0), proving the success-gate.

Import-path note (mirrors vla/test_eval_loop.py): pytest's prepend import mode
puts LIBERO/libero on sys.path (nearest test ancestor without __init__.py),
so ``libero.datasets.*`` resolves. We additionally insert the repo root so the
``LIBERO.libero.libero.envs`` ABSOLUTE imports inside env_wrapper.py resolve
when the real env is constructed.
"""

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
# also anchor LIBERO/libero so `libero.*` resolves regardless of invocation dir
_LIBERO_LIBERO = os.path.join(_REPO_ROOT, "LIBERO", "libero")
if _LIBERO_LIBERO not in sys.path:
    sys.path.insert(0, _LIBERO_LIBERO)

import numpy as np
import h5py
import pytest

from libero.datasets.raw_recorder import build_recording_env
from libero.datasets.hdf5_writer import gather_demonstrations_as_hdf5

# TASKS[0] verbatim from explorations/soarm_sanity.py (do NOT invent a name).
BDDL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "bddl_files",
    "libero_spatial",
    "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate.bddl",
)


@pytest.fixture
def paths(tmp_path):
    """pytest tmp_path-backed tmp_dir + hdf5_path to avoid cross-run collisions."""
    tmp_dir = str(tmp_path / "raw")
    os.makedirs(tmp_dir, exist_ok=True)
    hdf5_path = str(tmp_path / "demo.hdf5")
    return tmp_dir, hdf5_path


def test_schema_and_obs_key_naming(paths):
    tmp_dir, hdf5_path = paths

    env = build_recording_env(BDDL_PATH, tmp_dir, has_renderer=False)
    env.reset()
    for _ in range(5):
        env.step(np.zeros(7))
    # Plumbing test: zero actions won't complete the task, so flag success
    # manually to exercise the WRITE path. Test 2 exercises the gate honestly.
    env.successful = True
    env.close()

    n = gather_demonstrations_as_hdf5(tmp_dir, hdf5_path, BDDL_PATH)
    assert n == 1

    with h5py.File(hdf5_path, "r") as f:
        assert "demo_1" in f["data"]
        obs_keys = set(f["data"]["demo_1"]["obs"].keys())
        # LIBERO's renamed keys present; raw robosuite names absent.
        assert obs_keys == {
            "agentview_rgb",
            "eye_in_hand_rgb",
            "agentview_depth",
            "gripper_states",
            "joint_states",
        }
        assert "agentview_image" not in obs_keys
        assert "robot0_gripper_qpos" not in obs_keys
        # agentview-only depth per D-05 — the real eye_in_hand wrist cam has
        # no depth capability, so no eye_in_hand_depth key should ever appear.
        assert "eye_in_hand_depth" not in obs_keys
        assert "robot0_eye_in_hand_depth" not in obs_keys

        assert f["data"]["demo_1"]["obs"]["agentview_rgb"].shape[1:] == (128, 128, 3)
        # SOARM's 84mm 2-DOF parallel gripper (D-07 upgrade), not Panda's 2
        # in name only — happens to also be 2, but for a different mechanism.
        assert f["data"]["demo_1"]["obs"]["gripper_states"].shape[1] == 2

        depth_data = f["data"]["demo_1"]["obs"]["agentview_depth"][()]
        assert depth_data.dtype == np.float32
        assert depth_data.shape[1:] == (128, 128, 1)
        assert np.all((depth_data >= 0.0) & (depth_data <= 1.0))

        # off-by-one fix verified
        assert len(f["data"]["demo_1"]["states"]) == len(
            f["data"]["demo_1"]["actions"]
        )
        assert f["data"].attrs["total"] == 1
        assert isinstance(f["data"].attrs["bddl_file_name"], str)
        assert len(f["data"].attrs["bddl_file_name"]) > 0


def test_unsuccessful_episode_not_written(paths):
    tmp_dir, hdf5_path = paths

    env = build_recording_env(BDDL_PATH, tmp_dir, has_renderer=False)
    env.reset()
    for _ in range(3):
        env.step(np.zeros(7))
    # Leave env.successful at its real default (False) — a couple of zero-action
    # steps never complete the task, so this episode must be excluded.
    env.close()

    n = gather_demonstrations_as_hdf5(tmp_dir, hdf5_path, BDDL_PATH)
    assert n == 0

    with h5py.File(hdf5_path, "r") as f:
        assert f["data"].attrs["total"] == 0
        assert "demo_1" not in f["data"]


def test_schema_matches_across_sources(tmp_path):
    """Prove the writer's obs-key-set/dtype/shape output is a pure function of
    the recording mechanism (DataCollectionWrapper + set_init_state), NOT of
    how the actions were generated — the exact property DATA-04 needs ("land
    in the same HDF5 format" for both the scripted (04-02) and teleop (04-05)
    collection sources).

    Both "sources" here are driven headless (has_renderer=False) — this test
    proves the WRITER's schema is source-agnostic, not that a live on-screen
    teleop session is reproducible inside an automated test (that live-input
    part is proven separately by Task 3's real human session).
    """
    tmp_dir_a = str(tmp_path / "raw_a")
    tmp_dir_b = str(tmp_path / "raw_b")
    os.makedirs(tmp_dir_a, exist_ok=True)
    os.makedirs(tmp_dir_b, exist_ok=True)
    hdf5_path_a = str(tmp_path / "demo_a.hdf5")
    hdf5_path_b = str(tmp_path / "demo_b.hdf5")

    # Source A: stands in for the scripted collector's computed (zero) actions.
    env_a = build_recording_env(BDDL_PATH, tmp_dir_a, has_renderer=False)
    env_a.reset()
    for _ in range(5):
        env_a.step(np.zeros(7))
    env_a.successful = True  # schema-proof, not a task-completion proof.
    env_a.close()

    # Source B: stands in for keyboard-teleop-driven actions — the exact
    # values don't matter, only that they differ from source A's, proving the
    # schema doesn't depend on action content.
    env_b = build_recording_env(BDDL_PATH, tmp_dir_b, has_renderer=False)
    env_b.reset()
    nonzero_action = np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
    for _ in range(5):
        env_b.step(nonzero_action)
    env_b.successful = True
    env_b.close()

    n_a = gather_demonstrations_as_hdf5(tmp_dir_a, hdf5_path_a, BDDL_PATH)
    n_b = gather_demonstrations_as_hdf5(tmp_dir_b, hdf5_path_b, BDDL_PATH)
    assert n_a == 1
    assert n_b == 1

    with h5py.File(hdf5_path_a, "r") as fa, h5py.File(hdf5_path_b, "r") as fb:
        demo_a = fa["data"]["demo_1"]
        demo_b = fb["data"]["demo_1"]

        expected_keys = {
            "agentview_rgb",
            "eye_in_hand_rgb",
            "agentview_depth",
            "gripper_states",
            "joint_states",
        }
        assert set(demo_a["obs"].keys()) == expected_keys
        assert set(demo_b["obs"].keys()) == expected_keys

        for key in expected_keys:
            assert demo_a["obs"][key].dtype == demo_b["obs"][key].dtype
            # Same per-step shape; step COUNT may legitimately differ since
            # the two action sequences aren't identical.
            assert demo_a["obs"][key].shape[1:] == demo_b["obs"][key].shape[1:]
