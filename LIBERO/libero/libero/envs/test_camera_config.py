"""Real (no-mock) local-sim integration proof for camera config threading.

Classified the same way as datasets/test_hdf5_writer.py: integration, real
local sim, small N. Constructs a real SOARM OffScreenRenderEnv against the
existing (Phase-4-validated) put_the_cream_cheese_in_the_bowl task and
inspects the actual rendered observations -- no mocking of MuJoCo/robosuite.

Two tests:
  * ``test_rgb_cameras_non_degenerate_spatial`` (SPAT-01 re-verification) --
    both agentview and eye_in_hand RGB frames render as non-degenerate
    (128, 128, 3) uint8 arrays with camera_depths=False (the default).
  * ``test_depth_cameras_non_degenerate_spatial`` (SPAT-03) -- both cameras'
    depth buffers render as non-degenerate (128, 128, 1) float arrays with
    all values in [0, 1] when camera_depths=True.

Import-path note (mirrors datasets/test_hdf5_writer.py): pytest's prepend
import mode puts LIBERO/libero on sys.path (nearest test ancestor without
__init__.py), so ``libero.envs.*`` resolves. We additionally insert the repo
root so the ``LIBERO.libero.libero.envs`` ABSOLUTE imports inside
env_wrapper.py resolve when the real env is constructed.
"""

import os
import sys

os.environ.setdefault("MUJOCO_GL", "glfw")

# repo root = four levels up from this file's dir
# (envs -> libero -> libero -> LIBERO -> repo root)
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

from libero.envs import OffScreenRenderEnv

# BDDL_DIR convention (mirrors datasets/collector.py's BDDL_DIR -- do NOT use
# get_libero_path; the local ~/.libero/config.yaml is stale per collector.py's
# own comment).
BDDL_DIR = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "bddl_files",
        "libero_goal",
    )
)
BDDL_PATH = os.path.join(BDDL_DIR, "put_the_cream_cheese_in_the_bowl.bddl")

CAMERA_NAMES = ["agentview", "robot0_eye_in_hand"]
CAMERA_H = 128
CAMERA_W = 128


def test_rgb_cameras_non_degenerate_spatial():
    """SPAT-01 re-verification: both cameras render non-degenerate RGB."""
    env = OffScreenRenderEnv(
        bddl_file_name=BDDL_PATH,
        robots=["Soarm101"],
        camera_names=CAMERA_NAMES,
        camera_heights=CAMERA_H,
        camera_widths=CAMERA_W,
        has_renderer=False,
        has_offscreen_renderer=True,
    )
    try:
        obs = env.reset()
        for cam in CAMERA_NAMES:
            key = f"{cam}_image"
            assert key in obs, f"missing RGB obs key {key}"
            frame = obs[key]
            assert frame.shape == (CAMERA_H, CAMERA_W, 3)
            assert frame.dtype == np.uint8
            assert frame.std() > 0, f"{key} frame is degenerate (zero variance)"
    finally:
        env.close()


def test_depth_cameras_non_degenerate_spatial():
    """SPAT-03: both cameras' depth buffers are non-degenerate, in [0, 1]."""
    env = OffScreenRenderEnv(
        bddl_file_name=BDDL_PATH,
        robots=["Soarm101"],
        camera_names=CAMERA_NAMES,
        camera_heights=CAMERA_H,
        camera_widths=CAMERA_W,
        camera_depths=True,
        has_renderer=False,
        has_offscreen_renderer=True,
    )
    try:
        obs = env.reset()
        for cam in CAMERA_NAMES:
            key = f"{cam}_depth"
            assert key in obs, f"missing depth obs key {key}"
            depth = obs[key]
            assert depth.shape == (CAMERA_H, CAMERA_W, 1)
            assert np.all((depth >= 0.0) & (depth <= 1.0)), (
                f"{key} depth values out of [0,1] range"
            )
            assert depth.std() > 0, f"{key} depth is degenerate (zero variance)"
    finally:
        env.close()
