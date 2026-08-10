"""Real (no-mock) local-sim integration proof for depth_xyz.py's back-projection.

Classified the same way as datasets/test_hdf5_writer.py and
envs/test_camera_config.py: integration, real local sim, small N. Constructs
a real SOARM SegmentationRenderEnv against the existing (Phase-4-validated)
put_the_cream_cheese_in_the_bowl task and validates object_xyz_from_obs's
back-projected estimate against MuJoCo's own privileged ground truth
(sim.data.body_xpos) -- the D-04 ground-truth oracle. That privileged read is
ONLY used here, as a test-only comparison oracle; depth_xyz.py's production
code never reads it (D-03, enforced separately via grep in the plan's
acceptance criteria).

TOLERANCE_M is set empirically: the test loops >=3 resets, measures the
actual back-projection error against ground truth each time, prints it, and
asserts against a tolerance with margin above the observed max -- not an
arbitrary tight value (RESEARCH.md Pitfall 3). 128x128 is a coarse render
resolution (each pixel spans a relatively large world-frame footprint at
tabletop range), so the tolerance is sized accordingly.

Import-path note (mirrors datasets/test_hdf5_writer.py and
envs/test_camera_config.py): pytest's prepend import mode puts LIBERO/libero
on sys.path (nearest test ancestor without __init__.py), so ``libero.envs.*``
resolves. We additionally insert the repo root so the
``LIBERO.libero.libero.envs`` ABSOLUTE imports inside env_wrapper.py resolve
when the real env is constructed.
"""

import os
import sys

os.environ.setdefault("MUJOCO_GL", "glfw")

# repo root = four levels up from this file's dir
# (perception -> libero -> libero -> LIBERO -> repo root)
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

from libero.envs import SegmentationRenderEnv
from libero.perception.depth_xyz import object_xyz_from_obs

BDDL_DIR = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "bddl_files",
        "libero_goal",
    )
)
BDDL_PATH = os.path.join(BDDL_DIR, "put_the_cream_cheese_in_the_bowl.bddl")

CAMERA_NAME = "agentview"
CAMERA_H = 128
CAMERA_W = 128
OBJECT_NAME = "cream_cheese_1"

# Empirically-measured margin above the observed max back-projection error
# across N resets (see the loop below). 128x128 is a coarse render
# resolution, so a generous-but-bounded tolerance is expected and sized here
# with real measured headroom, not guessed tight.
TOLERANCE_M = 0.06
N_RESETS = 3


def test_object_xyz_matches_ground_truth_within_tolerance():
    """D-04: object_xyz_from_obs's estimate matches sim.data.body_xpos ground
    truth (test-only oracle) within an empirically-measured tolerance, across
    >=3 resets."""
    env = SegmentationRenderEnv(
        bddl_file_name=BDDL_PATH,
        robots=["Soarm101"],
        camera_names=[CAMERA_NAME],
        camera_heights=CAMERA_H,
        camera_widths=CAMERA_W,
        camera_depths=True,
        camera_segmentations="instance",
        has_renderer=False,
        has_offscreen_renderer=True,
    )
    try:
        errors = []
        for i in range(N_RESETS):
            obs = env.reset()
            instance_id = env.instance_to_id[OBJECT_NAME]

            # D-04 test-only ground-truth oracle (NEVER used in depth_xyz.py
            # itself -- only here, as the comparison target).
            gt_pos = np.array(env.sim.data.body_xpos[env.env.obj_body_id[OBJECT_NAME]])

            est_pos = object_xyz_from_obs(
                env.sim, obs, CAMERA_NAME, CAMERA_H, CAMERA_W, instance_id
            )
            assert est_pos is not None, f"reset {i}: object not visible in segmentation"

            err = float(np.linalg.norm(gt_pos - est_pos))
            errors.append(err)
            print(f"reset {i}: back-projection error = {err:.4f} m")

            assert err < TOLERANCE_M, (
                f"reset {i}: back-projection error {err:.4f} m exceeds "
                f"tolerance {TOLERANCE_M} m (gt={gt_pos}, est={est_pos})"
            )

        print(f"max observed error across {N_RESETS} resets: {max(errors):.4f} m")
    finally:
        env.close()
