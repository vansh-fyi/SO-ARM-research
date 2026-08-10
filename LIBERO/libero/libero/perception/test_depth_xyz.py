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

KNOWN LOCAL PLATFORM LIMITATION (macOS / Apple Silicon, documented during
05-02 test development, not a bug in this project's code):
MuJoCo's native instance-ID-color segmentation rendering (the
`mjRND_SEGMENT`/`mjRND_IDCOLOR` scene flags, exercised via
`camera_segmentations="instance"`) produces byte-IDENTICAL output to a
normal (non-segmentation) render on this machine -- confirmed with the
lowest-level MuJoCo Python API directly (`mjr_render`/`mjr_readPixels`),
completely independent of robosuite/LIBERO, and reproduced with BOTH the
GLFW and CGL macOS GL backends. This means `obs[f"{cam}_segmentation_instance"]`
is unusable (all-background) on this local dev machine regardless of code
correctness -- it is a MuJoCo/macOS OpenGL-driver-level limitation, not
something this task's code can fix (no source files were modified to work
around it, per this plan's "no changes to existing files" constraint).
`env.instance_to_id` itself is unaffected (computed purely from
`env.model.instances_to_ids`, a model-level property, not a render read), so
it is still exercised for real here.

To still provide a genuine, real-sim (not fully mocked) D-04 validation of
the actual back-projection pipeline given this constraint, each reset here
constructs a small, ground-truth-anchored synthetic segmentation patch (the
one piece of the obs dict MuJoCo cannot currently render correctly on this
machine) by forward-projecting the REAL sim.data.body_xpos ground truth
through the SAME robosuite.utils.camera_utils calibration depth_xyz.py uses,
then marks a small neighborhood of pixels around that location with the
object's real `instance_to_id` value -- everything else (the env, the
physics, the depth buffer, the camera calibration, and the back-projection
math under test) is exercised for real, with zero mocking of MuJoCo/
robosuite/depth_xyz.py itself. This should be re-verified with real
segmentation rendering on Colab (Linux osmesa/egl), where this GL-driver
limitation is not expected to reproduce.

TOLERANCE_M is set empirically: the test loops >=3 resets, measures the
actual back-projection error against ground truth each time, prints it, and
asserts against a tolerance with margin above the observed max -- not an
arbitrary tight value (RESEARCH.md Pitfall 3).

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
from libero.perception.depth_xyz import object_pixel_centroid, object_xyz_from_obs
from robosuite.utils import camera_utils as cu

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
# across N resets (see the loop below): observed max ~0.022m (a small
# object-center-vs-visible-surface offset, expected for a ~2x4x8cm object),
# very consistent across resets once the two real bugs found during 05-02
# test development were fixed in depth_xyz.py (see its module docstring/
# comments): (1) pixel_to_world_xyz was passing the bare camera extrinsic
# instead of the inverse full pixel<->world transform, and (2) robosuite's
# default macros.IMAGE_CONVENTION="opengl" stores camera obs arrays
# row-0-at-bottom, not the row-0-at-top order camera_utils's pixel math
# assumes.
TOLERANCE_M = 0.05
N_RESETS = 3


def _ground_truth_anchored_segmentation_patch(sim, gt_pos, instance_id, patch_radius=2):
    """Build a small (H,W,1) synthetic instance-segmentation array centered
    on the REAL projection of `gt_pos` through the SAME camera_utils
    calibration depth_xyz.py uses, in the SAME raw (row-0-at-bottom, macros.
    IMAGE_CONVENTION="opengl") pixel convention `obs[f"{cam}_segmentation_*"]`
    naturally uses.

    Test-only workaround for the local MuJoCo/macOS segmentation-rendering
    platform limitation documented in this file's module docstring -- NEVER
    used inside depth_xyz.py itself.
    """
    world_to_cam = cu.get_camera_transform_matrix(
        sim=sim, camera_name=CAMERA_NAME, camera_height=CAMERA_H, camera_width=CAMERA_W
    )
    pixel_top_origin = cu.project_points_from_world_to_camera(gt_pos, world_to_cam, CAMERA_H, CAMERA_W)
    # convert from camera_utils's top-origin convention to the raw
    # (bottom-origin) convention real obs arrays use (see module docstring).
    raw_row = (CAMERA_H - 1) - int(pixel_top_origin[0])
    raw_col = int(pixel_top_origin[1])

    seg = np.zeros((CAMERA_H, CAMERA_W, 1), dtype=np.int32)
    r0, r1 = max(0, raw_row - patch_radius), min(CAMERA_H, raw_row + patch_radius + 1)
    c0, c1 = max(0, raw_col - patch_radius), min(CAMERA_W, raw_col + patch_radius + 1)
    seg[r0:r1, c0:c1, 0] = instance_id
    return seg


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

            # Test-only substitution for the locally-broken native
            # segmentation render (see module docstring). Real depth
            # (`obs[f"{cam}_depth"]`) and everything downstream (camera
            # calibration, back-projection math) is exercised for real.
            obs = dict(obs)
            obs[f"{CAMERA_NAME}_segmentation_instance"] = _ground_truth_anchored_segmentation_patch(
                env.sim, gt_pos, instance_id
            )

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


def test_object_pixel_centroid_pure_logic():
    """Unit-level proof of object_pixel_centroid's own logic, independent of
    sim/rendering: exact centroid over a known synthetic mask, and the
    None-return path when the instance has zero matching pixels."""
    seg = np.zeros((10, 10, 1), dtype=np.int32)
    seg[2:5, 3:6, 0] = 7  # rows 2..4, cols 3..5 -> centroid (3.0, 4.0)

    centroid = object_pixel_centroid(seg, instance_id=7)
    assert centroid == (3.0, 4.0)

    assert object_pixel_centroid(seg, instance_id=99) is None
