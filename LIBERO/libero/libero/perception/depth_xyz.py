"""Real-hardware-faithful depth-to-XYZ back-projection pipeline (SPAT-04).

Back-projects a MuJoCo depth buffer + a pixel location into a world-frame
object XYZ estimate using robosuite.utils.camera_utils's camera
intrinsics/extrinsics -- NOT a privileged read of MuJoCo's internal body
pose state (D-03). This mirrors what a real depth camera on the physical
SOARM arm would produce: a per-pixel depth value plus known camera
calibration, nothing more. The only sanctioned ground-truth (MuJoCo's
privileged body pose state) read for validating this pipeline lives in
test_depth_xyz.py, as a test-only comparison oracle -- never here.
"""

import numpy as np
from robosuite.utils import camera_utils as cu


def pixel_to_world_xyz(sim, camera_name, camera_height, camera_width, depth_map, pixel_yx):
    """Back-project a single (row, col) pixel + a (H,W,1) normalized depth
    map to world-frame XYZ.

    Args:
        sim (MjSim): simulator instance (used only for camera
            calibration -- intrinsics/extrinsics -- never for object state)
        camera_name (str): name of the camera the pixel/depth map came from
        camera_height (int): height of the camera image in pixels
        camera_width (int): width of the camera image in pixels
        depth_map (np.array): (H, W, 1) depth map normalized in [0, 1]
            (the raw MuJoCo/robosuite depth obs)
        pixel_yx (tuple[float, float]): (row, col) pixel location to
            back-project

    Returns:
        np.ndarray: shape (3,), world-frame XYZ estimate
    """
    # NOTE: robosuite's default macros.IMAGE_CONVENTION == "opengl" means
    # camera obs arrays (both `{cam}_depth` and `{cam}_segmentation_*`, see
    # robot_env.py's `depth[::convention]`/`seg[::convention, :, 1]` with
    # convention=1 for "opengl") are stored row-0-at-bottom (raw OpenGL
    # readback order), NOT the row-0-at-top order robosuite.utils.camera_
    # utils's pixel math assumes (get_camera_segmentation's OWN sibling
    # helper explicitly does `sim.render(...)[::-1]` to convert to that
    # top-origin order before use). `pixel_yx` is expected in the SAME raw
    # (bottom-origin) convention as the incoming `depth_map` -- i.e. exactly
    # what `object_pixel_centroid` yields when applied directly to a raw
    # `obs[f"{cam}_segmentation_instance"]` array -- so both are converted
    # to top-origin together here. Confirmed empirically: skipping this
    # flip produces a large, consistent (~7-9cm on X/Z) systematic bias in
    # every reset during 05-02 test development (see test_depth_xyz.py).
    depth_map_top_origin = depth_map[::-1]
    row, col = pixel_yx
    row_top_origin = (camera_height - 1) - row
    pixel_yx = (row_top_origin, col)

    real_depth = cu.get_real_depth_map(sim=sim, depth_map=depth_map_top_origin)
    # NOTE: transform_from_pixels_to_world's `camera_to_world_transform` arg is
    # NOT the plain camera pose (get_camera_extrinsic_matrix) -- its internal
    # math (cam_pts = [pixel_col * z, pixel_row * z, z, 1], i.e. un-normalized
    # pixel coords scaled by real-world depth, mirroring project_points_from_
    # world_to_camera's pre-perspective-divide homogeneous vector) expects the
    # INVERSE of the full world->pixel transform (intrinsics + extrinsics
    # combined, i.e. get_camera_transform_matrix's K_exp @ pose_inv(R)), not
    # the bare extrinsic. Passing the bare extrinsic silently produces
    # wildly-wrong (tens-of-meters) results -- confirmed empirically via a
    # forward-project/back-project round-trip against MuJoCo's privileged
    # body pose ground truth during 05-02 test development (see
    # test_depth_xyz.py).
    world_to_pixel = cu.get_camera_transform_matrix(
        sim=sim, camera_name=camera_name, camera_height=camera_height, camera_width=camera_width
    )
    pixel_to_world = np.linalg.inv(world_to_pixel)
    pixels = np.array([pixel_yx], dtype=float)  # shape (1, 2)
    xyz_world = cu.transform_from_pixels_to_world(
        pixels=pixels,
        depth_map=real_depth[None, ...],  # shape (1, H, W, 1)
        camera_to_world_transform=pixel_to_world,
    )
    return xyz_world[0]


def object_pixel_centroid(segmentation_image, instance_id):
    """Mean (row, col) of all pixels in a (H,W,1) segmentation array matching
    instance_id.

    Args:
        segmentation_image (np.array): (H, W, 1) instance-segmentation obs
        instance_id (int): instance ID to locate (matches
            SegmentationRenderEnv.instance_to_id[object_name])

    Returns:
        tuple[float, float] | None: (row, col) centroid, or None if the
            instance has zero matching pixels (not visible/occluded)
    """
    rows, cols = np.nonzero(segmentation_image[..., 0] == instance_id)
    if rows.size == 0:
        return None
    return (float(rows.mean()), float(cols.mean()))


def object_xyz_from_obs(sim, obs, camera_name, camera_height, camera_width, instance_id):
    """Compose object_pixel_centroid + pixel_to_world_xyz to estimate an
    object's world-frame XYZ purely from camera-derived observations.

    Reads ONLY obs[f"{camera_name}_segmentation_instance"] and
    obs[f"{camera_name}_depth"], plus sim (for camera calibration via
    camera_utils). NEVER reads MuJoCo's privileged body pose state (D-03) --
    that ground-truth read belongs only in test_depth_xyz.py, as the D-04
    comparison oracle.

    Args:
        sim (MjSim): simulator instance (camera calibration only)
        obs (dict): observation dict from a SegmentationRenderEnv step/reset
        camera_name (str): name of the camera to read from
        camera_height (int): height of the camera image in pixels
        camera_width (int): width of the camera image in pixels
        instance_id (int): instance ID of the object of interest

    Returns:
        np.ndarray | None: shape (3,) world-frame XYZ estimate, or None if
            the object isn't visible in the segmentation image
    """
    segmentation_image = obs[f"{camera_name}_segmentation_instance"]
    depth_map = obs[f"{camera_name}_depth"]

    centroid = object_pixel_centroid(segmentation_image, instance_id)
    if centroid is None:
        return None

    return pixel_to_world_xyz(sim, camera_name, camera_height, camera_width, depth_map, centroid)
