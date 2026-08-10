try:
    import numpy as _np

    # numpy2/robosuite 1.4.1 compat shim: robosuite's instance-segmentation
    # render path (utils/binding_utils.py's MjRenderContextOffscreen.
    # read_pixels) combines a uint8 RGB buffer's channels via
    # `rgb_img[..., 1] * (2**8)`, relying on NumPy's legacy value-based
    # scalar promotion (uint8 array * python int -> widened dtype). Under
    # NumPy>=2.0's NEP 50 "weak" promotion, that multiplication instead
    # raises `OverflowError: Python integer 256 out of bounds for uint8`,
    # because the result dtype stays uint8. This project's requirements.txt
    # pins numpy==1.22.4 (pre-NEP-50); when a newer numpy is present in the
    # active environment (observed: numpy 2.0.2), restore legacy promotion
    # for this process only -- this does not touch the installed numpy
    # package or any vendored robosuite source, and is required for ANY
    # caller (test or production) that constructs a SegmentationRenderEnv
    # (camera_segmentations="instance") for use with this module's
    # object_xyz_from_obs. Only applied when the private API exists and the
    # legacy state isn't already active; a missing/renamed private API on a
    # future numpy is a silent no-op, not a hard failure.
    if hasattr(_np, "_set_promotion_state"):
        _np._set_promotion_state("legacy")
except Exception:
    pass

try:
    # depth_xyz.py transitively depends on robosuite/MuJoCo (via
    # robosuite.utils.camera_utils), same class of sim-dependent submodule as
    # vla/oft_backend.py and vla/pi0_backend.py. Degrade gracefully so any
    # pytest suite or notebook that imports libero.perception without a full
    # robosuite/MuJoCo stack installed doesn't hard-fail on import. Also
    # degrades gracefully when a present dependency's transitive import
    # chain raises a non-ImportError exception (e.g. a numpy binary-ABI
    # ValueError), matching vla/__init__.py's documented rationale.
    from .depth_xyz import object_pixel_centroid, object_xyz_from_obs, pixel_to_world_xyz
except Exception:
    pixel_to_world_xyz = None
    object_pixel_centroid = None
    object_xyz_from_obs = None
