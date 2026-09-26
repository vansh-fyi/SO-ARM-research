"""Checkerboard-based stereo calibration for the AR0144 stereo camera (Phase
12, Plan 12-05 -- step 1 of the 3-step depth-calibration extension, D-07).

Produces a saved calibration file (`control/stereo_calibration.json` by
default) that Plan 12-06's `depth_camera.py` loads to both build rectification
maps AND derive the flattened-3x3-intrinsics-plus-baseline-in-meters format
NVIDIA's Fast-FoundationStereo (FastFS) requires as input, alongside a
rectified, undistorted stereo pair.

This module reuses the existing, unmodified `StereoSplitCamera`
(`control/vla_bridge/stereo_camera.py`) for live capture -- it NEVER opens a
second, independent capture of the AR0144, which `stereo_camera.py`'s own
docstring documents as unsupported on this rig. `run_calibration_session()`
accepts an already-constructed `StereoSplitCamera`-shaped object and calls
only its `read_left()`/`read_right()` methods, exactly like every other
consumer of that class.

Standalone: zero dependency on `run_vla_episode.py`, `robot_client.py`, or the
bridge/PolicyServer path. Can be run and verified entirely on its own, ahead
of Plan 12-06.
"""

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

# Checkerboard pattern defaults. MUST match whatever physical checkerboard
# the human actually uses -- overridable via the CLI (see main()).
DEFAULT_PATTERN_SIZE = (9, 6)  # internal corners: (columns, rows)
DEFAULT_SQUARE_SIZE_M = 0.025  # 25mm squares

# Mirrors detect_devices.py's DEVICE_MAP_PATH module-level constant
# convention -- lives at control/stereo_calibration.json, a sibling of
# control/device_map.json, NOT nested under vla_bridge/.
DEFAULT_CALIBRATION_PATH = Path(__file__).resolve().parent.parent / "stereo_calibration.json"


def build_object_points(
    pattern_size: tuple[int, int] = DEFAULT_PATTERN_SIZE,
    square_size_m: float = DEFAULT_SQUARE_SIZE_M,
) -> np.ndarray:
    """Real-world (x, y, z=0) coordinates of a planar checkerboard's internal
    corners, in meters, row-major by (row, col). Row `r*cols+c` is
    `[c*square_size_m, r*square_size_m, 0.0]`."""
    cols, rows = pattern_size
    object_points = np.zeros((cols * rows, 3), dtype=np.float32)
    object_points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size_m
    return object_points


def detect_checkerboard_corners(
    image: np.ndarray, pattern_size: tuple[int, int] = DEFAULT_PATTERN_SIZE
) -> np.ndarray | None:
    """Finds and subpixel-refines checkerboard corners in `image`. Returns
    `None` (never raises) if the pattern is not found."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    found, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    if not found:
        return None
    # Normalize to the (N, 1, 2) float32 shape cv2.cornerSubPix()/
    # calibrateCamera()/stereoCalibrate() all expect -- some cv2 builds
    # return findChessboardCorners()'s output as (N, 2) instead.
    corners = np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return np.asarray(refined, dtype=np.float32).reshape(-1, 1, 2)


@dataclass
class StereoCalibration:
    """Saved stereo calibration geometry. `T` (inter-camera translation) is
    in METERS -- real-world units, because build_object_points()'s square
    size is specified in meters."""

    image_size: tuple[int, int]  # (width, height)
    left_K: list[float]  # flattened row-major 3x3
    left_D: list[float]  # 5 distortion coeffs
    right_K: list[float]  # flattened row-major 3x3
    right_D: list[float]  # 5 distortion coeffs
    R: list[float]  # flattened row-major 3x3, inter-camera rotation
    T: list[float]  # 3-element, inter-camera translation, meters
    reprojection_error_px: float
    calibrated_at_utc: str


def calibrate_single_camera(
    object_points_list: list[np.ndarray],
    image_points_list: list[np.ndarray],
    image_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, float]:
    """Per-camera intrinsic calibration. Returns `(K, D, rms)`, discarding
    the rvecs/tvecs cv2.calibrateCamera() also returns -- no caller in this
    module needs per-view extrinsics."""
    rms, K, D, _rvecs, _tvecs = cv2.calibrateCamera(
        object_points_list, image_points_list, image_size, None, None
    )
    return K, D, rms


def calibrate_stereo_pair(
    object_points_list: list[np.ndarray],
    left_points_list: list[np.ndarray],
    right_points_list: list[np.ndarray],
    image_size: tuple[int, int],
    left_K: np.ndarray,
    left_D: np.ndarray,
    right_K: np.ndarray,
    right_D: np.ndarray,
) -> StereoCalibration:
    """Solves for the inter-camera rotation/translation with each camera's
    own intrinsics held FIXED (`cv2.CALIB_FIX_INTRINSIC`). This two-step
    approach -- per-camera `calibrate_single_camera()` first, then stereo
    calibration solving ONLY for R/T -- is the standard, better-conditioned
    OpenCV stereo calibration recipe versus solving all parameters jointly
    from scratch."""
    rms, K1, D1, K2, D2, R, T, _E, _F = cv2.stereoCalibrate(
        object_points_list,
        left_points_list,
        right_points_list,
        left_K,
        left_D,
        right_K,
        right_D,
        image_size,
        flags=cv2.CALIB_FIX_INTRINSIC,
    )
    return StereoCalibration(
        image_size=tuple(image_size),
        left_K=np.asarray(K1).flatten().tolist(),
        left_D=np.asarray(D1).flatten().tolist(),
        right_K=np.asarray(K2).flatten().tolist(),
        right_D=np.asarray(D2).flatten().tolist(),
        R=np.asarray(R).flatten().tolist(),
        T=np.asarray(T).flatten().tolist(),
        reprojection_error_px=float(rms),
        calibrated_at_utc=datetime.now(timezone.utc).isoformat(),
    )


def save_calibration(calibration: StereoCalibration, path: Path) -> None:
    """Writes `calibration` to `path` as indented JSON."""
    path = Path(path)
    path.write_text(json.dumps(asdict(calibration), indent=2))


def load_calibration(path: Path) -> StereoCalibration:
    """Reads a `StereoCalibration` back from `path`. Coerces `image_size`
    (a JSON list) back to a tuple, since JSON has no tuple type."""
    data = json.loads(Path(path).read_text())
    data["image_size"] = tuple(data["image_size"])
    return StereoCalibration(**data)


def run_calibration_session(
    stereo_camera,
    num_views: int = 15,
    pattern_size: tuple[int, int] = DEFAULT_PATTERN_SIZE,
    square_size_m: float = DEFAULT_SQUARE_SIZE_M,
    prompt_fn=input,
) -> StereoCalibration:
    """Live-capture orchestration loop. For up to `num_views` iterations,
    prompts the human to reposition the checkerboard, reads one stereo pair
    from `stereo_camera` (an already-constructed `StereoSplitCamera`-shaped
    object -- never opened by this function), and detects checkerboard
    corners in BOTH halves. A view where either half fails detection is
    skipped (does not count toward `num_views`). Raises `RuntimeError` if
    fewer than 3 views yield valid corners in both halves."""
    object_points_list: list[np.ndarray] = []
    left_points_list: list[np.ndarray] = []
    right_points_list: list[np.ndarray] = []
    first_left_frame: np.ndarray | None = None

    for view_index in range(num_views):
        if view_index > 0:
            prompt_fn(
                f"Reposition the checkerboard for view {view_index + 1}/{num_views}, "
                "then press Enter to capture..."
            )

        left_frame = stereo_camera.read_left()
        right_frame = stereo_camera.read_right()

        left_corners = detect_checkerboard_corners(left_frame, pattern_size) if left_frame is not None else None
        right_corners = (
            detect_checkerboard_corners(right_frame, pattern_size) if right_frame is not None else None
        )

        if left_corners is None or right_corners is None:
            print(f"WARNING: view {view_index + 1}/{num_views}: checkerboard not detected in both halves, skipping")
            continue

        if first_left_frame is None:
            first_left_frame = left_frame

        object_points_list.append(build_object_points(pattern_size, square_size_m))
        left_points_list.append(left_corners)
        right_points_list.append(right_corners)

    if len(object_points_list) < 3:
        raise RuntimeError(
            f"Only {len(object_points_list)} valid calibration view(s) captured (need "
            f"at least 3) -- reposition the checkerboard and try again"
        )

    # cv2 image arrays are (height, width, channels); calibrateCamera()/
    # stereoCalibrate() want image_size as (width, height) -- the reverse
    # order. Getting this backwards is a classic off-by-transpose bug.
    height, width = first_left_frame.shape[:2]
    image_size = (width, height)

    left_K, left_D, _left_rms = calibrate_single_camera(object_points_list, left_points_list, image_size)
    right_K, right_D, _right_rms = calibrate_single_camera(object_points_list, right_points_list, image_size)

    return calibrate_stereo_pair(
        object_points_list,
        left_points_list,
        right_points_list,
        image_size,
        left_K,
        left_D,
        right_K,
        right_D,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Checkerboard-based stereo calibration for the AR0144 stereo camera."
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_CALIBRATION_PATH, help="Output calibration JSON path")
    parser.add_argument(
        "--stereo-camera-index",
        type=str,
        default="1",
        help="AVFoundation device NAME or numeric index for the AR0144 (prefer NAME -- numeric "
        "index has been confirmed to drift between process launches on macOS).",
    )
    parser.add_argument("--num-views", type=int, default=15, help="Number of calibration views to capture")
    parser.add_argument("--pattern-cols", type=int, default=DEFAULT_PATTERN_SIZE[0], help="Checkerboard internal corner columns")
    parser.add_argument("--pattern-rows", type=int, default=DEFAULT_PATTERN_SIZE[1], help="Checkerboard internal corner rows")
    parser.add_argument("--square-size-m", type=float, default=DEFAULT_SQUARE_SIZE_M, help="Checkerboard square size in meters")
    args = parser.parse_args()

    from vla_bridge.stereo_camera import StereoSplitCamera

    stereo_camera = StereoSplitCamera(index=args.stereo_camera_index)
    calibration = run_calibration_session(
        stereo_camera,
        num_views=args.num_views,
        pattern_size=(args.pattern_cols, args.pattern_rows),
        square_size_m=args.square_size_m,
    )
    save_calibration(calibration, args.out)
    print(f"Reprojection error: {calibration.reprojection_error_px:.3f} px")
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
