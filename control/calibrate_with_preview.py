"""Interactive stereo calibration WITH live preview (Phase 12, calibration UX aid).

Combines `preview_checkerboard.py`'s live corner-detection view with
`vla_bridge/stereo_calibration.py`'s real capture-and-calibrate session, so
you see exactly what the camera sees at the moment you capture each view --
`vla_bridge/stereo_calibration.py`'s own `run_calibration_session()` is
headless (an `input()` prompt with zero visual feedback).

Reuses the SAME calibration math from `vla_bridge/stereo_calibration.py`
(`build_object_points`, `calibrate_single_camera`, `calibrate_stereo_pair`,
`save_calibration`) -- this script only replaces the blind capture loop with
a live cv2 window: SPACE to capture the currently-displayed frame as a view,
q/ESC to stop early (e.g. once you already have enough valid views).

Usage:
    python calibrate_with_preview.py --stereo-camera-index "CCB Camera" --square-size-m 0.019
"""

import argparse
from pathlib import Path

import cv2

from vla_bridge.stereo_calibration import (
    DEFAULT_CALIBRATION_PATH,
    DEFAULT_PATTERN_SIZE,
    DEFAULT_SQUARE_SIZE_M,
    StereoCalibration,
    build_object_points,
    calibrate_single_camera,
    calibrate_stereo_pair,
    detect_checkerboard_corners,
    save_calibration,
)
from vla_bridge.stereo_camera import StereoSplitCamera


def run_interactive_calibration(
    stereo_camera,
    num_views: int = 15,
    pattern_size: tuple[int, int] = DEFAULT_PATTERN_SIZE,
    square_size_m: float = DEFAULT_SQUARE_SIZE_M,
) -> StereoCalibration:
    """Live-preview capture loop. Shows a continuously-updating window with
    detected-corners overlay; SPACE captures the CURRENTLY DISPLAYED frame as
    one calibration view (only if corners were detected in both halves at
    that instant), q/ESC stops early. Raises `RuntimeError` if fewer than 3
    views were captured. Same underlying math as
    `stereo_calibration.run_calibration_session()` -- only the capture UX
    differs.
    """
    object_points_list = []
    left_points_list = []
    right_points_list = []
    first_left_frame = None
    captured = 0

    print(f"Live preview started. SPACE = capture this view, q/ESC = stop early.")
    print(f"Need {num_views} valid views (each must detect corners in BOTH halves).")

    while captured < num_views:
        left = stereo_camera.read_left()
        right = stereo_camera.read_right()
        if left is None or right is None:
            continue

        left_corners = detect_checkerboard_corners(left, pattern_size)
        right_corners = detect_checkerboard_corners(right, pattern_size)
        both_ok = left_corners is not None and right_corners is not None

        left_vis = left.copy()
        right_vis = right.copy()
        if left_corners is not None:
            cv2.drawChessboardCorners(left_vis, pattern_size, left_corners, True)
        if right_corners is not None:
            cv2.drawChessboardCorners(right_vis, pattern_size, right_corners, True)

        status = (
            f"View {captured + 1}/{num_views} | "
            f"LEFT: {'OK' if left_corners is not None else 'no'}  "
            f"RIGHT: {'OK' if right_corners is not None else 'no'}  |  "
            f"SPACE=capture  q=stop"
        )
        color = (0, 255, 0) if both_ok else (0, 0, 255)
        cv2.putText(left_vis, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        combined = cv2.hconcat([left_vis, right_vis])
        cv2.imshow("Stereo calibration -- SPACE to capture, q to stop", combined)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):  # 'q' or ESC
            print("Stopped early by user.")
            break
        if key == ord(" "):
            if not both_ok:
                print("  Skipped -- checkerboard not detected in both halves at capture instant.")
                continue
            if first_left_frame is None:
                first_left_frame = left
            object_points_list.append(build_object_points(pattern_size, square_size_m))
            left_points_list.append(left_corners)
            right_points_list.append(right_corners)
            captured += 1
            print(f"  Captured view {captured}/{num_views}")

    cv2.destroyAllWindows()

    if len(object_points_list) < 3:
        raise RuntimeError(
            f"Only {len(object_points_list)} valid calibration view(s) captured (need "
            f"at least 3) -- reposition the checkerboard and try again"
        )

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
    parser = argparse.ArgumentParser(description=__doc__)
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

    print(f"Opening AR0144 via StereoSplitCamera(index={args.stereo_camera_index!r})...")
    stereo_camera = StereoSplitCamera(index=args.stereo_camera_index)

    calibration = run_interactive_calibration(
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
