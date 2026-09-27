"""Live checkerboard-detection preview for the AR0144 stereo camera (Phase 12,
manual calibration-session aid).

`vla_bridge/stereo_calibration.py`'s capture session is deliberately headless
(prompts for Enter, no visual feedback) -- useful for the actual calibration
run, but gives no way to confirm the checkerboard is even in frame before
capturing a view. This script opens the SAME AR0144 stereo camera (via
`StereoSplitCamera`, never a second concurrent capture -- see
`stereo_camera.py`'s own docstring on why that's unsafe on this rig) and shows
a live window with detected corners overlaid, so a human can find a working
checkerboard position/distance/angle before running the real calibration
session.

IMPORTANT: only one process can hold the AR0144 open at a time. Close this
preview (press 'q' or Ctrl+C) before running `stereo_calibration.py`.

Usage:
    python preview_checkerboard.py --stereo-camera-index "CCB Camera"
"""

import argparse

import cv2

from vla_bridge.stereo_calibration import DEFAULT_PATTERN_SIZE, detect_checkerboard_corners
from vla_bridge.stereo_camera import StereoSplitCamera


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stereo-camera-index",
        type=str,
        default="1",
        help="AVFoundation device NAME or numeric index for the AR0144 (prefer NAME -- "
        "numeric index has been confirmed to drift between process launches on macOS).",
    )
    parser.add_argument("--pattern-cols", type=int, default=DEFAULT_PATTERN_SIZE[0])
    parser.add_argument("--pattern-rows", type=int, default=DEFAULT_PATTERN_SIZE[1])
    args = parser.parse_args()

    pattern_size = (args.pattern_cols, args.pattern_rows)
    print(f"Opening AR0144 via StereoSplitCamera(index={args.stereo_camera_index!r})...")
    cam = StereoSplitCamera(index=args.stereo_camera_index)
    print("Press 'q' in the preview window (or Ctrl+C here) to quit.")
    print(f"Looking for a {pattern_size[0]}x{pattern_size[1]} internal-corner checkerboard.")

    try:
        while True:
            left = cam.read_left()
            right = cam.read_right()
            if left is None or right is None:
                print("WARNING: got a None frame, retrying...")
                continue

            left_corners = detect_checkerboard_corners(left, pattern_size)
            right_corners = detect_checkerboard_corners(right, pattern_size)

            left_vis = left.copy()
            right_vis = right.copy()
            if left_corners is not None:
                cv2.drawChessboardCorners(left_vis, pattern_size, left_corners, True)
            if right_corners is not None:
                cv2.drawChessboardCorners(right_vis, pattern_size, right_corners, True)

            both_detected = left_corners is not None and right_corners is not None
            status = (
                f"LEFT: {'DETECTED' if left_corners is not None else 'not found'} | "
                f"RIGHT: {'DETECTED' if right_corners is not None else 'not found'}"
            )
            cv2.putText(
                left_vis, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 0) if both_detected else (0, 0, 255), 2,
            )

            combined = cv2.hconcat([left_vis, right_vis])
            cv2.imshow("AR0144 stereo preview (left | right) -- press q to quit", combined)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        print("Preview closed. AR0144 released -- safe to run the real calibration script now.")


if __name__ == "__main__":
    main()
