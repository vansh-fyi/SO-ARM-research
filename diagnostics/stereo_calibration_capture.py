"""Interactive capture tool for stereo calibration image pairs.

Shows a live preview from the Waveshare AR0144 stereo module (appears as one
wide side-by-side frame -- split down the middle for left/right, same
convention as camera_test.py). Runs chessboard corner detection on both
halves every frame and overlays the result so you can see live whether a
pose will actually be usable before capturing it.

Move a printed checkerboard (see diagnostics/README or ask Claude to
regenerate one) around in front of the camera -- vary distance, tilt, and
position (including corners of the frame, not just center) -- and press
SPACE to save a pair whenever both overlays turn green. Aim for 15-20+ pairs
covering a real spread of poses; a pile of near-identical center-frame shots
gives a numerically well-conditioned-looking but practically useless
calibration.

Usage:
    python stereo_calibration_capture.py [--camera-index N] [--corners 9x6]
                                          [--out diagnostics/outputs/stereo_calib]

Keys: SPACE = capture pair (only saves if both sides detected)
      q      = quit
"""
import argparse
import time
from pathlib import Path

import cv2

OUT_DEFAULT = Path(__file__).resolve().parent / "outputs" / "stereo_calib"


def find_camera_index(preferred: int | None) -> int:
    if preferred is not None:
        return preferred
    # Heuristic: try indices 0-5, pick the first wide (stereo-shaped) frame.
    for idx in range(6):
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap.release()
            continue
        ok, frame = cap.read()
        cap.release()
        if ok and frame is not None and frame.shape[1] >= 2 * frame.shape[0]:
            print(f"Auto-detected stereo camera at index {idx} ({frame.shape[1]}x{frame.shape[0]})")
            return idx
    raise RuntimeError(
        "Could not auto-detect a stereo (wide-aspect) camera. Run "
        "`python camera_test.py` first to find the right --camera-index."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-index", type=int, default=None)
    parser.add_argument("--corners", type=str, default="9x6", help="internal corners, WxH")
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = parser.parse_args()

    corners_x, corners_y = (int(v) for v in args.corners.split("x"))
    pattern_size = (corners_x, corners_y)

    args.out.mkdir(parents=True, exist_ok=True)
    existing = sorted(args.out.glob("pair_*_left.png"))
    pair_idx = len(existing)

    idx = find_camera_index(args.camera_index)
    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {idx}")

    print(f"Looking for a {corners_x}x{corners_y}-corner checkerboard. "
          f"Starting from pair {pair_idx}. SPACE=capture, q=quit.")

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Frame grab failed, retrying...")
            time.sleep(0.05)
            continue

        h, w = frame.shape[:2]
        left = frame[:, : w // 2]
        right = frame[:, w // 2 :]

        gray_l = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

        found_l, corners_l = cv2.findChessboardCorners(gray_l, pattern_size, flags=flags)
        found_r, corners_r = cv2.findChessboardCorners(gray_r, pattern_size, flags=flags)

        disp_l, disp_r = left.copy(), right.copy()
        if found_l:
            corners_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(disp_l, pattern_size, corners_l, found_l)
        if found_r:
            corners_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(disp_r, pattern_size, corners_r, found_r)

        both_ok = found_l and found_r
        status = f"pairs saved: {pair_idx}   both detected: {'YES' if both_ok else 'no'}"
        color = (0, 200, 0) if both_ok else (0, 0, 200)
        combined = cv2.hconcat([disp_l, disp_r])
        cv2.putText(combined, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(combined, "SPACE=capture  q=quit", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow("stereo calibration capture", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            if both_ok:
                lp = args.out / f"pair_{pair_idx:03d}_left.png"
                rp = args.out / f"pair_{pair_idx:03d}_right.png"
                cv2.imwrite(str(lp), left)
                cv2.imwrite(str(rp), right)
                print(f"saved pair {pair_idx}: {lp.name}, {rp.name}")
                pair_idx += 1
            else:
                print("skip: checkerboard not fully detected in both lenses")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nCaptured {pair_idx} pairs total in {args.out}")
    print("Next: python stereo_calibrate.py")


if __name__ == "__main__":
    main()
