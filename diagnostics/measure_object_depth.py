"""Measure a dark object's distance from the camera using the stereo calibration.

Captures a fresh stereo pair, rectifies it with diagnostics/stereo_calibration.npz,
computes disparity, finds the darkest large blob in the scene (excluding the top of
frame, where the gripper usually is), and reports its depth (distance from camera)
plus a reliability check (brightness + specular glint on the object's own surface).

This is diagnostic, not production code -- object detection here is a simple
brightness threshold, good enough for a single dark object on a light mat. See
diagnostics/UAT/function/depth/UAT.md for the full story on why dark/low-contrast
objects can give a misleadingly confident-looking but fake depth reading, and how
to tell the difference (that's exactly what the reliability check below does).

Usage:
    python measure_object_depth.py [--camera-index N] [--min-area 2000]
                                    [--exclude-top-frac 0.28]

Requires diagnostics/.venv (plain OpenCV/numpy) and an existing
diagnostics/stereo_calibration.npz (run stereo_calibration_capture.py +
stereo_calibrate.py first if it doesn't exist yet).
"""
import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np

CALIB_PATH = Path(__file__).resolve().parent / "stereo_calibration.npz"


def find_camera_index(preferred):
    if preferred is not None:
        return preferred
    for idx in range(6):
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap.release()
            continue
        ok, frame = cap.read()
        cap.release()
        if ok and frame is not None and frame.shape[1] >= 2 * frame.shape[0]:
            return idx
    raise RuntimeError("Could not auto-detect the stereo camera. Pass --camera-index explicitly "
                        "(check with: ffmpeg -f avfoundation -list_devices true -i \"\").")


def capture_pair(camera_index):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")
    # Warm up a few reads so exposure/gain settle, matching camera_test.py's convention.
    frame = None
    for _ in range(15):
        ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError("Failed to grab a frame")
    h, w_full = frame.shape[:2]
    w = w_full // 2
    return frame[:, :w], frame[:, w:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-index", type=int, default=None)
    parser.add_argument("--min-area", type=int, default=2000)
    parser.add_argument("--exclude-top-frac", type=float, default=0.28,
                         help="Zero out this fraction of the top of frame (where the gripper usually is)")
    parser.add_argument("--dark-threshold", type=int, default=100,
                         help="Pixels darker than this (0-255) are considered part of an object")
    args = parser.parse_args()

    if not CALIB_PATH.exists():
        raise SystemExit(f"No calibration found at {CALIB_PATH}. Run "
                          "stereo_calibration_capture.py + stereo_calibrate.py first.")
    calib = np.load(CALIB_PATH)
    mtx_l, dist_l, mtx_r, dist_r = calib["mtx_l"], calib["dist_l"], calib["mtx_r"], calib["dist_r"]
    R1, R2, P1, P2, Q = calib["R1"], calib["R2"], calib["P1"], calib["P2"], calib["Q"]
    img_size = tuple(calib["img_size"])

    idx = find_camera_index(args.camera_index)
    print(f"Using camera index {idx}")
    left, right = capture_pair(idx)
    h, w = left.shape[:2]

    map1_l, map2_l = cv2.initUndistortRectifyMap(mtx_l, dist_l, R1, P1, img_size, cv2.CV_32FC1)
    map1_r, map2_r = cv2.initUndistortRectifyMap(mtx_r, dist_r, R2, P2, img_size, cv2.CV_32FC1)
    rect_l = cv2.remap(left, map1_l, map2_l, cv2.INTER_LINEAR)
    rect_r = cv2.remap(right, map1_r, map2_r, cv2.INTER_LINEAR)
    gray_l = cv2.cvtColor(rect_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(rect_r, cv2.COLOR_BGR2GRAY)

    stereo = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=16 * 8, blockSize=7,
        P1=8 * 3 * 7 ** 2, P2=32 * 3 * 7 ** 2, disp12MaxDiff=1,
        uniquenessRatio=10, speckleWindowSize=100, speckleRange=2,
    )
    disp = stereo.compute(gray_l, gray_r).astype(np.float32) / 16.0
    points_3d = cv2.reprojectImageTo3D(disp, Q)
    valid = (disp > 1.0) & np.isfinite(points_3d[..., 2]) & (np.abs(points_3d[..., 2]) < 5000)
    print(f"Overall scene valid depth: {100 * valid.mean():.1f}%")

    # Find the object: darkest large blob, excluding the top of frame (gripper).
    _, mask = cv2.threshold(gray_l, args.dark_threshold, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    mask[: int(h * args.exclude_top_frac), :] = 0
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > args.min_area]
    if not contours:
        raise SystemExit("No object found. Adjust --dark-threshold / --min-area, or check "
                          "the object is actually in frame (below the excluded top region).")
    contours.sort(key=cv2.contourArea, reverse=True)
    c = contours[0]
    rect = cv2.minAreaRect(c)
    (cx, cy), (rw, rh), angle = rect
    box = cv2.boxPoints(rect).astype(int)
    print(f"Object bbox: center=({cx:.0f},{cy:.0f}) size=({rw:.0f}x{rh:.0f})px angle={angle:.1f}")

    obj_mask_full = np.zeros((h, w), np.uint8)
    cv2.drawContours(obj_mask_full, [box], -1, 255, -1)
    erode_mask = cv2.erode(obj_mask_full, np.ones((9, 9), np.uint8)) > 0
    obj_core_mask = erode_mask & valid

    obj_px = gray_l[erode_mask]
    glint_pct = 100 * (obj_px > 200).mean()
    coverage_pct = 100 * obj_core_mask.sum() / erode_mask.sum()
    print(f"\nObject surface brightness: mean={obj_px.mean():.1f} std={obj_px.std():.1f}")
    print(f"Specular glint (%px > 200): {glint_pct:.2f}%")
    print(f"Valid-depth coverage on object: {coverage_pct:.1f}%")

    obj_z = points_3d[..., 2][obj_core_mask]
    p25, median, p75 = np.percentile(obj_z, 25), np.median(obj_z), np.percentile(obj_z, 75)
    print(f"\nDEPTH (distance from camera): median={median:.1f}mm "
          f"(p25={p25:.1f}, p75={p75:.1f}, std={obj_z.std():.1f})")

    # Reliability check: a real measurement has natural per-pixel variance.
    # p25 == median == p75 (to the decimal) means most "valid" pixels returned an
    # identical value -- a smoothness-filled fake plateau, not a real reading.
    degenerate = abs(p25 - median) < 0.5 and abs(p75 - median) < 0.5
    if degenerate:
        print("\n⚠ UNRELIABLE: depth values are degenerate (nearly identical across "
              "the object) -- this usually means the object's surface is too dark/low-"
              "contrast (see brightness above) or has specular glint for real stereo "
              "matching to work. Treat this number as NOT trustworthy. See "
              "diagnostics/UAT/function/depth/UAT.md.")
    elif obj_px.mean() < 60:
        print("\n⚠ CAUTION: object surface is quite dark (mean < 60) -- double check "
              "the depth reading against a rough manual estimate before trusting it.")
    elif glint_pct > 10:
        print(f"\n⚠ CAUTION: {glint_pct:.1f}% specular glint on the object -- depth near "
              "those bright spots may be unreliable even if the overall reading looks sane.")
    else:
        print("\n✓ Depth reading looks reliable (real variance, reasonable brightness, low glint).")

    disp_vis = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX)
    disp_color = cv2.applyColorMap(np.uint8(disp_vis), cv2.COLORMAP_JET)
    cv2.drawContours(disp_color, [box], -1, (255, 255, 255), 2)
    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "measure_object_depth_last.png"
    cv2.imwrite(str(out_path), disp_color)
    print(f"\nSaved disparity visualization: {out_path}")


if __name__ == "__main__":
    main()
