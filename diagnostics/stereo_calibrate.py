"""Compute stereo calibration from captured checkerboard pairs.

Reads diagnostics/outputs/stereo_calib/pair_NNN_{left,right}.png (produced by
stereo_calibration_capture.py), runs per-lens intrinsic calibration then
stereoCalibrate + stereoRectify, and writes the result to
diagnostics/stereo_calibration.npz + .json (both committed -- this is real
hardware config worth keeping, unlike the ephemeral snapshots in outputs/,
which stays gitignored).

Also renders one sanity disparity/depth map from the first captured pair
using the *rectified* result, so you can visually confirm the calibration
actually produces clean depth before trusting it.

Usage:
    python stereo_calibrate.py [--corners 9x6] [--square-mm 20.0]
                                [--in diagnostics/outputs/stereo_calib]
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

IN_DEFAULT = Path(__file__).resolve().parent / "outputs" / "stereo_calib"
OUTLIER_THRESHOLD_FACTOR = 2.0  # drop pairs whose per-image RMS exceeds this x the median


def per_image_errors(objpoints, imgpoints, mtx, dist, rvecs, tvecs):
    """Per-calibration-image reprojection RMS (px), one value per image."""
    errors = []
    for i in range(len(objpoints)):
        projected, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        diff = imgpoints[i].reshape(-1, 2) - projected.reshape(-1, 2)
        errors.append(float(np.sqrt(np.mean(np.sum(diff**2, axis=1)))))
    return errors
OUT_NPZ = Path(__file__).resolve().parent / "stereo_calibration.npz"
OUT_JSON = Path(__file__).resolve().parent / "stereo_calibration.json"
OUT_SANITY_PNG = Path(__file__).resolve().parent / "outputs" / "stereo_calib_sanity_disparity.png"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corners", type=str, default="9x6")
    parser.add_argument("--square-mm", type=float, default=20.0)
    parser.add_argument("--in", dest="in_dir", type=Path, default=IN_DEFAULT)
    args = parser.parse_args()

    corners_x, corners_y = (int(v) for v in args.corners.split("x"))
    pattern_size = (corners_x, corners_y)
    square_mm = args.square_mm

    left_paths = sorted(args.in_dir.glob("pair_*_left.png"))
    if len(left_paths) < 8:
        raise SystemExit(
            f"Only found {len(left_paths)} pairs in {args.in_dir} -- need at least "
            "8-10 (15-20+ recommended) for a reliable calibration. Run "
            "stereo_calibration_capture.py first."
        )
    print(f"Found {len(left_paths)} candidate pairs.")

    # 3D object points for the checkerboard, in board-local mm, z=0 plane.
    objp = np.zeros((corners_x * corners_y, 3), np.float32)
    objp[:, :2] = np.mgrid[0:corners_x, 0:corners_y].T.reshape(-1, 2) * square_mm

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK

    objpoints, imgpoints_l, imgpoints_r = [], [], []
    img_size = None
    used_pairs = []

    for lp in left_paths:
        rp = lp.with_name(lp.name.replace("_left.png", "_right.png"))
        if not rp.exists():
            continue
        img_l = cv2.imread(str(lp))
        img_r = cv2.imread(str(rp))
        gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
        img_size = gray_l.shape[::-1]

        ok_l, corners_l = cv2.findChessboardCorners(gray_l, pattern_size, flags=flags)
        ok_r, corners_r = cv2.findChessboardCorners(gray_r, pattern_size, flags=flags)
        if not (ok_l and ok_r):
            print(f"  {lp.name}: skipped (corners not found in both -- shouldn't happen if "
                  f"captured via stereo_calibration_capture.py's live overlay)")
            continue

        corners_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
        corners_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)

        objpoints.append(objp)
        imgpoints_l.append(corners_l)
        imgpoints_r.append(corners_r)
        used_pairs.append(lp.name)

    print(f"Using {len(objpoints)}/{len(left_paths)} pairs (rest failed corner re-detection).")
    if len(objpoints) < 8:
        raise SystemExit("Too few usable pairs after re-detection -- capture more.")

    print("Pass 1/2: calibrating with all pairs to find per-pair outliers "
          "(e.g. bent/non-flat board captures)...")
    rms_l0, mtx_l0, dist_l0, rvecs_l0, tvecs_l0 = cv2.calibrateCamera(
        objpoints, imgpoints_l, img_size, None, None
    )
    rms_r0, mtx_r0, dist_r0, rvecs_r0, tvecs_r0 = cv2.calibrateCamera(
        objpoints, imgpoints_r, img_size, None, None
    )
    err_l = per_image_errors(objpoints, imgpoints_l, mtx_l0, dist_l0, rvecs_l0, tvecs_l0)
    err_r = per_image_errors(objpoints, imgpoints_r, mtx_r0, dist_r0, rvecs_r0, tvecs_r0)
    per_pair_err = [max(a, b) for a, b in zip(err_l, err_r)]  # worst of the two lenses

    median_err = float(np.median(per_pair_err))
    threshold = median_err * OUTLIER_THRESHOLD_FACTOR
    print(f"  all-pairs RMS: left {rms_l0:.4f}px, right {rms_r0:.4f}px  "
          f"(median per-pair error: {median_err:.4f}px, outlier threshold: {threshold:.4f}px)")
    print("  per-pair reprojection error:")
    keep_idx = []
    for i, (name, e) in enumerate(zip(used_pairs, per_pair_err)):
        flag = " <-- OUTLIER, dropping" if e > threshold else ""
        print(f"    {name}: {e:.4f}px{flag}")
        if e <= threshold:
            keep_idx.append(i)

    n_dropped = len(used_pairs) - len(keep_idx)
    if n_dropped:
        print(f"\n  Dropping {n_dropped} outlier pair(s) (likely bent board / bad detection), "
              f"re-running calibration on the remaining {len(keep_idx)}.")
        objpoints = [objpoints[i] for i in keep_idx]
        imgpoints_l = [imgpoints_l[i] for i in keep_idx]
        imgpoints_r = [imgpoints_r[i] for i in keep_idx]
        used_pairs = [used_pairs[i] for i in keep_idx]
        if len(objpoints) < 8:
            raise SystemExit(
                f"Only {len(objpoints)} pairs survive outlier rejection -- need at least 8. "
                "Capture more (keeping the board flat this time)."
            )
    else:
        print("\n  No outliers above threshold -- all pairs look consistent.")

    print("\nPass 2/2: calibrating left lens intrinsics on the clean set...")
    rms_l, mtx_l, dist_l, _, _ = cv2.calibrateCamera(objpoints, imgpoints_l, img_size, None, None)
    print(f"  left reprojection RMS: {rms_l:.4f} px")

    print("Calibrating right lens intrinsics on the clean set...")
    rms_r, mtx_r, dist_r, _, _ = cv2.calibrateCamera(objpoints, imgpoints_r, img_size, None, None)
    print(f"  right reprojection RMS: {rms_r:.4f} px")

    print("Running stereo calibration (fixing per-lens intrinsics)...")
    stereo_flags = cv2.CALIB_FIX_INTRINSIC
    rms_stereo, mtx_l, dist_l, mtx_r, dist_r, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_l, imgpoints_r, mtx_l, dist_l, mtx_r, dist_r, img_size,
        criteria=criteria, flags=stereo_flags,
    )
    baseline_mm = float(np.linalg.norm(T))
    print(f"  stereo reprojection RMS: {rms_stereo:.4f} px")
    print(f"  baseline: {baseline_mm:.2f} mm")

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        mtx_l, dist_l, mtx_r, dist_r, img_size, R, T, alpha=0,
    )

    np.savez(
        OUT_NPZ,
        mtx_l=mtx_l, dist_l=dist_l, mtx_r=mtx_r, dist_r=dist_r,
        R=R, T=T, E=E, F=F, R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
        img_size=np.array(img_size), square_mm=square_mm,
        corners=np.array(pattern_size), rms_stereo=rms_stereo,
    )
    summary = {
        "img_size_px": list(img_size),
        "square_mm": square_mm,
        "pattern_corners": list(pattern_size),
        "num_pairs_used": len(objpoints),
        "num_pairs_captured": len(left_paths),
        "outlier_pairs_dropped": n_dropped,
        "pairs_used": used_pairs,
        "reprojection_rms_left_px": rms_l,
        "reprojection_rms_right_px": rms_r,
        "reprojection_rms_stereo_px": rms_stereo,
        "baseline_mm": baseline_mm,
        "camera_matrix_left": mtx_l.tolist(),
        "camera_matrix_right": mtx_r.tolist(),
        "dist_coeffs_left": dist_l.tolist(),
        "dist_coeffs_right": dist_r.tolist(),
        "R": R.tolist(),
        "T_mm": T.tolist(),
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved: {OUT_NPZ}")
    print(f"Saved: {OUT_JSON}")

    if rms_stereo > 1.0:
        print(f"\n⚠ stereo reprojection RMS ({rms_stereo:.2f}px) is high (>1px) -- calibration "
              "may be unreliable. Common causes: printed square size doesn't match --square-mm, "
              "board wasn't flat, or too few/too-similar poses. Consider re-capturing.")
    else:
        print(f"\n✓ stereo reprojection RMS ({rms_stereo:.2f}px) looks good (<1px).")

    # Sanity check: rectify + compute disparity on the first pair, visually confirm
    # the checkerboard's depth comes back flat and roughly at the right distance.
    print("\nRendering rectified disparity sanity check from the first captured pair...")
    map1_l, map2_l = cv2.initUndistortRectifyMap(mtx_l, dist_l, R1, P1, img_size, cv2.CV_32FC1)
    map1_r, map2_r = cv2.initUndistortRectifyMap(mtx_r, dist_r, R2, P2, img_size, cv2.CV_32FC1)

    img_l = cv2.imread(str(left_paths[0]))
    img_r = cv2.imread(str(left_paths[0]).replace("_left.png", "_right.png"))
    rect_l = cv2.remap(img_l, map1_l, map2_l, cv2.INTER_LINEAR)
    rect_r = cv2.remap(img_r, map1_r, map2_r, cv2.INTER_LINEAR)

    gray_rl = cv2.cvtColor(rect_l, cv2.COLOR_BGR2GRAY)
    gray_rr = cv2.cvtColor(rect_r, cv2.COLOR_BGR2GRAY)
    stereo = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=16 * 8, blockSize=7,
        P1=8 * 3 * 7**2, P2=32 * 3 * 7**2, disp12MaxDiff=1,
        uniquenessRatio=10, speckleWindowSize=100, speckleRange=2,
    )
    disparity = stereo.compute(gray_rl, gray_rr).astype(np.float32) / 16.0
    points_3d = cv2.reprojectImageTo3D(disparity, Q)
    valid = disparity > 0
    if valid.any():
        depths_mm = points_3d[..., 2][valid]
        depths_mm = depths_mm[np.isfinite(depths_mm)]
        print(f"  valid disparity pixels: {valid.sum()}/{disparity.size}")
        if depths_mm.size:
            print(f"  metric depth range on this pair: {depths_mm.min():.0f}mm - "
                  f"{depths_mm.max():.0f}mm (median {np.median(depths_mm):.0f}mm)")
            print("  -> if this roughly matches how far the board actually was from the "
                  "camera when pair_000 was captured, the metric calibration is trustworthy.")

    disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    disp_color = cv2.applyColorMap(np.uint8(disp_vis), cv2.COLORMAP_JET)
    OUT_SANITY_PNG.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT_SANITY_PNG), disp_color)
    print(f"  saved: {OUT_SANITY_PNG}")


if __name__ == "__main__":
    main()
