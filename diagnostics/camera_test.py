"""
Detect connected USB cameras and grab one test frame from each.

Covers both cameras in the parts list: the Waveshare IMX335 5MP (wrist/gripper,
mono) and the Waveshare AR0144 2MP stereo module (overhead, appears as a single
wide side-by-side frame — split it down the middle for left/right).

Usage:
    python camera_test.py [--max-index 5] [--out diagnostics/outputs]
"""
import argparse
import time
from pathlib import Path

import cv2

OUT_DEFAULT = Path(__file__).resolve().parent / "outputs"
WARMUP_READS = 15  # discard early frames while exposure/auto-gain settles


def probe(max_index: int, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    found = []
    for idx in range(max_index + 1):
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap.release()
            continue

        ok, frame, forced_note = False, None, ""
        for _ in range(WARMUP_READS):
            ok, frame = cap.read()
            if ok:
                time.sleep(0.05)  # let auto-exposure/gain keep converging
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        if not ok:
            # Default format negotiation failed - retry forcing MJPG + a low
            # resolution, which fixes some UVC stereo modules that don't like
            # the AVFoundation backend's default pick.
            cap = cv2.VideoCapture(idx)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            for _ in range(WARMUP_READS):
                ok, frame = cap.read()
                if ok:
                    time.sleep(0.05)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            forced_note = " (needed forced MJPG/640x480 to read)" if ok else ""

        if not ok:
            print(f"  index {idx}: opened but no frame (tried default + forced MJPG/640x480)")
            continue
        tag = " (looks like the stereo pair - wide frame)" if w >= 2 * h else ""
        print(f"  index {idx}: {w}x{h}{tag}{forced_note}")
        out_path = out_dir / f"camera_{idx}.png"
        cv2.imwrite(str(out_path), frame)
        print(f"    saved -> {out_path}")
        found.append(idx)
    if not found:
        print("No cameras responded. Check USB connections / macOS camera permission for Terminal.")
    return found


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-index", type=int, default=5)
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = parser.parse_args()

    print(f"Probing camera indices 0-{args.max_index}...")
    probe(args.max_index, args.out)
