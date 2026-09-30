"""Local Fast-FoundationStereo (FastFS) depth-computation client (Phase 12,
Plan 12-06 -- step 3 of the 3-step depth-calibration extension, D-07).

Rectifies a live AR0144 stereo pair (captured via the existing, unmodified
`StereoSplitCamera` -- never a second, independent camera open) using a
`stereo_calibration.StereoCalibration` (Plan 12-05), downsamples it to
Fast-FoundationStereo's required input constraints (<1000px width, dimensions
divisible by 32, scaling intrinsics proportionally), POSTs the rectified pair
plus reduced calibration to the Colab-hosted FastFS endpoint documented in
`policy_server_launch.md` (Plan 12-06 Task 2), and returns the resulting
per-pixel depth map (float32, meters).

Depth is RECORDING-ONLY this phase (D-08): this module is never imported by
`vla_bridge.robot_client`, and its output is never added to the observation
dict `BridgeActionSource`/`connect_bridge` send to the policy.
`run_vla_episode.py` (Plan 12-06 Task 3) calls this module purely to persist
depth maps for inspection. `compute_depth()` never raises on a
network/endpoint failure -- it always returns `None`, matching
`io_logger.capture_camera_frame()`'s fail-safe convention, since a flaky
Colab connection must not crash the live episode's real control loop.
"""

import argparse
import base64
import io
import hashlib
import json
from dataclasses import asdict
import time
from pathlib import Path

import cv2
import numpy as np
import requests

from vla_bridge import stereo_calibration

# < FastFS's 1000px input-width limit, already a multiple of 32.
FASTFS_MAX_WIDTH = 992
_DIVISOR = 32


def compute_target_size(
    orig_width: int, orig_height: int, max_width: int = FASTFS_MAX_WIDTH
) -> tuple[int, int]:
    """Computes a `(target_width, target_height)` downsample target satisfying
    Fast-FoundationStereo's input constraints: width < 1000px, both dimensions
    divisible by `_DIVISOR` (32).

    `scale = min(max_width / orig_width, 1.0)` never upscales past the
    original size. Each dimension is floored independently to the nearest
    multiple of `_DIVISOR` at or below its own scaled value -- the per-axis
    scale factors ACTUALLY APPLIED (`target_width / orig_width`,
    `target_height / orig_height`) will differ slightly from this initial
    `scale` (and from each other) once independent floor-rounding is applied;
    callers (`DepthCameraClient.compute_depth()`) must recompute the true
    per-axis ratios from the actual orig/target values before calling
    `scale_intrinsics()`, never reuse this function's internal `scale`.
    """
    scale = min(max_width / orig_width, 1.0)
    target_width = max(_DIVISOR, (int(orig_width * scale) // _DIVISOR) * _DIVISOR)
    target_height = max(_DIVISOR, (int(orig_height * scale) // _DIVISOR) * _DIVISOR)
    return target_width, target_height


def scale_intrinsics(intrinsics_flat: list[float], scale_x: float, scale_y: float) -> list[float]:
    """Scales a flattened row-major 3x3 camera intrinsics matrix
    `[fx, 0, cx, 0, fy, cy, 0, 0, 1]` by independent x/y scale factors: `fx`/
    `cx` scaled by `scale_x`, `fy`/`cy` scaled by `scale_y`, the rest (skew
    and homogeneous row) left unchanged."""
    fx, _, cx, _, fy, cy, *_rest = intrinsics_flat
    return [fx * scale_x, 0.0, cx * scale_x, 0.0, fy * scale_y, cy * scale_y, 0.0, 0.0, 1.0]


class DepthCameraClient:
    """Rectifies a live AR0144 stereo pair using a saved `StereoCalibration`,
    downsamples it to Fast-FoundationStereo's input constraints, and requests
    a metric depth map from a Colab-hosted FastFS HTTP endpoint.
    """

    def __init__(
        self,
        calibration: "stereo_calibration.StereoCalibration",
        endpoint_url: str,
        max_width: int = FASTFS_MAX_WIDTH,
    ):
        self.endpoint_url = endpoint_url
        self.timeout_s = 10.0
        self.calibration_metadata = asdict(calibration)
        self.calibration_sha256 = hashlib.sha256(json.dumps(self.calibration_metadata, sort_keys=True).encode()).hexdigest()
        self.max_width = max_width
        self._build_rectification_maps(calibration)

    def _build_rectification_maps(self, calibration: "stereo_calibration.StereoCalibration") -> None:
        K1 = np.asarray(calibration.left_K, dtype=np.float64).reshape(3, 3)
        D1 = np.asarray(calibration.left_D, dtype=np.float64).reshape(-1)
        K2 = np.asarray(calibration.right_K, dtype=np.float64).reshape(3, 3)
        D2 = np.asarray(calibration.right_D, dtype=np.float64).reshape(-1)
        R = np.asarray(calibration.R, dtype=np.float64).reshape(3, 3)
        T = np.asarray(calibration.T, dtype=np.float64).reshape(3, 1)
        image_size = (int(calibration.image_size[0]), int(calibration.image_size[1]))

        R1, R2, P1, P2, _Q, _roi1, _roi2 = cv2.stereoRectify(
            K1, D1, K2, D2, image_size, R, T, alpha=0
        )

        self._map1x, self._map1y = cv2.initUndistortRectifyMap(
            K1, D1, R1, P1, image_size, cv2.CV_32FC1
        )
        self._map2x, self._map2y = cv2.initUndistortRectifyMap(
            K2, D2, R2, P2, image_size, cv2.CV_32FC1
        )

        # Rectified camera's 3x3 intrinsics submatrix, dropping P1's 4th column.
        self._rectified_intrinsics_flat = np.asarray(P1[:, :3]).flatten().tolist()
        # The AR0144's baseline is predominantly along the horizontal axis,
        # matching T's first component in this project's stereo-calibration
        # convention from Plan 12-05.
        self._baseline_m = abs(float(T.flatten()[0]))

    def rectify(self, left_frame: np.ndarray, right_frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Remaps `left_frame`/`right_frame` through this instance's
        precomputed rectification maps. Preserves the configured
        `image_size`, regardless of `stereoRectify`'s internal valid-ROI
        cropping (not applied to the map's output size since `alpha=0`)."""
        rectified_left = cv2.remap(left_frame, self._map1x, self._map1y, cv2.INTER_LINEAR)
        rectified_right = cv2.remap(right_frame, self._map2x, self._map2y, cv2.INTER_LINEAR)
        return rectified_left, rectified_right

    @staticmethod
    def _encode_png(image: np.ndarray) -> str:
        _ok, buf = cv2.imencode(".png", image)
        return base64.b64encode(buf.tobytes()).decode("ascii")

    @staticmethod
    def _decode_depth_npy(depth_b64: str) -> np.ndarray:
        raw = base64.b64decode(depth_b64)
        return np.load(io.BytesIO(raw), allow_pickle=False)

    def check_ready(self) -> dict:
        health_url = self.endpoint_url.rsplit('/depth', 1)[0] + '/health'
        response = requests.get(health_url, timeout=self.timeout_s)
        response.raise_for_status()
        health = response.json()
        if health.get('status') != 'ready' or health.get('protocol_version') != 1:
            raise RuntimeError('Expected a warmed resident FastFS server (protocol 1)')
        return health

    def compute_depth(self, left_frame, right_frame):
        """Compatibility API for the depth inspection CLI."""
        return self.compute_depth_result(left_frame, right_frame).get('depth')

    def compute_depth_result(self, left_frame, right_frame, request_id=None):
        """Return metric depth, the exact aligned RGB input and provenance.

        All failures become explicit result records; the background recorder
        persists these without interrupting motion or labelling them as success.
        """
        result = dict(depth=None, calibration_sha256=self.calibration_sha256,
                      units='metres', pixel_frame='rectified_left_resized')
        try:
            expected = tuple(self.calibration_metadata['image_size'])
            if any((x.shape[1], x.shape[0]) != expected for x in (left_frame, right_frame)):
                raise ValueError('capture resolution does not match calibration')
            left, right = self.rectify(left_frame, right_frame)
            h, w = left.shape[:2]
            tw, th = compute_target_size(w, h, self.max_width)
            left, right = cv2.resize(left, (tw, th)), cv2.resize(right, (tw, th))
            K = scale_intrinsics(self._rectified_intrinsics_flat, tw / w, th / h)
            result.update(aligned_left=left, aligned_right=right,
                          intrinsics_flat=K, baseline_m=self._baseline_m)
            payload = dict(left_png_b64=self._encode_png(left), right_png_b64=self._encode_png(right),
                           intrinsics_flat=K, baseline_m=self._baseline_m, request_id=request_id)
            response = requests.post(self.endpoint_url, json=payload, timeout=self.timeout_s)
            response.raise_for_status()
            data = response.json()
            depth = self._decode_depth_npy(data['depth_npy_b64'])
            if depth.shape != (th, tw) or depth.dtype != np.float32:
                raise ValueError(f'invalid depth shape/dtype: {depth.shape}/{depth.dtype}')
            if request_id is not None and data.get('request_id') != request_id:
                raise ValueError('depth response request_id mismatch')
            valid = np.isfinite(depth) & (depth > 0)
            if not valid.any():
                raise ValueError('depth response contains no valid positive samples')
            depth = depth.copy()
            depth[~valid] = np.nan
            result.update(depth=depth, valid_fraction=float(valid.mean()),
                          model=data.get('model'), inference_ms=data.get('inference_ms'))
        except Exception as exc:
            result.update(depth=None, error=f'{type(exc).__name__}: {exc}')
        return result


def main() -> None:
    """CLI for the human-verifiable known-distance depth-accuracy checkpoint
    (Plan 12-06 Task 1/Task 2). Reads live stereo frames, requests depth from
    the given FastFS endpoint, and prints the center-pixel depth value
    alongside its min/max, so a human can compare it against a known,
    physically-measured real-world distance."""
    parser = argparse.ArgumentParser(
        description="Known-distance depth-accuracy checkpoint against a live Colab FastFS endpoint."
    )
    parser.add_argument("--calibration", type=Path, required=True, help="Path to a saved StereoCalibration JSON file")
    parser.add_argument("--endpoint", type=str, required=True, help="Colab-hosted FastFS HTTP endpoint URL")
    parser.add_argument(
        "--stereo-camera-index",
        type=str,
        default="1",
        help="AVFoundation device NAME or numeric index for the AR0144 (prefer NAME).",
    )
    args = parser.parse_args()

    from vla_bridge.stereo_camera import StereoSplitCamera

    stereo_camera = StereoSplitCamera(index=args.stereo_camera_index)
    depth_client = DepthCameraClient(
        stereo_calibration.load_calibration(args.calibration), args.endpoint
    )

    while True:
        try:
            left_frame = stereo_camera.read_left()
            right_frame = stereo_camera.read_right()
            depth_map = depth_client.compute_depth(left_frame, right_frame)
            if depth_map is None:
                print("WARNING: compute_depth() returned None, skipping this reading")
                time.sleep(1.0)
                continue

            center = depth_map[depth_map.shape[0] // 2, depth_map.shape[1] // 2]
            print(f"center={center:.3f}m  min={depth_map.min():.3f}m  max={depth_map.max():.3f}m")
            time.sleep(1.0)
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()


__all__ = ["compute_target_size", "scale_intrinsics", "DepthCameraClient", "FASTFS_MAX_WIDTH"]
