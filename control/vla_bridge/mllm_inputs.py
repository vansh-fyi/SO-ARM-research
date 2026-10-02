"""Per-chunk input collector for every MLLM backend (Phase 15, PIPE-02).

Decides what the model sees on each provider call. One `collect()` per chunk:

  1. Capture: flush the wrist capture with a few `grab()` calls (a plain
     `cv2.VideoCapture` can hold a stale frame after a long blocking call,
     research Pitfall 8), read it, then read the stereo pair with
     `read_left()` immediately followed by `read_right()` so both halves come
     from ONE physical AR0144 read.
  2. Depth (D-06): exactly ONE blocking `compute_depth_result()` request on the
     UNMODIFIED raw 800x600 left/right arrays (the client rejects any other
     size, research Pitfall 5). HTTP 429 "depth service busy" is retried up to
     `depth.retry_429` times with `depth.retry_sleep_s` sleeps; any other
     error is not retried. A failure never blocks the chunk.
  3. Images (D-05/D-26): wrist, left and right downsized to `image_max_side`
     for transport. The left slot carries the raw left frame, or the
     rectified `aligned_left` when `left_image_source` is `aligned` and depth
     succeeded (raw otherwise; the meta records which was sent).
  4. Depth presentation (D-28), all derived from the SAME depth array: a
     turbo heatmap with a drawn legend (image role `depth`, sent for
     `heatmap`/`both`) and a median-pooled numeric grid in metres with
     min/median/max stats (text in the depth note, sent for `grid`/`both`).
     The full 576x800 array is never sent. depth.npy, depth_heatmap.png,
     depth_grid.txt and aligned_left.png always go to the sidecar when depth
     succeeded.

A failed or unconfigured depth request always yields the fixed note
"Depth unavailable this call."; the error text (which can contain the
rotating endpoint URL) goes only into the truncated `meta["depth"]["error"]`
(threat T-15-14).

Preview CLI (keyless; captures one live set of inputs, makes NO model call):

    cd control
    python -m vla_bridge.mllm_inputs --out outputs/mllm_inputs_preview \\
        --wrist-camera-index 0 --stereo-camera-index "CCB Camera" \\
        --calibration stereo_calibration.json --endpoint https://<fastfs-host>/depth

Omit --calibration/--endpoint to preview the three camera images without
depth. Optional: --config <backend config name>, --left-image-source
raw|aligned, --depth-presentation heatmap|grid|both.
"""

import argparse
import io
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from vla_bridge.robot_client import _downsize_for_transport

# Same roles and order as base_mllm.IMAGE_ORDER (a test asserts equality; not
# imported from there because that would be a circular import, see collect()).
IMAGE_ROLES = ("wrist", "left", "right", "depth")

DEPTH_UNAVAILABLE_NOTE = "Depth unavailable this call."
DEPTH_HEATMAP_NOTE = (
    "DEPTH: a depth heatmap image is attached (label: depth); its colour scale is "
    "described in the instructions."
)
DEPTH_BOTH_HEATMAP_LINE = "A depth heatmap image of the same depth map is also attached (label: depth)."

MAX_ERROR_CHARS = 300
DEFAULT_FLUSH_GRABS = 3

LEGEND_WIDTH = 96
LEGEND_TICKS = 5
_LEGEND_BG = 40
_LEGEND_FONT = cv2.FONT_HERSHEY_SIMPLEX
_WHITE = (255, 255, 255)

_HTTP_429 = re.compile(r"\b429\b")


# --- Capture -------------------------------------------------------------------------------


def capture_mllm_inputs(wrist_cap, stereo_camera, flush_grabs: int = DEFAULT_FLUSH_GRABS) -> dict | None:
    """One wrist frame and ONE stereo pair, all raw (never downsized here).

    `read_left()` and `read_right()` are called back to back so both halves
    come from a single physical AR0144 read (a third call in between would
    force a fresh read). Returns None when any read failed.
    """
    grab = getattr(wrist_cap, "grab", None)
    if callable(grab):
        for _ in range(flush_grabs):
            grab()
    ok, wrist = wrist_cap.read()
    left = stereo_camera.read_left()
    right = stereo_camera.read_right()
    if not ok or wrist is None or left is None or right is None:
        return None
    return {
        "wrist": wrist,
        "left": left,
        "right": right,
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "stereo_pair_metadata": dict(getattr(stereo_camera, "last_pair_metadata", {}) or {}),
    }


def encode_image(frame, image_format: str, jpeg_quality: int = 90) -> tuple[bytes | None, str]:
    """Encode `frame` as PNG or JPEG; returns `(bytes or None, media_type)`."""
    if image_format == "png":
        ext, params, media_type = ".png", [], "image/png"
    elif image_format == "jpeg":
        ext, params, media_type = ".jpg", [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)], "image/jpeg"
    else:
        raise ValueError(f"image_format must be 'png' or 'jpeg', got {image_format!r}")
    try:
        ok, buffer = cv2.imencode(ext, np.ascontiguousarray(frame), params)
    except cv2.error:
        return None, media_type
    return (buffer.tobytes() if ok else None), media_type


# --- Depth presentation (D-28) -------------------------------------------------------------


def _valid_mask(depth: np.ndarray) -> np.ndarray:
    """A depth pixel is valid when finite and strictly positive."""
    with np.errstate(invalid="ignore"):
        return np.isfinite(depth) & (depth > 0)


def depth_stats(depth_m) -> dict:
    """min/median/max over the valid pixels of the FULL array, plus the valid fraction."""
    depth = np.asarray(depth_m, dtype=np.float32)
    valid = _valid_mask(depth)
    values = depth[valid]
    if values.size == 0:
        return {"min": None, "median": None, "max": None, "valid_fraction": 0.0}
    return {
        "min": float(values.min()),
        "median": float(np.median(values)),
        "max": float(values.max()),
        "valid_fraction": float(values.size / depth.size),
    }


def depth_grid(depth_m, rows: int, cols: int) -> np.ndarray:
    """Median-pool `depth_m` to a `(rows, cols)` grid; an all-invalid cell is NaN.

    Block edges are integers from `np.linspace`, so any input shape works (a
    576x800 map pools to 24x32 cells of exactly 24x25 pixels).
    """
    depth = np.asarray(depth_m, dtype=np.float32)
    valid = _valid_mask(depth)
    height, width = depth.shape[:2]
    row_edges = np.linspace(0, height, rows + 1).round().astype(int)
    col_edges = np.linspace(0, width, cols + 1).round().astype(int)
    grid = np.full((rows, cols), np.nan, dtype=np.float64)
    for i in range(rows):
        r0, r1 = row_edges[i], row_edges[i + 1]
        for j in range(cols):
            c0, c1 = col_edges[j], col_edges[j + 1]
            values = depth[r0:r1, c0:c1][valid[r0:r1, c0:c1]]
            if values.size:
                grid[i, j] = float(np.median(values))
    return grid


def _fmt(value, decimals: int) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.{decimals}f}"


def depth_grid_text(depth_m, rows: int, cols: int, decimals: int) -> str:
    """Header line, a stats line over the full array, then `rows` lines `rNN: v v ...`."""
    depth = np.asarray(depth_m, dtype=np.float32)
    valid = _valid_mask(depth)
    values = depth[valid]
    if values.size:
        lo, mid, hi = values.min(), np.median(values), values.max()
    else:
        lo = mid = hi = None
    pct = 100.0 * values.size / depth.size if depth.size else 0.0
    lines = [
        f"DEPTH GRID (metres, median-pooled to {rows} rows x {cols} columns; row 1 is the top of "
        "the depth image, column 1 is the left; n/a means no valid depth in that cell)",
        f"min={_fmt(lo, decimals)} median={_fmt(mid, decimals)} max={_fmt(hi, decimals)} valid={pct:.1f}%",
    ]
    grid = depth_grid(depth, rows, cols)
    for i in range(rows):
        lines.append(f"r{i + 1:02d}: " + " ".join(_fmt(v, decimals) for v in grid[i]))
    return "\n".join(lines)


def heatmap_legend_labels(range_m, ticks: int = LEGEND_TICKS) -> list[str]:
    """The legend tick labels, evenly spaced over `range_m` (metres)."""
    near, far = float(range_m[0]), float(range_m[1])
    return [f"{value:.2f} m" for value in np.linspace(near, far, ticks)]


def _legend_strip(height: int, near: float, far: float) -> np.ndarray:
    """Vertical turbo colour bar (near at the top) with labelled ticks and an n/a swatch."""
    strip = np.full((height, LEGEND_WIDTH, 3), _LEGEND_BG, dtype=np.uint8)
    top, bottom = 10, max(height - 34, 12)
    bar_h = bottom - top
    ramp = np.linspace(0, 255, bar_h).round().astype(np.uint8).reshape(-1, 1)
    strip[top:bottom, 8:26] = cv2.applyColorMap(ramp, cv2.COLORMAP_TURBO)
    labels = heatmap_legend_labels((near, far))
    for k, label in enumerate(labels):
        y = top + round(k * (bar_h - 1) / (len(labels) - 1))
        cv2.line(strip, (26, y), (31, y), _WHITE, 1)
        cv2.putText(strip, label, (34, y + 4), _LEGEND_FONT, 0.38, _WHITE, 1, cv2.LINE_AA)
    swatch_y = height - 24
    cv2.rectangle(strip, (8, swatch_y), (25, swatch_y + 14), (0, 0, 0), -1)
    cv2.rectangle(strip, (8, swatch_y), (25, swatch_y + 14), _WHITE, 1)
    cv2.putText(strip, "n/a", (34, swatch_y + 11), _LEGEND_FONT, 0.38, _WHITE, 1, cv2.LINE_AA)
    return strip


def depth_heatmap(depth_m, range_m, legend: bool = True, max_side: int | None = None) -> np.ndarray:
    """Turbo heatmap (blue near, red far) clipped to `range_m`, invalid pixels black.

    With `max_side`, the map is shrunk with INTER_NEAREST (so black cells never
    blend into colours) to fit `max_side` including the legend strip.
    """
    near, far = float(range_m[0]), float(range_m[1])
    depth = np.asarray(depth_m, dtype=np.float32)
    valid = _valid_mask(depth)
    clipped = np.clip(np.where(valid, depth, near), near, far)
    scaled = np.round((clipped - near) / (far - near) * 255.0).astype(np.uint8)
    body = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
    body[~valid] = 0
    if max_side is not None:
        height, width = body.shape[:2]
        avail_w = max(1, max_side - (LEGEND_WIDTH if legend else 0))
        scale = min(1.0, avail_w / width, max_side / height)
        if scale < 1.0:
            size = (max(1, round(width * scale)), max(1, round(height * scale)))
            body = cv2.resize(body, size, interpolation=cv2.INTER_NEAREST)
    if legend:
        body = np.hstack([body, _legend_strip(body.shape[0], near, far)])
    return body


# --- Depth request (D-06) ------------------------------------------------------------------


def fetch_depth(depth_client, left_raw, right_raw, retry_429: int, retry_sleep_s: float,
                sleep_fn=time.sleep, clock=time.monotonic, request_id: str | None = None) -> dict:
    """One `compute_depth_result()` call, retried only on HTTP 429 (depth service busy).

    Never raises: an unexpected exception becomes a failure record. Returns the
    result dict plus `attempts` and `request_ms` (total, including sleeps).
    """
    start = clock()
    attempts = 0
    while True:
        attempts += 1
        try:
            result = dict(depth_client.compute_depth_result(left_raw, right_raw, request_id=request_id))
        except Exception as exc:  # last resort; the real client never raises
            result = {"depth": None, "error": f"{type(exc).__name__}: {exc}"}
        if result.get("depth") is not None:
            break
        error = str(result.get("error") or "depth request failed")
        result["error"] = error
        if _HTTP_429.search(error) and attempts <= retry_429:
            sleep_fn(retry_sleep_s)
            continue
        break
    result["attempts"] = attempts
    result["request_ms"] = (clock() - start) * 1000.0
    return result


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(array), allow_pickle=False)
    return buffer.getvalue()


def _plain_float(value):
    if isinstance(value, bool) or value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


# --- Collector -----------------------------------------------------------------------------


class MllmInputCollector:
    """`collect(joint_state) -> MllmInputs | None` for MllmActionSourceBase.

    Owns the wrist capture (released by `close()`); the stereo camera is
    released by the caller that opened it.
    """

    def __init__(self, wrist_cap, stereo_camera, depth_client, settings: dict, *,
                 clock=time.monotonic, sleep_fn=time.sleep, flush_grabs: int = DEFAULT_FLUSH_GRABS):
        self.wrist_cap = wrist_cap
        self.stereo_camera = stereo_camera
        self.depth_client = depth_client
        self.settings = settings
        self.clock = clock
        self.sleep_fn = sleep_fn
        self.flush_grabs = flush_grabs
        self._depth_requests = 0

    def collect(self, joint_state):
        # Imported here, not at module top: importing vla_bridge.backends runs
        # backend discovery, which imports the MLLM builder, which imports this
        # module -- a top-level import would be circular.
        from vla_bridge.backends.base_mllm import MllmInputs

        captured = capture_mllm_inputs(self.wrist_cap, self.stereo_camera, self.flush_grabs)
        if captured is None:
            return None  # no depth request on a failed capture

        settings = self.settings
        depth_cfg = settings["depth"]
        presentation = settings["depth_presentation"]
        requested = settings["left_image_source"]
        image_format = settings["image_format"]
        max_side = int(settings["image_max_side"])
        near, far = (float(v) for v in depth_cfg["range_m"])
        rows, cols = int(depth_cfg["grid_rows"]), int(depth_cfg["grid_cols"])
        decimals = int(depth_cfg["grid_decimals"])

        result = None
        depth_fetch_ms = None
        request_id = None
        if self.depth_client is not None:
            self._depth_requests += 1
            request_id = f"mllm-depth-{self._depth_requests:06d}"
            result = fetch_depth(
                self.depth_client, captured["left"], captured["right"],
                retry_429=int(depth_cfg["retry_429"]),
                retry_sleep_s=float(depth_cfg["retry_sleep_s"]),
                sleep_fn=self.sleep_fn, clock=self.clock, request_id=request_id,
            )
            depth_fetch_ms = float(result["request_ms"])
        depth_m = result.get("depth") if result is not None else None
        depth_ok = depth_m is not None
        aligned_left = result.get("aligned_left") if depth_ok else None

        t_encode = self.clock()
        if requested == "aligned" and aligned_left is not None:
            left_frame, left_sent = aligned_left, "aligned"
        else:
            left_frame, left_sent = captured["left"], "raw"

        images: dict[str, bytes] = {}
        image_shapes = {}
        media_type = "image/png"
        for role, frame in (("wrist", captured["wrist"]), ("left", left_frame), ("right", captured["right"])):
            small = _downsize_for_transport(frame, max_side)
            data, media_type = encode_image(small, image_format)
            if data is None:
                return None
            images[role] = data
            image_shapes[role] = [int(small.shape[0]), int(small.shape[1])]

        sidecar_files: dict[str, bytes] = {}
        depth_note = DEPTH_UNAVAILABLE_NOTE
        if depth_ok:
            heatmap = depth_heatmap(depth_m, (near, far), legend=True, max_side=max_side)
            grid_text = depth_grid_text(depth_m, rows, cols, decimals)
            if presentation in ("heatmap", "both"):
                data, _ = encode_image(heatmap, image_format)
                if data is not None:
                    images["depth"] = data
                    image_shapes["depth"] = [int(heatmap.shape[0]), int(heatmap.shape[1])]
            sidecar_files["depth.npy"] = _npy_bytes(depth_m)
            heatmap_png, _ = encode_image(heatmap, "png")
            if heatmap_png is not None:
                sidecar_files["depth_heatmap.png"] = heatmap_png
            sidecar_files["depth_grid.txt"] = grid_text.encode("utf-8")
            if aligned_left is not None:
                aligned_png, _ = encode_image(aligned_left, "png")
                if aligned_png is not None:
                    sidecar_files["aligned_left.png"] = aligned_png
            depth_note = self._depth_note(presentation, grid_text, "depth" in images)
        image_encode_ms = (self.clock() - t_encode) * 1000.0

        meta = {
            "timings_ms": {"depth_fetch": depth_fetch_ms, "image_encode": float(image_encode_ms)},
            "depth": self._depth_meta(result, depth_m, presentation, rows, cols, near, far, request_id),
            "left_image_source_requested": requested,
            "left_image_source_sent": left_sent,
            "captured_utc": captured["captured_utc"],
            "stereo_pair_metadata": captured["stereo_pair_metadata"],
            "image_shapes": image_shapes,
        }
        return MllmInputs(
            images=images,
            depth_note=depth_note,
            meta=meta,
            sidecar_files=sidecar_files,
            image_media_type=media_type,
        )

    @staticmethod
    def _depth_note(presentation: str, grid_text: str, heatmap_sent: bool) -> str:
        if presentation == "grid":
            return grid_text
        if presentation == "both":
            return grid_text + ("\n" + DEPTH_BOTH_HEATMAP_LINE if heatmap_sent else "")
        return DEPTH_HEATMAP_NOTE if heatmap_sent else DEPTH_UNAVAILABLE_NOTE

    @staticmethod
    def _depth_meta(result, depth_m, presentation, rows, cols, near, far, request_id) -> dict:
        meta = {
            "status": "not_configured",
            "valid_fraction": None,
            "request_ms": None,
            "calibration_sha256": None,
            "presentation": presentation,
            "grid_shape": [rows, cols],
            "colour_range_m": [near, far],
            "attempts": 0,
            "error": None,
            "model": None,
            "inference_ms": None,
            "request_id": request_id,
        }
        if result is None:
            return meta
        sha = result.get("calibration_sha256")
        model = result.get("model")
        meta.update(
            request_ms=_plain_float(result.get("request_ms")),
            calibration_sha256=str(sha) if sha is not None else None,
            attempts=int(result.get("attempts", 1)),
            model=str(model) if model is not None else None,
            inference_ms=_plain_float(result.get("inference_ms")),
        )
        if depth_m is None:
            meta["status"] = "failed"
            meta["error"] = str(result.get("error") or "depth request failed")[:MAX_ERROR_CHARS]
        else:
            meta["status"] = "ok"
            valid_fraction = _plain_float(result.get("valid_fraction"))
            meta["valid_fraction"] = (valid_fraction if valid_fraction is not None
                                      else depth_stats(depth_m)["valid_fraction"])
        return meta

    def close(self) -> None:
        release = getattr(self.wrist_cap, "release", None)
        if callable(release):
            release()


# --- Preview CLI ---------------------------------------------------------------------------


def _save(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    print(f"Saved → {path}")


def main(argv=None) -> int:
    """Capture one live set of MLLM inputs and write them under --out (no model call)."""
    parser = argparse.ArgumentParser(
        description="Preview exactly what an MLLM backend would be shown for one chunk call "
        "(camera images, depth heatmap and grid) without calling any model."
    )
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    parser.add_argument("--stereo-camera-index", type=str, default="1",
                        help="AVFoundation device NAME (preferred) or index of the AR0144 stereo camera")
    parser.add_argument("--wrist-camera-index", type=int, required=True, help="cv2 index of the wrist camera")
    parser.add_argument("--calibration", type=Path, default=None,
                        help="Saved StereoCalibration JSON (give together with --endpoint)")
    parser.add_argument("--endpoint", type=str, default=None,
                        help="FastFS depth endpoint URL (give together with --calibration)")
    parser.add_argument("--config", type=str, default=None,
                        help="Optional backend config name overriding mllm_defaults.json")
    parser.add_argument("--left-image-source", choices=("raw", "aligned"), default=None)
    parser.add_argument("--depth-presentation", choices=("heatmap", "grid", "both"), default=None)
    args = parser.parse_args(argv)
    if (args.calibration is None) != (args.endpoint is None):
        parser.error("--calibration and --endpoint must be given together (or both omitted)")

    from vla_bridge.backends.base_mllm import load_mllm_settings
    from vla_bridge.depth_camera import DepthCameraClient
    from vla_bridge.stereo_calibration import load_calibration
    from vla_bridge.stereo_camera import StereoSplitCamera

    settings = load_mllm_settings(args.config)
    if args.left_image_source is not None:
        settings["left_image_source"] = args.left_image_source
    if args.depth_presentation is not None:
        settings["depth_presentation"] = args.depth_presentation

    depth_client = None
    if args.endpoint is not None:
        depth_client = DepthCameraClient(load_calibration(args.calibration), args.endpoint)
        try:
            health = depth_client.check_ready()
        except Exception as exc:
            raise SystemExit(f"Depth service is not ready: {type(exc).__name__}: {exc}") from exc
        print(f"Depth service ready (model: {health.get('model', '?')})")

    stereo_camera = StereoSplitCamera(index=args.stereo_camera_index)
    wrist_cap = cv2.VideoCapture(args.wrist_camera_index)
    if hasattr(wrist_cap, "isOpened") and not wrist_cap.isOpened():
        print(f"WARNING: wrist camera index {args.wrist_camera_index} did not open.")
    collector = MllmInputCollector(wrist_cap, stereo_camera, depth_client, settings)
    try:
        inputs = collector.collect({})
    finally:
        collector.close()
        stereo_camera.release()
    if inputs is None:
        print("Camera read failed; nothing saved.")
        return 1

    out = args.out
    ext = ".jpg" if inputs.image_media_type == "image/jpeg" else ".png"
    for role in IMAGE_ROLES:
        if role in inputs.images:
            _save(out / "images" / f"{role}{ext}", inputs.images[role])
    for name, data in inputs.sidecar_files.items():
        _save(out / name, data)
    _save(out / "depth_note.txt", inputs.depth_note.encode("utf-8"))
    _save(out / "meta.json", json.dumps(inputs.meta, indent=2, default=str).encode("utf-8"))

    depth = inputs.meta["depth"]
    fetch_ms = inputs.meta["timings_ms"]["depth_fetch"]
    fetch_text = f"{fetch_ms:.0f} ms" if fetch_ms is not None else "n/a"
    print(f"Depth: status={depth['status']} fetch={fetch_text} attempts={depth['attempts']} "
          f"valid_fraction={depth['valid_fraction']}")
    print(f"Left image sent: {inputs.meta['left_image_source_sent']} "
          f"(requested {inputs.meta['left_image_source_requested']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEPTH_UNAVAILABLE_NOTE",
    "IMAGE_ROLES",
    "LEGEND_WIDTH",
    "MllmInputCollector",
    "capture_mllm_inputs",
    "depth_grid",
    "depth_grid_text",
    "depth_heatmap",
    "depth_stats",
    "encode_image",
    "fetch_depth",
    "heatmap_legend_labels",
    "main",
]
