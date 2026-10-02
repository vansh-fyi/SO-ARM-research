"""Tests for the MLLM input collector (Phase 15, PIPE-02).

`vla_bridge.mllm_inputs`: the per-chunk capture of wrist, raw left and raw
right frames plus one depth request (D-05, D-06), the depth heatmap and the
24x32 median-pooled grid (D-28), the left_image_source toggle (D-26), depth
failure handling (HTTP 429 retries, URL never in the prompt, T-15-14/T-15-15),
and the keyless preview CLI. Every camera and the depth service are fakes from
`fake_providers.py`; nothing here touches hardware or the network.
"""

import ast
import copy
import json
import re
from pathlib import Path

import cv2
import numpy as np
import pytest

from fake_providers import FakeDepthClient, FakeStereoCamera, FakeWristCap
from vla_bridge import mllm_inputs
from vla_bridge.backends.base_mllm import IMAGE_ORDER, MllmInputs, load_mllm_settings
from vla_bridge.mllm_inputs import (
    DEPTH_UNAVAILABLE_NOTE,
    IMAGE_ROLES,
    MllmInputCollector,
    capture_mllm_inputs,
    depth_grid,
    depth_grid_text,
    depth_heatmap,
    depth_stats,
    encode_image,
    fetch_depth,
    heatmap_legend_labels,
)
from vla_bridge.mllm_prompt import lint_prompt_text

RAW_SHAPE = (600, 800, 3)
DEPTH_SHAPE = (576, 800)
SECRET_URL = "https://secret-rotating-host.trycloudflare.com/depth"
ERROR_429 = f"HTTPError: 429 Client Error: TOO MANY REQUESTS for url: {SECRET_URL}"
ERROR_500 = f"HTTPError: 500 Server Error: INTERNAL SERVER ERROR for url: {SECRET_URL}"
GRID_ROW = re.compile(r"^r\d\d: ", re.MULTILINE)


class FakeClock:
    """Monotonic clock advancing 0.25 s on every read."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        self.t += 0.25
        return self.t


class RecordingStereo(FakeStereoCamera):
    """Keeps the exact arrays it returned so identity can be asserted."""

    def read_left(self):
        self.left_out = super().read_left()
        return self.left_out

    def read_right(self):
        self.right_out = super().read_right()
        return self.right_out


def _settings(depth=None, **overrides):
    settings = copy.deepcopy(load_mllm_settings())
    settings.update(overrides)
    if depth:
        settings["depth"].update(depth)
    return settings


def _collector(settings=None, depth_client="default", wrist_ok=True, stereo_ok=True,
               calls=None, sleeps=None, stereo_cls=FakeStereoCamera):
    calls = [] if calls is None else calls
    wrist = FakeWristCap(ok=wrist_ok, shape=RAW_SHAPE, value=7, calls=calls)
    stereo = stereo_cls(ok=stereo_ok, shape=RAW_SHAPE, calls=calls)
    if depth_client == "default":
        depth_client = FakeDepthClient(calls=calls)
    sleep_log = [] if sleeps is None else sleeps
    collector = MllmInputCollector(
        wrist, stereo, depth_client, settings or _settings(),
        clock=FakeClock(), sleep_fn=sleep_log.append,
    )
    return collector, wrist, stereo, depth_client


def _decode(data: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    assert image is not None
    return image


def _block_depth(consts: np.ndarray) -> np.ndarray:
    """576x800 depth made of constant 24x25 blocks (one per grid cell)."""
    return np.repeat(np.repeat(consts, 24, axis=0), 25, axis=1).astype(np.float32)


def _grid_consts() -> np.ndarray:
    return (0.3 + 0.001 * np.arange(24 * 32)).reshape(24, 32)


# --- Contract -------------------------------------------------------------------------------


def test_image_roles_match_base_image_order():
    assert IMAGE_ROLES == IMAGE_ORDER


def test_mllm_inputs_import_is_lazy_to_avoid_circular_backend_discovery():
    """MllmInputs must be imported inside collect(), never at module top level."""
    tree = ast.parse(Path(mllm_inputs.__file__).read_text())
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            module = getattr(node, "module", "") or ""
            names = [alias.name for alias in node.names]
            assert "backends" not in module, f"top-level import of {module}"
            assert not any("backends" in name for name in names)


# --- Grid pooling (D-28) --------------------------------------------------------------------


def test_grid_pools_constant_blocks_to_24x32():
    consts = _grid_consts()
    grid = depth_grid(_block_depth(consts), 24, 32)
    assert grid.shape == (24, 32)
    np.testing.assert_allclose(grid, consts, rtol=1e-6)


def test_grid_fully_invalid_block_is_nan_and_na_text():
    depth = _block_depth(_grid_consts())
    depth[0:24, 0:25] = np.nan
    grid = depth_grid(depth, 24, 32)
    assert np.isnan(grid[0, 0])
    assert np.isfinite(grid[0, 1])
    text = depth_grid_text(depth, 24, 32, 2)
    first_row = [line for line in text.splitlines() if line.startswith("r01:")][0]
    assert first_row.split()[1] == "n/a"


def test_grid_half_invalid_block_uses_median_of_valid_pixels():
    depth = np.full(DEPTH_SHAPE, 1.0, dtype=np.float32)
    depth[0:12, 0:25] = np.nan
    values = np.linspace(0.4, 0.9, 12 * 25, dtype=np.float32).reshape(12, 25)
    depth[12:24, 0:25] = values
    grid = depth_grid(depth, 24, 32)
    assert grid[0, 0] == pytest.approx(float(np.median(values)), rel=1e-6)


def test_grid_zero_negative_and_infinite_values_are_invalid():
    depth = np.full(DEPTH_SHAPE, 1.0, dtype=np.float32)
    depth[24:48, 25:50] = 0.0
    depth[24:36, 25:50] = -1.0
    depth[24:30, 25:50] = np.inf
    depth[48:60, 50:75] = 0.0
    depth[60:72, 50:75] = 0.9
    grid = depth_grid(depth, 24, 32)
    assert np.isnan(grid[1, 1])
    assert grid[2, 2] == pytest.approx(0.9)


def test_grid_non_divisible_shape_still_24x32():
    depth = np.random.default_rng(0).uniform(0.3, 1.2, (100, 130)).astype(np.float32)
    assert depth_grid(depth, 24, 32).shape == (24, 32)
    tiny = np.full((10, 12), 0.5, dtype=np.float32)
    grid = depth_grid(tiny, 24, 32)
    assert grid.shape == (24, 32)
    text = depth_grid_text(tiny, 24, 32, 2)
    assert len(GRID_ROW.findall(text)) == 24


def test_grid_text_header_stats_and_24_rows():
    depth = _block_depth(_grid_consts())
    depth[0:24, 0:25] = np.nan
    text = depth_grid_text(depth, 24, 32, 2)
    lines = text.splitlines()
    header = lines[0]
    assert header.startswith("DEPTH GRID")
    for phrase in ("metres", "median", "24 rows", "32 columns", "row 1 is the top", "column 1 is the left", "n/a"):
        assert phrase in header
    stats = lines[1]
    valid = depth[np.isfinite(depth) & (depth > 0)]
    assert f"min={valid.min():.2f}" in stats
    assert f"median={float(np.median(valid)):.2f}" in stats
    assert f"max={valid.max():.2f}" in stats
    assert f"valid={100.0 * valid.size / depth.size:.1f}%" in stats
    rows = [line for line in lines if GRID_ROW.match(line)]
    assert len(rows) == 24
    assert [row[:3] for row in rows] == [f"r{i:02d}" for i in range(1, 25)]
    for row in rows:
        tokens = row.split()[1:]
        assert len(tokens) == 32
        assert all(tok == "n/a" or re.fullmatch(r"\d+\.\d{2}", tok) for tok in tokens)


def test_grid_text_decimals_follow_setting():
    text = depth_grid_text(_block_depth(_grid_consts()), 24, 32, 3)
    row = [line for line in text.splitlines() if line.startswith("r02:")][0]
    assert all(re.fullmatch(r"\d+\.\d{3}", tok) for tok in row.split()[1:])


def test_grid_text_full_valid_grid_under_6000_chars():
    text = depth_grid_text(_block_depth(_grid_consts()), 24, 32, 2)
    assert len(text) < 6000


def test_grid_stats_over_full_array_and_all_invalid():
    depth = np.full((10, 10), np.nan, dtype=np.float32)
    depth[0, 0:4] = [0.5, 0.7, 0.9, 1.1]
    stats = depth_stats(depth)
    assert stats["min"] == pytest.approx(0.5)
    assert stats["max"] == pytest.approx(1.1)
    assert stats["median"] == pytest.approx(0.8)
    assert stats["valid_fraction"] == pytest.approx(0.04)
    empty = depth_stats(np.zeros((4, 4), dtype=np.float32))
    assert empty == {"min": None, "median": None, "max": None, "valid_fraction": 0.0}
    text = depth_grid_text(np.zeros((48, 64), dtype=np.float32), 24, 32, 2)
    assert "min=n/a" in text and "valid=0.0%" in text


# --- Heatmap (D-28) ---------------------------------------------------------------------------


def test_heatmap_nan_pixels_black_and_legend_widens():
    depth = np.full(DEPTH_SHAPE, 1.0, dtype=np.float32)
    depth[0:40, 0:40] = np.nan
    plain = depth_heatmap(depth, (0.2, 1.5), legend=False)
    with_legend = depth_heatmap(depth, (0.2, 1.5), legend=True)
    assert plain.shape == (576, 800, 3)
    assert with_legend.shape[0] == 576 and with_legend.shape[1] > 800
    assert with_legend[10, 10].tolist() == [0, 0, 0]
    assert with_legend[300, 300].tolist() != [0, 0, 0]


def test_heatmap_near_and_far_get_different_colours_and_values_are_clipped():
    near, far = 0.2, 1.5
    depth = np.full((100, 200), near, dtype=np.float32)
    depth[:, 100:] = far
    hm = depth_heatmap(depth, (near, far), legend=False)
    assert hm[50, 10].tolist() != hm[50, 150].tolist()
    # Turbo: blue channel dominates near, red channel dominates far (BGR order).
    assert hm[50, 10, 0] > hm[50, 10, 2]
    assert hm[50, 150, 2] > hm[50, 150, 0]
    clipped = np.full((100, 200), 0.05, dtype=np.float32)
    clipped[:, 100:] = 9.0
    hm_clipped = depth_heatmap(clipped, (near, far), legend=False)
    np.testing.assert_array_equal(hm_clipped, hm)


def test_heatmap_legend_labels_come_from_range_m():
    labels = heatmap_legend_labels((0.2, 1.5))
    assert len(labels) == 5
    assert labels[0] == "0.20 m" and labels[-1] == "1.50 m"
    other = heatmap_legend_labels((0.5, 2.5))
    assert other[0] == "0.50 m" and other[2] == "1.50 m" and other[-1] == "2.50 m"


def test_heatmap_respects_max_side_with_nearest_resize():
    depth = np.full(DEPTH_SHAPE, 1.0, dtype=np.float32)
    depth[0:200, 0:200] = np.nan
    hm = depth_heatmap(depth, (0.2, 1.5), legend=True, max_side=640)
    assert max(hm.shape[:2]) <= 640
    assert hm[5, 5].tolist() == [0, 0, 0]


# --- Encoding --------------------------------------------------------------------------------


def test_encode_image_png_and_jpeg():
    frame = np.full((40, 60, 3), 120, dtype=np.uint8)
    png, png_type = encode_image(frame, "png")
    jpg, jpg_type = encode_image(frame, "jpeg")
    assert png_type == "image/png" and png.startswith(b"\x89PNG")
    assert jpg_type == "image/jpeg" and jpg[:2] == b"\xff\xd8"


def test_image_format_jpeg_gives_jpeg_media_type_and_bytes():
    collector, *_ = _collector(_settings(image_format="jpeg"))
    inputs = collector.collect({})
    assert inputs.image_media_type == "image/jpeg"
    assert set(inputs.images) == {"wrist", "left", "right", "depth"}
    for data in inputs.images.values():
        assert data[:2] == b"\xff\xd8"
    assert inputs.sidecar_files["depth_heatmap.png"].startswith(b"\x89PNG")


# --- Presentation modes (D-28) ---------------------------------------------------------------


def test_presentation_heatmap_sends_image_and_no_grid_rows():
    collector, *_ = _collector(_settings(depth_presentation="heatmap"))
    inputs = collector.collect({})
    assert isinstance(inputs, MllmInputs)
    assert list(inputs.images) == ["wrist", "left", "right", "depth"]
    assert "heatmap" in inputs.depth_note
    assert "DEPTH GRID" not in inputs.depth_note
    assert not GRID_ROW.search(inputs.depth_note)


def test_presentation_grid_sends_no_depth_image_and_grid_note():
    collector, *_ = _collector(_settings(depth_presentation="grid"))
    inputs = collector.collect({})
    assert list(inputs.images) == ["wrist", "left", "right"]
    assert "DEPTH GRID" in inputs.depth_note
    assert "min=" in inputs.depth_note
    assert len(GRID_ROW.findall(inputs.depth_note)) == 24
    assert "heatmap" not in inputs.depth_note


def test_presentation_both_sends_image_grid_and_heatmap_line():
    collector, *_ = _collector(_settings(depth_presentation="both"))
    inputs = collector.collect({})
    assert list(inputs.images) == ["wrist", "left", "right", "depth"]
    assert len(GRID_ROW.findall(inputs.depth_note)) == 24
    heatmap_lines = [line for line in inputs.depth_note.splitlines() if "heatmap" in line]
    assert len(heatmap_lines) == 1


@pytest.mark.parametrize("presentation", ["heatmap", "grid", "both"])
def test_presentation_saves_all_sidecars_from_one_depth_call(presentation):
    collector, _, _, client = _collector(_settings(depth_presentation=presentation))
    inputs = collector.collect({})
    assert client.calls == 1
    assert set(inputs.sidecar_files) == {"depth.npy", "depth_heatmap.png", "depth_grid.txt", "aligned_left.png"}
    import io
    saved = np.load(io.BytesIO(inputs.sidecar_files["depth.npy"]), allow_pickle=False)
    np.testing.assert_array_equal(saved, client.last_depth)
    grid_text = inputs.sidecar_files["depth_grid.txt"].decode("utf-8")
    assert len(GRID_ROW.findall(grid_text)) == 24
    assert _decode(inputs.sidecar_files["aligned_left.png"]).shape == (576, 800, 3)
    assert inputs.meta["depth"]["presentation"] == presentation
    assert inputs.meta["depth"]["grid_shape"] == [24, 32]
    assert inputs.meta["depth"]["colour_range_m"] == [0.2, 1.5]
    collector.collect({})
    assert client.calls == 2


def test_presentation_depth_image_is_the_saved_heatmap():
    collector, *_ = _collector(_settings(depth_presentation="both"))
    inputs = collector.collect({})
    sent = _decode(inputs.images["depth"])
    saved = _decode(inputs.sidecar_files["depth_heatmap.png"])
    np.testing.assert_array_equal(sent, saved)
    assert max(sent.shape[:2]) <= 640


# --- Raw frames and read order (Pitfalls 5 and 8) -------------------------------------------


def test_raw_frames_unmodified_to_depth_and_downsized_to_prompt():
    collector, _, stereo, client = _collector(stereo_cls=RecordingStereo)
    inputs = collector.collect({})
    left_sent, right_sent, _request_id = client.received[0]
    assert left_sent is stereo.left_out
    assert right_sent is stereo.right_out
    assert left_sent.shape == RAW_SHAPE
    for role in ("wrist", "left", "right"):
        assert _decode(inputs.images[role]).shape == (480, 640, 3)


def test_raw_frames_read_order_wrist_flush_then_one_stereo_pair():
    calls = []
    collector, *_ = _collector(calls=calls)
    collector.collect({})
    assert calls == ["wrist.grab"] * 3 + ["wrist.read", "stereo.read_left", "stereo.read_right", "depth.compute"]


def test_raw_frames_capture_returns_arrays_and_metadata():
    calls = []
    wrist = FakeWristCap(shape=RAW_SHAPE, calls=calls)
    stereo = FakeStereoCamera(shape=RAW_SHAPE, calls=calls)
    captured = capture_mllm_inputs(wrist, stereo, flush_grabs=2)
    assert calls == ["wrist.grab", "wrist.grab", "wrist.read", "stereo.read_left", "stereo.read_right"]
    assert captured["wrist"].shape == RAW_SHAPE
    assert captured["left"][0, 0, 0] == 11 and captured["right"][0, 0, 0] == 22
    assert captured["stereo_pair_metadata"] == {"pair_id": 1}
    assert isinstance(captured["captured_utc"], str)


# --- left_image_source (D-26) ----------------------------------------------------------------


def test_left_image_source_raw_sends_downsized_raw_left():
    collector, *_ = _collector(_settings(left_image_source="raw"))
    inputs = collector.collect({})
    left = _decode(inputs.images["left"])
    assert left.shape == (480, 640, 3) and int(left[0, 0, 0]) == 11
    assert inputs.meta["left_image_source_requested"] == "raw"
    assert inputs.meta["left_image_source_sent"] == "raw"
    assert "aligned_left.png" in inputs.sidecar_files


def test_left_image_source_aligned_sends_aligned_left_when_depth_ok():
    collector, *_ = _collector(_settings(left_image_source="aligned"))
    inputs = collector.collect({})
    left = _decode(inputs.images["left"])
    assert int(left[0, 0, 0]) == FakeDepthClient.ALIGNED_VALUE
    assert left.shape == (461, 640, 3)
    assert inputs.meta["left_image_source_sent"] == "aligned"


def test_left_image_source_aligned_falls_back_to_raw_when_depth_failed():
    client = FakeDepthClient(results=[ERROR_500])
    collector, *_ = _collector(_settings(left_image_source="aligned"), depth_client=client)
    inputs = collector.collect({})
    assert int(_decode(inputs.images["left"])[0, 0, 0]) == 11
    assert inputs.meta["left_image_source_requested"] == "aligned"
    assert inputs.meta["left_image_source_sent"] == "raw"
    assert "aligned_left.png" not in inputs.sidecar_files


# --- Failures --------------------------------------------------------------------------------


def test_capture_failure_wrist_returns_none_without_depth_call():
    collector, _, _, client = _collector(wrist_ok=False)
    assert collector.collect({}) is None
    assert client.calls == 0


def test_capture_failure_stereo_none_returns_none_without_depth_call():
    collector, _, _, client = _collector(stereo_ok=False)
    assert collector.collect({}) is None
    assert client.calls == 0


def test_depth_failure_note_hides_url_and_error_text():
    client = FakeDepthClient(results=[ERROR_500])
    collector, *_ = _collector(depth_client=client)
    inputs = collector.collect({})
    assert inputs.depth_note == DEPTH_UNAVAILABLE_NOTE == "Depth unavailable this call."
    assert "depth" not in inputs.images
    assert list(inputs.images) == ["wrist", "left", "right"]
    assert SECRET_URL not in inputs.depth_note
    assert inputs.sidecar_files == {}
    depth = inputs.meta["depth"]
    assert depth["status"] == "failed"
    assert "500" in depth["error"]
    assert depth["attempts"] == 1
    assert client.calls == 1


def test_depth_failure_error_truncated_to_300_chars():
    client = FakeDepthClient(results=["ValueError: " + "x" * 1000])
    collector, *_ = _collector(depth_client=client)
    inputs = collector.collect({})
    assert len(inputs.meta["depth"]["error"]) == 300


def test_429_retried_up_to_retry_429_with_sleep_between():
    client = FakeDepthClient(results=[ERROR_429] * 3)
    sleeps = []
    settings = _settings(depth={"retry_429": 2, "retry_sleep_s": 2.0})
    collector, *_ = _collector(settings, depth_client=client, sleeps=sleeps)
    inputs = collector.collect({})
    assert client.calls == 3
    assert sleeps == [2.0, 2.0]
    assert inputs.meta["depth"]["status"] == "failed"
    assert inputs.meta["depth"]["attempts"] == 3
    assert inputs.depth_note == DEPTH_UNAVAILABLE_NOTE


def test_429_then_success_records_attempts():
    client = FakeDepthClient(results=[ERROR_429, None])
    sleeps = []
    collector, *_ = _collector(_settings(depth={"retry_sleep_s": 0.5}), depth_client=client, sleeps=sleeps)
    inputs = collector.collect({})
    assert client.calls == 2
    assert sleeps == [0.5]
    assert inputs.meta["depth"]["status"] == "ok"
    assert inputs.meta["depth"]["attempts"] == 2
    assert "depth" in inputs.images


def test_429_retries_disabled_when_retry_429_is_zero():
    client = FakeDepthClient(results=[ERROR_429])
    sleeps = []
    result = fetch_depth(client, None, None, retry_429=0, retry_sleep_s=2.0,
                         sleep_fn=sleeps.append, clock=FakeClock())
    assert client.calls == 1 and sleeps == []
    assert result["depth"] is None and result["attempts"] == 1


def test_non_429_error_is_not_retried():
    client = FakeDepthClient(results=[ERROR_500])
    sleeps = []
    result = fetch_depth(client, None, None, retry_429=2, retry_sleep_s=2.0,
                         sleep_fn=sleeps.append, clock=FakeClock())
    assert client.calls == 1 and sleeps == []
    assert result["attempts"] == 1
    assert result["request_ms"] == pytest.approx(250.0)


def test_fetch_depth_exception_becomes_failed_result():
    client = FakeDepthClient(results=[RuntimeError("boom")])
    result = fetch_depth(client, None, None, retry_429=2, retry_sleep_s=2.0,
                         sleep_fn=lambda s: None, clock=FakeClock())
    assert result["depth"] is None
    assert result["error"] == "RuntimeError: boom"


def test_no_depth_client_is_not_configured():
    collector, *_ = _collector(depth_client=None)
    inputs = collector.collect({})
    assert list(inputs.images) == ["wrist", "left", "right"]
    assert inputs.depth_note == DEPTH_UNAVAILABLE_NOTE
    assert inputs.meta["depth"]["status"] == "not_configured"
    assert inputs.meta["timings_ms"]["depth_fetch"] is None
    assert isinstance(inputs.meta["timings_ms"]["image_encode"], float)
    assert inputs.sidecar_files == {}


# --- Meta, lint, lifecycle -------------------------------------------------------------------


def test_meta_contract_is_json_serialisable():
    collector, *_ = _collector()
    inputs = collector.collect({})
    meta = json.loads(json.dumps(inputs.meta))
    assert set(meta["timings_ms"]) == {"depth_fetch", "image_encode"}
    assert meta["timings_ms"]["depth_fetch"] > 0
    depth = meta["depth"]
    for key in ("status", "valid_fraction", "request_ms", "calibration_sha256", "presentation",
                "grid_shape", "colour_range_m", "attempts", "error", "model", "inference_ms"):
        assert key in depth
    assert depth["status"] == "ok" and depth["error"] is None
    assert depth["calibration_sha256"] == FakeDepthClient.CALIBRATION_SHA256
    for key in ("left_image_source_requested", "left_image_source_sent", "captured_utc",
                "stereo_pair_metadata"):
        assert key in meta


@pytest.mark.parametrize("presentation", ["heatmap", "grid", "both"])
def test_depth_note_and_legend_text_lint_clean(presentation):
    collector, *_ = _collector(_settings(depth_presentation=presentation))
    assert lint_prompt_text(collector.collect({}).depth_note) == []
    assert lint_prompt_text(DEPTH_UNAVAILABLE_NOTE) == []
    assert lint_prompt_text(" ".join(heatmap_legend_labels((0.2, 1.5)))) == []


def test_close_releases_wrist_capture():
    collector, wrist, stereo, _ = _collector()
    collector.close()
    assert wrist.released is True
    assert stereo.released is False  # the caller owns the stereo camera


# --- Preview CLI -----------------------------------------------------------------------------


def _patch_devices(monkeypatch, client):
    import vla_bridge.depth_camera as depth_camera
    import vla_bridge.stereo_calibration as stereo_calibration
    import vla_bridge.stereo_camera as stereo_camera

    opened = {}

    def fake_stereo(index=1, **kwargs):
        opened["stereo"] = FakeStereoCamera(shape=RAW_SHAPE)
        opened["stereo_index"] = index
        return opened["stereo"]

    def fake_capture(index, *args):
        opened["wrist"] = FakeWristCap(shape=RAW_SHAPE)
        opened["wrist_index"] = index
        return opened["wrist"]

    monkeypatch.setattr(stereo_camera, "StereoSplitCamera", fake_stereo)
    monkeypatch.setattr(cv2, "VideoCapture", fake_capture)
    monkeypatch.setattr(stereo_calibration, "load_calibration", lambda path: {"path": str(path)})
    monkeypatch.setattr(depth_camera, "DepthCameraClient", lambda calibration, endpoint: client)
    return opened


def test_preview_cli_main_writes_images_sidecars_note_and_meta(monkeypatch, tmp_path, capsys):
    client = FakeDepthClient()
    opened = _patch_devices(monkeypatch, client)
    out = tmp_path / "preview"
    code = mllm_inputs.main([
        "--out", str(out), "--wrist-camera-index", "0", "--stereo-camera-index", "CCB Camera",
        "--calibration", str(tmp_path / "stereo.json"), "--endpoint", "http://fake/depth",
    ])
    assert code == 0
    for role in ("wrist", "left", "right", "depth"):
        assert (out / "images" / f"{role}.png").is_file()
    for name in ("depth.npy", "depth_heatmap.png", "depth_grid.txt", "aligned_left.png",
                 "depth_note.txt", "meta.json"):
        assert (out / name).is_file()
    meta = json.loads((out / "meta.json").read_text())
    assert meta["depth"]["status"] == "ok"
    assert client.ready_checks == 1 and client.calls == 1
    assert opened["wrist_index"] == 0 and opened["stereo_index"] == "CCB Camera"
    assert opened["wrist"].released and opened["stereo"].released
    printed = capsys.readouterr().out
    saved_lines = [line for line in printed.splitlines() if line.startswith("Saved → ")]
    assert len(saved_lines) >= 10
    assert "Depth" in printed


def test_preview_cli_without_depth_and_with_overrides(monkeypatch, tmp_path, capsys):
    client = FakeDepthClient()
    _patch_devices(monkeypatch, client)
    out = tmp_path / "preview"
    code = mllm_inputs.main(["--out", str(out), "--wrist-camera-index", "2",
                             "--left-image-source", "aligned", "--depth-presentation", "grid"])
    assert code == 0
    assert client.calls == 0
    assert not (out / "images" / "depth.png").exists()
    meta = json.loads((out / "meta.json").read_text())
    assert meta["depth"]["status"] == "not_configured"
    assert meta["left_image_source_requested"] == "aligned"
    assert (out / "depth_note.txt").read_text() == DEPTH_UNAVAILABLE_NOTE


def test_preview_cli_requires_calibration_and_endpoint_together(monkeypatch, tmp_path):
    _patch_devices(monkeypatch, FakeDepthClient())
    with pytest.raises(SystemExit):
        mllm_inputs.main(["--out", str(tmp_path), "--wrist-camera-index", "0",
                          "--endpoint", "http://fake/depth"])
