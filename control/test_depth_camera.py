"""Tests for `control/vla_bridge/depth_camera.py` (Phase 12, Plan 12-06,
Task 1 -- DEPTH-CAL-02/03).

Uses a real `cv2` stereo-rectification pipeline (deterministic, no hardware)
plus a monkeypatched `requests.post` -- never a real Colab FastFS endpoint.
"""

import base64
import io

import numpy as np
import pytest

from vla_bridge import depth_camera, stereo_calibration
from vla_bridge.depth_camera import DepthCameraClient, compute_target_size, scale_intrinsics


def test_compute_target_size_exact_multiple_of_32_case():
    assert compute_target_size(orig_width=1280, orig_height=720, max_width=992) == (992, 544)


def test_compute_target_size_never_upscales_past_original_size():
    assert compute_target_size(orig_width=500, orig_height=300, max_width=992) == (480, 288)


def test_compute_target_size_clamps_to_minimum_one_block_size_for_tiny_input():
    assert compute_target_size(orig_width=20, orig_height=20) == (32, 32)


def test_scale_intrinsics_scales_fx_cx_by_scale_x_and_fy_cy_by_scale_y():
    intrinsics = [800.0, 0.0, 640.0, 0.0, 800.0, 360.0, 0.0, 0.0, 1.0]
    result = scale_intrinsics(intrinsics, scale_x=0.5, scale_y=0.6)
    assert result == [400.0, 0.0, 320.0, 0.0, 480.0, 216.0, 0.0, 0.0, 1.0]


def _make_calibration() -> "stereo_calibration.StereoCalibration":
    identity_K = [800.0, 0.0, 640.0, 0.0, 800.0, 360.0, 0.0, 0.0, 1.0]
    zero_D = [0.0, 0.0, 0.0, 0.0, 0.0]
    identity_R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    return stereo_calibration.StereoCalibration(
        image_size=(1280, 720),
        left_K=identity_K,
        left_D=zero_D,
        right_K=identity_K,
        right_D=zero_D,
        R=identity_R,
        T=[-0.12, 0.0, 0.0],
        reprojection_error_px=0.1,
        calibrated_at_utc="2026-01-01T00:00:00+00:00",
    )


def test_depth_camera_client_builds_rectification_maps_without_raising_and_derives_baseline():
    calibration = _make_calibration()
    client = DepthCameraClient(calibration, "http://fake-endpoint/depth")
    assert client._baseline_m == pytest.approx(0.12)


def test_rectify_preserves_configured_image_size():
    calibration = _make_calibration()
    client = DepthCameraClient(calibration, "http://fake-endpoint/depth")
    left_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    right_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    rectified_left, rectified_right = client.rectify(left_frame, right_frame)

    assert rectified_left.shape == (720, 1280, 3)
    assert rectified_right.shape == (720, 1280, 3)


class _FakeResponse:
    def __init__(self, json_data: dict):
        self._json_data = json_data

    def raise_for_status(self):
        pass

    def json(self):
        return self._json_data


def _encode_depth_array(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    np.save(buf, arr)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_compute_depth_round_trips_array_and_sends_expected_payload(monkeypatch):
    calibration = _make_calibration()
    client = DepthCameraClient(calibration, "http://fake-endpoint/depth")
    left_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    right_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

    original_depth = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    depth_b64 = _encode_depth_array(original_depth)

    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["payload"] = json
        captured["timeout"] = timeout
        return _FakeResponse({"depth_npy_b64": depth_b64})

    monkeypatch.setattr(depth_camera.requests, "post", fake_post)

    result = client.compute_depth(left_frame, right_frame)

    assert result is not None
    np.testing.assert_array_equal(result, original_depth)

    payload = captured["payload"]
    assert "left_png_b64" in payload and "right_png_b64" in payload

    import cv2

    left_bytes = base64.b64decode(payload["left_png_b64"])
    left_decoded = cv2.imdecode(np.frombuffer(left_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    expected_w, expected_h = compute_target_size(1280, 720)
    assert left_decoded.shape[:2] == (expected_h, expected_w)

    right_bytes = base64.b64decode(payload["right_png_b64"])
    right_decoded = cv2.imdecode(np.frombuffer(right_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert right_decoded.shape[:2] == (expected_h, expected_w)

    assert payload["intrinsics_flat"] != client._rectified_intrinsics_flat
    assert payload["baseline_m"] == pytest.approx(0.12)


def test_compute_depth_calls_requests_post_with_bounded_timeout(monkeypatch):
    calibration = _make_calibration()
    client = DepthCameraClient(calibration, "http://fake-endpoint/depth")
    left_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    right_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    original_depth = np.array([[1.0]], dtype=np.float32)
    depth_b64 = _encode_depth_array(original_depth)

    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured["timeout"] = timeout
        return _FakeResponse({"depth_npy_b64": depth_b64})

    monkeypatch.setattr(depth_camera.requests, "post", fake_post)

    client.compute_depth(left_frame, right_frame)

    assert captured["timeout"] == 10.0


def test_compute_depth_returns_none_on_request_exception(monkeypatch):
    calibration = _make_calibration()
    client = DepthCameraClient(calibration, "http://fake-endpoint/depth")
    left_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    right_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    def fake_post_raises(url, json=None, timeout=None):
        raise depth_camera.requests.exceptions.RequestException("connection refused")

    monkeypatch.setattr(depth_camera.requests, "post", fake_post_raises)

    result = client.compute_depth(left_frame, right_frame)

    assert result is None


def test_compute_depth_returns_none_when_response_missing_depth_key(monkeypatch):
    calibration = _make_calibration()
    client = DepthCameraClient(calibration, "http://fake-endpoint/depth")
    left_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    right_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    def fake_post(url, json=None, timeout=None):
        return _FakeResponse({"unexpected_key": "value"})

    monkeypatch.setattr(depth_camera.requests, "post", fake_post)

    result = client.compute_depth(left_frame, right_frame)

    assert result is None
