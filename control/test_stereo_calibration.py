"""Tests for `vla_bridge.stereo_calibration` (Plan 12-05).

Deterministic, hardware-free coverage for every pure/orchestration function --
no real ffmpeg subprocess or AR0144 device is ever opened. Mirrors
`test_stereo_camera.py`'s hand-rolled-fake convention throughout (never
`unittest.mock`).
"""

import json

import numpy as np
import pytest

from vla_bridge import stereo_calibration
from vla_bridge.stereo_calibration import (
    StereoCalibration,
    build_object_points,
    calibrate_single_camera,
    calibrate_stereo_pair,
    detect_checkerboard_corners,
    load_calibration,
    run_calibration_session,
    save_calibration,
)


def _render_synthetic_checkerboard(pattern_size, square_px=60, margin_px=60) -> np.ndarray:
    """A real, clean, axis-aligned checkerboard pattern `cv2.findChessboardCorners`
    reliably detects (no perspective distortion needed for corner detection to
    succeed). `pattern_size` is (cols, rows) of INTERNAL corners, so the grid
    itself is (cols+1) x (rows+1) squares. Each internal corner lands at an
    exact, computable integer pixel location: `margin_px + (col+1) * square_px`,
    `margin_px + (row+1) * square_px` for `col in range(cols)`, `row in
    range(rows)`. Mirrors `test_stereo_camera.py`'s `_synthetic_stereo_frame()`
    naming/docstring style."""
    import cv2

    cols, rows = pattern_size
    grid_cols = cols + 1
    grid_rows = rows + 1
    height = grid_rows * square_px + 2 * margin_px
    width = grid_cols * square_px + 2 * margin_px

    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    for row in range(grid_rows):
        for col in range(grid_cols):
            if (row + col) % 2 == 0:
                y0 = margin_px + row * square_px
                y1 = y0 + square_px
                x0 = margin_px + col * square_px
                x1 = x0 + square_px
                canvas[y0:y1, x0:x1] = 0
    return canvas


def _expected_corner_locations(pattern_size, square_px=60, margin_px=60):
    cols, rows = pattern_size
    return [
        (margin_px + (col + 1) * square_px, margin_px + (row + 1) * square_px)
        for row in range(rows)
        for col in range(cols)
    ]


# --- build_object_points ------------------------------------------------


def test_build_object_points_returns_flat_grid_scaled_by_square_size():
    points = build_object_points(pattern_size=(9, 6), square_size_m=0.025)

    assert points.shape == (54, 3)
    assert points.dtype == np.float32
    for row in range(6):
        for col in range(9):
            expected = [col * 0.025, row * 0.025, 0.0]
            assert list(points[row * 9 + col]) == expected


# --- detect_checkerboard_corners -----------------------------------------


def test_detect_checkerboard_corners_finds_corners_near_expected_locations():
    pattern_size = (9, 7)
    image = _render_synthetic_checkerboard(pattern_size)

    corners = detect_checkerboard_corners(image, pattern_size=pattern_size)

    assert corners is not None
    assert corners.shape == (63, 1, 2)

    expected_locations = _expected_corner_locations(pattern_size)
    detected = corners.reshape(-1, 2)
    for expected_xy in expected_locations:
        distances = np.linalg.norm(detected - np.array(expected_xy), axis=1)
        assert distances.min() <= 3.0


def test_detect_checkerboard_corners_returns_none_on_blank_image_never_raises():
    blank = np.full((480, 640, 3), 128, dtype=np.uint8)

    result = detect_checkerboard_corners(blank, pattern_size=(9, 6))

    assert result is None


# --- calibrate_single_camera ---------------------------------------------


def test_calibrate_single_camera_calls_cv2_calibrate_camera_once_and_reduces_to_3_tuple(monkeypatch):
    calls = []
    fixed_K = np.eye(3)
    fixed_D = np.zeros(5)

    def fake_calibrate_camera(object_points_list, image_points_list, image_size, camera_matrix, dist_coeffs):
        calls.append((object_points_list, image_points_list, image_size, camera_matrix, dist_coeffs))
        return 0.42, fixed_K, fixed_D, ["rvec"], ["tvec"]

    monkeypatch.setattr(stereo_calibration.cv2, "calibrateCamera", fake_calibrate_camera)

    object_points_list = ["obj"]
    image_points_list = ["img"]
    image_size = (640, 480)

    result = calibrate_single_camera(object_points_list, image_points_list, image_size)

    assert len(calls) == 1
    assert calls[0][:3] == (object_points_list, image_points_list, image_size)
    assert calls[0][3] is None
    assert calls[0][4] is None
    K, D, rms = result
    assert K is fixed_K
    assert D is fixed_D
    assert rms == 0.42


# --- calibrate_stereo_pair -------------------------------------------------


def test_calibrate_stereo_pair_calls_cv2_stereo_calibrate_once_with_fixed_intrinsics(monkeypatch):
    calls = []
    fixed_K1 = np.eye(3)
    fixed_D1 = np.zeros(5)
    fixed_K2 = np.eye(3) * 2
    fixed_D2 = np.zeros(5)
    fixed_R = np.eye(3)
    fixed_T = np.array([[-0.12], [0.0], [0.0]])
    fixed_E = np.zeros((3, 3))
    fixed_F = np.zeros((3, 3))

    def fake_stereo_calibrate(
        object_points_list,
        left_points_list,
        right_points_list,
        left_K,
        left_D,
        right_K,
        right_D,
        image_size,
        flags=None,
    ):
        calls.append(
            dict(
                left_K=left_K,
                left_D=left_D,
                right_K=right_K,
                right_D=right_D,
                image_size=image_size,
                flags=flags,
            )
        )
        return 0.31, fixed_K1, fixed_D1, fixed_K2, fixed_D2, fixed_R, fixed_T, fixed_E, fixed_F

    monkeypatch.setattr(stereo_calibration.cv2, "stereoCalibrate", fake_stereo_calibrate)

    image_size = (1280, 720)
    result = calibrate_stereo_pair(
        ["obj"], ["left"], ["right"], image_size, fixed_K1, fixed_D1, fixed_K2, fixed_D2
    )

    assert len(calls) == 1
    call = calls[0]
    assert call["left_K"] is fixed_K1
    assert call["left_D"] is fixed_D1
    assert call["right_K"] is fixed_K2
    assert call["right_D"] is fixed_D2
    assert call["image_size"] == image_size
    assert call["flags"] == stereo_calibration.cv2.CALIB_FIX_INTRINSIC

    assert isinstance(result, StereoCalibration)
    assert result.reprojection_error_px == 0.31
    assert result.left_K == fixed_K1.flatten().tolist()
    assert result.right_K == fixed_K2.flatten().tolist()
    assert result.R == fixed_R.flatten().tolist()
    assert result.T == [-0.12, 0.0, 0.0]
    assert result.image_size == image_size
    assert isinstance(result.calibrated_at_utc, str)
    assert len(result.calibrated_at_utc) > 0


# --- save_calibration / load_calibration -----------------------------------


def test_save_and_load_calibration_round_trips(tmp_path):
    calibration = StereoCalibration(
        image_size=(1280, 720),
        left_K=[1.0] * 9,
        left_D=[0.1] * 5,
        right_K=[2.0] * 9,
        right_D=[0.2] * 5,
        R=[0.0] * 9,
        T=[-0.12, 0.0, 0.0],
        reprojection_error_px=0.31,
        calibrated_at_utc="2026-09-26T00:00:00+00:00",
    )
    path = tmp_path / "calib.json"

    save_calibration(calibration, path)
    loaded = load_calibration(path)

    assert loaded == calibration
    assert isinstance(loaded.image_size, tuple)

    # File is genuinely JSON with a list for image_size (no tuple type in JSON).
    raw = json.loads(path.read_text())
    assert isinstance(raw["image_size"], list)


# --- run_calibration_session -----------------------------------------------


class FakeStereoCameraForCalibration:
    """Stand-in for `StereoSplitCamera` -- returns a pre-baked checkerboard
    frame pair (or a blank pair) on each call, without ever opening a real
    ffmpeg subprocess or AR0144 device. Mirrors `test_stereo_camera.py`'s
    `FakeCapture` hand-rolled-fake convention."""

    def __init__(self, pattern_size, blank_on_views=None):
        self._pattern_size = pattern_size
        self._blank_on_views = set(blank_on_views or [])
        self._call_index = 0
        self._checkerboard = _render_synthetic_checkerboard(pattern_size)
        self._blank = np.full_like(self._checkerboard, 128)

    def _current_view(self):
        # Two reads (left, right) per view.
        return self._call_index // 2

    def _frame(self):
        view = self._current_view()
        self._call_index += 1
        if view in self._blank_on_views:
            return self._blank
        return self._checkerboard

    def read_left(self):
        return self._frame()

    def read_right(self):
        return self._frame()


def _no_op_prompt(*args, **kwargs):
    return ""


def test_run_calibration_session_calls_calibrate_functions_correctly(monkeypatch):
    pattern_size = (9, 6)
    fake_camera = FakeStereoCameraForCalibration(pattern_size)

    single_calls = []
    stereo_calls = []

    fixed_K = np.eye(3)
    fixed_D = np.zeros(5)

    def fake_calibrate_single_camera(object_points_list, image_points_list, image_size):
        single_calls.append((list(object_points_list), list(image_points_list), image_size))
        return fixed_K, fixed_D, 0.1

    def fake_calibrate_stereo_pair(
        object_points_list, left_points_list, right_points_list, image_size, left_K, left_D, right_K, right_D
    ):
        stereo_calls.append(
            (list(object_points_list), list(left_points_list), list(right_points_list), image_size)
        )
        return StereoCalibration(
            image_size=image_size,
            left_K=[0.0] * 9,
            left_D=[0.0] * 5,
            right_K=[0.0] * 9,
            right_D=[0.0] * 5,
            R=[0.0] * 9,
            T=[0.0, 0.0, 0.0],
            reprojection_error_px=0.1,
            calibrated_at_utc="2026-09-26T00:00:00+00:00",
        )

    monkeypatch.setattr(stereo_calibration, "calibrate_single_camera", fake_calibrate_single_camera)
    monkeypatch.setattr(stereo_calibration, "calibrate_stereo_pair", fake_calibrate_stereo_pair)

    result = run_calibration_session(
        fake_camera, num_views=5, pattern_size=pattern_size, prompt_fn=_no_op_prompt
    )

    assert len(single_calls) == 2  # once for left, once for right
    assert len(stereo_calls) == 1
    assert isinstance(result, StereoCalibration)

    left_call = single_calls[0]
    right_call = single_calls[1]
    stereo_call = stereo_calls[0]

    # Each view's detected corners threaded through correctly: same count in
    # every accumulated list, and object points list matches views captured.
    assert len(left_call[0]) == 5
    assert len(left_call[1]) == 5
    assert len(right_call[1]) == 5
    assert len(stereo_call[0]) == 5
    assert len(stereo_call[1]) == 5
    assert len(stereo_call[2]) == 5


def test_run_calibration_session_skips_views_with_no_detected_corners(monkeypatch):
    pattern_size = (9, 6)
    # View index 3 (0-based, the 4th view) is blank; all others have a valid
    # checkerboard.
    fake_camera = FakeStereoCameraForCalibration(pattern_size, blank_on_views={3})

    stereo_calls = []

    def fake_calibrate_single_camera(object_points_list, image_points_list, image_size):
        return np.eye(3), np.zeros(5), 0.1

    def fake_calibrate_stereo_pair(
        object_points_list, left_points_list, right_points_list, image_size, left_K, left_D, right_K, right_D
    ):
        stereo_calls.append((list(object_points_list), list(left_points_list), list(right_points_list)))
        return StereoCalibration(
            image_size=image_size,
            left_K=[0.0] * 9,
            left_D=[0.0] * 5,
            right_K=[0.0] * 9,
            right_D=[0.0] * 5,
            R=[0.0] * 9,
            T=[0.0, 0.0, 0.0],
            reprojection_error_px=0.1,
            calibrated_at_utc="2026-09-26T00:00:00+00:00",
        )

    monkeypatch.setattr(stereo_calibration, "calibrate_single_camera", fake_calibrate_single_camera)
    monkeypatch.setattr(stereo_calibration, "calibrate_stereo_pair", fake_calibrate_stereo_pair)

    run_calibration_session(fake_camera, num_views=6, pattern_size=pattern_size, prompt_fn=_no_op_prompt)

    # 6 attempted views, 1 skipped (view index 3) -> 5 valid views accumulated.
    assert len(stereo_calls) == 1
    assert len(stereo_calls[0][0]) == 5


def test_run_calibration_session_raises_runtime_error_with_fewer_than_3_valid_views(monkeypatch):
    pattern_size = (9, 6)
    # All 4 attempted views are blank -> 0 valid views.
    fake_camera = FakeStereoCameraForCalibration(pattern_size, blank_on_views={0, 1, 2, 3})

    def fake_calibrate_single_camera(object_points_list, image_points_list, image_size):
        raise AssertionError("calibrate_single_camera should not be called with < 3 valid views")

    def fake_calibrate_stereo_pair(*args, **kwargs):
        raise AssertionError("calibrate_stereo_pair should not be called with < 3 valid views")

    monkeypatch.setattr(stereo_calibration, "calibrate_single_camera", fake_calibrate_single_camera)
    monkeypatch.setattr(stereo_calibration, "calibrate_stereo_pair", fake_calibrate_stereo_pair)

    with pytest.raises(RuntimeError, match=r"Only 0 valid calibration view\(s\) captured \(need at least 3\)"):
        run_calibration_session(fake_camera, num_views=4, pattern_size=pattern_size, prompt_fn=_no_op_prompt)


def test_run_calibration_session_never_opens_second_capture_no_ffmpeg_reference():
    """Module-level guard: this module never imports _open_stereo_capture or
    references ffmpeg -- confirmed structurally via source inspection, not
    just behaviorally, per the plan's acceptance criteria."""
    import inspect

    source = inspect.getsource(stereo_calibration)
    assert "_open_stereo_capture" not in source
    assert "ffmpeg" not in source
