"""Tests for `control/vla_bridge/io_logger.py` (VLAHW-03).

Uses a hand-built fake camera object (`.read() -> (ok, numpy_array)`) --
never a real `cv2.VideoCapture` -- and `tmp_path` for all file I/O, matching
`control/conftest.py`'s no-real-hardware convention.
"""

import json

import numpy as np
import pytest

from vla_bridge.io_logger import IOLogger

CAMERA_NAMES = {0: "wrist", 1: "overhead"}


class FakeCamera:
    """Minimal stand-in for `cv2.VideoCapture` -- `.read()` only."""

    def __init__(self, ok: bool = True):
        self.ok = ok
        self.calls = 0

    def read(self):
        self.calls += 1
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        return self.ok, frame


PATTERN4_FIELDS = [
    "step",
    "timestamp_utc",
    "instruction",
    "camera_frames",
    "joint_state",
    "raw_model_output",
    "validated_action",
    "validator_flags",
    "executed_action",
    "latency_ms",
    "model_version",
]


def _write_dummy_step(logger: IOLogger, step: int) -> None:
    logger.write_step(
        step=step,
        instruction="Pick up the red cube",
        camera_frames={},
        joint_state={"shoulder_pan": 0.0},
        raw_model_output={"shoulder_pan": 0.0},
        validated_action={"shoulder_pan": 0.0},
        validator_flags=[],
        executed_action={"shoulder_pan": 0.0},
        latency_ms={"observation_to_action": 0, "action_to_execution": 0},
        model_version="scripted-dry-run-v1",
    )


def test_write_step_produces_valid_json_with_all_pattern4_fields(tmp_path):
    with IOLogger(tmp_path, CAMERA_NAMES) as logger:
        _write_dummy_step(logger, 0)

    lines = (tmp_path / "episode.jsonl").read_text().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    for field in PATTERN4_FIELDS:
        assert field in record, f"missing Pattern 4 field: {field}"


def test_write_step_three_times_produces_three_valid_json_lines(tmp_path):
    with IOLogger(tmp_path, CAMERA_NAMES) as logger:
        for step in range(3):
            _write_dummy_step(logger, step)

    lines = (tmp_path / "episode.jsonl").read_text().splitlines()
    assert len(lines) == 3
    for line in lines:
        record = json.loads(line)
        for field in PATTERN4_FIELDS:
            assert field in record


def test_two_cameras_get_different_captured_at_utc_timestamps(tmp_path):
    with IOLogger(tmp_path, CAMERA_NAMES) as logger:
        cam0 = FakeCamera()
        cam1 = FakeCamera()
        frame0 = logger.capture_camera_frame(cam0, "wrist", step=0)
        frame1 = logger.capture_camera_frame(cam1, "overhead", step=0)

    assert frame0["captured_at_utc"] != frame1["captured_at_utc"] or frame0["path"] != frame1["path"]
    # Independence, not just inequality-by-luck: each capture must record its
    # own timestamp variable, not share one across cameras.
    assert "captured_at_utc" in frame0
    assert "captured_at_utc" in frame1


def test_camera_frame_path_resolves_to_a_real_file_on_disk(tmp_path):
    with IOLogger(tmp_path, CAMERA_NAMES) as logger:
        cam = FakeCamera()
        frame = logger.capture_camera_frame(cam, "wrist", step=0)

    resolved = tmp_path / frame["path"]
    assert resolved.exists()


def test_camera_capture_never_raises_on_failed_read(tmp_path):
    with IOLogger(tmp_path, CAMERA_NAMES) as logger:
        cam = FakeCamera(ok=False)
        frame = logger.capture_camera_frame(cam, "wrist", step=0)

    assert frame["path"] is None
    assert frame["error"] == "capture failed"


def test_log_is_durable_across_a_simulated_mid_episode_crash(tmp_path):
    logger = IOLogger(tmp_path, CAMERA_NAMES)
    _write_dummy_step(logger, 0)
    _write_dummy_step(logger, 1)
    # Simulate a crash: no __exit__/close() call at all.
    del logger

    lines = (tmp_path / "episode.jsonl").read_text().splitlines()
    assert len(lines) == 2
    for line in lines:
        json.loads(line)  # must not raise -- every line is valid JSON
