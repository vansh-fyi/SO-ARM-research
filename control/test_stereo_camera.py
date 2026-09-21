"""Tests for `vla_bridge.stereo_camera.StereoSplitCamera` (Plan 11-04, Task 3).

Mocks `cv2.VideoCapture` throughout -- no real AR0144 device is ever opened
during these tests.
"""

import numpy as np

from vla_bridge import robot_client, stereo_camera
from vla_bridge.stereo_camera import STEREO_WIDTH, StereoSplitCamera


def _synthetic_stereo_frame() -> np.ndarray:
    """A 720x2560 frame with distinguishable left/right halves: left half is
    all 10s, right half is all 200s -- lets tests assert the split lands on
    the correct side, not swapped or overlapping."""
    frame = np.zeros((720, STEREO_WIDTH, 3), dtype=np.uint8)
    frame[:, :1280] = 10
    frame[:, 1280:] = 200
    return frame


class FakeCapture:
    """Stand-in for `cv2.VideoCapture` -- tracks open/read call counts so
    tests can assert the single-open/single-read-per-tick behaviors."""

    instances_created = 0

    def __init__(self, index, opens=True, reads_ok=True):
        FakeCapture.instances_created += 1
        self._opens = opens
        self._reads_ok = reads_ok
        self.read_count = 0
        self.released = False

    def isOpened(self):
        return self._opens

    def read(self):
        self.read_count += 1
        if not self._reads_ok:
            return False, None
        return True, _synthetic_stereo_frame()

    def release(self):
        self.released = True


def _install_fake_capture(monkeypatch, **kwargs):
    FakeCapture.instances_created = 0
    created = {}

    def factory(index):
        cap = FakeCapture(index, **kwargs)
        created["cap"] = cap
        return cap

    monkeypatch.setattr(stereo_camera.cv2, "VideoCapture", factory)
    return created


# --- Behavior 1: single open, single read per tick --------------------------


def test_device_opened_exactly_once_per_instance(monkeypatch):
    created = _install_fake_capture(monkeypatch)

    StereoSplitCamera(index=1, warmup_frames=0)

    assert FakeCapture.instances_created == 1


def test_read_left_and_read_right_share_one_physical_read_per_tick(monkeypatch):
    created = _install_fake_capture(monkeypatch)

    cam = StereoSplitCamera(index=1, warmup_frames=0)
    left = cam.read_left()
    right = cam.read_right()

    assert created["cap"].read_count == 1
    assert left is not None
    assert right is not None


def test_next_tick_triggers_a_fresh_physical_read(monkeypatch):
    created = _install_fake_capture(monkeypatch)

    cam = StereoSplitCamera(index=1, warmup_frames=0)
    cam.read_left()
    cam.read_right()
    assert created["cap"].read_count == 1

    # A new tick: both halves have been consumed, so this pair triggers
    # exactly one more physical read (not zero, not two).
    cam.read_left()
    cam.read_right()
    assert created["cap"].read_count == 2


# --- Behavior 2: correct left/right crop split -------------------------------


def test_read_left_and_read_right_return_correct_non_swapped_halves(monkeypatch):
    _install_fake_capture(monkeypatch)

    cam = StereoSplitCamera(index=1, warmup_frames=0)
    left = cam.read_left()
    right = cam.read_right()

    assert left.shape == (720, 1280, 3)
    assert right.shape == (720, 1280, 3)
    assert np.all(left == 10)
    assert np.all(right == 200)


# --- Behavior 3: open/read failure handling ----------------------------------


def test_device_open_failure_returns_none_for_both_halves_not_crash(monkeypatch):
    _install_fake_capture(monkeypatch, opens=False)

    cam = StereoSplitCamera(index=1, warmup_frames=0)

    assert cam.is_opened is False
    assert cam.read_left() is None
    assert cam.read_right() is None


def test_device_read_failure_returns_none_not_stale_frame(monkeypatch):
    _install_fake_capture(monkeypatch, opens=True, reads_ok=False)

    cam = StereoSplitCamera(index=1, warmup_frames=0)

    assert cam.read_left() is None
    assert cam.read_right() is None


def test_release_closes_underlying_capture(monkeypatch):
    created = _install_fake_capture(monkeypatch)

    cam = StereoSplitCamera(index=1, warmup_frames=0)
    cam.release()

    assert created["cap"].released is True


# --- robot_client.py wiring (acceptance criteria) ----------------------------


class FakeRobotForWiring:
    def get_observation(self):
        return {"shoulder_pan.pos": 0.0, "camera1": "wrist-frame"}


class FakeClientForWiring:
    def __init__(self):
        self.robot = FakeRobotForWiring()


class FakeStereoForWiring:
    def read_left(self):
        return "left-frame"

    def read_right(self):
        return "right-frame"


def test_wire_stereo_split_cameras_adds_camera2_camera3_from_stereo_split():
    """robot_client.py's bridge camera config must reference
    StereoSplitCamera's feeds for camera2/camera3, not two independent
    cv2.VideoCapture(1) opens."""
    client = FakeClientForWiring()

    robot_client._wire_stereo_split_cameras(client, stereo_camera=FakeStereoForWiring())

    obs = client.robot.get_observation()
    assert obs["camera1"] == "wrist-frame"
    assert obs["camera2"] == "left-frame"
    assert obs["camera3"] == "right-frame"
