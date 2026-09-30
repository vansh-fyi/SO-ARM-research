"""Hardware-free regression tests for the standalone recorder."""
import json

import numpy as np
import pytest

import record_episode as recorder


class FakeCapture:
    def __init__(self, index, opened=True):
        self.index = index
        self.opened = opened
        self.reads = 0
        self.frame = np.full((4, 6, 3), 17, dtype=np.uint8)
        self.released = False

    def isOpened(self):
        return self.opened

    def read(self):
        self.reads += 1
        return True, self.frame

    def get(self, prop):
        return 6 if prop == recorder.cv2.CAP_PROP_FRAME_WIDTH else 4

    def release(self):
        self.released = True


def test_open_cameras_plain_indices(monkeypatch):
    opened = []
    def capture(index):
        opened.append(index)
        return FakeCapture(index, opened=index != 8)
    monkeypatch.setattr(recorder.cv2, "VideoCapture", capture)
    assert list(recorder.open_cameras([3, 8])) == [3]
    assert opened == [3, 8]


def test_record_still_generic_warmup_and_pixels(monkeypatch, tmp_path):
    cap = FakeCapture(3)
    monkeypatch.setattr(recorder.time, "sleep", lambda _: None)
    recorder.record_still({3: cap}, tmp_path)
    assert cap.reads == 15
    np.testing.assert_array_equal(recorder.cv2.imread(str(tmp_path / "camera_3.png")), cap.frame)


@pytest.mark.parametrize("contents", [None, "invalid json"])
def test_load_device_map_missing_or_invalid(monkeypatch, tmp_path, contents):
    path = tmp_path / "device_map.json"
    if contents is not None:
        path.write_text(contents)
    monkeypatch.setattr(recorder, "DEVICE_MAP_PATH", path)
    assert recorder._load_device_map() is None


def test_load_device_map_unreadable(monkeypatch, tmp_path):
    monkeypatch.setattr(recorder, "DEVICE_MAP_PATH", tmp_path)
    assert recorder._load_device_map() is None


def test_load_device_map_stale_warns_and_returns_map(monkeypatch, tmp_path, capsys):
    data = {"detected_at": "2000-01-01T00:00:00Z", "cameras": {"wrist": 3}}
    path = tmp_path / "device_map.json"
    path.write_text(json.dumps(data))
    monkeypatch.setattr(recorder, "DEVICE_MAP_PATH", path)
    assert recorder._load_device_map() == data
    assert "ports/camera indices may have shifted" in capsys.readouterr().out


@pytest.mark.parametrize("device_map,expected", [
    (None, {}), ({}, {}),
    ({"cameras": {"wrist": 3, "stereo_overhead": 4}}, {3: "wrist", 4: "overhead"}),
])
def test_build_camera_roles(device_map, expected):
    assert recorder._build_camera_roles([3, 4, 9], device_map) == expected
