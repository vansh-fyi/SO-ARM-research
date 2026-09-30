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
    assert list(recorder.open_cameras([3, 8], {}, None)) == [3]
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


class FakeStereo:
    def __init__(self, index, opened=True):
        self.index = index
        self.is_opened = opened
        self.released = False
        self.left = np.full((600, 800, 3), 10, dtype=np.uint8)
        self.right = np.full((600, 800, 3), 200, dtype=np.uint8)
        self.left_reads = self.right_reads = 0

    def read_left(self):
        self.left_reads += 1
        return self.left

    def read_right(self):
        self.right_reads += 1
        return self.right

    def release(self):
        self.released = True


@pytest.mark.parametrize("device_map", [None, {"cameras": {"wrist": 3, "stereo_overhead": 4}}])
def test_open_cameras_resolves_roles_once(monkeypatch, device_map):
    opened, stereo, resolved = [], [], []
    def resolve(role, mapping):
        assert mapping is device_map
        resolved.append(role)
        return "CCB Camera" if role == "stereo_overhead" else 7
    def capture(index):
        opened.append(index)
        return FakeCapture(index)
    def split(index):
        stereo.append(index)
        return FakeStereo(index)
    monkeypatch.setattr(recorder.detect_devices, "resolve_camera_identifier", resolve)
    monkeypatch.setattr(recorder.cv2, "VideoCapture", capture)
    monkeypatch.setattr(recorder, "StereoSplitCamera", split)
    caps = recorder.open_cameras([3, 4, 4, 9], {3: "wrist", 4: "overhead"}, device_map)
    assert list(caps) == [3, 4, 9]
    assert opened == ([7, 9] if device_map else [3, 9])
    assert stereo == (["CCB Camera"] if device_map else [4])
    assert resolved == (["wrist", "stereo_overhead"] if device_map else [])


def test_open_cameras_failed_overhead_is_released(monkeypatch, capsys):
    cap = FakeStereo(4, opened=False)
    monkeypatch.setattr(recorder, "StereoSplitCamera", lambda index: cap)
    assert recorder.open_cameras([4], {4: "overhead"}, None) == {}
    assert cap.released
    assert "overhead camera did not open" in capsys.readouterr().out


def test_record_still_stereo_pair(monkeypatch, tmp_path):
    monkeypatch.setattr(recorder, "StereoSplitCamera", FakeStereo)
    cap = FakeStereo(4)
    recorder.record_still({4: cap}, tmp_path)
    assert {p.name for p in tmp_path.iterdir()} == {"camera_overhead_left.png", "camera_overhead_right.png"}
    for side in ("left", "right"):
        np.testing.assert_array_equal(recorder.cv2.imread(str(tmp_path / f"camera_overhead_{side}.png")), getattr(cap, side))


def test_record_still_failed_stereo_is_skipped(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(recorder, "StereoSplitCamera", FakeStereo)
    cap = FakeStereo(4)
    cap.left = cap.right = None
    recorder.record_still({4: cap}, tmp_path)
    assert cap.left_reads == cap.right_reads == 15
    assert not list(tmp_path.iterdir())
    assert "failed to capture" in capsys.readouterr().out


@pytest.mark.parametrize("stereo", [False, True])
def test_record_episode_writer_dimensions_frames_and_release(monkeypatch, tmp_path, stereo):
    monkeypatch.setattr(recorder, "StereoSplitCamera", FakeStereo)
    cap = FakeStereo(4) if stereo else FakeCapture(3)
    writers = {}
    class Writer:
        def __init__(self, path, fourcc, fps, size):
            self.frames, self.size, self.released = [], size, False
            writers[recorder.Path(path).name] = self
        def write(self, frame):
            self.frames.append(frame)
        def release(self):
            self.released = True
    monkeypatch.setattr(recorder.cv2, "VideoWriter", Writer)
    ticks = iter([0, 0, 0, 0.01, 1, 1])
    monkeypatch.setattr(recorder.time, "perf_counter", lambda: next(ticks))
    monkeypatch.setattr(recorder.time, "sleep", lambda _: None)
    recorder.record_episode(None, {4 if stereo else 3: cap}, tmp_path, 0.5, 15, False)
    expected = {"camera_overhead_left.mp4": cap.left, "camera_overhead_right.mp4": cap.right} if stereo else {"camera_3.mp4": cap.frame}
    assert writers.keys() == expected.keys()
    for name, frame in expected.items():
        writer = writers[name]
        assert writer.size == ((recorder.SPLIT_COL, recorder.STEREO_HEIGHT) if stereo else (6, 4))
        assert writer.released and len(writer.frames) == 1
        np.testing.assert_array_equal(writer.frames[0], frame)
    if stereo:
        assert cap.left_reads == cap.right_reads == 1
