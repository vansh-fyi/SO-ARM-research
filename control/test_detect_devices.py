"""Tests for `control/detect_devices.py` (Plan 11-05 Task 2).

Mocks `subprocess.run` (system_profiler), `cv2.VideoCapture`, and the robot
connect calls throughout -- no real hardware or `system_profiler` call during
these tests.
"""

import json
import subprocess

import numpy as np
import pytest

import detect_devices
from detect_devices import (
    DeviceDetectionError,
    _correlate_names,
    _is_builtin_name,
    _parse_camera_names,
    detect_devices as run_detect_devices,
    probe_camera_candidates,
    resolve_cameras,
    resolve_port_identity,
)
from vla_bridge.stereo_camera import STEREO_HEIGHT, STEREO_WIDTH

SYSTEM_PROFILER_SAMPLE = """Camera:

    FaceTime HD Camera (Built-in):

      Model ID: FaceTime HD Camera
      Unique ID: 0x1

    CCB Camera:

      Model ID: UVC Camera VendorID_1234 ProductID_5678
      Unique ID: 0x2

    USB Camera:

      Model ID: UVC Camera VendorID_1111 ProductID_2222
      Unique ID: 0x3

"""


# --- _parse_camera_names ------------------------------------------------------------


def test_parse_camera_names_extracts_top_level_device_names_only():
    names = _parse_camera_names(SYSTEM_PROFILER_SAMPLE)
    assert names == ["FaceTime HD Camera (Built-in)", "CCB Camera", "USB Camera"]


def test_parse_camera_names_empty_output_returns_empty_list():
    assert _parse_camera_names("") == []


# --- get_camera_names -----------------------------------------------------------------


def test_get_camera_names_calls_system_profiler_and_parses_output(monkeypatch):
    def fake_run(cmd, **kwargs):
        assert cmd == ["system_profiler", "SPCameraDataType"]
        return subprocess.CompletedProcess(cmd, 0, stdout=SYSTEM_PROFILER_SAMPLE, stderr="")

    monkeypatch.setattr(detect_devices.subprocess, "run", fake_run)
    names = detect_devices.get_camera_names()
    assert names == ["FaceTime HD Camera (Built-in)", "CCB Camera", "USB Camera"]


def test_get_camera_names_returns_empty_list_on_subprocess_failure(monkeypatch):
    def fake_run(cmd, **kwargs):
        raise OSError("system_profiler not found")

    monkeypatch.setattr(detect_devices.subprocess, "run", fake_run)
    assert detect_devices.get_camera_names() == []


# --- _is_builtin_name -----------------------------------------------------------------


def test_is_builtin_name_matches_facetime_case_insensitively():
    assert _is_builtin_name("FaceTime HD Camera (Built-in)") is True
    assert _is_builtin_name("facetime hd camera") is True


def test_is_builtin_name_false_for_real_camera_names():
    assert _is_builtin_name("CCB Camera") is False
    assert _is_builtin_name("USB Camera") is False


def test_is_builtin_name_false_for_none():
    assert _is_builtin_name(None) is False


# --- _correlate_names -----------------------------------------------------------------


def test_correlate_names_positional_zip_of_indices_to_names():
    candidates = [{"index": 0, "width": 1920, "height": 1080}, {"index": 2, "width": 2560, "height": 720}]
    names = ["CCB Camera", "USB Camera"]
    assert _correlate_names(candidates, names) == {0: "CCB Camera", 2: "USB Camera"}


def test_correlate_names_extra_indices_beyond_names_get_none():
    candidates = [{"index": 0, "width": 1920, "height": 1080}, {"index": 1, "width": 1920, "height": 1080}]
    names = ["CCB Camera"]
    assert _correlate_names(candidates, names) == {0: "CCB Camera", 1: None}


# --- probe_camera_candidates -----------------------------------------------------------


class FakeCapture:
    def __init__(self, index, opens=True, width=1920, height=1080):
        self._opens = opens
        self._width = width
        self._height = height

    def isOpened(self):
        return self._opens

    def read(self):
        if not self._opens:
            return False, None
        return True, np.zeros((self._height, self._width, 3), dtype=np.uint8)

    def release(self):
        pass


def test_probe_camera_candidates_records_resolution_per_opened_index(monkeypatch):
    specs = {
        0: {"opens": True, "width": 2560, "height": 720},
        1: {"opens": True, "width": 1920, "height": 1080},
        2: {"opens": False},
    }

    def factory(idx):
        spec = specs.get(idx, {"opens": False})
        return FakeCapture(idx, **spec)

    monkeypatch.setattr(detect_devices.cv2, "VideoCapture", factory)

    candidates = probe_camera_candidates(max_index=3)
    assert candidates == [
        {"index": 0, "width": 2560, "height": 720},
        {"index": 1, "width": 1920, "height": 1080},
    ]


def test_probe_camera_candidates_skips_failed_read(monkeypatch):
    class ReadFailsCapture(FakeCapture):
        def read(self):
            return False, None

    monkeypatch.setattr(detect_devices.cv2, "VideoCapture", lambda idx: ReadFailsCapture(idx))
    assert probe_camera_candidates(max_index=2) == []


# --- resolve_cameras (acceptance criteria) ----------------------------------------------


def test_builtin_camera_never_assigned_wrist_or_stereo_even_if_only_extra_candidate():
    """A camera named 'FaceTime HD Camera' must never be assigned to wrist or
    stereo_overhead, even if it's the only 'extra' non-stereo camera detected."""
    candidates = [
        {"index": 0, "width": STEREO_WIDTH, "height": STEREO_HEIGHT},  # real AR0144 stereo
        {"index": 1, "width": 1920, "height": 1080},  # real wrist camera
        {"index": 2, "width": 1920, "height": 1080},  # built-in FaceTime webcam
    ]
    names_by_index = {0: "CCB Camera", 1: "USB Camera", 2: "FaceTime HD Camera (Built-in)"}

    result = resolve_cameras(candidates, names_by_index)

    assert result == {"wrist": 1, "stereo_overhead": 0}
    assert result["wrist"] != 2
    assert result["stereo_overhead"] != 2


def test_builtin_only_extra_candidate_raises_rather_than_assigning_it():
    """If the built-in webcam is the ONLY non-stereo candidate left after exclusion,
    there is no valid wrist camera -- must raise, not silently fall back to it."""
    candidates = [
        {"index": 0, "width": STEREO_WIDTH, "height": STEREO_HEIGHT},
        {"index": 1, "width": 1920, "height": 1080},  # FaceTime, to be excluded
    ]
    names_by_index = {0: "CCB Camera", 1: "FaceTime HD Camera (Built-in)"}

    with pytest.raises(DeviceDetectionError):
        resolve_cameras(candidates, names_by_index)


def test_2560x720_resolution_candidate_always_assigned_stereo_overhead_never_wrist():
    candidates = [
        {"index": 5, "width": 1920, "height": 1080},
        {"index": 3, "width": STEREO_WIDTH, "height": STEREO_HEIGHT},
    ]
    names_by_index = {5: "USB Camera", 3: "CCB Camera"}

    result = resolve_cameras(candidates, names_by_index)

    assert result["stereo_overhead"] == 3
    assert result["wrist"] == 5


def test_no_stereo_resolution_candidate_raises():
    candidates = [{"index": 0, "width": 1920, "height": 1080}, {"index": 1, "width": 1920, "height": 1080}]
    names_by_index = {0: "CCB Camera", 1: "USB Camera"}

    with pytest.raises(DeviceDetectionError):
        resolve_cameras(candidates, names_by_index)


def test_ambiguous_wrist_candidates_fall_back_to_interactive_brightness_check(monkeypatch):
    candidates = [
        {"index": 0, "width": STEREO_WIDTH, "height": STEREO_HEIGHT},
        {"index": 1, "width": 1920, "height": 1080},
        {"index": 2, "width": 1920, "height": 1080},
    ]
    names_by_index = {0: "CCB Camera", 1: "USB Camera A", 2: "USB Camera B"}

    # Index 2 is "covered" -- its brightness drops sharply after the prompt; index 1
    # stays roughly the same.
    frames = {
        1: [np.full((4, 4, 3), 200, dtype=np.uint8), np.full((4, 4, 3), 195, dtype=np.uint8)],
        2: [np.full((4, 4, 3), 200, dtype=np.uint8), np.zeros((4, 4, 3), dtype=np.uint8)],
    }
    call_counts = {1: 0, 2: 0}

    class ScriptedCapture:
        def __init__(self, idx):
            self.idx = idx

        def read(self):
            frame = frames[self.idx][call_counts[self.idx]]
            call_counts[self.idx] += 1
            return True, frame

        def release(self):
            pass

    monkeypatch.setattr(detect_devices.cv2, "VideoCapture", lambda idx: ScriptedCapture(idx))
    prompts = []

    def stub_prompt(msg):
        prompts.append(msg)

    result = resolve_cameras(candidates, names_by_index, prompt_fn=stub_prompt)

    assert result == {"wrist": 2, "stereo_overhead": 0}
    assert len(prompts) == 1


def test_ambiguous_stereo_resolution_candidates_raises():
    candidates = [
        {"index": 0, "width": STEREO_WIDTH, "height": STEREO_HEIGHT},
        {"index": 1, "width": STEREO_WIDTH, "height": STEREO_HEIGHT},
    ]
    names_by_index = {0: "CCB Camera", 1: "USB Camera"}

    with pytest.raises(DeviceDetectionError):
        resolve_cameras(candidates, names_by_index)


# --- resolve_port_identity ---------------------------------------------------------------


class FakeConnectableRobot:
    """Stand-in for `SO101Follower`/`SOLeader` -- never touches a real serial port."""

    def __init__(self):
        self.calibration = {}
        self.connected = False
        self.bus = _FakeBus()

    def connect(self, calibrate=False):
        self.connected = True

    def disconnect(self):
        self.connected = False


class _FakeBus:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def torque_disabled(self):
        return self

    def write_calibration(self, calibration):
        pass


class FakeUnreachableRobot:
    def __init__(self):
        pass

    def connect(self, calibrate=False):
        raise ConnectionError("no such port")


def test_resolve_port_identity_returns_true_on_successful_round_trip():
    def make_robot(port, robot_id):
        assert port == "/dev/cu.fake1"
        assert robot_id == "soarm_follower_02"
        return FakeConnectableRobot()

    assert resolve_port_identity("/dev/cu.fake1", "soarm_follower_02", make_robot=make_robot) is True


def test_resolve_port_identity_returns_false_never_raises_on_connect_failure():
    def make_robot(port, robot_id):
        return FakeUnreachableRobot()

    assert resolve_port_identity("/dev/cu.fake1", "soarm_follower_02", make_robot=make_robot) is False


# --- detect_devices (top-level) -----------------------------------------------------------


def _no_op_prompt(msg):
    pass


def test_detect_devices_resolves_follower_leader_and_cameras(monkeypatch):
    def make_follower(port, robot_id):
        if port == "/dev/cu.follower_port":
            return FakeConnectableRobot()
        raise ConnectionError("wrong port")

    def make_leader(port, robot_id):
        if port == "/dev/cu.leader_port":
            return FakeConnectableRobot()
        raise ConnectionError("wrong port")

    monkeypatch.setattr(
        detect_devices,
        "probe_camera_candidates",
        lambda max_index: [
            {"index": 0, "width": STEREO_WIDTH, "height": STEREO_HEIGHT},
            {"index": 1, "width": 1920, "height": 1080},
        ],
    )
    monkeypatch.setattr(detect_devices, "get_camera_names", lambda: ["CCB Camera", "USB Camera"])

    result = run_detect_devices(
        ports=["/dev/cu.follower_port", "/dev/cu.leader_port"],
        make_follower=make_follower,
        make_leader=make_leader,
        prompt_fn=_no_op_prompt,
    )

    assert result["follower"] == {"port": "/dev/cu.follower_port", "id": "soarm_follower_02"}
    assert result["leader"] == {"port": "/dev/cu.leader_port", "id": "soarm_leader_01"}
    assert result["cameras"] == {"wrist": 1, "stereo_overhead": 0}
    assert "detected_at" in result


def test_detect_devices_leader_is_none_when_not_connected(monkeypatch):
    def make_follower(port, robot_id):
        return FakeConnectableRobot()

    def make_leader(port, robot_id):
        raise ConnectionError("no leader connected")

    monkeypatch.setattr(
        detect_devices,
        "probe_camera_candidates",
        lambda max_index: [
            {"index": 0, "width": STEREO_WIDTH, "height": STEREO_HEIGHT},
            {"index": 1, "width": 1920, "height": 1080},
        ],
    )
    monkeypatch.setattr(detect_devices, "get_camera_names", lambda: [])

    result = run_detect_devices(
        ports=["/dev/cu.follower_port"],
        make_follower=make_follower,
        make_leader=make_leader,
        prompt_fn=_no_op_prompt,
    )

    assert result["follower"] == {"port": "/dev/cu.follower_port", "id": "soarm_follower_02"}
    assert result["leader"] is None


def test_detect_devices_raises_when_no_port_resolves_to_follower(monkeypatch):
    def always_fails(port, robot_id):
        raise ConnectionError("nope")

    with pytest.raises(DeviceDetectionError):
        run_detect_devices(
            ports=["/dev/cu.mystery"],
            make_follower=always_fails,
            make_leader=always_fails,
            prompt_fn=_no_op_prompt,
        )


# --- main() CLI (writes device_map.json) --------------------------------------------------


def test_main_writes_device_map_json_to_out_path(monkeypatch, tmp_path):
    out_path = tmp_path / "device_map.json"

    def fake_detect_devices(max_camera_index):
        return {
            "detected_at": "2026-09-22T10:00:00Z",
            "follower": {"port": "/dev/cu.fake", "id": "soarm_follower_02"},
            "leader": None,
            "cameras": {"wrist": 1, "stereo_overhead": 0},
        }

    monkeypatch.setattr(detect_devices, "detect_devices", fake_detect_devices)
    monkeypatch.setattr("sys.argv", ["detect_devices.py", "--out", str(out_path)])

    detect_devices.main()

    written = json.loads(out_path.read_text())
    assert written["follower"]["port"] == "/dev/cu.fake"
    assert written["cameras"] == {"wrist": 1, "stereo_overhead": 0}


def test_main_exits_nonzero_on_device_detection_error(monkeypatch, tmp_path):
    def fake_detect_devices(max_camera_index):
        raise DeviceDetectionError("no follower found")

    monkeypatch.setattr(detect_devices, "detect_devices", fake_detect_devices)
    monkeypatch.setattr("sys.argv", ["detect_devices.py", "--out", str(tmp_path / "device_map.json")])

    with pytest.raises(SystemExit):
        detect_devices.main()
