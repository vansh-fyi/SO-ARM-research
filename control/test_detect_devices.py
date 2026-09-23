"""Tests for `control/detect_devices.py` (Plan 11-05 Task 2).

Mocks `subprocess.run` (system_profiler), `cv2.VideoCapture`, and the robot
connect calls throughout -- no real hardware or `system_profiler` call during
these tests.
"""

import json
import subprocess
from dataclasses import dataclass

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


@dataclass
class _FakeMotorCalibration:
    """Stand-in for lerobot's `MotorCalibration` -- same fields `resolve_port_identity`
    reads (`homing_offset`, `range_min`, `range_max`)."""

    id: int
    drive_mode: int
    homing_offset: int
    range_min: int
    range_max: int


_SAMPLE_CALIBRATION = {
    "shoulder_pan": _FakeMotorCalibration(id=1, drive_mode=0, homing_offset=-1653, range_min=1269, range_max=2869),
    "gripper": _FakeMotorCalibration(id=6, drive_mode=1, homing_offset=-197, range_min=48, range_max=3637),
}

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

    assert result == {"wrist": 1, "stereo_overhead": 0, "stereo_overhead_name": "CCB Camera"}
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
    """Must stub verify_stereo_fn -- otherwise this falls through to the real
    ffmpeg fallback and touches actual hardware."""
    candidates = [{"index": 0, "width": 1920, "height": 1080}, {"index": 1, "width": 1920, "height": 1080}]
    names_by_index = {0: "CCB Camera", 1: "USB Camera"}

    with pytest.raises(DeviceDetectionError):
        resolve_cameras(candidates, names_by_index, verify_stereo_fn=lambda index: False)


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

    assert result == {"wrist": 2, "stereo_overhead": 0, "stereo_overhead_name": "CCB Camera"}
    assert len(prompts) == 1


def test_ambiguous_stereo_resolution_candidates_raises():
    candidates = [
        {"index": 0, "width": STEREO_WIDTH, "height": STEREO_HEIGHT},
        {"index": 1, "width": STEREO_WIDTH, "height": STEREO_HEIGHT},
    ]
    names_by_index = {0: "CCB Camera", 1: "USB Camera"}

    with pytest.raises(DeviceDetectionError):
        resolve_cameras(candidates, names_by_index)


# --- resolve_cameras: ffmpeg fallback (macOS cv2/AVFoundation resolution bug) ------------


def test_ffmpeg_fallback_used_when_no_cv2_candidate_matches_stereo_resolution():
    """The real 11-05 Task 3 scenario: cv2 reports the AR0144 at some OTHER
    resolution (never 2560x720, a confirmed macOS OpenCV/AVFoundation bug), so
    resolve_cameras() must fall back to verify_stereo_fn to find it."""
    candidates = [
        {"index": 1, "width": 1920, "height": 1080},  # AR0144, cv2 misreports its resolution
        {"index": 2, "width": 1920, "height": 1080},  # real wrist camera
    ]
    names_by_index = {1: "CCB Camera", 2: "USB Camera"}

    def verify_stereo_fn(index):
        return index == 1

    result = resolve_cameras(candidates, names_by_index, verify_stereo_fn=verify_stereo_fn)

    assert result == {"wrist": 2, "stereo_overhead": 1, "stereo_overhead_name": "CCB Camera"}


def test_ffmpeg_fallback_not_consulted_when_cv2_already_found_a_stereo_candidate():
    """Fast path (cv2 resolution match) must win when it succeeds -- the
    fallback should never even be called."""
    candidates = [
        {"index": 0, "width": STEREO_WIDTH, "height": STEREO_HEIGHT},
        {"index": 1, "width": 1920, "height": 1080},
    ]
    names_by_index = {0: "CCB Camera", 1: "USB Camera"}

    def verify_stereo_fn(index):
        raise AssertionError("verify_stereo_fn must not be called when the cv2 fast path succeeds")

    result = resolve_cameras(candidates, names_by_index, verify_stereo_fn=verify_stereo_fn)

    assert result == {"wrist": 1, "stereo_overhead": 0, "stereo_overhead_name": "CCB Camera"}


def test_ffmpeg_fallback_finding_no_match_raises_same_as_no_stereo_candidate():
    candidates = [
        {"index": 1, "width": 1920, "height": 1080},
        {"index": 2, "width": 1920, "height": 1080},
    ]
    names_by_index = {1: "CCB Camera", 2: "USB Camera"}

    def verify_stereo_fn(index):
        return False

    with pytest.raises(DeviceDetectionError):
        resolve_cameras(candidates, names_by_index, verify_stereo_fn=verify_stereo_fn)


def test_ffmpeg_fallback_finding_multiple_matches_raises_ambiguous():
    candidates = [
        {"index": 1, "width": 1920, "height": 1080},
        {"index": 2, "width": 1920, "height": 1080},
    ]
    names_by_index = {1: "CCB Camera", 2: "USB Camera"}

    def verify_stereo_fn(index):
        return True

    with pytest.raises(DeviceDetectionError):
        resolve_cameras(candidates, names_by_index, verify_stereo_fn=verify_stereo_fn)


def test_verify_stereo_via_ffmpeg_returns_true_on_matching_frame_size(monkeypatch):
    expected_bytes = STEREO_WIDTH * STEREO_HEIGHT * 3

    def fake_run(cmd, **kwargs):
        assert "avfoundation" in cmd
        return subprocess.CompletedProcess(cmd, 0, stdout=b"\x00" * expected_bytes, stderr=b"")

    monkeypatch.setattr(detect_devices.subprocess, "run", fake_run)

    assert detect_devices._verify_stereo_via_ffmpeg(1) is True


def test_verify_stereo_via_ffmpeg_returns_false_on_nonzero_exit(monkeypatch):
    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, stdout=b"", stderr=b"error")

    monkeypatch.setattr(detect_devices.subprocess, "run", fake_run)

    assert detect_devices._verify_stereo_via_ffmpeg(1) is False


def test_verify_stereo_via_ffmpeg_returns_false_on_wrong_byte_count(monkeypatch):
    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout=b"\x00" * 100, stderr=b"")

    monkeypatch.setattr(detect_devices.subprocess, "run", fake_run)

    assert detect_devices._verify_stereo_via_ffmpeg(1) is False


def test_verify_stereo_via_ffmpeg_returns_false_never_raises_on_subprocess_error(monkeypatch):
    def fake_run(cmd, **kwargs):
        raise OSError("ffmpeg not found")

    monkeypatch.setattr(detect_devices.subprocess, "run", fake_run)

    assert detect_devices._verify_stereo_via_ffmpeg(1) is False


# --- resolve_port_identity ---------------------------------------------------------------


class FakeConnectableRobot:
    """Stand-in for `SO101Follower`/`SOLeader` -- never touches a real serial port.

    `calibration` mirrors what the real robot loads from its saved calibration
    file at construction time; `live_calibration` mirrors what's actually on the
    servos right now (read via `bus.read_calibration()`). Tests set these
    independently to simulate a match (genuine identity) or a mismatch (wrong id
    resolved, or no saved file at all).
    """

    def __init__(self, calibration=None, live_calibration=None):
        self.calibration = calibration if calibration is not None else {}
        self.connected = False
        self.bus = _FakeBus(live_calibration if live_calibration is not None else self.calibration)

    def connect(self, calibrate=False):
        self.connected = True

    def disconnect(self):
        self.connected = False


class _FakeBus:
    def __init__(self, live_calibration):
        self._live_calibration = live_calibration

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def torque_disabled(self):
        return self

    def write_calibration(self, calibration):
        raise AssertionError(
            "resolve_port_identity() must never call write_calibration() -- "
            "identity resolution is read-only"
        )

    def read_calibration(self):
        return self._live_calibration


class FakeUnreachableRobot:
    def __init__(self):
        pass

    def connect(self, calibrate=False):
        raise ConnectionError("no such port")


def test_resolve_port_identity_returns_true_when_live_matches_saved_calibration():
    def make_robot(port, robot_id):
        assert port == "/dev/cu.fake1"
        assert robot_id == "soarm_follower_02"
        return FakeConnectableRobot(calibration=_SAMPLE_CALIBRATION, live_calibration=_SAMPLE_CALIBRATION)

    assert resolve_port_identity("/dev/cu.fake1", "soarm_follower_02", make_robot=make_robot) is True


def test_resolve_port_identity_returns_false_never_raises_on_connect_failure():
    def make_robot(port, robot_id):
        return FakeUnreachableRobot()

    assert resolve_port_identity("/dev/cu.fake1", "soarm_follower_02", make_robot=make_robot) is False


def test_resolve_port_identity_returns_false_when_no_saved_calibration_for_id():
    """The exact failure mode found live during 11-05 Task 3 prep: an id with no
    saved calibration file loads as an empty dict. Must fail closed, not silently
    'succeed' via a no-op write."""

    def make_robot(port, robot_id):
        return FakeConnectableRobot(calibration={}, live_calibration=_SAMPLE_CALIBRATION)

    assert resolve_port_identity("/dev/cu.fake1", "soarm_leader_01", make_robot=make_robot) is False


def test_resolve_port_identity_returns_false_when_live_calibration_does_not_match_saved():
    """Simulates the real cross-robot scenario: this id's saved file describes a
    DIFFERENT physical robot than what's actually connected on this port."""
    wrong_robot_calibration = {
        "shoulder_pan": _FakeMotorCalibration(id=1, drive_mode=0, homing_offset=-1978, range_min=917, range_max=3133),
        "gripper": _FakeMotorCalibration(id=6, drive_mode=0, homing_offset=1804, range_min=1611, range_max=2896),
    }

    def make_robot(port, robot_id):
        return FakeConnectableRobot(calibration=wrong_robot_calibration, live_calibration=_SAMPLE_CALIBRATION)

    assert resolve_port_identity("/dev/cu.fake1", "soarm_leader_01", make_robot=make_robot) is False


def test_resolve_port_identity_never_writes_to_the_bus():
    """`_FakeBus.write_calibration` raises if called -- this test passes only if
    resolve_port_identity() never invokes it, on success or failure paths."""

    def make_robot(port, robot_id):
        return FakeConnectableRobot(calibration=_SAMPLE_CALIBRATION, live_calibration=_SAMPLE_CALIBRATION)

    resolve_port_identity("/dev/cu.fake1", "soarm_follower_02", make_robot=make_robot)

    def make_mismatched_robot(port, robot_id):
        return FakeConnectableRobot(calibration=_SAMPLE_CALIBRATION, live_calibration={})

    resolve_port_identity("/dev/cu.fake1", "soarm_follower_02", make_robot=make_mismatched_robot)


# --- detect_devices (top-level) -----------------------------------------------------------


def _no_op_prompt(msg):
    pass


def test_detect_devices_resolves_follower_leader_and_cameras(monkeypatch):
    def make_follower(port, robot_id):
        if port == "/dev/cu.follower_port":
            return FakeConnectableRobot(calibration=_SAMPLE_CALIBRATION, live_calibration=_SAMPLE_CALIBRATION)
        raise ConnectionError("wrong port")

    def make_leader(port, robot_id):
        if port == "/dev/cu.leader_port":
            return FakeConnectableRobot(calibration=_SAMPLE_CALIBRATION, live_calibration=_SAMPLE_CALIBRATION)
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
    assert result["cameras"] == {"wrist": 1, "stereo_overhead": 0, "stereo_overhead_name": "CCB Camera"}
    assert "detected_at" in result


def test_detect_devices_leader_is_none_when_not_connected(monkeypatch):
    def make_follower(port, robot_id):
        return FakeConnectableRobot(calibration=_SAMPLE_CALIBRATION, live_calibration=_SAMPLE_CALIBRATION)

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


def test_detect_devices_default_leader_constructor_is_soleader_not_sofollower(monkeypatch):
    """Regression test for the bug found live re-testing the leader this
    session: main() never passed make_leader through detect_devices(), so
    the leader identity check silently defaulted to constructing an
    SO101Follower (resolve_port_identity()'s own generic fallback) even for
    the leader id -- whose calibration_fpath always resolves under
    robots/so_follower/, where the leader's real saved calibration never
    lives (it's under teleoperators/so_leader/). Leader auto-discovery was
    dead on arrival since Task 2 landed. detect_devices() must now default
    make_leader to the real _make_leader_robot (SOLeader) itself, not rely
    on every caller to remember to pass it."""

    def make_follower(port, robot_id):
        raise ConnectionError("not the follower port")

    calls = []

    def fake_make_leader_robot(port, robot_id):
        calls.append((port, robot_id))
        return FakeConnectableRobot(calibration=_SAMPLE_CALIBRATION, live_calibration=_SAMPLE_CALIBRATION)

    monkeypatch.setattr(detect_devices, "_make_leader_robot", fake_make_leader_robot)
    monkeypatch.setattr(
        detect_devices,
        "probe_camera_candidates",
        lambda max_index: [
            {"index": 0, "width": STEREO_WIDTH, "height": STEREO_HEIGHT},
            {"index": 1, "width": 1920, "height": 1080},
        ],
    )
    monkeypatch.setattr(detect_devices, "get_camera_names", lambda: ["CCB Camera", "USB Camera"])

    # make_follower always fails here, so the follower loop iteration falls
    # through to the leader check -- but detect_devices() requires SOME
    # follower to resolve or it raises, so give it a second port where
    # make_follower succeeds (via the real default _make_follower_robot,
    # left untouched) after the leader port has already been tried.
    def make_follower_second_port(port, robot_id):
        if port == "/dev/cu.follower_port":
            return FakeConnectableRobot(calibration=_SAMPLE_CALIBRATION, live_calibration=_SAMPLE_CALIBRATION)
        raise ConnectionError("not the follower port")

    result = run_detect_devices(
        ports=["/dev/cu.leader_port", "/dev/cu.follower_port"],
        make_follower=make_follower_second_port,
        prompt_fn=_no_op_prompt,
    )

    assert calls == [("/dev/cu.leader_port", "soarm_leader_01")]
    assert result["leader"] == {"port": "/dev/cu.leader_port", "id": "soarm_leader_01"}


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
