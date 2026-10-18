"""Tests for `control/run_vla_episode.py` (VLAHW-02, VLAHW-03).

Uses Plan 11-01's `mock_robot` fixture (from `conftest.py`) and a fake camera
object -- never a real robot or real `cv2.VideoCapture`.
"""

import json

import numpy as np
import pytest

import run_vla_episode
import detect_devices
from run_vla_episode import (
    ScriptedActionSource,
    _build_camera_names,
    _interpolate_waypoints,
    _open_cameras,
    run_episode,
)
from vla_bridge import action_contract
from vla_bridge import backends as vla_backends
from vla_bridge import robot_client
from vla_bridge.backends import BackendHandles
from vla_bridge.io_logger import IOLogger


class _NeverOpensCapture:
    """Stand-in for `cv2.VideoCapture` that never actually opens a device --
    keeps the bridge-selection tests below hermetic (no real camera probing)."""

    def __init__(self, *args, **kwargs):
        pass

    def isOpened(self):
        return False

    def release(self):
        pass


class FakeCamera:
    def read(self):
        return True, np.zeros((4, 4, 3), dtype=np.uint8)


CAMERA_NAMES = {0: "wrist", 1: "overhead"}


def _run_resolved_bridge_main(monkeypatch, tmp_path, resolver, cli_args=()):
    device_map = {"cameras": {"wrist": 3, "stereo_overhead": 4}}
    calls = {}
    monkeypatch.setattr(run_vla_episode, "_load_device_map", lambda: device_map)
    monkeypatch.setattr(detect_devices, "resolve_camera_identifier", resolver)
    monkeypatch.setattr(action_contract, "load_joint_limits_deg", lambda: {})
    monkeypatch.setattr(run_vla_episode.cv2, "VideoCapture", _TrackingCapture)
    _TrackingCapture.constructed_indices = []

    class Client:
        robot = object()
        _stereo_camera = object()

        def stop(self):
            calls["stopped"] = True

    client = Client()

    def connect(*args, **kwargs):
        calls.update(kwargs)
        return client

    def episode(*args, **kwargs):
        calls["recording_caps"] = args[1]
        calls["recording_stereo"] = kwargs["stereo_camera"]

    monkeypatch.setattr(robot_client, "connect_bridge", connect)
    monkeypatch.setattr(robot_client, "BridgeActionSource", lambda *a: None)
    monkeypatch.setattr(run_vla_episode, "run_episode", episode)
    monkeypatch.setattr("sys.argv", [
        "run_vla_episode.py", "/dev/fake", "fake_robot", "--out", str(tmp_path),
        "--server-address", "example.test:1234", "--checkpoint", "fake/checkpoint",
        *cli_args,
    ])
    run_vla_episode.main()
    return calls, client


def test_device_map_defaults_use_shared_resolved_identifiers(monkeypatch, tmp_path):
    roles = []

    def resolve(role, device_map):
        roles.append(role)
        assert device_map["cameras"] == {"wrist": 3, "stereo_overhead": 4}
        return {"wrist": 8, "stereo_overhead": "Resolved stereo"}[role]

    calls, _ = _run_resolved_bridge_main(monkeypatch, tmp_path, resolve)
    assert calls["wrist_camera_index"] == 8
    assert calls["stereo_camera_index"] == "Resolved stereo"
    assert roles == ["wrist", "stereo_overhead"]


def test_explicit_camera_identifiers_bypass_shared_resolver(monkeypatch, tmp_path):
    def forbidden(*args):
        raise AssertionError("Explicit camera identifiers must bypass resolution")

    calls, _ = _run_resolved_bridge_main(
        monkeypatch, tmp_path, forbidden,
        ["--camera", "5", "--camera", "7", "--stereo-camera-index", "Explicit stereo"],
    )
    assert calls["wrist_camera_index"] == 5
    assert calls["stereo_camera_index"] == "Explicit stereo"


@pytest.mark.parametrize("wrist_index", [8, 4])
def test_resolved_bridge_records_wrist_and_reuses_single_stereo(
    monkeypatch, tmp_path, wrist_index,
):
    # A resolved wrist index may equal the stale overhead index after USB replug.
    calls, client = _run_resolved_bridge_main(
        monkeypatch, tmp_path,
        lambda role, device_map: {
            "wrist": wrist_index, "stereo_overhead": "Resolved stereo",
        }[role],
    )
    assert _TrackingCapture.constructed_indices == [wrist_index]
    assert list(calls["recording_caps"]) == [wrist_index]
    assert calls["wrist_camera_index"] == wrist_index
    assert calls["stereo_camera_index"] == "Resolved stereo"
    assert calls["recording_stereo"] is client._stereo_camera
    assert calls["stopped"] is True


def _fake_caps():
    return {0: FakeCamera(), 1: FakeCamera()}


class _TrackingCapture:
    """Stand-in for `cv2.VideoCapture` that records the indices it was
    constructed with -- lets tests assert exactly which indices
    `_open_cameras()` attempted to open, matching this file's existing
    hand-rolled-fake convention (not `unittest.mock`)."""

    constructed_indices: list[int] = []

    def __init__(self, idx):
        _TrackingCapture.constructed_indices.append(idx)
        self._idx = idx

    def isOpened(self):
        return True

    def release(self):
        pass


class _FakeStereoCamera:
    """Stand-in for `StereoSplitCamera` -- returns fixed, distinct left/right
    frames instead of reading a real AR0144."""

    def __init__(self):
        self.left_frame = np.full((2, 2, 3), 1, dtype=np.uint8)
        self.right_frame = np.full((2, 2, 3), 2, dtype=np.uint8)

    def read_left(self):
        return self.left_frame

    def read_right(self):
        return self.right_frame


def test_scripted_action_source_toggles_gripper_after_30_steps():
    source = ScriptedActionSource()
    joint_state = {
        "shoulder_pan": 0.0,
        "shoulder_lift": 0.0,
        "elbow_flex": 0.0,
        "wrist_flex": 0.0,
        "wrist_roll": 0.0,
        "gripper": 50.0,
    }

    first_action, model_version = source.get_action(joint_state, "instruction")
    assert first_action["gripper"] == 0.0
    assert model_version == "scripted-dry-run-v1"
    for joint in joint_state:
        if joint != "gripper":
            assert first_action[joint] == joint_state[joint]

    # Calls 2..30 (internal step counter 1..29) stay in the same 30-step
    # block as the first call (internal step 0) -> gripper stays 0.0.
    for _ in range(29):
        action, _ = source.get_action(joint_state, "instruction")
        assert action["gripper"] == 0.0

    # Call 31 (internal step counter reaches 30) crosses into the next
    # 30-step block -> gripper toggles to 100.0.
    action, _ = source.get_action(joint_state, "instruction")
    assert action["gripper"] == 100.0


def test_loop_writes_termination_json_with_max_steps_reason(tmp_path, mock_robot, mock_calibration_file):
    action_source = ScriptedActionSource()
    caps = _fake_caps()
    joint_limits_deg = action_contract.load_joint_limits_deg(mock_calibration_file)
    with IOLogger(tmp_path, CAMERA_NAMES) as io_logger:
        run_episode(
            mock_robot,
            caps,
            CAMERA_NAMES,
            io_logger,
            action_source,
            instruction="Pick up the red cube",
            max_steps=3,
            control_hz=100.0,  # fast, so the test doesn't sleep meaningfully
            joint_limits_deg=joint_limits_deg,
        )

    termination = json.loads((tmp_path / "termination.json").read_text())
    assert termination["reason"] == "max_steps_reached"
    assert termination["steps_completed"] == 3


def test_latency_ms_reflects_real_monotonic_deltas_under_controlled_clock(
    tmp_path, mock_robot, mock_calibration_file, monkeypatch
):
    """LATENCY-03: `latency_ms` must carry real `time.monotonic()`-derived
    millisecond deltas, not the old hardcoded-zero dict. Monkeypatches
    `run_vla_episode.time.monotonic` to a known incrementing sequence
    (0.0 -> 0.1 -> 0.25 across the single tick's t0/t1/t2 brackets) and
    asserts the written `latency_ms` dict matches the expected deltas
    exactly (100.0ms observation_to_action, 150.0ms action_to_execution)."""
    action_source = ScriptedActionSource()
    caps = _fake_caps()
    joint_limits_deg = action_contract.load_joint_limits_deg(mock_calibration_file)

    monotonic_sequence = iter([0.0, 0.1, 0.25])
    monkeypatch.setattr(run_vla_episode.time, "monotonic", lambda: next(monotonic_sequence))

    with IOLogger(tmp_path, CAMERA_NAMES) as io_logger:
        run_episode(
            mock_robot,
            caps,
            CAMERA_NAMES,
            io_logger,
            action_source,
            instruction="Pick up the red cube",
            max_steps=1,
            control_hz=100.0,
            joint_limits_deg=joint_limits_deg,
        )

    lines = (tmp_path / "episode.jsonl").read_text().strip().splitlines()
    record = json.loads(lines[0])
    assert record["latency_ms"]["observation_to_action"] == pytest.approx(100.0)
    assert record["latency_ms"]["action_to_execution"] == pytest.approx(150.0)


def test_keyboard_interrupt_triggers_return_to_start_before_disconnect(
    tmp_path, mock_robot, mock_calibration_file, monkeypatch
):
    call_order = []

    def fake_move_to_positions(robot, targets, kp, control_freq, max_seconds, **kwargs):
        call_order.append("move_to_positions")

    monkeypatch.setattr(run_vla_episode, "move_to_positions", fake_move_to_positions)

    original_disconnect = mock_robot.disconnect

    def tracked_disconnect():
        call_order.append("disconnect")
        original_disconnect()

    mock_robot.disconnect = tracked_disconnect

    class InterruptingActionSource:
        def __init__(self):
            self.calls = 0

        def get_action(self, joint_state, instruction):
            self.calls += 1
            if self.calls == 2:
                raise KeyboardInterrupt
            return dict(joint_state), "scripted-dry-run-v1"

    caps = _fake_caps()
    joint_limits_deg = action_contract.load_joint_limits_deg(mock_calibration_file)
    with IOLogger(tmp_path, CAMERA_NAMES) as io_logger:
        run_episode(
            mock_robot,
            caps,
            CAMERA_NAMES,
            io_logger,
            InterruptingActionSource(),
            instruction="Pick up the red cube",
            max_steps=10,
            control_hz=100.0,
            joint_limits_deg=joint_limits_deg,
        )
    mock_robot.disconnect()  # simulate main()'s outer finally calling disconnect after run_episode

    assert call_order.index("move_to_positions") < call_order.index("disconnect")

    termination = json.loads((tmp_path / "termination.json").read_text())
    assert termination["reason"] == "keyboard_interrupt"


def test_keyboard_interrupt_reason_survives_broken_stdout_pipe(
    tmp_path, mock_robot, mock_calibration_file, monkeypatch
):
    # Ctrl-C also kills a piped reader (`... | grep -v objc`), so the handler's
    # print raises BrokenPipeError (Plan 15-10 Task 3 step D, 2026-10-03). The
    # reason and the return-to-start must not depend on that print.
    moved = []
    monkeypatch.setattr(run_vla_episode, "move_to_positions",
                        lambda robot, targets, kp, control_freq, max_seconds, **kw: moved.append(True))

    def broken_print(*args, **kwargs):
        if args and "Interrupted" in str(args[0]):
            raise BrokenPipeError(32, "Broken pipe")

    monkeypatch.setattr("builtins.print", broken_print)

    class InterruptingActionSource:
        def get_action(self, joint_state, instruction):
            raise KeyboardInterrupt

    joint_limits_deg = action_contract.load_joint_limits_deg(mock_calibration_file)
    with IOLogger(tmp_path, CAMERA_NAMES) as io_logger:
        run_episode(
            mock_robot,
            _fake_caps(),
            CAMERA_NAMES,
            io_logger,
            InterruptingActionSource(),
            instruction="Pick up the red cube",
            max_steps=10,
            control_hz=100.0,
            joint_limits_deg=joint_limits_deg,
        )

    assert moved
    termination = json.loads((tmp_path / "termination.json").read_text())
    assert termination["reason"] == "keyboard_interrupt"


# --- _build_camera_names (Plan 11-05, Task 1/2 prep -- positional relabeling) ---


def test_build_camera_names_assigns_positionally_not_by_raw_index():
    # Plan 11-05 Task 1 found the AR0144 stereo camera at index 0 and the
    # IMX335 wrist camera at index 1 on this session's hardware -- the
    # reverse of the previous fixed {0: "wrist", 1: "overhead"} table. Passing
    # --camera in physical-reality order must relabel correctly.
    assert _build_camera_names([1, 0]) == {1: "wrist", 0: "overhead"}


def test_build_camera_names_default_order_matches_prior_default_table():
    assert _build_camera_names([0, 1]) == {0: "wrist", 1: "overhead"}


def test_build_camera_names_drops_extra_indices_beyond_known_names():
    assert _build_camera_names([0, 1, 2]) == {0: "wrist", 1: "overhead"}


# --- --server-address / --checkpoint bridge selection (Plan 11-04, Task 2) ---


def _point_device_map_at_nonexistent_path(monkeypatch, tmp_path):
    """Keeps bridge-selection tests hermetic regardless of whether a real
    control/device_map.json happens to exist on the machine running these
    tests -- device_map.json resolution is exercised by its own dedicated
    tests below."""
    monkeypatch.setattr(run_vla_episode, "DEVICE_MAP_PATH", tmp_path / "nonexistent_device_map.json")


def test_run_vla_episode_selects_bridge_source_when_server_address_given(monkeypatch, tmp_path):
    calls = {}

    class FakeBridgeClient:
        def __init__(self):
            self.robot = object()
            # Plan 12-03's Gap 1 fix reads client._stereo_camera through to
            # run_episode()'s stereo_camera parameter -- every FakeBridgeClient
            # double needs this attribute even when a given test doesn't
            # assert on it directly.
            self._stereo_camera = object()

        def stop(self):
            calls["stopped"] = True

    fake_client = FakeBridgeClient()

    def fake_connect_bridge(
        server_address,
        checkpoint,
        robot_config,
        task,
        policy_device="cuda",
        stereo_camera_index=1,
        wrist_camera_index=None,
        policy_type="smolvla",
        camera_map=None,
    ):
        calls["connect_bridge_args"] = (server_address, checkpoint, task, policy_device)
        calls["stereo_camera_index"] = stereo_camera_index
        calls["wrist_camera_index"] = wrist_camera_index
        return fake_client

    class FakeBridgeActionSource:
        def __init__(self, client, checkpoint, joint_limits_deg):
            calls["bridge_action_source_client"] = client
            calls["bridge_action_source_checkpoint"] = checkpoint

    def fake_run_episode(
        robot,
        caps,
        camera_names,
        io_logger,
        action_source,
        instruction,
        max_steps,
        control_hz,
        joint_limits_deg,
        **kwargs,
    ):
        # **kwargs absorbs Plan 12-03's new stereo_camera/execution_hz
        # keyword arguments -- main() now always threads them through this
        # call site; this fake's assertions are unaffected.
        calls["action_source_type"] = type(action_source).__name__

    _point_device_map_at_nonexistent_path(monkeypatch, tmp_path)
    monkeypatch.setattr(robot_client, "connect_bridge", fake_connect_bridge)
    monkeypatch.setattr(robot_client, "BridgeActionSource", FakeBridgeActionSource)
    monkeypatch.setattr(run_vla_episode, "run_episode", fake_run_episode)
    monkeypatch.setattr(action_contract, "load_joint_limits_deg", lambda: {})
    monkeypatch.setattr(run_vla_episode.cv2, "VideoCapture", lambda idx: _NeverOpensCapture())
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_vla_episode.py",
            "/dev/fake_port",
            "fake_robot_id",
            "--out",
            str(tmp_path),
            "--camera",
            "0",
            "--server-address",
            "0.tcp.ngrok.io:12345",
            "--checkpoint",
            "victorvanhalst/smolvla_so101_cube",
        ],
    )

    run_vla_episode.main()

    assert calls["action_source_type"] == "FakeBridgeActionSource"
    assert calls["bridge_action_source_client"] is fake_client
    assert calls["bridge_action_source_checkpoint"] == "victorvanhalst/smolvla_so101_cube"
    assert calls["connect_bridge_args"][0] == "0.tcp.ngrok.io:12345"
    assert calls["connect_bridge_args"][1] == "victorvanhalst/smolvla_so101_cube"
    assert calls.get("stopped") is True
    # Default --stereo-camera-index is 1 when neither the CLI nor device_map.json
    # supplies one.
    assert calls["stereo_camera_index"] == 1
    # Single --camera 0 given -> _build_camera_names([0]) == {0: "wrist"}.
    assert calls["wrist_camera_index"] == 0


def test_stereo_camera_index_flag_threads_through_to_connect_bridge(monkeypatch, tmp_path):
    calls = {}

    class FakeBridgeClient:
        def __init__(self):
            self.robot = object()
            # Plan 12-03's Gap 1 fix reads client._stereo_camera through to
            # run_episode()'s stereo_camera parameter -- every FakeBridgeClient
            # double needs this attribute even when a given test doesn't
            # assert on it directly.
            self._stereo_camera = object()

        def stop(self):
            pass

    fake_client = FakeBridgeClient()

    def fake_connect_bridge(
        server_address,
        checkpoint,
        robot_config,
        task,
        policy_device="cuda",
        stereo_camera_index=1,
        wrist_camera_index=None,
        policy_type="smolvla",
        camera_map=None,
    ):
        calls["stereo_camera_index"] = stereo_camera_index
        return fake_client

    _point_device_map_at_nonexistent_path(monkeypatch, tmp_path)
    monkeypatch.setattr(robot_client, "connect_bridge", fake_connect_bridge)
    monkeypatch.setattr(robot_client, "BridgeActionSource", lambda *a, **k: None)
    monkeypatch.setattr(run_vla_episode, "run_episode", lambda *a, **k: None)
    monkeypatch.setattr(action_contract, "load_joint_limits_deg", lambda: {})
    monkeypatch.setattr(run_vla_episode.cv2, "VideoCapture", lambda idx: _NeverOpensCapture())
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_vla_episode.py",
            "/dev/fake_port",
            "fake_robot_id",
            "--out",
            str(tmp_path),
            "--camera",
            "0",
            "--server-address",
            "0.tcp.ngrok.io:12345",
            "--checkpoint",
            "victorvanhalst/smolvla_so101_cube",
            "--stereo-camera-index",
            "0",
        ],
    )

    run_vla_episode.main()

    # --stereo-camera-index is type=str now (accepts a device name, not just a
    # numeric index) -- an explicit "0" arrives as the string "0", not int 0.
    assert calls["stereo_camera_index"] == "0"


def _run_main_capturing_camera_map(monkeypatch, tmp_path, extra_argv):
    calls = {}

    class FakeBridgeClient:
        def __init__(self):
            self.robot = object()
            self._stereo_camera = object()

        def stop(self):
            pass

    def fake_connect_bridge(server_address, checkpoint, robot_config, task, **kwargs):
        calls["camera_map"] = kwargs.get("camera_map")
        calls["policy_type"] = kwargs.get("policy_type")
        return FakeBridgeClient()

    _point_device_map_at_nonexistent_path(monkeypatch, tmp_path)
    monkeypatch.setattr(robot_client, "connect_bridge", fake_connect_bridge)
    monkeypatch.setattr(robot_client, "BridgeActionSource", lambda *a, **k: None)
    monkeypatch.setattr(run_vla_episode, "run_episode", lambda *a, **k: None)
    monkeypatch.setattr(action_contract, "load_joint_limits_deg", lambda: {})
    monkeypatch.setattr(run_vla_episode.cv2, "VideoCapture", lambda idx: _NeverOpensCapture())
    monkeypatch.setattr(
        "sys.argv",
        ["run_vla_episode.py", "/dev/fake_port", "fake_robot_id", "--out", str(tmp_path),
         "--camera", "0", "--server-address", "0.tcp.ngrok.io:12345", *extra_argv],
    )
    run_vla_episode.main()
    return calls


def test_backend_config_camera_map_reaches_connect_bridge(monkeypatch, tmp_path):
    calls = _run_main_capturing_camera_map(
        monkeypatch, tmp_path,
        ["--backend", "groot-n17", "--checkpoint",
         "tsangb34/groot-n17-so101-soccer-red_bowl-40episodes"],
    )

    assert calls["policy_type"] == "groot"
    assert calls["camera_map"] == {"wrist": "wrist", "front": "overhead_left"}


def test_camera_map_flag_overrides_the_backend_config(monkeypatch, tmp_path):
    calls = _run_main_capturing_camera_map(
        monkeypatch, tmp_path,
        ["--backend", "act", "--checkpoint", "Alkatt/ACT_so101_CubePickPlace_v2",
         "--camera-map", "camera1=wrist,camera2=overhead_right"],
    )

    assert calls["camera_map"] == {"camera1": "wrist", "camera2": "overhead_right"}


@pytest.mark.parametrize(
    "argv",
    [
        ["--backend", "act", "--checkpoint", "x/y", "--camera-map", "camera1=elbow"],
        ["--backend", "act", "--checkpoint", "x/y", "--camera-map", "camera1"],
    ],
)
def test_camera_map_flag_rejects_bad_maps_at_parse_time(monkeypatch, tmp_path, argv):
    with pytest.raises(SystemExit):
        _run_main_capturing_camera_map(monkeypatch, tmp_path, argv)


def test_camera_map_flag_is_refused_for_non_bridge_backends(monkeypatch, tmp_path):
    _point_device_map_at_nonexistent_path(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        ["run_vla_episode.py", "/dev/fake_port", "fake_robot_id", "--out", str(tmp_path),
         "--backend", "scripted", "--camera-map", "camera1=wrist"],
    )
    with pytest.raises(SystemExit):
        run_vla_episode.main()


def test_wrist_camera_index_threads_through_from_camera_names(monkeypatch, tmp_path):
    """The camera1 (wrist) wiring gap fix: connect_bridge() must receive
    whichever physical index _build_camera_names() positionally assigned to
    "wrist" -- here, --camera 5 (first) --camera 7 (second) -> wrist=5."""
    calls = {}

    class FakeBridgeClient:
        def __init__(self):
            self.robot = object()
            # Plan 12-03's Gap 1 fix reads client._stereo_camera through to
            # run_episode()'s stereo_camera parameter -- every FakeBridgeClient
            # double needs this attribute even when a given test doesn't
            # assert on it directly.
            self._stereo_camera = object()

        def stop(self):
            pass

    fake_client = FakeBridgeClient()

    def fake_connect_bridge(
        server_address,
        checkpoint,
        robot_config,
        task,
        policy_device="cuda",
        stereo_camera_index=1,
        wrist_camera_index=None,
        policy_type="smolvla",
        camera_map=None,
    ):
        calls["wrist_camera_index"] = wrist_camera_index
        return fake_client

    _point_device_map_at_nonexistent_path(monkeypatch, tmp_path)
    monkeypatch.setattr(robot_client, "connect_bridge", fake_connect_bridge)
    monkeypatch.setattr(robot_client, "BridgeActionSource", lambda *a, **k: None)
    monkeypatch.setattr(run_vla_episode, "run_episode", lambda *a, **k: None)
    monkeypatch.setattr(action_contract, "load_joint_limits_deg", lambda: {})
    monkeypatch.setattr(run_vla_episode.cv2, "VideoCapture", lambda idx: _NeverOpensCapture())
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_vla_episode.py",
            "/dev/fake_port",
            "fake_robot_id",
            "--out",
            str(tmp_path),
            "--camera",
            "5",
            "--camera",
            "7",
            "--server-address",
            "0.tcp.ngrok.io:12345",
            "--checkpoint",
            "victorvanhalst/smolvla_so101_cube",
        ],
    )

    run_vla_episode.main()

    assert calls["wrist_camera_index"] == 5


def test_bridge_unreachable_writes_termination_reason_without_driving_robot(monkeypatch, tmp_path):
    calls = {"run_episode_called": False}

    def fake_connect_bridge_returns_none(
        server_address,
        checkpoint,
        robot_config,
        task,
        policy_device="cuda",
        stereo_camera_index=1,
        wrist_camera_index=None,
        policy_type="smolvla",
        camera_map=None,
    ):
        return None

    def fake_run_episode(*args, **kwargs):
        calls["run_episode_called"] = True

    _point_device_map_at_nonexistent_path(monkeypatch, tmp_path)
    monkeypatch.setattr(robot_client, "connect_bridge", fake_connect_bridge_returns_none)
    monkeypatch.setattr(run_vla_episode, "run_episode", fake_run_episode)
    monkeypatch.setattr(action_contract, "load_joint_limits_deg", lambda: {})
    monkeypatch.setattr(run_vla_episode.cv2, "VideoCapture", lambda idx: _NeverOpensCapture())
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_vla_episode.py",
            "/dev/fake_port",
            "fake_robot_id",
            "--out",
            str(tmp_path),
            "--camera",
            "0",
            "--server-address",
            "0.tcp.ngrok.io:12345",
            "--checkpoint",
            "victorvanhalst/smolvla_so101_cube",
        ],
    )

    run_vla_episode.main()

    assert calls["run_episode_called"] is False
    termination = json.loads((tmp_path / "termination.json").read_text())
    assert termination["reason"] == "bridge_unreachable"
    assert termination["steps_completed"] == 0


def test_checkpoint_without_server_address_errors_clearly(monkeypatch, tmp_path):
    _point_device_map_at_nonexistent_path(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_vla_episode.py",
            "/dev/fake_port",
            "fake_robot_id",
            "--out",
            str(tmp_path),
            "--checkpoint",
            "victorvanhalst/smolvla_so101_cube",
        ],
    )

    with pytest.raises(SystemExit):
        run_vla_episode.main()


# --- device_map.json integration (Plan 11-05 Task 2) --------------------------------


def test_device_map_supplies_port_robot_id_camera_defaults_when_cli_args_omitted(monkeypatch, tmp_path):
    device_map_path = tmp_path / "device_map.json"
    device_map_path.write_text(
        json.dumps(
            {
                "detected_at": "2026-09-22T10:00:00Z",
                "follower": {"port": "/dev/cu.fake999", "id": "soarm_follower_02"},
                "leader": None,
                "cameras": {"wrist": 3, "stereo_overhead": 4},
            }
        )
    )
    monkeypatch.setattr(run_vla_episode, "DEVICE_MAP_PATH", device_map_path)

    calls = {}

    class FakeBridgeClient:
        def __init__(self):
            self.robot = object()
            # Plan 12-03's Gap 1 fix reads client._stereo_camera through to
            # run_episode()'s stereo_camera parameter -- every FakeBridgeClient
            # double needs this attribute even when a given test doesn't
            # assert on it directly.
            self._stereo_camera = object()

        def stop(self):
            calls["stopped"] = True

    fake_client = FakeBridgeClient()

    def fake_connect_bridge(
        server_address,
        checkpoint,
        robot_config,
        task,
        policy_device="cuda",
        stereo_camera_index=1,
        wrist_camera_index=None,
        policy_type="smolvla",
        camera_map=None,
    ):
        calls["port"] = robot_config.port
        calls["robot_id"] = robot_config.id
        calls["stereo_camera_index"] = stereo_camera_index
        calls["wrist_camera_index"] = wrist_camera_index
        return fake_client

    monkeypatch.setattr(robot_client, "connect_bridge", fake_connect_bridge)
    monkeypatch.setattr(robot_client, "BridgeActionSource", lambda *a, **k: None)
    monkeypatch.setattr(run_vla_episode, "run_episode", lambda *a, **k: None)
    monkeypatch.setattr(action_contract, "load_joint_limits_deg", lambda: {})
    monkeypatch.setattr(run_vla_episode.cv2, "VideoCapture", lambda idx: _NeverOpensCapture())
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_vla_episode.py",
            "--out",
            str(tmp_path / "out"),
            "--server-address",
            "0.tcp.ngrok.io:12345",
            "--checkpoint",
            "victorvanhalst/smolvla_so101_cube",
        ],
    )

    run_vla_episode.main()

    assert calls["port"] == "/dev/cu.fake999"
    assert calls["robot_id"] == "soarm_follower_02"
    assert calls["stereo_camera_index"] == 4


def test_device_map_prefers_stereo_overhead_name_over_numeric_index_when_present(monkeypatch, tmp_path):
    """The 11-05 Task 3 fix: cameras.stereo_overhead_name (a device NAME
    string) must win over cameras.stereo_overhead (a numeric cv2 index) when
    both are present -- the numeric index has been confirmed to drift between
    process launches on macOS, the name string has not."""
    device_map_path = tmp_path / "device_map.json"
    device_map_path.write_text(
        json.dumps(
            {
                "detected_at": "2026-09-22T10:00:00Z",
                "follower": {"port": "/dev/cu.fake999", "id": "soarm_follower_02"},
                "leader": None,
                "cameras": {"wrist": 3, "stereo_overhead": 4, "stereo_overhead_name": "CCB Camera"},
            }
        )
    )
    monkeypatch.setattr(run_vla_episode, "DEVICE_MAP_PATH", device_map_path)

    calls = {}

    class FakeBridgeClient:
        def __init__(self):
            self.robot = object()
            # Plan 12-03's Gap 1 fix reads client._stereo_camera through to
            # run_episode()'s stereo_camera parameter -- every FakeBridgeClient
            # double needs this attribute even when a given test doesn't
            # assert on it directly.
            self._stereo_camera = object()

        def stop(self):
            pass

    fake_client = FakeBridgeClient()

    def fake_connect_bridge(
        server_address,
        checkpoint,
        robot_config,
        task,
        policy_device="cuda",
        stereo_camera_index=1,
        wrist_camera_index=None,
        policy_type="smolvla",
        camera_map=None,
    ):
        calls["stereo_camera_index"] = stereo_camera_index
        calls["wrist_camera_index"] = wrist_camera_index
        return fake_client

    monkeypatch.setattr(robot_client, "connect_bridge", fake_connect_bridge)
    monkeypatch.setattr(robot_client, "BridgeActionSource", lambda *a, **k: None)
    monkeypatch.setattr(run_vla_episode, "run_episode", lambda *a, **k: None)
    monkeypatch.setattr(action_contract, "load_joint_limits_deg", lambda: {})
    monkeypatch.setattr(run_vla_episode.cv2, "VideoCapture", lambda idx: _NeverOpensCapture())
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_vla_episode.py",
            "--out",
            str(tmp_path / "out"),
            "--server-address",
            "0.tcp.ngrok.io:12345",
            "--checkpoint",
            "victorvanhalst/smolvla_so101_cube",
        ],
    )

    run_vla_episode.main()

    assert calls["stereo_camera_index"] == "CCB Camera"
    assert calls["wrist_camera_index"] == 3


def test_explicit_cli_args_override_device_map_json(monkeypatch, tmp_path):
    device_map_path = tmp_path / "device_map.json"
    device_map_path.write_text(
        json.dumps(
            {
                "detected_at": "2026-09-22T10:00:00Z",
                "follower": {"port": "/dev/cu.should_not_be_used", "id": "wrong_id"},
                "leader": None,
                "cameras": {"wrist": 99, "stereo_overhead": 98},
            }
        )
    )
    monkeypatch.setattr(run_vla_episode, "DEVICE_MAP_PATH", device_map_path)

    calls = {}

    class FakeBridgeClient:
        def __init__(self):
            self.robot = object()
            # Plan 12-03's Gap 1 fix reads client._stereo_camera through to
            # run_episode()'s stereo_camera parameter -- every FakeBridgeClient
            # double needs this attribute even when a given test doesn't
            # assert on it directly.
            self._stereo_camera = object()

        def stop(self):
            pass

    fake_client = FakeBridgeClient()

    def fake_connect_bridge(
        server_address,
        checkpoint,
        robot_config,
        task,
        policy_device="cuda",
        stereo_camera_index=1,
        wrist_camera_index=None,
        policy_type="smolvla",
        camera_map=None,
    ):
        calls["port"] = robot_config.port
        calls["robot_id"] = robot_config.id
        calls["stereo_camera_index"] = stereo_camera_index
        return fake_client

    monkeypatch.setattr(robot_client, "connect_bridge", fake_connect_bridge)
    monkeypatch.setattr(robot_client, "BridgeActionSource", lambda *a, **k: None)
    monkeypatch.setattr(run_vla_episode, "run_episode", lambda *a, **k: None)
    monkeypatch.setattr(action_contract, "load_joint_limits_deg", lambda: {})
    monkeypatch.setattr(run_vla_episode.cv2, "VideoCapture", lambda idx: _NeverOpensCapture())
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_vla_episode.py",
            "/dev/cu.explicit_port",
            "explicit_robot_id",
            "--camera",
            "0",
            "--out",
            str(tmp_path / "out"),
            "--server-address",
            "0.tcp.ngrok.io:12345",
            "--checkpoint",
            "victorvanhalst/smolvla_so101_cube",
            "--stereo-camera-index",
            "1",
        ],
    )

    run_vla_episode.main()

    assert calls["port"] == "/dev/cu.explicit_port"
    assert calls["robot_id"] == "explicit_robot_id"
    # --stereo-camera-index is type=str now (accepts a device name like "CCB
    # Camera", not just a numeric index -- see the flag's help text), so an
    # explicit "1" arrives as the string "1", not int 1.
    assert calls["stereo_camera_index"] == "1"


def test_missing_port_and_device_map_errors_clearly(monkeypatch, tmp_path):
    _point_device_map_at_nonexistent_path(monkeypatch, tmp_path)
    monkeypatch.setattr("sys.argv", ["run_vla_episode.py", "--out", str(tmp_path / "out")])

    with pytest.raises(SystemExit):
        run_vla_episode.main()


def test_missing_camera_and_device_map_errors_clearly(monkeypatch, tmp_path):
    _point_device_map_at_nonexistent_path(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        ["run_vla_episode.py", "/dev/fake_port", "fake_robot_id", "--out", str(tmp_path / "out")],
    )

    with pytest.raises(SystemExit):
        run_vla_episode.main()


def test_help_still_documents_port_robot_id_camera_as_explicit_overrides(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["run_vla_episode.py", "--help"])

    with pytest.raises(SystemExit) as exc_info:
        run_vla_episode.main()

    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "port" in captured.out
    assert "robot_id" in captured.out
    assert "--camera" in captured.out


# --- _open_cameras (Plan 12-03, Task 1 -- Gap 1 closure) ---------------------


def test_open_cameras_skips_indices_whose_name_is_in_skip_names(monkeypatch):
    _TrackingCapture.constructed_indices = []
    monkeypatch.setattr(run_vla_episode.cv2, "VideoCapture", _TrackingCapture)
    camera_names = {0: "wrist", 1: "overhead"}

    caps = _open_cameras([0, 1], camera_names, skip_names=frozenset({"overhead"}))

    assert _TrackingCapture.constructed_indices == [0]
    assert list(caps.keys()) == [0]


def test_open_cameras_opens_everything_when_skip_names_is_empty(monkeypatch):
    _TrackingCapture.constructed_indices = []
    monkeypatch.setattr(run_vla_episode.cv2, "VideoCapture", _TrackingCapture)
    camera_names = {0: "wrist", 1: "overhead"}

    caps = _open_cameras([0, 1], camera_names)

    assert _TrackingCapture.constructed_indices == [0, 1]
    assert set(caps.keys()) == {0, 1}


def test_server_address_never_opens_overhead_camera_index_while_wrist_still_does(monkeypatch, tmp_path):
    """main()-level integration test: the "overhead"-labeled index must never
    reach cv2.VideoCapture in the --server-address (bridge) path, while the
    wrist index still does (Gap 1)."""
    _TrackingCapture.constructed_indices = []

    class FakeBridgeClient:
        def __init__(self):
            self.robot = object()
            self._stereo_camera = object()

        def stop(self):
            pass

    fake_client = FakeBridgeClient()

    def fake_connect_bridge(
        server_address,
        checkpoint,
        robot_config,
        task,
        policy_device="cuda",
        stereo_camera_index=1,
        wrist_camera_index=None,
        policy_type="smolvla",
        camera_map=None,
    ):
        return fake_client

    _point_device_map_at_nonexistent_path(monkeypatch, tmp_path)
    monkeypatch.setattr(robot_client, "connect_bridge", fake_connect_bridge)
    monkeypatch.setattr(robot_client, "BridgeActionSource", lambda *a, **k: None)
    monkeypatch.setattr(run_vla_episode, "run_episode", lambda *a, **k: None)
    monkeypatch.setattr(action_contract, "load_joint_limits_deg", lambda: {})
    monkeypatch.setattr(run_vla_episode.cv2, "VideoCapture", _TrackingCapture)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_vla_episode.py",
            "/dev/fake_port",
            "fake_robot_id",
            "--out",
            str(tmp_path),
            "--camera",
            "0",
            "--camera",
            "1",
            "--server-address",
            "0.tcp.ngrok.io:12345",
            "--checkpoint",
            "victorvanhalst/smolvla_so101_cube",
        ],
    )

    run_vla_episode.main()

    # _build_camera_names([0, 1]) == {0: "wrist", 1: "overhead"} -- index 1
    # ("overhead") must never reach cv2.VideoCapture; index 0 ("wrist") does.
    assert _TrackingCapture.constructed_indices == [0]


# --- run_episode() camera provenance (Plan 12-03, Task 1 -- Gap 1 closure) ---


def test_run_episode_records_overhead_left_and_right_from_shared_stereo_camera(
    tmp_path, mock_robot, mock_calibration_file
):
    action_source = ScriptedActionSource()
    caps = {0: FakeCamera()}  # only the wrist index -- matches the now-fixed bridge path
    camera_names = {0: "wrist"}
    joint_limits_deg = action_contract.load_joint_limits_deg(mock_calibration_file)
    stereo_camera = _FakeStereoCamera()

    with IOLogger(tmp_path, camera_names) as io_logger:
        run_episode(
            mock_robot,
            caps,
            camera_names,
            io_logger,
            action_source,
            instruction="Pick up the red cube",
            max_steps=2,
            control_hz=100.0,
            joint_limits_deg=joint_limits_deg,
            stereo_camera=stereo_camera,
        )

    lines = (tmp_path / "episode.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    for line in lines:
        record = json.loads(line)
        assert "overhead_left" in record["camera_frames"]
        assert "overhead_right" in record["camera_frames"]
        left_path = record["camera_frames"]["overhead_left"]["path"]
        right_path = record["camera_frames"]["overhead_right"]["path"]
        assert left_path is not None and (tmp_path / left_path).exists()
        assert right_path is not None and (tmp_path / right_path).exists()
        assert left_path.startswith("camera_overhead_left")
        assert right_path.startswith("camera_overhead_right")


def test_run_episode_without_stereo_camera_produces_no_overhead_left_right_keys(
    tmp_path, mock_robot, mock_calibration_file
):
    action_source = ScriptedActionSource()
    caps = _fake_caps()
    joint_limits_deg = action_contract.load_joint_limits_deg(mock_calibration_file)

    with IOLogger(tmp_path, CAMERA_NAMES) as io_logger:
        run_episode(
            mock_robot,
            caps,
            CAMERA_NAMES,
            io_logger,
            action_source,
            instruction="Pick up the red cube",
            max_steps=1,
            control_hz=100.0,
            joint_limits_deg=joint_limits_deg,
        )

    lines = (tmp_path / "episode.jsonl").read_text().strip().splitlines()
    record = json.loads(lines[0])
    assert "overhead_left" not in record["camera_frames"]
    assert "overhead_right" not in record["camera_frames"]


# --- _interpolate_waypoints (Plan 12-03, Task 2 -- Gap 2 closure) -----------


def test_interpolate_waypoints_returns_evenly_spaced_intermediate_values():
    start = {"j": 0.0}
    end = {"j": 10.0}

    waypoints = _interpolate_waypoints(start, end, num_steps=5)

    assert len(waypoints) == 5
    for wp, expected in zip(waypoints[:-1], [2.0, 4.0, 6.0, 8.0]):
        assert wp["j"] == pytest.approx(expected)
    assert waypoints[-1]["j"] == 10.0


def test_interpolate_waypoints_degenerate_single_step_returns_end_exactly():
    start = {"j": 0.0}
    end = {"j": 10.0}

    waypoints = _interpolate_waypoints(start, end, num_steps=1)

    assert waypoints == [end]
    assert waypoints[0]["j"] == end["j"]


# --- run_episode() waypoint interpolation (Plan 12-03, Task 2 -- Gap 2 closure) ---


def test_run_episode_sends_multiple_interpolated_waypoints_per_tick_when_execution_hz_exceeds_control_hz(
    tmp_path, mock_robot, mock_calibration_file, monkeypatch
):
    # move_to_positions() (the end-of-episode return-to-start move) also
    # calls robot.send_action() at least once -- stub it out so this test's
    # sent_actions count reflects only the main control loop's sends.
    monkeypatch.setattr(run_vla_episode, "move_to_positions", lambda *a, **k: None)
    action_source = ScriptedActionSource()
    caps = _fake_caps()
    joint_limits_deg = action_contract.load_joint_limits_deg(mock_calibration_file)

    with IOLogger(tmp_path, CAMERA_NAMES) as io_logger:
        run_episode(
            mock_robot,
            caps,
            CAMERA_NAMES,
            io_logger,
            action_source,
            instruction="Pick up the red cube",
            max_steps=2,
            control_hz=10.0,
            joint_limits_deg=joint_limits_deg,
            execution_hz=100.0,
        )

    # control_period / execution_period == 0.1 / 0.01 == 10 substeps per tick.
    assert len(mock_robot.sent_actions) == 20

    lines = (tmp_path / "episode.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    for tick_idx, line in enumerate(lines):
        record = json.loads(line)
        expected_target = {f"{j}.pos": v for j, v in record["validated_action"].items()}
        last_send_idx = (tick_idx + 1) * 10 - 1
        assert mock_robot.sent_actions[last_send_idx] == expected_target


def test_run_episode_default_execution_hz_sends_exactly_one_action_per_tick(
    tmp_path, mock_robot, mock_calibration_file, monkeypatch
):
    """Regression: execution_hz omitted (default 100.0) with the existing
    fast-test control_hz=100.0 convention (execution_hz <= control_hz) must
    send exactly ONE robot.send_action() call per tick -- identical to
    pre-this-plan behavior."""
    # move_to_positions() (the end-of-episode return-to-start move) also
    # calls robot.send_action() at least once -- stub it out so this test's
    # sent_actions count reflects only the main control loop's sends.
    monkeypatch.setattr(run_vla_episode, "move_to_positions", lambda *a, **k: None)
    action_source = ScriptedActionSource()
    caps = _fake_caps()
    joint_limits_deg = action_contract.load_joint_limits_deg(mock_calibration_file)

    with IOLogger(tmp_path, CAMERA_NAMES) as io_logger:
        run_episode(
            mock_robot,
            caps,
            CAMERA_NAMES,
            io_logger,
            action_source,
            instruction="Pick up the red cube",
            max_steps=3,
            control_hz=100.0,
            joint_limits_deg=joint_limits_deg,
        )

    assert len(mock_robot.sent_actions) == 3


# --- depth recording (Plan 12-06, Task 3 -- DEPTH-CAL-03) --------------------


class FakeStereoCameraCountingReads:
    """Stand-in for `StereoSplitCamera` that returns fixed left/right frames
    and counts how many times each read method is called -- confirms the
    per-tick stereo device is read exactly once, shared between the
    diagnostic recording and depth computation."""

    def __init__(self, left_frame, right_frame):
        self.left_frame = left_frame
        self.right_frame = right_frame
        self.read_left_calls = 0
        self.read_right_calls = 0

    def read_left(self):
        self.read_left_calls += 1
        return self.left_frame

    def read_right(self):
        self.read_right_calls += 1
        return self.right_frame


class FakeDepthClient:
    """Stand-in for `DepthCameraClient` -- records each call's arguments and
    returns a fixed depth map."""

    def __init__(self, depth_map):
        self.depth_map = depth_map
        self.compute_depth_calls = []

    def compute_depth(self, left, right):
        self.compute_depth_calls.append((left, right))
        return self.depth_map


def test_run_episode_records_depth_frame_when_depth_client_given(tmp_path, mock_robot, mock_calibration_file):
    action_source = ScriptedActionSource()
    caps = {0: FakeCamera()}
    camera_names = {0: "wrist"}
    joint_limits_deg = action_contract.load_joint_limits_deg(mock_calibration_file)
    left_frame = np.full((2, 2, 3), 1, dtype=np.uint8)
    right_frame = np.full((2, 2, 3), 2, dtype=np.uint8)
    stereo_camera = FakeStereoCameraCountingReads(left_frame, right_frame)
    depth_map = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    depth_client = FakeDepthClient(depth_map)

    with IOLogger(tmp_path, camera_names) as io_logger:
        run_episode(
            mock_robot,
            caps,
            camera_names,
            io_logger,
            action_source,
            instruction="Pick up the red cube",
            max_steps=1,
            control_hz=100.0,
            joint_limits_deg=joint_limits_deg,
            stereo_camera=stereo_camera,
            depth_client=depth_client,
            depth_every_n_steps=1,
        )

    lines = (tmp_path / "episode.jsonl").read_text().strip().splitlines()
    record = json.loads(lines[0])
    manifest = json.loads((tmp_path / record["depth_frames"]["overhead"]["result_path"]).read_text())
    assert manifest["status"] == "complete"
    depth_path = manifest["depth"]["path"]
    assert depth_path is not None
    loaded = np.load(tmp_path / depth_path)
    np.testing.assert_array_equal(loaded, depth_map)


def test_run_episode_depth_cadence_gate_skips_non_matching_ticks(tmp_path, mock_robot, mock_calibration_file):
    action_source = ScriptedActionSource()
    caps = {0: FakeCamera()}
    camera_names = {0: "wrist"}
    joint_limits_deg = action_contract.load_joint_limits_deg(mock_calibration_file)
    left_frame = np.full((2, 2, 3), 1, dtype=np.uint8)
    right_frame = np.full((2, 2, 3), 2, dtype=np.uint8)
    stereo_camera = FakeStereoCameraCountingReads(left_frame, right_frame)
    depth_map = np.array([[1.0]], dtype=np.float32)
    depth_client = FakeDepthClient(depth_map)

    with IOLogger(tmp_path, camera_names) as io_logger:
        run_episode(
            mock_robot,
            caps,
            camera_names,
            io_logger,
            action_source,
            instruction="Pick up the red cube",
            max_steps=2,
            control_hz=100.0,
            joint_limits_deg=joint_limits_deg,
            stereo_camera=stereo_camera,
            depth_client=depth_client,
            depth_every_n_steps=2,
        )

    # Only step i=0 satisfies 0 % 2 == 0 -- step i=1 (1 % 2 == 1) is skipped.
    assert len(depth_client.compute_depth_calls) == 1


def test_run_episode_without_depth_client_produces_empty_depth_frames_and_no_directory(
    tmp_path, mock_robot, mock_calibration_file
):
    action_source = ScriptedActionSource()
    caps = {0: FakeCamera()}
    camera_names = {0: "wrist"}
    joint_limits_deg = action_contract.load_joint_limits_deg(mock_calibration_file)
    left_frame = np.full((2, 2, 3), 1, dtype=np.uint8)
    right_frame = np.full((2, 2, 3), 2, dtype=np.uint8)
    stereo_camera = FakeStereoCameraCountingReads(left_frame, right_frame)

    with IOLogger(tmp_path, camera_names) as io_logger:
        run_episode(
            mock_robot,
            caps,
            camera_names,
            io_logger,
            action_source,
            instruction="Pick up the red cube",
            max_steps=1,
            control_hz=100.0,
            joint_limits_deg=joint_limits_deg,
            stereo_camera=stereo_camera,
        )

    lines = (tmp_path / "episode.jsonl").read_text().strip().splitlines()
    record = json.loads(lines[0])
    assert record["depth_frames"] == {}
    assert not (tmp_path / "depth_overhead").exists()


def test_run_episode_reads_stereo_camera_exactly_once_per_tick_and_shares_frame_with_depth_client(
    tmp_path, mock_robot, mock_calibration_file
):
    action_source = ScriptedActionSource()
    caps = {0: FakeCamera()}
    camera_names = {0: "wrist"}
    joint_limits_deg = action_contract.load_joint_limits_deg(mock_calibration_file)
    left_frame = np.full((2, 2, 3), 1, dtype=np.uint8)
    right_frame = np.full((2, 2, 3), 2, dtype=np.uint8)
    stereo_camera = FakeStereoCameraCountingReads(left_frame, right_frame)
    depth_map = np.array([[1.0]], dtype=np.float32)
    depth_client = FakeDepthClient(depth_map)

    with IOLogger(tmp_path, camera_names) as io_logger:
        run_episode(
            mock_robot,
            caps,
            camera_names,
            io_logger,
            action_source,
            instruction="Pick up the red cube",
            max_steps=1,
            control_hz=100.0,
            joint_limits_deg=joint_limits_deg,
            stereo_camera=stereo_camera,
            depth_client=depth_client,
            depth_every_n_steps=1,
        )

    assert stereo_camera.read_left_calls == 1
    assert stereo_camera.read_right_calls == 1
    assert len(depth_client.compute_depth_calls) == 1
    called_left, called_right = depth_client.compute_depth_calls[0]
    np.testing.assert_array_equal(called_left, left_frame)
    np.testing.assert_array_equal(called_right, right_frame)
    assert not np.shares_memory(called_left, left_frame)
    assert not np.shares_memory(called_right, right_frame)


def test_main_constructs_and_threads_depth_client_when_both_depth_flags_given(monkeypatch, tmp_path):
    calls = {}

    class FakeBridgeClient:
        def __init__(self):
            self.robot = object()
            self._stereo_camera = object()

        def stop(self):
            pass

    fake_client = FakeBridgeClient()

    def fake_connect_bridge(
        server_address,
        checkpoint,
        robot_config,
        task,
        policy_device="cuda",
        stereo_camera_index=1,
        wrist_camera_index=None,
        policy_type="smolvla",
        camera_map=None,
    ):
        return fake_client

    def fake_run_episode(
        robot,
        caps,
        camera_names,
        io_logger,
        action_source,
        instruction,
        max_steps,
        control_hz,
        joint_limits_deg,
        **kwargs,
    ):
        calls["depth_client"] = kwargs.get("depth_client")
        calls["depth_every_n_steps"] = kwargs.get("depth_every_n_steps")

    fake_calibration = object()

    def fake_load_calibration(path):
        calls["load_calibration_path"] = path
        return fake_calibration

    class FakeDepthCameraClient:
        calibration_metadata = {}

        def check_ready(self):
            calls["depth_ready"] = True

        def __init__(self, calibration, endpoint_url):
            calls["depth_camera_client_calibration"] = calibration
            calls["depth_camera_client_endpoint"] = endpoint_url

    _point_device_map_at_nonexistent_path(monkeypatch, tmp_path)
    monkeypatch.setattr(robot_client, "connect_bridge", fake_connect_bridge)
    monkeypatch.setattr(robot_client, "BridgeActionSource", lambda *a, **k: None)
    monkeypatch.setattr(run_vla_episode, "run_episode", fake_run_episode)
    monkeypatch.setattr(run_vla_episode, "load_calibration", fake_load_calibration)
    monkeypatch.setattr(run_vla_episode, "DepthCameraClient", FakeDepthCameraClient)
    monkeypatch.setattr(action_contract, "load_joint_limits_deg", lambda: {})
    monkeypatch.setattr(run_vla_episode.cv2, "VideoCapture", lambda idx: _NeverOpensCapture())
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_vla_episode.py",
            "/dev/fake_port",
            "fake_robot_id",
            "--out",
            str(tmp_path),
            "--camera",
            "0",
            "--server-address",
            "0.tcp.ngrok.io:12345",
            "--checkpoint",
            "victorvanhalst/smolvla_so101_cube",
            "--depth-endpoint",
            "https://fake.ngrok.io/depth",
            "--depth-calibration",
            str(tmp_path / "stereo_calibration.json"),
        ],
    )

    run_vla_episode.main()

    assert calls["load_calibration_path"] == tmp_path / "stereo_calibration.json"
    assert calls["depth_camera_client_calibration"] is fake_calibration
    assert calls["depth_camera_client_endpoint"] == "https://fake.ngrok.io/depth"
    assert isinstance(calls["depth_client"], FakeDepthCameraClient)
    assert calls["depth_every_n_steps"] == 10
    assert calls["depth_ready"]


def test_main_errors_when_only_depth_endpoint_given_without_calibration(monkeypatch, tmp_path):
    _point_device_map_at_nonexistent_path(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_vla_episode.py",
            "/dev/fake_port",
            "fake_robot_id",
            "--out",
            str(tmp_path),
            "--camera",
            "0",
            "--server-address",
            "0.tcp.ngrok.io:12345",
            "--checkpoint",
            "victorvanhalst/smolvla_so101_cube",
            "--depth-endpoint",
            "https://fake.ngrok.io/depth",
        ],
    )

    with pytest.raises(SystemExit):
        run_vla_episode.main()


def test_main_errors_when_depth_flags_given_without_server_address(monkeypatch, tmp_path, capsys):
    _point_device_map_at_nonexistent_path(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_vla_episode.py",
            "/dev/fake_port",
            "fake_robot_id",
            "--out",
            str(tmp_path),
            "--camera",
            "0",
            "--depth-endpoint",
            "https://fake.ngrok.io/depth",
            "--depth-calibration",
            str(tmp_path / "stereo_calibration.json"),
        ],
    )

    with pytest.raises(SystemExit):
        run_vla_episode.main()
    assert "--server-address" in capsys.readouterr().err


# --- Plan 14-02: --backend flag and BACKEND_REGISTRY dispatch ------------------


class _FakeBackendRobot:
    def __init__(self):
        self.disconnected = False

    def disconnect(self):
        self.disconnected = True


class _ClosableActionSource:
    def __init__(self):
        self.closed = False

    def get_action(self, joint_state, instruction):
        return dict(joint_state), "fake"

    def close(self):
        self.closed = True


class _ReleasableStereo:
    def __init__(self):
        self.released = False

    def release(self):
        self.released = True


def _fake_builder(built, name, handles="default"):
    """A fake `build()` that records which registry entry was dispatched to,
    plus how many cameras `main()` had opened by the time it ran."""

    def build(args, camera_names, joint_limits_deg):
        built.append(name)
        built_state[name] = dict(
            backend_arg=args.backend,
            cameras_opened_before_build=list(_TrackingCapture.constructed_indices),
        )
        if handles == "default":
            return BackendHandles(action_source=_ClosableActionSource(), robot=_FakeBackendRobot())
        return handles

    built_state = {}
    build.state = built_state
    return build


def _dispatch_main(monkeypatch, tmp_path, cli_args, builders):
    """Run main() with every external seam faked. `builders` maps a backend
    name to a fake build(), installed into the real BACKEND_REGISTRY for this
    test only (monkeypatch.setitem restores it afterwards)."""
    calls = {"run_episode_called": False}
    _point_device_map_at_nonexistent_path(monkeypatch, tmp_path)
    for name, build in builders.items():
        monkeypatch.setitem(vla_backends.BACKEND_REGISTRY, name, build)
    monkeypatch.setattr(action_contract, "load_joint_limits_deg", lambda: {})
    monkeypatch.setattr(run_vla_episode.cv2, "VideoCapture", _TrackingCapture)
    _TrackingCapture.constructed_indices = []

    def fake_run_episode(robot, caps, camera_names, io_logger, action_source, *args, **kwargs):
        calls.update(
            run_episode_called=True,
            robot=robot,
            caps=dict(caps),
            action_source=action_source,
            stereo_camera=kwargs.get("stereo_camera"),
            depth_client=kwargs.get("depth_client"),
        )

    monkeypatch.setattr(run_vla_episode, "run_episode", fake_run_episode)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_vla_episode.py", "/dev/fake_port", "fake_robot_id",
            "--out", str(tmp_path / "out"), "--camera", "0", "--camera", "1",
            *cli_args,
        ],
    )
    run_vla_episode.main()
    return calls


BRIDGE_ARGS = ["--server-address", "0.tcp.ngrok.io:12345", "--checkpoint", "victorvanhalst/smolvla_so101_cube"]


def test_omitted_backend_without_server_address_dispatches_to_scripted(monkeypatch, tmp_path):
    built = []
    calls = _dispatch_main(monkeypatch, tmp_path, [], {
        "scripted": _fake_builder(built, "scripted"),
        "smolvla": _fake_builder(built, "smolvla"),
    })
    assert built == ["scripted"]
    assert calls["run_episode_called"] is True


def test_omitted_backend_with_bridge_args_dispatches_to_smolvla(monkeypatch, tmp_path):
    """Backward-compat shim: every pre-Phase-14 `--server-address X
    --checkpoint Y` invocation keeps selecting SmolVLA."""
    built = []
    smolvla = _fake_builder(built, "smolvla")
    _dispatch_main(monkeypatch, tmp_path, BRIDGE_ARGS, {
        "scripted": _fake_builder(built, "scripted"),
        "smolvla": smolvla,
    })
    assert built == ["smolvla"]
    assert smolvla.state["smolvla"]["backend_arg"] == "smolvla"


@pytest.mark.parametrize("extra_args", [[], BRIDGE_ARGS], ids=["no-bridge-args", "with-bridge-args"])
def test_explicit_backend_is_dispatched_regardless_of_server_address(monkeypatch, tmp_path, extra_args):
    built = []
    _dispatch_main(monkeypatch, tmp_path, ["--backend", "fake-explicit", *extra_args], {
        "scripted": _fake_builder(built, "scripted"),
        "smolvla": _fake_builder(built, "smolvla"),
        "fake-explicit": _fake_builder(built, "fake-explicit"),
    })
    assert built == ["fake-explicit"]


@pytest.mark.parametrize(
    "partial_args",
    [["--checkpoint", "victorvanhalst/smolvla_so101_cube"], ["--server-address", "0.tcp.ngrok.io:12345"]],
    ids=["checkpoint-only", "server-address-only"],
)
@pytest.mark.parametrize("backend_args", [[], ["--backend", "smolvla"]], ids=["implicit", "explicit-smolvla"])
def test_partial_bridge_args_error_for_grpc_backends(monkeypatch, tmp_path, partial_args, backend_args):
    built = []
    with pytest.raises(SystemExit):
        _dispatch_main(monkeypatch, tmp_path, [*backend_args, *partial_args], {
            "scripted": _fake_builder(built, "scripted"),
            "smolvla": _fake_builder(built, "smolvla"),
        })
    assert built == []


@pytest.mark.parametrize(
    "partial_args",
    [["--checkpoint", "some/ckpt"], ["--server-address", "0.tcp.ngrok.io:12345"]],
    ids=["checkpoint-only", "server-address-only"],
)
def test_partial_bridge_args_do_not_error_for_non_grpc_backends(monkeypatch, tmp_path, partial_args):
    built = []
    _dispatch_main(monkeypatch, tmp_path, ["--backend", "fake-plain", *partial_args], {
        "fake-plain": _fake_builder(built, "fake-plain"),
    })
    assert built == ["fake-plain"]
    assert "fake-plain" not in vla_backends.GRPC_BRIDGE_BACKENDS


def test_unknown_backend_is_rejected_by_argparse_before_any_construction(monkeypatch, tmp_path):
    built = []
    with pytest.raises(SystemExit):
        _dispatch_main(monkeypatch, tmp_path, ["--backend", "not-a-backend"], {
            "scripted": _fake_builder(built, "scripted"),
        })
    assert built == []


def test_backend_returning_none_writes_bridge_unreachable_before_opening_logger(monkeypatch, tmp_path):
    """The unreachable path now applies to ANY backend whose build() returns
    None, not only the old --server-address branch."""
    built = []
    calls = _dispatch_main(monkeypatch, tmp_path, ["--backend", "fake-down"], {
        "fake-down": _fake_builder(built, "fake-down", handles=None),
    })
    out = tmp_path / "out"
    assert built == ["fake-down"]
    assert calls["run_episode_called"] is False
    termination = json.loads((out / "termination.json").read_text())
    assert termination == {**termination, "reason": "bridge_unreachable", "steps_completed": 0}
    assert not (out / "episode.jsonl").exists()
    assert _TrackingCapture.constructed_indices == []


def test_cameras_open_after_backend_build_honouring_its_skip_names(monkeypatch, tmp_path):
    built = []
    robot = _FakeBackendRobot()
    stereo = _ReleasableStereo()
    handles = BackendHandles(
        action_source=_ClosableActionSource(),
        robot=robot,
        stereo_camera=stereo,
        skip_camera_names=frozenset({"overhead"}),
    )
    builder = _fake_builder(built, "fake-skip", handles=handles)
    calls = _dispatch_main(monkeypatch, tmp_path, ["--backend", "fake-skip"], {"fake-skip": builder})

    assert builder.state["fake-skip"]["cameras_opened_before_build"] == []
    # --camera 0 --camera 1 -> {0: "wrist", 1: "overhead"}; overhead is skipped.
    assert _TrackingCapture.constructed_indices == [0]
    assert list(calls["caps"]) == [0]
    assert calls["robot"] is robot
    assert calls["action_source"] is handles.action_source
    assert calls["stereo_camera"] is stereo


def test_backend_without_shutdown_hook_is_disconnected_and_closed(monkeypatch, tmp_path):
    built = []
    robot = _FakeBackendRobot()
    source = _ClosableActionSource()
    stereo = _ReleasableStereo()
    handles = BackendHandles(action_source=source, robot=robot, stereo_camera=stereo)
    _dispatch_main(monkeypatch, tmp_path, ["--backend", "fake-http"], {
        "fake-http": _fake_builder(built, "fake-http", handles=handles),
    })
    assert robot.disconnected is True
    assert source.closed is True
    assert stereo.released is True


def test_backend_shutdown_hook_replaces_robot_disconnect(monkeypatch, tmp_path):
    """gRPC-lane backends own a RobotClient whose stop() disconnects the robot
    and closes the channel; main() must call that instead of a bare disconnect."""
    built = []
    order = []
    robot = _FakeBackendRobot()
    stereo = _ReleasableStereo()
    stereo.release = lambda: order.append("stereo_release")
    handles = BackendHandles(
        action_source=_ClosableActionSource(),
        robot=robot,
        stereo_camera=stereo,
        shutdown=lambda: order.append("shutdown"),
    )
    _dispatch_main(monkeypatch, tmp_path, ["--backend", "fake-grpc"], {
        "fake-grpc": _fake_builder(built, "fake-grpc", handles=handles),
    })
    assert robot.disconnected is False
    assert order == ["shutdown", "stereo_release"]


def test_explicit_grpc_backend_without_any_bridge_args_errors(monkeypatch, tmp_path):
    """`--backend smolvla` alone passes both pairing checks (neither flag is
    given) but has no bridge to dial -- fail at parse time instead."""
    built = []
    with pytest.raises(SystemExit):
        _dispatch_main(monkeypatch, tmp_path, ["--backend", "smolvla"], {
            "smolvla": _fake_builder(built, "smolvla"),
        })
    assert built == []


def test_bridge_path_end_to_end_through_the_real_smolvla_registry_entry(monkeypatch, tmp_path):
    """No fake builder: the real smolvla build() -> base_grpc -> (faked)
    robot_client.connect_bridge, asserting the handshake asks for smolvla and
    that RobotClient.stop() runs at teardown."""
    calls = {}

    class FakeClient:
        robot = object()
        _stereo_camera = None

        def stop(self):
            calls["stopped"] = True

    def fake_connect_bridge(server_address, checkpoint, **kwargs):
        calls.update(server_address=server_address, checkpoint=checkpoint, **kwargs)
        return FakeClient()

    monkeypatch.setattr(robot_client, "connect_bridge", fake_connect_bridge)
    calls.update(_dispatch_main(monkeypatch, tmp_path, BRIDGE_ARGS, {}))

    assert calls["policy_type"] == "smolvla"
    assert calls["server_address"] == "0.tcp.ngrok.io:12345"
    assert calls["checkpoint"] == "victorvanhalst/smolvla_so101_cube"
    assert isinstance(calls["action_source"], robot_client.BridgeActionSource)
    assert list(calls["caps"]) == [0]  # "overhead" (index 1) is skipped
    assert calls["stopped"] is True


# --- Plan 15-03: episode_complete_reason termination hook (D-18, Pattern 8) ------------


class _CompletingActionSource:
    """Echoes the live joint state; sets `episode_complete_reason` on call `complete_on`."""

    def __init__(self, complete_on=None, value="task_complete"):
        self.complete_on = complete_on
        self.value = value
        self.calls = 0

    def get_action(self, joint_state, instruction):
        if self.calls == self.complete_on:
            self.episode_complete_reason = self.value
        self.calls += 1
        return dict(joint_state), "fake-completing"


def _run_hook_episode(tmp_path, mock_robot, mock_calibration_file, action_source, max_steps=4):
    joint_limits_deg = action_contract.load_joint_limits_deg(mock_calibration_file)
    with IOLogger(tmp_path, CAMERA_NAMES) as io_logger:
        run_episode(
            mock_robot,
            _fake_caps(),
            CAMERA_NAMES,
            io_logger,
            action_source,
            instruction="Pick up the pen",
            max_steps=max_steps,
            control_hz=100.0,
            joint_limits_deg=joint_limits_deg,
        )
    lines = (tmp_path / "episode.jsonl").read_text().strip().splitlines()
    termination = json.loads((tmp_path / "termination.json").read_text())
    return [json.loads(line) for line in lines], termination


def test_run_episode_ends_when_source_sets_episode_complete_reason(
    tmp_path, mock_robot, mock_calibration_file, monkeypatch
):
    returned = []
    real_move = run_vla_episode.move_to_positions

    def tracking_move(robot, targets, **kwargs):
        returned.append(dict(targets))
        return real_move(robot, targets, **kwargs)

    monkeypatch.setattr(run_vla_episode, "move_to_positions", tracking_move)
    source = _CompletingActionSource(complete_on=1)
    records, termination = _run_hook_episode(tmp_path, mock_robot, mock_calibration_file, source)

    assert len(records) == 2
    assert termination == {"reason": "task_complete", "steps_completed": 2}
    assert source.calls == 2
    assert len(returned) == 1  # the return-to-start finally block still ran


@pytest.mark.parametrize(
    "value",
    ["absent", None, "", object(), 42],
    ids=["no-attribute", "none", "empty-string", "object", "int"],
)
def test_run_episode_ignores_absent_or_non_string_episode_complete_reason(
    value, tmp_path, mock_robot, mock_calibration_file
):
    if isinstance(value, str) and value == "absent":
        source = _CompletingActionSource(complete_on=None)
    else:
        source = _CompletingActionSource(complete_on=0, value=value)
    records, termination = _run_hook_episode(tmp_path, mock_robot, mock_calibration_file, source)
    assert len(records) == 4
    assert termination == {"reason": "max_steps_reached", "steps_completed": 4}


def _mllm_source(adapter, mock_calibration_file, chunk_len, cap=3):
    from fake_providers import FakeCollector
    from vla_bridge.backends.base_mllm import MllmActionSourceBase, load_mllm_settings

    settings = load_mllm_settings()
    settings["chunk_len"] = chunk_len
    settings["max_consecutive_failed_calls"] = cap
    return MllmActionSourceBase(
        adapter, FakeCollector(), settings,
        action_contract.load_joint_limits_deg(mock_calibration_file), "mllm-fake",
    )


def test_mllm_task_complete_ends_real_run_episode_after_one_held_step(
    tmp_path, mock_robot, mock_calibration_file
):
    from fake_providers import ScriptedAdapter, make_chunk_doc

    doc = make_chunk_doc(3, base=5.0, task_complete=True)
    adapter = ScriptedAdapter([json.dumps(doc)])
    source = _mllm_source(adapter, mock_calibration_file, chunk_len=3)
    records, termination = _run_hook_episode(
        tmp_path, mock_robot, mock_calibration_file, source, max_steps=10
    )

    assert termination == {"reason": "task_complete", "steps_completed": 1}
    assert len(records) == 1
    record = records[0]
    assert record["action_details"]["mllm_call"]["trace"]["task_complete"] is True
    assert record["model_version"] == "mllm-fake@task-complete-holding-position"
    assert record["raw_model_output"] == record["joint_state"]  # held, not the model's step
    assert record["executed_action"] == record["joint_state"]
    assert record["executed_action"] != doc["actions"][0]
    assert adapter.calls == 1


def test_mllm_backend_failure_cap_ends_real_run_episode(tmp_path, mock_robot, mock_calibration_file):
    from fake_providers import ScriptedAdapter

    adapter = ScriptedAdapter([RuntimeError("upstream 500") for _ in range(5)])
    source = _mllm_source(adapter, mock_calibration_file, chunk_len=2, cap=3)
    records, termination = _run_hook_episode(
        tmp_path, mock_robot, mock_calibration_file, source, max_steps=20
    )

    assert termination == {"reason": "mllm_backend_failed", "steps_completed": 5}
    assert len(records) == 5
    assert adapter.calls == 3
    for record in records:
        assert record["model_version"] == "mllm-fake@provider-error-holding-position"
        assert record["executed_action"] == record["joint_state"]


# --- Plan 15-03: depth flags scoped away from MLLM backends (D-06, research C5) ----------

DEPTH_URL = "https://fake.ngrok.io/depth"
FAKE_MLLM = "mllm-fake-depth-owner"


def _depth_args(tmp_path, *, calibration=True):
    args = ["--depth-endpoint", DEPTH_URL]
    if calibration:
        args += ["--depth-calibration", str(tmp_path / "stereo_calibration.json")]
    return args


@pytest.mark.parametrize(
    "backend_args",
    [[], ["--backend", "scripted"], ["--backend", "fake-plain"], ["--backend", "smolvla"]],
    ids=["default", "explicit-scripted", "plain-non-mllm", "explicit-grpc"],
)
def test_depth_endpoint_without_server_address_errors_for_non_mllm_backends(
    backend_args, monkeypatch, tmp_path, capsys
):
    built = []
    with pytest.raises(SystemExit):
        _dispatch_main(monkeypatch, tmp_path, [*backend_args, *_depth_args(tmp_path)], {
            "scripted": _fake_builder(built, "scripted"),
            "smolvla": _fake_builder(built, "smolvla"),
            "fake-plain": _fake_builder(built, "fake-plain"),
        })
    assert built == []
    assert "--server-address" in capsys.readouterr().err


def _register_fake_mllm(monkeypatch, built):
    from vla_bridge.backends import base_mllm

    seen = {}

    def build(args, camera_names, joint_limits_deg):
        built.append(FAKE_MLLM)
        seen.update(depth_endpoint=args.depth_endpoint, depth_calibration=args.depth_calibration)
        return BackendHandles(action_source=_ClosableActionSource(), robot=_FakeBackendRobot())

    monkeypatch.setitem(
        base_mllm.MLLM_BACKENDS, FAKE_MLLM,
        base_mllm.MllmBackendSpec(FAKE_MLLM, "mllm_defaults", lambda api_key, config: None),
    )
    return build, seen


def test_mllm_backend_accepts_depth_flags_without_server_address_and_owns_its_depth_client(
    monkeypatch, tmp_path
):
    def must_not_build(*args, **kwargs):
        pytest.fail("main() must not build a DepthCameraClient for an MLLM backend (FastFS 429)")

    monkeypatch.setattr(run_vla_episode, "DepthCameraClient", must_not_build)
    monkeypatch.setattr(run_vla_episode, "load_calibration", must_not_build)
    built = []
    build, seen = _register_fake_mllm(monkeypatch, built)

    calls = _dispatch_main(monkeypatch, tmp_path, ["--backend", FAKE_MLLM, *_depth_args(tmp_path)], {
        FAKE_MLLM: build,
    })

    assert built == [FAKE_MLLM]
    assert seen == {
        "depth_endpoint": DEPTH_URL,
        "depth_calibration": tmp_path / "stereo_calibration.json",
    }
    assert calls["run_episode_called"] is True
    assert calls["depth_client"] is None
    assert not (tmp_path / "out" / "depth_calibration.json").exists()


def test_mllm_backend_still_requires_depth_flags_together(monkeypatch, tmp_path, capsys):
    built = []
    build, _seen = _register_fake_mllm(monkeypatch, built)
    with pytest.raises(SystemExit):
        _dispatch_main(monkeypatch, tmp_path,
                       ["--backend", FAKE_MLLM, *_depth_args(tmp_path, calibration=False)],
                       {FAKE_MLLM: build})
    assert built == []
    assert "must be given together" in capsys.readouterr().err
