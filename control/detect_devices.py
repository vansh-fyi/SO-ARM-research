"""Auto-discovery of follower/leader serial ports and camera indices (VLAHW hardware
task, Plan 11-05 Task 2).

Supersedes `control/COMMANDS.md`'s hardcoded port/camera-index table -- live testing
this session found both serial ports AND camera cv2 indices drift across
sessions/replugs on this rig, and the laptop's own built-in webcam was silently being
selected as a real camera by `run_vla_episode.py`'s old `_probe_camera_indices()`
(no identity check at all).

Two independent detection problems, solved differently:

1. Serial port -> robot identity: NOT solved by fixed port-string matching (this
   session confirmed that drifts). Instead, for each candidate `/dev/cu.usbmodem*`
   port, this script connects and reads the LIVE calibration values off the
   servos, then compares them against each known robot id's saved calibration
   file -- a match is the signal, not the port string itself, and this check
   is READ-ONLY (never writes to a servo). See `resolve_port_identity()`.
   (An earlier version of this check wrote each candidate id's saved
   calibration onto the port being tested and treated "no exception" as
   proof of identity -- confirmed unsafe live during 11-05 Task 3 prep: a
   robot-class/id combination with no saved calibration file loads an EMPTY
   calibration, so the "test" write was silently a no-op that raised nothing,
   letting a wrong id "resolve" without ever proving identity. Had that id
   happened to have a real saved file for a DIFFERENT physical robot, the
   write would have silently pushed that robot's homing offsets/position
   limits onto this port's real servos.)

2. Camera identity: `system_profiler SPCameraDataType` names devices, but macOS does
   NOT guarantee its enumeration order matches cv2's ascending `VideoCapture` index
   order -- so name<->index correlation here is a best-effort positional heuristic
   (confirmed to hold on this rig this session with all cameras connected), used only
   to EXCLUDE the built-in camera by name ("FaceTime" in its name). Among the
   remaining candidates, the AR0144 stereo camera is identified by its distinctive
   2560x720 capture resolution (unique on this rig); if a second non-stereo
   candidate remains ambiguous after that, this script falls back to an interactive
   cover-the-lens brightness check -- the same manual technique used live this
   session -- rather than guessing. See `resolve_cameras()`.
   (Found live during 11-05 Task 3 prep: `cv2.VideoCapture`'s reported resolution
   for the AR0144 is UNRELIABLE on macOS -- a known unfixed OpenCV AVFoundation-
   backend bug, opencv/opencv#23368 -- so when no cv2-probed candidate matches
   2560x720, `resolve_cameras()` falls back to a real `ffmpeg`-based capture
   check per candidate. See `_verify_stereo_via_ffmpeg()`.)

Usage:
    python detect_devices.py [--out device_map.json] [--max-camera-index 6]
"""

import argparse
import glob
import json
import re
import subprocess
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path

import cv2

from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SO101Follower
from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig
from lerobot.teleoperators.so_leader.so_leader import SOLeader

from vla_bridge.stereo_camera import STEREO_HEIGHT, STEREO_WIDTH

FOLLOWER_ROBOT_ID = "soarm_follower_02"
LEADER_ROBOT_ID = "soarm_leader_01"

# Names containing any of these (case-insensitive) substrings are the laptop's own
# built-in camera -- confirmed this session: `system_profiler SPCameraDataType`
# reliably names it "FaceTime HD Camera" on this machine.
BUILT_IN_NAME_MARKERS = ("facetime",)

DEVICE_MAP_PATH = Path(__file__).resolve().parent / "device_map.json"

_DEVICE_NAME_LINE_RE = re.compile(r"^    (?! )(.+):\s*$")


class DeviceDetectionError(RuntimeError):
    """Raised when a MANDATORY device can't be resolved (the follower robot, or a
    wrist/stereo camera assignment). Never raised for the leader, which is optional
    (`run_vla_episode.py` never needs it) -- an unresolved leader is left `None`.
    """


# --- Serial port -> robot identity ------------------------------------------------


def list_serial_ports() -> list[str]:
    """`/dev/cu.usbmodem*` ports, sorted for deterministic iteration order --
    reuses `control/COMMANDS.md`'s own `ls`-based discovery convention."""
    return sorted(glob.glob("/dev/cu.usbmodem*"))


def _make_follower_robot(port: str, robot_id: str):
    return SO101Follower(SOFollowerRobotConfig(port=port, id=robot_id))


def _make_leader_robot(port: str, robot_id: str):
    return SOLeader(SOLeaderTeleopConfig(port=port, id=robot_id))


def resolve_port_identity(port: str, robot_id: str, make_robot=None) -> bool:
    """Connects to `port` and performs a READ-ONLY identity check: reads the
    calibration values LIVE off the servos (`bus.read_calibration()`) and compares
    them against `robot_id`'s own saved calibration file (loaded automatically at
    robot construction into `robot.calibration`). Returns True only if every motor
    in the saved file has an exact live match -- genuine proof this port really is
    `robot_id`, not merely "a robot answered on this port."

    NEVER writes to the servos. An earlier version called `bus.write_calibration()`
    as its "test" and treated "no exception" as proof of identity -- this is unsafe:
    when `robot_id` has no saved calibration file for this robot class,
    `robot.calibration` loads as an empty dict, so `write_calibration({})` is a
    silent no-op that never raises, letting a wrong id "resolve" without ever
    proving identity. Had that id instead resolved to a real saved file belonging
    to a DIFFERENT physical robot, the write would have silently pushed that
    robot's homing offsets and position limits onto this port's real servos.

    Returns False on ANY failure -- connect error, no saved calibration file for
    this id, or a live/saved mismatch on any motor -- never raises, so one bad
    port/id guess doesn't crash the whole detection run.

    `make_robot(port, robot_id) -> robot` is injectable (defaults to a real
    `SO101Follower`/`SOFollowerRobotConfig` round trip) so tests never touch a real
    serial port.
    """
    if make_robot is None:
        make_robot = _make_follower_robot
    try:
        robot = make_robot(port, robot_id)
        robot.connect(calibrate=False)
        try:
            saved = robot.calibration
            if not saved:
                print(
                    f"WARNING: port {port} did not resolve to id {robot_id!r}: "
                    f"no saved calibration file for this id, cannot confirm identity"
                )
                return False
            live = robot.bus.read_calibration()
            for motor, saved_cal in saved.items():
                live_cal = live.get(motor)
                if live_cal is None:
                    print(
                        f"WARNING: port {port} did not resolve to id {robot_id!r}: "
                        f"motor {motor!r} missing from live calibration readback"
                    )
                    return False
                if (
                    live_cal.homing_offset != saved_cal.homing_offset
                    or live_cal.range_min != saved_cal.range_min
                    or live_cal.range_max != saved_cal.range_max
                ):
                    print(
                        f"WARNING: port {port} did not resolve to id {robot_id!r}: "
                        f"motor {motor!r} calibration mismatch "
                        f"(live={live_cal}, saved={saved_cal})"
                    )
                    return False
            return True
        finally:
            robot.disconnect()
    except Exception as e:
        print(f"WARNING: port {port} did not resolve to id {robot_id!r}: {e}")
        return False


# --- Camera identity ---------------------------------------------------------------


def _parse_camera_names(system_profiler_output: str) -> list[str]:
    """Extracts top-level device names from `system_profiler SPCameraDataType`
    output, in the order `system_profiler` lists them. A device name line has
    EXACTLY 4 leading spaces and ends with ':' (nested attribute lines like
    "Model ID: ..." are indented 6+ spaces and are skipped)."""
    names = []
    for line in system_profiler_output.splitlines():
        match = _DEVICE_NAME_LINE_RE.match(line)
        if match:
            names.append(match.group(1).strip())
    return names


def get_camera_names(timeout_s: float = 10.0) -> list[str]:
    """Best-effort ordered list of camera device names via `system_profiler
    SPCameraDataType` (stdlib `subprocess` only, no new dependency) -- used only for
    built-in-camera name exclusion. Never raises: a failed/missing `system_profiler`
    call just means no name-based exclusion is possible (printed as a warning, not
    fatal -- resolution-based stereo/wrist disambiguation still applies)."""
    try:
        result = subprocess.run(
            ["system_profiler", "SPCameraDataType"],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        print(f"WARNING: system_profiler SPCameraDataType failed ({e}); camera name exclusion unavailable")
        return []
    return _parse_camera_names(result.stdout)


def probe_camera_candidates(max_index: int = 6) -> list[dict]:
    """Opens cv2 indices 0..max_index-1 (same open/probe pattern as
    `run_vla_episode.py`'s `_probe_camera_indices()`), capturing one frame per opened
    index and recording its resolution. Returns one `{"index", "width", "height"}`
    dict per index that both opened AND returned a real frame, in ascending index
    order."""
    candidates = []
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok and frame is not None:
                height, width = frame.shape[:2]
                candidates.append({"index": idx, "width": width, "height": height})
            else:
                print(f"WARNING: camera index {idx} opened but frame read failed, skipping")
        cap.release()
    return candidates


def _correlate_names(candidates: list[dict], names: list[str]) -> dict[int, str | None]:
    """Best-effort POSITIONAL correlation between `system_profiler`'s device-name
    order and cv2's ascending-index probe order. macOS does NOT guarantee these two
    enumeration orders actually match -- this is a heuristic (confirmed to hold on
    this rig this session with all cameras connected), not a verified 1:1 mapping.
    Extra opened indices beyond the number of names found get `None` (never
    excluded by name, since we can't confirm either way)."""
    indices = [c["index"] for c in candidates]
    return {idx: name for idx, name in zip_longest(indices, names, fillvalue=None) if idx is not None}


def _is_builtin_name(name: str | None) -> bool:
    if not name:
        return False
    lowered = name.lower()
    return any(marker in lowered for marker in BUILT_IN_NAME_MARKERS)


def _mean_brightness(cap) -> float | None:
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return float(frame.mean())


def _disambiguate_by_brightness(candidates: list[dict], prompt_fn=input) -> int:
    """Interactive cover-the-lens brightness check -- the same manual technique used
    live in this session's Task 1 investigation -- for telling the real wrist camera
    apart from a second ambiguous non-stereo, non-built-in candidate (name-based
    exclusion alone couldn't resolve it). `prompt_fn` is injectable so tests can stub
    the interactive prompt without blocking on real input()."""
    indices = [c["index"] for c in candidates]
    print(f"Ambiguous: {len(candidates)} non-stereo, non-built-in camera candidates found: {indices}")
    caps = {idx: cv2.VideoCapture(idx) for idx in indices}
    before = {idx: _mean_brightness(cap) for idx, cap in caps.items()}
    prompt_fn("Cover the WRIST camera's lens (only that one) and press Enter...")
    after = {idx: _mean_brightness(cap) for idx, cap in caps.items()}
    for cap in caps.values():
        cap.release()

    drops = {
        idx: before[idx] - after[idx]
        for idx in indices
        if before[idx] is not None and after[idx] is not None
    }
    if not drops:
        raise DeviceDetectionError(
            "Could not read frames from ambiguous camera candidates to disambiguate via brightness"
        )

    wrist_idx = max(drops, key=drops.get)
    print(f"Resolved wrist camera: index {wrist_idx} (brightness drop {drops[wrist_idx]:.1f})")
    return wrist_idx


def _verify_stereo_via_ffmpeg(index: int, timeout_s: float = 8.0) -> bool:
    """Confirms `index` is genuinely the AR0144 stereo camera via a real `ffmpeg`
    capture attempt at its native 2560x720 `uyvy422` mode.

    Needed as a fallback because `cv2.VideoCapture`'s reported resolution for
    this camera is UNRELIABLE on macOS: confirmed live during 11-05 Task 3 prep
    that cv2's AVFoundation backend serves 1920x1080 or 1280x720 for this device
    regardless of `CAP_PROP_FRAME_WIDTH`/`HEIGHT`/`FOURCC` requests -- a known
    unfixed OpenCV bug (opencv/opencv#23368), not a hardware or cable problem.
    AVFoundation itself confirms 2560x720 IS a genuinely supported mode for this
    device (verified via `ffmpeg -video_size 9999x9999 ...`'s "Supported modes"
    error listing), and `ffmpeg -f avfoundation -pixel_format uyvy422
    -video_size 2560x720` reliably captures the real frame where cv2 cannot.

    Never raises; returns False on any failure (wrong camera, ffmpeg missing,
    timeout, non-stereo device that can't produce this frame shape).
    """
    expected_bytes = STEREO_WIDTH * STEREO_HEIGHT * 3
    cmd = [
        "ffmpeg", "-loglevel", "error",
        "-f", "avfoundation",
        "-pixel_format", "uyvy422",
        "-video_size", f"{STEREO_WIDTH}x{STEREO_HEIGHT}",
        "-framerate", "30",
        "-i", str(index),
        "-frames:v", "1",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=timeout_s, check=False)
    except (OSError, subprocess.TimeoutExpired) as e:
        print(f"WARNING: ffmpeg stereo verification failed for index {index}: {e}")
        return False
    return result.returncode == 0 and len(result.stdout) == expected_bytes


def resolve_cameras(
    candidates: list[dict],
    names_by_index: dict[int, str | None],
    prompt_fn=input,
    verify_stereo_fn=None,
) -> dict[str, int]:
    """Resolves `{"wrist": <idx>, "stereo_overhead": <idx>, "stereo_overhead_name":
    <name-or-idx>}` from probed camera `candidates`, excluding any built-in-named
    device first, then splitting the remainder by the AR0144's distinctive
    2560x720 resolution. `stereo_overhead_name` is the identifier
    `StereoSplitCamera`'s ffmpeg backend should actually be opened with -- see
    the inline comment above its construction for why the numeric index isn't
    safe to persist across process launches.

    `verify_stereo_fn(index) -> bool` is injectable (defaults to
    `_verify_stereo_via_ffmpeg`) so tests never spawn a real ffmpeg subprocess.
    It's only consulted as a FALLBACK when no candidate's cv2-reported
    resolution matches 2560x720 -- see `_verify_stereo_via_ffmpeg`'s docstring
    for why cv2's reported resolution can't always be trusted for this camera.
    """
    non_builtin = [c for c in candidates if not _is_builtin_name(names_by_index.get(c["index"]))]

    stereo_candidates = [c for c in non_builtin if (c["width"], c["height"]) == (STEREO_WIDTH, STEREO_HEIGHT)]
    stereo_indices = {c["index"] for c in stereo_candidates}

    if not stereo_candidates:
        verify = verify_stereo_fn if verify_stereo_fn is not None else _verify_stereo_via_ffmpeg
        verified = [c for c in non_builtin if verify(c["index"])]
        if len(verified) == 1:
            stereo_candidates = verified
            stereo_indices = {verified[0]["index"]}
        elif len(verified) > 1:
            raise DeviceDetectionError(
                f"Ambiguous: {len(verified)} candidates verified as the AR0144 stereo camera "
                f"via the ffmpeg fallback check: {verified}"
            )

    other_candidates = [c for c in non_builtin if c["index"] not in stereo_indices]

    if len(stereo_candidates) == 0:
        raise DeviceDetectionError(
            f"No {STEREO_WIDTH}x{STEREO_HEIGHT} AR0144 stereo camera candidate found "
            f"among non-built-in cameras: {non_builtin}"
        )
    if len(stereo_candidates) > 1:
        raise DeviceDetectionError(
            f"Ambiguous: {len(stereo_candidates)} candidates matched the AR0144's "
            f"{STEREO_WIDTH}x{STEREO_HEIGHT} stereo resolution: {stereo_candidates}"
        )
    stereo_idx = stereo_candidates[0]["index"]

    if len(other_candidates) == 0:
        raise DeviceDetectionError(
            "No wrist camera candidate found -- every non-built-in camera matched the stereo resolution"
        )
    if len(other_candidates) == 1:
        wrist_idx = other_candidates[0]["index"]
    else:
        wrist_idx = _disambiguate_by_brightness(other_candidates, prompt_fn=prompt_fn)

    # `stereo_overhead` (cv2 index) is kept for backward compat / the IOLogger's
    # own cv2-based recording caps in run_vla_episode.py. `stereo_overhead_name`
    # is the AUTHORITATIVE identifier for StereoSplitCamera's ffmpeg capture
    # backend: found live during 11-05 Task 3 prep that BOTH cv2's AND ffmpeg's
    # own AVFoundation device index numbering can drift between process
    # launches on macOS (confirmed: the same physical AR0144 camera showed up
    # at ffmpeg index 2 in one process and index 0 moments later in another) --
    # a persisted numeric index is not safe to reuse across process
    # boundaries. Addressing the device by its AVFoundation NAME string
    # (`ffmpeg -i "CCB Camera"`) sidesteps the instability entirely. Falls
    # back to the numeric index only if no name could be correlated for it
    # (best-effort correlation, see `_correlate_names()`).
    stereo_name = names_by_index.get(stereo_idx)
    return {
        "wrist": wrist_idx,
        "stereo_overhead": stereo_idx,
        "stereo_overhead_name": stereo_name if stereo_name is not None else stereo_idx,
    }


# --- Top-level detection -------------------------------------------------------------


def detect_devices(
    ports: list[str] | None = None,
    max_camera_index: int = 6,
    make_follower=None,
    make_leader=None,
    prompt_fn=input,
    verify_stereo_fn=None,
) -> dict:
    """Runs both detection problems and returns the full `device_map.json`-shaped
    dict (not yet written to disk -- see `main()`)."""
    if ports is None:
        ports = list_serial_ports()

    follower = None
    leader = None
    for port in ports:
        if follower is None and resolve_port_identity(port, FOLLOWER_ROBOT_ID, make_robot=make_follower):
            follower = {"port": port, "id": FOLLOWER_ROBOT_ID}
            continue
        if leader is None and resolve_port_identity(port, LEADER_ROBOT_ID, make_robot=make_leader):
            leader = {"port": port, "id": LEADER_ROBOT_ID}
            continue
        print(
            f"WARNING: port {port} did not resolve to follower id {FOLLOWER_ROBOT_ID!r} "
            f"or leader id {LEADER_ROBOT_ID!r}; leaving unassigned"
        )

    if follower is None:
        raise DeviceDetectionError(
            f"No connected serial port resolved to the follower id {FOLLOWER_ROBOT_ID!r}. "
            f"Checked ports: {ports or '(none found)'}"
        )

    candidates = probe_camera_candidates(max_camera_index)
    names = get_camera_names()
    names_by_index = _correlate_names(candidates, names)
    cameras = resolve_cameras(candidates, names_by_index, prompt_fn=prompt_fn, verify_stereo_fn=verify_stereo_fn)

    return {
        "detected_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "follower": follower,
        "leader": leader,
        "cameras": cameras,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=DEVICE_MAP_PATH,
        help=f"Output path for device_map.json (default: {DEVICE_MAP_PATH})",
    )
    parser.add_argument(
        "--max-camera-index",
        type=int,
        default=6,
        help="Highest cv2 index (exclusive) to probe for cameras (default: 6, indices 0-5)",
    )
    args = parser.parse_args()

    try:
        device_map = detect_devices(max_camera_index=args.max_camera_index)
    except DeviceDetectionError as e:
        print(f"ERROR: {e}")
        raise SystemExit(1) from e

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(device_map, indent=2) + "\n")
    print(f"Saved -> {args.out}")
    print(json.dumps(device_map, indent=2))


if __name__ == "__main__":
    main()
