"""
Record a synced joint-position + camera-frame episode from the SO-101 follower.
Covers function UAT Steps 5 (still image), 6 (video during motion), and 7 (combined
episode recording) in one script — a still is just a 0-duration video, a video-only
run is an episode without joint logging.

We built our own recorder rather than fighting `lerobot-record`'s CLI (this session
already hit real gaps in LeRobot's generic CLI paths for so101_follower - see
keyboard_joint_control.py and diagnostics/UAT/function/UAT.md Step 3). This is a
straightforward, fully-understood alternative: cv2.VideoWriter per camera + a CSV of
joint positions, both timestamped against the same clock.

Usage:
    python record_episode.py PORT ROBOT_ID --duration 5 --camera 0 --camera 1 \
        --out outputs/episode_001 [--no-joints] [--fps 15]

    --duration 0 captures a single still frame per camera instead of video.
"""
import argparse
import csv
import json
from datetime import datetime, timezone
import time
from pathlib import Path

import cv2
import detect_devices
from vla_bridge.stereo_camera import SPLIT_COL, STEREO_HEIGHT, StereoSplitCamera

from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SO101Follower


DEVICE_MAP_PATH = Path(__file__).resolve().parent / "device_map.json"
DEVICE_MAP_MAX_AGE_HOURS = 24


def _load_device_map() -> dict | None:
    """Loads `control/device_map.json` (Plan 11-05 Task 2's `detect_devices.py`
    output) if present and parseable. Returns `None` -- never raises -- on
    any failure (missing file, unreadable, invalid JSON), so callers can
    fall through to requiring explicit CLI args instead of crashing."""
    if not DEVICE_MAP_PATH.exists():
        return None
    try:
        device_map = json.loads(DEVICE_MAP_PATH.read_text())
    except (json.JSONDecodeError, OSError) as e:
        print(f"WARNING: {DEVICE_MAP_PATH} exists but could not be read ({e}); ignoring it.")
        return None

    detected_at = device_map.get("detected_at")
    if detected_at:
        try:
            detected_dt = datetime.strptime(detected_at, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            age_hours = (datetime.now(timezone.utc) - detected_dt).total_seconds() / 3600.0
            if age_hours > DEVICE_MAP_MAX_AGE_HOURS:
                print(
                    f"WARNING: {DEVICE_MAP_PATH} is {age_hours:.1f}h old (>{DEVICE_MAP_MAX_AGE_HOURS}h) -- "
                    "ports/camera indices may have shifted since detection. Re-run detect_devices.py "
                    "if any USB device has been unplugged/replugged since then."
                )
        except ValueError:
            pass  # Unparseable timestamp -- use the file anyway, just skip the freshness check.

    return device_map


def _build_camera_roles(camera_indices: list[int], device_map: dict | None) -> dict[int, str]:
    if device_map is None or "cameras" not in device_map:
        return {}
    cameras = device_map["cameras"]
    return {cameras[key]: role for key, role in
            (("wrist", "wrist"), ("stereo_overhead", "overhead"))
            if cameras.get(key) in camera_indices}


def open_cameras(camera_indices: list[int], camera_roles: dict[int, str], device_map: dict | None) -> dict[int, object]:
    caps = {}
    for idx in dict.fromkeys(camera_indices):
        role = camera_roles.get(idx)
        if role == "overhead":
            identifier = detect_devices.resolve_camera_identifier("stereo_overhead", device_map) if device_map is not None else idx
            cap = StereoSplitCamera(index=identifier)
            if not cap.is_opened:
                print(f"WARNING: overhead camera did not open (index {idx}), skipping")
                cap.release()
                continue
            caps[idx] = cap
            continue
        identifier = detect_devices.resolve_camera_identifier("wrist", device_map) if role == "wrist" and device_map is not None else idx
        cap = cv2.VideoCapture(identifier)
        if not cap.isOpened():
            print(f"WARNING: camera index {idx} did not open, skipping")
            continue
        caps[idx] = cap
    return caps


def record_still(caps, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, cap in caps.items():
        if isinstance(cap, StereoSplitCamera):
            for _ in range(15):
                left, right = cap.read_left(), cap.read_right()
                if left is not None and right is not None:
                    break
            if left is None or right is None:
                print(f"overhead camera {idx}: failed to capture")
                continue
            for side, frame in (("left", left), ("right", right)):
                path = out_dir / f"camera_overhead_{side}.png"
                cv2.imwrite(str(path), frame)
                print(f"overhead camera {side}: saved -> {path}")
            continue
        ok, frame = False, None
        for _ in range(15):  # warm up, same lesson as diagnostics/camera_test.py
            ok, frame = cap.read()
            if ok:
                time.sleep(0.05)
        if not ok:
            print(f"camera {idx}: failed to capture")
            continue
        path = out_dir / f"camera_{idx}.png"
        cv2.imwrite(str(path), frame)
        print(f"camera {idx}: saved -> {path}")


def record_episode(robot, caps, out_dir: Path, duration: float, fps: float, log_joints: bool):
    out_dir.mkdir(parents=True, exist_ok=True)
    writers = {}
    for idx, cap in caps.items():
        if isinstance(cap, StereoSplitCamera):
            for side in ("left", "right"):
                path = out_dir / f"camera_overhead_{side}.mp4"
                writers[f"overhead_{side}"] = cv2.VideoWriter(
                    str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (SPLIT_COL, STEREO_HEIGHT)
                )
            continue
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        path = out_dir / f"camera_{idx}.mp4"
        writers[idx] = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    joints_file = None
    joints_writer = None
    if log_joints:
        joints_path = out_dir / "joints.csv"
        joints_file = open(joints_path, "w", newline="")
        joints_writer = csv.writer(joints_file)
        joints_writer.writerow(["timestamp_s"] + [f"{j}.pos" for j in
                                ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]])

    print(f"Recording for {duration}s at {fps} fps...")
    start = time.perf_counter()
    period = 1.0 / fps
    frame_count = 0
    while time.perf_counter() - start < duration:
        t0 = time.perf_counter()
        ts = t0 - start

        for idx, cap in caps.items():
            if isinstance(cap, StereoSplitCamera):
                for side, frame in (("left", cap.read_left()), ("right", cap.read_right())):
                    if frame is not None:
                        writers[f"overhead_{side}"].write(frame)
                continue
            ok, frame = cap.read()
            if ok:
                writers[idx].write(frame)

        if log_joints:
            obs = robot.get_observation()
            row = [ts] + [obs.get(f"{j}.pos", "") for j in
                          ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]]
            joints_writer.writerow(row)

        frame_count += 1
        elapsed = time.perf_counter() - t0
        if elapsed < period:
            time.sleep(period - elapsed)

    for idx, w in writers.items():
        w.release()
        print(f"camera {idx}: saved -> {out_dir / f'camera_{idx}.mp4'}")
    if joints_file:
        joints_file.close()
        print(f"joints: saved -> {out_dir / 'joints.csv'}")
    print(f"Captured {frame_count} frames over {time.perf_counter() - start:.1f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("port")
    parser.add_argument("robot_id")
    parser.add_argument("--camera", type=int, action="append", default=[], help="repeatable, e.g. --camera 0 --camera 1")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=5.0, help="seconds; 0 = single still frame")
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--no-joints", action="store_true", help="skip joint-position logging (camera-only capture)")
    args = parser.parse_args()
    device_map = _load_device_map()

    if not args.camera:
        print("No --camera given. Probing indices 0-5...")
        args.camera = []
        for idx in range(6):
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                args.camera.append(idx)
            cap.release()
        print(f"Found cameras: {args.camera}")

    camera_roles = _build_camera_roles(args.camera, device_map)
    caps = open_cameras(args.camera, camera_roles, device_map)
    if not caps:
        print("No cameras opened, aborting.")
        return

    if args.duration <= 0:
        record_still(caps, args.out)
        for cap in caps.values():
            cap.release()
        return

    robot = None
    if not args.no_joints:
        robot = SO101Follower(SOFollowerRobotConfig(port=args.port, id=args.robot_id))
        robot.connect(calibrate=False)
        robot.bus.write_calibration(robot.calibration)

    try:
        record_episode(robot, caps, args.out, args.duration, args.fps, log_joints=not args.no_joints)
    finally:
        for cap in caps.values():
            cap.release()
        if robot:
            robot.disconnect()


if __name__ == "__main__":
    main()
