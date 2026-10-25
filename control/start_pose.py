"""The one fixed home (start) pose every real-robot episode uses (quick 261025-wbw).

Contract: `run_vla_episode.py` (and `tune_execution_hz.py`, through
`run_episode()`) moves the arm to this pose after the robot connects and
before the backend's first `get_action`, refuses to start if any joint misses
it by more than `START_POSE_TOLERANCE`, and returns the arm here before
disconnect on every exit path once it has arrived. A missing or invalid pose
file means the episode refuses to run at all.

File format (`control/start_pose.json`, committed to git):

    {
      "captured_at": "2026-10-25T12:00:00Z",
      "units": {"shoulder_pan": "degrees", ..., "gripper": "percent_0_100"},
      "joints": {"shoulder_pan": 0.0, ..., "gripper": 20.0}
    }

`joints` holds the same calibrated values `keyboard_joint_control.read_positions()`
returns (degrees for the 5 arm joints, 0-100 percent for the gripper), keyed
by `action_contract.JOINT_ORDER`. Every value must lie inside
`action_contract.load_joint_limits_deg()`'s safety limits. There is one
physical arm, so no robot id is recorded or checked.

Capture CLI (run from control/, after placing the arm in the home pose by hand
or with keyboard_joint_control.py / joint_jog.py):

    python start_pose.py [PORT] [ROBOT_ID] [--force]

The capture only READS the arm's joint positions; it never moves it.
Overwriting an existing pose file requires `--force`.
"""

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SO101Follower

from keyboard_joint_control import read_positions
from vla_bridge import action_contract

START_POSE_PATH = Path(__file__).resolve().parent / "start_pose.json"
CAPTURE_COMMAND = "python start_pose.py"

# Per-joint arrival tolerance: degrees for arm joints, percent for the gripper.
# Absorbs the P-control residual left by gravity sag on the loaded joints.
START_POSE_TOLERANCE = 5.0
# Seconds allowed for the start move AND the return move -- generous, because
# home can be far from wherever the policy left the arm.
START_POSE_MOVE_SECONDS = 15.0
# Proportional gain of the move_to_positions() P-control step (same as the
# keyboard tool's return-to-start move).
START_POSE_KP = 0.2
# P-control loop rate in Hz; an int because move_to_positions() uses control_freq // 2.
START_POSE_CONTROL_HZ = 30

_TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


class StartPoseError(ValueError):
    """The start pose file is missing or invalid, or a pose fails validation."""


class StartPoseNotReachedError(RuntimeError):
    """The arm did not arrive within tolerance of the home pose."""

    def __init__(self, target, reached, missed, tolerance):
        self.target = dict(target)
        self.reached = dict(reached)
        self.missed = dict(missed)
        self.tolerance = tolerance
        lines = []
        for joint, error in self.missed.items():
            if error is None:
                lines.append(f"  {joint}: target {self.target.get(joint)}, reached not reported")
            else:
                lines.append(
                    f"  {joint}: target {self.target[joint]:.2f}, "
                    f"reached {self.reached[joint]:.2f}, error {error:+.2f}"
                )
        super().__init__(
            f"arm did not reach the start pose within +/-{tolerance} (deg / %) on "
            f"{len(self.missed)} joint(s) -- not starting the episode:\n"
            + "\n".join(lines)
            + "\nCheck for obstacles or servo faults (overload, torque off), or if the "
            f"home pose itself is unreachable, re-capture it with `{CAPTURE_COMMAND} --force`."
        )


def _capture_hint(path) -> str:
    return (
        f"Place the arm in the home pose, then run `{CAPTURE_COMMAND}` from control/ "
        f"to capture {path}."
    )


def validate_start_pose(joints, joint_limits_deg) -> dict[str, float]:
    """Return `joints` as `{joint: float}` in JOINT_ORDER order, or raise ONE
    StartPoseError listing every problem found (missing, unknown, non-numeric,
    non-finite, unlimited or out-of-limit joints)."""
    if not isinstance(joints, dict):
        raise StartPoseError(f"start pose joints must be an object keyed by joint, got {type(joints).__name__}")

    joint_order = action_contract.JOINT_ORDER
    problems = []
    for joint in joint_order:
        if joint not in joints:
            problems.append(f"missing joint {joint}")
    for joint in joints:
        if joint not in joint_order:
            problems.append(f"unknown joint {joint}")

    result = {}
    for joint in joint_order:
        if joint not in joints:
            continue
        value = joints[joint]
        # bool is an int subclass, so it must be rejected explicitly.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            problems.append(f"{joint} must be a number, got {value!r}")
            continue
        value = float(value)
        if not math.isfinite(value):
            problems.append(f"{joint} must be finite, got {value!r}")
            continue
        if joint not in joint_limits_deg:
            problems.append(f"{joint} has no safety limits entry")
            continue
        lo, hi = joint_limits_deg[joint]
        if not lo <= value <= hi:
            problems.append(f"{joint} = {value} is outside its safety limits ({lo:.2f}, {hi:.2f})")
            continue
        result[joint] = value

    if problems:
        raise StartPoseError("invalid start pose: " + "; ".join(problems))
    return result


def load_start_pose(joint_limits_deg, path=None) -> dict[str, float]:
    """Read and validate the committed home pose file (default START_POSE_PATH,
    resolved at call time so tests can monkeypatch it)."""
    path = START_POSE_PATH if path is None else Path(path)
    if not path.exists():
        raise StartPoseError(f"start pose file not found: {path}. {_capture_hint(path)}")
    try:
        doc = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError) as e:
        raise StartPoseError(f"start pose file {path} could not be read ({e}). {_capture_hint(path)}") from e
    except json.JSONDecodeError as e:
        raise StartPoseError(f"start pose file {path} is not valid JSON ({e}). {_capture_hint(path)}") from e

    if not isinstance(doc, dict):
        raise StartPoseError(f"start pose file {path} must contain a JSON object. {_capture_hint(path)}")
    captured_at = doc.get("captured_at")
    if not isinstance(captured_at, str) or not captured_at:
        raise StartPoseError(
            f"start pose file {path} has no captured_at timestamp. {_capture_hint(path)}"
        )
    if not isinstance(doc.get("joints"), dict):
        raise StartPoseError(f"start pose file {path} has no joints object. {_capture_hint(path)}")
    try:
        return validate_start_pose(doc["joints"], joint_limits_deg)
    except StartPoseError as e:
        raise StartPoseError(f"start pose file {path}: {e}. {_capture_hint(path)}") from e


def capture_start_pose(robot, joint_limits_deg, path=None, force=False) -> dict:
    """Read the arm's current joint positions (never moves it), validate them
    against the safety limits and write them as the home pose file."""
    path = START_POSE_PATH if path is None else Path(path)
    if path.exists() and not force:
        raise FileExistsError(
            f"start pose file already exists: {path}. Re-run with --force to overwrite it."
        )
    joints = validate_start_pose(read_positions(robot), joint_limits_deg)
    doc = {
        "captured_at": datetime.now(timezone.utc).strftime(_TIMESTAMP_FORMAT),
        "units": dict(action_contract.ACTION_UNITS),
        "joints": {joint: round(value, 2) for joint, value in joints.items()},
    }
    path.write_text(json.dumps(doc, indent=2) + "\n")
    return doc


def joints_out_of_tolerance(target, reached, tolerance=START_POSE_TOLERANCE) -> dict:
    """`{joint: reached - target}` for every target joint off by more than
    `tolerance` (inclusive bound), `{joint: None}` for one absent from `reached`."""
    missed = {}
    for joint, target_value in target.items():
        if joint not in reached:
            missed[joint] = None
            continue
        error = reached[joint] - target_value
        if abs(error) > tolerance:
            missed[joint] = error
    return missed


def main():
    parser = argparse.ArgumentParser(
        description="Capture the arm's current joint positions as the fixed home pose "
        "(reads only; never moves the arm)."
    )
    parser.add_argument(
        "port",
        nargs="?",
        default=None,
        help="Serial port, e.g. /dev/cu.usbmodemXXXX. If omitted, read from "
        "control/device_map.json's follower.port (run detect_devices.py first).",
    )
    parser.add_argument(
        "robot_id",
        nargs="?",
        default=None,
        help="Robot calibration id, e.g. soarm_follower_02. If omitted, read from "
        "control/device_map.json's follower.id.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite an existing start_pose.json.")
    args = parser.parse_args()

    path = START_POSE_PATH
    if path.exists() and not args.force:
        parser.error(f"{path} already exists; re-run with --force to overwrite it")

    if args.port is None or args.robot_id is None:
        # In-function import: run_vla_episode imports this module, so a
        # top-level import here would be circular.
        from run_vla_episode import _load_device_map

        device_map = _load_device_map()
        if device_map is None or device_map.get("follower") is None:
            parser.error(
                "PORT/ROBOT_ID not given and control/device_map.json's follower entry is "
                "unavailable -- run detect_devices.py first, or pass PORT ROBOT_ID explicitly."
            )
        if args.port is None:
            args.port = device_map["follower"]["port"]
        if args.robot_id is None:
            args.robot_id = device_map["follower"]["id"]

    joint_limits_deg = action_contract.load_joint_limits_deg()

    robot = SO101Follower(SOFollowerRobotConfig(port=args.port, id=args.robot_id))
    robot.connect(calibrate=False)
    try:
        # connect(calibrate=False) skips write_calibration(); apply it with
        # torque off, exactly like keyboard_joint_control.main().
        with robot.bus.torque_disabled():
            robot.bus.write_calibration(robot.calibration)
        try:
            doc = capture_start_pose(robot, joint_limits_deg, path, force=args.force)
        except (StartPoseError, FileExistsError) as e:
            raise SystemExit(str(e)) from e
        for joint, value in doc["joints"].items():
            print(f"{joint}: {value}")
        print(f"Saved → {path}")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
