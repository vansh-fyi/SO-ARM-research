"""
Move a single SO-101 follower joint by a relative amount, via LeRobot's calibrated
robot object (respects the calibration.json from `lerobot-calibrate`), not raw ticks.

Why this exists: LeRobot's built-in `keyboard` teleoperator emits raw key names, not
joint deltas, and isn't wired to translate into so101_follower's joint action space
via the generic `lerobot-teleoperate` CLI (confirmed by reading the installed
lerobot 0.6.1 source - `KeyboardTeleop.get_action()` returns `{key: None}` pairs,
and the CLI's default processor pipeline doesn't translate keys to joints for this
robot type, causing an immediate crash). `keyboard_ee` is built for a different
robot class (`So100FollowerEndEffector`), not plain `so101_follower`. This script is
the practical laptop-control path until a leader arm, gamepad, or a proper keyboard
mapping is added.

Units: arm joints in degrees (use_degrees=True to match calibration), gripper in
0-100 (RANGE_0_100 norm mode).

Usage:
    python joint_jog.py PORT ROBOT_ID JOINT DELTA [--max-relative-target 10]

    JOINT is one of: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex,
    wrist_roll, gripper
"""
import argparse
import sys
import time

from lerobot.robots.so_follower import SO101Follower
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

VALID_JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def jog(port: str, robot_id: str, joint: str, delta: float, max_relative_target: float):
    if joint not in VALID_JOINTS:
        print(f"Unknown joint '{joint}'. Valid: {VALID_JOINTS}")
        sys.exit(1)

    config = SOFollowerRobotConfig(
        port=port,
        id=robot_id,
        max_relative_target=max_relative_target,
    )
    robot = SO101Follower(config)
    robot.connect(calibrate=False)
    # connect(calibrate=False) skips write_calibration() entirely, leaving the
    # servo's own Min/Max_Position_Limit + Homing_Offset registers at whatever was
    # there before (can be a stale narrow range) - apply the saved calibration.json
    # explicitly every time so hardware limits always match what lerobot-calibrate
    # recorded.
    robot.bus.write_calibration(robot.calibration)

    try:
        obs = robot.get_observation()
        key = f"{joint}.pos"
        current = obs[key]
        target = current + delta
        print(f"{joint}: current={current:.2f} -> target={target:.2f} (delta {delta:+.2f})")

        sent = robot.send_action({key: target})
        print(f"Sent (possibly clamped by max_relative_target={max_relative_target}): {sent}")

        time.sleep(1.0)  # let the servo actually finish traveling before verifying
        new_obs = robot.get_observation()
        print(f"{joint}: now at {new_obs[key]:.2f}")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("port")
    parser.add_argument("robot_id")
    parser.add_argument("joint", choices=VALID_JOINTS)
    parser.add_argument("delta", type=float, help="degrees for arm joints, 0-100 units for gripper")
    parser.add_argument("--max-relative-target", type=float, default=10.0)
    args = parser.parse_args()

    jog(args.port, args.robot_id, args.joint, args.delta, args.max_relative_target)
