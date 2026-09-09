"""
Snapshot both arms' calibrated/normalized joint positions side by side, to check
whether "the same physical pose" on leader and follower actually maps to the same
normalized value per joint - if not, that joint's calibration direction/range
doesn't correspond between the two arms, and leader-follower teleop will jump to a
very different follower pose even when the arms look similarly posed by eye.

Usage:
    python compare_leader_follower.py FOLLOWER_PORT FOLLOWER_ID LEADER_PORT LEADER_ID
"""
import sys

from lerobot.robots.so_follower import SO101Follower
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.teleoperators.so_leader.so_leader import SOLeader
from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig


def main():
    f_port, f_id, l_port, l_id = sys.argv[1:5]

    # use_degrees=False so shoulder_lift/elbow_flex/etc. go through the
    # RANGE_M100_100 branch of normalize/unnormalize - the DEGREES branch (the
    # default for both configs) never applies drive_mode, so a real calibration
    # inversion on those joints is otherwise invisible to this comparison.
    robot = SO101Follower(SOFollowerRobotConfig(port=f_port, id=f_id, use_degrees=False))
    leader = SOLeader(SOLeaderTeleopConfig(port=l_port, id=l_id, use_degrees=False))

    robot.connect(calibrate=False)
    robot.bus.write_calibration(robot.calibration)
    leader.connect(calibrate=False)
    leader.bus.write_calibration(leader.calibration)

    follower_obs = robot.get_observation()
    leader_action = leader.get_action()

    print(f"{'joint':<15} | {'follower %':>12} | {'leader %':>12} | {'diff':>8}")
    print("-" * 55)
    for joint in ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]:
        f_val = follower_obs.get(f"{joint}.pos")
        l_val = leader_action.get(f"{joint}.pos")
        diff = f_val - l_val if (f_val is not None and l_val is not None) else float("nan")
        print(f"{joint:<15} | {f_val:>12.1f} | {l_val:>12.1f} | {diff:>8.1f}")

    robot.disconnect()
    leader.disconnect()


if __name__ == "__main__":
    main()
