#!/usr/bin/env python3
"""
Keyboard joint control for the SO-101 follower, adapted from XLeRobot's
0_so100_keyboard_joint_control.py (https://github.com/Vector-Wangel/XLeRobot).

Why this exists (not plain `lerobot-teleoperate --teleop.type=keyboard`): LeRobot's
generic keyboard teleoperator emits raw key names, not joint deltas, and the CLI's
default processor pipeline doesn't translate keys into so101_follower's joint action
space - confirmed by reading lerobot 0.6.1 source (see diagnostics/UAT/function/UAT.md
Step 3). This script drives the robot directly via LeRobot's SO101Follower/
KeyboardTeleop classes with a continuous P-control loop, which is what XLeRobot's
example does for so100_follower - same SOFollower class under the hood, so it applies
directly to our so101_follower hardware.

Controls (key: joint, direction):
  Q/A: shoulder_pan   -/+
  W/S: shoulder_lift   -/+
  E/D: elbow_flex      -/+
  R/F: wrist_flex       -/+
  T/G: wrist_roll        -/+
  Y/H: gripper (0-100%)  -/+ (5% steps)
  X:   exit (returns to start position first)

Robot continuously moves toward the current target at all times (P control), not
just on keypress - this is what makes motion smooth instead of stepping.

Usage:
    python keyboard_joint_control.py [PORT] [ROBOT_ID]
"""

import sys
import time
import traceback

from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SO101Follower
from lerobot.teleoperators.keyboard.configuration_keyboard import KeyboardTeleopConfig
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

JOINT_KEYS = {
    "q": ("shoulder_pan", -1),
    "a": ("shoulder_pan", 1),
    "w": ("shoulder_lift", -1),
    "s": ("shoulder_lift", 1),
    "e": ("elbow_flex", -1),
    "d": ("elbow_flex", 1),
    "r": ("wrist_flex", -1),
    "f": ("wrist_flex", 1),
    "t": ("wrist_roll", -1),
    "g": ("wrist_roll", 1),
    "y": ("gripper", -1),
    "h": ("gripper", 1),
}


def read_positions(robot, retries=5, backoff=0.05):
    """
    robot.get_observation() -> sync_read occasionally fails with "no status
    packet" - a known, unresolved LeRobot/Feetech issue (acknowledged by a
    HuggingFace maintainer as "very very hard to debug", see
    github.com/huggingface/lerobot/issues/3131). LeRobot's own num_read_retries
    already retries a few times internally before raising; this adds an outer
    retry with backoff so one transient dropout doesn't crash the whole control
    loop, matching the community workaround from that same issue thread.
    """
    last_error = None
    for attempt in range(retries):
        try:
            obs = robot.get_observation()
            return {key.removesuffix(".pos"): val for key, val in obs.items() if key.endswith(".pos")}
        except ConnectionError as e:
            last_error = e
            time.sleep(backoff)
    raise last_error


def move_to_positions(robot, targets, kp, control_freq, max_seconds, stop_error=2.0, label=""):
    control_period = 1.0 / control_freq
    max_steps = int(max_seconds * control_freq)
    for step in range(max_steps):
        try:
            current = read_positions(robot)
        except ConnectionError as e:
            print(f"Read failed, skipping this tick: {e}")
            time.sleep(control_period)
            continue
        action = {}
        total_error = 0.0
        for joint, target in targets.items():
            if joint not in current:
                continue
            error = target - current[joint]
            total_error += abs(error)
            action[f"{joint}.pos"] = current[joint] + kp * error
        if action:
            try:
                robot.send_action(action)
            except (ConnectionError, RuntimeError) as e:
                print(f"Write failed, skipping this tick: {e}")
        if step % (control_freq // 2) == 0 and label:
            print(f"{label}: {min(100.0, (step / max_steps) * 100):.0f}%")
        if total_error < stop_error:
            break
        time.sleep(control_period)


def control_loop(robot, keyboard, start_positions, kp=0.5, control_freq=30):
    control_period = 1.0 / control_freq
    # Start targets at wherever the arm currently is, not a forced zero pose - the
    # bulk "move everything to zero at once" step proved unreliable on this bus
    # (tripped an overload once, crashed mid-move another time, dropped comms a
    # third time) and isn't actually needed: the P-loop is stable driving from any
    # starting position via keyboard.
    target_positions = dict(start_positions)

    print("Starting P-control loop. Press X to exit (returns to start position first).")
    while True:
        try:
            keyboard_action = keyboard.get_action()
            if keyboard_action:
                for key in keyboard_action:
                    if key == "x":
                        print("Exit requested - returning to start position...")
                        move_to_positions(robot, start_positions, kp=0.2, control_freq=control_freq, max_seconds=5.0)
                        return
                    if key in JOINT_KEYS:
                        joint, direction = JOINT_KEYS[key]
                        current_target = target_positions[joint]
                        if joint == "gripper":
                            new_target = max(0.0, min(100.0, current_target + direction * 5.0))
                        else:
                            # DEGREES-mode joints (SOFollowerConfig.use_degrees=True default,
                            # confirmed by reading config_so_follower.py; see
                            # control/vla_bridge/action_contract.py for the full documented
                            # contract) - unclamped accumulation here let repeated presses drive
                            # the target past +-100 indefinitely, pushing the servo into its
                            # mechanical hard stop and tripping overload protection (observed on
                            # wrist_flex).
                            new_target = max(-100.0, min(100.0, current_target + direction))
                        target_positions[joint] = new_target
                        print(f"{joint}: target -> {new_target:.1f}")

            try:
                current = read_positions(robot)
            except ConnectionError as e:
                print(f"Read failed after retries, skipping this tick: {e}")
                time.sleep(control_period)
                continue

            action = {}
            for joint, target in target_positions.items():
                if joint not in current:
                    continue
                error = target - current[joint]
                action[f"{joint}.pos"] = current[joint] + kp * error
            if action:
                try:
                    robot.send_action(action)
                except (ConnectionError, RuntimeError) as e:
                    print(f"Write failed, skipping this tick: {e}")
            time.sleep(control_period)
        except KeyboardInterrupt:
            print("Interrupted.")
            break
        except Exception:
            traceback.print_exc()
            break


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else input("SO-101 port (e.g. /dev/cu.usbmodem5B8E1132011): ").strip()
    robot_id = sys.argv[2] if len(sys.argv) > 2 else input("Robot ID (e.g. soarm_follower_01): ").strip()

    robot = SO101Follower(SOFollowerRobotConfig(port=port, id=robot_id))
    keyboard = KeyboardTeleop(KeyboardTeleopConfig())

    robot.connect(calibrate=False)
    # connect(calibrate=False) skips write_calibration() - apply the saved
    # calibration.json explicitly so hardware position limits match what
    # lerobot-calibrate recorded (see joint_jog.py for the full explanation).
    # Min/Max_Position_Limit are EEPROM addresses the servo rejects while torque
    # is enabled ("Incorrect status packet"), so this must happen with torque off,
    # same as the library's own calibrate() path (so_follower.py disable_torque()
    # before its write_calibration() calls).
    with robot.bus.torque_disabled():
        robot.bus.write_calibration(robot.calibration)

    # Previously this reduced shoulder_lift/elbow_flex to P=10/D=24 (vs LeRobot's
    # default P=16/D=32) because the OLD 7.4V/19.5kg-cm servos overheated 3x under
    # sustained holding - a softer gain tolerated more sag to cut sustained current.
    # Follower has since been rebuilt with 12V/30kg-cm servos (see PARTS_LIST.md,
    # "Pending Hardware Change" - gains/protection settings explicitly flagged there
    # as needing re-verification, not carryover, after the swap). A soft gain is
    # actually the wrong direction on the new hardware: more sag under load means
    # the servo corrects harder/longer to hold position, building sustained current
    # over time - a plausible cause of the overload trip seen lifting a payload with
    # the old P=10/D=24 values still applied. Reverted to LeRobot's stock gains;
    # re-tune only if the new servos show their own overheating pattern.

    keyboard.connect()
    print("Connected.")

    start_positions = {k: v for k, v in read_positions(robot).items()}
    print("Start positions:", {k: round(v, 1) for k, v in start_positions.items()})
    print("Starting from current position (no forced move-to-zero) - drive from here via keyboard.")

    print("Controls: Q/A W/S E/D R/F T/G shoulder_pan/shoulder_lift/elbow_flex/wrist_flex/wrist_roll -/+, Y/H gripper -/+, X exit")
    control_loop(robot, keyboard, start_positions)

    robot.disconnect()
    keyboard.disconnect()
    print("Disconnected.")


if __name__ == "__main__":
    main()
