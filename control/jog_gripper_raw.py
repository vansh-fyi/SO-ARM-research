"""
Jog the gripper servo directly in raw encoder ticks (0-4095), under motor control
(small commanded steps), instead of hand-moving it with torque disabled.

Why this exists: hand-calibrating this gripper's true open/closed limits kept
failing because the mechanism has no tactile hard stop - a hand sweep overshoots
the real limit before you can feel it and stop, wrapping the raw counter around.
Driving it via small commanded position steps instead means every step is small,
deliberate, and stoppable the instant you see (visually) the jaws hit their real
limit - no momentum to overshoot with.

This intentionally has NO calibration applied (works in raw 0-4095 ticks, not
percent-open) and NO range clamping - by design, since figuring out the true
range is the point. That means it CAN be driven into the same wraparound zone as
a hand sweep if you hold a key down carelessly - use small steps and watch the
printed position after every single step, not multiple steps blind.

Controls: type "o"+ENTER = open a bit (raw -), "c"+ENTER = close a bit (raw +) -
direction is a guess, correct it after the first step once you see which way
position actually moved. "q"+ENTER = quit and print the final raw position.

Deliberately uses blocking line input, not a live keyboard listener: a held key
with the live-polling version fires many steps back-to-back with no way to stop
between them (confirmed - this is what caused an overload hitting a real stop at
speed). Requiring ENTER after each single step makes multi-step runaway
impossible regardless of how a key is pressed.

Usage:
    python jog_gripper_raw.py PORT ROBOT_ID [--step 15]
"""
import argparse
import sys

from lerobot.robots.so_follower import SO101Follower
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("port")
    parser.add_argument("robot_id")
    parser.add_argument("--step", type=int, default=15, help="raw ticks per keypress")
    args = parser.parse_args()

    robot = SO101Follower(SOFollowerRobotConfig(port=args.port, id=args.robot_id))
    robot.connect(calibrate=False)
    # Work in true raw ticks: zero the homing offset and open the position limits
    # to the full range, same as MotorsBus.reset_calibration() does before any
    # homing step. Calibration isn't finalized yet - that's what this tool is for.
    robot.bus.reset_calibration(["gripper"])
    robot.bus.enable_torque(["gripper"])

    pos = robot.bus.sync_read("Present_Position", ["gripper"], normalize=False, num_retry=5)["gripper"]
    print(f"Starting raw position: {pos}")
    print(f"o+ENTER = open a bit (-{args.step}), c+ENTER = close a bit (+{args.step}), q+ENTER = quit")
    print("Exactly one step per ENTER - no key repeat, no runaway.")

    try:
        while True:
            key = input("> ").strip().lower()
            if key == "q":
                print(f"Final raw position: {pos}")
                return
            elif key in ("o", "c"):
                delta = -args.step if key == "o" else args.step
                target = pos + delta
                if not (0 <= target <= 4095):
                    print(f"  refusing to send {target} - would immediately wrap, back off first")
                    continue
                try:
                    robot.bus.write("Goal_Position", "gripper", target, normalize=False)
                except RuntimeError as e:
                    print(f"  write failed ({e}) - likely hit a real resistance-based stop right at {pos}.")
                    continue
                new_pos = robot.bus.sync_read("Present_Position", ["gripper"], normalize=False, num_retry=5)["gripper"]
                jump = new_pos - pos
                if abs(jump) > 2048:
                    print(f"  WRAPPED ({pos} -> {new_pos}) - you just passed the real limit. Step back the other way.")
                else:
                    print(f"  pos: {pos} -> {new_pos}")
                pos = new_pos
            else:
                print("  unrecognized - use o, c, or q")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
