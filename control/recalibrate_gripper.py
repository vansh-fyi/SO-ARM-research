"""
Recalibrate only the gripper joint's Homing_Offset + range, leaving the other 5
joints' calibration untouched.

Why this exists: `record_ranges_of_motion` (lerobot/motors/motors_bus.py) tracks a
plain running min/max of raw Present_Position with no wraparound handling. If the
gripper's operating span straddles the 4095->0 rollover point (STS3215 is a 4096-
count single-turn encoder), the recorded range comes out as ~4000+ ticks (nearly a
full rotation) instead of the true short arc a gripper actually travels - which is
exactly what happened here (range_min=14, range_max=4090 recorded for a mechanism
that only swings a few hundred ticks in reality). This happens when the gripper
wasn't at its true mechanical half-open point during the shared "move every joint to
the middle of its range" step in the original full calibration.

Fix: re-home just the gripper with the operator holding it at its actual mechanical
half-open point, then re-record just its range - centering its true short span away
from the wrap boundary this time.

Usage:
    python recalibrate_gripper.py PORT ROBOT_ID
"""
import sys

from lerobot.robots.so_follower import SO101Follower
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.motors.motors_bus import MotorCalibration


def main():
    port = sys.argv[1]
    robot_id = sys.argv[2]

    robot = SO101Follower(SOFollowerRobotConfig(port=port, id=robot_id))
    robot.connect(calibrate=False)
    with robot.bus.torque_disabled():
        robot.bus.write_calibration(robot.calibration)

    print(f"Current gripper calibration: {robot.calibration['gripper']}")

    with robot.bus.torque_disabled():
        # Must zero Homing_Offset (and open Min/Max limits) BEFORE reading the
        # midpoint position, same as MotorsBus.set_half_turn_homings() does via
        # reset_calibration(). Without this, Present_Position is still shifted by
        # whatever stale offset was previously written, so the "raw" reading isn't
        # actually raw - this was the bug in the first version of this script.
        robot.bus.reset_calibration(["gripper"])

        input(
            "\nBy hand, move the GRIPPER only to its true mechanical half-open "
            "midpoint (roughly halfway between fully open and fully closed), "
            "then press ENTER..."
        )
        raw_mid = robot.bus.sync_read("Present_Position", ["gripper"], normalize=False, num_retry=5)["gripper"]
        # Same formula as MotorsBus._get_half_turn_homings(): center this physical
        # point at the encoder's mid-code (2047 for a 4096-count servo), so the
        # gripper's actual (short) travel arc sits away from the 0/4095 wrap.
        homing_offset = raw_mid - 2047
        robot.bus.write("Homing_Offset", "gripper", homing_offset)
        print(f"New Homing_Offset: {homing_offset} (raw midpoint was {raw_mid})")

        # This mechanism has no tactile hard stop (confirmed empirically - repeated
        # attempts at a continuous open<->closed sweep kept overshooting into the
        # wraparound zone before the operator could tell they'd passed the real
        # limit). A live sweep can't work reliably here: instead, sample each end
        # as one deliberate static point - move there, hold, press ENTER once.
        print("\nMove the gripper to fully OPEN (visually), hold it there, then press ENTER...")
        input()
        pos_open = robot.bus.sync_read("Present_Position", ["gripper"], normalize=False, num_retry=5)["gripper"]
        print(f"  captured open = {pos_open}")

        print("\nMove the gripper to fully CLOSED (visually), hold it there, then press ENTER...")
        input()
        pos_closed = robot.bus.sync_read("Present_Position", ["gripper"], normalize=False, num_retry=5)["gripper"]
        print(f"  captured closed = {pos_closed}")

        range_min, range_max = sorted((pos_open, pos_closed))

    span = range_max - range_min
    print(f"\nRecorded range: {range_min}-{range_max} (span {span})")
    if span > 1500:
        print(
            "WARNING: span looks too large for a gripper (>1500 ticks). Likely one "
            "of the two captures wasn't the true visual extreme. Re-run before "
            "trusting this."
        )
        sys.exit(1)

    robot.calibration["gripper"] = MotorCalibration(
        id=robot.calibration["gripper"].id,
        drive_mode=0,
        homing_offset=int(homing_offset),
        range_min=int(range_min),
        range_max=int(range_max),
    )
    with robot.bus.torque_disabled():
        robot.bus.write_calibration(robot.calibration)
    robot._save_calibration()
    print(f"Saved updated calibration (gripper only) to {robot.calibration_fpath}")

    robot.disconnect()


if __name__ == "__main__":
    main()
