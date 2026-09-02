"""
Move a single Feetech STS3215 servo a small, safe amount and confirm it responds.

Moves by a relative offset (default +100 ticks, ~8.8 degrees out of 4096/rev) from
its current position, polls until it actually stops moving (not a fixed timer - a
short fixed wait can misreport "done" while a heavy/loaded joint is still mid-travel),
then reads the final position.

Safety: torque is left ENABLED (holding) when this finishes, unless --release is
passed. For a weight-bearing arm joint, cutting torque while it's still elevated
mid-travel lets it drop under gravity - only release torque on joints that won't fall
or strain anything when unpowered (e.g. the gripper, or a joint you're physically
supporting by hand).

Usage:
    python servo_move_test.py PORT ID [--offset 100] [--speed 200] [--baud 1000000] [--release]
"""
import argparse
import sys
import time

from scservo_sdk import PacketHandler, PortHandler

STS_PROTOCOL_END = 0
ADDR_TORQUE_ENABLE = 40
ADDR_GOAL_POSITION = 42  # 2 bytes
ADDR_GOAL_SPEED = 46  # 2 bytes
ADDR_PRESENT_POSITION = 56  # 2 bytes

POLL_INTERVAL_S = 0.25
STABLE_POLLS_REQUIRED = 3  # ~0.75s of no movement = considered arrived/stalled
STABLE_TOLERANCE_TICKS = 2
MAX_WAIT_S = 10.0


def move_test(port_name: str, servo_id: int, offset: int, speed: int, baud: int, release: bool):
    port = PortHandler(port_name)
    if not port.openPort() or not port.setBaudRate(baud):
        print(f"Failed to open {port_name} @ {baud}")
        sys.exit(1)

    packet = PacketHandler(STS_PROTOCOL_END)

    model, result, error = packet.ping(port, servo_id)
    if result != 0:
        print(f"ID {servo_id}: no response, aborting")
        port.closePort()
        sys.exit(1)

    start_pos, r1, _ = packet.read2ByteTxRx(port, servo_id, ADDR_PRESENT_POSITION)
    if r1 != 0:
        print(f"ID {servo_id}: could not read start position, aborting")
        port.closePort()
        sys.exit(1)
    print(f"ID {servo_id}: start position = {start_pos}")

    target = max(0, min(4095, start_pos + offset))
    if target == start_pos and offset != 0:
        print(
            f"ID {servo_id}: requested offset {offset:+d} would push raw position "
            f"past the 0/4095 boundary - clamped to a no-op ({start_pos} -> {target}). "
            f"This is a script-side guard against encoder wraparound, not a servo/mechanical "
            f"limit. Move to a mid-range position first (e.g. via servo_torque.py or a smaller "
            f"offset in the other direction) before retrying this direction."
        )
        port.closePort()
        return
    packet.write1ByteTxRx(port, servo_id, ADDR_TORQUE_ENABLE, 1)
    packet.write2ByteTxRx(port, servo_id, ADDR_GOAL_SPEED, speed)
    packet.write2ByteTxRx(port, servo_id, ADDR_GOAL_POSITION, target)
    print(f"ID {servo_id}: commanded move to {target} (offset {offset:+d}), polling until it stops...")

    last_pos = start_pos
    stable_count = 0
    elapsed = 0.0
    while elapsed < MAX_WAIT_S:
        time.sleep(POLL_INTERVAL_S)
        elapsed += POLL_INTERVAL_S
        pos, r, _ = packet.read2ByteTxRx(port, servo_id, ADDR_PRESENT_POSITION)
        if r != 0:
            continue
        if abs(pos - last_pos) <= STABLE_TOLERANCE_TICKS:
            stable_count += 1
        else:
            stable_count = 0
        last_pos = pos
        if stable_count >= STABLE_POLLS_REQUIRED:
            break

    end_pos = last_pos
    if release:
        packet.write1ByteTxRx(port, servo_id, ADDR_TORQUE_ENABLE, 0)
        torque_note = "torque released"
    else:
        torque_note = "torque still holding"
    port.closePort()

    moved = abs(end_pos - start_pos)
    print(f"ID {servo_id}: end position = {end_pos} (moved {moved} ticks, {torque_note})")
    if moved >= abs(offset) * 0.5:
        print(f"ID {servo_id}: PASS - servo moved as commanded")
    else:
        print(
            f"ID {servo_id}: FAIL - servo did not move enough (check power/mechanics, "
            f"or it may still be under-torqued for the load - not necessarily a defect)"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("port")
    parser.add_argument("id", type=int)
    parser.add_argument("--offset", type=int, default=100)
    parser.add_argument("--speed", type=int, default=200)
    parser.add_argument("--baud", type=int, default=1000000)
    parser.add_argument("--release", action="store_true", help="release torque after the move (default: keep holding)")
    args = parser.parse_args()

    move_test(args.port, args.id, args.offset, args.speed, args.baud, args.release)
