"""
Move a single Feetech STS3215 servo a small, safe amount and confirm it responds.

Moves by a relative offset (default +100 ticks, ~8.8 degrees out of 4096/rev) from
its current position, waits, then reads the new position back. Small default offset
so an unpowered/uncalibrated joint doesn't slam into a hard stop.

Usage:
    python servo_move_test.py PORT ID [--offset 100] [--speed 200] [--baud 1000000]
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


def move_test(port_name: str, servo_id: int, offset: int, speed: int, baud: int):
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
    print(f"ID {servo_id}: commanded move to {target} (offset {offset:+d})")

    time.sleep(1.5)
    end_pos, r2, _ = packet.read2ByteTxRx(port, servo_id, ADDR_PRESENT_POSITION)
    packet.write1ByteTxRx(port, servo_id, ADDR_TORQUE_ENABLE, 0)  # release torque
    port.closePort()

    if r2 != 0:
        print(f"ID {servo_id}: could not read end position")
        sys.exit(1)

    moved = abs(end_pos - start_pos)
    print(f"ID {servo_id}: end position = {end_pos} (moved {moved} ticks)")
    if moved >= abs(offset) * 0.5:
        print(f"ID {servo_id}: PASS - servo moved as commanded")
    else:
        print(f"ID {servo_id}: FAIL - servo did not move enough (check power/mechanics)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("port")
    parser.add_argument("id", type=int)
    parser.add_argument("--offset", type=int, default=100)
    parser.add_argument("--speed", type=int, default=200)
    parser.add_argument("--baud", type=int, default=1000000)
    args = parser.parse_args()

    move_test(args.port, args.id, args.offset, args.speed, args.baud)
