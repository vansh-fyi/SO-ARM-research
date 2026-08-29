"""
Drive a servo toward one extreme until it genuinely stops moving (stalls against a
hard mechanical stop or reaches goal), rather than reading position after a fixed
timer. Use this for finding real open/closed limits during calibration - a fixed
short wait can misreport "closed" while the servo is still mid-travel on a large
commanded offset.

Safety: run this only with torque limit already capped (see servo_set_torque_limit.py)
so a real stall just holds position at the limit instead of grinding.

Usage:
    python servo_drive_to_stall.py PORT ID --offset 1500 [--speed 200] [--baud 1000000] [--max-wait 20]

A large --offset (e.g. 1500-2000) in the desired direction works fine even if the
mechanism can't travel that far - it'll just stall at the real limit and we read
where it actually stopped, not the requested target.
"""
import argparse
import sys
import time

from scservo_sdk import PacketHandler, PortHandler

STS_PROTOCOL_END = 0
ADDR_TORQUE_ENABLE = 40
ADDR_GOAL_POSITION = 42
ADDR_GOAL_SPEED = 46
ADDR_PRESENT_POSITION = 56

POLL_INTERVAL_S = 0.25
STALL_POLLS_REQUIRED = 4  # ~1s of no movement = considered stalled/arrived
STALL_TOLERANCE_TICKS = 2


def drive_to_stall(port_name: str, servo_id: int, offset: int, speed: int, baud: int, max_wait: float):
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
            f"ID {servo_id}: requested offset {offset:+d} clamps to a no-op at the "
            f"0/4095 boundary ({start_pos} -> {target}). Move away from that edge "
            f"first (opposite-direction offset) before probing this direction."
        )
        port.closePort()
        sys.exit(1)

    packet.write1ByteTxRx(port, servo_id, ADDR_TORQUE_ENABLE, 1)
    packet.write2ByteTxRx(port, servo_id, ADDR_GOAL_SPEED, speed)
    packet.write2ByteTxRx(port, servo_id, ADDR_GOAL_POSITION, target)
    print(f"ID {servo_id}: commanded move to {target} (offset {offset:+d}), polling until it stalls...")

    last_pos = start_pos
    stable_count = 0
    elapsed = 0.0
    while elapsed < max_wait:
        time.sleep(POLL_INTERVAL_S)
        elapsed += POLL_INTERVAL_S
        pos, r, _ = packet.read2ByteTxRx(port, servo_id, ADDR_PRESENT_POSITION)
        if r != 0:
            continue
        if abs(pos - last_pos) <= STALL_TOLERANCE_TICKS:
            stable_count += 1
        else:
            stable_count = 0
        last_pos = pos
        if stable_count >= STALL_POLLS_REQUIRED:
            break
    else:
        # loop exhausted MAX_WAIT_S without ever going stable - still moving, not stalled
        packet.write1ByteTxRx(port, servo_id, ADDR_TORQUE_ENABLE, 0)
        port.closePort()
        print(
            f"ID {servo_id}: TIMED OUT after {max_wait}s, still at {last_pos} (target {target}) "
            f"and still moving when we gave up - NOT a confirmed stall. Retry with a higher "
            f"--max-wait, a higher --speed, or a smaller --offset closer to the actual limit."
        )
        return

    packet.write1ByteTxRx(port, servo_id, ADDR_TORQUE_ENABLE, 0)  # release torque
    port.closePort()

    reached_target = abs(last_pos - target) <= STALL_TOLERANCE_TICKS
    if reached_target:
        print(f"ID {servo_id}: reached commanded target, resting at {last_pos} (no stall - didn't hit a limit)")
    else:
        print(f"ID {servo_id}: STALLED at {last_pos} (target was {target}) - this is the real mechanical limit")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("port")
    parser.add_argument("id", type=int)
    parser.add_argument("--offset", type=int, required=True)
    parser.add_argument("--speed", type=int, default=200)
    parser.add_argument("--baud", type=int, default=1000000)
    parser.add_argument("--max-wait", type=float, default=20.0, help="seconds to poll before giving up")
    args = parser.parse_args()

    drive_to_stall(args.port, args.id, args.offset, args.speed, args.baud, args.max_wait)
