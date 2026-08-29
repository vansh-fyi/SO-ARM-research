"""
Enable or disable torque on a servo, and/or print its current raw position.

With torque OFF, the servo can be moved by hand (useful for manually finding the
gripper's true open/closed extremes during calibration). Position is a raw 12-bit
encoder tick (0-4095 per revolution).

Usage:
    python servo_torque.py PORT ID --off      # release torque, print position
    python servo_torque.py PORT ID --on       # re-enable torque, print position
    python servo_torque.py PORT ID            # just print position, don't change torque
"""
import argparse
import sys

from scservo_sdk import PacketHandler, PortHandler

STS_PROTOCOL_END = 0
ADDR_TORQUE_ENABLE = 40
ADDR_PRESENT_POSITION = 56


def run(port_name: str, servo_id: int, set_torque: int | None, baud: int):
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

    if set_torque is not None:
        packet.write1ByteTxRx(port, servo_id, ADDR_TORQUE_ENABLE, set_torque)
        state = "released (movable by hand)" if set_torque == 0 else "enabled (holding position)"
        print(f"ID {servo_id}: torque {state}")

    pos, pos_result, pos_error = packet.read2ByteTxRx(port, servo_id, ADDR_PRESENT_POSITION)
    port.closePort()
    if pos_result == 0:
        print(f"ID {servo_id}: current position = {pos}")
    else:
        print(f"ID {servo_id}: could not read position")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("port")
    parser.add_argument("id", type=int)
    parser.add_argument("--baud", type=int, default=1000000)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--off", action="store_true", help="release torque")
    group.add_argument("--on", action="store_true", help="enable torque")
    args = parser.parse_args()

    set_torque = None
    if args.off:
        set_torque = 0
    elif args.on:
        set_torque = 1

    run(args.port, args.id, set_torque, args.baud)
