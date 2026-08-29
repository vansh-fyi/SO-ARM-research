"""
Set a Feetech STS3215 servo's max torque limit (EEPROM, persists across power cycles).

macOS/Linux replacement for the vendor gripper assembly guide's Windows-only FD.exe
step ("Programming" tab -> Max Torque Limit -> 500 -> Save). Value is 0-1000,
representing 0-100.0% of the servo's rated torque.

Usage:
    python servo_set_torque_limit.py PORT ID --limit 500 [--baud 1000000]
"""
import argparse
import sys

from scservo_sdk import PacketHandler, PortHandler

STS_PROTOCOL_END = 0
ADDR_LOCK = 55
ADDR_TORQUE_LIMIT = 48  # 2 bytes, EEPROM


def set_torque_limit(port_name: str, servo_id: int, limit: int, baud: int):
    if not 0 <= limit <= 1000:
        print("Limit must be 0-1000 (0-100.0% of rated torque)")
        sys.exit(1)

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

    packet.write1ByteTxRx(port, servo_id, ADDR_LOCK, 0)
    packet.write2ByteTxRx(port, servo_id, ADDR_TORQUE_LIMIT, limit)
    packet.write1ByteTxRx(port, servo_id, ADDR_LOCK, 1)

    readback, r_result, r_error = packet.read2ByteTxRx(port, servo_id, ADDR_TORQUE_LIMIT)
    port.closePort()

    if r_result == 0 and readback == limit:
        print(f"ID {servo_id}: torque limit set to {limit} (confirmed)")
    else:
        print(f"ID {servo_id}: wrote {limit} but readback was {readback} - verify manually")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("port")
    parser.add_argument("id", type=int)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--baud", type=int, default=1000000)
    args = parser.parse_args()

    set_torque_limit(args.port, args.id, args.limit, args.baud)
