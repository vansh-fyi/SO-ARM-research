"""
Assign a unique ID to a single Feetech STS3215 servo.

IMPORTANT: connect exactly ONE servo to the bus when running this. Brand-new
servos usually all share the same factory default ID (1) - if more than one is
connected, they'll collide answering the ping and this will fail or, worse, write
the new ID to the wrong/multiple servos.

Usage:
    python servo_set_id.py PORT --old-id 1 --new-id 3 [--baud 1000000]

Typical flow for 6 fresh servos: connect servo A alone -> set to ID 1 -> disconnect
-> connect servo B alone -> set to ID 2 (--old-id 1, since B is still factory-default
1) -> ... repeat through ID 6 -> THEN daisy-chain all 6 together and run
servo_scan.py to confirm all 6 respond.
"""
import argparse
import sys

from scservo_sdk import PacketHandler, PortHandler

STS_PROTOCOL_END = 0
ADDR_ID = 5
ADDR_LOCK = 55
BROADCAST_ID = 0xFE


def set_id(port_name: str, old_id: int, new_id: int, baud: int):
    port = PortHandler(port_name)
    if not port.openPort() or not port.setBaudRate(baud):
        print(f"Failed to open {port_name} @ {baud}")
        sys.exit(1)

    packet = PacketHandler(STS_PROTOCOL_END)

    model, result, error = packet.ping(port, old_id)
    if result != 0:
        print(
            f"No servo responded at ID {old_id}. Confirm exactly one servo is "
            f"connected and it's actually at that ID (try servo_scan.py first)."
        )
        port.closePort()
        sys.exit(1)
    print(f"Found servo at ID {old_id} (model {model}).")

    # Unlock EEPROM, write new ID, re-lock.
    packet.write1ByteTxRx(port, old_id, ADDR_LOCK, 0)
    w_result, w_error = packet.write1ByteTxRx(port, old_id, ADDR_ID, new_id)
    packet.write1ByteTxRx(port, new_id, ADDR_LOCK, 1)

    # Confirm under the new ID.
    model2, result2, error2 = packet.ping(port, new_id)
    port.closePort()

    if result2 == 0:
        print(f"OK: servo now responds at ID {new_id}.")
    else:
        print(f"Wrote new ID but servo did not respond at {new_id} - re-check with servo_scan.py.")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("port")
    parser.add_argument("--old-id", type=int, default=1, help="current/factory-default ID")
    parser.add_argument("--new-id", type=int, required=True)
    parser.add_argument("--baud", type=int, default=1000000)
    args = parser.parse_args()

    set_id(args.port, args.old_id, args.new_id, args.baud)
