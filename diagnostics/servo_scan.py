"""
Ping every servo on the Feetech STS3215 bus via a Waveshare Bus Servo Adapter.

Usage:
    python servo_scan.py [PORT] [--baud 1000000] [--ids 1-6]

If PORT is omitted, lists available serial ports and exits.
"""
import argparse
import sys

from scservo_sdk import PacketHandler, PortHandler

DEFAULT_BAUD = 1000000  # STS3215 factory default
COMMON_BAUDS = [1000000, 500000, 250000, 128000, 115200, 57600, 38400, 19200, 9600]
STS_PROTOCOL_END = 0  # STS/SMS series use protocol end 0 (vs SCS series = 1)
ADDR_PRESENT_POSITION = 56  # STS3215 control table, 2 bytes
BROADCAST_ID = 0xFE


def list_ports():
    from serial.tools import list_ports as lp

    ports = list(lp.comports())
    if not ports:
        print("No serial ports found. Is the Waveshare adapter plugged in?")
        return
    print("Available serial ports:")
    for p in ports:
        print(f"  {p.device}  ({p.description})")


def scan(port_name: str, baud: int, ids: range):
    port = PortHandler(port_name)
    if not port.openPort():
        print(f"Failed to open {port_name}")
        sys.exit(1)
    if not port.setBaudRate(baud):
        print(f"Failed to set baud rate {baud}")
        sys.exit(1)

    packet = PacketHandler(STS_PROTOCOL_END)
    found = []
    print(f"Scanning IDs {ids.start}-{ids.stop - 1} on {port_name} @ {baud} baud...")
    for servo_id in ids:
        model, result, error = packet.ping(port, servo_id)
        if result == 0:
            pos, pos_result, pos_error = packet.read2ByteTxRx(
                port, servo_id, ADDR_PRESENT_POSITION
            )
            pos_str = f"pos={pos}" if pos_result == 0 else "pos=<read failed>"
            print(f"  ID {servo_id}: FOUND (model {model}, {pos_str})")
            found.append(servo_id)
        else:
            print(f"  ID {servo_id}: no response")

    port.closePort()
    print(f"\n{len(found)} servo(s) responded: {found}")


def baud_sweep(port_name: str):
    """Try a broadcast ping at every common Feetech baud rate to find the live one."""
    port = PortHandler(port_name)
    if not port.openPort():
        print(f"Failed to open {port_name}")
        sys.exit(1)

    packet = PacketHandler(STS_PROTOCOL_END)
    print(f"Sweeping baud rates on {port_name} with a broadcast ping (ID {BROADCAST_ID})...")
    for baud in COMMON_BAUDS:
        if not port.setBaudRate(baud):
            continue
        model, result, error = packet.ping(port, BROADCAST_ID)
        tag = "response!" if result == 0 else "no response"
        print(f"  {baud:>8} baud -> {tag}")
    port.closePort()
    print(
        "\nA broadcast ping getting a clean response is unusual (multiple servos "
        "answering at once usually garbles the packet) - a per-ID scan at the "
        "matching baud is the real test. This is just to find which baud is live."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("port", nargs="?", help="e.g. /dev/tty.usbserial-XXXX")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument("--ids", default="1-6", help="ID range to scan, e.g. 1-6")
    parser.add_argument(
        "--sweep", action="store_true", help="try all common baud rates instead of scanning IDs"
    )
    args = parser.parse_args()

    if not args.port:
        list_ports()
        sys.exit(0)

    if args.sweep:
        baud_sweep(args.port)
        sys.exit(0)

    lo, hi = (int(x) for x in args.ids.split("-"))
    scan(args.port, args.baud, range(lo, hi + 1))
