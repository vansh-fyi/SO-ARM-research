"""
Set a Feetech STS3215 servo's thermal/overload protection registers. LeRobot's own
so_follower.py already does this for the gripper only (Max_Torque_Limit=500,
Protection_Current=250, Overload_Torque=25 - see configure() in
lerobot/robots/so_follower/so_follower.py); this generalizes the same idea to any
joint, since shoulder_lift has repeatedly overheated under sustained gravity load
and gets no such protection by default.

Registers (from lerobot's feetech/tables.py control table):
  Max_Torque_Limit (addr 16, 2 bytes, 0-1000): hard ceiling on commanded torque.
  Protection_Current (addr 28, 2 bytes): current threshold that triggers overload
    protection. NOTE: exact full-scale is not confirmed from Feetech's own docs
    here - LeRobot's comment for the gripper says "250 = 50% of max current",
    implying full scale ~500, NOT the 0-1000 scale Max_Torque_Limit/Torque_Limit
    use. Verify by reading back after writing, don't trust blindly.
  Overload_Torque (addr 36, 1 byte, 0-100%): torque the servo falls back to (not
    zero) once overload protection actually triggers - this is the useful part
    for a weight-bearing joint like shoulder_lift, since dropping to 0% torque
    while holding the arm's weight is itself dangerous (uncontrolled fall).

Unlike the gripper's aggressive 50% cap (fine for a small pinch grip), shoulder_lift
genuinely needs most of its available torque to function at moderate-to-extended
reach (confirmed empirically this session) - don't blindly reuse the gripper's
values. Defaults here are conservative: shave a little off the peak, keep a
non-zero fallback torque, leave real headroom for normal operation.

Usage:
    python servo_set_protection.py PORT ID [--max-torque 900] [--protection-current 400] [--overload-torque 25]
"""
import argparse
import sys

from scservo_sdk import PacketHandler, PortHandler

STS_PROTOCOL_END = 0
ADDR_MAX_TORQUE_LIMIT = 16  # 2 bytes, 0-1000
ADDR_PROTECTION_CURRENT = 28  # 2 bytes, scale unconfirmed - verify via readback
ADDR_OVERLOAD_TORQUE = 36  # 1 byte, 0-100 (%)
ADDR_LOCK = 55


def set_protection(port_name: str, servo_id: int, max_torque: int, protection_current: int, overload_torque: int, baud: int):
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
    packet.write2ByteTxRx(port, servo_id, ADDR_MAX_TORQUE_LIMIT, max_torque)
    packet.write2ByteTxRx(port, servo_id, ADDR_PROTECTION_CURRENT, protection_current)
    packet.write1ByteTxRx(port, servo_id, ADDR_OVERLOAD_TORQUE, overload_torque)
    packet.write1ByteTxRx(port, servo_id, ADDR_LOCK, 1)

    def r2(addr):
        v, r, _ = packet.read2ByteTxRx(port, servo_id, addr)
        return v if r == 0 else None

    def r1(addr):
        v, r, _ = packet.read1ByteTxRx(port, servo_id, addr)
        return v if r == 0 else None

    readback = {
        "Max_Torque_Limit": r2(ADDR_MAX_TORQUE_LIMIT),
        "Protection_Current": r2(ADDR_PROTECTION_CURRENT),
        "Overload_Torque": r1(ADDR_OVERLOAD_TORQUE),
    }
    port.closePort()

    print(f"ID {servo_id}: wrote max_torque={max_torque}, protection_current={protection_current}, overload_torque={overload_torque}")
    print(f"ID {servo_id}: readback -> {readback}")

    ok = (
        readback["Max_Torque_Limit"] == max_torque
        and readback["Protection_Current"] == protection_current
        and readback["Overload_Torque"] == overload_torque
    )
    if not ok:
        print(f"ID {servo_id}: WARNING - readback does not match what was written, verify manually")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("port")
    parser.add_argument("id", type=int)
    parser.add_argument("--max-torque", type=int, default=900, help="0-1000, ceiling on commanded torque")
    parser.add_argument("--protection-current", type=int, default=400, help="scale unconfirmed - verify via readback")
    parser.add_argument("--overload-torque", type=int, default=25, help="0-100%%, fallback torque once overload trips (not 0, to avoid an uncontrolled fall on a weight-bearing joint)")
    parser.add_argument("--baud", type=int, default=1000000)
    args = parser.parse_args()

    set_protection(args.port, args.id, args.max_torque, args.protection_current, args.overload_torque, args.baud)
