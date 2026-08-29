# diagnostics

Hardware bring-up scripts for the SOARM electronics — servo bus scans, single-servo
ping/move tests, camera checks — kept separate from the LIBERO/sim code.

Parts on hand: see [`PARTS_LIST.md`](./PARTS_LIST.md) (6x Feetech STS3215 servos on
a Waveshare Bus Servo Adapter, IMX335 wrist cam, AR0144 stereo overhead cam).

## Setup (one-time)

The Python env auto-loads via `direnv` when you `cd` into this directory (already
hooked into `~/.zshrc` and allowed for this repo). If it's your first time here:

```
cd diagnostics
direnv allow .
```

If `direnv` isn't hooked into your shell, activate manually instead:

```
cd diagnostics
source .venv/bin/activate
```

## What's installed

- `pyserial` — raw serial port access, for verifying the Waveshare adapter enumerates
  before going through the servo SDK.
- `feetech-servo-sdk` (`scservo_sdk`) — vendor SDK for the STS3215 bus servos
  (ping/read/write over the half-duplex TTL bus).
- `opencv-python` / `numpy` — grabbing test frames from the USB cameras.

## Power first

Before touching servos: 12V wall supply -> XL4015 buck converter -> should read
**~7.4V** on the converter's LED voltmeter (matches the STS3215 rating) before it
feeds the servo bus. Don't connect servos to unregulated 12V.

## 1. Assign unique servo IDs (do this before first bus scan)

Brand-new Feetech servos all ship with the same factory default ID (1). Connect
**one servo at a time** and give each a unique ID before daisy-chaining them —
otherwise they collide answering the same ping and a full-bus scan finds nothing:

```
python servo_set_id.py /dev/tty.usbserial-XXXX --old-id 1 --new-id 1   # first servo
python servo_set_id.py /dev/tty.usbserial-XXXX --old-id 1 --new-id 2   # next servo, still at factory default 1
...through --new-id 6
```

Only after all 6 have unique IDs should you daisy-chain them together.

## 2. Servo bus scan

Find the Waveshare adapter's port, then ping IDs 1-6 (arm joints 1-5 + gripper):

```
python servo_scan.py                                # lists ports
python servo_scan.py /dev/tty.usbserial-XXXX         # scans IDs 1-6 @ 1,000,000 baud
python servo_scan.py /dev/tty.usbserial-XXXX --sweep # try all common baud rates (broadcast ping)
```

A servo that doesn't respond usually means: not daisy-chained onto the bus, duplicate
ID (see step 1), or the buck converter isn't delivering 7.4V under load.

## 3. Single-servo move test

```
python servo_move_test.py /dev/tty.usbserial-XXXX 1
```

Nudges the given servo ID by a small offset (default +100 ticks, ~8.8°) and confirms
it actually moved. Releases torque afterward.

## 4. Set servo max torque limit

macOS-native replacement for the vendor gripper guide's Windows-only `FD.exe` step:

```
python servo_set_torque_limit.py /dev/tty.usbserial-XXXX 6 --limit 500
```

Value is 0-1000 (0-100.0% of rated torque). Persists across power cycles (EEPROM).

## 5. Camera test

Grabs one frame from every USB camera index found and saves it to
`diagnostics/outputs/`:

```
python camera_test.py
```

The AR0144 stereo module reports as a single wide frame (left+right side by side,
~2x width vs height) — split it down the middle for stereo pairs. On macOS you may
need to grant your terminal app Camera permission (System Settings -> Privacy &
Security -> Camera) the first time.

## UAT

Tracked step-by-step checklists live under `UAT/`:

- [`UAT/components/UAT.md`](./UAT/components/UAT.md) — electronics bring-up (power
  rail → adapter → bus scan → per-servo move → load check → cameras). **Do this
  first**, before assembling anything.
- [`UAT/assembly/gripper/UAT.md`](./UAT/assembly/gripper/UAT.md) — gripper assembly.
- `UAT/assembly/main/UAT.md` — full arm assembly (added once the gripper is done).

## Adding more diagnostics

Add new one-off scripts here (e.g. single-servo move test, torque/current check) and
extend `requirements.txt` + `pip install -r requirements.txt` as needed.
