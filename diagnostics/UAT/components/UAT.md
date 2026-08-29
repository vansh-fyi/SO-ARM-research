# Diagnostics UAT — SOARM Electronics Bring-Up

Human-in-the-loop checklist for testing the physical parts as they come off the
printer / out of the box. Every check here requires real hardware in hand — none of
it is verifiable by an agent, so results are tracked by you filling this in as you go.

Parts under test: see [`PARTS_LIST.md`](../../PARTS_LIST.md).

**How to use:** work top to bottom, run the command shown, fill in Result + Notes.
Use ✅ PASS / ❌ FAIL / ⏭️ SKIPPED. Don't skip Step 1 — powering servos before
confirming voltage risks frying a $20+ servo. All commands below assume you're
running from `diagnostics/` (not from inside `UAT/components/`) — `cd` up two
levels first if you're sitting in this file's directory.

---

## Step 0 — Env sanity

```
cd diagnostics
direnv allow .        # first time only
python -c "import serial, scservo_sdk, cv2; print('env ok')"
```

| Result | Date | Notes |
|---|---|---|
| ✅ PASS | 2026-08-24 | `env ok` printed |

---

## Step 1 — Power rail check (do this before connecting any servo)

1. Plug the 12V 5A wall supply into the XL4015 buck converter.
2. Read the converter's onboard LED voltmeter.
3. Adjust the trim pot (if needed) until it reads **7.4V ± 0.1V**.
4. Confirm output polarity at the servo-bus connector with a multimeter if you have
   one (red = +7.4V, black = GND) before plugging in any servo.

| Result | Measured voltage | Date | Notes |
|---|---|---|---|
| ⚠️ PASS (low) | 7.0V | 2026-08-24 | Buck converter wired into the Waveshare board's power input terminal. Reads 7.0V, not the targeted 7.4V — within STS3215's 6-8.4V tolerance and held steady under load (Step 6), so not blocking, but worth trimming up when convenient. |

**Do not proceed to Step 2 until this reads ~7.4V.** Feeding 12V directly to an
STS3215 will damage it.

---

## Step 2 — Waveshare adapter enumerates over USB

```
python servo_scan.py
```

Expect the adapter to show up as `/dev/tty.usbserial-XXXX` (macOS). Record the exact
port — you'll reuse it for every step below.

| Result | Port found | Date | Notes |
|---|---|---|---|
| ✅ PASS | `/dev/cu.usbmodem5B8E1132011` | 2026-08-24 | Shows as "USB Single Serial" |

---

## Step 3 — Assign unique servo IDs (one servo at a time)

Brand-new Feetech servos ship with the same factory default ID (1). Connect **one
servo at a time** (not daisy-chained) and give each a unique ID before wiring them
all together — otherwise they collide answering the same ping and the bus scan sees
nothing:

```
python servo_set_id.py /dev/tty.usbserial-XXXX --old-id 1 --new-id <N>
```

| Servo | Assigned ID | Result | Date | Notes |
|---|---|---|---|---|
| 1 | 1 | ✅ PASS | 2026-08-24 | model 777 |
| 2 | 2 | ✅ PASS | 2026-08-24 | model 777 |
| 3 | 3 | ✅ PASS | 2026-08-24 | model 777 |
| 4 | 4 | ✅ PASS | 2026-08-24 | model 777 |
| 5 | 5 | ✅ PASS | 2026-08-24 | model 777 |
| 6 | 6 | ✅ PASS | 2026-08-24 | model 777 |

---

## Step 4 — Servo bus scan (all IDs)

With the buck converter at 7.4V and all 6 (now uniquely-IDed) servos daisy-chained
onto the bus:

```
python servo_scan.py /dev/tty.usbserial-XXXX
```

Expect all 6 IDs (1-6: joints 1-5 + gripper) to respond `FOUND`.

| Result | IDs found | Date | Notes |
|---|---|---|---|
| ✅ PASS | [1, 2, 3, 4, 5, 6] | 2026-08-24 | All model 777, all responding after ID assignment (Step 3) |

---

## Step 5 — Individual servo move test

Run once per servo ID (1 through 6), **one joint at a time**, so a stuck/miswired
joint doesn't get masked by others moving correctly:

```
python servo_move_test.py /dev/tty.usbserial-XXXX 1
python servo_move_test.py /dev/tty.usbserial-XXXX 2
python servo_move_test.py /dev/tty.usbserial-XXXX 3
python servo_move_test.py /dev/tty.usbserial-XXXX 4
python servo_move_test.py /dev/tty.usbserial-XXXX 5
python servo_move_test.py /dev/tty.usbserial-XXXX 6   # gripper
```

Each moves the servo a small +100-tick nudge (~8.8°) and reports PASS/FAIL based on
whether it actually moved. Watch the physical joint, not just the terminal — confirm
direction/mechanics look sane, not just that the position register changed.

| Servo ID | Joint | Result | Date | Notes |
|---|---|---|---|---|
| 1 |  | ✅ PASS | 2026-08-24 | 1 -> 99 (98 ticks) |
| 2 |  | ✅ PASS | 2026-08-24 | 4 -> 103 (99 ticks) |
| 3 |  | ✅ PASS | 2026-08-24 | 0 -> 99 (99 ticks) |
| 4 |  | ✅ PASS | 2026-08-24 | 1 -> 100 (99 ticks) |
| 5 |  | ✅ PASS | 2026-08-24 | 0 -> 99 (99 ticks) |
| 6 | gripper | ✅ PASS | 2026-08-24 | 2 -> 102 (100 ticks) |

---

## Step 6 — Multi-joint load check

With all 6 servos connected, run Step 4 on 2-3 joints back-to-back within a few
seconds of each other, and watch the buck converter's voltmeter for sag.

**Why:** per the parts list, 2+ servos stalling simultaneously (2.7A stall current
each) can approach/exceed the 5A converter rating — this catches brownouts before
they cause random resets during real teleop/eval runs.

| Result | Voltage under load | Date | Notes |
|---|---|---|---|
| ✅ PASS | 7.0V (held steady, no sag) | 2026-08-24 | Servos 1-3 moved back-to-back, `-100` offset each. Voltmeter reads 7.0V — a bit under the 7.4V target from Step 1 (not urgent, STS3215 tolerates 6-8.4V) but consider trimming up if convenient. No sag observed under this load. |

---

## Step 7 — Camera test

```
python camera_test.py
```

Grant Terminal camera permission if prompted (macOS: System Settings → Privacy &
Security → Camera). Check `diagnostics/outputs/camera_*.png` after running.

| Camera | Expected | Result | Date | Notes |
|---|---|---|---|---|
| IMX335 (wrist) | Single frame, in-focus | ✅ PASS | 2026-08-24 | 1920x1080, `outputs/camera_0.png` |
| AR0144 (stereo) | One wide frame, ~2x width vs height (L+R side by side) | ✅ PASS | 2026-08-24 | 2560x720, `outputs/camera_1.png` — visible L/R parallax shift confirmed. Root-caused an earlier "opened but no frame" failure to a bad USB cable (swapped, fixed); also needed a dedicated USB port, not sharing with the servo adapter, to enumerate reliably. |

---

## Sign-off

| All steps pass? | Date | Tested by |
|---|---|---|
| ✅ Yes (7/7) | 2026-08-24 | Vansh |

**Known issues / follow-ups:**
- Buck converter reads 7.0V, not the targeted 7.4V (Step 1/6) — within STS3215 tolerance and stable under load, but worth trimming up when convenient.
- Camera + servo adapter contend for USB bandwidth/enumeration — run cameras and the Waveshare adapter on separate ports/controllers, not a shared hub, if issues resurface.
- Stereo camera required a full cable swap to work reliably — keep the known-good cable paired with it.
