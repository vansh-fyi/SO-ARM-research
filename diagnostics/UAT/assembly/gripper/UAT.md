# Gripper Assembly UAT — roboninecom Parallel Gripper (SO-ARM101)

Step-by-step build + verification checklist for the follower gripper. Based on the
vendor's own assembly guide (mirrored at
[`progress-documentation/hardware/gripper_upstream_full/docs/assembly-guide.md`](../../../../progress-documentation/hardware/gripper_upstream_full/docs/assembly-guide.md)),
adapted so each step has a concrete pass/fail check instead of just an instruction.

**Prerequisite:** [`UAT/components/UAT.md`](../../components/UAT.md) fully passed
first (servo IDs assigned, power rail confirmed, servo #6/gripper move-tested
standalone). Don't start assembly on an unverified servo.

**Gripper spec (for reference):** 84mm full stroke, 120N max gripping force, 14mm/s
max speed, 0.5mm repeatability, 128x109x130.5mm assembled. Mount flange Ø20mm,
center bore Ø7mm, 4x Ø3.2mm mounting holes.

**Tools:** Phillips PH1 screwdriver, hex keys M2 (H1.5) and M4 (H2.5).

**Which servo:** use the one you assigned **ID 6** in the components UAT — that's
the gripper joint by this project's convention (arm joints 1-5 + gripper 6).

All commands below assume you're in `diagnostics/` with the env active (`cd
diagnostics`, then `direnv allow .` or `source .venv/bin/activate`).

---

## Step 1 — Install gear on servo

**Parts:** 1x STS3215 servo (ID 6), 1x Gear (RB9.01.062.040), 1x servo disk (from
servo kit), 1x M3x6 servo mounting screw (from servo kit), 4x DIN 913 M3x4 set
screws.

1. Place the gear on the servo disk. Tighten the 4x M3x4 set screws from disk toward
   gear — no gap should remain between disk and gear.
2. Mount the gear assembly onto the servo output shaft.
3. Secure with the M3x6 screw from the servo kit.

| Result | Date | Notes |
|---|---|---|
| ✅ PASS | 2026-08-24 | Gear + disk installed on ID-6 servo |

---

## Step 2 — Servo torque limit (macOS-native, not the Windows FD.exe from the vendor guide)

The vendor guide uses a Windows-only tool (`FD.exe`) to cap max torque at 500 (50%)
before the gripper can pinch anything. Use our own script instead — same effect,
works on macOS:

```
python servo_set_torque_limit.py /dev/tty.usbserial-XXXX 6 --limit 500
```

Expect `ID 6: torque limit set to 500 (confirmed)`.

**Why this matters:** at full torque (30 kg·cm stall) the STS3215 can crush a 3D
printed clamp or pinch a finger before you've verified the mechanism moves freely.
Cap it now, raise it later once you trust the assembly.

| Result | Date | Notes |
|---|---|---|
| ✅ PASS | 2026-08-25 | `ID 6: torque limit set to 500 (confirmed)` |

---

## Step 3 — Clamps

**Parts:** 2x Clamps (RB9.01.062.021), 2x Gear racks (RB9.01.062.031), 2x Nails
(RB9.01.062.100).

1. Insert gear racks into clamps from the side.
2. Insert nails from the back to retain them.

| Result | Date | Notes |
|---|---|---|
| ✅ PASS | 2026-08-25 | Both clamps racked and nailed |

---

## Step 4 — Rods

**Parts:** 2x Round rods, D6x125mm.

1. Insert the rods into both clamps. If your rods are longer than 125mm, cut to
   length first.

| Result | Date | Notes |
|---|---|---|
| ✅ PASS | 2026-08-25 | Both rods inserted through both clamps |

---

## Step 5 — Bearings on main frame

**Parts:** 2x MF106ZZ bearings (6x10x3mm), 2x DIN 7991 M4x8 screws.

1. Insert the 2 bearings into the main frame (RB9.01.062.010) — flange faces up.
2. Secure each with an M4x8 screw.

| Result | Date | Notes |
|---|---|---|
| ✅ PASS | 2026-08-25 | Both bearings seated, flange up, secured with M4x8 |

---

## Step 6 — Connect rods to frame

1. Snap the rod+clamp assembly (Steps 3-4) into the frame (Step 5).
2. Spread both clamps to their extreme left/right positions and confirm they slide
   freely on the rods — no binding, no grinding.

| Result | Free sliding? | Date | Notes |
|---|---|---|---|
| ✅ PASS | Yes | 2026-08-25 | Slides freely full range, no binding at either extreme |

---

## Step 7 — Mount servo on main frame

**Parts:** 3x self-tapping screws (from servo kit).

1. Position the servo (with gear from Step 1) on the main frame, align mounting
   holes.
2. Secure with 3x self-tapping screws.
3. **Do not power the servo yet** — Step 8 covers the first powered test.

| Result | Date | Notes |
|---|---|---|
| ✅ PASS | 2026-08-25 | Mounted flush with 3x self-tapping screws, gear teeth visually meshed with both racks. Gear couldn't be turned by hand — expected, STS3215 has a 1/345 internal reduction; not a valid manual check, verified under power in Step 8 instead. |

---

## Step 8 — First powered test (torque-limited)

With the gripper's servo bus cable connected back to the Waveshare adapter (torque
limit from Step 2 still in effect):

```
python servo_move_test.py /dev/tty.usbserial-XXXX 6 --offset 300
python servo_move_test.py /dev/tty.usbserial-XXXX 6 --offset -300
```

Watch the jaws open and close through a meaningful portion of the 84mm stroke.
Confirm:
- Both jaws move symmetrically (parallel, not skewed).
- No grinding/binding at either extreme.
- Gear rack doesn't skip teeth under the (torque-limited) load.

| Result | Symmetric? | Date | Notes |
|---|---|---|---|
| ✅ PASS | Yes | 2026-08-25 | `servo_move_test.py` ±100 ticks confirmed clean movement both directions, no errors. Jaws move symmetrically and close at true geometric center (initial impression of a slight offset was rechecked and was not actually present). |

---

## Step 9 — Gripper holder attachment (to the arm wrist)

**Parts:** 1x Camera holder (RB9.01.060.074, if using a camera) or plain Holder
(RB9.01.060.080), 1x SO-ARM101 wrist joint horn, 4x M3x6 screws (servo kit), 4x
self-tapping screws (servo kit).

1. Attach the Camera holder (or plain Holder) to the wrist joint horn with 4x M3x6
   screws.
2. Attach the gripper's main frame to the holder with 4x self-tapping screws.
3. Route the gripper's servo cable into the arm's daisy chain at the wrist end (the
   gripper servo is whatever bus ID you assigned it in `UAT/components/UAT.md` — ID
   6 by this project's convention). The vendor's own doc calls this "servo #5" using
   their own numbering scheme — go by ID, not by that number, to avoid confusion.

| Result | Date | Notes |
|---|---|---|
| ✅ PASS | 2026-08-25 | Done during main arm assembly, see `UAT/assembly/main/UAT.md` Step 6: camera holder attached to wrist joint horn (4x M3x6mm), gripper main frame attached to holder (4x self-tapping), servo cable routed into the arm's daisy chain (ID 6). |

---

## Step 10 — Camera attachment (optional — IMX335)

**Parts:** 1x IMX335 USB camera (28x28mm mount holes), 1x Camera Spacer
(RB9.01.060.090), 4x DIN 912 M2x8 screws, 4x DIN 934 M2 hex nuts.

1. Insert 4x M2 nuts from the back of the camera holder.
2. Position the camera spacer + camera from the front, secure with 4x M2x8 screws.
3. Re-run the camera test to confirm it still captures once mounted:

```
python camera_test.py
```

| Result | Date | Notes |
|---|---|---|
| ✅ PASS | 2026-08-25 | IMX335 mounted with spacer + 4x M2x8 screws + 4x M2 nuts. Camera confirmed capturing via `camera_test.py` (1920x1080). Hit a transient OpenCV/AVFoundation crash on first two attempts (`raised unknown C++ exception!`) — resolved by unplugging/replugging the USB cable, forcing clean re-enumeration; hardware itself was never at fault (confirmed alive via QuickTime throughout). |

---

## Step 11 — Full-stroke functional test (torque limit removed)

Only after Steps 1-10 all pass. Raise torque back toward normal operating range and
retest full range of motion + grip on a real object, using the calibrated bounds
above (open = 3, closed = 3594):

```
python servo_set_torque_limit.py /dev/tty.usbserial-XXXX 6 --limit 1000
python servo_drive_to_stall.py /dev/tty.usbserial-XXXX 6 --offset -3600   # drive to fully open
python servo_drive_to_stall.py /dev/tty.usbserial-XXXX 6 --offset 3600   # drive to fully closed
```

Then manually place a small object between the jaws (something under 84mm, e.g. a
pen or small box) and command a close — confirm it grips without slipping and
without the frame flexing/cracking.

| Result | Object used | Gripped without slipping? | Date | Notes |
|---|---|---|---|---|
| ✅ PASS | Pomodoro timer (round dial) | Yes | 2026-08-25 | Full torque (1000) restored, full stroke exercised (open=3 to closed stall=3594). First close attempt via `servo_drive_to_stall.py` gripped but then visibly relaxed - traced to that script auto-releasing torque after detecting stall (script design, not a servo safety feature). Redone with `servo_move_test.py` (holds torque by default): stopped at 1280 (consistent with the earlier 1270 contact point), object held firm under tug-test, no visible frame flexing/cracking. |

---

## Sign-off

| All steps pass? | Date | Tested by |
|---|---|---|
| ✅ Yes (11/11) | 2026-08-25 | Vansh |

**Gripper raw-tick calibration (servo ID 6) — RECALIBRATED 2026-08-25 (current,
final):**
- **Fully open: raw 3** (confirmed via genuine stall detection, `servo_drive_to_stall.py`)
- **Fully closed: raw 3594** (confirmed via genuine stall detection)
- Full stroke ≈ 3591 raw ticks
- Direction convention: positive offset = closing, negative = opening (unchanged)

**Recalibration note (2026-08-25):** original calibration (open ~38, closed ~3533)
invalidated after a screw-related mishap during handling (since resolved) shifted the
gear's mesh phase on the shaft. Re-ran the open/closed discovery from scratch.
Hit two rounds of noisy intermediate readings during the process (an unexplained
242-tick position jump, then a one-off corrupted read reporting 1519 when the servo
was actually at 101) - both were communication glitches, not real position changes,
resolved by re-verifying with `servo_drive_to_stall.py`'s genuine polling-based stall
detection rather than trusting single reads or visual judgment alone. Final numbers
above are stall-confirmed and closely match the ~100/~3560 manually-observed values
from earlier in the process (within normal mechanical variance).

**Previous calibration (superseded, kept for history):**
- Fully open: raw ~38-39
- Fully closed: raw ~3533 (confirmed genuine stall via `servo_drive_to_stall.py`,
  torque capped at 500, offset 3500 with comfortable margin - not a timeout)
- Full stroke ≈ 3495 raw ticks, entirely within the encoder's 0-4095 range

**Calibration history (for context, not re-actionable):** the servo's factory zero
point originally landed *inside* the gripper's working range (near raw 0), making the
true full-open position uncommandable via plain position offsets (STS3215 is a
single-turn absolute encoder with no wraparound in position mode - see project
discussion). Fixed via a purely mechanical re-mesh: drove to the confirmed close
stall, released torque, un-meshed the gear from the rack (servo-to-frame screws only,
gear-to-shaft coupling untouched), manually slid the clamps to true fully-open, then
re-meshed the gear there. This shifted which raw values map to which physical
positions without touching any servo EEPROM. (A live EEPROM zero-offset
recalibration was attempted first and abandoned after it caused a transient comm
failure - not damaging, but not worth repeating when the mechanical fix is simpler
and has no ambiguity risk.)
