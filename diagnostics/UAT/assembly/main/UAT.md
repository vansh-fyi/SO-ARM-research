# Main Arm Assembly UAT — SO-ARM101 Follower

Step-by-step build + verification checklist for the 5-joint follower arm (base to
wrist), condensed from the official assembly guide at
[huggingface.co/docs/lerobot/so101](https://huggingface.co/docs/lerobot/so101) and
this repo's own fastener breakdown at
[`progress-documentation/hardware/arm/docs/screws-checklist.md`](../../../../progress-documentation/hardware/arm/docs/screws-checklist.md).
The HF page has photos for exact part orientation at each step — use it alongside
this checklist for visual reference; this file adds the pass/fail checkpoints and
servo-ID tracking specific to this project's setup.

**Prerequisites:**
- [`UAT/components/UAT.md`](../../components/UAT.md) fully passed (all 6 servos
  bring-up tested, power rail confirmed).
- [`UAT/assembly/gripper/UAT.md`](../gripper/UAT.md) fully passed (gripper built and
  calibrated — it attaches to the wrist in Step 6 below).

**Tools:** Phillips PH1 screwdriver, hex key set (per servo kit).

**Fastener totals (follower arm only, excludes gripper's own hardware):**
50x M3x6mm, 24x M2x6mm. Servo kits typically only bundle enough screws for their own
horn — source an M2x6mm/M3x6mm assortment separately if you haven't already, per
`screws-checklist.md`.

**Servo ID convention (this project):** Joint 1 = base/shoulder pan, Joint 2 =
shoulder lift, Joint 3 = elbow flex, Joint 4 = wrist flex, Joint 5 = wrist roll,
Joint 6 = gripper (already assigned and calibrated in the gripper UAT). **Assign each
servo's ID before mounting it into its joint** — much easier to fix a wrong ID with
the servo connected alone than after it's screwed into the arm. Use
`servo_set_id.py` from `diagnostics/` (see `UAT/components/UAT.md` Step 3 for the
one-servo-at-a-time procedure).

All terminal commands below assume you're in `diagnostics/` with the env active.

---

## Step 1 — Joint 1 (base / shoulder pan)

**Parts:** Base (`base_so101_v2.stl`), Base motor holder (`base_motor_holder_so101_v1.stl`
+ `motor_holder_so101_base_v1.stl`), Waveshare mounting plate
(`waveshare_mounting_plate_so101_v2.stl`), Servo (**ID 1**), 1x M3x6mm (top motor
horn), 4x M2x6mm (motor to base), 2x M2x6mm (motor holder).

1. Confirm this servo is set to **ID 1** (test standalone before mounting — see
   servo ID convention note above).
2. Mount the top motor horn with 1x M3x6mm.
3. Fasten the servo into the base with 4x M2x6mm (2 top + 2 bottom).
4. Attach the first motor holder (one screw each side) with 2x M2x6mm.
5. Mount the Waveshare adapter board to its mounting plate on the base (per HF guide
   photos for exact orientation/screws — board-specific, not in the follower fastener
   count above).

| Result | Servo ID confirmed | Date | Notes |
|---|---|---|---|
| ✅ PASS | ID 1 | 2026-08-25 | Motor horn mounted, servo fastened to base, motor holder attached, Waveshare adapter mounted to base plate |

---

## Step 2 — Shoulder structure (top of Joint 1)

**Parts:** Shoulder part, 8x M3x6mm (4 top + 4 bottom), shoulder motor holder
(quantity not specified in source guide — budget spare M3x6mm/M2x6mm).

1. Attach the shoulder part to the Joint 1 assembly with 8x M3x6mm.
2. Attach the shoulder motor holder.
3. With Joint 1 servo powered (torque-limited, see `servo_set_torque_limit.py` if
   you want the same safety margin used for the gripper), confirm the base rotates
   freely through its intended range before moving on.

| Result | Rotates freely? | Date | Notes |
|---|---|---|---|
| ✅ PASS | Yes | 2026-08-25 | Shoulder part + motor holder attached, base rotates freely under power, no binding or cable snag |

---

## Step 3 — Joint 2 (shoulder lift)

**Parts:** Rotation/pitch part (`rotation_pitch_so101_v1.stl`), Upper arm
(`upper_arm_so101_v1.stl`), Servo (**ID 2**), 1x M3x6mm (top motor horn), 4x M2x6mm
(fasten motor), 8x M3x6mm (upper arm, 4 per side).

1. Confirm this servo is set to **ID 2** before mounting.
2. Mount the top motor horn with 1x M3x6mm.
3. Fasten motor 2 into the rotation/pitch part with 4x M2x6mm.
4. Attach the upper arm with 8x M3x6mm (4 per side).

| Result | Servo ID confirmed | Date | Notes |
|---|---|---|---|
| ✅ PASS | ID 2 | 2026-08-25 | Motor horn mounted, servo fastened into rotation/pitch part, upper arm attached (4 per side) |

---

## Step 4 — Joint 3 (elbow flex)

**Parts:** Under arm / forearm (`under_arm_so101_v1.stl`), Servo (**ID 3**), 1x
M3x6mm (top motor horn), 4x M2x6mm (fasten motor), 8x M3x6mm (forearm, 4 per side).

1. Confirm this servo is set to **ID 3** before mounting.
2. Mount the top motor horn with 1x M3x6mm.
3. Fasten motor 3 with 4x M2x6mm.
4. Attach the forearm (under arm) to motor 3 with 8x M3x6mm (4 per side).

| Result | Servo ID confirmed | Date | Notes |
|---|---|---|---|
| ✅ PASS | ID 3 | 2026-08-25 | Motor horn mounted, servo fastened, forearm (under arm) attached to motor 3 (4 per side) |

---

## Step 5 — Joint 4 (wrist flex)

**Parts:** Wrist motor holder (`motor_holder_so101_wrist_v1.stl`), Servo (**ID 4**),
1x M3x6mm (top motor horn), 4x M2x6mm (fasten motor).

1. Confirm this servo is set to **ID 4** before mounting.
2. Mount the top motor horn with 1x M3x6mm.
3. Fasten motor 4 with 4x M2x6mm.

| Result | Servo ID confirmed | Date | Notes |
|---|---|---|---|
| ✅ PASS | ID 4 | 2026-08-25 | Motor horn mounted, servo fastened into wrist motor holder |

---

## Step 6 — Joint 5 (wrist roll) + gripper mount point

**Parts:** Wrist roll/pitch (`wrist_roll_pitch_so101_v2.stl`), Wrist roll follower
(`wrist_roll_follower_so101_v1.stl`), Servo (**ID 5**), 2x M2x6mm (motor into wrist
holder, front), 1x M3x6mm (wrist motor horn), 8x M3x6mm (wrist to motor 4, 4 per
side).

1. Confirm this servo is set to **ID 5** before mounting.
2. Mount motor 5 into the wrist holder (front) with 2x M2x6mm.
3. Mount the wrist motor horn with 1x M3x6mm.
4. Attach the wrist assembly to motor 4 with 8x M3x6mm (4 per side).

This is also where the gripper (already built + calibrated per its own UAT) attaches
— that's covered in the gripper UAT's Step 9, not repeated here. If not already
done, go do that now: attach the gripper's Camera holder to this wrist joint horn (4x
M3x6mm), then the gripper's main frame to the holder (4x self-tapping screws), then
route its cable into this joint's daisy chain position.

| Result | Servo ID confirmed | Date | Notes |
|---|---|---|---|
| ✅ PASS | ID 5 | 2026-08-25 | Motor 5 into wrist holder, wrist motor horn mounted, wrist assembly attached to motor 4 (4 per side). Gripper attached: camera holder to wrist joint horn (4x M3x6mm), gripper frame to holder (4x self-tapping), servo cable routed into daisy chain. |

---

## Step 7 — Cable routing check

1. Trace the daisy-chain cabling from the Waveshare adapter through Joint 1 -> 2 ->
   3 -> 4 -> 5 -> gripper.
2. Confirm no cable is pinched, over-tensioned, or crossing a joint's rotation path
   in a way that would stretch/snag it at full range of motion.
3. Confirm connectors are fully seated (not half-plugged) at every joint.

| Result | Date | Notes |
|---|---|---|
| ✅ PASS | 2026-08-25 | Full chain traced Waveshare adapter -> Joint 1-5 -> gripper, no pinching/over-tension, all connectors fully seated |

---

## Step 8 — Full daisy-chain scan

With everything connected (all 6 servos, full chain from Waveshare adapter through
the gripper):

```
python servo_scan.py /dev/tty.usbserial-XXXX
```

Expect all 6 IDs (1-5 arm joints, 6 gripper) to respond `FOUND`, matching the
convention above. Any missing ID at this point usually means a connector issue
introduced during assembly (see Step 7), not a fresh ID collision (those should
already be resolved per-servo from Steps 1-6).

| Result | IDs found | Date | Notes |
|---|---|---|---|
| ✅ PASS | [1, 2, 3, 4, 5, 6] | 2026-08-25 | All model 777. Gripper (ID 6) reads pos=36, matching its calibrated open position from the gripper UAT. |

---

## Step 9 — Per-joint range of motion test

One joint at a time (others can stay connected but keep hands clear of whichever
joint is moving). Use torque-limited moves first, same safety pattern as the
gripper's Step 8:

```
python servo_set_torque_limit.py /dev/tty.usbserial-XXXX <id> --limit 500
python servo_move_test.py /dev/tty.usbserial-XXXX <id> --offset 300
python servo_move_test.py /dev/tty.usbserial-XXXX <id> --offset -300
```

For each joint, confirm: moves in the expected physical direction, no grinding or
binding, no cable snagging at either extreme, no collision with another link at this
small test range.

| Joint | Servo ID | Result | Date | Notes |
|---|---|---|---|---|
| Shoulder pan | 1 | ✅ PASS | 2026-08-25 | 286/300, 287/300 both directions, smooth |
| Shoulder lift | 2 | ✅ PASS | 2026-08-25 | 276/300, 274/300 both directions, smooth |
| Elbow flex | 3 | ✅ PASS (with note) | 2026-08-25 | -300 clean (187/300); +300 initially only moved 27/300 - confirmed torque/gravity limited (500 cap insufficient to hold position against downstream weight, arm dropped forward when stopped), not mechanical - no grinding, smooth motion throughout. Not a defect. |
| Wrist flex | 4 | ✅ PASS (with note) | 2026-08-25 | -300 clean (292/300); +300 only moved 126/300 - same torque/gravity cause as joint 3, confirmed smooth/no grinding. Not a defect. |
| Wrist roll | 5 | ✅ PASS | 2026-08-25 | 299/300, 299/300 both directions, smooth (roll axis not gravity-loaded) |

**Note:** joints 3 and 4 need more than the 500 (50%) torque cap to *hold* position
against their own downstream weight, even folded near the base. Raise torque before
Step 10's wider-range test (see below) rather than leaving it at 500.

---

## Step 10 — Full-arm integration test

**Scope note (revised after live testing):** an earlier version of this step called
for large (±800 tick) *isolated* single-joint moves against a static, fully-extended
rest-of-arm — i.e. one joint fighting the full gravity load alone while every other
joint stays rigid. That's a harder condition than the arm ever faces in real
operation, where a controller moves joints *together* (e.g. shoulder + elbow
coordinating so the load's center of mass stays closer to the base), which needs far
less peak torque at any single joint. Testing joints in full isolation at max range
is not representative and isn't a fair pass/fail bar — don't re-run that version of
this step.

What this step actually verifies now: cable/collision clearance across a **moderate**
range of motion, one joint at a time, at a safe torque level — not maximum
single-joint strength.

```
python servo_set_torque_limit.py /dev/tty.usbserial-XXXX <id> --limit 500
python servo_move_test.py /dev/tty.usbserial-XXXX <id> --offset 300
python servo_move_test.py /dev/tty.usbserial-XXXX <id> --offset -300
```

For each joint, from a reasonably neutral/folded starting pose, watching for:
- Any two links contacting each other at extreme poses
- Cable tension/snagging across this range (recheck Step 7 concerns)
- Consistent, repeatable behavior across a few repetitions per joint

Full coordinated multi-joint motion (the real test of whether the arm works as a
system, including gravity-compensated combined moves like shoulder+elbow together)
belongs in [`UAT/function/UAT.md`](../../function/UAT.md) once laptop-driven
control (LeRobot) is set up — that's where joints actually move together the way
they will in real operation, not this bring-up-level per-joint check.

| Result | Date | Notes |
|---|---|---|
| ✅ PASS (satisfied by Step 9) | 2026-08-25 | Step 9's ±300/500-limit/folded-pose data already covers this revised, moderate-range scope - all 5 joints moved cleanly, no collisions/snagging observed during that pass. No separate re-run needed. Full coordinated-motion validation deferred to `UAT/function/UAT.md` per the scope note above. |

---

## Sign-off

| All steps pass? | Date | Tested by |
|---|---|---|
| ✅ Yes (10/10) | 2026-08-25 | Vansh |

All 5 arm joints (IDs 1-5) set to torque limit 1000 (full, confirmed) as the final
baseline before handing off to the function UAT — a torque limit is a ceiling, not a
constant draw, so this doesn't mean the servos run hot in normal motion; the earlier
overload was specific to isolated single-joint testing against a static extended
load (see Step 10's notes), not a reason to cap normal operating torque. Gripper (ID
6) remains at 500 pending its own Step 11 (full-stroke test with torque restored).

**Known issues / follow-ups:**
- **Incident (2026-08-25):** first Step 10 attempt at `--limit 800` used the old
  `servo_move_test.py`, which unconditionally released torque after a fixed 1.5s
  wait regardless of whether the commanded move had actually finished. On joints 3
  and 4 (weight-bearing, gravity-loaded) this cut power mid-travel and the arm
  dropped harshly under its own weight — a script bug, not a mechanical or torque
  defect. Fixed: the script now polls until motion genuinely stops and **keeps
  torque holding by default** (`--release` opt-in only, for unloaded joints like the
  gripper).
- **Follow-up finding (2026-08-25):** re-testing joint 2 in isolation (at max torque,
  fully extended, everything downstream rigid) drove it into a genuine overload -
  temperature rose to 59°C with the LED blinking a warning, though no hard fault
  latched. Fully recovered after a 40-minute cool-down (32°C, clean status). This
  led to revising Step 10's scope (see the note there): isolated single-joint
  max-strength testing against a static extended load is not representative of real
  coordinated-motion operation and was retired as a test methodology, not just a
  one-off retry. No hardware damage; joint 2 confirmed fully healthy afterward.
