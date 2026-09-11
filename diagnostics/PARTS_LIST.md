# SOARM Hardware — Parts List

Everything sourced for the physical SO-ARM101 follower arm + roboninecom parallel
gripper build. Kept here as the single source of truth for what's on hand.

STL files and per-vendor docs (BOMs, assembly guides) live in
`progress-documentation/hardware/` — this file is the flat purchased/printed
inventory.

Status: **order complete** for everything below. The 7.4V→12V/30kg·cm servo swap
described below is **done** (confirmed via `diagnostics/UAT/function/UAT.md`,
2026-09-09 — follower recalibrated on the new servos, all 7 function UAT steps
passing), and the leader arm (below, reusing the removed 7.4V servos) is built,
calibrated, and confirmed working via real leader-follower teleop.

## Servo swap — 7.4V → 12V/30kg·cm (complete)

**Decision (2026-09-02, finalized):** replacing all 6 Feetech STS3215 servos (7.4V,
C001, 19.5kg·cm stall per official datasheet) with **STS3215-C018 (12V, 30kg·cm,
dual-shaft)**. Not yet ordered/received — tracked here so it isn't lost before the
parts arrive.

**Compatibility confirmed against official Feetech datasheets for both parts**
(current C001 and replacement C018) — identical in every dimension that matters:
body size 45.2×24.7×35mm, 1/345 gear ratio, 25T/⌀5.9mm horn spline, 5264-3P
connector (GND/Vcc/Signal), M3×6 mounting screw, same serial protocol (0-4095
range, 1Mbps default, PID control). "Dual-shaft" just means the C018 has output
access on both sides of the case — same primary mounting interface, no fit issue
with existing printed mounts. Only real difference: input voltage range (C001:
4-7.4V rated, 30kg·cm-variant: wider, works down to 7.4V too at reduced torque)
and the resulting torque ceiling.

**Why:** shoulder_lift (servo ID 2) overheated repeatedly during the function UAT
(`diagnostics/UAT/function/UAT.md`) — measured peaks up to 70°C (the servo's own
documented overheat cutoff, confirmed via datasheet: torque output shuts off above
70°C), tripped overload protection more than once, and ultimately failed under
light load after a full cooldown cycle. Tried and exhausted within the 7.4V
constraint first: current/torque de-rating (`diagnostics/servo_set_protection.py`),
softened P/D holding gains on loaded joints, read/write retry+backoff for the bus's
known "no status packet" flakiness (a documented, unresolved upstream LeRobot/
Feetech issue, not specific to this build), and a genuine mechanical bind fix
(loosening a joint) that measurably improved movement range (32/50 ticks vs
12-15/50 before) but still generated a 19°C temperature rise in two small moves —
confirming the torque ceiling itself, not just friction, was the limiting factor.

**Correction, logged for the record:** mid-session, voltage was trimmed up to 7.5V
for "extra headroom" based on a remembered general STS3215 tolerance figure, not
the actual datasheet. The real datasheet-specified ceiling for the C001 (7.4V)
servos is exactly **4-7.4V input, with the servo's own firmware overvoltage
protection triggering above 7.4V**. Running at 7.5V was therefore past spec and
likely triggered spurious overvoltage protection on top of the genuine thermal/
torque issues. If testing the current servos again before the swap, keep the buck
converter at or below 7.4V (not above) — this does not change the case for the
12V swap, which is about torque ceiling, not this voltage mistake.

**What changes:**
- All 6 servos — swapping only some is not viable, since all 6 share one bus
  voltage; running 12V-rated servos at 7.4V gives no torque benefit over the
  current servos, and running 7.4V-rated servos at 12V would over-volt and likely
  damage them.
- Power chain **simplifies**: 12V-rated servos can run closer to directly off the
  existing 12V wall supply — the XL4015 buck converter (currently stepping 12V down
  to 7.4V) likely won't be needed in the chain at all. Bonus: same power draw at
  higher voltage means lower current, which should also ease the bus brownout/
  "no status packet" issue this build has repeatedly hit.
- Full recalibration required after the swap — both LeRobot's `lerobot-calibrate`
  and the gripper's raw-tick open/close calibration
  (`diagnostics/UAT/assembly/gripper/UAT.md`) — positions won't carry over.
- Free win while it's apart anyway: reseat every daisy-chain connector. This
  session independently hit two other connector/cable issues (corrupted
  calibration reads, the AR0144 camera needing a cable swap) — worth doing while
  the arm is already disassembled for the servo swap.
- Once repowered, the servo protection settings and holding-gain tweaks above
  should be re-verified/re-applied for the new servos — the values used tonight
  were tuned for the 7.4V servos' actual behavior, not assumed to carry over.

**Current 7.4V servo status (as of 2026-09-02, before removal):** all 6 confirmed
responsive, set to max torque limit (1000) with torque holding enabled at their
resting positions (not commanded to move). Servo 2 (shoulder_lift) has the
overheat/failure history above and should be re-tested for baseline health (not
assumed fine) before reuse in the leader arm below.

## Leader Arm (built, reusing removed 7.4V servos)

**Status (2026-09-09):** built, calibrated (`soarm_leader_01`), and confirmed
working via real leader-follower teleop in `diagnostics/UAT/function/UAT.md`
Step 3b. One assembly issue found and fixed along the way: servo IDs 2 and 3
were physically swapped (elbow_flex/shoulder_lift) — see that UAT's "Known
issues" for the diagnosis and fix.

**Plan (2026-09-02):** once the 6 C001 servos come out of the follower for the 12V
swap, reuse them to build a leader arm rather than discarding them. Leader arms are
lighter-duty by design — a human hand does the actual weight-bearing/positioning,
the servos mainly report joint angles and provide light resistance, not sustained
gravity-holding — a good match for what these servos can still do reliably even
given the follower-role failures above.

**Still needed:**
- Leader-specific printed parts (not previously printed — this build was
  follower-only): `Handle_SO101.stl`, `Trigger_SO101.stl`, `Wrist_Roll_SO101.stl`,
  from the same vendored STL source as the follower arm
  (`progress-documentation/hardware/arm/`).
- No additional servos to buy — 6 come out of the follower (5 arm joints + gripper),
  exactly enough for a 6-servo leader arm.

**Known trade-off:** the vendor's leader-arm BOM specifies a mixed gear ratio across
joints (lighter/faster gearing than the follower's uniform 1/345) for a leader arm
optimized to be easy to hand-move. These reused servos are all 1/345 (higher-torque,
slower variant) — will function, but will feel stiffer to hand-move than the
"ideal" spec calls for. Not a blocker, just a known limitation of reusing parts.

## Servos & electronics

| Part | Qty | Role |
|---|---|---|
| Feetech STS3215 Servo, 7.4V, 1/345 gear (C001), 19.5kg·cm stall (official datasheet — corrects an earlier 16.5kg·cm figure in this file) | 6 | Follower arm joints 1-5 + gripper. **Pending replacement — see above.** Rated input 4-7.4V; firmware overvoltage protection above 7.4V. |
| Waveshare Bus Servo Adapter Board | 1 | USB-to-servo-bus control. Has two physical bus-output slots, likely wired in parallel on the same bus (unconfirmed) — worth verifying once reassembling for the servo swap, since splitting the 6 servos across both slots would shorten the connector path to each servo. |
| USB-C cable | 1 | Adapter board <-> computer |
| Standard 12V 5A 60W power supply, 5.5mm DC plug | 1 | Wall input. Likely feeds the new 12V servos closer to directly post-swap — see Pending Hardware Change above. |
| XL4015 5A step-down adjustable converter w/ LED voltmeter + heatsink | 1 | Bucks 12V down to ~7.4V for the *current* C001 servos. Heatsink added — needed to sustain the full 5A rating; bare module is only ~4A continuous. Even at 5A, 2+ servos stalling simultaneously (2.7A stall current each) can still approach/exceed the limit — watch for brownouts during multi-joint moves or calibration. **Keep at or below 7.4V, not above** — was briefly trimmed to 7.5V (2026-09-02) which exceeds the servos' actual rated/firmware-protected ceiling (see Pending Hardware Change above); corrected. Likely removed from the chain entirely once 12V servos arrive. |

## Gripper-specific (roboninecom parallel gripper)

| Part | Qty | Role |
|---|---|---|
| MF106ZZ bearing, 6x10x3mm | 2 | Main frame bearing seats |
| Round rod, D6x125mm (steel or aluminum) | 2 | Linear guide rods for the two clamps |
| DIN 7991 M4x8 countersunk screw | 2 | Secures bearings to main frame |
| DIN 913 M3x4 set screw | 4 | Locks gear to servo disk |
| DIN 912 M2x8 screw | 2 | Camera mount (optional, in use) |
| DIN 934 M2 hex nut | 4 | Camera mount (optional, in use) |

## Fasteners — SO-ARM101 arm assembly

Full per-step breakdown in `progress-documentation/hardware/arm/docs/screws-checklist.md`.

| Part | Qty | Role |
|---|---|---|
| M3x6mm screw | 50 | Motor horns, shoulder/upper-arm/forearm/wrist joints, gripper claw |
| M2x6mm screw | 24 | Motor-to-frame fastening at every joint ("the smallest screws") |

## Cameras

| Part | Qty | Role |
|---|---|---|
| Waveshare IMX335 5MP USB Camera (C) — distortion-free lens | 1 | Wrist/gripper camera, bolts to printed Camera Holder + Camera Spacer |
| Waveshare 32695 AR0144 2MP Stereo USB Camera Module (52mm baseline) | 1 | Overhead/external camera — static, aimed at workspace, stereo depth for Phase 5 spatial-awareness work |

Ruled out: Arducam B0568 IMX335 (MIPI CSI, incompatible with USB-only pipeline);
Waveshare IMX335 (B) (same as (C) but -36% lens distortion vs (C)'s -1.22%).

## Tools

| Part | Role |
|---|---|
| Phillips screwdriver PH1 | Servo/frame screws |
| Hex key set, M2 (H1.5) and M4 (H2.5) | Gripper bearing/frame screws |

## Printed parts (not purchased — 3D printed from vendored STLs)

- SO-ARM101 arm: 13 STL parts, `LIBERO/libero/libero/assets/robots/soarm101/assets/`
  (mirrored in `progress-documentation/hardware/arm/`)
- roboninecom parallel gripper: 16 STL parts + STEP CAD,
  `progress-documentation/hardware/gripper_upstream_full/`

## Not yet resolved

- **Overhead camera mount** for the Waveshare 32695 stereo module — no purpose-built
  or generic mount exists (PCB screw-hole pattern isn't publicly documented; only the
  66x30x17.72mm body dimensions are known). Plan: generic camera-arm/tripod base
  (1/4-20 thread) + a custom-printed friction-fit cradle sized to the body. Mount
  style (desk clamp vs. tripod vs. fixed frame) still to be decided.
