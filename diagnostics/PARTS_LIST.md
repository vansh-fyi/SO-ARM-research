# SOARM Hardware — Parts List

Everything sourced for the physical SO-ARM101 follower arm + roboninecom parallel
gripper build. Kept here as the single source of truth for what's on hand.

STL files and per-vendor docs (BOMs, assembly guides) live in
`progress-documentation/3d_print_stls/` — this file is the flat purchased/printed
inventory.

Status: **order complete** — all items below purchased.

## Servos & electronics

| Part | Qty | Role |
|---|---|---|
| Feetech STS3215 Servo, 7.4V, 1/345 gear (C001) | 6 | Follower arm joints 1-5 + gripper |
| Waveshare Bus Servo Adapter Board | 1 | USB-to-servo-bus control |
| USB-C cable | 1 | Adapter board <-> computer |
| Standard 12V 5A 60W power supply, 5.5mm DC plug | 1 | Wall input to the buck converter below |
| XL4015 5A step-down adjustable converter w/ LED voltmeter + heatsink | 1 | Bucks 12V down to 7.4V (matches the C001 servos' 7.4V rating) to feed the servo bus. Heatsink added — needed to sustain the full 5A rating; bare module is only ~4A continuous. Even at 5A, 2+ servos stalling simultaneously (2.7A stall current each) can still approach/exceed the limit — watch for brownouts during multi-joint moves or calibration. |

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

Full per-step breakdown in `progress-documentation/3d_print_stls/arm/docs/screws-checklist.md`.

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
  (mirrored in `progress-documentation/3d_print_stls/arm/`)
- roboninecom parallel gripper: 16 STL parts + STEP CAD,
  `progress-documentation/3d_print_stls/gripper_upstream_full/`

## Not yet resolved

- **Overhead camera mount** for the Waveshare 32695 stereo module — no purpose-built
  or generic mount exists (PCB screw-hole pattern isn't publicly documented; only the
  66x30x17.72mm body dimensions are known). Plan: generic camera-arm/tripod base
  (1/4-20 thread) + a custom-printed friction-fit cradle sized to the body. Mount
  style (desk clamp vs. tripod vs. fixed frame) still to be decided.
