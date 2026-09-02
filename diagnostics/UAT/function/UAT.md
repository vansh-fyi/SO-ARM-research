# Functional UAT — Laptop Control + Camera Recording

Verifies the assembled SO-ARM101 follower can be driven from the laptop (not just
individually pinged/moved like the bring-up UATs) and that the mounted cameras can
capture stills and video while the arm operates. This is the layer needed before any
teleop/dataset-collection work for VLA fine-tuning.

**Prerequisites:**
- [`UAT/components/UAT.md`](../components/UAT.md) passed (electronics bring-up).
- [`UAT/assembly/gripper/UAT.md`](../assembly/gripper/UAT.md) passed (gripper built
  + calibrated).
- [`UAT/assembly/main/UAT.md`](../assembly/main/UAT.md) passed (full arm assembled,
  all 6 servo IDs confirmed, per-joint ROM tested).

**Scope note:** this project only has a **follower** arm (no leader/handle for
physical teleop) — "control from laptop" here means direct programmatic/keyboard
joint control, not leader-follower teleoperation.

---

## Step 0 — Control software environment

The `diagnostics/.venv` env is intentionally lightweight (pyserial, feetech SDK,
opencv) for hardware bring-up. Real robot control via
[LeRobot](https://huggingface.co/docs/lerobot/so101) pulls in much heavier
dependencies (torch, transformers, etc.) — set up a **separate** env for this rather
than bloating diagnostics' env.

**Use Python 3.12, not the system default `python3`.** This machine's default
`python3` resolves to 3.14, which crashes LeRobot's config parser (`draccus`) with
`TypeError: float | None is not callable` on any `--help`/CLI invocation — a real
Python-3.14 incompatibility in draccus, not a typo or bad install. Check
`python3.12 --version` works (`brew install python@3.12` if not) before proceeding.

```
mkdir -p ~/Desktop/Work/Repositories/SoARM-Research/control
cd ~/Desktop/Work/Repositories/SoARM-Research/control
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install "lerobot[feetech]" opencv-python
```

Confirm the CLI installed and check its current flags (LeRobot's CLI surface changes
between versions — treat `--help` as the source of truth over any specific command
below):

```
lerobot-teleoperate --help
lerobot-record --help
```

| Result | LeRobot version | Date | Notes |
|---|---|---|---|
| ✅ PASS | 0.6.1 | 2026-08-25 | Installed under Python 3.12.12 (not system default 3.14, which crashes `draccus`'s CLI help generation - see note above). `so101_follower` robot type and `keyboard` teleop type both confirmed available; robot config exposes `--robot.port`, `--robot.id`, `--robot.cameras` as expected. `lerobot-calibrate` also confirmed present. |

---

## Step 1 — Robot config: port + servo IDs

LeRobot needs to know the serial port and which servo ID maps to which joint. Reuse
what's already confirmed in the assembly UATs (Joint 1-5 = IDs 1-5, gripper = ID 6).

1. Find the Waveshare adapter's port (same one used throughout diagnostics):
   ```
   python -c "from serial.tools import list_ports; [print(p.device, p.description) for p in list_ports.comports()]"
   ```
2. Create/confirm a robot config identifying this arm as a `so101_follower` (LeRobot
   robot type) with that port and a `robot.id` you choose to name this specific arm
   (e.g. `soarm_follower_01`) — exact config mechanism (CLI flags vs. a config file)
   depends on your installed LeRobot version; check `lerobot-teleoperate --help`.

| Result | Port | Robot ID chosen | Date | Notes |
|---|---|---|---|---|
| ✅ PASS | `/dev/cu.usbmodem5B8E1132011` | `soarm_follower_01` | 2026-08-25 | Same port confirmed throughout all prior UATs. Servo IDs already match convention (1-5 arm, 6 gripper) from assembly UATs - no remapping needed. |

---

## Step 2 — LeRobot's own calibration

LeRobot runs its own calibration routine (separate from the raw tick calibration we
did for the gripper in its assembly UAT) to record joint zero-points and ranges in
its own config format. Run whatever calibration entry point your installed version
exposes (commonly surfaced via `lerobot-calibrate` or a `--calibrate` flag on
`lerobot-teleoperate` — confirm via `--help` from Step 0) and follow its prompts
(usually: move each joint through its full range while it records min/max, then a
reference "home" pose).

**Sanity check before trusting this step:** the gripper's raw open/close ticks
should roughly match what we already measured (open ~38, closed ~3533, see
[`UAT/assembly/gripper/UAT.md`](../assembly/gripper/UAT.md)) — if LeRobot's
calibration reports wildly different numbers for the gripper joint, stop and
investigate before proceeding (mismatched joint mapping is a common
misconfiguration here).

| Result | Gripper range matches prior calibration? | Date | Notes |
|---|---|---|---|
| ✅ PASS (with note) | Open: yes (4 vs our 3). Closed: no (4093 vs our 3594, ~500-tick gap) | 2026-08-25/26 | `lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/cu.usbmodem5B8E1132011 --robot.id=soarm_follower_01`. First attempt hit a transient `ConnectionError` on ID 3 during initial connect (torque-enable write got no status packet) - hardware confirmed healthy via `servo_scan.py` immediately after (all 6 responding cleanly), retry succeeded. Full per-joint ranges recorded: shoulder_pan 794-3418, shoulder_lift 821-2097, elbow_flex 814-3045, wrist_flex 1006-3031, gripper 4-4093. Saved to `~/.cache/huggingface/lerobot/calibration/robots/so_follower/soarm_follower_01.json`. Gripper's closed-end discrepancy (4093 vs our motor-stall-confirmed 3594) attributed to hand force during calibration exceeding what the STS3215's own stall torque (~16.5kg·cm at 7.4V) could achieve - not necessarily an error, but flagged for Step 3: LeRobot's PID control isn't capped the same conservative way our manual torque-limit testing was, so watch for strain/overheat commanding toward that closed extreme. |

---

## Step 3 — Keyboard/scripted joint control

**Update:** plain `lerobot-teleoperate --teleop.type=keyboard` does NOT work for
`so101_follower` in LeRobot 0.6.1 — confirmed by reading the installed source.
`KeyboardTeleop.get_action()` returns raw key names (`{'w': None}`), not joint
deltas, and the CLI's default processor pipeline never translates keys into this
robot's joint action space — every attempt crashes on the first loop tick
(`StopIteration` in `sync_write`, empty action dict). `keyboard_ee` is wired for a
*different* robot class (`So100FollowerEndEffector`), not plain `so101_follower`
either. This isn't a config mistake, it's a real gap in this LeRobot version's CLI.

**Fix:** `control/keyboard_joint_control.py`, adapted from XLeRobot's
[`0_so100_keyboard_joint_control.py`](https://github.com/Vector-Wangel/XLeRobot)
(same `SOFollower` class underlies both so100/so101, XLeRobot's script works
unmodified in principle — this port targets our `so101_follower` config/class names
explicitly). Continuous P-control loop, not one-shot commands:

```
cd control
source .venv/bin/activate
python keyboard_joint_control.py /dev/cu.usbmodem5B8E1132011 soarm_follower_01
```

Controls: Q/A W/S E/D R/F T/G = shoulder_pan/shoulder_lift/elbow_flex/wrist_flex/
wrist_roll (-/+), Y/H = gripper (-/+, 5% steps), X = exit (returns to start pose
first). Drive the arm to a moderate range on each joint, watch for smooth motion
and no runaway.

| Joint | Responds to laptop control? | Smooth (no jitter/runaway)? | Date | Notes |
|---|---|---|---|---|
| Shoulder pan (1) |  |  |  |  |
| Shoulder lift (2) |  |  |  |  |
| Elbow flex (3) |  |  |  |  |
| Wrist flex (4) |  |  |  |  |
| Wrist roll (5) |  |  |  |  |
| Gripper (6) |  |  |  |  |

---

## Step 4 — Camera recognized by control software

Confirm both mounted cameras (IMX335 wrist, AR0144 stereo overhead) are visible to
whatever software will record them — reuse the device-index discovery approach from
`diagnostics/camera_test.py` if LeRobot's own camera listing is unclear:

```
python -c "import cv2; [print(i, cv2.VideoCapture(i).isOpened()) for i in range(6)]"
```

Then confirm LeRobot's own camera config picks up the same devices (camera config is
typically part of the robot config from Step 1 — check `--help`/docs for the current
flag, e.g. `--robot.cameras`).

| Camera | Recognized by control software? | Date | Notes |
|---|---|---|---|
| IMX335 (wrist) | ✅ PASS | 2026-09-02 | Detected as cv2 index 0 by `control/record_episode.py`'s auto-probe (same approach `camera_test.py` uses) |
| AR0144 (stereo, overhead) | ✅ PASS | 2026-09-02 | Detected as cv2 index 1 |

---

## Step 5 — Still image capture

While the arm is idle (or mid-pose), capture a single still frame from each camera
through the control software's own capture path (not just `camera_test.py`, which
only proves the camera itself works — this step proves the *recording pipeline*
works end-to-end).

| Camera | Image captured? | Saved path | Date | Notes |
|---|---|---|---|---|
| IMX335 (wrist) |  |  |  |  |
| AR0144 (stereo) |  |  |  |  |

---

## Step 6 — Video recording during arm motion

Start a recording, move the arm through a few joints (Step 3-style manual control is
fine), stop the recording, then play back the saved file and confirm:
- Video is not corrupted / plays cleanly
- Frame rate is reasonable (no major stutter/drops)
- Arm motion visible in frame is smooth, not choppy — a choppy recording despite
  smooth real motion usually means the capture pipeline is dropping frames, not that
  the arm itself has a mechanical problem

| Camera | Result | Duration | Date | Notes |
|---|---|---|---|---|
| IMX335 (wrist) |  |  |  |  |
| AR0144 (stereo) |  |  |  |  |

---

## Step 7 — Combined episode recording (control + camera together)

This is the actual dataset-collection dry run: use `lerobot-record` (or equivalent)
to record one short "episode" — joint positions/actions AND camera frames captured
together, timestamped/synced, saved in LeRobot's dataset format. This is the format
later phases (VLA fine-tuning eval, per this project's `progress-documentation/`)
will actually consume, so this step is the real end-to-end proof the hardware is
usable for that purpose.

```
lerobot-record --robot.type=so101_follower ... --dataset.repo_id=<local-test> --dataset.num_episodes=1
```

| Result | Episode saved? | Camera + joint data both present? | Date | Notes |
|---|---|---|---|---|
|  |  |  |  |  |

---

## Sign-off

| All steps pass? | Date | Tested by |
|---|---|---|
|  |  |  |

**Known issues / follow-ups:**
-
