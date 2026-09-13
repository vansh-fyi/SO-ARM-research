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

**Update (2026-09-09):** a **leader** arm has now been assembled, so this UAT
covers two physical arms, not one. The original (lower-torque) servo set could not
lift the follower arm's own weight under load, so those servos + their driver chip
were repurposed as the **leader** (acceptable since a human does the lifting there —
the leader only needs to report position, not hold load). The follower arm is now
built with a new 30kg-load servo set (10kg more than the original), and driver chips
were swapped between the two arms accordingly. "Control from laptop" now includes
real leader-follower teleoperation via LeRobot, in addition to the keyboard/scripted
joint control from Step 3. All prior calibration data (recorded against the old
single-arm/servo configuration) is discarded — see Steps 1-2.

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

## Step 1 — Robot config: port + servo IDs (leader + follower)

LeRobot needs to know the serial port and which servo ID maps to which joint, for
**both** arms now. Reuse what's already confirmed in the assembly UATs (Joint 1-5 =
IDs 1-5, gripper = ID 6) — that convention holds independently on each arm.

1. Find both Waveshare adapters' ports:
   ```
   python -c "from serial.tools import list_ports; [print(p.device, p.description) for p in list_ports.comports()]"
   ```
   Since driver chips were swapped between the arms, don't assume port identity
   carries over from prior UATs — confirm by unplugging one arm and re-scanning to
   see which port disappears.
2. Confirm all 6 servo IDs respond on each port (`servo_scan.py <port>`).
3. Create/confirm configs: the follower as `so101_follower` (LeRobot `--robot.type`)
   and the leader as `so101_leader` (LeRobot `--teleop.type`), each with its own port
   and a `.id` you choose to name that specific arm.

| Arm | Result | Port | Robot/Teleop ID chosen | Date | Notes |
|---|---|---|---|---|---|
| Follower | ✅ PASS | `/dev/cu.usbmodem5B8E1139151` | `soarm_follower_02` | 2026-09-09 | New port after driver-chip swap — confirmed by unplug/replug test (this port persisted when leader was unplugged). New `_02` id since the 30kg servo set + swapped driver chip make this a distinct hardware config from `soarm_follower_01`. `servo_scan.py` found all 6 IDs (1-5 arm, 6 gripper) responding cleanly, no remapping needed. |
| Leader | ✅ PASS | `/dev/cu.usbmodem5B8E1132011` | `soarm_leader_01` | 2026-09-09 | This is the port previously labeled follower in the original (pre-leader) UAT — confirmed via unplug test that it now belongs to the leader arm (repurposed original low-torque servos + driver chip). `servo_scan.py` found all 6 IDs responding cleanly. |

---

## Step 2 — LeRobot's own calibration (leader + follower)

LeRobot runs its own calibration routine (separate from the raw tick calibration we
did for the gripper in its assembly UAT) to record joint zero-points and ranges in
its own config format, independently for each arm. Run whatever calibration entry
point your installed version exposes (`lerobot-calibrate`, using `--robot.*` flags
for the follower and `--teleop.*` flags for the leader) and follow its prompts
(usually: move each joint through its full range while it records min/max, then a
reference "home" pose).

```
lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/cu.usbmodem5B8E1139151 --robot.id=soarm_follower_02
lerobot-calibrate --teleop.type=so101_leader --teleop.port=/dev/cu.usbmodem5B8E1132011 --teleop.id=soarm_leader_01
```

**Prior calibration discarded:** `soarm_follower_01`'s calibration (from the
single-arm setup, old servo/driver config) no longer applies — new servos, swapped
driver chips, and a second physical arm mean this needs a clean run, not a diff
against old numbers.

| Arm | Result | Date | Notes |
|---|---|---|---|
| Follower (`soarm_follower_02`) | ✅ PASS | 2026-09-09 | Saved to `~/.cache/huggingface/lerobot/calibration/robots/so_follower/soarm_follower_02.json`. Final per-joint ranges: shoulder_pan 1269-2869, shoulder_lift 904-3349, elbow_flex 832-3044, wrist_flex 899-3220, wrist_roll 0-4095 (full turn, by design), gripper 48-3637. New 30kg servo set. **Gripper calibration needed real troubleshooting** — initial attempts wrongly spanned nearly the full 0-4095 circle because the raw encoder wrapped around during hand-calibration on a mechanism with no tactile hard stop; root-caused, then resolved via a servo-horn reassembly + full recalibration (all 6 follower joints ended up recalibrated together, hence all new homing offsets vs. the earlier per-joint table above). See "Known issues" at the end of this doc for the full story. Final gripper span (3589 ticks) reads wider than pre-reassembly estimates of true jaw travel suggested, but functional operation confirmed by hand afterward — flagged for awareness, not an open issue. |
| Leader (`soarm_leader_01`) | ✅ PASS | 2026-09-09 | Saved to `~/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/soarm_leader_01.json`. **Recalibrated a second time** after discovering servo IDs 2 and 3 were physically swapped on the leader (ID 2 was mounted as elbow_flex, ID 3 as shoulder_lift) — see Step 3b notes and "Known issues" for the full diagnosis. Fixed via `servo_set_id.py` (3-step swap through a temp ID), confirmed via `servo_scan.py`, then this recalibration. Final per-joint ranges: shoulder_pan 917-3133, shoulder_lift 798-3229, elbow_flex 880-3076, wrist_flex 842-3160, wrist_roll 0-4095 (full turn), gripper 1611-2896. These are the original (lower-torque) servos repurposed from the pre-leader follower build. Gripper range (span ~1285) is notably narrower than the follower's — see leader-gripper-span follow-up at the end of this doc. |

---

## Step 3 — Joint control: keyboard (follower) + leader-follower teleop

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

**Command now targets the new follower hardware/id** (30kg servos, `soarm_follower_02`,
port `/dev/cu.usbmodem5B8E1139151`):
```
python keyboard_joint_control.py /dev/cu.usbmodem5B8E1139151 soarm_follower_02
```

**Retest required** — new servo set (30kg vs. old, higher torque) and swapped driver
chip mean the prior PASS results no longer apply; the P-control gains tuned against
the old servos' response characteristics may also behave differently now.

| Joint | Responds to laptop control? | Smooth (no jitter/runaway)? | Date | Notes |
|---|---|---|---|---|
| Shoulder pan (1) | ✅ PASS | ✅ PASS | 2026-09-09 | |
| Shoulder lift (2) | ✅ PASS | ✅ PASS | 2026-09-09 | Stock P=16/D=32 gains (reverted from stale 7.4V-servo tuning, see Step 3 fix history) held up fine under the new 30kg servos. |
| Elbow flex (3) | ✅ PASS | ✅ PASS | 2026-09-09 | Same gain revert as shoulder_lift; previously tripped overload under sustained holding with the stale soft gains, confirmed fixed. |
| Wrist flex (4) | ✅ PASS | ✅ PASS | 2026-09-09 | Previously tripped overload from the unclamped-target bug (unbounded RANGE_M100_100 accumulation); confirmed fixed after clamping targets to +-100. |
| Wrist roll (5) | ✅ PASS | ✅ PASS | 2026-09-09 | |
| Gripper (6) | ✅ PASS | ✅ PASS | 2026-09-09 | Post-recalibration/reassembly (see Step 2 notes and Known Issues). |

### Step 3b — Real leader-follower teleoperation (new)

With both arms now assembled, LeRobot's native leader-follower teleop is possible
for the first time on this project — physically move the leader's joints and
confirm the follower mirrors them in real time.

```
lerobot-teleoperate \
  --robot.type=so101_follower --robot.port=/dev/cu.usbmodem5B8E1139151 --robot.id=soarm_follower_02 \
  --teleop.type=so101_leader --teleop.port=/dev/cu.usbmodem5B8E1132011 --teleop.id=soarm_leader_01
```

Move each leader joint through a moderate range one at a time and watch the
follower. Flag anything where the follower doesn't track, tracks with a large lag,
or where the differing gripper calibration spans (Step 2 note) cause the follower
gripper to not reach fully open/closed.

| Joint | Follower tracks leader? | Smooth (no jitter/lag/runaway)? | Date | Notes |
|---|---|---|---|---|
| Shoulder pan (1) | ✅ PASS | ✅ PASS | 2026-09-09 | |
| Shoulder lift (2) | ✅ PASS | ✅ PASS | 2026-09-09 | See "Known issues" — required discovering and fixing a leader servo ID 2/3 swap before this passed. |
| Elbow flex (3) | ✅ PASS | ✅ PASS | 2026-09-09 | Same ID-swap fix as shoulder_lift. |
| Wrist flex (4) | ✅ PASS | ✅ PASS | 2026-09-09 | |
| Wrist roll (5) | ✅ PASS | ✅ PASS | 2026-09-09 | |
| Gripper (6) | ✅ PASS | ✅ PASS | 2026-09-09 | Follower gripper needed `drive_mode=1` in calibration (direction was reversed from the sorted-open/closed capture in `recalibrate_gripper.py`) — see "Known issues". |

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
| IMX335 (wrist) | ✅ PASS | 2026-09-09 | cv2 index 0. Confirmed by capturing and visually inspecting a frame (green-tinted low-light image looking up from the gripper area) rather than trusting `isOpened()` alone. |
| AR0144 (stereo, overhead) | ✅ PASS | 2026-09-09 | cv2 index 1. Frame is 2560x720 - a side-by-side stereo pair, confirmed visually (desk/tools view from overhead). Note: cv2 index 2 also opens successfully but is the laptop's own built-in webcam, not a robot camera - don't confuse it with a robot feed if device indices shift in the future. |

---

## Step 5 — Still image capture

While the arm is idle (or mid-pose), capture a single still frame from each camera
through the control software's own capture path (not just `camera_test.py`, which
only proves the camera itself works — this step proves the *recording pipeline*
works end-to-end).

| Camera | Image captured? | Saved path | Date | Notes |
|---|---|---|---|---|
| IMX335 (wrist) | ✅ PASS | `control/outputs/step5_still/camera_0.png` | 2026-09-09 | Via `control/record_episode.py --duration 0`. Clean, correctly-exposed color image (the earlier green tint seen during raw `camera_test.py`-style probing was a transient/lighting artifact, not present here). |
| AR0144 (stereo) | ✅ PASS | `control/outputs/step5_still/camera_1.png` | 2026-09-09 | Via same command. 2560x720 side-by-side stereo pair, both halves clean. |

**Hardware note (camera hub power):** both robot cameras were initially on the same
bus-powered USB hub as the two arm serial adapters, which caused the IMX335 to fail
to open (`OpenCV: raised unknown C++ exception`) — a bus-powered hub splits limited
laptop USB power across all attached devices, and the camera (highest power draw)
was the one that lost out. Fixed by moving both cameras to a separate hub, keeping
only the two (lower-power) arm serial adapters on the original one. cv2 device
indices are not stable across USB topology changes — reconfirmed by frame shape each
time: IMX335 is 1920x1080 (color, ceiling-facing wrist view), AR0144 is 2560x720
(stereo pair), and the laptop's own webcam also enumerates (1920x1080, but a face,
not a robot view) — don't assume index 0/1 without checking.

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
| IMX335 (wrist) | ✅ PASS | 8.0s / 107 frames (~13.4fps actual vs. 15fps target) | 2026-09-09 | Motion driven via real leader-follower teleop, not keyboard. Plays cleanly, no corruption. First vs. last frame dramatically different (confirms real motion captured, some natural motion blur from a fast move - not dropped-frame choppiness). `control/outputs/step6_video_v3/camera_0.mp4`. |
| AR0144 (stereo) | ✅ PASS | 8.0s / 107 frames | 2026-09-09 | Same session. Both stereo halves clean, motion clearly visible between frames. `control/outputs/step6_video_v3/camera_1.mp4`. |

**Camera hardware flakiness (this session, resolved):** the IMX335 intermittently
failed to open (`OpenCV: raised unknown C++ exception`) across several attempts in
this step, independent of whether the arms were actively teleoperating (ruled out
as a shared-USB-power-with-the-arms issue by testing with teleop stopped - it still
failed). Resolved by: reseating the IMX335's USB cable AND switching the laptop from
battery to AC power (macOS/some laptops reduce USB port power delivery on battery,
which can matter for a power-hungry camera). If this recurs, check AC power first,
not just cabling.

---

## Step 7 — Combined episode recording (control + camera together)

This is the actual dataset-collection dry run: use `lerobot-record` (or equivalent)
to record one short "episode" — joint positions/actions AND camera frames captured
together, timestamped/synced, saved in LeRobot's dataset format. This is the format
later phases (VLA fine-tuning eval, per this project's `progress-documentation/`)
will actually consume, so this step is the real end-to-end proof the hardware is
usable for that purpose.

Now that a leader arm exists, record via real teleop rather than keyboard control —
this is closer to how actual dataset-collection episodes will be driven going
forward:

```
lerobot-record \
  --robot.type=so101_follower --robot.port=/dev/cu.usbmodem5B8E1139151 --robot.id=soarm_follower_02 \
  --robot.cameras '{"wrist": {"type": "opencv", "index_or_path": 0, "width": 1920, "height": 1080, "fps": 30}, "overhead": {"type": "opencv", "index_or_path": 1, "width": 2560, "height": 720, "fps": 30}}' \
  --teleop.type=so101_leader --teleop.port=/dev/cu.usbmodem5B8E1132011 --teleop.id=soarm_leader_01 \
  --dataset.repo_id=local_test/step7_episode --dataset.single_task="UAT Step 7 test episode" \
  --dataset.num_episodes=1 --dataset.episode_time_s=8 --dataset.reset_time_s=1 --dataset.push_to_hub=false
```

Notes on the command: `--robot.cameras` needs explicit `width`/`height`/`fps` per
camera (fps must match the camera's actual native rate - 30, not 15, for both of
ours - LeRobot validates strictly and errors if it doesn't match). `lerobot-record`
also required installing the `datasets` extra (`pip install 'lerobot[dataset]'`),
not present in the base `lerobot[feetech]` install from Step 0.

| Result | Episode saved? | Camera + joint data both present? | Date | Notes |
|---|---|---|---|---|
| ✅ PASS | ✅ PASS | ✅ PASS | 2026-09-09 | `local_test/step7_episode_20260909_140149`. 239 synced timesteps (fps=30, ~8s). Both `action` and `observation.state` present with correct per-joint names; values genuinely change start-to-end (e.g. shoulder_pan 2.9→29.4, shoulder_lift -105.7→-71.2), confirming real motion was captured, not a frozen/static episode. Both camera videos (`observation.images.wrist`, `observation.images.overhead`) verified playable with exactly 239 frames each, matching the joint-data row count. Driven via real leader-follower teleop. |

**Blockers hit before this passed (all resolved):** (1) missing `datasets` package
- installed via `pip install 'lerobot[dataset]'`; (2) `--robot.cameras` requires
`width`/`height` or the robot rejects the camera entirely; (3) the IMX335 camera
intermittently failed to open (`raised unknown C++ exception`) even after Step 6's
fix (AC power + cable reseat) - ruled out a `cv2`/`av` library collision (a genuine
but separate issue flagged by an objc runtime warning after installing the
`datasets` extra) and a QuickTime camera lock, before finding the real cause: the
camera hub was still bus-powered. Fixed by powering the hub from a USB-C PD source
(a power bank's PD-out port works fine) instead of relying on the laptop's own USB
port budget - if this recurs on a fresh setup, check hub power before anything
else, since it now has a track record of being the actual root cause here; (4) a
transient `Failed to write 'Torque_Enable' ... no status packet` - the same known
intermittent Feetech comms dropout seen throughout this UAT, cleared on retry.

---

## Sign-off

| All steps pass? | Date | Tested by |
|---|---|---|
| ✅ YES (Steps 0-7) | 2026-09-09 | vanshux23 |

**Known issues / follow-ups:**
- Leader gripper's mechanical travel is narrower than the follower's (calibrated
  span 1996-3318 vs. follower's 48-3637, confirmed by user to be due to the leader
  gripper handle's shape/linkage geometry). Not a functional blocker — LeRobot
  normalizes each arm to its own 0-100% range, so the follower still reaches full
  open/close — but it does mean coarser control resolution on the leader side.
  Increasing it requires a mechanical redesign (longer lever/different linkage) of
  the leader gripper handle, not a software fix. Revisit in a future hardware
  iteration if finer proportional grasp control is needed.
- Follower gripper (servo ID 6) had a real calibration bug during Step 2, worth
  recording in detail since it can recur on any future gripper recalibration:
  `record_ranges_of_motion` (LeRobot's live min/max tracker used during
  `lerobot-calibrate`) has no wraparound handling — if the gripper's true operating
  arc straddles the raw encoder's 0/4095 rollover point, or if the operator's hand
  overshoots the mechanism's real (non-tactile) end stop during recording, the
  recorded range comes out as nearly the full 4096-tick circle instead of the true
  short arc. This happened here (`14-4090` from the first calibration attempt) and
  the underlying gripper mechanism turned out to have **no tactile hard stop** —
  nothing resists the servo horn rotating well past the jaws' real functional
  range, so a hand sweep easily overshoots before the operator can feel it and
  stop. Fixed via: (1) `control/recalibrate_gripper.py` and
  `control/jog_gripper_raw.py` — custom tools written during this session that
  detect the wraparound signature (a >2048-tick jump between consecutive reads)
  live and stop immediately instead of silently corrupting the whole recording;
  (2) a physical reassembly of the gripper servo horn to re-center the servo's
  electrical zero away from the jaws' real working arc; (3) a full recalibration
  of all 6 follower joints afterward. Final gripper range (48-3637) confirmed
  functional by hand. If recalibrating this gripper again in the future, prefer
  `jog_gripper_raw.py`'s deliberate single-step-per-keypress approach over a live
  hand sweep, and re-verify the raw-tick open/close direction after any servo-horn
  remount (it's mount-orientation-dependent, not fixed). Its calibration also ended
  up needing `drive_mode: 1` (see gripper entry in Step 2's follower row) because
  `recalibrate_gripper.py` assigns `range_min`/`range_max` via `sorted()` on the two
  captured raw positions rather than preserving which one was physically open vs.
  closed — harmless as long as the resulting direction is checked against the
  other arm (as done here) and corrected via `drive_mode`, but worth fixing in the
  script itself if it gets reused.
- **Leader servo IDs 2 and 3 were physically swapped**: ID 2 was mounted as
  elbow_flex's servo and ID 3 as shoulder_lift's, backwards from the intended
  1-5=joint convention (shoulder_pan/shoulder_lift/elbow_flex/wrist_flex/wrist_roll)
  used everywhere else in this project. This was the real cause of a
  leader-follower teleop symptom that initially looked like a calibration-direction
  bug: with both arms visually posed the same, `shoulder_lift`/`elbow_flex` read as
  near-mirror-opposite normalized values between the two arms (confirmed via a new
  diagnostic, `control/compare_leader_follower.py`, which snapshots both arms'
  normalized joint values side by side — useful for any future leader/follower
  correspondence check). A `drive_mode=1` calibration patch was tried first and
  seemed plausible, but had **zero effect** on the leader's arm joints specifically
  because `so101_leader`/`so101_follower` both default to `use_degrees=True`, and
  the `DEGREES` branch of LeRobot's normalize/unnormalize math (`motors_bus.py`)
  never applies `drive_mode` at all (only `RANGE_M100_100`/`RANGE_0_100` do) — worth
  remembering if `drive_mode` ever silently seems to do nothing again. Root cause
  was only found by physically verifying each raw servo ID against the joint it
  actually moves, via the original bring-up UAT's own `servo_move_test.py`
  (`diagnostics/servo_move_test.py`) run one ID at a time. Fixed by reassigning IDs
  with `servo_set_id.py` through a temporary ID (3-step: 2→9, 3→2, 9→3, since
  writing 2→3 directly would collide with the servo already at 3), confirmed via
  `servo_scan.py`, then a full leader recalibration (which also reset the now-moot
  `drive_mode=1` patch back to 0). **Takeaway for any future ID/wiring suspicion**:
  don't trust calibration-side normalized-value comparisons alone to diagnose a
  direction mismatch — verify at the raw servo-ID level first with
  `servo_move_test.py`, since a wiring/ID problem and a calibration-direction
  problem can look identical from the normalized-value side.
