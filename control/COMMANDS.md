# SOARM Control Commands

Copy-paste reference for all `control/` scripts. Run from the `control/` directory
with the venv active (`source .venv/bin/activate` if not already).

**Device auto-discovery (preferred over the table below):** ports AND camera
indices were both confirmed to drift across sessions/replugs on this rig (Plan
11-05). Run this first, any time a USB device has been unplugged/replugged:

```bash
python detect_devices.py
```

Writes `control/device_map.json` (gitignored, machine-local) with resolved
follower/leader ports+ids and wrist/stereo-overhead camera indices.
`run_vla_episode.py`, run with no `PORT`/`ROBOT_ID`/`--camera` given, reads its
defaults from this file automatically; explicit CLI args still override it.

**Known hardware (stale reference only — update if used; prefer
`detect_devices.py` above. macOS reassigns `/dev/cu.usbmodem*` suffixes on
replug; check with `ls /dev/cu.usbmodem*`):**

| Arm | Port | ID |
|---|---|---|
| Follower (30kg servos) | `/dev/cu.usbmodem5B8E1139151` | `soarm_follower_02` |
| Leader | `/dev/cu.usbmodem5B8E1132011` | `soarm_leader_01` |

---

## Leader-follower teleoperation (native LeRobot)

```bash
lerobot-teleoperate \
  --robot.type=so101_follower --robot.port=/dev/cu.usbmodem5B8E1139151 --robot.id=soarm_follower_02 \
  --teleop.type=so101_leader --teleop.port=/dev/cu.usbmodem5B8E1132011 --teleop.id=soarm_leader_01
```

Show all flags:
```bash
lerobot-teleoperate --help
```

---

## Keyboard joint control (single arm, no leader needed)

```bash
python keyboard_joint_control.py /dev/cu.usbmodem5B8E1139151 soarm_follower_02
```

Controls: `Q/A W/S E/D R/F T/G` = shoulder_pan/shoulder_lift/elbow_flex/wrist_flex/
wrist_roll (-/+), `Y/H` = gripper (-/+, 5% steps), `X` = exit (returns to start pose first).

---

## Single joint jog (scripted, one move)

```bash
python joint_jog.py /dev/cu.usbmodem5B8E1139151 soarm_follower_02 shoulder_pan 10 --max-relative-target 10
```

`JOINT` choices: `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper`
`DELTA`: degrees for arm joints, 0–100 units for gripper.

---

## Raw gripper jog (bypasses calibration, ticks not %)

```bash
python jog_gripper_raw.py /dev/cu.usbmodem5B8E1139151 soarm_follower_02 --step 15
```

---

## Gripper recalibration

```bash
python recalibrate_gripper.py /dev/cu.usbmodem5B8E1139151 soarm_follower_02
```

---

## Leader vs. follower pose comparison (calibration sanity check)

```bash
python compare_leader_follower.py \
  /dev/cu.usbmodem5B8E1139151 soarm_follower_02 \
  /dev/cu.usbmodem5B8E1132011 soarm_leader_01
```

Args order: `FOLLOWER_PORT FOLLOWER_ID LEADER_PORT LEADER_ID`

---

## Episode recording (cameras + joint log)

```bash
python record_episode.py /dev/cu.usbmodem5B8E1139151 soarm_follower_02 \
  --duration 5 --camera 0 --camera 1 --out outputs/episode_001 --fps 15
```

- `--duration 0` → single still frame per camera instead of video
- `--no-joints` → skip joint-position logging (camera-only capture)
- `--camera` is repeatable (one flag per camera index)

---

## Camera device discovery

```bash
python -c "import cv2; [print(i, cv2.VideoCapture(i).isOpened()) for i in range(6)]"
```

---

## Port discovery (if a device stops responding)

```bash
ls /dev/cu.usbmodem*
```
