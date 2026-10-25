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

## Start (home) pose — captured once, used by every episode

One fixed pose in `control/start_pose.json`, committed to git: calibrated degrees
for the arm joints (gripper 0–100%), keyed by joint, plus `captured_at`.

Put the arm in the home pose (by hand, or with `keyboard_joint_control.py` /
`joint_jog.py`), then capture it and commit the file:

```bash
python start_pose.py                      # PORT ROBOT_ID default to device_map.json's follower
python start_pose.py /dev/cu.usbmodem5B8E1139151 soarm_follower_02 --force   # overwrite
```

- Overwriting an existing `start_pose.json` needs `--force`.
- Capture only reads the arm, never moves it, and refuses a pose outside the safety
  limits (e.g. a fully closed gripper reading slightly below 0%).
- Capture needs the serial port, so exit the jog tool first.
- `keyboard_joint_control.py`'s `X` returns to its own start pose before exiting, so
  the arm leaves the pose you jogged it to.
- Disconnecting releases torque, so choose a self-supporting pose or hold the arm
  while capturing.

`run_vla_episode.py` and `tune_execution_hz.py` always use it:

- Missing or invalid file → refuses to run, before touching the robot or backend.
- Moves home before the first model request; if any joint is more than ±5 (deg / %)
  off, refuses to start, naming the joints (termination reason
  `start_pose_not_reached`).
- Records target and reached joints in `<out>/start_pose.json`.
- Returns home before disconnect on max steps, task complete, Ctrl-C or error.
- No return move if home was never reached.
- Tolerance and move timeout are constants in `start_pose.py`.

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

---

## MLLM backends (Phase 15)

Five registered backends share one prompt, schema and chunk loop: `mllm-claude`,
`mllm-gpt`, `mllm-hf-qwen`, `mllm-hf-deepseek`, `mllm-hf-kimi`.

API keys are read only from environment variables exported in the shell, and are
never written to files (configs hold only the variable name):

```bash
export ANTHROPIC_API_KEY=...   # mllm-claude
export OPENAI_API_KEY=...      # mllm-gpt
export HF_TOKEN=...            # mllm-hf-* (fine-grained token with Inference Providers permission)
```

- **Spend:** about $1 per call for Claude/GPT at xhigh effort; HF variants need purchased HF Inference Providers credits (the free tier is $0.10/month), and Kimi-K3 is not cheap (about $0.35 per call).
- **Retention:** images leave this machine on every call (camera images and depth); Anthropic keeps request data for 30 days (30-day retention, not zero-data-retention), and each build prints the provider's retention NOTICE.

Show exactly what every backend is told (keyless, no provider call):

```bash
python -m vla_bridge.mllm_prompt --chunk-len 5
```

Preview one live set of model inputs (cameras plus depth heatmap/grid; no model call):

```bash
python -m vla_bridge.mllm_inputs --out outputs/mllm_inputs_preview \
  --wrist-camera-index 0 --stereo-camera-index "CCB Camera" \
  --calibration stereo_calibration.json --endpoint https://xxxx.ngrok-free.app/depth
```

One tiny paid smoke call per backend (synthetic images, 3-step chunk, no hardware;
refuses to spend without `--confirm-spend`):

```bash
python -m vla_bridge.mllm_smoke --backend mllm-claude --chunk-len 3 --out outputs/smoke --confirm-spend
```

Episode on the real arm:

```bash
python run_vla_episode.py --out outputs/mllm_001 --backend mllm-claude --max-steps 120 \
  --depth-endpoint https://xxxx.ngrok-free.app/depth --depth-calibration stereo_calibration.json \
  --instruction "Pick up the pen"
```

- Raise `--max-steps` for MLLM runs: with chunk_len 20, the default 60 ticks is only about 3 chunks (3 calls).
- Omit both depth flags to run without depth (the prompt then says depth is unavailable).
