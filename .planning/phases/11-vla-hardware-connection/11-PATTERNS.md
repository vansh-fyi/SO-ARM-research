# Phase 11: VLA Hardware Connection - Pattern Map

**Mapped:** 2026-09-20
**Files analyzed:** 6 (new) + 1 (modified)
**Analogs found:** 6 / 7

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|--------------------|------|-----------|-----------------|---------------|
| `control/vla_bridge/safety_validator.py` | utility (input-validation gate) | transform (validate/clamp) | `control/keyboard_joint_control.py` (`read_positions`, clamp logic in `control_loop`) | role-match (net-new logic, but retry/clamp idiom exists) |
| `control/vla_bridge/io_logger.py` | utility (structured log writer) | event-driven / batch (one record per step) | `control/record_episode.py` (`joints.csv` writer + per-frame capture loop) | role-match (CSV→JSONL swap, same per-step-record idea) |
| `control/vla_bridge/robot_client.py` | service (wraps `SO101Follower` + LeRobot `RobotClient`) | request-response (obs out, action in, over gRPC bridge) | `control/keyboard_joint_control.py` (whole file: connect/read/write loop) | role-match (control loop skeleton, network layer is net-new) |
| `control/run_vla_episode.py` | controller / entrypoint (wires client+validator+logger+e-stop) | event-driven (control loop) | `control/record_episode.py` `main()` (argparse + camera/robot setup/teardown) | exact (CLI shape, connect/disconnect lifecycle) |
| `control/vla_bridge/policy_server_launch.md` | config/docs (Colab notebook instructions, not executable) | n/a | none in-repo (new pattern) | no analog |
| `control/test_action_contract.py` | test | unit | `scripts/test_calibration_utils.py` (not read this session, but same pytest convention per RESEARCH.md) | role-match |
| `control/test_safety_validator.py` | test | unit | same as above | role-match |
| `control/test_io_logger.py` | test | unit + integration | same as above | role-match |
| `control/keyboard_joint_control.py` (modified: fix stale comment) | controller (existing) | request-response | itself | exact (in-place edit) |

## Pattern Assignments

### `control/vla_bridge/robot_client.py` (service, request-response)

**Analog:** `control/keyboard_joint_control.py`

**Imports pattern** (lines 35-38):
```python
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SO101Follower
from lerobot.teleoperators.keyboard.configuration_keyboard import KeyboardTeleopConfig
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop
```
For the VLA bridge, swap the keyboard teleop import for `lerobot.async_inference.robot_client.RobotClient` (per RESEARCH.md Pattern 1), but keep the `SOFollowerRobotConfig`/`SO101Follower` import block verbatim — this is the sanctioned, already-working hardware connection path.

**Connect/calibration pattern** (lines 176-186, reuse verbatim):
```python
robot = SO101Follower(SOFollowerRobotConfig(port=port, id=robot_id))
robot.connect(calibrate=False)
with robot.bus.torque_disabled():
    robot.bus.write_calibration(robot.calibration)
```
Note `record_episode.py` uses the simpler (non-torque-disabled) form `robot.connect(calibrate=False); robot.bus.write_calibration(robot.calibration)` — both exist in-repo; prefer the `torque_disabled()` form since it is the more defensive/newer pattern per its own inline comment (EEPROM write requires torque off).

**Comms-retry pattern (VLAHW-02's comms-failure handling)** (lines 58-76):
```python
def read_positions(robot, retries=5, backoff=0.05):
    last_error = None
    for attempt in range(retries):
        try:
            obs = robot.get_observation()
            return {key.removesuffix(".pos"): val for key, val in obs.items() if key.endswith(".pos")}
        except ConnectionError as e:
            last_error = e
            time.sleep(backoff)
    raise last_error
```
Copy this pattern directly for the robot-side of the bridge (reading joint state to send as an observation to the Colab server); extend the same `except (ConnectionError, RuntimeError)` guard from `move_to_positions`/`control_loop` (lines 84-88, 98-101, 142-159) around `send_action()` calls receiving validated actions back from the bridge.

**Control-loop / KeyboardInterrupt e-stop pattern** (lines 109, 119-166, especially 161-166):
```python
def control_loop(robot, keyboard, start_positions, kp=0.5, control_freq=30):
    ...
    while True:
        try:
            ...
        except KeyboardInterrupt:
            print("Interrupted.")
            break
        except Exception:
            traceback.print_exc()
            break
```
This is the exact e-stop skeleton D-03 says to extend, not replace. `run_vla_episode.py`'s main loop should wrap the observation→bridge→validate→execute cycle in the same `try/except KeyboardInterrupt: break` shape, then call the existing "return to start position" move (lines 125-126: `move_to_positions(robot, start_positions, kp=0.2, control_freq=control_freq, max_seconds=5.0)`) as the safe-shutdown action before disconnecting.

**Known stale comment to fix (low-risk cleanup per RESEARCH.md Pattern 2):** line 134 `# RANGE_M100_100 joints (see so_follower.py Motor() norm_mode) - unclamped ...` is factually wrong (real mode is DEGREES, not RANGE_M100_100) and should be corrected as part of this phase's file-modification list.

---

### `control/vla_bridge/safety_validator.py` (utility, transform)

**No direct in-repo analog for the validation function itself** — this is net-new per RESEARCH.md's own "Don't Hand-Roll" table. However, borrow two idioms from `keyboard_joint_control.py`:

**Clamping idiom** (lines 131-138, adapt for JOINT_LIMITS_DEG instead of a fixed ±100):
```python
if joint == "gripper":
    new_target = max(0.0, min(100.0, current_target + direction * 5.0))
else:
    new_target = max(-100.0, min(100.0, current_target + direction))
```
Replace the hardcoded `-100.0/100.0` with the per-joint `JOINT_LIMITS_DEG` bounds from RESEARCH.md Pattern 3 (`shoulder_pan`: ±70.33°, `shoulder_lift`: ±107.47°, `elbow_flex`: ±97.23°, `wrist_flex`: ±102.02°, `wrist_roll`: −157.21°/+162.79°, `gripper`: 0–100 RANGE_0_100 percent — NOT meters, see RESEARCH.md Pitfall 4).

**Error-handling idiom (guard + fallback rather than raise)** (lines 100-101, 156-159):
```python
except (ConnectionError, RuntimeError) as e:
    print(f"Write failed, skipping this tick: {e}")
```
Mirror this "log and hold last-known-good state" style (not a hard crash) for NaN/inf/malformed-action rejection in the validator — return the current/previous safe action rather than raising, consistent with the existing project convention of never letting a single bad tick kill the control loop.

**Reference implementation to start from** — RESEARCH.md's own sketch (already vetted against this project's calibration data):
```python
JOINT_LIMITS_DEG = {
    "shoulder_pan": (-70.33, 70.33),
    "shoulder_lift": (-107.47, 107.47),
    "elbow_flex": (-97.23, 97.23),
    "wrist_flex": (-102.02, 102.02),
    "wrist_roll": (-157.21, 162.79),
    "gripper": (0.0, 100.0),
}

def validate_action(action: dict, current_state: dict) -> tuple[dict, list[str]]:
    flags = []
    safe = {}
    for joint, value in action.items():
        if value is None or not math.isfinite(value):
            flags.append(f"{joint}: rejected non-finite value ({value}), holding current position")
            safe[joint] = current_state.get(joint, 0.0)
            continue
        lo, hi = JOINT_LIMITS_DEG[joint]
        clamped = max(lo, min(hi, value))
        if clamped != value:
            flags.append(f"{joint}: clamped {value} -> {clamped} (limit {lo}/{hi})")
        safe[joint] = clamped
    return safe, flags
```

**Calibration-derivation cross-reference:** `scripts/calibration_utils.py`'s `JOINTS_FROM_CALIBRATION` tuple and `calibration_ticks_to_radians()` are the authoritative formula these degree limits were derived from — re-derive from the live `soarm_follower_02.json` calibration file at execution time rather than hardcoding permanently (per RESEARCH.md Assumption A2), reusing this module's `CALIBRATION_PATH` constant and tick→degree math (drop the `math.radians()` conversion at the end since the validator needs degrees, not radians — everything else in the formula is identical).

---

### `control/vla_bridge/io_logger.py` (utility, event-driven/batch)

**Analog:** `control/record_episode.py`

**Imports pattern** (lines 19-27):
```python
import argparse
import csv
import time
from pathlib import Path

import cv2

from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SO101Follower
```
For the JSONL logger, swap `import csv` for `import json`, keep `pathlib.Path` for path handling (matches project convention favoring `Path` in newer scripts).

**Per-camera capture + write pattern** (lines 30-38, 83-97):
```python
def open_cameras(indices):
    caps = {}
    for idx in indices:
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            print(f"WARNING: camera index {idx} did not open, skipping")
            continue
        caps[idx] = cap
    return caps
```
```python
for idx, cap in caps.items():
    ok, frame = cap.read()
    if ok:
        writers[idx].write(frame)
```
Reuse `open_cameras()` verbatim; for Phase 11's per-step still-frame capture (not continuous video), adapt the `record_still()` warm-up loop (lines 41-54) but save one PNG per step per camera into `camera_wrist/NNNNNN.png` / `camera_overhead/NNNNNN.png`, and record the per-camera capture timestamp independently (per RESEARCH.md Pattern 4's explicit "per-camera timestamps captured independently" requirement) rather than the single shared `ts` this analog uses (line 81: `ts = t0 - start`, shared across all cameras — do not carry this shared-timestamp simplification forward).

**Per-record write pattern to swap CSV → JSONL** (lines 66-73, 88-92):
```python
joints_writer.writerow(["timestamp_s"] + [f"{j}.pos" for j in [...]])
...
row = [ts] + [obs.get(f"{j}.pos", "") for j in [...]]
joints_writer.writerow(row)
```
Replace with one `json.dumps(record) + "\n"` append per inference step, using the schema from RESEARCH.md Pattern 4 (step, timestamp_utc, instruction, camera_frames with per-camera path+timestamp, joint_state, raw_model_output, validated_action, validator_flags, executed_action, latency_ms, model_version).

**File lifecycle pattern (open/close bracketing the recording loop)** (lines 66-70, 102-104):
```python
joints_file = open(joints_path, "w", newline="")
joints_writer = csv.writer(joints_file)
...
if joints_file:
    joints_file.close()
    print(f"joints: saved -> {out_dir / 'joints.csv'}")
```
Mirror this open-before-loop / close-after-loop / print-confirmation shape for `episode.jsonl`, using `"a"` (append) mode per-step-write if the logger writes incrementally (recommended, so a mid-episode bridge drop doesn't lose already-logged steps — directly serves VLAHW-04's "durable log" requirement).

---

### `control/run_vla_episode.py` (controller/entrypoint, event-driven)

**Analog:** `control/record_episode.py` `main()` (lines 108-153)

**Argparse + camera-probe + connect/teardown pattern** (lines 109-117, 119-127, 140-152):
```python
parser = argparse.ArgumentParser()
parser.add_argument("port")
parser.add_argument("robot_id")
parser.add_argument("--camera", type=int, action="append", default=[], help="repeatable, e.g. --camera 0 --camera 1")
parser.add_argument("--out", type=Path, required=True)
...
if not args.camera:
    print("No --camera given. Probing indices 0-5...")
    ...
caps = open_cameras(args.camera)
if not caps:
    print("No cameras opened, aborting.")
    return
robot = SO101Follower(SOFollowerRobotConfig(port=args.port, id=args.robot_id))
robot.connect(calibrate=False)
robot.bus.write_calibration(robot.calibration)
try:
    record_episode(robot, caps, args.out, args.duration, args.fps, log_joints=not args.no_joints)
finally:
    for cap in caps.values():
        cap.release()
    if robot:
        robot.disconnect()
```
Copy this positional-`port`/`robot_id` + `--out`/`--camera` argparse shape verbatim (matches the project convention noted in CONTEXT.md: "`PORT ROBOT_ID` positional args"), add `--server-address` (ngrok tunnel host:port), `--task` (instruction string), and `--checkpoint` (HF Hub model id) as new flags. Keep the `try/finally` teardown structure to guarantee camera release + robot disconnect + tunnel/bridge cleanup even on a mid-episode error, and additionally wrap the main loop in `except KeyboardInterrupt` (from `keyboard_joint_control.py`, see above) for the e-stop path.

---

## Shared Patterns

### Positional CLI args (`PORT ROBOT_ID`)
**Source:** `control/keyboard_joint_control.py` lines 170-171, `control/record_episode.py` lines 110-111
**Apply to:** All new `control/` entrypoint scripts (`run_vla_episode.py`)
```python
port = sys.argv[1] if len(sys.argv) > 1 else input("SO-101 port (e.g. /dev/cu.usbmodem5B8E1132011): ").strip()
robot_id = sys.argv[2] if len(sys.argv) > 2 else input("Robot ID (e.g. soarm_follower_01): ").strip()
```
(or the `argparse` positional-args equivalent, as in `record_episode.py`) — either idiom is acceptable; `record_episode.py`'s argparse form is preferred for `run_vla_episode.py` since it also needs several optional flags.

### Comms-failure retry (VLAHW-02)
**Source:** `control/keyboard_joint_control.py` lines 58-76 (`read_positions`), lines 98-101 and 156-159 (`send_action` guards)
**Apply to:** `robot_client.py` (both the local observation read and the post-validation action write)

### Robot connect/calibration boilerplate
**Source:** `control/keyboard_joint_control.py` lines 176-186 (torque-disabled write_calibration) or `control/record_episode.py` lines 143-144 (simpler form)
**Apply to:** `robot_client.py`, `run_vla_episode.py`

### E-stop via KeyboardInterrupt
**Source:** `control/keyboard_joint_control.py` lines 109-166 (`control_loop`), especially lines 161-166 and the "return to start position" call at lines 125-126
**Apply to:** `run_vla_episode.py`'s main control loop — required by D-03/VLAHW-02

### Per-camera capture with warn-and-skip on failed open
**Source:** `control/record_episode.py` lines 30-38 (`open_cameras`)
**Apply to:** `robot_client.py` (observation capture) and `io_logger.py` (frame persistence)

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| `control/vla_bridge/policy_server_launch.md` | config/docs | n/a | Colab-side notebook instructions; no prior Colab-hosted `PolicyServer`/gRPC pattern exists anywhere in this repo (Phase 3's Colab VLA work was sim-only, feeding `OffScreenRenderEnv`, not a live hardware relay — RESEARCH.md confirms this gap explicitly). Planner should build this directly from RESEARCH.md Pattern 1's Colab-side code example rather than an in-repo analog. |
| `control/vla_bridge/robot_client.py`'s gRPC/tunnel wiring specifically (as opposed to the SO101Follower parts, which do have an analog) | service | request-response over network | No prior local↔Colab live bridge exists in `control/`; use `lerobot.async_inference.robot_client.RobotClient` directly per RESEARCH.md Pattern 1 rather than inventing a new transport. |

## Metadata

**Analog search scope:** `control/` (all `.py` files), `scripts/calibration_utils.py`; RESEARCH.md's own Code Examples and Don't-Hand-Roll table used as a secondary source where no in-repo analog exists (explicitly flagged above wherever this applies).
**Files scanned:** `control/keyboard_joint_control.py`, `control/record_episode.py`, `scripts/calibration_utils.py` (all read in full this session; each under 250 lines, no partial reads needed).
**Pattern extraction date:** 2026-09-20
