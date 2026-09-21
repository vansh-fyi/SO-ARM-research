"""Real-hardware SO-ARM101 VLA episode harness (VLAHW-02, VLAHW-03).

Wires together Plan 11-01's safety validator + action contract, Task 1's
JSON Lines I/O logger, real camera capture, and a software keyboard-interrupt
e-stop (D-03 -- no new physical hardware), driving the REAL robot end-to-end
via a pluggable `ActionSource` interface.

This script's default `ScriptedActionSource` is an explicit stand-in for the
not-yet-built Colab/SmolVLA bridge (Plan 11-03/11-04 replace it) -- used here
to prove the safety-critical plumbing (comms retry, logging durability,
e-stop) works against real hardware before adding network complexity. The
`ActionSource` seam is the exact interface Plan 11-04 swaps a real
`BridgeActionSource` into; nothing else in this script should need to change.

Reuses (does not reimplement):
  - `keyboard_joint_control.read_positions` / `move_to_positions` (comms
    retry/backoff, return-to-start P-control)
  - `vla_bridge.action_contract.load_joint_limits_deg`
  - `vla_bridge.safety_validator.validate_action`
  - `vla_bridge.io_logger.IOLogger`

Usage:
    python run_vla_episode.py PORT ROBOT_ID --out outputs/dry_run_001 \
        --max-steps 20 [--camera 0 --camera 1] [--instruction "Pick up the red cube"]
"""

import argparse
import json
import time
from pathlib import Path
from typing import Protocol

import cv2
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SO101Follower

from keyboard_joint_control import move_to_positions, read_positions
from vla_bridge import action_contract, safety_validator
from vla_bridge.io_logger import IOLogger

# Camera indices can shift on USB replug -- see control/COMMANDS.md's own
# caveat. This mapping is fixed for the default 2-camera rig (IMX335 wrist @
# index 0, AR0144 stereo overhead @ index 1); override at the CLI if indices
# have shifted.
DEFAULT_CAMERA_NAMES = {0: "wrist", 1: "overhead"}


class ActionSource(Protocol):
    """Pluggable seam for whatever produces the next candidate action.

    Plan 11-04 supplies the real `BridgeActionSource` (Colab-hosted SmolVLA,
    relayed over the async_inference gRPC bridge) implementing this exact
    interface -- everything else in this script's control loop is unaware of
    where the action came from.
    """

    def get_action(
        self, joint_state: dict[str, float], instruction: str
    ) -> tuple[dict[str, float], str]:
        """Return (raw_action, model_version) for the given observation."""
        ...


class ScriptedActionSource:
    """Explicit stand-in for the not-yet-built Colab/SmolVLA bridge.

    Returns the joint_state unchanged except it toggles `gripper` between
    0.0 and 100.0 every 30 steps, proving the write path is genuinely live
    (not a frozen no-op) while exercising the full safety-validator +
    logging + e-stop plumbing against the real robot today. This is NOT a
    real VLA and is not intended to remain in place once Plan 11-04 wires in
    `BridgeActionSource` through this same `ActionSource` seam.
    """

    def __init__(self):
        self._step = 0

    def get_action(
        self, joint_state: dict[str, float], instruction: str
    ) -> tuple[dict[str, float], str]:
        action = dict(joint_state)
        if (self._step // 30) % 2 == 0:
            action["gripper"] = 0.0
        else:
            action["gripper"] = 100.0
        self._step += 1
        return action, "scripted-dry-run-v1"


def _probe_camera_indices() -> list[int]:
    print("No --camera given. Probing indices 0-5...")
    found = []
    for idx in range(6):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            found.append(idx)
        cap.release()
    print(f"Found cameras: {found}")
    return found


def _write_termination(out_dir: Path, reason: str, steps_completed: int) -> None:
    (out_dir / "termination.json").write_text(
        json.dumps({"reason": reason, "steps_completed": steps_completed})
    )


def run_episode(
    robot,
    caps: dict[int, "cv2.VideoCapture"],
    camera_names: dict[int, str],
    io_logger: IOLogger,
    action_source: ActionSource,
    instruction: str,
    max_steps: int,
    control_hz: float,
    joint_limits_deg: dict[str, tuple[float, float]],
) -> None:
    """Main control loop -- one iteration per step, up to `max_steps`.

    Every write path to `robot.send_action()` passes through
    `safety_validator.validate_action()` first (VLAHW-02's core
    acceptance criterion). The e-stop (KeyboardInterrupt) always triggers a
    return-to-start move BEFORE disconnect, in every exit path (max-steps,
    interrupt, or any other exception via the caller's `finally`).

    `joint_limits_deg` is loaded once at startup by the caller (`main()`,
    via `action_contract.load_joint_limits_deg()`) and passed in here rather
    than loaded internally -- keeps this function hermetic/testable against
    a mocked calibration file, not this machine's real hardware cache.
    """
    control_period = 1.0 / control_hz
    start_positions = read_positions(robot)
    last_sent_action: dict[str, float] | None = None
    steps_completed = 0
    reason = "max_steps_reached"

    try:
        for i in range(max_steps):
            current = read_positions(robot)
            raw_action, model_version = action_source.get_action(current, instruction)
            validated_action, flags = safety_validator.validate_action(
                raw_action,
                current,
                joint_limits_deg,
                obs_age_s=0.0,
                prev_action=last_sent_action,
                dt_s=1.0 / control_hz,
            )
            try:
                robot.send_action({f"{j}.pos": v for j, v in validated_action.items()})
            except (ConnectionError, RuntimeError) as e:
                print(f"Write failed, skipping this tick: {e}")

            camera_frames = {
                name: io_logger.capture_camera_frame(caps[idx], name, step=i)
                for idx, name in camera_names.items()
                if idx in caps
            }

            io_logger.write_step(
                step=i,
                instruction=instruction,
                camera_frames=camera_frames,
                joint_state=current,
                raw_model_output=raw_action,
                validated_action=validated_action,
                validator_flags=flags,
                executed_action=validated_action,
                latency_ms={"observation_to_action": 0, "action_to_execution": 0},
                model_version=model_version,
            )
            last_sent_action = validated_action
            steps_completed = i + 1
            time.sleep(control_period)
    except KeyboardInterrupt:
        print("Interrupted -- returning to start position...")
        reason = "keyboard_interrupt"
    except Exception:
        reason = "exception"
        raise
    finally:
        # E-stop (D-03): return-to-start strictly before disconnect, in every
        # exit path -- max-steps, KeyboardInterrupt, or any other exception
        # (disconnect() itself happens in main()'s outer try/finally).
        move_to_positions(robot, start_positions, kp=0.2, control_freq=30, max_seconds=5.0)
        _write_termination(io_logger.out_dir, reason, steps_completed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("port")
    parser.add_argument("robot_id")
    parser.add_argument(
        "--camera", type=int, action="append", default=[], help="repeatable, e.g. --camera 0 --camera 1"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--instruction", type=str, default="Pick up the red cube")
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--control-hz", type=float, default=2.0)
    args = parser.parse_args()

    if not args.camera:
        args.camera = _probe_camera_indices()

    caps = {}
    for idx in args.camera:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            caps[idx] = cap
        else:
            print(f"WARNING: camera index {idx} did not open, skipping")

    robot = SO101Follower(
        SOFollowerRobotConfig(
            port=args.port,
            id=args.robot_id,
            max_relative_target=safety_validator.MAX_RELATIVE_TARGET_DEG,
        )
    )
    robot.connect(calibrate=False)
    with robot.bus.torque_disabled():
        robot.bus.write_calibration(robot.calibration)

    action_source = ScriptedActionSource()
    joint_limits_deg = action_contract.load_joint_limits_deg()

    try:
        with IOLogger(args.out, DEFAULT_CAMERA_NAMES) as io_logger:
            run_episode(
                robot,
                caps,
                DEFAULT_CAMERA_NAMES,
                io_logger,
                action_source,
                args.instruction,
                args.max_steps,
                args.control_hz,
                joint_limits_deg,
            )
    finally:
        for cap in caps.values():
            cap.release()
        robot.disconnect()


if __name__ == "__main__":
    main()
