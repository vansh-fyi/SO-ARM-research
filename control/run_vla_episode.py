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
from vla_bridge.robot_client import BridgeActionSource, connect_bridge

# Camera indices can shift on USB replug -- see control/COMMANDS.md's own
# caveat. Semantic names are assigned POSITIONALLY from the order `--camera`
# indices are given (or ascending probe order, if omitted) via
# `_build_camera_names()` below -- NOT from a fixed index->name table -- so
# `--camera <idx1> --camera <idx2>` genuinely relabels which physical device
# is "wrist" vs "overhead" when USB enumeration order shifts. This was a real
# bug found in Plan 11-05 Task 1: a previous fixed `{0: "wrist", 1:
# "overhead"}` table silently mislabeled recorded camera output whenever the
# physical index assignment didn't match the table, even though `--camera`
# was passed in the "corrected" order (the table ignored --camera's order
# entirely). Default assumption if `--camera` is omitted: index 0 = wrist,
# index 1 = overhead -- reverify with `ls /dev/cu.usbmodem*`-style physical
# checks (e.g. cover-the-lens test) before trusting it, since it has already
# been observed to invert on this rig.
CAMERA_SEMANTIC_NAMES = ["wrist", "overhead"]


def _build_camera_names(camera_indices: list[int]) -> dict[int, str]:
    """Maps physical camera indices to semantic names positionally: the Nth
    index in `camera_indices` (CLI `--camera` order, or ascending probe
    order) gets the Nth name in `CAMERA_SEMANTIC_NAMES`. Indices beyond the
    known semantic names are left unlabeled (still present in `caps` for
    direct iteration, just not recorded under a semantic name)."""
    return dict(zip(camera_indices, CAMERA_SEMANTIC_NAMES))


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
    parser.add_argument(
        "--server-address",
        type=str,
        default=None,
        help="Colab PolicyServer bridge address (host:port). When given, drives the robot "
        "with a real Colab-hosted VLA via BridgeActionSource instead of ScriptedActionSource.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="HF Hub SmolVLA checkpoint id. Required when --server-address is given.",
    )
    parser.add_argument(
        "--stereo-camera-index",
        type=int,
        default=1,
        help="cv2 index of the AR0144 stereo camera for the --server-address bridge path's "
        "camera2/camera3 split (vla_bridge.robot_client.connect_bridge's stereo_camera_index). "
        "Independent of --camera (which only controls this script's own IOLogger recording "
        "caps) -- verify which physical index is actually the AR0144 before a live run; it has "
        "been observed to shift (Plan 11-05 Task 1).",
    )
    args = parser.parse_args()

    if args.server_address and not args.checkpoint:
        parser.error("--checkpoint is required when --server-address is given")
    if args.checkpoint and not args.server_address:
        parser.error("--server-address is required when --checkpoint is given")

    if not args.camera:
        args.camera = _probe_camera_indices()

    camera_names = _build_camera_names(args.camera)

    caps = {}
    for idx in args.camera:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            caps[idx] = cap
        else:
            print(f"WARNING: camera index {idx} did not open, skipping")

    joint_limits_deg = action_contract.load_joint_limits_deg()

    if args.server_address:
        # Real, network-bridged SmolVLA path (Plan 11-04). `connect_bridge()`
        # constructs AND connects the physical robot internally -- its
        # returned client's `.robot` is the one true robot handle; do NOT
        # also construct a separate local SO101Follower here.
        robot_config = SOFollowerRobotConfig(
            port=args.port,
            id=args.robot_id,
            max_relative_target=safety_validator.MAX_RELATIVE_TARGET_DEG,
        )
        client = connect_bridge(
            args.server_address,
            args.checkpoint,
            robot_config=robot_config,
            task=args.instruction,
            policy_device="cuda",
            stereo_camera_index=args.stereo_camera_index,
        )
        if client is None:
            print(f"Bridge unreachable at {args.server_address}, aborting before touching the robot.")
            args.out.mkdir(parents=True, exist_ok=True)
            _write_termination(args.out, "bridge_unreachable", 0)
            for cap in caps.values():
                cap.release()
            return
        robot = client.robot
        action_source = BridgeActionSource(client, args.checkpoint, joint_limits_deg)
    else:
        # Plan 11-02's scripted dry-run path, unchanged.
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

    try:
        with IOLogger(args.out, camera_names) as io_logger:
            run_episode(
                robot,
                caps,
                camera_names,
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
        if args.server_address:
            client.stop()
        else:
            robot.disconnect()


if __name__ == "__main__":
    main()
