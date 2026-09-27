"""Motion-smoothness tuning tool for `run_vla_episode.py`'s `--execution-hz` waypoint
interpolation (Phase 12, live-hardware tuning aid).

Exercises `run_vla_episode.py`'s REAL `run_episode()` control loop
(interpolation, safety validation, e-stop) against the real SO-ARM101, using a
continuously-oscillating scripted target instead of the Colab bridge -- no
tunnel/PolicyServer needed, fast iterate loop for finding a good
`--execution-hz`.

Unlike `run_vla_episode.py`'s own `ScriptedActionSource` (which only toggles
the gripper every 30 steps and leaves all other joints frozen -- fine for
proving the e-stop/comms plumbing, useless for judging motion smoothness),
`SweepActionSource` below continuously moves multiple joints in staggered
sine waves every single tick, giving the interpolation code a real,
constantly-changing target to smooth between.

Usage (from control/, with .venv active):
    python tune_execution_hz.py --execution-hz 100 --max-steps 40

Ctrl+C at any time triggers the same e-stop return-to-start path as the real
episode script.
"""

import argparse
import math
from pathlib import Path

from keyboard_joint_control import read_positions
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SO101Follower
from run_vla_episode import _load_device_map, run_episode
from vla_bridge import action_contract, safety_validator
from vla_bridge.io_logger import IOLogger

# (joint, amplitude_deg, phase_offset_fraction_of_period) -- phase offsets are staggered
# so joints don't all peak/trough in lockstep, giving a real coordinated-reach feel
# instead of every joint just wagging in sync.
DEFAULT_JOINTS = [
    ("shoulder_pan", 15.0, 0.0),
    ("shoulder_lift", 12.0, 0.25),
    ("elbow_flex", 12.0, 0.5),
    ("wrist_flex", 10.0, 0.75),
    ("wrist_roll", 8.0, 0.15),
]


class SweepActionSource:
    """Continuously oscillates MULTIPLE joints in sine waves (staggered phases), one
    control tick at a time -- gives run_episode()'s waypoint interpolation a real,
    constantly-changing, multi-joint target to smooth between every single tick --
    much closer to what a real coordinated reach/pick motion looks like than a
    single-joint wag.
    """

    def __init__(self, start_positions, joints=None, period_ticks=6):
        self._start = dict(start_positions)
        self._joints = joints if joints is not None else DEFAULT_JOINTS
        self._period = period_ticks
        self._tick = 0

    def get_action(self, joint_state, instruction):
        action = dict(joint_state)
        for joint, amplitude_deg, phase_frac in self._joints:
            phase = self._tick / self._period + phase_frac
            offset = amplitude_deg * math.sin(2 * math.pi * phase)
            action[joint] = self._start[joint] + offset
        self._tick += 1
        return action, f"sweep-tuning-tick-{self._tick}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-hz", type=float, default=100.0,
                         help="Value to tune -- current run_vla_episode.py default is 100.0, "
                         "tuned live during Phase 12 UAT (20 was jerky, 60 better, 100 best).")
    parser.add_argument("--control-hz", type=float, default=2.0)
    parser.add_argument("--period-ticks", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--single-joint", type=str, default=None,
                         help="If given, sweep only this one joint (e.g. shoulder_pan) instead of the default multi-joint coordinated sweep")
    parser.add_argument("--amplitude-deg", type=float, default=15.0,
                         help="Amplitude for --single-joint mode only; ignored in multi-joint mode")
    args = parser.parse_args()

    device_map = _load_device_map()
    if device_map is None or device_map.get("follower") is None:
        raise SystemExit("device_map.json missing/no follower entry -- run detect_devices.py first")

    port = device_map["follower"]["port"]
    robot_id = device_map["follower"]["id"]
    print(f"Connecting to {robot_id} on {port}...")

    robot_config = SOFollowerRobotConfig(
        port=port, id=robot_id, max_relative_target=safety_validator.MAX_RELATIVE_TARGET_DEG
    )
    robot = SO101Follower(robot_config)
    robot.connect()

    joint_limits_deg = action_contract.load_joint_limits_deg()
    start_positions = read_positions(robot)
    print(f"Start positions: {start_positions}")

    if args.single_joint:
        joints = [(args.single_joint, args.amplitude_deg, 0.0)]
        joint_desc = f"{args.single_joint} only, amplitude={args.amplitude_deg}deg"
    else:
        joints = DEFAULT_JOINTS
        joint_desc = f"multi-joint coordinated sweep ({', '.join(j for j, _, _ in joints)})"

    action_source = SweepActionSource(start_positions, joints=joints, period_ticks=args.period_ticks)

    out_dir = Path(f"/tmp/motion_tune_hz{int(args.execution_hz)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Tuning run: execution_hz={args.execution_hz}, control_hz={args.control_hz}, "
          f"{joint_desc}, {args.max_steps} steps")
    print(f"(No cameras recorded -- pure motion tuning. Logs -> {out_dir})")

    try:
        with IOLogger(out_dir, {}) as io_logger:
            run_episode(
                robot=robot,
                caps={},
                camera_names={},
                io_logger=io_logger,
                action_source=action_source,
                instruction="motion-tuning sweep (not a real task)",
                max_steps=args.max_steps,
                control_hz=args.control_hz,
                joint_limits_deg=joint_limits_deg,
                stereo_camera=None,
                execution_hz=args.execution_hz,
            )
    finally:
        robot.disconnect()
        print("Disconnected.")


if __name__ == "__main__":
    main()
