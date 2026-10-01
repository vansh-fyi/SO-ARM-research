"""`--backend scripted`: the no-network dry-run backend (Plan 11-02's path).

Drives the real robot with `ScriptedActionSource` -- a fixed gripper toggle,
not a VLA -- to prove the safety-validator + logging + e-stop plumbing works
on hardware with no bridge involved.

`ScriptedActionSource` lives here (moved from `run_vla_episode.py` in Plan
14-02) so this module can construct it without importing
`run_vla_episode.py`, which itself imports `vla_bridge.backends` (a circular
import otherwise). `run_vla_episode.py` re-exports it, so
`run_vla_episode.ScriptedActionSource` keeps resolving for every existing
caller.
"""

from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SO101Follower

from vla_bridge import safety_validator
from vla_bridge.backends import BackendHandles, register


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


@register("scripted")
def build(args, camera_names: dict[int, str], joint_limits_deg: dict) -> BackendHandles:
    """Connect the robot directly (no bridge) and pair it with a
    `ScriptedActionSource`. `args` needs `.port` and `.robot_id`.

    Opens no camera of its own, so `run_vla_episode.py` opens every
    `--camera` index itself (`skip_camera_names` is empty).
    """
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
    return BackendHandles(
        action_source=ScriptedActionSource(),
        robot=robot,
        stereo_camera=None,
        skip_camera_names=frozenset(),
    )


__all__ = ["ScriptedActionSource", "build"]
