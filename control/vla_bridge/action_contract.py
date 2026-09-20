"""The real, verified SO-101 follower action contract (VLAHW-01).

This module documents -- and pins with automated tests -- the actual live
action contract `control/`'s real hardware bridge must speak, resolved
directly from installed `lerobot` 0.6.1 source, not from stale in-repo
comments:

- `SOFollowerConfig.use_degrees` defaults to `True` and is never overridden
  anywhere in `control/` (confirmed by reading
  `control/.venv/lib/python3.12/site-packages/lerobot/robots/so_follower/config_so_follower.py`
  this session) -- so the 5 arm joints are normalized in **DEGREES**
  (`MotorNormMode.DEGREES`), symmetric about each joint's calibrated
  midpoint.
- `gripper` is hardcoded to `MotorNormMode.RANGE_0_100` regardless of
  `use_degrees` -- it is a separate, always-percent motor mode in
  `so_follower.py`'s constructor.
- Actions are **absolute** `{joint}.pos` targets, never deltas -- deltas
  only exist internally, inside LeRobot's own `max_relative_target` clamp.

Mirrors `scripts/calibration_utils.py`'s tick->radian formula (same
`MotorsBus._normalize()` DEGREES-branch math) but stops at degrees instead
of converting to radians, since the safety validator that consumes this
module's output operates on the real robot's native degree-mode action
space, not the URDF's radians.
"""

import json
from pathlib import Path

# 6 DOF, arm joints first (in servo-chain order), gripper last -- matches
# so_follower.py's `_motors_ft` dict-insertion order.
JOINT_ORDER = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

# The real, per-joint unit each JOINT_ORDER entry is expressed in when sent
# to/read from `SO101Follower.send_action()`/`get_observation()`.
ACTION_UNITS = {
    "shoulder_pan": "degrees",
    "shoulder_lift": "degrees",
    "elbow_flex": "degrees",
    "wrist_flex": "degrees",
    "wrist_roll": "degrees",
    "gripper": "percent_0_100",
}

# `send_action()` takes absolute `{joint}.pos` targets, never deltas --
# deltas only exist internally inside LeRobot's own `max_relative_target`
# clamp (see `so_follower.py`'s `send_action()`/`ensure_safe_goal_position`).
ACTION_MODE = "absolute"

# Same path pattern as `scripts/calibration_utils.py`'s `CALIBRATION_PATH` --
# LeRobot writes/reads calibration from its local cache, keyed to the
# current follower's device name.
CALIBRATION_PATH = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "lerobot"
    / "calibration"
    / "robots"
    / "so_follower"
    / "soarm_follower_02.json"
)

# The 4 arm joints whose safety-validator limits are derived from live
# calibration ticks. `wrist_roll`/`gripper` are excluded for the same
# reasons `scripts/calibration_utils.py` excludes them: `wrist_roll`'s
# calibration range is a full-turn (0-4095) placeholder, never a real
# measured limit; `gripper`'s calibration entry describes a percent-mode
# normalization, not a distance/angle range.
JOINTS_FROM_CALIBRATION = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex")

# URDF mechanical-limit value (Phase 10) -- a fixed constant, never derived
# from calibration ticks, because `wrist_roll`'s calibration range is a
# full-turn placeholder (see JOINTS_FROM_CALIBRATION docstring above).
WRIST_ROLL_LIMIT_DEG = (-157.21, 162.79)

# RANGE_0_100 percent-open -- NOT meters. Distinct from Phase 10's
# simulation-side 0.036 m prismatic-joint value; the real robot's
# send_action()/get_observation() never sees meters for the gripper.
GRIPPER_LIMIT_PCT = (0.0, 100.0)


def calibration_ticks_to_degrees(
    range_min: int, range_max: int, max_res: int = 4095
) -> tuple[float, float]:
    """Convert a LeRobot calibration tick range to a symmetric degree range.

    Mirrors `scripts/calibration_utils.py::calibration_ticks_to_radians()`'s
    formula exactly (itself mirroring LeRobot's `MotorsBus._normalize()`
    DEGREES branch), stopping before the final `math.radians()` conversion
    that module applies -- this module's consumers (the safety validator)
    operate directly in degrees, matching the real robot's native DEGREES
    mode.
    """
    half_ticks = (range_max - range_min) / 2
    half_deg = half_ticks * 360 / max_res
    return -half_deg, half_deg


def load_joint_limits_deg(
    calibration_path: Path = CALIBRATION_PATH,
) -> dict[str, tuple[float, float]]:
    """Load per-joint degree-mode safety limits from a live calibration file.

    Computes the 4 calibration-derived joints (`JOINTS_FROM_CALIBRATION`)
    via `calibration_ticks_to_degrees()`, then merges in the two fixed
    constants (`wrist_roll`, `gripper`) which must never be derived from
    calibration ticks (see their module-level docstrings above).
    """
    if not calibration_path.exists():
        raise FileNotFoundError(
            f"Calibration file not found at expected path: {calibration_path}\n"
            "This should be the current follower's live LeRobot calibration "
            "cache file. Check control/COMMANDS.md's hardware device table -- "
            "the device ID (e.g. 'soarm_follower_02') should be stable even "
            "if the USB port suffix has changed on replug. If the device ID "
            "itself has changed, update CALIBRATION_PATH in "
            "control/vla_bridge/action_contract.py to match."
        )

    with open(calibration_path) as f:
        calibration = json.load(f)

    limits: dict[str, tuple[float, float]] = {}
    for joint in JOINTS_FROM_CALIBRATION:
        entry = calibration[joint]
        limits[joint] = calibration_ticks_to_degrees(entry["range_min"], entry["range_max"])

    limits["wrist_roll"] = WRIST_ROLL_LIMIT_DEG
    limits["gripper"] = GRIPPER_LIMIT_PCT

    return limits


__all__ = [
    "JOINT_ORDER",
    "ACTION_UNITS",
    "ACTION_MODE",
    "CALIBRATION_PATH",
    "JOINTS_FROM_CALIBRATION",
    "WRIST_ROLL_LIMIT_DEG",
    "GRIPPER_LIMIT_PCT",
    "calibration_ticks_to_degrees",
    "load_joint_limits_deg",
]


if __name__ == "__main__":
    limits = load_joint_limits_deg()
    for joint in JOINT_ORDER:
        lo, hi = limits[joint]
        print(f"{joint}: ({lo:.2f}, {hi:.2f}) {ACTION_UNITS[joint]}")
