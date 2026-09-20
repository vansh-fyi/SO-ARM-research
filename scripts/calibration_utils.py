"""LeRobot calibration tick->radian conversion for the SO-ARM101 follower.

Mirrors LeRobot's own ``MotorsBus._normalize()`` DEGREES-mode formula
(source read directly this session:
``control/.venv/lib/python3.12/site-packages/lerobot/motors/motors_bus.py``,
lines ~854-911) so joint limits hand-applied to `robot.xml` are consistent
with what `control/`'s live robot object would compute for the same
calibration data, not an independently-invented formula that merely looks
similar.

Reused as the single source of truth by:
  - This module's own `main()` CLI (prints all derived values for manual
    application to `robot.xml`, Phase 10 Plan 10-01 Task 2)
  - Phase 10 Plan 10-04's `test_joint_limits_match_calibration` regression
    test (re-derives the same values to assert `robot.xml` still matches)

See Phase 10 CONTEXT.md D-05/D-06 and RESEARCH.md Pattern 2/Pitfall 1 for
the design rationale (why `wrist_roll` and `gripper` are excluded below).
"""

import json
import math
from pathlib import Path

# Per D-05: no calibration JSON lives in this repo -- LeRobot writes/reads it
# from its local cache, keyed to the current follower's device name
# (`soarm_follower_02`, per control/COMMANDS.md's hardware table). This is
# the actual file `control/` presently uses to drive the real robot; the
# USB port suffix can change on replug, but the device ID/calibration
# filename should not.
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

# The 4 arm joints TWIN-05 derives limits for from live calibration.
# `wrist_roll` is excluded: LeRobot's calibrate() hardcodes its calibration
# range to 0-4095 (a full-turn placeholder -- the real servo spins
# continuously past a mechanical stop check and is never actually measured),
# per D-06/Pitfall 1. Feeding that placeholder through this formula would
# silently widen wrist_roll to an unrestricted full turn; robot.xml's
# existing mechanical-limit-derived value is kept instead.
# `gripper` is excluded: its calibration entry describes the real servo's
# rotation normalized into a RANGE_0_100 percent-open representation, not a
# linear-distance measurement -- it has no defined mapping to the
# roboninecom 84mm parallel-jaw's 0-0.042m prismatic stroke (a completely
# different aftermarket mechanism). soarm_gripper.xml's existing prismatic
# limits use the user-approved 36 mm-per-jaw cap (2026-09-20), not the full
# hardware stroke; retain the MJCF limits rather than deriving them from ticks.
JOINTS_FROM_CALIBRATION = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex")


def calibration_ticks_to_radians(
    range_min: int, range_max: int, max_res: int = 4095
) -> tuple[float, float]:
    """Convert a LeRobot calibration tick range to a symmetric radian range.

    Mirrors LeRobot's MotorsBus._normalize() DEGREES branch exactly (source:
    control/.venv/lib/python3.12/site-packages/lerobot/motors/motors_bus.py,
    lines 874-877): ``mid = (min_+max_)/2``, ``max_res =
    model_resolution_table[model] - 1`` (4096-1=4095 for the STS3215 servos
    used here, per control/.venv/.../lerobot/motors/feetech/tables.py line
    190), ``normalized = (val - mid) * 360 / max_res``. Evaluating this at
    ``val=range_max`` gives the positive half-range in degrees; the negative
    half-range is symmetric by construction since `range_min`/`range_max`
    are equidistant from `mid`.

    Returns ``(-half_rad, +half_rad)`` where ``half_ticks =
    (range_max-range_min)/2``, ``half_deg = half_ticks*360/max_res``,
    ``half_rad = math.radians(half_deg)``. This symmetric-around-midpoint
    range is also MuJoCo's qpos=0 home pose per robot.xml's header comment
    ("no keyframe: qpos=0 IS the documented new-calib home pose").
    """
    half_ticks = (range_max - range_min) / 2
    half_deg = half_ticks * 360 / max_res
    half_rad = math.radians(half_deg)
    return -half_rad, half_rad


def main() -> None:
    if not CALIBRATION_PATH.exists():
        raise SystemExit(
            f"Calibration file not found at expected path: {CALIBRATION_PATH}\n"
            "This should be the current follower's live LeRobot calibration "
            "cache file. Check control/COMMANDS.md's hardware device table -- "
            "the device ID (e.g. 'soarm_follower_02') should be stable even "
            "if the USB port suffix has changed on replug. If the device ID "
            "itself has changed, update CALIBRATION_PATH in "
            "scripts/calibration_utils.py to match."
        )

    with open(CALIBRATION_PATH) as f:
        calibration = json.load(f)

    for joint in JOINTS_FROM_CALIBRATION:
        entry = calibration[joint]
        rm, rM = entry["range_min"], entry["range_max"]
        lo, hi = calibration_ticks_to_radians(rm, rM)
        print(f"{joint}: range_min={rm} range_max={rM} -> ({lo:.6f}, {hi:.6f}) rad")

    print(
        "wrist_roll: EXCLUDED (calibration range 0-4095 is a full-turn "
        "placeholder per Pitfall 1, not a real limit -- robot.xml's existing "
        "mechanical value is kept)"
    )
    print("gripper: EXCLUDED (percent-mode calibration, not a URDF prismatic range)")


if __name__ == "__main__":
    main()
