"""The safety gate every candidate action must pass through before it can
reach the real SO-ARM101's servo bus (VLAHW-02).

Layered on top of (not a replacement for) LeRobot's own built-in
`max_relative_target`/`ensure_safe_goal_position` clamp -- this module adds
the checks that clamp does not cover: absolute joint-limit clamping
against this project's own live calibration, NaN/inf rejection (holding
current position instead), a velocity-aware (dt-scaled) displacement cap
in addition to the flat per-step cap, and a stale-observation override
that holds the entire action when the paired observation is too old to
trust.

Every candidate action -- whether from a scripted source, a network
bridge, or a bug anywhere upstream -- is treated as untrusted input
regardless of origin (see 11-01-PLAN.md's threat model, T-11-01).
"""

import math

from vla_bridge import action_contract

# Conservative starting per-step caps (degrees for the 5 arm joints,
# percentage-points for gripper). [ASSUMED] tunable defaults per
# 11-RESEARCH.md's Assumption A3 -- to be exercised and tuned during Plan
# 11-02's dry run.
MAX_RELATIVE_TARGET_DEG = {
    "shoulder_pan": 5.0,
    "shoulder_lift": 5.0,
    "elbow_flex": 5.0,
    "wrist_flex": 5.0,
    "wrist_roll": 5.0,
    "gripper": 15.0,
}

# A second, dt-aware cap (percentage-points/second for gripper) applied in
# addition to the flat per-step cap above, for the case where the interval
# between two actions is unexpectedly short.
MAX_VELOCITY_DEG_PER_S = {
    "shoulder_pan": 30.0,
    "shoulder_lift": 30.0,
    "elbow_flex": 30.0,
    "wrist_flex": 30.0,
    "wrist_roll": 30.0,
    "gripper": 50.0,
}

# An action paired with an observation older than this (seconds) is
# rejected wholesale -- the entire action is overridden with current_state
# (hold position on every joint).
STALE_OBSERVATION_S = 1.0

# A bridge-returned action itself older than this (seconds), covering
# network round-trip staleness -- used by a later plan's bridge path;
# defined here so the constant lives in one place.
STALE_ACTION_S = 3.0


def validate_action(
    raw_action: dict[str, float],
    current_state: dict[str, float],
    joint_limits_deg: dict[str, tuple[float, float]],
    obs_age_s: float = 0.0,
    prev_action: dict[str, float] | None = None,
    dt_s: float = 1.0,
) -> tuple[dict[str, float], list[str]]:
    """Validate a candidate action before it can reach the servo bus.

    Never raises on NaN/inf/out-of-range/stale input -- always returns a
    finite, in-bounds dict plus a human-readable list of what was
    clamped/rejected.
    """
    flags: list[str] = []
    safe_action: dict[str, float] = {}

    for joint in action_contract.JOINT_ORDER:
        if joint not in raw_action:
            continue

        value = raw_action[joint]

        # 1. NaN/inf/None -> hold current_state, flag, do not propagate.
        if value is None or not math.isfinite(value):
            flags.append(f"{joint}: rejected non-finite value ({value!r}), holding current state")
            safe_action[joint] = current_state[joint]
            continue

        result = value

        # 2. Clamp to the absolute joint-limit envelope.
        lo, hi = joint_limits_deg[joint]
        clamped = max(lo, min(hi, result))
        if clamped != result:
            flags.append(f"{joint}: clamped {result} -> {clamped} (limit {lo}/{hi})")
        result = clamped

        # 3. Clamp further to the flat per-step displacement cap.
        max_step = MAX_RELATIVE_TARGET_DEG[joint]
        current = current_state[joint]
        step_lo, step_hi = current - max_step, current + max_step
        step_clamped = max(step_lo, min(step_hi, result))
        if step_clamped != result:
            flags.append(
                f"{joint}: clamped {result} -> {step_clamped} "
                f"(max per-step displacement {max_step} from current_state {current})"
            )
        result = step_clamped

        # 4. If prev_action is given, clamp further to the velocity-implied bound.
        if prev_action is not None and joint in prev_action:
            max_velocity_step = MAX_VELOCITY_DEG_PER_S[joint] * dt_s
            prev = prev_action[joint]
            vel_lo, vel_hi = prev - max_velocity_step, prev + max_velocity_step
            vel_clamped = max(vel_lo, min(vel_hi, result))
            if vel_clamped != result:
                flags.append(
                    f"{joint}: clamped {result} -> {vel_clamped} "
                    f"(max velocity {MAX_VELOCITY_DEG_PER_S[joint]}/s over dt_s={dt_s} "
                    f"from prev_action {prev})"
                )
            result = vel_clamped

        safe_action[joint] = result

    # After the per-joint loop: stale observation overrides everything,
    # discarding all per-joint results computed above.
    if obs_age_s > STALE_OBSERVATION_S:
        return dict(current_state), [
            f"stale observation ({obs_age_s}s > {STALE_OBSERVATION_S}s), holding all joints"
        ]

    return safe_action, flags


__all__ = [
    "MAX_RELATIVE_TARGET_DEG",
    "MAX_VELOCITY_DEG_PER_S",
    "STALE_OBSERVATION_S",
    "STALE_ACTION_S",
    "validate_action",
]
