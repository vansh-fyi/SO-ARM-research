"""Tests for control/vla_bridge/safety_validator.py (VLAHW-02).

Covers clamp, NaN/inf rejection, per-step displacement cap, velocity cap,
stale-observation override, and the gripper-units regression (Pitfall 4:
gripper is percent, not meters).
"""

import math

import pytest

from vla_bridge.action_contract import load_joint_limits_deg
from vla_bridge.safety_validator import (
    MAX_RELATIVE_TARGET_DEG,
    MAX_VELOCITY_DEG_PER_S,
    STALE_OBSERVATION_S,
    validate_action,
)


@pytest.fixture
def joint_limits_deg(mock_calibration_file):
    return load_joint_limits_deg(mock_calibration_file)


def test_out_of_range_value_clamped_to_joint_limit(joint_limits_deg):
    """shoulder_pan=90.0 is outside its +-70.33 degree limit."""
    raw_action = {"shoulder_pan": 90.0}
    current_state = {"shoulder_pan": 70.0}

    safe, flags = validate_action(raw_action, current_state, joint_limits_deg)

    assert safe["shoulder_pan"] == pytest.approx(70.33, abs=0.01)
    assert any("clamped" in f for f in flags)


def test_nan_value_holds_current_state_and_does_not_propagate(joint_limits_deg):
    raw_action = {"shoulder_pan": float("nan")}
    current_state = {"shoulder_pan": 12.5}

    safe, flags = validate_action(raw_action, current_state, joint_limits_deg)

    assert safe["shoulder_pan"] == 12.5
    assert not math.isnan(safe["shoulder_pan"])
    assert any("rejected non-finite" in f for f in flags)


def test_infinite_value_holds_current_state_and_does_not_propagate(joint_limits_deg):
    raw_action = {"shoulder_pan": float("inf")}
    current_state = {"shoulder_pan": 12.5}

    safe, flags = validate_action(raw_action, current_state, joint_limits_deg)

    assert safe["shoulder_pan"] == 12.5
    assert math.isfinite(safe["shoulder_pan"])
    assert any("rejected non-finite" in f for f in flags)


def test_large_step_clamped_to_max_relative_target_default(joint_limits_deg):
    """A large jump on shoulder_pan in one step is clamped to a step from
    current_state bounded by MAX_RELATIVE_TARGET_DEG["shoulder_pan"]."""
    raw_action = {"shoulder_pan": 60.0}
    current_state = {"shoulder_pan": 0.0}

    safe, flags = validate_action(raw_action, current_state, joint_limits_deg)

    assert safe["shoulder_pan"] == pytest.approx(
        current_state["shoulder_pan"] + MAX_RELATIVE_TARGET_DEG["shoulder_pan"]
    )
    assert any("max per-step displacement" in f for f in flags)


def test_velocity_cap_clamps_further_given_short_dt(joint_limits_deg):
    """Given dt_s=0.01 and a requested step implying more than
    MAX_VELOCITY_DEG_PER_S["shoulder_pan"] deg/s relative to prev_action,
    the result is clamped to the velocity-implied bound."""
    raw_action = {"shoulder_pan": 3.0}
    current_state = {"shoulder_pan": 0.0}
    prev_action = {"shoulder_pan": 0.0}

    safe, flags = validate_action(
        raw_action,
        current_state,
        joint_limits_deg,
        prev_action=prev_action,
        dt_s=0.01,
    )

    assert safe["shoulder_pan"] == pytest.approx(
        MAX_VELOCITY_DEG_PER_S["shoulder_pan"] * 0.01, abs=1e-6
    )
    assert any("max velocity" in f for f in flags)


def test_stale_observation_holds_all_joints(joint_limits_deg):
    """obs_age_s greater than STALE_OBSERVATION_S causes the ENTIRE action
    to be overridden with current_state, holding position on every joint."""
    raw_action = {"shoulder_pan": 50.0, "gripper": 80.0}
    current_state = {"shoulder_pan": 10.0, "gripper": 20.0}

    safe, flags = validate_action(
        raw_action, current_state, joint_limits_deg, obs_age_s=STALE_OBSERVATION_S + 1.0
    )

    assert safe == current_state
    assert any("stale observation" in f for f in flags)


def test_gripper_percent_value_not_treated_as_meters(joint_limits_deg):
    """A gripper value of 0.05 (which would be an absurd out-of-range value
    if misread as meters) must be treated as a normal in-bounds percent
    value, NOT rejected -- regression test for Pitfall 4."""
    raw_action = {"gripper": 0.05}
    current_state = {"gripper": 0.0}

    safe, flags = validate_action(raw_action, current_state, joint_limits_deg)

    assert 0.0 <= safe["gripper"] <= 100.0
    assert safe["gripper"] == pytest.approx(0.05)
