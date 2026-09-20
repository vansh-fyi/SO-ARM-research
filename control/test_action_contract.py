"""Tests for control/vla_bridge/action_contract.py's documented action
contract (VLAHW-01).

Pins the real, verified contract (DEGREES for the 5 arm joints,
RANGE_0_100 percent for the gripper, absolute targets, 6-DOF joint order)
as automated tests, not tribal knowledge in a comment.
"""

import pytest

from vla_bridge.action_contract import (
    GRIPPER_LIMIT_PCT,
    JOINT_ORDER,
    WRIST_ROLL_LIMIT_DEG,
    calibration_ticks_to_degrees,
    load_joint_limits_deg,
)


def test_joint_order_is_six_dof_arm_then_gripper():
    assert JOINT_ORDER == (
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    )
    assert len(JOINT_ORDER) == 6


def test_calibration_ticks_to_degrees_matches_known_shoulder_pan():
    """Live soarm_follower_02.json shoulder_pan entry (range_min=1269,
    range_max=2869) must convert to (-70.33, 70.33) degrees within 0.01."""
    lo, hi = calibration_ticks_to_degrees(1269, 2869)
    assert lo == pytest.approx(-70.33, abs=0.01)
    assert hi == pytest.approx(70.33, abs=0.01)


def test_calibration_ticks_to_degrees_matches_known_shoulder_lift():
    lo, hi = calibration_ticks_to_degrees(904, 3349)
    assert lo == pytest.approx(-107.47, abs=0.01)
    assert hi == pytest.approx(107.47, abs=0.01)


def test_calibration_ticks_to_degrees_matches_known_elbow_flex():
    lo, hi = calibration_ticks_to_degrees(832, 3044)
    assert lo == pytest.approx(-97.23, abs=0.01)
    assert hi == pytest.approx(97.23, abs=0.01)


def test_calibration_ticks_to_degrees_matches_known_wrist_flex():
    lo, hi = calibration_ticks_to_degrees(899, 3220)
    assert lo == pytest.approx(-102.02, abs=0.01)
    assert hi == pytest.approx(102.02, abs=0.01)


def test_load_joint_limits_deg_uses_fixed_wrist_roll_and_gripper_bounds_not_calibration_derived(
    mock_calibration_file,
):
    """wrist_roll/gripper entries in the calibration file are known
    placeholders (full-turn ticks / percent-mode) -- load_joint_limits_deg()
    must ignore them and always return the fixed constants instead."""
    limits = load_joint_limits_deg(mock_calibration_file)
    assert limits["wrist_roll"] == WRIST_ROLL_LIMIT_DEG
    assert limits["gripper"] == GRIPPER_LIMIT_PCT


def test_load_joint_limits_deg_missing_file_raises_file_not_found_error(tmp_path):
    missing_path = tmp_path / "does_not_exist.json"
    with pytest.raises(FileNotFoundError):
        load_joint_limits_deg(missing_path)
