"""Tests for scripts/calibration_utils.py's LeRobot tick->radian formula.

Mirrors LeRobot's own MotorsBus._normalize() DEGREES branch (source:
control/.venv/lib/python3.12/site-packages/lerobot/motors/motors_bus.py,
lines 854-911) -- these tests pin known-good input/output pairs before any
value derived from `calibration_ticks_to_radians()` is hand-applied to
`robot.xml` (Phase 10 Plan 10-01 Task 2).
"""

import math

import pytest

from calibration_utils import calibration_ticks_to_radians


def test_shoulder_pan_matches_known_calibration():
    """The live soarm_follower_02.json shoulder_pan entry (range_min=1269,
    range_max=2869) must convert to the pre-computed table value in
    10-RESEARCH.md Pattern 2 (+-1.227484 rad)."""
    lo, hi = calibration_ticks_to_radians(1269, 2869)
    assert lo == pytest.approx(-1.227484, abs=1e-5)
    assert hi == pytest.approx(1.227484, abs=1e-5)


def test_full_range_is_full_turn_placeholder():
    """Regression/documentation test for Pitfall 1's exact warning sign: a
    derived range suspiciously close to +-pi is the wrist_roll calibration
    entry's full-turn placeholder (range_min=0, range_max=4095) leaking
    through -- demonstrating why it must NOT be fed through this formula
    and applied to robot.xml's wrist_roll joint."""
    lo, hi = calibration_ticks_to_radians(0, 4095)
    assert lo == pytest.approx(-math.pi, abs=1e-5)
    assert hi == pytest.approx(math.pi, abs=1e-5)
