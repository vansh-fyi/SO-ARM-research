"""Tests for vla_bridge.backends.eef_to_joint (Plan 14-03 Task 1).

The kinematic chain is loaded from the real So-101/So-101.urdf through
ikpy (no robot hardware, no network), so these exercise the actual Phase 10
URDF geometry rather than a fixture.
"""

import math

import numpy as np
import pytest

from vla_bridge.action_contract import GRIPPER_LIMIT_PCT, JOINT_ORDER
from vla_bridge.backends import eef_to_joint

# A comfortably in-range, non-singular arm pose (degrees / gripper percent).
HOME = {
    "shoulder_pan": 10.0,
    "shoulder_lift": -20.0,
    "elbow_flex": 30.0,
    "wrist_flex": 15.0,
    "wrist_roll": 5.0,
    "gripper": 37.0,
}

ZERO_DELTA = {"dx": 0.0, "dy": 0.0, "dz": 0.0, "droll": 0.0, "dpitch": 0.0, "dyaw": 0.0, "gripper": 37.0}


def _delta(**overrides):
    delta = dict(ZERO_DELTA)
    delta.update(overrides)
    return delta


def _assert_joint_dict(result):
    assert tuple(result) == JOINT_ORDER
    for joint, value in result.items():
        assert isinstance(value, float), (joint, value)
        assert math.isfinite(value), (joint, value)


def test_chain_is_loaded_from_the_phase10_urdf_not_hand_derived_constants():
    assert eef_to_joint.URDF_PATH.name == "So-101.urdf"
    assert eef_to_joint.URDF_PATH.parent.name == "So-101"
    assert eef_to_joint.URDF_PATH.exists()
    chain = eef_to_joint._load_chain()
    active = [link.name for link, on in zip(chain.links, chain.active_links_mask) if on]
    assert tuple(active) == JOINT_ORDER[:5]
    # URDF-derived limits, e.g. wrist_roll's asymmetric mechanical range.
    lo, hi = eef_to_joint.urdf_joint_limits_deg()["wrist_roll"]
    assert lo == pytest.approx(math.degrees(-2.7438472969992493))
    assert hi == pytest.approx(math.degrees(2.841206309382605))


def test_forward_kinematics_returns_six_finite_floats():
    pose = eef_to_joint.forward_kinematics(HOME)
    assert len(pose) == 6
    assert all(isinstance(v, float) and math.isfinite(v) for v in pose)


def test_zero_delta_round_trips_to_current_joints():
    result = eef_to_joint.solve_ik(HOME, ZERO_DELTA)
    _assert_joint_dict(result)
    for joint in JOINT_ORDER[:5]:
        assert result[joint] == pytest.approx(HOME[joint], abs=1e-3)


def test_reachable_position_delta_is_tracked_by_forward_kinematics():
    start = eef_to_joint.forward_kinematics(HOME)
    result = eef_to_joint.solve_ik(HOME, _delta(dx=0.01, dz=-0.01))
    end = eef_to_joint.forward_kinematics(result)
    # 5-DOF arm: position is weighted above orientation, so it should track
    # a small reachable translation to within a couple of millimetres.
    assert end[0] - start[0] == pytest.approx(0.01, abs=2e-3)
    assert end[1] - start[1] == pytest.approx(0.0, abs=2e-3)
    assert end[2] - start[2] == pytest.approx(-0.01, abs=2e-3)
    assert any(abs(result[j] - HOME[j]) > 0.1 for j in JOINT_ORDER[:5])


def test_target_reached_from_a_known_reachable_configuration():
    goal_joints = dict(HOME, shoulder_pan=18.0, elbow_flex=22.0, wrist_flex=25.0)
    start = np.array(eef_to_joint.forward_kinematics(HOME))
    goal = np.array(eef_to_joint.forward_kinematics(goal_joints))
    # Express the goal as a position-only delta and check position convergence.
    d = goal[:3] - start[:3]
    result = eef_to_joint.solve_ik(HOME, _delta(dx=d[0], dy=d[1], dz=d[2]), max_iters=200)
    end = np.array(eef_to_joint.forward_kinematics(result))
    assert np.linalg.norm(end[:3] - goal[:3]) < 2e-3


@pytest.mark.parametrize(
    "delta",
    [
        _delta(dx=10.0, dy=-10.0, dz=10.0),
        _delta(droll=50.0, dpitch=-50.0, dyaw=50.0),
        _delta(dx=float("nan")),
        _delta(dy=float("inf"), dyaw=float("-inf")),
        _delta(dx=1e300, dz=-1e300),
    ],
)
def test_output_is_always_finite_and_within_limits_even_for_unreachable_deltas(delta):
    result = eef_to_joint.solve_ik(HOME, delta)
    _assert_joint_dict(result)
    limits = eef_to_joint.urdf_joint_limits_deg()
    for joint in JOINT_ORDER[:5]:
        lo, hi = limits[joint]
        assert lo - 1e-9 <= result[joint] <= hi + 1e-9


def test_non_finite_delta_holds_the_current_arm_pose():
    result = eef_to_joint.solve_ik(HOME, _delta(dx=float("nan"), dz=0.05))
    for joint in JOINT_ORDER[:5]:
        assert result[joint] == pytest.approx(HOME[joint])


def test_output_is_clamped_to_caller_supplied_calibrated_limits():
    tight = {joint: (HOME[joint] - 1.0, HOME[joint] + 1.0) for joint in JOINT_ORDER[:5]}
    tight["gripper"] = GRIPPER_LIMIT_PCT
    result = eef_to_joint.solve_ik(HOME, _delta(dx=0.05, dz=0.05), joint_limits_deg=tight)
    for joint in JOINT_ORDER[:5]:
        lo, hi = tight[joint]
        assert lo - 1e-9 <= result[joint] <= hi + 1e-9


@pytest.mark.parametrize("gripper", [0.0, 42.5, 100.0])
def test_gripper_passes_straight_through(gripper):
    result = eef_to_joint.solve_ik(HOME, _delta(dx=0.01, gripper=gripper))
    assert result["gripper"] == gripper


def test_gripper_is_never_run_through_the_arm_solver(monkeypatch):
    seen = []
    real = eef_to_joint._arm_vector_rad

    def spy(joint_deg):
        seen.append(dict(joint_deg))
        return real(joint_deg)

    monkeypatch.setattr(eef_to_joint, "_arm_vector_rad", spy)
    eef_to_joint.solve_ik(HOME, _delta(gripper=99.0))
    assert seen and all(len(vec) == 6 for vec in seen)  # dicts given; only arm joints read
    assert eef_to_joint._arm_vector_rad(HOME).shape == (5,)


def test_gripper_out_of_range_is_clamped_and_missing_gripper_holds():
    assert eef_to_joint.solve_ik(HOME, _delta(gripper=250.0))["gripper"] == 100.0
    assert eef_to_joint.solve_ik(HOME, _delta(gripper=-3.0))["gripper"] == 0.0
    delta = dict(ZERO_DELTA)
    del delta["gripper"]
    assert eef_to_joint.solve_ik(HOME, delta)["gripper"] == HOME["gripper"]
    assert eef_to_joint.solve_ik(HOME, _delta(gripper=float("nan")))["gripper"] == HOME["gripper"]


def test_non_finite_current_joint_state_is_rejected():
    with pytest.raises(ValueError):
        eef_to_joint.solve_ik(dict(HOME, elbow_flex=float("nan")), ZERO_DELTA)


def test_missing_arm_joint_is_rejected():
    state = dict(HOME)
    del state["wrist_flex"]
    with pytest.raises(KeyError):
        eef_to_joint.solve_ik(state, ZERO_DELTA)
