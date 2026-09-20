"""Real (no-mock) load-and-assert tests for So-101/So-101.urdf against
TWIN-01..05.

Loads the generated URDF via `yourdfpy` (a standard, independent URDF
parser) and independently confirms the structural/kinematic assertions
Phase 10's earlier plans (10-01/10-02/10-03) claim to have fixed, rather
than trusting `scripts/mjcf_to_urdf.py`'s own output. Imports
`scripts/verify_urdf.py`'s tree-walk helpers rather than duplicating logic,
and imports `scripts/calibration_utils.py`'s formula for TWIN-05's
independent recomputation.

Since `scripts/` has no `conftest.py`, this file inserts its own directory
onto `sys.path` so `import verify_urdf` and `import calibration_utils`
resolve as sibling modules (simpler than
`LIBERO/libero/libero/envs/test_camera_config.py`'s four-level-up
repo-root anchoring pattern, since this test needs no `libero.*` imports).
"""
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest

import calibration_utils
import verify_urdf

# robot.xml's original, unchanged wrist_roll limit (see verify_urdf.py's own
# constants) -- kept as the expected value here too so this test doesn't
# silently pass if both modules' constants drifted together.
WRIST_ROLL_LOWER = -2.7438472969992493
WRIST_ROLL_UPPER = 2.841206309382605


def test_established_gripper_export_matches_approved_configuration():
    root = Path(__file__).resolve().parents[1]
    urdf = ET.parse(root / "So-101/So-101.urdf").getroot()
    mjcf = ET.parse(root / "LIBERO/libero/libero/assets/grippers/soarm_gripper.xml").getroot()
    for name in ("gripper_left", "gripper_right"):
        limit = urdf.find(f"./joint[@name='{name}']/limit")
        assert float(limit.get("lower")) == 0
        assert float(limit.get("upper")) == .036
        assert [float(x) for x in mjcf.find(f".//joint[@name='{name}']").get("range").split()] == [0, .036]
        assert [float(x) for x in mjcf.find(f"./actuator/position[@joint='{name}']").get("ctrlrange").split()] == [0, .036]
    mimic = urdf.find("./joint[@name='gripper_right']/mimic")
    assert mimic.get("joint") == "gripper_left"
    for name, rgba in (("parallel_gripper_mechanism", [0.1,0.1,0.1,1]),
                       ("gripper_left_jaw", [1,.82,.12,1]),
                       ("gripper_right_jaw", [1,.82,.12,1])):
        color = urdf.find(f"./link[@name='{name}']/visual/material/color")
        assert [float(x) for x in color.get("rgba").split()] == rgba
    mesh = urdf.find("./link[@name='right_hand']/visual/geometry/mesh")
    assert (root / "So-101" / mesh.get("filename")).read_bytes() == (root / "LIBERO/libero/libero/assets/robots/soarm101/assets/sts3215_03a_v1.stl").read_bytes()
    assert urdf.find("./link[@name='parallel_gripper_mechanism']/visual/geometry/cylinder") is not None
    assert urdf.find("./joint[@name='gripper_to_eef']/child").get("link") == "eef"

# Regression guard against Pitfall 1: the naive full-turn calibration-
# derived value that must NOT have leaked into wrist_roll's URDF limit.
NAIVE_FULL_TURN_LOWER = -math.pi
NAIVE_FULL_TURN_UPPER = math.pi


@pytest.fixture(scope="module")
def robot():
    return verify_urdf.load_generated_urdf()


def test_gripper_is_child_of_wrist(robot):
    """TWIN-01: the gripper is a proper descendant of the wrist (right_hand),
    not a disconnected root-level body."""
    assert verify_urdf.is_descendant(robot, "right_gripper", "right_hand") is True
    # right_gripper must not be a URDF root link -- it has an incoming joint
    # (right_hand_to_gripper), so it appears as a child in the parent map.
    parent_map = verify_urdf.build_parent_map(robot)
    assert "right_gripper" in parent_map


def test_wrist_roll_range(robot):
    """TWIN-02: wrist_roll is revolute with the exact asymmetric mechanical
    limit carried over from robot.xml."""
    joint = robot.joint_map["wrist_roll"]
    assert joint.type == "revolute"
    assert joint.limit.lower == pytest.approx(WRIST_ROLL_LOWER, abs=1e-6)
    assert joint.limit.upper == pytest.approx(WRIST_ROLL_UPPER, abs=1e-6)


def test_gripper_joint_axes(robot):
    """TWIN-03: gripper_left/gripper_right are prismatic with the correct,
    opposing jaw-travel axes."""
    left = robot.joint_map["gripper_left"]
    assert left.type == "prismatic"
    assert list(left.axis) == [0, -1, 0]

    right = robot.joint_map["gripper_right"]
    assert right.type == "prismatic"
    assert list(right.axis) == [0, 1, 0]


def test_no_absolute_mesh_paths(robot):
    """TWIN-04: no mesh filename is an absolute path or file:// URI."""
    assert verify_urdf.no_absolute_mesh_paths(robot) == []


def test_joint_limits_match_calibration(robot):
    """TWIN-05: for each of the 4 calibration-derived arm joints, recompute
    the expected (lo, hi) independently via calibration_utils' formula
    against the live calibration JSON, and assert the URDF's limit matches.
    Also regression-guards against Pitfall 1: wrist_roll's URDF limit must
    NOT equal the naive full-turn calibration-derived value, and must equal
    robot.xml's original unchanged value."""
    if not calibration_utils.CALIBRATION_PATH.exists():
        pytest.skip(
            f"Live calibration file not found at "
            f"{calibration_utils.CALIBRATION_PATH} -- TWIN-05 needs the "
            "real follower's calibration cache to independently recompute "
            "expected joint limits."
        )

    import json

    with open(calibration_utils.CALIBRATION_PATH) as f:
        calibration = json.load(f)

    for joint_name in calibration_utils.JOINTS_FROM_CALIBRATION:
        entry = calibration[joint_name]
        expected_lo, expected_hi = calibration_utils.calibration_ticks_to_radians(
            entry["range_min"], entry["range_max"]
        )
        joint = robot.joint_map[joint_name]
        assert joint.limit.lower == pytest.approx(expected_lo, abs=1e-5)
        assert joint.limit.upper == pytest.approx(expected_hi, abs=1e-5)

    # Regression guard (Pitfall 1): wrist_roll must NOT have been fed
    # through the naive full-turn calibration value...
    wrist_roll = robot.joint_map["wrist_roll"]
    assert wrist_roll.limit.lower != pytest.approx(NAIVE_FULL_TURN_LOWER, abs=1e-3)
    assert wrist_roll.limit.upper != pytest.approx(NAIVE_FULL_TURN_UPPER, abs=1e-3)
    # ...and DOES equal robot.xml's original unchanged mechanical-limit value.
    assert wrist_roll.limit.lower == pytest.approx(WRIST_ROLL_LOWER, abs=1e-6)
    assert wrist_roll.limit.upper == pytest.approx(WRIST_ROLL_UPPER, abs=1e-6)
