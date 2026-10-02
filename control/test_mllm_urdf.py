"""Tests for the compact URDF kinematics text (Phase 15, D-07).

`vla_bridge.mllm_urdf.urdf_kinematics_text()` renders every joint element of
So-101/So-101.urdf as one line (parent -> child, origin in metres and
degrees, axis, limits in degrees or metres) behind a header that states only
verified conventions (research conflict C8: the sign of positive rotation
against the URDF axis is unverified on hardware and must not be claimed).
Joint names are parsed independently here with xml.etree.
"""

import math
import xml.etree.ElementTree as ET

import pytest

from vla_bridge.mllm_prompt import lint_prompt_text
from vla_bridge.mllm_urdf import URDF_PATH, urdf_kinematics_text


def _urdf_joints():
    return ET.parse(URDF_PATH).getroot().findall("joint")


def _joint_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.startswith("- ")]


def test_urdf_path_resolves_to_repo_so101_file():
    assert URDF_PATH.name == "So-101.urdf"
    assert URDF_PATH.parent.name == "So-101"
    assert URDF_PATH.is_file()


def test_urdf_text_lists_every_joint_in_document_order():
    joints = _urdf_joints()
    lines = _joint_lines(urdf_kinematics_text())
    assert len(lines) == len(joints) > 0
    for joint, line in zip(joints, lines):
        assert line.startswith(f"- {joint.get('name')}: {joint.get('type')}, ")
        parent = joint.find("parent").get("link")
        child = joint.find("child").get("link")
        assert f"{parent} -> {child}" in line


def test_urdf_text_revolute_limits_in_degrees_and_origin_units():
    lines = {line.split(":")[0][2:]: line for line in _joint_lines(urdf_kinematics_text())}
    pan = lines["shoulder_pan"]
    expected = f"{math.degrees(1.227484):.2f}"
    assert expected == "70.33"
    assert f"limits -{expected} to {expected} deg" in pan
    assert "origin xyz 0.0388 0.0000 0.0624 m" in pan
    assert "rpy 180.00 0.00 -180.00 deg" in pan
    assert "axis 0 0 1" in pan
    roll = lines["wrist_roll"]
    assert f"limits {math.degrees(-2.7438472969992493):.2f} to {math.degrees(2.841206309382605):.2f} deg" in roll


def test_urdf_text_prismatic_limits_in_metres_and_fixed_has_no_limits():
    lines = {line.split(":")[0][2:]: line for line in _joint_lines(urdf_kinematics_text())}
    assert "limits 0.000 to 0.036 m" in lines["gripper_left"]
    assert "axis 0 -1 0" in lines["gripper_left"]
    assert "mimics gripper_left" in lines["gripper_right"]
    assert "limits" not in lines["right_hand_to_gripper"]


def test_urdf_text_is_compact_and_lint_clean():
    text = urdf_kinematics_text()
    assert len(text) < 4000
    assert lint_prompt_text(text) == []


def test_urdf_text_states_only_verified_conventions():
    text = urdf_kinematics_text()
    lowered = text.lower()
    assert "metres" in lowered and "degrees" in lowered
    assert "urdf joint angle 0 corresponds to 0 degrees in the joint targets" in lowered
    assert "0 percent closed" in lowered and "100 percent fully open" in lowered
    assert "0.036 m" in text
    # Research C8: never claim a sign relationship against the URDF axis.
    for phrase in ("right-hand", "right hand rule", "positive", "counterclockwise",
                   "counter-clockwise", "clockwise"):
        assert phrase not in lowered


def test_urdf_text_missing_file_raises_naming_the_path(tmp_path):
    missing = tmp_path / "nope" / "So-101.urdf"
    with pytest.raises(FileNotFoundError, match="nope"):
        urdf_kinematics_text(missing)


def test_urdf_text_is_deterministic():
    assert urdf_kinematics_text() == urdf_kinematics_text(URDF_PATH)
