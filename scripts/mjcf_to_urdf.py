"""Generate a correct So-101/So-101.urdf from the two already-correct MJCF files.

This script GENERATES `So-101/So-101.urdf` fresh from `robot.xml` (arm) +
`soarm_gripper.xml` (gripper)'s MJCF `<worldbody>` trees. It does NOT patch
the existing broken CoppeliaSim-exported URDF in place -- it overwrites it
wholesale, per Phase 10 D-02. D-02's rationale: generating from the MJCF
(already kinematically correct per D-01, and independently re-verified
against live servo calibration in Plans 10-01/10-02) guarantees the URDF and
MJCF agree by construction, instead of risking two independently-maintained
kinematic definitions drifting apart.

The broken artifact being replaced (`So-101/So-101.urdf`, a CoppeliaSim
export) has the gripper floating as a disconnected root-level object, 30
absolute `file:///Users/hp/Downloads/...` mesh paths, no `wrist_roll`/gripper
joints at all, and terminates at `wrist_link_respondable`. This script fixes
all four problems by construction: the gripper is parented under the arm's
terminal `right_hand` link (TWIN-01), `wrist_roll` + `gripper_left`/
`gripper_right` are emitted with the MJCF's own limits/axes (TWIN-02/03), and
every mesh `filename` is a bare in-repo-relative path (TWIN-04).

Per 10-RESEARCH.md D-03/Pattern 1: this is a hand-rolled ElementTree-based
converter (stdlib only: xml.etree.ElementTree, math, pathlib), not a
general-purpose MJCF->URDF library -- the one actively-maintained PyPI
candidate (`mjcf-urdf-simple-converter`) does not support prismatic joints,
which this gripper needs (`gripper_left`/`gripper_right` are MJCF `slide`
joints). The kinematic tree is small and fully known ahead of time (9 links,
8 joints total), so an explicit per-joint mapping table -- reading each
body's/joint's attributes directly out of the parsed MJCF trees -- is
simpler and easier to verify by hand than a generic recursive MJCF
interpreter.

Visual mesh sourcing (per D-02): the arm's DAE meshes come from the
CoppeliaSim export (`So-101/*.dae`, already in-repo, referenced here by bare
filename only -- this script does not read that broken URDF at runtime, the
filenames below were confirmed by a direct read of it during planning). The
gripper's STL meshes come from `coppelia/meshes/*.stl` (relative path
`../coppelia/meshes/...` from this URDF's own directory). Only the arm/
gripper MJCF files are read at runtime; visual mesh *placement* offsets for
the gripper meshes are read live from `soarm_gripper.xml` (so Plan 10-02's
clamp-visual fix is picked up automatically); the arm's per-link mesh
placement offsets are CoppeliaSim-export artifacts (the STL/DAE's own local
origin relative to the link frame) that don't exist in the MJCF at all --
those are hardcoded constants below, sourced from the broken URDF's own
per-link visual/collision <origin> (confirmed identical across an entire
link's mesh list in that file), which is exactly the "reuse mesh filename
list" carry-over this script's plan calls for.
"""

from __future__ import annotations

import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ROBOT_XML = ROOT / "LIBERO" / "libero" / "libero" / "assets" / "robots" / "soarm101" / "robot.xml"
GRIPPER_XML = ROOT / "LIBERO" / "libero" / "libero" / "assets" / "grippers" / "soarm_gripper.xml"
OUTPUT_URDF = ROOT / "So-101" / "So-101.urdf"

# STS3215 servo torque limit (robot.xml's <motor ctrlrange="-2.94 2.94" .../>
# for all 5 arm actuators, torq_shoulder_pan..torq_wrist_roll).
ARM_JOINT_EFFORT = "2.94"
# Placeholder only -- no calibrated velocity data exists for this arm (see
# Phase 10 Plan 10-03). Flagged here and in the emitted URDF's own comment
# rather than silently presented as a real spec value.
ARM_JOINT_VELOCITY_PLACEHOLDER = "10.0"

# Per-link CoppeliaSim-export mesh-placement offset (STL/DAE local origin
# relative to the link frame). Not present in the MJCF at all -- MJCF's
# per-geom pos/quat describe *visual sub-parts* of a rigid body assembly, not
# a single offset for an externally-authored respondable/visual mesh set.
# Confirmed identical across every visual/collision entry within one link in
# the broken So-101/So-101.urdf (read directly during Plan 10-03 planning);
# carried over verbatim -- this is mesh *placement*, not kinematic structure.
MESH_PLACEMENT = {
    "base": ("-0.300000 -0.225000 0.081000", "-1.570802 3.141593 -1.570796"),
    "shoulder": ("-0.030399 0.000422 -0.041700", "1.570800 1.570800 0.000000"),
    "upper_arm": ("-0.112570 -0.015500 0.018700", "0.000003 3.141593 1.570793"),
    "lower_arm": ("-0.064850 -0.032000 0.018200", "-0.000003 3.141593 -3.141593"),
    "wrist": ("0.042400 -0.000034 0.030600", "0.526021 1.570804 0.525218"),
}

# Bare filenames (sibling of this URDF, i.e. So-101/), per-link, confirmed
# against the broken So-101/So-101.urdf's own mesh list during planning.
LINK_MESHES = {
    "base": {
        "visual": [f"So-101_base_link_visual_vis_{i}.dae" for i in range(1, 5)],
        "collision": [f"So-101_robot_base_coll_{i}.dae" for i in range(1, 5)],
    },
    "shoulder": {
        "visual": [f"So-101_shoulder_link_visual_vis_{i}.dae" for i in range(1, 4)],
        "collision": [f"So-101_shoulder_link_respondable_coll_{i}.dae" for i in range(1, 4)],
    },
    "upper_arm": {
        "visual": [f"So-101_upper_arm_link_visual_vis_{i}.dae" for i in range(1, 3)],
        "collision": [f"So-101_upper_arm_link_respondable_coll_{i}.dae" for i in range(1, 3)],
    },
    "lower_arm": {
        "visual": [f"So-101_lower_arm_link_visual_vis_{i}.dae" for i in range(1, 4)],
        "collision": [f"So-101_lower_arm_link_respondable_coll_{i}.dae" for i in range(1, 4)],
    },
    "wrist": {
        "visual": [f"So-101_wrist_link_visual_vis_{i}.dae" for i in range(1, 3)],
        "collision": [f"So-101_wrist_link_respondable_coll_{i}.dae" for i in range(1, 3)],
    },
}

# Required MJCF body/joint names -- fail loudly (not silently emit a partial/
# malformed URDF) if any is missing from the parsed tree. Mirrors
# coppelia/export_model_library.py's fail-loud precondition-check pattern.
REQUIRED_ROBOT_BODIES = ("base", "shoulder", "upper_arm", "lower_arm", "wrist", "right_hand")
REQUIRED_ROBOT_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
REQUIRED_GRIPPER_BODIES = ("right_gripper", "gripper_left_jaw", "gripper_right_jaw")
REQUIRED_GRIPPER_JOINTS = ("gripper_left", "gripper_right")


def mjcf_quat_to_rpy(w: float, x: float, y: float, z: float) -> tuple[float, float, float]:
    """Closed-form MuJoCo quat (w,x,y,z) -> URDF roll,pitch,yaw (rad).

    Source: 10-RESEARCH.md Pattern 1, derived from MuJoCo's documented quat
    convention (w,x,y,z) and the standard Tait-Bryan intrinsic ZYX->rpy
    conversion URDF uses. No scipy/numpy needed -- every joint in this robot
    is a modest, well-documented single-axis rotation, not a general
    gimbal-lock-prone case.
    """
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else math.asin(sinp)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def parse_floats(attr: str) -> list[float]:
    return [float(v) for v in attr.split()]


def fmt(values) -> str:
    return " ".join(f"{v:.10g}" for v in values)


def find_body(root: ET.Element, name: str) -> ET.Element | None:
    for body in root.iter("body"):
        if body.get("name") == name:
            return body
    return None


def find_joint(body: ET.Element, name: str) -> ET.Element | None:
    for joint in body.findall("joint"):
        if joint.get("name") == name:
            return joint
    return None


def require_body(root: ET.Element, name: str, source: Path) -> ET.Element:
    body = find_body(root, name)
    if body is None:
        sys.exit(f"ERROR: missing body '{name}' in {source}")
    return body


def require_joint(body: ET.Element, name: str, source: Path) -> ET.Element:
    joint = find_joint(body, name)
    if joint is None:
        sys.exit(f"ERROR: missing joint '{name}' on body '{body.get('name')}' in {source}")
    return joint


def body_origin_rpy(body: ET.Element) -> tuple[str, str]:
    """A body's MJCF pos/quat -> URDF <origin xyz rpy> string pair."""
    pos = body.get("pos", "0 0 0")
    quat = parse_floats(body.get("quat", "1 0 0 0"))
    roll, pitch, yaw = mjcf_quat_to_rpy(*quat)
    return pos, fmt((roll, pitch, yaw))


def geom_origin_rpy(geom: ET.Element) -> tuple[str, str]:
    pos = geom.get("pos", "0 0 0")
    quat = parse_floats(geom.get("quat", "1 0 0 0"))
    roll, pitch, yaw = mjcf_quat_to_rpy(*quat)
    return pos, fmt((roll, pitch, yaw))


def inertial_from_fullinertia(body: ET.Element) -> ET.Element:
    """MJCF <inertial pos mass fullinertia> -> URDF <inertial>.

    fullinertia order (ixx iyy izz ixy ixz iyz) maps 1:1 onto URDF's named
    <inertia ixx iyy izz ixy ixz iyz> attributes -- both are already
    expressed in the body's own frame, so the inertial <origin> needs no
    rotation (rpy="0 0 0"), only the MJCF pos as the center-of-mass offset.
    """
    inertial = body.find("inertial")
    if inertial is None:
        sys.exit(f"ERROR: body '{body.get('name')}' has no <inertial> element")
    pos = inertial.get("pos", "0 0 0")
    mass = inertial.get("mass")
    ixx, iyy, izz, ixy, ixz, iyz = parse_floats(inertial.get("fullinertia"))
    return build_inertial(pos, mass, ixx, iyy, izz, ixy, ixz, iyz)


def inertial_from_diaginertia(body: ET.Element) -> ET.Element:
    """MJCF <inertial pos mass diaginertia="Ixx Iyy Izz"> -> URDF <inertial>.

    diaginertia gives the 3 diagonal moments directly (Ixx, Iyy, Izz, in that
    order) with all off-diagonal cross terms zero by construction.
    """
    inertial = body.find("inertial")
    if inertial is None:
        sys.exit(f"ERROR: body '{body.get('name')}' has no <inertial> element")
    pos = inertial.get("pos", "0 0 0")
    mass = inertial.get("mass")
    ixx, iyy, izz = parse_floats(inertial.get("diaginertia"))
    return build_inertial(pos, mass, ixx, iyy, izz, 0.0, 0.0, 0.0)


def build_inertial(pos, mass, ixx, iyy, izz, ixy, ixz, iyz) -> ET.Element:
    inertial = ET.Element("inertial")
    ET.SubElement(inertial, "origin", xyz=pos, rpy="0 0 0")
    ET.SubElement(inertial, "mass", value=str(mass))
    ET.SubElement(
        inertial,
        "inertia",
        ixx=f"{ixx:.10g}",
        ixy=f"{ixy:.10g}",
        ixz=f"{ixz:.10g}",
        iyy=f"{iyy:.10g}",
        iyz=f"{iyz:.10g}",
        izz=f"{izz:.10g}",
    )
    return inertial


def make_link(
    name: str,
    inertial: ET.Element,
    visual_meshes: list[str] | None = None,
    visual_origin: tuple[str, str] = ("0 0 0", "0 0 0"),
    collision_meshes: list[str] | None = None,
    collision_origin: tuple[str, str] = ("0 0 0", "0 0 0"),
    mesh_scale: str | None = None,
    no_mesh_comment: str | None = None,
) -> ET.Element:
    link = ET.Element("link", name=name)
    if no_mesh_comment:
        link.append(ET.Comment(f" {no_mesh_comment} "))
    vxyz, vrpy = visual_origin
    for mesh_file in visual_meshes or []:
        visual = ET.SubElement(link, "visual")
        ET.SubElement(visual, "origin", xyz=vxyz, rpy=vrpy)
        geometry = ET.SubElement(visual, "geometry")
        mesh_attrs = {"filename": mesh_file}
        if mesh_scale:
            mesh_attrs["scale"] = mesh_scale
        ET.SubElement(geometry, "mesh", **mesh_attrs)
    cxyz, crpy = collision_origin
    for mesh_file in collision_meshes or []:
        collision = ET.SubElement(link, "collision")
        ET.SubElement(collision, "origin", xyz=cxyz, rpy=crpy)
        geometry = ET.SubElement(collision, "geometry")
        mesh_attrs = {"filename": mesh_file}
        if mesh_scale:
            mesh_attrs["scale"] = mesh_scale
        ET.SubElement(geometry, "mesh", **mesh_attrs)
    link.append(inertial)
    return link


def make_box_collision_link(name: str, inertial: ET.Element, box_size: str, box_pos: str) -> ET.Element:
    link = ET.Element("link", name=name)
    collision = ET.SubElement(link, "collision")
    ET.SubElement(collision, "origin", xyz=box_pos, rpy="0 0 0")
    geometry = ET.SubElement(collision, "geometry")
    ET.SubElement(geometry, "box", size=box_size)
    link.append(inertial)
    return link


def make_joint(
    name: str,
    joint_type: str,
    parent: str,
    child: str,
    origin_xyz: str,
    origin_rpy: str,
    axis: str | None = None,
    lower: str | None = None,
    upper: str | None = None,
    effort: str | None = None,
    velocity: str | None = None,
    damping: str | None = None,
    friction: str | None = None,
) -> ET.Element:
    joint = ET.Element("joint", name=name, type=joint_type)
    ET.SubElement(joint, "parent", link=parent)
    ET.SubElement(joint, "child", link=child)
    ET.SubElement(joint, "origin", xyz=origin_xyz, rpy=origin_rpy)
    if axis is not None:
        ET.SubElement(joint, "axis", xyz=axis)
    if lower is not None:
        limit_attrs = {"lower": lower, "upper": upper}
        if effort is not None:
            limit_attrs["effort"] = effort
        if velocity is not None:
            limit_attrs["velocity"] = velocity
        ET.SubElement(joint, "limit", **limit_attrs)
    if damping is not None:
        ET.SubElement(joint, "dynamics", damping=damping, friction=friction)
    return joint


def main() -> None:
    for required in (ROBOT_XML, GRIPPER_XML):
        if not required.exists():
            sys.exit(f"ERROR: missing {required}")

    robot_tree = ET.parse(ROBOT_XML)
    robot_root = robot_tree.getroot()
    gripper_tree = ET.parse(GRIPPER_XML)
    gripper_root = gripper_tree.getroot()

    # Fail-loud precondition checks (T-10-05): every body/joint this
    # converter's mapping table depends on must exist before we emit
    # anything, per coppelia/export_model_library.py's own precondition
    # pattern -- never silently emit a partial/malformed URDF.
    bodies = {name: require_body(robot_root, name, ROBOT_XML) for name in REQUIRED_ROBOT_BODIES}
    joints = {}
    joints["shoulder_pan"] = require_joint(bodies["shoulder"], "shoulder_pan", ROBOT_XML)
    joints["shoulder_lift"] = require_joint(bodies["upper_arm"], "shoulder_lift", ROBOT_XML)
    joints["elbow_flex"] = require_joint(bodies["lower_arm"], "elbow_flex", ROBOT_XML)
    joints["wrist_flex"] = require_joint(bodies["wrist"], "wrist_flex", ROBOT_XML)
    joints["wrist_roll"] = require_joint(bodies["right_hand"], "wrist_roll", ROBOT_XML)

    gripper_bodies = {
        name: require_body(gripper_root, name, GRIPPER_XML) for name in REQUIRED_GRIPPER_BODIES
    }
    gripper_joints = {}
    gripper_joints["gripper_left"] = require_joint(
        gripper_bodies["gripper_left_jaw"], "gripper_left", GRIPPER_XML
    )
    gripper_joints["gripper_right"] = require_joint(
        gripper_bodies["gripper_right_jaw"], "gripper_right", GRIPPER_XML
    )

    right_gripper_body = gripper_bodies["right_gripper"]
    main_frame_geom = right_gripper_body.find("./geom[@name='main_frame_visual']")
    if main_frame_geom is None:
        sys.exit(f"ERROR: missing geom 'main_frame_visual' on body 'right_gripper' in {GRIPPER_XML}")
    left_jaw_visual_geom = gripper_bodies["gripper_left_jaw"].find("./geom[@name='left_jaw_visual']")
    if left_jaw_visual_geom is None:
        sys.exit(f"ERROR: missing geom 'left_jaw_visual' on body 'gripper_left_jaw' in {GRIPPER_XML}")
    right_jaw_visual_geom = gripper_bodies["gripper_right_jaw"].find("./geom[@name='right_jaw_visual']")
    if right_jaw_visual_geom is None:
        sys.exit(f"ERROR: missing geom 'right_jaw_visual' on body 'gripper_right_jaw' in {GRIPPER_XML}")

    robot = ET.Element("robot", name="so101")
    robot.append(
        ET.Comment(
            " Generated by scripts/mjcf_to_urdf.py from robot.xml + soarm_gripper.xml - "
            "do not hand-edit. Regenerate instead. See Phase 10 D-02/D-03. "
        )
    )
    robot.append(
        ET.Comment(
            " velocity=\"10.0\" on all 5 arm revolute joints below is a PLACEHOLDER - "
            "no calibrated velocity data exists for this arm (Phase 10 Plan 10-03). "
        )
    )

    # --- Link: base (root, no joint) ---
    base_body = bodies["base"]
    base_meshes = LINK_MESHES["base"]
    base_origin = MESH_PLACEMENT["base"]
    robot.append(
        make_link(
            "base",
            inertial_from_fullinertia(base_body),
            visual_meshes=base_meshes["visual"],
            visual_origin=base_origin,
            collision_meshes=base_meshes["collision"],
            collision_origin=base_origin,
        )
    )

    # --- shoulder_pan joint + shoulder link ---
    shoulder_body = bodies["shoulder"]
    xyz, rpy = body_origin_rpy(shoulder_body)
    j = joints["shoulder_pan"]
    lo, hi = j.get("range").split()
    robot.append(
        make_joint(
            "shoulder_pan",
            "revolute",
            "base",
            "shoulder",
            xyz,
            rpy,
            axis=j.get("axis"),
            lower=lo,
            upper=hi,
            effort=ARM_JOINT_EFFORT,
            velocity=ARM_JOINT_VELOCITY_PLACEHOLDER,
            damping=j.get("damping"),
            friction=j.get("frictionloss"),
        )
    )
    shoulder_meshes = LINK_MESHES["shoulder"]
    shoulder_origin = MESH_PLACEMENT["shoulder"]
    robot.append(
        make_link(
            "shoulder",
            inertial_from_fullinertia(shoulder_body),
            visual_meshes=shoulder_meshes["visual"],
            visual_origin=shoulder_origin,
            collision_meshes=shoulder_meshes["collision"],
            collision_origin=shoulder_origin,
        )
    )

    # --- shoulder_lift joint + upper_arm link ---
    upper_arm_body = bodies["upper_arm"]
    xyz, rpy = body_origin_rpy(upper_arm_body)
    j = joints["shoulder_lift"]
    lo, hi = j.get("range").split()
    robot.append(
        make_joint(
            "shoulder_lift",
            "revolute",
            "shoulder",
            "upper_arm",
            xyz,
            rpy,
            axis=j.get("axis"),
            lower=lo,
            upper=hi,
            effort=ARM_JOINT_EFFORT,
            velocity=ARM_JOINT_VELOCITY_PLACEHOLDER,
            damping=j.get("damping"),
            friction=j.get("frictionloss"),
        )
    )
    upper_arm_meshes = LINK_MESHES["upper_arm"]
    upper_arm_origin = MESH_PLACEMENT["upper_arm"]
    robot.append(
        make_link(
            "upper_arm",
            inertial_from_fullinertia(upper_arm_body),
            visual_meshes=upper_arm_meshes["visual"],
            visual_origin=upper_arm_origin,
            collision_meshes=upper_arm_meshes["collision"],
            collision_origin=upper_arm_origin,
        )
    )

    # --- elbow_flex joint + lower_arm link ---
    lower_arm_body = bodies["lower_arm"]
    xyz, rpy = body_origin_rpy(lower_arm_body)
    j = joints["elbow_flex"]
    lo, hi = j.get("range").split()
    robot.append(
        make_joint(
            "elbow_flex",
            "revolute",
            "upper_arm",
            "lower_arm",
            xyz,
            rpy,
            axis=j.get("axis"),
            lower=lo,
            upper=hi,
            effort=ARM_JOINT_EFFORT,
            velocity=ARM_JOINT_VELOCITY_PLACEHOLDER,
            damping=j.get("damping"),
            friction=j.get("frictionloss"),
        )
    )
    lower_arm_meshes = LINK_MESHES["lower_arm"]
    lower_arm_origin = MESH_PLACEMENT["lower_arm"]
    robot.append(
        make_link(
            "lower_arm",
            inertial_from_fullinertia(lower_arm_body),
            visual_meshes=lower_arm_meshes["visual"],
            visual_origin=lower_arm_origin,
            collision_meshes=lower_arm_meshes["collision"],
            collision_origin=lower_arm_origin,
        )
    )

    # --- wrist_flex joint + wrist link ---
    wrist_body = bodies["wrist"]
    xyz, rpy = body_origin_rpy(wrist_body)
    j = joints["wrist_flex"]
    lo, hi = j.get("range").split()
    robot.append(
        make_joint(
            "wrist_flex",
            "revolute",
            "lower_arm",
            "wrist",
            xyz,
            rpy,
            axis=j.get("axis"),
            lower=lo,
            upper=hi,
            effort=ARM_JOINT_EFFORT,
            velocity=ARM_JOINT_VELOCITY_PLACEHOLDER,
            damping=j.get("damping"),
            friction=j.get("frictionloss"),
        )
    )
    wrist_meshes = LINK_MESHES["wrist"]
    wrist_origin = MESH_PLACEMENT["wrist"]
    robot.append(
        make_link(
            "wrist",
            inertial_from_fullinertia(wrist_body),
            visual_meshes=wrist_meshes["visual"],
            visual_origin=wrist_origin,
            collision_meshes=wrist_meshes["collision"],
            collision_origin=wrist_origin,
        )
    )

    # --- wrist_roll joint + right_hand link (TWIN-02) ---
    right_hand_body = bodies["right_hand"]
    xyz, rpy = body_origin_rpy(right_hand_body)
    j = joints["wrist_roll"]
    lo, hi = j.get("range").split()
    robot.append(
        make_joint(
            "wrist_roll",
            "revolute",
            "wrist",
            "right_hand",
            xyz,
            rpy,
            axis=j.get("axis"),
            lower=lo,
            upper=hi,
            effort=ARM_JOINT_EFFORT,
            velocity=ARM_JOINT_VELOCITY_PLACEHOLDER,
            damping=j.get("damping"),
            friction=j.get("frictionloss"),
        )
    )
    robot.append(
        make_link(
            "right_hand",
            inertial_from_fullinertia(right_hand_body),
            no_mesh_comment=(
                "right_hand has no visual/collision mesh - no mesh exists at this "
                "split point in the CoppeliaSim export (arm/gripper boundary)"
            ),
        )
    )

    # --- right_hand_to_gripper fixed joint + right_gripper link (TWIN-01) ---
    xyz, rpy = geom_origin_rpy(main_frame_geom)
    robot.append(
        make_joint(
            "right_hand_to_gripper",
            "fixed",
            "right_hand",
            "right_gripper",
            "0 0 0",
            "0 0 0",
        )
    )
    robot.append(
        make_link(
            "right_gripper",
            inertial_from_diaginertia(right_gripper_body),
            visual_meshes=["../coppelia/meshes/main_frame_visual.stl"],
            visual_origin=(xyz, rpy),
            mesh_scale="0.001 0.001 0.001",
        )
    )

    # --- gripper_left joint + gripper_left_jaw link (TWIN-03) ---
    j = gripper_joints["gripper_left"]
    lo, hi = j.get("range").split()
    robot.append(
        make_joint(
            "gripper_left",
            "prismatic",
            "right_gripper",
            "gripper_left_jaw",
            "0 0 0",
            "0 0 0",
            axis=j.get("axis"),
            lower=lo,
            upper=hi,
            effort="60",
            velocity="0.5",
            damping=j.get("damping"),
            friction=j.get("frictionloss"),
        )
    )
    left_xyz, left_rpy = geom_origin_rpy(left_jaw_visual_geom)
    left_jaw_link = make_link(
        "gripper_left_jaw",
        inertial_from_diaginertia(gripper_bodies["gripper_left_jaw"]),
        visual_meshes=["../coppelia/meshes/clamp_1_visual.stl"],
        visual_origin=(left_xyz, left_rpy),
        mesh_scale="0.001 0.001 0.001",
    )
    # Append collision box (2x the MJCF left_jaw_collision box's half-extents)
    # into the same link, after the visual mesh already added by make_link.
    left_collision_geom = gripper_bodies["gripper_left_jaw"].find("./geom[@name='left_jaw_collision']")
    if left_collision_geom is None:
        sys.exit(f"ERROR: missing geom 'left_jaw_collision' on body 'gripper_left_jaw' in {GRIPPER_XML}")
    half_extents = parse_floats(left_collision_geom.get("size"))
    box_size = fmt(v * 2 for v in half_extents)
    box_pos = left_collision_geom.get("pos")
    collision = ET.SubElement(left_jaw_link, "collision")
    ET.SubElement(collision, "origin", xyz=box_pos, rpy="0 0 0")
    geometry = ET.SubElement(collision, "geometry")
    ET.SubElement(geometry, "box", size=box_size)
    robot.append(left_jaw_link)

    # --- gripper_right joint + gripper_right_jaw link (TWIN-03) ---
    j = gripper_joints["gripper_right"]
    lo, hi = j.get("range").split()
    robot.append(
        make_joint(
            "gripper_right",
            "prismatic",
            "right_gripper",
            "gripper_right_jaw",
            "0 0 0",
            "0 0 0",
            axis=j.get("axis"),
            lower=lo,
            upper=hi,
            effort="60",
            velocity="0.5",
            damping=j.get("damping"),
            friction=j.get("frictionloss"),
        )
    )
    right_xyz, right_rpy = geom_origin_rpy(right_jaw_visual_geom)
    right_jaw_link = make_link(
        "gripper_right_jaw",
        inertial_from_diaginertia(gripper_bodies["gripper_right_jaw"]),
        visual_meshes=["../coppelia/meshes/clamp_2_visual.stl"],
        visual_origin=(right_xyz, right_rpy),
        mesh_scale="0.001 0.001 0.001",
    )
    right_collision_geom = gripper_bodies["gripper_right_jaw"].find("./geom[@name='right_jaw_collision']")
    if right_collision_geom is None:
        sys.exit(f"ERROR: missing geom 'right_jaw_collision' on body 'gripper_right_jaw' in {GRIPPER_XML}")
    half_extents = parse_floats(right_collision_geom.get("size"))
    box_size = fmt(v * 2 for v in half_extents)
    box_pos = right_collision_geom.get("pos")
    collision = ET.SubElement(right_jaw_link, "collision")
    ET.SubElement(collision, "origin", xyz=box_pos, rpy="0 0 0")
    geometry = ET.SubElement(collision, "geometry")
    ET.SubElement(geometry, "box", size=box_size)
    robot.append(right_jaw_link)

    ET.indent(robot, space="  ")
    tree = ET.ElementTree(robot)
    OUTPUT_URDF.parent.mkdir(parents=True, exist_ok=True)
    tree.write(OUTPUT_URDF, encoding="unicode", xml_declaration=True)
    with open(OUTPUT_URDF, "a") as f:
        f.write("\n")

    print(f"Saved -> {OUTPUT_URDF}")


if __name__ == "__main__":
    main()
