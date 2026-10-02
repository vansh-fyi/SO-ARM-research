"""Compact URDF kinematics text for the MLLM system prompt (Phase 15, D-07).

Renders every `<joint>` element of So-101/So-101.urdf (the regenerated,
verified Phase 10 URDF) as one line, in document order:

    - shoulder_pan: revolute, base -> shoulder, origin xyz 0.0388 0.0000 0.0624 m
      rpy 180.00 0.00 -180.00 deg, axis 0 0 1, limits -70.33 to 70.33 deg

Lengths are metres (4 decimals), rotations degrees (2 decimals); revolute
limits are converted to degrees, prismatic limits stay in metres. The raw
XML would cost ~5-8K tokens; this table is under 4000 characters.

Only verified conventions are stated (research conflict C8): URDF joint angle
0 is 0 degrees in the joint targets, the jaw slide range, and the gripper
channel's 0-100 percent meaning. The direction of positive rotation of each
real joint against its URDF axis has NOT been checked on hardware (open human
check, Plan 15-10), so the text never claims one.

Parsed with stdlib `xml.etree.ElementTree` (joint elements only; visual and
collision origins are ignored), never ikpy.
"""

import math
import xml.etree.ElementTree as ET
from pathlib import Path

URDF_PATH = Path(__file__).resolve().parents[2] / "So-101" / "So-101.urdf"

HEADER_LINES = (
    "Lengths are in metres and rotations in degrees. URDF joint angle 0 corresponds to 0 degrees "
    "in the joint targets.",
    "Gripper: the jaw joints <<JAWS>> slide <<JAW_RANGE>>; the gripper channel in the joint "
    "targets is 0 percent closed to 100 percent fully open.",
    "One line per joint: type, parent link -> child link, origin of the child frame in the parent "
    "frame (xyz, then roll pitch yaw), joint axis in the child frame, limits.",
)


def _num(value: float, decimals: int) -> str:
    text = f"{value:.{decimals}f}"
    return f"{0.0:.{decimals}f}" if float(text) == 0 else text  # no "-0.0000"


def _floats(text: str | None, default: str) -> list[float]:
    return [float(v) for v in (text or default).split()]


def _joint_line(joint: ET.Element) -> str:
    name = joint.get("name")
    joint_type = joint.get("type")
    parent = joint.find("parent").get("link")
    child = joint.find("child").get("link")
    parts = [f"{joint_type}", f"{parent} -> {child}"]

    origin = joint.find("origin")
    xyz = _floats(origin.get("xyz") if origin is not None else None, "0 0 0")
    rpy = _floats(origin.get("rpy") if origin is not None else None, "0 0 0")
    parts.append(
        "origin xyz " + " ".join(_num(v, 4) for v in xyz) + " m rpy "
        + " ".join(_num(math.degrees(v), 2) for v in rpy) + " deg"
    )

    axis = joint.find("axis")
    if joint_type != "fixed":
        axis_xyz = _floats(axis.get("xyz") if axis is not None else None, "1 0 0")
        parts.append("axis " + " ".join(f"{v:g}" for v in axis_xyz))

    limit = joint.find("limit")
    if joint_type == "revolute" and limit is not None:
        lower, upper = float(limit.get("lower", 0)), float(limit.get("upper", 0))
        parts.append(f"limits {_num(math.degrees(lower), 2)} to {_num(math.degrees(upper), 2)} deg")
    elif joint_type == "prismatic" and limit is not None:
        lower, upper = float(limit.get("lower", 0)), float(limit.get("upper", 0))
        parts.append(f"limits {_num(lower, 3)} to {_num(upper, 3)} m")
    elif joint_type == "continuous":
        parts.append("no limits")

    mimic = joint.find("mimic")
    if mimic is not None:
        parts.append(f"mimics {mimic.get('joint')}")
    return f"- {name}: " + ", ".join(parts)


def urdf_kinematics_text(urdf_path: Path = URDF_PATH) -> str:
    """Header plus one line per joint element of `urdf_path`, in document order."""
    path = Path(urdf_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"URDF file not found: {path}")
    joints = ET.parse(path).getroot().findall("joint")

    jaws = [j for j in joints if j.get("type") == "prismatic"]
    jaw_names = ", ".join(j.get("name") for j in jaws) or "(none)"
    jaw_limits = [j.find("limit") for j in jaws if j.find("limit") is not None]
    if jaw_limits:
        lo = min(float(lim.get("lower", 0)) for lim in jaw_limits)
        hi = max(float(lim.get("upper", 0)) for lim in jaw_limits)
        jaw_range = f"{lo:g} to {hi:g} m"
    else:
        jaw_range = "over an unspecified range"

    header = [line.replace("<<JAWS>>", jaw_names).replace("<<JAW_RANGE>>", jaw_range)
              for line in HEADER_LINES]
    return "\n".join(header + [_joint_line(joint) for joint in joints])


__all__ = ["URDF_PATH", "urdf_kinematics_text"]
