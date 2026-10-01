"""Gemini Robotics-ER point/box -> one bounded joint-space step (Plan 14-13).

READ THIS FIRST: this is NOT a native VLA action-output policy
==============================================================
Gemini Robotics-ER 2 (the public model the Gemini backends use, per
14-CONTEXT.md D-02's supersession) returns embodied-reasoning output: image
points and boxes as JSON. It never returns joint or motor commands. This
module is a small hand-written controller that turns one such point into one
small arm motion. It is a **bounded, proportional, image-plane-only
visual-servo primitive**:

  - It moves the end effector a few millimetres per step, sideways or
    outward/inward in the horizontal plane, so that the target point drifts
    toward the image centre. It never jumps toward a far-away target in one
    step: the nudge is hard-capped at `MAX_NUDGE_M` (5 mm) no matter what the
    model returns (T-14-12).
  - It does not reach in depth (`dz = 0`), never changes orientation (all
    rotation deltas 0), and never decides when to grasp (the gripper target
    is the current gripper value).
  - It was not learned, has no task knowledge beyond "move toward the
    point", and is NOT equivalent to the other backends' trained
    action-output policies. Every action it produces is tagged
    `@gemini-er-conversion` by `base_gemini.GeminiActionSourceBase`.

Input (from `GeminiActionSourceBase`):
`{"er_entries": <parsed ER JSON>, "image_shape": [height, width]}`. A bare
entry list or a single entry dict is also accepted. Verified ER formats
(ai.google.dev/gemini-api/docs/robotics-spatial, 2026-10-01):
  - point: `{"point": [y, x], "label": ...}`, normalized 0-1000
  - box:   `{"y": ymin, "x": xmin, "y2": ymax, "x2": xmax, "label": ...}`, 0-1000
  - box:   `{"box_2d": [ymin, xmin, ymax, xmax], ...}`. This is the older
    Gemini detection convention, accepted for robustness.
The first entry with a valid point or box wins. A box is reduced to its
centre.

Image-plane -> robot-frame mapping: ASSUMPTIONS, not calibration
----------------------------------------------------------------
This repo has NO wrist-camera intrinsics and NO camera-to-robot (hand-eye)
extrinsics. `control/stereo_calibration.json` covers only the overhead
AR0144 stereo pair's intrinsics and its left-right transform. The URDF has
no camera link. So this module does not pretend to back-project pixels into
metres. Instead:

  1. The target's pixel offset `(u right, v down)` from `SERVO_SETPOINT_YX`
     (the image centre) is scaled by `PIXELS_TO_EEF_GAIN` (m/px), then
     magnitude-clamped to `MAX_NUDGE_M`. Offsets within `DEADBAND_PX` count
     as centred and produce no motion.
  2. The direction comes from the arm's own URDF forward kinematics, not
     from fixed base axes. "Image up" is taken to mean radially outward:
     horizontally away from the base, along the current end-effector
     heading. "Image right" is taken to mean the horizontal tangent to the
     right of that heading. This is what a downward-looking wrist camera with
     wrist_roll ~ 0 sees, and it stays correct as the arm pans.
     `IMAGE_UP_SIGN` / `IMAGE_RIGHT_SIGN` flip each axis if the real mount is
     mirrored.
  3. The base-frame `(dx, dy)` goes to `eef_to_joint.solve_ik()` (Plan
     14-03's URDF-driven IK). That call clamps to the URDF limits intersected
     with the calibrated `joint_limits_deg`.

These assumptions (camera mount, sign of each axis, setpoint = image centre,
and the same unverified real-joint vs URDF sign caveat as 14-03) are
human-check items in 14-13-SUMMARY.md. All are single named constants, so
fixing one does not touch the logic.

Never raises: any malformed or missing point, bad image shape, bad joint
state, FK or IK failure returns `dict(joint_state)`, i.e. hold position.
`safety_validator.validate_action()` re-validates everything downstream.
This module is the first bound, never the only one.
"""

import math
import numbers

from vla_bridge.backends import eef_to_joint

# Gemini ER's documented coordinate range: [y, x] normalized to 0-1000.
ER_COORD_MAX = 1000.0
# Shape used only when neither the caller nor the ER output supplies one.
DEFAULT_IMAGE_SHAPE = (480, 640)
# Where the target should end up, in ER-normalized [y, x]: the image centre.
SERVO_SETPOINT_YX = (500.0, 500.0)

# Proportional gain: metres of end-effector nudge per pixel of target offset.
# 100 px (about 1/6 of a 640-wide frame) -> 5 mm, i.e. the cap.
PIXELS_TO_EEF_GAIN = 5e-5
# Hard cap on the horizontal nudge per step, regardless of ER output (T-14-12).
MAX_NUDGE_M = 0.005
# Pixel radius around the setpoint that counts as "centred" (no motion).
DEADBAND_PX = 8.0
# +1: image-up = radially outward from the base; image-right = right of heading.
IMAGE_UP_SIGN = 1.0
IMAGE_RIGHT_SIGN = 1.0
# Below this horizontal EEF distance from the base axis, the heading is undefined.
MIN_HEADING_RADIUS_M = 0.02


def _coord(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return None
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= ER_COORD_MAX:
        return None
    return value


def _entry_target(entry) -> tuple[float, float] | None:
    if not isinstance(entry, dict):
        return None
    point = entry.get("point")
    if isinstance(point, (list, tuple)) and len(point) == 2:
        y, x = (_coord(v) for v in point)
        if y is not None and x is not None:
            return y, x
    box = entry.get("box_2d")
    if isinstance(box, (list, tuple)) and len(box) == 4:
        coords = [_coord(v) for v in box]
        if None not in coords:
            ymin, xmin, ymax, xmax = coords
            return (ymin + ymax) / 2.0, (xmin + xmax) / 2.0
    if all(key in entry for key in ("y", "x", "y2", "x2")):
        coords = [_coord(entry[key]) for key in ("y", "x", "y2", "x2")]
        if None not in coords:
            ymin, xmin, ymax, xmax = coords
            return (ymin + ymax) / 2.0, (xmin + xmax) / 2.0
    return None


def extract_target_yx(er_response) -> tuple[float, float] | None:
    """First valid target in an ER response, as ER-normalized `(y, x)`, or `None`."""
    if isinstance(er_response, dict) and "er_entries" in er_response:
        entries = er_response["er_entries"]
    else:
        entries = er_response
    if isinstance(entries, dict):
        entries = [entries]
    if not isinstance(entries, list):
        return None
    for entry in entries:
        target = _entry_target(entry)
        if target is not None:
            return target
    return None


def _resolve_image_shape(er_response, image_shape) -> tuple[float, float] | None:
    candidate = image_shape
    if candidate is None and isinstance(er_response, dict):
        candidate = er_response.get("image_shape")
    if candidate is None:
        candidate = DEFAULT_IMAGE_SHAPE
    if not isinstance(candidate, (list, tuple)) or len(candidate) < 2:
        return None
    height, width = candidate[0], candidate[1]
    for value in (height, width):
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            return None
        if not math.isfinite(value) or value <= 0:
            return None
    return float(height), float(width)


def _horizontal_heading(joint_state) -> tuple[float, float] | None:
    """Unit horizontal direction from the base axis to the current EEF (URDF FK)."""
    x, y = eef_to_joint.forward_kinematics(joint_state)[:2]
    radius = math.hypot(x, y)
    if not math.isfinite(radius) or radius < MIN_HEADING_RADIUS_M:
        return None
    return x / radius, y / radius


def image_target_to_base_nudge(target_yx, image_shape, joint_state) -> tuple[float, float] | None:
    """Bounded base-frame `(dx, dy)` metres for one servo step, or `None` when the
    target is already inside the deadband (or the heading is undefined)."""
    height, width = image_shape
    u = (target_yx[1] - SERVO_SETPOINT_YX[1]) / ER_COORD_MAX * width   # +: target right of setpoint
    v = (target_yx[0] - SERVO_SETPOINT_YX[0]) / ER_COORD_MAX * height  # +: target below setpoint
    if math.hypot(u, v) <= DEADBAND_PX:
        return None
    heading = _horizontal_heading(joint_state)
    if heading is None:
        return None
    right = (heading[1], -heading[0])  # heading x up(+z): the horizontal right-hand side
    outward = IMAGE_UP_SIGN * (-v) * PIXELS_TO_EEF_GAIN
    sideways = IMAGE_RIGHT_SIGN * u * PIXELS_TO_EEF_GAIN
    dx = outward * heading[0] + sideways * right[0]
    dy = outward * heading[1] + sideways * right[1]
    norm = math.hypot(dx, dy)
    if not math.isfinite(norm):
        return None
    if norm > MAX_NUDGE_M:
        dx, dy = dx * MAX_NUDGE_M / norm, dy * MAX_NUDGE_M / norm
    return dx, dy


def convert_er_output_to_action(
    er_response,
    joint_state: dict,
    image_shape: tuple[int, int] | None = None,
    joint_limits_deg: dict | None = None,
) -> dict:
    """One bounded visual-servo step toward the ER target, as an absolute
    `JOINT_ORDER` dict. Hold (`dict(joint_state)`) on anything unusable; never raises.

    `image_shape` (height, width) of the frame the ER model saw. Precedence:
    this argument, then `er_response["image_shape"]`, then
    `DEFAULT_IMAGE_SHAPE`.
    """
    try:
        target = extract_target_yx(er_response)
        if target is None:
            return dict(joint_state)
        shape = _resolve_image_shape(er_response, image_shape)
        if shape is None:
            return dict(joint_state)
        nudge = image_target_to_base_nudge(target, shape, joint_state)
        if nudge is None:
            return dict(joint_state)
        eef_delta = {
            "dx": float(nudge[0]),
            "dy": float(nudge[1]),
            "dz": 0.0,
            "droll": 0.0,
            "dpitch": 0.0,
            "dyaw": 0.0,
            "gripper": float(joint_state["gripper"]),  # held: this layer never grasps
        }
        return eef_to_joint.solve_ik(joint_state, eef_delta, joint_limits_deg=joint_limits_deg)
    except Exception:  # never let a conversion problem crash the live loop
        return dict(joint_state)


def make_er_convert_fn(joint_limits_deg: dict | None = None, image_shape: tuple[int, int] | None = None):
    """`convert_fn(er_output, joint_state)` for `GeminiActionSourceBase`, bound to
    the backend's calibrated joint limits (as `build(args, camera_names,
    joint_limits_deg)` receives them)."""

    def convert(er_output, joint_state):
        return convert_er_output_to_action(
            er_output, joint_state, image_shape=image_shape, joint_limits_deg=joint_limits_deg
        )

    return convert


__all__ = [
    "DEADBAND_PX",
    "DEFAULT_IMAGE_SHAPE",
    "MAX_NUDGE_M",
    "PIXELS_TO_EEF_GAIN",
    "convert_er_output_to_action",
    "extract_target_yx",
    "image_target_to_base_nudge",
    "make_er_convert_fn",
]
