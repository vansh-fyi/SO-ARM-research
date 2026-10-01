"""Shared end-effector-delta -> absolute-joint-target conversion (Plan 14-03, VLAB-07).

Several Tier C backends output Cartesian end-effector deltas, not joint
positions: OpenVLA-OFT's LIBERO checkpoints, OpenVLA, Octo's OXE base, and
possibly GR00T N1.7 (14-RESEARCH.md Pattern 3). `action_contract.JOINT_ORDER`
and `ACTION_UNITS` describe joint space only, so a backend's `convert_fn` has
to turn the delta into an absolute `JOINT_ORDER` dict BEFORE
`safety_validator.validate_action()` sees it.

Kinematic chain
---------------
The chain is NOT hand-derived. It is loaded at runtime from Phase 10's
regenerated, verified URDF:

    So-101/So-101.urdf  (URDF_PATH below, resolved relative to this file)

It goes through `ikpy` (`ikpy==4.1.0`, pinned in control/requirements.txt)
and `ikpy.chain.Chain.from_urdf_file()`. The URDF is a tree:
`right_gripper` branches into both the `parallel_gripper_mechanism` jaws and
the `eef` tool frame. So the path is given explicitly as `_CHAIN_ELEMENTS`:
`base` -> the 5 revolute arm joints -> `right_hand_to_gripper` ->
`gripper_to_eef` -> `eef`. Link lengths, joint axes, and mechanical limits
all come from that file. `_load_chain()` also checks that the active joints
are exactly `JOINT_ORDER[:5]`.

ikpy supplies URDF parsing and forward kinematics. The inverse solver in
`solve_ik()` is a small damped-least-squares (Levenberg-Marquardt-style)
loop over ikpy's FK frames, as Plan 14-03 Task 1 specifies. It is not
ikpy's own scipy-optimizer IK.

Units and conventions
---------------------
- Joints come in and go out in `action_contract` units: arm joints in
  degrees, gripper in percent 0-100. The URDF works in radians, and degree 0
  is taken to equal URDF angle 0. `scripts/calibration_utils.py` documents
  the calibrated midpoint (degree 0) as MuJoCo's `qpos=0` home pose, and
  `WRIST_ROLL_LIMIT_DEG` is the URDF `wrist_roll` limit converted to degrees.
  The SIGN of each real joint against its URDF axis has never been checked
  on hardware. Phase 10 checked limits and the gripper direction only. This
  is listed as a human check in 14-03-SUMMARY.md.
- `eef_delta` keys:
  - `dx, dy, dz`: metres, in the URDF `base` frame.
  - `droll, dpitch, dyaw`: radians, an extrinsic base-frame x-y-z rotation,
    composed on the LEFT of the current orientation (`R_goal = R_delta @
    R_current`). This is the same left-multiplication robosuite's OSC
    controller applies to its axis-angle deltas.
  - `gripper`: an ABSOLUTE percent-open target, passed straight through
    (clamped to `GRIPPER_LIMIT_PCT`). It never goes through the arm solver.
  Backend-specific scaling, such as robosuite's normalized [-1, 1] ->
  metres/radians, belongs in the backend's own `convert_fn`, not here.
- SO-ARM101 has 5 arm DOF, so a full 6-D pose target is generally
  over-constrained. The solver minimises a weighted task-space error:
  position in metres, plus orientation in radians scaled by
  `ORIENTATION_WEIGHT`. Position therefore wins whenever both cannot be
  met.

Safety (T-14-05)
----------------
- Every solver iterate is clamped to the URDF mechanical limits intersected
  with the caller's calibrated `joint_limits_deg`, when given.
- A non-finite delta component holds the current arm pose.
- The output is always finite.
- Bad current state raises. A non-finite current joint raises `ValueError`
  and a missing arm joint raises `KeyError`, so
  `HttpAdapterActionSource` degrades to hold-position.
- `safety_validator.validate_action()` still re-validates everything
  downstream. This module is the first layer, never the only one.
"""

import math
from functools import lru_cache
from pathlib import Path

import numpy as np

from vla_bridge.action_contract import GRIPPER_LIMIT_PCT, JOINT_ORDER

# control/vla_bridge/backends/eef_to_joint.py -> repo root is parents[3].
URDF_PATH = Path(__file__).resolve().parents[3] / "So-101" / "So-101.urdf"

ARM_JOINTS = JOINT_ORDER[:5]

# Explicit link/joint path through the URDF tree (see module docstring).
_CHAIN_ELEMENTS = (
    "base",
    "shoulder_pan", "shoulder",
    "shoulder_lift", "upper_arm",
    "elbow_flex", "lower_arm",
    "wrist_flex", "wrist",
    "wrist_roll", "right_hand",
    "right_hand_to_gripper", "right_gripper",
    "gripper_to_eef", "eef",
)

# Task-space weighting: 0.1 rad of orientation error costs as much as 1 cm
# of position error.
ORIENTATION_WEIGHT = 0.1
# Levenberg-Marquardt damping (metres-scale task error), keeps steps bounded
# near singularities (e.g. a fully stretched arm for an unreachable target).
DAMPING = 0.02
# Per-iteration joint step cap (radians), for stability on large deltas.
MAX_STEP_RAD = 0.2


@lru_cache(maxsize=None)
def _load_chain(urdf_path: str = str(URDF_PATH)):
    """Parse the URDF once per process and verify the active joints."""
    from ikpy.chain import Chain  # lazy: backend discovery imports this module

    if not Path(urdf_path).exists():
        raise FileNotFoundError(
            f"SO-101 URDF not found at {urdf_path}. It is Phase 10's generated "
            "artifact; regenerate with `python scripts/mjcf_to_urdf.py` (see So-101/README.md)."
        )
    # Mask: ikpy's synthetic origin link + 5 revolute joints + 2 fixed joints.
    mask = [False] + [True] * len(ARM_JOINTS) + [False, False]
    chain = Chain.from_urdf_file(
        urdf_path,
        base_elements=list(_CHAIN_ELEMENTS),
        active_links_mask=mask,
        symbolic=False,
        name="so101_eef",
    )
    active = tuple(link.name for link, on in zip(chain.links, chain.active_links_mask) if on)
    if active != ARM_JOINTS or len(chain.links) != len(mask):
        raise RuntimeError(
            f"{urdf_path} chain to 'eef' has active joints {active} (links "
            f"{[link.name for link in chain.links]}), expected {ARM_JOINTS}"
        )
    return chain


def _active_indices(chain) -> list[int]:
    return [i for i, on in enumerate(chain.active_links_mask) if on]


def urdf_joint_limits_deg() -> dict[str, tuple[float, float]]:
    """The URDF's own mechanical limits for the 5 arm joints, in degrees."""
    chain = _load_chain()
    return {
        chain.links[i].name: (math.degrees(chain.links[i].bounds[0]), math.degrees(chain.links[i].bounds[1]))
        for i in _active_indices(chain)
    }


def _arm_vector_rad(joint_deg: dict[str, float]) -> np.ndarray:
    """The 5 arm joints (never the gripper), degrees -> URDF radians."""
    values = np.array([float(joint_deg[j]) for j in ARM_JOINTS], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"non-finite arm joint state: {dict(zip(ARM_JOINTS, values))}")
    return np.radians(values)


def _full_vector(chain, q_arm: np.ndarray) -> np.ndarray:
    full = np.zeros(len(chain.links))
    full[_active_indices(chain)] = q_arm
    return full


def _fk_matrix(chain, q_arm: np.ndarray) -> np.ndarray:
    return chain.forward_kinematics(_full_vector(chain, q_arm))


def _rpy_from_matrix(rotation: np.ndarray) -> tuple[float, float, float]:
    from scipy.spatial.transform import Rotation

    roll, pitch, yaw = Rotation.from_matrix(rotation).as_euler("xyz")
    return float(roll), float(pitch), float(yaw)


def forward_kinematics(joint_deg: dict[str, float]) -> tuple[float, float, float, float, float, float]:
    """End-effector (`eef` link) pose in the URDF `base` frame.

    Returns `(x, y, z, roll, pitch, yaw)` with metres and radians. The angles
    are extrinsic x-y-z Euler angles. Only the 5 arm joints are read.
    """
    chain = _load_chain()
    frame = _fk_matrix(chain, _arm_vector_rad(joint_deg))
    x, y, z = (float(v) for v in frame[:3, 3])
    return (x, y, z, *_rpy_from_matrix(frame[:3, :3]))


def _limits_rad(joint_limits_deg: dict | None) -> tuple[np.ndarray, np.ndarray]:
    """URDF mechanical limits intersected with caller calibrated limits."""
    urdf = urdf_joint_limits_deg()
    lo = np.empty(len(ARM_JOINTS))
    hi = np.empty(len(ARM_JOINTS))
    for k, joint in enumerate(ARM_JOINTS):
        j_lo, j_hi = urdf[joint]
        if joint_limits_deg and joint in joint_limits_deg:
            c_lo, c_hi = (float(v) for v in joint_limits_deg[joint])
            if math.isfinite(c_lo) and math.isfinite(c_hi) and max(j_lo, c_lo) <= min(j_hi, c_hi):
                j_lo, j_hi = max(j_lo, c_lo), min(j_hi, c_hi)
        lo[k], hi[k] = math.radians(j_lo), math.radians(j_hi)
    return lo, hi


def _jacobian_and_pose(chain, q_arm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Geometric 6x5 Jacobian (linear; angular) of `eef` plus its 4x4 frame.

    ikpy's frame for each revolute link is origin-translation, then
    origin-rotation, then a rotation of q about `link.rotation`. The joint
    axis therefore passes through that frame's origin with world direction
    `R @ link.rotation`.
    """
    frames = chain.forward_kinematics(_full_vector(chain, q_arm), full_kinematics=True)
    tip = frames[-1]
    p_tip = tip[:3, 3]
    jac = np.zeros((6, len(ARM_JOINTS)))
    for col, idx in enumerate(_active_indices(chain)):
        frame = frames[idx]
        axis = frame[:3, :3] @ np.asarray(chain.links[idx].rotation, dtype=float)
        jac[:3, col] = np.cross(axis, p_tip - frame[:3, 3])
        jac[3:, col] = axis
    return jac, tip


def _task_error(tip: np.ndarray, target_pos: np.ndarray, target_rot: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    pos_err = target_pos - tip[:3, 3]
    rot_err = Rotation.from_matrix(target_rot @ tip[:3, :3].T).as_rotvec()  # base frame
    return np.concatenate([pos_err, rot_err])


def solve_ik(
    current_joint_deg: dict[str, float],
    eef_delta: dict[str, float],
    max_iters: int = 50,
    tol: float = 1e-4,
    joint_limits_deg: dict | None = None,
) -> dict[str, float]:
    """Convert an end-effector delta into an absolute `JOINT_ORDER` target.

    The solve is seeded at `current_joint_deg`. Its target is
    `forward_kinematics(current_joint_deg)` offset by `eef_delta`, using the
    conventions in the module docstring. It runs projected damped-least-
    squares iterations, clamped every iteration to the URDF limits
    intersected with `joint_limits_deg` (e.g.
    `action_contract.load_joint_limits_deg()`). It returns the
    lowest-weighted-error iterate.

    `tol` is the stop threshold on the weighted task-space error norm. The
    gripper is `eef_delta["gripper"]` clamped to percent 0-100, or the
    current gripper if that value is missing or non-finite.
    """
    from scipy.spatial.transform import Rotation

    chain = _load_chain()
    q0 = _arm_vector_rad(current_joint_deg)  # raises on missing / non-finite
    lo, hi = _limits_rad(joint_limits_deg)

    gripper = eef_delta.get("gripper")
    try:
        gripper = float(gripper)
    except (TypeError, ValueError):
        gripper = None
    if gripper is None or not math.isfinite(gripper):
        gripper = float(current_joint_deg.get("gripper", GRIPPER_LIMIT_PCT[0]))
    gripper = float(min(max(gripper, GRIPPER_LIMIT_PCT[0]), GRIPPER_LIMIT_PCT[1]))

    delta = np.array(
        [float(eef_delta.get(k, 0.0)) for k in ("dx", "dy", "dz", "droll", "dpitch", "dyaw")],
        dtype=float,
    )
    q_seed = np.clip(q0, lo, hi)
    if np.all(np.isfinite(delta)):
        weights = np.array([1.0, 1.0, 1.0] + [ORIENTATION_WEIGHT] * 3)
        start = _fk_matrix(chain, q0)
        target_pos = start[:3, 3] + delta[:3]
        target_rot = Rotation.from_euler("xyz", delta[3:]).as_matrix() @ start[:3, :3]

        q = q_seed
        jac, tip = _jacobian_and_pose(chain, q)
        err = weights * _task_error(tip, target_pos, target_rot)
        best_q, best_norm = q, float(np.linalg.norm(err))
        damping_sq = DAMPING**2 * np.eye(6)
        for _ in range(max_iters):
            if not math.isfinite(best_norm) or best_norm < tol:
                break
            jw = weights[:, None] * jac
            step = jw.T @ np.linalg.solve(jw @ jw.T + damping_sq, err)
            peak = float(np.max(np.abs(step)))
            if not math.isfinite(peak):
                break
            if peak > MAX_STEP_RAD:
                step *= MAX_STEP_RAD / peak
            q_next = np.clip(q + step, lo, hi)
            if float(np.max(np.abs(q_next - q))) < 1e-9:
                break  # pinned against limits / converged as far as 5 DOF allows
            q = q_next
            jac, tip = _jacobian_and_pose(chain, q)
            err = weights * _task_error(tip, target_pos, target_rot)
            norm = float(np.linalg.norm(err))
            if math.isfinite(norm) and norm < best_norm:
                best_q, best_norm = q, norm
        q_out = best_q
    else:
        q_out = q_seed  # non-finite delta: hold the current arm pose

    if not np.all(np.isfinite(q_out)):
        q_out = q_seed
    result = {joint: float(math.degrees(value)) for joint, value in zip(ARM_JOINTS, q_out)}
    result["gripper"] = gripper
    return {joint: result[joint] for joint in JOINT_ORDER}


__all__ = ["URDF_PATH", "forward_kinematics", "solve_ik", "urdf_joint_limits_deg"]
