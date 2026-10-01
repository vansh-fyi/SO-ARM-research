"""`--backend rdt1b`: RDT-1B over the Tier C resident HTTP adapter (VLAB-09).

Same lane and shape as `backends/octo.py` / `backends/openvla.py`:
- Colab side: `vla_bridge/rdt1b_server.py` serves POST /action. It returns the
  checkpoint's unified-action slots for SO-101, one step of RDT's 64-step chunk.
- Transport: `base_http_adapter.HttpAdapterActionSource`.
- Conversion: this module's `convert_fn` turns those slots into an absolute
  `JOINT_ORDER` target BEFORE `run_episode()`'s unmodified
  `safety_validator.validate_action()`.

The conversion is chosen from the checkpoint's reviewed profile in
`backend_configs/rdt1b.json["checkpoints"][<id>]`. A checkpoint without one is
refused rather than guessed.

- `joint_position`: every reviewed checkpoint, confirmed on 2026-10-01 against
  RDT's own `configs/state_vec.py`, `scripts/agilex_model.py`,
  `data/hdf5_vla_dataset.py` and README at the pinned commit, plus the weights'
  128-D output layer (see rdt1b_server.py). Slots 0-4 (`right_arm_joint_{i}_pos`)
  are ABSOLUTE arm joint targets in radians, and slot 10 (`right_gripper_open`)
  is the gripper, min-max normalized to [0, 1]. So no IK is needed. The values
  map one to one onto `JOINT_ORDER`, with only a unit change: radians -> degrees,
  and `lo + g * (hi - lo)` percent for the gripper. A literal identity would read
  0.5 rad as 0.5 degrees.
- `eef_pose`: the fallback the plan requires if a checkpoint's slots turn out to
  be end-effector based. RDT's EEF slots hold an ABSOLUTE pose:
  `[x, y, z]` (slots 30-32, metres) plus a 6-D rotation (slots 33-38, the first two
  columns of the rotation matrix; `docs/test_6drot.py`), plus the gripper.
  `rdt_eef_pose_to_eef_delta()` turns that into `eef_to_joint.solve_ik()`'s delta
  at the current FK pose. No reviewed checkpoint uses it today, and
  `rdt1b_server.py` only serves joint-position profiles (an EEF profile's state
  input would be the EEF pose, which the wire contract does not carry).

Endpoint: `--server-address` (the per-session ngrok `https://` URL) or
`backend_configs/rdt1b.json`'s `"endpoint_url"`. `--checkpoint` (or the config's
`"checkpoint"`) must name the checkpoint the Colab server was launched with.
Each step's logged `model_version` names the checkpoint the server actually ran.
"""

import math

from vla_bridge.action_contract import JOINT_ORDER
from vla_bridge.backends import BackendHandles, base_http_adapter, eef_to_joint, load_backend_config, register

ACTION_SPACES = ("joint_position", "eef_pose")
EEF_POSE_DIM = 10  # x, y, z, 6-D rotation, gripper
JOINT_UNITS = ("rad", "deg")


def _numbers(raw_action, dim: int) -> list[float]:
    if not isinstance(raw_action, (list, tuple)) or len(raw_action) != dim:
        raise ValueError(f"expected a list of {dim} numbers, got {raw_action!r}")
    return [float(v) for v in raw_action]


def _gripper_percent(value: float, gripper_norm_to_percent) -> float:
    lo, hi = (float(v) for v in gripper_norm_to_percent)
    return lo + value * (hi - lo) if math.isfinite(value) else float("nan")


def joint_position_action(raw_action, joint_units: str, gripper_norm_to_percent) -> dict[str, float]:
    """RDT's 6 SO-101 slots (5 arm joints, gripper_open) -> `JOINT_ORDER` dict (no IK).

    Raises `ValueError`/`TypeError` on a malformed action, which makes the HTTP
    source hold position. Non-finite values pass through to the validator.
    """
    values = _numbers(raw_action, len(JOINT_ORDER))
    arm = [math.degrees(v) if joint_units == "rad" else v for v in values[:5]]
    return dict(zip(JOINT_ORDER, arm + [_gripper_percent(values[5], gripper_norm_to_percent)]))


def rotation_from_ortho6d(ortho6d):
    """`docs/test_6drot.py`'s `compute_rotation_matrix_from_ortho6d`, for one vector."""
    import numpy as np

    a1, a2 = np.asarray(ortho6d[:3], dtype=float), np.asarray(ortho6d[3:6], dtype=float)
    x = a1 / max(np.linalg.norm(a1), 1e-8)
    z = np.cross(x, a2)
    z = z / max(np.linalg.norm(z), 1e-8)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


def rdt_eef_pose_to_eef_delta(raw_action, joint_state, gripper_norm_to_percent) -> dict[str, float]:
    """Absolute RDT EEF pose `[x, y, z, r6d (6), gripper]` -> `solve_ik()`'s `eef_delta`.

    The delta is taken at `forward_kinematics(joint_state)`: position difference
    in metres, and the extrinsic rotation `R_goal @ R_current^T` (applied on the
    left by `solve_ik()`) as x-y-z angles. Raises `ValueError`/`TypeError` on a
    malformed action. A non-finite pose comes back as NaN, which `solve_ik()`
    treats as hold-pose.
    """
    values = _numbers(raw_action, EEF_POSE_DIM)
    gripper = _gripper_percent(values[9], gripper_norm_to_percent)
    if not all(math.isfinite(v) for v in values[:9]):
        nan = float("nan")
        return {"dx": nan, "dy": nan, "dz": nan, "droll": nan, "dpitch": nan, "dyaw": nan, "gripper": gripper}
    from scipy.spatial.transform import Rotation

    current = eef_to_joint.forward_kinematics(joint_state)
    goal = Rotation.from_matrix(rotation_from_ortho6d(values[3:9]))
    droll, dpitch, dyaw = (float(a) for a in (goal * Rotation.from_euler("xyz", current[3:]).inv()).as_euler("xyz"))
    return {"dx": values[0] - current[0], "dy": values[1] - current[1], "dz": values[2] - current[2],
            "droll": droll, "dpitch": dpitch, "dyaw": dyaw, "gripper": gripper}


def make_convert_fn(profile: dict, joint_limits_deg: dict | None):
    """`convert_fn(raw_action, joint_state)` for `build_http_adapter_backend()`,
    selected by the profile's `action_space`."""
    action_space = profile.get("action_space")
    gripper_range = profile.get("gripper_norm_to_percent")
    if action_space not in ACTION_SPACES:
        raise ValueError(f"unknown RDT action_space {action_space!r}; expected one of {ACTION_SPACES}")
    if not isinstance(gripper_range, (list, tuple)) or len(gripper_range) != 2:
        raise ValueError("an RDT profile needs gripper_norm_to_percent: [lo, hi]")
    if action_space == "joint_position":
        joint_units = profile.get("joint_units")
        if joint_units not in JOINT_UNITS:
            raise ValueError(f"a joint_position RDT profile needs joint_units in {JOINT_UNITS}")

        def _convert(raw_action, joint_state):
            return joint_position_action(raw_action, joint_units, gripper_range)

    else:

        def _convert(raw_action, joint_state):
            delta = rdt_eef_pose_to_eef_delta(raw_action, joint_state, gripper_range)
            return eef_to_joint.solve_ik(joint_state, delta, joint_limits_deg=joint_limits_deg)

    return _convert


@register("rdt1b")
def build(args, camera_names: dict[int, str], joint_limits_deg: dict) -> BackendHandles:
    config = load_backend_config("rdt1b") or {}
    endpoint_url = args.server_address or config.get("endpoint_url")
    if not endpoint_url:
        raise SystemExit(
            "--backend rdt1b needs --server-address or backend_configs/rdt1b.json's 'endpoint_url' field"
        )
    if not str(endpoint_url).startswith(("http://", "https://")):
        raise SystemExit(
            f"--backend rdt1b endpoint {endpoint_url!r} must be the adapter's http(s):// URL "
            "(the ngrok HTTP tunnel to rdt1b_server.py), not a gRPC host:port"
        )
    checkpoint = args.checkpoint or config.get("checkpoint")
    profile = (config.get("checkpoints") or {}).get(checkpoint) if checkpoint else None
    if not isinstance(profile, dict) or profile.get("action_space") not in ACTION_SPACES:
        raise SystemExit(
            f"--backend rdt1b checkpoint {checkpoint!r} has no reviewed action_space profile in "
            f"backend_configs/rdt1b.json 'checkpoints' (known: {sorted(config.get('checkpoints') or {})})"
        )
    return base_http_adapter.build_http_adapter_backend(
        endpoint_url, checkpoint, args, camera_names,
        convert_fn=make_convert_fn(profile, joint_limits_deg),
    )


__all__ = ["ACTION_SPACES", "build", "joint_position_action", "make_convert_fn", "rdt_eef_pose_to_eef_delta",
           "rotation_from_ortho6d"]
