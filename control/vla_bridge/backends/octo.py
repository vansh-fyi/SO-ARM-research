"""`--backend octo`: Octo over the Tier C resident HTTP adapter (VLAB-08).

Same lane and shape as `backends/openvla.py` / `backends/openvla_oft.py`:
- Colab side: `vla_bridge/octo_server.py` serves POST /action and returns the
  first step of Octo's un-normalized action chunk.
- Transport: `base_http_adapter.HttpAdapterActionSource`.
- Conversion: this module's `convert_fn` turns that raw action into an absolute
  `JOINT_ORDER` target BEFORE `run_episode()`'s unmodified
  `safety_validator.validate_action()`.

The conversion is chosen from the checkpoint's reviewed profile in
`backend_configs/octo.json["checkpoints"][<id>]`. Its `action_space` was read
from that checkpoint's own `dataset_statistics.json` and dataset card, not
assumed from the model family (14-RESEARCH Assumption A4). A checkpoint without
a reviewed profile is refused rather than guessed.

- `eef_delta`: every reviewed checkpoint, verified live on 2026-10-01.
  - The default `sapanostic/octo-so101-3hz-finetune` is a real SO-101
    fine-tune. Its action is `[dx, dy, dz, droll, dpitch, dyaw, gripper]`:
    metres and radians, as differences of Pinocchio-FK poses on the SO101
    URDF, with the gripper absolute and min-max normalized over [-10, 100].
  - The official `rail-berkeley/octo-{small,base}-1.5` with `bridge_dataset`
    statistics use the same layout: WidowX state differences, gripper 1 = open.
  - Both are POSE DIFFERENCES: the angles are differences of roll/pitch/yaw.
    `eef_to_joint.solve_ik()` instead composes an extrinsic rotation on the
    left. So `octo_action_to_eef_delta()` turns `rpy + d` into that rotation
    exactly at the current FK pose, then `solve_ik()` runs.
  - Each profile's `gripper_norm_to_percent` maps the raw gripper value back
    to this rig's percent.
- `joint_absolute`: an identity pass-through for an Octo fine-tune whose action
  is 6 absolute `JOINT_ORDER` values (degrees, gripper percent). No reviewed
  official-format Octo checkpoint uses it today. The SO-101 joint-space Octo
  checkpoints on the Hub (`khanhnd61/octo*_so101*`) are a PyTorch port in an
  unofficial lerobot fork, not `octo-models/octo`.

Endpoint: `--server-address` (the per-session ngrok `https://` URL) or
`backend_configs/octo.json`'s `"endpoint_url"`. `--checkpoint` (or the config's
`"checkpoint"`) must name the checkpoint the Colab server was launched with,
because it selects the conversion. Each step's logged `model_version` names the
checkpoint the server actually ran.
"""

import math

from vla_bridge.action_contract import JOINT_ORDER
from vla_bridge.backends import BackendHandles, base_http_adapter, eef_to_joint, load_backend_config, register

EEF_ACTION_DIM = 7
ACTION_SPACES = ("eef_delta", "joint_absolute")


def _numbers(raw_action, dim: int) -> list[float]:
    if not isinstance(raw_action, (list, tuple)) or len(raw_action) != dim:
        raise ValueError(f"expected a list of {dim} numbers, got {raw_action!r}")
    return [float(v) for v in raw_action]


def joint_absolute_action(raw_action) -> dict[str, float]:
    """Raw 6-value joint-space action -> `JOINT_ORDER` dict (no IK).

    Raises `ValueError`/`TypeError` on a malformed action, which makes the HTTP
    source hold position. Non-finite values pass through to the validator.
    """
    return dict(zip(JOINT_ORDER, _numbers(raw_action, len(JOINT_ORDER))))


def octo_action_to_eef_delta(raw_action, joint_state, gripper_norm_to_percent) -> dict[str, float]:
    """Raw 7-D Octo pose-difference action -> `eef_to_joint.solve_ik()`'s `eef_delta`.

    The angular slots are differences of extrinsic x-y-z (roll, pitch, yaw)
    angles. The goal orientation is therefore `R(rpy_current + d)`, and the
    returned (droll, dpitch, dyaw) is the extrinsic rotation `R_goal @
    R_current^T` that `solve_ik()` applies on the left, evaluated at
    `forward_kinematics(joint_state)`. The gripper is
    `lo + g * (hi - lo)` with `(lo, hi) = gripper_norm_to_percent`.

    Raises `ValueError`/`TypeError` on a malformed action (the HTTP source
    then holds). Non-finite values come back as NaN, which `solve_ik()` treats
    as hold-pose / keep-gripper.
    """
    values = _numbers(raw_action, EEF_ACTION_DIM)
    angles = values[3:6]
    if all(math.isfinite(v) for v in angles):
        from scipy.spatial.transform import Rotation

        current = eef_to_joint.forward_kinematics(joint_state)[3:]
        goal = Rotation.from_euler("xyz", [c + d for c, d in zip(current, angles)])
        droll, dpitch, dyaw = (float(a) for a in (goal * Rotation.from_euler("xyz", current).inv()).as_euler("xyz"))
    else:
        droll = dpitch = dyaw = float("nan")
    lo, hi = (float(v) for v in gripper_norm_to_percent)
    grip = values[6]
    gripper = lo + grip * (hi - lo) if math.isfinite(grip) else float("nan")
    return {"dx": values[0], "dy": values[1], "dz": values[2],
            "droll": droll, "dpitch": dpitch, "dyaw": dyaw, "gripper": gripper}


def make_convert_fn(profile: dict, joint_limits_deg: dict | None):
    """`convert_fn(raw_action, joint_state)` for `build_http_adapter_backend()`,
    selected by the profile's `action_space`."""
    action_space = profile.get("action_space")
    if action_space == "joint_absolute":

        def _convert(raw_action, joint_state):
            return joint_absolute_action(raw_action)

    elif action_space == "eef_delta":
        gripper_range = profile.get("gripper_norm_to_percent")
        if not isinstance(gripper_range, (list, tuple)) or len(gripper_range) != 2:
            raise ValueError("an eef_delta Octo profile needs gripper_norm_to_percent: [lo, hi]")

        def _convert(raw_action, joint_state):
            delta = octo_action_to_eef_delta(raw_action, joint_state, gripper_range)
            return eef_to_joint.solve_ik(joint_state, delta, joint_limits_deg=joint_limits_deg)

    else:
        raise ValueError(f"unknown Octo action_space {action_space!r}; expected one of {ACTION_SPACES}")
    return _convert


@register("octo")
def build(args, camera_names: dict[int, str], joint_limits_deg: dict) -> BackendHandles:
    config = load_backend_config("octo") or {}
    endpoint_url = args.server_address or config.get("endpoint_url")
    if not endpoint_url:
        raise SystemExit(
            "--backend octo needs --server-address or backend_configs/octo.json's 'endpoint_url' field"
        )
    if not str(endpoint_url).startswith(("http://", "https://")):
        raise SystemExit(
            f"--backend octo endpoint {endpoint_url!r} must be the adapter's http(s):// URL "
            "(the ngrok HTTP tunnel to octo_server.py), not a gRPC host:port"
        )
    checkpoint = args.checkpoint or config.get("checkpoint")
    profile = (config.get("checkpoints") or {}).get(checkpoint) if checkpoint else None
    if not isinstance(profile, dict) or profile.get("action_space") not in ACTION_SPACES:
        raise SystemExit(
            f"--backend octo checkpoint {checkpoint!r} has no reviewed action_space profile in "
            f"backend_configs/octo.json 'checkpoints' (known: {sorted(config.get('checkpoints') or {})})"
        )
    return base_http_adapter.build_http_adapter_backend(
        endpoint_url, checkpoint, args, camera_names,
        convert_fn=make_convert_fn(profile, joint_limits_deg),
    )


__all__ = ["ACTION_SPACES", "build", "joint_absolute_action", "make_convert_fn", "octo_action_to_eef_delta"]
