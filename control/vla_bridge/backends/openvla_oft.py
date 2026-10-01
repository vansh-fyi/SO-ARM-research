"""`--backend openvla-oft`: OpenVLA-OFT over the Tier C resident HTTP adapter (VLAB-07).

This is the first backend on the HTTP-adapter lane (14-CONTEXT.md D-05).
Pieces:
- Colab side: `vla_bridge/openvla_oft_server.py` serves POST /action.
- Transport: `base_http_adapter.HttpAdapterActionSource`.
- Conversion: this module's `convert_fn` turns the model's raw 7-D delta into
  an absolute `JOINT_ORDER` target through `eef_to_joint.solve_ik()`, BEFORE
  `run_episode()`'s unmodified `safety_validator.validate_action()`.

Raw action -> `eef_delta`, as verified against live upstream source on
2026-10-01:
- OpenVLA-OFT's LIBERO checkpoints emit un-normalized robosuite OSC_POSE
  deltas in the controller's normalized [-1, 1] input units. LIBERO builds
  its env with `controller="OSC_POSE"` (LIBERO/libero/libero/envs/env_wrapper.py).
- robosuite v1.4.1 `controllers/config/osc_pose.json` maps that input range
  to `output_max = [0.05]*3 m + [0.5]*3 rad`, giving `OSC_POSITION_SCALE_M`
  and `OSC_ROTATION_SCALE_RAD`.
- The rotation part is an axis-angle vector that robosuite applies in the
  base frame (`control_utils.set_goal_orientation`: `R_goal = R(delta) @ R`).
  It is converted exactly to the extrinsic x-y-z angles `eef_to_joint`
  expects.
- Gripper: the model emits [0, 1] with 1 = open. openvla-oft's
  `run_libero_eval.process_action` binarizes it with `sign(2g - 1)`. This
  repo's SO-101 gripper is percent-open (increasing = opening, verified on
  hardware in Phase 10 TWIN-07). So open maps to 100, close to 0, and an
  exact 0.5 to 50.

The checkpoint is a LIBERO/Franka-sim policy, so these actions are
out-of-distribution for SO-101. The frame mapping is an identity: the
model's base frame is taken to be the SO-101 URDF `base` frame. That is
unverified on hardware and is a human check in 14-03-SUMMARY.md.

Endpoint: `--server-address` (the per-session ngrok `https://` URL) or
`backend_configs/openvla_oft.json`'s `"endpoint_url"`. The checkpoint
string is informational only (`model_version` / io_logger); the real
checkpoint loads server-side on Colab.
"""

import math

from vla_bridge.backends import BackendHandles, base_http_adapter, eef_to_joint, load_backend_config, register

# robosuite v1.4.1 controllers/config/osc_pose.json output_max/output_min magnitudes.
OSC_POSITION_SCALE_M = 0.05
OSC_ROTATION_SCALE_RAD = 0.5
ACTION_DIM = 7


def openvla_oft_action_to_eef_delta(raw_action) -> dict[str, float]:
    """Raw OpenVLA-OFT 7-float action -> `eef_to_joint.solve_ik()`'s `eef_delta`.

    Raises `ValueError`/`TypeError` on a malformed action, which makes the
    HTTP source hold position. Non-finite values pass through, and
    `solve_ik()` treats them as hold-position.
    """
    if not isinstance(raw_action, (list, tuple)) or len(raw_action) != ACTION_DIM:
        raise ValueError(f"expected a list of {ACTION_DIM} numbers, got {raw_action!r}")
    values = [float(v) for v in raw_action]

    rotvec = [v * OSC_ROTATION_SCALE_RAD for v in values[3:6]]
    if all(math.isfinite(v) for v in rotvec):
        from scipy.spatial.transform import Rotation

        droll, dpitch, dyaw = (float(a) for a in Rotation.from_rotvec(rotvec).as_euler("xyz"))
    else:
        droll = dpitch = dyaw = float("nan")

    grip = values[6]
    if math.isfinite(grip):
        sign = (2.0 * grip - 1.0 > 0) - (2.0 * grip - 1.0 < 0)
        gripper = 50.0 * (1.0 + sign)
    else:
        gripper = float("nan")  # solve_ik keeps the current gripper

    return {
        "dx": values[0] * OSC_POSITION_SCALE_M,
        "dy": values[1] * OSC_POSITION_SCALE_M,
        "dz": values[2] * OSC_POSITION_SCALE_M,
        "droll": droll,
        "dpitch": dpitch,
        "dyaw": dyaw,
        "gripper": gripper,
    }


def make_convert_fn(joint_limits_deg: dict | None):
    """`convert_fn(raw_action, joint_state)` for `build_http_adapter_backend()`."""

    def _convert(raw_action, joint_state):
        return eef_to_joint.solve_ik(
            joint_state, openvla_oft_action_to_eef_delta(raw_action), joint_limits_deg=joint_limits_deg
        )

    return _convert


@register("openvla-oft")
def build(args, camera_names: dict[int, str], joint_limits_deg: dict) -> BackendHandles:
    config = load_backend_config("openvla_oft") or {}
    endpoint_url = args.server_address or config.get("endpoint_url")
    if not endpoint_url:
        raise SystemExit(
            "--backend openvla-oft needs --server-address or backend_configs/openvla_oft.json's "
            "'endpoint_url' field"
        )
    if not str(endpoint_url).startswith(("http://", "https://")):
        raise SystemExit(
            f"--backend openvla-oft endpoint {endpoint_url!r} must be the adapter's http(s):// URL "
            "(the ngrok HTTP tunnel to openvla_oft_server.py), not a gRPC host:port"
        )
    checkpoint = args.checkpoint or config.get("checkpoint") or "openvla-oft"
    return base_http_adapter.build_http_adapter_backend(
        endpoint_url, checkpoint, args, camera_names, convert_fn=make_convert_fn(joint_limits_deg)
    )


__all__ = ["build", "make_convert_fn", "openvla_oft_action_to_eef_delta"]
