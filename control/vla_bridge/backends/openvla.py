"""`--backend openvla`: OpenVLA over the Tier C resident HTTP adapter (VLAB-06).

Same lane and shape as `backends/openvla_oft.py` (Plan 14-03):
- Colab side: `vla_bridge/openvla_server.py` serves POST /action and returns
  OpenVLA's raw 7-D `predict_action()` output.
- Transport: `base_http_adapter.HttpAdapterActionSource`.
- Conversion: this module's `convert_fn` turns that raw action into an absolute
  `JOINT_ORDER` target BEFORE `run_episode()`'s unmodified
  `safety_validator.validate_action()`.

The conversion depends on the checkpoint, so it is chosen per checkpoint from
`backend_configs/openvla.json["checkpoints"][<id>]["action_space"]`. A
checkpoint without a reviewed profile is refused rather than guessed. Both
spaces were verified against live upstream source on 2026-10-01:

- `absolute_joint_degrees_so101_7d` (the default,
  `toasta18/openvla-so101-cylinder`). An OpenVLA-7B fine-tune on a real
  SO-101 lerobot dataset whose `action` names are exactly
  `shoulder_pan.pos .. gripper.pos`, in degree mode with gripper percent.
  The RLDS conversion padded it to OpenVLA's 7 slots as [5 arm joints,
  constant 0, gripper], which the checkpoint's q01/q99 stats confirm. So slots
  0-4 and 6 map straight onto JOINT_ORDER and slot 5 is dropped. **No IK.**
  This is the SO-101 joint-space fine-tune 14-RESEARCH Open Question 3 asks
  each backend to prefer.
- `delta_eef_robosuite_osc` (the official MIT
  `openvla/openvla-7b-finetuned-libero-*`). LIBERO's robosuite OSC_POSE delta,
  the same units OpenVLA-OFT's LIBERO checkpoints emit:
  - Position: [-1, 1] input scaled by 0.05 m.
  - Rotation: axis-angle scaled by 0.5 rad, applied in the base frame.
  - Gripper: [0, 1] with 1 = open. openvla's `libero_dataset_transform` maps
    LIBERO's -1 = open to +1 = open, and `run_libero_eval` binarizes it with
    `sign(2g - 1)`.

  The conversion therefore reuses `openvla_oft.openvla_oft_action_to_eef_delta()`
  unchanged and then calls `eef_to_joint.solve_ik()`. The LIBERO->SO-101 frame
  mapping is the same unverified identity as 14-03's.

Endpoint: `--server-address` (the per-session ngrok `https://` URL) or
`backend_configs/openvla.json`'s `"endpoint_url"`. `--checkpoint` (or the
config's `"checkpoint"`) must name the checkpoint the Colab server was
launched with, because it selects the conversion. Each step's logged
`model_version` names the checkpoint the server actually ran.
"""

from vla_bridge.action_contract import JOINT_ORDER
from vla_bridge.backends import BackendHandles, base_http_adapter, eef_to_joint, load_backend_config, openvla_oft, register

ACTION_DIM = 7
ACTION_SPACES = ("absolute_joint_degrees_so101_7d", "delta_eef_robosuite_osc")
# Raw slot -> JOINT_ORDER joint for the SO-101 joint-space fine-tune; slot 5 is the pad.
SO101_SLOTS = (0, 1, 2, 3, 4, 6)


def so101_joint_action(raw_action) -> dict[str, float]:
    """Raw 7-slot OpenVLA SO-101 action -> absolute `JOINT_ORDER` dict.

    Raises `ValueError`/`TypeError` on a malformed action, which makes the HTTP
    source hold position. Non-finite values pass through to the validator.
    """
    if not isinstance(raw_action, (list, tuple)) or len(raw_action) != ACTION_DIM:
        raise ValueError(f"expected a list of {ACTION_DIM} numbers, got {raw_action!r}")
    values = [float(v) for v in raw_action]
    return {joint: values[slot] for joint, slot in zip(JOINT_ORDER, SO101_SLOTS)}


def make_convert_fn(action_space: str, joint_limits_deg: dict | None):
    """`convert_fn(raw_action, joint_state)` for `build_http_adapter_backend()`."""
    if action_space == "absolute_joint_degrees_so101_7d":

        def _convert(raw_action, joint_state):
            return so101_joint_action(raw_action)

    elif action_space == "delta_eef_robosuite_osc":

        def _convert(raw_action, joint_state):
            return eef_to_joint.solve_ik(
                joint_state, openvla_oft.openvla_oft_action_to_eef_delta(raw_action),
                joint_limits_deg=joint_limits_deg,
            )

    else:
        raise ValueError(f"unknown OpenVLA action_space {action_space!r}; expected one of {ACTION_SPACES}")
    return _convert


@register("openvla")
def build(args, camera_names: dict[int, str], joint_limits_deg: dict) -> BackendHandles:
    config = load_backend_config("openvla") or {}
    endpoint_url = args.server_address or config.get("endpoint_url")
    if not endpoint_url:
        raise SystemExit(
            "--backend openvla needs --server-address or backend_configs/openvla.json's 'endpoint_url' field"
        )
    if not str(endpoint_url).startswith(("http://", "https://")):
        raise SystemExit(
            f"--backend openvla endpoint {endpoint_url!r} must be the adapter's http(s):// URL "
            "(the ngrok HTTP tunnel to openvla_server.py), not a gRPC host:port"
        )
    checkpoint = args.checkpoint or config.get("checkpoint")
    profile = (config.get("checkpoints") or {}).get(checkpoint) if checkpoint else None
    if not isinstance(profile, dict) or profile.get("action_space") not in ACTION_SPACES:
        raise SystemExit(
            f"--backend openvla checkpoint {checkpoint!r} has no reviewed action_space profile in "
            f"backend_configs/openvla.json 'checkpoints' (known: {sorted(config.get('checkpoints') or {})})"
        )
    return base_http_adapter.build_http_adapter_backend(
        endpoint_url, checkpoint, args, camera_names,
        convert_fn=make_convert_fn(profile["action_space"], joint_limits_deg),
    )


__all__ = ["ACTION_SPACES", "build", "make_convert_fn", "so101_joint_action"]
