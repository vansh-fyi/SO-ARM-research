"""`--backend groot-n15`: NVIDIA GR00T N1.5 over the Tier C resident HTTP adapter (VLAB-11).

Same lane and shape as `backends/rdt1b.py` / `backends/octo.py`:
- Colab side: `vla_bridge/groot_n15_server.py` serves POST /action. It runs the
  checkpoint through NVIDIA's own `Isaac-GR00T` package (`Gr00tPolicy`, pinned
  n1.5-release commit) and returns one step of the un-normalized action chunk
  as 6 floats in `JOINT_ORDER` (`single_arm` [5] then `gripper` [1]).
- Transport: `base_http_adapter.HttpAdapterActionSource`.
- Conversion: this module's own `convert_fn`, BEFORE `run_episode()`'s
  unmodified `safety_validator.validate_action()`.

Deliberately independent of the GR00T N1.7 backend (14-RESEARCH Pitfall 3,
T-14-10). N1.7 rides the lerobot gRPC bridge with its own
`relative_ee_conversion.py` gate. This module imports neither of those, nor
lerobot: one adapter class with a version flag is exactly the failure mode
Pitfall 3 warns about.

Action convention, confirmed on 2026-10-01 rather than assumed (see
groot_n15_server.py for the full evidence):
- SO-101 N1.5 fine-tunes use Isaac-GR00T's `so100_dualcam` data config
  (NVIDIA's SO-101 tutorial). Its `min_max` normalization is un-applied by
  `Gr00tPolicy.get_action()`, so actions come back in the dataset's own units.
- Every SO-101 N1.5 checkpoint's `experiment_cfg/metadata.json` marks
  `single_arm` and `gripper` `absolute: true`.
- NVIDIA's own real-robot client (`examples/SO-100/eval_lerobot.py`) sends
  each chunk step straight to `robot.send_action()`.
So the action is an ABSOLUTE joint target: no delta, no IK, and the joint state
is never re-applied. The only open question is the unit mode, which is per
checkpoint. The reviewed profile pins `action_units: degrees` (current lerobot
`so_follower` degree mode, the mode `action_contract.py` speaks), and the
server re-derives it from the checkpoint's own statistics at startup. So the
conversion is a true identity: the 6 floats become the `JOINT_ORDER` dict.
Range-normalized (+/-100), radian and pre-0.2 offset-calibration checkpoints
are refused, not converted (see deferred-items.md).

Endpoint: `--server-address` (the per-session ngrok `https://` URL) or
`backend_configs/groot_n15.json`'s `"endpoint_url"`. `--checkpoint` (or the
config's `"checkpoint"`) must name the checkpoint the Colab server was launched
with, and must have a reviewed profile. Each step's logged `model_version`
names the checkpoint the server actually ran.
"""

import numbers

from vla_bridge.action_contract import JOINT_ORDER
from vla_bridge.backends import BackendHandles, base_http_adapter, load_backend_config, register

BACKEND_NAME = "groot-n15"
CONFIG_NAME = "groot_n15"
ACTION_SPACE = "absolute_joint"
ACTION_UNITS = "degrees"


def absolute_joint_action(raw_action) -> dict[str, float]:
    """6 absolute joint targets (`single_arm` [5] + `gripper` [1], degrees and
    gripper percent) -> `JOINT_ORDER` dict. An identity reshape.

    Raises `ValueError`/`TypeError` on a malformed action, which makes the HTTP
    source hold position. Non-finite values pass through to the validator,
    which rejects them.
    """
    if not isinstance(raw_action, (list, tuple)) or len(raw_action) != len(JOINT_ORDER):
        raise ValueError(f"expected a list of {len(JOINT_ORDER)} numbers, got {raw_action!r}")
    values = []
    for value in raw_action:
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise TypeError(f"expected a number, got {value!r}")
        values.append(float(value))
    return dict(zip(JOINT_ORDER, values))


def make_convert_fn(profile: dict):
    """`convert_fn(raw_action, joint_state)` for `build_http_adapter_backend()`.

    Only an `absolute_joint` profile in `degrees` is accepted: anything else
    would need a conversion this backend does not do, so it is refused rather
    than passed through as if it were degrees.
    """
    action_space = profile.get("action_space")
    action_units = profile.get("action_units")
    if action_space != ACTION_SPACE:
        raise ValueError(f"GR00T N1.5 action_space {action_space!r} is not supported; only {ACTION_SPACE!r} "
                         "(absolute joint targets, confirmed for SO-101 N1.5 fine-tunes) is")
    if action_units != ACTION_UNITS:
        raise ValueError(f"GR00T N1.5 action_units {action_units!r} is not supported; only {ACTION_UNITS!r} "
                         "(lerobot so_follower degree mode = action_contract.py) maps one to one")

    def _convert(raw_action, joint_state):
        # joint_state is unused on purpose: the targets are absolute (never re-applied).
        return absolute_joint_action(raw_action)

    return _convert


@register(BACKEND_NAME)
def build(args, camera_names: dict[int, str], joint_limits_deg: dict) -> BackendHandles:
    config = load_backend_config(CONFIG_NAME) or {}
    endpoint_url = args.server_address or config.get("endpoint_url")
    if not endpoint_url:
        raise SystemExit(
            f"--backend {BACKEND_NAME} needs --server-address or backend_configs/{CONFIG_NAME}.json's "
            "'endpoint_url' field"
        )
    if not str(endpoint_url).startswith(("http://", "https://")):
        raise SystemExit(
            f"--backend {BACKEND_NAME} endpoint {endpoint_url!r} must be the adapter's http(s):// URL "
            "(the ngrok HTTP tunnel to groot_n15_server.py), not a gRPC host:port"
        )
    checkpoint = args.checkpoint or config.get("checkpoint")
    profile = (config.get("checkpoints") or {}).get(checkpoint) if checkpoint else None
    if not isinstance(profile, dict):
        raise SystemExit(
            f"--backend {BACKEND_NAME} checkpoint {checkpoint!r} has no reviewed profile in "
            f"backend_configs/{CONFIG_NAME}.json 'checkpoints' (known: {sorted(config.get('checkpoints') or {})}). "
            "The base nvidia/GR00T-N1.5-3B has no SO-101 (new_embodiment) head or statistics."
        )
    try:
        convert_fn = make_convert_fn(profile)
    except ValueError as e:
        raise SystemExit(f"--backend {BACKEND_NAME}: backend_configs/{CONFIG_NAME}.json [{checkpoint}]: {e}") from e
    return base_http_adapter.build_http_adapter_backend(
        endpoint_url, checkpoint, args, camera_names, convert_fn=convert_fn,
    )


__all__ = ["ACTION_SPACE", "ACTION_UNITS", "BACKEND_NAME", "CONFIG_NAME", "absolute_joint_action", "build",
           "make_convert_fn"]
