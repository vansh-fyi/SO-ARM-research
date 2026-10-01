"""`--backend groot-n17`: NVIDIA GR00T N1.7 over the Colab `PolicyServer` gRPC
bridge (VLAB-10, Tier B -> confirmed gRPC tier).

Tier decision, read from the installed `lerobot==0.6.1` source (2026-10-01),
not from docs:
- `lerobot/async_inference/constants.py:26` is
  `SUPPORTED_POLICIES = ["act", "smolvla", "diffusion", "tdmpc", "vqbet", "pi0", "pi05", "groot"]`,
  enforced in `async_inference/policy_server.py:130` (`SendPolicyInstructions`).
  The GitHub-issue list 14-RESEARCH.md quoted (without `"groot"`) is older
  than the installed release.
- `policy_server.py` resolves it with `get_policy_class("groot")`, which is
  `lerobot/policies/groot/modeling_groot.py::GrootPolicy`. That wraps GR00T
  **N1.7** only: `configuration_groot.py` sets `GROOT_N1_7_BASE_MODEL =
  "nvidia/GR00T-N1.7-3B"`, `max_state_dim = max_action_dim = 132`, and rejects
  any N1.5 checkpoint ("GR00T N1.5 support was removed from LeRobot").
So GR00T N1.7 rides the existing gRPC bridge with `policy_type="groot"`; the
Tier C HTTP-adapter fallback is not needed. `build()` re-checks the dispatch
table at call time and refuses (instead of handshaking into a server-side
`ValueError`) if a lerobot upgrade ever drops `"groot"`.

Pitfalls found on the same code path, and what this backend does about them:
- **Action space.** SO-101 N1.7 checkpoints are joint-space (`single_arm`
  RELATIVE/NON_EEF + `gripper` ABSOLUTE, or plain absolute). The server's
  postprocessor turns them into ABSOLUTE joint targets, so the client must NOT
  apply a relative-EE/IK conversion. `backend_configs/groot_n17.json` declares
  `action_space: absolute_joint` + `action_units: degrees`, and the wired
  `convert_fn` (`relative_ee_conversion.py`) is the explicit gate for that.
- **Stock server cannot decode relative N1.7 chunks.** `PolicyServer.
  _predict_action_chunk()` post-processes one `(B, D)` action at a time, and
  `GrootN17ActionDecodeStep` raises `NotImplementedError("... cannot decode
  native relative actions one step at a time ...")` for native relative-action
  checkpoints (most SO-101 N1.7 fine-tunes). The Colab side therefore runs
  `vla_bridge/groot_n17_server.py`, which decodes the whole chunk in one call.
  Absolute-action checkpoints work on either server.
- **Cameras.** GR00T tolerates missing views (`GrootN17PackInputsStep` feeds
  whichever declared cameras arrive), but the server indexes
  `policy.config.image_features[key]` for every delivered key
  (`async_inference/helpers.py::prepare_raw_observation`), and today's bridge
  delivers only `observation.images.camera1`. No public SO-101 N1.7 checkpoint
  declares `camera1`, so no default checkpoint ships; `--checkpoint` is required
  until the bridge follow-up in `deferred-items.md` lands.

Actions flow through the unchanged `BridgeActionSource` ->
`pop_validated_action()` -> `safety_validator.validate_action()`, then the
convert_fn gate, then `run_episode()`'s own `validate_action()` (T-14-08).
"""

from lerobot.async_inference import constants as async_constants

from vla_bridge.backends import (
    BackendHandles,
    base_grpc,
    load_backend_config,
    register_grpc_backend,
    relative_ee_conversion,
)

POLICY_TYPE = "groot"
BACKEND_NAME = "groot-n17"
CONFIG_NAME = "groot_n17"


def policy_server_dispatches(policy_type: str) -> bool:
    """True if the installed lerobot's async `PolicyServer` accepts `policy_type`."""
    return policy_type in async_constants.SUPPORTED_POLICIES


@register_grpc_backend(BACKEND_NAME)
def build(args, camera_names: dict[int, str], joint_limits_deg: dict) -> "BackendHandles | None":
    """Returns `None` if the bridge is unreachable (see `build_grpc_action_source`)."""
    if not policy_server_dispatches(POLICY_TYPE):
        raise SystemExit(
            f"--backend {BACKEND_NAME}: the installed lerobot's async_inference SUPPORTED_POLICIES "
            f"({async_constants.SUPPORTED_POLICIES}) does not include {POLICY_TYPE!r}; the PolicyServer "
            "would reject the handshake. Re-verify the dispatch table before using this backend."
        )
    config = load_backend_config(CONFIG_NAME) or {}
    try:
        convert_fn = relative_ee_conversion.make_convert_fn(
            config.get("action_space"), config.get("action_units")
        )
    except ValueError as e:
        raise SystemExit(f"--backend {BACKEND_NAME}: backend_configs/{CONFIG_NAME}.json: {e}") from e
    checkpoint = args.checkpoint or config.get("checkpoint")
    if not checkpoint:
        raise SystemExit(
            f"--backend {BACKEND_NAME} needs --checkpoint or backend_configs/{CONFIG_NAME}.json's "
            "'checkpoint' field. None is shipped: no public SO-101 GR00T N1.7 checkpoint declares "
            "camera1, the only camera today's gRPC bridge delivers. See "
            f"backend_configs/{CONFIG_NAME}.json's notes."
        )
    return base_grpc.build_grpc_action_source(
        POLICY_TYPE, checkpoint, args, camera_names, joint_limits_deg, convert_fn=convert_fn
    )


__all__ = ["BACKEND_NAME", "CONFIG_NAME", "POLICY_TYPE", "build", "policy_server_dispatches"]
