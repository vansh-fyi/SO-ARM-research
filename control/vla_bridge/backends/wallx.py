"""`--backend wallx`: Wall-X / WALL-OSS (VLAB-04, Tier B -> confirmed Tier C).

Tier decision, read from the installed `lerobot==0.6.1` source (2026-10-01),
not from docs:
- `lerobot/async_inference/constants.py:26` is
  `SUPPORTED_POLICIES = ["act", "smolvla", "diffusion", "tdmpc", "vqbet", "pi0", "pi05", "groot"]`,
  enforced in `async_inference/policy_server.py:130` (`SendPolicyInstructions`
  raises "Policy type ... not supported"). No Wall-X spelling (`wall_x`,
  `wallx`, `wall_oss`) appears anywhere under `lerobot/async_inference/`.
- lerobot itself does ship Wall-X: `lerobot/policies/wall_x/` registers
  `@PreTrainedConfig.register_subclass("wall_x")` and `WallXPolicy.name =
  "wall_x"`, and `get_policy_class("wall_x")` resolves it. So `"wall_x"` is
  the policy_type string, and the gap is the async server's dispatch table
  only.
So the gRPC bridge would reject the handshake, and Wall-X runs on the Tier C
resident-HTTP-adapter lane: `vla_bridge/wallx_server.py` on Colab loads it
through lerobot's own `WallXPolicy` (no X-Square-Robot/wall-x GitHub install
and no CUDA-operator compile; see that file's docstring).

`build()` re-checks the dispatch table on every call, so the branch follows
the installed source rather than this note:
- `policy_server_dispatches("wall_x")` True (a future lerobot) AND a gRPC
  `host:port` endpoint -> `base_grpc.build_grpc_action_source("wall_x", ...)`,
  with no `convert_fn`, because Wall-X SO-101 actions are already absolute
  joints.
- otherwise an `http(s)://` endpoint -> `base_http_adapter.
  build_http_adapter_backend(..., convert_fn=None)`. The adapter returns the
  `JOINT_ORDER` dict itself.
- a `host:port` endpoint while dispatch is unsupported -> `SystemExit`,
  instead of a handshake the server would refuse.

Action space: joint-space, not EEF. Every SO-101 `wall_x` checkpoint found on
the HF Hub (2026-10-01) has `observation.state` [6] -> `action` [6] over
`shoulder_pan.pos .. gripper.pos`. The default
(Zekai-Chen/wallx_pick_place_3cam_200ep) is in degree mode, matching
`action_contract.ACTION_UNITS`. So `eef_to_joint.solve_ik` is NOT used.

Actions still pass through `run_episode()`'s unmodified
`safety_validator.validate_action()` (T-14-02).
"""

from lerobot.async_inference import constants as async_constants

from vla_bridge.backends import BackendHandles, base_grpc, base_http_adapter, load_backend_config, register

POLICY_TYPE = "wall_x"
BACKEND_NAME = "wallx"
CONFIG_NAME = "wallx"


def policy_server_dispatches(policy_type: str) -> bool:
    """True if the installed lerobot's async `PolicyServer` accepts `policy_type`."""
    return policy_type in async_constants.SUPPORTED_POLICIES


def _is_http(endpoint: str) -> bool:
    return str(endpoint).startswith(("http://", "https://"))


@register(BACKEND_NAME)
def build(args, camera_names: dict[int, str], joint_limits_deg: dict) -> "BackendHandles | None":
    config = load_backend_config(CONFIG_NAME) or {}
    endpoint = args.server_address or config.get("endpoint_url")
    if not endpoint:
        raise SystemExit(
            f"--backend {BACKEND_NAME} needs --server-address (the wallx_server.py ngrok https:// URL) "
            f"or backend_configs/{CONFIG_NAME}.json's 'endpoint_url' field"
        )
    checkpoint = args.checkpoint or config.get("checkpoint")
    if not checkpoint:
        raise SystemExit(
            f"--backend {BACKEND_NAME} needs --checkpoint or backend_configs/{CONFIG_NAME}.json's "
            "'checkpoint' field"
        )

    if _is_http(endpoint):
        return base_http_adapter.build_http_adapter_backend(
            endpoint, checkpoint, args, camera_names, convert_fn=None
        )
    if policy_server_dispatches(POLICY_TYPE):
        return base_grpc.build_grpc_action_source(
            POLICY_TYPE, checkpoint, args, camera_names, joint_limits_deg, convert_fn=None
        )
    raise SystemExit(
        f"--backend {BACKEND_NAME}: endpoint {endpoint!r} looks like a gRPC host:port, but the installed "
        f"lerobot's async_inference SUPPORTED_POLICIES ({async_constants.SUPPORTED_POLICIES}) does not "
        f"include {POLICY_TYPE!r}, so a PolicyServer would reject the handshake. Run "
        "vla_bridge/wallx_server.py on Colab and pass its http(s):// tunnel URL instead."
    )


__all__ = ["BACKEND_NAME", "CONFIG_NAME", "POLICY_TYPE", "build", "policy_server_dispatches"]
