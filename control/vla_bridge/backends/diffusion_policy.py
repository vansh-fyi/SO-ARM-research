"""`--backend diffusion`: Diffusion Policy over the Colab `PolicyServer` gRPC
bridge (VLAB-12, Tier A). It is the one non-VLA imitation-learning baseline in
Phase 14's backend set, included for contrast (14-CONTEXT.md D-00).

Same shape as `act.py` / `pi0.py`: only the `policy_type` string and the
checkpoint change, and everything else is shared in
`base_grpc.build_grpc_action_source()`. The registry key and `--backend`
value are `"diffusion"`, the exact `policy_type` lerobot dispatches on. The
config file is `backend_configs/diffusion_policy.json`.

Installed `lerobot==0.6.1` facts this relies on (read from the installed
source, 2026-10-01):
- `async_inference/constants.py::SUPPORTED_POLICIES` contains `"diffusion"`.
  The server resolves it with `get_policy_class("diffusion")` to
  `DiffusionPolicy`, which needs `diffusers` (`lerobot[diffusion]` extra) on
  the server.
- **The stock PolicyServer cannot run Diffusion Policy.** It calls
  `policy.predict_action_chunk(obs)` directly with one observation, so the
  state arrives as `(B, state_dim)`. `DiffusionModel.generate_actions()` reads
  `n_obs_steps` from `state.shape[1]` and asserts it equals
  `config.n_obs_steps`, so every inference fails. The server logs
  `Error in StreamActions` and the robot only holds position. This is
  upstream issue huggingface/lerobot#3861, with an open, unmerged fix PR
  #3886. The Colab side therefore runs `vla_bridge/diffusion_policy_server.py`,
  which applies PR #3886's fix before starting the same `PolicyServer`. See
  `policy_server.ipynb`'s Diffusion Policy section.
- **Diffusion Policy does not mask missing cameras.** `predict_action_chunk()`
  stacks `[batch[key] for key in self.config.image_features]`, like ACT. So the
  bridge must send exactly the declared cameras: the shipped SO-101 checkpoint
  declares `camera1` + `camera2`, and `backend_configs/diffusion_policy.json`'s
  `camera_map` sends our wrist frame and the left AR0144 overhead half under
  those names. Another checkpoint needs a matching `--camera-map`.
- Diffusion Policy ignores the language instruction. `task` still travels in
  the observation, but nothing in its preprocessor or model reads it.

Actions flow through the unchanged `BridgeActionSource` ->
`pop_validated_action()` -> `safety_validator.validate_action()`. This module
adds no path around it (T-14-02).
"""

from vla_bridge.backends import BackendHandles, base_grpc, load_backend_config, register_grpc_backend

POLICY_TYPE = "diffusion"
CONFIG_NAME = "diffusion_policy"


@register_grpc_backend(POLICY_TYPE)
def build(args, camera_names: dict[int, str], joint_limits_deg: dict) -> "BackendHandles | None":
    """Returns `None` if the bridge is unreachable (see `build_grpc_action_source`)."""
    config = load_backend_config(CONFIG_NAME) or {}
    checkpoint = args.checkpoint or config.get("checkpoint")
    if not checkpoint:
        raise SystemExit(
            f"--backend {POLICY_TYPE} needs --checkpoint or backend_configs/{CONFIG_NAME}.json's "
            "'checkpoint' field"
        )
    return base_grpc.build_grpc_action_source(
        POLICY_TYPE, checkpoint, args, camera_names, joint_limits_deg,
        camera_map=config.get("camera_map"),
    )


__all__ = ["build"]
