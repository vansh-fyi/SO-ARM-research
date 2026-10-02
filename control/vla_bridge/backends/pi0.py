"""`--backend pi0` / `--backend pi05`: Physical Intelligence's pi0 and pi0.5 over
the Colab `PolicyServer` gRPC bridge (VLAB-03, Tier A).

Same shape as `smolvla.py`: only the `policy_type` string and the checkpoint
change, and everything else is shared in `base_grpc.build_grpc_action_source()`.
There is no new Colab-side code. The notebook's existing `PolicyServer`
loads whatever `policy_type`/checkpoint the client sends in the handshake.

Installed `lerobot==0.6.1` facts this relies on (read from the installed
source, 2026-10-01):
- `async_inference/constants.py::SUPPORTED_POLICIES` contains `"pi0"` and
  `"pi05"`. The server resolves them with `get_policy_class()` to
  `PI0Policy` / `PI05Policy`, then loads the checkpoint's own
  `policy_preprocessor.json` / `policy_postprocessor.json`.
- **Cameras.** The bridge sends the camera names in the backend config's
  `camera_map` (or `--camera-map`): the shipped checkpoints declare `camera1` +
  `camera2`, which get our wrist frame and the left AR0144 overhead half. A
  declared camera that is not sent is masked by pi0/pi05 as an empty camera,
  but a sent name the checkpoint does not declare is a KeyError in
  `prepare_raw_observation()`, so the map must only use declared names.
- pi0/pi05 truncate the predicted chunk to the checkpoint's
  `output_features["action"]` dim (6 for the shipped checkpoints), and
  `actions_per_chunk=50` stays within their `chunk_size=50`.

Checkpoint resolution: `--checkpoint` wins; otherwise
`backend_configs/<pi0|pi05>.json`'s `"checkpoint"`. Both shipped checkpoints
are SO-101 joint-space fine-tunes whose state/action are exactly
`JOINT_ORDER` (`.pos`), so there is no IK or EEF conversion layer. The
selection evidence is in `checkpoint_candidates.md`. The configs'
`"verified_revision"` records the sha that was checked. The bridge's
`RemotePolicyConfig` has no revision field, so the server loads the repo's
current `main`.

Colab prerequisite: the pi0/pi05 tokenizer step loads
`google/paligemma-3b-pt-224`, which is gated on the HF Hub, so the
notebook's HF token must belong to an account that accepted its license.

Actions flow through the unchanged `BridgeActionSource` ->
`pop_validated_action()` -> `safety_validator.validate_action()`. This module
adds no path around it (T-14-02).
"""

from vla_bridge.backends import BackendHandles, base_grpc, load_backend_config, register_grpc_backend


def _make_build(policy_type: str):
    def build(args, camera_names: dict[int, str], joint_limits_deg: dict) -> "BackendHandles | None":
        """Returns `None` if the bridge is unreachable (see `build_grpc_action_source`)."""
        config = load_backend_config(policy_type) or {}
        checkpoint = args.checkpoint or config.get("checkpoint")
        if not checkpoint:
            raise SystemExit(
                f"--backend {policy_type} needs --checkpoint or backend_configs/{policy_type}.json's "
                "'checkpoint' field"
            )
        return base_grpc.build_grpc_action_source(
            policy_type, checkpoint, args, camera_names, joint_limits_deg,
            camera_map=config.get("camera_map"),
        )

    build.__name__ = build.__qualname__ = f"build_{policy_type}"
    return build


build_pi0 = register_grpc_backend("pi0")(_make_build("pi0"))
build_pi05 = register_grpc_backend("pi05")(_make_build("pi05"))


__all__ = ["build_pi0", "build_pi05"]
