"""`--backend act`: ACT (Action Chunking with Transformers) over the Colab
`PolicyServer` gRPC bridge (VLAB-05, Tier A).

Same shape as `smolvla.py` / `pi0.py`: only the `policy_type` string and the
checkpoint change, and everything else is shared in
`base_grpc.build_grpc_action_source()`. There is no new Colab-side code: the
notebook's existing `PolicyServer` loads whatever `policy_type`/checkpoint the
client sends in the handshake.

Installed `lerobot==0.6.1` facts this relies on (read from the installed
source, 2026-10-01):
- `async_inference/constants.py::SUPPORTED_POLICIES` contains `"act"`. The
  server resolves it with `get_policy_class("act")` to `ACTPolicy`, then loads
  the checkpoint's own `policy_preprocessor.json` / `policy_postprocessor.json`.
- **ACT does not mask missing cameras.** `ACTPolicy.predict_action_chunk()`
  builds `[batch[key] for key in self.config.image_features]`. pi0/pi05/SmolVLA
  instead fill a declared-but-absent camera with a masked empty image. So the
  bridge must send exactly the cameras the checkpoint declares: the shipped
  checkpoint declares only `camera1`, and `backend_configs/act.json`'s
  `camera_map` sends just our wrist frame under that name. A two-camera
  checkpoint needs `--camera-map` naming both.
- ACT ignores the language instruction. `task` still travels in the
  observation, but no ACT preprocessor step or model input reads it.
- `_get_action_chunk()` slices the chunk to `actions_per_chunk=50`, which is
  within the shipped checkpoint's `chunk_size=100`.

Checkpoint resolution: `--checkpoint` wins; otherwise
`backend_configs/act.json`'s `"checkpoint"`. The shipped checkpoint is an
SO-101 ACT fine-tune with a 6-D state/action and `camera1` as its only camera
(evidence in `checkpoint_candidates.md`). `"verified_revision"` records the sha that was
checked. The bridge's `RemotePolicyConfig` has no revision field, so the
server loads the repo's current `main`.

Actions flow through the unchanged `BridgeActionSource` ->
`pop_validated_action()` -> `safety_validator.validate_action()`. This module
adds no path around it (T-14-02).
"""

from vla_bridge.backends import BackendHandles, base_grpc, load_backend_config, register_grpc_backend

POLICY_TYPE = "act"


@register_grpc_backend(POLICY_TYPE)
def build(args, camera_names: dict[int, str], joint_limits_deg: dict) -> "BackendHandles | None":
    """Returns `None` if the bridge is unreachable (see `build_grpc_action_source`)."""
    config = load_backend_config(POLICY_TYPE) or {}
    checkpoint = args.checkpoint or config.get("checkpoint")
    if not checkpoint:
        raise SystemExit(
            f"--backend {POLICY_TYPE} needs --checkpoint or backend_configs/{POLICY_TYPE}.json's "
            "'checkpoint' field"
        )
    return base_grpc.build_grpc_action_source(
        POLICY_TYPE, checkpoint, args, camera_names, joint_limits_deg,
        camera_map=config.get("camera_map"),
    )


__all__ = ["build"]
