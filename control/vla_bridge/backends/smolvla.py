"""`--backend smolvla`: SmolVLA over the Colab `PolicyServer` gRPC bridge.

The Phase 11/12 production path, now one registry entry. Everything except
the `policy_type` string and the checkpoint is shared with the other Tier A/B
backends in `base_grpc.build_grpc_action_source()`.

Checkpoint resolution: `--checkpoint` wins; otherwise
`backend_configs/smolvla.json`'s `"checkpoint"` field
(`victorvanhalst/smolvla_so101_cube`, checked on the HF Hub on 2026-10-01:
public, lerobot `"type": "smolvla"`, inputs `observation.images.camera1/2/3`
matching `connect_bridge()`'s wrist + stereo-split wiring).

`"smolvla"` is in installed `lerobot==0.6.1`'s server-side
`async_inference/constants.py::SUPPORTED_POLICIES`.
"""

from vla_bridge.backends import BackendHandles, base_grpc, load_backend_config, register_grpc_backend


@register_grpc_backend("smolvla")
def build(args, camera_names: dict[int, str], joint_limits_deg: dict) -> "BackendHandles | None":
    """Returns `None` if the bridge is unreachable (see `build_grpc_action_source`)."""
    config = load_backend_config("smolvla") or {}
    checkpoint = args.checkpoint or config.get("checkpoint")
    if not checkpoint:
        raise SystemExit(
            "--backend smolvla needs --checkpoint or backend_configs/smolvla.json's "
            "'checkpoint' field"
        )
    return base_grpc.build_grpc_action_source(
        "smolvla", checkpoint, args, camera_names, joint_limits_deg,
        camera_map=config.get("camera_map"),
    )


__all__ = ["build"]
