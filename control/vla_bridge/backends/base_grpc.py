"""Shared construction helper for the Tier A/B gRPC-bridge lane (14-CONTEXT.md D-01/D-04).

Every backend that rides the existing Colab-hosted LeRobot
`async_inference.PolicyServer` bridge (SmolVLA today; pi0/pi05, ACT,
Diffusion Policy, and -- if the installed lerobot's async dispatch supports
them -- GR00T N1.7/Wall-X) differs from SmolVLA only by the `policy_type`
string and checkpoint it sends in the handshake. This module is the single
place that bridge is built, so per-backend modules stay one-call thin.

The returned `BackendHandles.action_source` is the UNMODIFIED
`robot_client.BridgeActionSource`: never calls `client.control_loop_action()`
/ `client.control_loop()` (both call `robot.send_action()` internally,
bypassing `safety_validator.validate_action()` -- VLAHW-02/T-11-09).

`policy_type` must be a value the Colab-side `PolicyServer` accepts:
installed `lerobot==0.6.1`'s `async_inference/constants.py::SUPPORTED_POLICIES`
is `["act", "smolvla", "diffusion", "tdmpc", "vqbet", "pi0", "pi05", "groot"]`,
enforced server-side (an unsupported value fails the handshake, so this
function returns `None`, never raises).

Depth is untouched here (D-03).
"""

from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

from vla_bridge import robot_client, safety_validator
from vla_bridge.backends import BackendHandles


def build_grpc_action_source(
    policy_type: str,
    checkpoint: str,
    args,
    camera_names: dict[int, str],
    joint_limits_deg: dict,
) -> "BackendHandles | None":
    """Connect the gRPC bridge for `policy_type`/`checkpoint` and wrap it.

    Mirrors `run_vla_episode.py`'s pre-Phase-14 bridge-path construction
    exactly. `args` needs `.port`, `.robot_id`, `.server_address`,
    `.instruction`, `.stereo_camera_index`.

    Returns `None` if the bridge is unreachable (handshake failed) -- the
    caller handles that exactly like today's `bridge_unreachable` path.
    `connect_bridge()` constructs AND connects the physical robot inside
    `RobotClient.__init__`, so `client.robot` is the one true robot handle;
    callers must not construct a second `SO101Follower`.
    """
    robot_config = SOFollowerRobotConfig(
        port=args.port,
        id=args.robot_id,
        max_relative_target=safety_validator.MAX_RELATIVE_TARGET_DEG,
    )
    # `_build_camera_names()` assigns the FIRST camera index to "wrist" positionally.
    wrist_camera_index = next((idx for idx, name in camera_names.items() if name == "wrist"), None)
    client = robot_client.connect_bridge(
        args.server_address,
        checkpoint,
        robot_config=robot_config,
        task=args.instruction,
        policy_device="cuda",
        stereo_camera_index=args.stereo_camera_index,
        wrist_camera_index=wrist_camera_index,
        policy_type=policy_type,
    )
    if client is None:
        return None
    return BackendHandles(
        action_source=robot_client.BridgeActionSource(client, checkpoint, joint_limits_deg),
        robot=client.robot,
        # Reuse the SAME StereoSplitCamera already feeding the policy (Gap 1).
        stereo_camera=client._stereo_camera,
        skip_camera_names=frozenset({"overhead"}),
        # lerobot 0.6.1 RobotClient.stop(): sets shutdown_event, disconnects
        # the robot, closes the gRPC channel -- the pre-Phase-14 teardown.
        shutdown=client.stop,
    )


__all__ = ["build_grpc_action_source"]
