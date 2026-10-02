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

Optional `convert_fn` (Plan 14-07): a backend may pass
`convert_fn(action, joint_state) -> dict`, and the action source is then a
`ConvertedBridgeActionSource` that applies it to each action the
`BridgeActionSource` returns. Omitting it (every backend before GR00T N1.7)
returns the plain `BridgeActionSource`, exactly as before. Scope: on this lane
`pop_validated_action()` has ALREADY run `safety_validator.validate_action()`
on the raw, positionally mapped joint dict before `convert_fn` sees it, so a
gRPC-lane `convert_fn` must be a joint-space -> joint-space transform (a shape,
unit or ordering gate). A Cartesian -> joint conversion would need a
pre-validation hook in `robot_client.pop_validated_action()`, which no backend
needs today. `run_episode()` still validates the converted action with the
unmodified validator, and a `convert_fn` exception holds position.
"""

from collections.abc import Callable

from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

from vla_bridge import robot_client, safety_validator
from vla_bridge.backends import BackendHandles


def build_grpc_action_source(
    policy_type: str,
    checkpoint: str,
    args,
    camera_names: dict[int, str],
    joint_limits_deg: dict,
    convert_fn: "Callable[[dict, dict], dict] | None" = None,
    camera_map: "dict[str, str] | None" = None,
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

    `convert_fn`, when given, wraps the source in `ConvertedBridgeActionSource`
    (see the module docstring for its joint-space-only scope).

    Camera wiring: `args.camera_map` (the `--camera-map` CLI override) wins,
    then the backend's own `camera_map` (from its backend config), then
    `robot_client.DEFAULT_CAMERA_MAP`. See `robot_client.connect_bridge()`.
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
        camera_map=resolve_camera_map(args, camera_map),
    )
    if client is None:
        return None
    action_source = robot_client.BridgeActionSource(client, checkpoint, joint_limits_deg)
    if convert_fn is not None:
        action_source = ConvertedBridgeActionSource(action_source, convert_fn, checkpoint)
    return BackendHandles(
        action_source=action_source,
        robot=client.robot,
        # Reuse the SAME StereoSplitCamera already feeding the policy (Gap 1).
        stereo_camera=client._stereo_camera,
        skip_camera_names=frozenset({"overhead"}),
        # lerobot 0.6.1 RobotClient.stop(): sets shutdown_event, disconnects
        # the robot, closes the gRPC channel -- the pre-Phase-14 teardown.
        shutdown=client.stop,
    )


def resolve_camera_map(args, config_camera_map: "dict[str, str] | None") -> "dict[str, str] | None":
    """`--camera-map` beats the backend config's `camera_map`; `None` means the default.

    A malformed config map exits with the config error rather than reaching
    the robot; the CLI form is already validated at argument parsing.
    """
    cli_map = getattr(args, "camera_map", None)
    if isinstance(cli_map, dict):
        return cli_map
    if config_camera_map is None:
        return None
    try:
        return robot_client.validate_camera_map(config_camera_map)
    except ValueError as e:
        raise SystemExit(f"backend config camera_map: {e}") from e


class ConvertedBridgeActionSource:
    """`BridgeActionSource` plus a per-action `convert_fn` (Plan 14-07).

    `get_action()` delegates to the wrapped source, then returns
    `convert_fn(action, joint_state)`. It never raises: if `convert_fn` raises,
    it holds position (`<checkpoint>@action-conversion-error-holding-position`),
    like `HttpAdapterActionSource`. `last_details` is the wrapped source's dict,
    so `run_episode()` logs the same raw model output and bridge flags. Any other
    attribute (`client`, `checkpoint`, ...) is read from the wrapped source.
    """

    def __init__(self, inner, convert_fn: "Callable[[dict, dict], dict]", checkpoint: str):
        self.inner = inner
        self.convert_fn = convert_fn
        self.checkpoint = checkpoint
        self.last_details = {}

    def get_action(self, joint_state: dict[str, float], instruction: str) -> tuple[dict[str, float], str]:
        action, model_version = self.inner.get_action(joint_state, instruction)
        details = getattr(self.inner, "last_details", None)
        self.last_details = details if isinstance(details, dict) else {}
        try:
            converted = self.convert_fn(action, joint_state)
        except Exception as e:  # noqa: BLE001 -- never raise into the control loop
            flags = list(self.last_details.get("bridge_flags") or [])
            self.last_details["bridge_flags"] = [*flags, f"action-conversion-error: {e}"]
            return dict(joint_state), f"{self.checkpoint}@action-conversion-error-holding-position"
        return converted, model_version

    def close(self) -> None:
        close = getattr(self.inner, "close", None)
        if callable(close):
            close()

    def __getattr__(self, name):
        inner = self.__dict__.get("inner")
        if inner is None:
            raise AttributeError(name)
        return getattr(inner, name)


__all__ = ["ConvertedBridgeActionSource", "build_grpc_action_source", "resolve_camera_map"]
