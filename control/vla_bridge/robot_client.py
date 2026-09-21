"""Validated bridge wrapper around `lerobot.async_inference.RobotClient` (VLAHW-02).

Wires a real, network-bridged SmolVLA policy (Colab-hosted `PolicyServer`,
reached over the D-01 gRPC bridge) into `run_vla_episode.py`'s pluggable
`ActionSource` seam -- but deliberately does NOT call the installed
`RobotClient`'s own `control_loop_action()`/`control_loop()` methods. Reading
the installed `lerobot==0.6.1` source directly (not just its CLI docs)
confirmed both of those call `self.robot.send_action(...)` internally,
completely bypassing `vla_bridge.safety_validator.validate_action()` -- using
either would silently violate VLAHW-02 ("VLA output actions pass a safety
validator ... before being sent to the real robot").

Instead, this module uses only the library's public queue/observation
primitives (`control_loop_observation()`, `action_queue`,
`_action_tensor_to_action_dict()`) and keeps every actual `send_action()`
call in the caller's hands (`run_vla_episode.py`'s control loop), strictly
after `pop_validated_action()` has run the candidate through
`safety_validator.validate_action()`. The validation gate and the execution
gate are visibly separate in this code, on purpose (T-11-09).
"""

import queue
import time
from typing import Any

import grpc

from lerobot.async_inference.configs import RobotClientConfig
from lerobot.async_inference.robot_client import RobotClient

from vla_bridge import safety_validator


def connect_bridge(
    server_address: str,
    checkpoint: str,
    robot_config: Any,
    task: str,
    actions_per_chunk: int = 50,
    chunk_size_threshold: float = 0.5,
    policy_device: str = "cuda",
):
    """Connect to a Colab-hosted `PolicyServer` bridge.

    Constructing `RobotClient(config)` connects the PHYSICAL robot as a side
    effect inside its own `__init__` -- the returned client's `.robot`
    attribute is the one true robot handle for the rest of the caller's
    control loop; callers must NOT also construct a second, separate
    `SO101Follower` for this path.

    Returns the connected, handshaken `RobotClient` on success, or `None`
    (never an exception) if the bridge handshake fails -- calling `.stop()`
    on the half-connected client first, for cleanup.
    """
    config = RobotClientConfig(
        policy_type="smolvla",
        pretrained_name_or_path=checkpoint,
        robot=robot_config,
        actions_per_chunk=actions_per_chunk,
        task=task,
        server_address=server_address,
        policy_device=policy_device,
        chunk_size_threshold=chunk_size_threshold,
    )
    client = RobotClient(config)

    if not client.start():
        client.stop()
        return None

    return client


def pop_validated_action(
    client,
    safety_validate_fn,
    joint_limits_deg: dict[str, tuple[float, float]],
    current_state: dict[str, float],
    prev_action: dict[str, float] | None,
    dt_s: float,
) -> tuple[dict[str, float], list[str], dict[str, float], float]:
    """Pop one candidate action from the bridge client's queue and validate it.

    Performs NO `send_action` call itself -- the caller (`BridgeActionSource`
    or `run_vla_episode.py`) is responsible for that, keeping the validation
    gate and the execution gate visibly separate in the code (T-11-09).

    Returns `(validated_action, flags, raw_action, obs_age_s)`. On an empty
    queue, returns `(dict(current_state), ["no action available, holding
    position"], {}, 0.0)` rather than raising.
    """
    with client.action_queue_lock:
        try:
            timed_action = client.action_queue.get_nowait()
        except queue.Empty:
            return dict(current_state), ["no action available, holding position"], {}, 0.0

    raw_action = client._action_tensor_to_action_dict(timed_action.get_action())
    # Network round-trip staleness -- feeds safety_validator's
    # STALE_ACTION_S/STALE_OBSERVATION_S holds.
    obs_age_s = time.time() - timed_action.get_timestamp()

    validated_action, flags = safety_validate_fn(
        raw_action,
        current_state,
        joint_limits_deg,
        obs_age_s=obs_age_s,
        prev_action=prev_action,
        dt_s=dt_s,
    )
    return validated_action, flags, raw_action, obs_age_s


class BridgeActionSource:
    """`run_vla_episode.py`'s `ActionSource` interface, backed by a real,
    network-bridged SmolVLA policy over an already-connected `RobotClient`.

    Deliberately never calls `client.control_loop_action()` or
    `client.control_loop()` -- both call `client.robot.send_action()`
    internally, bypassing `safety_validator.validate_action()`, which would
    silently violate VLAHW-02.
    """

    def __init__(
        self,
        client,
        checkpoint: str,
        joint_limits_deg: dict[str, tuple[float, float]],
        dt_s: float = 0.5,
    ):
        self.client = client
        self.checkpoint = checkpoint
        self.joint_limits_deg = joint_limits_deg
        self._dt_s = dt_s
        self._last_action: dict[str, float] | None = None

    def get_action(
        self, joint_state: dict[str, float], instruction: str
    ) -> tuple[dict[str, float], str]:
        try:
            self.client.control_loop_observation(task=instruction)
        except (grpc.RpcError, ConnectionError, RuntimeError):
            return joint_state, f"{self.checkpoint}@bridge-error-holding-position"

        try:
            validated_action, _flags, _raw_action, _obs_age_s = pop_validated_action(
                self.client,
                safety_validator.validate_action,
                self.joint_limits_deg,
                joint_state,
                self._last_action,
                self._dt_s,
            )
        except (grpc.RpcError, ConnectionError, RuntimeError):
            return joint_state, f"{self.checkpoint}@bridge-error-holding-position"

        self._last_action = validated_action
        return validated_action, f"{self.checkpoint}@unknown"


__all__ = [
    "connect_bridge",
    "pop_validated_action",
    "BridgeActionSource",
]
