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

`receive_actions()` (the library's background thread that pulls action
chunks off the gRPC stream and fills `client.action_queue`) is started
explicitly here, in its own daemon thread. It does NOT call
`robot.send_action()` -- only `control_loop_action()`/`control_loop()` do
that -- so running it does not reintroduce the T-11-09 bypass this module
otherwise avoids. Without this thread, `action_queue` is never populated
and every `pop_validated_action()` call silently falls back to "hold
current position" forever (found live: an episode ran to completion with
the robot never moving, `action_queue` permanently empty). Because
`receive_actions()` calls `self.start_barrier.wait()` expecting a second
thread (the library's own `control_loop()`, which this module deliberately
never runs), `client.start_barrier` is replaced with a 1-party barrier
before starting the thread so it doesn't block forever waiting for a
partner that will never arrive.
"""

import queue
import threading
import time
from typing import Any

import grpc

from lerobot.async_inference.configs import RobotClientConfig
from lerobot.async_inference.robot_client import RobotClient
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

from vla_bridge import safety_validator
from vla_bridge.stereo_camera import StereoSplitCamera

# Gap 3 (blocker-severity, `12-UAT.md`): the vendored `lerobot` library's own
# `must_go`/`action_queue`-empty recovery path is not guaranteed to converge
# quickly once its `PolicyServer`'s `observations_similar()` filter starts
# silently dropping every non-`must_go` observation -- the live UAT run
# showed this take 6+ minutes to self-recover (steps ~192-241 of a 300-step
# episode, `obs_age_s` growing unbounded from 30s to 227s). At a typical 2 Hz
# control tick, `STALE_WATCHDOG_CONSECUTIVE_LIMIT` consecutive stale/no-action
# results is ~5 seconds -- well under that 6+ minute window -- before this
# project's own code forces the bridge out of the stuck state directly,
# rather than waiting on the vendored library's internal timing.
STALE_WATCHDOG_CONSECUTIVE_LIMIT = 10


def connect_bridge(
    server_address: str,
    checkpoint: str,
    robot_config: Any,
    task: str,
    actions_per_chunk: int = 50,
    chunk_size_threshold: float = 0.5,
    policy_device: str = "cuda",
    stereo_camera_index: int | str = 1,
    wrist_camera_index: int | None = None,
):
    """Connect to a Colab-hosted `PolicyServer` bridge.

    Constructing `RobotClient(config)` connects the PHYSICAL robot as a side
    effect inside its own `__init__` -- the returned client's `.robot`
    attribute is the one true robot handle for the rest of the caller's
    control loop; callers must NOT also construct a second, separate
    `SO101Follower` for this path.

    Also wires the AR0144 split-stereo camera feeds (Plan 11-03's camera
    mapping, `policy_server_launch.md`) into the returned client's
    observation-building step -- see `_wire_stereo_split_cameras()`.

    `wrist_camera_index`, when given, wires a real `camera1` (wrist) entry
    into `robot_config.cameras` via lerobot's own `OpenCVCameraConfig` --
    the standard camera-config route (`SOFollower.__init__` calls
    `make_cameras_from_configs(config.cameras)`, and `RobotClient.__init__`'s
    own `self.robot.connect()` then opens every configured camera, including
    this one) -- BEFORE `RobotClient(config)` is constructed below, since
    `RobotClient.__init__` connects the robot as a side effect. This closes
    a real gap found live this session: `camera1` was never wired into the
    observation dict at all (only `camera2`/`camera3`, the AR0144 split),
    which would have silently starved the VLA checkpoint of its wrist view.

    This is a genuinely SEPARATE `cv2.VideoCapture` open from
    `run_vla_episode.py`'s own local wrist capture (used for its IOLogger
    recording path), not a shared handle. Reading the installed lerobot
    camera-config source (`lerobot/cameras/camera.py`,
    `configuration_opencv.py`) found no built-in mechanism for two
    independently-registered camera configs to share one physical device
    instance the way `StereoSplitCamera` shares the AR0144's single open for
    camera2/camera3 -- sharing would require deeper surgery (monkey-patching
    the wrist `Camera` object's own `connect()`/`read()`, mirroring
    `_wire_stereo_split_cameras()`'s approach but for a single-camera
    config rather than a get_observation() patch). Whether a standard USB
    webcam driver tolerates two independent opens of the same index (unlike
    the AR0144 stereo pair, which does not, per `stereo_camera.py`'s own
    docstring) could NOT be verified empirically in this session -- no real
    hardware was available in this coding environment -- so the simpler
    independent-open path was chosen, per this task's own documented
    fallback. This MUST be re-verified against the real camera before a
    live episode: a second open failing would surface as an exception
    inside `RobotClient.__init__` (raised by the wrist `Camera.connect()`
    call), before the bridge handshake even starts.

    When `wrist_camera_index` is `None` (default), `robot_config.cameras`
    is left untouched -- unchanged prior behavior.

    Returns the connected, handshaken `RobotClient` on success, or `None`
    (never an exception) if the bridge handshake fails -- calling `.stop()`
    on the half-connected client first, for cleanup.
    """
    if wrist_camera_index is not None:
        robot_config.cameras["camera1"] = OpenCVCameraConfig(index_or_path=wrist_camera_index)

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
    _wire_stereo_split_cameras(client, stereo_camera_index=stereo_camera_index)

    if not client.start():
        client.stop()
        return None

    client.start_barrier = threading.Barrier(1)
    action_thread = threading.Thread(target=client.receive_actions, daemon=True)
    action_thread.start()
    client._action_receiver_thread = action_thread

    return client


def _wire_stereo_split_cameras(client, stereo_camera_index: int | str = 1, stereo_camera=None) -> None:
    """Make every observation this client's `control_loop_observation()`
    sends include real `camera2`/`camera3` split-stereo feeds, sourced from
    a single shared `StereoSplitCamera` -- not two independent
    `cv2.VideoCapture(1)` opens of the same physical AR0144 device.

    `lerobot.cameras.camera.CameraConfig` IS a `draccus.ChoiceRegistry` (a
    custom camera `type` plugin route genuinely exists -- confirmed by
    reading `control/.venv/.../lerobot/cameras/camera.py` and
    `configuration_opencv.py` this session), but there is no built-in
    facility for two independently-registered camera configs to share one
    physical device instance -- a plugin route would still have to solve
    that same single-device-sharing problem `StereoSplitCamera` already
    solves internally. Monkey-patching the already-connected robot's own
    `get_observation()` (the actual observation-building step
    `control_loop_observation()` calls internally, per
    `lerobot/async_inference/robot_client.py`) is the simpler,
    directly-testable mechanism that's actually scriptable here, per
    `policy_server_launch.md`'s "observation-dict patch" option.

    `stereo_camera` is an injectable override (a already-constructed
    `StereoSplitCamera`-shaped double) so tests can verify this wiring
    without opening a real cv2 device.
    """
    stereo = stereo_camera if stereo_camera is not None else StereoSplitCamera(index=stereo_camera_index)
    original_get_observation = client.robot.get_observation

    def get_observation_with_stereo_split():
        obs = original_get_observation()
        obs["camera2"] = stereo.read_left()
        obs["camera3"] = stereo.read_right()
        return obs

    client.robot.get_observation = get_observation_with_stereo_split
    # Keep a reference so the StereoSplitCamera (and its open cv2 device) isn't
    # garbage-collected once this function returns.
    client._stereo_camera = stereo


def pop_validated_action(
    client,
    safety_validate_fn,
    joint_limits_deg: dict[str, tuple[float, float]],
    current_state: dict[str, float],
    prev_action: dict[str, float] | None,
    dt_s: float,
    stale_threshold_s: float = safety_validator.STALE_ACTION_S,
) -> tuple[dict[str, float], list[str], dict[str, float], float]:
    """Pop one candidate action from the bridge client's queue and validate it.

    Performs NO `send_action` call itself -- the caller (`BridgeActionSource`
    or `run_vla_episode.py`) is responsible for that, keeping the validation
    gate and the execution gate visibly separate in the code (T-11-09).

    Returns `(validated_action, flags, raw_action, obs_age_s)`. On an empty
    queue, returns `(dict(current_state), ["no action available, holding
    position"], {}, 0.0)` rather than raising.

    Defaults `stale_threshold_s` to `STALE_ACTION_S` (network round-trip
    staleness), not `STALE_OBSERVATION_S` (synchronous-loop staleness,
    `validate_action()`'s own default) -- found live this session:
    every bridge-returned action was being checked against the 1.0s
    synchronous threshold, which a real cloud round-trip (tens of seconds,
    worse on a cold-started policy server) can never meet, permanently
    holding position regardless of whether real actions were arriving.

    Normalizes `.pos`-suffixed keys out of `raw_action` immediately after
    `_action_tensor_to_action_dict()`, before validation. lerobot's real
    convention (`SO101Follower.action_features`, confirmed by reading the
    installed `lerobot==0.6.1` source) returns keys like `"shoulder_pan.pos"`,
    while this project's convention everywhere else (`action_contract.
    JOINT_ORDER`, `safety_validator.validate_action()`'s lookups) is plain
    names like `"shoulder_pan"`. Left unnormalized, every joint failed
    `validate_action()`'s `if joint not in raw_action: continue` membership
    check, so `validated_action` silently came back `{}` for every real
    bridge-returned action -- no exception, no flag -- which crashed the
    last live episode at step 1 (`robot.send_action({f"{j}.pos": v for j, v
    in validated_action.items()})` sent an empty `goal_pos`, raising
    `ValueError: max_relative_target keys must match those of
    goal_present_pos.` from lerobot's `ensure_safe_goal_position()`).
    """
    with client.action_queue_lock:
        try:
            timed_action = client.action_queue.get_nowait()
        except queue.Empty:
            return dict(current_state), ["no action available, holding position"], {}, 0.0

    with client.latest_action_lock:
        client.latest_action = timed_action.get_timestep()

    raw_action = client._action_tensor_to_action_dict(timed_action.get_action())
    raw_action = {key.removesuffix(".pos"): value for key, value in raw_action.items()}
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
        stale_threshold_s=stale_threshold_s,
    )
    return validated_action, flags, raw_action, obs_age_s


def _is_stale_or_empty(flags: list[str]) -> bool:
    """`True` if `flags` (as returned by `pop_validated_action()`) represents
    a held, not-genuinely-fresh result -- the two exact shapes
    `pop_validated_action()` can return for a held result: its own
    empty-queue early return (`"no action available, holding position"`) or
    `safety_validator.validate_action()`'s stale-observation override (a flag
    containing the substring `"stale observation"`)."""
    return any(
        flag == "no action available, holding position" or "stale observation" in flag
        for flag in flags
    )


def force_bridge_recovery(client) -> None:
    """Force the vendored `RobotClient` out of a stuck stale/similarity
    deadlock (Gap 3, blocker) directly -- this project's own code -- instead
    of waiting on its own internal `must_go`/`action_queue`-empty convergence,
    confirmed live to take 6+ minutes once the server's `observations_similar()`
    filter starts dropping every non-`must_go` observation
    (`policy_server.py`'s `_enqueue_observation()`, read-only reference, never
    patched in place).

    Drains `client.action_queue` to empty via repeated `get_nowait()` under
    `client.action_queue_lock` -- never `client.action_queue.empty()` as a
    loop condition, since that check-then-act pattern would race against
    `receive_actions()`'s background thread refilling the queue between the
    check and the pop. Then sets `client.must_go` so the next
    `control_loop_observation()` call sends an observation the server's own
    `_enqueue_observation()` unconditionally processes, bypassing
    `observations_similar()`.
    """
    with client.action_queue_lock:
        while True:
            try:
                client.action_queue.get_nowait()
            except queue.Empty:
                break
    client.must_go.set()


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
        stale_watchdog_limit: int = STALE_WATCHDOG_CONSECUTIVE_LIMIT,
    ):
        self.client = client
        self.checkpoint = checkpoint
        self.joint_limits_deg = joint_limits_deg
        self._dt_s = dt_s
        self._last_action: dict[str, float] | None = None
        self._inflight_lock = threading.Lock()
        self._stale_watchdog_limit = stale_watchdog_limit
        self._consecutive_stale_count = 0

    def get_action(
        self, joint_state: dict[str, float], instruction: str
    ) -> tuple[dict[str, float], str]:
        # LATENCY-04: defensive-only guard (per CONTEXT.md D-05) -- get_action()
        # is called synchronously, once per tick, from a single-threaded
        # control loop today, so no active overlap bug exists. This structurally
        # prevents a *future* change (e.g. adding a call timeout) from silently
        # reintroducing an overlapping in-flight request.
        if not self._inflight_lock.acquire(blocking=False):
            return joint_state, f"{self.checkpoint}@bridge-request-inflight-holding-position"

        try:
            if self.client._ready_to_send_observation():
                try:
                    self.client.control_loop_observation(task=instruction)
                except (grpc.RpcError, ConnectionError, RuntimeError):
                    return joint_state, f"{self.checkpoint}@bridge-error-holding-position"

            try:
                validated_action, flags, _raw_action, _obs_age_s = pop_validated_action(
                    self.client,
                    safety_validator.validate_action,
                    self.joint_limits_deg,
                    joint_state,
                    self._last_action,
                    self._dt_s,
                )
            except (grpc.RpcError, ConnectionError, RuntimeError):
                return joint_state, f"{self.checkpoint}@bridge-error-holding-position"

            # Gap 3 (blocker) watchdog: count consecutive stale/no-action
            # results; after `_stale_watchdog_limit` in a row, force the
            # bridge out of the stuck state directly rather than waiting on
            # the vendored library's own must_go/action_queue-empty
            # convergence (confirmed live to take 6+ minutes). Any genuine
            # fresh result always resets the counter to zero.
            if _is_stale_or_empty(flags):
                self._consecutive_stale_count += 1
                if self._consecutive_stale_count >= self._stale_watchdog_limit:
                    force_bridge_recovery(self.client)
                    self._consecutive_stale_count = 0
                    flags = [*flags, "staleness-watchdog: forced must_go + cleared action_queue"]
            else:
                self._consecutive_stale_count = 0

            self._last_action = validated_action
            if flags:
                return validated_action, f"{self.checkpoint}@{'+'.join(flags)}"
            return validated_action, f"{self.checkpoint}@unknown"
        finally:
            self._inflight_lock.release()


__all__ = [
    "connect_bridge",
    "pop_validated_action",
    "BridgeActionSource",
]
