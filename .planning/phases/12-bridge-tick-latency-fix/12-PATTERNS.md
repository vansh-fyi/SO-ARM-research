# Phase 12: Bridge Tick-Latency Fix - Pattern Map

**Mapped:** 2026-09-25
**Files analyzed:** 4 (all existing, modified in place — no new files created)
**Analogs found:** 4 / 4

This phase modifies existing files rather than creating new ones. The
"analog" for each file is the vendored `lerobot` reference implementation
(or, for DEBT-02, the file's own pre-fix docstring context) that the fix
must mirror.

## File Classification

| File to Modify | Role | Data Flow | Closest Analog | Match Quality |
|---|---|---|---|---|
| `control/vla_bridge/robot_client.py` (`BridgeActionSource.get_action`, `pop_validated_action`, `connect_bridge`) | service/controller (control-loop gate + queue pop) | streaming / event-driven (chunk-drain over gRPC) | `control/.venv/.../lerobot/async_inference/robot_client.py` `control_loop()` (line 458), `control_loop_action()` (line 370), `_ready_to_send_observation()` (line 403) | exact — this module already wraps that class; the fix ports specific vendored methods into the existing wrapper |
| `control/vla_bridge/io_logger.py` (`latency_ms` field / `write_step` caller contract) | utility (structured JSONL writer) | request-response (per-tick record write) | `control/run_vla_episode.py` line 227 (current call site constructing the `latency_ms` dict) | exact — same file's own existing call-site convention, just fed real values instead of hardcoded 0s |
| `control/run_vla_episode.py` (control loop, lines ~190-232) | controller (top-level per-tick orchestration) | streaming / event-driven (tick loop: read state → get action → validate → send → log) | itself (existing loop) + `control_loop()` gate pattern from vendored `robot_client.py` (line 458-479) | exact — no new file; existing loop gets two `time.monotonic()` brackets added |
| `LIBERO/libero/libero/envs/grippers/soarm_gripper.py` (`format_action` docstring, lines 41-53) | model (gripper kinematics wrapper) | transform (action → jaw position mapping) | itself — docstring already exists, needs correction to match already-correct code (lines 54-59) | exact — the polarity fix already landed in code; only the docstring is stale |

## Pattern Assignments

### `control/vla_bridge/robot_client.py` — LATENCY-01 (observation gate)

**Analog:** vendored `lerobot/async_inference/robot_client.py`, `control_loop()` (lines 458-479)

**Reference gate pattern to mirror** (vendored, lines ~469-476):
```python
while self.running:
    control_loop_start = time.perf_counter()
    """Control loop: (1) Performing actions, when available"""
    if self.actions_available():
        _performed_action = self.control_loop_action(verbose)

    """Control loop: (2) Streaming observations to the remote policy server"""
    if self._ready_to_send_observation():
        _captured_observation = self.control_loop_observation(task, verbose)
```

**Current code being replaced** — `BridgeActionSource.get_action()`, `control/vla_bridge/robot_client.py` lines 270-293, currently calls `control_loop_observation()` unconditionally every tick:
```python
def get_action(
    self, joint_state: dict[str, float], instruction: str
) -> tuple[dict[str, float], str]:
    try:
        self.client.control_loop_observation(task=instruction)
    except (grpc.RpcError, ConnectionError, RuntimeError):
        return joint_state, f"{self.checkpoint}@bridge-error-holding-position"
    ...
```

**Fix shape (per D-01/D-05, no new gate logic to write — only wire in the existing `_ready_to_send_observation()`):**
```python
def get_action(self, joint_state, instruction):
    if self.client._ready_to_send_observation():
        try:
            self.client.control_loop_observation(task=instruction)
        except (grpc.RpcError, ConnectionError, RuntimeError):
            return joint_state, f"{self.checkpoint}@bridge-error-holding-position"
    # ... pop_validated_action() unconditionally, same as today
```
Note: `_ready_to_send_observation()` reads `self.action_queue_lock` internally (vendored
line 403-405) — do not double-lock around it.

---

### `control/vla_bridge/robot_client.py` — LATENCY-02 (`client.latest_action` update)

**Analog:** vendored `control_loop_action()` (lines 370-390), specifically:
```python
_performed_action = self.robot.send_action(
    self._action_tensor_to_action_dict(timed_action.get_action())
)
with self.latest_action_lock:
    self.latest_action = timed_action.get_timestep()
```

**Target:** `pop_validated_action()` in `control/vla_bridge/robot_client.py`, lines 181-244.
This project's `pop_validated_action()` is the de facto replacement for `control_loop_action()`
(per module docstring lines 1-35 — it deliberately never calls `robot.send_action()` itself,
keeping the validation/execution gate split). The existing pop happens at lines 223-227:
```python
with client.action_queue_lock:
    try:
        timed_action = client.action_queue.get_nowait()
    except queue.Empty:
        return dict(current_state), ["no action available, holding position"], {}, 0.0
```
**Fix shape** — after a successful pop (mirroring vendored lines 384-385, using the project's
existing `client.latest_action_lock` which is already present on the vendored `RobotClient`
instance since `connect_bridge()` returns a real `RobotClient`):
```python
with client.latest_action_lock:
    client.latest_action = timed_action.get_timestep()
```
Place this right after the `get_nowait()` succeeds (after line 227's early-return branch,
before/after building `raw_action` — either is fine since `timed_action` is already available).

---

### `control/vla_bridge/robot_client.py` — LATENCY-04 (in-flight-request guard)

**Analog:** none needed from vendored lib (defensive-only per D-05) — pattern is a standard
guard-flag idiom. Closest in-repo precedent is the existing `client.action_queue_lock` usage
pattern already in this file (lines 223, `with client.action_queue_lock:`), i.e. this codebase's
established idiom for guarding a shared-state section is a `with <lock>:` block, not manual
acquire/release.

**Fix shape** (implementer's discretion per D-05 — boolean flag or `threading.Lock`; the
existing codebase idiom favors an explicit lock object for consistency with `action_queue_lock`):
```python
class BridgeActionSource:
    def __init__(self, client, checkpoint, joint_limits_deg, dt_s=0.5):
        ...
        self._inflight_lock = threading.Lock()

    def get_action(self, joint_state, instruction):
        if not self._inflight_lock.acquire(blocking=False):
            return joint_state, f"{self.checkpoint}@bridge-request-inflight-holding-position"
        try:
            # existing gated observation-send + pop_validated_action() body
            ...
        finally:
            self._inflight_lock.release()
```
`threading` is already imported at module top (`control/vla_bridge/robot_client.py` line 38).

---

### `control/vla_bridge/io_logger.py` + `control/run_vla_episode.py` — LATENCY-03 (real latency)

**Analog:** the existing call site itself, `control/run_vla_episode.py` line 227:
```python
latency_ms={"observation_to_action": 0, "action_to_execution": 0},
```
**Surrounding loop context** (lines 196-229) shows the two bracket points to instrument:
```python
for i in range(max_steps):
    current = read_positions(robot)
    raw_action, model_version = action_source.get_action(current, instruction)   # <- bracket 1 start/end
    validated_action, flags = safety_validator.validate_action(...)
    try:
        robot.send_action({f"{j}.pos": v for j, v in validated_action.items()})  # <- bracket 2 end
    except (ConnectionError, RuntimeError) as e:
        print(f"Write failed, skipping this tick: {e}")
    ...
    io_logger.write_step(
        ...,
        latency_ms={"observation_to_action": 0, "action_to_execution": 0},  # <- replace with real deltas
        ...,
    )
```
**Fix shape** (per CONTEXT.md Claude's Discretion — `time.monotonic()` deltas, `time` already
imported in `run_vla_episode.py`):
```python
t0 = time.monotonic()
raw_action, model_version = action_source.get_action(current, instruction)
t1 = time.monotonic()
validated_action, flags = safety_validator.validate_action(...)
try:
    robot.send_action({f"{j}.pos": v for j, v in validated_action.items()})
except (ConnectionError, RuntimeError) as e:
    print(f"Write failed, skipping this tick: {e}")
t2 = time.monotonic()
...
io_logger.write_step(
    ...,
    latency_ms={
        "observation_to_action": (t1 - t0) * 1000,
        "action_to_execution": (t2 - t1) * 1000,
    },
    ...,
)
```
`io_logger.py`'s `write_step()` signature (lines 42-54) takes `latency_ms: dict` as-is — no
signature change needed, only the caller's payload changes.

---

### `LIBERO/libero/libero/envs/grippers/soarm_gripper.py` — DEBT-02 (docstring fix)

**Analog:** the file's own already-correct code, lines 54-59:
```python
def format_action(self, action):
    """Maps the 1-D gripper action into the two jaw position targets.

    External +1 opens and -1 closes. robosuite scales current_action
    from [-1, +1] to the actuator range [0, 0.036] metres: -1 is closed,
    +1 is open. Both symmetric jaws integrate in the same direction.
    ...
    """
```
This docstring (lines 44-46) is actually already correct in the currently-read version
(states "+1 opens and -1 closes", matching `init_qpos` open-endpoint convention at line 64
and `speed` comment at line 68). **Verify against CONTEXT.md's claimed line range 41-53
before editing** — CONTEXT.md states it "currently mislabels OPEN/CLOSE direction
post-polarity-fix," but the text read in this pass already states the corrected polarity.
Re-read lines 41-53 fresh at plan/implementation time in case a different version is on disk,
or confirm CONTEXT.md's claim refers to a since-superseded state. If already correct, this
edit may be a no-op / already resolved — flag to planner rather than assume unconfirmed drift.

## Shared Patterns

### Lock-guarded shared-state access
**Source:** `control/vla_bridge/robot_client.py` line 223 (`with client.action_queue_lock:`)
**Apply to:** LATENCY-02 (`client.latest_action_lock`) and LATENCY-04 (new `_inflight_lock`) —
this codebase's established idiom is `with <lock>:` context-manager blocks, not manual
acquire/release, except where non-blocking `acquire(blocking=False)` is specifically needed
(LATENCY-04's guard, which cannot use `with` directly).

### Exception handling on bridge calls
**Source:** `control/vla_bridge/robot_client.py` lines 273-276, 287-288
```python
except (grpc.RpcError, ConnectionError, RuntimeError):
    return joint_state, f"{self.checkpoint}@bridge-error-holding-position"
```
**Apply to:** Any new code path added inside `BridgeActionSource.get_action()` — preserve this
exact exception tuple and the "@bridge-error-holding-position" flag-string convention for
consistency with existing error flags (`@unknown`, `@bridge-error-holding-position`).

### Validation/execution gate separation (T-11-09)
**Source:** module docstring, `control/vla_bridge/robot_client.py` lines 1-35
**Apply to:** All LATENCY-01/02/04 changes — `pop_validated_action()` must continue to never
call `send_action()`; `robot.send_action()` stays exclusively in `run_vla_episode.py`'s loop
(line 208). Do not let LATENCY-02's `client.latest_action` update or LATENCY-04's guard blur
this boundary.

## No Analog Found

None — all 4 files have a clear analog (vendored lerobot source or their own existing
call-site/docstring context).

## Metadata

**Analog search scope:** `control/vla_bridge/`, `control/run_vla_episode.py`,
`control/.venv/lib/python3.12/site-packages/lerobot/async_inference/robot_client.py`,
`LIBERO/libero/libero/envs/grippers/soarm_gripper.py`
**Files scanned:** 4 target files + 1 vendored reference file
**Pattern extraction date:** 2026-09-25
