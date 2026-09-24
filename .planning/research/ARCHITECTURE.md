# Architecture Research

**Domain:** Real-hardware robot control loop — integrating an MLLM (Claude) control path alongside an existing VLA bridge, fixing an observation/action timing bug, and unifying two camera-resolution code paths
**Researched:** 2026-09-24
**Confidence:** HIGH — every claim below is grounded in the actual project source (`control/vla_bridge/*.py`, `control/run_vla_episode.py`) and the actual installed `lerobot==0.6.1` library source (`control/.venv/lib/python3.12/site-packages/lerobot/async_inference/{robot_client.py,configs.py}`), not framework docs or assumption. File:line citations are given throughout.

## Standard Architecture

### System Overview — Current (Phase 11, as built)

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    control/run_vla_episode.py (main())                    │
│  builds robot_config, caps{}, camera_names{}, joint_limits_deg            │
└───────────────┬─────────────────────────────────────────┬────────────────┘
                │ constructs                               │ constructs
                ▼                                           ▼
┌───────────────────────────────┐          ┌───────────────────────────────┐
│ vla_bridge.robot_client        │          │ run_vla_episode.py's own      │
│  .connect_bridge()             │          │  caps = {idx: cv2.VideoCapture│
│  → RobotClient (lerobot)       │          │          (idx) for idx in     │
│  → BridgeActionSource          │          │          args.camera}         │
│  wires StereoSplitCamera into  │          │  (SEPARATE cv2 opens, numeric │
│  client.robot.get_observation()│          │  index, NOT name-based)       │
│  (camera2/camera3 ONLY)        │          └───────────────┬───────────────┘
└───────────────┬─────────────────                            │
                │                                              │
                ▼                                              ▼
┌───────────────────────────────────────────┐   ┌─────────────────────────┐
│      run_episode() control loop             │   │  io_logger.IOLogger      │
│      (run_vla_episode.py:165-244)            │   │  .capture_camera_frame() │
│                                               │──▶│  writes camera_overhead/ │
│  for i in range(max_steps):                  │   │  {step}.png from `caps`  │
│    current = read_positions(robot)           │   │  (the WRONG device — cv2 │
│    raw_action, model_version =               │   │  numeric index drifted   │
│      action_source.get_action(current, ...)  │   │  to the laptop webcam)   │
│    validated_action, flags =                 │   └─────────────────────────┘
│      safety_validator.validate_action(...)   │
│    robot.send_action(...)                    │
│    io_logger.write_step(...)                 │
└───────────────┬───────────────────────────────┘
                │ action_source.get_action() calls, EVERY tick:
                ▼
┌───────────────────────────────────────────────────────────┐
│ BridgeActionSource.get_action()  (robot_client.py:270-293)  │
│   self.client.control_loop_observation(task=instruction)    │  ← UNCONDITIONAL,
│   pop_validated_action(...) → client.action_queue.get_nowait()  every tick,
└───────────────────────────────────────────────────────────┘     regardless of
                │                                                  queue state
                ▼ (over ngrok tunnel, every tick)
        Colab-hosted PolicyServer (SmolVLA inference)
```

**The bug, precisely located:** `control_loop_observation()` (installed `lerobot`'s `robot_client.py:408-453`) does a full `self.robot.get_observation()` capture + pickle + gRPC `SendObservations` round-trip. `BridgeActionSource.get_action()` (this project's `robot_client.py:270-293`) calls it **unconditionally on line 274**, once per control-loop tick, even while `pop_validated_action()` is just draining an already-fetched 50-action chunk from a prior inference call. That is the entire ~11–20s/tick cost documented in `FINDINGS.md §5` — not model inference time, not robot I/O time, but a redundant network round-trip repeated on every tick.

### Component Responsibilities

| Component | Responsibility | File:line |
|-----------|-----------------|-----------|
| `run_episode()` | Control loop driver: read state → get action → validate → send → log. Provider-agnostic — has no idea whether `action_source` is scripted, VLA-bridged, or MLLM. | `control/run_vla_episode.py:165-244` |
| `ActionSource` Protocol | The seam that makes `run_episode()` provider-agnostic. `get_action(joint_state, instruction) -> (raw_action, model_version)`. | `control/run_vla_episode.py:75-88` |
| `BridgeActionSource` | Wraps `RobotClient` + `pop_validated_action()` behind `ActionSource`. Deliberately never calls the library's own `control_loop()`/`control_loop_action()` because both call `robot.send_action()` internally, bypassing the safety validator. | `control/vla_bridge/robot_client.py:247-293` |
| `safety_validator.validate_action()` | The ONE gate every candidate action (any source) must pass before `robot.send_action()`. NaN/inf rejection, absolute joint-limit clamp, per-step + velocity displacement caps, staleness override. | `control/vla_bridge/safety_validator.py:58-132` |
| `IOLogger` | One JSONL record per step: instruction, camera frame refs, joint state, raw/validated/executed action, latency, model version. Provider-agnostic — takes plain dicts, doesn't know the action's origin. | `control/vla_bridge/io_logger.py:20-93` |
| `action_contract` | Single source of truth for `JOINT_ORDER`, per-joint units, and live joint limits (`load_joint_limits_deg()`). Consumed by both the validator and (for the MLLM path) the model's own output schema. | `control/vla_bridge/action_contract.py` |
| `StereoSplitCamera` | Owns the ONE ffmpeg/AVFoundation process for the AR0144 stereo device, resolved by AVFoundation **name** (not numeric index, which is confirmed to drift on macOS). Splits one 2560×720 read into `read_left()`/`read_right()`. | `control/vla_bridge/stereo_camera.py:110-176` |
| `RobotClient` (vendored `lerobot`) | Owns the gRPC channel, `action_queue`, and — critically — `_ready_to_send_observation()` / `must_go`, the library's own built-in observation/action decoupling mechanism that this project's bridge code does not currently use. | `.venv/.../lerobot/async_inference/robot_client.py:83-481` |

## The Latency Fix: Use the Library's Own Decoupling, Don't Hand-Roll One

### The gate already exists in `lerobot` — it's just not called

Reading the installed library (not assumed from its CLI docs) turns up exactly the mechanism PROJECT.md's candidate fix (a) is asking for, already implemented and already configured:

```python
# robot_client.py:403-406 (installed lerobot 0.6.1)
def _ready_to_send_observation(self):
    """Flags when the client is ready to send an observation"""
    with self.action_queue_lock:
        return self.action_queue.qsize() / self.action_chunk_size <= self._chunk_size_threshold
```

```python
# robot_client.py:458-481 — the library's OWN control_loop(), which this project
# deliberately never calls (it bypasses safety_validator, per robot_client.py:1-35)
while self.running:
    if self.actions_available():
        _performed_action = self.control_loop_action(verbose)      # drain
    if self._ready_to_send_observation():
        _captured_observation = self.control_loop_observation(...)  # only send when queue is low
    time.sleep(max(0, self.config.environment_dt - elapsed))
```

`connect_bridge()` already passes `chunk_size_threshold=0.5` into `RobotClientConfig` (`robot_client.py:57-58,117-126` in this project's module) — so the 50%-drained threshold this project wants is **already wired into the config**. It's just never consulted, because `BridgeActionSource.get_action()` (this project's code, `robot_client.py:270-293`) calls `control_loop_observation()` directly and unconditionally instead of gating it behind `_ready_to_send_observation()`.

### Recommended fix (Pattern 1)

**What:** Gate the `control_loop_observation()` call in `BridgeActionSource.get_action()` behind `self.client._ready_to_send_observation()`, mirroring the vendored library's own `control_loop()` — without adopting `control_loop()` itself (which still can't be used, for the documented safety-bypass reason).

**Where:** `control/vla_bridge/robot_client.py:270-293`, minimal diff:

```python
def get_action(self, joint_state, instruction):
    try:
        if self.client._ready_to_send_observation():
            self.client.control_loop_observation(task=instruction)
    except (grpc.RpcError, ConnectionError, RuntimeError):
        return joint_state, f"{self.checkpoint}@bridge-error-holding-position"
    ...  # pop_validated_action() unchanged — drains whatever is already queued
```

**Why this is the cleanest option, not a new design:** it reuses a mechanism the library authors already built and this project already configures (`chunk_size_threshold`), rather than inventing project-local timer/counter logic. It requires touching exactly one call site. It needs no new state, no new config surface, and it keeps the file's existing, well-documented convention of reaching into `client._`-prefixed internals with a cited rationale (the file already does this for `client.action_queue_lock`, `client._action_tensor_to_action_dict`, `client._stereo_camera`).

**A second bug in the same neighborhood, worth fixing in the same change:** `pop_validated_action()` (`robot_client.py:181-244`) never updates `client.latest_action` after popping — only the library's own `control_loop_action()` does that (`robot_client.py:384-385`). Left as-is, `client.latest_action` stays at its `__init__` sentinel of `-1` for the whole episode, which means every `TimedObservation` sent by `control_loop_observation()` reports `timestep=max(latest_action, 0)` → always `0` (`robot_client.py:422`), and `_aggregate_action_queues()`'s "skip actions older than the latest performed one" dedup logic (`robot_client.py:246`) never actually skips anything. Neither is fatal on its own, but both silently degrade the server's view of trajectory progress and the client's own aggregation. Fix: add `with self.action_queue_lock: ... self.latest_action = timed_action.get_timestep()`-equivalent bookkeeping to `pop_validated_action()` after the pop, mirroring `control_loop_action()`'s own update (needs a `latest_action_lock`-protected write via a small helper, since `pop_validated_action()` is a free function, not a `RobotClient` method).

**Candidate fix (b) (tune `control_hz`) becomes secondary, not parallel:** once (a) lands, ticks where the queue is still well-stocked skip the network round-trip entirely — `pop_validated_action()`'s `action_queue.get_nowait()` plus `safety_validator.validate_action()` plus `robot.send_action()` are all local/serial-bus operations, sub-100ms. `run_vla_episode.py`'s `--control-hz` (default 2.0, `run_vla_episode.py:274`) then only needs to be fast enough for smooth local drain-and-execute, completely decoupled from the ~11–20s Colab/ngrok round-trip. Recommend tuning `control_hz` only after (a) is live and measured — don't pre-guess a number now.

### Data Flow — After the Fix

```
Tick N (queue well-stocked):
  read_positions → pop_validated_action (local, <10ms) → validate → send_action → log
  [control_loop_observation() SKIPPED — _ready_to_send_observation() is False]

Tick N+k (queue drained to ≤50% of last chunk):
  read_positions → control_loop_observation() [network, ~11-20s] → pop_validated_action → validate → send_action → log
  [this tick pays the round-trip cost; the next ~25 ticks (50 * 0.5) don't]
```

This is the direct fix for the 1/60 (1.7%) real-action yield documented in `FINDINGS.md §2,5`: actions stop going stale before they're popped because most ticks are no longer artificially inflated to 11–20s by a redundant network call.

## MLLM Integration: New `ActionSource`, Zero Duplication of Safety/Logging

### The integration point is exactly the seam already built for this

`run_episode()` (`run_vla_episode.py:165-244`) is already provider-agnostic: it calls `action_source.get_action(current, instruction)`, then unconditionally runs the result through `safety_validator.validate_action()`, then `robot.send_action()`, then `io_logger.write_step()`. Plan 11-04 proved this seam works by swapping `ScriptedActionSource` → `BridgeActionSource` with **no changes to `run_episode()` itself**. The MLLM path should be the same swap: a new `ClaudeActionSource` implementing the same `ActionSource` Protocol (`run_vla_episode.py:75-88`), selected via a new CLI flag (e.g. `--mllm-model`) alongside the existing `--server-address`/`--checkpoint` flags, mutually exclusive with them.

**What must NOT happen:** a second safety-check path, a second JSONL writer, or a second `robot.send_action()` call site. The existing `safety_validator.py`/`io_logger.py`/`action_contract.py` trio is provider-agnostic by construction (plain dicts in, plain dicts/flags out) — there is nothing MLLM-specific to add to any of them except one field (reasoning trace — see below).

### Pattern 2: Claude call produces a *chunk*, not a single action — needs its own local queue

The VLA bridge's `action_queue` is filled by a network thread (`receive_actions()`) and drained by `pop_validated_action()` one action per tick. The MLLM path has no equivalent background thread — a Claude API call is a single synchronous request that should return a **sequenced chunk** of actions (per PROJECT.md's committed design: "emits a sequenced chunk of JSON actions ... chunk size tuned from latency-fix timing data"). The cleanest shape, mirroring the VLA bridge's own request/drain split instead of inventing a new one:

```python
class ClaudeActionSource:
    """ActionSource backed by Claude (Anthropic API). Mirrors BridgeActionSource's
    request-a-chunk/drain-locally split, but the chunk source is a synchronous
    API call instead of a background gRPC thread."""

    def __init__(self, client, model, joint_limits_deg, dt_s=0.5, chunk_size=10):
        self._client = client            # anthropic.Anthropic()
        self._model = model
        self._joint_limits_deg = joint_limits_deg
        self._dt_s = dt_s
        self._chunk_size = chunk_size
        self._queue: list[dict] = []     # local, synchronous — no thread/lock needed
        self._last_reasoning = ""

    def get_action(self, joint_state, instruction):
        if not self._queue:
            self._queue, self._last_reasoning = self._request_chunk(joint_state, instruction)
        raw_action = self._queue.pop(0) if self._queue else dict(joint_state)
        return raw_action, f"{self._model}@chunk"

    def _request_chunk(self, joint_state, instruction) -> tuple[list[dict], str]:
        # client.messages.create(..., output_config={"format": {"type": "json_schema",
        # "schema": ACTION_CHUNK_SCHEMA}}, messages=[...task/calibration/joint_state
        # text..., *camera_frame_image_blocks]) -- see Integration Points below.
        ...
```

No lock is needed (unlike `BridgeActionSource`'s `action_queue_lock`) because the Claude request is synchronous and single-threaded from the control loop's own perspective — there's no background receiver thread analogous to `receive_actions()`. This is simpler than the VLA bridge path, not more complex, precisely because the bridge's complexity (locks, barriers, a daemon thread) exists to cover an async gRPC stream that the MLLM path doesn't have.

**Chunk-size tuning depends on the latency fix's timing data (explicit dependency):** PROJECT.md's Key Decisions table states this outright — "Chunk size for the MLLM's action sequences will be informed by real timing/chunking data gathered while fixing the tick-latency bug." Concretely: once fix (a) above lands and is measured, the phase will know the real Colab/ngrok round-trip distribution: use a comparable order-of-magnitude for `ClaudeActionSource`'s `chunk_size` (an MLLM sub-goal-level call has its own multi-second latency; a chunk sized too small re-triggers an API call almost every tick — the same anti-pattern the VLA-side fix removes — while too large risks committing to a stale plan if the visual scene changes mid-chunk).

### Pattern 3: Action JSON schema mirrors `action_contract.JOINT_ORDER` exactly — reuse, don't reinvent

The safety validator's per-joint lookups (`safety_validator.py:76-123`) iterate `action_contract.JOINT_ORDER` and index `raw_action[joint]`/`joint_limits_deg[joint]`. For the MLLM's JSON output to pass through the *same, unmodified* validator, the `output_config.format` JSON schema given to Claude must produce objects keyed exactly by `action_contract.JOINT_ORDER`'s six plain names (`shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper`), in the same units (`action_contract.ACTION_UNITS`: degrees for the five arm joints, `percent_0_100` for gripper). Build the schema programmatically from `action_contract.JOINT_ORDER`/`ACTION_UNITS` rather than hand-writing it, so a future joint-order change can't silently desync the model's output shape from the validator's expectations — the exact class of bug `pop_validated_action()`'s `.pos`-suffix normalization already had to fix once for the VLA path (`robot_client.py:208-221`; `FINDINGS.md §1.3`).

**Anthropic API design decisions for this call** (per `claude-api` skill, current as of 2026-09):

- **Structured output:** use `output_config: {"format": {"type": "json_schema", "schema": ...}}` on a plain `client.messages.create()` call — not tool-use/`tool_choice`. This is a single-shot "produce this JSON shape" request, not an agentic tool-calling loop (there's nothing for Claude to call — it isn't executing code, per the already-committed decision to skip sandboxing entirely). `output_config.format` guarantees the first content block is text containing valid JSON matching the schema (`json.loads()` it directly) — this is GA, no beta header.
- **Vision input (multiple camera frames):** each camera frame is one `{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": ...}}` content block in the same user message, alongside the task instruction, calibration data, and current joint-state text — i.e. the same multi-image-in-one-message shape the project's io_logger already saves frames for. This is the direct MLLM analogue of what `BridgeActionSource` already does by wiring `camera2`/`camera3` into `get_observation()` (`robot_client.py:142-179`) for the VLA's native dual-image API — reuse the same camera capture calls (`StereoSplitCamera.read_left()`/`read_right()`, plus wrist) as the frame source for both.
- **Model choice:** given this milestone is explicitly a "deep-reasoning" comparison against SmolVLA (PROJECT.md: "prove out... then run a comparable experiment with a genuine deep-reasoning multimodal model"), default to `claude-opus-5` for the actual Pen Transfer comparison run — the stronger reasoning tier is the point of the experiment. `claude-sonnet-5` is a reasonable cost-conscious substitute for iterating on the control-loop plumbing itself (schema correctness, camera wiring, reasoning-trace logging) before spending the more expensive tier on real hardware episodes — this matches the project's own established pattern of piloting cheap before committing paid-tier budget (PROJECT.md Key Decisions: "Provider-agnostic MLLM router, piloted first on a free/cheap-tier model before wiring paid providers").
- **Streaming:** not required for this call shape. Sub-goal-level chunk requests (not per-tick) with a bounded `max_tokens` (schema-constrained JSON plus a reasoning-trace text block) fit well inside non-streaming's default request timeout (10 min). Reach for streaming only if the reasoning-trace text turns out to run long enough to risk `max_tokens` truncation — decide from real token counts observed in the first live calls, not pre-emptively.

### Pattern 4: Reasoning-trace capture — extend the `ActionSource` contract, not `IOLogger`'s internals

D-05 (`FINDINGS.md §4`) already establishes that SmolVLA has no reasoning trace and that `io_logger.py`'s existing "complete I/O" fields are the correct, complete substitute for the VLA. The MLLM path is where reasoning-trace capture actually applies. Cleanest integration, consistent with this codebase's existing seams:

1. Widen the `ActionSource` Protocol's return type from `tuple[dict, str]` to `tuple[dict, str, str]` (action, model_version, reasoning) — `ScriptedActionSource` and `BridgeActionSource` both return `""` for reasoning (explicit, not absent — preserves D-05's framing that the VLA genuinely has none, rather than looking like an oversight).
2. Add one new parameter to `IOLogger.write_step()` — `reasoning: str = ""` — and one new key to the JSONL record. Backward compatible: existing episode-reading code that doesn't know about the field is unaffected, and old episodes (`vla_episode_001`, the Phase 11 retry) remain valid JSONL without it.
3. `run_episode()`'s loop (`run_vla_episode.py:198,218-229`) picks up the third tuple element and threads it into `write_step()` — a 2-line change to the control loop, nothing new to `safety_validator.py` or `action_contract.py`.

This keeps reasoning-trace capture entirely inside the `ActionSource`↔`IOLogger` boundary that already exists, rather than bolting a parallel logging path onto the MLLM source.

## Camera-Resolution Unification: Share the Proven Device Handle, Don't Just Copy Its Resolution Method

### Root cause, precisely located

`connect_bridge()`'s inference-input path resolves the AR0144 by **name** via `StereoSplitCamera` (`stereo_camera.py:110-176`, backed by `ffmpeg -f avfoundation -i "<name>"`). `run_vla_episode.py`'s own recording path (`main()`:348-354) opens cameras by **plain numeric `cv2.VideoCapture(idx)`**, where `idx` comes from `device_map.json`'s `cameras.stereo_overhead` (a bare int) — the exact index-drift failure mode `stereo_camera.py`'s own module docstring warns about, and the one confirmed live to be capturing the laptop's FaceTime camera instead of the robot workspace (PROJECT.md Active scope, confirmed 2026-09-24).

**The fix should not stop at "resolve by name instead of index" for the recording path** — a second independent `StereoSplitCamera`/ffmpeg process opening the *same* AVFoundation device the inference path already opened would very likely fail outright: `stereo_camera.py`'s own docstring states "most webcam drivers reject a second concurrent open of the same index," which is exactly why `StereoSplitCamera` exists as a single shared reader in the first place (`connect_bridge()`'s comment at `robot_client.py:142-165` makes the same point about `camera1`/wrist). The correct fix is to **share one `StereoSplitCamera` instance** between the inference path and the recording path, not to independently re-resolve the name twice.

### Pattern 5: Thread a single `StereoSplitCamera` through both call sites

**New:** two small `.read()`-shaped adapter objects (or one dual-purpose helper) around `StereoSplitCamera.read_left()`/`read_right()`, matching the existing `_FFmpegAVFoundationCapture`'s own documented convention ("Minimal `cv2.VideoCapture`-shaped wrapper... `isOpened()`/`read()`/`release()`", `stereo_camera.py:44-47`) so `IOLogger.capture_camera_frame(cap, name, step)` (`io_logger.py:71-90`, which calls `cap.read()`) can consume a `StereoSplitCamera` half exactly like it already consumes a `cv2.VideoCapture`:

```python
class _StereoHalfReader:
    """cv2.VideoCapture-shaped (.read()/.release()) view of one half of a
    shared StereoSplitCamera -- lets IOLogger.capture_camera_frame() treat
    it identically to any other camera handle."""
    def __init__(self, stereo: StereoSplitCamera, half: str):
        self._stereo, self._half = stereo, half  # half in {"left", "right"}
    def read(self):
        frame = (self._stereo.read_left() if self._half == "left"
                  else self._stereo.read_right())
        return (frame is not None), frame
    def release(self):
        pass  # StereoSplitCamera owns the underlying process; released by its owner
```

**Modified:**
- `stereo_camera.py`: add `_StereoHalfReader` (or equivalent) — new class in an existing module, no change to `StereoSplitCamera` itself.
- `robot_client.py`'s `connect_bridge()`: add a passthrough `stereo_camera=None` parameter to `_wire_stereo_split_cameras()` (which already accepts this — `_wire_stereo_split_cameras(client, stereo_camera_index=1, stereo_camera=None)`, `robot_client.py:142` — the injectable-override seam already exists for tests; extend `connect_bridge()`'s own signature to expose it to callers instead of only constructing internally).
- `run_vla_episode.py`'s `main()`: construct **one** `StereoSplitCamera(index=args.stereo_camera_index)` before branching on `args.server_address`, pass it into `connect_bridge(..., stereo_camera=stereo)` on the bridge path, and use `_StereoHalfReader(stereo, "left")` (or whichever half is confirmed to be the correct overhead view) in place of the current `cv2.VideoCapture(idx)` open for the `"overhead"` semantic camera slot in `caps`/`camera_names` — on **both** the bridge path and the `ScriptedActionSource` no-bridge path, so dry-run/e-stop recordings get the same fix.

This closes the bug by construction: recording and inference physically cannot diverge in which device they capture, because after this change they're reading from the same open handle.

**Wrist camera note (scope boundary):** PROJECT.md's Active scope only flags `camera_overhead` as confirmed-broken; the wrist camera (`args.camera[0]`) is unaffected by this specific bug report and stays on its current `cv2.VideoCapture` path for now. The same index-drift risk technically applies to it too (per `device_map.json`'s own freshness warning, `run_vla_episode.py:129-156`), but migrating it is out of this milestone's stated scope — note it as a follow-up, don't fold it into this fix silently.

## Anti-Patterns to Avoid

### Anti-Pattern 1: Calling `RobotClient.control_loop()` / `control_loop_action()` for the MLLM path

**What people might do:** since `control_loop()` already contains the observation/action decoupling logic this milestone wants, it's tempting to just call it directly for a "simpler" implementation.
**Why it's wrong:** both `control_loop()` and `control_loop_action()` call `self.robot.send_action(...)` internally (`robot_client.py:381-383,458-481`), completely bypassing `safety_validator.validate_action()`. This is the exact bypass `robot_client.py`'s own module docstring (lines 1-35) explains was deliberately avoided for the VLA path, and it applies identically to any MLLM path built on the same `RobotClient`.
**Instead:** only ever reuse the library's read-only/query primitives (`_ready_to_send_observation()`, `action_queue`, `_action_tensor_to_action_dict()`) — never the methods that touch `robot.send_action()` themselves. The MLLM path doesn't even use `RobotClient` at all (Claude is called directly via the Anthropic SDK, not through the gRPC bridge), so this risk doesn't arise there — but it's the reason `BridgeActionSource` is built the way it is, and any future refactor of the VLA path must preserve it.

### Anti-Pattern 2: A parallel safety/logging path "just for the MLLM, since it's different"

**What people might do:** because the MLLM path is architecturally different (synchronous API call vs. background-thread gRPC queue), build a bespoke validation/logging path alongside it "since it doesn't fit the VLA bridge's shape anyway."
**Why it's wrong:** `safety_validator.validate_action()` and `IOLogger.write_step()` are already provider-agnostic — they operate on plain `dict[str, float]` actions and don't know or care where the dict came from. The *shape mismatch* is entirely upstream, inside how each `ActionSource` fills its local queue (background thread vs. synchronous call) — it never needs to leak downstream of `get_action()`'s return value.
**Instead:** the only new code is the `ActionSource` implementation itself (`ClaudeActionSource`) and, per Pattern 4, one small widening of the `ActionSource` Protocol's return type to carry a reasoning string through the same existing logging call site.

### Anti-Pattern 3: Two independent camera-device opens for the same physical AR0144

**What people might do:** "fix" the recording path in isolation by swapping its `cv2.VideoCapture(idx)` for a *second*, independently-constructed `StereoSplitCamera(index=name)` — technically also name-based, so it looks fixed.
**Why it's wrong:** `stereo_camera.py`'s own docstring states most webcam drivers reject a second concurrent device open — a second `StereoSplitCamera` instance risks either failing to open at all, or (worse, silently) succeeding but reading a desynced frame from a second, competing ffmpeg process against the same physical camera, reintroducing a milder version of the exact "recorded footage isn't what the model actually saw" trust gap this fix is meant to close.
**Instead:** Pattern 5 above — one shared instance, injected into both consumers.

## Integration Points

### External Services

| Service | Integration Pattern | Notes |
|---------|---------------------|-------|
| Colab-hosted PolicyServer (SmolVLA) | gRPC over ngrok tunnel, via vendored `lerobot.async_inference.RobotClient` | Existing, unchanged by this milestone except the `_ready_to_send_observation()` gating fix inside `BridgeActionSource` |
| Anthropic API (Claude) | `anthropic` Python SDK, `client.messages.create(model=..., output_config={"format": {"type": "json_schema", ...}}, messages=[...text..., *image_blocks])` | New. Single-shot structured-output call per action chunk — no tool-use loop, no agentic runner, since the model never executes code (sandboxing explicitly out of scope per PROJECT.md) |

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| `ActionSource` ↔ `run_episode()` | `get_action(joint_state, instruction) -> (action, model_version[, reasoning])` | The one seam every provider (scripted/VLA/MLLM) must speak; widened by Pattern 4, not replaced |
| `ActionSource` ↔ `safety_validator` | Plain `dict[str, float]` in `action_contract.JOINT_ORDER`'s keys/units | MLLM's JSON schema must be generated from `action_contract`, not hand-authored, to guarantee this contract holds |
| `RobotClient` (vendored) ↔ `BridgeActionSource` | `_ready_to_send_observation()`, `action_queue`/`action_queue_lock`, `_action_tensor_to_action_dict()` — never `send_action()` | Unchanged boundary, just one new call site (`get_action()`) consulting an already-existing library method it previously ignored |
| `StereoSplitCamera` ↔ (`connect_bridge()`'s observation patch, `IOLogger`'s recording) | Both consumers hold a reference to the *same* instance; one via `get_observation_with_stereo_split()`'s closure (`robot_client.py:169-176`), the other via a new `_StereoHalfReader` adapter | New shared-ownership relationship — previously only the inference path held a `StereoSplitCamera`; the recording path held nothing of the sort |

## Recommended Build Order

PROJECT.md's own scoping already fixes the tick-latency fix as first (Key Decision: "the MLLM phase's action-chunk sizing depends on real timing/chunking data that only comes out of fixing and measuring the latency bug first"). Confirmed correct by the source-level analysis above — chunk-size tuning for `ClaudeActionSource` genuinely cannot be reasoned about until the real drain-vs-network-round-trip ratio is measured. Recommended full order, with the camera fix moved earlier than a literal reading of the milestone's active-item list might suggest:

1. **Tick-latency fix** (`_ready_to_send_observation()` gating in `BridgeActionSource.get_action()`, plus the `pop_validated_action()`/`client.latest_action` bookkeeping fix). No dependency on anything else in this milestone. Directly testable against the existing SmolVLA baseline — re-run the same Pen Transfer-adjacent scripted/VLA episode and confirm real-action yield rises well above 1/60 before touching anything else.
2. **Safety-validator cap re-tightening**, immediately after (1), not deferred. The caps were loosened *specifically* to compensate for staleness caused by the latency bug (`FINDINGS.md §2`, commits `d8c2785`/`8e2060a`) — leaving them loosened after the root cause is fixed re-opens exactly the safety margin this milestone's own PROJECT.md Active list wants restored, and doing it as a distinct, quickly-verified step (not bundled into a bigger change) keeps the causal link between "why were these loosened" and "why are they now safe to tighten" auditable in the commit history.
3. **Camera-resolution unification** (Pattern 5), argued out of the milestone's tentative "third" position into second. Rationale: it is completely independent of both (1) and (4) — it touches only `stereo_camera.py` and `run_vla_episode.py`'s camera-opening code, never `safety_validator.py`/`robot_client.py`'s control-loop logic. Landing it before the MLLM work means (a) the *next* live SmolVLA re-verification run (needed anyway, to confirm fix (1) worked) also produces a trustworthy recorded episode instead of another laptop-webcam capture, and (b) the MLLM phase's own camera-frame input (Pattern 3) and the "verify what was actually sent to Colab" thumbnail/hash logging (a separate, still-open Active item) both build on top of camera plumbing already known to be correct, rather than debugging two unfamiliar systems (Claude's vision input AND a still-broken camera path) at once.
4. **MLLM raw-JSON control loop** (`ClaudeActionSource` + reasoning-trace capture, Patterns 2-4). Depends on (1) for chunk-size tuning data and benefits from (3) being already landed for trustworthy camera input. Build the `ClaudeActionSource`/schema/reasoning-trace-logging plumbing as one unit — they're small, tightly coupled, and the Protocol widening (Pattern 4) touches the same call sites as the new source itself.
5. **Pen Transfer comparison run.** Depends on all four above: needs a fixed, trustworthy control loop (1,2), correct recorded camera evidence to actually judge the comparison against (3), and the MLLM source to exist (4).
6. **v2.0 tech-debt cleanup items** (`WRIST_ROLL_LIMIT_DEG` pin, `latency_ms` fix, docstring fix, untracked mesh dirs) — orthogonal to all of the above; can be interleaved wherever convenient, but note the `latency_ms` fix (currently always `{0,0}`, `FINDINGS.md §2`) is genuinely useful to land alongside (1), since it's the natural instrumentation point to *prove* the latency fix worked with real per-step numbers instead of only `timestamp_utc` deltas.

## Sources

- `control/vla_bridge/robot_client.py` (this project) — `connect_bridge()`, `_wire_stereo_split_cameras()`, `pop_validated_action()`, `BridgeActionSource` — read in full, 2026-09-24
- `control/vla_bridge/safety_validator.py` (this project) — `validate_action()` and its cap constants — read in full
- `control/vla_bridge/io_logger.py` (this project) — `IOLogger.write_step()`/`capture_camera_frame()` — read in full
- `control/vla_bridge/action_contract.py` (this project) — `JOINT_ORDER`, `ACTION_UNITS`, `load_joint_limits_deg()` — read in full
- `control/vla_bridge/stereo_camera.py` (this project) — `StereoSplitCamera`, `_FFmpegAVFoundationCapture` — read in full
- `control/run_vla_episode.py` (this project) — `run_episode()`, `main()`, `ActionSource` Protocol, `_build_camera_names()` — read in full
- `control/vla_bridge/FINDINGS.md` (this project) — Phase 11 go/no-go, root cause of the tick-latency bug — read in full
- `control/.venv/lib/python3.12/site-packages/lerobot/async_inference/robot_client.py` (installed `lerobot==0.6.1`, HIGH confidence — primary vendored source, not docs) — `RobotClient.__init__`, `receive_actions()`, `_aggregate_action_queues()`, `control_loop_action()`, `_ready_to_send_observation()`, `control_loop_observation()`, `control_loop()` — read in full, confirms the gating mechanism this fix reuses
- `control/.venv/lib/python3.12/site-packages/lerobot/async_inference/configs.py` (installed `lerobot==0.6.1`) — `RobotClientConfig.chunk_size_threshold`/`fps`/`environment_dt` — read in full
- `.planning/PROJECT.md` (this project) — v2.1 milestone scope, Active requirements, Key Decisions table — read in full
- Anthropic `claude-api` skill reference (bundled skill, cached 2026-06-24 model/pricing table) — `output_config.format` structured-output pattern, multi-image vision input shape, model selection guidance — consulted for the MLLM integration design (Patterns 2-3)

---
*Architecture research for: SoARM VLA Research v2.1 (MLLM Raw-Autonomy Benchmark)*
*Researched: 2026-09-24*
