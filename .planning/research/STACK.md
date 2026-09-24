# Stack Research

**Domain:** MLLM raw-autonomy robot control loop + async control-loop latency fix + macOS camera device resolution (v2.1 milestone additions to `control/vla_bridge/`)
**Researched:** 2026-09-24
**Confidence:** HIGH

This document covers only the **three new v2.1 capabilities**. It does not re-research the already-working VLA/safety-validator/bridge stack (`lerobot==0.6.1`, `robot_client.py`, `safety_validator.py`, `io_logger.py`, `stereo_camera.py`) — those are treated as fixed integration points the new work plugs into.

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| `anthropic` (Python SDK) | `1.8.0` (current on PyPI as of 2026-09-24; verified directly via `pip index versions anthropic`) | Calls the Claude Messages API for the MLLM raw-JSON control loop | Official first-party SDK. Project mandates the Anthropic Python SDK for any Claude/Anthropic integration (per project skill policy) — never raw `requests`/`httpx` calls to `/v1/messages`. `control/` already has `httpx==0.28.1` and Python 3.12 installed, both compatible with `anthropic` 1.x (SDK requires Python ≥3.10, uses `httpx` internally — no version conflict introduced). |
| `output_config: {"format": {"type": "json_schema", "schema": {...}}}` (Messages API structured-output feature, not a package) | GA, no beta header | Forces Claude's response to validate against an explicit JSON schema for the action-chunk output | This is the mechanism that satisfies the milestone's "no pre-built movement primitives" constraint: **use structured output, not tool-use/function-calling.** Tool-use would reintroduce a `move_to()`/`grasp()`-shaped abstraction (a named tool the model calls); structured output makes the model emit the JSON action chunk directly as its response text, which is what "reasons directly to raw joint-level JSON" requires. `client.messages.parse()` (Pydantic-model variant) is the simplest way to get this with a validated Python object back (`response.parsed_output`); the raw `output_config.format` + `json.loads()` path works identically without a Pydantic dependency. |
| `claude-opus-5` (model ID) | current | The MLLM "brain" reasoning over instruction + calibration + joint state + camera frames | Anthropic's current most-capable widely-released model for demanding reasoning/vision tasks — appropriate default for a research comparison against a fine-tuned VLA baseline, where reasoning quality (not per-call cost) is the variable under test. `claude-sonnet-5` ($3/$15 per 1M vs. Opus 5's $5/$25) is a reasonable cost-conscious substitution for iterating on the control-loop code itself (schema debugging, chunk-size tuning) before running the real Pen Transfer comparison episode on Opus 5 — swap the model string only, no other code changes required. |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `pydantic` | any modern 2.x (not yet in `control/requirements.txt` — add explicitly, `anthropic` does not vendor it) | Defines the action-chunk response schema (`reasoning: str`, `actions: list[JointAction]`) for `client.messages.parse()` | Use if you want a validated Python object back directly (`response.parsed_output`) instead of hand-parsing `json.loads()` on the raw-schema path. Either is fine; Pydantic buys you IDE/type-checking on `JOINT_ORDER`-shaped fields and is the SDK's own "recommended" structured-output path. |
| `pillow` (already installed, `12.3.0`) | already pinned | Encode `numpy` camera frames (from `StereoSplitCamera.read_left()`/`read_right()` or `cv2.VideoCapture.read()`) to PNG/JPEG bytes before base64-encoding for the `image` content block | No new dependency — already in `control/`'s environment. `cv2.imencode(".png", frame)` (already used by `io_logger.py`) works equally well and avoids adding a BGR→RGB conversion step; either is acceptable, prefer reusing `cv2.imencode` since `io_logger.py` already establishes that convention. |
| `queue.Queue` / `threading` (Python stdlib) | stdlib | Draining-aware control loop for the tick-latency fix | **No new package needed.** `RobotClient.action_queue` (from the existing `lerobot` bridge) is already a stdlib `queue.Queue` drained via `action_queue.get_nowait()` under `action_queue_lock` (see `robot_client.py::pop_validated_action`). The fix is a control-flow change — check `client.action_queue.qsize()` before deciding whether to call `client.control_loop_observation()` again — not a new async/networking library. |
| `time.monotonic()` (Python stdlib) | stdlib | Correctly measuring real tick/round-trip latency to populate `io_logger.py`'s currently-always-`{0,0}` `latency_ms` field | Use `monotonic()`, not `time.time()`/wall-clock deltas (which is what `FINDINGS.md`'s ~11-20s/tick numbers were derived from post-hoc, via `timestamp_utc` diffing, because `latency_ms` was never actually populated) — wall-clock is subject to NTP adjustment; monotonic is immune and is the correct primitive for interval timing. |
| `ffmpeg` (system binary, already a runtime dependency via `stereo_camera.py`) | system-installed (macOS: `brew install ffmpeg`) | Robust name-based AVFoundation camera capture for the `camera_overhead` recording-path fix | **No new library** — `stereo_camera.py`'s `_FFmpegAVFoundationCapture` already solves exactly this problem (`ffmpeg -f avfoundation -i "<device name>"`, not a numeric index) for the *inference-input* path. Reuse it (or its underlying subprocess pattern) for the *IOLogger recording* path instead of `run_vla_episode.py`'s current bare `cv2.VideoCapture(numeric_index)`. |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| `client.messages.count_tokens(...)` (Anthropic SDK method, no new package) | Estimate per-call cost before running a live Pen Transfer episode | Useful given this is a paid-tier API and the milestone explicitly wants a small, analyzable first comparison run — check token cost per multi-image + calibration-data prompt before committing to a full 60-step episode budget. |
| `ant auth status` / `ant auth login` (Anthropic CLI, optional) | Credential resolution without hardcoding `ANTHROPIC_API_KEY` | Optional convenience; a plain `ANTHROPIC_API_KEY` env var (matching this repo's existing no-`.env`, inline-env-var convention per `CLAUDE.md`) works identically with a bare `anthropic.Anthropic()` client. |

## Installation

```bash
# In control/'s existing venv (Python 3.12) — add to control/requirements.txt
pip install anthropic pydantic

# ffmpeg is a system binary, not a pip package — already required by
# stereo_camera.py; verify it's present (macOS):
brew install ffmpeg   # if not already installed
```

`control/requirements.txt` addition:
```
anthropic==1.8.0
pydantic>=2.0
```

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|--------------------------|
| Structured output (`output_config.format` / `messages.parse()`) for the action-chunk JSON | Tool-use (`tools=[...]`, forced `tool_choice`) with a single `emit_action_chunk` tool wrapping the same schema | Only if a future milestone phase reintroduces multi-turn tool-calling (e.g. the model deciding to call a "look closer" camera-zoom tool mid-reasoning). For v2.1's raw-autonomy design, tool-use is explicitly the wrong shape — it's a named-function abstraction, which is exactly what "no pre-built movement primitives" rules out. Structured output is the correct mechanism here, not a stylistic preference. |
| `client.messages.parse()` (Pydantic) | Raw `output_config: {"format": {"type": "json_schema", ...}}` + `json.loads()` | Use raw schema if you want to avoid adding `pydantic` as a dependency, or need a schema shape Pydantic can't express directly (e.g. deeply dynamic per-joint bounds pulled from `action_contract.JOINT_ORDER` at runtime — trivial to build as a raw JSON-schema dict, slightly more code as a Pydantic model with a variable field set). |
| `claude-opus-5` for the real comparison episode | `claude-sonnet-5` | Use Sonnet 5 while iterating on the control loop's plumbing (schema shape, chunk-size tuning, prompt structure) to cut cost ~40% per call; switch to Opus 5 for the actual Pen Transfer run being compared against the SmolVLA baseline, since reasoning quality is the variable under test. |
| Fix `camera_overhead` recording by reusing the existing ffmpeg/AVFoundation name-based capture | A dedicated macOS camera library (e.g. `pyobjc`/`AVFoundation` Python bindings, `imageio-ffmpeg`) | Not needed. `stereo_camera.py` already has a working, tested `_FFmpegAVFoundationCapture` class with a `cv2`-compatible `isOpened()`/`read()`/`release()` shape — the fix is reuse, not a new dependency. Only reach for a dedicated binding if a future need requires querying AVFoundation device metadata (e.g. enumerating available devices by name programmatically) beyond what `ffmpeg -f avfoundation -list_devices true -i ""` already provides via subprocess. |
| Polling `action_queue.qsize()` to decide when to re-request inference | Rewriting the bridge on `asyncio`/`aiohttp` | Not warranted. The existing `lerobot` `RobotClient` bridge is a synchronous, `threading`+`grpc`-based design (`receive_actions()` runs in a daemon thread, `action_queue` is a stdlib `queue.Queue`). Introducing `asyncio` here would mean rewriting the vendored `lerobot` integration, not scoping a fix — the FINDINGS.md root cause is a control-flow bug (`control_loop_observation()` called unconditionally every tick), fully fixable within the existing threading model. |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|--------------|
| Tool-use / function-calling (`tools=[{"name": "move_to", ...}]`) for the MLLM's action output | Reintroduces exactly the pre-built-movement-primitive abstraction the milestone explicitly rules out ("no `move_to()`/`grasp()` helpers... reasons directly to raw joint-level JSON") | `output_config.format` (structured JSON output) — the model's entire response *is* the action chunk, not a tool call describing one |
| `client.messages.create()` with unconstrained free-text output + regex/manual JSON extraction from prose | Fragile — no schema guarantee, and the existing `safety_validator.py` expects a clean `dict[str, float]` per joint; parsing failures would need their own error-handling path that structured output makes unnecessary | `output_config.format` / `client.messages.parse()` — the SDK guarantees the first text block is valid JSON matching your schema |
| `cv2.VideoCapture(numeric_index)` for the AR0144 stereo/overhead camera, in *any* code path (recording or inference) | Confirmed live twice this session to silently capture the laptop's built-in webcam instead of the robot workspace — macOS AVFoundation numeric indices are known to drift across process launches (already documented in `stereo_camera.py` and `run_vla_episode.py`'s own `--stereo-camera-index` help text) | Device-**name**-keyed capture (`ffmpeg -f avfoundation -i "CCB Camera"`, i.e. `stereo_camera.py`'s existing `_FFmpegAVFoundationCapture`), reading the name from `device_map.json`'s `cameras.stereo_overhead_name` |
| Opening a second, independent `StereoSplitCamera`/ffmpeg capture of the AR0144 for the IOLogger recording path while `connect_bridge()`'s own `StereoSplitCamera` is already open for inference | `stereo_camera.py`'s own docstring: "most webcam drivers reject a second concurrent open of the same index" — a second independent open of the same physical AVFoundation device is likely to fail or contend, not just be redundant | Reuse the **same** `StereoSplitCamera` instance `connect_bridge()` already created (reachable via `client._stereo_camera`) for both the inference-input path and the recording path — one shared capture, two consumers, same pattern `stereo_camera.py` already uses internally for `read_left()`/`read_right()` |
| `asyncio`/`aiohttp` rewrite of the control loop to fix tick latency | Existing bridge is synchronous/threaded via vendored `lerobot`; an asyncio rewrite is a disproportionate architecture change for a scoped control-flow bug | Conditional re-request logic (`if client.action_queue.qsize() < N: client.control_loop_observation(...)`) within the existing threading model, and/or tuning `control_hz` to the measured real round-trip |
| `time.time()` for latency measurement in the fixed `io_logger.py` `latency_ms` field | Wall-clock time is subject to NTP adjustment and was already the (indirect, via `timestamp_utc` diffing) source of FINDINGS.md's imprecise ~11-20s/tick numbers | `time.monotonic()` deltas, captured at the actual request-send and response-receive points, not reconstructed after the fact from log timestamps |

## Stack Patterns by Variant

**If iterating on the MLLM control-loop code/schema before spending real API budget:**
- Use `claude-sonnet-5` with a short `--max-steps` test episode (or a scripted/mocked camera+joint-state fixture, no real hardware)
- Because Sonnet 5 is ~40% cheaper per call and the plumbing (schema validation, `safety_validator.py` handoff, `io_logger.py` reasoning-trace field) is identical between Sonnet 5 and Opus 5 — no code path differs by model choice

**If running the real Pen Transfer comparison episode against the Phase 11 SmolVLA baseline:**
- Use `claude-opus-5`
- Because reasoning quality (not cost) is the variable under test in this comparison, and this is a single, deliberately small (per Key Decisions) live-hardware run, not a high-volume workload

**If a future phase adds a provider-agnostic router (explicitly deferred, Future Requirements):**
- Structured output is not uniformly available across providers the same way — re-verify each provider's JSON-schema/structured-output support before assuming this pattern ports unchanged
- Out of scope for v2.1; noted here only so the router design doesn't silently assume Anthropic-specific `output_config` semantics

## Version Compatibility

| Package A | Compatible With | Notes |
|-----------|------------------|-------|
| `anthropic==1.8.0` | Python 3.12 (this project's `control/` venv), `httpx==0.28.1` (already installed) | `anthropic` 1.x requires Python ≥3.10 and uses `httpx` internally; both constraints already satisfied by `control/`'s existing environment — no version bump needed elsewhere. |
| `anthropic==1.8.0` | `grpcio==1.84.0`, `lerobot==0.6.1` (already installed) | No shared dependency conflict — `anthropic`'s HTTP transport (`httpx`) and `lerobot`'s bridge transport (`grpc`) are independent stacks; the MLLM call and the robot bridge call are separate network paths that never share a client object. |
| Structured output (`output_config.format`) | Extended thinking (`thinking: {"type": "adaptive"}`) | Compatible together — thinking blocks (if `display: "summarized"` is set) appear before the schema-constrained text block in `response.content`; the schema guarantee applies only to the final text block, not to thinking. Not compatible with `citations: {enabled: true}` on document blocks (returns 400) — irrelevant here since no PDF/document input is used. |
| `pydantic>=2.0` | `anthropic==1.8.0`'s `messages.parse()` | The SDK's structured-output helper is built for Pydantic v2-style `BaseModel`s; do not pin an old Pydantic v1 model anywhere else in `control/` if adding this. |

## Sources

- Bundled `claude-api` skill (`python/claude-api/README.md`, `python/claude-api/tool-use.md`) — Anthropic-authored reference covering current Messages API shape (`output_config.format`, `messages.parse()`, vision content blocks, model IDs/pricing, thinking/effort). Confidence: HIGH (official-equivalent, cross-checked against the skill's own "verify against `{lang}/` files, not training-prior" instruction).
- `pip index versions anthropic` / `pip download anthropic` run directly against PyPI (2026-09-24) — confirmed `anthropic` 1.8.0 is current, Python SDK is on the 1.x major line. Confidence: HIGH (direct registry observation, not a cached/provider claim).
- Direct reads of this repo's own `control/vla_bridge/robot_client.py`, `stereo_camera.py`, `safety_validator.py`, `io_logger.py`, `action_contract.py`, `run_vla_episode.py`, `device_map.json`, `FINDINGS.md`, and `control/requirements.txt` (2026-09-24) — ground truth for integration points, existing dependency versions, and the exact root cause the latency/camera fixes target. Confidence: HIGH (primary source, this project's own code).
- WebSearch cross-check on `anthropic` PyPI latest version — corroborated the direct pip check but itself returned a stale cached claim (`0.116.0`); **the direct pip index check (1.8.0) is authoritative**, not the search result. Confidence of the WebSearch result alone: LOW — included only to note the discrepancy, not relied upon.

---
*Stack research for: MLLM raw-autonomy robot control loop, async control-loop latency fix, macOS camera device resolution (SoARM VLA Research v2.1)*
*Researched: 2026-09-24*
