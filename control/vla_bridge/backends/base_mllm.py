"""Shared chunk-drain base for every MLLM-style backend (Phase 15, PIPE-02, MLLM-04..07).

What this is
============
`MllmActionSourceBase` is a `run_vla_episode.ActionSource` that asks a general
multimodal reasoning model (Claude, GPT, an HF-router model) for a CHUNK of N
raw absolute joint targets, then drains that chunk one step per control tick:

  - When its queue is empty it makes ONE blocking provider call (the robot
    holds meanwhile, because `get_action()` is synchronous), parses the JSON,
    pre-validates every step and enqueues exactly N items. Every tick pops
    one item. There are no per-tick calls (D-00d, D-08).
  - Every call sends the same system prompt (`mllm_prompt.build_system_prompt`,
    built once) and the same schema (`mllm_schema.build_action_chunk_schema`,
    built once). Providers differ only in a small adapter object and a config
    file; `register_mllm_backend()` adds one without editing this module
    (D-00a).
  - The model emits raw absolute joint targets only. There are no movement
    primitives and nothing from the model is executed except those numbers,
    which still pass through the UNMODIFIED `safety_validator.validate_action()`
    inside `run_episode()` (D-04). Nothing here pre-clamps a value.

Adapter contract (duck-typed; Plans 07-08 supply the real ones)
---------------------------------------------------------------
Attributes `provider: str`, `model: str`, `sdk: {"name", "version"}`,
`settings_summary: {"requested": {}, "applied": {}, "omitted": {param: reason}}`;
method `call(request: ProviderRequest) -> ProviderResult` (any `Exception` is a
transport failure); optional `close()`. Every adapter sends, for each image in
IMAGE_ORDER that is present, a text label "Image: <role>" immediately followed
by the image, then the user text block.

Collector contract: `collect(joint_state) -> MllmInputs | None` (None = camera
read failed); optional `close()`. Recorder contract (optional, Plan 04):
`finalize(record, *, chunk_id, request, result, inputs, error) -> dict`.

Hold semantics (D-00c, D-03, MLLM-07)
-------------------------------------
`get_action()` never raises (except `KeyboardInterrupt`, which must reach
`run_episode()`'s return-to-start path). A hold returns
`(dict(joint_state), "<label>@<reason>-holding-position")`, where
`joint_state` is the live reading passed to THAT tick: never zeros, never a
previous target. Reasons:

  - `malformed-step`: step k of the chunk has a missing key, a non-numeric,
    null, boolean, NaN/Infinity or float-overflowing value. Steps before k
    run; step k and the rest of the chunk hold. No retry call.
  - `short-chunk`: fewer than N steps came back; the missing tail holds.
    Extra steps beyond N are dropped and counted (`parse.extra_dropped`).
  - `provider-error`, `refusal`, `truncated`, `empty-response`,
    `invalid-json`, `missing-actions`, `internal-error`: the whole call
    failed, so N holds are enqueued and the next paid call happens only after
    N ticks (research Pitfall 4).
  - `camera-read-failed`: the collector returned None; a single hold, no
    provider call, no chunk id consumed.
  - `request-inflight`: a concurrent call is in progress; a single hold.
  - `task-complete`: the model answered `task_complete: true` (see below);
    the whole chunk holds.
  - `episode-complete`: `episode_complete_reason` is already set and the
    queue is empty; a single hold with no collector and no provider call
    (a runaway-spend guard for a loop that ignores the flag).

End of episode (D-18, D-19, D-25, research Pitfall 15)
-------------------------------------------------------
`episode_complete_reason` starts as None. `run_vla_episode.run_episode()`
reads it after every tick and ends the episode with that termination reason
once it is a non-empty string:

  - `"mllm_backend_failed"` (EPISODE_BACKEND_FAILED): every whole-call failure
    (provider exception, refusal/truncated/other stop, empty, invalid JSON,
    missing actions, internal error) adds one to a consecutive-failure
    counter; reaching `max_consecutive_failed_calls` sets this reason. A call
    whose answer contained an `actions` list resets the counter to 0. Camera
    read failures never touch it (no call was made, nothing was paid).
  - `"task_complete"` (EPISODE_TASK_COMPLETE): the parsed answer's
    `task_complete` is the boolean `true` (identity check: the string "true",
    1 or null do not count; the raw value is kept in
    `trace.task_complete_raw`). D-25 resolution: the claim is the model's
    judgement about the CURRENT views, so no motion may follow it. All N
    queue items become `task-complete` holds; the model's actions are never
    executed and survive only in the raw payload/sidecar. A failed call never
    sets `task_complete_claimed`. The claim is recorded on its own
    (`trace.task_complete`, `task_complete_claimed`) and is not an outcome
    label (D-19): Phase 16 judges success separately.

Per-call record (D-12, D-22)
----------------------------
On the tick that refills the queue, `last_details["mllm_call"]` holds the full
record (staged fields `task_restatement`/`plan`/`scene_claims`/`task_complete`,
native thinking text when the provider returns it, usage, stop, parse result,
provenance hashes, latency, image hashes). Every tick carries only
`last_details["chunk_ref"] = {chunk_id, step_index, chunk_len}`. Everything in
`last_details` is plain JSON (research Pitfall 6).

Secret discipline (T-15-03; mirrors base_gemini.py)
---------------------------------------------------
This class never sees a client or request headers. The `secrets` it is given
are used only to redact error text and the record; they never appear in
`last_details` or `repr()`. Fuller record and sidecar redaction lives in the
Plan 04 recorder.

This module deliberately imports neither `lerobot` nor `cv2` (nor any
provider SDK): it is pure orchestration, and the hardware builder lives in a
separate module (Plan 06).
"""

import copy
import hashlib
import json
import math
import numbers
import re
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from vla_bridge.action_contract import JOINT_ORDER
from vla_bridge.backends import load_backend_config, register
from vla_bridge.mllm_prompt import (
    PROMPT_TEMPLATE_VERSION,
    build_system_prompt,
    build_user_text,
    template_sha256,
    text_sha256,
)
from vla_bridge.mllm_schema import SCHEMA_NAME, build_action_chunk_schema, schema_sha256

HOLD_SUFFIX = "-holding-position"
CHUNK_SUFFIX = "mllm-chunk"

# Normalised stop reasons every adapter returns in ProviderResult.stop["reason"].
STOP_COMPLETE = "complete"
STOP_REFUSAL = "refusal"
STOP_TRUNCATED = "truncated"
STOP_OTHER = "other"
# Record-only stop reason for a call that raised (never returned by an adapter).
_STOP_ERROR = "error"

# Episode-end reasons, read by run_vla_episode.run_episode()'s termination hook.
EPISODE_TASK_COMPLETE = "task_complete"
EPISODE_BACKEND_FAILED = "mllm_backend_failed"

IMAGE_ORDER = ("wrist", "left", "right", "depth")

MAX_LOGGED_TEXT_CHARS = 2000
MAX_INLINE_THINKING_CHARS = 20000

ACTION_PATH = (
    "mllm raw absolute joint targets (JSON chunk of N steps from a general multimodal model), "
    "drained one step per control tick; no movement primitives; every step passes the "
    "unmodified safety validator"
)

DEFAULTS_CONFIG_NAME = "mllm_defaults"
MAX_CHUNK_LEN = 200

# Hold reasons.
REASON_INFLIGHT = "request-inflight"
REASON_CAMERA = "camera-read-failed"
REASON_PROVIDER = "provider-error"
REASON_REFUSAL = "refusal"
REASON_TRUNCATED = "truncated"
REASON_MALFORMED = "malformed-step"
REASON_SHORT = "short-chunk"
REASON_INTERNAL = "internal-error"
REASON_TASK_COMPLETE = "task-complete"
REASON_EPISODE_COMPLETE = "episode-complete"

_STOP_TO_REASON = {STOP_REFUSAL: REASON_REFUSAL, STOP_TRUNCATED: REASON_TRUNCATED}
_STAGED_FIELDS = ("task_restatement", "plan", "scene_claims", "task_complete")
# Config keys merged one level deep when a per-backend file overrides them (D-02).
_MERGE_DICT_KEYS = ("sampling", "depth", "prices")
_FENCE_RE = re.compile(r"^```[A-Za-z]*[ \t]*\n?(.*?)\n?[ \t]*```$", re.S)


# --- Contract dataclasses -------------------------------------------------------------


@dataclass(frozen=True)
class MllmBackendSpec:
    """One registered MLLM backend: `adapter_factory(api_key, config) -> adapter`."""

    name: str
    config_name: str
    adapter_factory: Callable


@dataclass
class MllmInputs:
    """What a collector gathered for one call (images keyed by IMAGE_ORDER role)."""

    images: dict[str, bytes] = field(repr=False)
    depth_note: str
    meta: dict
    sidecar_files: dict[str, bytes] = field(repr=False)
    image_media_type: str = "image/png"


@dataclass
class ProviderRequest:
    """Everything an adapter sends: identical prompt/schema on every call."""

    system_prompt: str = field(repr=False)
    user_text: str
    images: list[tuple[str, bytes]] = field(repr=False)
    schema: dict = field(repr=False)
    schema_name: str
    image_media_type: str


@dataclass
class ProviderResult:
    """One provider answer, normalised.

    `usage` keys: input_tokens, output_tokens, reasoning_tokens (None if
    unreported), cached_input_tokens (None if unknown), raw (plain dict).
    `stop` keys: reason (a STOP_* value), refusal_category, incomplete_reason.
    """

    text: str | None
    raw_payload: Any = field(repr=False)
    usage: dict
    stop: dict
    native_thinking_text: str | None = field(default=None, repr=False)
    native_thinking_kind: str = "none"
    request_id: str | None = None
    response_id: str | None = None
    model_returned: str | None = None


@dataclass
class ParsedChunk:
    doc: dict = field(repr=False)
    steps: list[dict]
    first_bad_index: int | None
    reason: str | None
    reason_detail: str | None
    n_returned: int
    missing_top_level: list[str]
    extra_dropped: int
    fence_stripped: bool


class ChunkParseError(ValueError):
    """A whole-response failure: `reason` is empty-response, invalid-json or missing-actions."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


# --- Registration (D-00a) ---------------------------------------------------------------

MLLM_BACKENDS: dict[str, MllmBackendSpec] = {}


def register_mllm_backend(name: str, *, config_name: str, adapter_factory: Callable):
    """Decorator: register the decorated `build()` under `name` in the shared
    BACKEND_REGISTRY (so `--backend name` works) and record its spec in
    MLLM_BACKENDS."""

    def decorator(build):
        register(name)(build)
        MLLM_BACKENDS[name] = MllmBackendSpec(name, config_name, adapter_factory)
        return build

    return decorator


# --- Settings (D-02) ------------------------------------------------------------------------


def _config_path(config_name: str) -> str:
    return f"control/vla_bridge/backend_configs/{config_name}.json"


def _is_real(value) -> bool:
    return (not isinstance(value, bool) and isinstance(value, numbers.Real)
            and math.isfinite(value))


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_settings(settings: dict, where: str) -> None:
    def fail(field_name: str, requirement: str):
        raise SystemExit(f"MLLM settings ({where}): '{field_name}' must be {requirement}.")

    chunk_len = settings.get("chunk_len")
    if not _is_int(chunk_len) or not 1 <= chunk_len <= MAX_CHUNK_LEN:
        fail("chunk_len", f"an int between 1 and {MAX_CHUNK_LEN}")
    if not _is_real(settings.get("step_dt_s")) or settings["step_dt_s"] <= 0:
        fail("step_dt_s", "a finite positive number of seconds")
    for key in ("include_urdf", "disclose_safety_caps"):
        if not isinstance(settings.get(key), bool):
            fail(key, "true or false")
    if settings.get("left_image_source") not in ("raw", "aligned"):
        fail("left_image_source", "'raw' or 'aligned'")
    if settings.get("depth_presentation") not in ("heatmap", "grid", "both"):
        fail("depth_presentation", "'heatmap', 'grid' or 'both'")
    if settings.get("image_format") not in ("png", "jpeg"):
        fail("image_format", "'png' or 'jpeg'")
    if not _is_int(settings.get("image_max_side")) or settings["image_max_side"] < 64:
        fail("image_max_side", "an int >= 64")
    if not _is_int(settings.get("max_consecutive_failed_calls")) or settings["max_consecutive_failed_calls"] < 1:
        fail("max_consecutive_failed_calls", "an int >= 1")
    if not _is_real(settings.get("timeout_s")) or settings["timeout_s"] <= 0:
        fail("timeout_s", "a finite positive number of seconds")

    sampling = settings.get("sampling")
    if not isinstance(sampling, dict):
        fail("sampling", "an object with 'temperature' and 'top_p'")
    temperature = sampling.get("temperature")
    if temperature is not None and (not _is_real(temperature) or temperature < 0):
        fail("sampling.temperature", "a finite number >= 0 or null")
    top_p = sampling.get("top_p")
    if top_p is not None and (not _is_real(top_p) or not 0 < top_p <= 1):
        fail("sampling.top_p", "a number in (0, 1] or null")

    depth = settings.get("depth")
    if not isinstance(depth, dict):
        fail("depth", "an object")
    range_m = depth.get("range_m")
    if (not isinstance(range_m, list) or len(range_m) != 2 or not all(_is_real(v) for v in range_m)
            or not 0 < range_m[0] < range_m[1]):
        fail("depth.range_m", "two increasing positive numbers of metres")
    for key in ("grid_rows", "grid_cols"):
        if not _is_int(depth.get(key)) or depth[key] < 1:
            fail(f"depth.{key}", "an int >= 1")
    if not _is_int(depth.get("retry_429")) or depth["retry_429"] < 0:
        fail("depth.retry_429", "an int >= 0")
    if not _is_real(depth.get("retry_sleep_s")) or depth["retry_sleep_s"] < 0:
        fail("depth.retry_sleep_s", "a finite number >= 0")


def load_mllm_settings(config_name: str | None = None) -> dict:
    """Shared `mllm_defaults.json`, optionally overridden by `<config_name>.json`.

    A per-backend value wins; dict-valued keys (sampling, depth, prices) merge
    one level deep. Extra provider keys (model, api_key_env, ...) pass through
    untouched. Raises `SystemExit` naming the file or the bad field.
    """
    defaults = load_backend_config(DEFAULTS_CONFIG_NAME)
    if defaults is None:
        raise SystemExit(
            f"{_config_path(DEFAULTS_CONFIG_NAME)} is missing or unreadable; every MLLM backend "
            "needs these shared defaults."
        )
    settings = copy.deepcopy(defaults)
    where = _config_path(DEFAULTS_CONFIG_NAME)
    if config_name is not None:
        override = load_backend_config(config_name)
        if override is None:
            raise SystemExit(
                f"{_config_path(config_name)} is missing or unreadable (a JSON object of "
                "per-backend MLLM settings is required)."
            )
        for key, value in override.items():
            if key in _MERGE_DICT_KEYS and isinstance(value, dict) and isinstance(settings.get(key), dict):
                settings[key] = {**settings[key], **copy.deepcopy(value)}
            else:
                settings[key] = copy.deepcopy(value)
        where = f"{where} + {_config_path(config_name)}"
    _validate_settings(settings, where)
    return settings


# --- Parsing and pre-validation (D-03) -------------------------------------------------------


def _validate_step(item) -> tuple[dict | None, str | None]:
    """A clean `{joint: float}` step, or `(None, "<joint> <kind>")` naming the
    first problem (never the raw value)."""
    if not isinstance(item, dict):
        return None, "not an object"
    step = {}
    for joint in JOINT_ORDER:
        if joint not in item:
            return None, f"{joint} missing"
        value = item[joint]
        if value is None:
            return None, f"{joint} null"
        if isinstance(value, bool):
            return None, f"{joint} boolean"
        if not isinstance(value, numbers.Real):
            return None, f"{joint} non-numeric"
        try:
            number = float(value)
        except (OverflowError, ValueError, TypeError):
            return None, f"{joint} overflow"
        if not math.isfinite(number):
            return None, f"{joint} non-finite"
        step[joint] = number
    return step, None


def parse_chunk(text, chunk_len: int) -> ParsedChunk:
    """Parse and pre-validate one provider answer.

    Raises `ChunkParseError` only for whole-response failures (empty,
    not JSON / not an object, no `actions` list). Everything else is reported
    in the returned `ParsedChunk`: the valid prefix of steps, the first bad
    index and why, extra steps dropped, absent staged fields.
    """
    if not isinstance(text, str) or not text.strip():
        raise ChunkParseError("empty-response")
    candidate = text.strip()
    fence_stripped = False
    fenced = _FENCE_RE.match(candidate)
    if fenced:
        candidate = fenced.group(1).strip()
        fence_stripped = True
    try:
        # Default parser on purpose: a NaN/Infinity literal in step k must not
        # discard the good steps before it; the per-step check rejects it.
        doc = json.loads(candidate)
    except (ValueError, RecursionError) as exc:
        raise ChunkParseError("invalid-json", type(exc).__name__) from None
    if not isinstance(doc, dict):
        raise ChunkParseError("invalid-json", "top-level JSON is not an object")
    actions = doc.get("actions")
    if not isinstance(actions, list):
        raise ChunkParseError("missing-actions", "no 'actions' list")

    steps: list[dict] = []
    first_bad_index = reason = reason_detail = None
    for k, item in enumerate(actions[:chunk_len]):
        step, problem = _validate_step(item)
        if step is None:
            first_bad_index, reason = k, REASON_MALFORMED
            reason_detail = f"step {k}: {problem}"
            break
        steps.append(step)
    if first_bad_index is None and len(steps) < chunk_len:
        first_bad_index, reason = len(steps), REASON_SHORT
        reason_detail = f"{len(actions)} steps returned, {chunk_len} expected"
    return ParsedChunk(
        doc=doc,
        steps=steps,
        first_bad_index=first_bad_index,
        reason=reason,
        reason_detail=reason_detail,
        n_returned=len(actions),
        missing_top_level=[key for key in _STAGED_FIELDS if key not in doc],
        extra_dropped=max(0, len(actions) - chunk_len),
        fence_stripped=fence_stripped,
    )


# --- Helpers -------------------------------------------------------------------------------------


def _plain(value):
    """Coerce to plain JSON types (SDK objects become strings)."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        return float(value)
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return str(value)


def _ms(seconds: float) -> float:
    return round(seconds * 1000.0, 3)


def _logged_raw_step(item):
    """The offending raw step for `raw_model_output`, kept strict-JSON and small."""
    try:
        strict = json.dumps(item, allow_nan=False)
        if isinstance(item, dict) and len(strict) <= MAX_LOGGED_TEXT_CHARS:
            return item
    except (ValueError, TypeError):
        pass
    try:
        text = json.dumps(item)
    except (ValueError, TypeError):
        return None
    if len(text) > MAX_LOGGED_TEXT_CHARS:
        return None
    return {"malformed_step_json": text}


def _logged_raw_value(value):
    """A model-supplied scalar or object for the record: plain JSON, size-capped."""
    value = _plain(value)
    try:
        text = json.dumps(value, allow_nan=False)
    except (ValueError, TypeError):
        return {"unloggable_value_type": type(value).__name__}
    if len(text) > MAX_LOGGED_TEXT_CHARS:
        return {"omitted_json_chars": len(text)}
    return value


@dataclass
class _QueueItem:
    chunk_id: int
    step_index: int
    action: dict | None = None  # a step to execute; None means hold
    raw: Any = None  # what is logged as raw_model_output
    reason: str | None = None  # hold reason


# --- The action source --------------------------------------------------------------------------


class MllmActionSourceBase:
    """`run_vla_episode.ActionSource` that drains MLLM-produced joint-target chunks."""

    def __init__(
        self,
        adapter,
        collector,
        settings: dict,
        joint_limits_deg: dict,
        label: str,
        *,
        urdf_text: str | None = None,
        recorder=None,
        secrets=(),
        clock=time.monotonic,
    ):
        self.adapter = adapter
        self.collector = collector
        self.settings = settings
        self.label = label
        self.urdf_text = urdf_text
        self.recorder = recorder
        self.chunk_len = settings["chunk_len"]
        self._secrets = tuple(s for s in secrets if isinstance(s, str) and s)
        self._clock = clock

        # Built ONCE: identical on every call (Plan 15-09 compares these across backends).
        self.system_prompt = build_system_prompt(settings, joint_limits_deg, urdf_text)
        self.schema = build_action_chunk_schema(self.chunk_len)
        self.schema_hash = schema_sha256(self.schema)
        self.prompt_template_version = PROMPT_TEMPLATE_VERSION
        self._prompt_sha256 = text_sha256(self.system_prompt)
        self._template_sha256 = template_sha256()

        self._inflight_lock = threading.Lock()
        self._queue: deque[_QueueItem] = deque()
        self._chunk_counter = 0
        self._previous_plan: str | None = None
        self._pop_times: list[float] = []

        # End-of-episode state (D-18, D-25, Pitfall 15), read by run_episode().
        self.max_consecutive_failed_calls = settings["max_consecutive_failed_calls"]
        self._consecutive_failed_calls = 0
        self.episode_complete_reason: str | None = None
        self.task_complete_claimed = False
        self.last_details = self._details()

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(label={self.label!r}, "
                f"provider={getattr(self.adapter, 'provider', None)!r}, "
                f"model={getattr(self.adapter, 'model', None)!r}, chunk_len={self.chunk_len})")

    # -- small helpers --

    def _details(self) -> dict:
        return {"action_path": ACTION_PATH, "mllm_backend": self.label, "bridge_flags": []}

    def _redact(self, text: str) -> str:
        for secret in self._secrets:
            text = text.replace(secret, "<redacted>")
        return text

    def _redact_obj(self, value):
        if isinstance(value, str):
            return self._redact(value)
        if isinstance(value, dict):
            return {key: self._redact_obj(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._redact_obj(item) for item in value]
        return value

    def _error_text(self, exc: BaseException) -> str:
        return self._redact(str(exc))[:MAX_LOGGED_TEXT_CHARS]

    def _hold(self, joint_state: dict, reason: str):
        self.last_details["bridge_flags"] = [f"{reason}{HOLD_SUFFIX}"]
        return dict(joint_state), f"{self.label}@{reason}{HOLD_SUFFIX}"

    def _unchunked_ref(self) -> dict:
        return {"chunk_id": None, "step_index": None, "chunk_len": self.chunk_len}

    # -- the ActionSource method --

    def get_action(self, joint_state: dict[str, float], instruction: str) -> tuple[dict[str, float], str]:
        self.last_details = self._details()
        if not self._inflight_lock.acquire(blocking=False):
            self.last_details["chunk_ref"] = self._unchunked_ref()
            return self._hold(joint_state, REASON_INFLIGHT)
        try:
            if not self._queue and self.episode_complete_reason is not None:
                # Runaway-spend guard: once the episode is over, never capture or call again.
                self.last_details["chunk_ref"] = self._unchunked_ref()
                return self._hold(joint_state, REASON_EPISODE_COMPLETE)
            if not self._queue:
                try:
                    self._refill(joint_state, instruction)
                except Exception as exc:  # a bug here must not crash the live loop or re-call per tick
                    self.last_details["error_type"] = type(exc).__name__
                    self.last_details["error"] = self._error_text(exc)
                    if not self._queue:
                        # _refill counts a call's outcome right before enqueueing, so an
                        # empty queue means this failure has not been counted yet.
                        self._count_call_outcome(failed=True)
                        chunk_id = self._chunk_counter
                        self._chunk_counter += 1
                        self._enqueue_holds(chunk_id, 0, REASON_INTERNAL)
            if not self._queue:  # the collector failed: no call was made
                self.last_details["chunk_ref"] = self._unchunked_ref()
                return self._hold(joint_state, REASON_CAMERA)
            item = self._queue.popleft()
            self._pop_times.append(self._clock())
            self.last_details["chunk_ref"] = {
                "chunk_id": item.chunk_id,
                "step_index": item.step_index,
                "chunk_len": self.chunk_len,
            }
            if item.action is None:
                if item.raw is not None:
                    self.last_details["raw_model_output"] = item.raw
                return self._hold(joint_state, item.reason)
            self.last_details["raw_model_output"] = dict(item.raw)
            return dict(item.action), f"{self.label}@{CHUNK_SUFFIX}"
        finally:
            self._inflight_lock.release()

    # -- refill --

    def _enqueue_holds(self, chunk_id: int, start: int, reason: str, first_raw=None) -> None:
        for k in range(start, self.chunk_len):
            raw = first_raw if k == start else None
            self._queue.append(_QueueItem(chunk_id, k, raw=raw, reason=reason))

    def _count_call_outcome(self, *, failed: bool) -> None:
        """Track consecutive whole-call failures; at the cap, end the episode (Pitfall 15)."""
        if not failed:
            self._consecutive_failed_calls = 0
            return
        self._consecutive_failed_calls += 1
        if (self._consecutive_failed_calls >= self.max_consecutive_failed_calls
                and self.episode_complete_reason is None):
            self.episode_complete_reason = EPISODE_BACKEND_FAILED
            print(f"MLLM backend {self.label}: {self._consecutive_failed_calls} consecutive failed "
                  f"calls reached max_consecutive_failed_calls={self.max_consecutive_failed_calls}; "
                  f"ending the episode ({EPISODE_BACKEND_FAILED})")

    def _prev_chunk_tick_period(self) -> float | None:
        pops = self._pop_times
        if len(pops) < 2:
            return None
        return (pops[-1] - pops[0]) / (len(pops) - 1)

    def _refill(self, joint_state: dict, instruction: str) -> None:
        t_start = time.perf_counter()
        started_utc = datetime.now(timezone.utc).isoformat()
        try:
            inputs = self.collector.collect(dict(joint_state))
        except Exception as exc:
            self.last_details["error_type"] = type(exc).__name__
            self.last_details["error"] = self._error_text(exc)
            inputs = None
        if inputs is None:
            return  # caller holds once; no chunk id consumed, nothing paid

        chunk_id = self._chunk_counter
        self._chunk_counter += 1
        prev_period = self._prev_chunk_tick_period()
        self._pop_times = []

        request = ProviderRequest(
            system_prompt=self.system_prompt,
            user_text=build_user_text(joint_state, instruction, self.settings,
                                      self._previous_plan, inputs.depth_note),
            images=[(role, inputs.images[role]) for role in IMAGE_ORDER if role in inputs.images],
            schema=self.schema,
            schema_name=SCHEMA_NAME,
            image_media_type=inputs.image_media_type,
        )

        model = getattr(self.adapter, "model", "?")
        print(f"MLLM chunk {chunk_id}: calling {model} ...")
        result = error = parsed = None
        failure_reason = failure_detail = None
        t_api = time.perf_counter()
        try:
            result = self.adapter.call(request)
        except Exception as exc:  # transport/SDK: never crash the live loop
            error = exc
            failure_reason = REASON_PROVIDER
            failure_detail = self._error_text(exc)
            self.last_details["error_type"] = type(exc).__name__
            self.last_details["error"] = failure_detail
        api_s = time.perf_counter() - t_api

        parse_ms = None
        if result is not None:
            stop_reason = (result.stop or {}).get("reason")
            if stop_reason != STOP_COMPLETE:
                failure_reason = _STOP_TO_REASON.get(stop_reason, REASON_PROVIDER)
                failure_detail = f"provider stop reason {stop_reason!r}"
                self.last_details["error_type"] = "stop"
                self.last_details["error"] = failure_detail
            else:
                t_parse = time.perf_counter()
                try:
                    parsed = parse_chunk(result.text, self.chunk_len)
                except ChunkParseError as exc:
                    failure_reason = exc.reason
                    failure_detail = self._error_text(exc)
                    self.last_details["error_type"] = type(exc).__name__
                    self.last_details["error"] = failure_detail
                parse_ms = _ms(time.perf_counter() - t_parse)

        # A call whose answer parsed (an actions list was present) resets the counter;
        # every whole-call failure counts. Counted immediately before enqueueing (see
        # get_action's internal-error path, which relies on this ordering).
        self._count_call_outcome(failed=parsed is None)
        claimed = parsed is not None and parsed.doc.get("task_complete") is True
        if parsed is None:
            self._enqueue_holds(chunk_id, 0, failure_reason or REASON_PROVIDER)
        elif claimed:
            # D-25: "done" is a judgement about the CURRENT views, so no motion follows
            # it. The whole chunk holds; the model's actions are never executed.
            self._enqueue_holds(chunk_id, 0, REASON_TASK_COMPLETE)
            self.task_complete_claimed = True
            if self.episode_complete_reason is None:
                self.episode_complete_reason = EPISODE_TASK_COMPLETE
        else:
            for k, step in enumerate(parsed.steps):
                self._queue.append(_QueueItem(chunk_id, k, action=step, raw=step))
            if parsed.first_bad_index is not None:
                actions = parsed.doc["actions"]
                first_raw = (_logged_raw_step(actions[parsed.first_bad_index])
                             if parsed.first_bad_index < len(actions) else None)
                self._enqueue_holds(chunk_id, parsed.first_bad_index, parsed.reason, first_raw)
        if parsed is not None:
            plan = parsed.doc.get("plan")
            if isinstance(plan, str) and plan.strip():
                self._previous_plan = plan  # D-09: only the previous plan is carried

        n_valid = len(parsed.steps) if parsed is not None else 0
        if claimed:
            print(f"MLLM chunk {chunk_id}: {model} claimed task_complete after {api_s:.1f} s; "
                  f"holding position for {self.chunk_len} ticks and ending the episode")
        elif failure_reason is None:
            print(f"MLLM chunk {chunk_id}: {model} answered in {api_s:.1f} s, "
                  f"{n_valid}/{self.chunk_len} valid steps")
        else:
            print(f"MLLM chunk {chunk_id}: {model} call failed after {api_s:.1f} s "
                  f"({failure_reason}); holding position for {self.chunk_len} ticks")

        timings = inputs.meta.get("timings_ms", {}) if isinstance(inputs.meta, dict) else {}
        record = self._build_record(
            chunk_id=chunk_id,
            started_utc=started_utc,
            request=request,
            result=result,
            inputs=inputs,
            parsed=parsed,
            failure_reason=failure_reason,
            failure_detail=failure_detail,
            prev_period=prev_period,
            latency_ms={
                "depth_fetch": _plain(timings.get("depth_fetch")),
                "image_encode": _plain(timings.get("image_encode")),
                "api": _ms(api_s),
                "parse": parse_ms,
                "total": _ms(time.perf_counter() - t_start),
            },
        )
        if self.recorder is not None:
            try:
                finalized = self.recorder.finalize(record, chunk_id=chunk_id, request=request,
                                                   result=result, inputs=inputs, error=error)
                if isinstance(finalized, dict):
                    record = finalized
            except Exception as exc:
                record["recorder_error"] = f"{type(exc).__name__}: {self._error_text(exc)}"
        self.last_details["mllm_call"] = self._redact_obj(_plain(record))

    def _build_record(self, *, chunk_id, started_utc, request, result, inputs, parsed,
                      failure_reason, failure_detail, prev_period, latency_ms) -> dict:
        meta = inputs.meta if isinstance(inputs.meta, dict) else {}
        doc = parsed.doc if parsed is not None else {}

        if result is None:
            stop = {"reason": _STOP_ERROR, "refusal_category": None, "incomplete_reason": None}
            usage = None
        else:
            stop = _plain(result.stop)
            usage = _plain(result.usage)

        thinking = result.native_thinking_text if result is not None else None
        trace = {
            "task_restatement": doc.get("task_restatement"),
            "plan": doc.get("plan"),
            "scene_claims": doc.get("scene_claims"),  # verbatim (D-11)
            "task_complete": doc.get("task_complete") is True,  # strict boolean (D-18)
            "task_complete_raw": _logged_raw_value(doc.get("task_complete")),
            "native_thinking_text": thinking[:MAX_INLINE_THINKING_CHARS] if isinstance(thinking, str) else None,
            "native_thinking_kind": result.native_thinking_kind if result is not None else "none",
        }

        if parsed is not None:
            parse = {
                "ok": parsed.first_bad_index is None,
                "n_returned": parsed.n_returned,
                "n_valid": len(parsed.steps),
                "first_bad_index": parsed.first_bad_index,
                "reason": parsed.reason,
                "reason_detail": parsed.reason_detail,
                "missing_top_level": parsed.missing_top_level,
                "extra_dropped": parsed.extra_dropped,
                "fence_stripped": parsed.fence_stripped,
            }
        else:
            parse = {
                "ok": False,
                "n_returned": None,
                "n_valid": 0,
                "first_bad_index": None,
                "reason": failure_reason,
                "reason_detail": failure_detail,
                "missing_top_level": [],
                "extra_dropped": 0,
                "fence_stripped": False,
            }

        capture_keys = ("left_image_source_requested", "left_image_source_sent",
                        "captured_utc", "stereo_pair_metadata")
        return {
            "chunk_id": chunk_id,
            "started_utc": started_utc,
            "backend": self.label,
            "provider": getattr(self.adapter, "provider", None),
            "model_requested": getattr(self.adapter, "model", None),
            "model_returned": result.model_returned if result is not None else None,
            "request_id": result.request_id if result is not None else None,
            "response_id": result.response_id if result is not None else None,
            "prompt_template_version": self.prompt_template_version,
            "prompt_sha256": self._prompt_sha256,
            "prompt_template_sha256": self._template_sha256,
            "schema_sha256": self.schema_hash,
            "include_urdf": self.urdf_text is not None,
            "chunk_len": self.chunk_len,
            "step_dt_s": self.settings["step_dt_s"],
            "sdk": getattr(self.adapter, "sdk", None),
            "settings": getattr(self.adapter, "settings_summary", None),
            "usage": usage,
            "stop": stop,
            "trace": trace,
            "parse": parse,
            "consecutive_failed_calls": self._consecutive_failed_calls,
            "depth": meta.get("depth"),
            "capture": {key: meta[key] for key in capture_keys if key in meta},
            "latency_ms": latency_ms,
            "prev_chunk_tick_period_s": prev_period,
            "images": [
                {"role": role, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
                for role, data in request.images
            ],
        }

    def close(self) -> None:
        """Close the adapter and collector when they expose `close()`."""
        for owner in (self.adapter, self.collector):
            close = getattr(owner, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as exc:
                    print(f"WARNING: closing {type(owner).__name__} failed: {self._error_text(exc)}")


__all__ = [
    "ACTION_PATH",
    "CHUNK_SUFFIX",
    "ChunkParseError",
    "EPISODE_BACKEND_FAILED",
    "EPISODE_TASK_COMPLETE",
    "HOLD_SUFFIX",
    "IMAGE_ORDER",
    "MAX_INLINE_THINKING_CHARS",
    "MAX_LOGGED_TEXT_CHARS",
    "MLLM_BACKENDS",
    "MllmActionSourceBase",
    "MllmBackendSpec",
    "MllmInputs",
    "ParsedChunk",
    "ProviderRequest",
    "ProviderResult",
    "STOP_COMPLETE",
    "STOP_OTHER",
    "STOP_REFUSAL",
    "STOP_TRUNCATED",
    "load_mllm_settings",
    "parse_chunk",
    "register_mllm_backend",
]
