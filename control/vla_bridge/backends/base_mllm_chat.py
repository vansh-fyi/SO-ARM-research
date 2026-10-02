"""Shared OpenAI-compatible chat adapter for the HF Inference Providers MLLM lane (Phase 15, MLLM-01).

What this is
============
One adapter class (`OpenAIChatAdapter`) and one adapter factory
(`make_chat_adapter`) shared by every HF-router variant (`mllm_hf_qwen.py`,
`mllm_hf_deepseek.py`, `mllm_hf_kimi.py`). Each variant module is a thin
registration plus its own `backend_configs/<name>.json`; the model, the
provider pin, the effort and the price table all live in that config, so no
model-specific code reaches the control loop (PIPE-02, D-00a, D-23).

The `base_` prefix keeps backend discovery from importing this module on its
own: it registers nothing.

Router facts: verified, not guessed (2026-10-02, research "Provider Facts")
---------------------------------------------------------------------------
  - router.huggingface.co/v1 is OpenAI-compatible (chat completions). The
    model string may carry a provider suffix `<hf-model-id>:<provider>`; with
    no suffix, routing defaults to the `:fastest` policy
    (huggingface.co/docs/inference-providers).
  - `supports_structured_output` is PER PROVIDER (router.huggingface.co/v1/
    models): e.g. DeepSeek-V4.1-Flash is structured on deepinfra but NOT on
    novita, and `:fastest` can land it on novita. A silently unpinned
    request therefore loses JSON-schema mode (research C2, Pitfall 2), which
    is why every config must pin one provider by name and the routing-policy
    names fastest/cheapest/preferred are rejected.
  - No model on the router is free (research C1). Prices differ per provider
    and are pinned in each config with `verified_on`.
  - Model cards publish thinking-mode sampling temperature 1.0 / top_p 0.95,
    the shared nominal values (D-20), so the HF lane is the only one that
    actually APPLIES them (research C3).

Installed SDK (openai==3.23.0, read from site-packages on 2026-10-02)
---------------------------------------------------------------------
`Completions.create(messages, model, ..., max_tokens, reasoning_effort,
response_format, stream, stream_options, temperature, top_p, ...,
extra_body, timeout)`. With `stream=True` it returns `openai.Stream`
(`openai/_streaming.py`): iterating it yields typed `ChatCompletionChunk`s
built without validation (unknown finish reasons and provider extras
survive), `.response` is the underlying HTTP response (its headers carry
`x-request-id`), and `.close()` releases the connection. An SSE event whose
JSON has an `error` key makes the iterator raise `openai.APIError` mid-stream.
`ChoiceDelta` has `content` and `refusal`; provider extras such as
`reasoning` or `reasoning_content` exist only as pydantic extras
(`model_extra`). `stream_options={"include_usage": True}` adds a final chunk
with empty `choices` and the `usage`;
`CompletionUsage.completion_tokens_details.reasoning_tokens` and
`prompt_tokens_details.cached_tokens` are optional.

Why the adapter streams (Plan 15-10 live smoke, 2026-10-02)
-----------------------------------------------------------
A non-streaming reasoning call sends no bytes until the whole answer is
done. The HF router's edge proxy cut such idle connections at about 120 s
(live HTTP 504 Gateway Time-out at 120.9 s and 121.2 s for mllm-hf-deepseek,
with timeout_s 900), so every long reasoning call failed. Streaming keeps
bytes flowing (reasoning deltas) for the whole call, as the Claude and GPT
adapters already do. The adapter assembles the deltas into one answer:
`delta.content` into the text, `reasoning` / `reasoning_content` (first
non-empty, attribute or `model_extra`) into the raw thinking text,
`delta.refusal` into the refusal, `finish_reason` from the last choice chunk
that carries one, and `usage` from the include_usage chunk. The sidecar's
raw payload is a JSON-safe reconstruction (the assembled message, ids,
model, finish_reason, usage dump and chunk counts), never SDK objects; the
recorder redacts it before writing. `timeout_s` now bounds each wait for the
next streamed byte rather than one silent wait for the whole answer.

`reasoning_effort` travels inside `extra_body`, not as the typed keyword:
the effort names are per model (Qwen `xhigh`, Kimi-K3 and DeepSeek `max`),
and keeping them out of the SDK's typed parameter means a model-specific
level can never be rejected or rewritten by SDK typing on a future SDK
version; the router receives the same JSON key either way.

Structured-output modes (research C2, Pitfall 3)
------------------------------------------------
  - `json_schema` (default): `response_format = {"type": "json_schema",
    "json_schema": {"name", "strict": true, "schema"}}` with the shared
    schema.
  - `json_object`: `response_format = {"type": "json_object"}` and the schema
    JSON text appended to the system message; the base still parses and
    pre-validates every step, and tolerates one surrounding json code fence.
The mode is a config value a human sets after a live rejection. There is no
try/fallback loop: a rejected request is one failed (held) call.
`structured_mode` and `schema_in_prompt` are logged in every call record's
`settings.applied`.

Config keys this module validates (SystemExit naming the field)
---------------------------------------------------------------
provider "huggingface"; model (no ':' suffix; the pin is separate);
hf_provider (lowercase provider name, never a routing policy); base_url
(https, default DEFAULT_BASE_URL); structured_mode (json_schema|json_object,
default json_schema); extra_body (object, default {}, may not override a
key the adapter sets); context_window (optional int; max_tokens +
PROMPT_TOKEN_RESERVE must fit inside it). The shared keys (api_key_env,
effort, max_tokens, prices, capabilities, ...) are validated earlier by
`base_mllm_build.load_mllm_backend_config`.

Secret rules (T-15-27)
----------------------
The HF token reaches this module only as the `api_key` argument of
`make_chat_adapter()`, which hands it straight to the SDK client built with
`max_retries=0` (no hidden paid retry). The adapter keeps no copy, never puts
it in `repr()`, `settings_summary` or a result, and never catches exceptions:
the base turns a failure into chunk_len holds and redacts any error text,
and the recorder redacts the reconstructed raw payload before writing the sidecar.
"""

import base64
import importlib.metadata
import json
import re

from vla_bridge.backends.base_mllm import (
    STOP_COMPLETE,
    STOP_OTHER,
    STOP_REFUSAL,
    STOP_TRUNCATED,
    ProviderResult,
)

PROVIDER = "huggingface"
# The prompt is about 15K tokens; it must fit next to the reasoning budget
# (max_tokens) inside the pinned provider's context window (research Pitfall 10).
PROMPT_TOKEN_RESERVE = 16000
DEFAULT_BASE_URL = "https://router.huggingface.co/v1"
ROUTING_POLICY_NAMES = ("fastest", "cheapest", "preferred")
STRUCTURED_MODES = ("json_schema", "json_object")
DEFAULT_STRUCTURED_MODE = "json_schema"

# Keys the adapter itself sends; extra_body may not override them silently.
_RESERVED_EXTRA_BODY_KEYS = ("model", "messages", "response_format", "max_tokens", "temperature",
                             "top_p", "reasoning_effort", "stream", "stream_options", "tools",
                             "tool_choice")
_PROVIDER_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_HTTPS_URL_RE = re.compile(r"^https://[^/\s]+(/\S*)?$")

_FINISH_MAP = {
    "stop": STOP_COMPLETE,
    "length": STOP_TRUNCATED,
    "content_filter": STOP_REFUSAL,
}
_REASONING_FIELDS = ("reasoning", "reasoning_content")
STREAM_OPTIONS = {"include_usage": True}
REQUEST_ID_HEADER = "x-request-id"

JSON_OBJECT_SCHEMA_BLOCK = (
    "\n\nRESPONSE FORMAT\n"
    "Reply with exactly one JSON object and nothing else. The object must conform to this "
    "JSON Schema (named {name}):\n{schema}"
)
_APPLIED_NOTE = (
    "sent to the HF router; whether the pinned provider honours each value (sampling, "
    "reasoning_effort, json_schema) is provider-dependent and is confirmed by the live smoke "
    "(research A3/A4)"
)
_SAMPLING_UNSUPPORTED_REASON = "capabilities.sampling_supported is false for this config, so {param} is not sent"
_SAMPLING_NULL_REASON = "the shared nominal sampling sets {param} to null, so it is not sent"
_EFFORT_NULL_REASON = "effort is null in the config, so reasoning_effort is not sent (provider default applies)"


def _make_chat_client(api_key: str, base_url: str, timeout_s: float):
    """The real OpenAI-compatible SDK client, imported lazily; `max_retries=0` so no hidden paid retry."""
    try:
        import openai
    except ImportError as exc:  # pragma: no cover - the pinned SDK is installed in control/.venv
        raise SystemExit(
            f"The HF MLLM backends need the pinned openai SDK (control/requirements.txt): {exc}"
        ) from None
    return openai.OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_s, max_retries=0)


def _sdk_version() -> str | None:
    try:
        return importlib.metadata.version("openai")
    except importlib.metadata.PackageNotFoundError:
        return None


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validate_chat_config(config: dict, backend_name: str) -> None:
    """Check the HF-specific keys; raise `SystemExit` naming the bad field (never a secret)."""

    def fail(field_name: str, requirement: str):
        raise SystemExit(f"--backend {backend_name} config: '{field_name}' must be {requirement}.")

    if config.get("provider") != PROVIDER:
        fail("provider", f"'{PROVIDER}' for the HF router chat adapter")
    model = config.get("model")
    if not isinstance(model, str) or not model.strip() or ":" in model:
        fail("model", "a non-blank HF model id without a ':' suffix (the provider pin goes in "
                      "'hf_provider')")

    hf_provider = config.get("hf_provider")
    if not isinstance(hf_provider, str) or not hf_provider.strip():
        fail("hf_provider", "a non-blank provider name (e.g. deepinfra, baseten)")
    if hf_provider in ROUTING_POLICY_NAMES:
        fail("hf_provider", f"a pinned provider name, not the routing policy {hf_provider!r} "
                            "(policy routing can land on a provider without structured output; "
                            "research C2)")
    if not _PROVIDER_NAME_RE.match(hf_provider):
        fail("hf_provider", "a lowercase provider name of letters, digits, '-' or '_' "
                            "(no '/', ':' or uppercase)")

    base_url = config.get("base_url", DEFAULT_BASE_URL)
    if not isinstance(base_url, str) or not _HTTPS_URL_RE.match(base_url):
        fail("base_url", f"an https:// URL (default {DEFAULT_BASE_URL})")

    structured_mode = config.get("structured_mode", DEFAULT_STRUCTURED_MODE)
    if structured_mode not in STRUCTURED_MODES:
        fail("structured_mode", f"one of {list(STRUCTURED_MODES)}")

    extra_body = config.get("extra_body", {})
    if not isinstance(extra_body, dict):
        fail("extra_body", "an object of extra JSON body keys (default {})")
    clashes = sorted(key for key in extra_body if key in _RESERVED_EXTRA_BODY_KEYS)
    if clashes:
        fail("extra_body", f"free of keys the adapter sets itself ({', '.join(clashes)}); use "
                           "'effort', 'max_tokens', 'structured_mode' or the shared sampling instead")

    if "context_window" in config and config["context_window"] is not None:
        context_window = config["context_window"]
        if not _is_int(context_window) or context_window < 1:
            fail("context_window", "a positive int (the pinned provider's context length in tokens)")
        max_tokens = config.get("max_tokens")
        if _is_int(max_tokens) and max_tokens + PROMPT_TOKEN_RESERVE > context_window:
            fail("context_window", f">= max_tokens ({max_tokens}) + the {PROMPT_TOKEN_RESERVE}-token "
                                   f"prompt reserve; lower max_tokens or pin a provider with a larger "
                                   f"context")


def _get(obj, key):
    """Attribute or mapping access (typed SDK objects, dict extras, SimpleNamespace fakes)."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _int_or_none(value):
    return value if _is_int(value) else None


def _plain_dump(obj):
    if obj is None:
        return None
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
    if isinstance(obj, dict):
        return dict(obj)
    return {key: value for key, value in vars(obj).items() if not key.startswith("_")}


def _normalise_usage(usage) -> dict:
    if usage is None:
        return {"input_tokens": None, "output_tokens": None, "reasoning_tokens": None,
                "cached_input_tokens": None, "raw": None}
    return {
        "input_tokens": _int_or_none(_get(usage, "prompt_tokens")),
        "output_tokens": _int_or_none(_get(usage, "completion_tokens")),
        "reasoning_tokens": _int_or_none(_get(_get(usage, "completion_tokens_details"), "reasoning_tokens")),
        "cached_input_tokens": _int_or_none(_get(_get(usage, "prompt_tokens_details"), "cached_tokens")),
        "raw": _plain_dump(usage),
    }


def _delta_reasoning(delta) -> str | None:
    """`delta.reasoning`, else `reasoning_content`, else either one from `model_extra` (first non-empty)."""
    extra = getattr(delta, "model_extra", None) or {}
    for source in (delta, extra):
        for name in _REASONING_FIELDS:
            value = _get(source, name)
            if isinstance(value, str) and value:
                return value
    return None


def _stop(finish_reason, refusal) -> dict:
    if finish_reason == "content_filter":
        reason, category = STOP_REFUSAL, "content_filter"
    elif isinstance(refusal, str) and refusal.strip():
        reason, category = STOP_REFUSAL, "message_refusal"
    else:
        reason, category = _FINISH_MAP.get(finish_reason, STOP_OTHER), None
    return {
        "reason": reason,
        "refusal_category": category,
        "incomplete_reason": None if reason == STOP_COMPLETE else finish_reason,
        "provider_stop_reason": finish_reason,
    }


class StreamAssembly:
    """Folds streamed `ChatCompletionChunk`s into one answer (choice index 0 only)."""

    def __init__(self):
        self._content: list[str] = []
        self._reasoning: list[str] = []
        self._refusal: list[str] = []
        self.finish_reason = None
        self.usage = None
        self.model = None
        self.response_id = None
        self.created = None
        self.chunk_count = 0
        self.choice_chunk_count = 0

    def add(self, chunk) -> None:
        self.chunk_count += 1
        for key in ("model", "id"):
            value = _get(chunk, key)
            if isinstance(value, str) and value:
                setattr(self, "response_id" if key == "id" else "model", value)
        created = _get(chunk, "created")
        if _is_int(created):
            self.created = created
        usage = _get(chunk, "usage")
        if usage is not None:
            self.usage = usage
        for choice in _get(chunk, "choices") or []:
            if _get(choice, "index") not in (None, 0):
                continue
            self.choice_chunk_count += 1
            delta = _get(choice, "delta")
            if delta is not None:
                content = _get(delta, "content")
                if isinstance(content, str):
                    self._content.append(content)
                refusal = _get(delta, "refusal")
                if isinstance(refusal, str):
                    self._refusal.append(refusal)
                reasoning = _delta_reasoning(delta)
                if reasoning is not None:
                    self._reasoning.append(reasoning)
            finish_reason = _get(choice, "finish_reason")
            if finish_reason is not None:
                self.finish_reason = finish_reason

    @property
    def text(self) -> str | None:
        return "".join(self._content) if self._content else None

    @property
    def refusal(self) -> str | None:
        return "".join(self._refusal) if self._refusal else None

    @property
    def thinking(self) -> str | None:
        joined = "".join(self._reasoning)
        return joined if joined.strip() else None

    def raw_record(self) -> dict:
        """A JSON-safe reconstruction of the streamed answer for the (redacted) sidecar."""
        choices = []
        if self.choice_chunk_count:
            choices.append({
                "index": 0,
                "finish_reason": self.finish_reason if isinstance(self.finish_reason, str) else None,
                "message": {"role": "assistant", "content": self.text, "refusal": self.refusal,
                            "reasoning": self.thinking},
            })
        return {
            "object": "chat.completion",
            "reconstructed_from_stream": True,
            "id": self.response_id,
            "model": self.model,
            "created": self.created,
            "choices": choices,
            "usage": _normalise_usage(self.usage)["raw"],
            "stream": {"chunk_count": self.chunk_count, "choice_chunk_count": self.choice_chunk_count},
        }

    def to_result(self, request_id: str | None) -> ProviderResult:
        thinking = self.thinking
        return ProviderResult(
            text=self.text,
            raw_payload=self.raw_record(),
            usage=_normalise_usage(self.usage),
            stop=_stop(self.finish_reason, self.refusal),
            native_thinking_text=thinking,
            native_thinking_kind="raw" if thinking is not None else "none",
            request_id=request_id if isinstance(request_id, str) and request_id else None,
            response_id=self.response_id,
            model_returned=self.model,
        )


def _header_request_id(stream) -> str | None:
    """The `x-request-id` header of the stream's HTTP response (`.response`, else `._response`)."""
    for attr in ("response", "_response"):
        try:
            headers = getattr(getattr(stream, attr, None), "headers", None)
            value = headers.get(REQUEST_ID_HEADER) if headers is not None else None
        except Exception:
            value = None
        if isinstance(value, str) and value:
            return value
    return None


class OpenAIChatAdapter:
    """MLLM adapter for the HF router's OpenAI-compatible chat completions, pinned to one provider."""

    provider = PROVIDER

    def __init__(self, client, config: dict):
        self._client = client
        self.base_model = config["model"]
        self.hf_provider = config["hf_provider"]
        self.model = f"{self.base_model}:{self.hf_provider}"  # the pinned request model string (D-23)
        self.base_url = config.get("base_url", DEFAULT_BASE_URL)
        self.structured_mode = config.get("structured_mode", DEFAULT_STRUCTURED_MODE)
        self.extra_body = dict(config.get("extra_body") or {})
        self.effort = config.get("effort")
        self.max_tokens = config["max_tokens"]
        self.sdk = {"name": "openai", "version": _sdk_version()}

        sampling = dict(config.get("sampling") or {})
        sampling_supported = bool((config.get("capabilities") or {}).get("sampling_supported"))
        self._sampling: dict = {}
        omitted: dict = {}
        for param in ("temperature", "top_p"):
            value = sampling.get(param)
            if not sampling_supported:
                omitted[param] = _SAMPLING_UNSUPPORTED_REASON.format(param=param)
            elif value is None:
                omitted[param] = _SAMPLING_NULL_REASON.format(param=param)
            else:
                self._sampling[param] = value
        if self.effort is None:
            omitted["reasoning_effort"] = _EFFORT_NULL_REASON

        applied = {"model": self.model, "max_tokens": self.max_tokens, **self._sampling}
        if self.effort is not None:
            applied["reasoning_effort"] = self.effort
        if self.extra_body:
            applied["extra_body"] = dict(self.extra_body)
        applied.update({
            "response_format": self.structured_mode,
            "structured_mode": self.structured_mode,
            "schema_in_prompt": self.structured_mode == "json_object",
            "note": _APPLIED_NOTE,
        })
        self.settings_summary = {
            "requested": {
                **sampling,
                "effort": self.effort,
                "max_tokens": self.max_tokens,
                "structured_mode": self.structured_mode,
                "hf_provider": self.hf_provider,
                "model": self.model,
            },
            "applied": applied,
            "omitted": omitted,
        }

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(model={self.model!r}, effort={self.effort!r}, "
                f"max_tokens={self.max_tokens}, structured_mode={self.structured_mode!r})")

    def _system_content(self, request) -> str:
        if self.structured_mode != "json_object":
            return request.system_prompt
        return request.system_prompt + JSON_OBJECT_SCHEMA_BLOCK.format(
            name=request.schema_name, schema=json.dumps(request.schema))

    def _user_content(self, request) -> list[dict]:
        content = []
        for role, data in request.images:
            content.append({"type": "text", "text": f"Image: {role}"})
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{request.image_media_type};base64,"
                                     + base64.b64encode(data).decode("ascii")},
            })
        content.append({"type": "text", "text": request.user_text})
        return content

    def _response_format(self, request) -> dict:
        if self.structured_mode == "json_object":
            return {"type": "json_object"}
        return {"type": "json_schema",
                "json_schema": {"name": request.schema_name, "strict": True, "schema": request.schema}}

    def request_kwargs(self, request) -> dict:
        """Exactly what `chat.completions.create()` receives (streamed; no tools, no prefill)."""
        extra_body = dict(self.extra_body)
        if self.effort is not None:
            extra_body["reasoning_effort"] = self.effort
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self._system_content(request)},
                {"role": "user", "content": self._user_content(request)},
            ],
            "response_format": self._response_format(request),
            "max_tokens": self.max_tokens,
            **self._sampling,
            "stream": True,
            "stream_options": dict(STREAM_OPTIONS),
            "extra_body": extra_body,
        }

    def call(self, request) -> ProviderResult:
        # Exceptions propagate on purpose (including a mid-stream APIError): the base holds,
        # counts and redacts them. The stream is always closed so the connection is released.
        stream = self._client.chat.completions.create(**self.request_kwargs(request))
        assembly = StreamAssembly()
        try:
            request_id = _header_request_id(stream)
            for chunk in stream:
                assembly.add(chunk)
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                close()
        return assembly.to_result(request_id)

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            close()


def make_chat_adapter(api_key: str, config: dict) -> OpenAIChatAdapter:
    """Adapter factory for `build_mllm_backend` (validates, then builds; runs before the robot)."""
    validate_chat_config(config, f"HF model {config.get('model')!r}")
    client = _make_chat_client(api_key, config.get("base_url", DEFAULT_BASE_URL), config["timeout_s"])
    return OpenAIChatAdapter(client, config)


__all__ = [
    "DEFAULT_BASE_URL",
    "JSON_OBJECT_SCHEMA_BLOCK",
    "OpenAIChatAdapter",
    "PROMPT_TOKEN_RESERVE",
    "ROUTING_POLICY_NAMES",
    "STREAM_OPTIONS",
    "STRUCTURED_MODES",
    "StreamAssembly",
    "make_chat_adapter",
    "validate_chat_config",
]
