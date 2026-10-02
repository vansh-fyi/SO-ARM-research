"""`--backend mllm-claude`: Claude as a raw-autonomy joint-target controller (Phase 15, MLLM-02).

What this is
============
One provider adapter (`AnthropicAdapter`) plus one config file
(`backend_configs/mllm_claude.json`), registered with
`base_mllm.register_mllm_backend`. Everything else (the chunk-drain loop,
prompt, schema, parsing, holds, failure cap, recorder, inputs and hardware)
is shared: `build()` is a single call to `base_mllm_build.build_mllm_backend`
(PIPE-02, D-00a). The model returns a JSON chunk of absolute joint targets
that still passes the unmodified safety validator on every tick (MLLM-04).

Model facts: verified, not guessed (2026-10-02)
-----------------------------------------------
  - platform.claude.com/docs/en/models/overview and models/fable-5-1:
    `claude-fable-5-1`, released 2026-09-01, 1M context, 128K max output,
    vision input, $10 input / $50 output / $0.25 cache read per million
    tokens. It is a Covered Model with 30-day retention, not ZDR.
    `claude-mythos-5-1` is invitation-only; `claude-opus-5-5` is Anthropic's
    default recommendation, but D-14 asks for the top available flagship.
  - build-with-claude/structured-outputs: native structured output through
    `output_config.format = {"type": "json_schema", "schema": ...}`. It works
    with streaming and thinking, and is incompatible with prefill. No tool use
    and no assistant prefill are sent (forced tool_choice and prefill -> 400).
  - build-with-claude/thinking: adaptive thinking is always on for this model;
    `display: "summarized"` returns readable summaries (the full raw thinking
    is billed and counts toward `max_tokens`).
  - build-with-claude/effort: `output_config.effort` in
    low/medium/high/xhigh/max.
  - Installed SDK source (anthropic==1.11.0): `Messages.stream(max_tokens,
    messages, model, ..., output_config, system, thinking, ...)` has no
    temperature/top_p/top_k parameter; `Usage.output_tokens_details
    .thinking_tokens`, `cache_read_input_tokens`,
    `cache_creation_input_tokens`; `Message.stop_reason`, `Message
    .stop_details.category` (`RefusalStopDetails`). The streamed final
    message does not carry `_request_id`, so the stream's `request-id`
    header value is the fallback.

Config keys (backend_configs/mllm_claude.json, over mllm_defaults.json)
-----------------------------------------------------------------------
provider, model, api_key_env (a NAME), effort, max_tokens, thinking
{type, display}, prices {input_per_mtok, output_per_mtok,
cached_input_per_mtok, verified_on, source}, capabilities
{sampling_supported, effort_levels, summary_supported}, data_retention_note,
requirement, model_verified, notes. timeout_s (900 s) and the shared sampling
come from the defaults.

Transport
---------
Direct HTTPS to api.anthropic.com through the official SDK, streaming
(`messages.stream(...)` then `get_final_message()`) because the SDK refuses
long non-streaming requests (research Pitfall 9). The client is built lazily
inside `_make_anthropic_client` with `max_retries=0`: a failed call is never
silently retried and paid for twice; the base turns it into chunk_len holds
and counts it toward the consecutive-failure cap (T-15-19).

Secret rules (T-15-18)
----------------------
The key reaches this module only as the `api_key` argument of
`make_adapter()`, which hands it straight to the SDK client. The adapter keeps
no copy of it, never puts it in `repr()`, `settings_summary` or a result, and
never catches exceptions: the base redacts any error text that might echo it.

Conflict notes
--------------
  - C3 (D-20 shared sampling): the provider fixes sampling, so temperature,
    top_p and top_k are never sent; `settings_summary` logs the shared nominal
    values as requested and each one as omitted with a provider-fixed reason.
  - C4 (staged fields vs reasoning extraction): the prompt and schema stay
    lint-clean (Plan 15-01); a `refusal` stop (category preserved, e.g.
    `reasoning_extraction`) or a `max_tokens` stop is reported through
    `stop` only, and the base turns it into a held, failed call.
"""

import base64
import importlib.metadata

from vla_bridge.backends import base_mllm_build
from vla_bridge.backends.base_mllm import (
    STOP_COMPLETE,
    STOP_OTHER,
    STOP_REFUSAL,
    STOP_TRUNCATED,
    ProviderResult,
    register_mllm_backend,
)

BACKEND_NAME = "mllm-claude"
CONFIG_NAME = "mllm_claude"
_CONFIG_WHERE = f"control/vla_bridge/backend_configs/{CONFIG_NAME}.json"

_STOP_MAP = {
    "end_turn": STOP_COMPLETE,
    "refusal": STOP_REFUSAL,
    "max_tokens": STOP_TRUNCATED,
}
_THINKING_TYPES = ("adaptive",)
_THINKING_DISPLAYS = ("summarized", "omitted")
_PROVIDER_FIXED_REASON = (
    "provider-fixed: {model} rejects any non-default {param} with HTTP 400 and the anthropic "
    "SDK has no such parameter, so it is never sent (research C3)"
)


def _make_anthropic_client(api_key: str, timeout_s: float):
    """The real SDK client, imported lazily; `max_retries=0` so no hidden paid retry."""
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - the pinned SDK is installed in control/.venv
        raise SystemExit(
            f"--backend {BACKEND_NAME} needs the pinned anthropic SDK (control/requirements.txt): {exc}"
        ) from None
    return anthropic.Anthropic(api_key=api_key, timeout=timeout_s, max_retries=0)


def _sdk_version() -> str | None:
    try:
        return importlib.metadata.version("anthropic")
    except importlib.metadata.PackageNotFoundError:
        return None


def _int_or_none(value):
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _plain_dump(obj):
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
    if obj is None:
        return None
    return {key: value for key, value in vars(obj).items() if not key.startswith("_")}


def _normalise_usage(usage) -> dict:
    """input_tokens counts everything sent (uncached + cache read + cache write)."""
    if usage is None:
        return {"input_tokens": None, "output_tokens": None, "reasoning_tokens": None,
                "cached_input_tokens": None, "raw": None}
    uncached = _int_or_none(getattr(usage, "input_tokens", None))
    cache_read = _int_or_none(getattr(usage, "cache_read_input_tokens", None))
    cache_write = _int_or_none(getattr(usage, "cache_creation_input_tokens", None))
    details = getattr(usage, "output_tokens_details", None)
    thinking_tokens = _int_or_none(getattr(details, "thinking_tokens", None)) if details is not None else None
    return {
        "input_tokens": None if uncached is None else uncached + (cache_read or 0) + (cache_write or 0),
        "output_tokens": _int_or_none(getattr(usage, "output_tokens", None)),
        "reasoning_tokens": thinking_tokens,
        "cached_input_tokens": cache_read,
        "raw": _plain_dump(usage),
    }


def _thinking(blocks) -> tuple[str | None, str]:
    """(text, kind): 'summary' with text, 'omitted' when blocks carry no text, 'none' without blocks."""
    thinking_blocks = [b for b in blocks if getattr(b, "type", None) in ("thinking", "redacted_thinking")]
    if not thinking_blocks:
        return None, "none"
    parts = [b.thinking for b in thinking_blocks
             if getattr(b, "type", None) == "thinking" and isinstance(getattr(b, "thinking", None), str)
             and b.thinking.strip()]
    if not parts:
        return None, "omitted"
    return "\n\n".join(parts), "summary"


def message_to_result(message, stream_request_id: str | None = None) -> ProviderResult:
    """Map a final `anthropic.types.Message` to the shared `ProviderResult`."""
    blocks = list(getattr(message, "content", None) or [])
    texts = [b.text for b in blocks
             if getattr(b, "type", None) == "text" and isinstance(getattr(b, "text", None), str)]
    thinking_text, thinking_kind = _thinking(blocks)

    provider_stop = getattr(message, "stop_reason", None)
    reason = _STOP_MAP.get(provider_stop, STOP_OTHER)
    stop_details = getattr(message, "stop_details", None)
    category = getattr(stop_details, "category", None) if stop_details is not None else None
    stop = {
        "reason": reason,
        "refusal_category": category,
        "incomplete_reason": None if reason == STOP_COMPLETE else provider_stop,
        "provider_stop_reason": provider_stop,
    }
    request_id = getattr(message, "_request_id", None) or stream_request_id
    return ProviderResult(
        text="".join(texts) if texts else None,
        raw_payload=_plain_dump(message),
        usage=_normalise_usage(getattr(message, "usage", None)),
        stop=stop,
        native_thinking_text=thinking_text,
        native_thinking_kind=thinking_kind,
        request_id=request_id,
        response_id=getattr(message, "id", None),
        model_returned=getattr(message, "model", None),
    )


def _validated_thinking(thinking) -> dict:
    if (not isinstance(thinking, dict) or thinking.get("type") not in _THINKING_TYPES
            or thinking.get("display", "summarized") not in _THINKING_DISPLAYS):
        raise SystemExit(
            f"{_CONFIG_WHERE}: 'thinking' must be an object with type {list(_THINKING_TYPES)} and "
            f"display {list(_THINKING_DISPLAYS)} (adaptive thinking is always on for this model; "
            "'enabled'/'disabled' return HTTP 400)."
        )
    return dict(thinking)


class AnthropicAdapter:
    """MLLM adapter for the Anthropic Messages API (structured output + adaptive thinking)."""

    provider = "anthropic"

    def __init__(self, client, config: dict):
        self._client = client
        self.model = config["model"]
        self.effort = config.get("effort")
        self.max_tokens = config["max_tokens"]
        self.thinking = _validated_thinking(config.get("thinking"))
        self.sdk = {"name": "anthropic", "version": _sdk_version()}

        applied = {"max_tokens": self.max_tokens, "thinking": dict(self.thinking)}
        if self.effort is not None:
            applied = {"effort": self.effort, **applied}
        requested = {**dict(config.get("sampling") or {}), "effort": self.effort,
                     "max_tokens": self.max_tokens, "thinking": dict(self.thinking)}
        self.settings_summary = {
            "requested": requested,
            "applied": applied,
            "omitted": {
                param: _PROVIDER_FIXED_REASON.format(model=self.model, param=param)
                for param in ("temperature", "top_p", "top_k")
            },
        }

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(model={self.model!r}, effort={self.effort!r}, "
                f"max_tokens={self.max_tokens})")

    def _content(self, request) -> list[dict]:
        content = []
        for role, data in request.images:
            content.append({"type": "text", "text": f"Image: {role}"})
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": request.image_media_type,
                    "data": base64.b64encode(data).decode("ascii"),
                },
            })
        content.append({"type": "text", "text": request.user_text})
        return content

    def request_kwargs(self, request) -> dict:
        """Exactly what `messages.stream()` receives (no sampling, tools or prefill)."""
        output_config = {"format": {"type": "json_schema", "schema": request.schema}}
        if self.effort is not None:
            output_config = {"effort": self.effort, **output_config}
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": request.system_prompt,
            "messages": [{"role": "user", "content": self._content(request)}],
            "thinking": dict(self.thinking),
            "output_config": output_config,
        }

    def call(self, request) -> ProviderResult:
        # Exceptions propagate on purpose: the base holds, counts and redacts them.
        with self._client.messages.stream(**self.request_kwargs(request)) as stream:
            message = stream.get_final_message()
            try:
                stream_request_id = getattr(stream, "request_id", None)
            except Exception:
                stream_request_id = None
        if not isinstance(stream_request_id, str):
            stream_request_id = None
        return message_to_result(message, stream_request_id)

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            close()


def make_adapter(api_key: str, config: dict) -> AnthropicAdapter:
    """Adapter factory for `build_mllm_backend` (called before the robot is touched)."""
    return AnthropicAdapter(_make_anthropic_client(api_key, config["timeout_s"]), config)


@register_mllm_backend(BACKEND_NAME, config_name=CONFIG_NAME, adapter_factory=make_adapter)
def build(args, camera_names, joint_limits_deg):
    return base_mllm_build.build_mllm_backend(BACKEND_NAME, args, camera_names, joint_limits_deg)


__all__ = [
    "AnthropicAdapter",
    "BACKEND_NAME",
    "CONFIG_NAME",
    "build",
    "make_adapter",
    "message_to_result",
]
