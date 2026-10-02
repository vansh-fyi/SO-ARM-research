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
    models): e.g. DeepSeek-V4-Flash-Vision-Exp is structured on deepinfra but
    NOT on novita, and `:fastest` lands it on novita. A silently unpinned
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
response_format, temperature, top_p, ..., extra_body, timeout)`;
`client.chat.completions.with_raw_response.create(...)` returns a
`LegacyAPIResponse` with `.http_response.text` (the exact wire body),
`.headers` and `.parse()` (a typed `ChatCompletion` built without
validation, so unknown finish reasons survive). `ChatCompletionMessage` has
`content` and `refusal`; provider extras such as `reasoning` or
`reasoning_content` exist only as pydantic extras (`model_extra`).
`CompletionUsage.completion_tokens_details.reasoning_tokens` and
`prompt_tokens_details.cached_tokens` are optional.

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
and the recorder redacts the wire body before writing the sidecar.
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
                             "top_p", "reasoning_effort", "stream", "tools", "tool_choice")
_PROVIDER_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_HTTPS_URL_RE = re.compile(r"^https://[^/\s]+(/\S*)?$")

_FINISH_MAP = {
    "stop": STOP_COMPLETE,
    "length": STOP_TRUNCATED,
    "content_filter": STOP_REFUSAL,
}
_REASONING_FIELDS = ("reasoning", "reasoning_content")

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
        fail("hf_provider", "a non-blank provider name (e.g. cerebras, deepinfra, baseten)")
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


def _reasoning_text(message) -> str | None:
    """`message.reasoning`, else `reasoning_content`, else either one from `model_extra`."""
    extra = getattr(message, "model_extra", None) or {}
    for source in (message, extra):
        for name in _REASONING_FIELDS:
            value = _get(source, name)
            if isinstance(value, str) and value.strip():
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


def completion_to_result(completion, wire_text: str, request_id: str | None) -> ProviderResult:
    """Map a typed chat completion plus its verbatim wire body to the shared `ProviderResult`."""
    choices = list(_get(completion, "choices") or [])
    choice = choices[0] if choices else None
    message = _get(choice, "message")
    content = _get(message, "content")
    thinking = _reasoning_text(message) if message is not None else None
    return ProviderResult(
        text=content if isinstance(content, str) else None,
        raw_payload=wire_text,
        usage=_normalise_usage(_get(completion, "usage")),
        stop=_stop(_get(choice, "finish_reason"), _get(message, "refusal")),
        native_thinking_text=thinking,
        native_thinking_kind="raw" if thinking is not None else "none",
        request_id=request_id if isinstance(request_id, str) and request_id else None,
        response_id=_get(completion, "id"),
        model_returned=_get(completion, "model"),
    )


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
        """Exactly what `chat.completions.with_raw_response.create()` receives (no tools, no prefill)."""
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
            "extra_body": extra_body,
        }

    def call(self, request) -> ProviderResult:
        # Exceptions propagate on purpose: the base holds, counts and redacts them.
        raw = self._client.chat.completions.with_raw_response.create(**self.request_kwargs(request))
        wire_text = raw.http_response.text
        completion = raw.parse()
        headers = getattr(raw, "headers", None) or getattr(raw.http_response, "headers", None) or {}
        request_id = headers.get("x-request-id") if hasattr(headers, "get") else None
        return completion_to_result(completion, wire_text, request_id)

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
    "STRUCTURED_MODES",
    "completion_to_result",
    "make_chat_adapter",
    "validate_chat_config",
]
