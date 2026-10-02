"""`--backend mllm-gpt`: GPT as a raw-autonomy joint-target controller (Phase 15, MLLM-03).

What this is
============
One provider adapter (`OpenAIResponsesAdapter`) plus one config file
(`backend_configs/mllm_gpt.json`), registered with
`base_mllm.register_mllm_backend`. Everything else (the chunk-drain loop,
prompt, schema, parsing, holds, failure cap, recorder, inputs and hardware)
is shared: `build()` is a single call to `base_mllm_build.build_mllm_backend`
(PIPE-02, D-00a). The model returns a JSON chunk of absolute joint targets
that still passes the unmodified safety validator on every tick (MLLM-04).

Model facts: verified, not guessed (2026-10-02)
-----------------------------------------------
  - developers.openai.com/api/docs/models/gpt-6-astra: `gpt-6-astra`, 1.05M
    context, 128K max output, image input, structured outputs, streaming;
    $10 input / $1 cached input / $50 output per million tokens. The
    Responses API is the recommended endpoint.
  - guides/reasoning and guides/latest-model: `reasoning.effort` in
    low/medium/high/xhigh/max (none/minimal are rejected with HTTP 400);
    `reasoning.summary` (auto/concise/detailed) is opt-in. Reasoning tokens
    are billed as output and count toward `max_output_tokens`; a response can
    end `status: "incomplete"` (`incomplete_details.reason:
    "max_output_tokens"`) before any visible output.
  - guides/latest-model "Unsupported parameters": remove `temperature`,
    `top_p` and `top_logprobs` whenever reasoning effort is not `none`.
  - guides/structured-outputs: `text.format = {"type": "json_schema",
    "name", "strict": true, "schema"}`; a refusal arrives as a `refusal`
    content part instead of `output_text`.
  - Installed SDK source (openai==3.23.0): `Responses.stream(input, model,
    instructions, max_output_tokens, reasoning, store, text, ...)` returns a
    `ResponseStreamManager`; `ResponseUsage.input_tokens` (cached included),
    `.input_tokens_details.cached_tokens`, `.output_tokens`,
    `.output_tokens_details.reasoning_tokens`; `Response.status` in
    completed/failed/in_progress/cancelled/queued/incomplete.
    `ResponseStream.get_final_response()` only finalises on a
    `response.completed` event and raises RuntimeError for a
    `response.incomplete` or `response.failed` stream, so this adapter reads
    the terminal stream event itself. Streamed responses carry no
    `_request_id`; the stream's `x-request-id` header is the fallback.

Config keys (backend_configs/mllm_gpt.json, over mllm_defaults.json)
--------------------------------------------------------------------
provider, model, api_key_env (a NAME), effort, max_tokens (sent as
max_output_tokens), prices {input_per_mtok, output_per_mtok,
cached_input_per_mtok, verified_on, source}, capabilities
{sampling_supported, effort_levels, summary_supported}, data_retention_note,
requirement, model_verified, notes. timeout_s (900 s) and the shared sampling
come from the defaults.

Transport
---------
Direct HTTPS to api.openai.com through the official SDK, streaming
(`responses.stream(...)`), with `store=False` on every request. The client is
built lazily inside `_make_openai_client` with `max_retries=0`: a failed call
is never silently retried and paid for twice; the base turns it into
chunk_len holds and counts it toward the consecutive-failure cap (T-15-24).

Secret rules (T-15-23)
----------------------
The key reaches this module only as the `api_key` argument of
`make_adapter()`, which hands it straight to the SDK client. The adapter keeps
no copy of it, never puts it in `repr()`, `settings_summary` or a result, and
never catches exceptions: the base redacts any error text that might echo it.

Conflict notes
--------------
  - C3 (D-20 shared sampling): temperature and top_p are never sent because
    effort is never `none` on this model; `settings_summary` logs the shared
    nominal values as requested and each one as omitted with that reason.
  - C11 (D-21 highest sensible effort): `xhigh` is the default; `max` is a
    Phase 16 ablation (a one-line config change).
  - A5: whether `reasoning.summary` is accepted for this organisation is
    unverified until the first live call (Plan 15-10). The static
    `capabilities.summary_supported` flag decides; when false the parameter
    is omitted up front and logged, never negotiated by a retry loop.
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

BACKEND_NAME = "mllm-gpt"
CONFIG_NAME = "mllm_gpt"

IMAGE_DETAIL = "high"
REASONING_SUMMARY = "auto"
_TERMINAL_EVENTS = ("response.completed", "response.incomplete", "response.failed")
_REFUSAL_INCOMPLETE_REASONS = ("content_filter",)
_SAMPLING_REMOVED_REASON = (
    "removed: OpenAI requires {param} to be removed whenever reasoning effort is not none, and "
    "{model} does not support none (research C3), so it is never sent"
)
_SUMMARY_OMITTED_REASON = (
    "omitted: capabilities.summary_supported is false in backend_configs/mllm_gpt.json "
    "(assumption A5); decided from the static table, never negotiated by retrying"
)


def _make_openai_client(api_key: str, timeout_s: float):
    """The real SDK client, imported lazily; `max_retries=0` so no hidden paid retry."""
    try:
        import openai
    except ImportError as exc:  # pragma: no cover - the pinned SDK is installed in control/.venv
        raise SystemExit(
            f"--backend {BACKEND_NAME} needs the pinned openai SDK (control/requirements.txt): {exc}"
        ) from None
    return openai.OpenAI(api_key=api_key, timeout=timeout_s, max_retries=0)


def _sdk_version() -> str | None:
    try:
        return importlib.metadata.version("openai")
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
    """OpenAI's input_tokens already includes cached tokens; output_tokens includes reasoning."""
    if usage is None:
        return {"input_tokens": None, "output_tokens": None, "reasoning_tokens": None,
                "cached_input_tokens": None, "raw": None}
    in_details = getattr(usage, "input_tokens_details", None)
    out_details = getattr(usage, "output_tokens_details", None)
    return {
        "input_tokens": _int_or_none(getattr(usage, "input_tokens", None)),
        "output_tokens": _int_or_none(getattr(usage, "output_tokens", None)),
        "reasoning_tokens": (_int_or_none(getattr(out_details, "reasoning_tokens", None))
                             if out_details is not None else None),
        "cached_input_tokens": (_int_or_none(getattr(in_details, "cached_tokens", None))
                                if in_details is not None else None),
        "raw": _plain_dump(usage),
    }


def _items(response, item_type: str) -> list:
    return [item for item in (getattr(response, "output", None) or [])
            if getattr(item, "type", None) == item_type]


def _message_parts(response) -> tuple[list[str], bool]:
    """(output_text strings, whether any refusal part is present) across all message items."""
    texts, refused = [], False
    for message in _items(response, "message"):
        for part in getattr(message, "content", None) or []:
            part_type = getattr(part, "type", None)
            if part_type == "output_text" and isinstance(getattr(part, "text", None), str):
                texts.append(part.text)
            elif part_type == "refusal":
                refused = True
    return texts, refused


def _thinking(response) -> tuple[str | None, str]:
    """(text, kind): 'summary' with summary text, 'omitted' when reasoning items carry none, 'none' without."""
    reasoning = _items(response, "reasoning")
    if not reasoning:
        return None, "none"
    parts = [entry.text for item in reasoning for entry in (getattr(item, "summary", None) or [])
             if isinstance(getattr(entry, "text", None), str) and entry.text.strip()]
    if not parts:
        return None, "omitted"
    return "\n\n".join(parts), "summary"


def _stop(response, refused: bool) -> dict:
    status = getattr(response, "status", None)
    details = getattr(response, "incomplete_details", None)
    detail_reason = getattr(details, "reason", None) if details is not None else None
    category = None
    if status == "incomplete":
        if detail_reason in _REFUSAL_INCOMPLETE_REASONS:
            reason, category = STOP_REFUSAL, detail_reason
        else:
            reason = STOP_TRUNCATED
        incomplete_reason = detail_reason
    elif status == "completed" and refused:
        reason, incomplete_reason = STOP_REFUSAL, "refusal"
    elif status == "completed":
        reason, incomplete_reason = STOP_COMPLETE, None
    else:  # failed / cancelled / in_progress / queued / unknown
        reason = STOP_OTHER
        error = getattr(response, "error", None)
        code = getattr(error, "code", None) if error is not None else None
        incomplete_reason = code if isinstance(code, str) and code else status
    return {
        "reason": reason,
        "refusal_category": category,
        "incomplete_reason": incomplete_reason,
        "provider_stop_reason": status,
    }


def response_to_result(response, header_request_id: str | None = None) -> ProviderResult:
    """Map a final `openai.types.responses.Response` to the shared `ProviderResult`."""
    texts, refused = _message_parts(response)
    thinking_text, thinking_kind = _thinking(response)
    request_id = getattr(response, "_request_id", None)
    if not isinstance(request_id, str):
        request_id = header_request_id
    return ProviderResult(
        text="".join(texts) if texts else None,
        raw_payload=_plain_dump(response),
        usage=_normalise_usage(getattr(response, "usage", None)),
        stop=_stop(response, refused),
        native_thinking_text=thinking_text,
        native_thinking_kind=thinking_kind,
        request_id=request_id,
        response_id=getattr(response, "id", None),
        model_returned=getattr(response, "model", None),
    )


def _header_request_id(stream) -> str | None:
    """The `x-request-id` header of the underlying HTTP response, if the stream exposes it."""
    try:
        headers = getattr(getattr(stream, "_response", None), "headers", None)
        value = headers.get("x-request-id") if headers is not None else None
    except Exception:
        return None
    return value if isinstance(value, str) else None


class OpenAIResponsesAdapter:
    """MLLM adapter for the OpenAI Responses API (strict JSON schema + reasoning)."""

    provider = "openai"

    def __init__(self, client, config: dict):
        self._client = client
        self.model = config["model"]
        self.effort = config.get("effort")
        self.max_output_tokens = config["max_tokens"]
        capabilities = config.get("capabilities") or {}
        self.summary_supported = capabilities.get("summary_supported") is True
        self.sdk = {"name": "openai", "version": _sdk_version()}

        reasoning = {}
        if self.effort is not None:
            reasoning["effort"] = self.effort
        if self.summary_supported:
            reasoning["summary"] = REASONING_SUMMARY
        self.reasoning = reasoning

        omitted = {param: _SAMPLING_REMOVED_REASON.format(param=param, model=self.model)
                   for param in ("temperature", "top_p")}
        if not self.summary_supported:
            omitted["reasoning.summary"] = _SUMMARY_OMITTED_REASON
        self.settings_summary = {
            "requested": {**dict(config.get("sampling") or {}), "effort": self.effort,
                          "summary": REASONING_SUMMARY, "max_output_tokens": self.max_output_tokens,
                          "store": False},
            "applied": {**dict(reasoning), "max_output_tokens": self.max_output_tokens, "store": False},
            "omitted": omitted,
        }

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(model={self.model!r}, effort={self.effort!r}, "
                f"max_output_tokens={self.max_output_tokens})")

    def _content(self, request) -> list[dict]:
        content = []
        for role, data in request.images:
            content.append({"type": "input_text", "text": f"Image: {role}"})
            content.append({
                "type": "input_image",
                "image_url": f"data:{request.image_media_type};base64,"
                             f"{base64.b64encode(data).decode('ascii')}",
                "detail": IMAGE_DETAIL,
            })
        content.append({"type": "input_text", "text": request.user_text})
        return content

    def request_kwargs(self, request) -> dict:
        """Exactly what `responses.stream()` receives (no sampling, no tools)."""
        kwargs = {
            "model": self.model,
            "instructions": request.system_prompt,
            "input": [{"role": "user", "content": self._content(request)}],
        }
        if self.reasoning:
            kwargs["reasoning"] = dict(self.reasoning)
        kwargs["text"] = {"format": {"type": "json_schema", "name": request.schema_name,
                                     "strict": True, "schema": request.schema}}
        kwargs["max_output_tokens"] = self.max_output_tokens
        kwargs["store"] = False
        return kwargs

    def call(self, request) -> ProviderResult:
        # Exceptions propagate on purpose: the base holds, counts and redacts them.
        response = error_event = None
        with self._client.responses.stream(**self.request_kwargs(request)) as stream:
            # Not get_final_response(): it raises for incomplete/failed streams (openai 3.23.0).
            for event in stream:
                event_type = getattr(event, "type", None)
                if event_type in _TERMINAL_EVENTS:
                    response = event.response
                elif event_type == "error":
                    error_event = event
            header_request_id = _header_request_id(stream)
        if response is None:
            detail = ""
            if error_event is not None:
                detail = (f"; stream error event code={getattr(error_event, 'code', None)!r} "
                          f"message={getattr(error_event, 'message', None)!r}")
            raise RuntimeError("OpenAI stream ended without a terminal response event "
                               f"({', '.join(_TERMINAL_EVENTS)}){detail}")
        return response_to_result(response, header_request_id)

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            close()


def make_adapter(api_key: str, config: dict) -> OpenAIResponsesAdapter:
    """Adapter factory for `build_mllm_backend` (called before the robot is touched)."""
    return OpenAIResponsesAdapter(_make_openai_client(api_key, config["timeout_s"]), config)


@register_mllm_backend(BACKEND_NAME, config_name=CONFIG_NAME, adapter_factory=make_adapter)
def build(args, camera_names, joint_limits_deg):
    return base_mllm_build.build_mllm_backend(BACKEND_NAME, args, camera_names, joint_limits_deg)


__all__ = [
    "BACKEND_NAME",
    "CONFIG_NAME",
    "OpenAIResponsesAdapter",
    "build",
    "make_adapter",
    "response_to_result",
]
