"""Tests for the HF Inference Providers MLLM lane (Plan 15-08, MLLM-01, PIPE-02).

Covers the shared OpenAI-compatible chat adapter
(`vla_bridge.backends.base_mllm_chat`) and the three registered HF variants
(`mllm-hf-qwen`, `mllm-hf-deepseek`, `mllm-hf-kimi`) with their shipped
configs. Nothing here touches the network.

Streaming (Plan 15-10 fix, 2026-10-02): the adapter calls
`chat.completions.create(..., stream=True, stream_options={"include_usage": True})`
because a non-streaming reasoning call sends no bytes until it is done and the
HF router's edge proxy cut the idle connection at ~120 s (live HTTP 504 for
mllm-hf-deepseek, 15-LIVE-RESULTS.md section 1). Most tests use a fake client
whose `chat.completions.create(**kwargs)` returns a `FakeStream` of typed
`ChatCompletionChunk`s (built with the installed SDK's own recursive,
non-validating `construct`, as the SDK does). A few tests drive the REAL
`openai.OpenAI` client over an in-process `httpx2.MockTransport` that answers
with server-sent events, so the SDK's own stream decoding, header access and
mid-stream error handling are exercised with no network. The request kwargs
are bound to the INSTALLED `Completions.create` signature (openai==3.23.0) so
drift fails here, not on a paid call.
"""

import ast
import base64
import inspect
import json
import math
from pathlib import Path
from types import SimpleNamespace

import httpx2
import openai
import pytest
from openai.types.chat import ChatCompletionChunk

from conftest import MockRobot
from fake_providers import (
    FakeCaptureCamera,
    FakeCollector,
    FakeStereoCamera,
    FakeWristCap,
    make_chunk_text,
    png_bytes,
)
from run_vla_episode import run_episode
from vla_bridge import action_contract, backends
from vla_bridge.action_contract import JOINT_ORDER
from vla_bridge.backends import (
    BACKEND_REGISTRY,
    GRPC_BRIDGE_BACKENDS,
    base_mllm,
    base_mllm_build,
    base_mllm_chat,
    mllm_claude,
    mllm_hf_deepseek,
    mllm_hf_kimi,
    mllm_hf_qwen,
)
from vla_bridge.io_logger import IOLogger
from vla_bridge.backends.base_mllm import (
    IMAGE_ORDER,
    STOP_COMPLETE,
    STOP_OTHER,
    STOP_REFUSAL,
    STOP_TRUNCATED,
    MllmActionSourceBase,
    ProviderRequest,
    load_mllm_settings,
)
from vla_bridge.mllm_prompt import lint_prompt_text
from vla_bridge.mllm_schema import SCHEMA_NAME, build_action_chunk_schema

SECRET = "hf_TEST_SECRET_mllm_hf_0123456789abcdefABCDEF"
TEST_MODEL = "Org/Test-Model-7B"
TEST_PIN = "deepinfra"
MODULE_PATH = Path(base_mllm_chat.__file__)
EXPECTED_KWARGS = {"model", "messages", "response_format", "max_tokens", "temperature", "top_p",
                   "extra_body", "stream", "stream_options"}


# --- fake streaming OpenAI-compatible chat client ------------------------------------------------


def _pieces(text, count):
    """Split `text` into up to `count` consecutive non-empty pieces ("" stays one piece)."""
    if text is None:
        return []
    if text == "":
        return [""]
    size = max(1, -(-len(text) // count))
    return [text[i:i + size] for i in range(0, len(text), size)]


def make_chunks(content=None, finish_reason="stop", reasoning=None, reasoning_content=None,
                refusal=None, prompt_tokens=1200, completion_tokens=300, reasoning_tokens=None,
                cached_tokens=None, model=TEST_MODEL, choices=True, usage=True, pieces=3):
    """The router's `chat.completion.chunk` SSE payloads (dicts) for one streamed answer.

    Order as providers send it: a role delta, reasoning deltas, content deltas,
    an optional refusal delta, a finish delta, then the include_usage chunk
    (empty `choices`, `usage` set).
    """
    head = {"id": "chatcmpl-fake-1", "object": "chat.completion.chunk", "created": 1790000000,
            "model": model}

    def chunk(delta, finish=None):
        return {**head, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

    chunks = []
    if choices:
        chunks.append(chunk({"role": "assistant"}))
        reasoning_fields = {name: _pieces(text, pieces) for name, text in
                            (("reasoning", reasoning), ("reasoning_content", reasoning_content))
                            if text is not None}
        for i in range(max((len(v) for v in reasoning_fields.values()), default=0)):
            chunks.append(chunk({name: parts[i] for name, parts in reasoning_fields.items()
                                 if i < len(parts)}))
        for piece in _pieces(content, pieces):
            chunks.append(chunk({"content": piece}))
        if refusal is not None:
            chunks.append(chunk({"refusal": refusal}))
        chunks.append(chunk({}, finish_reason))
    if usage:
        chunks.append({**head, "choices": [], "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "completion_tokens_details": (None if reasoning_tokens is None
                                          else {"reasoning_tokens": reasoning_tokens}),
            "prompt_tokens_details": None if cached_tokens is None else {"cached_tokens": cached_tokens},
        }})
    return chunks


class FakeStream:
    """Stands in for openai's `Stream[ChatCompletionChunk]`: iterable, `.response.headers`, `.close()`.

    `exc` is raised after `fail_after` chunks have been yielded (a mid-stream
    error, as the SDK raises `APIError` on an error event).
    """

    def __init__(self, chunks, headers=None, exc=None, fail_after=None):
        headers = {"x-request-id": "req_hf_fake_1"} if headers is None else headers
        self.response = SimpleNamespace(headers=headers)
        self.chunks = [ChatCompletionChunk.construct(**c) if isinstance(c, dict) else c for c in chunks]
        self.exc = exc
        self.fail_after = len(self.chunks) if fail_after is None else fail_after
        self.yielded = 0
        self.closed = False

    def __iter__(self):
        for chunk in self.chunks:
            if self.exc is not None and self.yielded >= self.fail_after:
                raise self.exc
            self.yielded += 1
            yield chunk
        if self.exc is not None:
            raise self.exc

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def close(self):
        self.closed = True


def stream_from(chunks, **kwargs):
    return FakeStream(chunks, **kwargs)


class _StreamingCompletions:
    def __init__(self, state):
        self.state = state

    def create(self, **kwargs):
        self.state.calls.append(kwargs)
        if self.state.exc is not None:
            raise self.state.exc
        if not self.state.responses:
            raise AssertionError("fake chat client: unexpected extra (paid) call")
        return self.state.responses.pop(0)


def fake_chat_client(responses=(), exc=None):
    """A fake client whose `chat.completions.create(**kwargs)` replays `responses` (FakeStreams).

    It deliberately has no `with_raw_response`: the adapter must stream.
    """
    state = SimpleNamespace(calls=[], responses=list(responses), exc=exc, closed=False)
    client = SimpleNamespace(chat=SimpleNamespace(completions=_StreamingCompletions(state)), state=state)
    client.close = lambda: setattr(state, "closed", True)
    return client


def hf_config(**overrides):
    """A generated HF config merged over the shared defaults (no shipped file needed)."""
    config = load_mllm_settings()
    config.update({
        "provider": "huggingface",
        "model": TEST_MODEL,
        "hf_provider": TEST_PIN,
        "base_url": "https://router.huggingface.co/v1",
        "api_key_env": "HF_TOKEN",
        "effort": "max",
        "max_tokens": 32000,
        "structured_mode": "json_schema",
        "extra_body": {},
        "capabilities": {"sampling_supported": True, "effort_levels": None, "summary_supported": False},
        "prices": {"input_per_mtok": 0.44, "output_per_mtok": 1.32, "cached_input_per_mtok": None,
                   "verified_on": "2026-10-02", "source": "test"},
        "data_retention_note": "test note",
    })
    config.update(overrides)
    return config


def make_request(roles=("wrist", "left", "right"), media_type="image/png", chunk_len=20):
    images = [(role, png_bytes(10 + i)) for i, role in enumerate(roles)]
    return ProviderRequest(
        system_prompt="SYSTEM PROMPT TEXT",
        user_text="USER TEXT: joint state and instruction",
        images=images,
        schema=build_action_chunk_schema(chunk_len),
        schema_name=SCHEMA_NAME,
        image_media_type=media_type,
    )


def make_adapter(responses=(), exc=None, **overrides):
    client = fake_chat_client(responses, exc=exc)
    return base_mllm_chat.OpenAIChatAdapter(client, hf_config(**overrides)), client


def ok_stream(text=None, **kwargs):
    return stream_from(make_chunks(content=make_chunk_text(20) if text is None else text, **kwargs))


# --- the real openai SDK over an in-process mock transport (no network) --------------------------


def sse_body(events) -> bytes:
    """Server-sent events as the router sends them; dict events are JSON-encoded, then [DONE]."""
    out = ""
    for event in events:
        out += "data: " + (event if isinstance(event, str) else json.dumps(event)) + "\n\n"
    return (out + "data: [DONE]\n\n").encode("utf-8")


def real_sdk_client(handler):
    """A real `openai.OpenAI` whose HTTP layer is `httpx2.MockTransport(handler)` (no network)."""
    return openai.OpenAI(base_url="https://router.huggingface.co/v1", api_key=SECRET, max_retries=0,
                         timeout=900, http_client=httpx2.Client(transport=httpx2.MockTransport(handler)))


def sse_handler(events, sent, status=200, headers=None, content=None):
    def handler(request):
        sent.append(json.loads(request.content))
        response_headers = {"content-type": "text/event-stream", "x-request-id": "req_hf_sse_1"}
        response_headers.update(headers or {})
        return httpx2.Response(status, headers=response_headers,
                               content=sse_body(events) if content is None else content)
    return handler


# =============================================================================
# Config validation (SystemExit naming the field, before any hardware)
# =============================================================================


@pytest.mark.parametrize("overrides,field", [
    ({"hf_provider": ""}, "hf_provider"),
    ({"hf_provider": "   "}, "hf_provider"),
    ({"hf_provider": None}, "hf_provider"),
    ({"hf_provider": "fastest"}, "hf_provider"),
    ({"hf_provider": "cheapest"}, "hf_provider"),
    ({"hf_provider": "preferred"}, "hf_provider"),
    ({"hf_provider": "Cerebras"}, "hf_provider"),
    ({"hf_provider": "org/deepinfra"}, "hf_provider"),
    ({"hf_provider": "deepinfra:fastest"}, "hf_provider"),
    ({"base_url": "http://router.huggingface.co/v1"}, "base_url"),
    ({"base_url": "router.huggingface.co/v1"}, "base_url"),
    ({"base_url": "https://"}, "base_url"),
    ({"base_url": 5}, "base_url"),
    ({"structured_mode": "json"}, "structured_mode"),
    ({"structured_mode": None}, "structured_mode"),
    ({"extra_body": []}, "extra_body"),
    ({"extra_body": "x"}, "extra_body"),
    ({"extra_body": {"reasoning_effort": "low"}}, "extra_body"),
    ({"extra_body": {"temperature": 0.2}}, "extra_body"),
    ({"context_window": 40000}, "context_window"),
    ({"context_window": "65536"}, "context_window"),
    ({"context_window": True}, "context_window"),
    ({"provider": "openai"}, "provider"),
    ({"model": "Org/Test-Model-7B:novita"}, "model"),
])
def test_invalid_hf_config_raises_system_exit_naming_the_field(overrides, field):
    with pytest.raises(SystemExit) as excinfo:
        base_mllm_chat.validate_chat_config(hf_config(**overrides), "mllm-hf-test")
    assert f"'{field}'" in str(excinfo.value)


def test_hf_provider_fastest_is_rejected_by_the_adapter_factory():
    with pytest.raises(SystemExit, match="hf_provider"):
        base_mllm_chat.make_chat_adapter(SECRET, hf_config(hf_provider="fastest"))


def test_valid_config_passes_and_optional_keys_default():
    config = hf_config(context_window=65536)
    for key in ("base_url", "structured_mode", "extra_body"):
        config.pop(key)
    base_mllm_chat.validate_chat_config(config, "mllm-hf-test")  # no SystemExit
    adapter = base_mllm_chat.OpenAIChatAdapter(fake_chat_client(), config)
    assert adapter.structured_mode == "json_schema"
    assert adapter.base_url == base_mllm_chat.DEFAULT_BASE_URL == "https://router.huggingface.co/v1"
    assert adapter.extra_body == {}


def test_context_window_reserve_rule_boundary():
    reserve = base_mllm_chat.PROMPT_TOKEN_RESERVE
    assert reserve == 16000
    base_mllm_chat.validate_chat_config(hf_config(max_tokens=32000, context_window=32000 + reserve), "x")
    with pytest.raises(SystemExit, match="context_window"):
        base_mllm_chat.validate_chat_config(hf_config(max_tokens=32000, context_window=32000 + reserve - 1), "x")


def test_module_constants():
    assert base_mllm_chat.ROUTING_POLICY_NAMES == ("fastest", "cheapest", "preferred")
    assert base_mllm_chat.STRUCTURED_MODES == ("json_schema", "json_object")


# =============================================================================
# Request shape
# =============================================================================


def test_json_schema_request_kwargs_are_exactly_the_verified_set():
    adapter, client = make_adapter([ok_stream()])
    request = make_request()

    adapter.call(request)

    (kwargs,) = client.state.calls
    assert set(kwargs) == EXPECTED_KWARGS
    assert kwargs["model"] == f"{TEST_MODEL}:{TEST_PIN}"
    assert kwargs["max_tokens"] == 32000
    assert kwargs["temperature"] == 1.0
    assert kwargs["top_p"] == 0.95
    assert kwargs["extra_body"] == {"reasoning_effort": "max"}
    assert kwargs["stream"] is True
    assert kwargs["stream_options"] == {"include_usage": True}
    assert kwargs["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": SCHEMA_NAME, "strict": True, "schema": request.schema},
    }
    system, user = kwargs["messages"]
    assert system == {"role": "system", "content": request.system_prompt}
    assert user["role"] == "user"
    expected = []
    for role, data in request.images:
        expected.append({"type": "text", "text": f"Image: {role}"})
        expected.append({"type": "image_url", "image_url": {
            "url": "data:image/png;base64," + base64.b64encode(data).decode("ascii")}})
    expected.append({"type": "text", "text": request.user_text})
    assert user["content"] == expected
    for forbidden in ("tools", "tool_choice", "reasoning_effort"):
        assert forbidden not in kwargs


def test_request_kwargs_bind_to_the_installed_sdk_create_signature():
    from openai.resources.chat.completions import Completions

    adapter, client = make_adapter([ok_stream()])
    adapter.call(make_request())
    (kwargs,) = client.state.calls
    inspect.signature(Completions.create).bind(object(), **kwargs)  # TypeError on drift
    signature = inspect.signature(Completions.create)
    assert "stream" in signature.parameters and "stream_options" in signature.parameters
    from openai.types.chat import ChatCompletionStreamOptionsParam
    assert "include_usage" in ChatCompletionStreamOptionsParam.__annotations__


def test_call_sends_exactly_request_kwargs_and_never_the_raw_non_streaming_path():
    adapter, client = make_adapter([ok_stream()])
    request = make_request()
    adapter.call(request)
    (kwargs,) = client.state.calls
    assert kwargs == adapter.request_kwargs(request)
    assert not hasattr(client.chat.completions, "with_raw_response")


@pytest.mark.parametrize("key", ["stream", "stream_options"])
def test_extra_body_may_not_override_streaming(key):
    with pytest.raises(SystemExit, match="extra_body"):
        base_mllm_chat.validate_chat_config(hf_config(extra_body={key: False}), "mllm-hf-test")


def test_config_extra_body_merges_with_reasoning_effort():
    adapter, client = make_adapter([ok_stream()], extra_body={"chat_template_kwargs": {"enable_thinking": True}})
    adapter.call(make_request())
    assert client.state.calls[0]["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": True}, "reasoning_effort": "max"}


def test_request_labels_every_image_including_depth_and_jpeg_media_type():
    adapter, client = make_adapter([ok_stream()])
    adapter.call(make_request(roles=IMAGE_ORDER, media_type="image/jpeg"))
    content = client.state.calls[0]["messages"][1]["content"]
    labels = [part["text"] for part in content[:-1] if part["type"] == "text"]
    assert labels == [f"Image: {role}" for role in IMAGE_ORDER]
    urls = [part["image_url"]["url"] for part in content if part["type"] == "image_url"]
    assert len(urls) == 4 and all(url.startswith("data:image/jpeg;base64,") for url in urls)


def test_json_object_mode_appends_the_schema_to_the_system_message_only():
    adapter, client = make_adapter([ok_stream()], structured_mode="json_object")
    request = make_request()

    adapter.call(request)

    kwargs = client.state.calls[0]
    assert kwargs["response_format"] == {"type": "json_object"}
    system_content = kwargs["messages"][0]["content"]
    assert system_content.startswith(request.system_prompt)
    appended = system_content[len(request.system_prompt):]
    assert appended and json.dumps(request.schema) in appended
    assert lint_prompt_text(appended) == []
    assert kwargs["messages"][1]["content"][-1] == {"type": "text", "text": request.user_text}
    applied = adapter.settings_summary["applied"]
    assert applied["structured_mode"] == "json_object"
    assert applied["schema_in_prompt"] is True


def test_sampling_unsupported_omits_temperature_and_top_p_with_reasons():
    config_caps = {"sampling_supported": False, "effort_levels": None, "summary_supported": False}
    adapter, client = make_adapter([ok_stream()], capabilities=config_caps)
    adapter.call(make_request())
    kwargs = client.state.calls[0]
    assert "temperature" not in kwargs and "top_p" not in kwargs
    omitted = adapter.settings_summary["omitted"]
    assert set(omitted) >= {"temperature", "top_p"}
    assert all(isinstance(reason, str) and reason for reason in omitted.values())
    assert "temperature" not in adapter.settings_summary["applied"]


def test_effort_none_omits_reasoning_effort_and_records_it():
    adapter, client = make_adapter([ok_stream()], effort=None)
    adapter.call(make_request())
    assert client.state.calls[0]["extra_body"] == {}
    assert "reasoning_effort" in adapter.settings_summary["omitted"]
    assert "reasoning_effort" not in adapter.settings_summary["applied"]


def test_settings_summary_logs_requested_applied_and_nothing_omitted_by_default():
    adapter, _client = make_adapter()
    summary = adapter.settings_summary
    requested = summary["requested"]
    assert requested["temperature"] == 1.0 and requested["top_p"] == 0.95
    assert requested["effort"] == "max"
    assert requested["max_tokens"] == 32000
    assert requested["structured_mode"] == "json_schema"
    assert requested["hf_provider"] == TEST_PIN
    applied = summary["applied"]
    assert applied["model"] == f"{TEST_MODEL}:{TEST_PIN}"
    assert (applied["temperature"], applied["top_p"]) == (1.0, 0.95)
    assert applied["reasoning_effort"] == "max"
    assert applied["max_tokens"] == 32000
    assert applied["structured_mode"] == "json_schema"
    assert applied["schema_in_prompt"] is False
    assert "provider" in applied["note"]
    assert summary["omitted"] == {}
    json.dumps(summary)
    assert adapter.provider == "huggingface"
    assert adapter.model == f"{TEST_MODEL}:{TEST_PIN}"
    assert adapter.sdk == {"name": "openai", "version": "3.23.0"}


# =============================================================================
# Result mapping
# =============================================================================


@pytest.mark.parametrize("finish_reason,expected,category", [
    ("stop", STOP_COMPLETE, None),
    ("length", STOP_TRUNCATED, None),
    ("content_filter", STOP_REFUSAL, "content_filter"),
    ("tool_calls", STOP_OTHER, None),
    ("eos", STOP_OTHER, None),
    (None, STOP_OTHER, None),
])
def test_finish_reason_mapping(finish_reason, expected, category):
    adapter, _client = make_adapter([ok_stream(finish_reason=finish_reason)])
    stop = adapter.call(make_request()).stop
    assert stop["reason"] == expected
    assert stop["refusal_category"] == category
    assert stop["provider_stop_reason"] == finish_reason
    assert stop["incomplete_reason"] == (None if expected == STOP_COMPLETE else finish_reason)
    if finish_reason == "length":
        assert stop["incomplete_reason"] == "length"


def test_non_empty_refusal_field_maps_to_refusal():
    adapter, _client = make_adapter([ok_stream(text=None, refusal="I can't help with that.")])
    adapter_text_none = make_adapter([stream_from(make_chunks(content=None, refusal="I can't help with that."))])[0]
    for result in (adapter.call(make_request()), adapter_text_none.call(make_request())):
        assert result.stop["reason"] == STOP_REFUSAL
        assert result.stop["refusal_category"] == "message_refusal"
        assert result.stop["provider_stop_reason"] == "stop"


def test_refusal_split_across_deltas_is_joined_and_recorded():
    chunks = make_chunks(content=None, finish_reason="stop")
    head = {key: chunks[0][key] for key in ("id", "object", "created", "model")}
    refusal_deltas = [{**head, "choices": [{"index": 0, "delta": {"refusal": part}, "finish_reason": None}]}
                      for part in ("I can't ", "control robots.")]
    chunks = chunks[:1] + refusal_deltas + chunks[1:]
    adapter, _client = make_adapter([stream_from(chunks)])
    result = adapter.call(make_request())
    assert result.stop["reason"] == STOP_REFUSAL
    assert result.stop["refusal_category"] == "message_refusal"
    assert result.text is None
    assert result.raw_payload["choices"][0]["message"]["refusal"] == "I can't control robots."


def test_empty_refusal_field_is_not_a_refusal():
    adapter, _client = make_adapter([ok_stream(refusal="")])
    assert adapter.call(make_request()).stop["reason"] == STOP_COMPLETE


def test_streamed_content_is_assembled_in_order_from_many_deltas():
    text = make_chunk_text(20)
    stream = stream_from(make_chunks(content=text, pieces=17))
    adapter, _client = make_adapter([stream])
    result = adapter.call(make_request())
    assert result.text == text
    assert stream.yielded == len(stream.chunks)
    assert stream.closed is True


def test_truncated_stream_keeps_partial_text_and_usage():
    stream = stream_from(make_chunks(content='{"actions": [', finish_reason="length",
                                     completion_tokens=32000, reasoning_tokens=31000))
    adapter, _client = make_adapter([stream])
    result = adapter.call(make_request())
    assert result.text == '{"actions": ['
    assert result.stop["reason"] == STOP_TRUNCATED
    assert result.stop["incomplete_reason"] == "length"
    assert result.usage["output_tokens"] == 32000
    assert result.usage["reasoning_tokens"] == 31000


def test_finish_reason_comes_from_the_last_choice_chunk_that_carries_one():
    chunks = make_chunks(content="{}", finish_reason="length")
    head = {key: chunks[0][key] for key in ("id", "object", "created", "model")}
    # a trailing choice delta with finish_reason null must not erase the earlier value
    chunks.insert(-1, {**head, "choices": [{"index": 0, "delta": {}, "finish_reason": None}]})
    adapter, _client = make_adapter([stream_from(chunks)])
    assert adapter.call(make_request()).stop["provider_stop_reason"] == "length"


@pytest.mark.parametrize("wire_kwargs,expected", [
    ({"reasoning": "look at the wrist view first"}, "look at the wrist view first"),
    ({"reasoning_content": "the pen is left of the cup"}, "the pen is left of the cup"),
    ({"reasoning": "from reasoning", "reasoning_content": "from reasoning_content"}, "from reasoning"),
])
def test_reasoning_text_is_read_as_raw_kind(wire_kwargs, expected):
    adapter, _client = make_adapter([stream_from(make_chunks(content="{}", **wire_kwargs))])
    result = adapter.call(make_request())
    assert result.native_thinking_text == expected
    assert result.native_thinking_kind == "raw"
    assert result.text == "{}"


def test_reasoning_split_across_many_deltas_is_assembled_verbatim():
    reasoning = "First the wrist view.\nThen the left view: the pen is left of the cup.\n"
    adapter, _client = make_adapter([stream_from(make_chunks(content="{}", reasoning_content=reasoning,
                                                              pieces=11))])
    assert adapter.call(make_request()).native_thinking_text == reasoning


def test_reasoning_text_falls_back_to_model_extra():
    delta = SimpleNamespace(content=None, refusal=None,
                            model_extra={"reasoning_content": "only in model_extra"})
    done = SimpleNamespace(content="{}", refusal=None, model_extra={})
    chunks = [
        SimpleNamespace(id="c1", model="m", created=1, usage=None,
                        choices=[SimpleNamespace(index=0, finish_reason=None, delta=delta)]),
        SimpleNamespace(id="c1", model="m", created=1, usage=None,
                        choices=[SimpleNamespace(index=0, finish_reason="stop", delta=done)]),
    ]
    adapter, _client = make_adapter([FakeStream(chunks)])
    result = adapter.call(make_request())
    assert (result.native_thinking_text, result.native_thinking_kind) == ("only in model_extra", "raw")
    assert result.text == "{}"


@pytest.mark.parametrize("wire_kwargs", [{}, {"reasoning": ""}, {"reasoning_content": "   "}])
def test_absent_or_blank_reasoning_text_is_none_kind(wire_kwargs):
    adapter, _client = make_adapter([stream_from(make_chunks(content="{}", **wire_kwargs))])
    result = adapter.call(make_request())
    assert (result.native_thinking_text, result.native_thinking_kind) == (None, "none")


def test_usage_maps_prompt_completion_reasoning_and_cached_tokens():
    body = make_chunks(content="{}", prompt_tokens=1500, completion_tokens=400, reasoning_tokens=250,
                       cached_tokens=100)
    adapter, _client = make_adapter([stream_from(body)])
    usage = adapter.call(make_request()).usage
    assert usage["input_tokens"] == 1500
    assert usage["output_tokens"] == 400
    assert usage["reasoning_tokens"] == 250
    assert usage["cached_input_tokens"] == 100
    assert usage["raw"]["prompt_tokens"] == 1500
    assert usage["raw"]["completion_tokens_details"]["reasoning_tokens"] == 250
    json.dumps(usage)


def test_usage_without_details_has_reasoning_and_cached_none():
    adapter, _client = make_adapter([stream_from(make_chunks(content="{}"))])
    usage = adapter.call(make_request()).usage
    assert usage["reasoning_tokens"] is None
    assert usage["cached_input_tokens"] is None
    assert (usage["input_tokens"], usage["output_tokens"]) == (1200, 300)


def test_missing_usage_is_all_none():
    adapter, _client = make_adapter([stream_from(make_chunks(content="{}", usage=False))])
    usage = adapter.call(make_request()).usage
    assert usage == {"input_tokens": None, "output_tokens": None, "reasoning_tokens": None,
                     "cached_input_tokens": None, "raw": None}


def test_raw_payload_is_a_json_safe_reconstruction_and_ids_are_mapped():
    chunks = make_chunks(content='{"actions": []}', reasoning_content="check the wrist view",
                         reasoning_tokens=5, cached_tokens=64, model="Org/Test-Model-7B")
    adapter, _client = make_adapter([stream_from(chunks, headers={"x-request-id": "req_abc"})])
    result = adapter.call(make_request())

    payload = result.raw_payload
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload  # plain JSON, no SDK objects
    assert payload["object"] == "chat.completion"
    assert payload["reconstructed_from_stream"] is True
    assert payload["id"] == "chatcmpl-fake-1"
    assert payload["model"] == "Org/Test-Model-7B"
    assert payload["created"] == 1790000000
    (choice,) = payload["choices"]
    assert choice["finish_reason"] == "stop"
    assert choice["message"] == {"role": "assistant", "content": '{"actions": []}', "refusal": None,
                                 "reasoning": "check the wrist view"}
    assert payload["usage"]["completion_tokens_details"]["reasoning_tokens"] == 5
    assert payload["usage"]["prompt_tokens_details"]["cached_tokens"] == 64
    assert payload["stream"] == {"chunk_count": len(chunks), "choice_chunk_count": len(chunks) - 1}

    assert result.text == '{"actions": []}'
    assert result.request_id == "req_abc"
    assert result.response_id == "chatcmpl-fake-1"
    assert result.model_returned == "Org/Test-Model-7B"


def test_missing_request_id_header_is_none():
    adapter, _client = make_adapter([stream_from(make_chunks(content="{}"), headers={})])
    assert adapter.call(make_request()).request_id is None


def test_empty_choices_yield_no_text_and_other_stop():
    adapter, _client = make_adapter([stream_from(make_chunks(choices=False))])
    result = adapter.call(make_request())
    assert result.text is None
    assert result.stop["reason"] == STOP_OTHER
    assert result.stop["provider_stop_reason"] is None
    assert result.raw_payload["choices"] == []
    assert result.usage["input_tokens"] == 1200


def test_an_empty_stream_yields_no_text_and_other_stop():
    adapter, _client = make_adapter([stream_from([])])
    result = adapter.call(make_request())
    assert (result.text, result.stop["reason"], result.request_id) == (None, STOP_OTHER, "req_hf_fake_1")
    assert result.raw_payload["stream"] == {"chunk_count": 0, "choice_chunk_count": 0}


def test_client_exceptions_propagate_to_the_base():
    adapter, _client = make_adapter(exc=RuntimeError("boom"))
    with pytest.raises(RuntimeError, match="boom"):
        adapter.call(make_request())


def test_mid_stream_error_propagates_and_the_stream_is_closed():
    stream = stream_from(make_chunks(content=make_chunk_text(20)), exc=RuntimeError("stream reset"),
                         fail_after=3)
    adapter, _client = make_adapter([stream])
    with pytest.raises(RuntimeError, match="stream reset"):
        adapter.call(make_request())
    assert stream.yielded == 3
    assert stream.closed is True


# --- the real SDK stream over a mock transport (no network) ---------------------------------------


def _sse_chunks(**kwargs):
    return make_chunks(**kwargs)


def test_real_sdk_stream_is_requested_and_assembled_end_to_end():
    sent = []
    content = make_chunk_text(20)
    client = real_sdk_client(sse_handler(_sse_chunks(content=content, reasoning_content="wrist first",
                                                     reasoning_tokens=42, model="served-model-name"),
                                         sent))
    adapter = base_mllm_chat.OpenAIChatAdapter(client, hf_config())
    try:
        result = adapter.call(make_request())
    finally:
        adapter.close()

    (body,) = sent
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}
    assert body["model"] == f"{TEST_MODEL}:{TEST_PIN}"
    assert body["reasoning_effort"] == "max"  # extra_body is merged into the JSON body
    assert body["response_format"]["type"] == "json_schema"
    assert result.text == content
    assert result.native_thinking_text == "wrist first"
    assert result.usage["reasoning_tokens"] == 42
    assert result.stop["reason"] == STOP_COMPLETE
    assert result.request_id == "req_hf_sse_1"
    assert result.model_returned == "served-model-name"
    json.dumps(result.raw_payload, allow_nan=False)


def _real_sdk_source(handler, chunk_len=3):
    adapter = base_mllm_chat.OpenAIChatAdapter(real_sdk_client(handler), hf_config())
    return _base_source(adapter, chunk_len=chunk_len)


@pytest.mark.parametrize("handler_kwargs", [
    # an error event after some content: the SDK raises APIError mid-iteration
    {"events": "mid_stream_error"},
    # the router's edge proxy gateway time-out seen live (HTML body, no stream)
    {"status": 504, "content": b"<html><body><h1>504 Gateway Time-out</h1></body></html>",
     "headers": {"content-type": "text/html"}},
])
def test_real_sdk_errors_become_provider_error_holds_through_the_base(handler_kwargs):
    sent = []
    head = make_chunks(content="{")[0]
    events = [head, {**head, "choices": [{"index": 0, "delta": {"content": "{"}, "finish_reason": None}]},
              {"error": {"message": f"upstream provider reset (token {SECRET})", "code": 502}}]
    kwargs = dict(handler_kwargs)
    if kwargs.pop("events", None) is None:
        events = []
    source = _real_sdk_source(sse_handler(events, sent, **kwargs))
    joint_state = {joint: 3.0 for joint in JOINT_ORDER}

    results = [source.get_action(joint_state, "pick up the pen") for _ in range(3)]

    assert [version for _action, version in results] == ["mllm-hf-test@provider-error-holding-position"] * 3
    assert all(action == joint_state for action, _version in results)
    assert len(sent) == 1  # one call, no hidden retry
    assert SECRET not in json.dumps(source.last_details)
    source.close()


# =============================================================================
# Integration with the shared base
# =============================================================================


def _base_source(adapter, chunk_len=3):
    settings = hf_config()
    settings["chunk_len"] = chunk_len
    limits = {joint: (-180.0, 180.0) for joint in JOINT_ORDER}
    return MllmActionSourceBase(adapter, FakeCollector(), settings, limits, "mllm-hf-test", secrets=[SECRET])


def test_fenced_json_from_json_object_mode_still_yields_valid_steps():
    fenced = "```json\n" + make_chunk_text(3) + "\n```"
    adapter, client = make_adapter([ok_stream(text=fenced)], structured_mode="json_object")
    source = _base_source(adapter, chunk_len=3)
    joint_state = {joint: 3.0 for joint in JOINT_ORDER}

    versions = [source.get_action(joint_state, "pick up the pen")[1] for _ in range(1)]

    assert versions == ["mllm-hf-test@mllm-chunk"]
    call = source.last_details["mllm_call"]
    assert call["parse"]["ok"] is True
    assert call["parse"]["fence_stripped"] is True
    assert call["settings"]["applied"]["structured_mode"] == "json_object"
    assert len(client.state.calls) == 1


# =============================================================================
# Real client construction, module hygiene and secret handling
# =============================================================================


def test_make_chat_client_builds_a_real_openai_client_without_retries():
    import openai

    client = base_mllm_chat._make_chat_client(SECRET, "https://router.huggingface.co/v1", 123.0)
    try:
        assert isinstance(client, openai.OpenAI)
        assert str(client.base_url).rstrip("/") == "https://router.huggingface.co/v1"
        assert client.timeout == 123.0
        assert client.max_retries == 0
    finally:
        client.close()


def test_make_chat_adapter_uses_config_base_url_and_timeout_and_hides_the_token():
    adapter = base_mllm_chat.make_chat_adapter(SECRET, hf_config(timeout_s=321))
    try:
        assert str(adapter._client.base_url).rstrip("/") == "https://router.huggingface.co/v1"
        assert adapter._client.timeout == 321
        assert adapter._client.max_retries == 0
        assert SECRET not in repr(adapter)
        assert SECRET not in json.dumps(adapter.settings_summary)
        assert SECRET not in json.dumps(adapter.sdk)
        assert TEST_MODEL in repr(adapter)
    finally:
        adapter.close()


def test_adapter_close_closes_the_client():
    adapter, client = make_adapter()
    adapter.close()
    assert client.state.closed is True


def test_module_imports_openai_only_inside_the_client_factory():
    tree = ast.parse(MODULE_PATH.read_text())
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert all(alias.name.split(".")[0] != "openai" for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] != "openai"
    owners = []
    for func in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        for node in ast.walk(func):
            if isinstance(node, ast.Import) and any(a.name == "openai" for a in node.names):
                owners.append(func.name)
    assert owners == ["_make_chat_client"]


# =============================================================================
# The three registered HF variants (Task 2)
# =============================================================================

CAMERA_NAMES = {5: "wrist", 3: "overhead"}

# name -> (module, config name, model, pin, effort, effort_levels, (in, out) prices, context_window)
VARIANTS = {
    "mllm-hf-qwen": (mllm_hf_qwen, "mllm_hf_qwen", "Qwen/Qwen3.8-27B", "cerebras", "xhigh",
                     ["low", "medium", "xhigh"], (0.99, 1.49), 65536),
    "mllm-hf-deepseek": (mllm_hf_deepseek, "mllm_hf_deepseek", "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp",
                         "deepinfra", "max", None, (0.44, 1.32), None),
    "mllm-hf-kimi": (mllm_hf_kimi, "mllm_hf_kimi", "moonshotai/Kimi-K3", "baseten", "max",
                     ["low", "high", "max"], (3.0, 15.0), None),
}
VARIANT_NAMES = list(VARIANTS)
PINNED_MODEL_STRINGS = {
    "mllm-hf-qwen": "Qwen/Qwen3.8-27B:cerebras",
    "mllm-hf-deepseek": "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp:deepinfra",
    "mllm-hf-kimi": "moonshotai/Kimi-K3:baseten",
}


def shipped(name):
    return base_mllm_build.load_mllm_backend_config(VARIANTS[name][1], name)


@pytest.mark.parametrize("name", VARIANT_NAMES)
def test_variant_is_registered_as_an_mllm_backend_on_the_shared_chat_factory(name):
    module, config_name, *_rest = VARIANTS[name]
    assert BACKEND_REGISTRY[name] is module.build
    spec = base_mllm.MLLM_BACKENDS[name]
    assert spec.config_name == config_name
    assert spec.adapter_factory is base_mllm_chat.make_chat_adapter
    assert name not in GRPC_BRIDGE_BACKENDS
    assert (module.BACKEND_NAME, module.CONFIG_NAME) == (name, config_name)


@pytest.mark.parametrize("name", VARIANT_NAMES)
def test_variant_module_is_thin_with_no_sdk_import(name):
    source = Path(VARIANTS[name][0].__file__).read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(alias.name.split(".")[0] != "openai" for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] != "openai"
    assert not [n for n in tree.body if isinstance(n, ast.ClassDef)]
    assert [n.name for n in tree.body if isinstance(n, ast.FunctionDef)] == ["build"]
    assert "base_mllm_chat.make_chat_adapter" in source


@pytest.mark.parametrize("name", VARIANT_NAMES)
def test_variant_build_delegates_to_the_shared_builder(name, monkeypatch):
    calls = []
    sentinel = object()
    monkeypatch.setattr(base_mllm_build, "build_mllm_backend", lambda *a: calls.append(a) or sentinel)
    args = SimpleNamespace()
    assert BACKEND_REGISTRY[name](args, CAMERA_NAMES, {"x": (0, 1)}) is sentinel
    assert calls == [(name, args, CAMERA_NAMES, {"x": (0, 1)})]


@pytest.mark.parametrize("name", VARIANT_NAMES)
def test_shipped_config_passes_both_validators_with_pinned_values(name):
    _module, _config_name, model, pin, effort, levels, (price_in, price_out), context_window = VARIANTS[name]
    settings = shipped(name)
    base_mllm_chat.validate_chat_config(settings, name)  # no SystemExit

    assert settings["provider"] == "huggingface"
    assert settings["model"] == model
    assert settings["hf_provider"] == pin
    assert settings["base_url"] == "https://router.huggingface.co/v1"
    assert settings["api_key_env"] == "HF_TOKEN"
    assert settings["effort"] == effort
    assert settings["max_tokens"] == 32000
    assert settings.get("context_window") == context_window
    if context_window is not None:
        assert settings["max_tokens"] + base_mllm_chat.PROMPT_TOKEN_RESERVE <= context_window
    assert settings["structured_mode"] == "json_schema"
    assert settings["extra_body"] == {}
    caps = settings["capabilities"]
    assert caps["sampling_supported"] is True
    assert caps["effort_levels"] == levels
    assert caps["summary_supported"] is False
    prices = settings["prices"]
    assert (prices["input_per_mtok"], prices["output_per_mtok"]) == (price_in, price_out)
    assert prices["cached_input_per_mtok"] is None
    assert prices["verified_on"] == "2026-10-02"
    assert prices["source"]
    assert settings["sampling"] == {"temperature": 1.0, "top_p": 0.95}
    assert settings["requirement"] == "MLLM-01"
    assert "2026-10-02" in settings["model_verified"]
    assert "third-party" in settings["data_retention_note"]
    notes = settings["notes"]
    assert pin in notes
    assert "gemma-4-31B-it" in notes and "GLM-5.3-Flash" in notes
    assert "15-10" in notes


@pytest.mark.parametrize("name", VARIANT_NAMES)
def test_request_model_string_is_exactly_the_pinned_value(name):
    adapter = base_mllm_chat.OpenAIChatAdapter(fake_chat_client(), shipped(name))
    assert adapter.model == PINNED_MODEL_STRINGS[name]
    assert not adapter.model.endswith(":fastest")


@pytest.mark.parametrize("name", VARIANT_NAMES)
def test_shipped_config_holds_a_key_name_never_a_token(name):
    raw = (backends.BACKEND_CONFIG_DIR / f"{VARIANTS[name][1]}.json").read_text()
    assert json.loads(raw)["api_key_env"] == "HF_TOKEN"
    assert "hf_" not in raw.replace("hf_provider", "")


def test_deepseek_notes_record_the_novita_and_v4_1_decisions():
    notes = shipped("mllm-hf-deepseek")["notes"]
    assert "D-23" in notes and "Novita" in notes and "structured output" in notes
    assert "D-27" in notes and "V4.1-Flash" in notes


def test_kimi_notes_flag_cost_and_qwen_notes_name_the_deepinfra_alternative():
    assert "not cheap" in shipped("mllm-hf-kimi")["notes"]
    qwen_notes = shipped("mllm-hf-qwen")["notes"]
    assert "deepinfra" in qwen_notes and "65,536" in qwen_notes


# --- end to end: registered build -> real run_episode loop ----------------------------------------


class RecordingMockRobot(MockRobot):
    instances = []

    def __init__(self, config):
        super().__init__()
        self.config = config
        RecordingMockRobot.instances.append(self)


def _patch_hardware(monkeypatch):
    RecordingMockRobot.instances = []
    monkeypatch.setattr(base_mllm_build, "SO101Follower", RecordingMockRobot)
    monkeypatch.setattr(base_mllm_build, "StereoSplitCamera", lambda index=1: FakeStereoCamera())
    monkeypatch.setattr(base_mllm_build.cv2, "VideoCapture", lambda index: FakeWristCap())


@pytest.mark.parametrize("name", VARIANT_NAMES)
def test_end_to_end_one_chunk_through_run_episode(name, monkeypatch, tmp_path, mock_calibration_file, capsys):
    _module, _config_name, model, _pin, _effort, _levels, (price_in, price_out), _cw = VARIANTS[name]
    _patch_hardware(monkeypatch)
    monkeypatch.setenv("HF_TOKEN", SECRET)
    content = make_chunk_text(20, plan=f"Reach the pen (never echo {SECRET})")
    # pieces=7 splits the secret across deltas: redaction must act on the assembled text
    chunks = make_chunks(content=content, reasoning_content=f"scene check {SECRET}", prompt_tokens=1200,
                         completion_tokens=300, reasoning_tokens=180, model=model, pieces=7)
    client = fake_chat_client([stream_from(chunks)])
    made = []

    def fake_client_factory(api_key, base_url, timeout_s):
        made.append((api_key, base_url, timeout_s))
        return client

    monkeypatch.setattr(base_mllm_chat, "_make_chat_client", fake_client_factory)
    out = tmp_path / "out"
    args = SimpleNamespace(port="/dev/null", robot_id="soarm_follower_02", instruction="pick up the pen",
                           stereo_camera_index=1, out=out, depth_endpoint=None, depth_calibration=None)
    limits = action_contract.load_joint_limits_deg(mock_calibration_file)

    handles = BACKEND_REGISTRY[name](args, dict(CAMERA_NAMES), limits)
    (robot,) = RecordingMockRobot.instances
    assert handles.robot is robot
    assert made == [(SECRET, "https://router.huggingface.co/v1", 900)]

    with IOLogger(out, {0: "wrist"}) as io_logger:
        run_episode(robot, {0: FakeCaptureCamera()}, {0: "wrist"}, io_logger, handles.action_source,
                    "pick up the pen", 20, 100.0, limits)
    handles.action_source.close()

    assert len(client.state.calls) == 1
    assert client.state.calls[0]["model"] == PINNED_MODEL_STRINGS[name]
    records = [json.loads(line) for line in (out / "episode.jsonl").read_text().strip().splitlines()]
    assert len(records) == 20
    call = records[0]["action_details"]["mllm_call"]
    assert call["provider"] == "huggingface"
    assert call["model_requested"] == PINNED_MODEL_STRINGS[name]
    assert call["model_returned"] == model
    assert call["request_id"] == "req_hf_fake_1"
    assert call["sdk"] == {"name": "openai", "version": "3.23.0"}
    assert call["usage"]["reasoning_tokens"] == 180
    assert call["trace"]["native_thinking_kind"] == "raw"
    assert call["settings"]["applied"]["structured_mode"] == "json_schema"
    assert math.isclose(call["cost_usd_est"], (1200 * price_in + 300 * price_out) / 1e6)
    assert call["sidecar_dir"] == "mllm_calls/0000"
    assert records[0]["model_version"] == f"{name}@mllm-chunk"
    assert all("mllm_call" not in r["action_details"] for r in records[1:])
    response_raw = json.loads((out / "mllm_calls" / "0000" / "response_raw.json").read_text())
    assert response_raw["reconstructed_from_stream"] is True
    assert response_raw["model"] == model
    message = response_raw["choices"][0]["message"]
    assert message["content"] == content.replace(SECRET, "<redacted>")
    assert message["reasoning"] == "scene check <redacted>"
    assert response_raw["usage"]["completion_tokens_details"]["reasoning_tokens"] == 180
    assert response_raw["stream"]["chunk_count"] == len(chunks)
    thinking = (out / "mllm_calls" / "0000" / "thinking.txt").read_text()
    assert thinking == "scene check <redacted>"

    for path in out.rglob("*"):
        if path.is_file():
            assert SECRET.encode() not in path.read_bytes(), path
    assert SECRET not in capsys.readouterr().out


# --- failure paths through one variant ----------------------------------------------------------


@pytest.mark.parametrize("kind,expected_reason", [
    ("exception", "provider-error"),
    ("length", "truncated"),
    ("refusal", "refusal"),
    ("non_json", "invalid-json"),
    ("mid_stream_error", "provider-error"),
])
def test_failures_hold_the_whole_chunk_with_one_call(kind, expected_reason):
    chunk_len = 4
    if kind == "exception":
        client = fake_chat_client(exc=RuntimeError(f"401 Unauthorized: bearer {SECRET}"))
    elif kind == "mid_stream_error":
        client = fake_chat_client([stream_from(make_chunks(content=make_chunk_text(chunk_len)),
                                               exc=RuntimeError(f"stream reset {SECRET}"), fail_after=2)])
    elif kind == "length":
        client = fake_chat_client([stream_from(make_chunks(content='{"actions": [', finish_reason="length"))])
    elif kind == "refusal":
        client = fake_chat_client([stream_from(make_chunks(content=None, refusal="I can't control robots."))])
    else:
        client = fake_chat_client([stream_from(make_chunks(content="Sure! Here is the plan in prose."))])
    settings = shipped("mllm-hf-qwen")
    settings["chunk_len"] = chunk_len
    adapter = base_mllm_chat.OpenAIChatAdapter(client, settings)
    limits = {joint: (-180.0, 180.0) for joint in JOINT_ORDER}
    source = MllmActionSourceBase(adapter, FakeCollector(), settings, limits, "mllm-hf-qwen",
                                  secrets=[SECRET])
    joint_state = {joint: 3.0 for joint in JOINT_ORDER}

    results = [source.get_action(joint_state, "pick up the pen") for _ in range(chunk_len)]

    assert [version for _action, version in results] == [
        f"mllm-hf-qwen@{expected_reason}-holding-position"] * chunk_len
    assert all(action == joint_state for action, _version in results)
    assert len(client.state.calls) == 1
    assert SECRET not in json.dumps(source.last_details)


# --- cross-provider prompt/schema identity (PIPE-02) ----------------------------------------------


def _fake_anthropic_client():
    calls = []

    class _Stream:
        request_id = "req_claude_fake"

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def get_final_message(self):
            return SimpleNamespace(id="msg_1", model="claude-fable-5-1", stop_reason="end_turn",
                                   stop_details=None, usage=None,
                                   content=[SimpleNamespace(type="text", text=make_chunk_text(20))])

    def stream(**kwargs):
        calls.append(kwargs)
        return _Stream()

    return SimpleNamespace(messages=SimpleNamespace(stream=stream)), calls


def _hf_labelled_images(content):
    pairs = []
    for label, image in zip(content[:-1:2], content[1:-1:2]):
        header, data = image["image_url"]["url"].split(",", 1)
        pairs.append((label["text"], header[len("data:"):-len(";base64")], data))
    return pairs, content[-1]["text"]


def _claude_labelled_images(content):
    pairs = []
    for label, image in zip(content[:-1:2], content[1:-1:2]):
        pairs.append((label["text"], image["source"]["media_type"], image["source"]["data"]))
    return pairs, content[-1]["text"]


def test_hf_variants_and_claude_receive_identical_prompt_schema_and_images():
    request = make_request(roles=IMAGE_ORDER)

    claude_client, claude_calls = _fake_anthropic_client()
    mllm_claude.AnthropicAdapter(claude_client, load_mllm_settings("mllm_claude")).call(request)
    (claude_kwargs,) = claude_calls
    claude_images, claude_user_text = _claude_labelled_images(claude_kwargs["messages"][0]["content"])
    claude_schema = claude_kwargs["output_config"]["format"]["schema"]
    assert len(claude_images) == len(IMAGE_ORDER)

    for name in VARIANT_NAMES:
        client = fake_chat_client([ok_stream()])
        base_mllm_chat.OpenAIChatAdapter(client, shipped(name)).call(request)
        (kwargs,) = client.state.calls
        system, user = kwargs["messages"]
        assert system["content"].encode("utf-8") == claude_kwargs["system"].encode("utf-8")
        assert kwargs["response_format"]["json_schema"]["schema"] == claude_schema == request.schema
        hf_images, hf_user_text = _hf_labelled_images(user["content"])
        assert hf_images == claude_images, name
        assert [label for label, _media, _data in hf_images] == [f"Image: {role}" for role in IMAGE_ORDER]
        assert hf_user_text == claude_user_text == request.user_text
