"""Tests for the HF Inference Providers MLLM lane (Plan 15-08, MLLM-01, PIPE-02).

Covers the shared OpenAI-compatible chat adapter
(`vla_bridge.backends.base_mllm_chat`) and the three registered HF variants
(`mllm-hf-qwen`, `mllm-hf-deepseek`, `mllm-hf-kimi`) with their shipped
configs. Nothing here touches the network: the SDK client is a fake exposing
`chat.completions.with_raw_response.create(**kwargs)`, whose raw wrapper
returns the exact wire body as `http_response.text` and a typed completion
from `parse()` (built with the installed SDK's own non-validating
`ChatCompletion.construct`, as the SDK does). The one test that builds a real
`openai.OpenAI` client never sends a request. The request kwargs are bound to
the INSTALLED `Completions.create` signature (openai==3.23.0) so drift fails
here, not on a paid call.
"""

import ast
import base64
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from openai.types.chat import ChatCompletion

from fake_providers import FakeCollector, make_chunk_text, png_bytes
from vla_bridge.action_contract import JOINT_ORDER
from vla_bridge.backends import base_mllm_chat
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
                   "extra_body"}


# --- fake OpenAI-compatible chat client ---------------------------------------------------------


def make_wire(content=None, finish_reason="stop", reasoning=None, reasoning_content=None,
              refusal=None, prompt_tokens=1200, completion_tokens=300, reasoning_tokens=None,
              cached_tokens=None, model="Org/Test-Model-7B", choices=True, usage=True):
    """A chat.completion wire body as a dict (the router's JSON shape)."""
    message = {"role": "assistant", "content": content, "refusal": refusal}
    if reasoning is not None:
        message["reasoning"] = reasoning
    if reasoning_content is not None:
        message["reasoning_content"] = reasoning_content
    body = {
        "id": "chatcmpl-fake-1",
        "object": "chat.completion",
        "created": 1790000000,
        "model": model,
        "choices": [{"index": 0, "finish_reason": finish_reason, "message": message}] if choices else [],
    }
    if usage:
        body["usage"] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "completion_tokens_details": (None if reasoning_tokens is None
                                          else {"reasoning_tokens": reasoning_tokens}),
            "prompt_tokens_details": None if cached_tokens is None else {"cached_tokens": cached_tokens},
        }
    return body


class FakeRaw:
    """Stands in for openai's LegacyAPIResponse: `.http_response.text`, `.headers`, `.parse()`."""

    def __init__(self, wire_text, headers=None, parsed=None):
        headers = {"x-request-id": "req_hf_fake_1"} if headers is None else headers
        self.http_response = SimpleNamespace(text=wire_text, headers=headers)
        self.headers = headers
        self._parsed = parsed
        self.parse_calls = 0

    def parse(self):
        self.parse_calls += 1
        if self._parsed is not None:
            return self._parsed
        return ChatCompletion.construct(**json.loads(self.http_response.text))


def raw_from(body, **kwargs):
    return FakeRaw(json.dumps(body), **kwargs)


class _RawCompletions:
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
    """A fake client whose `chat.completions.with_raw_response.create(...)` replays `responses`."""
    state = SimpleNamespace(calls=[], responses=list(responses), exc=exc, closed=False)
    completions = SimpleNamespace(with_raw_response=_RawCompletions(state))
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions), state=state)
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


def ok_raw(text=None, **kwargs):
    return raw_from(make_wire(content=make_chunk_text(20) if text is None else text, **kwargs))


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
    adapter, client = make_adapter([ok_raw()])
    request = make_request()

    adapter.call(request)

    (kwargs,) = client.state.calls
    assert set(kwargs) == EXPECTED_KWARGS
    assert kwargs["model"] == f"{TEST_MODEL}:{TEST_PIN}"
    assert kwargs["max_tokens"] == 32000
    assert kwargs["temperature"] == 1.0
    assert kwargs["top_p"] == 0.95
    assert kwargs["extra_body"] == {"reasoning_effort": "max"}
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

    adapter, client = make_adapter([ok_raw()])
    adapter.call(make_request())
    (kwargs,) = client.state.calls
    inspect.signature(Completions.create).bind(object(), **kwargs)  # TypeError on drift


def test_config_extra_body_merges_with_reasoning_effort():
    adapter, client = make_adapter([ok_raw()], extra_body={"chat_template_kwargs": {"enable_thinking": True}})
    adapter.call(make_request())
    assert client.state.calls[0]["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": True}, "reasoning_effort": "max"}


def test_request_labels_every_image_including_depth_and_jpeg_media_type():
    adapter, client = make_adapter([ok_raw()])
    adapter.call(make_request(roles=IMAGE_ORDER, media_type="image/jpeg"))
    content = client.state.calls[0]["messages"][1]["content"]
    labels = [part["text"] for part in content[:-1] if part["type"] == "text"]
    assert labels == [f"Image: {role}" for role in IMAGE_ORDER]
    urls = [part["image_url"]["url"] for part in content if part["type"] == "image_url"]
    assert len(urls) == 4 and all(url.startswith("data:image/jpeg;base64,") for url in urls)


def test_json_object_mode_appends_the_schema_to_the_system_message_only():
    adapter, client = make_adapter([ok_raw()], structured_mode="json_object")
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
    adapter, client = make_adapter([ok_raw()], capabilities=config_caps)
    adapter.call(make_request())
    kwargs = client.state.calls[0]
    assert "temperature" not in kwargs and "top_p" not in kwargs
    omitted = adapter.settings_summary["omitted"]
    assert set(omitted) >= {"temperature", "top_p"}
    assert all(isinstance(reason, str) and reason for reason in omitted.values())
    assert "temperature" not in adapter.settings_summary["applied"]


def test_effort_none_omits_reasoning_effort_and_records_it():
    adapter, client = make_adapter([ok_raw()], effort=None)
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
    adapter, _client = make_adapter([ok_raw(finish_reason=finish_reason)])
    stop = adapter.call(make_request()).stop
    assert stop["reason"] == expected
    assert stop["refusal_category"] == category
    assert stop["provider_stop_reason"] == finish_reason
    assert stop["incomplete_reason"] == (None if expected == STOP_COMPLETE else finish_reason)
    if finish_reason == "length":
        assert stop["incomplete_reason"] == "length"


def test_non_empty_refusal_field_maps_to_refusal():
    adapter, _client = make_adapter([ok_raw(text=None, refusal="I can't help with that.")])
    adapter_text_none = make_adapter([raw_from(make_wire(content=None, refusal="I can't help with that."))])[0]
    for result in (adapter.call(make_request()), adapter_text_none.call(make_request())):
        assert result.stop["reason"] == STOP_REFUSAL
        assert result.stop["refusal_category"] == "message_refusal"


def test_empty_refusal_field_is_not_a_refusal():
    adapter, _client = make_adapter([ok_raw(refusal="")])
    assert adapter.call(make_request()).stop["reason"] == STOP_COMPLETE


@pytest.mark.parametrize("wire_kwargs,expected", [
    ({"reasoning": "look at the wrist view first"}, "look at the wrist view first"),
    ({"reasoning_content": "the pen is left of the cup"}, "the pen is left of the cup"),
    ({"reasoning": "from reasoning", "reasoning_content": "from reasoning_content"}, "from reasoning"),
])
def test_reasoning_text_is_read_as_raw_kind(wire_kwargs, expected):
    adapter, _client = make_adapter([raw_from(make_wire(content="{}", **wire_kwargs))])
    result = adapter.call(make_request())
    assert result.native_thinking_text == expected
    assert result.native_thinking_kind == "raw"


def test_reasoning_text_falls_back_to_model_extra():
    message = SimpleNamespace(content="{}", refusal=None,
                              model_extra={"reasoning_content": "only in model_extra"})
    completion = SimpleNamespace(id="c1", model="m", usage=None,
                                 choices=[SimpleNamespace(finish_reason="stop", message=message)])
    adapter, _client = make_adapter([FakeRaw("{}", parsed=completion)])
    result = adapter.call(make_request())
    assert (result.native_thinking_text, result.native_thinking_kind) == ("only in model_extra", "raw")


@pytest.mark.parametrize("wire_kwargs", [{}, {"reasoning": ""}, {"reasoning_content": "   "}])
def test_absent_or_blank_reasoning_text_is_none_kind(wire_kwargs):
    adapter, _client = make_adapter([raw_from(make_wire(content="{}", **wire_kwargs))])
    result = adapter.call(make_request())
    assert (result.native_thinking_text, result.native_thinking_kind) == (None, "none")


def test_usage_maps_prompt_completion_reasoning_and_cached_tokens():
    body = make_wire(content="{}", prompt_tokens=1500, completion_tokens=400, reasoning_tokens=250,
                     cached_tokens=100)
    adapter, _client = make_adapter([raw_from(body)])
    usage = adapter.call(make_request()).usage
    assert usage["input_tokens"] == 1500
    assert usage["output_tokens"] == 400
    assert usage["reasoning_tokens"] == 250
    assert usage["cached_input_tokens"] == 100
    assert usage["raw"]["prompt_tokens"] == 1500
    assert usage["raw"]["completion_tokens_details"]["reasoning_tokens"] == 250
    json.dumps(usage)


def test_usage_without_details_has_reasoning_and_cached_none():
    adapter, _client = make_adapter([raw_from(make_wire(content="{}"))])
    usage = adapter.call(make_request()).usage
    assert usage["reasoning_tokens"] is None
    assert usage["cached_input_tokens"] is None
    assert (usage["input_tokens"], usage["output_tokens"]) == (1200, 300)


def test_missing_usage_is_all_none():
    adapter, _client = make_adapter([raw_from(make_wire(content="{}", usage=False))])
    usage = adapter.call(make_request()).usage
    assert usage == {"input_tokens": None, "output_tokens": None, "reasoning_tokens": None,
                     "cached_input_tokens": None, "raw": None}


def test_raw_payload_is_the_verbatim_wire_body_and_ids_are_mapped():
    wire_text = json.dumps(make_wire(content='{"actions": []}', model="Org/Test-Model-7B"), indent=1)
    adapter, _client = make_adapter([FakeRaw(wire_text, headers={"x-request-id": "req_abc"})])
    result = adapter.call(make_request())
    assert result.raw_payload == wire_text
    assert isinstance(result.raw_payload, str)
    assert result.text == '{"actions": []}'
    assert result.request_id == "req_abc"
    assert result.response_id == "chatcmpl-fake-1"
    assert result.model_returned == "Org/Test-Model-7B"


def test_missing_request_id_header_is_none():
    adapter, _client = make_adapter([raw_from(make_wire(content="{}"), headers={})])
    assert adapter.call(make_request()).request_id is None


def test_empty_choices_yield_no_text_and_other_stop():
    adapter, _client = make_adapter([raw_from(make_wire(choices=False))])
    result = adapter.call(make_request())
    assert result.text is None
    assert result.stop["reason"] == STOP_OTHER


def test_client_exceptions_propagate_to_the_base():
    adapter, _client = make_adapter(exc=RuntimeError("boom"))
    with pytest.raises(RuntimeError, match="boom"):
        adapter.call(make_request())


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
    adapter, client = make_adapter([ok_raw(text=fenced)], structured_mode="json_object")
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
