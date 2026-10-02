"""Tests for `--backend mllm-gpt` (Plan 15-07, MLLM-03, PIPE-02, MLLM-04).

The GPT adapter (`vla_bridge.backends.mllm_gpt`), its registration and its
shipped config. No test calls the real OpenAI API: the SDK client is a fake
exposing `responses.stream(**kwargs)` as a context manager that yields stream
events, and the tests that build a real `openai.OpenAI` client or a real
`ResponseStream` never open a connection. The request kwargs are bound to the
INSTALLED SDK's `Responses.stream` signature so drift from openai==3.23.0
fails here, not on a paid call.
"""

import ast
import base64
import inspect
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

import test_backend_mllm_claude as claude_tests
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
from vla_bridge.backends import BACKEND_REGISTRY, GRPC_BRIDGE_BACKENDS, base_mllm, base_mllm_build
from vla_bridge.backends import mllm_claude, mllm_gpt
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
from vla_bridge.io_logger import IOLogger
from vla_bridge.mllm_schema import SCHEMA_NAME, build_action_chunk_schema

BACKEND_NAME = "mllm-gpt"
CONFIG_NAME = "mllm_gpt"
MODEL = "gpt-6-astra"
SECRET = "sk-proj-TEST-SECRET-mllm-gpt-0123456789abcdefABCDEF"
CAMERA_NAMES = {5: "wrist", 3: "overhead"}
MODULE_PATH = Path(mllm_gpt.__file__)
SHIPPED_CONFIG_PATH = backends.BACKEND_CONFIG_DIR / f"{CONFIG_NAME}.json"
EXPECTED_KWARGS = {"model", "instructions", "input", "reasoning", "text", "max_output_tokens", "store"}


# --- fake OpenAI SDK client --------------------------------------------------------------------


def _plain(obj):
    """Recursively turn SimpleNamespace fakes into JSON-able dicts (like model_dump(mode='json'))."""
    if isinstance(obj, SimpleNamespace):
        return {k: _plain(v) for k, v in vars(obj).items() if not k.startswith("_") and not callable(v)}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    return obj


def _usage(input_tokens=1200, output_tokens=300, cached_tokens=200, reasoning_tokens=120):
    in_details = None if cached_tokens is None else SimpleNamespace(cached_tokens=cached_tokens)
    out_details = None if reasoning_tokens is None else SimpleNamespace(reasoning_tokens=reasoning_tokens)
    usage = SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=(input_tokens or 0) + (output_tokens or 0),
        input_tokens_details=in_details,
        output_tokens_details=out_details,
    )
    usage.model_dump = lambda mode="python", _u=usage: _plain(_u)
    return usage


class FakeResponse(SimpleNamespace):
    def model_dump(self, mode="python"):
        dumped = _plain(self)
        dumped["dumped_mode"] = mode
        return dumped


def message_item(text=None, refusal=None):
    parts = []
    if text is not None:
        parts.append(SimpleNamespace(type="output_text", text=text, annotations=[]))
    if refusal is not None:
        parts.append(SimpleNamespace(type="refusal", refusal=refusal))
    return SimpleNamespace(type="message", id="msg_1", role="assistant", status="completed", content=parts)


def reasoning_item(*summaries):
    return SimpleNamespace(type="reasoning", id="rs_1",
                           summary=[SimpleNamespace(type="summary_text", text=s) for s in summaries])


def make_response(text=None, summary=None, status="completed", incomplete_reason=None, refusal=None,
                  usage=None, request_id="req_fake_gpt_123", output=None, error=None):
    if output is None:
        output = []
        if summary is not None:
            output.append(reasoning_item(*([summary] if isinstance(summary, str) else summary)))
        if text is not None or refusal is not None:
            output.append(message_item(text=text, refusal=refusal))
    response = FakeResponse(
        id="resp_fake_1",
        object="response",
        model=f"{MODEL}-2026-09-15",
        status=status,
        incomplete_details=None if incomplete_reason is None else SimpleNamespace(reason=incomplete_reason),
        error=error,
        output=output,
        usage=usage if usage is not None else _usage(),
    )
    response._request_id = request_id
    return response


_TERMINAL_EVENT = {"completed": "response.completed", "incomplete": "response.incomplete",
                   "failed": "response.failed"}


class FakeStream:
    """Mimics openai's ResponseStream: iterable stream events, raw httpx response in `_response`."""

    def __init__(self, response, header_request_id=None, terminal=True):
        self.response = response
        self._response = SimpleNamespace(headers={"x-request-id": header_request_id} if header_request_id else {})
        self.terminal = terminal
        self.entered = False
        self.exited = False

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, *exc):
        self.exited = True
        return False

    def __iter__(self):
        yield SimpleNamespace(type="response.created", sequence_number=0)
        yield SimpleNamespace(type="response.output_text.delta", sequence_number=1, delta="{")
        if self.terminal:
            event_type = _TERMINAL_EVENT.get(self.response.status, "response.completed")
            yield SimpleNamespace(type=event_type, sequence_number=2, response=self.response)

    def get_final_response(self):  # pragma: no cover - the adapter must not depend on it (see Rule 1 note)
        raise AssertionError("adapter must read the terminal stream event, not get_final_response()")


class FakeResponses:
    def __init__(self, owner):
        self.owner = owner

    def stream(self, **kwargs):
        self.owner.calls.append(kwargs)
        if self.owner.exc is not None:
            raise self.owner.exc
        if not self.owner.responses:
            raise AssertionError("fake OpenAI client: unexpected extra (paid) call")
        stream = FakeStream(self.owner.responses.pop(0), self.owner.header_request_id, self.owner.terminal)
        self.owner.streams.append(stream)
        return stream


def _client(responses=(), exc=None, header_request_id=None, terminal=True):
    """A fake client whose `.responses.stream(...)` replays `responses`."""
    state = SimpleNamespace(calls=[], streams=[], closed=False, exc=exc, responses=list(responses),
                            header_request_id=header_request_id, terminal=terminal)
    client = SimpleNamespace(responses=FakeResponses(state), state=state)
    client.close = lambda: setattr(state, "closed", True)
    return client


def _config(**overrides):
    config = load_mllm_settings(CONFIG_NAME)
    config.update(overrides)
    return config


def _config_with_caps(**cap_overrides):
    config = _config()
    config["capabilities"] = {**config["capabilities"], **cap_overrides}
    return config


def _request(roles=("wrist", "left", "right"), media_type="image/png", chunk_len=20):
    images = [(role, png_bytes(10 + i)) for i, role in enumerate(roles)]
    return ProviderRequest(
        system_prompt="SYSTEM PROMPT TEXT",
        user_text="USER TEXT: joint state and instruction",
        images=images,
        schema=build_action_chunk_schema(chunk_len),
        schema_name=SCHEMA_NAME,
        image_media_type=media_type,
    )


def _adapter(responses=(), config=None, **kwargs):
    client = _client(responses, **kwargs)
    return mllm_gpt.OpenAIResponsesAdapter(client, config if config is not None else _config()), client


# =============================================================================
# Registration and module hygiene
# =============================================================================


def test_mllm_gpt_is_registered_as_an_mllm_backend_not_grpc():
    assert BACKEND_REGISTRY[BACKEND_NAME] is mllm_gpt.build
    spec = base_mllm.MLLM_BACKENDS[BACKEND_NAME]
    assert spec.config_name == CONFIG_NAME
    assert spec.adapter_factory is mllm_gpt.make_adapter
    assert BACKEND_NAME not in GRPC_BRIDGE_BACKENDS
    assert (mllm_gpt.BACKEND_NAME, mllm_gpt.CONFIG_NAME) == (BACKEND_NAME, CONFIG_NAME)


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
    assert owners == ["_make_openai_client"]


def test_build_delegates_to_the_shared_builder(monkeypatch):
    calls = []
    sentinel = object()
    monkeypatch.setattr(base_mllm_build, "build_mllm_backend", lambda *a: calls.append(a) or sentinel)
    args = SimpleNamespace()
    assert BACKEND_REGISTRY[BACKEND_NAME](args, CAMERA_NAMES, {"x": (0, 1)}) is sentinel
    assert calls == [(BACKEND_NAME, args, CAMERA_NAMES, {"x": (0, 1)})]


# =============================================================================
# Shipped config
# =============================================================================


def test_shipped_config_passes_the_strict_validator_with_pinned_values():
    settings = base_mllm_build.load_mllm_backend_config(CONFIG_NAME, BACKEND_NAME)

    assert settings["provider"] == "openai"
    assert settings["model"] == MODEL
    assert settings["api_key_env"] == "OPENAI_API_KEY"
    assert settings["effort"] == "xhigh"
    assert settings["max_tokens"] == 64000
    prices = settings["prices"]
    assert (prices["input_per_mtok"], prices["output_per_mtok"], prices["cached_input_per_mtok"]) == (10.0, 50.0, 1.0)
    assert prices["verified_on"] == "2026-10-02"
    assert "developers.openai.com" in prices["source"]
    caps = settings["capabilities"]
    assert caps["sampling_supported"] is False
    assert caps["effort_levels"] == ["low", "medium", "high", "xhigh", "max"]
    assert caps["summary_supported"] is True
    assert "store=false" in settings["data_retention_note"]
    assert settings["data_retention_note"].strip()
    assert "2026-10-02" in settings["model_verified"] and MODEL in settings["model_verified"]
    assert settings["requirement"] == "MLLM-03"
    assert settings["timeout_s"] == 900  # inherited from mllm_defaults.json
    assert settings["sampling"] == {"temperature": 1.0, "top_p": 0.95}  # shared nominal, never sent


def test_shipped_config_holds_a_key_name_never_a_key():
    raw = SHIPPED_CONFIG_PATH.read_text()
    config = json.loads(raw)
    assert config["api_key_env"] == "OPENAI_API_KEY"
    assert "sk-" not in raw


# =============================================================================
# Request shape
# =============================================================================


def _expected_content(request, detail="high"):
    expected = []
    for role, data in request.images:
        expected.append({"type": "input_text", "text": f"Image: {role}"})
        expected.append({"type": "input_image",
                         "image_url": f"data:{request.image_media_type};base64,"
                                      f"{base64.b64encode(data).decode('ascii')}",
                         "detail": detail})
    expected.append({"type": "input_text", "text": request.user_text})
    return expected


def test_request_kwargs_are_exactly_the_verified_set_in_canonical_order():
    adapter, client = _adapter([make_response(text=make_chunk_text(20))])
    request = _request()

    adapter.call(request)

    (kwargs,) = client.state.calls
    assert set(kwargs) == EXPECTED_KWARGS
    assert kwargs["model"] == MODEL
    assert kwargs["instructions"] == request.system_prompt
    assert kwargs["reasoning"] == {"effort": "xhigh", "summary": "auto"}
    assert kwargs["text"] == {"format": {"type": "json_schema", "name": SCHEMA_NAME, "strict": True,
                                         "schema": request.schema}}
    assert kwargs["text"]["format"]["schema"] is request.schema
    assert kwargs["max_output_tokens"] == 64000
    assert kwargs["store"] is False
    (message,) = kwargs["input"]
    assert message["role"] == "user"
    assert message["content"] == _expected_content(request)
    stream = client.state.streams[0]
    assert stream.entered and stream.exited


def test_request_sends_no_sampling_or_tools():
    adapter, client = _adapter([make_response(text=make_chunk_text(20))])
    adapter.call(_request())
    (kwargs,) = client.state.calls
    for forbidden in ("temperature", "top_p", "top_logprobs", "tools", "tool_choice", "previous_response_id",
                      "text_format", "background"):
        assert forbidden not in kwargs
    assert [m["role"] for m in kwargs["input"]] == ["user"]


def test_request_kwargs_bind_to_the_installed_sdk_stream_signature():
    from openai.resources.responses import Responses

    adapter, client = _adapter([make_response(text=make_chunk_text(20))])
    adapter.call(_request())
    (kwargs,) = client.state.calls
    inspect.signature(Responses.stream).bind(object(), **kwargs)  # raises TypeError on drift


def test_request_labels_every_image_including_depth_and_jpeg_media_type():
    adapter, client = _adapter([make_response(text=make_chunk_text(20))])
    request = _request(roles=IMAGE_ORDER, media_type="image/jpeg")
    adapter.call(request)
    content = client.state.calls[0]["input"][0]["content"]
    assert content == _expected_content(request)
    labels = [part["text"] for part in content[:-1] if part["type"] == "input_text"]
    assert labels == [f"Image: {role}" for role in IMAGE_ORDER]
    images = [part for part in content if part["type"] == "input_image"]
    assert len(images) == 4
    assert all(img["image_url"].startswith("data:image/jpeg;base64,") for img in images)
    assert content[-1] == {"type": "input_text", "text": request.user_text}


def test_summary_unsupported_sends_effort_only_and_logs_the_omission():
    """Assumption A5: an unsupported parameter is omitted up front from the static table, never negotiated."""
    adapter, client = _adapter([make_response(text=make_chunk_text(20))],
                               config=_config_with_caps(summary_supported=False))
    adapter.call(_request())

    (kwargs,) = client.state.calls
    assert kwargs["reasoning"] == {"effort": "xhigh"}
    assert len(client.state.calls) == 1  # no try/400/retry anywhere
    summary = adapter.settings_summary
    assert "reasoning.summary" in summary["omitted"]
    assert "summary_supported" in summary["omitted"]["reasoning.summary"]
    assert "summary" not in summary["applied"]
    assert summary["requested"]["summary"] == "auto"


def test_effort_none_is_left_out_of_reasoning():
    adapter, client = _adapter([make_response(text=make_chunk_text(20))], config=_config(effort=None))
    adapter.call(_request())
    assert client.state.calls[0]["reasoning"] == {"summary": "auto"}


def test_a_400_for_an_unsupported_parameter_is_never_retried():
    adapter, client = _adapter(exc=RuntimeError("400 Unsupported parameter: 'reasoning.summary'"))
    with pytest.raises(RuntimeError, match="reasoning.summary"):
        adapter.call(_request())
    assert len(client.state.calls) == 1


# =============================================================================
# Result mapping
# =============================================================================


def test_result_maps_text_summary_thinking_ids_and_raw_payload():
    response = make_response(text='{"actions": []}', summary=["Look at the wrist image.", "Then reach."])
    adapter, _c = _adapter([response])

    result = adapter.call(_request())

    assert result.text == '{"actions": []}'
    assert result.native_thinking_text == "Look at the wrist image.\n\nThen reach."
    assert result.native_thinking_kind == "summary"
    assert result.request_id == "req_fake_gpt_123"
    assert result.response_id == "resp_fake_1"
    assert result.model_returned == f"{MODEL}-2026-09-15"
    assert result.raw_payload == response.model_dump(mode="json")
    assert result.raw_payload["dumped_mode"] == "json"
    assert result.stop == {"reason": STOP_COMPLETE, "refusal_category": None, "incomplete_reason": None,
                           "provider_stop_reason": "completed"}


def test_reasoning_without_summary_text_is_omitted_and_no_reasoning_is_none():
    adapter, _c = _adapter([make_response(text="{}", summary=[]),
                            make_response(text="{}", summary=["   "]),
                            make_response(text="{}")])
    first, second, third = (adapter.call(_request()) for _ in range(3))
    assert (first.native_thinking_kind, first.native_thinking_text) == ("omitted", None)
    assert (second.native_thinking_kind, second.native_thinking_text) == ("omitted", None)
    assert (third.native_thinking_kind, third.native_thinking_text) == ("none", None)


def test_multiple_output_text_parts_join_and_no_message_gives_text_none():
    message = SimpleNamespace(type="message", role="assistant", content=[
        SimpleNamespace(type="output_text", text='{"a": '), SimpleNamespace(type="output_text", text="1}")])
    adapter, _c = _adapter([make_response(output=[message]), make_response(summary="only reasoning")])
    assert adapter.call(_request()).text == '{"a": 1}'
    assert adapter.call(_request()).text is None


def test_incomplete_max_output_tokens_is_truncated():
    response = make_response(text='{"actions": [', status="incomplete", incomplete_reason="max_output_tokens")
    adapter, _c = _adapter([response])
    stop = adapter.call(_request()).stop
    assert stop == {"reason": STOP_TRUNCATED, "refusal_category": None, "incomplete_reason": "max_output_tokens",
                    "provider_stop_reason": "incomplete"}


def test_incomplete_before_any_visible_output_is_truncated():
    response = make_response(summary=[], status="incomplete", incomplete_reason="max_output_tokens")
    result = _adapter([response])[0].call(_request())
    assert result.stop["reason"] == STOP_TRUNCATED
    assert result.text is None


def test_incomplete_content_filter_is_a_refusal_with_its_category():
    response = make_response(text="", status="incomplete", incomplete_reason="content_filter")
    stop = _adapter([response])[0].call(_request()).stop
    assert stop["reason"] == STOP_REFUSAL
    assert stop["refusal_category"] == "content_filter"
    assert stop["incomplete_reason"] == "content_filter"


def test_refusal_content_part_is_a_refusal():
    response = make_response(refusal="I can't help with that.")
    result = _adapter([response])[0].call(_request())
    assert result.stop["reason"] == STOP_REFUSAL
    assert result.stop["refusal_category"] is None
    assert result.stop["incomplete_reason"] == "refusal"
    assert result.text is None


@pytest.mark.parametrize("status", ["failed", "cancelled", "in_progress", "queued", None])
def test_failed_cancelled_and_unknown_status_map_to_other(status):
    response = make_response(text="{}", status=status,
                             error=SimpleNamespace(code="server_error", message="boom") if status == "failed" else None)
    result = mllm_gpt.response_to_result(response)
    assert result.stop["reason"] == STOP_OTHER
    assert result.stop["provider_stop_reason"] == status
    expected_incomplete = "server_error" if status == "failed" else status
    assert result.stop["incomplete_reason"] == expected_incomplete


def test_failed_status_through_the_stream_maps_to_other():
    response = make_response(text="{}", status="failed", error=SimpleNamespace(code="server_error", message="x"))
    assert _adapter([response])[0].call(_request()).stop["reason"] == STOP_OTHER


def test_usage_is_normalised_with_cached_and_reasoning_tokens():
    usage = _usage(input_tokens=1200, output_tokens=300, cached_tokens=200, reasoning_tokens=120)
    result = _adapter([make_response(text="{}", usage=usage)])[0].call(_request())
    assert result.usage["input_tokens"] == 1200  # already includes cached tokens
    assert result.usage["cached_input_tokens"] == 200
    assert result.usage["output_tokens"] == 300
    assert result.usage["reasoning_tokens"] == 120
    assert result.usage["raw"]["input_tokens_details"] == {"cached_tokens": 200}
    json.dumps(result.usage)


def test_usage_without_details_has_none_and_missing_usage_is_all_none():
    usage = _usage(cached_tokens=None, reasoning_tokens=None)
    result = _adapter([make_response(text="{}", usage=usage)])[0].call(_request())
    assert result.usage["cached_input_tokens"] is None
    assert result.usage["reasoning_tokens"] is None
    assert result.usage["input_tokens"] == 1200
    response = make_response(text="{}")
    response.usage = None
    assert mllm_gpt.response_to_result(response).usage == {
        "input_tokens": None, "output_tokens": None, "reasoning_tokens": None,
        "cached_input_tokens": None, "raw": None}


def test_request_id_falls_back_to_the_stream_header():
    response = make_response(text="{}", request_id=None)
    adapter, _c = _adapter([response], header_request_id="req_from_header")
    assert adapter.call(_request()).request_id == "req_from_header"


def test_stream_without_a_terminal_event_raises_for_the_base_to_hold():
    adapter, client = _adapter([make_response(text="{}")], terminal=False)
    with pytest.raises(RuntimeError, match="terminal"):
        adapter.call(_request())
    assert len(client.state.calls) == 1


def test_stream_error_event_detail_is_surfaced_when_no_terminal_event_arrives():
    class ErrorStream(FakeStream):
        def __iter__(self):
            yield SimpleNamespace(type="response.created", sequence_number=0)
            yield SimpleNamespace(type="error", sequence_number=1, code="server_error", message="overloaded")

    adapter, client = _adapter([make_response(text="{}")])
    client.responses.stream = lambda **kwargs: client.state.calls.append(kwargs) or ErrorStream(make_response())
    with pytest.raises(RuntimeError, match="server_error.*overloaded"):
        adapter.call(_request())
    assert len(client.state.calls) == 1


# =============================================================================
# The real SDK stream helper (openai 3.23.0) drives the adapter correctly
# =============================================================================


def _sdk_response(status, output, incomplete=None):
    return {"id": "resp_sdk_1", "object": "response", "created_at": 1, "model": MODEL, "status": status,
            "output": output, "parallel_tool_calls": False, "tool_choice": "auto", "tools": [],
            "incomplete_details": incomplete,
            "usage": {"input_tokens": 900, "output_tokens": 64000, "total_tokens": 64900,
                      "input_tokens_details": {"cached_tokens": 0},
                      "output_tokens_details": {"reasoning_tokens": 63990}}}


def _real_sdk_stream(terminal_event):
    """A real `openai...ResponseStream` over typed events; no network."""
    from openai._models import construct_type
    from openai._types import omit
    from openai.lib.streaming.responses._responses import ResponseStream
    from openai.types.responses.response_stream_event import ResponseStreamEvent

    events = [construct_type(type_=ResponseStreamEvent, value={
                  "type": "response.created", "sequence_number": 0,
                  "response": _sdk_response("in_progress", [])}),
              construct_type(type_=ResponseStreamEvent, value=terminal_event)]
    http_response = SimpleNamespace(close=lambda: None, headers={"x-request-id": "req_sdk_hdr"})
    raw_stream = type("RawStream", (), {"__iter__": lambda self: iter(events), "response": http_response})()
    return ResponseStream(raw_stream=raw_stream, text_format=omit, input_tools=omit, starting_after=None)


class _SdkStreamManager:
    def __init__(self, stream):
        self.stream = stream

    def __enter__(self):
        return self.stream

    def __exit__(self, *exc):
        self.stream.close()
        return False


def _sdk_client(terminal_event):
    calls = []

    def stream(**kwargs):
        calls.append(kwargs)
        return _SdkStreamManager(_real_sdk_stream(terminal_event))

    return SimpleNamespace(responses=SimpleNamespace(stream=stream), calls=calls)


INCOMPLETE_EVENT = {"type": "response.incomplete", "sequence_number": 1,
                    "response": _sdk_response("incomplete", [{"type": "reasoning", "id": "rs_1", "summary": []}],
                                              {"reason": "max_output_tokens"})}


def test_sdk_get_final_response_cannot_see_an_incomplete_response():
    """Why the adapter reads the terminal event: the helper only finalises on response.completed."""
    stream = _real_sdk_stream(INCOMPLETE_EVENT)
    with pytest.raises(RuntimeError, match="response.completed"):
        stream.get_final_response()


def test_real_sdk_stream_incomplete_maps_to_truncated_with_header_request_id():
    client = _sdk_client(INCOMPLETE_EVENT)
    adapter = mllm_gpt.OpenAIResponsesAdapter(client, _config())
    result = adapter.call(_request())
    assert result.stop["reason"] == STOP_TRUNCATED
    assert result.stop["incomplete_reason"] == "max_output_tokens"
    assert result.native_thinking_kind == "omitted"
    assert result.usage["reasoning_tokens"] == 63990
    assert result.request_id == "req_sdk_hdr"
    assert result.raw_payload["status"] == "incomplete"
    json.dumps(result.raw_payload)


def test_real_sdk_stream_completed_maps_text_and_summary():
    output = [{"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "Plan the reach."}]},
              {"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
               "content": [{"type": "output_text", "text": '{"actions": []}', "annotations": []}]}]
    completed = {"type": "response.completed", "sequence_number": 1,
                 "response": _sdk_response("completed", output)}
    result = mllm_gpt.OpenAIResponsesAdapter(_sdk_client(completed), _config()).call(_request())
    assert result.stop["reason"] == STOP_COMPLETE
    assert result.text == '{"actions": []}'
    assert (result.native_thinking_kind, result.native_thinking_text) == ("summary", "Plan the reach.")
    assert result.response_id == "resp_sdk_1"


# =============================================================================
# Settings summary (D-20 / D-21)
# =============================================================================


def test_settings_summary_logs_requested_applied_and_omitted_with_reasons():
    adapter, _c = _adapter()
    summary = adapter.settings_summary
    assert summary["requested"] == {"temperature": 1.0, "top_p": 0.95, "effort": "xhigh", "summary": "auto",
                                    "max_output_tokens": 64000, "store": False}
    assert summary["applied"] == {"effort": "xhigh", "summary": "auto", "max_output_tokens": 64000,
                                  "store": False}
    assert set(summary["omitted"]) == {"temperature", "top_p"}
    for reason in summary["omitted"].values():
        assert "reasoning effort is not none" in reason
        assert "does not support none" in reason
    json.dumps(summary)
    assert adapter.provider == "openai"
    assert adapter.model == MODEL
    assert adapter.sdk == {"name": "openai", "version": "3.23.0"}


# =============================================================================
# Real client construction and secret handling
# =============================================================================


def test_make_openai_client_builds_a_real_client_without_retries():
    import openai

    client = mllm_gpt._make_openai_client(SECRET, 123.0)
    try:
        assert isinstance(client, openai.OpenAI)
        assert client.max_retries == 0
        assert client.timeout == 123.0
    finally:
        client.close()


def test_make_adapter_uses_the_configured_timeout_and_hides_the_key():
    adapter = mllm_gpt.make_adapter(SECRET, _config())
    try:
        assert adapter._client.max_retries == 0
        assert adapter._client.timeout == 900
        assert SECRET not in repr(adapter)
        assert SECRET not in json.dumps(adapter.settings_summary)
        assert SECRET not in json.dumps(adapter.sdk)
        assert not any(isinstance(v, str) and SECRET in v for v in vars(adapter).values())
    finally:
        adapter.close()


def test_adapter_close_closes_the_client():
    adapter, client = _adapter()
    adapter.close()
    assert client.state.closed is True


# =============================================================================
# End to end: registered build -> shared builder -> real run_episode loop
# =============================================================================


RecordingMockRobot = claude_tests.RecordingMockRobot  # one copy of the helper, shared with 15-06


def _run_registered(monkeypatch, tmp_path, mock_calibration_file, client, ticks=20):
    """Build `--backend mllm-gpt` through BACKEND_REGISTRY with fake hardware and run `ticks` ticks."""
    RecordingMockRobot.instances = []
    monkeypatch.setattr(base_mllm_build, "SO101Follower", RecordingMockRobot)
    monkeypatch.setattr(base_mllm_build, "StereoSplitCamera", lambda index=1: FakeStereoCamera())
    monkeypatch.setattr(base_mllm_build.cv2, "VideoCapture", lambda index: FakeWristCap())
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    made = []

    def fake_client_factory(api_key, timeout_s):
        made.append((api_key, timeout_s))
        return client

    monkeypatch.setattr(mllm_gpt, "_make_openai_client", fake_client_factory)
    out = tmp_path / "out"
    args = SimpleNamespace(port="/dev/null", robot_id="soarm_follower_02", instruction="pick up the pen",
                           stereo_camera_index=1, out=out, depth_endpoint=None, depth_calibration=None)
    limits = action_contract.load_joint_limits_deg(mock_calibration_file)

    handles = BACKEND_REGISTRY[BACKEND_NAME](args, dict(CAMERA_NAMES), limits)
    (robot,) = RecordingMockRobot.instances
    assert handles.robot is robot
    assert made == [(SECRET, 900)]

    with IOLogger(out, {0: "wrist"}) as io_logger:
        run_episode(robot, {0: FakeCaptureCamera()}, {0: "wrist"}, io_logger, handles.action_source,
                    "pick up the pen", ticks, 100.0, limits)
    handles.action_source.close()
    lines = (out / "episode.jsonl").read_text().strip().splitlines()
    return out, [json.loads(line) for line in lines]


def _assert_secret_absent(out, capsys):
    scanned = 0
    for path in out.rglob("*"):
        if path.is_file():
            scanned += 1
            assert SECRET.encode() not in path.read_bytes(), path
    assert scanned > 0
    assert SECRET not in capsys.readouterr().out


def test_end_to_end_one_chunk_through_run_episode(monkeypatch, tmp_path, mock_calibration_file, capsys):
    usage = _usage(input_tokens=1200, output_tokens=300, cached_tokens=200, reasoning_tokens=120)
    plan_with_secret = make_chunk_text(20, plan=f"Reach the pen (never echo {SECRET})")
    client = _client([make_response(text=plan_with_secret, summary=f"summary mentions {SECRET}", usage=usage)])

    out, records = _run_registered(monkeypatch, tmp_path, mock_calibration_file, client)

    assert len(client.state.calls) == 1  # one paid call across 20 ticks
    assert len(records) == 20
    call = records[0]["action_details"]["mllm_call"]
    assert call["provider"] == "openai"
    assert call["model_requested"] == MODEL
    assert call["model_returned"] == f"{MODEL}-2026-09-15"
    assert call["request_id"] == "req_fake_gpt_123"
    assert call["sdk"] == {"name": "openai", "version": "3.23.0"}
    assert call["usage"]["reasoning_tokens"] == 120
    expected_cost = ((1200 - 200) * 10.0 + 200 * 1.0 + 300 * 50.0) / 1e6
    assert math.isclose(call["cost_usd_est"], expected_cost)
    assert call["sidecar_dir"] == "mllm_calls/0000"
    assert call["settings"]["omitted"].keys() == {"temperature", "top_p"}
    assert call["settings"]["applied"]["store"] is False
    assert call["trace"]["native_thinking_kind"] == "summary"
    assert call["stop"]["reason"] == STOP_COMPLETE
    assert records[0]["model_version"] == f"{BACKEND_NAME}@mllm-chunk"
    assert all("mllm_call" not in r["action_details"] for r in records[1:])
    assert all(r["action_details"].get("chunk_ref") is not None for r in records[1:])
    sidecar = out / "mllm_calls" / "0000"
    assert {"prompt.txt", "request_meta.json", "response_raw.json", "thinking.txt"} <= {
        p.name for p in sidecar.iterdir() if p.is_file()}
    sent = client.state.calls[0]
    assert sent["store"] is False and "temperature" not in sent and "top_p" not in sent
    _assert_secret_absent(out, capsys)


@pytest.mark.parametrize("case,reason", [
    ("exception", "provider-error"),
    ("incomplete", "truncated"),
    ("refusal", "refusal"),
])
def test_failed_call_holds_the_whole_chunk_and_never_leaks_the_key(
        monkeypatch, tmp_path, mock_calibration_file, capsys, case, reason):
    if case == "exception":
        client = _client(exc=RuntimeError(f"401 Incorrect API key provided: {SECRET}"))
    elif case == "incomplete":
        client = _client([make_response(text='{"actions": [', summary=[], status="incomplete",
                                         incomplete_reason="max_output_tokens")])
    else:
        client = _client([make_response(refusal=f"I can't help with that ({SECRET}).")])

    out, records = _run_registered(monkeypatch, tmp_path, mock_calibration_file, client)

    assert len(client.state.calls) == 1  # one call per chunk_len ticks, never retried
    assert len(records) == 20
    assert [r["model_version"] for r in records] == [f"{BACKEND_NAME}@{reason}-holding-position"] * 20
    call = records[0]["action_details"]["mllm_call"]
    assert call["parse"]["ok"] is False
    if case == "incomplete":
        assert call["stop"]["incomplete_reason"] == "max_output_tokens"
    _assert_secret_absent(out, capsys)


# =============================================================================
# Cross-provider identity (PIPE-02, success criterion 1)
# =============================================================================


def _claude_sequence(kwargs):
    seq = []
    for block in kwargs["messages"][0]["content"]:
        if block["type"] == "text":
            seq.append(("text", block["text"]))
        else:
            assert block["type"] == "image"
            seq.append(("image", base64.b64decode(block["source"]["data"])))
    return seq


def _gpt_sequence(kwargs):
    seq = []
    for part in kwargs["input"][0]["content"]:
        if part["type"] == "input_text":
            seq.append(("text", part["text"]))
        else:
            assert part["type"] == "input_image"
            seq.append(("image", base64.b64decode(part["image_url"].split(",", 1)[1])))
    return seq


def _both_providers_capture(request):
    claude_client = claude_tests._client([claude_tests.make_message(text=make_chunk_text(20))])
    gpt_client = _client([make_response(text=make_chunk_text(20))])
    mllm_claude.AnthropicAdapter(claude_client, load_mllm_settings("mllm_claude")).call(request)
    mllm_gpt.OpenAIResponsesAdapter(gpt_client, _config()).call(request)
    (claude_kwargs,) = claude_client.state.calls
    (gpt_kwargs,) = gpt_client.state.calls
    return claude_kwargs, gpt_kwargs


def test_payload_order_matches_the_claude_adapter_element_for_element():
    request = _request(roles=IMAGE_ORDER)
    claude_kwargs, gpt_kwargs = _both_providers_capture(request)

    claude_seq, gpt_seq = _claude_sequence(claude_kwargs), _gpt_sequence(gpt_kwargs)

    assert gpt_seq == claude_seq
    expected = []
    for role, data in request.images:
        expected += [("text", f"Image: {role}"), ("image", data)]
    expected.append(("text", request.user_text))
    assert gpt_seq == expected
    assert gpt_seq[-1] == ("text", request.user_text)


def test_the_same_system_prompt_and_schema_objects_reach_both_providers_unchanged():
    request = _request(roles=IMAGE_ORDER)
    claude_kwargs, gpt_kwargs = _both_providers_capture(request)

    assert claude_kwargs["system"] is request.system_prompt
    assert gpt_kwargs["instructions"] is request.system_prompt
    assert claude_kwargs["output_config"]["format"]["schema"] is request.schema
    assert gpt_kwargs["text"]["format"]["schema"] is request.schema
    assert gpt_kwargs["text"]["format"]["name"] == request.schema_name


def test_the_shared_base_builds_identical_prompt_and_schema_for_both_providers():
    """Through MllmActionSourceBase (real prompt + schema), both vendors receive the same text and schema."""
    claude_client = claude_tests._client([claude_tests.make_message(text=make_chunk_text(2))])
    gpt_client = _client([make_response(text=make_chunk_text(2))])
    limits = {joint: (-180.0, 180.0) for joint in JOINT_ORDER}
    joint_state = {joint: 3.0 for joint in JOINT_ORDER}
    for config_name, adapter in (
            ("mllm_claude", mllm_claude.AnthropicAdapter(claude_client, load_mllm_settings("mllm_claude"))),
            (CONFIG_NAME, mllm_gpt.OpenAIResponsesAdapter(gpt_client, _config()))):
        settings = load_mllm_settings(config_name)
        settings["chunk_len"] = 2
        source = MllmActionSourceBase(adapter, FakeCollector(include_depth=True), settings, limits,
                                      "cross-check", secrets=[SECRET])
        source.get_action(joint_state, "pick up the pen")

    (claude_kwargs,) = claude_client.state.calls
    (gpt_kwargs,) = gpt_client.state.calls
    assert gpt_kwargs["instructions"] == claude_kwargs["system"]
    assert gpt_kwargs["text"]["format"]["schema"] == claude_kwargs["output_config"]["format"]["schema"]
    assert _gpt_sequence(gpt_kwargs) == _claude_sequence(claude_kwargs)
    assert [item for item in _gpt_sequence(gpt_kwargs) if item[0] == "text"][:-1] == [
        ("text", f"Image: {role}") for role in IMAGE_ORDER]
