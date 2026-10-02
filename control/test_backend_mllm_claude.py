"""Tests for `--backend mllm-claude` (Plan 15-06, MLLM-02, PIPE-02, MLLM-04).

The Claude adapter (`vla_bridge.backends.mllm_claude`), its registration and
its shipped config. No test calls the real Anthropic API: the SDK client is a
fake exposing `messages.stream(**kwargs)` as a context manager, and the one
test that builds a real `anthropic.Anthropic` client never sends a request.
The request kwargs are bound to the INSTALLED SDK's `Messages.stream`
signature so drift from anthropic==1.11.0 fails here, not on a paid call.
"""

import ast
import base64
import inspect
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

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
from vla_bridge.backends import BACKEND_REGISTRY, GRPC_BRIDGE_BACKENDS, base_mllm, base_mllm_build
from vla_bridge.backends import mllm_claude
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

BACKEND_NAME = "mllm-claude"
CONFIG_NAME = "mllm_claude"
MODEL = "claude-opus-5-5"  # user decision 2026-10-02 (was claude-fable-5-1)
SECRET = "sk-ant-api03-TEST-SECRET-mllm-claude-0123456789abcdef"
CAMERA_NAMES = {5: "wrist", 3: "overhead"}
MODULE_PATH = Path(mllm_claude.__file__)
SHIPPED_CONFIG_PATH = backends.BACKEND_CONFIG_DIR / f"{CONFIG_NAME}.json"
EXPECTED_KWARGS = {"model", "max_tokens", "system", "messages", "thinking", "output_config"}


# --- fake Anthropic SDK client -----------------------------------------------------------------


def _usage(input_tokens=1000, output_tokens=300, cache_read=200, cache_creation=0, thinking_tokens=120):
    details = None if thinking_tokens is None else SimpleNamespace(thinking_tokens=thinking_tokens)
    raw = {"input_tokens": input_tokens, "output_tokens": output_tokens,
           "cache_read_input_tokens": cache_read, "cache_creation_input_tokens": cache_creation,
           "output_tokens_details": None if thinking_tokens is None else {"thinking_tokens": thinking_tokens}}
    return SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=cache_creation,
        output_tokens_details=details,
        model_dump=lambda mode="python": dict(raw),
    )


class FakeMessage(SimpleNamespace):
    def model_dump(self, mode="python"):
        return {
            "id": self.id,
            "type": "message",
            "model": self.model,
            "stop_reason": self.stop_reason,
            "content": [vars(block) for block in self.content],
            "dumped_mode": mode,
        }


def make_message(text=None, thinking=None, stop_reason="end_turn", stop_details=None, usage=None,
                 request_id="req_fake_123", blocks=None):
    if blocks is None:
        blocks = []
        if thinking is not None:
            blocks.append(SimpleNamespace(type="thinking", thinking=thinking, signature="sig"))
        if text is not None:
            blocks.append(SimpleNamespace(type="text", text=text))
    message = FakeMessage(
        id="msg_fake_1",
        model=MODEL,
        content=blocks,
        stop_reason=stop_reason,
        stop_details=stop_details,
        usage=usage if usage is not None else _usage(),
    )
    message._request_id = request_id
    return message


class FakeStream:
    def __init__(self, message, request_id=None):
        self.message = message
        self.request_id = request_id
        self.entered = False
        self.exited = False

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, *exc):
        self.exited = True
        return False

    def get_final_message(self):
        return self.message


class FakeMessages:
    def __init__(self, owner):
        self.owner = owner

    def stream(self, **kwargs):
        self.owner.calls.append(kwargs)
        if self.owner.exc is not None:
            raise self.owner.exc
        if not self.owner.messages:
            raise AssertionError("fake Anthropic client: unexpected extra (paid) call")
        stream = FakeStream(self.owner.messages.pop(0), self.owner.stream_request_id)
        self.owner.streams.append(stream)
        return stream


def _client(messages=(), exc=None, stream_request_id=None):
    """A fake client whose `.messages.stream(...)` replays `messages`."""
    queue = list(messages)
    state = SimpleNamespace(calls=[], streams=[], closed=False, exc=exc, messages=queue,
                            stream_request_id=stream_request_id)
    client = SimpleNamespace(messages=FakeMessages(state), state=state)
    client.close = lambda: setattr(state, "closed", True)
    return client


def _config(**overrides):
    config = load_mllm_settings(CONFIG_NAME)
    config.update(overrides)
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


def _adapter(messages=(), **kwargs):
    client = _client(messages, **kwargs)
    return mllm_claude.AnthropicAdapter(client, _config()), client


# =============================================================================
# Registration and module hygiene
# =============================================================================


def test_mllm_claude_is_registered_as_an_mllm_backend_not_grpc():
    assert BACKEND_REGISTRY[BACKEND_NAME] is mllm_claude.build
    spec = base_mllm.MLLM_BACKENDS[BACKEND_NAME]
    assert spec.config_name == CONFIG_NAME
    assert spec.adapter_factory is mllm_claude.make_adapter
    assert BACKEND_NAME not in GRPC_BRIDGE_BACKENDS
    assert (mllm_claude.BACKEND_NAME, mllm_claude.CONFIG_NAME) == (BACKEND_NAME, CONFIG_NAME)


def test_module_imports_anthropic_only_inside_the_client_factory():
    tree = ast.parse(MODULE_PATH.read_text())
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert all(alias.name.split(".")[0] != "anthropic" for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] != "anthropic"
    owners = []
    for func in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        for node in ast.walk(func):
            if isinstance(node, ast.Import) and any(a.name == "anthropic" for a in node.names):
                owners.append(func.name)
    assert owners == ["_make_anthropic_client"]


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

    assert settings["provider"] == "anthropic"
    assert settings["model"] == MODEL
    assert settings["api_key_env"] == "ANTHROPIC_API_KEY"
    assert settings["effort"] == "xhigh"
    assert settings["max_tokens"] == 64000
    assert settings["thinking"] == {"type": "adaptive", "display": "summarized"}
    prices = settings["prices"]
    assert (prices["input_per_mtok"], prices["output_per_mtok"], prices["cached_input_per_mtok"]) == (4.0, 20.0, 0.20)
    assert prices["verified_on"] == "2026-10-02"
    assert prices["source"]
    caps = settings["capabilities"]
    assert caps["sampling_supported"] is False
    assert caps["effort_levels"] == ["low", "medium", "high", "xhigh", "max"]
    assert caps["summary_supported"] is True
    assert "30-day" in settings["data_retention_note"]
    assert "2026" in settings["model_verified"] and MODEL in settings["model_verified"]
    assert settings["requirement"] == "MLLM-02"
    assert settings["timeout_s"] == 900  # inherited from mllm_defaults.json


def test_shipped_config_records_the_opus_5_5_switch_and_its_sources():
    raw = json.loads(SHIPPED_CONFIG_PATH.read_text())
    assert raw["prices"]["source"].startswith("https://platform.claude.com/docs/en/models/opus-5-5/overview")
    notes = raw["notes"]
    assert "Switched from claude-fable-5-1 to claude-opus-5-5 by user decision 2026-10-02" in notes
    assert "15-LIVE-RESULTS.md section 2" in notes
    assert "effort kept xhigh explicitly (Opus 5.5 default is medium)" in notes
    assert "re-smoke required" in notes
    assert "claude-fable-5-1 rejects" not in notes
    assert "whats-new-opus-5-5" in raw["model_verified"]
    # Fable 5.1's Covered Model retention requirement is not documented for Opus 5.5.
    assert "claude-fable-5-1 is" not in raw["data_retention_note"]
    assert "claude-opus-5-5" in raw["data_retention_note"]


def test_shipped_config_holds_a_key_name_never_a_key():
    raw = SHIPPED_CONFIG_PATH.read_text()
    config = json.loads(raw)
    assert config["api_key_env"] == "ANTHROPIC_API_KEY"
    assert "sk-ant" not in raw


# =============================================================================
# Request shape
# =============================================================================


def test_request_kwargs_are_exactly_the_verified_set_in_canonical_order():
    adapter, client = _adapter([make_message(text=make_chunk_text(20))])
    request = _request()

    adapter.call(request)

    (kwargs,) = client.state.calls
    assert set(kwargs) == EXPECTED_KWARGS
    assert kwargs["model"] == MODEL
    assert kwargs["max_tokens"] == 64000
    assert kwargs["system"] == request.system_prompt
    assert kwargs["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert kwargs["output_config"] == {"effort": "xhigh",
                                       "format": {"type": "json_schema", "schema": request.schema}}
    (message,) = kwargs["messages"]
    assert message["role"] == "user"
    content = message["content"]
    expected = []
    for role, data in request.images:
        expected.append({"type": "text", "text": f"Image: {role}"})
        expected.append({"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                     "data": base64.b64encode(data).decode("ascii")}})
    expected.append({"type": "text", "text": request.user_text})
    assert content == expected
    stream = client.state.streams[0]
    assert stream.entered and stream.exited


def test_request_sends_no_sampling_tools_or_prefill():
    adapter, client = _adapter([make_message(text=make_chunk_text(20))])
    adapter.call(_request())
    (kwargs,) = client.state.calls
    for forbidden in ("temperature", "top_p", "top_k", "tools", "tool_choice", "output_format"):
        assert forbidden not in kwargs
    assert [m["role"] for m in kwargs["messages"]] == ["user"]  # no assistant prefill


def test_request_kwargs_bind_to_the_installed_sdk_stream_signature():
    from anthropic.resources.messages import Messages

    adapter, client = _adapter([make_message(text=make_chunk_text(20))])
    adapter.call(_request())
    (kwargs,) = client.state.calls
    inspect.signature(Messages.stream).bind(object(), **kwargs)  # raises TypeError on drift


def test_request_labels_every_image_including_depth_and_jpeg_media_type():
    adapter, client = _adapter([make_message(text=make_chunk_text(20))])
    adapter.call(_request(roles=IMAGE_ORDER, media_type="image/jpeg"))
    content = client.state.calls[0]["messages"][0]["content"]
    labels = [block["text"] for block in content[:-1] if block["type"] == "text"]
    assert labels == [f"Image: {role}" for role in IMAGE_ORDER]
    images = [block for block in content if block["type"] == "image"]
    assert len(images) == 4 and {img["source"]["media_type"] for img in images} == {"image/jpeg"}


def test_effort_none_is_left_out_of_output_config():
    client = _client([make_message(text=make_chunk_text(20))])
    adapter = mllm_claude.AnthropicAdapter(client, _config(effort=None))
    adapter.call(_request())
    assert "effort" not in client.state.calls[0]["output_config"]


@pytest.mark.parametrize("thinking", [None, "adaptive", {"type": "enabled"}, {"display": "summarized"},
                                      {"type": "adaptive", "display": "full"}])
def test_invalid_thinking_config_is_rejected_at_build_time(thinking):
    with pytest.raises(SystemExit, match="thinking"):
        mllm_claude.AnthropicAdapter(_client(), _config(thinking=thinking))


# =============================================================================
# Result mapping
# =============================================================================


def test_result_maps_text_summary_thinking_ids_and_raw_payload():
    message = make_message(text='{"actions": []}', thinking="I will look at the wrist image first.")
    adapter, _client_obj = _adapter([message])

    result = adapter.call(_request())

    assert result.text == '{"actions": []}'
    assert result.native_thinking_text == "I will look at the wrist image first."
    assert result.native_thinking_kind == "summary"
    assert result.request_id == "req_fake_123"
    assert result.response_id == "msg_fake_1"
    assert result.model_returned == MODEL
    assert result.raw_payload == message.model_dump(mode="json")
    assert result.raw_payload["dumped_mode"] == "json"
    assert result.stop == {"reason": STOP_COMPLETE, "refusal_category": None, "incomplete_reason": None,
                           "provider_stop_reason": "end_turn"}


def test_result_with_empty_thinking_is_omitted_and_without_thinking_is_none():
    adapter, _c = _adapter([make_message(text="{}", thinking=""), make_message(text="{}")])
    first = adapter.call(_request())
    second = adapter.call(_request())
    assert (first.native_thinking_kind, first.native_thinking_text) == ("omitted", None)
    assert (second.native_thinking_kind, second.native_thinking_text) == ("none", None)


def test_redacted_thinking_only_is_omitted_and_multiple_text_blocks_join():
    blocks = [SimpleNamespace(type="redacted_thinking", data="opaque"),
              SimpleNamespace(type="text", text='{"a": '), SimpleNamespace(type="text", text="1}")]
    adapter, _c = _adapter([make_message(blocks=blocks)])
    result = adapter.call(_request())
    assert result.native_thinking_kind == "omitted"
    assert result.text == '{"a": 1}'


def test_result_without_text_blocks_has_text_none():
    adapter, _c = _adapter([make_message(thinking="only thinking")])
    assert adapter.call(_request()).text is None


def test_usage_is_normalised_with_cache_tokens_and_thinking_tokens():
    usage = _usage(input_tokens=1000, output_tokens=300, cache_read=200, cache_creation=50, thinking_tokens=120)
    adapter, _c = _adapter([make_message(text="{}", usage=usage)])
    result = adapter.call(_request())
    assert result.usage["input_tokens"] == 1250
    assert result.usage["cached_input_tokens"] == 200
    assert result.usage["output_tokens"] == 300
    assert result.usage["reasoning_tokens"] == 120
    assert result.usage["raw"]["cache_creation_input_tokens"] == 50
    json.dumps(result.usage)  # plain JSON


def test_usage_without_thinking_details_has_reasoning_none():
    usage = _usage(cache_read=None, cache_creation=None, thinking_tokens=None)
    adapter, _c = _adapter([make_message(text="{}", usage=usage)])
    result = adapter.call(_request())
    assert result.usage["reasoning_tokens"] is None
    assert result.usage["input_tokens"] == 1000
    assert result.usage["cached_input_tokens"] is None


@pytest.mark.parametrize("stop_reason,expected", [
    ("end_turn", STOP_COMPLETE),
    ("refusal", STOP_REFUSAL),
    ("max_tokens", STOP_TRUNCATED),
    ("pause_turn", STOP_OTHER),
    ("tool_use", STOP_OTHER),
    (None, STOP_OTHER),
])
def test_stop_reason_mapping(stop_reason, expected):
    adapter, _c = _adapter([make_message(text="{}", stop_reason=stop_reason)])
    stop = adapter.call(_request()).stop
    assert stop["reason"] == expected
    assert stop["provider_stop_reason"] == stop_reason
    assert stop["incomplete_reason"] == (None if expected == STOP_COMPLETE else stop_reason)


def test_refusal_keeps_the_stop_details_category():
    details = SimpleNamespace(category="reasoning_extraction", explanation="asks for reasoning", type="refusal")
    adapter, _c = _adapter([make_message(text="", stop_reason="refusal", stop_details=details)])
    stop = adapter.call(_request()).stop
    assert stop["reason"] == STOP_REFUSAL
    assert stop["refusal_category"] == "reasoning_extraction"


def test_request_id_falls_back_to_the_stream_header_value():
    message = make_message(text="{}", request_id=None)
    adapter, _c = _adapter([message], stream_request_id="req_from_header")
    assert adapter.call(_request()).request_id == "req_from_header"


# =============================================================================
# Settings summary (D-20 / D-21)
# =============================================================================


def test_settings_summary_logs_requested_applied_and_never_sent_sampling():
    adapter, _c = _adapter()
    summary = adapter.settings_summary
    thinking = {"type": "adaptive", "display": "summarized"}
    assert summary["requested"] == {"temperature": 1.0, "top_p": 0.95, "effort": "xhigh",
                                    "max_tokens": 64000, "thinking": thinking}
    assert summary["applied"] == {"effort": "xhigh", "max_tokens": 64000, "thinking": thinking}
    assert set(summary["omitted"]) == {"temperature", "top_p", "top_k"}
    for reason in summary["omitted"].values():
        # Never claims a provider HTTP 400 for sampling (not stated in the Opus 5.5 docs).
        assert reason.startswith("not sent:"), reason
        assert "400" not in reason and "rejects" not in reason, reason
        assert "never sends sampling parameters" in reason
    json.dumps(summary)
    assert adapter.provider == "anthropic"
    assert adapter.model == MODEL
    assert adapter.sdk == {"name": "anthropic", "version": "1.11.0"}


# =============================================================================
# Real client construction and secret handling
# =============================================================================


def test_make_anthropic_client_builds_a_real_client_without_retries():
    import anthropic

    client = mllm_claude._make_anthropic_client(SECRET, 123.0)
    assert isinstance(client, anthropic.Anthropic)
    assert client.max_retries == 0
    assert client.timeout == 123.0
    client.close()


def test_make_adapter_uses_the_configured_timeout_and_hides_the_key():
    adapter = mllm_claude.make_adapter(SECRET, _config())
    try:
        assert adapter._client.max_retries == 0
        assert adapter._client.timeout == 900
        assert SECRET not in repr(adapter)
        assert SECRET not in json.dumps(adapter.settings_summary)
        assert SECRET not in json.dumps(adapter.sdk)
    finally:
        adapter.close()


def test_adapter_close_closes_the_client():
    adapter, client = _adapter()
    adapter.close()
    assert client.state.closed is True


def _base_source(adapter, chunk_len=2):
    settings = load_mllm_settings(CONFIG_NAME)
    settings["chunk_len"] = chunk_len
    limits = {joint: (-180.0, 180.0) for joint in JOINT_ORDER}
    return MllmActionSourceBase(adapter, FakeCollector(), settings, limits, BACKEND_NAME, secrets=[SECRET])


def test_client_exception_containing_the_key_is_redacted_by_the_base():
    client = _client(exc=RuntimeError(f"401 invalid x-api-key {SECRET}"))
    source = _base_source(mllm_claude.AnthropicAdapter(client, _config()))
    joint_state = {joint: 3.0 for joint in JOINT_ORDER}

    action, version = source.get_action(joint_state, "pick up the pen")

    assert action == joint_state
    assert version == f"{BACKEND_NAME}@provider-error-holding-position"
    dumped = json.dumps(source.last_details)
    assert SECRET not in dumped
    assert "<redacted>" in source.last_details["error"]


def test_refusal_through_the_base_holds_and_records_the_category():
    details = SimpleNamespace(category="reasoning_extraction", explanation=None, type="refusal")
    client = _client([make_message(text="", stop_reason="refusal", stop_details=details)])
    source = _base_source(mllm_claude.AnthropicAdapter(client, _config()))
    joint_state = {joint: 3.0 for joint in JOINT_ORDER}

    _action, version = source.get_action(joint_state, "pick up the pen")

    assert version == f"{BACKEND_NAME}@refusal-holding-position"
    call = source.last_details["mllm_call"]
    assert call["stop"]["refusal_category"] == "reasoning_extraction"
    assert call["parse"]["ok"] is False


def test_max_tokens_truncation_through_the_base_holds_the_whole_chunk():
    client = _client([make_message(text='{"actions": [', stop_reason="max_tokens")])
    source = _base_source(mllm_claude.AnthropicAdapter(client, _config()), chunk_len=2)
    joint_state = {joint: 3.0 for joint in JOINT_ORDER}
    versions = [source.get_action(joint_state, "x")[1] for _ in range(2)]
    assert versions == [f"{BACKEND_NAME}@truncated-holding-position"] * 2
    assert len(client.state.calls) == 1


# =============================================================================
# End to end: registered build -> real run_episode loop
# =============================================================================


class RecordingMockRobot(MockRobot):
    instances = []

    def __init__(self, config):
        super().__init__()
        self.config = config
        RecordingMockRobot.instances.append(self)


def test_end_to_end_one_chunk_through_run_episode(monkeypatch, tmp_path, mock_calibration_file, capsys):
    RecordingMockRobot.instances = []
    monkeypatch.setattr(base_mllm_build, "SO101Follower", RecordingMockRobot)
    monkeypatch.setattr(base_mllm_build, "StereoSplitCamera", lambda index=1: FakeStereoCamera())
    monkeypatch.setattr(base_mllm_build.cv2, "VideoCapture", lambda index: FakeWristCap())
    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)
    usage = _usage(input_tokens=1000, output_tokens=300, cache_read=200, cache_creation=0, thinking_tokens=120)
    plan_with_secret = make_chunk_text(20, plan=f"Reach the pen (never echo {SECRET})")
    client = _client([make_message(text=plan_with_secret, thinking=f"thinking mentions {SECRET}", usage=usage)])
    made = []

    def fake_client_factory(api_key, timeout_s):
        made.append((api_key, timeout_s))
        return client

    monkeypatch.setattr(mllm_claude, "_make_anthropic_client", fake_client_factory)
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
                    "pick up the pen", 20, 100.0, limits)
    handles.action_source.close()

    assert len(client.state.calls) == 1
    lines = (out / "episode.jsonl").read_text().strip().splitlines()
    assert len(lines) == 20
    records = [json.loads(line) for line in lines]
    call = records[0]["action_details"]["mllm_call"]
    assert call["provider"] == "anthropic"
    assert call["model_requested"] == MODEL
    assert call["model_returned"] == MODEL
    assert call["request_id"] == "req_fake_123"
    assert call["sdk"] == {"name": "anthropic", "version": "1.11.0"}
    assert call["usage"]["reasoning_tokens"] == 120
    expected_cost = ((1200 - 200) * 4.0 + 200 * 0.20 + 300 * 20.0) / 1e6
    assert math.isclose(call["cost_usd_est"], expected_cost)
    assert call["sidecar_dir"] == "mllm_calls/0000"
    assert call["settings"]["omitted"].keys() == {"temperature", "top_p", "top_k"}
    assert call["trace"]["native_thinking_kind"] == "summary"
    assert records[0]["model_version"] == f"{BACKEND_NAME}@mllm-chunk"
    assert all("mllm_call" not in r["action_details"] for r in records[1:])
    sidecar = out / "mllm_calls" / "0000"
    assert {"prompt.txt", "request_meta.json", "response_raw.json", "thinking.txt"} <= {
        p.name for p in sidecar.iterdir() if p.is_file()}

    for path in out.rglob("*"):
        if path.is_file():
            assert SECRET.encode() not in path.read_bytes(), path
    assert SECRET not in capsys.readouterr().out
