"""Tests for the shared MLLM chunk-drain action source (Phase 15, PIPE-02, MLLM-04..07).

`vla_bridge.backends.base_mllm` (MllmActionSourceBase, parse_chunk,
load_mllm_settings, register_mllm_backend) driven by the scripted fakes in
`fake_providers.py`.

The `test_slice_` tests run the REAL `run_episode()` loop with `MockRobot`,
the real `IOLogger` and the UNMODIFIED `safety_validator`, and read the
resulting `episode.jsonl`. Nothing here touches real hardware, a real camera
or the network: the provider is a `ScriptedAdapter` and the inputs come from
a `FakeCollector`.
"""

import json
import re

import pytest

from fake_providers import (
    FakeCaptureCamera,
    FakeCollector,
    ScriptedAdapter,
    make_chunk_doc,
)
from run_vla_episode import run_episode
from vla_bridge import action_contract
from vla_bridge.action_contract import JOINT_ORDER
from vla_bridge.backends.base_mllm import MllmActionSourceBase, load_mllm_settings
from vla_bridge.io_logger import IOLogger

LABEL = "mllm-fake"
INSTRUCTION = "Pick up the pen"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

PLAN_1 = "Lower the arm toward the pen, keeping the gripper open."
PLAN_2 = "Close the gripper around the pen and lift it slightly."


def _settings(chunk_len=3, **overrides):
    settings = load_mllm_settings()
    settings["chunk_len"] = chunk_len
    settings.update(overrides)
    return settings


def _limits(mock_calibration_file):
    return action_contract.load_joint_limits_deg(mock_calibration_file)


def _source(adapter, mock_calibration_file, collector=None, chunk_len=3, **kwargs):
    return MllmActionSourceBase(
        adapter,
        collector if collector is not None else FakeCollector(),
        _settings(chunk_len),
        _limits(mock_calibration_file),
        LABEL,
        **kwargs,
    )


def _run(tmp_path, mock_robot, source, limits, max_steps):
    with IOLogger(tmp_path, {0: "wrist"}) as io_logger:
        run_episode(
            mock_robot,
            {0: FakeCaptureCamera()},
            {0: "wrist"},
            io_logger,
            source,
            INSTRUCTION,
            max_steps,
            100.0,
            limits,
        )
    lines = (tmp_path / "episode.jsonl").read_text().strip().splitlines()
    return [json.loads(line) for line in lines]


def _two_chunk_episode(tmp_path, mock_robot, mock_calibration_file):
    for joint in JOINT_ORDER:
        mock_robot._positions[joint] = 5.0
    doc1 = make_chunk_doc(3, plan=PLAN_1, base=5.0, step=0.5)
    doc2 = make_chunk_doc(3, plan=PLAN_2, base=7.0, step=0.5)
    adapter = ScriptedAdapter([json.dumps(doc1), json.dumps(doc2)])
    limits = _limits(mock_calibration_file)
    source = MllmActionSourceBase(adapter, FakeCollector(), _settings(3), limits, LABEL)
    records = _run(tmp_path, mock_robot, source, limits, 6)
    return adapter, source, records, (doc1, doc2)


# --- The end-to-end slice (D-00d, D-08, D-12) ---------------------------------


def test_slice_e2e_chunk_drain_through_real_run_episode(tmp_path, mock_robot, mock_calibration_file):
    adapter, _source_obj, records, (doc1, doc2) = _two_chunk_episode(
        tmp_path, mock_robot, mock_calibration_file
    )

    assert adapter.calls == 2  # once per chunk_len ticks, never per tick
    assert len(records) == 6
    scripted_steps = doc1["actions"] + doc2["actions"]
    for i, record in enumerate(records):
        details = record["action_details"]
        if i in (0, 3):
            assert details["mllm_call"]["chunk_id"] == i // 3
        else:
            assert "mllm_call" not in details
        assert details["chunk_ref"] == {"chunk_id": i // 3, "step_index": i % 3, "chunk_len": 3}
        assert record["model_version"] == "mllm-fake@mllm-chunk"
        assert record["raw_model_output"] == scripted_steps[i]
        assert record["validated_action"] == record["raw_model_output"]  # no clamp
        assert record["executed_action"] == record["validated_action"]
        assert record["validator_flags"] == []
        assert "policy_observation_frames" not in details


def test_slice_trace_written_once_per_chunk(tmp_path, mock_robot, mock_calibration_file):
    _adapter, _source_obj, records, (doc1, _doc2) = _two_chunk_episode(
        tmp_path, mock_robot, mock_calibration_file
    )
    call = records[0]["action_details"]["mllm_call"]

    trace = call["trace"]
    assert trace["task_restatement"] == doc1["task_restatement"]
    assert trace["plan"] == doc1["plan"]
    assert trace["scene_claims"] == doc1["scene_claims"]
    for claim in trace["scene_claims"]:
        assert set(claim) == {"claim", "source", "confidence"}
    assert trace["task_complete"] is False

    assert call["parse"]["ok"] is True
    assert call["parse"]["n_valid"] == 3
    assert call["stop"]["reason"] == "complete"
    assert call["usage"]["input_tokens"] == 100
    assert call["prompt_template_version"]
    assert SHA256_RE.match(call["prompt_sha256"])
    assert SHA256_RE.match(call["schema_sha256"])
    assert call["include_urdf"] is False
    assert call["chunk_len"] == 3

    for i in (1, 2, 4, 5):
        assert "trace" not in json.dumps(records[i]["action_details"])


def test_slice_previous_plan_is_carried_into_next_prompt(tmp_path, mock_robot, mock_calibration_file):
    adapter, _source_obj, _records, (doc1, _doc2) = _two_chunk_episode(
        tmp_path, mock_robot, mock_calibration_file
    )
    first, second = adapter.requests
    assert "none (this is the first call)" in first.user_text
    assert doc1["plan"] not in first.user_text
    assert doc1["plan"] in second.user_text  # D-09: only the previous plan is carried


def test_slice_shared_contract_identical_every_call(tmp_path, mock_robot, mock_calibration_file):
    adapter, _source_obj, _records, _docs = _two_chunk_episode(
        tmp_path, mock_robot, mock_calibration_file
    )
    first, second = adapter.requests
    assert first.system_prompt == second.system_prompt
    assert first.schema == second.schema


def test_slice_last_details_is_json_serialisable_every_tick(mock_calibration_file):
    docs = [make_chunk_doc(3, base=5.0), make_chunk_doc(3, base=7.0)]
    adapter = ScriptedAdapter([json.dumps(d) for d in docs])
    source = _source(adapter, mock_calibration_file)
    joint_state = {joint: 5.0 for joint in JOINT_ORDER}
    for _tick in range(6):
        source.get_action(dict(joint_state), INSTRUCTION)
        json.dumps(source.last_details, allow_nan=False)
        assert "observation" not in source.last_details
    assert adapter.calls == 2


# --- Helpers for the behavior tests ------------------------------------------------

SECRET = "sk-TEST-SECRET-do-not-log-0123456789"


def _js(tick):
    """A distinct, non-zero live joint state per tick (holds must echo THIS one)."""
    return {joint: 10.0 * (tick + 1) + i for i, joint in enumerate(JOINT_ORDER)}


def _drive(source, ticks, start=0):
    outputs = []
    for tick in range(start, start + ticks):
        action, model_version = source.get_action(_js(tick), INSTRUCTION)
        outputs.append((action, model_version, json.loads(json.dumps(source.last_details))))
    return outputs


def _assert_hold(output, tick, reason):
    action, model_version, details = output
    assert action == _js(tick)
    assert model_version == f"{LABEL}@{reason}-holding-position"
    assert details["bridge_flags"] == [f"{reason}-holding-position"]


# --- Malformed steps inside a chunk (D-03, MLLM-07, T-15-01) -------------------------


def _bad_step_text(kind):
    doc = make_chunk_doc(3, base=5.0)
    bad = doc["actions"][1]
    if kind == "missing-key":
        del bad["elbow_flex"]
    elif kind == "string":
        bad["elbow_flex"] = "12.5"
    elif kind == "null":
        bad["elbow_flex"] = None
    elif kind == "bool":
        bad["elbow_flex"] = True
    elif kind == "nan":
        bad["elbow_flex"] = float("nan")  # json.dumps writes the NaN literal
    elif kind == "infinity":
        bad["elbow_flex"] = float("inf")  # json.dumps writes the Infinity literal
    elif kind == "overflow":
        bad["elbow_flex"] = 10**400  # a valid JSON integer too large for float
    elif kind == "short":
        doc["actions"] = doc["actions"][:1]
    elif kind == "extra":
        doc["actions"] = make_chunk_doc(5, base=5.0)["actions"]
    return json.dumps(doc), doc


MALFORMED_KINDS = ["missing-key", "string", "null", "bool", "nan", "infinity", "overflow", "short", "extra"]


@pytest.mark.parametrize("kind", MALFORMED_KINDS)
def test_malformed_step_holds_live_state_for_rest_of_chunk_without_retry(kind, mock_calibration_file):
    text, doc = _bad_step_text(kind)
    adapter = ScriptedAdapter([text])
    source = _source(adapter, mock_calibration_file)
    out = _drive(source, 3)

    assert adapter.calls == 1  # never a retry call for a malformed chunk
    first_action, first_version, first_details = out[0]
    assert first_action == doc["actions"][0]
    assert first_version == f"{LABEL}@mllm-chunk"
    record = first_details["mllm_call"]

    if kind == "extra":
        for tick in range(3):
            assert out[tick][0] == doc["actions"][tick]
            assert out[tick][1] == f"{LABEL}@mllm-chunk"
        assert record["parse"]["extra_dropped"] == 2
        assert record["parse"]["n_returned"] == 5
        assert record["parse"]["ok"] is True
        return

    reason = "short-chunk" if kind == "short" else "malformed-step"
    for tick in (1, 2):
        _assert_hold(out[tick], tick, reason)
        assert out[tick][0] != {joint: 0.0 for joint in JOINT_ORDER}
    assert record["parse"]["ok"] is False
    assert record["parse"]["first_bad_index"] == 1
    assert record["parse"]["n_valid"] == 1
    assert record["parse"]["reason"] == reason
    if kind != "short":
        assert "elbow_flex" in record["parse"]["reason_detail"]


def test_malformed_step_raw_item_is_logged_on_the_first_bad_tick(mock_calibration_file):
    text, doc = _bad_step_text("string")
    source = _source(ScriptedAdapter([text]), mock_calibration_file)
    out = _drive(source, 3)
    assert out[1][2]["raw_model_output"] == doc["actions"][1]
    assert "raw_model_output" not in out[2][2]


def test_nan_step_is_logged_without_breaking_strict_json(mock_calibration_file):
    text, _doc = _bad_step_text("nan")
    source = _source(ScriptedAdapter([text]), mock_calibration_file)
    out = _drive(source, 3)
    detail = out[0][2]["mllm_call"]["parse"]["reason_detail"]
    assert detail == "step 1: elbow_flex non-finite"  # names joint and kind, never the value
    json.dumps(source.last_details, allow_nan=False)
    raw = out[1][2]["raw_model_output"]
    assert isinstance(raw["malformed_step_json"], str) and "NaN" in raw["malformed_step_json"]


# --- parse_chunk whole-response failures ------------------------------------------------


@pytest.mark.parametrize(
    "text,reason",
    [
        ("", "empty-response"),
        ("   \n", "empty-response"),
        (None, "empty-response"),
        ("not json at all", "invalid-json"),
        ("[1, 2, 3]", "invalid-json"),
        ('{"plan": "x"}', "missing-actions"),
        ('{"actions": {"shoulder_pan": 1}}', "missing-actions"),
    ],
)
def test_parse_chunk_whole_response_failures(text, reason):
    from vla_bridge.backends.base_mllm import ChunkParseError, parse_chunk

    with pytest.raises(ChunkParseError) as excinfo:
        parse_chunk(text, 3)
    assert excinfo.value.reason == reason
    assert isinstance(excinfo.value, ValueError)


def test_parse_chunk_tolerates_a_json_code_fence_and_missing_top_level_fields():
    from vla_bridge.backends.base_mllm import parse_chunk

    doc = make_chunk_doc(3)
    fenced = parse_chunk("```json\n" + json.dumps(doc) + "\n```", 3)
    assert fenced.fence_stripped is True
    assert fenced.steps == doc["actions"]
    assert fenced.first_bad_index is None and fenced.reason is None

    bare = parse_chunk(json.dumps({"actions": doc["actions"]}), 3)
    assert bare.fence_stripped is False
    assert bare.steps == doc["actions"]  # absent staged fields never void good actions
    assert bare.missing_top_level == ["task_restatement", "plan", "scene_claims", "task_complete"]


# --- Whole-call failures hold N ticks, never per-tick paid retries (T-15-02) --------------


def test_provider_exception_holds_whole_chunk_and_redacts_secret(mock_calibration_file):
    adapter = ScriptedAdapter([
        RuntimeError(f"401 Unauthorized: key {SECRET} rejected"),
        json.dumps(make_chunk_doc(3, base=5.0)),
    ])
    source = _source(adapter, mock_calibration_file, secrets=(SECRET,))
    out = _drive(source, 3)
    assert adapter.calls == 1
    for tick in range(3):
        _assert_hold(out[tick], tick, "provider-error")
    details = out[0][2]
    assert details["error_type"] == "RuntimeError"
    assert SECRET not in details["error"] and "<redacted>" in details["error"]
    record = details["mllm_call"]
    assert record["stop"] == {"reason": "error", "refusal_category": None, "incomplete_reason": None}
    assert record["usage"] is None
    assert record["parse"]["ok"] is False and record["parse"]["reason"] == "provider-error"
    assert SECRET not in json.dumps(out)

    out2 = _drive(source, 1, start=3)  # next call only after chunk_len ticks
    assert adapter.calls == 2
    assert out2[0][1] == f"{LABEL}@mllm-chunk"


@pytest.mark.parametrize(
    "stop_reason,hold_reason",
    [("refusal", "refusal"), ("truncated", "truncated"), ("other", "provider-error")],
)
def test_non_complete_stop_holds_whole_chunk_and_keeps_stop_and_usage(
    stop_reason, hold_reason, mock_calibration_file
):
    from fake_providers import make_provider_result

    result = make_provider_result(json.dumps(make_chunk_doc(3)), stop_reason=stop_reason)
    result.stop["refusal_category"] = "cyber" if stop_reason == "refusal" else None
    adapter = ScriptedAdapter([result])
    source = _source(adapter, mock_calibration_file)
    out = _drive(source, 3)
    assert adapter.calls == 1
    for tick in range(3):
        _assert_hold(out[tick], tick, hold_reason)
    record = out[0][2]["mllm_call"]
    assert record["stop"]["reason"] == stop_reason
    assert record["usage"]["input_tokens"] == 100
    assert record["parse"]["ok"] is False


@pytest.mark.parametrize("text,reason", [("", "empty-response"), ("{oops", "invalid-json"),
                                         ('{"plan": "p"}', "missing-actions")])
def test_unparseable_response_holds_whole_chunk(text, reason, mock_calibration_file):
    adapter = ScriptedAdapter([text])
    source = _source(adapter, mock_calibration_file)
    out = _drive(source, 3)
    assert adapter.calls == 1
    for tick in range(3):
        _assert_hold(out[tick], tick, reason)


def test_keyboard_interrupt_propagates_and_releases_lock(mock_calibration_file):
    adapter = ScriptedAdapter([KeyboardInterrupt()])
    source = _source(adapter, mock_calibration_file)
    with pytest.raises(KeyboardInterrupt):
        source.get_action(_js(0), INSTRUCTION)
    assert source._inflight_lock.acquire(blocking=False)
    source._inflight_lock.release()


def test_collector_failure_is_a_single_hold_without_provider_call(mock_calibration_file):
    adapter = ScriptedAdapter([json.dumps(make_chunk_doc(3))])
    collector = FakeCollector(fail=True)
    source = _source(adapter, mock_calibration_file, collector=collector)
    out = _drive(source, 2)
    for tick in range(2):
        _assert_hold(out[tick], tick, "camera-read-failed")
        assert out[tick][2]["chunk_ref"]["chunk_id"] is None
    assert adapter.calls == 0
    assert collector.calls == 2  # retried next tick: no paid call was spent

    collector.fail = False
    out2 = _drive(source, 1, start=2)
    assert out2[0][2]["mllm_call"]["chunk_id"] == 0  # no chunk id consumed by the failures


def test_inflight_lock_held_gives_hold_without_any_call(mock_calibration_file):
    adapter = ScriptedAdapter([json.dumps(make_chunk_doc(3))])
    collector = FakeCollector()
    source = _source(adapter, mock_calibration_file, collector=collector)
    assert source._inflight_lock.acquire(blocking=False)
    try:
        out = _drive(source, 1)
    finally:
        source._inflight_lock.release()
    _assert_hold(out[0], 0, "request-inflight")
    assert adapter.calls == 0 and collector.calls == 0


# --- Request contents and the per-call record (D-12, D-22, D-24) --------------------------


def test_request_images_follow_image_order_and_record_lists_them(mock_calibration_file):
    from vla_bridge.backends.base_mllm import IMAGE_ORDER
    from vla_bridge.mllm_schema import SCHEMA_NAME

    adapter = ScriptedAdapter([json.dumps(make_chunk_doc(3))])
    source = _source(adapter, mock_calibration_file, collector=FakeCollector(include_depth=True))
    out = _drive(source, 1)
    request = adapter.requests[0]
    assert IMAGE_ORDER == ("wrist", "left", "right", "depth")
    assert [role for role, _data in request.images] == list(IMAGE_ORDER)
    assert request.schema_name == SCHEMA_NAME
    assert request.image_media_type == "image/png"
    assert request.system_prompt == source.system_prompt
    assert request.schema is source.schema
    assert "Depth unavailable this call (fake collector)." in request.user_text

    record = out[0][2]["mllm_call"]
    assert [image["role"] for image in record["images"]] == list(IMAGE_ORDER)
    for image, (_role, data) in zip(record["images"], request.images):
        assert image["bytes"] == len(data)
        assert SHA256_RE.match(image["sha256"])
    assert record["backend"] == LABEL
    assert record["provider"] == "fake"
    assert record["model_requested"] == "fake-model"
    assert record["sdk"] == {"name": "fake", "version": "0"}
    assert record["settings"]["omitted"] == {"temperature": "fake provider"}
    assert record["depth"] == {"status": "not_configured"}
    assert record["capture"] == {"left_image_source_requested": "raw", "left_image_source_sent": "raw"}
    assert set(record["latency_ms"]) == {"depth_fetch", "image_encode", "api", "parse", "total"}
    assert record["step_dt_s"] == 0.785
    assert record["prompt_template_version"] == source.prompt_template_version
    assert record["schema_sha256"] == source.schema_hash
    assert record["prev_chunk_tick_period_s"] is None
    assert record["trace"]["native_thinking_kind"] == "none"


def test_prev_chunk_tick_period_from_injected_clock(mock_calibration_file):
    times = iter([100.0, 100.8, 101.6, 200.0, 200.5, 201.0])
    adapter = ScriptedAdapter([json.dumps(make_chunk_doc(3)), json.dumps(make_chunk_doc(3))])
    source = _source(adapter, mock_calibration_file, clock=lambda: next(times))
    out = _drive(source, 4)
    assert out[0][2]["mllm_call"]["prev_chunk_tick_period_s"] is None
    assert out[3][2]["mllm_call"]["prev_chunk_tick_period_s"] == pytest.approx(0.8)


def test_task_complete_is_true_only_for_boolean_true(mock_calibration_file):
    doc_str = make_chunk_doc(1)
    doc_str["task_complete"] = "true"
    doc_true = make_chunk_doc(1, task_complete=True)
    adapter = ScriptedAdapter([json.dumps(doc_str), json.dumps(doc_true)])
    source = _source(adapter, mock_calibration_file, chunk_len=1)
    out = _drive(source, 2)
    assert out[0][2]["mllm_call"]["trace"]["task_complete"] is False
    assert out[1][2]["mllm_call"]["trace"]["task_complete"] is True


def test_native_thinking_text_is_capped_inline(mock_calibration_file):
    from fake_providers import make_provider_result
    from vla_bridge.backends.base_mllm import MAX_INLINE_THINKING_CHARS

    result = make_provider_result(json.dumps(make_chunk_doc(1)),
                                  native_thinking_text="z" * (MAX_INLINE_THINKING_CHARS + 50),
                                  native_thinking_kind="summary")
    source = _source(ScriptedAdapter([result]), mock_calibration_file, chunk_len=1)
    trace = _drive(source, 1)[0][2]["mllm_call"]["trace"]
    assert len(trace["native_thinking_text"]) == MAX_INLINE_THINKING_CHARS
    assert trace["native_thinking_kind"] == "summary"


def test_recorder_hook_can_extend_the_record_and_its_failure_is_redacted(mock_calibration_file):
    seen = {}

    class Recorder:
        def finalize(self, record, *, chunk_id, request, result, inputs, error):
            seen.update(chunk_id=chunk_id, request=request, result=result, inputs=inputs, error=error)
            return dict(record, sidecar_dir=f"mllm_calls/{chunk_id:04d}")

    class BrokenRecorder:
        def finalize(self, record, **kwargs):
            raise OSError(f"disk full writing {SECRET}")

    source = _source(ScriptedAdapter([json.dumps(make_chunk_doc(1))]), mock_calibration_file,
                     chunk_len=1, recorder=Recorder())
    record = _drive(source, 1)[0][2]["mllm_call"]
    assert record["sidecar_dir"] == "mllm_calls/0000"
    assert seen["chunk_id"] == 0 and seen["error"] is None
    assert seen["result"].text and seen["inputs"].images

    broken = _source(ScriptedAdapter([json.dumps(make_chunk_doc(1))]), mock_calibration_file,
                     chunk_len=1, recorder=BrokenRecorder(), secrets=(SECRET,))
    record = _drive(broken, 1)[0][2]["mllm_call"]
    assert "recorder_error" in record and SECRET not in record["recorder_error"]


def test_repr_and_last_details_never_hold_secrets(mock_calibration_file):
    source = _source(ScriptedAdapter([json.dumps(make_chunk_doc(1))]), mock_calibration_file,
                     chunk_len=1, secrets=(SECRET,))
    assert SECRET not in repr(source)
    out = _drive(source, 1)
    assert SECRET not in json.dumps(out[0][2])
    assert out[0][2]["action_path"] and out[0][2]["mllm_backend"] == LABEL


def test_close_closes_adapter_and_collector(mock_calibration_file):
    adapter = ScriptedAdapter([])
    collector = FakeCollector()
    source = _source(adapter, mock_calibration_file, collector=collector)
    source.close()
    assert adapter.closed and collector.closed


def test_public_attributes_for_later_plans(mock_calibration_file):
    from vla_bridge.mllm_prompt import PROMPT_TEMPLATE_VERSION
    from vla_bridge.mllm_schema import build_action_chunk_schema, schema_sha256

    adapter = ScriptedAdapter([])
    source = _source(adapter, mock_calibration_file, urdf_text="<robot/>")
    assert source.adapter is adapter
    assert source.label == LABEL and source.chunk_len == 3
    assert source.urdf_text == "<robot/>" and "ARM KINEMATICS (URDF)" in source.system_prompt
    assert source.schema == build_action_chunk_schema(3)
    assert source.schema_hash == schema_sha256(source.schema)
    assert source.prompt_template_version == PROMPT_TEMPLATE_VERSION


# --- Out-of-range values reach the UNMODIFIED validator (D-04) ------------------------------


def test_out_of_range_value_is_logged_raw_and_clamped_only_by_the_validator(
    tmp_path, mock_robot, mock_calibration_file
):
    for joint in JOINT_ORDER:
        mock_robot._positions[joint] = 5.0
    mock_robot._positions["shoulder_pan"] = 50.0
    doc = make_chunk_doc(1, base=5.0)
    doc["actions"][0]["shoulder_pan"] = 500.0
    adapter = ScriptedAdapter([json.dumps(doc)])
    limits = _limits(mock_calibration_file)
    source = MllmActionSourceBase(adapter, FakeCollector(), _settings(1), limits, LABEL)
    record = _run(tmp_path, mock_robot, source, limits, 1)[0]

    assert record["model_version"] == f"{LABEL}@mllm-chunk"  # no pre-clamp, no hold
    assert record["raw_model_output"]["shoulder_pan"] == 500.0
    assert record["validated_action"]["shoulder_pan"] == pytest.approx(limits["shoulder_pan"][1])
    assert any("shoulder_pan: clamped 500.0" in flag for flag in record["validator_flags"])


# --- Settings, registration and module hygiene (D-00a, D-02) -----------------------------------


@pytest.fixture
def config_dir(tmp_path, monkeypatch):
    from vla_bridge import backends

    configs = tmp_path / "configs"
    configs.mkdir()
    shipped = backends.BACKEND_CONFIG_DIR / "mllm_defaults.json"
    (configs / "mllm_defaults.json").write_text(shipped.read_text())
    monkeypatch.setattr(backends, "BACKEND_CONFIG_DIR", configs)
    return configs


def _write_config(config_dir, name, data):
    (config_dir / f"{name}.json").write_text(json.dumps(data))


def test_load_mllm_settings_defaults():
    settings = load_mllm_settings()
    assert settings["chunk_len"] == 20 and settings["step_dt_s"] == 0.785
    assert settings["sampling"] == {"temperature": 1.0, "top_p": 0.95}


def test_load_mllm_settings_per_backend_override_wins_and_merges_dicts(config_dir):
    _write_config(config_dir, "mllm_fake", {
        "chunk_len": 7, "sampling": {"temperature": None}, "depth": {"grid_rows": 12},
        "model": "fake-model",
    })
    settings = load_mllm_settings("mllm_fake")
    assert settings["chunk_len"] == 7
    assert settings["sampling"] == {"temperature": None, "top_p": 0.95}
    assert settings["depth"]["grid_rows"] == 12 and settings["depth"]["grid_cols"] == 32
    assert settings["model"] == "fake-model"  # provider keys pass through untouched


def test_load_mllm_settings_missing_files_name_the_path(config_dir):
    with pytest.raises(SystemExit, match="backend_configs/mllm_nope.json"):
        load_mllm_settings("mllm_nope")
    (config_dir / "mllm_defaults.json").unlink()
    with pytest.raises(SystemExit, match="backend_configs/mllm_defaults.json"):
        load_mllm_settings()


@pytest.mark.parametrize(
    "override,field",
    [
        ({"chunk_len": 0}, "chunk_len"),
        ({"chunk_len": True}, "chunk_len"),
        ({"chunk_len": "20"}, "chunk_len"),
        ({"chunk_len": 201}, "chunk_len"),
        ({"step_dt_s": 0}, "step_dt_s"),
        ({"step_dt_s": "0.785"}, "step_dt_s"),
        ({"depth_presentation": "video"}, "depth_presentation"),
        ({"left_image_source": "rectified"}, "left_image_source"),
        ({"image_format": "gif"}, "image_format"),
        ({"sampling": {"temperature": -1}}, "sampling"),
        ({"sampling": {"top_p": 0}}, "sampling"),
        ({"sampling": {"top_p": 1.5}}, "sampling"),
        ({"include_urdf": "yes"}, "include_urdf"),
        ({"max_consecutive_failed_calls": 0}, "max_consecutive_failed_calls"),
        ({"depth": {"range_m": [1.5, 0.2]}}, "depth"),
    ],
)
def test_load_mllm_settings_rejects_invalid_fields_by_name(config_dir, override, field):
    _write_config(config_dir, "mllm_bad", override)
    with pytest.raises(SystemExit, match=field):
        load_mllm_settings("mllm_bad")


def test_register_mllm_backend_records_spec_and_registers_name(monkeypatch):
    from vla_bridge import backends
    from vla_bridge.backends import base_mllm

    monkeypatch.setattr(backends, "BACKEND_REGISTRY", {})
    monkeypatch.setattr(base_mllm, "MLLM_BACKENDS", {})

    def factory(api_key, config):
        return ScriptedAdapter([])

    @base_mllm.register_mllm_backend("mllm-test", config_name="mllm_test", adapter_factory=factory)
    def build(args, camera_names, joint_limits_deg):
        return None

    assert backends.BACKEND_REGISTRY["mllm-test"] is build
    spec = base_mllm.MLLM_BACKENDS["mllm-test"]
    assert spec == base_mllm.MllmBackendSpec("mllm-test", "mllm_test", factory)
    with pytest.raises(Exception):
        spec.name = "other"  # frozen


def test_base_module_is_not_registered_by_discovery():
    from vla_bridge import backends

    for name, build in backends.BACKEND_REGISTRY.items():
        assert build.__module__ != "vla_bridge.backends.base_mllm", name


def test_get_action_signature_matches_the_protocol():
    import inspect

    params = list(inspect.signature(MllmActionSourceBase.get_action).parameters)
    assert params == ["self", "joint_state", "instruction"]


def test_base_mllm_imports_no_hardware_or_provider_sdk():
    import ast
    from pathlib import Path

    import vla_bridge.backends.base_mllm as module

    tree = ast.parse(Path(module.__file__).read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not imported & {"lerobot", "cv2", "anthropic", "openai"}
