"""Failure and end-of-episode semantics of the MLLM chunk-drain base (Phase 15).

Covers MLLM-07 (every answer is a fresh failure opportunity; no failure may
crash, zero-fill or spend per tick) and D-25 (a strict boolean
`task_complete: true` holds the whole chunk and ends the episode, because the
claim is about the CURRENT views, so no motion may follow it).

Research pitfalls exercised here:
  - Pitfall 4: a whole-call failure enqueues `chunk_len` holds; the next paid
    call happens only after the chunk drains, never on the next tick.
  - Pitfall 12: a refusal is a failed call whose stop details (including
    `refusal_category`) are kept in the record, not an exception.
  - Pitfall 15: after `max_consecutive_failed_calls` failed calls the source
    sets `episode_complete_reason = "mllm_backend_failed"`, and a guard stops
    any further collector or provider call once a terminal reason is set.

Every hold must echo the live joint state of THAT tick (distinct, non-zero per
tick), never zeros and never a previous target. Nothing here touches hardware,
a camera or the network: the provider is a `ScriptedAdapter`.
"""

import json

import pytest

from fake_providers import FakeCollector, ScriptedAdapter, make_chunk_doc, make_provider_result
from vla_bridge import action_contract
from vla_bridge.action_contract import JOINT_ORDER
from vla_bridge.backends.base_mllm import MllmActionSourceBase, load_mllm_settings

LABEL = "mllm-fake"
INSTRUCTION = "Pick up the pen"
SECRET = "sk-FAILURE-SECRET-never-log-9876543210"


def _js(tick):
    """A distinct, non-zero live joint state per tick (holds must echo THIS one)."""
    return {joint: 10.0 * (tick + 1) + i for i, joint in enumerate(JOINT_ORDER)}


def _source(adapter, mock_calibration_file, *, chunk_len=3, cap=3, collector=None):
    settings = load_mllm_settings()
    settings["chunk_len"] = chunk_len
    settings["max_consecutive_failed_calls"] = cap
    return MllmActionSourceBase(
        adapter,
        collector if collector is not None else FakeCollector(),
        settings,
        action_contract.load_joint_limits_deg(mock_calibration_file),
        LABEL,
        secrets=(SECRET,),
    )


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


def _ok_text(n=3, **kwargs):
    return json.dumps(make_chunk_doc(n, **kwargs))


def _stop_result(stop_reason, refusal_category=None):
    result = make_provider_result(_ok_text(), stop_reason=stop_reason)
    result.stop["refusal_category"] = refusal_category
    return result


# --- The whole-call failure matrix (MLLM-07, Pitfall 4) ----------------------------------

FAILURE_MATRIX = [
    pytest.param(lambda: "{oops", "invalid-json", id="invalid-json-text"),
    pytest.param(lambda: "", "empty-response", id="empty-string"),
    pytest.param(lambda: "   \n\t ", "empty-response", id="whitespace-only"),
    pytest.param(lambda: "[1, 2, 3]", "invalid-json", id="json-array"),
    pytest.param(lambda: '{"plan": "p", "task_complete": false}', "missing-actions", id="no-actions"),
    pytest.param(lambda: '{"actions": {"shoulder_pan": 1.0}}', "missing-actions", id="actions-not-list"),
    pytest.param(lambda: _stop_result("refusal", "reasoning_extraction"), "refusal", id="refusal"),
    pytest.param(lambda: _stop_result("truncated"), "truncated", id="truncated"),
    pytest.param(lambda: _stop_result("other"), "provider-error", id="other-stop"),
    pytest.param(lambda: RuntimeError(f"500 upstream failure for key {SECRET}"), "provider-error",
                 id="runtime-error-with-secret"),
    pytest.param(lambda: TimeoutError("read timed out after 900 s"), "provider-error", id="timeout"),
]


@pytest.mark.parametrize("make_item,reason", FAILURE_MATRIX)
def test_whole_call_failure_holds_live_state_for_whole_chunk_then_calls_again(
    make_item, reason, mock_calibration_file
):
    chunk_len = 3
    success = make_chunk_doc(chunk_len, base=5.0)
    adapter = ScriptedAdapter([make_item(), json.dumps(success)])
    source = _source(adapter, mock_calibration_file, chunk_len=chunk_len)

    out = _drive(source, chunk_len)
    for tick in range(chunk_len):
        _assert_hold(out[tick], tick, reason)
        assert out[tick][0] != {joint: 0.0 for joint in JOINT_ORDER}
        assert SECRET not in json.dumps(out[tick][2])
    assert adapter.calls == 1  # no per-tick retry while the failed chunk drains

    record = out[0][2]["mllm_call"]
    assert record["parse"]["ok"] is False
    assert record["parse"]["reason"] == reason
    assert record["consecutive_failed_calls"] == 1
    if reason == "refusal":
        assert record["stop"]["refusal_category"] == "reasoning_extraction"
        assert out[0][1] == f"{LABEL}@refusal-holding-position"
    assert source.episode_complete_reason is None
    assert source.task_complete_claimed is False

    after = _drive(source, 1, start=chunk_len)
    assert adapter.calls == 2  # the next paid call happens exactly at tick chunk_len
    action, model_version, details = after[0]
    assert model_version == f"{LABEL}@mllm-chunk"
    assert action == success["actions"][0]
    assert details["mllm_call"]["consecutive_failed_calls"] == 0
    assert SECRET not in json.dumps(details)


def test_keyboard_interrupt_propagates_and_releases_lock_so_next_call_works(mock_calibration_file):
    success = make_chunk_doc(3, base=5.0)
    adapter = ScriptedAdapter([KeyboardInterrupt(), json.dumps(success)])
    source = _source(adapter, mock_calibration_file)
    with pytest.raises(KeyboardInterrupt):
        source.get_action(_js(0), INSTRUCTION)
    assert source._inflight_lock.acquire(blocking=False)
    source._inflight_lock.release()

    action, model_version = source.get_action(_js(1), INSTRUCTION)
    assert model_version == f"{LABEL}@mllm-chunk"
    assert action == success["actions"][0]
    assert adapter.calls == 2
    assert source.episode_complete_reason is None


# --- Consecutive-failure cap and runaway-spend guard (Pitfall 15, T-15-06) ------------------


def test_three_consecutive_failed_calls_end_the_episode(mock_calibration_file):
    adapter = ScriptedAdapter([RuntimeError("boom 1"), "{oops", _stop_result("truncated")])
    source = _source(adapter, mock_calibration_file, chunk_len=2, cap=3)

    out = _drive(source, 4)
    assert adapter.calls == 2
    assert source.episode_complete_reason is None
    assert out[0][2]["mllm_call"]["consecutive_failed_calls"] == 1
    assert out[2][2]["mllm_call"]["consecutive_failed_calls"] == 2

    out = _drive(source, 1, start=4)
    assert adapter.calls == 3
    assert out[0][2]["mllm_call"]["consecutive_failed_calls"] == 3
    assert source.episode_complete_reason == "mllm_backend_failed"
    assert source.task_complete_claimed is False


def test_a_success_between_failures_resets_the_counter(mock_calibration_file):
    adapter = ScriptedAdapter([
        RuntimeError("boom"), "{oops", _ok_text(2), _stop_result("other"), "",
    ])
    source = _source(adapter, mock_calibration_file, chunk_len=2, cap=3)
    out = _drive(source, 10)
    assert adapter.calls == 5
    counts = [out[tick][2]["mllm_call"]["consecutive_failed_calls"] for tick in range(0, 10, 2)]
    assert counts == [1, 2, 0, 1, 2]
    assert source.episode_complete_reason is None


def test_camera_failures_never_count_toward_the_cap(mock_calibration_file):
    collector = FakeCollector()
    adapter = ScriptedAdapter([RuntimeError("boom 1"), RuntimeError("boom 2")])
    source = _source(adapter, mock_calibration_file, chunk_len=1, cap=2, collector=collector)

    _drive(source, 1)
    assert adapter.calls == 1

    collector.fail = True
    out = _drive(source, 4, start=1)
    for k, output in enumerate(out):
        _assert_hold(output, k + 1, "camera-read-failed")
    assert adapter.calls == 1
    assert source.episode_complete_reason is None  # camera failures did not increment

    collector.fail = False
    out = _drive(source, 1, start=5)
    assert out[0][2]["mllm_call"]["consecutive_failed_calls"] == 2  # and did not reset
    assert source.episode_complete_reason == "mllm_backend_failed"


def test_guard_blocks_any_further_call_once_episode_is_complete(mock_calibration_file):
    collector = FakeCollector()
    adapter = ScriptedAdapter([RuntimeError("a"), RuntimeError("b"), RuntimeError("c")])
    source = _source(adapter, mock_calibration_file, chunk_len=2, cap=3, collector=collector)
    _drive(source, 6)  # three failed calls, each chunk fully drained
    assert source.episode_complete_reason == "mllm_backend_failed"
    assert adapter.calls == 3 and collector.calls == 3

    out = _drive(source, 5, start=6)
    for k, output in enumerate(out):
        _assert_hold(output, 6 + k, "episode-complete")
        assert "mllm_call" not in output[2]
    assert adapter.calls == 3  # no paid call after a terminal flag
    assert collector.calls == 3  # and no camera capture either


def test_failed_call_record_has_stop_parse_usage_and_counter(mock_calibration_file):
    adapter = ScriptedAdapter([RuntimeError(f"bad key {SECRET}"), _stop_result("refusal", "cyber")])
    source = _source(adapter, mock_calibration_file, chunk_len=1, cap=5)
    out = _drive(source, 2)

    first = out[0][2]["mllm_call"]
    assert first["stop"] == {"reason": "error", "refusal_category": None, "incomplete_reason": None}
    assert first["usage"] is None  # no response arrived
    assert first["parse"]["ok"] is False and first["parse"]["reason"] == "provider-error"
    assert first["consecutive_failed_calls"] == 1
    assert first["trace"]["task_complete"] is False
    assert SECRET not in json.dumps(first)

    second = out[1][2]["mllm_call"]
    assert second["stop"]["reason"] == "refusal" and second["stop"]["refusal_category"] == "cyber"
    assert second["usage"]["input_tokens"] == 100  # a response arrived
    assert second["parse"]["ok"] is False and second["parse"]["reason"] == "refusal"
    assert second["consecutive_failed_calls"] == 2


# --- task_complete (D-18, D-19, D-25) ---------------------------------------------------------


def test_task_complete_true_holds_the_whole_chunk_and_ends_the_episode(mock_calibration_file):
    doc = make_chunk_doc(3, base=5.0, task_complete=True)
    collector = FakeCollector()
    adapter = ScriptedAdapter([json.dumps(doc)])
    source = _source(adapter, mock_calibration_file, chunk_len=3, collector=collector)

    out = _drive(source, 3)
    for tick in range(3):
        _assert_hold(out[tick], tick, "task-complete")
        assert out[tick][0] not in doc["actions"]  # the model's motion is never executed
    assert adapter.calls == 1
    assert source.episode_complete_reason == "task_complete"
    assert source.task_complete_claimed is True

    record = out[0][2]["mllm_call"]
    assert record["trace"]["task_complete"] is True
    assert record["parse"]["ok"] is True
    assert record["parse"]["n_valid"] == 3
    assert record["consecutive_failed_calls"] == 0

    after = _drive(source, 2, start=3)
    for k, output in enumerate(after):
        _assert_hold(output, 3 + k, "episode-complete")
    assert adapter.calls == 1 and collector.calls == 1


def test_task_complete_true_with_a_malformed_step_still_holds_and_claims(mock_calibration_file):
    doc = make_chunk_doc(3, base=5.0, task_complete=True)
    doc["actions"][1]["elbow_flex"] = "bad"
    source = _source(ScriptedAdapter([json.dumps(doc)]), mock_calibration_file, chunk_len=3)
    out = _drive(source, 3)
    for tick in range(3):
        _assert_hold(out[tick], tick, "task-complete")
    assert source.episode_complete_reason == "task_complete"
    assert out[0][2]["mllm_call"]["parse"]["ok"] is False


@pytest.mark.parametrize("raw", ["true", 1, None, "yes"])
def test_non_boolean_true_task_complete_does_not_end_the_episode(raw, mock_calibration_file):
    doc = make_chunk_doc(3, base=5.0)
    doc["task_complete"] = raw
    adapter = ScriptedAdapter([json.dumps(doc)])
    source = _source(adapter, mock_calibration_file, chunk_len=3)
    out = _drive(source, 3)
    for tick, (action, model_version, _details) in enumerate(out):
        assert model_version == f"{LABEL}@mllm-chunk"
        assert action == doc["actions"][tick]
    trace = out[0][2]["mllm_call"]["trace"]
    assert trace["task_complete"] is False
    assert trace["task_complete_raw"] == raw
    assert source.episode_complete_reason is None
    assert source.task_complete_claimed is False


@pytest.mark.parametrize("stop_reason,hold_reason", [("refusal", "refusal"), ("truncated", "truncated")])
def test_a_failed_call_never_sets_the_claim(stop_reason, hold_reason, mock_calibration_file):
    result = make_provider_result(_ok_text(3, task_complete=True), stop_reason=stop_reason)
    source = _source(ScriptedAdapter([result]), mock_calibration_file, chunk_len=3)
    out = _drive(source, 3)
    for tick in range(3):
        _assert_hold(out[tick], tick, hold_reason)
    assert source.task_complete_claimed is False
    assert source.episode_complete_reason is None
    assert out[0][2]["mllm_call"]["trace"]["task_complete"] is False
