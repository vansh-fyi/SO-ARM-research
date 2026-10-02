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
