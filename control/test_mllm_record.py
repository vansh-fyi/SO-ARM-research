"""Tests for the per-call MLLM audit layer (Phase 15, MLLM-06, PIPE-02, D-20..D-22).

`vla_bridge.mllm_record`: secret redaction, JSON-safety, cost estimation, the
whitelisted atomic sidecar writer and (Task 2) the `CallRecorder` that plugs
into `MllmActionSourceBase`'s recorder hook.

Nothing here touches real hardware, a real camera or the network. SECRET is a
sentinel API-key value; the integration tests scan every byte the run writes
for it.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from fake_providers import (
    FakeCaptureCamera,
    FakeCollector,
    ScriptedAdapter,
    make_chunk_doc,
    make_provider_result,
    png_bytes,
)
from run_vla_episode import run_episode
from vla_bridge import action_contract
from vla_bridge.action_contract import JOINT_ORDER
from vla_bridge.backends.base_mllm import (
    MAX_INLINE_THINKING_CHARS,
    MllmActionSourceBase,
    load_mllm_settings,
)
from vla_bridge.io_logger import IOLogger
from vla_bridge.mllm_record import (
    IMAGE_EXTENSIONS,
    REDACTED,
    CallRecorder,
    SECRET_HEADER_KEYS,
    SIDECAR_FILE_NAMES,
    SIDECAR_IMAGE_ROLES,
    estimate_cost_usd,
    redact,
    redact_text,
    sha256_hex,
    to_json_safe,
    write_call_sidecar,
)

SECRET = "sk-TEST-SECRET-do-not-log-0123456789"
OTHER_SECRET = "hf_ANOTHER_token_abcdef"


def _all_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file())


# --- redact_text / redact (D-00b, D-22, T-15-10) -----------------------------------


def test_redact_text_replaces_every_occurrence_of_each_secret():
    text = f"key={SECRET} again {SECRET} and {OTHER_SECRET}"
    out = redact_text(text, [SECRET, OTHER_SECRET])
    assert SECRET not in out and OTHER_SECRET not in out
    assert out.count(REDACTED) == 3


def test_redact_text_ignores_empty_and_too_short_secrets():
    text = "abc a ab the quick brown fox"
    assert redact_text(text, ["", "a", "ab", "abc", None]) == text


def test_redact_text_removes_json_escaped_form_of_a_secret():
    secret = 'pa"ss/wo\\rd-1234'
    encoded = json.dumps({"token": secret})
    out = redact_text(encoded, [secret])
    assert secret not in out
    assert json.dumps(secret)[1:-1] not in out


def test_redact_text_redacts_header_style_values_in_free_text():
    text = 'request failed: {"Authorization": "Bearer abc.def-123", "x-api-key": "zzz999"}'
    out = redact_text(text, [])
    assert "abc.def-123" not in out and "zzz999" not in out
    json.loads(out.split(": ", 1)[1])  # still valid JSON after redaction
    assert redact_text("api_key_env is ANTHROPIC_API_KEY", []) == "api_key_env is ANTHROPIC_API_KEY"


def test_redact_walks_nested_structures_and_json_encoded_strings():
    obj = {
        "a": [f"x {SECRET} y", ("t", SECRET), {"deep": {"deeper": [SECRET]}}],
        "json_text": json.dumps({"echo": SECRET}),
        "n": 3,
        "f": 1.5,
        "none": None,
        SECRET: "secret used as a key",
    }
    out = redact(obj, [SECRET])
    assert SECRET not in json.dumps(out)
    assert out["n"] == 3 and out["f"] == 1.5 and out["none"] is None
    assert isinstance(out["a"][1], tuple)
    assert REDACTED in out["json_text"]


@pytest.mark.parametrize("key", ["authorization", "Authorization", "X-API-Key", "api-key", "API_KEY", "apikey"])
def test_redact_replaces_secret_header_values_regardless_of_content(key):
    out = redact({"headers": {key: "anything-at-all", "content-type": "application/json"}}, [])
    assert out["headers"][key] == REDACTED
    assert out["headers"]["content-type"] == "application/json"
    out_nested = redact({key: {"nested": [1, 2]}}, [])
    assert out_nested[key] == REDACTED


def test_secret_header_keys_constant():
    assert set(SECRET_HEADER_KEYS) == {"authorization", "x-api-key", "api-key", "api_key", "apikey"}


# --- to_json_safe / sha256_hex -----------------------------------------------------


class _Model:
    def model_dump(self, mode="python"):
        assert mode == "json"
        return {"id": "resp_1", "content": [{"type": "text", "text": "hi"}]}


class _Opaque:
    def __repr__(self):
        return "Opaque(" + "x" * 5000 + ")"


def test_to_json_safe_converts_known_types_and_survives_json_dumps():
    payload = {
        "model": _Model(),
        "np_int": np.int64(7),
        "np_float": np.float32(0.5),
        "array": np.arange(4).reshape(2, 2),
        "path": Path("/tmp/x/y.png"),
        "blob": b"\x00\x01\x02",
        "tuple": (1, 2),
        "opaque": _Opaque(),
        "nan": float("nan"),
    }
    out = to_json_safe(payload)
    text = json.dumps(out, allow_nan=False)
    assert out["model"] == {"id": "resp_1", "content": [{"type": "text", "text": "hi"}]}
    assert out["np_int"] == 7 and isinstance(out["np_int"], int)
    assert out["np_float"] == 0.5
    assert out["array"] == [[0, 1], [2, 3]]
    assert out["path"] == "/tmp/x/y.png"
    assert out["blob"] == {"bytes": 3, "sha256": sha256_hex(b"\x00\x01\x02")}
    assert out["tuple"] == [1, 2]
    assert isinstance(out["opaque"], str) and out["opaque"].startswith("Opaque(")
    assert len(out["opaque"]) <= 2100
    assert isinstance(out["nan"], str)
    assert text


def test_sha256_hex_matches_hashlib():
    import hashlib

    assert sha256_hex(b"abc") == hashlib.sha256(b"abc").hexdigest()


# --- estimate_cost_usd (D-22c) -------------------------------------------------------

PRICES = {"input_per_mtok": 10, "cached_input_per_mtok": 1, "output_per_mtok": 50}


def test_estimate_cost_usd_bills_cached_and_uncached_input_and_output():
    usage = {"input_tokens": 1000, "cached_input_tokens": 200, "output_tokens": 500}
    expected = (800 * 10 + 200 * 1 + 500 * 50) / 1e6
    assert math.isclose(estimate_cost_usd(usage, PRICES), expected)
    assert math.isclose(estimate_cost_usd(usage, PRICES), 0.0332)


def test_estimate_cost_usd_cached_price_falls_back_to_input_price():
    usage = {"input_tokens": 1000, "cached_input_tokens": 200, "output_tokens": 500}
    prices = {"input_per_mtok": 10, "output_per_mtok": 50, "verified_on": "2026-10-02"}
    assert math.isclose(estimate_cost_usd(usage, prices), (1000 * 10 + 500 * 50) / 1e6)


def test_estimate_cost_usd_does_not_add_reasoning_tokens_again():
    base = {"input_tokens": 1000, "cached_input_tokens": 0, "output_tokens": 500}
    with_reasoning = dict(base, reasoning_tokens=400)
    assert estimate_cost_usd(with_reasoning, PRICES) == estimate_cost_usd(base, PRICES)


def test_estimate_cost_usd_unknown_cached_count_bills_all_input_at_input_price():
    usage = {"input_tokens": 1000, "cached_input_tokens": None, "output_tokens": 500}
    assert math.isclose(estimate_cost_usd(usage, PRICES), (1000 * 10 + 500 * 50) / 1e6)


@pytest.mark.parametrize(
    "usage,prices",
    [
        ({"input_tokens": None, "output_tokens": 500}, PRICES),
        ({"input_tokens": 1000, "output_tokens": None}, PRICES),
        ({"input_tokens": 1000}, PRICES),
        ({"input_tokens": -1, "output_tokens": 5}, PRICES),
        ({"input_tokens": 10, "cached_input_tokens": 20, "output_tokens": 5}, PRICES),
        ({"input_tokens": True, "output_tokens": 5}, PRICES),
        (None, PRICES),
        ({"input_tokens": 1000, "output_tokens": 500}, None),
        ({"input_tokens": 1000, "output_tokens": 500}, {}),
        ({"input_tokens": 1000, "output_tokens": 500}, {"input_per_mtok": 10}),
        ({"input_tokens": 1000, "output_tokens": 500}, {"input_per_mtok": "10", "output_per_mtok": 50}),
        ({"input_tokens": 1000, "output_tokens": 500}, {"input_per_mtok": -1, "output_per_mtok": 50}),
    ],
)
def test_estimate_cost_usd_returns_none_on_missing_or_invalid_inputs(usage, prices):
    assert estimate_cost_usd(usage, prices) is None


# --- write_call_sidecar (D-22d, ASVS V5, T-15-11) ------------------------------------


def test_sidecar_whitelists_and_image_extensions():
    assert set(SIDECAR_FILE_NAMES) == {
        "prompt.txt", "request_meta.json", "response_raw.json", "thinking.txt",
        "depth.npy", "aligned_left.png", "depth_heatmap.png", "depth_grid.txt",
    }
    assert tuple(SIDECAR_IMAGE_ROLES) == ("wrist", "left", "right", "depth")
    assert IMAGE_EXTENSIONS == {"image/png": "png", "image/jpeg": "jpg"}


def test_write_call_sidecar_writes_whitelisted_files_and_images_atomically(tmp_path):
    root = tmp_path / "mllm_calls"
    out = write_call_sidecar(
        root, 3,
        files={"prompt.txt": "hello\nworld", "depth.npy": b"\x93NUMPY-bytes"},
        images={"wrist": b"PNGW", "left": b"PNGL"},
        image_media_type="image/png",
    )
    chunk_dir = root / "0003"
    assert out["sidecar_dir"] == "mllm_calls/0003"
    assert sorted(out["files"]) == ["depth.npy", "prompt.txt"]
    assert out["images"] == {"wrist": "mllm_calls/0003/images/wrist.png",
                             "left": "mllm_calls/0003/images/left.png"}
    assert (chunk_dir / "prompt.txt").read_text() == "hello\nworld"  # str verbatim
    assert (chunk_dir / "depth.npy").read_bytes() == b"\x93NUMPY-bytes"  # bytes as bytes
    for role, rel in out["images"].items():
        assert (tmp_path / rel).read_bytes() == {"wrist": b"PNGW", "left": b"PNGL"}[role]
    assert not [p for p in _all_files(tmp_path) if p.name.endswith(".tmp")]


def test_write_call_sidecar_jpeg_extension(tmp_path):
    out = write_call_sidecar(tmp_path / "mllm_calls", 0, files={},
                             images={"right": b"JPG"}, image_media_type="image/jpeg")
    assert out["images"] == {"right": "mllm_calls/0000/images/right.jpg"}
    assert (tmp_path / "mllm_calls/0000/images/right.jpg").read_bytes() == b"JPG"


def test_write_call_sidecar_second_write_overwrites_same_fixed_names(tmp_path):
    root = tmp_path / "mllm_calls"
    write_call_sidecar(root, 1, files={"thinking.txt": "first"}, images={"wrist": b"A"},
                       image_media_type="image/png")
    write_call_sidecar(root, 1, files={"thinking.txt": "second"}, images={"wrist": b"B"},
                       image_media_type="image/png")
    assert (root / "0001" / "thinking.txt").read_text() == "second"
    assert (root / "0001" / "images" / "wrist.png").read_bytes() == b"B"
    names = sorted(p.relative_to(root).as_posix() for p in _all_files(root))
    assert names == ["0001/images/wrist.png", "0001/thinking.txt"]


@pytest.mark.parametrize("chunk_id", [-1, True, False, 1.0, "3", None, np.int64(2)])
def test_write_call_sidecar_rejects_bad_chunk_id_before_writing(tmp_path, chunk_id):
    with pytest.raises(ValueError):
        write_call_sidecar(tmp_path / "mllm_calls", chunk_id, files={"prompt.txt": "x"},
                           images={}, image_media_type="image/png")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "name",
    ["../x", "x", "/etc/passwd", str(Path("/tmp") / "prompt.txt"), "images/wrist.png",
     "prompt.txt/../../x", "PROMPT.TXT", "", 3],
)
def test_write_call_sidecar_rejects_unknown_filename_before_writing(tmp_path, name):
    with pytest.raises(ValueError):
        write_call_sidecar(tmp_path / "mllm_calls", 3,
                           files={"prompt.txt": "ok", name: "y"},
                           images={"wrist": b"W"}, image_media_type="image/png")
    assert list(tmp_path.iterdir()) == []


def test_write_call_sidecar_traversal_acceptance_case(tmp_path):
    with pytest.raises(ValueError):
        write_call_sidecar(tmp_path / "mllm_calls", 3, files={"../x": "y"}, images={},
                           image_media_type="image/png")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("role", ["../wrist", "top", "WRIST", ""])
def test_write_call_sidecar_rejects_unknown_image_role_before_writing(tmp_path, role):
    with pytest.raises(ValueError):
        write_call_sidecar(tmp_path / "mllm_calls", 0, files={"prompt.txt": "x"},
                           images={"wrist": b"W", role: b"X"}, image_media_type="image/png")
    assert list(tmp_path.iterdir()) == []


def test_write_call_sidecar_rejects_unknown_media_type_and_bad_content(tmp_path):
    with pytest.raises(ValueError):
        write_call_sidecar(tmp_path / "mllm_calls", 0, files={}, images={"wrist": b"W"},
                           image_media_type="image/../../gif")
    with pytest.raises((TypeError, ValueError)):
        write_call_sidecar(tmp_path / "mllm_calls", 0, files={"prompt.txt": 12345},
                           images={}, image_media_type="image/png")
    assert list(tmp_path.iterdir()) == []


# --- CallRecorder through the real base and loop (D-20, D-21, D-22) ---------------------

LABEL = "mllm-fake"
INSTRUCTION = "Pick up the pen"
PRICE_TABLE = {
    "input_per_mtok": 3.0,
    "cached_input_per_mtok": 0.3,
    "output_per_mtok": 15.0,
    "currency": "USD",
    "verified_on": "2026-10-02",
    "source": "test fixture",
}
USAGE = {
    "input_tokens": 1200,
    "output_tokens": 300,
    "reasoning_tokens": 120,
    "cached_input_tokens": 200,
    "raw": {"prompt_tokens": 1200, "completion_tokens": 300},
}
EXPECTED_COST = (1000 * 3.0 + 200 * 0.3 + 300 * 15.0) / 1e6
HEADER_ONLY_VALUE = "header-value-that-is-not-the-secret"
THINKING = "Considering the pen and the gripper. " * 700 + f" the key was {SECRET} end"
EXPECTED_SIDECAR = {"prompt.txt", "request_meta.json", "response_raw.json", "thinking.txt"}


def _settings(chunk_len=3):
    settings = load_mllm_settings()
    settings["chunk_len"] = chunk_len
    return settings


def _limits(mock_calibration_file):
    return action_contract.load_joint_limits_deg(mock_calibration_file)


def _secret_result(doc):
    raw_payload = {
        "id": "resp_fake_1",
        "headers": {"x-api-key": HEADER_ONLY_VALUE, "Authorization": f"Bearer {SECRET}"},
        "echo": f"request used key {SECRET}",
        "output": [{"type": "text", "text": json.dumps(doc)}],
    }
    return make_provider_result(
        json.dumps(doc),
        raw_payload=raw_payload,
        usage=dict(USAGE),
        native_thinking_text=THINKING,
        native_thinking_kind="raw",
        request_id="req_1",
        response_id="resp_fake_1",
        model_returned="fake-model-2026-10-01",
    )


def _source(adapter, mock_calibration_file, recorder, collector=None, chunk_len=3, **kwargs):
    return MllmActionSourceBase(
        adapter,
        collector if collector is not None else FakeCollector(),
        _settings(chunk_len),
        _limits(mock_calibration_file),
        LABEL,
        recorder=recorder,
        **kwargs,
    )


def _run(tmp_path, mock_robot, source, limits, max_steps):
    with IOLogger(tmp_path, {0: "wrist"}) as io_logger:
        run_episode(mock_robot, {0: FakeCaptureCamera()}, {0: "wrist"}, io_logger, source,
                    INSTRUCTION, max_steps, 100.0, limits,
                    home_pose={joint: mock_robot._positions[joint] for joint in JOINT_ORDER})
    lines = (tmp_path / "episode.jsonl").read_text().strip().splitlines()
    return [json.loads(line) for line in lines]


def _secret_episode(tmp_path, mock_robot, mock_calibration_file, prices=PRICE_TABLE):
    for joint in JOINT_ORDER:
        mock_robot._positions[joint] = 5.0
    doc1 = make_chunk_doc(3, plan=f"Lower toward the pen (do not echo {SECRET})", base=5.0)
    doc2 = make_chunk_doc(3, base=7.0)
    adapter = ScriptedAdapter([_secret_result(doc1), json.dumps(doc2)])
    recorder = CallRecorder(tmp_path / "mllm_calls", secrets=[SECRET], prices=prices)
    limits = _limits(mock_calibration_file)
    # Secrets go ONLY to the recorder: it alone must keep them out of every artifact.
    source = _source(adapter, mock_calibration_file, recorder)
    records = _run(tmp_path, mock_robot, source, limits, 6)
    return adapter, source, records


def test_record_integration_sidecar_layout_cost_and_image_paths(tmp_path, mock_robot, mock_calibration_file):
    adapter, _source_obj, records = _secret_episode(tmp_path, mock_robot, mock_calibration_file)
    call = records[0]["action_details"]["mllm_call"]

    assert call["sidecar_dir"] == "mllm_calls/0000"
    assert math.isclose(call["cost_usd_est"], EXPECTED_COST)
    assert call["price_table"] == PRICE_TABLE
    assert "sidecar_error" not in call

    chunk_dir = tmp_path / "mllm_calls" / "0000"
    names = {p.name for p in chunk_dir.iterdir() if p.is_file()}
    assert EXPECTED_SIDECAR <= names
    assert set(call["sidecar_files"]) == names
    images = sorted(p.name for p in (chunk_dir / "images").iterdir())
    assert images == ["left.png", "right.png", "wrist.png"]

    sent = dict(adapter.requests[0].images)
    assert [img["role"] for img in call["images"]] == ["wrist", "left", "right"]
    for img in call["images"]:
        path = tmp_path / img["path"]
        assert path.is_file()
        data = path.read_bytes()
        assert data == sent[img["role"]]  # the exact bytes sent
        assert sha256_hex(data) == img["sha256"] and len(data) == img["bytes"]

    # The second chunk: default usage has cached None -> all input at the input price.
    second = records[3]["action_details"]["mllm_call"]
    assert second["sidecar_dir"] == "mllm_calls/0001"
    assert math.isclose(second["cost_usd_est"], (100 * 3.0 + 50 * 15.0) / 1e6)
    assert (tmp_path / "mllm_calls" / "0001" / "response_raw.json").is_file()
    assert not (tmp_path / "mllm_calls" / "0001" / "thinking.txt").exists()


def test_record_request_meta_holds_settings_hashes_schema_and_capture(tmp_path, mock_robot, mock_calibration_file):
    adapter, source, records = _secret_episode(tmp_path, mock_robot, mock_calibration_file)
    call = records[0]["action_details"]["mllm_call"]
    meta = json.loads((tmp_path / "mllm_calls/0000/request_meta.json").read_text())

    assert meta["chunk_id"] == 0
    assert meta["backend"] == LABEL and meta["provider"] == "fake"
    assert meta["model_requested"] == "fake-model"
    assert meta["settings"] == adapter.settings_summary  # requested / applied / omitted (D-20)
    assert set(meta["settings"]) == {"requested", "applied", "omitted"}
    for key in ("prompt_sha256", "prompt_template_sha256", "schema_sha256"):
        assert meta[key] == call[key]
    assert meta["schema_name"] == adapter.requests[0].schema_name
    assert meta["schema"] == source.schema
    assert meta["image_roles"] == ["wrist", "left", "right"]
    assert meta["image_media_type"] == "image/png"
    assert meta["depth"] == {"status": "not_configured"}
    assert meta["capture"]["left_image_source_sent"] == "raw"
    assert "prev_chunk_tick_period_s" in meta
    assert meta.get("error") is None


def test_record_prompt_txt_has_system_user_and_image_list(tmp_path, mock_robot, mock_calibration_file):
    adapter, _source_obj, records = _secret_episode(tmp_path, mock_robot, mock_calibration_file)
    request = adapter.requests[0]
    text = (tmp_path / "mllm_calls/0000/prompt.txt").read_text()
    assert text.startswith(request.system_prompt)
    assert "USER MESSAGE" in text
    assert request.user_text in text
    assert text.index("USER MESSAGE") < text.index(request.user_text)
    positions = [text.index(f"{img['role']}: sha256 {img['sha256']}, {img['bytes']} bytes")
                 for img in records[0]["action_details"]["mllm_call"]["images"]]
    assert positions == sorted(positions)  # send order


def test_record_thinking_txt_keeps_full_text_while_record_is_capped(tmp_path, mock_robot, mock_calibration_file):
    _adapter, _source_obj, records = _secret_episode(tmp_path, mock_robot, mock_calibration_file)
    assert len(THINKING) > MAX_INLINE_THINKING_CHARS
    full = (tmp_path / "mllm_calls/0000/thinking.txt").read_text()
    assert full == THINKING.replace(SECRET, REDACTED)
    inline = records[0]["action_details"]["mllm_call"]["trace"]["native_thinking_text"]
    assert len(inline) == MAX_INLINE_THINKING_CHARS


def test_record_response_raw_is_redacted_including_header_values(tmp_path, mock_robot, mock_calibration_file):
    _secret_episode(tmp_path, mock_robot, mock_calibration_file)
    raw = json.loads((tmp_path / "mllm_calls/0000/response_raw.json").read_text())
    assert raw["id"] == "resp_fake_1"
    assert raw["headers"]["x-api-key"] == REDACTED  # key-based, content irrelevant
    assert raw["headers"]["Authorization"] == REDACTED
    assert REDACTED in raw["echo"]
    assert HEADER_ONLY_VALUE not in (tmp_path / "mllm_calls/0000/response_raw.json").read_text()


def test_secret_absent_from_every_artifact_of_the_run(tmp_path, mock_robot, mock_calibration_file):
    _secret_episode(tmp_path, mock_robot, mock_calibration_file)
    files = _all_files(tmp_path)
    assert tmp_path / "episode.jsonl" in files
    assert len([p for p in files if "mllm_calls" in p.parts]) >= 2 * 4
    needles = {SECRET.encode(), json.dumps(SECRET)[1:-1].encode(), HEADER_ONLY_VALUE.encode()}
    for path in files:
        data = path.read_bytes()
        for needle in needles:
            assert needle not in data, f"secret leaked into {path.relative_to(tmp_path)}"


def test_record_secret_absent_from_second_prompt_carrying_previous_plan(tmp_path, mock_robot, mock_calibration_file):
    adapter, _source_obj, _records = _secret_episode(tmp_path, mock_robot, mock_calibration_file)
    assert SECRET in adapter.requests[1].user_text  # the carried plan really contained it
    text = (tmp_path / "mllm_calls/0001/prompt.txt").read_text()
    assert SECRET not in text and REDACTED in text


class _SidecarCollector(FakeCollector):
    def collect(self, joint_state):
        inputs = super().collect(joint_state)
        inputs.sidecar_files = {
            "depth.npy": b"\x93NUMPY fake depth",
            "aligned_left.png": png_bytes(50),
            "depth_heatmap.png": png_bytes(60),
            "depth_grid.txt": f"grid rows ... {SECRET}",
            "../escape.txt": b"must not be written",
            "notes.txt": "not whitelisted",
            "prompt.txt": "a collector must not replace the recorder's prompt",
        }
        return inputs


def test_sidecar_collector_files_written_when_whitelisted_and_ignored_otherwise(tmp_path, mock_calibration_file):
    adapter = ScriptedAdapter([json.dumps(make_chunk_doc(1))])
    recorder = CallRecorder(tmp_path / "mllm_calls", secrets=[SECRET], prices=PRICE_TABLE)
    source = _source(adapter, mock_calibration_file, recorder, collector=_SidecarCollector(), chunk_len=1)
    source.get_action({joint: 5.0 for joint in JOINT_ORDER}, INSTRUCTION)

    call = source.last_details["mllm_call"]
    chunk_dir = tmp_path / "mllm_calls" / "0000"
    names = {p.name for p in chunk_dir.iterdir() if p.is_file()}
    no_thinking = EXPECTED_SIDECAR - {"thinking.txt"}  # this result carries no native thinking
    assert names == no_thinking | {"depth.npy", "aligned_left.png", "depth_heatmap.png", "depth_grid.txt"}
    assert set(call["sidecar_files"]) == names
    assert (chunk_dir / "depth.npy").read_bytes() == b"\x93NUMPY fake depth"
    assert (chunk_dir / "aligned_left.png").read_bytes() == png_bytes(50)
    assert (chunk_dir / "depth_grid.txt").read_text() == f"grid rows ... {REDACTED}"
    assert "collector must not replace" not in (chunk_dir / "prompt.txt").read_text()
    assert not (tmp_path / "mllm_calls" / "escape.txt").exists()
    assert not (tmp_path / "escape.txt").exists()
    assert "sidecar_error" not in call


def test_sidecar_on_provider_error_has_prompt_meta_images_and_no_response(tmp_path, mock_calibration_file):
    adapter = ScriptedAdapter([RuntimeError(f"connection reset (api key {SECRET})")])
    recorder = CallRecorder(tmp_path / "mllm_calls", secrets=[SECRET], prices=PRICE_TABLE)
    source = _source(adapter, mock_calibration_file, recorder)
    _action, version = source.get_action({joint: 5.0 for joint in JOINT_ORDER}, INSTRUCTION)
    assert version == f"{LABEL}@provider-error-holding-position"

    call = source.last_details["mllm_call"]
    assert call["usage"] is None and call["stop"]["reason"] == "error"
    assert call["cost_usd_est"] is None
    assert call["price_table"] == PRICE_TABLE
    chunk_dir = tmp_path / "mllm_calls" / "0000"
    names = {p.name for p in chunk_dir.iterdir() if p.is_file()}
    assert names == {"prompt.txt", "request_meta.json"}
    assert sorted(p.name for p in (chunk_dir / "images").iterdir()) == ["left.png", "right.png", "wrist.png"]
    meta = json.loads((chunk_dir / "request_meta.json").read_text())
    assert meta["error"].startswith("RuntimeError: connection reset")
    assert SECRET not in meta["error"] and REDACTED in meta["error"]
    for path in _all_files(tmp_path):
        assert SECRET.encode() not in path.read_bytes()


def test_sidecar_error_when_root_unwritable_does_not_break_the_loop(tmp_path, mock_robot, mock_calibration_file):
    for joint in JOINT_ORDER:
        mock_robot._positions[joint] = 5.0
    blocker = tmp_path / "mllm_calls"
    blocker.write_text("a regular file where the sidecar directory should be")
    adapter = ScriptedAdapter([json.dumps(make_chunk_doc(3, base=5.0))])
    recorder = CallRecorder(blocker, secrets=[SECRET], prices=PRICE_TABLE)
    source = _source(adapter, mock_calibration_file, recorder)
    records = _run(tmp_path, mock_robot, source, _limits(mock_calibration_file), 3)

    assert len(records) == 3
    assert [r["model_version"] for r in records] == [f"{LABEL}@mllm-chunk"] * 3
    call = records[0]["action_details"]["mllm_call"]
    assert isinstance(call["sidecar_error"], str) and call["sidecar_error"]
    assert call["sidecar_dir"] is None
    assert math.isclose(call["cost_usd_est"], (100 * 3.0 + 50 * 15.0) / 1e6)
    assert "recorder_error" not in call


def test_finalize_redacts_sidecar_error_and_never_raises(tmp_path, mock_calibration_file):
    blocker = tmp_path / f"calls-{SECRET}"
    blocker.write_text("not a directory")
    adapter = ScriptedAdapter([json.dumps(make_chunk_doc(1))])
    recorder = CallRecorder(blocker, secrets=[SECRET])
    source = _source(adapter, mock_calibration_file, recorder, chunk_len=1)
    source.get_action({joint: 5.0 for joint in JOINT_ORDER}, INSTRUCTION)
    call = source.last_details["mllm_call"]
    assert call["sidecar_error"] and SECRET not in call["sidecar_error"]
    assert call["cost_usd_est"] is None and call["price_table"] is None
    assert len(call["sidecar_error"]) <= 2000


def test_recorder_last_details_json_every_tick_and_later_ticks_only_chunk_ref(tmp_path, mock_calibration_file):
    docs = [make_chunk_doc(3, base=5.0), make_chunk_doc(3, base=7.0)]
    adapter = ScriptedAdapter([_secret_result(docs[0]), json.dumps(docs[1])])
    recorder = CallRecorder(tmp_path / "mllm_calls", secrets=[SECRET], prices=PRICE_TABLE)
    source = _source(adapter, mock_calibration_file, recorder)
    for tick in range(6):
        source.get_action({joint: 5.0 for joint in JOINT_ORDER}, INSTRUCTION)
        text = json.dumps(source.last_details, allow_nan=False)
        assert SECRET not in text
        if tick % 3 == 0:
            assert source.last_details["mllm_call"]["sidecar_dir"] == f"mllm_calls/{tick // 3:04d}"
        else:
            assert "mllm_call" not in source.last_details
            assert "sidecar" not in text and "mllm_calls/" not in text
            assert source.last_details["chunk_ref"]["step_index"] == tick % 3
    assert adapter.calls == 2


def test_recorder_price_table_is_a_snapshot(tmp_path, mock_calibration_file):
    prices = dict(PRICE_TABLE)
    adapter = ScriptedAdapter([json.dumps(make_chunk_doc(1))])
    source = _source(adapter, mock_calibration_file,
                     CallRecorder(tmp_path / "mllm_calls", prices=prices), chunk_len=1)
    source.get_action({joint: 5.0 for joint in JOINT_ORDER}, INSTRUCTION)
    prices["input_per_mtok"] = 999.0
    assert source.last_details["mllm_call"]["price_table"]["input_per_mtok"] == 3.0


class _Payload:
    def model_dump(self, mode="python"):
        return {"id": "msg_1", "usage": {"input_tokens": np.int64(5)},
                "headers": {"authorization": "Bearer xyz"}, "note": f"echo {SECRET}"}


@pytest.mark.parametrize("payload_kind", ["str", "model"])
def test_response_raw_str_written_verbatim_and_objects_dumped(tmp_path, mock_calibration_file, payload_kind):
    wire = '{"id": "chatcmpl-1", "choices": []}' + f"  {SECRET}"
    payload = wire if payload_kind == "str" else _Payload()
    result = make_provider_result(json.dumps(make_chunk_doc(1)), raw_payload=payload)
    adapter = ScriptedAdapter([result])
    source = _source(adapter, mock_calibration_file,
                     CallRecorder(tmp_path / "mllm_calls", secrets=[SECRET]), chunk_len=1)
    source.get_action({joint: 5.0 for joint in JOINT_ORDER}, INSTRUCTION)
    text = (tmp_path / "mllm_calls/0000/response_raw.json").read_text()
    assert SECRET not in text
    if payload_kind == "str":
        assert text == wire.replace(SECRET, REDACTED)
    else:
        doc = json.loads(text)
        assert doc["id"] == "msg_1" and doc["usage"]["input_tokens"] == 5
        assert doc["headers"]["authorization"] == REDACTED
