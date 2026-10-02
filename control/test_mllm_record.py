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

from vla_bridge.mllm_record import (
    IMAGE_EXTENSIONS,
    REDACTED,
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
