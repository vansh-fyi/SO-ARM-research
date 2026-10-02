"""Tests for the live MLLM smoke CLI (`python -m vla_bridge.mllm_smoke`, Plan 15-09).

The smoke tool makes ONE tiny paid call per backend (Plan 15-10), so these
tests pin its spend gate and its secret discipline without ever spending:

  - without `--confirm-spend` it prints the plan and returns 2 without reading
    the key or building a client;
  - with `--confirm-spend` it runs the registered adapter factory (here a fake
    spec registered on `base_mllm.MLLM_BACKENDS`) through the real
    `MllmActionSourceBase`, `MllmInputCollector` and `CallRecorder` with
    synthetic images and a synthetic depth map, writes the sidecar and prints
    PASS/WARN/FAIL lines;
  - the key sentinel never appears in stdout or in any file under --out.

No real provider, robot, camera or depth service is contacted.
"""

import ast
import json
import shutil
from pathlib import Path

import pytest

from fake_providers import ScriptedAdapter, make_chunk_text, make_provider_result
from vla_bridge import action_contract, backends
from vla_bridge.backends import base_mllm, base_mllm_build, base_mllm_chat, mllm_claude, mllm_gpt
from vla_bridge.backends.base_mllm import MllmBackendSpec
from vla_bridge.mllm_prompt import build_system_prompt
from vla_bridge.mllm_urdf import urdf_kinematics_text

from vla_bridge import mllm_smoke

BACKEND_NAME = "mllm-fake-smoke"
CONFIG_NAME = "mllm_fake_smoke"
API_KEY_ENV = "SOARM_TEST_MLLM_SMOKE_KEY"
SECRET = "sk-test-SMOKE-SECRET-0123456789abcdefghij"
MODEL = "fake-smoke-model-1"
REAL_DEFAULTS = backends.BACKEND_CONFIG_DIR / "mllm_defaults.json"
MODULE_PATH = Path(mllm_smoke.__file__)
FIVE_BACKENDS = ("mllm-claude", "mllm-gpt", "mllm-hf-qwen", "mllm-hf-deepseek", "mllm-hf-kimi")
REQUEST_MODEL_STRINGS = {
    "mllm-claude": "claude-fable-5-1",
    "mllm-gpt": "gpt-6-astra",
    "mllm-hf-qwen": "Qwen/Qwen3.8-27B:deepinfra",
    "mllm-hf-deepseek": "deepseek-ai/DeepSeek-V4.1-Flash:deepinfra",
    "mllm-hf-kimi": "moonshotai/Kimi-K3:baseten",
}


def _config(**overrides) -> dict:
    config = {
        "provider": "anthropic",
        "model": MODEL,
        "api_key_env": API_KEY_ENV,
        "effort": "high",
        "max_tokens": 4096,
        "prices": {"input_per_mtok": 3.0, "output_per_mtok": 15.0, "cached_input_per_mtok": None,
                   "verified_on": "2026-10-02", "source": "test"},
        "capabilities": {"sampling_supported": False, "effort_levels": ["low", "high", "xhigh"],
                         "summary_supported": True},
        "data_retention_note": "Fake smoke provider keeps nothing (test note).",
    }
    config.update(overrides)
    return config


@pytest.fixture
def config_dir(monkeypatch, tmp_path):
    directory = tmp_path / "configs"
    directory.mkdir()
    shutil.copy(REAL_DEFAULTS, directory / "mllm_defaults.json")
    monkeypatch.setattr(backends, "BACKEND_CONFIG_DIR", directory)

    def write(config=None):
        (directory / f"{CONFIG_NAME}.json").write_text(json.dumps(_config() if config is None else config))

    write()
    return write


class FakeFactory:
    """Adapter factory recording (api_key, config) and returning a ScriptedAdapter."""

    def __init__(self):
        self.calls = []
        self.adapters = []
        self.script = None

    def __call__(self, api_key, config):
        self.calls.append((api_key, config))
        script = self.script if self.script is not None else [_ok_result(config["chunk_len"])]
        adapter = ScriptedAdapter(script, model=config["model"])
        adapter.provider = config["provider"]
        adapter.settings_summary = {
            "requested": {"effort": config["effort"], "max_tokens": config["max_tokens"]},
            "applied": {"effort": config["effort"], "max_tokens": config["max_tokens"]},
            "omitted": {"temperature": "provider-fixed (fake)"},
        }
        self.adapters.append(adapter)
        return adapter


def _ok_result(n, reasoning_tokens=42, **kwargs):
    usage = {"input_tokens": 1000, "output_tokens": 200, "reasoning_tokens": reasoning_tokens,
             "cached_input_tokens": None, "raw": {}}
    return make_provider_result(
        make_chunk_text(n, plan=f"Reach toward the pen (never echo {SECRET})"),
        raw_payload={"echo": f"payload mentions {SECRET}"},
        usage=usage,
        native_thinking_text=f"thinking mentions {SECRET}",
        request_id="req_smoke_123",
        model_returned=MODEL,
        **kwargs,
    )


@pytest.fixture
def factory(monkeypatch, config_dir):
    fake = FakeFactory()
    monkeypatch.setitem(base_mllm.MLLM_BACKENDS, BACKEND_NAME, MllmBackendSpec(BACKEND_NAME, CONFIG_NAME, fake))
    return fake


@pytest.fixture
def api_key(monkeypatch):
    monkeypatch.setenv(API_KEY_ENV, SECRET)
    return SECRET


def _run(tmp_path, *extra, calibration="missing"):
    out = tmp_path / "out"
    argv = ["--backend", BACKEND_NAME, "--out", str(out)]
    if calibration == "missing":
        argv += ["--joint-calibration", str(tmp_path / "no_such_calibration.json")]
    elif calibration is not None:
        argv += ["--joint-calibration", str(calibration)]
    return mllm_smoke.main(argv + list(extra)), out


def _files(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.is_file()]


# =============================================================================
# Spend gate (T-15-32)
# =============================================================================


def test_without_confirm_spend_prints_the_plan_and_makes_no_call(factory, tmp_path, monkeypatch, capsys):
    monkeypatch.delenv(API_KEY_ENV, raising=False)  # the key is never read on a dry run
    code, out = _run(tmp_path)
    printed = capsys.readouterr().out

    assert code == 2
    assert factory.calls == []
    assert BACKEND_NAME in printed and MODEL in printed
    assert "chunk_len: 3" in printed
    assert "effort: high" in printed
    assert "--confirm-spend" in printed and "paid" in printed
    assert not (out / "mllm_calls").exists()


@pytest.mark.parametrize("name", FIVE_BACKENDS)
def test_dry_run_on_each_shipped_backend_names_the_request_model_and_builds_no_client(
        name, monkeypatch, tmp_path, capsys):
    def no_client(*args, **kwargs):
        raise AssertionError("a dry run must not build an SDK client")

    monkeypatch.setattr(mllm_claude, "_make_anthropic_client", no_client)
    monkeypatch.setattr(mllm_gpt, "_make_openai_client", no_client)
    monkeypatch.setattr(base_mllm_chat, "_make_chat_client", no_client)
    for env_name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "HF_TOKEN"):
        monkeypatch.delenv(env_name, raising=False)

    code = mllm_smoke.main(["--backend", name, "--out", str(tmp_path / "out"),
                            "--joint-calibration", str(tmp_path / "missing.json")])
    printed = capsys.readouterr().out
    assert code == 2
    assert REQUEST_MODEL_STRINGS[name] in printed
    assert "chunk_len: 3" in printed


def test_unknown_backend_raises_system_exit_naming_it(factory, tmp_path):
    with pytest.raises(SystemExit, match="mllm-nope"):
        mllm_smoke.main(["--backend", "mllm-nope", "--out", str(tmp_path / "out"), "--confirm-spend"])
    assert factory.calls == []


@pytest.mark.parametrize("value", [None, "   "])
def test_missing_or_blank_key_raises_system_exit_naming_the_variable(factory, tmp_path, monkeypatch, value):
    if value is None:
        monkeypatch.delenv(API_KEY_ENV, raising=False)
    else:
        monkeypatch.setenv(API_KEY_ENV, value)
    with pytest.raises(SystemExit, match=API_KEY_ENV):
        _run(tmp_path, "--confirm-spend")
    assert factory.calls == []


def test_effort_outside_the_configured_levels_is_rejected_before_any_call(factory, api_key, tmp_path):
    with pytest.raises(SystemExit, match="effort"):
        _run(tmp_path, "--effort", "ultra", "--confirm-spend")
    assert factory.calls == []


def test_chunk_len_below_one_is_rejected(factory, api_key, tmp_path):
    with pytest.raises(SystemExit):
        _run(tmp_path, "--chunk-len", "0", "--confirm-spend")
    assert factory.calls == []


# =============================================================================
# Confirmed run (fake adapter, real base/collector/recorder)
# =============================================================================


def test_confirmed_run_makes_one_call_writes_the_sidecar_and_passes(factory, api_key, tmp_path, capsys):
    code, out = _run(tmp_path, "--confirm-spend")
    printed = capsys.readouterr().out

    assert code == 0
    (adapter,) = factory.adapters
    assert adapter.calls == 1
    ((key, config),) = factory.calls
    assert key == SECRET
    assert config["chunk_len"] == 3

    sidecar = out / "mllm_calls" / "0000"
    assert f"Saved → {sidecar}" in printed
    names = {p.relative_to(sidecar).as_posix() for p in _files(sidecar)}
    assert {"prompt.txt", "request_meta.json", "response_raw.json", "thinking.txt",
            "depth.npy", "depth_heatmap.png", "depth_grid.txt"} <= names
    assert {"images/wrist.png", "images/left.png", "images/right.png", "images/depth.png"} <= names

    assert "PASS schema honoured: 3/3 valid steps" in printed
    assert "PASS reasoning tokens reported: 42" in printed
    assert "PASS stop reason: complete" in printed
    assert "api latency:" in printed
    assert "estimated cost: $" in printed
    assert "request id: req_smoke_123" in printed
    assert "settings requested:" in printed and "settings applied:" in printed
    assert "settings omitted:" in printed
    assert "RESULT: PASS" in printed


def test_confirmed_run_sends_four_labelled_images_and_the_depth_grid(factory, api_key, tmp_path):
    _run(tmp_path, "--confirm-spend")
    (request,) = factory.adapters[0].requests
    assert [role for role, _data in request.images] == ["wrist", "left", "right", "depth"]
    assert len({data for _role, data in request.images}) == 4  # distinct synthetic images
    assert "DEPTH GRID" in request.user_text
    assert "Move the arm slightly toward the pen" in request.user_text
    assert request.schema["properties"]["actions"]["description"].startswith("Exactly 3 steps")
    meta = json.loads((tmp_path / "out" / "mllm_calls" / "0000" / "request_meta.json").read_text())
    assert meta["depth"]["status"] == "ok"
    assert meta["capture"]["left_image_source_sent"] == "raw"


@pytest.mark.parametrize("presentation,depth_image,grid_text", [
    ("heatmap", True, False),
    ("grid", False, True),
    ("both", True, True),
])
def test_depth_presentation_is_honoured(factory, config_dir, api_key, tmp_path, presentation, depth_image,
                                        grid_text):
    config_dir(_config(depth_presentation=presentation))
    _run(tmp_path, "--confirm-spend")
    (request,) = factory.adapters[0].requests
    assert ("depth" in dict(request.images)) is depth_image
    assert ("DEPTH GRID" in request.user_text) is grid_text
    sidecar = tmp_path / "out" / "mllm_calls" / "0000"
    assert (sidecar / "depth_grid.txt").exists() and (sidecar / "depth.npy").exists()


def test_the_secret_appears_nowhere_in_stdout_or_any_written_file(factory, api_key, tmp_path, capsys):
    _run(tmp_path, "--confirm-spend")
    assert SECRET not in capsys.readouterr().out
    files = _files(tmp_path / "out")
    assert files
    for path in files:
        assert SECRET.encode() not in path.read_bytes(), path


def test_effort_override_applies_to_this_run_only_and_is_printed(factory, api_key, tmp_path, capsys):
    code, _out = _run(tmp_path, "--effort", "xhigh", "--confirm-spend")
    printed = capsys.readouterr().out

    assert code == 0
    ((_key, config),) = factory.calls
    assert config["effort"] == "xhigh"
    requested = next(line for line in printed.splitlines() if line.startswith("settings requested:"))
    assert '"effort": "xhigh"' in requested
    assert "effort: xhigh (--effort override; config has high)" in printed
    reloaded = base_mllm_build.load_mllm_backend_config(CONFIG_NAME, BACKEND_NAME)
    assert reloaded["effort"] == "high" and reloaded["chunk_len"] == 20


def test_chunk_len_flag_overrides_the_default_of_three(factory, api_key, tmp_path, capsys):
    code, _out = _run(tmp_path, "--chunk-len", "5", "--confirm-spend")
    assert code == 0
    ((_key, config),) = factory.calls
    assert config["chunk_len"] == 5
    assert "PASS schema honoured: 5/5 valid steps" in capsys.readouterr().out


def test_unreported_reasoning_tokens_warn_but_still_pass(factory, api_key, tmp_path, capsys):
    factory.script = [_ok_result(3, reasoning_tokens=None)]
    code, _out = _run(tmp_path, "--confirm-spend")
    printed = capsys.readouterr().out
    assert code == 0
    assert "WARN reasoning tokens not reported" in printed


def test_short_chunk_fails_the_schema_check(factory, api_key, tmp_path, capsys):
    factory.script = [make_provider_result(make_chunk_text(2))]
    code, _out = _run(tmp_path, "--confirm-spend")
    printed = capsys.readouterr().out
    assert code == 1
    assert "FAIL schema not honoured: 2/3 valid steps" in printed
    assert "RESULT: FAIL" in printed


def test_reasoning_extraction_refusal_fails_with_an_explicit_message(factory, api_key, tmp_path, capsys):
    refusal = make_provider_result(None, stop_reason="refusal")
    refusal.stop["refusal_category"] = "reasoning_extraction"
    factory.script = [refusal]
    code, _out = _run(tmp_path, "--confirm-spend")
    printed = capsys.readouterr().out
    assert code == 1
    assert "FAIL stop reason: refusal" in printed
    assert "FAIL refusal category reasoning_extraction" in printed
    assert "lint_prompt_text" in printed


def test_provider_exception_fails_and_is_redacted(factory, api_key, tmp_path, capsys):
    factory.script = [RuntimeError(f"401 Unauthorized: bearer {SECRET}")]
    code, out = _run(tmp_path, "--confirm-spend")
    printed = capsys.readouterr().out
    assert code == 1
    assert "FAIL stop reason: error" in printed
    assert "RuntimeError" in printed
    assert SECRET not in printed
    for path in _files(out):
        assert SECRET.encode() not in path.read_bytes(), path


# =============================================================================
# Joint limits
# =============================================================================


def _prompt(tmp_path) -> str:
    return (tmp_path / "out" / "mllm_calls" / "0000" / "prompt.txt").read_text()


def _settings(chunk_len=3) -> dict:
    settings = base_mllm_build.load_mllm_backend_config(CONFIG_NAME, BACKEND_NAME)
    settings["chunk_len"] = chunk_len
    return settings


def test_missing_calibration_warns_and_uses_neutral_limits(factory, api_key, tmp_path, capsys):
    code, _out = _run(tmp_path, "--confirm-spend")
    printed = capsys.readouterr().out
    assert code == 0
    assert "WARNING" in printed and "neutral joint limits" in printed
    expected = build_system_prompt(_settings(), mllm_smoke.NEUTRAL_JOINT_LIMITS_DEG, urdf_kinematics_text())
    assert _prompt(tmp_path).startswith(expected)
    assert mllm_smoke.NEUTRAL_JOINT_LIMITS_DEG["gripper"] == (0.0, 100.0)
    assert {mllm_smoke.NEUTRAL_JOINT_LIMITS_DEG[j] for j in action_contract.JOINT_ORDER if j != "gripper"} == {
        (-90.0, 90.0)}


def test_joint_calibration_file_is_loaded_through_action_contract(factory, api_key, tmp_path,
                                                                  mock_calibration_file, capsys):
    code, _out = _run(tmp_path, "--confirm-spend", calibration=mock_calibration_file)
    assert code == 0
    assert "neutral joint limits" not in capsys.readouterr().out
    limits = action_contract.load_joint_limits_deg(mock_calibration_file)
    assert _prompt(tmp_path).startswith(build_system_prompt(_settings(), limits, urdf_kinematics_text()))


def test_include_urdf_false_leaves_the_urdf_block_out(factory, config_dir, api_key, tmp_path):
    config_dir(_config(include_urdf=False))
    _run(tmp_path, "--confirm-spend")
    assert "ARM KINEMATICS (URDF)" not in _prompt(tmp_path)


# =============================================================================
# Module hygiene
# =============================================================================


def test_module_imports_no_provider_sdk_and_uses_the_registered_factory():
    source = MODULE_PATH.read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(alias.name.split(".")[0] not in ("anthropic", "openai") for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] not in ("anthropic", "openai")
    assert "adapter_factory" in source
    assert "--confirm-spend" in ast.get_docstring(tree)
