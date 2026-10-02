"""Cross-backend conformance for the five registered MLLM backends (Plan 15-09, PIPE-02).

The proof of Phase 15 success criterion 1: every MLLM backend (mllm-claude,
mllm-gpt, mllm-hf-qwen, mllm-hf-deepseek, mllm-hf-kimi) plugs into the shared
loop through the shared registry and builder, and reuses the shared system
prompt, schema, schema hash, prompt template version, chunk length and
labelled-image payload order UNCHANGED. Each backend is built through the
REAL `BACKEND_REGISTRY[name]` -> `build_mllm_backend` path, so a backend that
quietly built its own prompt or schema would fail here.

Also here:
  - hygiene scans (D-15, research C4, Security Domain V10/V14): no MLLM module
    passes `tools`/`tool_choice`, calls eval/exec or imports a provider SDK
    at module level; no shipped MLLM config holds a key-shaped value; every
    prompt, schema dump, depth note, URDF text and json_object-mode appended
    schema block is free of the reasoning-extraction and movement-primitive
    vocabulary;
  - config inheritance (D-28): every backend inherits the shared depth, left
    image, URDF and safety-cap disclosure defaults unless a documented
    override exists;
  - documentation assertions (research C9, D-06, D-23): REQUIREMENTS.md,
    ROADMAP.md, COMMANDS.md and the backends package docstring describe the
    five-backend reality.

Nothing here touches hardware or the network: the robot, cameras and every
provider SDK client are fakes (the client factories are monkeypatched), and
API keys are sentinel values.
"""

import ast
import base64
import json
import pkgutil
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

import test_backend_mllm_claude as claude_fakes
import test_backend_mllm_gpt as gpt_fakes
import test_backend_mllm_hf as hf_fakes
from fake_providers import make_chunk_text, make_depth_array, png_bytes
from test_mllm_build import RecordingMockRobot, hardware  # noqa: F401  (pytest fixture import)
from test_mllm_schema import GOLDEN_SCHEMA_SHA256
from vla_bridge import action_contract, backends
from vla_bridge.action_contract import JOINT_ORDER
from vla_bridge.backends import (
    BACKEND_REGISTRY,
    GRPC_BRIDGE_BACKENDS,
    base_mllm,
    base_mllm_build,
    base_mllm_chat,
    mllm_claude,
    mllm_gpt,
)
from vla_bridge.backends.base_mllm import IMAGE_ORDER, MllmActionSourceBase, ProviderRequest
from vla_bridge.mllm_inputs import (
    DEPTH_BOTH_HEATMAP_LINE,
    DEPTH_HEATMAP_NOTE,
    DEPTH_UNAVAILABLE_NOTE,
    depth_grid_text,
)
from vla_bridge.mllm_prompt import (
    PROMPT_TEMPLATE_VERSION,
    build_user_text,
    lint_prompt_text,
)
from vla_bridge.mllm_schema import SCHEMA_NAME, build_action_chunk_schema, lint_schema_subset, schema_sha256
from vla_bridge.mllm_urdf import urdf_kinematics_text

CONTROL_DIR = Path(__file__).resolve().parent
REPO_ROOT = CONTROL_DIR.parent
VLA_BRIDGE_DIR = CONTROL_DIR / "vla_bridge"
BACKENDS_DIR = VLA_BRIDGE_DIR / "backends"
CONFIG_DIR = backends.BACKEND_CONFIG_DIR

FIVE_BACKENDS = ("mllm-claude", "mllm-gpt", "mllm-hf-qwen", "mllm-hf-deepseek", "mllm-hf-kimi")
HF_BACKENDS = ("mllm-hf-qwen", "mllm-hf-deepseek", "mllm-hf-kimi")
BASE_MODULES = ("base_mllm", "base_mllm_build", "base_mllm_chat")
SHIPPED_CHUNK_LEN = 20
KEY_ENVS = {"ANTHROPIC_API_KEY": "sk-ant-api03-CONFORMANCE-SENTINEL-0123456789",
            "OPENAI_API_KEY": "sk-proj-CONFORMANCE-SENTINEL-0123456789",
            "HF_TOKEN": "hf_CONFORMANCE_SENTINEL_0123456789abcdef"}
CAMERA_NAMES = {5: "wrist", 3: "overhead"}
INSTRUCTION = "pick up the pen"

# Shared defaults every backend must inherit unless it overrides them with a notes explanation (D-28).
INHERITED_DEFAULTS = {
    "depth_presentation": "both",
    "left_image_source": "raw",
    "include_urdf": True,
    "disclose_safety_caps": False,
}
ENV_VAR_NAME_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")
KEY_SHAPE_RES = (
    re.compile(r"(?<![A-Za-z0-9])sk-ant-[\w-]{20,}"),
    re.compile(r"(?<![A-Za-z0-9])sk-[\w-]{20,}"),
    re.compile(r"(?<![A-Za-z0-9])AIza[\w-]{20,}"),
    re.compile(r"(?<![A-Za-z0-9])hf_\w{20,}"),
)
SDK_NAMES = {"anthropic", "openai"}


def _mllm_source_files() -> list[Path]:
    files = (sorted(VLA_BRIDGE_DIR.glob("mllm_*.py"))
             + sorted(BACKENDS_DIR.glob("base_mllm*.py"))
             + sorted(BACKENDS_DIR.glob("mllm_*.py")))
    assert files, "no MLLM modules found"
    return files


def _backend_module_files() -> list[Path]:
    return sorted(BACKENDS_DIR.glob("mllm_*.py")) + [BACKENDS_DIR / "base_mllm_chat.py"]


def _shipped_config_files() -> list[Path]:
    return sorted(CONFIG_DIR.glob("mllm_*.json"))


def _config_name(backend_name: str) -> str:
    return base_mllm.MLLM_BACKENDS[backend_name].config_name


def _limits(calibration_file) -> dict:
    return action_contract.load_joint_limits_deg(calibration_file)


# =============================================================================
# Registration (PIPE-02, D-00a)
# =============================================================================


def test_exactly_the_five_mllm_backends_are_registered():
    assert set(base_mllm.MLLM_BACKENDS) == set(FIVE_BACKENDS)
    for name in FIVE_BACKENDS:
        assert name in BACKEND_REGISTRY, name
        assert name not in GRPC_BRIDGE_BACKENDS, name
        assert base_mllm.MLLM_BACKENDS[name].name == name


def test_no_base_module_callable_is_registered():
    for name, build in BACKEND_REGISTRY.items():
        assert not build.__module__.startswith("vla_bridge.backends.base_"), (name, build.__module__)


def test_discovery_skips_every_mllm_base_module_and_imports_every_backend_module(monkeypatch):
    listed = {info.name for info in pkgutil.iter_modules(backends.__path__)}
    assert set(BASE_MODULES) <= listed
    imported = []
    monkeypatch.setattr(backends, "importlib", SimpleNamespace(import_module=imported.append))
    backends._discover_backends()

    skipped = sorted(listed - {name.rsplit(".", 1)[1] for name in imported})
    for module_name in BASE_MODULES:
        assert module_name in skipped, module_name
        assert f"vla_bridge.backends.{module_name}" not in imported
    for module_name in ("mllm_claude", "mllm_gpt", "mllm_hf_qwen", "mllm_hf_deepseek", "mllm_hf_kimi"):
        assert f"vla_bridge.backends.{module_name}" in imported, module_name


# =============================================================================
# Success criterion 1: one prompt, one schema, one payload order across all five
# =============================================================================


def _fake_client_for(name: str):
    """A scripted fake SDK client for `name` (no network) and an extractor of what it was sent."""
    if name == "mllm-claude":
        client = claude_fakes._client([claude_fakes.make_message(text=make_chunk_text(SHIPPED_CHUNK_LEN))])
        return client, lambda: _claude_payload(client.state.calls)
    if name == "mllm-gpt":
        client = gpt_fakes._client([gpt_fakes.make_response(text=make_chunk_text(SHIPPED_CHUNK_LEN))])
        return client, lambda: _gpt_payload(client.state.calls)
    client = hf_fakes.fake_chat_client([hf_fakes.ok_raw(make_chunk_text(SHIPPED_CHUNK_LEN))])
    return client, lambda: _hf_payload(client.state.calls)


def _data_url_parts(url: str) -> tuple[str, str]:
    header, data = url.split(",", 1)
    assert header.startswith("data:") and header.endswith(";base64"), header
    return header[len("data:"):-len(";base64")], data


def _labelled(content, label_of, image_of) -> tuple[list[tuple[str, str, str]], str]:
    """[(label, media_type, base64)] for each label+image pair, plus the trailing user text."""
    body, last = content[:-1], content[-1]
    assert len(body) % 2 == 0, "every image must be preceded by exactly one label"
    pairs = [(label_of(label), *image_of(image)) for label, image in zip(body[0::2], body[1::2])]
    return pairs, label_of(last)


def _claude_payload(calls):
    (kwargs,) = calls
    content = kwargs["messages"][0]["content"]
    images, user_text = _labelled(
        content, lambda block: block["text"],
        lambda block: (block["source"]["media_type"], block["source"]["data"]))
    return {"system": kwargs["system"], "images": images, "user_text": user_text,
            "schema": kwargs["output_config"]["format"]["schema"]}


def _gpt_payload(calls):
    (kwargs,) = calls
    content = kwargs["input"][0]["content"]
    images, user_text = _labelled(
        content, lambda block: block["text"], lambda block: _data_url_parts(block["image_url"]))
    return {"system": kwargs["instructions"], "images": images, "user_text": user_text,
            "schema": kwargs["text"]["format"]["schema"]}


def _hf_payload(calls):
    (kwargs,) = calls
    system, user = kwargs["messages"]
    images, user_text = _labelled(
        user["content"], lambda block: block["text"],
        lambda block: _data_url_parts(block["image_url"]["url"]))
    return {"system": system["content"], "images": images, "user_text": user_text,
            "schema": kwargs["response_format"]["json_schema"]["schema"]}


@pytest.fixture
def five_built(monkeypatch, tmp_path, hardware, mock_calibration_file):  # noqa: F811
    """All five backends built through BACKEND_REGISTRY with fake hardware and fake SDK clients."""
    for env_name, sentinel in KEY_ENVS.items():
        monkeypatch.setenv(env_name, sentinel)
    current = {}
    made = []

    def claude_factory(api_key, timeout_s):
        made.append(("anthropic", api_key))
        return current["client"]

    def gpt_factory(api_key, timeout_s):
        made.append(("openai", api_key))
        return current["client"]

    def chat_factory(api_key, base_url, timeout_s):
        made.append(("huggingface", api_key))
        return current["client"]

    monkeypatch.setattr(mllm_claude, "_make_anthropic_client", claude_factory)
    monkeypatch.setattr(mllm_gpt, "_make_openai_client", gpt_factory)
    monkeypatch.setattr(base_mllm_chat, "_make_chat_client", chat_factory)

    limits = _limits(mock_calibration_file)
    built = {}
    for name in FIVE_BACKENDS:
        client, payload = _fake_client_for(name)
        current["client"] = client
        args = SimpleNamespace(port="/dev/null", robot_id="soarm_follower_02", instruction=INSTRUCTION,
                               stereo_camera_index=1, out=tmp_path / name, depth_endpoint=None,
                               depth_calibration=None)
        handles = BACKEND_REGISTRY[name](args, dict(CAMERA_NAMES), limits)
        built[name] = SimpleNamespace(handles=handles, source=handles.action_source, client=client,
                                      payload=payload)
    yield SimpleNamespace(built=built, made=made, limits=limits)
    for entry in built.values():
        entry.source.close()


def test_all_five_build_through_the_shared_builder_with_their_own_key(five_built):
    assert [provider for provider, _key in five_built.made] == [
        "anthropic", "openai", "huggingface", "huggingface", "huggingface"]
    expected_keys = [KEY_ENVS["ANTHROPIC_API_KEY"], KEY_ENVS["OPENAI_API_KEY"]] + [KEY_ENVS["HF_TOKEN"]] * 3
    assert [key for _provider, key in five_built.made] == expected_keys
    assert len(RecordingMockRobot.instances) == len(FIVE_BACKENDS)
    for name, entry in five_built.built.items():
        assert isinstance(entry.source, MllmActionSourceBase), name
        assert entry.source.label == name
        assert entry.handles.skip_camera_names == frozenset({"overhead"})


def test_success_criterion_1_prompt_schema_version_and_chunk_len_are_identical(five_built):
    sources = [entry.source for entry in five_built.built.values()]

    assert len({source.system_prompt for source in sources}) == 1
    assert len({source.schema_hash for source in sources}) == 1
    assert len({json.dumps(source.schema, sort_keys=True) for source in sources}) == 1
    assert {source.prompt_template_version for source in sources} == {PROMPT_TEMPLATE_VERSION}
    assert {source.chunk_len for source in sources} == {SHIPPED_CHUNK_LEN}
    # Golden hash pinned by Plan 15-01 (test_mllm_schema.GOLDEN_SCHEMA_SHA256).
    assert sources[0].schema_hash == GOLDEN_SCHEMA_SHA256
    assert schema_sha256(sources[0].schema) == GOLDEN_SCHEMA_SHA256
    assert {source.urdf_text for source in sources} == {urdf_kinematics_text()}


def test_one_real_chunk_per_backend_sends_byte_identical_payloads(five_built):
    joint_state = {joint: 1.0 for joint in JOINT_ORDER}
    payloads = {}
    for name, entry in five_built.built.items():
        _action, version = entry.source.get_action(joint_state, INSTRUCTION)
        assert version == f"{name}@mllm-chunk", name
        payloads[name] = entry.payload()

    reference = payloads["mllm-claude"]
    assert [label for label, _media, _data in reference["images"]] == [
        "Image: wrist", "Image: left", "Image: right"]
    assert reference["system"] == five_built.built["mllm-claude"].source.system_prompt
    for name, payload in payloads.items():
        assert payload["system"].encode("utf-8") == reference["system"].encode("utf-8"), name
        assert payload["schema"] == reference["schema"], name
        assert payload["images"] == reference["images"], name
        assert payload["user_text"] == reference["user_text"], name

    records = {name: entry.source.last_details["mllm_call"] for name, entry in five_built.built.items()}
    assert len({r["prompt_sha256"] for r in records.values()}) == 1
    assert len({r["schema_sha256"] for r in records.values()}) == 1
    assert len({r["prompt_template_sha256"] for r in records.values()}) == 1
    assert {r["sidecar_dir"] for r in records.values()} == {"mllm_calls/0000"}


def _four_role_request() -> ProviderRequest:
    return ProviderRequest(
        system_prompt="SYSTEM PROMPT TEXT (conformance)",
        user_text="USER TEXT: joint state and instruction (conformance)",
        images=[(role, png_bytes(40 + i)) for i, role in enumerate(IMAGE_ORDER)],
        schema=build_action_chunk_schema(SHIPPED_CHUNK_LEN),
        schema_name=SCHEMA_NAME,
        image_media_type="image/png",
    )


def _adapter_for(name: str, client):
    config = base_mllm_build.load_mllm_backend_config(_config_name(name), name)
    if name == "mllm-claude":
        return mllm_claude.AnthropicAdapter(client, config)
    if name == "mllm-gpt":
        return mllm_gpt.OpenAIResponsesAdapter(client, config)
    return base_mllm_chat.OpenAIChatAdapter(client, config)


def test_four_role_labelled_image_sequence_is_identical_across_all_five_adapters():
    request = _four_role_request()
    expected = [(f"Image: {role}", "image/png", base64.b64encode(data).decode("ascii"))
                for role, data in request.images]

    for name in FIVE_BACKENDS:
        client, payload = _fake_client_for(name)
        _adapter_for(name, client).call(request)
        sent = payload()
        assert sent["images"] == expected, name
        assert sent["user_text"] == request.user_text, name
        assert sent["system"] == request.system_prompt, name
        assert sent["schema"] == request.schema, name


# =============================================================================
# Hygiene: vocabulary (research C4), schema subset, tools, eval/exec, SDK imports
# =============================================================================


def _example_texts(source, limits) -> dict[str, str]:
    settings = source.settings
    midpoints = {joint: (float(lo) + float(hi)) / 2 for joint, (lo, hi) in limits.items()}
    grid = depth_grid_text(make_depth_array(), settings["depth"]["grid_rows"], settings["depth"]["grid_cols"],
                           settings["depth"]["grid_decimals"])
    texts = {
        "system_prompt": source.system_prompt,
        "schema_dump": json.dumps(source.schema, indent=2),
        "user_text_first_call": build_user_text(midpoints, INSTRUCTION, settings, None, DEPTH_UNAVAILABLE_NOTE),
        "user_text_with_plan": build_user_text(midpoints, INSTRUCTION, settings,
                                               "Open the gripper, then lower toward the pen.",
                                               grid + "\n" + DEPTH_BOTH_HEATMAP_LINE),
        "depth_note_heatmap": DEPTH_HEATMAP_NOTE,
        "depth_note_grid": grid,
        "urdf_text": urdf_kinematics_text(),
    }
    return texts


def test_every_prompt_text_and_schema_dump_passes_the_vocabulary_lint(five_built):
    for name, entry in five_built.built.items():
        for what, text in _example_texts(entry.source, five_built.limits).items():
            assert lint_prompt_text(text) == [], (name, what)
        lint_schema_subset(entry.source.schema)  # raises ValueError on any violation


def test_json_object_mode_appended_schema_block_passes_the_lint():
    for name in HF_BACKENDS:
        config = base_mllm_build.load_mllm_backend_config(_config_name(name), name)
        config["structured_mode"] = "json_object"
        adapter = base_mllm_chat.OpenAIChatAdapter(hf_fakes.fake_chat_client(), config)
        request = _four_role_request()
        system_text = adapter.request_kwargs(request)["messages"][0]["content"]
        appended = system_text[len(request.system_prompt):]
        assert "RESPONSE FORMAT" in appended
        assert lint_prompt_text(appended) == [], name


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def _string_key(node) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _tools_hits(paths) -> list[tuple]:
    banned = {"tools", "tool_choice"}
    hits = []
    for path in paths:
        for node in ast.walk(_parse(path)):
            if isinstance(node, ast.Call):
                hits += [(path.name, node.lineno, kw.arg) for kw in node.keywords if kw.arg in banned]
            elif isinstance(node, ast.Dict):
                hits += [(path.name, node.lineno, _string_key(key)) for key in node.keys
                         if _string_key(key) in banned]
            elif isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store):
                if _string_key(node.slice) in banned:
                    hits.append((path.name, node.lineno, _string_key(node.slice)))
    return hits


def _eval_exec_hits(paths) -> list[tuple]:
    hits = []
    for path in paths:
        for node in ast.walk(_parse(path)):
            if isinstance(node, ast.Name) and node.id in ("eval", "exec"):
                hits.append((path.name, node.lineno, node.id))
            elif isinstance(node, ast.Attribute) and node.attr in ("eval", "exec") and isinstance(
                    node.value, ast.Name) and node.value.id == "builtins":
                hits.append((path.name, node.lineno, node.attr))
    return hits


def test_no_mllm_module_sends_tools_or_tool_choice():
    assert _tools_hits(_mllm_source_files()) == []


def test_no_mllm_module_uses_eval_or_exec():
    assert _eval_exec_hits(_mllm_source_files()) == []


def test_the_source_scans_catch_planted_violations(tmp_path):
    planted = tmp_path / "mllm_planted.py"
    planted.write_text(
        "import openai\n"
        "def call(client, request):\n"
        "    kwargs = {\"model\": \"m\", \"tool_choice\": \"auto\"}\n"
        "    kwargs[\"tools\"] = []\n"
        "    eval(request)\n"
        "    return client.create(tools=[], **kwargs)\n"
    )
    assert sorted(hit[2] for hit in _tools_hits([planted])) == ["tool_choice", "tools", "tools"]
    assert [hit[2] for hit in _eval_exec_hits([planted])] == ["eval"]
    assert "openai" in _top_level_imports(_parse(planted))


def _top_level_imports(tree: ast.Module) -> set[str]:
    names = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def _all_imports(tree: ast.Module) -> set[str]:
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_backend_modules_import_no_provider_sdk_at_module_level():
    files = _backend_module_files() + [BACKENDS_DIR / "base_mllm.py", BACKENDS_DIR / "base_mllm_build.py"]
    assert {path.name for path in files} >= {
        "mllm_claude.py", "mllm_gpt.py", "mllm_hf_qwen.py", "mllm_hf_deepseek.py", "mllm_hf_kimi.py",
        "base_mllm_chat.py"}
    for path in files:
        assert not SDK_NAMES & _top_level_imports(_parse(path)), path.name


def test_base_mllm_imports_neither_lerobot_nor_cv2_nor_any_sdk():
    imports = _all_imports(_parse(BACKENDS_DIR / "base_mllm.py"))
    assert not {"lerobot", "cv2"} & imports
    assert not SDK_NAMES & imports


# =============================================================================
# Config scan: key names only, no key-shaped values, inheritance (T-15-33, D-28)
# =============================================================================


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def test_six_shipped_mllm_config_files_are_present():
    names = {path.stem for path in _shipped_config_files()}
    assert names == {"mllm_defaults"} | {_config_name(name) for name in FIVE_BACKENDS}


def test_per_backend_configs_name_an_env_var_never_a_key():
    for name in FIVE_BACKENDS:
        raw = json.loads((CONFIG_DIR / f"{_config_name(name)}.json").read_text())
        assert ENV_VAR_NAME_RE.match(raw["api_key_env"]), name
    assert {json.loads((CONFIG_DIR / f"{_config_name(n)}.json").read_text())["api_key_env"]
            for n in FIVE_BACKENDS} == set(KEY_ENVS)


def test_no_shipped_mllm_config_contains_a_key_shaped_string():
    for path in _shipped_config_files():
        text = path.read_text()
        for pattern in KEY_SHAPE_RES:
            assert pattern.search(text) is None, (path.name, pattern.pattern)
        for value in _strings(json.loads(text)):
            for pattern in KEY_SHAPE_RES:
                assert pattern.search(value) is None, (path.name, pattern.pattern)


def test_key_shape_patterns_catch_real_shapes_and_spare_the_hf_provider_key():
    for sample in ("sk-ant-api03-abcdefghijklmnopqrstuvwxyz", "sk-proj-abcdefghijklmnopqrstuvwx",
                   "AIzaSyA-abcdefghijklmnopqrstuvwxyz", "hf_abcdefghijklmnopqrstuvwxyz"):
        assert any(p.search(sample) for p in KEY_SHAPE_RES), sample
    for sample in ("hf_provider", "task-complete-holding-position", "HF_TOKEN", "ANTHROPIC_API_KEY"):
        assert not any(p.search(sample) for p in KEY_SHAPE_RES), sample


def test_every_per_backend_config_validates_through_the_shared_loader():
    for name in FIVE_BACKENDS:
        settings = base_mllm_build.load_mllm_backend_config(_config_name(name), name)
        assert settings["chunk_len"] == SHIPPED_CHUNK_LEN, name
        if settings["provider"] == "huggingface":
            base_mllm_chat.validate_chat_config(settings, name)


def test_every_backend_inherits_the_shared_presentation_defaults_or_documents_an_override():
    defaults = json.loads((CONFIG_DIR / "mllm_defaults.json").read_text())
    for key, value in INHERITED_DEFAULTS.items():
        assert defaults[key] == value, key
    for name in FIVE_BACKENDS:
        raw = json.loads((CONFIG_DIR / f"{_config_name(name)}.json").read_text())
        settings = base_mllm_build.load_mllm_backend_config(_config_name(name), name)
        for key, value in INHERITED_DEFAULTS.items():
            if key in raw:
                assert key in raw.get("notes", ""), (name, key, "override without a notes explanation")
            else:
                assert settings[key] == value, (name, key)
