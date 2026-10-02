"""Tests for the shared MLLM hardware builder and backend-config validator (Plan 15-06).

`vla_bridge.backends.base_mllm_build` (load_mllm_backend_config,
build_mllm_backend). Every MLLM backend (Claude, GPT, HF) builds through this
one function, so these tests pin its fail-closed ordering (D-00b): every
config, key, camera and depth failure raises `SystemExit` BEFORE a robot is
constructed, and the key never reaches a printed line, a repr, a record or a
sidecar.

Nothing here touches real hardware, a real camera, the depth service or a
provider API: the robot, cameras and depth client are fakes patched into the
module under test, and the adapter is `fake_providers.ScriptedAdapter`.
"""

import ast
import json
import math
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from conftest import MockRobot
from fake_providers import FakeStereoCamera, FakeWristCap, ScriptedAdapter, make_chunk_text
from vla_bridge import backends, safety_validator
from vla_bridge.action_contract import JOINT_ORDER
from vla_bridge.backends import BackendHandles, base_mllm
from vla_bridge.backends.base_mllm import MllmActionSourceBase, MllmBackendSpec
from vla_bridge.mllm_prompt import build_system_prompt
from vla_bridge.mllm_urdf import urdf_kinematics_text

from vla_bridge.backends import base_mllm_build

BACKEND_NAME = "mllm-fake-build"
CONFIG_NAME = "mllm_fake"
API_KEY_ENV = "SOARM_TEST_MLLM_BUILD_KEY"
SECRET = "sk-test-SECRET-mllm-build-0123456789abcdef"
PASTED_KEY = "sk-ant-api03-PASTEDKEY-abcdefghijklmnop0123456789"
CAMERA_NAMES = {5: "wrist", 3: "overhead"}
JOINT_LIMITS_DEG = {joint: (-180.0, 180.0) for joint in JOINT_ORDER}
JOINT_LIMITS_DEG["gripper"] = (0.0, 100.0)
RETENTION_NOTE = "Fake provider keeps request data for 30 days (test note)."
REAL_DEFAULTS = backends.BACKEND_CONFIG_DIR / "mllm_defaults.json"
MODULE_PATH = Path(__file__).resolve().parent / "vla_bridge" / "backends" / "base_mllm_build.py"


def _valid_config(**overrides) -> dict:
    config = {
        "provider": "anthropic",
        "model": "fake-model-1",
        "api_key_env": API_KEY_ENV,
        "effort": "high",
        "max_tokens": 4096,
        "chunk_len": 3,
        "prices": {
            "input_per_mtok": 3.0,
            "output_per_mtok": 15.0,
            "cached_input_per_mtok": 0.3,
            "verified_on": "2026-10-02",
            "source": "test",
        },
        "capabilities": {
            "sampling_supported": False,
            "effort_levels": ["low", "medium", "high"],
            "summary_supported": True,
        },
        "data_retention_note": RETENTION_NOTE,
    }
    config.update(overrides)
    return config


# --- fixtures ----------------------------------------------------------------------------------


@pytest.fixture
def config_dir(monkeypatch, tmp_path):
    """Point backend config loading at a tmp dir holding the REAL shared
    defaults plus a generated per-backend file; returns a writer."""
    directory = tmp_path / "configs"
    directory.mkdir()
    shutil.copy(REAL_DEFAULTS, directory / "mllm_defaults.json")
    monkeypatch.setattr(backends, "BACKEND_CONFIG_DIR", directory)

    def write(config=None):
        config = _valid_config() if config is None else config
        path = directory / f"{CONFIG_NAME}.json"
        path.write_text(config if isinstance(config, str) else json.dumps(config))
        return path

    write()
    return write


@pytest.fixture
def factory_calls():
    return []


@pytest.fixture
def fake_spec(monkeypatch, factory_calls):
    """Register a fake MLLM backend spec whose factory returns a ScriptedAdapter."""

    def factory(api_key, config):
        factory_calls.append((api_key, config))
        adapter = ScriptedAdapter([make_chunk_text(3)], model=config["model"])
        adapter.provider = config["provider"]
        return adapter

    spec = MllmBackendSpec(BACKEND_NAME, CONFIG_NAME, factory)
    monkeypatch.setitem(base_mllm.MLLM_BACKENDS, BACKEND_NAME, spec)
    return spec


@pytest.fixture
def api_key(monkeypatch):
    monkeypatch.setenv(API_KEY_ENV, SECRET)
    return SECRET


class RecordingMockRobot(MockRobot):
    instances = []
    events = None

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.connect_kwargs = None
        self.calibration_written = False
        self.bus.write_calibration = self._write_calibration
        RecordingMockRobot.instances.append(self)
        if RecordingMockRobot.events is not None:
            RecordingMockRobot.events.append("robot")

    def _write_calibration(self, calibration):
        self.calibration_written = True

    def connect(self, calibrate: bool = False) -> None:
        self.connect_kwargs = dict(calibrate=calibrate)
        self.connected = True


@pytest.fixture
def hardware(monkeypatch):
    """Patch SO101Follower, StereoSplitCamera and cv2.VideoCapture in the module under test."""
    opened = {"events": []}
    RecordingMockRobot.instances = []
    RecordingMockRobot.events = opened["events"]
    monkeypatch.setattr(base_mllm_build, "SO101Follower", RecordingMockRobot)

    def fake_stereo(index=1):
        opened["stereo_index"] = index
        opened["stereo"] = FakeStereoCamera()
        return opened["stereo"]

    def fake_video_capture(index):
        opened["wrist_index"] = index
        opened["wrist"] = FakeWristCap()
        return opened["wrist"]

    monkeypatch.setattr(base_mllm_build, "StereoSplitCamera", fake_stereo)
    monkeypatch.setattr(base_mllm_build.cv2, "VideoCapture", fake_video_capture)
    yield opened
    RecordingMockRobot.events = None


class FakeDepthCameraClient:
    instances = []
    events = None
    fail_ready = None

    def __init__(self, calibration, endpoint):
        self.calibration = calibration
        self.endpoint = endpoint
        self.ready_checks = 0
        self.calibration_metadata = {"image_size": [800, 600], "baseline_m": 0.06}
        FakeDepthCameraClient.instances.append(self)

    def check_ready(self):
        self.ready_checks += 1
        if FakeDepthCameraClient.events is not None:
            FakeDepthCameraClient.events.append("depth.check_ready")
        if FakeDepthCameraClient.fail_ready is not None:
            raise FakeDepthCameraClient.fail_ready
        return {"status": "ready"}


@pytest.fixture
def fake_depth(monkeypatch, hardware):
    FakeDepthCameraClient.instances = []
    FakeDepthCameraClient.events = hardware["events"]
    FakeDepthCameraClient.fail_ready = None
    loaded = []

    def fake_load_calibration(path):
        loaded.append(path)
        return SimpleNamespace(sentinel="calibration", path=path)

    monkeypatch.setattr(base_mllm_build, "DepthCameraClient", FakeDepthCameraClient)
    monkeypatch.setattr(base_mllm_build, "load_calibration", fake_load_calibration)
    yield SimpleNamespace(cls=FakeDepthCameraClient, loaded=loaded)
    FakeDepthCameraClient.events = None
    FakeDepthCameraClient.fail_ready = None


def _args(tmp_path, **overrides):
    defaults = dict(
        port="/dev/tty.usbmodem1",
        robot_id="soarm_follower_02",
        instruction="pick up the pen",
        stereo_camera_index=1,
        out=tmp_path / "out",
        depth_endpoint=None,
        depth_calibration=None,
        server_address=None,
        checkpoint=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _build(tmp_path, **overrides):
    return base_mllm_build.build_mllm_backend(
        BACKEND_NAME, _args(tmp_path, **overrides), dict(CAMERA_NAMES), JOINT_LIMITS_DEG
    )


def _all_bytes_under(root: Path) -> list[tuple[Path, bytes]]:
    return [(p, p.read_bytes()) for p in root.rglob("*") if p.is_file()]


# =============================================================================
# load_mllm_backend_config
# =============================================================================


def test_config_valid_returns_merged_settings(config_dir):
    settings = base_mllm_build.load_mllm_backend_config(CONFIG_NAME, BACKEND_NAME)

    assert settings["provider"] == "anthropic"
    assert settings["model"] == "fake-model-1"
    assert settings["api_key_env"] == API_KEY_ENV
    assert settings["chunk_len"] == 3  # per-backend override wins
    assert settings["step_dt_s"] == 0.785  # inherited from the shared defaults
    assert settings["timeout_s"] == 900
    assert settings["prices"]["input_per_mtok"] == 3.0


def test_config_missing_per_backend_file_names_the_path(config_dir):
    (backends.BACKEND_CONFIG_DIR / f"{CONFIG_NAME}.json").unlink()
    with pytest.raises(SystemExit, match=f"backend_configs/{CONFIG_NAME}.json"):
        base_mllm_build.load_mllm_backend_config(CONFIG_NAME, BACKEND_NAME)


def _without(key):
    config = _valid_config()
    del config[key]
    return config


def _prices(**overrides):
    prices = dict(_valid_config()["prices"])
    for key, value in overrides.items():
        if value is _DROP:
            prices.pop(key)
        else:
            prices[key] = value
    return prices


def _caps(**overrides):
    caps = dict(_valid_config()["capabilities"])
    caps.update(overrides)
    return caps


_DROP = object()

BAD_CONFIGS = [
    ("provider", _valid_config(provider="google")),
    ("provider", _without("provider")),
    ("model", _valid_config(model="")),
    ("model", _valid_config(model="   ")),
    ("model", _valid_config(model=7)),
    ("api_key_env", _valid_config(api_key_env="anthropic_api_key")),
    ("api_key_env", _without("api_key_env")),
    ("effort", _valid_config(effort="ultra")),
    ("effort", _valid_config(effort="")),
    ("max_tokens", _valid_config(max_tokens=1000)),
    ("max_tokens", _valid_config(max_tokens=True)),
    ("max_tokens", _valid_config(max_tokens=4096.0)),
    ("max_tokens", _without("max_tokens")),
    ("prices", _without("prices")),
    ("prices", _valid_config(prices="cheap")),
    ("prices.input_per_mtok", _valid_config(prices=_prices(input_per_mtok=_DROP))),
    ("prices.output_per_mtok", _valid_config(prices=_prices(output_per_mtok=_DROP))),
    ("prices.input_per_mtok", _valid_config(prices=_prices(input_per_mtok=-1.0))),
    ("prices.output_per_mtok", _valid_config(prices=_prices(output_per_mtok=math.inf))),
    ("prices.input_per_mtok", _valid_config(prices=_prices(input_per_mtok=True))),
    ("prices.cached_input_per_mtok", _valid_config(prices=_prices(cached_input_per_mtok=-0.1))),
    ("prices.verified_on", _valid_config(prices=_prices(verified_on=" "))),
    ("prices.verified_on", _valid_config(prices=_prices(verified_on=_DROP))),
    ("capabilities", _without("capabilities")),
    ("capabilities.sampling_supported", _valid_config(capabilities=_caps(sampling_supported="no"))),
    ("capabilities.summary_supported", _valid_config(capabilities=_caps(summary_supported="yes"))),
    ("capabilities.effort_levels", _valid_config(capabilities=_caps(effort_levels="high"))),
    ("data_retention_note", _valid_config(data_retention_note="  ")),
    ("data_retention_note", _without("data_retention_note")),
]


@pytest.mark.parametrize("field,config", BAD_CONFIGS, ids=[f"{i}-{f}" for i, (f, _c) in enumerate(BAD_CONFIGS)])
def test_config_rejects_bad_field_naming_it(config_dir, field, config):
    config_dir(config)
    with pytest.raises(SystemExit) as excinfo:
        base_mllm_build.load_mllm_backend_config(CONFIG_NAME, BACKEND_NAME)
    assert f"'{field}'" in str(excinfo.value)


def test_config_rejects_a_pasted_key_without_echoing_it(config_dir):
    config_dir(_valid_config(api_key_env=PASTED_KEY))
    with pytest.raises(SystemExit) as excinfo:
        base_mllm_build.load_mllm_backend_config(CONFIG_NAME, BACKEND_NAME)
    message = str(excinfo.value)
    assert "api_key_env" in message
    assert PASTED_KEY not in message
    assert "PASTEDKEY" not in message


def test_config_accepts_null_effort_null_cached_price_and_no_effort_levels(config_dir):
    config_dir(_valid_config(effort=None, prices=_prices(cached_input_per_mtok=None),
                             capabilities={"sampling_supported": True}))
    settings = base_mllm_build.load_mllm_backend_config(CONFIG_NAME, BACKEND_NAME)
    assert settings["effort"] is None
    assert settings["capabilities"] == {"sampling_supported": True}


def test_config_accepts_any_non_blank_effort_when_effort_levels_is_absent(config_dir):
    config_dir(_valid_config(effort="xhigh", capabilities={"sampling_supported": False}))
    assert base_mllm_build.load_mllm_backend_config(CONFIG_NAME, BACKEND_NAME)["effort"] == "xhigh"


def test_config_still_runs_the_shared_settings_validation(config_dir):
    config_dir(_valid_config(chunk_len=0))
    with pytest.raises(SystemExit, match="chunk_len"):
        base_mllm_build.load_mllm_backend_config(CONFIG_NAME, BACKEND_NAME)


# =============================================================================
# build_mllm_backend: fail-closed ordering (D-00b)
# =============================================================================


def test_build_unknown_backend_name_exits_before_robot(config_dir, hardware, api_key, tmp_path):
    with pytest.raises(SystemExit, match="not-registered"):
        base_mllm_build.build_mllm_backend("not-registered", _args(tmp_path), CAMERA_NAMES, JOINT_LIMITS_DEG)
    assert RecordingMockRobot.instances == []


def test_build_invalid_config_exits_before_adapter_and_robot(config_dir, fake_spec, factory_calls,
                                                            hardware, api_key, tmp_path):
    config_dir(_valid_config(provider="google"))
    with pytest.raises(SystemExit, match="provider"):
        _build(tmp_path)
    assert factory_calls == []
    assert RecordingMockRobot.instances == []


def test_build_unset_key_exits_with_env_name_before_robot(config_dir, fake_spec, factory_calls,
                                                          hardware, monkeypatch, tmp_path):
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    with pytest.raises(SystemExit, match=API_KEY_ENV):
        _build(tmp_path)
    assert RecordingMockRobot.instances == []
    assert factory_calls == []


def test_build_blank_key_exits_with_env_name_before_robot(config_dir, fake_spec, factory_calls,
                                                          hardware, monkeypatch, tmp_path):
    monkeypatch.setenv(API_KEY_ENV, "   ")
    with pytest.raises(SystemExit, match=API_KEY_ENV):
        _build(tmp_path)
    assert RecordingMockRobot.instances == []
    assert factory_calls == []


def test_build_without_wrist_camera_exits_before_robot(config_dir, fake_spec, hardware, api_key, tmp_path):
    with pytest.raises(SystemExit, match="wrist"):
        base_mllm_build.build_mllm_backend(BACKEND_NAME, _args(tmp_path), {3: "overhead"}, JOINT_LIMITS_DEG)
    assert RecordingMockRobot.instances == []


@pytest.mark.parametrize("which", ["endpoint-only", "calibration-only"])
def test_build_with_only_one_depth_flag_exits_before_robot(config_dir, fake_spec, fake_depth,
                                                           api_key, tmp_path, which):
    overrides = ({"depth_endpoint": "https://depth.example/depth"} if which == "endpoint-only"
                 else {"depth_calibration": tmp_path / "cal.json"})
    with pytest.raises(SystemExit, match="together"):
        _build(tmp_path, **overrides)
    assert RecordingMockRobot.instances == []
    assert fake_depth.cls.instances == []


def test_build_depth_not_ready_exits_before_robot_naming_the_error(config_dir, fake_spec, fake_depth,
                                                                    api_key, tmp_path, capsys):
    fake_depth.cls.fail_ready = ConnectionError(f"depth service down (key {SECRET})")
    with pytest.raises(SystemExit) as excinfo:
        _build(tmp_path, depth_endpoint="https://depth.example/depth", depth_calibration=tmp_path / "cal.json")
    message = str(excinfo.value)
    assert "ConnectionError" in message and "depth service down" in message
    assert SECRET not in message
    assert SECRET not in capsys.readouterr().out
    assert RecordingMockRobot.instances == []


# =============================================================================
# build_mllm_backend: construction
# =============================================================================


def test_build_constructs_robot_cameras_and_handles(config_dir, fake_spec, hardware, api_key, tmp_path):
    handles = _build(tmp_path, stereo_camera_index=4)

    assert isinstance(handles, BackendHandles)
    (robot,) = RecordingMockRobot.instances
    assert handles.robot is robot
    assert robot.config.port == "/dev/tty.usbmodem1"
    assert robot.config.id == "soarm_follower_02"
    assert robot.config.max_relative_target == safety_validator.MAX_RELATIVE_TARGET_DEG
    assert robot.connect_kwargs == dict(calibrate=False)
    assert robot.calibration_written is True
    assert hardware["stereo_index"] == 4
    assert hardware["wrist_index"] == 5
    assert handles.stereo_camera is hardware["stereo"]
    assert handles.skip_camera_names == frozenset({"overhead"})
    assert handles.shutdown is None


def test_build_wires_the_shared_base_with_label_recorder_secrets_and_urdf(config_dir, fake_spec, factory_calls,
                                                                         hardware, api_key, tmp_path):
    handles = _build(tmp_path)
    source = handles.action_source

    assert isinstance(source, MllmActionSourceBase)
    assert source.label == BACKEND_NAME
    assert source.recorder.root == tmp_path / "out" / "mllm_calls"
    assert source.recorder.prices["input_per_mtok"] == 3.0
    assert SECRET in source._secrets
    assert SECRET in source.recorder._secrets
    assert source.urdf_text == urdf_kinematics_text()
    settings = base_mllm_build.load_mllm_backend_config(CONFIG_NAME, BACKEND_NAME)
    assert source.system_prompt == build_system_prompt(settings, JOINT_LIMITS_DEG, urdf_kinematics_text())
    assert source.collector.wrist_cap is hardware["wrist"]
    assert source.collector.stereo_camera is hardware["stereo"]
    ((key, config),) = factory_calls
    assert key == SECRET
    assert config == settings
    assert source.adapter.model == "fake-model-1"


def test_build_omits_urdf_text_when_include_urdf_is_false(config_dir, fake_spec, hardware, api_key, tmp_path):
    config_dir(_valid_config(include_urdf=False))
    source = _build(tmp_path).action_source
    assert source.urdf_text is None
    assert "ARM KINEMATICS (URDF)" not in source.system_prompt


def test_build_with_depth_checks_ready_once_before_robot_and_writes_calibration(config_dir, fake_spec, fake_depth,
                                                                              api_key, tmp_path):
    calibration_path = tmp_path / "cal.json"
    handles = _build(tmp_path, depth_endpoint="https://depth.example/depth", depth_calibration=calibration_path)

    (client,) = fake_depth.cls.instances
    assert fake_depth.loaded == [calibration_path]
    assert client.calibration.path == calibration_path
    assert client.endpoint == "https://depth.example/depth"
    assert client.ready_checks == 1
    assert fake_depth.cls.events.index("depth.check_ready") < fake_depth.cls.events.index("robot")
    written = json.loads((tmp_path / "out" / "depth_calibration.json").read_text())
    assert written == client.calibration_metadata
    assert handles.action_source.collector.depth_client is client


def test_build_without_depth_passes_none_and_warns(config_dir, fake_spec, fake_depth, api_key, tmp_path, capsys):
    handles = _build(tmp_path)
    out = capsys.readouterr().out
    assert handles.action_source.collector.depth_client is None
    assert fake_depth.cls.instances == []
    assert "WARNING" in out and "no depth image will be sent" in out
    assert not (tmp_path / "out" / "depth_calibration.json").exists()


def test_build_prints_operator_notice_with_provider_model_and_retention(config_dir, fake_spec, hardware,
                                                                       api_key, tmp_path, capsys):
    _build(tmp_path)
    out = capsys.readouterr().out
    notice = [line for line in out.splitlines() if "NOTICE" in line]
    assert notice, out
    text = "\n".join(out.splitlines()[out.splitlines().index(notice[0]):])
    assert "anthropic" in text and "fake-model-1" in text
    assert RETENTION_NOTE in text
    assert SECRET not in out


def test_build_warns_when_wrist_capture_did_not_open(config_dir, fake_spec, hardware, api_key,
                                                     monkeypatch, tmp_path, capsys):
    closed = FakeWristCap()
    closed.isOpened = lambda: False
    monkeypatch.setattr(base_mllm_build.cv2, "VideoCapture", lambda index: closed)
    _build(tmp_path)
    assert "wrist camera index 5 did not open" in capsys.readouterr().out


def test_built_source_repr_and_details_never_contain_the_key(config_dir, fake_spec, hardware, api_key, tmp_path):
    source = _build(tmp_path).action_source
    assert SECRET not in repr(source)
    assert SECRET not in repr(source.adapter)
    assert SECRET not in repr(source.recorder)
    assert SECRET not in json.dumps(source.last_details)


def test_built_source_close_releases_the_wrist_capture(config_dir, fake_spec, hardware, api_key, tmp_path):
    source = _build(tmp_path).action_source
    source.close()
    assert hardware["wrist"].released is True


def test_built_source_runs_one_chunk_and_keeps_the_key_out_of_every_artifact(config_dir, fake_spec, hardware,
                                                                            api_key, tmp_path, capsys):
    source = _build(tmp_path).action_source
    joint_state = {joint: 1.0 for joint in JOINT_ORDER}

    action, version = source.get_action(joint_state, "pick up the pen")

    assert version == f"{BACKEND_NAME}@mllm-chunk"
    assert source.adapter.calls == 1
    call = source.last_details["mllm_call"]
    assert call["sidecar_dir"] == "mllm_calls/0000"
    assert math.isclose(call["cost_usd_est"], (100 * 3.0 + 50 * 15.0) / 1e6)
    assert [img["role"] for img in call["images"]] == ["wrist", "left", "right"]
    assert SECRET not in json.dumps(source.last_details)
    assert SECRET not in capsys.readouterr().out
    files = _all_bytes_under(tmp_path / "out")
    assert any(p.name == "prompt.txt" for p, _data in files)
    for path, data in files:
        assert SECRET.encode() not in data, path


# =============================================================================
# Module hygiene
# =============================================================================


def test_builder_module_imports_no_provider_sdk_at_top_level():
    tree = ast.parse(MODULE_PATH.read_text())
    top = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top.add(node.module.split(".")[0])
    assert not {"anthropic", "openai", "google"} & top


def test_builder_module_is_not_registered_as_a_backend():
    for name, build in backends.BACKEND_REGISTRY.items():
        assert build.__module__ != "vla_bridge.backends.base_mllm_build", name
