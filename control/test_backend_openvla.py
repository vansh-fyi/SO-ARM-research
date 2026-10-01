"""Tests for the OpenVLA Tier C backend (Plan 14-09).

Covers the Colab-side resident adapter (`vla_bridge/openvla_server.py`, with
the HF download, the remote-code fetch and the 7B model load all mocked) and
the control-side client (`vla_bridge/backends/openvla.py`), with mocked HTTP.
Nothing here touches real hardware, a GPU, or the network.
"""

import base64
import json
import math
import threading
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PIL import Image

from vla_bridge import backends
from vla_bridge import openvla_server
from vla_bridge.action_contract import JOINT_ORDER
from vla_bridge.backends import base_http_adapter, eef_to_joint, openvla, openvla_oft
from vla_bridge.openvla_server import OpenVLAModel, create_app

SO101_CKPT = "toasta18/openvla-so101-cylinder"
SO101_REV = "f00833ce58510a955b981ee7e9ba8e9ef4e2ef2e"
LIBERO_CKPT = "openvla/openvla-7b-finetuned-libero-object"
LIBERO_REV = "287d6cfdf12d07b1449505f66d9bf3550257e9b3"

JOINT_STATE = {
    "shoulder_pan": 10.0,
    "shoulder_lift": -20.0,
    "elbow_flex": 30.0,
    "wrist_flex": 15.0,
    "wrist_roll": 5.0,
    "gripper": 37.0,
}

# predict_action() output for the SO-101 joint-space checkpoint: 5 arm joints
# (degrees), the constant pad slot (index 5), gripper (percent).
SO101_RAW = [12.0, -18.5, 33.0, 20.0, 4.0, -0.996, 41.0]
# predict_action() output for a LIBERO checkpoint: robosuite OSC_POSE delta.
LIBERO_RAW = [0.2, -0.1, 0.3, 0.02, -0.04, 0.06, 0.9]


def _b64_png(value=50, shape=(24, 32, 3)):
    ok, buf = cv2.imencode(".png", np.full(shape, value, dtype=np.uint8))
    assert ok
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _payload(**overrides):
    body = {
        "images": {"wrist": _b64_png(10), "overhead_left": _b64_png(20), "overhead_right": _b64_png(30)},
        "joint_state": dict(JOINT_STATE),
        "instruction": "Pick up the black cylinder and place it on the plate.",
    }
    body.update(overrides)
    return body


# --- Colab-side server ---------------------------------------------------------


def _norm_stats(key):
    return {key: {"action": {"q01": [0.0] * 7, "q99": [1.0] * 7, "mask": [True] * 6 + [False]}}}


class FakeInputs(dict):
    def __init__(self, prompt, image):
        super().__init__(input_ids="ids", pixel_values="pixels")
        self.prompt = prompt
        self.image = image
        self.moved_to = None

    def to(self, device, dtype=None):
        self.moved_to = (device, dtype)
        return self


class FakeProcessor:
    def __init__(self):
        self.calls = []

    def __call__(self, prompt, image):
        inputs = FakeInputs(prompt, image)
        self.calls.append(inputs)
        return inputs


class FakeVLA:
    def __init__(self, norm_stats, raw):
        self.norm_stats = norm_stats
        self.raw = raw
        self.calls = []

    def predict_action(self, unnorm_key=None, do_sample=None, **inputs):
        self.calls.append(dict(unnorm_key=unnorm_key, do_sample=do_sample, inputs=inputs))
        return np.array(self.raw, dtype=np.float64)


@pytest.fixture
def fake_policy(monkeypatch):
    """Mock `_download_pinned` (snapshot_download), `_localize_remote_code`
    (the pinned openvla/openvla-7b code fetch) and `_load_policy` (the
    AutoProcessor / AutoModelForVision2Seq load)."""
    state = SimpleNamespace(loads=[], downloads=[], localized=[], norm_key="so101_pick_cylinder",
                            raw=list(SO101_RAW), extra_keys=(), vla=None, processor=None)

    def download(checkpoint, revision, local_dir):
        state.downloads.append((checkpoint, revision, local_dir))
        return f"{local_dir}/snapshot"

    def localize(model_dir, code_repo=openvla_server.CODE_REPO, code_revision=openvla_server.CODE_REVISION):
        state.localized.append((model_dir, code_repo, code_revision))

    def load_policy(model_dir, attn_implementation):
        stats = _norm_stats(state.norm_key)
        for key in state.extra_keys:
            stats.update(_norm_stats(key))
        state.vla = FakeVLA(stats, state.raw)
        state.processor = FakeProcessor()
        state.loads.append((model_dir, attn_implementation))
        return dict(vla=state.vla, processor=state.processor, device="cuda:0", dtype="bf16")

    monkeypatch.setattr(openvla_server, "_download_pinned", download)
    monkeypatch.setattr(openvla_server, "_localize_remote_code", localize)
    monkeypatch.setattr(openvla_server, "_load_policy", load_policy)
    return state


def test_model_loads_once_across_many_action_requests(fake_policy):
    model = OpenVLAModel(SO101_CKPT, revision=SO101_REV, local_dir="/ckpt")
    http = create_app(model).test_client()
    for _ in range(3):
        assert http.post("/action", json=_payload()).status_code == 200
    assert len(fake_policy.loads) == 1
    assert len(fake_policy.downloads) == 1
    assert len(fake_policy.vla.calls) == 3


def test_pinned_revision_is_downloaded_localized_and_loaded_from_disk(fake_policy):
    model = OpenVLAModel(SO101_CKPT, revision=SO101_REV, local_dir="/ckpt")
    assert fake_policy.downloads == [(SO101_CKPT, SO101_REV, "/ckpt")]
    # The remote code is pinned to the reviewed openvla/openvla-7b revision.
    assert fake_policy.localized == [("/ckpt/snapshot", "openvla/openvla-7b",
                                      "47a0ec7fc4ec123775a391911046cf33cf9ed83f")]
    assert fake_policy.loads == [("/ckpt/snapshot", "sdpa")]
    assert model.model_version == f"{SO101_CKPT}@{SO101_REV[:12]}"
    assert model.metadata["revision"] == SO101_REV
    assert model.metadata["code_revision"] == openvla_server.CODE_REVISION


def test_unpinned_checkpoint_loads_by_repo_id_without_download(fake_policy):
    model = OpenVLAModel(SO101_CKPT, revision=None)
    assert fake_policy.downloads == [] and fake_policy.localized == []
    assert fake_policy.loads[0][0] == SO101_CKPT
    assert model.model_version.endswith("@unpinned")


def test_known_checkpoint_uses_its_profile_unnorm_key_among_many(fake_policy):
    # toasta18's config embeds the 25 base OXE keys plus its own.
    fake_policy.extra_keys = ("bridge_orig", "fractal20220817_data")
    model = OpenVLAModel(SO101_CKPT, revision=SO101_REV)
    assert model.unnorm_key == "so101_pick_cylinder"
    assert model.metadata["action_space"] == "absolute_joint_degrees_so101_7d"


def test_unknown_checkpoint_needs_an_unambiguous_unnorm_key(fake_policy):
    fake_policy.norm_key = "my_task"
    fake_policy.extra_keys = ("bridge_orig",)
    with pytest.raises(ValueError, match="unnorm"):
        OpenVLAModel("someone/openvla-custom", revision=None)
    model = OpenVLAModel("someone/openvla-custom", revision=None, unnorm_key="my_task")
    assert model.unnorm_key == "my_task"
    assert model.metadata["action_space"] == "unknown"
    with pytest.raises(ValueError, match="unnorm"):
        OpenVLAModel("someone/openvla-custom", revision=None, unnorm_key="not_there")


def test_single_key_checkpoint_auto_selects_it(fake_policy):
    fake_policy.norm_key = "libero_object"
    fake_policy.raw = list(LIBERO_RAW)
    model = OpenVLAModel(LIBERO_CKPT, revision=LIBERO_REV)
    assert model.unnorm_key == "libero_object"
    assert model.metadata["action_space"] == "delta_eef_robosuite_osc"


def test_non_seven_dim_action_stats_are_rejected(fake_policy, monkeypatch):
    def load_policy(model_dir, attn_implementation):
        stats = {"so101_pick_cylinder": {"action": {"q01": [0.0] * 6, "q99": [1.0] * 6}}}
        return dict(vla=FakeVLA(stats, SO101_RAW), processor=FakeProcessor(), device="cpu", dtype=None)

    monkeypatch.setattr(openvla_server, "_load_policy", load_policy)
    with pytest.raises(RuntimeError, match="7"):
        OpenVLAModel(SO101_CKPT, revision=SO101_REV)


def test_action_endpoint_returns_seven_floats_and_model_version(fake_policy):
    model = OpenVLAModel(SO101_CKPT, revision=SO101_REV, local_dir="/ckpt")
    response = create_app(model).test_client().post("/action", json=_payload())
    assert response.status_code == 200
    body = response.get_json()
    assert len(body["action"]) == 7
    assert all(isinstance(v, float) for v in body["action"])
    assert body["action"] == pytest.approx(SO101_RAW)
    assert body["model_version"] == f"{SO101_CKPT}@{SO101_REV[:12]}"


def test_model_gets_official_prompt_rgb_overhead_and_unnorm_key(fake_policy):
    model = OpenVLAModel(SO101_CKPT, revision=SO101_REV)
    bgr = np.zeros((24, 32, 3), dtype=np.uint8)
    bgr[..., 0] = 200  # pure blue in OpenCV BGR order
    ok, buf = cv2.imencode(".png", bgr)
    images = dict(_payload()["images"], overhead_left=base64.b64encode(buf.tobytes()).decode("ascii"))
    create_app(model).test_client().post("/action", json=_payload(images=images))
    inputs = fake_policy.processor.calls[0]
    # vla-scripts/deploy.py / openvla_utils.get_vla_action prompt, lower-cased instruction.
    assert inputs.prompt == ("In: What action should the robot take to "
                             "pick up the black cylinder and place it on the plate.?\nOut:")
    assert isinstance(inputs.image, Image.Image) and inputs.image.mode == "RGB"
    assert inputs.image.size == (224, 224)
    r, g, b = inputs.image.getpixel((112, 112))
    assert b > 150 and r < 30 and g < 30  # decoded BGR PNG -> RGB
    assert inputs.moved_to == ("cuda:0", "bf16")
    call = fake_policy.vla.calls[0]
    assert call["unnorm_key"] == "so101_pick_cylinder" and call["do_sample"] is False


def test_image_key_selects_which_frame_the_model_sees(fake_policy):
    model = OpenVLAModel(SO101_CKPT, revision=SO101_REV, image_key="wrist")
    images = dict(_payload()["images"], wrist=_b64_png(200), overhead_left=_b64_png(5))
    body = {"images": {"wrist": images["wrist"]}, "instruction": "x"}
    assert create_app(model).test_client().post("/action", json=body).status_code == 200
    assert fake_policy.processor.calls[0].image.getpixel((5, 5))[0] > 150


def test_center_crop_matches_openvla_crop_scale():
    rgb = np.zeros((1000, 1000, 3), dtype=np.uint8)
    rgb[:12, :, :] = 255  # a thin white band (1.2 % of the height) at the very top edge
    plain = openvla_server.preprocess_image(rgb, center_crop=False)
    cropped = openvla_server.preprocess_image(rgb, center_crop=True)
    assert plain.size == cropped.size == (224, 224)
    assert plain.getpixel((112, 0))[0] > 200
    # sqrt(0.9) side crop removes about 2.6 % of each edge, so the band is gone.
    assert cropped.getpixel((112, 0))[0] < 30


@pytest.mark.parametrize(
    "body",
    [
        {"instruction": "x"},  # missing images
        _payload(images={"wrist": _b64_png()}),  # missing overhead_left
        _payload(images={"overhead_left": "!!!not-base64!!!"}),
        _payload(images={"overhead_left": base64.b64encode(b"not a png").decode()}),
        _payload(images=["not", "a", "dict"]),
        _payload(images={"overhead_left": _b64_png(shape=(1100, 20, 3))}),
        _payload(instruction=""),
        _payload(instruction=7),
        _payload(joint_state=[1, 2, 3]),
        ["not", "an", "object"],
    ],
    ids=["no-images", "no-overhead", "bad-base64", "bad-png", "images-not-dict", "oversized-image",
         "empty-instruction", "non-str-instruction", "joint-state-not-dict", "non-object-body"],
)
def test_malformed_body_returns_400(fake_policy, body):
    model = OpenVLAModel(SO101_CKPT, revision=SO101_REV)
    response = create_app(model).test_client().post("/action", json=body)
    assert response.status_code == 400
    assert "error" in response.get_json()
    assert fake_policy.vla.calls == []


def test_non_json_body_returns_400(fake_policy):
    model = OpenVLAModel(SO101_CKPT, revision=SO101_REV)
    response = create_app(model).test_client().post("/action", data="garbage", content_type="text/plain")
    assert response.status_code == 400


def test_non_finite_model_output_returns_500(fake_policy):
    fake_policy.raw = [float("nan")] * 7
    model = OpenVLAModel(SO101_CKPT, revision=SO101_REV)
    response = create_app(model).test_client().post("/action", json=_payload())
    assert response.status_code == 500
    assert "openvla inference failed" in response.get_json()["error"]


def test_health_reports_ready_with_metadata(fake_policy):
    model = OpenVLAModel(SO101_CKPT, revision=SO101_REV)
    body = create_app(model).test_client().get("/health").get_json()
    assert body["status"] == "ready"
    assert body["model"]["checkpoint"] == SO101_CKPT
    assert body["model"]["unnorm_key"] == "so101_pick_cylinder"
    assert body["model"]["action_dim"] == 7
    assert body["model"]["image_key"] == "overhead_left"


def test_overlapping_request_gets_429_not_queued():
    started, release = threading.Event(), threading.Event()

    class Slow:
        metadata = {"checkpoint": "slow"}
        model_version = "slow@test"
        image_key = "overhead_left"

        def __call__(self, images, instruction):
            started.set()
            release.wait(5)
            return list(SO101_RAW)

    http = create_app(Slow()).test_client()
    results = []
    worker = threading.Thread(target=lambda: results.append(http.post("/action", json=_payload()).status_code))
    worker.start()
    assert started.wait(5)
    assert http.post("/action", json=_payload()).status_code == 429
    release.set()
    worker.join(5)
    assert results == [200]


def test_localize_remote_code_pins_official_code_and_rewrites_auto_map(tmp_path, monkeypatch):
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    fetched = []

    def fetch(code_repo, filename, code_revision):
        fetched.append((code_repo, filename, code_revision))
        path = code_dir / filename
        path.write_text(f"# official {filename}\n")
        return str(path)

    monkeypatch.setattr(openvla_server, "_fetch_code_file", fetch)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    prefix = "openvla/openvla-7b--"
    (model_dir / "config.json").write_text(json.dumps({"auto_map": {
        "AutoConfig": prefix + "configuration_prismatic.OpenVLAConfig",
        "AutoModelForVision2Seq": prefix + "modeling_prismatic.OpenVLAForActionPrediction"},
        "norm_stats": {"k": {}}}))
    (model_dir / "preprocessor_config.json").write_text(json.dumps({"auto_map": {
        "AutoImageProcessor": prefix + "processing_prismatic.PrismaticImageProcessor",
        "AutoProcessor": prefix + "processing_prismatic.PrismaticProcessor"}}))
    (model_dir / "tokenizer_config.json").write_text(json.dumps({"auto_map": {
        "AutoProcessor": prefix + "processing_prismatic.PrismaticProcessor"}, "tokenizer_class": "LlamaTokenizer"}))
    (model_dir / "modeling_prismatic.py").write_text("# a checkpoint-shipped copy\n")

    openvla_server._localize_remote_code(str(model_dir))
    openvla_server._localize_remote_code(str(model_dir))  # idempotent

    assert {f for _, f, _ in fetched} == set(openvla_server.REMOTE_CODE_FILES)
    assert all(repo == "openvla/openvla-7b" and rev == openvla_server.CODE_REVISION for repo, _, rev in fetched)
    for name in openvla_server.REMOTE_CODE_FILES:
        assert (model_dir / name).read_text() == f"# official {name}\n"  # shipped copy overwritten
    config = json.loads((model_dir / "config.json").read_text())
    assert config["auto_map"] == {"AutoConfig": "configuration_prismatic.OpenVLAConfig",
                                  "AutoModelForVision2Seq": "modeling_prismatic.OpenVLAForActionPrediction"}
    assert config["norm_stats"] == {"k": {}}
    assert json.loads((model_dir / "preprocessor_config.json").read_text())["auto_map"]["AutoImageProcessor"] == \
        "processing_prismatic.PrismaticImageProcessor"
    tokenizer = json.loads((model_dir / "tokenizer_config.json").read_text())
    assert tokenizer["auto_map"] == {"AutoProcessor": "processing_prismatic.PrismaticProcessor"}
    assert tokenizer["tokenizer_class"] == "LlamaTokenizer"


def test_localize_remote_code_refuses_foreign_remote_code(tmp_path, monkeypatch):
    monkeypatch.setattr(openvla_server, "_fetch_code_file",
                        lambda repo, name, rev: str((tmp_path / name).write_text("x") and tmp_path / name))
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"auto_map": {
        "AutoModelForVision2Seq": "someone/evil--modeling_x.Model"}}))
    with pytest.raises(RuntimeError, match="someone/evil"):
        openvla_server._localize_remote_code(str(model_dir))


def test_server_main_checks_deps_and_warms_model_before_serving(monkeypatch):
    events = []

    class FakeModel:
        metadata = {"checkpoint": "c"}
        model_version = "c@r"
        image_key = "overhead_left"

        def __init__(self, checkpoint, **kwargs):
            events.append(("init", checkpoint, kwargs))

        def __call__(self, images, instruction):
            events.append(("infer", sorted(images), instruction))
            return list(SO101_RAW)

    class FakeApp:
        def run(self, **kwargs):
            events.append(("run", kwargs))

    monkeypatch.setattr(openvla_server, "_check_dependency_versions", lambda: events.append(("deps",)))
    monkeypatch.setattr(openvla_server, "OpenVLAModel", FakeModel)
    monkeypatch.setattr(openvla_server, "create_app", lambda model: FakeApp())
    monkeypatch.setattr("sys.argv", ["openvla_server.py", "--port", "3003"])
    openvla_server.main()
    assert [e[0] for e in events] == ["deps", "init", "infer", "run"]
    assert events[1][1] == openvla_server.DEFAULT_CHECKPOINT == SO101_CKPT
    assert events[1][2]["revision"] == SO101_REV
    assert events[2][1] == ["overhead_left"]
    assert events[3][1]["port"] == 3003 and events[3][1]["use_reloader"] is False


def test_server_main_uses_profile_revision_for_a_known_alternate(monkeypatch):
    seen = {}

    class FakeModel:
        metadata, model_version, image_key = {}, "m", "overhead_left"

        def __init__(self, checkpoint, **kwargs):
            seen.update(kwargs, checkpoint=checkpoint)

        def __call__(self, images, instruction):
            return list(LIBERO_RAW)

    monkeypatch.setattr(openvla_server, "_check_dependency_versions", lambda: None)
    monkeypatch.setattr(openvla_server, "OpenVLAModel", FakeModel)
    monkeypatch.setattr(openvla_server, "create_app", lambda model: SimpleNamespace(run=lambda **k: None))
    monkeypatch.setattr("sys.argv", ["openvla_server.py", "--checkpoint", LIBERO_CKPT])
    openvla_server.main()
    assert seen["checkpoint"] == LIBERO_CKPT and seen["revision"] == LIBERO_REV


def test_server_main_refuses_unknown_checkpoint_without_explicit_revision(monkeypatch):
    monkeypatch.setattr(openvla_server, "_check_dependency_versions", lambda: None)
    monkeypatch.setattr("sys.argv", ["openvla_server.py", "--checkpoint", "someone/openvla-custom"])
    with pytest.raises(SystemExit, match="--revision"):
        openvla_server.main()


def test_dependency_check_refuses_unpinned_transformers(monkeypatch):
    monkeypatch.setattr(openvla_server, "_installed_versions",
                        lambda: {"transformers": "4.57.0", "tokenizers": "0.19.1", "timm": "0.9.10"})
    with pytest.raises(SystemExit, match="transformers"):
        openvla_server._check_dependency_versions()
    monkeypatch.setattr(openvla_server, "_installed_versions",
                        lambda: {"transformers": "4.40.1", "tokenizers": "0.19.1", "timm": "0.9.10"})
    openvla_server._check_dependency_versions()


# --- control-side client ------------------------------------------------------


def test_openvla_is_registered_on_the_http_lane():
    assert backends.BACKEND_REGISTRY["openvla"] is openvla.build
    assert "openvla" not in backends.GRPC_BRIDGE_BACKENDS


def test_backend_config_pins_checkpoint_and_leaves_endpoint_to_runtime():
    config = backends.load_backend_config("openvla")
    assert config["checkpoint"] == SO101_CKPT == openvla_server.DEFAULT_CHECKPOINT
    assert config["revision"] == SO101_REV == openvla_server.DEFAULT_REVISION
    assert config["endpoint_url"] is None


def test_client_and_server_checkpoint_profiles_agree():
    profiles = backends.load_backend_config("openvla")["checkpoints"]
    assert set(profiles) == set(openvla_server.CHECKPOINT_PROFILES)
    for checkpoint, profile in profiles.items():
        server = openvla_server.CHECKPOINT_PROFILES[checkpoint]
        assert profile["revision"] == server["revision"]
        assert profile["unnorm_key"] == server["unnorm_key"]
        assert profile["action_space"] == server["action_space"]
        assert profile["action_space"] in openvla.ACTION_SPACES


def test_so101_joint_action_maps_seven_slots_to_joint_order():
    action = openvla.so101_joint_action(SO101_RAW)
    assert tuple(action) == JOINT_ORDER
    assert [action[j] for j in JOINT_ORDER] == [12.0, -18.5, 33.0, 20.0, 4.0, 41.0]  # pad slot 5 dropped


@pytest.mark.parametrize("raw", [[0.0] * 6, [0.0] * 8, ["a"] * 7, {"dx": 0.1}, None])
def test_malformed_raw_action_raises(raw):
    with pytest.raises((ValueError, TypeError)):
        openvla.so101_joint_action(raw)


def test_joint_space_convert_fn_is_pass_through_without_ik(monkeypatch):
    monkeypatch.setattr(eef_to_joint, "solve_ik", lambda *a, **k: pytest.fail("joint-space must not run IK"))
    convert = openvla.make_convert_fn("absolute_joint_degrees_so101_7d", {"shoulder_pan": (-90.0, 90.0)})
    assert convert(SO101_RAW, JOINT_STATE) == openvla.so101_joint_action(SO101_RAW)


def test_delta_eef_convert_fn_calls_solve_ik_with_joint_state_and_eef_delta(monkeypatch):
    calls = []
    sentinel = {joint: 1.0 for joint in JOINT_ORDER}

    def fake_solve_ik(joint_state, eef_delta, **kwargs):
        calls.append((joint_state, eef_delta, kwargs))
        return sentinel

    monkeypatch.setattr(eef_to_joint, "solve_ik", fake_solve_ik)
    limits = {"shoulder_pan": (-90.0, 90.0)}
    convert = openvla.make_convert_fn("delta_eef_robosuite_osc", limits)
    assert convert(LIBERO_RAW, JOINT_STATE) is sentinel
    joint_state, eef_delta, kwargs = calls[0]
    assert joint_state is JOINT_STATE
    assert set(eef_delta) == {"dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"}
    # Same LIBERO robosuite OSC_POSE units and gripper convention as OpenVLA-OFT.
    assert eef_delta == openvla_oft.openvla_oft_action_to_eef_delta(LIBERO_RAW)
    assert kwargs == {"joint_limits_deg": limits}


def test_unknown_action_space_is_rejected():
    with pytest.raises(ValueError, match="action_space"):
        openvla.make_convert_fn("joint_velocity", None)


def _args(**overrides):
    defaults = dict(server_address=None, checkpoint=None, port="/dev/null", robot_id="r",
                    stereo_camera_index=1)
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


@pytest.fixture
def no_config(monkeypatch, tmp_path):
    monkeypatch.setattr(backends, "BACKEND_CONFIG_DIR", tmp_path)
    return tmp_path


def test_build_without_any_endpoint_raises_clear_systemexit(no_config):
    (no_config / "openvla.json").write_text(json.dumps({"checkpoint": "c", "endpoint_url": None}))
    with pytest.raises(SystemExit, match="--server-address"):
        openvla.build(_args(), {0: "wrist"}, {})


def test_build_without_config_file_raises_clear_systemexit(no_config):
    with pytest.raises(SystemExit, match="endpoint_url"):
        openvla.build(_args(), {0: "wrist"}, {})


def test_build_rejects_a_grpc_style_host_port_endpoint(no_config):
    with pytest.raises(SystemExit, match="http"):
        openvla.build(_args(server_address="0.tcp.ngrok.io:12345"), {0: "wrist"}, {})


def test_build_rejects_a_checkpoint_without_a_profile():
    with pytest.raises(SystemExit, match="action_space"):
        openvla.build(_args(server_address="https://x.ngrok-free.app", checkpoint="someone/other"), {0: "wrist"}, {})


def _capture_build(monkeypatch):
    seen = {}

    def fake_build(endpoint_url, checkpoint, args, camera_names, convert_fn=None):
        seen.update(endpoint_url=endpoint_url, checkpoint=checkpoint, camera_names=camera_names,
                    convert_fn=convert_fn)
        return "handles"

    monkeypatch.setattr(base_http_adapter, "build_http_adapter_backend", fake_build)
    return seen


def test_build_defaults_to_the_so101_joint_space_checkpoint(monkeypatch):
    seen = _capture_build(monkeypatch)
    monkeypatch.setattr(eef_to_joint, "solve_ik", lambda *a, **k: pytest.fail("no IK for joint space"))
    assert openvla.build(_args(server_address="https://abc.ngrok-free.app"), {0: "wrist"}, {}) == "handles"
    assert seen["endpoint_url"] == "https://abc.ngrok-free.app"
    assert seen["checkpoint"] == SO101_CKPT
    assert seen["camera_names"] == {0: "wrist"}
    assert seen["convert_fn"](SO101_RAW, JOINT_STATE) == openvla.so101_joint_action(SO101_RAW)


def test_build_with_libero_checkpoint_wires_the_ik_convert_fn(monkeypatch):
    seen = _capture_build(monkeypatch)
    calls = []
    monkeypatch.setattr(eef_to_joint, "solve_ik", lambda js, delta, **k: calls.append((js, delta, k)) or "ik")
    limits = {"elbow_flex": (-80.0, 80.0)}
    openvla.build(_args(server_address="http://h:3003", checkpoint=LIBERO_CKPT), {0: "wrist"}, limits)
    assert seen["checkpoint"] == LIBERO_CKPT
    assert seen["convert_fn"](LIBERO_RAW, JOINT_STATE) == "ik"
    assert calls[0][1] == openvla_oft.openvla_oft_action_to_eef_delta(LIBERO_RAW)
    assert calls[0][2] == {"joint_limits_deg": limits}


def test_build_falls_back_to_config_endpoint(monkeypatch, no_config):
    (no_config / "openvla.json").write_text(json.dumps({
        "checkpoint": "c", "endpoint_url": "https://cfg.example",
        "checkpoints": {"c": {"action_space": "absolute_joint_degrees_so101_7d"}}}))
    seen = _capture_build(monkeypatch)
    openvla.build(_args(), {0: "wrist"}, {})
    assert seen["endpoint_url"] == "https://cfg.example" and seen["checkpoint"] == "c"
