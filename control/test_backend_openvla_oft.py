"""Tests for the OpenVLA-OFT Tier C backend (Plan 14-03 Task 2).

Covers the Colab-side resident adapter (`vla_bridge/openvla_oft_server.py`,
with the real model loading mocked, so the 7B checkpoint is never downloaded)
and the control-side client (`vla_bridge/backends/openvla_oft.py`), with
mocked HTTP. Neither touches real hardware or the network.
"""

import base64
import json
import math
import threading
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from test_backend_registry import FakeStereo, FakeWristCap
from vla_bridge import backends
from vla_bridge import openvla_oft_server
from vla_bridge.action_contract import JOINT_ORDER
from vla_bridge.backends import base_http_adapter, eef_to_joint, openvla_oft
from vla_bridge.openvla_oft_server import OpenVLAOFTModel, create_app

JOINT_STATE = {
    "shoulder_pan": 10.0,
    "shoulder_lift": -20.0,
    "elbow_flex": 30.0,
    "wrist_flex": 15.0,
    "wrist_roll": 5.0,
    "gripper": 37.0,
}

RAW_ACTION = [0.2, -0.1, 0.3, 0.02, -0.04, 0.06, 0.9]


def _b64_png(value=50, shape=(24, 32, 3)):
    ok, buf = cv2.imencode(".png", np.full(shape, value, dtype=np.uint8))
    assert ok
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _payload(**overrides):
    body = {
        "images": {"wrist": _b64_png(10), "overhead_left": _b64_png(20), "overhead_right": _b64_png(30)},
        "proprio_state": [0.0] * 8,
        "joint_state": dict(JOINT_STATE),
        "instruction": "put the cube in the basket",
    }
    body.update(overrides)
    return body


# --- Colab-side server ---------------------------------------------------------


class FakeVLA:
    llm_dim = 4096
    norm_stats = {
        "libero_object_no_noops": {
            "action": {},
            "proprio": {"mean": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]},
        }
    }


@pytest.fixture
def fake_policy(monkeypatch):
    """Mock `_load_policy` (openvla-oft's get_vla/get_action_head/... imports)
    and `_download_pinned` (huggingface_hub.snapshot_download)."""
    loads, downloads, calls = [], [], []

    def get_vla_action(cfg, vla, processor, obs, task_label, action_head=None,
                       proprio_projector=None, noisy_action_projector=None, use_film=False):
        calls.append(dict(cfg=cfg, obs=obs, task_label=task_label, action_head=action_head,
                          proprio_projector=proprio_projector, use_film=use_film))
        chunk = np.tile(np.array(RAW_ACTION, dtype=np.float32), (8, 1))
        return [chunk[i] for i in range(len(chunk))]

    def load_policy(cfg):
        loads.append(cfg)
        return dict(vla=FakeVLA(), processor="processor", action_head="action_head",
                    proprio_projector="proprio_projector", get_vla_action=get_vla_action)

    def download(checkpoint, revision, local_dir):
        downloads.append((checkpoint, revision, local_dir))
        return f"{local_dir}/snapshot"

    monkeypatch.setattr(openvla_oft_server, "_load_policy", load_policy)
    monkeypatch.setattr(openvla_oft_server, "_download_pinned", download)
    return SimpleNamespace(loads=loads, downloads=downloads, calls=calls)


def test_model_loads_once_across_many_action_requests(fake_policy):
    model = OpenVLAOFTModel("moojink/openvla-7b-oft-finetuned-libero-object",
                            revision="4c89574e1c538b6c102f43f0526d60a9d3650148", local_dir="/ckpt")
    http = create_app(model).test_client()
    for _ in range(3):
        assert http.post("/action", json=_payload()).status_code == 200
    assert len(fake_policy.loads) == 1
    assert len(fake_policy.downloads) == 1
    assert len(fake_policy.calls) == 3


def test_pinned_revision_is_downloaded_and_loaded_from_local_snapshot(fake_policy):
    model = OpenVLAOFTModel("moojink/openvla-7b-oft-finetuned-libero-object",
                            revision="4c89574e1c538b6c102f43f0526d60a9d3650148", local_dir="/ckpt")
    assert fake_policy.downloads == [(
        "moojink/openvla-7b-oft-finetuned-libero-object", "4c89574e1c538b6c102f43f0526d60a9d3650148", "/ckpt",
    )]
    cfg = fake_policy.loads[0]
    assert cfg.pretrained_checkpoint == "/ckpt/snapshot"
    # Mirrors openvla-oft's LIBERO GenerateConfig / model card quick start.
    assert cfg.model_family == "openvla"
    assert cfg.use_l1_regression is True and cfg.use_diffusion is False
    assert cfg.use_film is False and cfg.num_images_in_input == 2
    assert cfg.use_proprio is True and cfg.center_crop is True
    assert cfg.unnorm_key == "libero_object_no_noops"  # sole key in dataset_statistics
    assert model.model_version == "moojink/openvla-7b-oft-finetuned-libero-object@4c89574e1c53"
    assert model.metadata["revision"] == "4c89574e1c538b6c102f43f0526d60a9d3650148"


def test_unpinned_checkpoint_loads_by_repo_id_without_download(fake_policy):
    model = OpenVLAOFTModel("moojink/openvla-7b-oft-finetuned-libero-object", revision=None)
    assert fake_policy.downloads == []
    assert fake_policy.loads[0].pretrained_checkpoint == "moojink/openvla-7b-oft-finetuned-libero-object"
    assert model.model_version.endswith("@unpinned")


def test_ambiguous_unnorm_key_requires_explicit_choice(fake_policy, monkeypatch):
    two = dict(FakeVLA.norm_stats, libero_goal_no_noops=FakeVLA.norm_stats["libero_object_no_noops"])
    monkeypatch.setattr(FakeVLA, "norm_stats", two)
    with pytest.raises(ValueError, match="unnorm"):
        OpenVLAOFTModel("moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10")
    model = OpenVLAOFTModel("moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10",
                            unnorm_key="libero_goal_no_noops")
    assert fake_policy.loads[-1].unnorm_key == "libero_goal_no_noops"
    with pytest.raises(ValueError, match="unnorm"):
        OpenVLAOFTModel("moojink/openvla-7b-oft-finetuned-libero-object", unnorm_key="bridge_orig")
    assert model.metadata["unnorm_key"] == "libero_goal_no_noops"


def test_action_endpoint_returns_seven_floats_and_model_version(fake_policy):
    model = OpenVLAOFTModel("moojink/openvla-7b-oft-finetuned-libero-object",
                            revision="4c89574e1c538b6c102f43f0526d60a9d3650148", local_dir="/ckpt")
    response = create_app(model).test_client().post("/action", json=_payload())
    assert response.status_code == 200
    body = response.get_json()
    assert len(body["action"]) == 7
    assert all(isinstance(v, float) for v in body["action"])
    assert body["action"] == pytest.approx(RAW_ACTION, abs=1e-6)
    assert body["model_version"] == "moojink/openvla-7b-oft-finetuned-libero-object@4c89574e1c53"


def test_model_receives_rgb_overhead_as_full_image_and_wrist_image(fake_policy):
    model = OpenVLAOFTModel("moojink/openvla-7b-oft-finetuned-libero-object")
    bgr = np.zeros((24, 32, 3), dtype=np.uint8)
    bgr[..., 0] = 200  # pure blue in OpenCV BGR order
    ok, buf = cv2.imencode(".png", bgr)
    images = dict(_payload()["images"], overhead_left=base64.b64encode(buf.tobytes()).decode("ascii"))
    create_app(model).test_client().post("/action", json=_payload(images=images))
    call = fake_policy.calls[0]
    obs = call["obs"]
    assert set(obs) == {"full_image", "wrist_image", "state"}
    assert obs["full_image"].dtype == np.uint8 and obs["full_image"].shape == (24, 32, 3)
    assert tuple(obs["full_image"][0, 0]) == (0, 0, 200)  # decoded to RGB
    assert int(obs["wrist_image"][0, 0, 0]) == 10
    assert call["task_label"] == "put the cube in the basket"
    assert call["action_head"] == "action_head" and call["proprio_projector"] == "proprio_projector"


def test_missing_proprio_falls_back_to_checkpoint_mean_proprio(fake_policy):
    model = OpenVLAOFTModel("moojink/openvla-7b-oft-finetuned-libero-object")
    body = _payload()
    del body["proprio_state"]
    assert create_app(model).test_client().post("/action", json=body).status_code == 200
    np.testing.assert_allclose(fake_policy.calls[0]["obs"]["state"], FakeVLA.norm_stats[
        "libero_object_no_noops"]["proprio"]["mean"])
    explicit = [1.0] * 8
    create_app(model).test_client().post("/action", json=_payload(proprio_state=explicit))
    np.testing.assert_allclose(fake_policy.calls[1]["obs"]["state"], explicit)


@pytest.mark.parametrize(
    "body",
    [
        {"instruction": "x", "proprio_state": [0.0] * 8},  # missing images
        _payload(images={"wrist": _b64_png()}),  # missing overhead_left
        _payload(images={"wrist": "!!!not-base64!!!", "overhead_left": _b64_png()}),
        _payload(images={"wrist": base64.b64encode(b"not a png").decode(), "overhead_left": _b64_png()}),
        _payload(images=["not", "a", "dict"]),
        _payload(instruction=""),
        _payload(instruction=7),
        _payload(proprio_state=[0.0] * 3),  # wrong proprio dim
        _payload(proprio_state=[float("nan")] * 8),
        _payload(proprio_state="zeros"),
        _payload(joint_state=[1, 2, 3]),
        ["not", "an", "object"],
    ],
    ids=["no-images", "no-overhead", "bad-base64", "bad-png", "images-not-dict", "empty-instruction",
         "non-str-instruction", "bad-proprio-dim", "nan-proprio", "proprio-not-list",
         "joint-state-not-dict", "non-object-body"],
)
def test_malformed_body_returns_400(fake_policy, body):
    model = OpenVLAOFTModel("moojink/openvla-7b-oft-finetuned-libero-object")
    response = create_app(model).test_client().post("/action", json=body)
    assert response.status_code == 400
    assert "error" in response.get_json()
    assert fake_policy.calls == []


def test_non_json_body_returns_400(fake_policy):
    model = OpenVLAOFTModel("moojink/openvla-7b-oft-finetuned-libero-object")
    response = create_app(model).test_client().post("/action", data="garbage", content_type="text/plain")
    assert response.status_code == 400


def test_health_reports_ready_with_metadata(fake_policy):
    model = OpenVLAOFTModel("moojink/openvla-7b-oft-finetuned-libero-object")
    body = create_app(model).test_client().get("/health").get_json()
    assert body["status"] == "ready"
    assert body["model"]["checkpoint"] == "moojink/openvla-7b-oft-finetuned-libero-object"
    assert body["model"]["action_space"] == "delta_eef_robosuite_osc_normalized"


def test_overlapping_request_gets_429_not_queued():
    started, release = threading.Event(), threading.Event()

    class Slow:
        metadata = {"checkpoint": "slow"}
        model_version = "slow@test"

        def __call__(self, images, proprio_state, instruction):
            started.set()
            release.wait(5)
            return list(RAW_ACTION)

    http = create_app(Slow()).test_client()
    results = []
    worker = threading.Thread(target=lambda: results.append(http.post("/action", json=_payload()).status_code))
    worker.start()
    assert started.wait(5)
    assert http.post("/action", json=_payload()).status_code == 429
    release.set()
    worker.join(5)
    assert results == [200]


def test_server_main_warms_model_before_serving(monkeypatch, tmp_path):
    events = []

    class FakeModel:
        metadata = {"checkpoint": "c"}
        model_version = "c@r"

        def __init__(self, checkpoint, **kwargs):
            events.append(("init", checkpoint, kwargs))

        def __call__(self, images, proprio_state, instruction):
            events.append(("infer", sorted(images), instruction))
            return list(RAW_ACTION)

    class FakeApp:
        def run(self, **kwargs):
            events.append(("run", kwargs))

    monkeypatch.setattr(openvla_oft_server, "OpenVLAOFTModel", FakeModel)
    monkeypatch.setattr(openvla_oft_server, "create_app", lambda model: FakeApp())
    monkeypatch.setattr(openvla_oft_server.os, "chdir", lambda path: events.append(("chdir", path)))
    monkeypatch.setattr("sys.argv", ["openvla_oft_server.py", "--repo", str(tmp_path), "--port", "3001"])
    openvla_oft_server.main()
    kinds = [e[0] for e in events]
    assert kinds == ["chdir", "init", "infer", "run"]
    assert events[1][1] == openvla_oft_server.DEFAULT_CHECKPOINT
    assert events[1][2]["revision"] == openvla_oft_server.DEFAULT_REVISION
    assert events[2][1] == ["overhead_left", "wrist"]
    assert events[3][1]["port"] == 3001 and events[3][1]["use_reloader"] is False


# --- control-side client ------------------------------------------------------


def test_openvla_oft_is_registered_on_the_http_lane():
    assert backends.BACKEND_REGISTRY["openvla-oft"] is openvla_oft.build
    assert "openvla-oft" not in backends.GRPC_BRIDGE_BACKENDS


def test_backend_config_pins_checkpoint_and_leaves_endpoint_to_runtime():
    config = backends.load_backend_config("openvla_oft")
    assert config["checkpoint"] == "moojink/openvla-7b-oft-finetuned-libero-object"
    assert config["revision"] == openvla_oft_server.DEFAULT_REVISION
    assert config["endpoint_url"] is None
    assert openvla_oft_server.DEFAULT_CHECKPOINT == config["checkpoint"]


def test_raw_action_is_scaled_from_robosuite_osc_units():
    from scipy.spatial.transform import Rotation

    delta = openvla_oft.openvla_oft_action_to_eef_delta(RAW_ACTION)
    assert delta["dx"] == pytest.approx(0.2 * 0.05)
    assert delta["dy"] == pytest.approx(-0.1 * 0.05)
    assert delta["dz"] == pytest.approx(0.3 * 0.05)
    expected = Rotation.from_rotvec(np.array(RAW_ACTION[3:6]) * 0.5).as_euler("xyz")
    assert [delta["droll"], delta["dpitch"], delta["dyaw"]] == pytest.approx(list(expected))
    assert delta["gripper"] == 100.0  # model 1 = open -> SO-101 100 % open


@pytest.mark.parametrize("raw_gripper,expected", [(0.9, 100.0), (0.1, 0.0), (0.5, 50.0), (1.0, 100.0), (0.0, 0.0)])
def test_gripper_follows_openvla_oft_binarized_convention(raw_gripper, expected):
    delta = openvla_oft.openvla_oft_action_to_eef_delta(RAW_ACTION[:6] + [raw_gripper])
    assert delta["gripper"] == expected


@pytest.mark.parametrize("raw", [[0.0] * 6, [0.0] * 8, ["a"] * 7, {"dx": 0.1}])
def test_malformed_raw_action_raises(raw):
    with pytest.raises((ValueError, TypeError)):
        openvla_oft.openvla_oft_action_to_eef_delta(raw)


def test_convert_fn_calls_solve_ik_with_joint_state_and_scaled_delta(monkeypatch):
    calls = []
    sentinel = {joint: 1.0 for joint in JOINT_ORDER}

    def fake_solve_ik(joint_state, eef_delta, **kwargs):
        calls.append((joint_state, eef_delta, kwargs))
        return sentinel

    monkeypatch.setattr(eef_to_joint, "solve_ik", fake_solve_ik)
    limits = {"shoulder_pan": (-90.0, 90.0)}
    convert = openvla_oft.make_convert_fn(limits)
    assert convert(RAW_ACTION, JOINT_STATE) is sentinel
    joint_state, eef_delta, kwargs = calls[0]
    assert joint_state is JOINT_STATE
    assert set(eef_delta) == {"dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"}
    assert eef_delta == openvla_oft.openvla_oft_action_to_eef_delta(RAW_ACTION)
    assert kwargs == {"joint_limits_deg": limits}


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
    (no_config / "openvla_oft.json").write_text(json.dumps({"checkpoint": "c", "endpoint_url": None}))
    with pytest.raises(SystemExit, match="--server-address"):
        openvla_oft.build(_args(), {0: "wrist"}, {})


def test_build_without_config_file_raises_clear_systemexit(no_config):
    with pytest.raises(SystemExit, match="endpoint_url"):
        openvla_oft.build(_args(), {0: "wrist"}, {})


def test_build_rejects_a_grpc_style_host_port_endpoint(no_config):
    with pytest.raises(SystemExit, match="http"):
        openvla_oft.build(_args(server_address="0.tcp.ngrok.io:12345"), {0: "wrist"}, {})


def test_build_wires_http_adapter_with_ik_convert_fn(monkeypatch):
    seen = {}

    def fake_build(endpoint_url, checkpoint, args, camera_names, convert_fn=None):
        seen.update(endpoint_url=endpoint_url, checkpoint=checkpoint, args=args,
                    camera_names=camera_names, convert_fn=convert_fn)
        return "handles"

    monkeypatch.setattr(base_http_adapter, "build_http_adapter_backend", fake_build)
    args = _args(server_address="https://abc.ngrok-free.app")
    assert openvla_oft.build(args, {0: "wrist"}, {}) == "handles"
    assert seen["endpoint_url"] == "https://abc.ngrok-free.app"
    assert seen["checkpoint"] == "moojink/openvla-7b-oft-finetuned-libero-object"  # from config
    assert seen["camera_names"] == {0: "wrist"}
    assert callable(seen["convert_fn"])

    openvla_oft.build(_args(server_address="http://h:3001", checkpoint="local/ckpt"), {0: "wrist"}, {})
    assert seen["checkpoint"] == "local/ckpt"


def test_build_falls_back_to_config_endpoint(monkeypatch, no_config):
    (no_config / "openvla_oft.json").write_text(json.dumps({"checkpoint": "c", "endpoint_url": "https://cfg.example"}))
    seen = {}
    monkeypatch.setattr(base_http_adapter, "build_http_adapter_backend",
                        lambda endpoint_url, checkpoint, *a, **k: seen.update(url=endpoint_url, ckpt=checkpoint))
    openvla_oft.build(_args(), {0: "wrist"}, {})
    assert seen == {"url": "https://cfg.example", "ckpt": "c"}


def test_end_to_end_server_response_becomes_a_valid_joint_action(fake_policy, monkeypatch):
    """Real create_app() behind the real HttpAdapterActionSource, with the
    real ikpy-backed convert_fn: raw 7-D delta -> finite JOINT_ORDER dict."""
    model = OpenVLAOFTModel("moojink/openvla-7b-oft-finetuned-libero-object",
                            revision="4c89574e1c538b6c102f43f0526d60a9d3650148", local_dir="/ckpt")
    http = create_app(model).test_client()

    class Routed:
        def __init__(self, response):
            self._response = response
            self.status_code = response.status_code

        def raise_for_status(self):
            assert self.status_code == 200

        def json(self):
            return self._response.get_json()

    posted = []

    def fake_post(url, json=None, timeout=None):
        posted.append(url)
        return Routed(http.post("/action", json=json))

    monkeypatch.setattr(base_http_adapter.requests, "post", fake_post)
    source = base_http_adapter.HttpAdapterActionSource(
        "https://abc.ngrok-free.app", "moojink/openvla-7b-oft-finetuned-libero-object",
        FakeWristCap(), FakeStereo(), convert_fn=openvla_oft.make_convert_fn(None),
    )
    action, model_version = source.get_action(JOINT_STATE, "put the cube in the basket")
    assert posted == ["https://abc.ngrok-free.app/action"]
    assert model_version == "moojink/openvla-7b-oft-finetuned-libero-object@4c89574e1c53"
    assert tuple(action) == JOINT_ORDER
    assert all(isinstance(v, float) and math.isfinite(v) for v in action.values())
    assert action["gripper"] == 100.0
    assert any(abs(action[j] - JOINT_STATE[j]) > 1e-3 for j in JOINT_ORDER[:5])
    assert source.last_details["raw_model_output"] == pytest.approx(RAW_ACTION, abs=1e-6)
    # The client's joint_state dict reaches the server but is not fed to the model.
    assert "state" in fake_policy.calls[0]["obs"]
