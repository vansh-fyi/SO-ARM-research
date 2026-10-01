"""Tests for the Octo Tier C backend (Plan 14-10, VLAB-08).

Covers the Colab-side resident adapter (`vla_bridge/octo_server.py`, with the
pinned HF download, the pinned T5 tokenizer fetch, the JAX model load and the
JAX sampling call all mocked) and the control-side client
(`vla_bridge/backends/octo.py`), with mocked HTTP. Nothing here touches real
hardware, a GPU, JAX, or the network.
"""

import base64
import json
import math
import threading
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from vla_bridge import backends
from vla_bridge import octo_server
from vla_bridge.action_contract import JOINT_ORDER
from vla_bridge.backends import base_http_adapter, eef_to_joint, octo
from vla_bridge.octo_server import OctoModel, create_app

SO101_CKPT = "sapanostic/octo-so101-3hz-finetune"
SO101_REV = "a45ff970242a66e268871a5f64530947143f9a6e"
SMALL_CKPT = "rail-berkeley/octo-small-1.5"
SMALL_REV = "dc9aa3019f764726c770814b27e4ab0fc6e32a58"

JOINT_STATE = {
    "shoulder_pan": 10.0,
    "shoulder_lift": -20.0,
    "elbow_flex": 30.0,
    "wrist_flex": 15.0,
    "wrist_roll": 5.0,
    "gripper": 37.0,
}

# One un-normalized Octo action: [dx, dy, dz (m), droll, dpitch, dyaw (rad, rpy
# differences), gripper (dataset-normalized, mask=False so returned raw)].
EEF_RAW = [0.01, -0.005, 0.008, 0.02, -0.03, 0.04, 0.5]
# A (hypothetical) joint-space Octo fine-tune: 6 values in JOINT_ORDER.
JOINT_RAW = [12.0, -18.5, 33.0, 20.0, 4.0, 41.0]


def _b64_png(value=50, shape=(24, 32, 3)):
    ok, buf = cv2.imencode(".png", np.full(shape, value, dtype=np.uint8))
    assert ok
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _payload(**overrides):
    body = {
        "images": {"wrist": _b64_png(10), "overhead_left": _b64_png(20), "overhead_right": _b64_png(30)},
        "joint_state": dict(JOINT_STATE),
        "instruction": "pick up the cube",
    }
    body.update(overrides)
    return body


# --- Colab-side server ---------------------------------------------------------


def _example_batch(window=2, horizon=4, wrist=True, proprio=False):
    observation = {
        "image_primary": np.zeros((1, window, 256, 256, 3), dtype=np.uint8),
        "timestep_pad_mask": np.ones((1, window), dtype=bool),
    }
    if wrist:
        observation["image_wrist"] = np.zeros((1, window, 128, 128, 3), dtype=np.uint8)
    if proprio:
        observation["proprio"] = np.zeros((1, window, 7), dtype=np.float32)
    return {"observation": observation, "action": np.zeros((1, window, horizon, 7), dtype=np.float32)}


def _action_stats(dim=7):
    return {"mean": np.zeros(dim), "std": np.ones(dim), "mask": np.array([True] * (dim - 1) + [False])}


class FakeOcto:
    """Stands in for upstream `octo.model.octo_model.OctoModel`."""

    def __init__(self, dataset_statistics, example_batch):
        self.dataset_statistics = dataset_statistics
        self.example_batch = example_batch
        self.task_calls = []

    def create_tasks(self, texts=None, goals=None):
        self.task_calls.append(list(texts))
        return {"language_instruction": {"input_ids": np.array([[1, 2, 3]])}}


@pytest.fixture
def fake_policy(monkeypatch):
    """Mock `_download_pinned` (snapshot_download), `_pin_text_encoder` (the
    pinned google-t5/t5-base fetch + config rewrite), `_load_policy`
    (`OctoModel.load_pretrained`) and `_sample_actions` (the jitted
    `sample_actions` call)."""
    state = SimpleNamespace(loads=[], downloads=[], pinned=[], samples=[], stats=None,
                            example_batch=_example_batch(), chunk=None, model=None)
    state.stats = _action_stats()  # a single-dataset fine-tune: flat {"action": ...}

    def download(checkpoint, revision, local_dir, step):
        state.downloads.append((checkpoint, revision, local_dir, step))
        return f"{local_dir}/snapshot"

    def pin_text_encoder(model_dir, local_dir):
        state.pinned.append((model_dir, local_dir))

    def load_policy(model_dir, step):
        stats = state.stats if "mean" not in state.stats else {"action": state.stats}
        state.model = FakeOcto(stats, state.example_batch)
        state.loads.append((model_dir, step))
        return state.model

    def sample_actions(model, observations, tasks, unnormalization_statistics, seed):
        state.samples.append(dict(observations=observations, tasks=tasks, stats=unnormalization_statistics,
                                  seed=seed))
        dim = len(unnormalization_statistics["mean"])
        if state.chunk is not None:
            return np.asarray(state.chunk, dtype=np.float32)
        horizon = model.example_batch["action"].shape[2]
        chunk = np.tile(np.asarray(EEF_RAW[:dim] if dim == 7 else JOINT_RAW, dtype=np.float32), (1, horizon, 1))
        chunk[0, 1:] += 1.0  # later chunk steps differ, so "first step" is observable
        return chunk

    monkeypatch.setattr(octo_server, "_download_pinned", download)
    monkeypatch.setattr(octo_server, "_pin_text_encoder", pin_text_encoder)
    monkeypatch.setattr(octo_server, "_load_policy", load_policy)
    monkeypatch.setattr(octo_server, "_sample_actions", sample_actions)
    return state


def test_model_loads_once_across_many_action_requests(fake_policy):
    model = OctoModel(SO101_CKPT, revision=SO101_REV, local_dir="/ckpt")
    http = create_app(model).test_client()
    for _ in range(3):
        assert http.post("/action", json=_payload()).status_code == 200
    assert len(fake_policy.loads) == 1
    assert len(fake_policy.downloads) == 1
    assert len(fake_policy.samples) == 3


def test_pinned_revision_and_step_are_downloaded_text_encoder_pinned_and_loaded_from_disk(fake_policy):
    model = OctoModel(SO101_CKPT, revision=SO101_REV, local_dir="/ckpt")
    # Profile step 10000 is the only checkpoint step that repo ships.
    assert fake_policy.downloads == [(SO101_CKPT, SO101_REV, "/ckpt", 10000)]
    assert fake_policy.pinned == [("/ckpt/snapshot", "/ckpt")]
    assert fake_policy.loads == [("/ckpt/snapshot", 10000)]
    assert model.model_version == f"{SO101_CKPT}@{SO101_REV[:12]}"
    assert model.metadata["revision"] == SO101_REV
    assert model.metadata["step"] == 10000
    assert model.metadata["text_encoder"] == f"{octo_server.T5_REPO}@{octo_server.T5_REVISION}"


def test_unpinned_checkpoint_loads_through_octos_own_hf_path(fake_policy):
    model = OctoModel(SO101_CKPT, revision=None)
    assert fake_policy.downloads == [] and fake_policy.pinned == []
    assert fake_policy.loads == [(f"hf://{SO101_CKPT}", None)]
    assert model.model_version.endswith("@unpinned")


def test_flat_single_dataset_statistics_are_used_directly(fake_policy):
    model = OctoModel(SO101_CKPT, revision=SO101_REV)
    assert model.unnorm_key is None
    assert model.metadata["action_space"] == "eef_delta"
    assert model.metadata["action_dim"] == 7


def test_official_checkpoint_uses_its_profile_dataset_key(fake_policy):
    # The OXE-pretrained base embeds one stats entry per pretraining dataset.
    fake_policy.stats = {"bridge_dataset": {"action": _action_stats()},
                         "fractal20220817_data": {"action": _action_stats()}}
    model = OctoModel(SMALL_CKPT, revision=SMALL_REV)
    assert model.unnorm_key == "bridge_dataset"
    assert fake_policy.downloads[0][3] == 300000
    create_app(model).test_client().post("/action", json=_payload())
    assert fake_policy.samples[0]["stats"] is model.model.dataset_statistics["bridge_dataset"]["action"]


def test_unknown_multi_dataset_checkpoint_needs_an_unambiguous_unnorm_key(fake_policy):
    fake_policy.stats = {"a": {"action": _action_stats()}, "b": {"action": _action_stats()}}
    with pytest.raises(ValueError, match="unnorm"):
        OctoModel("someone/octo-custom", revision=None)
    model = OctoModel("someone/octo-custom", revision=None, unnorm_key="b")
    assert model.unnorm_key == "b"
    assert model.metadata["action_space"] == "unknown"
    with pytest.raises(ValueError, match="unnorm"):
        OctoModel("someone/octo-custom", revision=None, unnorm_key="missing")


def test_checkpoint_needing_proprio_is_refused(fake_policy):
    fake_policy.example_batch = _example_batch(proprio=True)
    with pytest.raises(RuntimeError, match="proprio"):
        OctoModel(SO101_CKPT, revision=SO101_REV)


def test_action_endpoint_returns_first_chunk_step_and_model_version(fake_policy):
    model = OctoModel(SO101_CKPT, revision=SO101_REV, local_dir="/ckpt")
    response = create_app(model).test_client().post("/action", json=_payload())
    assert response.status_code == 200
    body = response.get_json()
    assert len(body["action"]) == 7  # the checkpoint's own action dim
    assert all(isinstance(v, float) for v in body["action"])
    assert body["action"] == pytest.approx(EEF_RAW)  # step 0 of the 4-step chunk
    assert body["model_version"] == f"{SO101_CKPT}@{SO101_REV[:12]}"


def test_action_dimension_follows_the_checkpoint_statistics(fake_policy):
    fake_policy.stats = _action_stats(dim=6)
    model = OctoModel("someone/octo-joint", revision=None)
    body = create_app(model).test_client().post("/action", json=_payload()).get_json()
    assert len(body["action"]) == 6
    assert model.metadata["action_dim"] == 6


def test_model_gets_resized_rgb_frames_window_and_task(fake_policy):
    model = OctoModel(SO101_CKPT, revision=SO101_REV)
    bgr = np.zeros((48, 64, 3), dtype=np.uint8)
    bgr[..., 0] = 200  # pure blue in OpenCV BGR order
    ok, buf = cv2.imencode(".png", bgr)
    images = dict(_payload()["images"], overhead_left=base64.b64encode(buf.tobytes()).decode("ascii"))
    create_app(model).test_client().post("/action", json=_payload(images=images))
    sample = fake_policy.samples[0]
    obs = sample["observations"]
    assert set(obs) == {"image_primary", "image_wrist", "timestep_pad_mask"}
    assert obs["image_primary"].shape == (1, 2, 256, 256, 3) and obs["image_primary"].dtype == np.uint8
    assert obs["image_wrist"].shape == (1, 2, 128, 128, 3) and obs["image_wrist"].dtype == np.uint8
    r, g, b = obs["image_primary"][0, -1, 128, 128]
    assert b > 150 and r < 30 and g < 30  # decoded BGR PNG -> RGB, overhead_left is primary
    assert obs["image_wrist"][0, -1, 64, 64, 0] == 10  # wrist frame is the wrist camera
    # First call: HistoryWrapper semantics -> the frame is repeated, older slot padded.
    assert obs["timestep_pad_mask"].tolist() == [[False, True]]
    assert fake_policy.model.task_calls == [["pick up the cube"]]
    assert sample["stats"] is model.action_stats
    assert fake_policy.samples[0]["seed"] != 0


def test_history_window_fills_then_resets_on_new_instruction(fake_policy):
    model = OctoModel(SO101_CKPT, revision=SO101_REV)
    http = create_app(model).test_client()
    http.post("/action", json=_payload(images=dict(_payload()["images"], overhead_left=_b64_png(40))))
    http.post("/action", json=_payload(images=dict(_payload()["images"], overhead_left=_b64_png(80))))
    second = fake_policy.samples[1]["observations"]
    assert second["timestep_pad_mask"].tolist() == [[True, True]]
    assert second["image_primary"][0, 0, 128, 128, 0] == 40  # previous frame
    assert second["image_primary"][0, 1, 128, 128, 0] == 80  # current frame
    http.post("/action", json=_payload(instruction="put the cube in the box"))
    assert fake_policy.samples[2]["observations"]["timestep_pad_mask"].tolist() == [[False, True]]
    http.post("/action", json=_payload(instruction="put the cube in the box", reset=True))
    assert fake_policy.samples[3]["observations"]["timestep_pad_mask"].tolist() == [[False, True]]


def test_history_resets_after_an_idle_gap(fake_policy, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(octo_server.time, "monotonic", lambda: clock[0])
    model = OctoModel(SO101_CKPT, revision=SO101_REV)
    http = create_app(model).test_client()
    http.post("/action", json=_payload())
    clock[0] += octo_server.HISTORY_RESET_S + 1
    http.post("/action", json=_payload())
    assert fake_policy.samples[1]["observations"]["timestep_pad_mask"].tolist() == [[False, True]]


def test_window_one_checkpoint_gets_a_single_frame(fake_policy):
    fake_policy.example_batch = _example_batch(window=1, wrist=False)
    model = OctoModel("someone/octo-w1", revision=None)
    create_app(model).test_client().post("/action", json=_payload())
    obs = fake_policy.samples[0]["observations"]
    assert set(obs) == {"image_primary", "timestep_pad_mask"}
    assert obs["image_primary"].shape == (1, 1, 256, 256, 3)
    assert obs["timestep_pad_mask"].tolist() == [[True]]


def test_center_crop_applies_to_primary_only():
    rgb = np.zeros((1000, 1000, 3), dtype=np.uint8)
    rgb[:12, :, :] = 255  # a thin white band (1.2 % of the height) at the very top edge
    plain = octo_server.preprocess_image(rgb, (256, 256), center_crop=False)
    cropped = octo_server.preprocess_image(rgb, (256, 256), center_crop=True)
    assert plain.shape == cropped.shape == (256, 256, 3)
    assert plain[0, 128, 0] > 200
    # ResizeImageWrapper's average crop (scale 0.9): sqrt(0.9) of each side, band gone.
    assert cropped[0, 128, 0] < 30


@pytest.mark.parametrize(
    "body",
    [
        {"instruction": "x"},  # missing images
        _payload(images={"wrist": _b64_png()}),  # missing overhead_left (primary)
        _payload(images={"overhead_left": _b64_png()}),  # missing wrist
        _payload(images={"overhead_left": "!!!not-base64!!!", "wrist": _b64_png()}),
        _payload(images={"overhead_left": base64.b64encode(b"not a png").decode(), "wrist": _b64_png()}),
        _payload(images=["not", "a", "dict"]),
        _payload(images={"overhead_left": _b64_png(shape=(1100, 20, 3)), "wrist": _b64_png()}),
        _payload(instruction=""),
        _payload(instruction=7),
        _payload(joint_state=[1, 2, 3]),
        _payload(reset="yes"),
        ["not", "an", "object"],
    ],
    ids=["no-images", "no-primary", "no-wrist", "bad-base64", "bad-png", "images-not-dict", "oversized-image",
         "empty-instruction", "non-str-instruction", "joint-state-not-dict", "reset-not-bool", "non-object-body"],
)
def test_malformed_body_returns_400(fake_policy, body):
    model = OctoModel(SO101_CKPT, revision=SO101_REV)
    response = create_app(model).test_client().post("/action", json=body)
    assert response.status_code == 400
    assert "error" in response.get_json()
    assert fake_policy.samples == []


def test_non_json_body_returns_400(fake_policy):
    model = OctoModel(SO101_CKPT, revision=SO101_REV)
    response = create_app(model).test_client().post("/action", data="garbage", content_type="text/plain")
    assert response.status_code == 400


@pytest.mark.parametrize("chunk", [np.full((1, 4, 7), np.nan), np.zeros((1, 4, 5)), np.zeros((4, 7))],
                         ids=["nan", "wrong-dim", "no-batch"])
def test_bad_model_output_returns_500(fake_policy, chunk):
    fake_policy.chunk = chunk
    model = OctoModel(SO101_CKPT, revision=SO101_REV)
    response = create_app(model).test_client().post("/action", json=_payload())
    assert response.status_code == 500
    assert "octo inference failed" in response.get_json()["error"]


def test_health_reports_ready_with_metadata(fake_policy):
    model = OctoModel(SO101_CKPT, revision=SO101_REV)
    body = create_app(model).test_client().get("/health").get_json()
    assert body["status"] == "ready"
    meta = body["model"]
    assert meta["checkpoint"] == SO101_CKPT
    assert meta["action_space"] == "eef_delta"
    assert meta["window_size"] == 2 and meta["action_horizon"] == 4
    assert meta["image_keys"] == {"image_primary": "overhead_left", "image_wrist": "wrist"}
    assert meta["action_mask"] == [True] * 6 + [False]


def test_overlapping_request_gets_429_not_queued():
    started, release = threading.Event(), threading.Event()

    class Slow:
        metadata = {"checkpoint": "slow"}
        model_version = "slow@test"
        required_image_keys = ("overhead_left",)

        def __call__(self, images, instruction):
            started.set()
            release.wait(5)
            return list(EEF_RAW)

    http = create_app(Slow()).test_client()
    results = []
    worker = threading.Thread(target=lambda: results.append(http.post("/action", json=_payload()).status_code))
    worker.start()
    assert started.wait(5)
    assert http.post("/action", json=_payload()).status_code == 429
    release.set()
    worker.join(5)
    assert results == [200]


def test_download_pinned_fetches_only_the_chosen_step(monkeypatch, tmp_path):
    import huggingface_hub

    calls = []
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **kw: calls.append(kw) or kw["local_dir"])
    path = octo_server._download_pinned(SO101_CKPT, SO101_REV, str(tmp_path), 10000)
    (call,) = calls
    assert call["repo_id"] == SO101_CKPT and call["revision"] == SO101_REV
    assert set(call["allow_patterns"]) == {"10000/*", "config.json", "dataset_statistics.json",
                                           "example_batch.msgpack"}
    assert path == call["local_dir"] and SO101_REV in path


def _octo_config(encoder="t5-base", tokenizer="t5-base"):
    return {
        "model": {"task_tokenizers": {"language": {
            "module": "octo.model.components.tokenizers", "name": "LanguageTokenizer",
            "kwargs": {"encoder": encoder, "finetune_encoder": False}}}, "heads": {}},
        "text_processor": {"module": "octo.data.utils.text_processing", "name": "HFTokenizer",
                           "kwargs": {"tokenizer_name": tokenizer, "encode_with_model": False}},
    }


def test_pin_text_encoder_rewrites_t5_base_to_the_pinned_local_copy(monkeypatch, tmp_path):
    import huggingface_hub

    calls = []

    def fake_snapshot(**kw):
        calls.append(kw)
        return kw["local_dir"]

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps(_octo_config()))
    octo_server._pin_text_encoder(str(model_dir), str(tmp_path))
    octo_server._pin_text_encoder(str(model_dir), str(tmp_path))  # idempotent
    assert calls[0]["repo_id"] == "google-t5/t5-base"
    assert calls[0]["revision"] == octo_server.T5_REVISION
    config = json.loads((model_dir / "config.json").read_text())
    encoder = config["model"]["task_tokenizers"]["language"]["kwargs"]["encoder"]
    assert encoder == calls[0]["local_dir"] == config["text_processor"]["kwargs"]["tokenizer_name"]
    assert "t5" in encoder  # LanguageTokenizer picks FlaxT5EncoderModel on '"t5" in encoder'


def test_pin_text_encoder_refuses_an_unreviewed_encoder(monkeypatch, tmp_path):
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **kw: kw["local_dir"])
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps(_octo_config(encoder="someone/evil-encoder")))
    with pytest.raises(RuntimeError, match="someone/evil-encoder"):
        octo_server._pin_text_encoder(str(model_dir), str(tmp_path))


def test_server_main_checks_deps_warms_then_clears_history_before_serving(monkeypatch):
    events = []

    class FakeModel:
        metadata = {"checkpoint": "c"}
        model_version = "c@r"
        required_image_keys = ("overhead_left", "wrist")

        def __init__(self, checkpoint, **kwargs):
            events.append(("init", checkpoint, kwargs))

        def __call__(self, images, instruction):
            events.append(("infer", sorted(images), instruction))
            return list(EEF_RAW)

        def reset(self):
            events.append(("reset",))

    class FakeApp:
        def run(self, **kwargs):
            events.append(("run", kwargs))

    monkeypatch.setattr(octo_server, "_check_dependency_versions", lambda: events.append(("deps",)))
    monkeypatch.setattr(octo_server, "OctoModel", FakeModel)
    monkeypatch.setattr(octo_server, "create_app", lambda model: FakeApp())
    monkeypatch.setattr("sys.argv", ["octo_server.py", "--port", "3004"])
    octo_server.main()
    assert [e[0] for e in events] == ["deps", "init", "infer", "reset", "run"]
    assert events[1][1] == octo_server.DEFAULT_CHECKPOINT == SO101_CKPT
    assert events[1][2]["revision"] == SO101_REV and events[1][2]["step"] == 10000
    assert events[2][1] == ["overhead_left", "wrist"]
    assert events[4][1]["port"] == 3004 == octo_server.DEFAULT_PORT
    assert events[4][1]["use_reloader"] is False


def test_server_main_uses_profile_revision_and_step_for_a_known_alternate(monkeypatch):
    seen = {}

    class FakeModel:
        metadata, model_version, required_image_keys = {}, "m", ("overhead_left",)

        def __init__(self, checkpoint, **kwargs):
            seen.update(kwargs, checkpoint=checkpoint)

        def __call__(self, images, instruction):
            return list(EEF_RAW)

        def reset(self):
            pass

    monkeypatch.setattr(octo_server, "_check_dependency_versions", lambda: None)
    monkeypatch.setattr(octo_server, "OctoModel", FakeModel)
    monkeypatch.setattr(octo_server, "create_app", lambda model: SimpleNamespace(run=lambda **k: None))
    monkeypatch.setattr("sys.argv", ["octo_server.py", "--checkpoint", SMALL_CKPT])
    octo_server.main()
    assert seen["checkpoint"] == SMALL_CKPT and seen["revision"] == SMALL_REV and seen["step"] == 300000


@pytest.mark.parametrize("argv,match", [
    (["--checkpoint", "someone/octo-custom"], "--revision"),
    (["--checkpoint", "someone/octo-custom", "--revision", "abc123"], "--step"),
])
def test_server_main_refuses_an_unreviewed_checkpoint_without_explicit_pins(monkeypatch, argv, match):
    monkeypatch.setattr(octo_server, "_check_dependency_versions", lambda: None)
    monkeypatch.setattr("sys.argv", ["octo_server.py", *argv])
    with pytest.raises(SystemExit, match=match):
        octo_server.main()


def test_dependency_check_refuses_an_unpinned_jax(monkeypatch):
    good = {name: sorted(versions)[0] for name, versions in octo_server.PINNED_DEPENDENCIES.items()}
    monkeypatch.setattr(octo_server, "_installed_versions", lambda: dict(good, jax="0.4.30"))
    with pytest.raises(SystemExit, match="jax"):
        octo_server._check_dependency_versions()
    monkeypatch.setattr(octo_server, "_installed_versions", lambda: dict(good))
    octo_server._check_dependency_versions()


# --- control-side client ------------------------------------------------------


def test_octo_is_registered_on_the_http_lane():
    assert backends.BACKEND_REGISTRY["octo"] is octo.build
    assert "octo" not in backends.GRPC_BRIDGE_BACKENDS


def test_backend_config_pins_checkpoint_and_records_the_confirmed_action_space():
    config = backends.load_backend_config("octo")
    assert config["checkpoint"] == SO101_CKPT == octo_server.DEFAULT_CHECKPOINT
    assert config["revision"] == SO101_REV == octo_server.DEFAULT_REVISION
    assert config["endpoint_url"] is None
    # Read from the checkpoint's own dataset_statistics.json / dataset card: a
    # 7-D [dx, dy, dz, droll, dpitch, dyaw, gripper] EEF delta (mask [T]*6+[F]).
    assert config["action_space"] == "eef_delta"
    assert config["checkpoints"][config["checkpoint"]]["action_space"] == config["action_space"]


def test_client_and_server_checkpoint_profiles_agree():
    profiles = backends.load_backend_config("octo")["checkpoints"]
    assert set(profiles) == set(octo_server.CHECKPOINT_PROFILES)
    for checkpoint, profile in profiles.items():
        server = octo_server.CHECKPOINT_PROFILES[checkpoint]
        for key in ("revision", "step", "unnorm_key", "action_space"):
            assert profile[key] == server[key], (checkpoint, key)
        assert profile["action_space"] in octo.ACTION_SPACES
        if profile["action_space"] == "eef_delta":
            lo, hi = profile["gripper_norm_to_percent"]
            assert hi > lo


def test_eef_delta_rpy_difference_is_composed_exactly_at_the_current_pose():
    delta = octo.octo_action_to_eef_delta(EEF_RAW, JOINT_STATE, (-10.0, 100.0))
    assert (delta["dx"], delta["dy"], delta["dz"]) == pytest.approx(tuple(EEF_RAW[:3]))
    pose = eef_to_joint.forward_kinematics(JOINT_STATE)
    expected = Rotation.from_euler("xyz", np.add(pose[3:], EEF_RAW[3:6])).as_matrix()
    # solve_ik's goal is from_euler("xyz", delta) @ R_current.
    got = Rotation.from_euler("xyz", [delta["droll"], delta["dpitch"], delta["dyaw"]]).as_matrix() \
        @ Rotation.from_euler("xyz", pose[3:]).as_matrix()
    assert got == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("norm,lo_hi,percent", [
    (0.5, (-10.0, 100.0), 45.0),  # SO-101 3 Hz conversion: [-10, 100] min-max
    (0.0, (-10.0, 100.0), -10.0),  # below 0 %: solve_ik clamps it
    (1.0, (0.0, 100.0), 100.0),  # bridge_dataset: 1 = open
])
def test_gripper_is_mapped_back_through_the_profile_range(norm, lo_hi, percent):
    raw = list(EEF_RAW[:6]) + [norm]
    assert octo.octo_action_to_eef_delta(raw, JOINT_STATE, lo_hi)["gripper"] == pytest.approx(percent)


def test_non_finite_eef_values_pass_through_as_hold():
    raw = [0.01, 0.0, 0.0, float("nan"), 0.0, 0.0, float("inf")]
    delta = octo.octo_action_to_eef_delta(raw, JOINT_STATE, (0.0, 100.0))
    assert math.isnan(delta["droll"]) and math.isnan(delta["dyaw"])  # solve_ik: hold the pose
    assert math.isnan(delta["gripper"])  # solve_ik: keep the current gripper


@pytest.mark.parametrize("raw", [[0.0] * 6, [0.0] * 8, ["a"] * 7, {"dx": 0.1}, None])
def test_malformed_eef_action_raises(raw):
    with pytest.raises((ValueError, TypeError)):
        octo.octo_action_to_eef_delta(raw, JOINT_STATE, (0.0, 100.0))


@pytest.mark.parametrize("raw", [[0.0] * 7, ["a"] * 6, {"shoulder_pan": 1.0}, None])
def test_malformed_joint_action_raises(raw):
    with pytest.raises((ValueError, TypeError)):
        octo.joint_absolute_action(raw)


def test_eef_delta_convert_fn_runs_solve_ik_with_joint_state_and_limits(monkeypatch):
    calls = []
    sentinel = {joint: 1.0 for joint in JOINT_ORDER}

    def fake_solve_ik(joint_state, eef_delta, **kwargs):
        calls.append((joint_state, eef_delta, kwargs))
        return sentinel

    monkeypatch.setattr(eef_to_joint, "solve_ik", fake_solve_ik)
    limits = {"shoulder_pan": (-90.0, 90.0)}
    profile = {"action_space": "eef_delta", "gripper_norm_to_percent": [-10.0, 100.0]}
    convert = octo.make_convert_fn(profile, limits)
    assert convert(EEF_RAW, JOINT_STATE) is sentinel
    joint_state, eef_delta, kwargs = calls[0]
    assert joint_state is JOINT_STATE
    assert eef_delta == octo.octo_action_to_eef_delta(EEF_RAW, JOINT_STATE, (-10.0, 100.0))
    assert kwargs == {"joint_limits_deg": limits}


def test_joint_absolute_convert_fn_is_identity_without_ik(monkeypatch):
    monkeypatch.setattr(eef_to_joint, "solve_ik", lambda *a, **k: pytest.fail("joint space must not run IK"))
    convert = octo.make_convert_fn({"action_space": "joint_absolute"}, {"shoulder_pan": (-90.0, 90.0)})
    action = convert(JOINT_RAW, JOINT_STATE)
    assert tuple(action) == JOINT_ORDER
    assert [action[j] for j in JOINT_ORDER] == JOINT_RAW


@pytest.mark.parametrize("profile", [{"action_space": "joint_velocity"}, {"action_space": "eef_delta"}, {}])
def test_unknown_or_incomplete_profile_is_rejected(profile):
    with pytest.raises(ValueError, match="action_space|gripper_norm_to_percent"):
        octo.make_convert_fn(profile, None)


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
    (no_config / "octo.json").write_text(json.dumps({"checkpoint": "c", "endpoint_url": None}))
    with pytest.raises(SystemExit, match="--server-address"):
        octo.build(_args(), {0: "wrist"}, {})


def test_build_without_config_file_raises_clear_systemexit(no_config):
    with pytest.raises(SystemExit, match="endpoint_url"):
        octo.build(_args(), {0: "wrist"}, {})


def test_build_rejects_a_grpc_style_host_port_endpoint(no_config):
    with pytest.raises(SystemExit, match="http"):
        octo.build(_args(server_address="0.tcp.ngrok.io:12345"), {0: "wrist"}, {})


def test_build_rejects_a_checkpoint_without_a_profile():
    with pytest.raises(SystemExit, match="action_space"):
        octo.build(_args(server_address="https://x.ngrok-free.app", checkpoint="someone/other"), {0: "wrist"}, {})


def _capture_build(monkeypatch):
    seen = {}

    def fake_build(endpoint_url, checkpoint, args, camera_names, convert_fn=None):
        seen.update(endpoint_url=endpoint_url, checkpoint=checkpoint, camera_names=camera_names,
                    convert_fn=convert_fn)
        return "handles"

    monkeypatch.setattr(base_http_adapter, "build_http_adapter_backend", fake_build)
    return seen


def test_build_defaults_to_the_so101_eef_delta_checkpoint_and_wires_ik(monkeypatch):
    seen = _capture_build(monkeypatch)
    calls = []
    monkeypatch.setattr(eef_to_joint, "solve_ik", lambda js, delta, **k: calls.append((js, delta, k)) or "ik")
    limits = {"elbow_flex": (-80.0, 80.0)}
    assert octo.build(_args(server_address="https://abc.ngrok-free.app"), {0: "wrist"}, limits) == "handles"
    assert seen["endpoint_url"] == "https://abc.ngrok-free.app"
    assert seen["checkpoint"] == SO101_CKPT
    assert seen["camera_names"] == {0: "wrist"}
    assert seen["convert_fn"](EEF_RAW, JOINT_STATE) == "ik"
    assert calls[0][1] == octo.octo_action_to_eef_delta(EEF_RAW, JOINT_STATE, (-10.0, 100.0))
    assert calls[0][2] == {"joint_limits_deg": limits}


def test_build_action_space_from_config_selects_the_branch(monkeypatch, no_config):
    """The config's declared action_space drives the conversion: a
    joint_absolute profile is an identity pass-through, never IK."""
    (no_config / "octo.json").write_text(json.dumps({
        "checkpoint": "someone/octo-joint", "endpoint_url": "https://cfg.example", "action_space": "joint_absolute",
        "checkpoints": {"someone/octo-joint": {"action_space": "joint_absolute"}}}))
    seen = _capture_build(monkeypatch)
    monkeypatch.setattr(eef_to_joint, "solve_ik", lambda *a, **k: pytest.fail("joint space must not run IK"))
    octo.build(_args(), {0: "wrist"}, {})
    assert seen["endpoint_url"] == "https://cfg.example" and seen["checkpoint"] == "someone/octo-joint"
    assert seen["convert_fn"](JOINT_RAW, JOINT_STATE) == dict(zip(JOINT_ORDER, JOINT_RAW))


def test_build_with_official_alternate_uses_its_gripper_range(monkeypatch):
    seen = _capture_build(monkeypatch)
    calls = []
    monkeypatch.setattr(eef_to_joint, "solve_ik", lambda js, delta, **k: calls.append(delta) or "ik")
    octo.build(_args(server_address="http://h:3004", checkpoint=SMALL_CKPT), {0: "wrist"}, {})
    seen["convert_fn"](list(EEF_RAW[:6]) + [1.0], JOINT_STATE)
    assert calls[0]["gripper"] == pytest.approx(100.0)
