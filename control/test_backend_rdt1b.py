"""Plan 14-11 (VLAB-09): RDT-1B over the Tier C resident HTTP adapter.

Colab-side `rdt1b_server.py` is tested with the upstream RDT wrapper, the
pinned downloads, the T5-XXL encoder and `step()` mocked. Client-side
`backends/rdt1b.py` is tested with `build_http_adapter_backend` (robot and
camera construction) mocked. The joint-position conversion and the eef_pose
fallback's real ikpy `solve_ik` run unmocked.
"""

import base64
import json
import math
import sys
import threading
import types
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from vla_bridge import backends
from vla_bridge import rdt1b_server
from vla_bridge.action_contract import JOINT_ORDER
from vla_bridge.backends import base_http_adapter, eef_to_joint, rdt1b
from vla_bridge.rdt1b_server import RDT1BModel, create_app

RDT_CKPT = "robotics-diffusion-transformer/rdt-1b"
RDT_REV = "eb09036cc64ca4945051acbd1bd581d30a1d7711"
SMALL_CKPT = "robotics-diffusion-transformer/rdt-170m"
SMALL_REV = "8aa386cac3bbfd9540676c75b3d767cc7f88a10a"

JOINT_STATE = {
    "shoulder_pan": 10.0,
    "shoulder_lift": -20.0,
    "elbow_flex": 30.0,
    "wrist_flex": 15.0,
    "wrist_roll": 5.0,
    "gripper": 37.0,
}
# One chunk step of RDT's SO-101 slots: 5 arm joints (rad), gripper_open (0-1).
JOINT_RAW = [math.radians(v) for v in (12.0, -18.5, 33.0, 20.0, 4.0)] + [0.41]
JOINT_EXPECTED = dict(zip(JOINT_ORDER, [12.0, -18.5, 33.0, 20.0, 4.0, 41.0]))


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


class FakeRDT:
    """Stands in for upstream `scripts.agilex_model.RoboticDiffusionTransformerModel`."""

    def __init__(self, img_history_size=2, num_cameras=3):
        self.args = {"common": {"img_history_size": img_history_size, "action_chunk_size": 64,
                                "num_cameras": num_cameras, "state_dim": 128},
                     "dataset": {"tokenizer_max_length": 1024},
                     "model": {"state_token_dim": 128}}


@pytest.fixture
def fake_policy(monkeypatch):
    """Mock `_download_pinned` (snapshot_download), `_load_policy` (the upstream
    wrapper + RDTRunner.from_pretrained), `_load_text_encoder` (T5Embedder),
    `_encode_instruction` (the T5-XXL forward) and `_step` (`model.step()`)."""
    state = SimpleNamespace(loads=[], downloads=[], encoders=[], encodes=[], steps=[], chunk=None, model=None)

    def download(repo_id, revision, local_dir, patterns):
        state.downloads.append((repo_id, revision, local_dir, tuple(patterns)))
        return f"{local_dir}/{repo_id}@{revision}"

    def load_policy(rdt_repo, model_dir, vision_dir, control_hz, state_indices, device):
        state.model = FakeRDT()
        state.loads.append(dict(rdt_repo=rdt_repo, model_dir=model_dir, vision_dir=vision_dir,
                                control_hz=control_hz, state_indices=list(state_indices), device=device))
        return state.model

    def load_text_encoder(rdt_repo, t5_dir, offload_dir, max_length, device):
        state.encoders.append(dict(rdt_repo=rdt_repo, t5_dir=t5_dir, offload_dir=offload_dir,
                                   max_length=max_length, device=device))
        return "t5-embedder"

    def encode(embedder, instruction, device):
        state.encodes.append((embedder, instruction))
        return f"emb:{instruction}"

    def step(model, proprio, images, text_embeds):
        state.steps.append(dict(proprio=list(proprio), images=list(images), text_embeds=text_embeds))
        if state.chunk is not None:
            return np.asarray(state.chunk, dtype=np.float32)
        chunk = np.tile(np.asarray(JOINT_RAW, dtype=np.float32), (1, 64, 1))
        chunk[0, :14] -= 1.0  # only the chosen chunk step (14 at 2 Hz) carries JOINT_RAW
        chunk[0, 15:] += 1.0
        return chunk

    monkeypatch.setattr(rdt1b_server, "_download_pinned", download)
    monkeypatch.setattr(rdt1b_server, "_load_policy", load_policy)
    monkeypatch.setattr(rdt1b_server, "_load_text_encoder", load_text_encoder)
    monkeypatch.setattr(rdt1b_server, "_encode_instruction", encode)
    monkeypatch.setattr(rdt1b_server, "_step", step)
    return state


def test_model_loads_once_across_many_action_requests(fake_policy):
    model = RDT1BModel(local_dir="/ckpt")
    client = create_app(model).test_client()
    for _ in range(3):
        assert client.post("/action", json=_payload()).status_code == 200
    assert len(fake_policy.loads) == 1
    assert len(fake_policy.encoders) == 1
    assert len(fake_policy.steps) == 3


def test_pinned_checkpoint_and_encoders_are_downloaded_at_reviewed_revisions(fake_policy):
    RDT1BModel(local_dir="/ckpt", rdt_repo="/repo")
    assert fake_policy.downloads == [
        (RDT_CKPT, RDT_REV, "/ckpt", ("config.json", "pytorch_model.bin")),
        (rdt1b_server.SIGLIP_REPO, rdt1b_server.SIGLIP_REVISION, "/ckpt",
         ("config.json", "preprocessor_config.json", "model.safetensors")),
        (rdt1b_server.T5_REPO, rdt1b_server.T5_REVISION, "/ckpt", rdt1b_server.T5_FILES),
    ]
    load = fake_policy.loads[0]
    assert load["model_dir"] == f"/ckpt/{RDT_CKPT}@{RDT_REV}"
    assert load["vision_dir"] == f"/ckpt/{rdt1b_server.SIGLIP_REPO}@{rdt1b_server.SIGLIP_REVISION}"
    assert load["rdt_repo"] == "/repo" and load["control_hz"] == 30
    assert load["state_indices"] == [0, 1, 2, 3, 4, 10]


def test_alternate_checkpoint_uses_its_own_pinned_revision(fake_policy):
    model = RDT1BModel(SMALL_CKPT, local_dir="/ckpt")
    assert fake_policy.downloads[0][:2] == (SMALL_CKPT, SMALL_REV)
    assert model.model_version == f"{SMALL_CKPT}@{SMALL_REV[:12]}"


def test_offload_dir_is_threaded_into_the_text_encoder_load(fake_policy):
    model = RDT1BModel(local_dir="/ckpt", offload_dir="/content/rdt_t5_offload")
    encoder = fake_policy.encoders[0]
    assert encoder["offload_dir"] == "/content/rdt_t5_offload"
    assert encoder["t5_dir"] == f"/ckpt/{rdt1b_server.T5_REPO}@{rdt1b_server.T5_REVISION}"
    assert encoder["max_length"] == 1024
    assert model.metadata["t5_offload_dir"] == "/content/rdt_t5_offload"


@pytest.mark.parametrize("mode,total_gib,expected", [
    ("auto", 15.0, "/off"), ("auto", 22.5, "/off"), ("auto", 40.0, None), ("auto", 80.0, None),
    ("on", 80.0, "/off"), ("off", 15.0, None),
])
def test_offload_follows_the_upstream_24gb_guidance(mode, total_gib, expected):
    assert rdt1b_server.resolve_offload_dir(mode, "/off", total_gib) == expected


@pytest.mark.parametrize("control_hz,client_hz,expected", [(30, 2.0, 14), (30, 30.0, 0), (30, 1.0, 29),
                                                           (30, 0.1, 63), (30, 60.0, 0)])
def test_chunk_step_is_about_one_client_period_ahead(control_hz, client_hz, expected):
    assert rdt1b_server.chunk_step_for(control_hz, client_hz, 64) == expected


def test_action_endpoint_returns_the_chosen_chunk_step_and_model_version(fake_policy):
    model = RDT1BModel(local_dir="/ckpt")
    response = create_app(model).test_client().post("/action", json=_payload())
    assert response.status_code == 200
    body = response.get_json()
    assert body["action"] == pytest.approx(JOINT_RAW, abs=1e-6)
    assert body["model_version"] == f"{RDT_CKPT}@{RDT_REV[:12]}"
    assert body["inference_ms"] >= 0


def test_model_gets_model_unit_proprio_rdt_image_order_and_cached_text(fake_policy):
    model = RDT1BModel(local_dir="/ckpt")
    client = create_app(model).test_client()
    client.post("/action", json=_payload())
    client.post("/action", json=_payload())
    first, second = fake_policy.steps
    expected = [math.radians(JOINT_STATE[j]) for j in JOINT_ORDER[:5]] + [0.37]
    assert first["proprio"] == pytest.approx(expected)
    # [ext, right_wrist, left_wrist] x [t-1, t]; single arm: left wrist = background (None)
    assert first["images"][:3] == [None, None, None]
    ext, right_wrist, left_wrist = first["images"][3:]
    assert ext.shape == (24, 32, 3) and int(ext[0, 0, 0]) == 20  # overhead_left, RGB
    assert int(right_wrist[0, 0, 0]) == 10 and left_wrist is None  # wrist
    assert second["images"][0] is ext and second["images"][1] is right_wrist
    assert first["text_embeds"] == "emb:pick up the cube"
    assert fake_policy.encodes == [("t5-embedder", "pick up the cube")]


def test_instruction_is_encoded_once_and_cached(fake_policy):
    model = RDT1BModel(local_dir="/ckpt")
    client = create_app(model).test_client()
    for instruction in ("a", "b", "a", "b"):
        client.post("/action", json=_payload(instruction=instruction))
    assert [i for _, i in fake_policy.encodes] == ["a", "b"]


def test_history_resets_on_new_instruction_reset_flag_and_idle(fake_policy, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(rdt1b_server.time, "monotonic", lambda: clock[0])
    model = RDT1BModel(local_dir="/ckpt")
    client = create_app(model).test_client()
    client.post("/action", json=_payload())                      # fresh
    client.post("/action", json=_payload())                      # history
    client.post("/action", json=_payload(instruction="other"))   # new instruction
    client.post("/action", json=_payload(instruction="other", reset=True))
    clock[0] += rdt1b_server.HISTORY_RESET_S + 1
    client.post("/action", json=_payload(instruction="other"))   # idle gap
    fresh = [step["images"][0] is None for step in fake_policy.steps]
    assert fresh == [True, False, True, True, True]


def test_ext_image_key_selects_the_exterior_camera(fake_policy):
    model = RDT1BModel(local_dir="/ckpt", ext_image_key="overhead_right")
    create_app(model).test_client().post("/action", json=_payload())
    assert int(fake_policy.steps[0]["images"][3][0, 0, 0]) == 30
    assert model.required_image_keys == ("overhead_right", "wrist")


def test_precomputed_only_serves_known_instructions(fake_policy, monkeypatch):
    monkeypatch.setattr(rdt1b_server, "_load_lang_embedding", lambda path: ("pick up the cube", f"pre:{path}"))
    model = RDT1BModel(local_dir="/ckpt", lang_embeddings=["/e/cube.pt"], text_encoder=False)
    client = create_app(model).test_client()
    assert client.post("/action", json=_payload()).status_code == 200
    assert fake_policy.steps[0]["text_embeds"] == "pre:/e/cube.pt"
    response = client.post("/action", json=_payload(instruction="stack the blocks"))
    assert response.status_code == 400 and "precomputed" in response.get_json()["error"]
    assert fake_policy.encoders == [] and len(fake_policy.steps) == 1
    assert model.metadata["text_encoder"] == "precomputed only"
    assert model.metadata["precomputed_instructions"] == ["pick up the cube"]
    assert [d[0] for d in fake_policy.downloads] == [RDT_CKPT, rdt1b_server.SIGLIP_REPO]  # no 44.5 GB T5


def test_precomputed_only_without_embeddings_is_refused(fake_policy):
    with pytest.raises(ValueError, match="lang-embeddings"):
        RDT1BModel(local_dir="/ckpt", text_encoder=False)


def test_unreviewed_checkpoint_is_refused(fake_policy):
    with pytest.raises(ValueError, match="reviewed"):
        RDT1BModel("someone/rdt-finetune", local_dir="/ckpt")


def test_eef_profile_is_refused_server_side(fake_policy, monkeypatch):
    monkeypatch.setitem(rdt1b_server.CHECKPOINT_PROFILES, "someone/rdt-eef",
                        dict(rdt1b_server.CHECKPOINT_PROFILES[RDT_CKPT], action_space="eef_pose"))
    with pytest.raises(RuntimeError, match="EEF"):
        RDT1BModel("someone/rdt-eef", local_dir="/ckpt")


def test_unexpected_image_history_layout_is_refused(fake_policy, monkeypatch):
    monkeypatch.setattr(rdt1b_server, "_load_policy", lambda *a: FakeRDT(img_history_size=3))
    with pytest.raises(RuntimeError, match="2 x 3"):
        RDT1BModel(local_dir="/ckpt")


@pytest.mark.parametrize("body", [
    {"images": {}, "instruction": "x", "joint_state": JOINT_STATE},
    {"images": {"wrist": "%%%", "overhead_left": "%%%"}, "instruction": "x", "joint_state": JOINT_STATE},
    {"images": "nope", "instruction": "x", "joint_state": JOINT_STATE},
    _payload(instruction=""),
    _payload(instruction=7),
    _payload(instruction="x" * 1001),
    _payload(reset="yes"),
    {k: v for k, v in _payload().items() if k != "joint_state"},
    _payload(joint_state="nope"),
    _payload(joint_state={k: v for k, v in JOINT_STATE.items() if k != "gripper"}),
    _payload(joint_state=dict(JOINT_STATE, elbow_flex=None)),
    _payload(joint_state=dict(JOINT_STATE, elbow_flex=True)),
    _payload(joint_state=dict(JOINT_STATE, elbow_flex="1")),
])
def test_malformed_body_returns_400(fake_policy, body):
    model = RDT1BModel(local_dir="/ckpt")
    response = create_app(model).test_client().post("/action", json=body)
    assert response.status_code == 400
    assert fake_policy.steps == []


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_non_finite_joint_state_is_a_bad_request_not_a_model_call(fake_policy, bad):
    model = RDT1BModel(local_dir="/ckpt")
    with pytest.raises(ValueError, match="finite"):
        model({}, "x", dict(JOINT_STATE, wrist_roll=bad))
    assert fake_policy.steps == []


def test_non_json_body_returns_400(fake_policy):
    model = RDT1BModel(local_dir="/ckpt")
    response = create_app(model).test_client().post("/action", data="not json", content_type="text/plain")
    assert response.status_code == 400


@pytest.mark.parametrize("chunk", [np.full((1, 64, 6), np.nan), np.zeros((1, 64, 7)), np.zeros((64, 6)),
                                   np.zeros((1, 10, 6))])
def test_bad_model_output_returns_500(fake_policy, chunk):
    fake_policy.chunk = chunk
    model = RDT1BModel(local_dir="/ckpt")
    response = create_app(model).test_client().post("/action", json=_payload())
    assert response.status_code == 500
    assert "rdt inference failed" in response.get_json()["error"]


def test_health_reports_ready_with_metadata(fake_policy):
    model = RDT1BModel(local_dir="/ckpt", offload_dir="/off")
    body = create_app(model).test_client().get("/health").get_json()
    assert body["status"] == "ready"
    meta = body["model"]
    assert meta["checkpoint"] == RDT_CKPT and meta["revision"] == RDT_REV
    assert meta["rdt_commit"] == rdt1b_server.RDT_COMMIT
    assert meta["action_space"] == "joint_position"
    assert meta["state_indices"] == [0, 1, 2, 3, 4, 10]
    assert meta["joint_units"] == "rad" and meta["gripper_norm_to_percent"] == [0.0, 100.0]
    assert meta["action_horizon"] == 64 and meta["chunk_step"] == 14 and meta["control_hz"] == 30
    assert meta["text_encoder"] == f"{rdt1b_server.T5_REPO}@{rdt1b_server.T5_REVISION}"
    assert meta["vision_encoder"] == f"{rdt1b_server.SIGLIP_REPO}@{rdt1b_server.SIGLIP_REVISION}"
    assert meta["image_order"] == {"ext": "overhead_left", "right_wrist": "wrist", "left_wrist": "background"}


def test_overlapping_request_gets_429_not_queued():
    started, release = threading.Event(), threading.Event()

    class SlowModel:
        metadata = {}
        model_version = "slow"
        required_image_keys = ("overhead_left", "wrist")

        def __call__(self, images, instruction, joint_state, reset=False):
            started.set()
            release.wait(5)
            return JOINT_RAW

    app = create_app(SlowModel())
    results = []
    worker = threading.Thread(target=lambda: results.append(app.test_client().post("/action", json=_payload())))
    worker.start()
    assert started.wait(5)
    second = app.test_client().post("/action", json=_payload())
    release.set()
    worker.join(5)
    assert second.status_code == 429
    assert results[0].status_code == 200


# --- upstream seams (fake upstream modules, real torch) ------------------------


def _fake_repo(tmp_path, monkeypatch, modules):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "agilex_model.py").write_text("# fake\n")
    monkeypatch.setattr(sys, "path", list(sys.path))
    for name, attrs in modules.items():
        parts = name.split(".")
        for i in range(1, len(parts)):
            parent = ".".join(parts[:i])
            if parent not in sys.modules or not getattr(sys.modules[parent], "_rdt_fake", False):
                package = types.ModuleType(parent)
                package._rdt_fake = True
                package.__path__ = []
                monkeypatch.setitem(sys.modules, parent, package)
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    return str(tmp_path)


def test_load_text_encoder_passes_the_offload_folder_to_upstream_t5embedder(tmp_path, monkeypatch):
    calls = []

    class FakeT5Embedder:
        available_models = ["google/t5-v1_1-xxl"]

        def __init__(self, device, from_pretrained=None, **kwargs):
            assert from_pretrained in self.available_models  # upstream's own assert
            calls.append(dict(device=device, from_pretrained=from_pretrained, **kwargs))

    repo = _fake_repo(tmp_path, monkeypatch, {"models.multimodal_encoder.t5_encoder": {"T5Embedder": FakeT5Embedder}})
    offload = tmp_path / "offload"
    rdt1b_server._load_text_encoder(repo, "/ckpt/t5", str(offload), 1024, "cuda")
    assert calls == [dict(device="cuda", from_pretrained="/ckpt/t5", model_max_length=1024,
                          use_offload_folder=str(offload))]
    assert offload.is_dir()
    rdt1b_server._load_text_encoder(repo, "/ckpt/t5", None, 1024, "cuda")
    assert calls[-1]["use_offload_folder"] is None
    assert FakeT5Embedder.available_models.count("/ckpt/t5") == 1


def test_load_policy_overrides_the_documented_format_hooks_and_verifies_weights(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    seen = {}

    class FakeUpstream:
        def __init__(self, args, device="cuda", dtype=None, image_size=None, control_frequency=25,
                     pretrained=None, pretrained_vision_encoder_name_or_path=None):
            seen.update(args=args, device=device, dtype=dtype, control_frequency=control_frequency,
                        pretrained=pretrained, vision=pretrained_vision_encoder_name_or_path)
            self.args = args
            self.policy = torch.nn.Linear(2, 2)

    repo = _fake_repo(tmp_path, monkeypatch, {"scripts.agilex_model": {"RoboticDiffusionTransformerModel": FakeUpstream}})
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "base.yaml").write_text("model:\n  state_token_dim: 128\n")
    model_dir = tmp_path / "ckpt"
    model_dir.mkdir()
    torch.save(torch.nn.Linear(2, 2).state_dict(), model_dir / "pytorch_model.bin")

    model = rdt1b_server._load_policy(repo, str(model_dir), "/vision", 30, [0, 1, 2, 3, 4, 10], "cpu")
    assert seen["dtype"] is torch.bfloat16 and seen["control_frequency"] == 30
    assert seen["pretrained"] == str(model_dir) and seen["vision"] == "/vision" and seen["device"] == "cpu"
    assert seen["args"] == {"model": {"state_token_dim": 128}}

    joints = torch.tensor([[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]])
    state, mask = model._format_joint_to_state(joints)
    assert state.shape == (1, 1, 128) and mask.shape == (1, 128)
    assert state[0, 0, [0, 1, 2, 3, 4, 10]].tolist() == pytest.approx(joints[0, 0].tolist())
    assert float(state.sum()) == pytest.approx(float(joints.sum()))
    assert mask[0].nonzero().flatten().tolist() == [0, 1, 2, 3, 4, 10]
    action = torch.arange(128.0).reshape(1, 1, 128).repeat(1, 64, 1)
    assert model._unformat_action_to_joint(action)[0, 0].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 10.0]

    torch.save({"weight": torch.zeros(2, 2)}, model_dir / "pytorch_model.bin")  # missing "bias"
    with pytest.raises(RuntimeError, match="missing"):
        rdt1b_server._load_policy(repo, str(model_dir), "/vision", 30, [0, 1, 2, 3, 4, 10], "cpu")


def test_add_repo_to_path_refuses_a_directory_that_is_not_the_rdt_repo(tmp_path):
    with pytest.raises(RuntimeError, match="RoboticsDiffusionTransformer"):
        rdt1b_server._add_repo_to_path(str(tmp_path))


def _state_vec_module(**overrides):
    mapping = {f"right_arm_joint_{i}_pos": i for i in range(10)}
    mapping.update(right_gripper_open=10)
    mapping.update(overrides)
    return {"STATE_VEC_IDX_MAPPING": mapping, "STATE_VEC_LEN": 128}


def test_state_vec_layout_check_accepts_the_reviewed_layout(tmp_path, monkeypatch):
    repo = _fake_repo(tmp_path, monkeypatch, {"configs.state_vec": _state_vec_module()})
    rdt1b_server._check_state_vec_layout(repo)


def test_state_vec_layout_check_refuses_a_changed_layout(tmp_path, monkeypatch):
    repo = _fake_repo(tmp_path, monkeypatch, {"configs.state_vec": _state_vec_module(right_gripper_open=11)})
    with pytest.raises(SystemExit, match="state_vec"):
        rdt1b_server._check_state_vec_layout(repo)


def test_download_pinned_fetches_only_the_listed_files(monkeypatch, tmp_path):
    import huggingface_hub

    calls = []
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **kw: calls.append(kw) or kw["local_dir"])
    path = rdt1b_server._download_pinned(RDT_CKPT, RDT_REV, str(tmp_path), ("config.json", "pytorch_model.bin"))
    assert calls == [dict(repo_id=RDT_CKPT, revision=RDT_REV,
                          local_dir=str(tmp_path / "robotics-diffusion-transformer--rdt-1b" / RDT_REV),
                          allow_patterns=["config.json", "pytorch_model.bin"])]
    assert path == calls[0]["local_dir"]


def _main_harness(monkeypatch, argv, gpu=None):
    events = []

    class FakeModel:
        def __init__(self, checkpoint, **kwargs):
            events.append(("init", checkpoint, kwargs))
            self.required_image_keys = ("overhead_left", "wrist")
            self.metadata = {"precomputed_instructions": ["pre"]}
            self.model_version = "m"

        def __call__(self, images, instruction, joint_state, reset=False):
            events.append(("warm", sorted(images), instruction, joint_state))
            return JOINT_RAW

        def reset(self):
            events.append(("reset",))

    class FakeApp:
        def run(self, **kwargs):
            events.append(("run", kwargs["port"]))

    monkeypatch.setattr(rdt1b_server, "_check_dependency_versions", lambda: events.append(("deps",)))
    monkeypatch.setattr(rdt1b_server, "_check_state_vec_layout", lambda repo: events.append(("layout", repo)))
    monkeypatch.setattr(rdt1b_server, "_gpu_info",
                        lambda: gpu or dict(cuda=True, name="A100", total_gib=39.6, bf16=True))
    monkeypatch.setattr(rdt1b_server, "_peak_vram_gib", lambda: 14.2)
    monkeypatch.setattr(rdt1b_server, "RDT1BModel", FakeModel)
    monkeypatch.setattr(rdt1b_server, "create_app", lambda model: FakeApp())
    monkeypatch.setattr("sys.argv", ["rdt1b_server.py", *argv])
    return events


def test_server_main_checks_deps_layout_gpu_then_warms_and_clears_history(monkeypatch):
    events = _main_harness(monkeypatch, [])
    rdt1b_server.main()
    assert [e[0] for e in events] == ["deps", "layout", "init", "warm", "reset", "run"]
    init = events[2]
    assert init[1] == RDT_CKPT and init[2]["offload_dir"] is None  # 39.6 GiB A100: no offload
    assert init[2]["text_encoder"] is True and init[2]["client_control_hz"] == 2.0
    assert events[3][1:] == (["overhead_left", "wrist"], "pick up the cube", {j: 0.0 for j in JOINT_ORDER})
    assert events[-1] == ("run", 3005)


def test_server_main_offloads_t5_below_24gb_and_honours_explicit_dir(monkeypatch):
    events = _main_harness(monkeypatch, ["--offload-dir", "/content/off"],
                           gpu=dict(cuda=True, name="L4", total_gib=22.0, bf16=True))
    rdt1b_server.main()
    assert events[2][2]["offload_dir"] == "/content/off"


def test_server_main_precomputed_only_warms_with_a_precomputed_instruction(monkeypatch):
    events = _main_harness(monkeypatch, ["--no-text-encoder", "--lang-embeddings", "/e/a.pt"])
    rdt1b_server.main()
    assert events[2][2]["text_encoder"] is False and events[2][2]["lang_embeddings"] == ["/e/a.pt"]
    assert events[3][2] == "pre"


@pytest.mark.parametrize("gpu", [dict(cuda=True, name="Tesla T4", total_gib=14.7, bf16=False),
                                 dict(cuda=False, name=None, total_gib=0.0, bf16=False)])
def test_server_main_refuses_a_gpu_without_bf16(monkeypatch, gpu):
    events = _main_harness(monkeypatch, [], gpu=gpu)
    with pytest.raises(SystemExit, match="bf16"):
        rdt1b_server.main()
    assert "init" not in [e[0] for e in events]


def test_server_main_refuses_an_unreviewed_checkpoint(monkeypatch):
    _main_harness(monkeypatch, ["--checkpoint", "someone/rdt"])
    with pytest.raises(SystemExit):
        rdt1b_server.main()


def test_dependency_check_refuses_other_versions_and_accepts_local_build_tags(monkeypatch):
    good = {name: next(iter(v)) for name, v in rdt1b_server.PINNED_DEPENDENCIES.items()}
    monkeypatch.setattr(rdt1b_server, "_installed_versions", lambda: dict(good, diffusers="0.30.0"))
    with pytest.raises(SystemExit, match="diffusers"):
        rdt1b_server._check_dependency_versions()
    monkeypatch.setattr(rdt1b_server, "_installed_versions", lambda: dict(good))
    rdt1b_server._check_dependency_versions()

    import importlib.metadata as metadata

    monkeypatch.setattr(metadata, "version", lambda name: "2.1.0+cu121" if name == "torch" else good[name])
    assert rdt1b_server._installed_versions()["torch"] == "2.1.0"


def test_server_joint_order_matches_the_action_contract():
    assert rdt1b_server.SO101_JOINTS == JOINT_ORDER


# --- client backend --------------------------------------------------------------


def test_rdt1b_is_registered_on_the_http_lane():
    assert backends.BACKEND_REGISTRY["rdt1b"] is rdt1b.build
    assert "rdt1b" not in backends.GRPC_BRIDGE_BACKENDS


def test_backend_config_pins_checkpoint_and_records_the_confirmed_action_space():
    config = backends.load_backend_config("rdt1b")
    assert config["checkpoint"] == RDT_CKPT and config["revision"] == RDT_REV
    assert config["endpoint_url"] is None
    assert config["action_space"] == "joint_position"
    assert config["checkpoints"][RDT_CKPT]["action_space"] == config["action_space"]


def test_client_and_server_checkpoint_profiles_agree():
    client = backends.load_backend_config("rdt1b")["checkpoints"]
    server = rdt1b_server.CHECKPOINT_PROFILES
    assert set(client) == set(server)
    for checkpoint, profile in server.items():
        assert client[checkpoint] == profile
        assert profile["state_indices"] == list(rdt1b_server.SO101_JOINT_SLOTS)


def test_joint_position_slots_become_degrees_and_gripper_percent():
    action = rdt1b.joint_position_action(JOINT_RAW, "rad", [0.0, 100.0])
    assert tuple(action) == JOINT_ORDER
    assert action == pytest.approx(JOINT_EXPECTED)


def test_server_and_client_unit_conversions_are_inverse():
    profile = rdt1b_server.CHECKPOINT_PROFILES[RDT_CKPT]
    model_units = rdt1b_server.joint_state_to_model(JOINT_STATE, profile)
    back = rdt1b.joint_position_action(model_units, profile["joint_units"], profile["gripper_norm_to_percent"])
    assert back == pytest.approx(JOINT_STATE)


def test_joint_position_non_finite_values_pass_through_to_the_validator():
    action = rdt1b.joint_position_action([float("nan")] * 6, "rad", [0.0, 100.0])
    assert all(math.isnan(v) for v in action.values())


@pytest.mark.parametrize("raw", [None, [0.1] * 5, [0.1] * 7, "abc", [0.1] * 5 + ["x"]])
def test_malformed_joint_action_raises(raw):
    with pytest.raises((ValueError, TypeError)):
        rdt1b.joint_position_action(raw, "rad", [0.0, 100.0])


def _ortho6d(rotation):
    matrix = Rotation.from_euler("xyz", rotation).as_matrix()
    return list(matrix[:, 0]) + list(matrix[:, 1])


def test_rotation_from_ortho6d_matches_upstream_round_trip():
    euler = [0.3, -0.7, 1.9]
    matrix = rdt1b.rotation_from_ortho6d(_ortho6d(euler))
    assert Rotation.from_matrix(matrix).as_euler("xyz") == pytest.approx(euler)


def test_eef_pose_becomes_a_delta_at_the_current_fk_pose():
    current = eef_to_joint.forward_kinematics(JOINT_STATE)
    goal_rpy = [current[3] + 0.05, current[4] - 0.04, current[5] + 0.03]
    raw = [current[0] + 0.01, current[1] - 0.02, current[2] + 0.005, *_ortho6d(goal_rpy), 0.6]
    delta = rdt1b.rdt_eef_pose_to_eef_delta(raw, JOINT_STATE, [0.0, 100.0])
    assert [delta["dx"], delta["dy"], delta["dz"]] == pytest.approx([0.01, -0.02, 0.005], abs=1e-9)
    composed = Rotation.from_euler("xyz", [delta["droll"], delta["dpitch"], delta["dyaw"]]) * \
        Rotation.from_euler("xyz", current[3:])
    assert composed.as_matrix() == pytest.approx(Rotation.from_euler("xyz", goal_rpy).as_matrix(), abs=1e-9)
    assert delta["gripper"] == pytest.approx(60.0)


def test_non_finite_eef_pose_holds():
    delta = rdt1b.rdt_eef_pose_to_eef_delta([float("nan")] * 9 + [0.5], JOINT_STATE, [0.0, 100.0])
    assert all(math.isnan(delta[k]) for k in ("dx", "dy", "dz", "droll", "dpitch", "dyaw"))
    assert delta["gripper"] == pytest.approx(50.0)


@pytest.mark.parametrize("raw", [None, [0.1] * 9, [0.1] * 11, "abc"])
def test_malformed_eef_pose_raises(raw):
    with pytest.raises((ValueError, TypeError)):
        rdt1b.rdt_eef_pose_to_eef_delta(raw, JOINT_STATE, [0.0, 100.0])


def test_joint_position_convert_fn_is_a_direct_slot_mapping_without_ik(monkeypatch):
    monkeypatch.setattr(eef_to_joint, "solve_ik", lambda *a, **k: pytest.fail("joint space must not run IK"))
    profile = backends.load_backend_config("rdt1b")["checkpoints"][RDT_CKPT]
    convert = rdt1b.make_convert_fn(profile, {"elbow_flex": (-80.0, 80.0)})
    assert convert(JOINT_RAW, JOINT_STATE) == pytest.approx(JOINT_EXPECTED)


def test_eef_pose_convert_fn_runs_solve_ik_with_joint_state_and_limits(monkeypatch):
    calls = []
    monkeypatch.setattr(eef_to_joint, "solve_ik", lambda js, delta, **k: calls.append((js, delta, k)) or "ik")
    limits = {"elbow_flex": (-80.0, 80.0)}
    convert = rdt1b.make_convert_fn({"action_space": "eef_pose", "gripper_norm_to_percent": [0.0, 100.0]}, limits)
    current = eef_to_joint.forward_kinematics(JOINT_STATE)
    raw = [*current[:3], *_ortho6d(current[3:]), 0.2]
    assert convert(raw, JOINT_STATE) == "ik"
    assert calls[0][0] == JOINT_STATE and calls[0][2] == {"joint_limits_deg": limits}
    assert calls[0][1]["dx"] == pytest.approx(0.0, abs=1e-12) and calls[0][1]["gripper"] == pytest.approx(20.0)


@pytest.mark.parametrize("profile", [
    {"action_space": "joint_velocity", "gripper_norm_to_percent": [0, 100]},
    {"action_space": "joint_position", "gripper_norm_to_percent": [0, 100]},  # no joint_units
    {"action_space": "joint_position", "joint_units": "rad"},  # no gripper range
    {"action_space": "eef_pose"},
    {},
])
def test_unknown_or_incomplete_profile_is_rejected(profile):
    with pytest.raises(ValueError, match="action_space|gripper_norm_to_percent|joint_units"):
        rdt1b.make_convert_fn(profile, None)


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
    (no_config / "rdt1b.json").write_text(json.dumps({"checkpoint": "c", "endpoint_url": None}))
    with pytest.raises(SystemExit, match="--server-address"):
        rdt1b.build(_args(), {0: "wrist"}, {})


def test_build_without_config_file_raises_clear_systemexit(no_config):
    with pytest.raises(SystemExit, match="endpoint_url"):
        rdt1b.build(_args(), {0: "wrist"}, {})


def test_build_rejects_a_grpc_style_host_port_endpoint(no_config):
    with pytest.raises(SystemExit, match="http"):
        rdt1b.build(_args(server_address="0.tcp.ngrok.io:12345"), {0: "wrist"}, {})


def test_build_rejects_a_checkpoint_without_a_profile():
    with pytest.raises(SystemExit, match="action_space"):
        rdt1b.build(_args(server_address="https://x.ngrok-free.app", checkpoint="someone/other"), {0: "wrist"}, {})


def _capture_build(monkeypatch):
    seen = {}

    def fake_build(endpoint_url, checkpoint, args, camera_names, convert_fn=None):
        seen.update(endpoint_url=endpoint_url, checkpoint=checkpoint, camera_names=camera_names,
                    convert_fn=convert_fn)
        return "handles"

    monkeypatch.setattr(base_http_adapter, "build_http_adapter_backend", fake_build)
    return seen


def test_build_defaults_to_rdt_1b_with_the_joint_position_mapping(monkeypatch):
    seen = _capture_build(monkeypatch)
    monkeypatch.setattr(eef_to_joint, "solve_ik", lambda *a, **k: pytest.fail("joint space must not run IK"))
    assert rdt1b.build(_args(server_address="https://abc.ngrok-free.app"), {0: "wrist"}, {}) == "handles"
    assert seen["endpoint_url"] == "https://abc.ngrok-free.app"
    assert seen["checkpoint"] == RDT_CKPT
    assert seen["camera_names"] == {0: "wrist"}
    assert seen["convert_fn"](JOINT_RAW, JOINT_STATE) == pytest.approx(JOINT_EXPECTED)


def test_build_action_space_from_config_selects_the_ik_branch(monkeypatch, no_config):
    """The config's declared action_space drives the conversion: an eef_pose
    profile goes through eef_to_joint.solve_ik()."""
    (no_config / "rdt1b.json").write_text(json.dumps({
        "checkpoint": "someone/rdt-eef", "endpoint_url": "https://cfg.example", "action_space": "eef_pose",
        "checkpoints": {"someone/rdt-eef": {"action_space": "eef_pose", "gripper_norm_to_percent": [0, 100]}}}))
    seen = _capture_build(monkeypatch)
    calls = []
    monkeypatch.setattr(eef_to_joint, "solve_ik", lambda js, delta, **k: calls.append(delta) or "ik")
    limits = {"wrist_flex": (-90.0, 90.0)}
    rdt1b.build(_args(), {0: "wrist"}, limits)
    assert seen["endpoint_url"] == "https://cfg.example" and seen["checkpoint"] == "someone/rdt-eef"
    current = eef_to_joint.forward_kinematics(JOINT_STATE)
    assert seen["convert_fn"]([*current[:3], *_ortho6d(current[3:]), 1.0], JOINT_STATE) == "ik"
    assert calls[0]["gripper"] == pytest.approx(100.0)


def test_build_with_the_170m_alternate(monkeypatch):
    seen = _capture_build(monkeypatch)
    rdt1b.build(_args(server_address="http://h:3005", checkpoint=SMALL_CKPT), {0: "wrist"}, {})
    assert seen["checkpoint"] == SMALL_CKPT
    assert seen["convert_fn"](JOINT_RAW, JOINT_STATE) == pytest.approx(JOINT_EXPECTED)


# --- Task 2: mocked full round trip (regression) ------------------------------


def _route_requests_to(app, monkeypatch):
    """Send the client's `requests.post` straight into the Flask test client."""
    http = app.test_client()
    posted = []

    class Routed:
        def __init__(self, response):
            self._response = response
            self.status_code = response.status_code

        def raise_for_status(self):
            if self.status_code >= 400:
                raise base_http_adapter.requests.HTTPError(f"{self.status_code}")

        def json(self):
            return self._response.get_json()

    def fake_post(url, json=None, timeout=None):
        posted.append((url, json))
        return Routed(http.post("/action", json=json))

    monkeypatch.setattr(base_http_adapter.requests, "post", fake_post)
    return posted


def _source_via_build(monkeypatch, checkpoint=None, limits=None):
    """Run the real `rdt1b.build()` (robot/camera construction stubbed) and
    return a real `HttpAdapterActionSource` with the convert_fn it wired."""
    from test_backend_registry import FakeStereo, FakeWristCap

    seen = _capture_build(monkeypatch)
    rdt1b.build(_args(server_address="https://abc.ngrok-free.app", checkpoint=checkpoint), {0: "wrist"},
                limits or {})
    return base_http_adapter.HttpAdapterActionSource(
        seen["endpoint_url"], seen["checkpoint"], FakeWristCap(), FakeStereo(), convert_fn=seen["convert_fn"])


LIMITS = {j: (-180.0, 180.0) for j in JOINT_ORDER[:5]} | {"gripper": (0.0, 100.0)}


def test_round_trip_default_joint_position_into_a_validated_action(fake_policy, monkeypatch):
    """Real create_app() -> real HttpAdapterActionSource with build()'s
    joint_position convert_fn -> real safety_validator.validate_action()."""
    from vla_bridge import safety_validator

    monkeypatch.setattr(eef_to_joint, "solve_ik", lambda *a, **k: pytest.fail("joint space must not run IK"))
    model = RDT1BModel(local_dir="/ckpt")
    posted = _route_requests_to(create_app(model), monkeypatch)
    source = _source_via_build(monkeypatch, limits=LIMITS)
    action, model_version = source.get_action(JOINT_STATE, "pick up the cube")

    assert [url for url, _ in posted] == ["https://abc.ngrok-free.app/action"]
    assert posted[0][1]["joint_state"] == JOINT_STATE
    assert set(posted[0][1]["images"]) == {"wrist", "overhead_left", "overhead_right"}
    # The model was conditioned on the rig's state in RDT units (rad, gripper 0-1).
    assert fake_policy.steps[0]["proprio"] == pytest.approx(
        [math.radians(JOINT_STATE[j]) for j in JOINT_ORDER[:5]] + [0.37])
    assert model_version == f"{RDT_CKPT}@{RDT_REV[:12]}"
    assert source.last_details["bridge_flags"] == []
    assert source.last_details["raw_model_output"] == pytest.approx(JOINT_RAW, abs=1e-6)
    assert tuple(action) == JOINT_ORDER
    assert all(isinstance(v, float) and math.isfinite(v) for v in action.values())
    assert action == pytest.approx(JOINT_EXPECTED, abs=1e-4)

    validated, _flags = safety_validator.validate_action(action, JOINT_STATE, LIMITS)
    assert set(validated) == set(JOINT_ORDER)
    assert all(math.isfinite(v) for v in validated.values())
    assert any(validated[j] != JOINT_STATE[j] for j in JOINT_ORDER)  # a real, non-held action


def test_round_trip_170m_alternate(fake_policy, monkeypatch):
    from vla_bridge import safety_validator

    model = RDT1BModel(SMALL_CKPT, local_dir="/ckpt")
    _route_requests_to(create_app(model), monkeypatch)
    source = _source_via_build(monkeypatch, checkpoint=SMALL_CKPT, limits=LIMITS)
    action, model_version = source.get_action(JOINT_STATE, "pick up the cube")
    assert model_version == f"{SMALL_CKPT}@{SMALL_REV[:12]}"
    assert action == pytest.approx(JOINT_EXPECTED, abs=1e-4)
    validated, _flags = safety_validator.validate_action(action, JOINT_STATE, LIMITS)
    assert all(math.isfinite(v) for v in validated.values())


def test_round_trip_consecutive_steps_fill_the_two_frame_history(fake_policy, monkeypatch):
    model = RDT1BModel(local_dir="/ckpt")
    _route_requests_to(create_app(model), monkeypatch)
    source = _source_via_build(monkeypatch, limits=LIMITS)
    for _ in range(3):
        source.get_action(JOINT_STATE, "pick up the cube")
    previous_present = [step["images"][0] is not None for step in fake_policy.steps]
    assert previous_present == [False, True, True]
    assert all(step["images"][2] is None and step["images"][5] is None for step in fake_policy.steps)


def test_round_trip_client_rate_picks_the_returned_chunk_step(fake_policy, monkeypatch):
    chunk = np.zeros((1, 64, 6), dtype=np.float32)
    chunk[0, :, 0] = np.radians(np.arange(64))  # shoulder_pan target = chunk index in degrees
    fake_policy.chunk = chunk
    model = RDT1BModel(local_dir="/ckpt", client_control_hz=30.0)
    _route_requests_to(create_app(model), monkeypatch)
    source = _source_via_build(monkeypatch, limits=LIMITS)
    action, _ = source.get_action(JOINT_STATE, "pick up the cube")
    assert action["shoulder_pan"] == pytest.approx(0.0, abs=1e-4)  # 30 Hz loop -> step 0
    assert model.metadata["chunk_step"] == 0


def test_round_trip_eef_pose_fallback_runs_real_ik_into_a_validated_action(monkeypatch, no_config):
    """The fallback branch end to end: an eef_pose profile's absolute pose goes
    through real ikpy solve_ik() and the real validator."""
    from vla_bridge import safety_validator

    (no_config / "rdt1b.json").write_text(json.dumps({
        "checkpoint": "someone/rdt-eef", "endpoint_url": None, "action_space": "eef_pose",
        "checkpoints": {"someone/rdt-eef": {"action_space": "eef_pose", "gripper_norm_to_percent": [0, 100]}}}))
    current = eef_to_joint.forward_kinematics(JOINT_STATE)
    target = [current[0] + 0.01, current[1] - 0.008, current[2] + 0.012]
    raw = [*target, *_ortho6d(current[3:]), 0.45]

    class Response:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"action": raw, "model_version": "someone/rdt-eef@abc"}

    monkeypatch.setattr(base_http_adapter.requests, "post", lambda url, json=None, timeout=None: Response())
    source = _source_via_build(monkeypatch, limits=LIMITS)
    action, model_version = source.get_action(JOINT_STATE, "pick up the cube")

    assert model_version == "someone/rdt-eef@abc" and source.last_details["bridge_flags"] == []
    assert tuple(action) == JOINT_ORDER
    assert action["gripper"] == pytest.approx(45.0)
    after = eef_to_joint.forward_kinematics(action)
    assert list(after[:3]) == pytest.approx(target, abs=3e-3)  # IK reached the absolute pose
    validated, _flags = safety_validator.validate_action(action, JOINT_STATE, LIMITS)
    assert all(math.isfinite(v) for v in validated.values())
    assert any(validated[j] != JOINT_STATE[j] for j in JOINT_ORDER)


def test_round_trip_holds_position_when_the_model_fails(fake_policy, monkeypatch):
    fake_policy.chunk = np.full((1, 64, 6), np.nan)
    model = RDT1BModel(local_dir="/ckpt")
    _route_requests_to(create_app(model), monkeypatch)
    source = _source_via_build(monkeypatch)
    action, model_version = source.get_action(JOINT_STATE, "x")
    assert action == JOINT_STATE
    assert model_version == f"{RDT_CKPT}@http-error-holding-position"
