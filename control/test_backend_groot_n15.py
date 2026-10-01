"""Plan 14-12 (VLAB-11): GR00T N1.5 over the Tier C resident HTTP adapter.

Colab-side `groot_n15_server.py` is tested with NVIDIA Isaac-GR00T's
`Gr00tPolicy` load, the pinned HF download and `get_action()` mocked; the
checkpoint's `experiment_cfg/metadata.json` is written for real to a tmp dir
(copied from the reviewed checkpoint's own file) so the layout/units checks run
unmocked. Client-side `backends/groot_n15.py` is tested with
`build_http_adapter_backend` (robot and camera construction) mocked. Nothing
here touches real hardware, a GPU, Isaac-GR00T, or the network.
"""

import ast
import base64
import json
import math
import threading
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from vla_bridge import backends
from vla_bridge import groot_n15_server
from vla_bridge.action_contract import JOINT_ORDER
from vla_bridge.backends import base_http_adapter, groot_n15
from vla_bridge.groot_n15_server import GrootN15Model, create_app

CKPT = "quangnd58/GR00T-N1.5-3B-so101-multitask"
REV = "488f3b6da322e55a316a562a00567fb4a1fe130a"
ISAAC_GROOT_COMMIT = "4af2b622892f7dcb5aae5a3fb70bcb02dc217b96"
VLA_BRIDGE = Path(__file__).resolve().parent / "vla_bridge"

JOINT_STATE = {
    "shoulder_pan": 10.0,
    "shoulder_lift": -20.0,
    "elbow_flex": 30.0,
    "wrist_flex": 15.0,
    "wrist_roll": 5.0,
    "gripper": 37.0,
}
# One chunk step of the checkpoint's un-normalized action: absolute joint
# targets in lerobot degree mode (5 arm joints) + gripper percent.
JOINT_RAW = [12.0, -18.5, 33.0, 20.0, 4.0, 41.0]
JOINT_EXPECTED = dict(zip(JOINT_ORDER, JOINT_RAW))


def _stats(lo, hi):
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    return {"min": lo.tolist(), "max": hi.tolist(), "mean": ((lo + hi) / 2).tolist(),
            "std": ((hi - lo) / 4).tolist(), "q01": lo.tolist(), "q99": hi.tolist()}


# The reviewed checkpoint's experiment_cfg/metadata.json (rev 488f3b6), rounded.
DEGREE_ARM = _stats([-63.253, -94.901, -69.275, -21.714, -0.747], [49.626, 63.692, 89.055, 100.396, 3.121])
# nvidia/SO_ARM_Starter_Gr00t @ 5c01635: a range-normalized (+/-100 clamp) checkpoint.
RANGE_ARM = _stats([-33.399, -100.0, -48.159, -33.304, -88.424], [41.809, 49.575, 100.0, 90.168, -29.436])
GRIPPER = _stats([0.162], [60.94])


def _metadata(arm=DEGREE_ARM, video=("front", "wrist"), absolute=True, arm_shape=5, tag="new_embodiment"):
    def modality(shape):
        return {"absolute": absolute, "rotation_type": None, "shape": [shape], "continuous": True}

    return {tag: {
        "statistics": {"state": {"single_arm": arm, "gripper": GRIPPER},
                       "action": {"single_arm": arm, "gripper": GRIPPER}},
        "modalities": {
            "video": {key: {"resolution": [640, 480], "channels": 3, "fps": 30.0} for key in video},
            "state": {"single_arm": modality(arm_shape), "gripper": modality(1)},
            "action": {"single_arm": modality(arm_shape), "gripper": modality(1)},
        },
        "embodiment_tag": tag,
    }}


def _b64_png(value=50, shape=(24, 32, 3)):
    frame = np.full(shape, value, dtype=np.uint8) if np.isscalar(value) else np.broadcast_to(
        np.asarray(value, dtype=np.uint8), shape).copy()
    ok, buf = cv2.imencode(".png", frame)
    assert ok
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _payload(**overrides):
    body = {
        "images": {"wrist": _b64_png(10), "overhead_left": _b64_png(20), "overhead_right": _b64_png(30)},
        "joint_state": dict(JOINT_STATE),
        "instruction": "Pick blue cube and place on red cube",
    }
    body.update(overrides)
    return body


def _chunk(horizon=16):
    """Un-normalized `Gr00tPolicy.get_action()` output: step k = JOINT_RAW + k."""
    steps = np.arange(horizon, dtype=np.float64)[:, None]
    full = np.asarray(JOINT_RAW, dtype=np.float64)[None, :] + steps
    return {"action.single_arm": full[:, :5].copy(), "action.gripper": full[:, 5:].copy()}


# --- Colab-side server ---------------------------------------------------------


class FakeDataConfig:
    """`gr00t.experiment.data_config.So100DualCamDataConfig`'s keys."""

    video_keys = ["video.front", "video.wrist"]
    state_keys = ["state.single_arm", "state.gripper"]
    action_keys = ["action.single_arm", "action.gripper"]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))


class FakePolicy:
    """Stands in for Isaac-GR00T's `gr00t.model.policy.Gr00tPolicy`."""

    def __init__(self, state):
        self._state = state
        self.model = SimpleNamespace(name="GR00T_N1_5")

    def get_action(self, observation):
        self._state.calls.append(observation)
        return self._state.chunk if self._state.chunk is not None else _chunk()


@pytest.fixture
def fake_policy(monkeypatch, tmp_path):
    """Mock `_download_pinned` (snapshot_download at the pinned revision),
    `_load_policy` (Isaac-GR00T's `Gr00tPolicy(...)`) and `_verify_weights`.
    The checkpoint's metadata.json is a real file under tmp_path."""
    state = SimpleNamespace(downloads=[], loads=[], verified=[], calls=[], chunk=None,
                            metadata=_metadata(), model_dir=tmp_path / "ckpt")

    def download(checkpoint, revision, local_dir, subfolder):
        state.downloads.append((checkpoint, revision, local_dir, subfolder))
        (state.model_dir / "experiment_cfg").mkdir(parents=True, exist_ok=True)
        (state.model_dir / "experiment_cfg" / "metadata.json").write_text(json.dumps(state.metadata))
        return str(state.model_dir)

    def load_policy(model_dir, profile, denoising_steps, groot_repo):
        state.loads.append(dict(model_dir=model_dir, data_config=profile["data_config"],
                                embodiment_tag=profile["embodiment_tag"], denoising_steps=denoising_steps,
                                groot_repo=groot_repo))
        return FakePolicy(state), FakeDataConfig()

    monkeypatch.setattr(groot_n15_server, "_download_pinned", download)
    monkeypatch.setattr(groot_n15_server, "_load_policy", load_policy)
    monkeypatch.setattr(groot_n15_server, "_verify_weights",
                        lambda model, model_dir: state.verified.append((model, model_dir)))
    return state


def test_model_loads_once_across_many_action_requests(fake_policy):
    model = GrootN15Model(local_dir="/ckpt")
    http = create_app(model).test_client()
    for _ in range(3):
        assert http.post("/action", json=_payload()).status_code == 200
    assert len(fake_policy.loads) == 1
    assert len(fake_policy.downloads) == 1
    assert len(fake_policy.calls) == 3


def test_pinned_revision_is_downloaded_and_loaded_through_isaac_groot(fake_policy):
    model = GrootN15Model(local_dir="/ckpt", groot_repo="/content/Isaac-GR00T")
    assert fake_policy.downloads == [(CKPT, REV, "/ckpt", "")]
    assert fake_policy.loads == [dict(model_dir=str(fake_policy.model_dir), data_config="so100_dualcam",
                                      embodiment_tag="new_embodiment", denoising_steps=4,
                                      groot_repo="/content/Isaac-GR00T")]
    assert len(fake_policy.verified) == 1  # the strict weight check ran on the loaded model
    assert model.model_version == f"{CKPT}@{REV[:12]}"


def test_unreviewed_checkpoint_is_refused(fake_policy):
    with pytest.raises(ValueError, match="reviewed"):
        GrootN15Model("nvidia/GR00T-N1.5-3B")
    assert fake_policy.downloads == []


def test_server_has_no_lerobot_or_n17_code_path():
    """Pitfall 3: N1.5 runs through NVIDIA Isaac-GR00T, never lerobot.policies
    (whose `groot` policy is N1.7-only and rejects N1.5)."""
    source = (VLA_BRIDGE / "groot_n15_server.py").read_text()
    imported = _imported_modules(VLA_BRIDGE / "groot_n15_server.py")
    assert not any(name.startswith("lerobot") for name in imported)
    assert "gr00t.model.policy" in imported and "gr00t.experiment.data_config" in imported
    assert "Gr00tPolicy(" in source


def test_action_endpoint_returns_absolute_joint_targets_and_model_version(fake_policy):
    model = GrootN15Model(local_dir="/ckpt")
    response = create_app(model).test_client().post("/action", json=_payload())
    assert response.status_code == 200
    body = response.get_json()
    # Default client loop is 2 Hz against 30 Hz training data: step round(30/2) - 1 = 14.
    assert body["action"] == pytest.approx([v + 14 for v in JOINT_RAW])
    assert body["model_version"] == f"{CKPT}@{REV[:12]}"


def test_observation_matches_isaac_groots_so100_dualcam_layout(fake_policy):
    model = GrootN15Model(local_dir="/ckpt")
    # BGR-encoded like the client: (b, g, r) = (200, 100, 50) must arrive as RGB (50, 100, 200).
    images = {"wrist": _b64_png(10, shape=(480, 640, 3)), "overhead_left": _b64_png((200, 100, 50), (480, 640, 3)),
              "overhead_right": _b64_png(30, shape=(480, 640, 3))}
    create_app(model).test_client().post("/action", json=_payload(images=images))
    obs = fake_policy.calls[0]
    assert set(obs) == {"video.front", "video.wrist", "state.single_arm", "state.gripper",
                        "annotation.human.task_description"}
    assert obs["video.front"].shape == (1, 480, 640, 3) and obs["video.front"].dtype == np.uint8
    assert obs["video.front"][0, 0, 0].tolist() == [50, 100, 200]  # overhead_left, RGB
    assert obs["video.wrist"][0, 0, 0].tolist() == [10, 10, 10]
    # The checkpoint's state is lerobot degree mode, the same units as joint_state: no conversion.
    assert obs["state.single_arm"].shape == (1, 5)
    assert obs["state.single_arm"][0].tolist() == [JOINT_STATE[j] for j in JOINT_ORDER[:5]]
    assert obs["state.gripper"].tolist() == [[37.0]]
    assert obs["annotation.human.task_description"] == ["Pick blue cube and place on red cube"]


def test_frames_are_resized_to_the_checkpoints_training_resolution(fake_policy):
    model = GrootN15Model(local_dir="/ckpt")
    create_app(model).test_client().post("/action", json=_payload())  # 24x32 frames
    obs = fake_policy.calls[0]
    assert obs["video.front"].shape == (1, 480, 640, 3)
    assert obs["video.wrist"].shape == (1, 480, 640, 3)


def test_front_image_key_selects_the_rig_camera(fake_policy):
    model = GrootN15Model(local_dir="/ckpt", front_image_key="overhead_right")
    create_app(model).test_client().post("/action", json=_payload())
    assert fake_policy.calls[0]["video.front"][0, 0, 0].tolist() == [30, 30, 30]
    assert model.metadata["video_sources"] == {"front": "overhead_right", "wrist": "wrist"}


@pytest.mark.parametrize("control_hz,client_hz,horizon,expected", [
    (30, 2.0, 16, 14), (30, 30.0, 16, 0), (30, 1.0, 16, 15), (30, 0.5, 16, 15), (30, 10.0, 16, 2)])
def test_chunk_step_is_about_one_client_period_ahead(control_hz, client_hz, horizon, expected):
    assert groot_n15_server.chunk_step_for(control_hz, client_hz, horizon) == expected


def test_client_control_hz_selects_the_returned_chunk_step(fake_policy):
    model = GrootN15Model(local_dir="/ckpt", client_control_hz=10.0)
    body = create_app(model).test_client().post("/action", json=_payload()).get_json()
    assert body["action"] == pytest.approx([v + 2 for v in JOINT_RAW])


@pytest.mark.parametrize("body", [
    {"joint_state": JOINT_STATE, "instruction": "x"},  # no images
    _payload(images="not-a-dict"),
    _payload(images={"wrist": "@@@", "overhead_left": _b64_png()}),
    _payload(images={"wrist": _b64_png()}),  # front camera missing
    _payload(instruction=""),
    _payload(instruction=5),
    {k: v for k, v in _payload().items() if k != "instruction"},
    {k: v for k, v in _payload().items() if k != "joint_state"},
    _payload(joint_state="x"),
    _payload(joint_state={k: v for k, v in JOINT_STATE.items() if k != "gripper"}),
    _payload(joint_state=dict(JOINT_STATE, elbow_flex="12")),
    _payload(joint_state=dict(JOINT_STATE, elbow_flex=True)),
    _payload(reset="yes"),
])
def test_malformed_body_returns_400(fake_policy, body):
    model = GrootN15Model(local_dir="/ckpt")
    response = create_app(model).test_client().post("/action", json=body)
    assert response.status_code == 400
    assert fake_policy.calls == []


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_non_finite_joint_state_is_a_bad_request_not_a_model_call(fake_policy, bad):
    model = GrootN15Model(local_dir="/ckpt")
    response = create_app(model).test_client().post(
        "/action", data=json.dumps(_payload(joint_state=dict(JOINT_STATE, wrist_flex=bad))),
        content_type="application/json")
    assert response.status_code == 400
    assert fake_policy.calls == []


def test_non_json_body_returns_400(fake_policy):
    model = GrootN15Model(local_dir="/ckpt")
    response = create_app(model).test_client().post("/action", data="nope", content_type="text/plain")
    assert response.status_code == 400


@pytest.mark.parametrize("chunk", [
    {"action.single_arm": np.full((16, 5), np.nan), "action.gripper": np.zeros((16, 1))},
    {"action.single_arm": np.zeros((16, 4)), "action.gripper": np.zeros((16, 1))},
    {"action.single_arm": np.zeros((16, 5)), "action.gripper": np.zeros((16, 2))},
    {"action.single_arm": np.zeros((3, 5)), "action.gripper": np.zeros((3, 1))},
    {"action.single_arm": np.zeros((16, 5))},
])
def test_bad_model_output_returns_500(fake_policy, chunk):
    fake_policy.chunk = chunk
    model = GrootN15Model(local_dir="/ckpt")
    response = create_app(model).test_client().post("/action", json=_payload())
    assert response.status_code == 500
    assert "groot" in response.get_json()["error"].lower()


def test_one_dimensional_gripper_chunk_is_accepted(fake_policy):
    chunk = _chunk()
    chunk["action.gripper"] = chunk["action.gripper"][:, 0]
    fake_policy.chunk = chunk
    model = GrootN15Model(local_dir="/ckpt")
    body = create_app(model).test_client().post("/action", json=_payload()).get_json()
    assert body["action"][5] == pytest.approx(JOINT_RAW[5] + 14)


def test_health_reports_ready_with_metadata(fake_policy):
    model = GrootN15Model(local_dir="/ckpt")
    body = create_app(model).test_client().get("/health").get_json()
    assert body["status"] == "ready"
    meta = body["model"]
    assert meta["checkpoint"] == CKPT and meta["revision"] == REV
    assert meta["isaac_groot_commit"] == ISAAC_GROOT_COMMIT
    assert meta["data_config"] == "so100_dualcam" and meta["embodiment_tag"] == "new_embodiment"
    assert meta["action_space"] == "absolute_joint" and meta["action_units"] == "degrees"
    assert meta["detected_action_units"] == "degrees"
    assert meta["action_horizon"] == 16 and meta["chunk_step"] == 14 and meta["control_hz"] == 30
    assert meta["denoising_steps"] == 4
    assert meta["video_sources"] == {"front": "overhead_left", "wrist": "wrist"}
    assert meta["video_resolution"] == {"front": [640, 480], "wrist": [640, 480]}


def test_overlapping_request_gets_429_not_queued():
    started, release = threading.Event(), threading.Event()

    class SlowModel:
        metadata = {}
        model_version = "slow"
        required_image_keys = ("overhead_left", "wrist")

        def __call__(self, images, instruction, joint_state):
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


# --- checkpoint layout / units checks (real metadata.json) -----------------------


@pytest.mark.parametrize("metadata,match", [
    (_metadata(arm=RANGE_ARM), "range_m100_100"),
    (_metadata(absolute=False), "absolute"),
    (_metadata(video=("front", "laptop")), "video"),
    (_metadata(arm_shape=6), "shape"),
    (_metadata(tag="so100"), "new_embodiment"),
])
def test_checkpoint_whose_layout_or_units_differ_from_its_profile_is_refused(fake_policy, metadata, match):
    fake_policy.metadata = metadata
    with pytest.raises(RuntimeError, match=match):
        GrootN15Model(local_dir="/ckpt")
    assert fake_policy.loads == []  # refused before the 3B model is loaded


@pytest.mark.parametrize("arm,expected", [
    (DEGREE_ARM, "degrees"),
    (RANGE_ARM, "range_m100_100"),
    (_stats([-1.2, -1.7, -0.3, -1.1, -2.9], [1.0, 0.4, 1.6, 1.3, 2.8]), "radians"),
    # Pre-0.2 lerobot SO-100 calibration (shenjianliang/so101_pick_GR00T-N1.5-3B): offset zero, up to ~199.
    (_stats([-57.0, 35.8, 33.2, -9.3, -137.5], [47.6, 199.4, 183.6, 103.0, 22.8]), "legacy_offset"),
    (_stats([-50.0, -60.0, -40.0, -20.0, -5.0], [50.0, 60.0, 40.0, 20.0, 5.0]), "ambiguous"),
])
def test_action_units_are_classified_from_the_checkpoints_own_statistics(arm, expected):
    assert groot_n15_server.classify_action_units({"single_arm": arm, "gripper": GRIPPER}) == expected


# --- upstream seams ---------------------------------------------------------------


def test_download_pinned_fetches_only_the_inference_files(monkeypatch, tmp_path):
    import huggingface_hub

    calls = []
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **kw: calls.append(kw) or kw["local_dir"])
    path = groot_n15_server._download_pinned(CKPT, REV, str(tmp_path), "checkpoint-10000/")
    assert calls[0]["repo_id"] == CKPT and calls[0]["revision"] == REV
    assert calls[0]["local_dir"] == str(tmp_path / CKPT.replace("/", "--") / REV)
    assert sorted(calls[0]["allow_patterns"]) == sorted([
        "checkpoint-10000/config.json", "checkpoint-10000/experiment_cfg/*",
        "checkpoint-10000/model*.safetensors", "checkpoint-10000/model.safetensors.index.json"])
    assert path == str(tmp_path / CKPT.replace("/", "--") / REV / "checkpoint-10000")


def test_load_policy_builds_gr00tpolicy_from_the_profiles_data_config(monkeypatch, tmp_path):
    """The real `_load_policy` against fake `gr00t.*` modules shaped like the
    pinned Isaac-GR00T commit's: a LOCAL path (no hub re-resolution of `main`)."""
    import sys
    import types

    (tmp_path / "gr00t" / "model").mkdir(parents=True)
    (tmp_path / "gr00t" / "model" / "policy.py").write_text("# fake\n")
    built = {}

    class FakeGr00tPolicy:
        def __init__(self, **kwargs):
            built.update(kwargs)

    class FakeConfig(FakeDataConfig):
        def modality_config(self):
            return "modality-config"

        def transform(self):
            return "transform"

    gr00t = types.ModuleType("gr00t")
    gr00t.__file__ = str(tmp_path / "gr00t" / "__init__.py")
    policy_mod = types.ModuleType("gr00t.model.policy")
    policy_mod.Gr00tPolicy = FakeGr00tPolicy
    data_config_mod = types.ModuleType("gr00t.experiment.data_config")
    data_config_mod.DATA_CONFIG_MAP = {"so100_dualcam": FakeConfig()}
    for name, module in {"gr00t": gr00t, "gr00t.model": types.ModuleType("gr00t.model"),
                         "gr00t.model.policy": policy_mod,
                         "gr00t.experiment": types.ModuleType("gr00t.experiment"),
                         "gr00t.experiment.data_config": data_config_mod}.items():
        monkeypatch.setitem(sys.modules, name, module)

    profile = groot_n15_server.CHECKPOINT_PROFILES[CKPT]
    policy, data_config = groot_n15_server._load_policy("/ckpt/local", profile, 4, str(tmp_path))
    assert isinstance(policy, FakeGr00tPolicy)
    assert data_config is data_config_mod.DATA_CONFIG_MAP["so100_dualcam"]
    assert built == dict(model_path="/ckpt/local", embodiment_tag="new_embodiment",
                         modality_config="modality-config", modality_transform="transform",
                         denoising_steps=4, device="cuda")


def test_load_policy_refuses_a_gr00t_package_outside_the_pinned_checkout(monkeypatch, tmp_path):
    import sys
    import types

    (tmp_path / "gr00t" / "model").mkdir(parents=True)
    (tmp_path / "gr00t" / "model" / "policy.py").write_text("# fake\n")
    gr00t = types.ModuleType("gr00t")
    gr00t.__file__ = "/usr/lib/python3/site-packages/gr00t/__init__.py"
    monkeypatch.setitem(sys.modules, "gr00t", gr00t)
    with pytest.raises(RuntimeError, match="not the pinned"):
        groot_n15_server._load_policy("/ckpt", groot_n15_server.CHECKPOINT_PROFILES[CKPT], 4, str(tmp_path))


def _weights_dir(tmp_path, keys):
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": {k: "model-00001-of-00001.safetensors" for k in keys}}))
    return str(tmp_path)


def test_verify_weights_accepts_a_matching_checkpoint_and_tied_weights(tmp_path):
    import torch

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(4, 3)
            self.head = torch.nn.Linear(3, 4, bias=False)
            self.head.weight = self.embed.weight  # tied: saved once, like Qwen's lm_head

    model = Tiny()
    groot_n15_server._verify_weights(model, _weights_dir(tmp_path, ["embed.weight"]))


@pytest.mark.parametrize("keys,match", [
    (["embed.weight", "head.weight", "extra.weight"], "unexpected"),
    (["embed.weight"], "missing"),
])
def test_verify_weights_refuses_missing_or_unexpected_keys(tmp_path, keys, match):
    import torch

    model = torch.nn.Sequential()
    model.embed = torch.nn.Embedding(4, 3)
    model.head = torch.nn.Linear(3, 4, bias=False)
    with pytest.raises(RuntimeError, match=match):
        groot_n15_server._verify_weights(model, _weights_dir(tmp_path, keys))


def test_groot_checkout_must_be_the_pinned_commit(monkeypatch, tmp_path):
    (tmp_path / "gr00t" / "model").mkdir(parents=True)
    (tmp_path / "gr00t" / "model" / "policy.py").write_text("# fake\n")
    monkeypatch.setattr(groot_n15_server, "_git_head", lambda repo: ISAAC_GROOT_COMMIT)
    groot_n15_server._check_groot_checkout(str(tmp_path))
    monkeypatch.setattr(groot_n15_server, "_git_head", lambda repo: "0" * 40)
    with pytest.raises(SystemExit, match=ISAAC_GROOT_COMMIT):
        groot_n15_server._check_groot_checkout(str(tmp_path))
    with pytest.raises(SystemExit, match="Isaac-GR00T checkout"):
        groot_n15_server._check_groot_checkout(str(tmp_path / "missing"))


def _main_harness(monkeypatch, argv, gpu=None):
    events = []

    class FakeModel:
        def __init__(self, checkpoint, **kwargs):
            events.append(("init", checkpoint, kwargs))
            self.required_image_keys = ("overhead_left", "wrist")
            self.warmup_shape = (480, 640, 3)
            self.metadata = {}
            self.model_version = "m"

        def __call__(self, images, instruction, joint_state):
            events.append(("warm", sorted(images), {k: v.shape for k, v in images.items()}, instruction,
                           joint_state))
            return JOINT_RAW

    class FakeApp:
        def run(self, **kwargs):
            events.append(("run", kwargs["port"]))

    monkeypatch.setattr(groot_n15_server, "_check_dependency_versions", lambda: events.append(("deps",)))
    monkeypatch.setattr(groot_n15_server, "_check_groot_checkout", lambda repo: events.append(("checkout", repo)))
    monkeypatch.setattr(groot_n15_server, "_gpu_info",
                        lambda: gpu or dict(cuda=True, name="NVIDIA L4", total_gib=22.0, capability=(8, 9),
                                            bf16=True))
    monkeypatch.setattr(groot_n15_server, "_peak_vram_gib", lambda: 6.1)
    monkeypatch.setattr(groot_n15_server, "GrootN15Model", FakeModel)
    monkeypatch.setattr(groot_n15_server, "create_app", lambda model: FakeApp())
    monkeypatch.setattr("sys.argv", ["groot_n15_server.py", *argv])
    return events


def test_server_main_checks_deps_checkout_gpu_then_warms_and_serves_on_3006(monkeypatch):
    events = _main_harness(monkeypatch, [])
    groot_n15_server.main()
    assert [e[0] for e in events] == ["deps", "checkout", "init", "warm", "run"]
    assert events[1] == ("checkout", "/content/Isaac-GR00T")
    init = events[2]
    assert init[1] == CKPT
    # front_image_key None = the profile's own mapping (overhead_left); denoising None = the profile's 4.
    assert init[2]["client_control_hz"] == 2.0 and init[2]["front_image_key"] is None
    assert init[2]["denoising_steps"] is None and init[2]["groot_repo"] == "/content/Isaac-GR00T"
    assert events[3][1] == ["overhead_left", "wrist"]
    assert set(events[3][2].values()) == {(480, 640, 3)}
    assert events[3][4] == {j: 0.0 for j in JOINT_ORDER}
    assert events[-1] == ("run", 3006)


@pytest.mark.parametrize("gpu", [
    dict(cuda=True, name="Tesla T4", total_gib=14.7, capability=(7, 5), bf16=False),
    dict(cuda=False, name=None, total_gib=0.0, capability=None, bf16=False),
])
def test_server_main_refuses_a_gpu_flash_attention_2_cannot_run_on(monkeypatch, gpu):
    events = _main_harness(monkeypatch, [], gpu=gpu)
    with pytest.raises(SystemExit, match="flash"):
        groot_n15_server.main()
    assert "init" not in [e[0] for e in events]


def test_server_main_refuses_an_unreviewed_checkpoint(monkeypatch):
    _main_harness(monkeypatch, ["--checkpoint", "nvidia/SO_ARM_Starter_Gr00t"])
    with pytest.raises(SystemExit):
        groot_n15_server.main()


def test_dependency_check_refuses_other_versions_and_accepts_local_build_tags(monkeypatch):
    good = {name: next(iter(v)) for name, v in groot_n15_server.PINNED_DEPENDENCIES.items()}
    monkeypatch.setattr(groot_n15_server, "_installed_versions", lambda: dict(good, transformers="4.57.0"))
    with pytest.raises(SystemExit, match="transformers"):
        groot_n15_server._check_dependency_versions()
    monkeypatch.setattr(groot_n15_server, "_installed_versions", lambda: dict(good))
    groot_n15_server._check_dependency_versions()

    import importlib.metadata as metadata

    monkeypatch.setattr(metadata, "version", lambda name: "2.5.1+cu124" if name == "torch" else good[name])
    assert groot_n15_server._installed_versions()["torch"] == "2.5.1"


def test_server_joint_order_matches_the_action_contract():
    assert groot_n15_server.SO101_JOINTS == JOINT_ORDER


# --- client backend --------------------------------------------------------------


def _imported_modules(path):
    tree = ast.parse(Path(path).read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def test_client_shares_no_code_with_the_n17_backend():
    """Pitfall 3 / T-14-10: no import of N1.7's relative_ee_conversion or groot_n17."""
    imported = _imported_modules(VLA_BRIDGE / "backends" / "groot_n15.py")
    assert not any("relative_ee_conversion" in name or "groot_n17" in name for name in imported)
    assert not any(name.startswith("lerobot") for name in imported)


def test_groot_n15_is_registered_on_the_http_lane():
    assert backends.BACKEND_REGISTRY["groot-n15"] is groot_n15.build
    assert "groot-n15" not in backends.GRPC_BRIDGE_BACKENDS
    assert "groot-n15" in backends.get_backend_names()


def test_backend_config_pins_the_checkpoint_and_records_the_confirmed_action_space():
    config = backends.load_backend_config("groot_n15")
    assert config["checkpoint"] == CKPT and config["revision"] == REV
    assert config["endpoint_url"] is None
    assert config["action_space"] == "absolute_joint" and config["action_units"] == "degrees"
    assert config["base_model"] == "nvidia/GR00T-N1.5-3B"
    assert config["isaac_groot_commit"] == ISAAC_GROOT_COMMIT == groot_n15_server.ISAAC_GROOT_COMMIT
    profile = config["checkpoints"][CKPT]
    assert profile["action_space"] == config["action_space"] and profile["action_units"] == config["action_units"]


def test_client_and_server_checkpoint_profiles_agree():
    client = backends.load_backend_config("groot_n15")["checkpoints"]
    server = groot_n15_server.CHECKPOINT_PROFILES
    assert set(client) == set(server)
    for checkpoint, profile in server.items():
        assert client[checkpoint] == profile


def test_absolute_joint_action_is_an_identity_reshape_into_joint_order():
    action = groot_n15.absolute_joint_action(JOINT_RAW)
    assert tuple(action) == JOINT_ORDER
    assert action == JOINT_EXPECTED
    assert all(type(v) is float for v in action.values())


def test_absolute_joint_non_finite_values_pass_through_to_the_validator():
    action = groot_n15.absolute_joint_action([float("nan")] * 5 + [float("inf")])
    assert math.isnan(action["shoulder_pan"]) and math.isinf(action["gripper"])


@pytest.mark.parametrize("raw", [None, [1.0] * 5, [1.0] * 7, "abcdef", [1.0] * 5 + ["x"], [1.0] * 5 + [True],
                                 {"shoulder_pan": 1.0}])
def test_malformed_absolute_joint_action_raises(raw):
    with pytest.raises((ValueError, TypeError)):
        groot_n15.absolute_joint_action(raw)


def test_convert_fn_never_reapplies_the_joint_state():
    """Absolute targets: the same raw action maps to the same target from any pose."""
    convert = groot_n15.make_convert_fn(backends.load_backend_config("groot_n15")["checkpoints"][CKPT])
    far = {j: v + 40.0 for j, v in JOINT_STATE.items()}
    assert convert(JOINT_RAW, JOINT_STATE) == JOINT_EXPECTED
    assert convert(JOINT_RAW, far) == JOINT_EXPECTED


@pytest.mark.parametrize("profile,match", [
    ({"action_space": "relative_ee", "action_units": "degrees"}, "action_space"),
    ({"action_space": "eef_delta", "action_units": "degrees"}, "action_space"),
    ({"action_units": "degrees"}, "action_space"),
    ({"action_space": "absolute_joint", "action_units": "range_m100_100"}, "action_units"),
    ({"action_space": "absolute_joint", "action_units": "radians"}, "action_units"),
    ({"action_space": "absolute_joint"}, "action_units"),
])
def test_unsupported_action_space_or_units_profile_is_rejected(profile, match):
    with pytest.raises(ValueError, match=match):
        groot_n15.make_convert_fn(profile)


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
    (no_config / "groot_n15.json").write_text(json.dumps({"checkpoint": "c", "endpoint_url": None}))
    with pytest.raises(SystemExit, match="--server-address"):
        groot_n15.build(_args(), {0: "wrist"}, {})


def test_build_without_config_file_raises_clear_systemexit(no_config):
    with pytest.raises(SystemExit, match="endpoint_url"):
        groot_n15.build(_args(), {0: "wrist"}, {})


def test_build_rejects_a_grpc_style_host_port_endpoint(no_config):
    with pytest.raises(SystemExit, match="http"):
        groot_n15.build(_args(server_address="0.tcp.ngrok.io:12345"), {0: "wrist"}, {})


@pytest.mark.parametrize("checkpoint", ["nvidia/GR00T-N1.5-3B", "nvidia/SO_ARM_Starter_Gr00t"])
def test_build_rejects_a_checkpoint_without_a_reviewed_profile(checkpoint):
    with pytest.raises(SystemExit, match="reviewed"):
        groot_n15.build(_args(server_address="https://x.ngrok-free.app", checkpoint=checkpoint), {0: "wrist"}, {})


def test_build_rejects_a_profile_with_unsupported_units(no_config):
    (no_config / "groot_n15.json").write_text(json.dumps({
        "checkpoint": "someone/range-ckpt", "endpoint_url": "https://cfg.example",
        "checkpoints": {"someone/range-ckpt": {"action_space": "absolute_joint",
                                               "action_units": "range_m100_100"}}}))
    with pytest.raises(SystemExit, match="action_units"):
        groot_n15.build(_args(), {0: "wrist"}, {})


def _capture_build(monkeypatch):
    seen = {}

    def fake_build(endpoint_url, checkpoint, args, camera_names, convert_fn=None):
        seen.update(endpoint_url=endpoint_url, checkpoint=checkpoint, camera_names=camera_names,
                    convert_fn=convert_fn)
        return "handles"

    monkeypatch.setattr(base_http_adapter, "build_http_adapter_backend", fake_build)
    return seen


def test_build_defaults_to_the_reviewed_so101_checkpoint_with_the_identity_mapping(monkeypatch):
    seen = _capture_build(monkeypatch)
    assert groot_n15.build(_args(server_address="https://abc.ngrok-free.app"), {0: "wrist"}, {}) == "handles"
    assert seen["endpoint_url"] == "https://abc.ngrok-free.app"
    assert seen["checkpoint"] == CKPT
    assert seen["camera_names"] == {0: "wrist"}
    assert seen["convert_fn"](JOINT_RAW, JOINT_STATE) == JOINT_EXPECTED


def test_build_uses_the_config_endpoint_when_no_server_address(monkeypatch, no_config):
    (no_config / "groot_n15.json").write_text(json.dumps({
        "checkpoint": CKPT, "endpoint_url": "https://cfg.example",
        "checkpoints": {CKPT: {"action_space": "absolute_joint", "action_units": "degrees"}}}))
    seen = _capture_build(monkeypatch)
    groot_n15.build(_args(), {0: "wrist"}, {})
    assert seen["endpoint_url"] == "https://cfg.example" and seen["checkpoint"] == CKPT
