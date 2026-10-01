"""Tests for the Wall-X / WALL-OSS backend (Plan 14-08, VLAB-04).

Tier decision: the installed `lerobot==0.6.1` async `PolicyServer` does not
dispatch `wall_x` (`async_inference/constants.py::SUPPORTED_POLICIES`), so
Wall-X runs on the Tier C resident-HTTP-adapter lane. Covers:
- `vla_bridge/backends/wallx.py`: the dispatch-gated branch between the gRPC
  bridge and the HTTP adapter, config handling, and clear `SystemExit`s.
- `vla_bridge/wallx_server.py`: the Colab-side adapter, with lerobot's real
  `WallXPolicy` loading mocked so no 8 GB checkpoint is downloaded.
Nothing here touches real hardware or the network.
"""

import base64
import json
import math
import threading
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from vla_bridge import backends
from vla_bridge import wallx_server
from vla_bridge.action_contract import JOINT_ORDER
from vla_bridge.backends import base_grpc, base_http_adapter, wallx
from vla_bridge.wallx_server import WallXModel, create_app

JOINT_STATE = {
    "shoulder_pan": 10.0,
    "shoulder_lift": -20.0,
    "elbow_flex": 30.0,
    "wrist_flex": 15.0,
    "wrist_roll": 5.0,
    "gripper": 37.0,
}

# What the fake policy predicts for chunk step 0 (normalized space); the fake
# postprocessor "unnormalizes" by adding POST_OFFSET.
CHUNK_STEP0 = [0.5, -0.25, 1.0, 2.0, -1.5, 3.0]
POST_OFFSET = 100.0


def _b64_png(value=50, shape=(24, 32, 3)):
    ok, buf = cv2.imencode(".png", np.full(shape, value, dtype=np.uint8))
    assert ok
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _payload(**overrides):
    body = {
        "images": {"wrist": _b64_png(10), "overhead_left": _b64_png(20), "overhead_right": _b64_png(30)},
        "joint_state": dict(JOINT_STATE),
        "instruction": "Put the object in the bin",
    }
    body.update(overrides)
    return body


# --- Colab-side server: fake lerobot policy stack -------------------------------


def _fake_config(image_keys=("wrist", "top", "side"), action_dim=6, state_dim=6, policy_type="wall_x"):
    input_features = {f"observation.images.{k}": SimpleNamespace(shape=(3, 480, 640)) for k in image_keys}
    input_features["observation.state"] = SimpleNamespace(shape=(state_dim,))
    return SimpleNamespace(
        type=policy_type,
        image_features={f"observation.images.{k}": None for k in image_keys},
        input_features=input_features,
        output_features={"action": SimpleNamespace(shape=(action_dim,))},
        chunk_size=32,
        prediction_mode="diffusion",
        attn_implementation="eager",
    )


class FakePolicy:
    def __init__(self, chunk_step0=CHUNK_STEP0, horizon=32):
        self.calls = []
        self.chunk_step0 = chunk_step0
        self.horizon = horizon

    def predict_action_chunk(self, batch):
        self.calls.append(batch)
        chunk = torch.zeros(1, self.horizon, len(self.chunk_step0))
        chunk[0, 0] = torch.tensor(self.chunk_step0)
        chunk[0, 1:] = 999.0  # later chunk steps must never be returned
        return chunk


@pytest.fixture
def fake_stack(monkeypatch):
    """Mock `_load_policy` (lerobot WallXPolicy + processors) and
    `_download_pinned` (huggingface_hub.snapshot_download)."""
    state = SimpleNamespace(loads=[], downloads=[], prepared=[], policy=FakePolicy(), config=_fake_config())

    def prepare_observation(observation, device, task, robot_type):
        state.prepared.append(dict(observation=observation, device=device, task=task, robot_type=robot_type))
        batch = {k: torch.as_tensor(np.asarray(v)).unsqueeze(0) for k, v in observation.items()}
        batch["task"] = task
        return batch

    def preprocessor(batch):
        return dict(batch, preprocessed=True)

    def postprocessor(action):
        assert action.shape == (1, len(CHUNK_STEP0))
        return action + POST_OFFSET

    def load_policy(checkpoint_dir, base_dir, device):
        state.loads.append(dict(checkpoint_dir=checkpoint_dir, base_dir=base_dir, device=device))
        return dict(config=state.config, policy=state.policy, preprocessor=preprocessor,
                    postprocessor=postprocessor, prepare_observation=prepare_observation)

    def download(repo_id, revision, local_dir):
        state.downloads.append((repo_id, revision, local_dir))
        return f"{local_dir}/{repo_id.replace('/', '--')}/{revision}"

    monkeypatch.setattr(wallx_server, "_load_policy", load_policy)
    monkeypatch.setattr(wallx_server, "_download_pinned", download)
    return state


def _model(**kwargs):
    kwargs.setdefault("device", "cpu")
    return WallXModel(**kwargs)


def test_server_joint_order_mirrors_the_action_contract():
    # The server is uploaded to Colab standalone, so it carries its own copy.
    assert wallx_server.JOINT_ORDER == JOINT_ORDER


def test_server_pins_a_lerobot_wall_x_checkpoint_and_its_base_model():
    assert wallx_server.POLICY_TYPE == "wall_x"
    assert wallx_server.DEFAULT_CHECKPOINT == "Zekai-Chen/wallx_pick_place_3cam_200ep"
    assert wallx_server.DEFAULT_REVISION == "ef8ce55f9772c4c76728228324eebe1d2c373ff6"
    assert wallx_server.DEFAULT_BASE_MODEL == "x-square-robot/wall-oss-flow"
    assert wallx_server.DEFAULT_BASE_REVISION == "44e827683819957d8c574e8b746a1a97e77f518a"
    assert wallx_server.DEFAULT_CAMERA_MAP == {"wrist": "wrist", "top": "overhead_left", "side": "overhead_right"}


def test_config_overrides_force_eager_attention_and_the_pinned_base_dir():
    overrides = wallx_server.config_overrides("/w/base", "cuda")
    assert "--attn_implementation=eager" in overrides
    assert "--pretrained_name_or_path=/w/base" in overrides
    assert "--device=cuda" in overrides


@pytest.mark.parametrize(
    "text, expected",
    [
        ("top=overhead_left,side=overhead_right,wrist=wrist",
         {"top": "overhead_left", "side": "overhead_right", "wrist": "wrist"}),
        (" front = overhead_left , gripper=wrist ", {"front": "overhead_left", "gripper": "wrist"}),
    ],
)
def test_parse_camera_map(text, expected):
    assert wallx_server.parse_camera_map(text) == expected


@pytest.mark.parametrize("text", ["", "top", "top=depth", "top=wrist,top=overhead_left", "=wrist"])
def test_parse_camera_map_rejects_bad_specs(text):
    with pytest.raises(ValueError):
        wallx_server.parse_camera_map(text)


def test_model_downloads_both_pinned_revisions_and_loads_once(fake_stack):
    model = _model(local_dir="/w")
    assert fake_stack.downloads == [
        ("Zekai-Chen/wallx_pick_place_3cam_200ep", "ef8ce55f9772c4c76728228324eebe1d2c373ff6", "/w"),
        ("x-square-robot/wall-oss-flow", "44e827683819957d8c574e8b746a1a97e77f518a", "/w"),
    ]
    assert fake_stack.loads == [dict(
        checkpoint_dir="/w/Zekai-Chen--wallx_pick_place_3cam_200ep/ef8ce55f9772c4c76728228324eebe1d2c373ff6",
        base_dir="/w/x-square-robot--wall-oss-flow/44e827683819957d8c574e8b746a1a97e77f518a",
        device="cpu",
    )]
    for _ in range(3):
        model({k: np.zeros((8, 8, 3), np.uint8) for k in wallx_server.CAMERA_SOURCES},
              dict(JOINT_STATE), "Put the object in the bin")
    assert len(fake_stack.loads) == 1
    assert model.model_version == "Zekai-Chen/wallx_pick_place_3cam_200ep@ef8ce55f9772"


def test_unpinned_model_loads_repo_ids_without_download(fake_stack):
    model = _model(checkpoint="someone/wallx", revision=None, base_revision=None)
    assert fake_stack.downloads == []
    assert fake_stack.loads[0]["checkpoint_dir"] == "someone/wallx"
    assert fake_stack.loads[0]["base_dir"] == "x-square-robot/wall-oss-flow"
    assert model.model_version == "someone/wallx@unpinned"


def test_model_rejects_checkpoint_camera_missing_from_camera_map(fake_stack):
    fake_stack.config = _fake_config(image_keys=("front", "wrist"))
    with pytest.raises(ValueError, match="front"):
        _model()
    _model(camera_map={"front": "overhead_left", "wrist": "wrist"})  # explicit mapping works


@pytest.mark.parametrize(
    "config",
    [_fake_config(action_dim=7), _fake_config(state_dim=7), _fake_config(policy_type="act")],
)
def test_model_rejects_non_so101_or_non_wall_x_checkpoints(fake_stack, config):
    fake_stack.config = config
    with pytest.raises(ValueError):
        _model()


def test_model_feeds_mapped_rgb_cameras_state_and_task_and_returns_first_chunk_step(fake_stack):
    model = _model()
    images = {
        "wrist": np.full((8, 8, 3), 1, np.uint8),
        "overhead_left": np.full((8, 8, 3), 2, np.uint8),
        "overhead_right": np.full((8, 8, 3), 3, np.uint8),
    }
    action = model(images, dict(JOINT_STATE), "Put the object in the bin")

    observation = fake_stack.prepared[0]["observation"]
    assert observation["observation.images.wrist"] is images["wrist"]
    assert observation["observation.images.top"] is images["overhead_left"]
    assert observation["observation.images.side"] is images["overhead_right"]
    assert observation["observation.state"].dtype == np.float32
    assert observation["observation.state"].tolist() == [JOINT_STATE[j] for j in JOINT_ORDER]
    assert fake_stack.prepared[0]["task"] == "Put the object in the bin"
    assert fake_stack.policy.calls[0]["preprocessed"] is True

    assert tuple(action) == JOINT_ORDER
    assert action == pytest.approx({j: v + POST_OFFSET for j, v in zip(JOINT_ORDER, CHUNK_STEP0)})
    assert all(isinstance(v, float) for v in action.values())


def test_model_raises_runtime_error_on_non_finite_output(fake_stack):
    fake_stack.policy = FakePolicy(chunk_step0=[float("nan")] + CHUNK_STEP0[1:])
    model = _model()
    with pytest.raises(RuntimeError, match="non-finite"):
        model({k: np.zeros((8, 8, 3), np.uint8) for k in wallx_server.CAMERA_SOURCES},
              dict(JOINT_STATE), "x")


def test_action_endpoint_returns_joint_dict_and_model_version(fake_stack):
    http = create_app(_model()).test_client()
    response = http.post("/action", json=_payload())
    assert response.status_code == 200
    body = response.get_json()
    assert set(body["action"]) == set(JOINT_ORDER)  # Flask sorts JSON keys; order is irrelevant
    assert body["action"] == pytest.approx({j: v + POST_OFFSET for j, v in zip(JOINT_ORDER, CHUNK_STEP0)})
    assert body["model_version"] == "Zekai-Chen/wallx_pick_place_3cam_200ep@ef8ce55f9772"
    assert body["inference_ms"] >= 0


def test_action_endpoint_decodes_png_to_rgb(fake_stack):
    http = create_app(_model()).test_client()
    bgr = np.zeros((4, 6, 3), np.uint8)
    bgr[..., 0] = 200  # blue in OpenCV's BGR order
    ok, buf = cv2.imencode(".png", bgr)
    assert ok
    images = dict(_payload()["images"], wrist=base64.b64encode(buf.tobytes()).decode("ascii"))
    assert http.post("/action", json=_payload(images=images)).status_code == 200
    wrist = fake_stack.prepared[0]["observation"]["observation.images.wrist"]
    assert wrist[0, 0].tolist() == [0, 0, 200]


@pytest.mark.parametrize(
    "body",
    [
        _payload(images={"wrist": _b64_png(), "overhead_left": _b64_png()}),  # side source missing
        _payload(images={"wrist": "not-base64!", "overhead_left": _b64_png(), "overhead_right": _b64_png()}),
        _payload(images={"wrist": base64.b64encode(b"nope").decode(), "overhead_left": _b64_png(),
                         "overhead_right": _b64_png()}),
        _payload(images=["wrist"]),
        _payload(instruction=""),
        _payload(instruction=7),
        _payload(joint_state=None),
        _payload(joint_state=[1, 2, 3, 4, 5, 6]),
        _payload(joint_state={k: v for k, v in JOINT_STATE.items() if k != "gripper"}),
        _payload(joint_state=dict(JOINT_STATE, elbow_flex=float("nan"))),
        _payload(joint_state=dict(JOINT_STATE, elbow_flex=True)),
        _payload(joint_state=dict(JOINT_STATE, elbow_flex="30")),
        {k: v for k, v in _payload().items() if k != "instruction"},
    ],
)
def test_malformed_body_returns_400(fake_stack, body):
    http = create_app(_model()).test_client()
    response = http.post("/action", json=body)
    assert response.status_code == 400
    assert fake_stack.policy.calls == []


def test_non_json_body_returns_400(fake_stack):
    http = create_app(_model()).test_client()
    assert http.post("/action", data="nope", content_type="text/plain").status_code == 400


def test_model_failure_returns_500_not_a_bogus_action(fake_stack):
    fake_stack.policy = FakePolicy(chunk_step0=[float("inf")] * 6)
    http = create_app(_model()).test_client()
    response = http.post("/action", json=_payload())
    assert response.status_code == 500
    assert "action" not in response.get_json()


def test_health_reports_ready_with_metadata(fake_stack):
    body = create_app(_model()).test_client().get("/health").get_json()
    assert body["status"] == "ready"
    meta = body["model"]
    assert meta["checkpoint"] == "Zekai-Chen/wallx_pick_place_3cam_200ep"
    assert meta["base_model"] == "x-square-robot/wall-oss-flow"
    assert meta["action_space"] == "absolute_joint"
    assert meta["joint_order"] == list(JOINT_ORDER)
    assert meta["camera_map"] == wallx_server.DEFAULT_CAMERA_MAP


def test_overlapping_request_gets_429_not_queued():
    entered, release = threading.Event(), threading.Event()

    class Slow:
        model_version = "slow"
        metadata = {}
        required_sources = ("wrist",)

        def __call__(self, images, joint_state, instruction):
            entered.set()
            release.wait(5)
            return dict(JOINT_STATE)

    app = create_app(Slow())
    first = {}
    worker = threading.Thread(target=lambda: first.update(r=app.test_client().post("/action", json=_payload())))
    worker.start()
    assert entered.wait(5)
    assert app.test_client().post("/action", json=_payload()).status_code == 429
    release.set()
    worker.join(5)
    assert first["r"].status_code == 200


def test_server_main_refuses_an_unverified_lerobot_version(monkeypatch):
    monkeypatch.setattr(wallx_server, "_installed_lerobot_version", lambda: "0.7.0")
    monkeypatch.setattr("sys.argv", ["wallx_server.py"])
    with pytest.raises(SystemExit, match="0.6.1"):
        wallx_server.main()


def test_server_main_warms_model_before_serving(monkeypatch):
    events = []

    class FakeModel:
        def __init__(self, **kwargs):
            events.append(("load", kwargs))

        def __call__(self, images, joint_state, instruction):
            events.append(("infer", sorted(images), instruction))
            return dict(JOINT_STATE)

    class FakeApp:
        def run(self, **kwargs):
            events.append(("serve", kwargs))

    monkeypatch.setattr(wallx_server, "_installed_lerobot_version", lambda: "0.6.1")
    monkeypatch.setattr(wallx_server, "WallXModel", FakeModel)
    monkeypatch.setattr(wallx_server, "create_app", lambda model: FakeApp())
    monkeypatch.setattr("sys.argv", ["wallx_server.py", "--camera-map", "front=overhead_left,wrist=wrist",
                                     "--port", "3999"])
    wallx_server.main()
    assert [e[0] for e in events] == ["load", "infer", "serve"]
    load_kwargs = events[0][1]
    assert load_kwargs["checkpoint"] == wallx_server.DEFAULT_CHECKPOINT
    assert load_kwargs["revision"] == wallx_server.DEFAULT_REVISION
    assert load_kwargs["camera_map"] == {"front": "overhead_left", "wrist": "wrist"}
    assert events[1][1] == sorted(wallx_server.CAMERA_SOURCES)
    assert events[2][1]["port"] == 3999


# --- control-side backend ------------------------------------------------------


def _args(**overrides):
    defaults = dict(server_address=None, checkpoint=None, port="/dev/null", robot_id="r",
                    stereo_camera_index=1)
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


@pytest.fixture
def no_config(monkeypatch, tmp_path):
    monkeypatch.setattr(backends, "BACKEND_CONFIG_DIR", tmp_path)
    return tmp_path


@pytest.fixture
def spies(monkeypatch):
    calls = SimpleNamespace(grpc=[], http=[])

    def fake_grpc(policy_type, checkpoint, args, camera_names, joint_limits_deg, convert_fn=None):
        calls.grpc.append(dict(policy_type=policy_type, checkpoint=checkpoint, args=args,
                               camera_names=camera_names, joint_limits_deg=joint_limits_deg,
                               convert_fn=convert_fn))
        return "grpc-handles"

    def fake_http(endpoint_url, checkpoint, args, camera_names, convert_fn=None):
        calls.http.append(dict(endpoint_url=endpoint_url, checkpoint=checkpoint, args=args,
                               camera_names=camera_names, convert_fn=convert_fn))
        return "http-handles"

    monkeypatch.setattr(base_grpc, "build_grpc_action_source", fake_grpc)
    monkeypatch.setattr(base_http_adapter, "build_http_adapter_backend", fake_http)
    return calls


def test_wallx_is_registered_on_the_http_lane():
    assert backends.BACKEND_REGISTRY["wallx"] is wallx.build
    assert "wallx" not in backends.GRPC_BRIDGE_BACKENDS


def test_installed_lerobot_policy_server_does_not_dispatch_wall_x():
    """Canary for the tier decision. If a lerobot upgrade adds `wall_x` to the
    async server's dispatch table, this fails: re-run the 14-08 source check."""
    from lerobot.async_inference import constants

    assert wallx.POLICY_TYPE == "wall_x"
    assert "wall_x" not in constants.SUPPORTED_POLICIES
    assert wallx.policy_server_dispatches("wall_x") is False
    assert wallx.policy_server_dispatches("act") is True


def test_module_docstring_records_the_installed_source_finding():
    doc = wallx.__doc__
    assert "lerobot==0.6.1" in doc
    assert "SUPPORTED_POLICIES" in doc
    assert "wall_x" in doc


def test_backend_config_matches_server_defaults_and_leaves_endpoint_to_runtime():
    config = backends.load_backend_config("wallx")
    assert config["checkpoint"] == wallx_server.DEFAULT_CHECKPOINT
    assert config["revision"] == wallx_server.DEFAULT_REVISION
    assert config["endpoint_url"] is None


def test_unsupported_dispatch_builds_http_adapter_without_convert_fn(monkeypatch, spies):
    monkeypatch.setattr(wallx, "policy_server_dispatches", lambda policy_type: False)
    args = _args(server_address="https://abc.ngrok-free.app")
    assert backends.BACKEND_REGISTRY["wallx"](args, {0: "wrist"}, {"gripper": (0, 100)}) == "http-handles"
    assert spies.grpc == []
    assert spies.http == [dict(endpoint_url="https://abc.ngrok-free.app",
                               checkpoint="Zekai-Chen/wallx_pick_place_3cam_200ep",
                               args=args, camera_names={0: "wrist"}, convert_fn=None)]


def test_supported_dispatch_builds_grpc_source_with_wall_x_policy_type(monkeypatch, spies):
    monkeypatch.setattr(wallx, "policy_server_dispatches", lambda policy_type: policy_type == "wall_x")
    args = _args(server_address="0.tcp.ngrok.io:12345", checkpoint="someone/wallx-so101")
    limits = {"gripper": (0, 100)}
    assert backends.BACKEND_REGISTRY["wallx"](args, {0: "wrist"}, limits) == "grpc-handles"
    assert spies.http == []
    assert spies.grpc == [dict(policy_type="wall_x", checkpoint="someone/wallx-so101", args=args,
                               camera_names={0: "wrist"}, joint_limits_deg=limits, convert_fn=None)]


def test_supported_dispatch_still_uses_the_http_adapter_for_an_http_endpoint(monkeypatch, spies):
    monkeypatch.setattr(wallx, "policy_server_dispatches", lambda policy_type: True)
    wallx.build(_args(server_address="http://h:3002"), {0: "wrist"}, {})
    assert spies.grpc == [] and spies.http[0]["endpoint_url"] == "http://h:3002"


def test_unsupported_dispatch_rejects_a_grpc_style_endpoint(monkeypatch, spies):
    monkeypatch.setattr(wallx, "policy_server_dispatches", lambda policy_type: False)
    with pytest.raises(SystemExit, match="SUPPORTED_POLICIES"):
        wallx.build(_args(server_address="0.tcp.ngrok.io:12345"), {0: "wrist"}, {})
    assert spies.grpc == [] and spies.http == []


def test_build_without_any_endpoint_raises_clear_systemexit(no_config, spies):
    (no_config / "wallx.json").write_text(json.dumps({"checkpoint": "c", "endpoint_url": None}))
    with pytest.raises(SystemExit, match="--server-address"):
        wallx.build(_args(), {0: "wrist"}, {})


def test_build_without_checkpoint_raises_clear_systemexit(no_config, spies):
    (no_config / "wallx.json").write_text(json.dumps({"endpoint_url": "https://cfg.example"}))
    with pytest.raises(SystemExit, match="--checkpoint"):
        wallx.build(_args(), {0: "wrist"}, {})


def test_build_without_config_file_raises_clear_systemexit(no_config, spies):
    with pytest.raises(SystemExit):
        wallx.build(_args(), {0: "wrist"}, {})


def test_build_falls_back_to_config_endpoint_and_cli_checkpoint_wins(no_config, spies):
    (no_config / "wallx.json").write_text(json.dumps({"checkpoint": "c", "endpoint_url": "https://cfg.example"}))
    wallx.build(_args(), {0: "wrist"}, {})
    wallx.build(_args(checkpoint="cli/ckpt"), {0: "wrist"}, {})
    assert [(c["endpoint_url"], c["checkpoint"]) for c in spies.http] == [
        ("https://cfg.example", "c"), ("https://cfg.example", "cli/ckpt")]


# --- Task 2: mocked full round trip (Tier C) ------------------------------------


class _Routed:
    """`requests.Response` stand-in wrapping a Flask test-client response."""

    def __init__(self, response):
        self._response = response
        self.status_code = response.status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise base_http_adapter.requests.HTTPError(f"{self.status_code}")

    def json(self):
        return self._response.get_json()


def _route_requests_to(app, monkeypatch):
    client, posted = app.test_client(), []

    def fake_post(url, json=None, timeout=None):
        posted.append((url, json))
        return _Routed(client.post("/action", json=json))

    monkeypatch.setattr(base_http_adapter.requests, "post", fake_post)
    return posted


def test_round_trip_yields_a_validated_finite_joint_action(fake_stack, monkeypatch):
    """Real create_app() behind the real HttpAdapterActionSource with
    convert_fn=None (as wallx.build() wires it), then the real validator."""
    from test_backend_registry import FakeStereo, FakeWristCap
    from vla_bridge import safety_validator

    step0 = [1.0, -2.0, 3.0, -1.0, 2.0, -3.0]
    fake_stack.policy = FakePolicy(chunk_step0=step0)
    posted = _route_requests_to(create_app(_model()), monkeypatch)
    source = base_http_adapter.HttpAdapterActionSource(
        "https://abc.ngrok-free.app", "Zekai-Chen/wallx_pick_place_3cam_200ep",
        FakeWristCap(), FakeStereo(), convert_fn=None,
    )
    action, model_version = source.get_action(JOINT_STATE, "Put the object in the bin")

    assert [url for url, _ in posted] == ["https://abc.ngrok-free.app/action"]
    sent = posted[0][1]
    assert sent["joint_state"] == JOINT_STATE
    assert set(sent["images"]) == set(wallx_server.CAMERA_SOURCES)
    assert model_version == "Zekai-Chen/wallx_pick_place_3cam_200ep@ef8ce55f9772"
    assert source.last_details["bridge_flags"] == []
    assert set(action) == set(JOINT_ORDER)
    assert all(isinstance(v, float) and math.isfinite(v) for v in action.values())
    assert action == pytest.approx({j: v + POST_OFFSET for j, v in zip(JOINT_ORDER, step0)})
    # The server fed the client's joint dict to the model as observation.state, in JOINT_ORDER.
    state = fake_stack.prepared[0]["observation"]["observation.state"]
    assert state.tolist() == [JOINT_STATE[j] for j in JOINT_ORDER]

    limits = {j: (-180.0, 180.0) for j in JOINT_ORDER[:5]} | {"gripper": (0.0, 100.0)}
    validated, _flags = safety_validator.validate_action(action, JOINT_STATE, limits)
    assert set(validated) == set(JOINT_ORDER)
    assert all(math.isfinite(v) for v in validated.values())
    assert any(validated[j] != JOINT_STATE[j] for j in JOINT_ORDER)  # a real, non-held action


def test_round_trip_holds_position_when_the_model_fails(fake_stack, monkeypatch):
    from test_backend_registry import FakeStereo, FakeWristCap

    fake_stack.policy = FakePolicy(chunk_step0=[float("nan")] * 6)
    _route_requests_to(create_app(_model()), monkeypatch)
    source = base_http_adapter.HttpAdapterActionSource(
        "https://abc.ngrok-free.app", "wallx", FakeWristCap(), FakeStereo(), convert_fn=None)
    action, model_version = source.get_action(JOINT_STATE, "Put the object in the bin")
    assert action == JOINT_STATE
    assert model_version == "wallx@http-error-holding-position"
