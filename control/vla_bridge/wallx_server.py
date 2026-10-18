"""Resident Wall-X / WALL-OSS HTTP adapter (Tier C, Plan 14-08). Run on Colab
with `lerobot[wallx]==0.6.1` installed in the same Python.

Why an HTTP adapter and not the gRPC bridge: the installed `lerobot==0.6.1`
async `PolicyServer` only dispatches `async_inference/constants.py::
SUPPORTED_POLICIES` = act, smolvla, diffusion, tdmpc, vqbet, pi0, pi05, groot
(enforced in `policy_server.py:130`). `wall_x` is not in it, even though
`lerobot/policies/wall_x/` ships a full `WallXPolicy`.

This mirrors `openvla_oft_server.py` / `fastfs_server.py`:
- the model loads ONCE
- one throwaway warmup inference runs before the socket opens
- `/health` reports ready
- a non-blocking `inference_lock` answers concurrent requests with 429
- `ValueError`/`KeyError`/`TypeError` on a request become 400, and a model
  failure (`RuntimeError`, e.g. a non-finite action) becomes 500, so the
  client holds position instead of executing a bogus action

Unlike OpenVLA-OFT it returns the `action_contract.JOINT_ORDER`-keyed absolute
joint dict directly (Wall-X SO-101 fine-tunes are joint-space), so the client
runs with `convert_fn=None` (`base_http_adapter` contract).

Model loading: lerobot's own WALL-OSS integration, NOT the X-Square-Robot/wall-x
GitHub install
-----------------------------------------------------------------------------
14-RESEARCH.md scoped the `github.com/X-Square-Robot/wall-x` source install to
the case where Wall-X is NOT reachable through `lerobot.policies`. It is:
`lerobot/policies/wall_x/` is "adapted from their open-source WallX repository"
(lerobot docs/source/walloss.mdx @ v0.6.1), and the `lerobot[wallx]` extra is
pure Python (transformers>=5.4,<5.6, peft, scipy, torchdiffeq, qwen-vl-utils).
The GitHub route would instead need a source build of flash-attn 2.8.3,
`dmuon` from a second GitHub repo, lerobot `main` with `--no-deps`, and an
nvcc `CUDAExtension` compile (wall-x `setup.py` @ ce2a1ae). That compile is
14-RESEARCH.md Pitfall 5, and this path does not have it.

Loading follows lerobot 0.6.1's own inference path (`async_inference/
policy_server.py` SendPolicyInstructions + `common/control_utils.predict_action`):
`PreTrainedConfig.from_pretrained` -> `get_policy_class("wall_x").from_pretrained`
-> `make_pre_post_processors(..., pretrained_path=...)` ->
`prepare_observation_for_inference` -> preprocessor -> policy -> postprocessor.
Two 0.6.1-specific details, both verified against installed source:
- `WallXConfig.__post_init__` raises unless `attn_implementation == "eager"`,
  and some hub checkpoints saved `"sdpa"` (e.g. Zekai-Chen/wallx_pick_place_3cam).
  The config is therefore parsed with the draccus override
  `--attn_implementation=eager`.
- `WallXPolicy.__init__` loads the base model from
  `config.pretrained_name_or_path` (`x-square-robot/wall-oss-flow`) with no
  revision argument, i.e. `main`. To honor the pinned base revision (T-14-06)
  the base is snapshot-downloaded at its sha and the override points
  `--pretrained_name_or_path` at that local directory.

Default checkpoint (HF API read 2026-10-01)
-------------------------------------------
Zekai-Chen/wallx_pick_place_3cam_200ep @ ef8ce55f9772c4c76728228324eebe1d2c373ff6
- Apache-2.0, `type: wall_x`, `prediction_mode: diffusion`,
  `attn_implementation: eager`, chunk 32, 60k steps on 200 episodes.
- Inputs: `observation.state` [6] plus `observation.images.{wrist,top,side}`
  (3x480x640). Output: `action` [6].
- Dataset Zekai-Chen/so101_pi05_pick_place_3cam (Apache-2.0, `so_follower`,
  300 episodes, 30 fps, one task "Put the object in the bin"). Its state and
  action names are exactly `shoulder_pan.pos .. gripper.pos` in JOINT_ORDER,
  in degree mode (shoulder_lift down to -103.5, wrist_roll down to -135.3,
  gripper 0..63). This matches `action_contract.ACTION_UNITS`.
- Base: x-square-robot/wall-oss-flow @ 44e827683819957d8c574e8b746a1a97e77f518a
  (Qwen2.5-VL MoE, model.safetensors 8.4 GB; the fine-tune is another 8.4 GB).
`x-square-robot/wall-oss-0.5` (the plan's suggestion) is the raw foundation
model: `Qwen2_5_VLForConditionalGeneration`, no `type: wall_x`, and no SO-101
normalizer stats, so it cannot produce an SO-101 joint action.

Camera mapping: the client sends `wrist`, `overhead_left` and `overhead_right`
(the AR0144 stereo halves). `--camera-map` assigns each checkpoint image key
to one of those, and startup refuses a checkpoint whose cameras are not all
mapped. The default map (wrist->wrist, top->overhead_left,
side->overhead_right) fits the default checkpoint. Our rig has no side camera,
so `side` gets the right stereo half: out of distribution, which is acceptable
for this phase's "one validated live action" bar.

Only the FIRST action of each 32-step chunk is returned (receding horizon,
same as openvla_oft_server.py). The control loop re-queries every tick.

Colab launch: see vla_bridge/notebooks/http_wallx.ipynb.
"""
import argparse
import base64
import math
import threading
import time
from pathlib import Path

import cv2
import numpy as np

POLICY_TYPE = "wall_x"
VERIFIED_LEROBOT_VERSION = "0.6.1"
DEFAULT_CHECKPOINT = "Zekai-Chen/wallx_pick_place_3cam_200ep"
# T-14-06: pin the reviewed HF revisions (HF API sha read 2026-10-01), never `main`.
DEFAULT_REVISION = "ef8ce55f9772c4c76728228324eebe1d2c373ff6"
DEFAULT_BASE_MODEL = "x-square-robot/wall-oss-flow"
DEFAULT_BASE_REVISION = "44e827683819957d8c574e8b746a1a97e77f518a"
DEFAULT_LOCAL_DIR = "/content/wallx_checkpoints"
DEFAULT_PORT = 3002
# Copy of action_contract.JOINT_ORDER: this file is uploaded to Colab on its own.
JOINT_ORDER = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
CAMERA_SOURCES = ("wrist", "overhead_left", "overhead_right")
DEFAULT_CAMERA_MAP = {"wrist": "wrist", "top": "overhead_left", "side": "overhead_right"}
IMAGE_PREFIX = "observation.images."
STATE_KEY = "observation.state"
ACTION_KEY = "action"
MAX_IMAGE_SIDE = 1024


def _installed_lerobot_version():
    from importlib.metadata import version

    return version("lerobot")


def _download_pinned(repo_id, revision, local_dir):
    """Materialise the exact pinned HF revision as a real directory."""
    from huggingface_hub import snapshot_download

    target = Path(local_dir) / repo_id.replace("/", "--") / revision
    return snapshot_download(repo_id=repo_id, revision=revision, local_dir=str(target))


def config_overrides(base_dir, device):
    """draccus overrides applied when parsing the checkpoint's config.json."""
    return [
        "--attn_implementation=eager",
        f"--pretrained_name_or_path={base_dir}",
        f"--device={device}",
    ]


def parse_camera_map(text):
    """`"top=overhead_left,side=overhead_right,wrist=wrist"` -> dict."""
    camera_map = {}
    for item in str(text).split(","):
        if not item.strip():
            continue
        key, sep, source = (part.strip() for part in item.partition("="))
        if not sep or not key or source not in CAMERA_SOURCES:
            raise ValueError(f"camera map entry {item!r} must be <checkpoint key>=<one of {CAMERA_SOURCES}>")
        if key in camera_map:
            raise ValueError(f"camera map assigns {key!r} twice")
        camera_map[key] = source
    if not camera_map:
        raise ValueError("camera map is empty")
    return camera_map


def _load_policy(checkpoint_dir, base_dir, device):
    """lerobot 0.6.1's own Wall-X load path (see module docstring)."""
    import lerobot.policies  # noqa: F401  (registers WallXConfig with draccus)
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    from lerobot.policies.utils import prepare_observation_for_inference

    config = PreTrainedConfig.from_pretrained(checkpoint_dir, cli_overrides=config_overrides(base_dir, device))
    if config.type != POLICY_TYPE:
        raise ValueError(f"{checkpoint_dir} is a {config.type!r} checkpoint, not {POLICY_TYPE!r}")
    policy = get_policy_class(POLICY_TYPE).from_pretrained(checkpoint_dir, config=config)
    policy.to(device)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=checkpoint_dir,
        preprocessor_overrides={
            "device_processor": {"device": device},
            "rename_observations_processor": {"rename_map": {}},
        },
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    return dict(config=policy.config, policy=policy, preprocessor=preprocessor,
                postprocessor=postprocessor, prepare_observation=prepare_observation_for_inference)


def _feature_dim(feature):
    shape = tuple(getattr(feature, "shape", ()) or ())
    return shape[0] if len(shape) == 1 else None


class WallXModel:
    def __init__(self, checkpoint=DEFAULT_CHECKPOINT, revision=DEFAULT_REVISION,
                 base_model=DEFAULT_BASE_MODEL, base_revision=DEFAULT_BASE_REVISION,
                 camera_map=None, local_dir=DEFAULT_LOCAL_DIR, device="cuda"):
        checkpoint_dir = _download_pinned(checkpoint, revision, local_dir) if revision else checkpoint
        base_dir = _download_pinned(base_model, base_revision, local_dir) if base_revision else base_model
        loaded = _load_policy(checkpoint_dir, base_dir, device)
        config = loaded["config"]
        self._validate_config(checkpoint, config)

        self.camera_map = dict(DEFAULT_CAMERA_MAP if camera_map is None else camera_map)
        image_keys = [key[len(IMAGE_PREFIX):] for key in config.image_features]
        unmapped = [key for key in image_keys if key not in self.camera_map]
        if unmapped:
            raise ValueError(
                f"{checkpoint} declares cameras {image_keys}; {unmapped} have no --camera-map entry "
                f"(map each to one of {CAMERA_SOURCES})"
            )
        self.image_keys = image_keys
        self.required_sources = tuple(dict.fromkeys(self.camera_map[key] for key in image_keys))

        self.device = device
        self.policy = loaded["policy"]
        self.preprocessor = loaded["preprocessor"]
        self.postprocessor = loaded["postprocessor"]
        self._prepare_observation = loaded["prepare_observation"]
        self.model_version = f"{checkpoint}@{revision[:12] if revision else 'unpinned'}"
        self.metadata = dict(
            checkpoint=checkpoint, revision=revision or "unpinned",
            base_model=base_model, base_revision=base_revision or "unpinned",
            policy_type=POLICY_TYPE, prediction_mode=getattr(config, "prediction_mode", None),
            chunk_size=getattr(config, "chunk_size", None), action_space="absolute_joint",
            action_units="degrees, gripper percent", joint_order=list(JOINT_ORDER),
            camera_map={key: self.camera_map[key] for key in image_keys},
            returns="first action of chunk",
        )

    @staticmethod
    def _validate_config(checkpoint, config):
        if getattr(config, "type", None) != POLICY_TYPE:
            raise ValueError(f"{checkpoint} is a {getattr(config, 'type', None)!r} checkpoint, not {POLICY_TYPE!r}")
        action_dim = _feature_dim(config.output_features.get(ACTION_KEY))
        state_dim = _feature_dim(config.input_features.get(STATE_KEY))
        if action_dim != len(JOINT_ORDER) or state_dim != len(JOINT_ORDER):
            raise ValueError(
                f"{checkpoint} has state dim {state_dim} / action dim {action_dim}; an SO-101 "
                f"joint-space checkpoint needs {len(JOINT_ORDER)} / {len(JOINT_ORDER)}"
            )

    def __call__(self, images, joint_state, instruction):
        """`images`: source name -> HxWx3 uint8 RGB. `joint_state`: JOINT_ORDER
        dict (degrees, gripper percent). Returns the absolute JOINT_ORDER dict."""
        import torch

        observation = {IMAGE_PREFIX + key: images[self.camera_map[key]] for key in self.image_keys}
        observation[STATE_KEY] = np.asarray([joint_state[j] for j in JOINT_ORDER], dtype=np.float32)
        with torch.inference_mode():
            batch = self._prepare_observation(observation, torch.device(self.device), instruction, "")
            batch = self.preprocessor(batch)
            chunk = self.policy.predict_action_chunk(batch)
            first = self.postprocessor(chunk[:, 0, :])
        values = np.asarray(first.detach().cpu().float().numpy() if hasattr(first, "detach") else first,
                            dtype=np.float64).reshape(-1)
        if values.shape != (len(JOINT_ORDER),):
            raise RuntimeError(f"model returned action of shape {values.shape}, expected ({len(JOINT_ORDER)},)")
        if not np.all(np.isfinite(values)):
            raise RuntimeError(f"model returned a non-finite action {values.tolist()}")
        return {joint: float(v) for joint, v in zip(JOINT_ORDER, values)}


def _decode_rgb(b64_png):
    """base64 PNG (OpenCV BGR-encoded by the client) -> HxWx3 uint8 RGB."""
    if not isinstance(b64_png, str):
        raise TypeError("image must be a base64 string")
    raw = base64.b64decode(b64_png, validate=True)
    frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("invalid PNG")
    if max(frame.shape[:2]) > MAX_IMAGE_SIDE:
        raise ValueError(f"image dimensions must be <= {MAX_IMAGE_SIDE}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _parse_joint_state(value):
    if not isinstance(value, dict):
        raise TypeError("joint_state must be an object keyed by joint name")
    state = {}
    for joint in JOINT_ORDER:
        v = value[joint]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError(f"joint_state[{joint!r}] must be a finite number")
        state[joint] = float(v)
    return state


def create_app(infer):
    from flask import Flask, jsonify, request
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024
    inference_lock = threading.Lock()

    @app.get('/health')
    def health():
        return jsonify(status='ready', protocol_version=1, model=infer.metadata)

    @app.post('/action')
    def action():
        if not inference_lock.acquire(blocking=False):
            return jsonify(error='wall-x service busy'), 429
        try:
            body = request.get_json(silent=True)
            if not isinstance(body, dict):
                raise ValueError('expected a JSON object body')
            images_in = body['images']
            if not isinstance(images_in, dict):
                raise TypeError('images must be an object')
            images = {key: _decode_rgb(images_in[key]) for key in infer.required_sources}
            instruction = body['instruction']
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError('instruction must be a non-empty string')
            joint_state = _parse_joint_state(body['joint_state'])
        except (ValueError, KeyError, TypeError) as exc:
            inference_lock.release()
            return jsonify(error=str(exc)), 400
        try:
            started = time.perf_counter()
            result = infer(images, joint_state, instruction)
            elapsed = (time.perf_counter() - started) * 1000
            return jsonify(action={j: float(result[j]) for j in JOINT_ORDER},
                           model_version=infer.model_version, inference_ms=elapsed)
        except RuntimeError as exc:
            return jsonify(error=f'wall-x inference failed: {exc}'), 500
        finally:
            inference_lock.release()
    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT)
    parser.add_argument('--revision', default=DEFAULT_REVISION,
                        help="pinned HF commit sha; pass '' only to deliberately load the unpinned repo id")
    parser.add_argument('--base-model', default=DEFAULT_BASE_MODEL)
    parser.add_argument('--base-revision', default=DEFAULT_BASE_REVISION,
                        help="pinned sha of the base model WallXPolicy loads; '' loads its main branch")
    parser.add_argument('--camera-map', default=None,
                        help="checkpoint camera key -> client source, e.g. 'top=overhead_left,wrist=wrist'")
    parser.add_argument('--local-dir', default=DEFAULT_LOCAL_DIR)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--port', type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    installed = _installed_lerobot_version()
    if installed != VERIFIED_LEROBOT_VERSION:
        raise SystemExit(
            f"wallx_server.py was verified against lerobot=={VERIFIED_LEROBOT_VERSION} (WallXConfig's eager-only "
            f"check, processor step names, load path); found {installed}. Install lerobot[wallx]=="
            f"{VERIFIED_LEROBOT_VERSION} or re-verify this file against the new source."
        )
    camera_map = parse_camera_map(args.camera_map) if args.camera_map else None
    model = WallXModel(checkpoint=args.checkpoint, revision=args.revision or None,
                       base_model=args.base_model, base_revision=args.base_revision or None,
                       camera_map=camera_map, local_dir=args.local_dir, device=args.device)
    # CUDA kernels / lazy init happen before the socket opens and readiness is reported.
    warmup = np.zeros((480, 640, 3), dtype=np.uint8)
    model({source: warmup for source in CAMERA_SOURCES}, {j: 0.0 for j in JOINT_ORDER}, 'warm up')
    print('Wall-X model loaded and warmed; ready', flush=True)
    create_app(model).run(host='0.0.0.0', port=args.port, threaded=True, use_reloader=False)


if __name__ == '__main__':
    main()
