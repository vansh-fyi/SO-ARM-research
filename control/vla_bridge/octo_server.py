"""Resident Octo HTTP adapter (Tier C, Plan 14-10, VLAB-08). Run on Colab inside
the isolated Octo venv (see policy_server_launch.md, "Tier C: Octo").

Same shape as `openvla_server.py` / `openvla_oft_server.py` / `fastfs_server.py`:
- the model loads ONCE
- one throwaway warmup inference runs before the socket opens
- `/health` reports ready
- a non-blocking `inference_lock` answers concurrent requests with 429
- `ValueError`/`KeyError`/`TypeError` on a request become 400
- a model failure (`RuntimeError`, e.g. a non-finite action) becomes 500,
  so the client holds position instead of parsing NaN JSON

The server stays model-format-only. It returns the first step of Octo's
un-normalized action chunk, and `backends/octo.py` turns that into a
`JOINT_ORDER` target for the checkpoint's action space.

Verified against live upstream source (2026-10-01, execution discipline)
------------------------------------------------------------------------
github.com/octo-models/octo @ 241fb35 (main HEAD; tag v1.5 = 5eaa5c6 + a
setup.py packaging fix), MIT. Octo is JAX/Flax, not PyTorch, and is NOT on
PyPI (14-RESEARCH audit). Files read: `README.md`, `requirements.txt`,
`octo/model/octo_model.py`, `octo/utils/gym_wrappers.py`,
`octo/model/components/tokenizers.py`, `octo/data/utils/text_processing.py`,
`octo/data/oxe/oxe_standardization_transforms.py`,
`octo/data/utils/data_utils.py`, `examples/03_eval_finetuned.py`,
`examples/04_eval_finetuned_on_robot.py`, `examples/envs/widowx_env.py`.

- Loading: `OctoModel.load_pretrained(checkpoint_path, step)` reads
  `config.json`, `example_batch.msgpack`, `dataset_statistics.json` and the
  orbax params at `<step>/`. Its own `hf://` path calls
  `snapshot_download(repo)` with NO revision (resolves `main`) and downloads
  every step a repo ships. So `_download_pinned` fetches only the pinned
  revision's chosen step into a local dir and loads that (T-14-06).
- Text: the config's `HFTokenizer` and `LanguageTokenizer` both resolve
  `"t5-base"` from the HF Hub at `main` (tokenizer files + `AutoConfig`; the
  T5 encoder weights are inside the Octo checkpoint). `_pin_text_encoder`
  fetches `google-t5/t5-base` at a reviewed sha and rewrites the local
  `config.json` to point at it. Any other encoder name is refused.
- Inference (examples 03/04): `tasks = model.create_tasks(texts=[...])`,
  `model.sample_actions(obs, tasks, unnormalization_statistics=<stats>["action"],
  rng=...)` -> `(batch, action_horizon, action_dim)`. Observations are
  `(batch, window, ...)` with `timestep_pad_mask`. `HistoryWrapper` repeats the
  first frame and pads the older slots on reset; missing keys count as padded.
- Images (`ResizeImageWrapper`): Lanczos resize to the example batch's size,
  then for `image_primary` only, the average of the training random-resized-crop
  (scale 0.9): a centre crop of sqrt(0.9) of each side, resized back. Reproduced
  with PIL (like openvla_server.py), so the pixels differ slightly from TF's.
- Un-normalization: `NormalizationType.NORMAL` (`mean`/`std`) where `mask` is
  true; masked dims (the gripper) come back raw.
- Dependency pins (README + requirements.txt): Python 3.10, `jax[cuda11_pip]==0.4.20`,
  `flax==0.7.5`, `tensorflow==2.15.0`, `numpy==1.24.3`. `main()` refuses
  to start on other versions.

Checkpoints (HF API, read 2026-10-01; see `CHECKPOINT_PROFILES`)
----------------------------------------------------------------
- Default: `sapanostic/octo-so101-3hz-finetune` @ a45ff97, step 10000. It is
  `rail-berkeley/octo-base-1.5` fine-tuned (`encoder_and_head`, window 2, horizon 4)
  on a real SO-101 (`so_follower`) lerobot dataset,
  `sapanostic/so101_pick_cube_v2_merged`: 100 episodes, 15 Hz, `right`/`left`/`wrist`
  cameras, task "pick up the cube". It was converted to RLDS at 3 Hz
  (`ivlabs/so101_pick_cube_v2_merged_rlds_3hz`, Apache-2.0, documented card).
  Its action, confirmed from the checkpoint's own `dataset_statistics.json`
  (7-D, mask `[T]*6+[F]`, deltas of a few cm / 0.1 rad) and the dataset card, is
  `action/eef_delta_m_rad` = [dx, dy, dz (m), droll, dpitch, dyaw (rad),
  gripper_normalized]. The pose is Pinocchio FK on the SO101 URDF
  (`so101_new_calib.urdf`), and the gripper is absolute, min-max normalized
  over [-10, 100]. So it is an **EEF delta: the client runs IK**.
  `finetune_config.json`: primary = `image_1`, wrist = `image_2`.
- Alternates: the official MIT `rail-berkeley/octo-{small,base}-1.5`
  (OXE-pretrained) with the `bridge_dataset` statistics, which is what
  `examples/04` uses on the WidowX. Bridge's action is `relabel_actions`'s
  state differences: [dx, dy, dz, droll, dpitch, dyaw] in the WidowX base
  frame at 5 Hz, plus the binarized gripper (1 = open).

Known limitations (documented, not hidden)
------------------------------------------
- The default checkpoint has no model card or license, and its training code
  (`so101_dataset_transform`) is unpublished. Its action space is confirmed
  from its statistics and the dataset card, not from code.
- Which source camera `image_1` is, is inferred as `left` (the dataset's camera
  order is right, left, wrist). So the default primary is this rig's
  `overhead_left`, and `--primary-image-key` overrides it.
- Actions are 3 Hz steps. The control loop applies each one as a delta from the
  current pose, so it can run faster than trained. The safety validator's
  per-step clamp bounds that.
- The server keeps a 2-frame history (Octo's window). It resets on a new
  instruction, on `"reset": true`, or after `HISTORY_RESET_S` seconds idle.
- It returns only the first step of each chunk (receding horizon, execute 1),
  not `examples/04`'s temporal ensembling.

Colab launch: see policy_server_launch.md, section "Tier C: Octo resident adapter".
"""
import argparse
import base64
import collections
import json
import math
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

DEFAULT_CHECKPOINT = "sapanostic/octo-so101-3hz-finetune"
DEFAULT_REVISION = "a45ff970242a66e268871a5f64530947143f9a6e"
DEFAULT_LOCAL_DIR = "/content/octo_checkpoints"
DEFAULT_PORT = 3004  # 3000 FastFS, 3001 OpenVLA-OFT, 3002 Wall-X, 3003 OpenVLA
OCTO_COMMIT = "241fb3514b7c40957a86d869fecb7c7fc353f540"  # github.com/octo-models/octo main HEAD

# T-14-06: the only tokenizer/encoder config Octo may resolve (HF API sha, read 2026-10-01).
T5_REPO = "google-t5/t5-base"
T5_REVISION = "a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1"
T5_FILES = ("config.json", "spiece.model", "tokenizer.json")
REVIEWED_ENCODER = "t5-base"

# Reviewed checkpoints. `backend_configs/octo.json["checkpoints"]` must agree
# (pinned by test_client_and_server_checkpoint_profiles_agree).
CHECKPOINT_PROFILES = {
    "sapanostic/octo-so101-3hz-finetune": dict(
        revision="a45ff970242a66e268871a5f64530947143f9a6e", step=10000, unnorm_key=None,
        action_space="eef_delta"),
    "rail-berkeley/octo-small-1.5": dict(
        revision="dc9aa3019f764726c770814b27e4ab0fc6e32a58", step=300000, unnorm_key="bridge_dataset",
        action_space="eef_delta"),
    "rail-berkeley/octo-base-1.5": dict(
        revision="ee3c10e8edd6ce2e8b1e8744d3c6fba4097bed48", step=300000, unnorm_key="bridge_dataset",
        action_space="eef_delta"),
}

CENTER_CROP_SCALE = 0.9  # ResizeImageWrapper avg_scale
MAX_IMAGE_SIDE = 1024
IMAGE_KEYS = ("overhead_left", "overhead_right", "wrist")
HISTORY_RESET_S = 10.0
PINNED_DEPENDENCIES = {"jax": {"0.4.20"}, "flax": {"0.7.5"}, "tensorflow": {"2.15.0"}, "numpy": {"1.24.3"}}


def _download_pinned(checkpoint, revision, local_dir, step):
    """Materialise the pinned revision's metadata plus ONE step of params as a
    real directory (Octo's own `hf://` path resolves `main` and fetches every step)."""
    from huggingface_hub import snapshot_download

    target = Path(local_dir) / checkpoint.replace("/", "--") / revision
    patterns = ["config.json", "dataset_statistics.json", "example_batch.msgpack", f"{step}/*"]
    return snapshot_download(repo_id=checkpoint, revision=revision, local_dir=str(target),
                             allow_patterns=patterns)


def _pin_text_encoder(model_dir, local_dir):
    """Point the checkpoint's `t5-base` tokenizer and encoder config at a local
    copy of the reviewed `google-t5/t5-base` revision. Refuses any other encoder."""
    from huggingface_hub import snapshot_download

    t5_dir = Path(local_dir) / "google-t5--t5-base" / T5_REVISION
    path = Path(model_dir) / "config.json"
    config = json.loads(path.read_text())
    local = str(t5_dir)
    refs = []
    if config.get("text_processor"):
        refs.append((config["text_processor"]["kwargs"], "tokenizer_name"))
    for spec in (config["model"].get("task_tokenizers") or {}).values():
        kwargs = spec.get("kwargs") or {}
        if kwargs.get("encoder") is not None:
            refs.append((kwargs, "encoder"))
    for holder, key in refs:
        if holder[key] not in (REVIEWED_ENCODER, local):
            raise RuntimeError(f"{path}: {key}={holder[key]!r} is not the reviewed {REVIEWED_ENCODER!r}")
    snapshot_download(repo_id=T5_REPO, revision=T5_REVISION, local_dir=local, allow_patterns=list(T5_FILES))
    for holder, key in refs:
        holder[key] = local  # contains "t5": LanguageTokenizer then builds FlaxT5EncoderModel
    path.write_text(json.dumps(config) + "\n")


def _load_policy(model_dir, step):
    """The official load (README / examples 03-04)."""
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")  # Octo imports TF for I/O only; JAX owns the GPU
    from octo.model.octo_model import OctoModel as UpstreamOctoModel

    return UpstreamOctoModel.load_pretrained(model_dir, step)


def _sample_actions(model, observations, tasks, unnormalization_statistics, seed):
    """`model.sample_actions(...)` with a fresh PRNG key, as examples 03/04 call it
    (default `NormalizationType.NORMAL`, the reviewed checkpoints' type); returns numpy."""
    import jax

    actions = model.sample_actions(observations, tasks, unnormalization_statistics=unnormalization_statistics,
                                   rng=jax.random.PRNGKey(seed))
    return np.asarray(actions)


def preprocess_image(rgb, size, center_crop):
    """HxWx3 uint8 RGB -> `size` (h, w) uint8 RGB: Lanczos resize, then
    optionally ResizeImageWrapper's average crop (sqrt(0.9) per side)."""
    height, width = size
    image = Image.fromarray(np.ascontiguousarray(rgb)).convert("RGB")
    image = image.resize((width, height), Image.LANCZOS)
    if center_crop:
        side = math.sqrt(CENTER_CROP_SCALE)
        dx, dy = width * (1 - side) / 2, height * (1 - side) / 2
        image = image.resize((width, height), Image.BILINEAR, box=(dx, dy, width - dx, height - dy))
    return np.asarray(image, dtype=np.uint8)


def _resolve_action_stats(statistics, unnorm_key, checkpoint):
    if unnorm_key is not None:
        if unnorm_key not in statistics or not isinstance(statistics[unnorm_key], dict):
            raise ValueError(f"unnorm key {unnorm_key!r} not in {checkpoint}'s dataset_statistics "
                             f"{sorted(statistics)}")
        return unnorm_key, statistics[unnorm_key]["action"]
    if isinstance(statistics.get("action"), dict) and "mean" in statistics["action"]:
        return None, statistics["action"]  # a single-dataset fine-tune
    keys = sorted(statistics)
    if len(keys) != 1:
        raise ValueError(f"{checkpoint} has unnorm keys {keys}; pass --unnorm-key to choose one")
    return keys[0], statistics[keys[0]]["action"]


class OctoModel:
    def __init__(self, checkpoint=DEFAULT_CHECKPOINT, revision=None, step=None, unnorm_key=None,
                 local_dir=DEFAULT_LOCAL_DIR, primary_image_key="overhead_left", wrist_image_key="wrist",
                 center_crop=True, seed=0):
        for key in (primary_image_key, wrist_image_key):
            if key not in IMAGE_KEYS:
                raise ValueError(f"image keys must be among {IMAGE_KEYS}")
        profile = CHECKPOINT_PROFILES.get(checkpoint, {})
        if step is None:
            step = profile.get("step")
        if revision:
            model_dir = _download_pinned(checkpoint, revision, local_dir, step)
            _pin_text_encoder(model_dir, local_dir)
        else:
            model_dir = f"hf://{checkpoint}"  # explicit opt-out: Octo's own unpinned path
            step = None  # load_pretrained refuses a step for hf:// paths
        self.model = _load_policy(model_dir, step)

        observation = self.model.example_batch["observation"]
        if "proprio" in observation:
            raise RuntimeError(f"{checkpoint} expects a proprio observation, which this adapter does not "
                               "supply (it would be the checkpoint's own FK pose); not supported")
        if "image_primary" not in observation:
            raise RuntimeError(f"{checkpoint} has no image_primary observation")
        self.window = int(observation["timestep_pad_mask"].shape[1])
        self.image_sizes = {k: tuple(int(v) for v in observation[k].shape[2:4])
                            for k in ("image_primary", "image_wrist") if k in observation}
        self.image_sources = {"image_primary": primary_image_key, "image_wrist": wrist_image_key}
        self.image_sources = {k: v for k, v in self.image_sources.items() if k in self.image_sizes}
        self.required_image_keys = tuple(dict.fromkeys(self.image_sources.values()))

        if unnorm_key is None:
            unnorm_key = profile.get("unnorm_key")
        self.unnorm_key, self.action_stats = _resolve_action_stats(
            self.model.dataset_statistics, unnorm_key, checkpoint)
        self.action_dim = len(self.action_stats["mean"])
        action = self.model.example_batch.get("action")
        self.action_horizon = int(action.shape[-2]) if action is not None else None
        self.center_crop = center_crop
        self._seed = int(seed)
        self._history = collections.deque(maxlen=self.window)
        self._num_obs = 0
        self._last_instruction = None
        self._last_time = None
        self._tasks = {}
        self.model_version = f"{checkpoint}@{revision[:12] if revision else 'unpinned'}"
        stats = {k: np.asarray(v).tolist() for k, v in self.action_stats.items()}
        self.metadata = dict(
            checkpoint=checkpoint, revision=revision or "unpinned", step=step,
            octo_commit=OCTO_COMMIT, text_encoder=f"{T5_REPO}@{T5_REVISION}" if revision else "unpinned",
            unnorm_key=self.unnorm_key, action_space=profile.get("action_space", "unknown"),
            action_dim=self.action_dim, action_horizon=self.action_horizon, window_size=self.window,
            action_mean=stats["mean"], action_std=stats["std"],
            action_mask=stats.get("mask", [True] * self.action_dim),
            normalization_type="normal", image_keys=self.image_sources,
            image_sizes={k: list(v) for k, v in self.image_sizes.items()}, center_crop=center_crop,
            returns="first step of the un-normalized action chunk",
        )

    def reset(self):
        self._history.clear()
        self._num_obs = 0
        self._last_instruction = None

    def _observation(self, images, instruction, reset):
        now = time.monotonic()
        stale = self._last_time is not None and now - self._last_time > HISTORY_RESET_S
        if reset or stale or instruction != self._last_instruction:
            self.reset()
        self._last_time = now
        self._last_instruction = instruction
        frame = {key: preprocess_image(images[source], self.image_sizes[key],
                                       self.center_crop and key == "image_primary")
                 for key, source in self.image_sources.items()}
        if self._num_obs == 0:
            self._history.extend([frame] * self.window)  # HistoryWrapper.reset
        else:
            self._history.append(frame)
        self._num_obs += 1
        observation = {key: np.stack([f[key] for f in self._history])[None] for key in frame}
        pad = np.ones((1, self.window), dtype=bool)
        pad[0, : self.window - min(self._num_obs, self.window)] = False
        observation["timestep_pad_mask"] = pad
        return observation

    def __call__(self, images, instruction, reset=False):
        observation = self._observation(images, instruction, reset)
        if instruction not in self._tasks:
            self._tasks[instruction] = self.model.create_tasks(texts=[instruction])
        self._seed += 1
        chunk = np.asarray(_sample_actions(self.model, observation, self._tasks[instruction], self.action_stats,
                                           self._seed), dtype=np.float64)
        if chunk.ndim != 3 or chunk.shape[0] != 1 or chunk.shape[2] != self.action_dim:
            raise RuntimeError(f"model returned actions of shape {chunk.shape}, "
                               f"expected (1, horizon, {self.action_dim})")
        values = chunk[0, 0]
        if not np.all(np.isfinite(values)):
            raise RuntimeError(f"model returned a non-finite action {values.tolist()}")
        return [float(v) for v in values]


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


def create_app(infer):
    from flask import Flask, jsonify, request
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024
    inference_lock = threading.Lock()
    image_keys = tuple(getattr(infer, 'required_image_keys', ('overhead_left', 'wrist')))

    @app.get('/health')
    def health():
        return jsonify(status='ready', protocol_version=1, model=infer.metadata)

    @app.post('/action')
    def action():
        if not inference_lock.acquire(blocking=False):
            return jsonify(error='octo service busy'), 429
        try:
            body = request.get_json(silent=True)
            if not isinstance(body, dict):
                raise ValueError('expected a JSON object body')
            images_in = body['images']
            if not isinstance(images_in, dict):
                raise TypeError('images must be an object')
            images = {key: _decode_rgb(images_in[key]) for key in image_keys}
            instruction = body['instruction']
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError('instruction must be a non-empty string')
            joint_state = body.get('joint_state')
            if joint_state is not None and not isinstance(joint_state, dict):
                raise TypeError('joint_state must be an object')
            reset = body.get('reset', False)
            if not isinstance(reset, bool):
                raise TypeError('reset must be a boolean')
            if reset and hasattr(infer, 'reset'):
                infer.reset()
            started = time.perf_counter()
            result = infer(images, instruction)
            elapsed = (time.perf_counter() - started) * 1000
            return jsonify(action=[float(v) for v in result], model_version=infer.model_version,
                           inference_ms=elapsed)
        except (ValueError, KeyError, TypeError) as exc:
            return jsonify(error=str(exc)), 400
        except RuntimeError as exc:
            return jsonify(error=f'octo inference failed: {exc}'), 500
        finally:
            inference_lock.release()
    return app


def _installed_versions():
    import importlib.metadata as metadata

    return {name: metadata.version(name) for name in PINNED_DEPENDENCIES}


def _check_dependency_versions():
    versions = _installed_versions()
    bad = {name: version for name, version in versions.items() if version not in PINNED_DEPENDENCIES[name]}
    if bad:
        raise SystemExit(
            f"Octo needs jax==0.4.20, flax==0.7.5, tensorflow==2.15.0, numpy==1.24.3 "
            f"(octo README / requirements.txt); found {bad}. Use the isolated Octo venv."
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT)
    parser.add_argument('--revision', default=None,
                        help="pinned HF commit sha (defaults to the reviewed profile's); "
                             "pass '' only to deliberately load the unpinned repo id")
    parser.add_argument('--step', type=int, default=None, help="checkpoint step (defaults to the profile's)")
    parser.add_argument('--unnorm-key', default=None)
    parser.add_argument('--local-dir', default=DEFAULT_LOCAL_DIR)
    parser.add_argument('--primary-image-key', default='overhead_left', choices=IMAGE_KEYS)
    parser.add_argument('--wrist-image-key', default='wrist', choices=IMAGE_KEYS)
    parser.add_argument('--no-center-crop', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--port', type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    profile = CHECKPOINT_PROFILES.get(args.checkpoint, {})
    revision = args.revision if args.revision is not None else profile.get('revision')
    if revision is None:
        raise SystemExit(f"{args.checkpoint} has no reviewed profile; pass --revision <sha> "
                         "(or --revision '' to deliberately load it unpinned)")
    step = args.step if args.step is not None else profile.get('step')
    if revision and step is None:
        raise SystemExit(f"{args.checkpoint} has no reviewed profile; pass --step <n> "
                         "(a pinned download fetches exactly one checkpoint step)")
    _check_dependency_versions()
    # JAX shares the GPU with nothing else in this process, but keep it from
    # grabbing 75 % of VRAM up front (FastFS may run on the same runtime).
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    model = OctoModel(args.checkpoint, revision=revision or None, step=step, unnorm_key=args.unnorm_key,
                      local_dir=args.local_dir, primary_image_key=args.primary_image_key,
                      wrist_image_key=args.wrist_image_key, center_crop=not args.no_center_crop,
                      seed=args.seed)
    # JIT compilation happens before the socket opens and readiness is reported.
    model({key: np.zeros((256, 256, 3), dtype=np.uint8) for key in model.required_image_keys}, 'warm up')
    model.reset()  # the warmup frame must not become the first real request's history
    print(f'Octo model loaded and warmed ({model.model_version}); ready', flush=True)
    create_app(model).run(host='0.0.0.0', port=args.port, threaded=True, use_reloader=False)


if __name__ == '__main__':
    main()
