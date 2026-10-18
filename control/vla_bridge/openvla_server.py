"""Resident OpenVLA HTTP adapter (Tier C, Plan 14-09). Run on Colab inside the
isolated OpenVLA venv (see vla_bridge/notebooks/http_openvla.ipynb).

Same shape as `openvla_oft_server.py` and `fastfs_server.py`:
- the model loads ONCE
- one throwaway warmup inference runs before the socket opens
- `/health` reports ready
- a non-blocking `inference_lock` answers concurrent requests with 429
- `ValueError`/`KeyError`/`TypeError` on a request become 400
- a model failure (`RuntimeError`, e.g. a non-finite action) becomes 500,
  so the client holds position instead of parsing NaN JSON

The server stays model-format-only. It returns OpenVLA's raw 7-D
`predict_action()` output, and `backends/openvla.py` turns that into a
`JOINT_ORDER` target for the checkpoint's action space.

Verified against live upstream source (2026-10-01, execution discipline)
------------------------------------------------------------------------
- Loading: OpenVLA (not OFT) needs no repo install. github.com/openvla/openvla
  @ c8f03f4 (`README.md` quick start, `vla-scripts/deploy.py`,
  `experiments/robot/openvla_utils.get_vla/get_processor/get_vla_action`) loads
  with `AutoProcessor.from_pretrained(path, trust_remote_code=True)` and
  `AutoModelForVision2Seq.from_pretrained(path, torch_dtype=bfloat16,
  low_cpu_mem_usage=True, trust_remote_code=True)`, then calls
  `vla.predict_action(**processor(prompt, image), unnorm_key=..., do_sample=False)`.
- Prompt: `"In: What action should the robot take to {instruction.lower()}?\\nOut:"`
  (deploy.py `get_openvla_prompt`; the fine-tune's `PurePromptBuilder`).
- One image only. `predict_action` takes a single third-person image. The LIBERO
  eval passes `full_image` only; finetune.py loads `load_camera_views=("primary",)`.
- `predict_action` (modeling_prismatic.py) decodes 256-bin tokens and
  un-normalizes with the chosen key's q01/q99 where `mask` is true. Masked-off
  dims come back as the raw bin centre in [-1, 1].
- Center crop: the LIBERO checkpoints and finetune.py's default `image_aug=True`
  train with random crops, so evaluation center-crops `sqrt(0.9)` of each side and
  resizes back to 224 (`openvla_utils.crop_and_resize`). That is reproduced here
  with PIL (a bilinear box resize), so TensorFlow isn't needed. The pixels differ
  slightly from TF's `crop_and_resize`.
- Remote code pin (T-14-06): the fine-tuned checkpoints ship NO
  `*_prismatic.py`. Their `config.json` / `preprocessor_config.json` /
  `tokenizer_config.json` `auto_map` point at
  `openvla/openvla-7b--<module>.<Class>`. transformers 4.40.1
  (`dynamic_module_utils.get_class_from_dynamic_module`) resolves that
  cross-repo code at `code_revision`, which `AutoProcessor` drops before its
  inner `AutoImageProcessor` call, so it would load `main`. Instead
  `_localize_remote_code` copies the three files from openvla/openvla-7b at the
  reviewed sha into the pinned snapshot and rewrites `auto_map` to local
  module names. `trust_remote_code` then only ever executes that reviewed code.
  The sha is 47a0ec7f. Its only commit after 31f090d ("Update modeling_prismatic.py
  to match public GitHub repo") is a README edit. Its `modeling_prismatic.py` is
  byte-identical to github.com/openvla/openvla @ c8f03f4's
  `prismatic/extern/hf/modeling_prismatic.py`.
- Dependency pins: `modeling_prismatic.py` raises unless `timm` is 0.9.10-0.9.16 and
  warns unless `transformers==4.40.1` / `tokenizers==0.19.1` (the README cites
  inference regressions on other versions). `main()` refuses to start otherwise.

Checkpoints (HF API, read 2026-10-01; see `CHECKPOINT_PROFILES`)
----------------------------------------------------------------
- Default: `toasta18/openvla-so101-cylinder` @ f00833c. It is a full OpenVLA-7B
  fine-tune on a real SO-101 (`so_follower`) lerobot dataset,
  `toasta18/so101_task_yellow_cylinder`. That dataset has 50 episodes, 30 fps,
  `overhead` + `wrist` cameras, and the single task "Pick up the black cylinder and
  place it on the plate.". It was converted to RLDS (`toasta18/so101_pick_cylinder_rlds`:
  `image`, `wrist_image`, a 7-D action).
  Its `so101_pick_cylinder` action stats are ABSOLUTE joint positions in degree
  mode: slots 0-4 are the arm joints in JOINT_ORDER, slot 5 is a constant 0 pad,
  and slot 6 is the gripper percent. So no IK is needed. Because its config embeds the
  25 base OXE keys as well, the unnorm key must be explicit.
- Alternates: the official MIT `openvla/openvla-7b-finetuned-libero-{object,
  spatial,goal,10}`. They emit robosuite OSC_POSE deltas, and the client converts
  those through `eef_to_joint.solve_ik()`.

Known limitations (documented, not hidden)
------------------------------------------
- toasta18's model card is empty and its repo has no license. Its config's action
  `mask` was edited after upload (commits a545284 -> f00833c) to
  `[T,T,T,T,T,F,T]`. Its own `dataset_statistics.json` still records OpenVLA's forced
  EEF_POS training mask `[T]*6+[F]`
  (`materialize.make_oxe_dataset_kwargs`). Under that mask the gripper target
  (1-53 %) was never normalized, so the action tokenizer clipped it to 1.0. The
  gripper output is therefore expected to sit near q99 (about 52.7 %) whatever the
  scene. The arm joints were trained normally. Inferred from the published
  stats, not from running the model.
- Which RLDS camera (`image`) the fine-tune saw is assumed to be the dataset's
  `overhead` view (LIBERO-style naming). The default `--image-key` is therefore
  `overhead_left`, one half of this rig's AR0144 stereo pair.
- The LIBERO checkpoints are Franka-sim policies, so those are out of distribution
  on SO-101 hardware.
- Each request is stateless. OpenVLA predicts one action per call, with no chunk.

Colab launch: see vla_bridge/notebooks/http_openvla.ipynb.
"""
import argparse
import base64
import json
import math
import shutil
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

DEFAULT_CHECKPOINT = "toasta18/openvla-so101-cylinder"
DEFAULT_REVISION = "f00833ce58510a955b981ee7e9ba8e9ef4e2ef2e"
DEFAULT_LOCAL_DIR = "/content/openvla_checkpoints"
DEFAULT_IMAGE_KEY = "overhead_left"
DEFAULT_PORT = 3003

# T-14-06: the only code trust_remote_code may execute (HF API sha, read 2026-10-01).
CODE_REPO = "openvla/openvla-7b"
CODE_REVISION = "47a0ec7fc4ec123775a391911046cf33cf9ed83f"
REMOTE_CODE_FILES = ("configuration_prismatic.py", "modeling_prismatic.py", "processing_prismatic.py")
AUTO_MAP_FILES = ("config.json", "preprocessor_config.json", "processor_config.json", "tokenizer_config.json")
_LOCAL_MODULES = tuple(name[: -len(".py")] for name in REMOTE_CODE_FILES)

# Reviewed checkpoints. `backend_configs/openvla.json["checkpoints"]` must agree
# (pinned by test_client_and_server_checkpoint_profiles_agree).
CHECKPOINT_PROFILES = {
    "toasta18/openvla-so101-cylinder": dict(
        revision="f00833ce58510a955b981ee7e9ba8e9ef4e2ef2e", unnorm_key="so101_pick_cylinder",
        action_space="absolute_joint_degrees_so101_7d"),
    "openvla/openvla-7b-finetuned-libero-object": dict(
        revision="287d6cfdf12d07b1449505f66d9bf3550257e9b3", unnorm_key="libero_object",
        action_space="delta_eef_robosuite_osc"),
    "openvla/openvla-7b-finetuned-libero-spatial": dict(
        revision="962318cec55ac10993ff0f5f43eda9a270b4c873", unnorm_key="libero_spatial",
        action_space="delta_eef_robosuite_osc"),
    "openvla/openvla-7b-finetuned-libero-goal": dict(
        revision="fa5ae1e7509348889295bba8e08621d8b55e9baf", unnorm_key="libero_goal",
        action_space="delta_eef_robosuite_osc"),
    "openvla/openvla-7b-finetuned-libero-10": dict(
        revision="80970322773f81baa2e22fe495d0487b93a05cfa", unnorm_key="libero_10",
        action_space="delta_eef_robosuite_osc"),
}

ACTION_DIM = 7
INPUT_SIZE = 224
CENTER_CROP_SCALE = 0.9  # area fraction, as in openvla_utils.get_vla_action
MAX_IMAGE_SIDE = 1024
IMAGE_KEYS = ("overhead_left", "overhead_right", "wrist")
PINNED_DEPENDENCIES = {"transformers": {"4.40.1"}, "tokenizers": {"0.19.1"},
                       "timm": {"0.9.10", "0.9.11", "0.9.12", "0.9.16"}}


def _download_pinned(checkpoint, revision, local_dir):
    """Materialise the exact pinned HF revision as a real directory (a repo-id
    load would resolve `main`)."""
    from huggingface_hub import snapshot_download

    target = Path(local_dir) / checkpoint.replace("/", "--") / revision
    return snapshot_download(repo_id=checkpoint, revision=revision, local_dir=str(target))


def _fetch_code_file(code_repo, filename, code_revision):
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=code_repo, filename=filename, revision=code_revision)


def _localize_auto_map(auto_map, code_repo, path):
    localized = {}
    for name, ref in auto_map.items():
        if not isinstance(ref, str):
            raise RuntimeError(f"{path}: unexpected auto_map entry {name}={ref!r}")
        repo, sep, local_ref = ref.rpartition("--")
        if sep and repo != code_repo:
            raise RuntimeError(f"{path}: auto_map {name} points at {repo!r}, not the reviewed {code_repo!r}")
        if local_ref.split(".", 1)[0] not in _LOCAL_MODULES:
            raise RuntimeError(f"{path}: auto_map {name}={ref!r} is not one of {_LOCAL_MODULES}")
        localized[name] = local_ref
    return localized


def _localize_remote_code(model_dir, code_repo=CODE_REPO, code_revision=CODE_REVISION):
    """Make `model_dir` self-contained: write the reviewed OpenVLA remote code
    into it (overwriting any checkpoint-shipped copy) and point every `auto_map`
    at those local modules. Refuses an `auto_map` naming any other repo."""
    model_dir = Path(model_dir)
    pending = {}
    for name in AUTO_MAP_FILES:
        path = model_dir / name
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        if isinstance(data.get("auto_map"), dict):
            data["auto_map"] = _localize_auto_map(data["auto_map"], code_repo, path)
            pending[path] = data
    for filename in REMOTE_CODE_FILES:
        shutil.copyfile(_fetch_code_file(code_repo, filename, code_revision), model_dir / filename)
    for path, data in pending.items():
        path.write_text(json.dumps(data, indent=2) + "\n")


def _load_policy(model_dir, attn_implementation):
    """The official HF AutoClass load (README quick start / deploy.py)."""
    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        model_dir, attn_implementation=attn_implementation, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True,
    ).to(device)
    vla.eval()
    return dict(vla=vla, processor=processor, device=device, dtype=torch.bfloat16)


def build_prompt(instruction):
    return f"In: What action should the robot take to {instruction.lower()}?\nOut:"


def preprocess_image(rgb, center_crop=True):
    """HxWx3 uint8 RGB -> 224x224 PIL RGB, optionally with OpenVLA's eval-time
    center crop (sqrt(CENTER_CROP_SCALE) of each side, resized back up)."""
    image = Image.fromarray(np.ascontiguousarray(rgb)).convert("RGB")
    image = image.resize((INPUT_SIZE, INPUT_SIZE), Image.LANCZOS)
    if center_crop:
        side = INPUT_SIZE * math.sqrt(CENTER_CROP_SCALE)
        offset = (INPUT_SIZE - side) / 2
        box = (offset, offset, offset + side, offset + side)
        image = image.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR, box=box)
    return image


class OpenVLAModel:
    def __init__(self, checkpoint=DEFAULT_CHECKPOINT, revision=None, unnorm_key=None,
                 local_dir=DEFAULT_LOCAL_DIR, center_crop=True, image_key=DEFAULT_IMAGE_KEY,
                 attn_implementation="sdpa"):
        if image_key not in IMAGE_KEYS:
            raise ValueError(f"image_key must be one of {IMAGE_KEYS}")
        profile = CHECKPOINT_PROFILES.get(checkpoint, {})
        if revision:
            model_dir = _download_pinned(checkpoint, revision, local_dir)
            _localize_remote_code(model_dir)
        else:
            model_dir = checkpoint  # explicit opt-out: remote code then resolves openvla/openvla-7b@main
        policy = _load_policy(model_dir, attn_implementation)
        self.vla = policy["vla"]
        self.processor = policy["processor"]
        self.device = policy["device"]
        self.dtype = policy["dtype"]

        keys = sorted(getattr(self.vla, "norm_stats", None) or {})
        if unnorm_key is None:
            unnorm_key = profile.get("unnorm_key")
        if unnorm_key is None:
            if len(keys) != 1:
                raise ValueError(f"{checkpoint} has unnorm keys {keys}; pass --unnorm-key to choose one")
            unnorm_key = keys[0]
        if unnorm_key not in keys:
            raise ValueError(f"unnorm key {unnorm_key!r} not in {checkpoint}'s norm_stats {keys}")
        action_stats = self.vla.norm_stats[unnorm_key]["action"]
        if len(action_stats.get("q01", [])) != ACTION_DIM:
            raise RuntimeError(f"{checkpoint}/{unnorm_key} action stats are not {ACTION_DIM}-D")
        self.unnorm_key = unnorm_key
        self.center_crop = center_crop
        self.image_key = image_key
        self.model_version = f"{checkpoint}@{revision[:12] if revision else 'unpinned'}"
        self.metadata = dict(
            checkpoint=checkpoint, revision=revision or "unpinned",
            code_repo=CODE_REPO, code_revision=CODE_REVISION if revision else "unpinned",
            unnorm_key=unnorm_key, action_space=profile.get("action_space", "unknown"),
            action_dim=ACTION_DIM, action_mask=list(action_stats.get("mask", [])),
            action_q01=list(action_stats["q01"]), action_q99=list(action_stats["q99"]),
            image_key=image_key, center_crop=center_crop, prompt=build_prompt("<instruction>"),
            returns="raw predict_action() output (one step, no chunk)",
        )

    def __call__(self, images, instruction):
        image = preprocess_image(images[self.image_key], self.center_crop)
        inputs = self.processor(build_prompt(instruction), image).to(self.device, dtype=self.dtype)
        action = self.vla.predict_action(**inputs, unnorm_key=self.unnorm_key, do_sample=False)
        values = np.asarray(action, dtype=np.float64).reshape(-1)
        if values.shape != (ACTION_DIM,):
            raise RuntimeError(f"model returned action of shape {values.shape}, expected ({ACTION_DIM},)")
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
    image_key = getattr(infer, 'image_key', DEFAULT_IMAGE_KEY)

    @app.get('/health')
    def health():
        return jsonify(status='ready', protocol_version=1, model=infer.metadata)

    @app.post('/action')
    def action():
        if not inference_lock.acquire(blocking=False):
            return jsonify(error='openvla service busy'), 429
        try:
            body = request.get_json(silent=True)
            if not isinstance(body, dict):
                raise ValueError('expected a JSON object body')
            images_in = body['images']
            if not isinstance(images_in, dict):
                raise TypeError('images must be an object')
            images = {image_key: _decode_rgb(images_in[image_key])}
            instruction = body['instruction']
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError('instruction must be a non-empty string')
            joint_state = body.get('joint_state')
            if joint_state is not None and not isinstance(joint_state, dict):
                raise TypeError('joint_state must be an object')
            started = time.perf_counter()
            result = infer(images, instruction)
            elapsed = (time.perf_counter() - started) * 1000
            return jsonify(action=[float(v) for v in result], model_version=infer.model_version,
                           inference_ms=elapsed)
        except (ValueError, KeyError, TypeError) as exc:
            return jsonify(error=str(exc)), 400
        except RuntimeError as exc:
            return jsonify(error=f'openvla inference failed: {exc}'), 500
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
            f"OpenVLA needs transformers==4.40.1, tokenizers==0.19.1, timm 0.9.10-0.9.16 "
            f"(openvla README / modeling_prismatic.py); found {bad}. Use the isolated OpenVLA venv."
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT)
    parser.add_argument('--revision', default=None,
                        help="pinned HF commit sha (defaults to the reviewed profile's); "
                             "pass '' only to deliberately load the unpinned repo id")
    parser.add_argument('--unnorm-key', default=None)
    parser.add_argument('--local-dir', default=DEFAULT_LOCAL_DIR)
    parser.add_argument('--image-key', default=DEFAULT_IMAGE_KEY, choices=IMAGE_KEYS)
    parser.add_argument('--no-center-crop', action='store_true')
    parser.add_argument('--attn-implementation', default='sdpa', choices=('sdpa', 'eager', 'flash_attention_2'))
    parser.add_argument('--port', type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    revision = args.revision
    if revision is None:
        revision = CHECKPOINT_PROFILES.get(args.checkpoint, {}).get('revision')
        if revision is None:
            raise SystemExit(f"{args.checkpoint} has no reviewed profile; pass --revision <sha> "
                             "(or --revision '' to deliberately load it unpinned)")
    _check_dependency_versions()
    model = OpenVLAModel(args.checkpoint, revision=revision or None, unnorm_key=args.unnorm_key,
                         local_dir=args.local_dir, center_crop=not args.no_center_crop,
                         image_key=args.image_key, attn_implementation=args.attn_implementation)
    # CUDA kernels / lazy init happen before the socket opens and readiness is reported.
    model({model.image_key: np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)}, 'warm up')
    print(f'OpenVLA model loaded and warmed ({model.model_version}); ready', flush=True)
    create_app(model).run(host='0.0.0.0', port=args.port, threaded=True, use_reloader=False)


if __name__ == '__main__':
    main()
