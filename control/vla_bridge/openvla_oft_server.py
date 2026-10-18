"""Resident OpenVLA-OFT HTTP adapter (Tier C, Plan 14-03). Run on Colab with the
openvla-oft environment's Python interpreter.

This mirrors `fastfs_server.py`:
- the model loads ONCE
- one throwaway warmup inference runs before the socket opens
- `/health` reports ready
- a non-blocking `inference_lock` answers concurrent requests with 429
- `ValueError`/`KeyError`/`TypeError` on a request become 400

The server stays model-format-only. It returns OpenVLA-OFT's raw 7-D action,
and every `action_contract` convention is applied control-side by
`backends/openvla_oft.py` through `backends/eef_to_joint.py`.

Verified against live upstream source (2026-10-01, execution discipline)
------------------------------------------------------------------------
- Code: github.com/moojink/openvla-oft @ e4287e9 (the model card's
  "github.com/openvla-oft/openvla-oft" link returns 404). Loading follows its
  own server, `vla-scripts/deploy.py`, through
  `experiments/robot/openvla_utils.py`:
  - `get_vla(cfg)` calls `AutoModelForVision2Seq.from_pretrained(...,
    torch_dtype=bfloat16, trust_remote_code=True)`.
  - `get_proprio_projector` and `get_action_head` load the separate
    `proprio_projector--*_checkpoint.pt` / `action_head--*_checkpoint.pt`
    MLPs, which a bare `from_pretrained()` does NOT load.
  - `get_processor(cfg)` and `get_vla_action(...)` run inference.
- Config: copied from the HF model card quick start and LIBERO
  `GenerateConfig`: L1-regression head, no FiLM, 2 input images
  (third-person `full_image` + `wrist_image`), proprio on, center crop on,
  `lora_rank=32`.
- Checkpoint: moojink/openvla-7b-oft-finetuned-libero-object. HF API sha
  4c89574e1c538b6c102f43f0526d60a9d3650148, read 2026-10-01.
  `dataset_statistics.json` has the single key `libero_object_no_noops`.
  - Action: 7-D, BOUNDS_Q99-normalized, chunk of `NUM_ACTIONS_CHUNK=8`.
    After un-normalization it is a robosuite OSC_POSE delta in the
    controller's normalized [-1, 1] units (dataset min/max reach +/-0.9375).
    The gripper is in [0, 1] with 1 = open (`run_libero_eval.process_action`
    flips it for the env).
  - Proprio: 8-D LIBERO Franka state (eef pos 3 + axis-angle 3 + gripper
    qpos 2).

Known limitations (documented, not hidden)
------------------------------------------
- These public checkpoints are LIBERO/Franka-sim policies. Driving SO-101
  with them is out-of-distribution. Plan 14-03's bar is "a validated,
  non-held live action", not task success.
- SO-101 joint degrees are not LIBERO Franka EEF proprio. The client's wire
  contract (`base_http_adapter`) sends `joint_state`, which is accepted and
  ignored. The model gets the request's optional 8-float `proprio_state` if
  present, otherwise the checkpoint's own dataset-mean proprio as a neutral
  input.
- Only the FIRST action of each 8-step chunk is returned (receding horizon).
  The control loop re-queries every tick.

Colab launch: see vla_bridge/notebooks/http_openvla_oft.ipynb.
"""
import argparse
import base64
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

DEFAULT_CHECKPOINT = "moojink/openvla-7b-oft-finetuned-libero-object"
# T-14-06: pin the reviewed HF revision (HF API sha read 2026-10-01), never `main`.
DEFAULT_REVISION = "4c89574e1c538b6c102f43f0526d60a9d3650148"
DEFAULT_LOCAL_DIR = "/content/openvla_oft_checkpoints"
ACTION_DIM = 7
PROPRIO_DIM = 8
MAX_IMAGE_SIDE = 1024
REQUIRED_IMAGES = ("overhead_left", "wrist")  # third-person full_image + wrist_image


def _download_pinned(checkpoint, revision, local_dir):
    """Materialise the exact pinned HF revision as a real directory.

    openvla-oft's local-checkpoint path (`get_vla` / `find_checkpoint_file`)
    then loads that directory. A repo-id load would always resolve `main`.
    """
    from huggingface_hub import snapshot_download

    target = Path(local_dir) / checkpoint.replace("/", "--") / revision
    return snapshot_download(repo_id=checkpoint, revision=revision, local_dir=str(target))


def _load_policy(cfg):
    """Load every OpenVLA-OFT component exactly as `vla-scripts/deploy.py` does.

    Needs the openvla-oft repo on `sys.path` and as the CWD (see `main()`).
    """
    from experiments.robot.openvla_utils import (
        get_action_head,
        get_processor,
        get_proprio_projector,
        get_vla,
        get_vla_action,
    )
    from prismatic.vla.constants import ACTION_DIM as REPO_ACTION_DIM
    from prismatic.vla.constants import PROPRIO_DIM as REPO_PROPRIO_DIM

    if (REPO_ACTION_DIM, REPO_PROPRIO_DIM) != (ACTION_DIM, PROPRIO_DIM):
        raise RuntimeError(
            f"openvla-oft constants resolved to ACTION_DIM={REPO_ACTION_DIM}, "
            f"PROPRIO_DIM={REPO_PROPRIO_DIM}; the LIBERO checkpoints need {ACTION_DIM}/{PROPRIO_DIM}"
        )
    vla = get_vla(cfg)
    proprio_projector = get_proprio_projector(cfg, vla.llm_dim, REPO_PROPRIO_DIM) if cfg.use_proprio else None
    action_head = get_action_head(cfg, vla.llm_dim)
    processor = get_processor(cfg)
    return dict(vla=vla, processor=processor, action_head=action_head,
                proprio_projector=proprio_projector, get_vla_action=get_vla_action)


class OpenVLAOFTModel:
    def __init__(self, checkpoint, revision=None, unnorm_key=None, local_dir=DEFAULT_LOCAL_DIR,
                 use_proprio=True, num_images_in_input=2, center_crop=True):
        pretrained = _download_pinned(checkpoint, revision, local_dir) if revision else checkpoint
        # The fields openvla_utils reads, with LIBERO GenerateConfig values.
        self.cfg = SimpleNamespace(
            model_family="openvla", pretrained_checkpoint=pretrained,
            use_l1_regression=True, use_diffusion=False,
            num_diffusion_steps_train=50, num_diffusion_steps_inference=50,
            use_film=False, num_images_in_input=num_images_in_input, use_proprio=use_proprio,
            center_crop=center_crop, num_open_loop_steps=8, lora_rank=32,
            unnorm_key=unnorm_key or "", load_in_8bit=False, load_in_4bit=False,
        )
        policy = _load_policy(self.cfg)
        self.vla = policy["vla"]
        self.processor = policy["processor"]
        self.action_head = policy["action_head"]
        self.proprio_projector = policy["proprio_projector"]
        self._get_vla_action = policy["get_vla_action"]

        keys = sorted(getattr(self.vla, "norm_stats", None) or {})
        if unnorm_key is None:
            if len(keys) != 1:
                raise ValueError(f"{checkpoint} has unnorm keys {keys}; pass --unnorm-key to choose one")
            unnorm_key = keys[0]
        elif unnorm_key not in keys:
            raise ValueError(f"unnorm key {unnorm_key!r} not in {checkpoint}'s dataset statistics {keys}")
        self.cfg.unnorm_key = unnorm_key
        self.neutral_proprio = np.asarray(
            self.vla.norm_stats[unnorm_key]["proprio"]["mean"], dtype=np.float64
        )
        self.model_version = f"{checkpoint}@{revision[:12] if revision else 'unpinned'}"
        self.metadata = dict(
            checkpoint=checkpoint, revision=revision or "unpinned", unnorm_key=unnorm_key,
            action_space="delta_eef_robosuite_osc_normalized", action_dim=ACTION_DIM,
            gripper="0_close_1_open", proprio_dim=PROPRIO_DIM, use_proprio=use_proprio,
            num_images_in_input=num_images_in_input, returns="first action of chunk",
        )

    def __call__(self, images, proprio_state, instruction):
        obs = {"full_image": images["overhead_left"], "wrist_image": images["wrist"]}
        if self.cfg.use_proprio:
            if proprio_state is None:
                state = self.neutral_proprio.copy()
            else:
                state = np.asarray(proprio_state, dtype=np.float64)
            if state.shape != (PROPRIO_DIM,) or not np.all(np.isfinite(state)):
                raise ValueError(f"proprio_state must be {PROPRIO_DIM} finite floats")
            obs["state"] = state
        actions = self._get_vla_action(
            self.cfg, self.vla, self.processor, obs, instruction,
            action_head=self.action_head, proprio_projector=self.proprio_projector, use_film=False,
        )
        first = np.asarray(actions[0], dtype=np.float64).reshape(-1)
        if first.shape != (ACTION_DIM,):
            raise RuntimeError(f"model returned action of shape {first.shape}, expected ({ACTION_DIM},)")
        return [float(v) for v in first]


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


def _parse_proprio(value):
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != PROPRIO_DIM:
        raise ValueError(f"proprio_state must be a list of {PROPRIO_DIM} numbers")
    state = [float(v) for v in value]
    if not all(np.isfinite(state)):
        raise ValueError("proprio_state must be finite")
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
            return jsonify(error='openvla-oft service busy'), 429
        try:
            body = request.get_json(silent=True)
            if not isinstance(body, dict):
                raise ValueError('expected a JSON object body')
            images_in = body['images']
            if not isinstance(images_in, dict):
                raise TypeError('images must be an object')
            images = {key: _decode_rgb(images_in[key]) for key in REQUIRED_IMAGES}
            instruction = body['instruction']
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError('instruction must be a non-empty string')
            proprio_state = _parse_proprio(body.get('proprio_state'))
            joint_state = body.get('joint_state')
            if joint_state is not None and not isinstance(joint_state, dict):
                raise TypeError('joint_state must be an object')
            started = time.perf_counter()
            result = infer(images, proprio_state, instruction)
            elapsed = (time.perf_counter() - started) * 1000
            return jsonify(action=[float(v) for v in result], model_version=infer.model_version,
                           inference_ms=elapsed)
        except (ValueError, KeyError, TypeError) as exc:
            return jsonify(error=str(exc)), 400
        finally:
            inference_lock.release()
    return app


def main():
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', required=True, help='openvla-oft checkout (github.com/moojink/openvla-oft)')
    parser.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT)
    parser.add_argument('--revision', default=DEFAULT_REVISION,
                        help="pinned HF commit sha; pass '' only to deliberately load the unpinned repo id")
    parser.add_argument('--unnorm-key', default=None)
    parser.add_argument('--local-dir', default=DEFAULT_LOCAL_DIR)
    parser.add_argument('--port', type=int, default=3001)
    args = parser.parse_args()
    repo = str(Path(args.repo).resolve())
    sys.path.insert(0, repo)
    # openvla_utils.check_model_logic_mismatch() walks ./prismatic/ relative to the CWD.
    os.chdir(repo)
    model = OpenVLAOFTModel(args.checkpoint, revision=args.revision or None,
                            unnorm_key=args.unnorm_key, local_dir=args.local_dir)
    # CUDA kernels / lazy init happen before the socket opens and readiness is reported.
    warmup = np.zeros((224, 224, 3), dtype=np.uint8)
    model({'overhead_left': warmup, 'wrist': warmup}, None, 'warm up')
    print('OpenVLA-OFT model loaded and warmed; ready', flush=True)
    create_app(model).run(host='0.0.0.0', port=args.port, threaded=True, use_reloader=False)


if __name__ == '__main__':
    main()
