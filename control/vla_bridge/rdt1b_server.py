"""Resident RDT-1B HTTP adapter (Tier C, Plan 14-11, VLAB-09). Run on Colab inside
the isolated RDT venv (see vla_bridge/notebooks/http_rdt1b.ipynb).

Same shape as `octo_server.py` / `openvla_server.py` / `fastfs_server.py`:
- the model loads ONCE
- one throwaway warmup inference runs before the socket opens
- `/health` reports ready
- a non-blocking `inference_lock` answers concurrent requests with 429
- `ValueError`/`KeyError`/`TypeError` on a request become 400
- a model failure (`RuntimeError`, e.g. a non-finite action) becomes 500,
  so the client holds position instead of parsing NaN JSON

The server stays model-format-only. It returns the checkpoint's unified-action
slots for SO-101 (radians and a [0, 1] gripper), and `backends/rdt1b.py` turns
them into a `JOINT_ORDER` target in this rig's units.

Verified against live upstream source (2026-10-01, execution discipline)
------------------------------------------------------------------------
github.com/thu-ml/RoboticsDiffusionTransformer @ cd79363 (main HEAD, MIT). It is
NOT on PyPI. Files read: `README.md`, `requirements.txt`, `configs/base.yaml`,
`configs/state_vec.py`, `configs/dataset_control_freq.json`,
`scripts/agilex_model.py`, `scripts/agilex_inference.py`,
`scripts/encode_lang.py`, `models/rdt_runner.py`, `models/hub_mixin.py`,
`models/rdt/blocks.py`, `models/multimodal_encoder/t5_encoder.py`,
`models/multimodal_encoder/siglip_encoder.py`, `data/hdf5_vla_dataset.py`,
`train/train.py`, `train/sample.py`, `docs/test_6drot.py`.

- Unified action space (`configs/state_vec.py`, `STATE_VEC_LEN = 128`):
  `[0, 10)` right-arm joint positions, `10` right gripper open, `[30, 33)` EEF
  position, `[33, 39)` EEF 6-D rotation, `[50, 60)` left-arm joints, ...
  The README says a single-arm robot fills the RIGHT-arm part, physical
  quantities stay unnormalized in SI units (radians), and only the gripper is
  min-max normalized to [0, 1]. State and action share the same slots: training
  uses the state indicator as the action mask (`train/train.py`, `train/sample.py`),
  and `data/hdf5_vla_dataset.py` fills the ABSOLUTE target qpos into the same
  slots as the state. So SO-101's 5 arm joints map directly to slots 0-4 and
  its gripper to slot 10: a **joint-position** action space, no IK.
  `_check_state_vec_layout()` re-reads the mapping from the cloned repo at startup.
- Checkpoint, read from the weights themselves (HTTP range reads of
  `pytorch_model.bin`'s pickle, no full download): `rdt-1b` @ eb09036 has
  1,228,319,872 bf16 parameters. The final layer outputs 128 values, the state
  adaptor takes 256 (128 state + 128 mask), the image condition is 4374 = 2
  history x 3 cameras x 729 SigLIP patches, and the language input is 4096
  (T5-XXL). `rdt-170m` @ 8aa386c uses the same 128-D space (166M parameters).
- Loading: the official wrapper `scripts.agilex_model.RoboticDiffusionTransformerModel`.
  The README says to override `_format_joint_to_state()` and
  `_unformat_action_to_joint()` for a new robot; `_wrapper_class()` does exactly
  that. With a directory, `get_policy()` calls `RDTRunner.from_pretrained(dir)`
  (the README's documented path). `huggingface_hub==0.25.2`'s mixin injects
  `config.json` as `config` and loads `pytorch_model.bin` with `strict=False`, so
  `_verify_weights()` re-checks that no key is missing or unexpected.
- Inference (`step()`): 6 images `[ext, right_wrist, left_wrist] x [t-1, t]`.
  `None` becomes a background image, as `agilex_inference.py` does for the first
  step. The proprio state goes in as `(1, n)`, `ctrl_freqs` is the profile's
  control rate, and the output is `(1, 64, n)` absolute targets.
- Language: T5-XXL (`google/t5-v1_1-xxl`, 4096-D). The official deployment
  precomputes embeddings because of VRAM. `T5Embedder(use_offload_folder=...)`
  puts encoder blocks 12-23 on disk (`scripts/encode_lang.py`: "if your GPU VRAM
  is less than 24GB, it is recommended to enable offloading"). This adapter
  encodes live with `--offload-dir` (Pitfall 4), or serves only precomputed
  embeddings (`--lang-embeddings`, `--no-text-encoder`). Encoding mirrors
  `encode_lang.py` and is cached per instruction.
- Pins: Python 3.10, `torch==2.1.0`, `transformers==4.41.0`, `diffusers==0.27.2`,
  `timm==1.0.3`, `accelerate==0.30.1` (README / requirements.txt), plus
  `huggingface_hub==0.25.2`, because diffusers 0.27.2 imports `cached_download`,
  which huggingface_hub removed in 0.26. `main()` refuses other versions.

Checkpoints (HF API, read 2026-10-01; see `CHECKPOINT_PROFILES`)
----------------------------------------------------------------
- Default: `robotics-diffusion-transformer/rdt-1b` @ eb09036 (MIT, official).
  No RDT fine-tune for SO-101 (or SO-100 / Koch) exists on the Hub. The model card
  says RDT "cannot yet generalize to new robot platforms (not seen in the
  pre-training datasets)", so this is out of distribution until fine-tuned.
- Alternate: `robotics-diffusion-transformer/rdt-170m` @ 8aa386c (MIT, official).

Known limitations (documented, not hidden)
------------------------------------------
- The SO-101 slot profile follows the README's own guidance (right-arm slots, SI
  units, gripper = percent / 100, 0 % = closed). The base checkpoint never saw
  SO-101 joint conventions, so a fine-tune is required for meaningful motion.
- The camera order is ext = `overhead_left`, right wrist = `wrist`, and left
  wrist = background (single arm). `--ext-image-key` overrides ext.
- RDT predicts 64 steps at the profile's control rate (30 Hz). The server returns
  the step closest to one client control period ahead (`--client-control-hz`,
  default 2 Hz = `run_vla_episode.py`'s default -> step 14), not step 0, which
  would be only 1/30 s of motion per loop.
- The server keeps the 2-frame image history. It resets on a new instruction, on
  `"reset": true`, or after `HISTORY_RESET_S` seconds idle.
- bfloat16 only (the wrapper's dtype): `main()` refuses GPUs without native bf16
  (T4). T5-XXL alone is ~9.5 GB in bf16, plus a 44.5 GB fp32 download.

Colab launch: see vla_bridge/notebooks/http_rdt1b.ipynb.
"""
import argparse
import base64
import collections
import math
import os
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

DEFAULT_CHECKPOINT = "robotics-diffusion-transformer/rdt-1b"
DEFAULT_LOCAL_DIR = "/content/rdt_checkpoints"
DEFAULT_RDT_REPO = "/content/RoboticsDiffusionTransformer"
DEFAULT_OFFLOAD_DIR = "/content/rdt_t5_offload"
DEFAULT_PORT = 3005  # 3000 FastFS, 3001 OpenVLA-OFT, 3002 Wall-X, 3003 OpenVLA, 3004 Octo
DEFAULT_INSTRUCTION = "pick up the cube"
DEFAULT_CLIENT_CONTROL_HZ = 2.0  # run_vla_episode.py --control-hz default
RDT_COMMIT = "cd79363a1387e8f81c7724d070ef7e45fd23150f"  # github.com/thu-ml/RoboticsDiffusionTransformer main HEAD

# T-14-06: the only encoders RDT may load (HF API shas, read 2026-10-01).
T5_REPO = "google/t5-v1_1-xxl"
T5_REVISION = "3db67ab1af984cf10548a73467f0e5bca2aaaeb2"
T5_FILES = ("config.json", "pytorch_model.bin", "spiece.model", "special_tokens_map.json", "tokenizer_config.json")
SIGLIP_REPO = "google/siglip-so400m-patch14-384"
SIGLIP_REVISION = "9fdffc58afc957d1a03a25b10dba0329ab15c2a3"
SIGLIP_FILES = ("config.json", "preprocessor_config.json", "model.safetensors")
CHECKPOINT_FILES = ("config.json", "pytorch_model.bin")
OFFLOAD_VRAM_THRESHOLD_GIB = 24.0  # scripts/encode_lang.py: offload below 24 GB

# action_contract.JOINT_ORDER (pinned equal by test_server_joint_order_matches_the_action_contract);
# this file runs standalone on Colab, so it does not import vla_bridge.
SO101_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
# configs/state_vec.py @ RDT_COMMIT: right_arm_joint_{0..4}_pos -> 0..4, right_gripper_open -> 10.
SO101_JOINT_SLOTS = (0, 1, 2, 3, 4, 10)
STATE_VEC_LEN = 128

# Reviewed checkpoints. `backend_configs/rdt1b.json["checkpoints"]` must agree
# (pinned by test_client_and_server_checkpoint_profiles_agree).
_SO101_JOINT_PROFILE = dict(action_space="joint_position", state_indices=list(SO101_JOINT_SLOTS),
                            joint_units="rad", gripper_norm_to_percent=[0.0, 100.0], control_hz=30)
CHECKPOINT_PROFILES = {
    "robotics-diffusion-transformer/rdt-1b": dict(
        revision="eb09036cc64ca4945051acbd1bd581d30a1d7711", **_SO101_JOINT_PROFILE),
    "robotics-diffusion-transformer/rdt-170m": dict(
        revision="8aa386cac3bbfd9540676c75b3d767cc7f88a10a", **_SO101_JOINT_PROFILE),
}

MAX_IMAGE_SIDE = 1024
MAX_INSTRUCTION_CHARS = 1000
MAX_LIVE_EMBEDDINGS = 64
IMAGE_KEYS = ("overhead_left", "overhead_right", "wrist")
HISTORY_RESET_S = 10.0
PINNED_DEPENDENCIES = {"torch": {"2.1.0"}, "transformers": {"4.41.0"}, "diffusers": {"0.27.2"},
                       "timm": {"1.0.3"}, "accelerate": {"0.30.1"}, "huggingface_hub": {"0.25.2"}}


def _download_pinned(repo_id, revision, local_dir, patterns):
    """Materialise exactly `patterns` of `repo_id` at the pinned `revision` as a
    real directory (the upstream loaders would otherwise resolve `main`)."""
    from huggingface_hub import snapshot_download

    target = Path(local_dir) / repo_id.replace("/", "--") / revision
    return snapshot_download(repo_id=repo_id, revision=revision, local_dir=str(target),
                             allow_patterns=list(patterns))


def _add_repo_to_path(rdt_repo):
    """The upstream code imports `configs.*` / `models.*` / `scripts.*` from its repo root."""
    root = str(Path(rdt_repo).resolve())
    if not (Path(root) / "scripts" / "agilex_model.py").is_file():
        raise RuntimeError(f"{root} is not a RoboticsDiffusionTransformer checkout (no scripts/agilex_model.py)")
    if root not in sys.path:
        sys.path.insert(0, root)


def _check_state_vec_layout(rdt_repo):
    """Refuse to run if the cloned repo's unified layout is not the one this
    adapter (and backends/rdt1b.py) was reviewed against."""
    _add_repo_to_path(rdt_repo)
    from configs.state_vec import STATE_VEC_IDX_MAPPING, STATE_VEC_LEN as upstream_len

    names = [f"right_arm_joint_{i}_pos" for i in range(5)] + ["right_gripper_open"]
    found = tuple(STATE_VEC_IDX_MAPPING.get(name) for name in names)
    if found != SO101_JOINT_SLOTS or upstream_len != STATE_VEC_LEN:
        raise SystemExit(f"{rdt_repo}/configs/state_vec.py maps {dict(zip(names, found))} (len {upstream_len}); "
                         f"expected {SO101_JOINT_SLOTS} of {STATE_VEC_LEN}. Check out {RDT_COMMIT}.")


def _wrapper_class(base, state_indices):
    """The README's documented robot adaptation: override `_format_joint_to_state()`
    and `_unformat_action_to_joint()` of `RoboticDiffusionTransformerModel`.
    Values arrive already in model units (radians, gripper 0-1)."""
    indices = list(state_indices)

    class SO101RoboticDiffusionTransformerModel(base):
        def _format_joint_to_state(self, joints):
            import torch

            batch, steps, _ = joints.shape
            dim = self.args["model"]["state_token_dim"]
            state = torch.zeros((batch, steps, dim), device=joints.device, dtype=joints.dtype)
            state[:, :, indices] = joints
            state_elem_mask = torch.zeros((batch, dim), device=joints.device, dtype=joints.dtype)
            state_elem_mask[:, indices] = 1
            return state, state_elem_mask

        def _unformat_action_to_joint(self, action):
            return action[:, :, indices]

    return SO101RoboticDiffusionTransformerModel


def _verify_weights(policy, weights_file):
    """`CompatiblePyTorchModelHubMixin` loads with strict=False; a silent key
    mismatch would leave random weights, so re-check it strictly here."""
    import torch

    state_dict = torch.load(str(weights_file), map_location="cpu", weights_only=True)
    result = policy.load_state_dict(state_dict, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"{weights_file} does not match RDTRunner: missing {result.missing_keys[:5]}, "
                           f"unexpected {result.unexpected_keys[:5]}")


def _load_base_config(rdt_repo):
    import yaml

    with open(Path(rdt_repo) / "configs" / "base.yaml") as handle:
        return yaml.safe_load(handle)


def _load_policy(rdt_repo, model_dir, vision_dir, control_hz, state_indices, device):
    """The official load (README "Deployment on Real-Robots" / scripts/agilex_inference.py)."""
    import torch

    _add_repo_to_path(rdt_repo)
    from scripts.agilex_model import RoboticDiffusionTransformerModel

    model = _wrapper_class(RoboticDiffusionTransformerModel, state_indices)(
        _load_base_config(rdt_repo), device=device, dtype=torch.bfloat16, control_frequency=control_hz,
        pretrained=str(model_dir), pretrained_vision_encoder_name_or_path=str(vision_dir))
    _verify_weights(model.policy, Path(model_dir) / "pytorch_model.bin")
    return model


def _load_text_encoder(rdt_repo, t5_dir, offload_dir, max_length, device):
    """Upstream `T5Embedder`, with the offload folder threaded into its
    `use_offload_folder` (encoder blocks 12-23 on disk). Its `available_models`
    allow-list holds only the repo id, so the pinned local copy is added to it."""
    _add_repo_to_path(rdt_repo)
    from models.multimodal_encoder.t5_encoder import T5Embedder

    t5_dir = str(t5_dir)
    if t5_dir not in T5Embedder.available_models:
        T5Embedder.available_models = [*T5Embedder.available_models, t5_dir]
    if offload_dir:
        os.makedirs(offload_dir, exist_ok=True)
    return T5Embedder(device=device, from_pretrained=t5_dir, model_max_length=max_length,
                      use_offload_folder=offload_dir or None)


def _encode_instruction(embedder, instruction, device):
    """Exactly `scripts/encode_lang.py`: (1, L, 4096) last hidden state, on CPU."""
    import torch

    tokens = embedder.tokenizer(instruction, return_tensors="pt", padding="longest",
                                truncation=True)["input_ids"].to(device)
    tokens = tokens.view(1, -1)
    with torch.no_grad():
        return embedder.model(tokens).last_hidden_state.detach().cpu()


def _load_lang_embedding(path):
    """A `scripts/encode_lang.py` output: {"name", "instruction", "embeddings": (1, L, 4096)}."""
    import torch

    saved = torch.load(str(path), map_location="cpu")
    instruction, embeddings = saved.get("instruction"), saved.get("embeddings")
    if not isinstance(instruction, str) or getattr(embeddings, "ndim", None) != 3 or embeddings.shape[0] != 1:
        raise ValueError(f"{path} is not an encode_lang.py embedding file")
    return instruction, embeddings


def _step(model, proprio, images, text_embeds):
    """`RoboticDiffusionTransformerModel.step()`; returns (1, horizon, n) numpy."""
    import torch

    pil_images = [Image.fromarray(image) if image is not None else None for image in images]
    actions = model.step(proprio=torch.tensor([proprio], dtype=torch.float32), images=pil_images,
                         text_embeds=text_embeds)
    return actions.float().cpu().numpy()


def chunk_step_for(control_hz, client_control_hz, horizon):
    """Index of the chunk step about one client control period ahead."""
    if client_control_hz <= 0:
        raise ValueError("client control rate must be positive")
    return int(min(max(round(control_hz / client_control_hz) - 1, 0), horizon - 1))


def joint_state_to_model(joint_state, profile):
    """Rig units (degrees, gripper percent) -> the profile's slot units."""
    if not isinstance(joint_state, dict):
        raise TypeError("joint_state must be an object")
    values = []
    for name in SO101_JOINTS:
        value = joint_state[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"joint_state[{name!r}] must be a finite number")
        values.append(float(value))
    arm = [math.radians(v) if profile["joint_units"] == "rad" else v for v in values[:5]]
    lo, hi = (float(v) for v in profile["gripper_norm_to_percent"])
    return arm + [(values[5] - lo) / (hi - lo)]


class RDT1BModel:
    def __init__(self, checkpoint=DEFAULT_CHECKPOINT, revision=None, rdt_repo=DEFAULT_RDT_REPO,
                 local_dir=DEFAULT_LOCAL_DIR, offload_dir=None, ext_image_key="overhead_left",
                 client_control_hz=DEFAULT_CLIENT_CONTROL_HZ, lang_embeddings=(), text_encoder=True,
                 device="cuda"):
        profile = CHECKPOINT_PROFILES.get(checkpoint)
        if profile is None:
            raise ValueError(f"{checkpoint} has no reviewed slot profile (known: {sorted(CHECKPOINT_PROFILES)})")
        if profile["action_space"] != "joint_position":
            raise RuntimeError(f"{checkpoint}: only joint_position profiles are served; an EEF profile needs "
                               "the current EEF pose as state, which the wire contract does not carry")
        if ext_image_key not in IMAGE_KEYS:
            raise ValueError(f"ext image key must be one of {IMAGE_KEYS}")
        revision = revision or profile["revision"]
        model_dir = _download_pinned(checkpoint, revision, local_dir, CHECKPOINT_FILES)
        vision_dir = _download_pinned(SIGLIP_REPO, SIGLIP_REVISION, local_dir, SIGLIP_FILES)
        self.model = _load_policy(rdt_repo, model_dir, vision_dir, profile["control_hz"],
                                  profile["state_indices"], device)
        common = self.model.args["common"]
        if int(common["img_history_size"]) != 2 or int(common["num_cameras"]) != 3:
            raise RuntimeError(f"base.yaml expects {common['img_history_size']} frames x {common['num_cameras']} "
                               "cameras; this adapter feeds 2 x 3")
        self.horizon = int(common["action_chunk_size"])
        self.chunk_step = chunk_step_for(profile["control_hz"], client_control_hz, self.horizon)
        self.profile = profile
        self.device = device
        self._precomputed = {}
        for path in lang_embeddings:
            instruction, embeddings = _load_lang_embedding(path)
            self._precomputed[instruction] = embeddings
        self._live = collections.OrderedDict()
        self.text_encoder = None
        if text_encoder:
            t5_dir = _download_pinned(T5_REPO, T5_REVISION, local_dir, T5_FILES)
            self.text_encoder = _load_text_encoder(rdt_repo, t5_dir, offload_dir,
                                                   self.model.args["dataset"]["tokenizer_max_length"], device)
        elif not self._precomputed:
            raise ValueError("without the live text encoder, at least one --lang-embeddings file is required")
        self.image_sources = (ext_image_key, "wrist", None)  # [ext, right_wrist, left_wrist]
        self.required_image_keys = tuple(dict.fromkeys(k for k in self.image_sources if k))
        self._history = collections.deque(maxlen=2)
        self._last_instruction = None
        self._last_time = None
        self.model_version = f"{checkpoint}@{revision[:12]}"
        self.metadata = dict(
            checkpoint=checkpoint, revision=revision, rdt_commit=RDT_COMMIT,
            vision_encoder=f"{SIGLIP_REPO}@{SIGLIP_REVISION}",
            text_encoder=f"{T5_REPO}@{T5_REVISION}" if text_encoder else "precomputed only",
            t5_offload_dir=(offload_dir or None) if text_encoder else None,
            precomputed_instructions=sorted(self._precomputed),
            action_space=profile["action_space"], state_indices=list(profile["state_indices"]),
            joint_units=profile["joint_units"], gripper_norm_to_percent=list(profile["gripper_norm_to_percent"]),
            control_hz=profile["control_hz"], action_horizon=self.horizon, chunk_step=self.chunk_step,
            client_control_hz=client_control_hz, img_history_size=2,
            image_order=dict(ext=ext_image_key, right_wrist="wrist", left_wrist="background"),
            returns="chunk[chunk_step] of the profile's slots: 5 arm joints (rad), gripper_open (0-1)",
        )

    def reset(self):
        self._history.clear()
        self._last_instruction = None

    def _text_embeds(self, instruction):
        if instruction in self._precomputed:
            return self._precomputed[instruction]
        if instruction in self._live:
            self._live.move_to_end(instruction)
            return self._live[instruction]
        if self.text_encoder is None:
            raise ValueError(f"no precomputed embedding for {instruction!r} (known: {sorted(self._precomputed)})")
        embeddings = _encode_instruction(self.text_encoder, instruction, self.device)
        self._live[instruction] = embeddings
        while len(self._live) > MAX_LIVE_EMBEDDINGS:
            self._live.popitem(last=False)
        return embeddings

    def __call__(self, images, instruction, joint_state, reset=False):
        proprio = joint_state_to_model(joint_state, self.profile)  # 400 before touching history
        if len(instruction) > MAX_INSTRUCTION_CHARS:
            raise ValueError(f"instruction must be <= {MAX_INSTRUCTION_CHARS} characters")
        text_embeds = self._text_embeds(instruction)
        now = time.monotonic()
        stale = self._last_time is not None and now - self._last_time > HISTORY_RESET_S
        if reset or stale or instruction != self._last_instruction:
            self.reset()
        self._last_time = now
        self._last_instruction = instruction
        current = tuple(images[key] if key else None for key in self.image_sources)
        previous = self._history[-1] if self._history else (None, None, None)  # agilex: background at t-1
        self._history.append(current)
        chunk = np.asarray(_step(self.model, proprio, [*previous, *current], text_embeds), dtype=np.float64)
        width = len(self.profile["state_indices"])
        if chunk.ndim != 3 or chunk.shape[0] != 1 or chunk.shape[2] != width or chunk.shape[1] <= self.chunk_step:
            raise RuntimeError(f"model returned actions of shape {chunk.shape}, "
                               f"expected (1, {self.horizon}, {width})")
        values = chunk[0, self.chunk_step]
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
            return jsonify(error='rdt service busy'), 429
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
            joint_state = body['joint_state']  # RDT conditions on proprio: required
            reset = body.get('reset', False)
            if not isinstance(reset, bool):
                raise TypeError('reset must be a boolean')
            started = time.perf_counter()
            result = infer(images, instruction, joint_state, reset=reset)
            elapsed = (time.perf_counter() - started) * 1000
            return jsonify(action=[float(v) for v in result], model_version=infer.model_version,
                           inference_ms=elapsed)
        except (ValueError, KeyError, TypeError) as exc:
            return jsonify(error=str(exc)), 400
        except RuntimeError as exc:
            return jsonify(error=f'rdt inference failed: {exc}'), 500
        finally:
            inference_lock.release()
    return app


def _installed_versions():
    import importlib.metadata as metadata

    return {name: metadata.version(name).split("+")[0] for name in PINNED_DEPENDENCIES}


def _check_dependency_versions():
    versions = _installed_versions()
    bad = {name: version for name, version in versions.items() if version not in PINNED_DEPENDENCIES[name]}
    if bad:
        raise SystemExit(
            "RDT needs torch==2.1.0, transformers==4.41.0, diffusers==0.27.2, timm==1.0.3, accelerate==0.30.1, "
            f"huggingface_hub==0.25.2 (RDT README / requirements.txt); found {bad}. Use the isolated RDT venv."
        )


def _gpu_info():
    import torch

    if not torch.cuda.is_available():
        return dict(cuda=False, name=None, total_gib=0.0, bf16=False)
    props = torch.cuda.get_device_properties(0)
    return dict(cuda=True, name=props.name, total_gib=props.total_memory / 2**30,
                bf16=bool(torch.cuda.is_bf16_supported()))


def _peak_vram_gib():
    import torch

    return torch.cuda.max_memory_allocated() / 2**30


def resolve_offload_dir(mode, offload_dir, total_gib):
    """`auto` follows encode_lang.py's guidance: offload T5-XXL below 24 GB of VRAM."""
    if mode == "off":
        return None
    if mode == "on" or total_gib < OFFLOAD_VRAM_THRESHOLD_GIB:
        return offload_dir
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT, choices=sorted(CHECKPOINT_PROFILES))
    parser.add_argument('--revision', default=None, help="pinned HF commit sha (defaults to the reviewed profile's)")
    parser.add_argument('--rdt-repo', default=DEFAULT_RDT_REPO, help=f'RoboticsDiffusionTransformer checkout ({RDT_COMMIT[:7]})')
    parser.add_argument('--local-dir', default=DEFAULT_LOCAL_DIR)
    parser.add_argument('--offload-dir', default=DEFAULT_OFFLOAD_DIR,
                        help='T5-XXL offload folder (T5Embedder use_offload_folder); needs ~5 GB of free disk')
    parser.add_argument('--t5-offload', choices=('auto', 'on', 'off'), default='auto',
                        help=f'auto = offload when the GPU has < {OFFLOAD_VRAM_THRESHOLD_GIB:g} GiB')
    parser.add_argument('--lang-embeddings', action='append', default=[],
                        help='scripts/encode_lang.py output (.pt); repeatable')
    parser.add_argument('--no-text-encoder', action='store_true', help='serve precomputed instructions only')
    parser.add_argument('--ext-image-key', default='overhead_left', choices=IMAGE_KEYS)
    parser.add_argument('--client-control-hz', type=float, default=DEFAULT_CLIENT_CONTROL_HZ,
                        help="run_vla_episode.py's --control-hz; picks the returned chunk step")
    parser.add_argument('--warmup-instruction', default=DEFAULT_INSTRUCTION)
    parser.add_argument('--port', type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    _check_dependency_versions()
    _check_state_vec_layout(args.rdt_repo)
    gpu = _gpu_info()
    if not gpu['cuda'] or not gpu['bf16']:
        raise SystemExit(f"RDT runs in bfloat16 and needs a GPU with native bf16 (A100/L4 class); found {gpu}. "
                         "A T4 cannot run it (14-RESEARCH Pitfall 4).")
    offload_dir = resolve_offload_dir(args.t5_offload, args.offload_dir, gpu['total_gib'])
    print(f"GPU {gpu['name']} {gpu['total_gib']:.1f} GiB; T5-XXL offload: {offload_dir or 'off'}", flush=True)
    model = RDT1BModel(args.checkpoint, revision=args.revision, rdt_repo=args.rdt_repo, local_dir=args.local_dir,
                       offload_dir=offload_dir, ext_image_key=args.ext_image_key,
                       client_control_hz=args.client_control_hz, lang_embeddings=args.lang_embeddings,
                       text_encoder=not args.no_text_encoder)
    warmup = args.warmup_instruction if not args.no_text_encoder else model.metadata['precomputed_instructions'][0]
    model({key: np.zeros((384, 384, 3), dtype=np.uint8) for key in model.required_image_keys}, warmup,
          {name: 0.0 for name in SO101_JOINTS})
    model.reset()  # the warmup frame must not become the first real request's history
    print(f'RDT model loaded and warmed ({model.model_version}); peak VRAM {_peak_vram_gib():.1f} GiB; ready',
          flush=True)
    create_app(model).run(host='0.0.0.0', port=args.port, threaded=True, use_reloader=False)


if __name__ == '__main__':
    main()
